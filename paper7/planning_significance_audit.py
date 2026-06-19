"""Planning significance audit for Paper 7 CEUS revision evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PLANNING_FIELDS = (
    "reward",
    "reward_real",
    "slope_change_pct",
    "cont_change",
    "baimu_count_change",
    "baimu_area_change_ha",
    "budget_used",
    "training_time_s",
)


def concentration_metrics(selected_blocks: list[int]) -> dict[str, Any]:
    """Summarize how concentrated a sequence of selected block IDs is."""
    if not selected_blocks:
        return {
            "n_actions": 0,
            "n_unique_blocks": 0,
            "unique_share": None,
            "top1_share": None,
            "top3_share": None,
            "hhi": None,
            "entropy_norm": None,
        }

    counts = Counter(int(block_id) for block_id in selected_blocks)
    total = len(selected_blocks)
    shares = [count / total for _, count in counts.most_common()]
    entropy = -sum(share * math.log(share) for share in shares)
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 0.0
    top3 = sum(count for _, count in counts.most_common(3)) / total
    return {
        "n_actions": int(total),
        "n_unique_blocks": int(len(counts)),
        "unique_share": round(len(counts) / total, 6),
        "top1_share": round(shares[0], 6),
        "top3_share": round(top3, 6),
        "hhi": round(sum(share * share for share in shares), 6),
        "entropy_norm": round(entropy / max_entropy, 6) if max_entropy > 0 else 0.0,
    }


def summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize core planning outcomes for a set of policy evaluation rows."""
    summary: dict[str, Any] = {"n": len(rows)}
    for field_name in PLANNING_FIELDS:
        values = [
            float(row[field_name])
            for row in rows
            if row.get(field_name) is not None and math.isfinite(float(row[field_name]))
        ]
        if values:
            summary[f"{field_name}_mean"] = round(mean(values), 6)
            summary[f"{field_name}_sd"] = round(pstdev(values), 6) if len(values) > 1 else 0.0
            summary[f"{field_name}_min"] = round(min(values), 6)
            summary[f"{field_name}_max"] = round(max(values), 6)
    return summary


def build_report(
    seed_dir: Path,
    baselines_path: Path,
    output_path: Path | None = None,
    policy_induced_path: Path | None = None,
) -> dict[str, Any]:
    with_cal = _load_seed_rows(seed_dir, "with_cal")
    no_cal = _load_seed_rows(seed_dir, "no_cal")
    baselines = json.loads(baselines_path.read_text(encoding="utf-8"))

    report: dict[str, Any] = {
        "description": (
            "Planning significance audit for Paper 7. Summaries combine slope, "
            "contiguity, baimu, reward, budget, and available action-concentration "
            "evidence rather than relying on a single slope percentage."
        ),
        "seed_dir": str(seed_dir),
        "baselines_path": str(baselines_path),
        "calibrated_policy": summarize_policy_rows(with_cal),
        "uncalibrated_policy": summarize_policy_rows(no_cal),
        "paired_calibration_effects": _paired_deltas(with_cal, no_cal),
        "non_learning_baselines": _summarize_baselines(baselines),
        "action_concentration": {
            "status": "unavailable_full_sequence",
            "reason": (
                "Seed evaluation files store final outcomes but not complete action "
                "sequences. Head-only action concentration is reported when the "
                "policy-induced diagnostic artifact is available."
            ),
        },
    }

    if policy_induced_path is not None and policy_induced_path.exists():
        report["policy_induced_path"] = str(policy_induced_path)
        report["action_concentration"] = _summarize_policy_induced_actions(
            json.loads(policy_induced_path.read_text(encoding="utf-8"))
        )

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _load_seed_rows(seed_dir: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(seed_dir.glob(f"{label}_eval_seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for item in payload:
                row = dict(item)
                row.setdefault("seed", _seed_from_path(path, label))
                row.setdefault("label", label)
                rows.append(row)
        else:
            row = dict(payload)
            row.setdefault("seed", _seed_from_path(path, label))
            row.setdefault("label", label)
            rows.append(row)
    return sorted(rows, key=lambda row: int(row.get("seed", 0)))


def _seed_from_path(path: Path, label: str) -> int:
    return int(path.stem.replace(f"{label}_eval_seed", ""))


def _paired_deltas(
    with_cal: list[dict[str, Any]], no_cal: list[dict[str, Any]]
) -> dict[str, Any]:
    with_by_seed = {int(row["seed"]): row for row in with_cal if "seed" in row}
    no_by_seed = {int(row["seed"]): row for row in no_cal if "seed" in row}
    paired_seeds = sorted(set(with_by_seed).intersection(no_by_seed))
    deltas: list[dict[str, Any]] = []
    for seed in paired_seeds:
        item: dict[str, Any] = {"seed": seed}
        for field_name in (
            "slope_change_pct",
            "reward_real",
            "cont_change",
            "baimu_count_change",
            "baimu_area_change_ha",
        ):
            if field_name in with_by_seed[seed] and field_name in no_by_seed[seed]:
                item[f"{field_name}_delta_with_minus_no"] = round(
                    float(with_by_seed[seed][field_name])
                    - float(no_by_seed[seed][field_name]),
                    6,
                )
        deltas.append(item)

    summary = {"n_paired_seeds": len(paired_seeds), "paired_seeds": paired_seeds}
    for field_name in (
        "slope_change_pct",
        "reward_real",
        "cont_change",
        "baimu_count_change",
        "baimu_area_change_ha",
    ):
        key = f"{field_name}_delta_with_minus_no"
        values = [float(row[key]) for row in deltas if key in row]
        if values:
            summary[f"{key}_mean"] = round(mean(values), 6)
            summary[f"{key}_sd"] = round(pstdev(values), 6) if len(values) > 1 else 0.0
    summary["per_seed"] = deltas
    return summary


def _summarize_baselines(payload: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "description": payload.get("description"),
        "budget": payload.get("budget"),
        "policies": {},
    }
    if "summary" in payload:
        for row in payload["summary"]:
            summary["policies"][row["policy"]] = row
    elif "runs" in payload:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in payload["runs"]:
            grouped.setdefault(row["policy"], []).append(row)
        summary["policies"] = {
            policy: summarize_policy_rows(rows) for policy, rows in sorted(grouped.items())
        }
    return summary


def _summarize_policy_induced_actions(payload: dict[str, Any]) -> dict[str, Any]:
    per_seed = []
    combined: list[int] = []
    for episode in payload.get("episodes", []):
        summary = episode.get("summary", {})
        actions = [int(action) for action in summary.get("selected_actions_head", [])]
        seed = summary.get("seed")
        combined.extend(actions)
        row = {"seed": seed, "head_only": True}
        row.update(concentration_metrics(actions))
        per_seed.append(row)

    aggregate = concentration_metrics(combined)
    aggregate["status"] = "head_only_available"
    aggregate["head_only"] = True
    aggregate["n_policy_episodes"] = len(per_seed)
    aggregate["per_seed"] = per_seed
    aggregate["interpretation"] = (
        "Concentration uses selected_actions_head from the policy-induced "
        "diagnostic, not full 100-step action sequences."
    )
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-dir",
        type=Path,
        default=Path("paper7/results/revision/seeds"),
    )
    parser.add_argument(
        "--baselines",
        type=Path,
        default=Path("paper7/results/revision/bishan_strong_baselines.json"),
    )
    parser.add_argument(
        "--policy-induced",
        type=Path,
        default=Path("paper7/results/revision/policy_induced_diagnostics_15seed.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/revision/planning_significance_audit.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        seed_dir=args.seed_dir,
        baselines_path=args.baselines,
        policy_induced_path=args.policy_induced,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
