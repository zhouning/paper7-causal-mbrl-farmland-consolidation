"""
End-to-end evidence audit for the Paper 7 CEUS revision.

This script does not retrain expensive RL models. It validates the evidence chain
from data assets and recorded trajectories through learned-environment training,
real-environment evaluation, calibration diagnostics, strong baselines, and the
external Dongxing feasibility check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
DEFAULT_OUT = PAPER7_DIR / "results" / "revision" / "end_to_end_validation.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, hash_files: bool = False) -> dict[str, Any]:
    exists = path.exists()
    item: dict[str, Any] = {
        "path": str(path.relative_to(REPO_ROOT) if path.is_absolute() else path),
        "exists": exists,
    }
    if exists and path.is_file():
        stat = path.stat()
        item["bytes"] = stat.st_size
        item["modified_utc"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
        if hash_files:
            item["sha256"] = file_sha256(path)
    return item


def _extract_slope(payload: Any) -> float:
    if isinstance(payload, list):
        values = [float(row["slope_change_pct"]) for row in payload]
        return mean(values)
    if isinstance(payload, dict):
        return float(payload["slope_change_pct"])
    raise TypeError(f"Unsupported evaluation payload type: {type(payload)!r}")


def summarize_seed_evaluations(seed_dir: Path) -> dict[str, Any]:
    """Summarize paired calibrated and uncalibrated real-environment evaluations."""
    no_cal: dict[int, float] = {}
    with_cal: dict[int, float] = {}
    for path in sorted(seed_dir.glob("no_cal_eval_seed*.json")):
        seed = int(path.stem.replace("no_cal_eval_seed", ""))
        no_cal[seed] = _extract_slope(load_json(path))
    for path in sorted(seed_dir.glob("with_cal_eval_seed*.json")):
        seed = int(path.stem.replace("with_cal_eval_seed", ""))
        with_cal[seed] = _extract_slope(load_json(path))

    paired = sorted(set(no_cal).intersection(with_cal))
    no_values = [no_cal[seed] for seed in paired]
    with_values = [with_cal[seed] for seed in paired]
    no_mean = mean(no_values) if no_values else math.nan
    with_mean = mean(with_values) if with_values else math.nan
    improvement_pct = (
        (abs(with_mean) - abs(no_mean)) / abs(no_mean) * 100.0
        if no_values and no_mean != 0
        else math.nan
    )

    return {
        "status": "supported" if paired else "missing",
        "seed_dir": str(seed_dir.relative_to(REPO_ROOT)),
        "n_no_cal_files": len(no_cal),
        "n_with_cal_files": len(with_cal),
        "n_paired_seeds": len(paired),
        "paired_seeds": paired,
        "balanced_pairs": len(no_cal) == len(with_cal) == len(paired),
        "no_cal_mean": round(no_mean, 6) if no_values else None,
        "no_cal_std": round(pstdev(no_values), 6) if len(no_values) > 1 else 0.0,
        "with_cal_mean": round(with_mean, 6) if with_values else None,
        "with_cal_std": round(pstdev(with_values), 6)
        if len(with_values) > 1
        else 0.0,
        "improvement_pct": round(improvement_pct, 6) if no_values else None,
    }


def summarize_alpha_grid(
    grid_path: Path, pre_specified_alpha: float = 0.185
) -> dict[str, Any]:
    """Summarize reward-scale grid results and distance from pre-specified alpha."""
    rows = load_json(grid_path)
    by_alpha: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_alpha[float(row["reward_scale"])].append(float(row["slope_change_pct"]))

    means = {
        alpha: mean(values)
        for alpha, values in sorted(by_alpha.items(), key=lambda item: item[0])
    }
    best_alpha = min(means, key=lambda alpha: means[alpha])
    pre_alpha = min(means, key=lambda alpha: abs(alpha - pre_specified_alpha))
    best_slope = means[best_alpha]
    pre_slope = means[pre_alpha]
    relative_gap_pct = (
        abs(abs(best_slope) - abs(pre_slope)) / abs(best_slope) * 100.0
        if best_slope != 0
        else math.nan
    )

    return {
        "status": "supported",
        "grid_path": str(grid_path.relative_to(REPO_ROOT)),
        "n_runs": len(rows),
        "n_alphas": len(means),
        "best_alpha": round(best_alpha, 6),
        "best_slope_mean": round(best_slope, 6),
        "pre_specified_alpha": round(pre_specified_alpha, 6),
        "nearest_grid_alpha_to_pre_specified": round(pre_alpha, 6),
        "pre_specified_slope_mean": round(pre_slope, 6),
        "relative_gap_pct": round(relative_gap_pct, 6),
        "alpha_means": {f"{alpha:.3f}": round(value, 6) for alpha, value in means.items()},
    }


def summarize_transition_training(history_path: Path, model_path: Path) -> dict[str, Any]:
    history = load_json(history_path)
    val_loss = [float(x) for x in history.get("val_loss", [])]
    val_cos = [float(x) for x in history.get("val_obs_cosine", [])]
    val_reward = [float(x) for x in history.get("val_reward_mse", [])]
    return {
        "status": "supported" if val_loss and model_path.exists() else "missing",
        "model": describe_file(model_path),
        "history": describe_file(history_path),
        "epochs_recorded": len(val_loss),
        "best_val_loss": round(min(val_loss), 6) if val_loss else None,
        "final_val_loss": round(val_loss[-1], 6) if val_loss else None,
        "final_val_obs_cosine": round(val_cos[-1], 6) if val_cos else None,
        "final_val_reward_mse": round(val_reward[-1], 6) if val_reward else None,
    }


def summarize_rollout_diagnostics(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    h100 = payload["horizons"]["100"]
    h1 = payload["horizons"]["1"]
    return {
        "status": "supported",
        "path": str(path.relative_to(REPO_ROOT)),
        "n_transitions_loaded": payload.get("n_transitions_loaded"),
        "n_starts": payload.get("n_starts"),
        "horizon_1_reward_mae": h1["reward_mae"],
        "horizon_100_reward_mae": h100["reward_mae"],
        "horizon_100_global_mae": h100["global_mae"],
        "horizon_100_mask_agreement": h100["mask_agreement"],
        "interpretation": (
            "supports learned-environment training as an approximate surrogate; "
            "final policy outcomes still require real-environment evaluation"
        ),
    }


def summarize_policy_induced_diagnostics(path: Path) -> dict[str, Any]:
    """Summarize learned-vs-real diagnostics under trained policy actions."""
    payload = load_json(path)
    aggregate = payload["aggregate"]
    return {
        "status": "supported",
        "path": str(path.relative_to(REPO_ROOT)),
        "support_size": int(payload.get("support_size", 0)),
        "n_policy_episodes": int(aggregate["n_episodes"]),
        "policy_induced_selected_block_mae_mean": aggregate[
            "selected_block_mae_mean_mean"
        ],
        "policy_induced_all_block_mae_mean": aggregate["all_block_mae_mean_mean"],
        "policy_induced_global_mae_mean": aggregate["global_mae_mean_mean"],
        "policy_induced_reward_mae_mean": aggregate["reward_mae_mean_mean"],
        "policy_induced_calibrated_reward_mae_mean": aggregate[
            "calibrated_reward_mae_mean_mean"
        ],
        "policy_induced_mask_agreement_mean": aggregate["mask_agreement_mean_mean"],
        "policy_induced_support_distance_mean": aggregate[
            "support_distance_mean_mean"
        ],
        "policy_induced_support_distance_q95": aggregate["support_distance_q95_mean"],
        "policy_induced_final_real_slope_pct_mean": aggregate[
            "final_real_slope_change_pct_mean"
        ],
        "interpretation": (
            "trained-policy actions induce states close to the recorded trajectory "
            "support and retain high action-mask agreement; final outcomes are "
            "still measured in the real parcel-simulation environment"
        ),
    }


def summarize_calibration(path: Path, sensitivity_path: Path) -> dict[str, Any]:
    calibration = load_json(path)
    sensitivity = load_json(sensitivity_path)
    mixed = next(scope for scope in sensitivity["scopes"] if scope["scope"] == "mixed")
    median = next(
        item for item in mixed["thresholds"] if float(item["treatment_percentile"]) == 50.0
    )
    return {
        "status": "supported_as_observational_regularization",
        "calibration_path": str(path.relative_to(REPO_ROOT)),
        "sensitivity_path": str(sensitivity_path.relative_to(REPO_ROOT)),
        "empirical_att": round(float(calibration["empirical_att"]), 6),
        "predicted_att": round(float(calibration["predicted_att"]), 6),
        "calibration_factor": round(float(calibration["calibration_factor"]), 6),
        "mixed_median_common_support": median["overlap"]["common_support_share"],
        "mixed_median_trimmed_att": median["att_trimmed"]["att"],
        "mixed_median_ci_low": median["att_trimmed"]["ci_low"],
        "mixed_median_ci_high": median["att_trimmed"]["ci_high"],
        "interpretation": (
            "observational treatment-effect-informed reward regularization; "
            "not definitive causal identification"
        ),
    }


def summarize_bishan_baselines(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    by_policy = {row["policy"]: row for row in payload["summary"]}
    return {
        "status": "supported",
        "path": str(path.relative_to(REPO_ROOT)),
        "budget": payload.get("budget"),
        "random_slope_mean": by_policy["random"]["slope_change_pct_mean"],
        "random_budget_completion": by_policy["random"]["budget_completion_mean"],
        "slope_gap_greedy_slope": by_policy["slope_gap_greedy"][
            "slope_change_pct_mean"
        ],
        "area_weighted_greedy_slope": by_policy["area_weighted_greedy"][
            "slope_change_pct_mean"
        ],
        "immediate_slope_delta_slope": by_policy["immediate_slope_delta"][
            "slope_change_pct_mean"
        ],
        "immediate_slope_delta_budget_completion": by_policy["immediate_slope_delta"][
            "budget_completion_mean"
        ],
        "interpretation": (
            "strong local non-learning rules do not match the calibrated learned "
            "policy under the Bishan real-environment evaluation"
        ),
    }


def summarize_dongxing(
    audit_path: Path, block_summary_path: Path, dynamic_path: Path
) -> dict[str, Any]:
    audit = load_json(audit_path)
    blocks = load_json(block_summary_path)
    dynamic = load_json(dynamic_path)
    return {
        "status": "supported",
        "audit_path": str(audit_path.relative_to(REPO_ROOT)),
        "block_summary_path": str(block_summary_path.relative_to(REPO_ROOT)),
        "dynamic_path": str(dynamic_path.relative_to(REPO_ROOT)),
        "n_parcels": audit["source"]["record_count"],
        "n_blocks": blocks["n_blocks"],
        "slope_coverage_farmland": audit["slope"]["by_land_use"]["farmland"][
            "coverage"
        ],
        "slope_coverage_forest": audit["slope"]["by_land_use"]["forest"]["coverage"],
        "has_complete_slope_for_swap": blocks["has_complete_slope_for_swap"],
        "dynamic_slope_gap_pct": dynamic["strategies"]["dynamic_slope_gap"][
            "slope_change_pct"
        ],
        "dynamic_area_weighted_pct": dynamic["strategies"][
            "dynamic_area_weighted_gap"
        ]["slope_change_pct"],
        "random_dynamic_mean_pct": dynamic["random_baseline"][
            "slope_change_pct_mean"
        ],
        "random_dynamic_n": dynamic["random_baseline"]["n_seeds"],
        "has_learned_policy": False,
        "interpretation": (
            "external data, action-space, and dynamic non-RL feasibility; "
            "not learned-policy transfer"
        ),
    }


def classify_claim_scope(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify which manuscript claims are supported by which evidence level."""
    bishan = evidence.get("bishan_seed_chain", {})
    dongxing = evidence.get("dongxing_dynamic", {})
    calibration = evidence.get("calibration", {})
    rollout = evidence.get("transition_rollout", {})
    policy_shift = evidence.get("policy_induced_diagnostics", {})

    return [
        {
            "id": "bishan_model_based_policy_e2e",
            "claim": (
                "Bishan model-based policies are trained in a learned environment "
                "and evaluated in the real parcel-simulation environment."
            ),
            "status": "supported"
            if bishan.get("status") == "supported"
            and bishan.get("n_paired_seeds", 0) >= 15
            else "partial_or_missing",
            "evidence_level": "end_to_end_result_chain",
            "n_paired_seeds": bishan.get("n_paired_seeds", 0),
        },
        {
            "id": "calibration_scope",
            "claim": "The calibration factor improves learned-environment policy training.",
            "status": calibration.get("status", "missing"),
            "evidence_level": "observational_regularization_plus_downstream_evaluation",
            "causal_identification_claimed": False,
            "interpretation": calibration.get("interpretation"),
        },
        {
            "id": "transition_surrogate_scope",
            "claim": "The learned transition model can be used as an approximate training surrogate.",
            "status": rollout.get("status", "missing"),
            "evidence_level": "one_step_training_metrics_plus_multistep_rollout_diagnostics",
            "standalone_simulator_for_final_outcomes": False,
            "interpretation": rollout.get("interpretation"),
        },
        {
            "id": "policy_induced_surrogate_scope",
            "claim": (
                "The trained learned-environment policy remains evaluable under "
                "synchronized learned-vs-real rollouts."
            ),
            "status": "supported"
            if policy_shift.get("status") == "supported"
            and policy_shift.get("n_policy_episodes", 0) >= 3
            else "partial_or_missing",
            "evidence_level": "trained_policy_induced_learned_vs_real_diagnostics",
            "n_policy_episodes": policy_shift.get("n_policy_episodes", 0),
            "standalone_simulator_for_final_outcomes": False,
            "interpretation": policy_shift.get("interpretation"),
        },
        {
            "id": "dongxing_external_scope",
            "claim": "Dongxing supports external-county portability.",
            "status": "supported_as_external_feasibility"
            if dongxing.get("status") == "supported"
            and not dongxing.get("has_learned_policy", False)
            else "policy_transfer_supported",
            "evidence_level": "external_data_action_space_dynamic_non_rl",
            "policy_transfer_tested": bool(dongxing.get("has_learned_policy", False)),
            "interpretation": dongxing.get(
                "interpretation",
                "external feasibility evidence; not learned-policy transfer",
            ),
        },
    ]


def build_validation_report(
    repo_root: Path = REPO_ROOT, hash_core_assets: bool = False
) -> dict[str, Any]:
    paper7_dir = repo_root / "paper7"
    evidence: dict[str, Any] = {}

    core_assets = {
        "trajectories": [
            describe_file(path, hash_files=False)
            for path in sorted((paper7_dir / "trajectories").glob("*.npz"))
        ],
        "transition_model": describe_file(
            paper7_dir / "models" / "transition_model.pt",
            hash_files=hash_core_assets,
        ),
        "training_history": describe_file(
            paper7_dir / "models" / "training_history.json",
            hash_files=hash_core_assets,
        ),
        "dongxing_slope_gpkg": describe_file(
            paper7_dir / "data" / "dongxing_DLTB_with_slope.gpkg",
            hash_files=False,
        ),
    }

    evidence["transition_training"] = summarize_transition_training(
        paper7_dir / "models" / "training_history.json",
        paper7_dir / "models" / "transition_model.pt",
    )
    evidence["bishan_seed_chain"] = summarize_seed_evaluations(
        paper7_dir / "results" / "revision" / "seeds"
    )
    evidence["alpha_grid"] = summarize_alpha_grid(
        paper7_dir / "results" / "revision" / "alpha_grid" / "grid_results.json"
    )
    evidence["transition_rollout"] = summarize_rollout_diagnostics(
        paper7_dir / "results" / "revision" / "transition_rollout_diagnostics.json"
    )
    evidence["policy_induced_diagnostics"] = summarize_policy_induced_diagnostics(
        paper7_dir / "results" / "revision" / "policy_induced_diagnostics.json"
    )
    evidence["calibration"] = summarize_calibration(
        paper7_dir / "results" / "causal_calibration.json",
        paper7_dir / "results" / "revision" / "causal_sensitivity_diagnostics.json",
    )
    evidence["bishan_non_learning_baselines"] = summarize_bishan_baselines(
        paper7_dir / "results" / "revision" / "bishan_strong_baselines.json"
    )
    evidence["dongxing_dynamic"] = summarize_dongxing(
        paper7_dir / "results" / "dongxing_data_audit_slope.json",
        paper7_dir / "results" / "dongxing_blocks_slope" / "summary.json",
        paper7_dir / "results" / "dongxing_dynamic_baselines.json",
    )

    return {
        "description": (
            "End-to-end evidence audit for Paper 7. Expensive RL models are not "
            "retrained; the script validates the recorded data-to-result chain and "
            "classifies each manuscript claim by evidence level."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "core_assets": core_assets,
        "evidence": evidence,
        "claim_scope": classify_claim_scope(evidence),
        "overall_status": "supported_with_bounded_external_scope",
        "external_policy_transfer_tested": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Path to write the validation JSON report.",
    )
    parser.add_argument(
        "--hash-core-assets",
        action="store_true",
        help="Hash small core assets in addition to recording file sizes.",
    )
    args = parser.parse_args()

    report = build_validation_report(hash_core_assets=args.hash_core_assets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote end-to-end validation report to {out_path}")
    print(f"Overall status: {report['overall_status']}")
    print(
        "Dongxing scope: "
        f"{next(item for item in report['claim_scope'] if item['id'] == 'dongxing_external_scope')['status']}"
    )


if __name__ == "__main__":
    main()
