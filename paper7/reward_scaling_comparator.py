"""Reward-scaling comparator for Paper 7 CEUS revision evidence."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


def summarize_by_scale(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group reward-grid rows by scale and summarize planning outcomes."""
    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        scale = float(row["reward_scale"])
        grouped.setdefault(scale, []).append(row)

    summary: dict[str, dict[str, Any]] = {}
    for scale, items in sorted(grouped.items()):
        item: dict[str, Any] = {
            "reward_scale": round(scale, 6),
            "n": len(items),
            "seeds": sorted(int(row["seed"]) for row in items if "seed" in row),
        }
        for field_name in (
            "slope_change_pct",
            "reward_real",
            "cont_change",
            "baimu_count_change",
            "baimu_area_change_ha",
            "training_time_s",
        ):
            values = [
                float(row[field_name])
                for row in items
                if row.get(field_name) is not None and math.isfinite(float(row[field_name]))
            ]
            if values:
                item[f"{field_name}_mean"] = round(mean(values), 6)
                item[f"{field_name}_sd"] = round(pstdev(values), 6) if len(values) > 1 else 0.0
                item[f"{field_name}_min"] = round(min(values), 6)
                item[f"{field_name}_max"] = round(max(values), 6)
        summary[f"{scale:.3f}"] = item
    return summary


def compare_reward_scales(
    rows: list[dict[str, Any]], pre_specified_alpha: float = 0.185
) -> dict[str, Any]:
    """Compare pre-specified observational alpha with heuristic grid scales.

    Lower slope_change_pct is better because negative values indicate slope
    reduction. The returned rank is one-indexed.
    """
    by_scale = summarize_by_scale(rows)
    if not by_scale:
        raise ValueError("No reward-scale rows were provided")

    scale_rows = list(by_scale.values())
    ranked = sorted(
        scale_rows,
        key=lambda item: (float(item["slope_change_pct_mean"]), float(item["reward_scale"])),
    )
    best = ranked[0]
    pre = min(
        scale_rows,
        key=lambda item: abs(float(item["reward_scale"]) - float(pre_specified_alpha)),
    )
    unscaled = min(scale_rows, key=lambda item: abs(float(item["reward_scale"]) - 1.0))

    best_slope = float(best["slope_change_pct_mean"])
    pre_slope = float(pre["slope_change_pct_mean"])
    unscaled_slope = float(unscaled["slope_change_pct_mean"])
    pre_rank = next(
        idx + 1
        for idx, item in enumerate(ranked)
        if float(item["reward_scale"]) == float(pre["reward_scale"])
    )

    return {
        "description": (
            "Comparison of the pre-specified observational reward calibration "
            "factor against ordinary heuristic reward-scaling grid values. "
            "Lower slope_change_pct is better."
        ),
        "n_runs": len(rows),
        "n_scales": len(scale_rows),
        "pre_specified_alpha": round(float(pre_specified_alpha), 6),
        "pre_specified_scale": round(float(pre["reward_scale"]), 6),
        "pre_specified_rank_by_slope": int(pre_rank),
        "best_scale": round(float(best["reward_scale"]), 6),
        "best_slope_change_pct_mean": round(best_slope, 6),
        "pre_specified_slope_change_pct_mean": round(pre_slope, 6),
        "unscaled_scale": round(float(unscaled["reward_scale"]), 6),
        "unscaled_slope_change_pct_mean": round(unscaled_slope, 6),
        "pre_vs_best_relative_gap_pct": _relative_abs_gap(pre_slope, best_slope),
        "pre_vs_unscaled_slope_gain_pct": _relative_abs_gain(pre_slope, unscaled_slope),
        "scale_summaries": by_scale,
    }


def build_report(
    grid_path: Path,
    output_path: Path | None = None,
    pre_specified_alpha: float = 0.185,
) -> dict[str, Any]:
    rows = json.loads(grid_path.read_text(encoding="utf-8"))
    report = compare_reward_scales(rows, pre_specified_alpha=pre_specified_alpha)
    report["grid_path"] = str(grid_path)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _relative_abs_gap(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return round(abs(abs(value) - abs(reference)) / abs(reference) * 100.0, 6)


def _relative_abs_gain(value: float, reference: float) -> float | None:
    if reference == 0:
        return None
    return round((abs(value) - abs(reference)) / abs(reference) * 100.0, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid",
        type=Path,
        default=Path("paper7/results/revision/alpha_grid/grid_results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/revision/reward_scaling_comparator.json"),
    )
    parser.add_argument("--pre-specified-alpha", type=float, default=0.185)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(
        grid_path=args.grid,
        output_path=args.output,
        pre_specified_alpha=args.pre_specified_alpha,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
