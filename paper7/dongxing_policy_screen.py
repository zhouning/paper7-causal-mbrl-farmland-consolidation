"""Screen Dongxing candidate blocks with transparent non-RL baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SCREEN_STRATEGIES = ("slope_gap", "area_weighted_gap")


def rank_blocks(blocks: list[dict[str, Any]], strategy: str) -> list[dict[str, Any]]:
    """Return complete-slope blocks ranked by a transparent opportunity score."""
    if strategy not in SCREEN_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {SCREEN_STRATEGIES}")

    ranked = []
    for block in blocks:
        farm_slope = block.get("avg_farm_slope")
        forest_slope = block.get("avg_forest_slope")
        if farm_slope is None or forest_slope is None:
            continue
        slope_gap = float(farm_slope) - float(forest_slope)
        exchange_area_ha = min(float(block.get("farm_area_ha", 0.0)), float(block.get("forest_area_ha", 0.0)))
        enriched = dict(block)
        enriched["slope_gap"] = round(slope_gap, 6)
        enriched["exchange_area_ha"] = round(exchange_area_ha, 6)
        enriched["opportunity_score"] = round(slope_gap * exchange_area_ha, 6)
        ranked.append(enriched)

    score_field = "slope_gap" if strategy == "slope_gap" else "opportunity_score"
    return sorted(
        ranked,
        key=lambda block: (
            block[score_field],
            block["exchange_area_ha"],
            block.get("total_area_ha", 0.0),
        ),
        reverse=True,
    )


def summarize_policy_screen(
    blocks: list[dict[str, Any]],
    top_k: int = 100,
    random_seeds: list[int] | None = None,
) -> dict[str, Any]:
    """Summarize deterministic and random block-selection screens."""
    complete_blocks = rank_blocks(blocks, strategy="slope_gap")
    selected_k = min(int(top_k), len(complete_blocks))

    strategies = {}
    for strategy in SCREEN_STRATEGIES:
        ranked = rank_blocks(blocks, strategy=strategy)
        selected = ranked[:selected_k]
        strategies[strategy] = _summarize_selection(selected)
        strategies[strategy]["top_blocks"] = _top_block_records(selected, limit=min(20, selected_k))

    if random_seeds is None:
        random_seeds = list(range(20))
    random_results = []
    for seed in random_seeds:
        rng = random.Random(int(seed))
        sample = complete_blocks.copy()
        rng.shuffle(sample)
        random_results.append({"seed": int(seed), **_summarize_selection(sample[:selected_k])})

    random_baseline = _summarize_random(random_results)
    for strategy_summary in strategies.values():
        _attach_random_comparison(strategy_summary, random_baseline)

    return {
        "description": (
            "Static Dongxing block-level opportunity screen. This is not a dynamic RL "
            "environment evaluation; it ranks candidate blocks using initial block slopes."
        ),
        "top_k": selected_k,
        "n_candidate_blocks": int(len(blocks)),
        "n_complete_slope_blocks": int(len(complete_blocks)),
        "strategies": strategies,
        "random_baseline": random_baseline,
    }


def _summarize_selection(selected: list[dict[str, Any]]) -> dict[str, Any]:
    if not selected:
        return {
            "selected_blocks": 0,
            "mean_slope_gap": None,
            "positive_gap_blocks": 0,
            "positive_gap_share": None,
            "exchange_area_ha": 0.0,
            "opportunity_score_sum": 0.0,
            "unique_units": 0,
        }

    gaps = [float(block["slope_gap"]) for block in selected]
    positive = [gap for gap in gaps if gap > 0]
    return {
        "selected_blocks": int(len(selected)),
        "mean_slope_gap": round(sum(gaps) / len(gaps), 6),
        "median_slope_gap": round(_median(gaps), 6),
        "min_slope_gap": round(min(gaps), 6),
        "max_slope_gap": round(max(gaps), 6),
        "positive_gap_blocks": int(len(positive)),
        "positive_gap_share": round(len(positive) / len(selected), 6),
        "exchange_area_ha": round(sum(float(block["exchange_area_ha"]) for block in selected), 6),
        "opportunity_score_sum": round(sum(float(block["opportunity_score"]) for block in selected), 6),
        "unique_units": int(len({block.get("unit_id") for block in selected if block.get("unit_id") is not None})),
    }


def _summarize_random(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("mean_slope_gap", "positive_gap_share", "exchange_area_ha", "opportunity_score_sum", "unique_units")
    summary: dict[str, Any] = {
        "n_seeds": len(per_seed),
        "per_seed": per_seed,
    }
    for field in fields:
        values = [float(row[field]) for row in per_seed if row[field] is not None]
        stats = _distribution_stats(values)
        for key, value in stats.items():
            summary[f"{field}_{key}"] = value
    return summary


def _distribution_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "sd": None,
            "q05": None,
            "q50": None,
            "q95": None,
        }

    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": round(mean, 6),
        "sd": round(variance ** 0.5, 6),
        "q05": round(_quantile(values, 0.05), 6),
        "q50": round(_quantile(values, 0.50), 6),
        "q95": round(_quantile(values, 0.95), 6),
    }


def _attach_random_comparison(strategy_summary: dict[str, Any], random_baseline: dict[str, Any]) -> None:
    per_seed = random_baseline.get("per_seed", [])
    for field in ("mean_slope_gap", "positive_gap_share", "opportunity_score_sum"):
        observed = strategy_summary.get(field)
        values = [float(row[field]) for row in per_seed if row.get(field) is not None]
        if observed is None or not values:
            strategy_summary[f"random_p_{field}"] = None
            continue
        # One-sided empirical p-value: probability that a random top-k selection
        # matches or exceeds the deterministic screen on the same metric.
        exceedances = sum(1 for value in values if value >= float(observed))
        strategy_summary[f"random_p_{field}"] = round((exceedances + 1) / (len(values) + 1), 6)


def _top_block_records(selected: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    fields = (
        "block_id",
        "unit_id",
        "n_parcels",
        "n_farmland",
        "n_forest",
        "total_area_ha",
        "farm_area_ha",
        "forest_area_ha",
        "avg_farm_slope",
        "avg_forest_slope",
        "slope_gap",
        "exchange_area_ha",
        "opportunity_score",
        "compactness",
    )
    return [
        {field: block.get(field) for field in fields}
        for block in selected[:limit]
    ]


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty sequence")
    if not 0 <= q <= 1:
        raise ValueError(f"Quantile must be in [0, 1], got {q}")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--block-features",
        type=Path,
        default=Path("paper7/results/dongxing_blocks_slope/block_features.json"),
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--random-seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/dongxing_policy_screen.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    blocks = json.loads(args.block_features.read_text(encoding="utf-8"))
    summary = summarize_policy_screen(
        blocks,
        top_k=args.top_k,
        random_seeds=list(range(args.random_seeds)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
