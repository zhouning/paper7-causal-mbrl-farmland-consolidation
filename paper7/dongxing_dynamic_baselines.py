"""Dynamic paired-swap baselines for Dongxing external validation."""

from __future__ import annotations

import argparse
import heapq
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.dongxing_data_audit import classify_land_use


DETERMINISTIC_STRATEGIES = ("dynamic_slope_gap", "dynamic_area_weighted_gap")


@dataclass
class DynamicSwapState:
    """Minimal county state for parcel-label paired slope swaps."""

    land_use: list[str]
    areas_m2: list[float]
    slopes: list[float]
    block_compositions: dict[str, list[int]]
    swapped: list[bool] = field(init=False)
    completed_pairs: int = field(default=0, init=False)
    pair_records: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.swapped = [False] * len(self.land_use)
        self.block_stats = {
            str(block_id): self._compute_block_stats(indices)
            for block_id, indices in self.block_compositions.items()
        }
        self.initial_farmland_area_m2 = self._farmland_area()
        self.initial_weighted_slope = self._farmland_weighted_slope()
        self.current_farmland_area_m2 = self.initial_farmland_area_m2
        self.current_weighted_slope = self.initial_weighted_slope
        self.initial_avg_farmland_slope = self.avg_farmland_slope

    @classmethod
    def from_records(
        cls,
        parcels: list[dict[str, Any]],
        block_compositions: dict[str, list[int]],
    ) -> "DynamicSwapState":
        land_use = [str(parcel["land_use"]) for parcel in parcels]
        areas_m2 = [float(parcel["area_m2"]) for parcel in parcels]
        slopes = [float(parcel["slope"]) for parcel in parcels]
        return cls(
            land_use=land_use,
            areas_m2=areas_m2,
            slopes=slopes,
            block_compositions={str(k): [int(i) for i in v] for k, v in block_compositions.items()},
        )

    @property
    def avg_farmland_slope(self) -> float:
        if self.current_farmland_area_m2 <= 0:
            return 0.0
        return self.current_weighted_slope / self.current_farmland_area_m2

    @property
    def slope_change_pct(self) -> float:
        denom = abs(self.initial_avg_farmland_slope) + 1e-12
        return 100.0 * (self.avg_farmland_slope - self.initial_avg_farmland_slope) / denom

    def land_use_label(self, swappable_index: int) -> str:
        return self.land_use[int(swappable_index)]

    def feasible_gain(self, block_id: int) -> float:
        stats = self.block_stats.get(str(block_id))
        if stats is None or not stats["farm"] or not stats["forest"]:
            return 0.0
        return self.slopes[stats["farm"][0]] - self.slopes[stats["forest"][0]]

    def feasible_exchange_area_m2(self, block_id: int) -> float:
        stats = self.block_stats.get(str(block_id))
        if stats is None:
            return 0.0
        return min(
            stats["farm_area_m2"],
            stats["forest_area_m2"],
        )

    def execute_block(self, block_id: int, max_pairs: int) -> int:
        completed = 0
        for _ in range(max_pairs):
            stats = self.block_stats.get(str(block_id))
            if stats is None or not stats["farm"] or not stats["forest"]:
                break

            best_farm = stats["farm"][0]
            best_forest = stats["forest"][0]
            slope_gap = self.slopes[best_farm] - self.slopes[best_forest]
            if slope_gap <= 0:
                break

            self.land_use[best_farm] = "forest"
            self.land_use[best_forest] = "farmland"
            self.swapped[best_farm] = True
            self.swapped[best_forest] = True
            self.current_weighted_slope += (
                self.slopes[best_forest] * self.areas_m2[best_forest]
                - self.slopes[best_farm] * self.areas_m2[best_farm]
            )
            self.current_farmland_area_m2 += self.areas_m2[best_forest] - self.areas_m2[best_farm]
            self._remove_available_pair(str(block_id), best_farm, best_forest)
            completed += 1
            self.completed_pairs += 1
            self.pair_records.append(
                {
                    "block_id": int(block_id),
                    "farmland_out": int(best_farm),
                    "forest_in": int(best_forest),
                    "farmland_out_slope": round(float(self.slopes[best_farm]), 6),
                    "forest_in_slope": round(float(self.slopes[best_forest]), 6),
                    "slope_gap": round(float(slope_gap), 6),
                    "farmland_out_area_m2": round(float(self.areas_m2[best_farm]), 2),
                    "forest_in_area_m2": round(float(self.areas_m2[best_forest]), 2),
                }
            )
        return completed

    def _compute_block_stats(self, indices: list[int]) -> dict[str, Any]:
        farm = [i for i in indices if self.land_use[i] == "farmland"]
        forest = [i for i in indices if self.land_use[i] == "forest"]
        farm.sort(key=lambda i: (self.slopes[i], self.areas_m2[i]), reverse=True)
        forest.sort(key=lambda i: (self.slopes[i], -self.areas_m2[i]))
        return {
            "farm": farm,
            "forest": forest,
            "farm_area_m2": sum(self.areas_m2[i] for i in farm),
            "forest_area_m2": sum(self.areas_m2[i] for i in forest),
        }

    def _remove_available_pair(self, block_key: str, farm_idx: int, forest_idx: int) -> None:
        stats = self.block_stats[block_key]
        stats["farm"].remove(farm_idx)
        stats["forest"].remove(forest_idx)
        stats["farm_area_m2"] -= self.areas_m2[farm_idx]
        stats["forest_area_m2"] -= self.areas_m2[forest_idx]

    def _available_indices(self, block_id: int) -> tuple[list[int], list[int]]:
        indices = self.block_compositions.get(str(block_id), [])
        farm_idx = [
            i for i in indices
            if not self.swapped[i] and self.land_use[i] == "farmland"
        ]
        forest_idx = [
            i for i in indices
            if not self.swapped[i] and self.land_use[i] == "forest"
        ]
        return farm_idx, forest_idx

    def _farmland_area(self) -> float:
        return sum(area for area, land_use in zip(self.areas_m2, self.land_use) if land_use == "farmland")

    def _farmland_weighted_slope(self) -> float:
        return sum(
            area * slope
            for area, slope, land_use in zip(self.areas_m2, self.slopes, self.land_use)
            if land_use == "farmland"
        )


def load_swappable_parcels(
    dltb_path: Path,
    mapping_path: Path,
    slope_field: str = "slope_mean",
) -> list[dict[str, Any]]:
    """Load swappable Dongxing parcels in the same order as parcel_block_mapping.csv."""
    frame = gpd.read_file(dltb_path, engine="pyogrio")
    frame["paper7_land_use"] = frame.apply(
        lambda row: classify_land_use(row.get("DLBM"), row.get("DLMC")), axis=1
    )
    mapping = pd.read_csv(mapping_path)
    parcels = []
    for _, row in mapping.sort_values("swappable_index").iterrows():
        source_index = int(row["source_index"])
        record = frame.iloc[source_index]
        slope = pd.to_numeric(pd.Series([record.get(slope_field)]), errors="coerce").iloc[0]
        if pd.isna(slope):
            raise ValueError(f"Missing {slope_field} for source_index={source_index}")
        parcels.append(
            {
                "swappable_index": int(row["swappable_index"]),
                "source_index": source_index,
                "land_use": str(row["land_use"]),
                "area_m2": float(pd.to_numeric(pd.Series([record.get("TBMJ")]), errors="coerce").fillna(0.0).iloc[0]),
                "slope": float(slope),
                "block_id": int(row["block_id"]),
            }
        )
    return parcels


def run_episode(
    parcels: list[dict[str, Any]],
    block_compositions: dict[str, list[int]],
    block_features: list[dict[str, Any]],
    strategy: str,
    max_steps: int,
    swaps_per_step: int,
    seed: int | None = None,
) -> dict[str, Any]:
    state = DynamicSwapState.from_records(parcels, block_compositions)
    rng = random.Random(seed)
    block_ids = [int(block["block_id"]) for block in block_features]
    selected_blocks: list[int] = []
    selector = _EpisodeSelector(state, block_ids, strategy, rng)

    for _ in range(max_steps):
        block_id = selector.select()
        if block_id is None:
            break
        completed = state.execute_block(block_id, max_pairs=swaps_per_step)
        if completed <= 0:
            break
        selector.update(block_id)
        selected_blocks.append(int(block_id))

    gaps = [record["slope_gap"] for record in state.pair_records]
    return {
        "strategy": strategy,
        "seed": seed,
        "steps": len(selected_blocks),
        "selected_blocks": selected_blocks,
        "completed_pairs": int(state.completed_pairs),
        "initial_avg_farmland_slope": round(float(state.initial_avg_farmland_slope), 6),
        "final_avg_farmland_slope": round(float(state.avg_farmland_slope), 6),
        "slope_change_pct": round(float(state.slope_change_pct), 6),
        "mean_pair_slope_gap": round(sum(gaps) / len(gaps), 6) if gaps else None,
        "total_pair_slope_gap": round(sum(gaps), 6),
        "unique_blocks": len(set(selected_blocks)),
        "pair_records_head": state.pair_records[:20],
    }


def run_baseline_suite(
    parcels: list[dict[str, Any]],
    block_compositions: dict[str, list[int]],
    block_features: list[dict[str, Any]],
    max_steps: int = 100,
    swaps_per_step: int = 5,
    random_seeds: list[int] | None = None,
) -> dict[str, Any]:
    strategies = {
        strategy: run_episode(
            parcels=parcels,
            block_compositions=block_compositions,
            block_features=block_features,
            strategy=strategy,
            max_steps=max_steps,
            swaps_per_step=swaps_per_step,
        )
        for strategy in DETERMINISTIC_STRATEGIES
    }

    if random_seeds is None:
        random_seeds = list(range(100))
    random_runs = [
        run_episode(
            parcels=parcels,
            block_compositions=block_compositions,
            block_features=block_features,
            strategy="random",
            max_steps=max_steps,
            swaps_per_step=swaps_per_step,
            seed=int(seed),
        )
        for seed in random_seeds
    ]
    random_summary = _summarize_runs(random_runs)
    for result in strategies.values():
        _attach_random_p_values(result, random_runs)

    return {
        "description": (
            "Dongxing dynamic paired-swap baselines. This is a deterministic/non-RL "
            "external-county environment check, not learned-policy transfer."
        ),
        "max_steps": int(max_steps),
        "swaps_per_step": int(swaps_per_step),
        "n_parcels": int(len(parcels)),
        "n_blocks": int(len(block_features)),
        "strategies": strategies,
        "random_baseline": random_summary,
    }


class _EpisodeSelector:
    def __init__(
        self,
        state: DynamicSwapState,
        block_ids: list[int],
        strategy: str,
        rng: random.Random,
    ) -> None:
        self.state = state
        self.strategy = strategy
        self.rng = rng
        self.versions = {int(block_id): 0 for block_id in block_ids}
        self.heap: list[tuple[float, float, int, int]] = []
        self.random_feasible: list[int] = []
        self.random_positions: dict[int, int] = {}
        if strategy in DETERMINISTIC_STRATEGIES:
            for block_id in block_ids:
                self._push_heap(block_id)
        elif strategy == "random":
            for block_id in block_ids:
                if self.state.feasible_gain(block_id) > 0:
                    self._add_random(block_id)
        else:
            raise ValueError(f"Unknown dynamic strategy {strategy!r}")

    def select(self) -> int | None:
        if self.strategy in DETERMINISTIC_STRATEGIES:
            while self.heap:
                _, _, neg_block_id, version = self.heap[0]
                block_id = -neg_block_id
                if version != self.versions[block_id] or self.state.feasible_gain(block_id) <= 0:
                    heapq.heappop(self.heap)
                    continue
                return block_id
            return None
        if not self.random_feasible:
            return None
        return self.rng.choice(self.random_feasible)

    def update(self, block_id: int) -> None:
        self.versions[block_id] += 1
        if self.strategy in DETERMINISTIC_STRATEGIES:
            self._push_heap(block_id)
        elif self.state.feasible_gain(block_id) <= 0:
            self._remove_random(block_id)

    def _push_heap(self, block_id: int) -> None:
        gain = self.state.feasible_gain(block_id)
        if gain <= 0:
            return
        area = self.state.feasible_exchange_area_m2(block_id)
        if self.strategy == "dynamic_slope_gap":
            primary = gain
            secondary = area
        else:
            primary = gain * area
            secondary = gain
        heapq.heappush(self.heap, (-primary, -secondary, -int(block_id), self.versions[int(block_id)]))

    def _add_random(self, block_id: int) -> None:
        block_id = int(block_id)
        self.random_positions[block_id] = len(self.random_feasible)
        self.random_feasible.append(block_id)

    def _remove_random(self, block_id: int) -> None:
        block_id = int(block_id)
        pos = self.random_positions.pop(block_id, None)
        if pos is None:
            return
        last = self.random_feasible.pop()
        if pos < len(self.random_feasible):
            self.random_feasible[pos] = last
            self.random_positions[last] = pos


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_seeds": len(runs), "per_seed": [_compact_run(run) for run in runs]}
    for field_name in (
        "slope_change_pct",
        "completed_pairs",
        "mean_pair_slope_gap",
        "total_pair_slope_gap",
        "unique_blocks",
    ):
        values = [
            float(run[field_name])
            for run in runs
            if run.get(field_name) is not None
        ]
        stats = _distribution_stats(values)
        for key, value in stats.items():
            summary[f"{field_name}_{key}"] = value
    return summary


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "strategy",
        "seed",
        "steps",
        "completed_pairs",
        "initial_avg_farmland_slope",
        "final_avg_farmland_slope",
        "slope_change_pct",
        "mean_pair_slope_gap",
        "total_pair_slope_gap",
        "unique_blocks",
    )
    return {field_name: run.get(field_name) for field_name in fields}


def _attach_random_p_values(result: dict[str, Any], random_runs: list[dict[str, Any]]) -> None:
    # Lower slope_change_pct is better. Higher pair counts/gaps are better.
    comparisons = {
        "slope_change_pct": "lower",
        "completed_pairs": "higher",
        "mean_pair_slope_gap": "higher",
        "total_pair_slope_gap": "higher",
    }
    for field_name, direction in comparisons.items():
        observed = result.get(field_name)
        values = [float(run[field_name]) for run in random_runs if run.get(field_name) is not None]
        if observed is None or not values:
            result[f"random_p_{field_name}"] = None
            continue
        if direction == "lower":
            exceedances = sum(1 for value in values if value <= float(observed))
        else:
            exceedances = sum(1 for value in values if value >= float(observed))
        result[f"random_p_{field_name}"] = round((exceedances + 1) / (len(values) + 1), 6)


def _distribution_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sd": None, "q05": None, "q50": None, "q95": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": round(mean, 6),
        "sd": round(variance**0.5, 6),
        "q05": round(_quantile(values, 0.05), 6),
        "q50": round(_quantile(values, 0.50), 6),
        "q95": round(_quantile(values, 0.95), 6),
    }


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--random-seeds", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/dongxing_dynamic_baselines.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_features = json.loads((args.block_dir / "block_features.json").read_text(encoding="utf-8"))
    block_compositions = json.loads((args.block_dir / "block_compositions.json").read_text(encoding="utf-8"))
    parcels = load_swappable_parcels(args.dltb, args.block_dir / "parcel_block_mapping.csv")
    summary = run_baseline_suite(
        parcels=parcels,
        block_compositions=block_compositions,
        block_features=block_features,
        max_steps=args.max_steps,
        swaps_per_step=args.swaps_per_step,
        random_seeds=list(range(args.random_seeds)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
