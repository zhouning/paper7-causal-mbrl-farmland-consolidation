"""Build a full multi-objective Dongxing environment for Paper 7."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from paper7.dongxing_data_audit import classify_land_use
from paper7.generic_county_env import GenericCountyEnv


def load_dongxing_parcels_for_full_env(
    dltb_path: Path,
    block_dir: Path,
    slope_field: str = "slope_mean",
) -> list[dict[str, Any]]:
    frame = gpd.read_file(dltb_path, engine="pyogrio")
    return _parcels_from_frame_and_mapping(frame, block_dir, slope_field=slope_field)


def build_env_from_frame_and_block_package(
    frame: gpd.GeoDataFrame,
    block_dir: Path,
    total_budget: int = 500,
    swaps_per_step: int = 5,
    slope_field: str = "slope_mean",
) -> GenericCountyEnv:
    parcels = _parcels_from_frame_and_mapping(frame, block_dir, slope_field=slope_field)
    block_compositions = json.loads((block_dir / "block_compositions.json").read_text(encoding="utf-8"))
    block_features = json.loads((block_dir / "block_features.json").read_text(encoding="utf-8"))
    block_ids = [int(item["block_id"]) for item in block_features]
    return GenericCountyEnv(
        parcels=parcels,
        block_compositions=block_compositions,
        block_ids=block_ids,
        total_budget=total_budget,
        swaps_per_step=swaps_per_step,
    )


def build_dongxing_full_env(
    dltb_path: Path,
    block_dir: Path,
    total_budget: int = 500,
    swaps_per_step: int = 5,
    slope_field: str = "slope_mean",
) -> GenericCountyEnv:
    frame = gpd.read_file(dltb_path, engine="pyogrio")
    return build_env_from_frame_and_block_package(
        frame,
        block_dir,
        total_budget=total_budget,
        swaps_per_step=swaps_per_step,
        slope_field=slope_field,
    )


def summarize_env(env: GenericCountyEnv) -> dict[str, Any]:
    _, info = env.reset(seed=0)
    return {
        "status": "constructed",
        "n_parcels": int(env.n_parcels),
        "n_blocks": int(env.n_blocks),
        "valid_action_count": int(env.action_masks().sum()),
        "initial_avg_farmland_slope": info["avg_slope"],
        "initial_contiguity": info["contiguity"],
        "initial_baimu_count": info["baimu_count"],
        "initial_baimu_area_ha": info["baimu_area_ha"],
        "total_budget": int(env.total_budget),
        "swaps_per_step": int(env.swaps_per_step),
        "max_steps": int(env.max_steps),
    }


def _parcels_from_frame_and_mapping(
    frame: gpd.GeoDataFrame,
    block_dir: Path,
    slope_field: str,
) -> list[dict[str, Any]]:
    mapping = pd.read_csv(block_dir / "parcel_block_mapping.csv")
    mapping = mapping.sort_values("swappable_index")
    parcels: list[dict[str, Any]] = []
    for expected_index, (_, row) in enumerate(mapping.iterrows()):
        swappable_index = int(row["swappable_index"])
        if swappable_index != expected_index:
            raise ValueError(
                f"Expected contiguous swappable_index {expected_index}, got {swappable_index}"
            )
        source_index = int(row["source_index"])
        record = frame.iloc[source_index]
        slope = pd.to_numeric(pd.Series([record.get(slope_field)]), errors="coerce").iloc[0]
        area = pd.to_numeric(pd.Series([record.get("TBMJ")]), errors="coerce").iloc[0]
        if pd.isna(slope):
            raise ValueError(f"Missing {slope_field} for source_index={source_index}")
        if pd.isna(area):
            raise ValueError(f"Missing TBMJ for source_index={source_index}")
        land_use = str(row.get("land_use") or "").strip().lower()
        if land_use not in {"farmland", "forest"}:
            land_use = classify_land_use(record.get("DLBM"), record.get("DLMC"))
        parcels.append(
            {
                "swappable_index": swappable_index,
                "source_index": source_index,
                "land_use": land_use,
                "area_m2": float(area),
                "slope": float(slope),
                "geometry": record.geometry,
                "block_id": int(row["block_id"]),
            }
        )
    return parcels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--slope-field", default="slope_mean")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("paper7/results/full_rigor/dongxing_full_env_smoke.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = build_dongxing_full_env(
        dltb_path=args.dltb,
        block_dir=args.block_dir,
        total_budget=args.total_budget,
        swaps_per_step=args.swaps_per_step,
        slope_field=args.slope_field,
    )
    summary = summarize_env(env)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
