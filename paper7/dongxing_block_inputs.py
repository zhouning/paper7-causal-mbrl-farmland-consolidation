"""Prepare Dongxing District block inputs for external Paper 7 validation."""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import unary_union

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.dongxing_data_audit import audit_geodataframe, classify_land_use, find_dltb_shapefile


def prepare_block_inputs(
    frame: gpd.GeoDataFrame,
    source_path: Path,
    unit_field: str = "ZLDWDM",
    max_parcels: int = 30,
    min_parcels: int = 3,
    min_area_ha: float = 0.5,
) -> dict[str, Any]:
    """Build swappable parcel blocks grouped by administrative/ownership unit."""
    audit = audit_geodataframe(frame, source_path=source_path)
    prepared = _prepare_swappable_frame(frame)

    swappable = prepared[prepared["paper7_land_use"].isin(["farmland", "forest"])].copy()
    swappable = swappable.reset_index(drop=False).rename(columns={"index": "source_index"})
    if unit_field not in swappable.columns:
        raise ValueError(f"Unit field {unit_field!r} not found in Dongxing DLTB data")

    block_features: list[dict[str, Any]] = []
    block_compositions: dict[str, list[int]] = {}
    parcel_mapping = [
        {
            "swappable_index": int(i),
            "source_index": int(row["source_index"]),
            "unit_id": _clean_scalar(row.get(unit_field)),
            "land_use": str(row["paper7_land_use"]),
            "block_id": -1,
        }
        for i, row in swappable.iterrows()
    ]

    projected = swappable
    block_id = 0
    units_processed = 0
    units_with_blocks = set()

    for unit_id, unit_frame in projected.groupby(unit_field, dropna=True, sort=True):
        unit_frame = unit_frame.reset_index()
        units_processed += 1
        adjacency = _build_intersection_adjacency(unit_frame)
        components = _connected_components(adjacency, len(unit_frame))
        for component in components:
            for sub_component in _split_component(component, unit_frame, max_parcels=max_parcels):
                feature = _compute_block_feature(
                    unit_frame=unit_frame,
                    local_indices=sub_component,
                    block_id=block_id,
                    unit_id=unit_id,
                    min_parcels=min_parcels,
                    min_area_ha=min_area_ha,
                )
                if feature is None:
                    continue
                global_indices = [int(unit_frame.iloc[i]["index"]) for i in sub_component]
                block_compositions[str(block_id)] = global_indices
                for global_idx in global_indices:
                    parcel_mapping[global_idx]["block_id"] = block_id
                block_features.append(feature)
                units_with_blocks.add(str(unit_id))
                block_id += 1

    summary = _build_summary(
        audit=audit,
        swappable=swappable,
        block_features=block_features,
        n_units_processed=units_processed,
        n_units_with_blocks=len(units_with_blocks),
        unit_field=unit_field,
        max_parcels=max_parcels,
        min_parcels=min_parcels,
        min_area_ha=min_area_ha,
    )

    return {
        "source": audit["source"],
        "summary": summary,
        "block_features": block_features,
        "block_compositions": block_compositions,
        "parcel_mapping": parcel_mapping,
    }


def write_block_package(package: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(package["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "block_features.json").write_text(
        json.dumps(package["block_features"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "block_compositions.json").write_text(
        json.dumps(package["block_compositions"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(package["parcel_mapping"]).to_csv(output_dir / "parcel_block_mapping.csv", index=False)


def _prepare_swappable_frame(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    prepared = frame.copy()
    prepared["paper7_land_use"] = prepared.apply(
        lambda row: classify_land_use(row.get("DLBM"), row.get("DLMC")), axis=1
    )
    prepared["paper7_area_m2"] = pd.to_numeric(prepared.get("TBMJ"), errors="coerce").fillna(0.0)
    if "GDPDJ" in prepared.columns:
        prepared["paper7_slope_grade"] = pd.to_numeric(prepared["GDPDJ"], errors="coerce")
    else:
        prepared["paper7_slope_grade"] = np.nan
    if "slope_mean" in prepared.columns:
        prepared["paper7_slope"] = pd.to_numeric(prepared["slope_mean"], errors="coerce")
    else:
        prepared["paper7_slope"] = np.nan
    return prepared


def _build_intersection_adjacency(frame: gpd.GeoDataFrame) -> list[list[int]]:
    geometries = list(frame.geometry)
    spatial_index = frame.sindex
    adjacency = [set() for _ in geometries]
    for i, geometry in enumerate(geometries):
        if geometry is None or geometry.is_empty:
            continue
        candidates = spatial_index.query(geometry, predicate="intersects")
        for j in candidates:
            j = int(j)
            if i == j:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
    return [sorted(neighbors) for neighbors in adjacency]


def _connected_components(adjacency: list[list[int]], n_nodes: int) -> list[list[int]]:
    seen = [False] * n_nodes
    components = []
    for start in range(n_nodes):
        if seen[start]:
            continue
        queue = deque([start])
        seen[start] = True
        component = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
        components.append(component)
    return components


def _split_component(component: list[int], frame: gpd.GeoDataFrame, max_parcels: int) -> list[list[int]]:
    if len(component) <= max_parcels:
        return [component]
    centroids = frame.iloc[component].geometry.centroid
    order = np.lexsort((centroids.y.to_numpy(), centroids.x.to_numpy()))
    ordered = [component[int(i)] for i in order]
    return [ordered[i : i + max_parcels] for i in range(0, len(ordered), max_parcels)]


def _compute_block_feature(
    unit_frame: gpd.GeoDataFrame,
    local_indices: list[int],
    block_id: int,
    unit_id: object,
    min_parcels: int,
    min_area_ha: float,
) -> dict[str, Any] | None:
    parcels = unit_frame.iloc[local_indices]
    farmland = parcels["paper7_land_use"] == "farmland"
    forest = parcels["paper7_land_use"] == "forest"
    if len(parcels) < min_parcels or not farmland.any() or not forest.any():
        return None

    total_area_m2 = float(parcels["paper7_area_m2"].sum())
    if total_area_m2 / 10000.0 < min_area_ha:
        return None

    return {
        "block_id": int(block_id),
        "unit_id": _clean_scalar(unit_id),
        "n_parcels": int(len(parcels)),
        "n_farmland": int(farmland.sum()),
        "n_forest": int(forest.sum()),
        "total_area_m2": round(total_area_m2, 2),
        "total_area_ha": round(total_area_m2 / 10000.0, 6),
        "farm_area_ha": round(float(parcels.loc[farmland, "paper7_area_m2"].sum()) / 10000.0, 6),
        "forest_area_ha": round(float(parcels.loc[forest, "paper7_area_m2"].sum()) / 10000.0, 6),
        "avg_farm_slope": _weighted_average_or_none(parcels.loc[farmland], "paper7_slope"),
        "avg_forest_slope": _weighted_average_or_none(parcels.loc[forest], "paper7_slope"),
        "avg_farm_slope_grade": _weighted_average_or_none(parcels.loc[farmland], "paper7_slope_grade"),
        "avg_forest_slope_grade": _weighted_average_or_none(parcels.loc[forest], "paper7_slope_grade"),
        "compactness": _compactness(parcels),
    }


def _weighted_average_or_none(parcels: gpd.GeoDataFrame, value_field: str) -> float | None:
    valid = parcels.dropna(subset=[value_field])
    if valid.empty:
        return None
    weights = valid["paper7_area_m2"].to_numpy(dtype=float)
    values = valid[value_field].to_numpy(dtype=float)
    if float(weights.sum()) <= 0:
        return round(float(values.mean()), 6)
    return round(float(np.average(values, weights=weights)), 6)


def _compactness(parcels: gpd.GeoDataFrame) -> float:
    geometry = unary_union(parcels.geometry.values)
    if geometry.is_empty or geometry.length <= 0:
        return 0.0
    return round(float(4.0 * np.pi * geometry.area / (geometry.length**2)), 8)


def _build_summary(
    audit: dict[str, Any],
    swappable: gpd.GeoDataFrame,
    block_features: list[dict[str, Any]],
    n_units_processed: int,
    n_units_with_blocks: int,
    unit_field: str,
    max_parcels: int,
    min_parcels: int,
    min_area_ha: float,
) -> dict[str, Any]:
    slopes = audit["slope"]["by_land_use"]
    has_complete_slope = slopes["farmland"]["non_null_count"] > 0 and slopes["forest"]["non_null_count"] > 0
    slope_gap = None if has_complete_slope else "forest slope values are missing"
    return {
        "unit_field": unit_field,
        "n_swappable_parcels": int(len(swappable)),
        "n_blocks": int(len(block_features)),
        "n_units_processed": int(n_units_processed),
        "n_units_with_blocks": int(n_units_with_blocks),
        "min_parcels": int(min_parcels),
        "max_parcels": int(max_parcels),
        "min_area_ha": float(min_area_ha),
        "has_complete_slope_for_swap": bool(has_complete_slope),
        "slope_gap": slope_gap,
        "slope_mode": audit["readiness"]["slope_mode"],
        "readiness": audit["readiness"],
    }


def _clean_scalar(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Root directory containing Dongxing shapefiles.")
    parser.add_argument("--dltb", type=Path, help="Path to DLTB.shp. Overrides --root search.")
    parser.add_argument("--unit-field", default="ZLDWDM")
    parser.add_argument("--max-parcels", type=int, default=30)
    parser.add_argument("--min-parcels", type=int, default=3)
    parser.add_argument("--min-area-ha", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("paper7/results/dongxing_blocks"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dltb_path = args.dltb if args.dltb is not None else find_dltb_shapefile(args.root)
    frame = gpd.read_file(dltb_path, engine="pyogrio")
    package = prepare_block_inputs(
        frame,
        source_path=dltb_path,
        unit_field=args.unit_field,
        max_parcels=args.max_parcels,
        min_parcels=args.min_parcels,
        min_area_ha=args.min_area_ha,
    )
    write_block_package(package, args.output_dir)
    print(json.dumps(package["summary"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
