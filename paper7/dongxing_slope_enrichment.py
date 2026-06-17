"""DEM-based slope enrichment helpers for Dongxing external validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.transform import rowcol
from rasterio.features import geometry_mask, rasterize

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.dongxing_data_audit import find_dltb_shapefile


def bounds_cover(
    raster_bounds: tuple[float, float, float, float],
    vector_bounds: tuple[float, float, float, float],
    tolerance: float = 0.0,
) -> bool:
    """Return True when raster bounds fully cover vector bounds."""
    r_left, r_bottom, r_right, r_top = raster_bounds
    v_left, v_bottom, v_right, v_top = vector_bounds
    return (
        r_left <= v_left + tolerance
        and r_bottom <= v_bottom + tolerance
        and r_right >= v_right - tolerance
        and r_top >= v_top - tolerance
    )


def compute_slope_degrees(
    elevation: np.ndarray,
    x_resolution: float,
    y_resolution: float,
) -> np.ndarray:
    """Compute slope in degrees from a DEM array and cell resolution."""
    dem = elevation.astype(float)
    grad_y, grad_x = np.gradient(dem, abs(y_resolution), abs(x_resolution))
    slope_radians = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
    return np.degrees(slope_radians)


def transformed_bounds(
    bounds: Iterable[float],
    source_crs: object,
    target_crs: object,
) -> tuple[float, float, float, float]:
    """Transform rectangle corner bounds between CRSs."""
    left, bottom, right, top = [float(value) for value in bounds]
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    points = [
        transformer.transform(left, bottom),
        transformer.transform(left, top),
        transformer.transform(right, bottom),
        transformer.transform(right, top),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def audit_dem_coverage(dem_path: Path, vector_path: Path) -> dict[str, object]:
    """Check whether a DEM covers the Dongxing parcel layer."""
    parcels = gpd.read_file(vector_path, rows=1, engine="pyogrio")
    full_bounds = _read_vector_bounds(vector_path)
    with rasterio.open(dem_path) as src:
        vector_bounds_in_dem_crs = transformed_bounds(full_bounds, parcels.crs, src.crs)
        raster_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        return {
            "dem_path": str(dem_path),
            "vector_path": str(vector_path),
            "dem_crs": str(src.crs),
            "vector_crs": str(parcels.crs),
            "dem_bounds": [round(float(value), 8) for value in raster_bounds],
            "vector_bounds_in_dem_crs": [round(float(value), 8) for value in vector_bounds_in_dem_crs],
            "covers_vector": bounds_cover(raster_bounds, vector_bounds_in_dem_crs),
            "dem_shape": [int(src.height), int(src.width)],
            "dem_resolution": [float(src.res[0]), float(src.res[1])],
        }


def console_json(data: object) -> str:
    """Return ASCII-safe JSON for Windows console output."""
    return json.dumps(data, ensure_ascii=True, indent=2)


def enrich_slope_mean(
    parcels: gpd.GeoDataFrame,
    dem_path: Path,
    output_path: Path,
    slope_field: str = "slope_mean",
    all_touched: bool = False,
) -> None:
    """Write a GeoPackage with parcel-level mean slope values from a covering DEM."""
    with rasterio.open(dem_path) as src:
        dem_bounds = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
        vector_bounds = transformed_bounds(parcels.total_bounds, parcels.crs, src.crs)
        if not bounds_cover(dem_bounds, vector_bounds):
            raise ValueError("DEM does not fully cover the parcel layer bounds")

        dem = src.read(1, masked=True).astype(float)
        slope = compute_slope_degrees(dem.filled(np.nan), src.res[0], src.res[1])
        parcels_in_dem_crs = parcels.to_crs(src.crs)
        parcels[slope_field] = mean_raster_values_by_geometry(
            slope,
            src.transform,
            parcels_in_dem_crs.geometry,
            all_touched=all_touched,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    parcels.to_file(output_path, driver="GPKG")


def mean_raster_values_by_geometry(
    raster: np.ndarray,
    transform: object,
    geometries: Iterable[object],
    all_touched: bool = False,
) -> list[float | None]:
    """Compute per-geometry raster means via one rasterization pass."""
    geometry_list = list(geometries)
    shapes = [
        (geometry, index + 1)
        for index, geometry in enumerate(geometry_list)
        if geometry is not None and not geometry.is_empty
    ]
    if not shapes:
        return [None for _ in geometry_list]

    labels = rasterize(
        shapes,
        out_shape=raster.shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=all_touched,
    )
    valid = (labels > 0) & np.isfinite(raster)
    if not np.any(valid):
        return [
            _sample_raster_at_representative_point(raster, transform, geometry)
            for geometry in geometry_list
        ]

    label_values = labels[valid]
    raster_values = raster[valid].astype(float, copy=False)
    sums = np.bincount(label_values, weights=raster_values, minlength=len(geometry_list) + 1)
    counts = np.bincount(label_values, minlength=len(geometry_list) + 1)

    means: list[float | None] = []
    for label in range(1, len(geometry_list) + 1):
        if counts[label] == 0:
            means.append(_sample_raster_at_representative_point(raster, transform, geometry_list[label - 1]))
        else:
            means.append(float(sums[label] / counts[label]))
    return means


def _sample_raster_at_representative_point(
    raster: np.ndarray,
    transform: object,
    geometry: object,
) -> float | None:
    if geometry is None or geometry.is_empty:
        return None
    point = geometry.representative_point()
    row, col = rowcol(transform, point.x, point.y)
    if row < 0 or col < 0 or row >= raster.shape[0] or col >= raster.shape[1]:
        return None
    value = float(raster[row, col])
    if not np.isfinite(value):
        return None
    return value


def _read_vector_bounds(vector_path: Path) -> tuple[float, float, float, float]:
    frame = gpd.read_file(vector_path, engine="pyogrio")
    return tuple(float(value) for value in frame.total_bounds)


def _mean_raster_value_for_geometry(
    raster: np.ndarray,
    transform: object,
    geometry: object,
) -> float | None:
    if geometry is None or geometry.is_empty:
        return None
    mask = geometry_mask([geometry], out_shape=raster.shape, transform=transform, invert=True)
    values = raster[mask]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(values.mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, required=True)
    parser.add_argument("--root", type=Path, help="Root directory containing Dongxing shapefiles.")
    parser.add_argument("--dltb", type=Path, help="Path to DLTB.shp. Overrides --root search.")
    parser.add_argument("--coverage-json", type=Path, default=Path("paper7/results/dongxing_dem_coverage.json"))
    parser.add_argument("--output-gpkg", type=Path, help="Optional output GeoPackage with slope_mean.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dltb_path = args.dltb if args.dltb is not None else find_dltb_shapefile(args.root)
    coverage = audit_dem_coverage(args.dem, dltb_path)
    args.coverage_json.parent.mkdir(parents=True, exist_ok=True)
    args.coverage_json.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(console_json(coverage))

    if args.output_gpkg is not None:
        if not coverage["covers_vector"]:
            raise SystemExit("DEM does not cover Dongxing; slope enrichment skipped.")
        parcels = gpd.read_file(dltb_path, engine="pyogrio")
        enrich_slope_mean(parcels, args.dem, args.output_gpkg)
        print(f"Wrote {args.output_gpkg}")


if __name__ == "__main__":
    main()
