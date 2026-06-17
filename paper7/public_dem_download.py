"""Download public DEM tiles for Paper 7 external validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from urllib.parse import urlencode

import geopandas as gpd
import requests
from pyproj import Transformer
import rasterio
from rasterio.merge import merge

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.dongxing_data_audit import find_dltb_shapefile


OPENTOPOGRAPHY_GLOBALDEM_URL = "https://portal.opentopography.org/API/globaldem"
AWS_TERRAIN_GEOTIFF_URL = "https://s3.amazonaws.com/elevation-tiles-prod/geotiff"


def padded_bounds(
    bounds: tuple[float, float, float, float],
    pad_degrees: float,
) -> tuple[float, float, float, float]:
    west, south, east, north = bounds
    return (
        round(west - pad_degrees, 8),
        round(south - pad_degrees, 8),
        round(east + pad_degrees, 8),
        round(north + pad_degrees, 8),
    )


def build_opentopography_url(
    bounds: tuple[float, float, float, float],
    dem_type: str = "SRTMGL1",
    api_key: str | None = None,
) -> str:
    west, south, east, north = bounds
    query = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
    }
    if api_key:
        query["API_Key"] = api_key
    return f"{OPENTOPOGRAPHY_GLOBALDEM_URL}?{urlencode(query)}"


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tiles_for_bounds(bounds: tuple[float, float, float, float], zoom: int) -> list[tuple[int, int, int]]:
    west, south, east, north = bounds
    x_min, y_south = lonlat_to_tile(west, south, zoom)
    x_max, y_north = lonlat_to_tile(east, north, zoom)
    y_min = min(y_north, y_south)
    y_max = max(y_north, y_south)
    return [(zoom, x, y) for x in range(x_min, x_max + 1) for y in range(y_min, y_max + 1)]


def aws_terrain_tile_url(tile: tuple[int, int, int]) -> str:
    zoom, x, y = tile
    return f"{AWS_TERRAIN_GEOTIFF_URL}/{zoom}/{x}/{y}.tif"


def vector_bounds_wgs84(vector_path: Path) -> tuple[float, float, float, float]:
    frame = gpd.read_file(vector_path, engine="pyogrio")
    if frame.crs is None:
        raise ValueError(f"Vector layer has no CRS: {vector_path}")
    bounds = frame.total_bounds
    transformer = Transformer.from_crs(frame.crs, "EPSG:4326", always_xy=True)
    left, bottom, right, top = [float(value) for value in bounds]
    points = [
        transformer.transform(left, bottom),
        transformer.transform(left, top),
        transformer.transform(right, bottom),
        transformer.transform(right, top),
    ]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def download_file(url: str, output_path: Path, timeout_s: int = 300) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type.lower():
            text = response.text[:500]
            raise RuntimeError(f"DEM endpoint returned HTML instead of raster: {text}")
        bytes_written = 0
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bytes_written += len(chunk)
    return {
        "url": url,
        "output_path": str(output_path),
        "bytes": bytes_written,
    }


def download_aws_terrain_mosaic(
    bounds: tuple[float, float, float, float],
    output_path: Path,
    tile_dir: Path,
    zoom: int = 10,
) -> dict[str, object]:
    tile_dir.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tiles = tiles_for_bounds(bounds, zoom)
    tile_paths = []
    downloads = []

    for tile in tiles:
        tile_path = tile_dir / f"{tile[0]}_{tile[1]}_{tile[2]}.tif"
        if not tile_path.exists():
            downloads.append(download_file(aws_terrain_tile_url(tile), tile_path))
        tile_paths.append(tile_path)

    datasets = [rasterio.open(path) for path in tile_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            {
                "height": int(mosaic.shape[1]),
                "width": int(mosaic.shape[2]),
                "transform": transform,
                "driver": "GTiff",
            }
        )
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mosaic)
    finally:
        for dataset in datasets:
            dataset.close()

    return {
        "provider": "aws-terrain-tiles",
        "zoom": int(zoom),
        "tile_count": len(tiles),
        "tiles": [{"z": z, "x": x, "y": y} for z, x, y in tiles],
        "downloaded_tiles": downloads,
        "tile_dir": str(tile_dir),
        "output_path": str(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Root directory containing Dongxing shapefiles.")
    parser.add_argument("--dltb", type=Path, help="Path to DLTB.shp. Overrides --root search.")
    parser.add_argument("--output", type=Path, default=Path("paper7/data/dongxing_dem_srtmgl1.tif"))
    parser.add_argument("--metadata", type=Path, default=Path("paper7/results/dongxing_dem_download.json"))
    parser.add_argument("--provider", choices=["opentopography", "aws-terrain"], default="aws-terrain")
    parser.add_argument("--dem-type", default="SRTMGL1")
    parser.add_argument("--zoom", type=int, default=10)
    parser.add_argument("--tile-dir", type=Path, default=Path("paper7/data/dongxing_dem_tiles"))
    parser.add_argument("--pad-degrees", type=float, default=0.02)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dltb_path = args.dltb if args.dltb is not None else find_dltb_shapefile(args.root)
    base_bounds = vector_bounds_wgs84(dltb_path)
    request_bounds = padded_bounds(base_bounds, args.pad_degrees)
    url = build_opentopography_url(request_bounds, dem_type=args.dem_type, api_key=args.api_key)

    metadata = {
        "source": "OpenTopography Global DEM API" if args.provider == "opentopography" else "AWS Terrain Tiles",
        "provider": args.provider,
        "dem_type": args.dem_type,
        "vector_path": str(dltb_path),
        "vector_bounds_wgs84": [round(float(value), 8) for value in base_bounds],
        "request_bounds_wgs84": [round(float(value), 8) for value in request_bounds],
        "url": url if args.provider == "opentopography" else None,
        "output": str(args.output),
        "dry_run": bool(args.dry_run),
    }

    if not args.dry_run:
        if args.provider == "opentopography":
            metadata["download"] = download_file(url, args.output)
        else:
            metadata["download"] = download_aws_terrain_mosaic(
                request_bounds,
                output_path=args.output,
                tile_dir=args.tile_dir,
                zoom=args.zoom,
            )

    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
