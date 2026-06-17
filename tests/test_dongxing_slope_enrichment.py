import numpy as np
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from paper7.dongxing_slope_enrichment import (
    bounds_cover,
    console_json,
    compute_slope_degrees,
    enrich_slope_mean,
    mean_raster_values_by_geometry,
)


def test_bounds_cover_detects_full_coverage():
    raster_bounds = (0.0, 0.0, 10.0, 10.0)
    vector_bounds = (1.0, 2.0, 9.0, 8.0)

    assert bounds_cover(raster_bounds, vector_bounds) is True


def test_bounds_cover_rejects_non_overlapping_bounds():
    raster_bounds = (105.7, 29.2, 106.4, 29.9)
    dongxing_bounds = (104.98, 29.44, 105.42, 29.85)

    assert bounds_cover(raster_bounds, dongxing_bounds) is False


def test_compute_slope_degrees_returns_zero_for_flat_dem():
    dem = np.ones((5, 5), dtype=float) * 100.0

    slope = compute_slope_degrees(dem, x_resolution=30.0, y_resolution=30.0)

    assert slope.shape == dem.shape
    assert np.allclose(slope, 0.0)


def test_console_json_is_ascii_safe_for_windows_stdout():
    text = console_json({"path": "D:\\\u5317\u5927MEM\\\u5185\u6c5f"})

    text.encode("ascii")
    assert "\\u5317\\u5927" in text


def test_mean_raster_values_by_geometry_batches_polygon_means():
    raster = np.array(
        [
            [1.0, 3.0, 5.0, 7.0],
            [2.0, np.nan, 6.0, 8.0],
            [10.0, 12.0, 20.0, 22.0],
            [14.0, 16.0, 24.0, 26.0],
        ]
    )
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    geometries = [
        box(0.0, 2.0, 2.0, 4.0),
        box(2.0, 0.0, 4.0, 2.0),
        box(10.0, 10.0, 11.0, 11.0),
    ]

    means = mean_raster_values_by_geometry(raster, transform, geometries)

    assert means == [2.0, 23.0, None]


def test_mean_raster_values_by_geometry_can_include_touched_pixels_for_small_parcels():
    raster = np.array([[10.0]])
    transform = from_origin(0.0, 1.0, 1.0, 1.0)
    small_corner_geometry = box(0.01, 0.01, 0.2, 0.2)

    means = mean_raster_values_by_geometry(raster, transform, [small_corner_geometry], all_touched=True)

    assert means == [10.0]


def test_mean_raster_values_by_geometry_samples_representative_point_for_small_parcels():
    raster = np.array([[10.0]])
    transform = from_origin(0.0, 1.0, 1.0, 1.0)
    small_corner_geometry = box(0.01, 0.01, 0.2, 0.2)

    means = mean_raster_values_by_geometry(raster, transform, [small_corner_geometry])

    assert means == [10.0]


def test_enrich_slope_mean_handles_integer_dem_with_nodata(tmp_path):
    dem_path = tmp_path / "integer_dem.tif"
    output_path = tmp_path / "parcels_with_slope.gpkg"
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    dem = np.array(
        [
            [100, 100, 100, 100],
            [100, -9999, 100, 100],
            [110, 110, 120, 120],
            [110, 110, 120, 120],
        ],
        dtype=np.int16,
    )
    with rasterio.open(
        dem_path,
        "w",
        driver="GTiff",
        height=4,
        width=4,
        count=1,
        dtype=dem.dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(dem, 1)

    parcels = gpd.GeoDataFrame(
        {"DLBM": ["013", "031"], "DLMC": ["dryland", "forest"], "TBMJ": [1.0, 1.0]},
        geometry=[box(0.0, 2.0, 2.0, 4.0), box(2.0, 0.0, 4.0, 2.0)],
        crs="EPSG:3857",
    )

    enrich_slope_mean(parcels, dem_path, output_path)

    enriched = gpd.read_file(output_path, engine="pyogrio")
    assert "slope_mean" in enriched.columns
    assert enriched["slope_mean"].notna().all()
