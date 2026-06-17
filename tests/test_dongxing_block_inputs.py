from pathlib import Path
import subprocess
import sys

import geopandas as gpd
from shapely.geometry import box

from paper7.dongxing_block_inputs import prepare_block_inputs


def test_prepare_block_inputs_builds_mixed_swappable_blocks():
    frame = gpd.GeoDataFrame(
        {
            "DLBM": ["013", "011", "031", "013", "013"],
            "DLMC": ["旱地", "水田", "有林地", "旱地", "旱地"],
            "TBMJ": [100.0, 100.0, 100.0, 100.0, 100.0],
            "GDPDJ": ["3", "4", "", "2", "3"],
            "ZLDWDM": ["unit_a", "unit_a", "unit_a", "unit_b", "unit_b"],
            "ZLDWMC": ["A", "A", "A", "B", "B"],
        },
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(2, 0, 3, 1),
            box(10, 0, 11, 1),
            box(11, 0, 12, 1),
        ],
        crs="EPSG:2359",
    )

    package = prepare_block_inputs(
        frame,
        source_path=Path("DLTB.shp"),
        unit_field="ZLDWDM",
        max_parcels=10,
        min_parcels=2,
        min_area_ha=0.0,
    )

    assert package["summary"]["n_swappable_parcels"] == 5
    assert package["summary"]["n_blocks"] == 1
    assert package["summary"]["n_units_processed"] == 2
    assert package["summary"]["n_units_with_blocks"] == 1
    assert package["summary"]["has_complete_slope_for_swap"] is False
    assert package["summary"]["slope_gap"] == "forest slope values are missing"
    assert package["block_features"][0]["unit_id"] == "unit_a"
    assert package["block_features"][0]["n_farmland"] == 2
    assert package["block_features"][0]["n_forest"] == 1
    assert package["block_features"][0]["avg_farm_slope_grade"] == 3.5
    assert package["block_features"][0]["avg_forest_slope_grade"] is None
    assert package["block_compositions"]["0"] == [0, 1, 2]
    assert package["parcel_mapping"][0]["block_id"] == 0
    assert package["parcel_mapping"][3]["block_id"] == -1


def test_prepare_block_inputs_prefers_dem_slope_mean_when_available():
    frame = gpd.GeoDataFrame(
        {
            "DLBM": ["013", "011", "031"],
            "DLMC": ["dryland", "paddy", "forest"],
            "TBMJ": [100.0, 300.0, 600.0],
            "GDPDJ": ["5", "5", ""],
            "slope_mean": [9.0, 3.0, 12.0],
            "ZLDWDM": ["unit_a", "unit_a", "unit_a"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
        crs="EPSG:2359",
    )

    package = prepare_block_inputs(
        frame,
        source_path=Path("dongxing_DLTB_with_slope.gpkg"),
        unit_field="ZLDWDM",
        max_parcels=10,
        min_parcels=2,
        min_area_ha=0.0,
    )

    feature = package["block_features"][0]
    assert package["summary"]["has_complete_slope_for_swap"] is True
    assert package["summary"]["slope_gap"] is None
    assert package["summary"]["slope_mode"] == "continuous_slope"
    assert feature["avg_farm_slope"] == 4.5
    assert feature["avg_forest_slope"] == 12.0
    assert feature["avg_farm_slope_grade"] == 5.0
    assert feature["avg_forest_slope_grade"] is None


def test_dongxing_block_inputs_script_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "paper7/dongxing_block_inputs.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--output-dir" in result.stdout
