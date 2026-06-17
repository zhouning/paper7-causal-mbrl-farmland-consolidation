from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

from paper7.dongxing_data_audit import audit_geodataframe, classify_land_use


def test_classify_land_use_maps_chinese_land_codes():
    assert classify_land_use("013", "旱地") == "farmland"
    assert classify_land_use("031", "有林地") == "forest"
    assert classify_land_use("032", "灌木林地") == "forest"
    assert classify_land_use("203", "村庄") == "other"


def test_audit_geodataframe_reports_external_county_readiness():
    frame = gpd.GeoDataFrame(
        {
            "DLBM": ["013", "011", "031", "032", "203"],
            "DLMC": ["旱地", "水田", "有林地", "灌木林地", "村庄"],
            "TBMJ": [1000.0, 2000.0, 1500.0, 500.0, 300.0],
            "GDPDJ": ["2", "4", None, "", "5"],
            "ZLDWDM": ["a", "a", "b", "b", "c"],
            "QSDWDM": ["qa", "qa", "qb", "qb", "qc"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(0, 1, 1, 2), box(1, 1, 2, 2), box(2, 2, 3, 3)],
        crs="EPSG:2359",
    )

    audit = audit_geodataframe(frame, source_path=Path("DLTB.shp"))

    assert audit["source"]["record_count"] == 5
    assert audit["schema"]["has_required_fields"] is True
    assert audit["land_use"]["farmland"]["count"] == 2
    assert audit["land_use"]["forest"]["count"] == 2
    assert audit["land_use"]["other"]["count"] == 1
    assert audit["land_use"]["farmland"]["area_m2"] == 3000.0
    assert audit["land_use"]["forest"]["area_m2"] == 2000.0
    assert audit["slope"]["field"] == "GDPDJ"
    assert audit["slope"]["non_null_count"] == 3
    assert audit["administrative_units"]["ZLDWDM"]["unique_count"] == 3
    assert audit["slope"]["by_land_use"]["farmland"]["non_null_count"] == 2
    assert audit["slope"]["by_land_use"]["forest"]["non_null_count"] == 0
    assert audit["readiness"]["can_build_block_inputs"] is True
    assert audit["readiness"]["can_run_slope_swap_environment"] is False
    assert audit["readiness"]["slope_mode"] == "slope_grade_proxy"
    assert "forest parcels lack slope values" in audit["readiness"]["blockers"]


def test_audit_geodataframe_allows_slope_swap_environment_when_forest_slopes_exist():
    frame = gpd.GeoDataFrame(
        {
            "DLBM": ["013", "031"],
            "DLMC": ["旱地", "有林地"],
            "TBMJ": [1000.0, 1500.0],
            "GDPDJ": ["4", "2"],
            "ZLDWDM": ["a", "a"],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1)],
        crs="EPSG:2359",
    )

    audit = audit_geodataframe(frame, source_path=Path("DLTB.shp"))

    assert audit["readiness"]["can_build_block_inputs"] is True
    assert audit["readiness"]["can_run_slope_swap_environment"] is True
    assert audit["readiness"]["blockers"] == []
