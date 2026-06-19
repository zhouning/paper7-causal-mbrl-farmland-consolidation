from shapely.geometry import box

from paper7.generic_county_env import GenericCountyEnv


def _toy_parcels():
    return [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 4.0, "geometry": box(3, 0, 4, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 8.0, "geometry": box(4, 0, 5, 1)},
    ]


def test_generic_env_masks_positive_gain_blocks_and_runs_full_reward():
    env = GenericCountyEnv(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        total_budget=2,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )

    _, info = env.reset(seed=0)
    assert env.action_masks().tolist() == [True, False]
    assert info["baimu_count"] == 0

    _, reward, terminated, truncated, info = env.step(0)

    assert reward > 0
    assert terminated is True
    assert truncated is False
    assert info["completed_swaps"] == 1
    assert info["slope_change_pct"] < 0
    assert "reward_components" in info


def test_generic_env_counts_baimu_components_from_adjacency():
    parcels = [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 1.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 1.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 50.0, "slope": 1.0, "geometry": box(4, 0, 5, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 0.5, "geometry": box(5, 0, 6, 1)},
    ]
    env = GenericCountyEnv(
        parcels=parcels,
        block_compositions={"0": [2, 3]},
        block_ids=[0],
        total_budget=1,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )

    _, info = env.reset(seed=0)

    assert info["baimu_count"] == 1
    assert info["baimu_area_ha"] == 0.02


def test_dongxing_loader_uses_swappable_index_order(tmp_path):
    import geopandas as gpd
    import pandas as pd

    from paper7.dongxing_full_env import build_env_from_frame_and_block_package

    frame = gpd.GeoDataFrame(
        {
            "DLBM": ["011", "031", "011", "031"],
            "DLMC": ["farmland", "forest", "farmland", "forest"],
            "TBMJ": [100.0, 100.0, 100.0, 100.0],
            "slope_mean": [10.0, 2.0, 4.0, 8.0],
        },
        geometry=[
            box(0, 0, 1, 1),
            box(1, 0, 2, 1),
            box(3, 0, 4, 1),
            box(4, 0, 5, 1),
        ],
        crs="EPSG:3857",
    )
    block_dir = tmp_path / "blocks"
    block_dir.mkdir()
    pd.DataFrame(
        [
            {"swappable_index": 0, "source_index": 0, "land_use": "farmland", "block_id": 0},
            {"swappable_index": 1, "source_index": 1, "land_use": "forest", "block_id": 0},
            {"swappable_index": 2, "source_index": 2, "land_use": "farmland", "block_id": 1},
            {"swappable_index": 3, "source_index": 3, "land_use": "forest", "block_id": 1},
        ]
    ).to_csv(block_dir / "parcel_block_mapping.csv", index=False)
    (block_dir / "block_compositions.json").write_text(
        '{"0": [0, 1], "1": [2, 3]}',
        encoding="utf-8",
    )
    (block_dir / "block_features.json").write_text('[{"block_id": 0}, {"block_id": 1}]', encoding="utf-8")

    env = build_env_from_frame_and_block_package(frame, block_dir, total_budget=2, swaps_per_step=1)

    assert env.n_blocks == 2
    assert env.action_masks().tolist() == [True, False]
