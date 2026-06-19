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
