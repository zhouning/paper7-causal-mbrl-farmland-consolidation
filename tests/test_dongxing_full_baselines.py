import numpy as np
from shapely.geometry import box

from paper7.dongxing_full_baselines import choose_full_env_action, run_policy_episode, summarize_runs
from paper7.generic_county_env import GenericCountyEnv


def test_choose_full_env_action_respects_mask():
    features = np.zeros((3, 8), dtype=np.float32)
    features[:, 0] = [0.5, 0.9, 0.3]
    features[:, 1] = [0.2, 0.4, 0.9]
    mask = np.array([True, False, True])
    rng = np.random.default_rng(0)

    assert choose_full_env_action("dynamic_slope_gap", features, mask, rng) == 0
    assert choose_full_env_action("area_weighted_slope_gap", features, mask, rng) == 2


def test_summarize_runs_reports_multi_objective_means():
    runs = [
        {
            "slope_change_pct": -1.0,
            "cont_change": 0.1,
            "baimu_count_change": 1,
            "baimu_area_change_ha": 2,
            "reward": 10,
        },
        {
            "slope_change_pct": -2.0,
            "cont_change": 0.3,
            "baimu_count_change": 3,
            "baimu_area_change_ha": 6,
            "reward": 20,
        },
    ]

    summary = summarize_runs(runs)

    assert summary["n"] == 2
    assert summary["slope_change_pct_mean"] == -1.5
    assert summary["cont_change_mean"] == 0.2
    assert summary["baimu_count_change_mean"] == 2.0
    assert summary["reward_mean"] == 15.0


def test_run_policy_episode_reuses_observation_features(monkeypatch):
    parcels = [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 4.0, "geometry": box(3, 0, 4, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 8.0, "geometry": box(4, 0, 5, 1)},
    ]
    env = GenericCountyEnv(
        parcels=parcels,
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        total_budget=2,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )
    original = env.block_feature_matrix
    calls = {"n": 0}

    def counted_features():
        calls["n"] += 1
        return original()

    monkeypatch.setattr(env, "block_feature_matrix", counted_features)

    result = run_policy_episode(env, "dynamic_slope_gap", seed=0)

    assert result["completed_swaps"] == 1
    assert calls["n"] == result["steps"] + 1
