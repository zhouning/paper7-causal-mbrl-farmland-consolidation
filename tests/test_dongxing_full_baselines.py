import numpy as np

from paper7.dongxing_full_baselines import choose_full_env_action, summarize_runs


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
