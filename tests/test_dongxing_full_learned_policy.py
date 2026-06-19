from shapely.geometry import box

from paper7.dongxing_full_learned_policy import (
    compare_to_full_baselines,
    evaluate_preference_policy,
    train_preference_policy,
)
from paper7.generic_county_env import GenericCountyEnv


def _toy_env() -> GenericCountyEnv:
    parcels = [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 4.0, "geometry": box(3, 0, 4, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 8.0, "geometry": box(4, 0, 5, 1)},
    ]
    return GenericCountyEnv(
        parcels=parcels,
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        total_budget=2,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )


def test_train_preference_policy_learns_positive_gain_weight():
    policy = train_preference_policy(
        env_factory=_toy_env,
        train_seeds=[0, 1],
        episodes=12,
        learning_rate=0.05,
        epsilon=0.25,
    )

    assert policy["learner_type"] == "linear_preference_full_reward"
    assert policy["weights"][0] > 0
    assert policy["training"]["episodes"] == 12


def test_evaluate_preference_policy_reports_full_metrics():
    policy = {
        "learner_type": "linear_preference_full_reward",
        "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    result = evaluate_preference_policy(_toy_env(), policy, seed=0)

    assert result["policy"] == "learned_full_reward_preference"
    assert result["completed_swaps"] == 1
    assert result["slope_change_pct"] < 0
    assert "cont_change" in result
    assert "baimu_area_change_ha" in result


def test_compare_to_full_baselines_reports_deltas():
    learned_summary = {
        "reward_mean": 12.0,
        "slope_change_pct_mean": -1.0,
        "cont_change_mean": 0.1,
        "baimu_area_change_ha_mean": 5.0,
    }
    baselines = {
        "policy_summaries": {
            "random": {"reward_mean": 2.0, "slope_change_pct_mean": -0.2},
            "scalarized_default": {"reward_mean": 10.0, "slope_change_pct_mean": -0.8},
        }
    }

    comparisons = compare_to_full_baselines(learned_summary, baselines)

    assert comparisons["learned_minus_random_reward_mean"] == 10.0
    assert comparisons["learned_minus_scalarized_default_reward_mean"] == 2.0
    assert comparisons["learned_minus_scalarized_default_slope_change_pct_mean"] == -0.2
