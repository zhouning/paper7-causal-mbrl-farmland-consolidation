import numpy as np

from paper7.reward_component_rollouts import (
    choose_action,
    component_from_step_state,
    summarize_episode,
)


def test_component_from_step_state_matches_county_env_delta_definitions():
    component = component_from_step_state(
        prev_slope=10.0,
        cur_slope=9.0,
        initial_slope=10.0,
        prev_cont=2.0,
        cur_cont=2.2,
        initial_cont=2.0,
        prev_baimu_area=100.0,
        cur_baimu_area=120.0,
        initial_farm_area=1000.0,
        prev_baimu_count=3,
        cur_baimu_count=5,
        completed_swaps=4,
    )

    assert round(component.slope_delta, 6) == 0.1
    assert round(component.cont_delta, 6) == 0.1
    assert round(component.baimu_area_delta, 6) == 0.02
    assert component.baimu_new_count == 2
    assert component.completed_swaps == 4


def test_choose_action_obeys_mask_for_supported_heuristics():
    block_features = np.zeros((3, 17), dtype=np.float32)
    block_features[:, 3] = [0.5, 0.9, 0.4]
    block_features[:, 7] = [0.2, 0.4, 0.9]
    block_features[:, 8] = [0.2, 0.4, 0.9]
    block_features[:, 13] = [0.1, 0.8, 0.2]
    mask = np.array([True, False, True])
    rng = np.random.default_rng(0)

    assert choose_action("dynamic_slope_gap", block_features, mask, rng) == 0
    assert choose_action("area_weighted_slope_gap", block_features, mask, rng) == 2
    assert choose_action("contiguity_aware", block_features, mask, rng) == 0
    assert choose_action("baimu_aware", block_features, mask, rng) in {0, 2}
    assert choose_action("scalarized_default", block_features, mask, rng) in {0, 2}


def test_summarize_episode_reports_final_metrics_and_total_components():
    steps = [
        {
            "reward_default": 1.0,
            "slope_delta": 0.1,
            "cont_delta": 0.2,
            "baimu_area_delta": -0.01,
            "baimu_new_count": 0,
            "completed_swaps": 5,
            "slope_change_pct": -0.5,
            "cont_change": 0.01,
            "baimu_count_change": 1,
            "baimu_area_change_ha": -2.0,
            "budget_used": 5,
        },
        {
            "reward_default": 2.0,
            "slope_delta": 0.2,
            "cont_delta": 0.1,
            "baimu_area_delta": 0.03,
            "baimu_new_count": 1,
            "completed_swaps": 5,
            "slope_change_pct": -0.8,
            "cont_change": 0.03,
            "baimu_count_change": 2,
            "baimu_area_change_ha": 4.0,
            "budget_used": 10,
        },
    ]

    summary = summarize_episode("x", 7, steps)

    assert summary["policy"] == "x"
    assert summary["seed"] == 7
    assert summary["steps"] == 2
    assert summary["reward_default"] == 3.0
    assert summary["slope_delta_total"] == 0.3
    assert summary["final_slope_change_pct"] == -0.8
