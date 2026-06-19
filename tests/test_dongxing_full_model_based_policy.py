import json

from shapely.geometry import box

from paper7.dongxing_full_model_based_policy import (
    evaluate_model_based_policy,
    fit_one_step_model,
    predict_action_rewards,
    run_experiment,
    select_model_based_action,
    summarize_model_based_runs,
)
from paper7.dongxing_full_transition_diagnostics import collect_transition_rows
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


def test_fit_one_step_model_predicts_reward_and_features():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )

    model = fit_one_step_model(rows, ridge=1e-3)

    assert model["model_type"] == "one_step_ridge_transition_reward"
    assert model["feature_standardization"]["enabled"] is True
    assert model["n_training_transitions"] == len(rows)
    assert len(model["reward_coef"]) == 17
    assert len(model["selected_coef"]) == 17
    assert len(model["selected_coef"][0]) == 8


def test_select_model_based_action_prefers_positive_reward_block():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )
    model = fit_one_step_model(rows, ridge=1e-3)
    env = _toy_env()
    obs, _ = env.reset(seed=0)

    action = select_model_based_action(obs, env.n_blocks, model)

    assert action == 0


def test_predict_action_rewards_masks_invalid_actions():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )
    model = fit_one_step_model(rows, ridge=1e-3)
    env = _toy_env()
    obs, _ = env.reset(seed=0)

    rewards = predict_action_rewards(obs, env.n_blocks, model)

    assert rewards.shape == (2,)
    assert rewards[0] > rewards[1]


def test_evaluate_model_based_policy_reports_real_environment_metrics():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )
    model = fit_one_step_model(rows, ridge=1e-3)

    result = evaluate_model_based_policy(_toy_env(), model, seed=0)

    assert result["policy"] == "one_step_model_based_reward"
    assert result["completed_swaps"] == 1
    assert result["slope_change_pct"] < 0
    assert "model_predicted_reward_sum" in result


def test_run_experiment_marks_bounded_model_based_scope(tmp_path):
    baseline_path = tmp_path / "baselines.json"
    baseline_path.write_text(
        json.dumps(
            {
                "policy_summaries": {
                    "random": {"reward_mean": 0.0, "slope_change_pct_mean": 0.0}
                }
            }
        ),
        encoding="utf-8",
    )

    result = run_experiment(
        env_factory=_toy_env,
        baseline_path=baseline_path,
        collection_policies=["dynamic_slope_gap", "random"],
        train_seeds=[0, 1, 2, 3],
        eval_seeds=[0, 1],
        max_steps=2,
        ridge=1e-3,
    )

    assert result["status"] == "supported_as_dongxing_full_one_step_model_based_policy"
    assert result["mbrl_transition_model_used"] is True
    assert result["policy_transfer_tested"] is False
    assert result["model_based_policy"]["summary"]["n"] == 2


def test_summarize_model_based_runs_compares_to_baselines():
    runs = [
        {
            "reward": 3.0,
            "slope_change_pct": -1.0,
            "cont_change": 0.1,
            "baimu_area_change_ha": 2.0,
            "completed_swaps": 2,
            "unique_blocks": 2,
        }
    ]
    baselines = {
        "policy_summaries": {
            "random": {
                "reward_mean": 1.0,
                "slope_change_pct_mean": -0.2,
                "cont_change_mean": 0.0,
                "baimu_area_change_ha_mean": 1.0,
            }
        }
    }

    summary, comparisons = summarize_model_based_runs(runs, baselines)

    assert summary["reward_mean"] == 3.0
    assert comparisons["model_based_minus_random_reward_mean"] == 2.0
    assert comparisons["model_based_minus_random_slope_change_pct_mean"] == -0.8
