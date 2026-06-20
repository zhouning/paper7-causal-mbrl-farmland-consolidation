import json

import numpy as np
from shapely.geometry import box

from paper7.dongxing_full_model_based_policy import fit_one_step_model
from paper7.dongxing_full_transition_diagnostics import collect_transition_rows
from paper7.dongxing_multistep_mbrl_policy import (
    evaluate_multistep_policy_real,
    optimize_policy_weights_cem,
    rollout_surrogate_policy,
    run_experiment,
)
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC


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


def _toy_model():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )
    return fit_one_step_model(rows, ridge=1e-3)


def test_rollout_surrogate_policy_updates_state_over_multiple_steps():
    env = _toy_env()
    obs, _ = env.reset(seed=0)
    model = _toy_model()
    weights = np.zeros(K_BLOCK_GENERIC + 1, dtype=np.float64)
    weights[0] = 1.0
    weights[-1] = 1.0

    rollout = rollout_surrogate_policy(
        initial_obs=obs,
        n_blocks=env.n_blocks,
        model=model,
        weights=weights,
        horizon=2,
    )

    assert rollout["steps"] >= 1
    assert np.isfinite(rollout["predicted_reward_sum"])
    assert len(rollout["selected_actions"]) == rollout["steps"]
    assert rollout["final_obs"].shape == obs.shape


def test_optimize_policy_weights_cem_returns_elite_weights_and_history():
    env = _toy_env()
    obs, _ = env.reset(seed=0)
    model = _toy_model()

    result = optimize_policy_weights_cem(
        initial_observations=[obs],
        n_blocks=env.n_blocks,
        model=model,
        horizon=2,
        iterations=2,
        population_size=8,
        elite_frac=0.25,
        seed=7,
    )

    assert result["optimizer"] == "cross_entropy_method"
    assert len(result["weights"]) == K_BLOCK_GENERIC + 1
    assert len(result["history"]) == 2
    assert result["history"][-1]["elite_mean_score"] >= result["history"][0]["elite_mean_score"]


def test_evaluate_multistep_policy_real_reports_real_environment_metrics():
    model = _toy_model()
    weights = np.zeros(K_BLOCK_GENERIC + 1, dtype=np.float64)
    weights[0] = 1.0
    weights[-1] = 1.0

    result = evaluate_multistep_policy_real(
        env=_toy_env(),
        model=model,
        weights=weights,
        seed=0,
    )

    assert result["policy"] == "multistep_learned_env_optimized"
    assert result["completed_swaps"] == 1
    assert result["slope_change_pct"] < 0
    assert result["unique_blocks"] >= 1


def test_run_experiment_marks_multi_step_learned_environment_scope(tmp_path):
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
        train_seeds=[0, 1],
        eval_seeds=[2, 3],
        max_steps=2,
        ridge=1e-3,
        cem_iterations=2,
        population_size=8,
        elite_frac=0.25,
        optimizer_seed=11,
    )

    assert result["status"] == "supported_as_dongxing_multistep_learned_environment_policy"
    assert result["mbrl_transition_model_used"] is True
    assert result["multi_step_mbrl_planning_tested"] is True
    assert result["policy_transfer_tested"] is False
    assert result["planning_horizon"] == 2
    assert result["real_environment_eval"]["summary"]["n"] == 2
