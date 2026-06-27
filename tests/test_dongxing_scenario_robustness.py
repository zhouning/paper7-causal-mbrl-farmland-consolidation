import numpy as np
import pytest

from paper7.generic_county_env import GenericCountyEnv
from paper7.dongxing_scenario_robustness import (
    ScenarioSpec,
    apply_slope_perturbation,
    build_default_scenario_specs,
    evaluate_linear_weight_policy,
    optimize_scenario_robust_linear_policy,
    run_scenario_robustness_experiment,
    summarize_policy_scenario_runs,
)


def _toy_parcels():
    return [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": None},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": None},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 9.0, "geometry": None},
        {"land_use": "forest", "area_m2": 100.0, "slope": 1.0, "geometry": None},
    ]


def _toy_env(parcels=None, total_budget=4, swaps_per_step=1):
    return GenericCountyEnv(
        parcels=parcels or _toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        total_budget=total_budget,
        swaps_per_step=swaps_per_step,
    )


def test_default_scenario_specs_have_stable_ids_and_split():
    specs = build_default_scenario_specs()
    scenario_ids = [spec.scenario_id for spec in specs]

    assert "base" in scenario_ids
    assert len(scenario_ids) == len(set(scenario_ids))
    assert {spec.split for spec in specs}.issuperset({"selection", "heldout"})
    assert all(spec.total_budget > 0 for spec in specs)
    assert all(spec.swaps_per_step > 0 for spec in specs)


def test_apply_slope_perturbation_is_reproducible_nonnegative_and_shape_preserving():
    base = _toy_parcels()
    spec = ScenarioSpec(
        scenario_id="noise_seed7",
        split="heldout",
        slope_scale=1.0,
        slope_noise_sd=0.05,
        slope_noise_seed=7,
        total_budget=4,
        swaps_per_step=1,
        description="test noise",
    )

    first = apply_slope_perturbation(base, spec)
    second = apply_slope_perturbation(base, spec)

    assert len(first) == len(base)
    assert [row["land_use"] for row in first] == [row["land_use"] for row in base]
    assert all(row["slope"] >= 0.0 for row in first)
    assert [row["slope"] for row in first] == [row["slope"] for row in second]
    assert [row["slope"] for row in first] != [row["slope"] for row in base]


def test_summarize_policy_scenario_runs_uses_scenario_variation_for_deterministic_rows():
    runs = [
        {
            "policy": "deterministic_rule",
            "scenario_id": "base",
            "deterministic_policy": True,
            "reward": 10.0,
            "slope_change_pct": -1.0,
        },
        {
            "policy": "deterministic_rule",
            "scenario_id": "budget_low",
            "deterministic_policy": True,
            "reward": 5.0,
            "slope_change_pct": -0.5,
        },
    ]

    summary = summarize_policy_scenario_runs(runs)["deterministic_rule"]

    assert summary["scenario_count"] == 2
    assert summary["deterministic_policy"] is True
    assert summary["seed_replication_is_independent"] is False
    assert summary["reward_mean"] == 7.5
    assert summary["reward_worst"] == 5.0
    assert summary["slope_change_pct_worst"] == -0.5


def test_summarize_policy_scenario_runs_deduplicates_deterministic_seed_repeats():
    runs = [
        {
            "policy": "deterministic_rule",
            "scenario_id": "base",
            "seed": 0,
            "deterministic_policy": True,
            "reward": 10.0,
            "slope_change_pct": -1.0,
        },
        {
            "policy": "deterministic_rule",
            "scenario_id": "base",
            "seed": 1,
            "deterministic_policy": True,
            "reward": 99.0,
            "slope_change_pct": -9.9,
        },
        {
            "policy": "deterministic_rule",
            "scenario_id": "budget_low",
            "seed": 0,
            "deterministic_policy": True,
            "reward": 5.0,
            "slope_change_pct": -0.5,
        },
    ]

    summary = summarize_policy_scenario_runs(runs)["deterministic_rule"]

    assert summary["scenario_count"] == 2
    assert summary["n"] == 2
    assert summary["deterministic_policy"] is True
    assert summary["seed_replication_is_independent"] is False
    assert summary["reward_mean"] == 7.5
    assert summary["reward_worst"] == 5.0


def test_evaluate_linear_weight_policy_runs_on_toy_env():
    env = _toy_env()
    weights = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    result = evaluate_linear_weight_policy(
        env=env,
        weights=weights,
        policy_name="toy_linear",
        scenario_id="base",
    )

    assert result["policy"] == "toy_linear"
    assert result["scenario_id"] == "base"
    assert result["deterministic_policy"] is True
    assert result["steps"] > 0
    assert result["completed_swaps"] > 0


def test_optimize_scenario_robust_linear_policy_returns_weight_vector():
    envs = [_toy_env(), _toy_env(total_budget=3)]
    weights, optimizer = optimize_scenario_robust_linear_policy(
        envs=envs,
        iterations=2,
        population_size=6,
        elite_frac=0.5,
        seed=3,
    )

    assert weights.shape == (9,)
    assert optimizer["optimizer"] == "cross_entropy_method_real_scenario_smoke"
    assert len(optimizer["history"]) == 2


def test_optimize_scenario_robust_linear_policy_rejects_invalid_parameters():
    envs = [_toy_env()]

    with pytest.raises(ValueError, match="iterations must be positive"):
        optimize_scenario_robust_linear_policy(envs=envs, iterations=0)

    with pytest.raises(ValueError, match="population_size must be positive"):
        optimize_scenario_robust_linear_policy(envs=envs, population_size=0)

    with pytest.raises(ValueError, match="elite_frac must be in"):
        optimize_scenario_robust_linear_policy(envs=envs, elite_frac=0.0)


def test_run_scenario_robustness_experiment_smoke_uses_scenarios_not_seed_replication():
    parcels = _toy_parcels()
    block_compositions = {"0": [0, 1], "1": [2, 3]}
    block_ids = [0, 1]
    scenarios = [
        ScenarioSpec("base", "selection", 1.0, 0.0, 0, 4, 1, "base"),
        ScenarioSpec("scaled", "heldout", 1.1, 0.0, 0, 4, 1, "scaled"),
    ]

    result = run_scenario_robustness_experiment(
        parcels=parcels,
        block_compositions=block_compositions,
        block_ids=block_ids,
        scenarios=scenarios,
        baseline_policies=["dynamic_slope_gap"],
        random_seeds=[0, 1],
        cem_iterations=2,
        cem_population_size=6,
        output_path=None,
    )

    robust = result["policy_summaries"]["scenario_robust_mbrl"]
    deterministic = result["policy_summaries"]["dynamic_slope_gap"]

    assert result["status"] == "supported_as_dongxing_scenario_robustness"
    assert result["scenario_count"] == 2
    assert result["deterministic_seed_repetition_avoided"] is True
    assert robust["scenario_count"] == 2
    assert deterministic["seed_replication_is_independent"] is False


def test_run_scenario_robustness_experiment_requires_selection_scenario():
    scenarios = [
        ScenarioSpec("heldout_only", "heldout", 1.0, 0.0, 0, 4, 1, "heldout"),
    ]

    with pytest.raises(ValueError, match="At least one selection scenario"):
        run_scenario_robustness_experiment(
            parcels=_toy_parcels(),
            block_compositions={"0": [0, 1], "1": [2, 3]},
            block_ids=[0, 1],
            scenarios=scenarios,
            baseline_policies=["dynamic_slope_gap"],
            random_seeds=[0, 1],
            cem_iterations=2,
            cem_population_size=6,
            output_path=None,
        )
