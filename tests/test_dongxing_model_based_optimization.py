import json

from shapely.geometry import box

from paper7.dongxing_full_model_based_policy import fit_one_step_model
from paper7.dongxing_full_transition_diagnostics import collect_transition_rows
from paper7.dongxing_model_based_optimization import (
    evaluate_candidate_policy,
    make_candidate_grid,
    run_optimization_experiment,
    score_actions,
    select_best_candidate,
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


def _toy_model():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )
    return fit_one_step_model(rows, ridge=1e-3)


def test_make_candidate_grid_contains_pre_specified_baseline_and_variants():
    candidates = make_candidate_grid()

    names = {candidate["name"] for candidate in candidates}
    assert "reward_only" in names
    assert "reward_slope_bonus" in names
    assert "reward_diversity_penalty" in names
    assert all("reward_weight" in candidate for candidate in candidates)


def test_score_actions_combines_reward_slope_and_diversity_terms():
    env = _toy_env()
    obs, _ = env.reset(seed=0)
    model = _toy_model()
    candidate = {
        "name": "test",
        "reward_weight": 1.0,
        "slope_weight": 10.0,
        "current_farm_weight": 0.0,
        "neighbor_weight": 0.0,
        "diversity_penalty": 0.0,
    }

    scores = score_actions(obs, env.n_blocks, model, candidate, selected_counts={})

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_evaluate_candidate_policy_reports_candidate_metrics():
    model = _toy_model()
    candidate = make_candidate_grid()[0]

    result = evaluate_candidate_policy(_toy_env(), model, candidate, seed=0)

    assert result["policy"] == "optimized_one_step_model_based"
    assert result["candidate"] == candidate["name"]
    assert result["completed_swaps"] == 1
    assert result["slope_change_pct"] < 0


def test_select_best_candidate_uses_selection_reward_then_slope():
    summaries = [
        {"candidate": "a", "reward_mean": 1.0, "slope_change_pct_mean": -3.0},
        {"candidate": "b", "reward_mean": 2.0, "slope_change_pct_mean": -1.0},
        {"candidate": "c", "reward_mean": 2.0, "slope_change_pct_mean": -2.0},
    ]

    best = select_best_candidate(summaries)

    assert best["candidate"] == "c"


def test_run_optimization_experiment_separates_selection_and_eval(tmp_path):
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
    candidates = [
        {
            "name": "reward_only",
            "reward_weight": 1.0,
            "slope_weight": 0.0,
            "current_farm_weight": 0.0,
            "neighbor_weight": 0.0,
            "diversity_penalty": 0.0,
        },
        {
            "name": "reward_slope_bonus",
            "reward_weight": 1.0,
            "slope_weight": 10.0,
            "current_farm_weight": 0.0,
            "neighbor_weight": 0.0,
            "diversity_penalty": 0.0,
        },
    ]

    result = run_optimization_experiment(
        env_factory=_toy_env,
        baseline_path=baseline_path,
        collection_policies=["dynamic_slope_gap", "random"],
        train_seeds=[0, 1],
        selection_seeds=[2, 3],
        eval_seeds=[4, 5],
        max_steps=2,
        ridge=1e-3,
        candidates=candidates,
    )

    assert result["status"] == "supported_as_dongxing_model_based_scoring_optimization"
    assert result["selection_seeds"] == [2, 3]
    assert result["eval_seeds"] == [4, 5]
    assert result["best_candidate"]["name"] in {"reward_only", "reward_slope_bonus"}
    assert result["heldout_eval"]["summary"]["n"] == 2
    assert len(result["candidate_selection_summaries"]) == 2
