from shapely.geometry import box

from paper7.dongxing_full_transition_diagnostics import (
    collect_transition_rows,
    summarize_transition_diagnostics,
    train_policy_holdout_diagnostics,
    train_ridge_transition,
)
from paper7.generic_county_env import GenericCountyEnv


def _toy_env() -> GenericCountyEnv:
    parcels = [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 6.0, "geometry": box(2, 0, 3, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 1.0, "geometry": box(3, 0, 4, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 4.0, "geometry": box(6, 0, 7, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 8.0, "geometry": box(7, 0, 8, 1)},
    ]
    return GenericCountyEnv(
        parcels=parcels,
        block_compositions={"0": [0, 1], "1": [2, 3], "2": [4, 5]},
        block_ids=[0, 1, 2],
        total_budget=3,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )


def test_collect_transition_rows_records_selected_and_next_features():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap"],
        seeds=[0],
        max_steps=2,
    )

    assert len(rows) == 2
    first = rows[0]
    assert first["policy"] == "dynamic_slope_gap"
    assert first["action"] == 0
    assert first["selected_block_features"].shape == (8,)
    assert first["next_selected_block_features"].shape == (8,)
    assert first["global_features"].shape == (8,)
    assert first["reward"] > 0


def test_train_ridge_transition_beats_persistence_on_toy_rows():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )

    result = train_ridge_transition(rows, train_fraction=0.75, ridge=1e-3, split_seed=42)

    assert result["n_train"] > 0
    assert result["n_val"] > 0
    assert result["split"] == "seeded_random_transition_split"
    assert result["feature_standardization"]["enabled"] is True
    assert result["selected_feature_mae"] <= result["selected_feature_persistence_mae"]
    assert result["reward_mae"] < result["reward_persistence_mae"]


def test_policy_holdout_diagnostics_reports_generalization_boundary():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )

    result = train_policy_holdout_diagnostics(
        rows,
        holdout_policy="random",
        ridge=1e-3,
    )

    assert result["split"] == "policy_holdout"
    assert result["holdout_policy"] == "random"
    assert result["n_train"] > 0
    assert result["n_val"] > 0
    assert result["feature_standardization"]["enabled"] is True


def test_summarize_transition_diagnostics_marks_not_policy_evidence():
    rows = collect_transition_rows(
        env_factory=_toy_env,
        policies=["dynamic_slope_gap", "random"],
        seeds=[0, 1, 2, 3],
        max_steps=2,
    )
    model_result = train_ridge_transition(rows, train_fraction=0.75, ridge=1e-3)

    summary = summarize_transition_diagnostics(rows, model_result)

    assert summary["status"] == "supported_as_dongxing_full_transition_diagnostic"
    assert summary["mbrl_policy_trained"] is False
    assert summary["n_transitions"] == len(rows)
