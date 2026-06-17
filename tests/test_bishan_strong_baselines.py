import numpy as np
import pytest

from paper7.bishan_strong_baselines import (
    choose_area_weighted_action,
    choose_contiguity_aware_action,
    choose_immediate_slope_delta_action,
    choose_slope_gap_action,
    summarize_policy_runs,
)


def test_choose_slope_gap_action_respects_valid_mask():
    block_features = np.zeros((3, 17), dtype=np.float32)
    block_features[:, 3] = [0.8, 0.9, 0.7]
    mask = np.array([True, False, True])

    assert choose_slope_gap_action(block_features, mask) == 0


def test_choose_area_weighted_action_uses_swap_potential_and_area():
    block_features = np.zeros((3, 17), dtype=np.float32)
    block_features[:, 3] = [0.9, 0.6, 0.1]
    block_features[:, 9] = [0.2, 0.9, 1.0]
    block_features[:, 12] = [0.2, 0.7, 0.1]
    mask = np.array([True, True, True])

    assert choose_area_weighted_action(block_features, mask) == 1


def test_choose_contiguity_aware_action_adds_spatial_coordination_terms():
    block_features = np.zeros((3, 17), dtype=np.float32)
    block_features[:, 3] = [0.7, 0.7, 0.1]
    block_features[:, 11] = [0.1, 0.8, 0.0]
    block_features[:, 13] = [0.0, 0.6, 0.0]
    mask = np.array([True, True, True])

    assert choose_contiguity_aware_action(block_features, mask) == 1


def test_choose_immediate_slope_delta_action_uses_global_area_weighted_slope():
    class FakeEnv:
        block_parcels = [np.array([0, 1]), np.array([2, 3])]
        land_use = np.array([1, 2, 1, 2])
        swapped = np.array([False, False, False, False])
        slopes = np.array([6.0, 5.0, 10.0, 1.0])
        areas = np.array([100.0, 1.0, 1.0, 1.0])
        total_weighted_slope = 610.0
        total_farm_area = 101.0
        swaps_per_step = 1

    mask = np.array([True, True])

    assert choose_immediate_slope_delta_action(FakeEnv(), mask) == 1


def test_summarize_policy_runs_reports_mean_std_and_budget_completion():
    runs = [
        {"slope_change_pct": -1.0, "reward": 10.0, "budget_used": 500, "budget": 500},
        {"slope_change_pct": -2.0, "reward": 20.0, "budget_used": 400, "budget": 500},
    ]

    summary = summarize_policy_runs("random", runs)

    assert summary["policy"] == "random"
    assert summary["n_runs"] == 2
    assert summary["slope_change_pct_mean"] == pytest.approx(-1.5)
    assert summary["budget_completion_mean"] == pytest.approx(0.9)
