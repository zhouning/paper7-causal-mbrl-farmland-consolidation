import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from paper7.policy_induced_diagnostics import (
    compute_nearest_neighbor_distances,
    compute_policy_step_metrics,
    summarize_policy_diagnostics,
    summarize_episode_metrics,
)


def test_compute_nearest_neighbor_distances_reports_distance_to_training_support():
    query = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32)
    support = np.array([[0.0, 0.0], [1.0, 0.0], [4.0, 0.0]], dtype=np.float32)

    distances = compute_nearest_neighbor_distances(query, support)

    assert distances.tolist() == pytest.approx([0.0, 1.0])


def test_compute_policy_step_metrics_compares_learned_and_real_states():
    learned_block = np.zeros((3, 4), dtype=np.float32)
    real_block = learned_block.copy()
    learned_global = np.array([1.0, 3.0], dtype=np.float32)
    real_global = np.array([2.0, 1.0], dtype=np.float32)
    learned_mask = np.array([True, False, True])
    real_mask = np.array([True, True, False])

    metrics = compute_policy_step_metrics(
        step=4,
        action=2,
        learned_block=learned_block,
        real_block=real_block,
        learned_global=learned_global,
        real_global=real_global,
        learned_reward=1.25,
        real_reward=1.0,
        learned_mask=learned_mask,
        real_mask=real_mask,
        support_distance=0.5,
        reward_scale=0.2,
    )

    assert metrics["step"] == 4
    assert metrics["action"] == 2
    assert metrics["global_mae"] == pytest.approx(1.5)
    assert metrics["reward_abs_error"] == pytest.approx(0.25)
    assert metrics["calibrated_reward_abs_error"] == pytest.approx(0.05)
    assert metrics["mask_agreement"] == pytest.approx(1 / 3)
    assert metrics["support_distance"] == pytest.approx(0.5)


def test_summarize_episode_metrics_preserves_final_real_outcomes():
    step_metrics = [
        {
            "global_mae": 0.1,
            "reward_abs_error": 0.2,
            "calibrated_reward_abs_error": 0.02,
            "mask_agreement": 1.0,
            "support_distance": 0.3,
        },
        {
            "global_mae": 0.3,
            "reward_abs_error": 0.4,
            "calibrated_reward_abs_error": 0.04,
            "mask_agreement": 0.5,
            "support_distance": 0.7,
        },
    ]
    final_info = {"slope_change_pct": -1.2, "cont_change": 0.01, "budget_used": 500}

    summary = summarize_episode_metrics(step_metrics, final_info)

    assert summary["n_steps"] == 2
    assert summary["global_mae_mean"] == pytest.approx(0.2)
    assert summary["reward_mae_mean"] == pytest.approx(0.3)
    assert summary["calibrated_reward_mae_mean"] == pytest.approx(0.03)
    assert summary["mask_agreement_mean"] == pytest.approx(0.75)
    assert summary["support_distance_mean"] == pytest.approx(0.5)
    assert summary["final_real_slope_change_pct"] == pytest.approx(-1.2)


def test_summarize_policy_diagnostics_aggregates_block_error_metrics():
    episodes = [
        {
            "summary": {
                "selected_block_mae_mean": 0.07,
                "all_block_mae_mean": 0.002,
                "global_mae_mean": 0.05,
                "reward_mae_mean": 0.6,
                "calibrated_reward_mae_mean": 0.11,
                "mask_agreement_mean": 0.997,
                "support_distance_mean": 0.01,
                "support_distance_q95": 0.015,
                "final_real_slope_change_pct": -1.1,
            }
        },
        {
            "summary": {
                "selected_block_mae_mean": 0.09,
                "all_block_mae_mean": 0.004,
                "global_mae_mean": 0.07,
                "reward_mae_mean": 0.8,
                "calibrated_reward_mae_mean": 0.15,
                "mask_agreement_mean": 0.999,
                "support_distance_mean": 0.03,
                "support_distance_q95": 0.035,
                "final_real_slope_change_pct": -1.3,
            }
        },
    ]

    aggregate = summarize_policy_diagnostics(episodes)

    assert aggregate["selected_block_mae_mean_mean"] == pytest.approx(0.08)
    assert aggregate["all_block_mae_mean_mean"] == pytest.approx(0.003)
    assert aggregate["global_mae_mean_mean"] == pytest.approx(0.06)


def test_policy_induced_diagnostics_script_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "paper7/policy_induced_diagnostics.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--policy-models" in result.stdout
