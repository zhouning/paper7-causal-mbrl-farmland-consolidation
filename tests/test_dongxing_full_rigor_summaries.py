import json
import subprocess
import sys
from pathlib import Path

import pytest

from paper7.dongxing_full_rigor_summaries import (
    build_dongxing_mbrl_results_summary,
    build_dongxing_trajectory_summary,
    build_transfer_finetune_summary,
    write_full_rigor_summaries,
)


def test_build_dongxing_trajectory_summary_compacts_transition_diagnostics():
    summary = build_dongxing_trajectory_summary(
        {
            "status": "supported_as_dongxing_full_transition_diagnostic",
            "n_transitions": 3000,
            "policies": [
                "random",
                "dynamic_slope_gap",
                "area_weighted_slope_gap",
            ],
            "seeds": [0, 1, 2, 3, 4],
            "feature_dims": {"selected_block": 8, "global": 8},
            "model": {"reward_mae": 0.887175},
            "policy_holdout_count": 6,
            "policy_holdout_reward_beats_baseline_count": 2,
            "mbrl_policy_trained": False,
            "policy_transfer_tested": False,
        }
    )

    assert summary["status"] == "supported_as_dongxing_trajectory_summary"
    assert summary["n_transitions"] == 3000
    assert summary["n_policies"] == 3
    assert summary["n_seeds"] == 5
    assert summary["policy_holdout_reward_beats_baseline_count"] == 2
    assert summary["mbrl_policy_trained"] is False


def test_build_dongxing_mbrl_results_summary_compacts_local_mbrl_evidence():
    summary = build_dongxing_mbrl_results_summary(
        {
            "status": "supported_as_dongxing_full_transition_diagnostic",
            "n_transitions": 3000,
            "random_split_reward_mae": 0.887175,
            "policy_holdout_count": 6,
            "policy_holdout_reward_beats_baseline_count": 2,
        },
        {
            "status": "supported_as_dongxing_full_one_step_model_based_policy",
            "n_training_transitions": 3000,
            "n_eval_seeds": 10,
            "planning_horizon": 1,
            "model_based_reward_mean": 72.665825,
            "model_based_slope_change_pct_mean": -1.999939,
            "model_based_cont_change_mean": -0.041937,
            "model_based_baimu_area_change_ha_mean": -2698.58047,
            "comparisons": {
                "model_based_minus_scalarized_default_reward_mean": 35.871697,
                "model_based_minus_baimu_aware_reward_mean": -59.064267,
            },
            "mbrl_transition_model_used": True,
            "policy_transfer_tested": False,
            "multi_step_mbrl_planning_tested": False,
        },
        {
            "status": "supported_as_dongxing_model_based_scoring_optimization",
            "n_training_transitions": 3000,
            "n_candidates": 10,
            "best_candidate": "reward_slope_bonus_x2",
            "n_eval_seeds": 5,
            "heldout_reward_mean": 116.547453,
            "heldout_slope_change_pct_mean": -2.237549,
            "heldout_cont_change_mean": -0.036515,
            "heldout_baimu_area_change_ha_mean": -2506.355936,
            "beats_scalarized_default_reward": True,
            "beats_baimu_aware_reward": False,
            "selection_eval_split": True,
        },
        {
            "status": "supported_as_dongxing_multistep_learned_environment_policy",
            "n_training_transitions": 3000,
            "n_eval_seeds": 5,
            "planning_horizon": 100,
            "real_eval_reward_mean": 88.0,
            "real_eval_slope_change_pct_mean": -2.1,
            "mbrl_transition_model_used": True,
            "multi_step_mbrl_planning_tested": True,
            "policy_transfer_tested": False,
        },
        {
            "status": "supported_as_dongxing_scenario_robustness",
            "scenario_count": 3,
            "policy_summaries": {
                "scenario_robust_mbrl": {
                    "reward_mean": 12.0,
                    "reward_worst": 8.0,
                    "slope_change_pct_mean": -1.2,
                    "scenario_count": 3,
                }
            },
            "deterministic_seed_repetition_avoided": True,
            "policy_transfer_tested": False,
        },
    )

    assert summary["status"] == "supported_as_local_dongxing_mbrl_results"
    assert summary["transition_diagnostics"]["n_transitions"] == 3000
    assert summary["full_model_based_policy"]["n_eval_seeds"] == 10
    assert summary["model_based_optimization"]["best_candidate"] == "reward_slope_bonus_x2"
    assert summary["multistep_mbrl_policy"]["planning_horizon"] == 100
    assert summary["mbrl_transition_model_used"] is True
    assert summary["policy_transfer_tested"] is False
    assert summary["multi_step_mbrl_planning_tested"] is True
    assert summary["scenario_robustness"]["scenario_count"] == 3
    assert summary["scenario_robustness"]["deterministic_seed_repetition_avoided"] is True


def test_build_transfer_finetune_summary_marks_direct_transfer_structurally_invalid():
    summary = build_transfer_finetune_summary(
        {
            "n_blocks": 2600,
            "k_block": 17,
            "k_global": 12,
            "observation_dim": 44212,
            "action_dim": 2600,
        },
        {
            "n_blocks": 2978,
            "k_block": 8,
            "k_global": 8,
            "observation_dim": 23832,
            "action_dim": 2978,
        },
    )

    assert summary["status"] == "structurally_invalid_for_direct_policy_transfer"
    assert summary["direct_policy_transfer_tested"] is False
    assert summary["fine_tuning_tested"] is False
    assert summary["dimension_mismatch"]["observation_dim_match"] is False
    assert summary["dimension_mismatch"]["action_dim_match"] is False


def test_write_full_rigor_summaries_writes_expected_files(tmp_path):
    repo_root = tmp_path
    full_rigor = repo_root / "paper7" / "results" / "full_rigor"
    revision = repo_root / "paper7" / "results" / "revision"
    full_rigor.mkdir(parents=True, exist_ok=True)
    revision.mkdir(parents=True, exist_ok=True)

    (full_rigor / "dongxing_transition_diagnostics.json").write_text(
        json.dumps(
            {
                "status": "supported_as_dongxing_full_transition_diagnostic",
                "n_transitions": 12,
                "policies": ["random", "dynamic_slope_gap"],
                "seeds": [0, 1],
                "feature_dims": {"selected_block": 8, "global": 8},
                "model": {
                    "reward_mae": 0.5,
                    "reward_persistence_mae": 0.75,
                },
                "policy_holdout_diagnostics": [
                    {
                        "holdout_policy": "random",
                        "reward_mae": 0.9,
                        "reward_persistence_mae": 1.0,
                    },
                    {
                        "holdout_policy": "dynamic_slope_gap",
                        "reward_mae": 0.7,
                        "reward_persistence_mae": 0.8,
                    },
                    {
                        "holdout_policy": "area_weighted_slope_gap",
                        "reward_mae": 0.6,
                        "reward_persistence_mae": 0.55,
                    },
                ],
                "policy_holdout_count": 2,
                "policy_holdout_reward_beats_baseline_count": 1,
                "mbrl_policy_trained": False,
                "policy_transfer_tested": False,
            }
        ),
        encoding="utf-8",
    )
    (full_rigor / "dongxing_full_model_based_policy.json").write_text(
        json.dumps(
            {
                "status": "supported_as_dongxing_full_one_step_model_based_policy",
                "n_training_transitions": 12,
                "model_based_policy": {"summary": {"n": 3, "reward_mean": 7.0}},
                "comparisons": {},
                "mbrl_transition_model_used": True,
                "policy_transfer_tested": False,
            }
        ),
        encoding="utf-8",
    )
    (full_rigor / "dongxing_model_based_optimization.json").write_text(
        json.dumps(
            {
                "status": "supported_as_dongxing_model_based_scoring_optimization",
                "n_training_transitions": 12,
                "n_candidates": 2,
                "best_candidate": {"name": "reward_slope_bonus_x2"},
                "heldout_eval": {"summary": {"n": 2, "reward_mean": 9.0}},
                "comparisons": {},
                "selection_eval_split": True,
                "mbrl_transition_model_used": True,
                "policy_transfer_tested": False,
            }
        ),
        encoding="utf-8",
    )
    (full_rigor / "dongxing_multistep_mbrl_policy.json").write_text(
        json.dumps(
            {
                "status": "supported_as_dongxing_multistep_learned_environment_policy",
                "n_training_transitions": 12,
                "planning_horizon": 2,
                "real_environment_eval": {"summary": {"n": 2, "reward_mean": 8.0}},
                "mbrl_transition_model_used": True,
                "multi_step_mbrl_planning_tested": True,
                "policy_transfer_tested": False,
            }
        ),
        encoding="utf-8",
    )
    (full_rigor / "dongxing_full_env_smoke.json").write_text(
        json.dumps(
            {
                "status": "constructed",
                "n_parcels": 76378,
                "n_blocks": 2978,
                "valid_action_count": 100,
            }
        ),
        encoding="utf-8",
    )
    (revision / "transition_rollout_diagnostics.json").write_text(
        json.dumps({"n_blocks": 2600}),
        encoding="utf-8",
    )

    outputs = write_full_rigor_summaries(repo_root=repo_root)

    expected = {
        "dongxing_trajectories_summary",
        "dongxing_mbrl_results",
        "transfer_finetune_results",
    }
    assert set(outputs) == expected
    for path in outputs.values():
        assert path.exists()
        assert path.suffix == ".json"

    summary = json.loads(outputs["dongxing_trajectories_summary"].read_text(encoding="utf-8"))
    assert summary["random_split_reward_mae"] == pytest.approx(0.5)
    assert summary["policy_holdout_count"] == 3
    assert summary["policy_holdout_reward_beats_baseline_count"] == 2
    mbrl_summary = json.loads(outputs["dongxing_mbrl_results"].read_text(encoding="utf-8"))
    assert mbrl_summary["multi_step_mbrl_planning_tested"] is True


def test_dongxing_full_rigor_summaries_script_help_imports_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "paper7/dongxing_full_rigor_summaries.py", "--help"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--repo-root" in result.stdout
