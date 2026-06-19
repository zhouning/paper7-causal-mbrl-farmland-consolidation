import json
import subprocess
import sys
from pathlib import Path

import pytest

from paper7.end_to_end_validation import (
    classify_claim_scope,
    load_json,
    select_policy_induced_diagnostics_path,
    summarize_alpha_grid,
    summarize_dongxing_rl_lite,
    summarize_dongxing_full_baselines,
    summarize_dongxing_full_learned_policy,
    summarize_dongxing_full_model_based_policy,
    summarize_dongxing_model_based_optimization,
    summarize_dongxing_transition_diagnostics,
    summarize_policy_induced_diagnostics,
    summarize_planning_significance,
    summarize_reward_scaling_comparator,
    summarize_reward_weight_sensitivity,
    summarize_seed_evaluations,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_json_accepts_utf8_bom_files(tmp_path):
    path = tmp_path / "bom.json"
    path.write_text('{"status": "ok"}', encoding="utf-8-sig")

    assert load_json(path) == {"status": "ok"}


def test_summarize_seed_evaluations_requires_balanced_calibrated_pairs(tmp_path):
    seed_dir = tmp_path / "seeds"
    for seed, no_cal, with_cal in [
        (0, -1.0, -1.2),
        (1, -0.8, -1.0),
        (2, -0.9, -1.1),
    ]:
        _write_json(seed_dir / f"no_cal_eval_seed{seed}.json", {"slope_change_pct": no_cal})
        _write_json(seed_dir / f"with_cal_eval_seed{seed}.json", {"slope_change_pct": with_cal})

    summary = summarize_seed_evaluations(seed_dir)

    assert summary["n_paired_seeds"] == 3
    assert summary["no_cal_mean"] == pytest.approx(-0.9)
    assert summary["with_cal_mean"] == pytest.approx(-1.1)
    assert summary["improvement_pct"] == pytest.approx(22.222222, rel=1e-6)
    assert summary["balanced_pairs"] is True


def test_summarize_alpha_grid_reports_pre_specified_distance_to_best(tmp_path):
    grid_path = tmp_path / "grid_results.json"
    rows = [
        {"reward_scale": 0.185, "slope_change_pct": -1.0},
        {"reward_scale": 0.185, "slope_change_pct": -1.2},
        {"reward_scale": 0.2, "slope_change_pct": -1.25},
        {"reward_scale": 0.2, "slope_change_pct": -1.15},
        {"reward_scale": 1.0, "slope_change_pct": -0.8},
    ]
    _write_json(grid_path, rows)

    summary = summarize_alpha_grid(grid_path, pre_specified_alpha=0.185)

    assert summary["n_runs"] == 5
    assert summary["best_alpha"] == pytest.approx(0.2)
    assert summary["pre_specified_alpha"] == pytest.approx(0.185)
    assert summary["best_slope_mean"] == pytest.approx(-1.2)
    assert summary["pre_specified_slope_mean"] == pytest.approx(-1.1)
    assert summary["relative_gap_pct"] == pytest.approx(8.333333, rel=1e-6)


def test_summarize_policy_induced_diagnostics_reports_policy_shift_metrics(tmp_path):
    path = tmp_path / "policy_induced_diagnostics.json"
    _write_json(
        path,
        {
            "support_size": 12000,
            "aggregate": {
                "n_episodes": 3,
                "selected_block_mae_mean_mean": 0.075143,
                "all_block_mae_mean_mean": 0.0027,
                "global_mae_mean_mean": 0.0521,
                "reward_mae_mean_mean": 0.601632,
                "calibrated_reward_mae_mean_mean": 0.111302,
                "mask_agreement_mean_mean": 0.997491,
                "support_distance_mean_mean": 0.011306,
                "support_distance_q95_mean": 0.016308,
                "final_real_slope_change_pct_mean": -1.10901,
            },
            "validation": {
                "passes_all_thresholds": True,
                "passes_mask_agreement_threshold": True,
                "passes_support_distance_threshold": True,
                "passes_reward_calibration_check": True,
            },
        },
    )

    summary = summarize_policy_induced_diagnostics(path)

    assert summary["status"] == "supported"
    assert summary["n_policy_episodes"] == 3
    assert summary["policy_induced_selected_block_mae_mean"] == pytest.approx(0.075143)
    assert summary["policy_induced_all_block_mae_mean"] == pytest.approx(0.0027)
    assert summary["policy_induced_global_mae_mean"] == pytest.approx(0.0521)
    assert summary["policy_induced_calibrated_reward_mae_mean"] == pytest.approx(0.111302)
    assert summary["policy_induced_final_real_slope_pct_mean"] == pytest.approx(-1.10901)
    assert summary["validation_passes_all_thresholds"] is True


def test_select_policy_induced_diagnostics_path_prefers_15_seed_file(tmp_path):
    revision_dir = tmp_path / "results" / "revision"
    legacy = revision_dir / "policy_induced_diagnostics.json"
    expanded = revision_dir / "policy_induced_diagnostics_15seed.json"
    _write_json(legacy, {"aggregate": {"n_episodes": 3}})
    _write_json(expanded, {"aggregate": {"n_episodes": 15}})

    selected = select_policy_induced_diagnostics_path(tmp_path)

    assert selected == expanded


def test_classify_claim_scope_marks_dongxing_as_feasibility_not_policy_transfer():
    evidence = {
        "bishan_seed_chain": {"status": "supported", "n_paired_seeds": 15},
        "policy_induced_diagnostics": {
            "status": "supported",
            "n_policy_episodes": 3,
        },
        "dongxing_dynamic": {"status": "supported", "has_learned_policy": False},
    }

    claims = classify_claim_scope(evidence)
    dongxing = next(item for item in claims if item["id"] == "dongxing_external_scope")
    policy_shift = next(item for item in claims if item["id"] == "policy_induced_surrogate_scope")

    assert dongxing["status"] == "supported_as_external_feasibility"
    assert dongxing["policy_transfer_tested"] is False
    assert "not learned-policy transfer" in dongxing["interpretation"]
    assert policy_shift["status"] == "supported"
    assert policy_shift["n_policy_episodes"] == 3


def test_classify_claim_scope_marks_dongxing_rl_lite_as_slope_only_not_transfer():
    evidence = {
        "dongxing_dynamic": {"status": "supported", "has_learned_policy": False},
        "dongxing_rl_lite": {
            "status": "supported_as_slope_only_rl_actionability",
            "learner_type": "tabular_preference_fallback",
        },
    }

    claims = classify_claim_scope(evidence)
    dongxing = next(item for item in claims if item["id"] == "dongxing_external_scope")

    assert dongxing["policy_transfer_tested"] is False
    assert dongxing["slope_only_rl_actionability_tested"] is True
    assert dongxing["status"] == "supported_as_external_slope_only_actionability"


def test_reward_rigor_scope_is_bounded_when_weight_sensitivity_exists():
    evidence = {
        "reward_weight_sensitivity": {
            "status": "supported_as_fixed_policy_reward_sensitivity",
            "n_episodes": 60,
            "n_weight_settings": 14,
            "interpretation": "fixed-policy reward-component replay",
        }
    }

    scopes = classify_claim_scope(evidence)
    reward_scope = next(item for item in scopes if item["id"] == "reward_function_scope")

    assert reward_scope["status"] == "supported_as_fixed_policy_reward_sensitivity"
    assert reward_scope["policy_retraining_under_all_weights"] is False


def test_new_evidence_summarizers_extract_core_metrics(tmp_path):
    reward_path = tmp_path / "reward.json"
    planning_path = tmp_path / "planning.json"
    dongxing_path = tmp_path / "dongxing.json"
    _write_json(
        reward_path,
        {
            "pre_specified_rank_by_slope": 2,
            "pre_vs_best_relative_gap_pct": 3.1,
            "pre_vs_unscaled_slope_gain_pct": 22.1,
        },
    )
    _write_json(
        planning_path,
        {
            "calibrated_policy": {"slope_change_pct_mean": -1.1},
            "paired_calibration_effects": {
                "n_paired_seeds": 15,
                "slope_change_pct_delta_with_minus_no_mean": -0.12,
            },
            "action_concentration": {"status": "head_only_available"},
        },
    )
    _write_json(
        dongxing_path,
        {
            "status": "supported_as_slope_only_rl_actionability",
            "learner_type": "tabular_preference_fallback",
            "n_blocks": 2978,
            "learned_policy": {"summary": {"slope_change_pct_mean": -2.0}},
            "comparisons": {
                "learned_minus_random_slope_change_pct": -1.5,
                "learned_minus_dynamic_slope_gap_slope_change_pct": -0.7,
            },
        },
    )

    reward = summarize_reward_scaling_comparator(reward_path)
    planning = summarize_planning_significance(planning_path)
    dongxing = summarize_dongxing_rl_lite(dongxing_path)

    assert reward["pre_specified_rank_by_slope"] == 2
    assert planning["n_paired_seeds"] == 15
    assert dongxing["status"] == "supported_as_slope_only_rl_actionability"
    assert dongxing["learned_slope_change_pct_mean"] == -2.0


def test_summarize_reward_weight_sensitivity_extracts_bounded_metrics(tmp_path):
    path = tmp_path / "reward_weight_sensitivity.json"
    _write_json(
        path,
        {
            "n_episodes": 60,
            "n_weight_settings": 14,
            "policy_weight_summaries": [{"policy": "a"}] * 84,
            "policy_metric_summaries": [{"policy": "a"}] * 6,
            "pareto_front": [{"policy": "a"}, {"policy": "b"}],
            "best_policy_by_weight": [{"weight_name": "default", "policy": "a"}],
        },
    )

    summary = summarize_reward_weight_sensitivity(path)

    assert summary["status"] == "supported_as_fixed_policy_reward_sensitivity"
    assert summary["n_episodes"] == 60
    assert summary["n_weight_settings"] == 14
    assert summary["n_policy_metric_summaries"] == 6
    assert summary["n_pareto_rows"] == 2
    assert summary["policy_retraining_under_all_weights"] is False


def test_summarize_dongxing_full_baselines_extracts_scope(tmp_path):
    path = tmp_path / "dongxing_full_baselines.json"
    _write_json(
        path,
        {
            "status": "supported_as_full_real_environment_baseline_pilot",
            "n_runs": 12,
            "n_policies": 6,
            "policy_summaries": {
                "random": {"slope_change_pct_mean": -0.1},
                "scalarized_default": {"slope_change_pct_mean": -0.5},
            },
        },
    )

    summary = summarize_dongxing_full_baselines(path)

    assert summary["status"] == "supported_as_full_real_environment_baseline_pilot"
    assert summary["n_runs"] == 12
    assert summary["has_full_reward_metrics"] is True


def test_classify_claim_scope_marks_dongxing_full_baseline_level():
    scopes = classify_claim_scope(
        {
            "dongxing_full_baselines": {
                "status": "supported_as_full_real_environment_baselines",
                "interpretation": "full baseline evidence",
            }
        }
    )

    dongxing_full = next(
        item for item in scopes if item["id"] == "dongxing_full_real_environment_scope"
    )

    assert dongxing_full["status"] == "supported_as_full_real_environment_baselines"
    assert dongxing_full["evidence_level"] == "external_full_real_environment_baselines"


def test_summarize_dongxing_full_learned_policy_extracts_scope(tmp_path):
    path = tmp_path / "dongxing_full_learned_policy.json"
    _write_json(
        path,
        {
            "status": "supported_as_dongxing_full_reward_learned_policy",
            "learner_type": "linear_preference_full_reward",
            "train_seeds": [0, 1],
            "eval_seeds": [0, 1, 2],
            "learned_policy": {"summary": {"n": 3, "reward_mean": 12.0}},
            "comparisons": {"learned_minus_random_reward_mean": 10.0},
            "claim_boundary": "Local Dongxing full-reward learned actionability",
        },
    )

    summary = summarize_dongxing_full_learned_policy(path)

    assert summary["status"] == "supported_as_dongxing_full_reward_learned_policy"
    assert summary["learner_type"] == "linear_preference_full_reward"
    assert summary["n_eval_seeds"] == 3
    assert summary["learned_policy_tested"] is True
    assert summary["transfer_tested"] is False


def test_classify_claim_scope_marks_dongxing_full_learned_policy_as_local_not_transfer():
    scopes = classify_claim_scope(
        {
            "dongxing_full_learned_policy": {
                "status": "supported_as_dongxing_full_reward_learned_policy",
                "n_eval_seeds": 10,
                "interpretation": "local full-reward learned policy",
            }
        }
    )

    scope = next(item for item in scopes if item["id"] == "dongxing_full_learned_policy_scope")

    assert scope["status"] == "supported_as_dongxing_full_reward_learned_policy"
    assert scope["evidence_level"] == "external_full_reward_local_learned_policy"
    assert scope["policy_transfer_tested"] is False


def test_summarize_dongxing_transition_diagnostics_extracts_bounded_metrics(tmp_path):
    path = tmp_path / "dongxing_transition_diagnostics.json"
    _write_json(
        path,
        {
            "status": "supported_as_dongxing_full_transition_diagnostic",
            "n_transitions": 3000,
            "policies": ["random", "dynamic_slope_gap"],
            "model": {
                "selected_feature_mae": 0.01,
                "selected_feature_persistence_mae": 0.03,
                "global_feature_mae": 0.001,
                "global_feature_persistence_mae": 0.004,
                "reward_mae": 0.8,
                "reward_persistence_mae": 1.0,
            },
            "policy_holdout_diagnostics": [
                {
                    "holdout_policy": "random",
                    "selected_feature_mae": 0.02,
                    "selected_feature_persistence_mae": 0.03,
                    "reward_mae": 2.0,
                    "reward_persistence_mae": 1.0,
                },
                {
                    "holdout_policy": "dynamic_slope_gap",
                    "selected_feature_mae": 0.02,
                    "selected_feature_persistence_mae": 0.03,
                    "reward_mae": 0.8,
                    "reward_persistence_mae": 1.0,
                },
            ],
        },
    )

    summary = summarize_dongxing_transition_diagnostics(path)

    assert summary["status"] == "supported_as_dongxing_full_transition_diagnostic"
    assert summary["n_transitions"] == 3000
    assert summary["random_split_reward_beats_baseline"] is True
    assert summary["policy_holdout_count"] == 2
    assert summary["policy_holdout_reward_beats_baseline_count"] == 1
    assert summary["mbrl_policy_trained"] is False


def test_classify_claim_scope_marks_dongxing_transition_as_diagnostic_not_mbrl():
    scopes = classify_claim_scope(
        {
            "dongxing_transition_diagnostics": {
                "status": "supported_as_dongxing_full_transition_diagnostic",
                "n_transitions": 3000,
                "policy_holdout_reward_beats_baseline_count": 3,
                "policy_holdout_count": 6,
                "interpretation": "transition learnability diagnostic",
            }
        }
    )

    scope = next(item for item in scopes if item["id"] == "dongxing_full_transition_diagnostic_scope")

    assert scope["status"] == "supported_as_dongxing_full_transition_diagnostic"
    assert scope["evidence_level"] == "external_full_transition_learnability_diagnostic"
    assert scope["mbrl_policy_trained"] is False
    assert scope["policy_transfer_tested"] is False


def test_summarize_dongxing_full_model_based_policy_extracts_scope(tmp_path):
    path = tmp_path / "dongxing_full_model_based_policy.json"
    _write_json(
        path,
        {
            "status": "supported_as_dongxing_full_one_step_model_based_policy",
            "n_training_transitions": 3000,
            "planning_horizon": 1,
            "model_based_policy": {
                "summary": {
                    "n": 10,
                    "reward_mean": 72.0,
                    "slope_change_pct_mean": -2.0,
                }
            },
            "comparisons": {
                "model_based_minus_random_reward_mean": 62.0,
                "model_based_minus_scalarized_default_reward_mean": -36.0,
            },
            "mbrl_transition_model_used": True,
            "policy_transfer_tested": False,
            "claim_boundary": "one-step model-based action selection",
        },
    )

    summary = summarize_dongxing_full_model_based_policy(path)

    assert summary["status"] == "supported_as_dongxing_full_one_step_model_based_policy"
    assert summary["n_training_transitions"] == 3000
    assert summary["n_eval_seeds"] == 10
    assert summary["planning_horizon"] == 1
    assert summary["model_based_reward_mean"] == 72.0
    assert summary["mbrl_transition_model_used"] is True
    assert summary["policy_transfer_tested"] is False


def test_classify_claim_scope_marks_model_based_policy_as_one_step_bounded():
    scopes = classify_claim_scope(
        {
            "dongxing_full_model_based_policy": {
                "status": "supported_as_dongxing_full_one_step_model_based_policy",
                "n_eval_seeds": 10,
                "planning_horizon": 1,
                "interpretation": "one-step model-based action selection",
            }
        }
    )

    scope = next(item for item in scopes if item["id"] == "dongxing_full_model_based_policy_scope")

    assert scope["status"] == "supported_as_dongxing_full_one_step_model_based_policy"
    assert scope["evidence_level"] == "external_full_one_step_model_based_policy"
    assert scope["planning_horizon"] == 1
    assert scope["policy_transfer_tested"] is False
    assert scope["multi_step_mbrl_planning_tested"] is False


def test_summarize_dongxing_model_based_optimization_extracts_best_heldout_result(tmp_path):
    path = tmp_path / "dongxing_model_based_optimization.json"
    _write_json(
        path,
        {
            "status": "supported_as_dongxing_model_based_scoring_optimization",
            "n_training_transitions": 3000,
            "selection_eval_split": True,
            "best_candidate": {"name": "reward_slope_bonus_x2"},
            "candidate_selection_summaries": [{"candidate": "a"}, {"candidate": "b"}],
            "heldout_eval": {
                "summary": {
                    "n": 5,
                    "reward_mean": 116.5,
                    "slope_change_pct_mean": -2.2,
                }
            },
            "comparisons": {
                "model_based_minus_scalarized_default_reward_mean": 8.0,
                "model_based_minus_baimu_aware_reward_mean": -15.0,
            },
            "planning_horizon": 1,
            "mbrl_transition_model_used": True,
            "policy_transfer_tested": False,
            "claim_boundary": "held-out one-step scoring optimization",
        },
    )

    summary = summarize_dongxing_model_based_optimization(path)

    assert summary["status"] == "supported_as_dongxing_model_based_scoring_optimization"
    assert summary["best_candidate"] == "reward_slope_bonus_x2"
    assert summary["n_candidates"] == 2
    assert summary["heldout_reward_mean"] == 116.5
    assert summary["beats_scalarized_default_reward"] is True
    assert summary["beats_baimu_aware_reward"] is False
    assert summary["selection_eval_split"] is True


def test_classify_claim_scope_marks_scoring_optimization_as_heldout_bounded():
    scopes = classify_claim_scope(
        {
            "dongxing_model_based_optimization": {
                "status": "supported_as_dongxing_model_based_scoring_optimization",
                "best_candidate": "reward_slope_bonus_x2",
                "n_eval_seeds": 5,
                "selection_eval_split": True,
                "interpretation": "held-out one-step scoring optimization",
            }
        }
    )

    scope = next(item for item in scopes if item["id"] == "dongxing_model_based_optimization_scope")

    assert scope["status"] == "supported_as_dongxing_model_based_scoring_optimization"
    assert scope["evidence_level"] == "external_full_heldout_scoring_optimization"
    assert scope["selection_eval_split"] is True
    assert scope["policy_transfer_tested"] is False
    assert scope["multi_step_mbrl_planning_tested"] is False


def test_end_to_end_validation_script_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "paper7/end_to_end_validation.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--out" in result.stdout
