import json
from pathlib import Path

import pytest

from paper7.manuscript_evidence_ledger import (
    build_manuscript_evidence_ledger,
    render_ledger_markdown,
    write_manuscript_evidence_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_AUDIT_PATH = (
    REPO_ROOT / "paper7" / "results" / "revision" / "end_to_end_validation.json"
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_ledger_from_real_audit_contains_required_claims():
    ledger = build_manuscript_evidence_ledger(REAL_AUDIT_PATH, repo_root=REPO_ROOT)
    rows = {row["claim_id"]: row for row in ledger["claims"]}

    expected = {
        "bishan_learned_environment_e2e",
        "calibration_effect",
        "reward_scaling_comparator",
        "transition_surrogate_diagnostics",
        "planning_tradeoff_boundary",
        "dongxing_local_counterpart",
        "direct_transfer_boundary",
        "reward_weight_replay_boundary",
    }

    assert expected.issubset(rows)
    assert rows["calibration_effect"]["metrics"]["one_sided_p"] == pytest.approx(0.011963)
    assert rows["calibration_effect"]["metrics"]["two_sided_p"] == pytest.approx(0.023926)
    assert rows["calibration_effect"]["metrics"]["improvement_pct"] == pytest.approx(
        12.993898
    )
    assert (
        rows["direct_transfer_boundary"]["claim_strength"]
        == "structural_boundary"
    )
    assert ledger["overall_status"] == "supported_with_bounded_external_scope"
    assert "observational reward regularization" in ledger["required_boundaries"]
    assert "not definitive causal identification" in ledger["required_boundaries"]


def test_real_ledger_artifact_paths_exist():
    ledger = build_manuscript_evidence_ledger(REAL_AUDIT_PATH, repo_root=REPO_ROOT)

    missing = []
    for row in ledger["claims"]:
        for rel_path in row["artifact_paths"]:
            path = REPO_ROOT / rel_path
            if not path.exists():
                missing.append(f"{row['claim_id']} -> {rel_path}")

    assert missing == []


def test_markdown_renderer_includes_claim_ids_metrics_and_boundaries():
    ledger = build_manuscript_evidence_ledger(REAL_AUDIT_PATH, repo_root=REPO_ROOT)
    markdown = render_ledger_markdown(ledger)

    assert "# Paper 7 Manuscript Evidence Ledger" in markdown
    assert "bishan_learned_environment_e2e" in markdown
    assert "calibration_effect" in markdown
    assert "one_sided_p" in markdown
    assert "not direct Bishan-to-Dongxing policy transfer" in markdown
    assert "fixed-policy replay" in markdown


def test_write_ledger_outputs_machine_and_human_readable_files(tmp_path):
    json_out = tmp_path / "manuscript_evidence_ledger.json"
    md_out = tmp_path / "manuscript_evidence_ledger.md"

    written = write_manuscript_evidence_ledger(
        audit_path=REAL_AUDIT_PATH,
        json_out=json_out,
        md_out=md_out,
        repo_root=REPO_ROOT,
    )

    assert written["json"] == json_out
    assert written["markdown"] == md_out
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    markdown = md_out.read_text(encoding="utf-8")
    assert payload["claims"]
    assert "Dongxing" in markdown


def test_fixture_ledger_keeps_claim_strengths_bounded(tmp_path):
    audit_path = tmp_path / "audit.json"
    _write_json(
        audit_path,
        {
            "overall_status": "supported_with_bounded_external_scope",
            "evidence": {
                "transition_training": {
                    "status": "supported",
                    "model": {"path": "paper7/models/transition_model.pt"},
                    "history": {"path": "paper7/models/training_history.json"},
                    "final_val_obs_cosine": 0.999794,
                    "final_val_reward_mse": 1.096993,
                },
                "bishan_seed_chain": {
                    "status": "supported",
                    "seed_dir": "paper7/results/revision/seeds",
                    "n_paired_seeds": 15,
                    "no_cal_mean": -0.975683,
                    "with_cal_mean": -1.102462,
                    "improvement_pct": 12.993898,
                    "paired_slope_test": {
                        "one_sided_p": 0.011963,
                        "two_sided_p": 0.023926,
                    },
                },
                "reward_scaling_comparator": {
                    "path": "paper7/results/revision/reward_scaling_comparator.json",
                    "best_scale": 0.2,
                    "pre_specified_scale": 0.185,
                    "pre_vs_best_relative_gap_pct": 3.12135,
                    "pre_vs_unscaled_slope_gain_pct": 22.144879,
                },
                "alpha_grid": {
                    "grid_path": "paper7/results/revision/alpha_grid/grid_results.json",
                    "n_runs": 40,
                    "n_alphas": 8,
                },
                "transition_rollout": {
                    "path": "paper7/results/revision/transition_rollout_diagnostics.json",
                    "horizon_100_mask_agreement": 0.997384,
                    "horizon_100_reward_mae": 0.234012,
                },
                "policy_induced_diagnostics": {
                    "path": "paper7/results/revision/policy_induced_diagnostics_15seed.json",
                    "n_policy_episodes": 15,
                    "policy_induced_mask_agreement_mean": 0.997629,
                    "policy_induced_support_distance_mean": 0.012601,
                },
                "calibration": {
                    "calibration_path": "paper7/results/causal_calibration.json",
                    "sensitivity_path": "paper7/results/revision/causal_sensitivity_diagnostics.json",
                    "calibration_factor": 0.185005,
                    "interpretation": "observational treatment-effect-informed reward regularization; not definitive causal identification",
                },
                "planning_significance": {
                    "path": "paper7/results/revision/planning_significance_audit.json",
                    "contiguity_delta_with_minus_no_mean": -0.001977,
                    "baimu_count_delta_with_minus_no_mean": -0.2,
                    "baimu_area_delta_with_minus_no_mean": -67.208419,
                },
                "reward_weight_sensitivity": {
                    "path": "paper7/results/full_rigor/reward_weight_sensitivity.json",
                    "n_weight_settings": 14,
                    "policy_retraining_under_all_weights": False,
                },
                "bishan_non_learning_baselines": {
                    "path": "paper7/results/revision/bishan_strong_baselines.json"
                },
                "dongxing_full_baselines": {
                    "path": "paper7/results/full_rigor/dongxing_full_baselines.json",
                    "status": "supported_as_full_real_environment_baselines",
                },
                "dongxing_full_learned_policy": {
                    "path": "paper7/results/full_rigor/dongxing_full_learned_policy.json",
                    "n_eval_seeds": 10,
                },
                "dongxing_mbrl_results": {
                    "path": "paper7/results/full_rigor/dongxing_mbrl_results.json",
                    "status": "supported_as_local_dongxing_mbrl_results",
                    "multi_step_mbrl_planning_tested": True,
                },
                "dongxing_multistep_mbrl_policy": {
                    "path": "paper7/results/full_rigor/dongxing_multistep_mbrl_policy.json",
                    "real_eval_reward_mean": 61.287306,
                    "real_eval_slope_change_pct_mean": -1.882392,
                },
                "transfer_finetune_results": {
                    "path": "paper7/results/full_rigor/transfer_finetune_results.json",
                    "status": "structurally_invalid_for_direct_policy_transfer",
                    "dimension_mismatch": {
                        "observation_dim_match": False,
                        "action_dim_match": False,
                    },
                },
            },
        },
    )

    ledger = build_manuscript_evidence_ledger(audit_path, repo_root=REPO_ROOT)
    rows = {row["claim_id"]: row for row in ledger["claims"]}

    assert rows["calibration_effect"]["claim_strength"] == "supported_bounded"
    assert rows["dongxing_local_counterpart"]["claim_strength"] == "supported_bounded"
    assert rows["reward_weight_replay_boundary"]["metrics"][
        "policy_retraining_under_all_weights"
    ] is False
