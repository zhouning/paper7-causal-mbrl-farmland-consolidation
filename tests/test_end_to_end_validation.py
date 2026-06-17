import json
import subprocess
import sys
from pathlib import Path

import pytest

from paper7.end_to_end_validation import (
    classify_claim_scope,
    summarize_alpha_grid,
    summarize_policy_induced_diagnostics,
    summarize_seed_evaluations,
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


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
