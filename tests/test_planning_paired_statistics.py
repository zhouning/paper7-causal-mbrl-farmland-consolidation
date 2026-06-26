from pathlib import Path

import pytest

from paper7.end_to_end_validation import summarize_seed_evaluations
from paper7.planning_significance_audit import _paired_deltas, exact_sign_flip_test


def test_exact_sign_flip_test_reports_seed_paired_one_sided_result():
    result = exact_sign_flip_test([-2.0, -1.0, 0.5], alternative="less")

    assert result["paired_test"] == "exact_sign_flip"
    assert result["alternative"] == "less"
    assert result["n_pairs"] == 3
    assert result["n_nonzero_pairs"] == 3
    assert result["negative_delta_count"] == 2
    assert result["positive_delta_count"] == 1
    assert result["zero_delta_count"] == 0
    assert result["one_sided_p"] == 0.25
    assert result["two_sided_p"] == 0.5
    assert "Mann-Whitney" in result["interpretation_boundary"]


def test_paired_deltas_include_slope_paired_test():
    with_cal = [
        {"seed": 0, "slope_change_pct": -1.2, "reward_real": 10.0},
        {"seed": 1, "slope_change_pct": -1.0, "reward_real": 11.0},
        {"seed": 2, "slope_change_pct": -0.9, "reward_real": 12.0},
    ]
    no_cal = [
        {"seed": 0, "slope_change_pct": -1.0, "reward_real": 9.0},
        {"seed": 1, "slope_change_pct": -0.8, "reward_real": 12.0},
        {"seed": 2, "slope_change_pct": -1.1, "reward_real": 11.0},
    ]

    summary = _paired_deltas(with_cal, no_cal)

    test = summary["slope_change_pct_delta_with_minus_no_paired_test"]
    assert test["paired_test"] == "exact_sign_flip"
    assert test["alternative"] == "less"
    assert test["negative_delta_count"] == 2
    assert test["positive_delta_count"] == 1


def test_summarize_seed_evaluations_exposes_paired_calibration_test(tmp_path):
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    for seed, no_cal, with_cal in [
        (0, 0.0, -2.0),
        (1, 0.0, -1.0),
        (2, 0.0, 0.5),
    ]:
        (seed_dir / f"no_cal_eval_seed{seed}.json").write_text(
            f'{{"slope_change_pct": {no_cal}}}',
            encoding="utf-8",
        )
        (seed_dir / f"with_cal_eval_seed{seed}.json").write_text(
            f'{{"slope_change_pct": {with_cal}}}',
            encoding="utf-8",
        )

    summary = summarize_seed_evaluations(seed_dir)

    assert summary["paired_test_method"] == "exact_sign_flip"
    assert summary["calibration_slope_one_sided_p"] == pytest.approx(0.25)
    assert summary["calibrated_slope_win_count"] == 2
    assert summary["uncalibrated_slope_win_count"] == 1


def test_batch_revision_summary_uses_paired_test_not_mann_whitney():
    source = Path("paper7/batch_revision.py").read_text(encoding="utf-8")

    assert "mannwhitneyu" not in source
    assert "mann_whitney_U" not in source
    assert "mann_whitney_p" not in source
    assert "Mann-Whitney U" not in source
    assert "exact_sign_flip_test" in source
