from paper7.reward_scaling_comparator import (
    compare_reward_scales,
    summarize_by_scale,
)


def test_summarize_by_scale_groups_seed_rows_and_prefers_lower_slope():
    rows = [
        {
            "reward_scale": 0.1,
            "seed": 0,
            "slope_change_pct": -1.0,
            "reward_real": 10.0,
        },
        {
            "reward_scale": 0.1,
            "seed": 1,
            "slope_change_pct": -1.2,
            "reward_real": 12.0,
        },
        {
            "reward_scale": 1.0,
            "seed": 0,
            "slope_change_pct": -0.5,
            "reward_real": 8.0,
        },
    ]

    summary = summarize_by_scale(rows)

    assert summary["0.100"]["n"] == 2
    assert summary["0.100"]["slope_change_pct_mean"] == -1.1
    assert summary["1.000"]["slope_change_pct_mean"] == -0.5


def test_compare_reward_scales_reports_pre_specified_gap_and_rank():
    rows = [
        {
            "reward_scale": 0.1,
            "seed": 0,
            "slope_change_pct": -1.0,
            "reward_real": 10.0,
        },
        {
            "reward_scale": 0.185,
            "seed": 0,
            "slope_change_pct": -1.2,
            "reward_real": 11.0,
        },
        {
            "reward_scale": 0.2,
            "seed": 0,
            "slope_change_pct": -1.3,
            "reward_real": 12.0,
        },
        {
            "reward_scale": 1.0,
            "seed": 0,
            "slope_change_pct": -0.6,
            "reward_real": 8.0,
        },
    ]

    comparison = compare_reward_scales(rows, pre_specified_alpha=0.185)

    assert comparison["best_scale"] == 0.2
    assert comparison["pre_specified_scale"] == 0.185
    assert comparison["pre_specified_rank_by_slope"] == 2
    assert comparison["unscaled_scale"] == 1.0
    assert comparison["pre_vs_unscaled_slope_gain_pct"] > 0
