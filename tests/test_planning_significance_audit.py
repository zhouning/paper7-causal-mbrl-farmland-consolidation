from paper7.planning_significance_audit import (
    concentration_metrics,
    summarize_policy_rows,
)


def test_concentration_metrics_reports_unique_share_and_top_share():
    selected = [1, 1, 2, 3]

    metrics = concentration_metrics(selected)

    assert metrics["n_actions"] == 4
    assert metrics["n_unique_blocks"] == 3
    assert metrics["unique_share"] == 0.75
    assert metrics["top1_share"] == 0.5


def test_summarize_policy_rows_handles_core_planning_fields():
    rows = [
        {
            "slope_change_pct": -1.0,
            "cont_change": 0.01,
            "baimu_count_change": 1,
            "baimu_area_change_ha": -10,
            "reward": 20,
        },
        {
            "slope_change_pct": -1.2,
            "cont_change": 0.03,
            "baimu_count_change": 3,
            "baimu_area_change_ha": -20,
            "reward": 30,
        },
    ]

    summary = summarize_policy_rows(rows)

    assert summary["n"] == 2
    assert summary["slope_change_pct_mean"] == -1.1
    assert summary["cont_change_mean"] == 0.02
    assert summary["baimu_count_change_mean"] == 2.0
