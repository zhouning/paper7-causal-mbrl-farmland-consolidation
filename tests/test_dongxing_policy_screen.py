from paper7.dongxing_policy_screen import (
    rank_blocks,
    summarize_policy_screen,
)


def test_rank_blocks_orders_by_slope_gap_then_area():
    blocks = [
        {
            "block_id": 1,
            "avg_farm_slope": 8.0,
            "avg_forest_slope": 4.0,
            "farm_area_ha": 2.0,
            "forest_area_ha": 2.0,
        },
        {
            "block_id": 2,
            "avg_farm_slope": 6.0,
            "avg_forest_slope": 1.0,
            "farm_area_ha": 1.0,
            "forest_area_ha": 1.0,
        },
        {
            "block_id": 3,
            "avg_farm_slope": 10.0,
            "avg_forest_slope": 8.0,
            "farm_area_ha": 5.0,
            "forest_area_ha": 5.0,
        },
    ]

    ranked = rank_blocks(blocks, strategy="slope_gap")

    assert [block["block_id"] for block in ranked] == [2, 1, 3]
    assert ranked[0]["slope_gap"] == 5.0


def test_rank_blocks_orders_by_area_weighted_opportunity():
    blocks = [
        {
            "block_id": 1,
            "avg_farm_slope": 8.0,
            "avg_forest_slope": 4.0,
            "farm_area_ha": 1.0,
            "forest_area_ha": 1.0,
        },
        {
            "block_id": 2,
            "avg_farm_slope": 6.0,
            "avg_forest_slope": 3.0,
            "farm_area_ha": 5.0,
            "forest_area_ha": 5.0,
        },
    ]

    ranked = rank_blocks(blocks, strategy="area_weighted_gap")

    assert [block["block_id"] for block in ranked] == [2, 1]
    assert ranked[0]["opportunity_score"] == 15.0


def test_summarize_policy_screen_reports_budgeted_strategies_and_random_baseline():
    blocks = [
        {
            "block_id": 1,
            "unit_id": "a",
            "avg_farm_slope": 8.0,
            "avg_forest_slope": 4.0,
            "farm_area_ha": 2.0,
            "forest_area_ha": 2.0,
            "total_area_ha": 4.0,
        },
        {
            "block_id": 2,
            "unit_id": "b",
            "avg_farm_slope": 6.0,
            "avg_forest_slope": 1.0,
            "farm_area_ha": 1.0,
            "forest_area_ha": 1.0,
            "total_area_ha": 2.0,
        },
        {
            "block_id": 3,
            "unit_id": "c",
            "avg_farm_slope": 10.0,
            "avg_forest_slope": 8.0,
            "farm_area_ha": 5.0,
            "forest_area_ha": 5.0,
            "total_area_ha": 10.0,
        },
    ]

    summary = summarize_policy_screen(blocks, top_k=2, random_seeds=[0, 1])

    assert summary["n_candidate_blocks"] == 3
    assert summary["strategies"]["slope_gap"]["selected_blocks"] == 2
    assert summary["strategies"]["slope_gap"]["mean_slope_gap"] == 4.5
    assert summary["strategies"]["area_weighted_gap"]["selected_blocks"] == 2
    assert summary["random_baseline"]["n_seeds"] == 2
    assert len(summary["random_baseline"]["per_seed"]) == 2


def test_summarize_policy_screen_reports_random_distribution_and_empirical_p_values():
    blocks = [
        {
            "block_id": 1,
            "unit_id": "a",
            "avg_farm_slope": 10.0,
            "avg_forest_slope": 1.0,
            "farm_area_ha": 2.0,
            "forest_area_ha": 2.0,
            "total_area_ha": 4.0,
        },
        {
            "block_id": 2,
            "unit_id": "b",
            "avg_farm_slope": 8.0,
            "avg_forest_slope": 2.0,
            "farm_area_ha": 3.0,
            "forest_area_ha": 3.0,
            "total_area_ha": 6.0,
        },
        {
            "block_id": 3,
            "unit_id": "c",
            "avg_farm_slope": 3.0,
            "avg_forest_slope": 8.0,
            "farm_area_ha": 10.0,
            "forest_area_ha": 10.0,
            "total_area_ha": 20.0,
        },
        {
            "block_id": 4,
            "unit_id": "d",
            "avg_farm_slope": 2.0,
            "avg_forest_slope": 6.0,
            "farm_area_ha": 8.0,
            "forest_area_ha": 8.0,
            "total_area_ha": 16.0,
        },
    ]

    summary = summarize_policy_screen(blocks, top_k=2, random_seeds=[0, 1, 2, 3, 4])
    random = summary["random_baseline"]

    assert random["mean_slope_gap_sd"] is not None
    assert random["mean_slope_gap_q05"] is not None
    assert random["mean_slope_gap_q50"] is not None
    assert random["mean_slope_gap_q95"] is not None

    slope_gap = summary["strategies"]["slope_gap"]
    assert slope_gap["random_p_mean_slope_gap"] is not None
    assert slope_gap["random_p_positive_gap_share"] is not None
    assert slope_gap["random_p_opportunity_score_sum"] is not None
