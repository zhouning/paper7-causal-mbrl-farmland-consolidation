from paper7.trajectory_source_ablation import (
    compare_source_reports,
    source_name_from_policies,
)


def test_source_name_from_policies_distinguishes_mixed_random_and_greedy():
    assert source_name_from_policies(None) == "mixed"
    assert source_name_from_policies(["random"]) == "random_only"
    assert source_name_from_policies(["greedy"]) == "greedy_only"
    assert source_name_from_policies(["random", "greedy"]) == "random_greedy"


def test_compare_source_reports_prefers_mixed_for_rollout_reward_mae():
    reports = [
        {
            "source": "random_only",
            "evaluation": {
                "all": {"horizon_100_reward_mae": 0.31, "horizon_100_global_mae": 0.12},
                "random": {"horizon_100_reward_mae": 0.35},
                "greedy": {"horizon_100_reward_mae": 0.28},
            },
        },
        {
            "source": "greedy_only",
            "evaluation": {
                "all": {"horizon_100_reward_mae": 0.27, "horizon_100_global_mae": 0.11},
                "random": {"horizon_100_reward_mae": 0.29},
                "greedy": {"horizon_100_reward_mae": 0.26},
            },
        },
        {
            "source": "mixed",
            "evaluation": {
                "all": {"horizon_100_reward_mae": 0.19, "horizon_100_global_mae": 0.08},
                "random": {"horizon_100_reward_mae": 0.21},
                "greedy": {"horizon_100_reward_mae": 0.18},
            },
        },
    ]

    comparison = compare_source_reports(reports)

    assert comparison["best_source_by_all_horizon_100_reward_mae"] == "mixed"
    assert comparison["best_source_by_random_horizon_100_reward_mae"] == "mixed"
    assert comparison["best_source_by_greedy_horizon_100_reward_mae"] == "mixed"
    assert comparison["mixed_minus_random_all_horizon_100_reward_mae"] == -0.12
    assert comparison["mixed_minus_greedy_all_horizon_100_reward_mae"] == -0.08
