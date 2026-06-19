from paper7.reward_components import RewardWeights
from paper7.reward_weight_sensitivity import (
    replay_episode_reward,
    summarize_replayed_episodes,
)


def _episode(policy, seed, slope, cont, baimu_area, baimu_count):
    return {
        "policy": policy,
        "seed": seed,
        "summary": {
            "final_slope_change_pct": slope,
            "final_cont_change": cont,
            "final_baimu_area_change_ha": baimu_area,
            "final_baimu_count_change": baimu_count,
        },
        "steps": [
            {
                "slope_delta": 0.01,
                "cont_delta": 0.02,
                "baimu_area_delta": 0.03,
                "baimu_new_count": 1,
                "completed_swaps": 5,
            }
        ],
    }


def test_replay_episode_reward_uses_requested_weights():
    episode = _episode("p", 0, -1.0, 0.01, 2.0, 1)
    reward = replay_episode_reward(
        episode,
        RewardWeights(slope_weight=1, cont_weight=1, baimu_weight=1, baimu_bonus=1),
    )

    assert round(reward, 6) == 1.06


def test_summarize_replayed_episodes_groups_by_policy_and_weight_name():
    episodes = [
        _episode("a", 0, -1.0, 0.01, 2.0, 1),
        _episode("a", 1, -1.2, 0.02, 3.0, 2),
        _episode("b", 0, -0.5, 0.03, 5.0, 3),
    ]

    report = summarize_replayed_episodes(
        episodes,
        weight_grid=[
            {
                "name": "unit",
                "weights": RewardWeights(
                    slope_weight=1,
                    cont_weight=1,
                    baimu_weight=1,
                    baimu_bonus=1,
                ),
            },
            {
                "name": "slope_only",
                "weights": RewardWeights(
                    slope_weight=10,
                    cont_weight=0,
                    baimu_weight=0,
                    baimu_bonus=0,
                ),
            },
        ],
    )

    rows = report["policy_weight_summaries"]
    row = next(item for item in rows if item["policy"] == "a" and item["weight_name"] == "unit")
    assert row["n"] == 2
    assert row["slope_change_pct_mean"] == -1.1
    assert row["baimu_count_change_mean"] == 1.5
    assert len(report["policy_metric_summaries"]) == 2
    assert report["pareto_front"]
    assert len({row["policy"] for row in report["pareto_front"]}) == len(report["pareto_front"])
    assert {item["weight_name"] for item in report["best_policy_by_weight"]} == {"unit", "slope_only"}
