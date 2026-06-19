from paper7.dongxing_rl_lite import (
    DongxingSlopeEnv,
    evaluate_preference_policy,
    train_tabular_preference_policy,
)


def _toy_parcels():
    return [
        {"swappable_index": 0, "land_use": "farmland", "area_m2": 100.0, "slope": 10.0},
        {"swappable_index": 1, "land_use": "forest", "area_m2": 100.0, "slope": 2.0},
        {"swappable_index": 2, "land_use": "farmland", "area_m2": 100.0, "slope": 4.0},
        {"swappable_index": 3, "land_use": "forest", "area_m2": 100.0, "slope": 8.0},
    ]


def test_env_masks_only_positive_gain_blocks_and_rewards_slope_improvement():
    env = DongxingSlopeEnv(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        max_steps=3,
        swaps_per_step=1,
    )

    _, _ = env.reset(seed=0)
    mask = env.action_masks()
    assert mask.tolist() == [True, False]

    _, reward, terminated, truncated, info = env.step(0)
    assert reward > 0
    assert terminated is False
    assert truncated is False
    assert info["completed_pairs"] == 1
    assert info["slope_change_pct"] < 0


def test_preference_policy_learns_positive_weight_for_gain():
    env = DongxingSlopeEnv(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        max_steps=3,
        swaps_per_step=1,
    )

    policy = train_tabular_preference_policy(env, seeds=[0, 1], episodes=10)
    result = evaluate_preference_policy(env, policy, seed=0)

    assert policy["weights"][0] > 0
    assert result["completed_pairs"] == 1
