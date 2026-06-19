from paper7.reward_components import (
    RewardComponents,
    RewardWeights,
    compute_scalar_reward,
    default_reward_weights,
    generate_weight_grid,
    pareto_front,
)


def test_default_reward_matches_county_env_formula_with_negative_baimu_area():
    components = RewardComponents(
        slope_delta=0.001,
        cont_delta=0.002,
        baimu_area_delta=-0.003,
        baimu_new_count=2,
        completed_swaps=5,
    )

    reward = compute_scalar_reward(components, default_reward_weights())

    expected = (
        4000.0 * 0.001
        + 500.0 * 0.002
        + 1500.0 * -0.003
        + 5.0 * 2
        + 2000.0 * -0.003
    )
    assert reward == expected


def test_invalid_or_zero_swap_action_receives_penalty():
    components = RewardComponents(
        slope_delta=0.0,
        cont_delta=0.0,
        baimu_area_delta=0.0,
        baimu_new_count=0,
        completed_swaps=0,
    )

    reward = compute_scalar_reward(components, default_reward_weights())

    assert reward == -1.0


def test_generate_weight_grid_includes_default_and_named_variants():
    grid = generate_weight_grid()
    names = {item["name"] for item in grid}

    assert "default" in names
    assert "slope_x2" in names
    assert "contiguity_x2" in names
    assert "baimu_area_x2" in names
    assert "baimu_count_x2" in names

    default = next(item for item in grid if item["name"] == "default")
    assert default["weights"].slope_weight == 4000.0


def test_pareto_front_keeps_non_dominated_rows_for_mixed_directions():
    rows = [
        {
            "id": "a",
            "slope_change_pct": -1.0,
            "cont_change": 0.01,
            "baimu_area_change_ha": 1.0,
        },
        {
            "id": "b",
            "slope_change_pct": -1.2,
            "cont_change": 0.02,
            "baimu_area_change_ha": 2.0,
        },
        {
            "id": "c",
            "slope_change_pct": -1.4,
            "cont_change": 0.005,
            "baimu_area_change_ha": 1.5,
        },
    ]

    front = pareto_front(
        rows,
        objectives={
            "slope_change_pct": "min",
            "cont_change": "max",
            "baimu_area_change_ha": "max",
        },
    )

    ids = {row["id"] for row in front}
    assert ids == {"b", "c"}
