import numpy as np
import pandas as pd
import pytest

from paper7.causal_sensitivity_diagnostics import (
    build_rows_from_trajectory_arrays,
    filter_dataset_by_policy,
    estimate_att_from_propensity,
    standardized_mean_differences,
)


def test_build_rows_assigns_percentile_treatment_with_policy_flags():
    block_features = np.zeros((2, 3, 17), dtype=np.float32)
    block_features[0, :, 3] = [0.1, 0.5, 0.9]
    block_features[1, :, 3] = [0.2, 0.3, 0.4]
    block_features[:, :, 9] = 0.2
    global_features = np.zeros((2, 12), dtype=np.float32)
    global_features[:, 0] = [1.0, 0.9]
    actions = np.array([2, 0], dtype=np.int64)
    rewards = np.array([3.0, -1.0], dtype=np.float32)

    rows = build_rows_from_trajectory_arrays(
        block_features=block_features,
        global_features=global_features,
        actions=actions,
        rewards=rewards,
        policy_name="greedy",
        treatment_percentile=50,
    )

    assert [row["treatment"] for row in rows] == [1, 0]
    assert rows[0]["policy_greedy"] == 1.0
    assert rows[0]["block_best_swap_gain_norm"] == pytest.approx(0.9)
    assert rows[1]["outcome"] == pytest.approx(-1.0)


def test_estimate_att_from_propensity_uses_att_weights_and_trim():
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "outcome": [5.0, 7.0, 1.0, 10.0],
            "propensity": [0.6, 0.6, 0.5, 0.1],
        }
    )

    untrimmed = estimate_att_from_propensity(df, trim_bounds=(0.0, 1.0))
    trimmed = estimate_att_from_propensity(df, trim_bounds=(0.2, 0.95))

    assert untrimmed["n_used"] == 4
    assert untrimmed["att"] == pytest.approx(4.1)
    assert trimmed["n_used"] == 3
    assert trimmed["att"] == pytest.approx(5.0)


def test_standardized_mean_differences_report_before_and_after_att_weighting():
    df = pd.DataFrame(
        {
            "treatment": [1, 1, 0, 0],
            "x": [2.0, 4.0, 0.0, 2.0],
            "propensity": [0.7, 0.7, 0.2, 0.8],
        }
    )

    smd = standardized_mean_differences(df, ["x"])

    assert smd["x"]["before"] == pytest.approx(2.0, rel=1e-6)
    assert abs(smd["x"]["after_att_weighting"]) < abs(smd["x"]["before"])
    assert smd["max_abs_before"] == pytest.approx(abs(smd["x"]["before"]))


def test_filter_dataset_by_policy_keeps_requested_policy_names():
    df = pd.DataFrame(
        {
            "policy": ["random", "greedy", "random"],
            "outcome": [0.0, 1.0, 2.0],
        }
    )

    filtered = filter_dataset_by_policy(df, ["random"])

    assert filtered["policy"].tolist() == ["random", "random"]
    assert filter_dataset_by_policy(df, None).equals(df)
