import numpy as np
import torch

from paper7.transition_rollout_diagnostics import (
    compute_action_mask_agreement,
    compute_step_metrics,
    rollout_model,
    summarize_feature_groups,
    summarize_step_metrics,
)


class ToyTransitionModel(torch.nn.Module):
    def forward(self, block_features, global_features, action, geofm=None):
        next_block = block_features.clone()
        next_global = global_features.clone()
        rewards = []
        for row, act in enumerate(action.tolist()):
            next_block[row, act, 0] += 1.0
            next_global[row, 0] += 0.5
            rewards.append(float(act) + 0.25)
        return next_block, next_global, torch.tensor(rewards, dtype=torch.float32)


def test_compute_action_mask_agreement_uses_swap_potential_feature():
    pred = np.array([[0.0] * 10, [0.0] * 10, [0.0] * 10], dtype=np.float32)
    true = pred.copy()
    pred[:, 9] = [0.02, 0.0, 0.03]
    true[:, 9] = [0.02, 0.0, 0.04]

    agreement = compute_action_mask_agreement(pred, true, threshold=0.01)

    assert agreement == 1.0


def test_compute_step_metrics_reports_selected_global_reward_and_mask_errors():
    current_block = np.zeros((2, 10), dtype=np.float32)
    pred_block = current_block.copy()
    true_block = current_block.copy()
    pred_block[1, 0] = 2.0
    true_block[1, 0] = 1.0
    pred_block[:, 9] = [0.02, 0.0]
    true_block[:, 9] = [0.0, 0.0]

    metrics = compute_step_metrics(
        pred_block=pred_block,
        pred_global=np.array([2.0, 1.0], dtype=np.float32),
        pred_reward=1.5,
        true_block=true_block,
        true_global=np.array([1.0, 1.0], dtype=np.float32),
        true_reward=1.0,
        action=1,
    )

    assert metrics["selected_block_mae"] == 0.1
    assert metrics["global_mae"] == 0.5
    assert metrics["reward_abs_error"] == 0.5
    assert metrics["mask_agreement"] == 0.5


def test_rollout_model_accumulates_predictions_against_recorded_actions():
    block_features = np.zeros((3, 2, 10), dtype=np.float32)
    global_features = np.zeros((3, 2), dtype=np.float32)
    actions = np.array([0, 1, 0], dtype=np.int64)
    rewards = np.array([0.25, 1.25, 0.25], dtype=np.float32)
    true_next_block = block_features.copy()
    true_next_global = global_features.copy()
    for step, action in enumerate(actions):
        true_next_block[step, action, 0] = 1.0
        true_next_global[step, 0] = 0.5

    results = rollout_model(
        model=ToyTransitionModel(),
        block_features=block_features,
        global_features=global_features,
        actions=actions,
        rewards=rewards,
        next_block_features=true_next_block,
        next_global_features=true_next_global,
        horizons=[1, 2],
        start_indices=[0],
    )

    assert results["horizons"]["1"]["n_steps"] == 1
    assert results["horizons"]["1"]["reward_mae"] == 0.0
    assert results["horizons"]["2"]["n_steps"] == 2
    assert results["horizons"]["2"]["selected_block_mae"] == 0.0


def test_summarize_step_metrics_adds_q50_and_q95_for_error_fields():
    metrics = [
        {
            "selected_block_mae": 1.0,
            "all_block_mae": 0.1,
            "global_mae": 2.0,
            "global_rmse": 2.0,
            "reward_abs_error": 3.0,
            "mask_agreement": 1.0,
        },
        {
            "selected_block_mae": 3.0,
            "all_block_mae": 0.3,
            "global_mae": 4.0,
            "global_rmse": 4.0,
            "reward_abs_error": 5.0,
            "mask_agreement": 0.5,
        },
    ]

    summary = summarize_step_metrics(metrics)

    assert summary["selected_block_mae_q50"] == 2.0
    assert summary["selected_block_mae_q95"] > 2.0
    assert summary["reward_mae"] == 4.0


def test_summarize_feature_groups_reports_named_global_groups():
    pred_global = [1.0, 3.0, 10.0, 10.0]
    true_global = [2.0, 1.0, 7.0, 12.0]

    groups = summarize_feature_groups(
        pred_global,
        true_global,
        {"first_two": [0, 1], "last_two": [2, 3]},
    )

    assert groups["first_two_mae"] == 1.5
    assert groups["last_two_mae"] == 2.5
