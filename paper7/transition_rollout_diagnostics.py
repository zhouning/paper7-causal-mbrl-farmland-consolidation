"""Multi-step diagnostics for Paper 7 transition-model validation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper7.train_model_based import load_transition_model


DEFAULT_GLOBAL_FEATURE_GROUPS = {
    "budget_progress": [0, 3],
    "slope_contiguity": [1, 2, 4, 5],
    "baimu": [6, 7],
    "investment_spread": [8, 9, 10, 11],
}


def compute_action_mask_agreement(
    pred_block: np.ndarray,
    true_block: np.ndarray,
    threshold: float = 0.01,
    mask_feature_index: int = 9,
) -> float:
    """Return the fraction of blocks with matching valid-action masks."""
    pred_mask = pred_block[:, mask_feature_index] > threshold
    true_mask = true_block[:, mask_feature_index] > threshold
    return float(np.mean(pred_mask == true_mask))


def summarize_feature_groups(
    pred_values: np.ndarray | list[float],
    true_values: np.ndarray | list[float],
    groups: dict[str, list[int]],
) -> dict[str, float]:
    """Return MAE for named feature-index groups."""
    pred = np.asarray(pred_values, dtype=np.float64)
    true = np.asarray(true_values, dtype=np.float64)
    result: dict[str, float] = {}
    for name, indices in groups.items():
        valid_indices = [int(index) for index in indices if int(index) < len(pred) and int(index) < len(true)]
        if not valid_indices:
            continue
        result[f"{name}_mae"] = round(
            float(np.abs(pred[valid_indices] - true[valid_indices]).mean()), 6
        )
    return result


def compute_step_metrics(
    pred_block: np.ndarray,
    pred_global: np.ndarray,
    pred_reward: float,
    true_block: np.ndarray,
    true_global: np.ndarray,
    true_reward: float,
    action: int,
) -> dict[str, float]:
    """Compute per-step diagnostics that avoid whole-observation cosine inflation."""
    selected_error = np.abs(pred_block[int(action)] - true_block[int(action)])
    block_delta_error = np.abs(pred_block - true_block)
    global_error = np.abs(pred_global - true_global)
    metrics = {
        "selected_block_mae": round(float(selected_error.mean()), 6),
        "selected_block_rmse": round(float(np.sqrt(np.mean(selected_error**2))), 6),
        "all_block_mae": round(float(block_delta_error.mean()), 6),
        "global_mae": round(float(global_error.mean()), 6),
        "global_rmse": round(float(np.sqrt(np.mean(global_error**2))), 6),
        "reward_abs_error": round(float(abs(float(pred_reward) - float(true_reward))), 6),
        "mask_agreement": round(float(compute_action_mask_agreement(pred_block, true_block)), 6),
    }
    metrics.update(
        summarize_feature_groups(
            pred_values=pred_global,
            true_values=true_global,
            groups=DEFAULT_GLOBAL_FEATURE_GROUPS,
        )
    )
    return metrics


def rollout_model(
    model: torch.nn.Module,
    block_features: np.ndarray,
    global_features: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_block_features: np.ndarray,
    next_global_features: np.ndarray,
    horizons: list[int],
    start_indices: list[int],
) -> dict[str, Any]:
    """Roll out a transition model over recorded actions and summarize errors."""
    model.eval()
    results: dict[str, Any] = {"horizons": {}}
    max_start = len(actions) - 1

    for horizon in horizons:
        step_metrics: list[dict[str, float]] = []
        usable_starts = [
            int(start)
            for start in start_indices
            if 0 <= int(start) <= max_start and int(start) + int(horizon) <= len(actions)
        ]
        for start in usable_starts:
            pred_block = block_features[start].astype(np.float32).copy()
            pred_global = global_features[start].astype(np.float32).copy()

            for offset in range(int(horizon)):
                idx = start + offset
                action = int(actions[idx])
                with torch.no_grad():
                    bf_t = torch.tensor(pred_block, dtype=torch.float32).unsqueeze(0)
                    gf_t = torch.tensor(pred_global, dtype=torch.float32).unsqueeze(0)
                    action_t = torch.tensor([action], dtype=torch.long)
                    next_bf_t, next_gf_t, pred_reward_t = model(bf_t, gf_t, action_t)
                pred_block = next_bf_t.squeeze(0).cpu().numpy()
                pred_global = next_gf_t.squeeze(0).cpu().numpy()
                pred_reward = float(pred_reward_t.squeeze(0).cpu().item())
                step_metrics.append(
                    compute_step_metrics(
                        pred_block=pred_block,
                        pred_global=pred_global,
                        pred_reward=pred_reward,
                        true_block=next_block_features[idx].astype(np.float32),
                        true_global=next_global_features[idx].astype(np.float32),
                        true_reward=float(rewards[idx]),
                        action=action,
                    )
                )

        results["horizons"][str(horizon)] = summarize_step_metrics(step_metrics)

    return results


def summarize_step_metrics(step_metrics: list[dict[str, float]]) -> dict[str, float | int | None]:
    """Aggregate per-step metrics."""
    if not step_metrics:
        return {"n_steps": 0}
    summary: dict[str, float | int | None] = {"n_steps": len(step_metrics)}
    keys = step_metrics[0].keys()
    for key in keys:
        values = np.array([float(metric[key]) for metric in step_metrics], dtype=np.float64)
        summary[key] = round(float(values.mean()), 6)
        summary[f"{key}_q50"] = round(float(np.quantile(values, 0.50)), 6)
        summary[f"{key}_q95"] = round(float(np.quantile(values, 0.95)), 6)
    # Short aliases used in the manuscript table.
    summary["reward_mae"] = summary["reward_abs_error"]
    return summary


def load_trajectory_arrays(trajectory_dir: Path, max_transitions: int | None = None) -> dict[str, np.ndarray]:
    """Load and concatenate trajectory arrays from compressed NPZ files."""
    arrays: dict[str, list[np.ndarray]] = {
        "block_features": [],
        "global_features": [],
        "actions": [],
        "rewards": [],
        "next_block_features": [],
        "next_global_features": [],
    }
    for path in sorted(trajectory_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        for key in arrays:
            arrays[key].append(data[key])
    merged = {key: np.concatenate(values, axis=0) for key, values in arrays.items()}
    if max_transitions is not None:
        n = min(int(max_transitions), len(merged["actions"]))
        merged = {key: value[:n] for key, value in merged.items()}
    return merged


def choose_start_indices(n_transitions: int, horizons: list[int], n_starts: int) -> list[int]:
    """Choose deterministic start indices with enough room for the largest horizon."""
    max_horizon = max(horizons)
    max_start = max(0, n_transitions - max_horizon)
    if max_start == 0:
        return [0]
    count = min(int(n_starts), max_start + 1)
    return sorted(set(int(x) for x in np.linspace(0, max_start, count)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("paper7/models/transition_model.pt"))
    parser.add_argument("--trajectory-dir", type=Path, default=Path("paper7/trajectories"))
    parser.add_argument("--horizons", default="1,5,10,25,50,100")
    parser.add_argument("--n-starts", type=int, default=120)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/revision/transition_rollout_diagnostics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(item) for item in str(args.horizons).split(",") if item.strip()]
    model, ckpt = load_transition_model(args.model)
    arrays = load_trajectory_arrays(args.trajectory_dir, max_transitions=args.max_transitions)
    start_indices = choose_start_indices(len(arrays["actions"]), horizons, args.n_starts)
    results = rollout_model(
        model=model,
        block_features=arrays["block_features"].astype(np.float32),
        global_features=arrays["global_features"].astype(np.float32),
        actions=arrays["actions"].astype(np.int64),
        rewards=arrays["rewards"].astype(np.float32),
        next_block_features=arrays["next_block_features"].astype(np.float32),
        next_global_features=arrays["next_global_features"].astype(np.float32),
        horizons=horizons,
        start_indices=start_indices,
    )
    results.update(
        {
            "description": (
                "Multi-step transition-model diagnostics over recorded real-environment actions. "
                "Selected-block, global, reward, and action-mask metrics are reported to avoid "
                "inflated whole-observation cosine similarity."
            ),
            "model": os.fspath(args.model),
            "trajectory_dir": os.fspath(args.trajectory_dir),
            "n_transitions_loaded": int(len(arrays["actions"])),
            "n_starts": int(len(start_indices)),
            "horizons_requested": horizons,
            "n_blocks": int(ckpt["n_blocks"]),
            "k_block": int(ckpt["k_block"]),
            "k_global": int(ckpt["k_global"]),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
