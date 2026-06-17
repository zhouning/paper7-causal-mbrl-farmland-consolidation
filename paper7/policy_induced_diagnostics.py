"""Policy-induced learned-vs-real environment diagnostics for Paper 7.

The recorded-action rollout diagnostics test transition error on behaviour-policy
actions. This script tests a stricter case: a trained model-based policy chooses
actions from the learned environment state, and the same actions are replayed in
the real county environment. The resulting state, reward, mask, and support
metrics diagnose whether the final policy visits states far from the transition
model's trajectory support.
"""

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

torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO

from county_env import CountyLevelEnv
from paper7.learned_env import LearnedCountyEnv
from paper7.train_model_based import load_transition_model
from paper7.transition_rollout_diagnostics import load_trajectory_arrays


PAPER7_DIR = Path(__file__).resolve().parent


def compute_nearest_neighbor_distances(
    query: np.ndarray,
    support: np.ndarray,
    chunk_size: int = 512,
) -> np.ndarray:
    """Return Euclidean distance from each query row to its nearest support row."""
    query = np.asarray(query, dtype=np.float32)
    support = np.asarray(support, dtype=np.float32)
    if query.ndim == 1:
        query = query.reshape(1, -1)
    if support.ndim == 1:
        support = support.reshape(1, -1)
    distances: list[np.ndarray] = []
    for start in range(0, len(query), int(chunk_size)):
        q = query[start : start + int(chunk_size)]
        diff = q[:, None, :] - support[None, :, :]
        sq = np.sum(diff * diff, axis=2)
        distances.append(np.sqrt(np.min(sq, axis=1)))
    return np.concatenate(distances, axis=0)


def compute_policy_step_metrics(
    step: int,
    action: int,
    learned_block: np.ndarray,
    real_block: np.ndarray,
    learned_global: np.ndarray,
    real_global: np.ndarray,
    learned_reward: float,
    real_reward: float,
    learned_mask: np.ndarray,
    real_mask: np.ndarray,
    support_distance: float,
    reward_scale: float = 1.0,
) -> dict[str, float | int]:
    """Compare one synchronized learned-env and real-env policy step."""
    learned_block = np.asarray(learned_block, dtype=np.float32)
    real_block = np.asarray(real_block, dtype=np.float32)
    learned_global = np.asarray(learned_global, dtype=np.float32)
    real_global = np.asarray(real_global, dtype=np.float32)
    learned_mask = np.asarray(learned_mask, dtype=bool)
    real_mask = np.asarray(real_mask, dtype=bool)
    return {
        "step": int(step),
        "action": int(action),
        "selected_block_mae": round(
            float(np.abs(learned_block[int(action)] - real_block[int(action)]).mean()), 6
        ),
        "all_block_mae": round(float(np.abs(learned_block - real_block).mean()), 6),
        "global_mae": round(float(np.abs(learned_global - real_global).mean()), 6),
        "reward_abs_error": round(abs(float(learned_reward) - float(real_reward)), 6),
        "calibrated_reward_abs_error": round(
            abs(float(reward_scale) * (float(learned_reward) - float(real_reward))),
            6,
        ),
        "mask_agreement": round(float(np.mean(learned_mask == real_mask)), 6),
        "support_distance": round(float(support_distance), 6),
    }


def summarize_episode_metrics(
    step_metrics: list[dict[str, float | int]],
    final_real_info: dict[str, Any],
) -> dict[str, float | int | None]:
    """Aggregate one policy-induced diagnostic episode."""
    if not step_metrics:
        return {
            "n_steps": 0,
            "final_real_slope_change_pct": final_real_info.get("slope_change_pct"),
        }

    def _values(key: str) -> list[float]:
        return [float(row[key]) for row in step_metrics if key in row]

    def _mean(key: str) -> float | None:
        values = _values(key)
        if not values:
            return None
        return round(float(np.mean(values)), 6)

    def _q95(key: str) -> float | None:
        values = _values(key)
        if not values:
            return None
        return round(float(np.quantile(values, 0.95)), 6)

    summary: dict[str, float | int | None] = {
        "n_steps": len(step_metrics),
        "global_mae_mean": _mean("global_mae"),
        "global_mae_q95": _q95("global_mae"),
        "reward_mae_mean": _mean("reward_abs_error"),
        "reward_mae_q95": _q95("reward_abs_error"),
        "calibrated_reward_mae_mean": _mean("calibrated_reward_abs_error"),
        "calibrated_reward_mae_q95": _q95("calibrated_reward_abs_error"),
        "mask_agreement_mean": _mean("mask_agreement"),
        "mask_agreement_min": round(float(np.min(_values("mask_agreement"))), 6)
        if _values("mask_agreement")
        else None,
        "support_distance_mean": _mean("support_distance"),
        "support_distance_q95": _q95("support_distance"),
        "final_real_slope_change_pct": final_real_info.get("slope_change_pct"),
        "final_real_cont_change": final_real_info.get("cont_change"),
        "final_real_budget_used": final_real_info.get("budget_used"),
    }
    if _values("selected_block_mae"):
        summary["selected_block_mae_mean"] = _mean("selected_block_mae")
        summary["selected_block_mae_q95"] = _q95("selected_block_mae")
    if _values("all_block_mae"):
        summary["all_block_mae_mean"] = _mean("all_block_mae")
    return summary


def _split_flat_obs(obs: np.ndarray, n_blocks: int, k_block: int, k_global: int) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(obs, dtype=np.float32)
    block_size = int(n_blocks) * int(k_block)
    return obs[:block_size].reshape(int(n_blocks), int(k_block)), obs[block_size : block_size + int(k_global)]


def _make_learned_env(
    transition_model_path: Path,
    trajectory_dir: Path,
) -> tuple[LearnedCountyEnv, dict[str, Any]]:
    model, ckpt = load_transition_model(transition_model_path)
    first_traj = sorted(trajectory_dir.glob("*.npz"))[0]
    data = np.load(first_traj, allow_pickle=False)
    env = LearnedCountyEnv(
        transition_model=model,
        initial_block_features=data["block_features"][0].astype(np.float32),
        initial_global_features=data["global_features"][0].astype(np.float32),
        n_blocks=int(data["n_blocks"]),
        k_block=int(data["k_block"]),
        k_global=int(data["k_global"]),
        max_steps=100,
    )
    return env, ckpt


def _support_global_features(trajectory_dir: Path, max_support: int | None = None) -> np.ndarray:
    arrays = load_trajectory_arrays(trajectory_dir, max_transitions=max_support)
    return arrays["global_features"].astype(np.float32)


def run_policy_diagnostic_episode(
    policy_model_path: Path,
    transition_model_path: Path,
    trajectory_dir: Path,
    reward_scale: float = 0.185,
    max_steps: int = 100,
    support_globals: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run one trained policy in learned env while replaying actions in real env."""
    learned_env, ckpt = _make_learned_env(
        transition_model_path=transition_model_path,
        trajectory_dir=trajectory_dir,
    )
    real_env = CountyLevelEnv(total_budget=500, swaps_per_step=5)
    policy = MaskablePPO.load(policy_model_path, env=learned_env)

    learned_obs, _ = learned_env.reset()
    real_obs, real_info = real_env.reset()
    n_blocks = int(ckpt["n_blocks"])
    k_block = int(ckpt["k_block"])
    k_global = int(ckpt["k_global"])
    if support_globals is None:
        support_globals = _support_global_features(trajectory_dir)

    step_metrics: list[dict[str, float | int]] = []
    selected_actions: list[int] = []
    done = False
    for step in range(int(max_steps)):
        learned_mask = learned_env.action_masks()
        real_mask = real_env.action_masks()
        valid_mask = learned_mask & real_mask
        if not valid_mask.any():
            break

        action, _ = policy.predict(
            learned_obs,
            action_masks=valid_mask,
            deterministic=True,
        )
        action_int = int(action)
        selected_actions.append(action_int)

        learned_obs, learned_reward, learned_done, learned_trunc, _ = learned_env.step(action_int)
        real_obs, real_reward, real_done, real_trunc, real_info = real_env.step(action_int)

        learned_block, learned_global = _split_flat_obs(learned_obs, n_blocks, k_block, k_global)
        real_block, real_global = _split_flat_obs(real_obs, n_blocks, k_block, k_global)
        support_distance = compute_nearest_neighbor_distances(real_global, support_globals)[0]
        step_metrics.append(
            compute_policy_step_metrics(
                step=step + 1,
                action=action_int,
                learned_block=learned_block,
                real_block=real_block,
                learned_global=learned_global,
                real_global=real_global,
                learned_reward=float(learned_reward),
                real_reward=float(real_reward),
                learned_mask=learned_env.action_masks(),
                real_mask=real_env.action_masks(),
                support_distance=float(support_distance),
                reward_scale=float(reward_scale),
            )
        )

        done = bool(learned_done or learned_trunc or real_done or real_trunc)
        if done:
            break

    summary = summarize_episode_metrics(step_metrics, real_info)
    summary.update(
        {
            "policy_model": os.fspath(policy_model_path),
            "reward_scale": float(reward_scale),
            "selected_actions_head": selected_actions[:20],
            "terminated": done,
        }
    )
    return {"summary": summary, "step_metrics_head": step_metrics[:20]}


def summarize_policy_diagnostics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = [episode["summary"] for episode in episodes]
    keys = [
        "selected_block_mae_mean",
        "all_block_mae_mean",
        "global_mae_mean",
        "reward_mae_mean",
        "calibrated_reward_mae_mean",
        "mask_agreement_mean",
        "support_distance_mean",
        "support_distance_q95",
        "final_real_slope_change_pct",
    ]
    aggregate: dict[str, Any] = {"n_episodes": len(summaries)}
    for key in keys:
        values = [float(row[key]) for row in summaries if row.get(key) is not None]
        if values:
            aggregate[f"{key}_mean"] = round(float(np.mean(values)), 6)
            aggregate[f"{key}_min"] = round(float(np.min(values)), 6)
            aggregate[f"{key}_max"] = round(float(np.max(values)), 6)
    return aggregate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-models",
        nargs="+",
        type=Path,
        default=[
            Path("paper7/results/revision/seeds/with_cal_model_seed0.zip"),
            Path("paper7/results/revision/seeds/with_cal_model_seed1.zip"),
            Path("paper7/results/revision/seeds/with_cal_model_seed2.zip"),
        ],
    )
    parser.add_argument("--transition-model", type=Path, default=Path("paper7/models/transition_model.pt"))
    parser.add_argument("--trajectory-dir", type=Path, default=Path("paper7/trajectories"))
    parser.add_argument("--reward-scale", type=float, default=0.185)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-support", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/revision/policy_induced_diagnostics.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    support_globals = _support_global_features(args.trajectory_dir, max_support=args.max_support)
    episodes = [
        run_policy_diagnostic_episode(
            policy_model_path=path,
            transition_model_path=args.transition_model,
            trajectory_dir=args.trajectory_dir,
            reward_scale=args.reward_scale,
            max_steps=args.max_steps,
            support_globals=support_globals,
        )
        for path in args.policy_models
    ]
    result = {
        "description": (
            "Policy-induced diagnostics: trained model-based policies choose actions "
            "from learned-environment states while the same actions are replayed in "
            "the real county environment."
        ),
        "transition_model": os.fspath(args.transition_model),
        "trajectory_dir": os.fspath(args.trajectory_dir),
        "reward_scale": float(args.reward_scale),
        "support_size": int(len(support_globals)),
        "episodes": episodes,
        "aggregate": summarize_policy_diagnostics(episodes),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
