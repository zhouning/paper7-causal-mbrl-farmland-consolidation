"""Dongxing full-reward learned preference policy experiment."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper7.dongxing_full_baselines import _block_features_from_obs, summarize_runs
from paper7.dongxing_full_env import build_dongxing_full_env
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC


FEATURE_NAMES = (
    "feasible_slope_gain",
    "exchange_area_share",
    "available_farm_area_share",
    "available_forest_area_share",
    "current_farm_area_share",
    "neighbor_farmland_context",
    "used_share",
    "remaining_step_share",
)


def select_action(
    block_features: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    epsilon: float,
) -> int:
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    if epsilon > 0 and float(rng.random()) < float(epsilon):
        return int(rng.choice(valid))
    scores = block_features.astype(np.float64) @ weights.astype(np.float64)
    scores[~mask] = -np.inf
    scores += rng.normal(0.0, 1e-9, size=scores.shape)
    return int(np.argmax(scores))


def train_preference_policy(
    env_factory: Callable[[], GenericCountyEnv],
    train_seeds: list[int],
    episodes: int,
    learning_rate: float = 0.03,
    epsilon: float = 0.20,
) -> dict[str, Any]:
    weights = np.zeros(K_BLOCK_GENERIC, dtype=np.float64)
    history: list[dict[str, Any]] = []
    for seed in train_seeds:
        rng = np.random.default_rng(int(seed))
        env = env_factory()
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) * 100_000 + episode)
            total_reward = 0.0
            steps = 0
            last_info: dict[str, Any] = {}
            done = False
            while not done:
                features = _block_features_from_obs(obs, env.n_blocks).astype(np.float64)
                mask = features[:, 0] > 0.0
                if not bool(mask.any()):
                    break
                action = select_action(features, weights, mask, rng, epsilon=epsilon)
                chosen = features[action].copy()
                obs, reward, terminated, truncated, info = env.step(action)
                weights += float(learning_rate) * float(reward) * chosen
                total_reward += float(reward)
                steps += 1
                last_info = info
                done = terminated or truncated
            history.append(
                {
                    "seed": int(seed),
                    "episode": int(episode),
                    "steps": int(steps),
                    "reward": round(float(total_reward), 6),
                    "slope_change_pct": float(last_info.get("slope_change_pct", 0.0)),
                    "completed_swaps": int(last_info.get("budget_used", 0)),
                }
            )
    return {
        "learner_type": "linear_preference_full_reward",
        "feature_names": list(FEATURE_NAMES),
        "weights": [round(float(value), 10) for value in weights.tolist()],
        "training": {
            "episodes": int(episodes),
            "train_seeds": [int(seed) for seed in train_seeds],
            "learning_rate": float(learning_rate),
            "epsilon": float(epsilon),
            "history_tail": history[-20:],
        },
    }


def evaluate_preference_policy(
    env: GenericCountyEnv,
    policy: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    weights = np.asarray(policy["weights"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    obs, _ = env.reset(seed=int(seed))
    done = False
    total_reward = 0.0
    selected_blocks: list[int] = []
    last_info: dict[str, Any] = {}
    while not done:
        features = _block_features_from_obs(obs, env.n_blocks)
        mask = features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        action = select_action(features, weights, mask, rng, epsilon=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return {
        "policy": "learned_full_reward_preference",
        "seed": int(seed),
        "steps": int(last_info.get("step", 0)),
        "reward": round(float(total_reward), 6),
        "slope_change_pct": float(last_info.get("slope_change_pct", 0.0)),
        "cont_change": float(last_info.get("cont_change", 0.0)),
        "baimu_count_change": int(last_info.get("baimu_count_change", 0)),
        "baimu_area_change_ha": float(last_info.get("baimu_area_change_ha", 0.0)),
        "completed_swaps": int(last_info.get("budget_used", 0)),
        "unique_blocks": len(set(selected_blocks)),
        "selected_blocks_head": selected_blocks[:20],
    }


def compare_to_full_baselines(
    learned_summary: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {
        "higher_reward_is_better": True,
        "lower_slope_change_pct_is_better": True,
    }
    for policy, summary in baseline_report.get("policy_summaries", {}).items():
        for field in (
            "reward_mean",
            "slope_change_pct_mean",
            "cont_change_mean",
            "baimu_area_change_ha_mean",
        ):
            if learned_summary.get(field) is None or summary.get(field) is None:
                continue
            comparisons[f"learned_minus_{policy}_{field}"] = round(
                float(learned_summary[field]) - float(summary[field]),
                6,
            )
    return comparisons


def run_experiment(
    env_factory: Callable[[], GenericCountyEnv],
    baseline_path: Path,
    train_seeds: list[int],
    eval_seeds: list[int],
    episodes: int,
    learning_rate: float,
    epsilon: float,
) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    t0 = time.time()
    policy = train_preference_policy(
        env_factory=env_factory,
        train_seeds=train_seeds,
        episodes=episodes,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )
    training_time_s = time.time() - t0
    runs = [
        evaluate_preference_policy(env_factory(), policy, seed=int(seed))
        for seed in eval_seeds
    ]
    summary = summarize_runs(runs)
    return {
        "description": (
            "Dongxing full-reward learned preference policy evaluated in the full "
            "real environment."
        ),
        "status": "supported_as_dongxing_full_reward_learned_policy",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "learner_type": policy["learner_type"],
        "feature_names": list(FEATURE_NAMES),
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "episodes": int(episodes),
        "learning_rate": float(learning_rate),
        "epsilon": float(epsilon),
        "training_time_s": round(float(training_time_s), 6),
        "policy": policy,
        "learned_policy": {"summary": summary, "runs": runs},
        "baseline_path": os.fspath(baseline_path),
        "comparisons": compare_to_full_baselines(summary, baseline_report),
        "claim_boundary": (
            "Local Dongxing full-reward learned actionability; not Bishan-to-Dongxing "
            "policy transfer and not learned-transition MBRL."
        ),
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--baseline", type=Path, default=Path("paper7/results/full_rigor/dongxing_full_baselines.json"))
    parser.add_argument("--train-seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/full_rigor/dongxing_full_learned_policy.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def env_factory() -> GenericCountyEnv:
        return build_dongxing_full_env(
            dltb_path=args.dltb,
            block_dir=args.block_dir,
            total_budget=args.total_budget,
            swaps_per_step=args.swaps_per_step,
        )

    report = run_experiment(
        env_factory=env_factory,
        baseline_path=args.baseline,
        train_seeds=_parse_int_list(args.train_seeds),
        eval_seeds=_parse_int_list(args.eval_seeds),
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "n_eval": len(report["learned_policy"]["runs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
