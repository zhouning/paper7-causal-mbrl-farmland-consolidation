"""Dongxing slope-only RL-lite experiment for Paper 7 CEUS revision."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.dongxing_dynamic_baselines import (  # noqa: E402
    DynamicSwapState,
    load_swappable_parcels,
    run_baseline_suite,
)


FEATURE_NAMES = (
    "feasible_gain_norm",
    "exchange_area_norm",
    "farm_area_norm",
    "forest_area_norm",
    "used_pair_share",
    "remaining_step_share",
)


class DongxingSlopeEnv(gym.Env):
    """Slope-only Dongxing block-selection environment.

    This environment reuses the external-county paired-swap dynamics and exposes
    a compact masked block-action MDP. It is intentionally slope-only and does
    not claim to reproduce Bishan's full contiguity/baimu reward.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        parcels: list[dict[str, Any]],
        block_compositions: dict[str, list[int]],
        block_ids: list[int],
        max_steps: int = 100,
        swaps_per_step: int = 5,
    ) -> None:
        super().__init__()
        self.parcels = [dict(parcel) for parcel in parcels]
        self.block_compositions = {
            str(block_id): [int(index) for index in indices]
            for block_id, indices in block_compositions.items()
        }
        self.block_ids = [int(block_id) for block_id in block_ids]
        self.max_steps = int(max_steps)
        self.swaps_per_step = int(swaps_per_step)
        self.feature_names = FEATURE_NAMES
        self.feature_dim = len(FEATURE_NAMES)
        self.action_space = spaces.Discrete(len(self.block_ids))
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(self.block_ids) * self.feature_dim,),
            dtype=np.float32,
        )
        self._max_area_m2 = self._compute_max_area_m2()
        self._max_gain = self._compute_initial_max_gain()
        self._block_position = {int(block_id): idx for idx, block_id in enumerate(self.block_ids)}
        self._rng = np.random.default_rng(0)
        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._rng = np.random.default_rng(seed)
        self.state = DynamicSwapState.from_records(
            parcels=self.parcels,
            block_compositions=self.block_compositions,
        )
        self.step_count = 0
        self._pairs_by_block = {int(block_id): 0 for block_id in self.block_ids}
        self._features = np.zeros((len(self.block_ids), self.feature_dim), dtype=np.float32)
        self._mask = np.zeros(len(self.block_ids), dtype=bool)
        for position in range(len(self.block_ids)):
            self._refresh_block_feature(position)
        self._set_remaining_step_share()
        return self._get_obs(), self._info()

    def action_masks(self) -> np.ndarray:
        return self._mask.copy()

    def block_feature_matrix(self) -> np.ndarray:
        return self._features.copy()

    def step(self, action: int):
        action = int(action)
        old_slope = float(self.state.avg_farmland_slope)
        if action < 0 or action >= len(self.block_ids):
            completed = 0
            reward = -0.01
            block_id = None
        else:
            block_id = self.block_ids[action]
            if self.state.feasible_gain(block_id) <= 0:
                completed = 0
                reward = -0.01
            else:
                completed = self.state.execute_block(block_id, max_pairs=self.swaps_per_step)
                self._pairs_by_block[block_id] += completed
                reward = float(old_slope - self.state.avg_farmland_slope)
                if completed <= 0:
                    reward = -0.01
                self._refresh_block_feature(action)

        self.step_count += 1
        self._set_remaining_step_share()
        terminated = self.step_count >= self.max_steps
        truncated = False
        info = self._info()
        info.update(
            {
                "selected_block": block_id,
                "completed_pairs": int(completed),
                "reward_slope_decrease": round(float(reward), 8),
            }
        )
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return self._features.reshape(-1).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "step": int(self.step_count),
            "completed_pairs_total": int(self.state.completed_pairs),
            "initial_avg_farmland_slope": round(float(self.state.initial_avg_farmland_slope), 6),
            "final_avg_farmland_slope": round(float(self.state.avg_farmland_slope), 6),
            "slope_change_pct": round(float(self.state.slope_change_pct), 6),
        }

    def _compute_max_area_m2(self) -> float:
        max_area = 0.0
        for indices in self.block_compositions.values():
            area = sum(float(self.parcels[index]["area_m2"]) for index in indices)
            max_area = max(max_area, area)
        return max(max_area, 1.0)

    def _compute_initial_max_gain(self) -> float:
        state = DynamicSwapState.from_records(self.parcels, self.block_compositions)
        max_gain = max((state.feasible_gain(block_id) for block_id in self.block_ids), default=0.0)
        return max(float(max_gain), 1e-6)

    def _refresh_block_feature(self, position: int) -> None:
        block_id = self.block_ids[int(position)]
        stats = self.state.block_stats.get(str(block_id), {})
        gain = max(0.0, float(self.state.feasible_gain(block_id)))
        exchange_area = max(0.0, float(self.state.feasible_exchange_area_m2(block_id)))
        farm_area = max(0.0, float(stats.get("farm_area_m2", 0.0)))
        forest_area = max(0.0, float(stats.get("forest_area_m2", 0.0)))
        max_pairs = max(1, self.max_steps * self.swaps_per_step)
        self._features[int(position), :5] = np.asarray(
            [
                gain / self._max_gain,
                exchange_area / self._max_area_m2,
                farm_area / self._max_area_m2,
                forest_area / self._max_area_m2,
                self._pairs_by_block.get(int(block_id), 0) / max_pairs,
            ],
            dtype=np.float32,
        )
        self._mask[int(position)] = gain > 0

    def _set_remaining_step_share(self) -> None:
        self._features[:, 5] = np.float32(1.0 - self.step_count / max(1, self.max_steps))


def train_tabular_preference_policy(
    env: DongxingSlopeEnv,
    seeds: list[int],
    episodes: int,
    learning_rate: float = 0.05,
    epsilon: float = 0.25,
) -> dict[str, Any]:
    """Learn linear block preferences from slope-improvement rewards."""
    weights = np.zeros(env.feature_dim, dtype=np.float64)
    episode_history: list[dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        for episode in range(int(episodes)):
            env.reset(seed=int(seed) * 100_000 + episode)
            total_reward = 0.0
            steps = 0
            while steps < env.max_steps:
                mask = env.action_masks()
                if not bool(mask.any()):
                    break
                features = env.block_feature_matrix().astype(np.float64)
                action = _select_action(features, weights, mask, rng, epsilon=epsilon)
                chosen_features = features[action].copy()
                _, reward, terminated, truncated, info = env.step(action)
                weights += float(learning_rate) * float(reward) * chosen_features
                total_reward += float(reward)
                steps += 1
                if terminated or truncated:
                    break
            episode_history.append(
                {
                    "seed": int(seed),
                    "episode": int(episode),
                    "steps": int(steps),
                    "total_reward": round(float(total_reward), 8),
                    "completed_pairs": int(env.state.completed_pairs),
                    "slope_change_pct": float(info["slope_change_pct"]) if steps else 0.0,
                }
            )

    return {
        "learner_type": "tabular_preference_fallback",
        "feature_names": list(env.feature_names),
        "weights": [round(float(value), 10) for value in weights.tolist()],
        "training": {
            "episodes": int(episodes),
            "seeds": [int(seed) for seed in seeds],
            "learning_rate": float(learning_rate),
            "epsilon": float(epsilon),
            "history_tail": episode_history[-20:],
        },
    }


def evaluate_preference_policy(
    env: DongxingSlopeEnv,
    policy: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Evaluate a learned linear preference policy in DongxingSlopeEnv."""
    weights = np.asarray(policy["weights"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    env.reset(seed=int(seed))
    total_reward = 0.0
    selected_blocks: list[int] = []
    steps = 0

    while steps < env.max_steps:
        mask = env.action_masks()
        if not bool(mask.any()):
            break
        features = env.block_feature_matrix().astype(np.float64)
        action = _select_action(features, weights, mask, rng, epsilon=0.0)
        _, reward, terminated, truncated, info = env.step(action)
        selected_blocks.append(int(env.block_ids[action]))
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    gaps = [float(record["slope_gap"]) for record in env.state.pair_records]
    return {
        "strategy": "rl_lite_preference",
        "learner_type": policy.get("learner_type", "tabular_preference_fallback"),
        "seed": int(seed),
        "steps": int(steps),
        "selected_blocks": selected_blocks,
        "completed_pairs": int(env.state.completed_pairs),
        "initial_avg_farmland_slope": round(float(env.state.initial_avg_farmland_slope), 6),
        "final_avg_farmland_slope": round(float(env.state.avg_farmland_slope), 6),
        "slope_change_pct": round(float(env.state.slope_change_pct), 6),
        "total_reward": round(float(total_reward), 8),
        "mean_pair_slope_gap": round(mean(gaps), 6) if gaps else None,
        "total_pair_slope_gap": round(sum(gaps), 6),
        "unique_blocks": len(set(selected_blocks)),
    }


def run_experiment(
    parcels: list[dict[str, Any]],
    block_compositions: dict[str, list[int]],
    block_features: list[dict[str, Any]],
    train_seeds: list[int],
    eval_seeds: list[int],
    episodes: int,
    max_steps: int = 100,
    swaps_per_step: int = 5,
) -> dict[str, Any]:
    block_ids = [int(block["block_id"]) for block in block_features]
    env = DongxingSlopeEnv(
        parcels=parcels,
        block_compositions=block_compositions,
        block_ids=block_ids,
        max_steps=max_steps,
        swaps_per_step=swaps_per_step,
    )
    t0 = time.time()
    policy = train_tabular_preference_policy(
        env=env,
        seeds=train_seeds,
        episodes=episodes,
    )
    training_time_s = time.time() - t0

    learned_runs = [
        evaluate_preference_policy(
            env=DongxingSlopeEnv(
                parcels=parcels,
                block_compositions=block_compositions,
                block_ids=block_ids,
                max_steps=max_steps,
                swaps_per_step=swaps_per_step,
            ),
            policy=policy,
            seed=int(seed),
        )
        for seed in eval_seeds
    ]
    baselines = run_baseline_suite(
        parcels=parcels,
        block_compositions=block_compositions,
        block_features=block_features,
        max_steps=max_steps,
        swaps_per_step=swaps_per_step,
        random_seeds=eval_seeds,
    )

    return {
        "description": (
            "Dongxing slope-only RL-lite experiment. This tests external-county "
            "learned block-selection actionability under paired-swap slope "
            "dynamics; it is not full Bishan reward transfer."
        ),
        "status": "supported_as_slope_only_rl_actionability",
        "learner_type": policy["learner_type"],
        "claim_boundary": (
            "External slope-only learned policy evaluation, not full cross-county "
            "learned-policy transfer and not a contiguity/baimu reward test."
        ),
        "max_steps": int(max_steps),
        "swaps_per_step": int(swaps_per_step),
        "n_parcels": int(len(parcels)),
        "n_blocks": int(len(block_ids)),
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "training_time_s": round(float(training_time_s), 6),
        "policy": policy,
        "learned_policy": {
            "summary": _summarize_runs(learned_runs),
            "per_seed": [_compact_run(run) for run in learned_runs],
        },
        "baselines": baselines,
        "comparisons": _compare_to_baselines(learned_runs, baselines),
    }


def _select_action(
    features: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    epsilon: float,
) -> int:
    feasible = np.flatnonzero(mask)
    if len(feasible) == 0:
        raise ValueError("No feasible action is available")
    if epsilon > 0 and float(rng.random()) < float(epsilon):
        return int(rng.choice(feasible))
    scores = features @ weights
    scores = scores.astype(np.float64)
    scores[~mask] = -np.inf
    scores += rng.normal(0.0, 1e-9, size=scores.shape)
    return int(np.argmax(scores))


def _summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_seeds": len(runs)}
    for field_name in (
        "slope_change_pct",
        "completed_pairs",
        "mean_pair_slope_gap",
        "total_pair_slope_gap",
        "total_reward",
        "unique_blocks",
    ):
        values = [
            float(run[field_name])
            for run in runs
            if run.get(field_name) is not None and math.isfinite(float(run[field_name]))
        ]
        for key, value in _distribution_stats(values).items():
            summary[f"{field_name}_{key}"] = value
    return summary


def _compact_run(run: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "strategy",
        "learner_type",
        "seed",
        "steps",
        "completed_pairs",
        "initial_avg_farmland_slope",
        "final_avg_farmland_slope",
        "slope_change_pct",
        "total_reward",
        "mean_pair_slope_gap",
        "total_pair_slope_gap",
        "unique_blocks",
    )
    return {field_name: run.get(field_name) for field_name in fields}


def _compare_to_baselines(
    learned_runs: list[dict[str, Any]], baselines: dict[str, Any]
) -> dict[str, Any]:
    learned_summary = _summarize_runs(learned_runs)
    learned_slope = learned_summary.get("slope_change_pct_mean")
    comparisons: dict[str, Any] = {
        "lower_slope_change_pct_is_better": True,
        "learned_slope_change_pct_mean": learned_slope,
    }
    if learned_slope is None:
        return comparisons

    random_mean = baselines["random_baseline"].get("slope_change_pct_mean")
    if random_mean is not None:
        comparisons["random_slope_change_pct_mean"] = random_mean
        comparisons["learned_minus_random_slope_change_pct"] = round(
            float(learned_slope) - float(random_mean), 6
        )

    for strategy, result in baselines.get("strategies", {}).items():
        baseline_slope = result.get("slope_change_pct")
        if baseline_slope is None:
            continue
        comparisons[f"learned_minus_{strategy}_slope_change_pct"] = round(
            float(learned_slope) - float(baseline_slope), 6
        )
    return comparisons


def _distribution_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None}
    return {
        "mean": round(mean(values), 6),
        "sd": round(pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--train-seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/revision/dongxing_rl_lite.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    block_features = json.loads((args.block_dir / "block_features.json").read_text(encoding="utf-8"))
    block_compositions = json.loads((args.block_dir / "block_compositions.json").read_text(encoding="utf-8"))
    parcels = load_swappable_parcels(args.dltb, args.block_dir / "parcel_block_mapping.csv")
    report = run_experiment(
        parcels=parcels,
        block_compositions=block_compositions,
        block_features=block_features,
        train_seeds=_parse_int_list(args.train_seeds),
        eval_seeds=_parse_int_list(args.eval_seeds),
        episodes=args.episodes,
        max_steps=args.max_steps,
        swaps_per_step=args.swaps_per_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
