"""Strong non-learning baselines for the Bishan real environment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY


PolicyFn = Callable[[np.ndarray, np.ndarray, np.random.Generator], int]


def _masked_argmax(scores: np.ndarray, mask: np.ndarray) -> int:
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return 0
    masked_scores = np.full(len(scores), -np.inf, dtype=float)
    masked_scores[valid] = scores[valid]
    return int(np.argmax(masked_scores))


def choose_slope_gap_action(block_features: np.ndarray, mask: np.ndarray) -> int:
    """Choose the valid block with the largest normalized immediate slope gain."""
    return _masked_argmax(block_features[:, 3].astype(float), mask)


def choose_area_weighted_action(block_features: np.ndarray, mask: np.ndarray) -> int:
    """Choose high slope gain weighted by available swap potential and block area."""
    gain = np.maximum(block_features[:, 3].astype(float), 0.0)
    swap_potential = np.maximum(block_features[:, 9].astype(float), 0.0)
    area = np.maximum(block_features[:, 12].astype(float), 0.0)
    scores = gain * np.sqrt(swap_potential + 1e-8) * np.sqrt(area + 1e-8)
    return _masked_argmax(scores, mask)


def choose_contiguity_aware_action(block_features: np.ndarray, mask: np.ndarray) -> int:
    """Choose slope-gain blocks with compactness and adjacent-investment terms."""
    gain = np.maximum(block_features[:, 3].astype(float), 0.0)
    compactness = np.nan_to_num(block_features[:, 11].astype(float), nan=0.0)
    neighbor_invested = np.nan_to_num(block_features[:, 13].astype(float), nan=0.0)
    swap_potential = np.maximum(block_features[:, 9].astype(float), 0.0)
    scores = gain + 0.15 * compactness + 0.25 * neighbor_invested + 0.05 * swap_potential
    return _masked_argmax(scores, mask)


def estimate_immediate_slope_delta(env: Any, block_id: int) -> float:
    """Estimate immediate change in area-weighted county slope for one block."""
    parcels = env.block_parcels[int(block_id)]
    types = env.land_use[parcels]
    avail = ~env.swapped[parcels]
    farm_idx = parcels[(types == 1) & avail]
    forest_idx = parcels[(types == 2) & avail]
    if len(farm_idx) == 0 or len(forest_idx) == 0:
        return float("-inf")

    farm_order = farm_idx[np.argsort(env.slopes[farm_idx])[::-1]]
    forest_order = forest_idx[np.argsort(env.slopes[forest_idx])]
    n_pairs = min(int(getattr(env, "swaps_per_step", 1)), len(farm_order), len(forest_order))
    weighted = float(env.total_weighted_slope)
    area = float(env.total_farm_area)
    for farm, forest in zip(farm_order[:n_pairs], forest_order[:n_pairs]):
        if float(env.slopes[farm]) <= float(env.slopes[forest]):
            break
        weighted -= float(env.slopes[farm]) * float(env.areas[farm])
        weighted += float(env.slopes[forest]) * float(env.areas[forest])
        area -= float(env.areas[farm])
        area += float(env.areas[forest])
    if area <= 0:
        return float("inf")
    current = float(env.total_weighted_slope) / max(float(env.total_farm_area), 1e-8)
    next_slope = weighted / max(area, 1e-8)
    return current - next_slope


def choose_immediate_slope_delta_action(env: Any, mask: np.ndarray) -> int:
    """Choose the block with the best estimated immediate county-slope reduction."""
    scores = np.array([estimate_immediate_slope_delta(env, idx) for idx in range(len(mask))], dtype=float)
    return _masked_argmax(scores, mask)


def _split_obs(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_blocks = (len(obs) - K_GLOBAL_COUNTY) // K_BLOCK
    block_features = obs[: n_blocks * K_BLOCK].reshape(n_blocks, K_BLOCK)
    global_features = obs[n_blocks * K_BLOCK :]
    return block_features, global_features


def make_policy(name: str) -> PolicyFn:
    """Return a policy function by name."""
    key = name.lower()

    def random_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        valid = np.where(mask)[0]
        return int(rng.choice(valid)) if len(valid) else 0

    def slope_gap_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        block_features, _ = _split_obs(obs)
        return choose_slope_gap_action(block_features, mask[: len(block_features)])

    def area_weighted_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        block_features, _ = _split_obs(obs)
        return choose_area_weighted_action(block_features, mask[: len(block_features)])

    def contiguity_aware_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        block_features, _ = _split_obs(obs)
        return choose_contiguity_aware_action(block_features, mask[: len(block_features)])

    def immediate_slope_policy(obs: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int:
        raise RuntimeError("immediate_slope_delta requires direct environment access")

    policies: dict[str, PolicyFn] = {
        "random": random_policy,
        "slope_gap_greedy": slope_gap_policy,
        "area_weighted_greedy": area_weighted_policy,
        "contiguity_aware_greedy": contiguity_aware_policy,
        "immediate_slope_delta": immediate_slope_policy,
    }
    if key not in policies:
        raise ValueError(f"Unknown policy: {name}")
    return policies[key]


def run_policy_episode(
    policy_name: str,
    seed: int,
    budget: int = 500,
    swaps_per_step: int = 5,
) -> dict[str, float | int | str]:
    """Run one deterministic or seeded policy episode in CountyLevelEnv."""
    rng = np.random.default_rng(seed)
    env = CountyLevelEnv(total_budget=budget, swaps_per_step=swaps_per_step)
    policy = make_policy(policy_name)
    obs, info = env.reset(seed=seed)
    initial_info = dict(info)
    done = False
    total_reward = 0.0
    last_info = dict(info)
    while not done:
        mask = env.action_masks()
        if policy_name == "immediate_slope_delta":
            action = choose_immediate_slope_delta_action(env, mask)
        else:
            action = policy(obs, mask, rng)
        obs, reward, terminated, truncated, last_info = env.step(int(action))
        total_reward += float(reward)
        done = bool(terminated or truncated)
    return {
        "policy": policy_name,
        "seed": int(seed),
        "budget": int(budget),
        "reward": round(float(total_reward), 6),
        "initial_avg_slope": round(float(initial_info.get("avg_slope", 0.0)), 6),
        "final_avg_slope": round(float(last_info.get("avg_slope", 0.0)), 6),
        "slope_change_pct": round(float(last_info.get("slope_change_pct", 0.0)), 6),
        "initial_contiguity": round(float(initial_info.get("contiguity", 0.0)), 6),
        "final_contiguity": round(float(last_info.get("contiguity", 0.0)), 6),
        "cont_change": round(float(last_info.get("cont_change", 0.0)), 6),
        "initial_baimu_count": int(initial_info.get("baimu_count", 0)),
        "final_baimu_count": int(last_info.get("baimu_count", 0)),
        "baimu_count_change": int(last_info.get("baimu_count_change", 0)),
        "final_baimu_area_ha": round(float(last_info.get("baimu_area_ha", 0.0)), 6),
        "baimu_area_change_ha": round(float(last_info.get("baimu_area_change_ha", 0.0)), 6),
        "budget_used": int(last_info.get("budget_used", 0)),
        "steps": int(last_info.get("step", 0)),
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def summarize_policy_runs(policy_name: str, runs: list[dict[str, Any]]) -> dict[str, float | int | str]:
    """Summarize repeated runs for one policy."""
    summary: dict[str, float | int | str] = {"policy": policy_name, "n_runs": int(len(runs))}
    metrics = [
        "reward",
        "slope_change_pct",
        "cont_change",
        "baimu_count_change",
        "baimu_area_change_ha",
        "budget_used",
    ]
    for metric in metrics:
        if not all(metric in run for run in runs):
            continue
        mean, std = _mean_std([float(run[metric]) for run in runs])
        summary[f"{metric}_mean"] = round(mean, 6)
        summary[f"{metric}_std"] = round(std, 6)
    completion = [float(run["budget_used"]) / max(float(run.get("budget", 1)), 1.0) for run in runs]
    mean, std = _mean_std(completion)
    summary["budget_completion_mean"] = round(mean, 6)
    summary["budget_completion_std"] = round(std, 6)
    return summary


def run_baselines(
    policies: list[str],
    random_seeds: list[int],
    deterministic_seeds: list[int],
    budget: int = 500,
    swaps_per_step: int = 5,
) -> dict[str, Any]:
    """Run all requested baseline policies."""
    all_runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for policy in policies:
        seeds = random_seeds if policy == "random" else deterministic_seeds
        policy_runs = [
            run_policy_episode(policy, seed=seed, budget=budget, swaps_per_step=swaps_per_step)
            for seed in seeds
        ]
        all_runs.extend(policy_runs)
        summaries.append(summarize_policy_runs(policy, policy_runs))
    return {
        "description": "Bishan real-environment non-learning baselines.",
        "budget": int(budget),
        "swaps_per_step": int(swaps_per_step),
        "runs": all_runs,
        "summary": summaries,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies",
        default="random,slope_gap_greedy,area_weighted_greedy,contiguity_aware_greedy,immediate_slope_delta",
    )
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    parser.add_argument("--deterministic-seeds", default="0")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/revision/bishan_strong_baselines.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [item.strip() for item in str(args.policies).split(",") if item.strip()]
    random_seeds = [int(item) for item in str(args.random_seeds).split(",") if item.strip()]
    deterministic_seeds = [int(item) for item in str(args.deterministic_seeds).split(",") if item.strip()]
    results = run_baselines(
        policies=policies,
        random_seeds=random_seeds,
        deterministic_seeds=deterministic_seeds,
        budget=args.budget,
        swaps_per_step=args.swaps_per_step,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
