"""Collect real-environment reward component rollouts for Paper 7."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from county_env import K_BLOCK, K_GLOBAL_COUNTY, CountyLevelEnv
from paper7.reward_components import (
    RewardComponents,
    compute_scalar_reward,
    default_reward_weights,
)


POLICIES = (
    "random",
    "dynamic_slope_gap",
    "area_weighted_slope_gap",
    "contiguity_aware",
    "baimu_aware",
    "scalarized_default",
)


def component_from_step_state(
    *,
    prev_slope: float,
    cur_slope: float,
    initial_slope: float,
    prev_cont: float,
    cur_cont: float,
    initial_cont: float,
    prev_baimu_area: float,
    cur_baimu_area: float,
    initial_farm_area: float,
    prev_baimu_count: int,
    cur_baimu_count: int,
    completed_swaps: int,
) -> RewardComponents:
    return RewardComponents(
        slope_delta=(float(prev_slope) - float(cur_slope)) / (abs(float(initial_slope)) + 1e-8),
        cont_delta=(float(cur_cont) - float(prev_cont)) / (abs(float(initial_cont)) + 1e-8),
        baimu_area_delta=(float(cur_baimu_area) - float(prev_baimu_area))
        / (float(initial_farm_area) + 1e-8),
        baimu_new_count=max(0, int(cur_baimu_count) - int(prev_baimu_count)),
        completed_swaps=int(completed_swaps),
    )


def choose_action(
    policy: str,
    block_features: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return 0
    if policy == "random":
        return int(rng.choice(valid))

    scores = np.full(block_features.shape[0], -np.inf, dtype=np.float64)
    gain = block_features[:, 3].astype(np.float64)
    farm_area = block_features[:, 7].astype(np.float64)
    forest_area = block_features[:, 8].astype(np.float64)
    swap_potential = block_features[:, 9].astype(np.float64)
    neighbor_invest = block_features[:, 13].astype(np.float64)
    neighbor_farm = block_features[:, 14].astype(np.float64)
    current_farm = block_features[:, 15].astype(np.float64)

    if policy == "dynamic_slope_gap":
        raw = gain
    elif policy == "area_weighted_slope_gap":
        raw = gain * np.minimum(farm_area, forest_area)
    elif policy == "contiguity_aware":
        raw = gain + 0.25 * neighbor_farm + 0.10 * neighbor_invest
    elif policy == "baimu_aware":
        raw = gain + 0.50 * current_farm + 0.25 * swap_potential
    elif policy == "scalarized_default":
        raw = 4000.0 * gain + 500.0 * neighbor_farm + 1500.0 * current_farm + 5.0 * swap_potential
    else:
        raise ValueError(f"Unsupported policy {policy!r}")

    scores[valid] = raw[valid]
    return int(np.argmax(scores))


def run_episode(
    policy: str,
    seed: int,
    budget: int = 500,
    swaps_per_step: int = 5,
    env: CountyLevelEnv | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if env is None:
        env = CountyLevelEnv(total_budget=budget, swaps_per_step=swaps_per_step)
    obs, _ = env.reset(seed=seed)
    done = False
    steps: list[dict[str, Any]] = []

    while not done:
        prev_slope = float(env.prev_slope)
        prev_cont = float(env.prev_cont)
        prev_baimu_area = float(env.prev_baimu_area)
        prev_baimu_count = int(env.prev_baimu_count)

        block_features = obs[: env.n_blocks * K_BLOCK].reshape(env.n_blocks, K_BLOCK)
        mask = env.action_masks()
        action = choose_action(policy, block_features, mask, rng)
        next_obs, real_reward, terminated, truncated, info = env.step(action)
        component = component_from_step_state(
            prev_slope=prev_slope,
            cur_slope=float(env.avg_farmland_slope),
            initial_slope=float(env.initial_slope),
            prev_cont=prev_cont,
            cur_cont=float(env.contiguity),
            initial_cont=float(env.initial_cont),
            prev_baimu_area=prev_baimu_area,
            cur_baimu_area=float(env.baimu_total_area),
            initial_farm_area=float(env.initial_farm_area),
            prev_baimu_count=prev_baimu_count,
            cur_baimu_count=int(env.baimu_count),
            completed_swaps=int(info.get("completed_swaps", 0)),
        )
        recomputed_reward = compute_scalar_reward(component, default_reward_weights())
        step = {
            "step": int(info.get("step", len(steps) + 1)),
            "action": int(action),
            "real_reward": float(real_reward),
            "reward_default": float(recomputed_reward),
            "reward_diff_default_vs_env": float(recomputed_reward - real_reward),
            **component.to_dict(),
            "slope_change_pct": float(info.get("slope_change_pct", 0.0)),
            "cont_change": float(info.get("cont_change", 0.0)),
            "baimu_count_change": int(info.get("baimu_count_change", 0)),
            "baimu_area_change_ha": float(info.get("baimu_area_change_ha", 0.0)),
            "budget_used": int(info.get("budget_used", 0)),
            "completed_swaps_info": int(info.get("completed_swaps", 0)),
        }
        steps.append(step)
        obs = next_obs
        done = terminated or truncated

    return {
        "policy": policy,
        "seed": int(seed),
        "budget": int(budget),
        "swaps_per_step": int(swaps_per_step),
        "summary": summarize_episode(policy, seed, steps),
        "steps": steps,
    }


def summarize_episode(policy: str, seed: int, steps: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "policy": policy,
        "seed": int(seed),
        "steps": len(steps),
        "reward_default": round(sum(float(step["reward_default"]) for step in steps), 6),
        "real_reward": round(sum(float(step.get("real_reward", 0.0)) for step in steps), 6),
        "max_abs_reward_recompute_error": round(
            max([abs(float(step.get("reward_diff_default_vs_env", 0.0))) for step in steps] or [0.0]),
            12,
        ),
        "slope_delta_total": round(sum(float(step["slope_delta"]) for step in steps), 6),
        "cont_delta_total": round(sum(float(step["cont_delta"]) for step in steps), 6),
        "baimu_area_delta_total": round(sum(float(step["baimu_area_delta"]) for step in steps), 6),
        "baimu_new_count_total": int(sum(int(step["baimu_new_count"]) for step in steps)),
        "completed_swaps_total": int(sum(int(step["completed_swaps"]) for step in steps)),
    }
    if steps:
        last = steps[-1]
        summary.update(
            {
                "final_slope_change_pct": round(float(last["slope_change_pct"]), 6),
                "final_cont_change": round(float(last["cont_change"]), 6),
                "final_baimu_count_change": int(last["baimu_count_change"]),
                "final_baimu_area_change_ha": round(float(last["baimu_area_change_ha"]), 6),
                "final_budget_used": int(last["budget_used"]),
            }
        )
    return summary


def run_suite(policies: list[str], seeds: list[int], budget: int, swaps_per_step: int) -> dict[str, Any]:
    episodes = []
    env = CountyLevelEnv(total_budget=budget, swaps_per_step=swaps_per_step)
    for policy in policies:
        for seed in seeds:
            episodes.append(
                run_episode(
                    policy,
                    seed,
                    budget=budget,
                    swaps_per_step=swaps_per_step,
                    env=env,
                )
            )
    return {
        "description": "Real CountyLevelEnv reward-component rollouts for reward-weight sensitivity.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policies": policies,
        "seeds": seeds,
        "budget": int(budget),
        "swaps_per_step": int(swaps_per_step),
        "k_block": K_BLOCK,
        "k_global": K_GLOBAL_COUNTY,
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/full_rigor/reward_component_rollouts.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"Unsupported policies: {unknown}")
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    report = run_suite(policies, seeds, args.budget, args.swaps_per_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "n_episodes": len(report["episodes"])}, indent=2))


if __name__ == "__main__":
    main()
