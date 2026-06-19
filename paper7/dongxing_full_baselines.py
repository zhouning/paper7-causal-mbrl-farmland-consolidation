"""Full multi-objective Dongxing real-environment baselines."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper7.dongxing_full_env import build_dongxing_full_env
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC


POLICIES = (
    "random",
    "dynamic_slope_gap",
    "area_weighted_slope_gap",
    "contiguity_aware",
    "baimu_aware",
    "scalarized_default",
)


def choose_full_env_action(
    policy: str,
    block_features: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    if policy == "random":
        return int(rng.choice(valid))

    scores = np.full(block_features.shape[0], -np.inf, dtype=np.float64)
    gain = block_features[:, 0].astype(np.float64)
    exchange_area = block_features[:, 1].astype(np.float64)
    farm_area = block_features[:, 2].astype(np.float64)
    forest_area = block_features[:, 3].astype(np.float64)
    current_farm = block_features[:, 4].astype(np.float64)
    neighbor_farm = block_features[:, 5].astype(np.float64)

    if policy == "dynamic_slope_gap":
        raw = gain
    elif policy == "area_weighted_slope_gap":
        raw = gain * exchange_area
    elif policy == "contiguity_aware":
        raw = gain + 0.50 * neighbor_farm + 0.10 * current_farm
    elif policy == "baimu_aware":
        raw = gain + 0.50 * current_farm + 0.25 * np.minimum(farm_area, forest_area)
    elif policy == "scalarized_default":
        raw = 4000.0 * gain + 500.0 * neighbor_farm + 1500.0 * current_farm
    else:
        raise ValueError(f"Unsupported full-env policy {policy!r}")

    scores[valid] = raw[valid]
    return int(np.argmax(scores))


def run_policy_episode(env: GenericCountyEnv, policy: str, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0.0
    selected_blocks: list[int] = []
    last_info: dict[str, Any] = {}
    while not done:
        features = _block_features_from_obs(obs, env.n_blocks)
        mask = features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        action = choose_full_env_action(policy, features, mask, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))

    return {
        "policy": policy,
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


def _block_features_from_obs(obs: np.ndarray, n_blocks: int) -> np.ndarray:
    return np.asarray(obs[: n_blocks * K_BLOCK_GENERIC], dtype=np.float32).reshape(
        n_blocks,
        K_BLOCK_GENERIC,
    )


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(runs)}
    for field in (
        "slope_change_pct",
        "cont_change",
        "baimu_count_change",
        "baimu_area_change_ha",
        "reward",
        "completed_swaps",
        "unique_blocks",
    ):
        values = [float(row[field]) for row in runs if row.get(field) is not None]
        for key, value in _stats(values).items():
            summary[f"{field}_{key}"] = value
    return summary


def run_suite(
    env_factory: Callable[[], GenericCountyEnv],
    policies: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    env = env_factory()
    _, initial_info = env.reset(seed=0)
    runs = []
    for policy in policies:
        for seed in seeds:
            runs.append(run_policy_episode(env, policy, seed))
    policy_summaries = {
        policy: summarize_runs([row for row in runs if row["policy"] == policy])
        for policy in policies
    }
    return {
        "description": (
            "Dongxing full real-environment baselines with slope, contiguity, "
            "baimu fang, and default scalar reward."
        ),
        "status": "supported_as_full_real_environment_baselines",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "n_policies": len(policies),
        "policies": list(policies),
        "seeds": [int(seed) for seed in seeds],
        "initial_environment": initial_info,
        "policy_summaries": policy_summaries,
        "runs": runs,
        "claim_boundary": (
            "Full real-environment baseline evidence; no learned Dongxing policy "
            "or cross-county transfer is tested here."
        ),
    }


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sd": None, "min": None, "max": None}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": round(mean, 6),
        "sd": round(variance**0.5, 6),
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/full_rigor/dongxing_full_baselines.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = _parse_csv_list(args.policies)
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"Unsupported policies: {unknown}")
    seeds = _parse_int_list(args.seeds)

    def env_factory() -> GenericCountyEnv:
        return build_dongxing_full_env(
            dltb_path=args.dltb,
            block_dir=args.block_dir,
            total_budget=args.total_budget,
            swaps_per_step=args.swaps_per_step,
        )

    report = run_suite(env_factory, policies, seeds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "n_runs": report["n_runs"]}, indent=2))


if __name__ == "__main__":
    main()
