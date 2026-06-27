"""Scenario-based Dongxing robustness experiments for Paper 7."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper7.dongxing_full_baselines import (
    POLICIES as BASELINE_POLICIES,
    _block_features_from_obs,
    choose_full_env_action,
    summarize_runs,
)
from paper7.dongxing_full_env import load_dongxing_parcels_for_full_env
from paper7.dongxing_full_learned_policy import evaluate_preference_policy
from paper7.dongxing_full_model_based_policy import (
    evaluate_model_based_policy,
    fit_one_step_model,
)
from paper7.dongxing_full_transition_diagnostics import (
    TRANSITION_POLICIES,
    collect_transition_rows,
)
from paper7.dongxing_model_based_optimization import evaluate_candidate_policy
from paper7.dongxing_multistep_mbrl_policy import (
    _score_policy,
    optimize_policy_weights_cem,
)
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: str
    slope_scale: float
    slope_noise_sd: float
    slope_noise_seed: int
    total_budget: int
    swaps_per_step: int
    description: str


def build_default_scenario_specs() -> list[ScenarioSpec]:
    return [
        ScenarioSpec("base", "selection", 1.0, 0.0, 0, 500, 5, "Base Dongxing setting"),
        ScenarioSpec("slope_scale_low", "selection", 0.95, 0.0, 0, 500, 5, "Five percent lower slopes"),
        ScenarioSpec("slope_scale_high", "selection", 1.05, 0.0, 0, 500, 5, "Five percent higher slopes"),
        ScenarioSpec("slope_noise_0", "selection", 1.0, 0.05, 0, 500, 5, "DEM-like slope noise seed 0"),
        ScenarioSpec("slope_noise_1", "heldout", 1.0, 0.05, 1, 500, 5, "DEM-like slope noise seed 1"),
        ScenarioSpec("slope_noise_2", "heldout", 1.0, 0.05, 2, 500, 5, "DEM-like slope noise seed 2"),
        ScenarioSpec("budget_low", "heldout", 1.0, 0.0, 0, 350, 5, "Lower total swap budget"),
        ScenarioSpec("budget_high", "heldout", 1.0, 0.0, 0, 650, 5, "Higher total swap budget"),
        ScenarioSpec("swap_fine", "heldout", 1.0, 0.0, 0, 500, 3, "Finer per-step execution"),
        ScenarioSpec("swap_coarse", "heldout", 1.0, 0.0, 0, 500, 7, "Coarser per-step execution"),
    ]


def apply_slope_perturbation(
    parcels: list[dict[str, Any]],
    scenario: ScenarioSpec,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(scenario.slope_noise_seed))
    updated: list[dict[str, Any]] = []
    for parcel in parcels:
        row = dict(parcel)
        slope = float(row["slope"]) * float(scenario.slope_scale)
        if float(scenario.slope_noise_sd) > 0:
            slope *= 1.0 + float(rng.normal(0.0, float(scenario.slope_noise_sd)))
        row["slope"] = max(0.0, float(slope))
        updated.append(row)
    return updated


def build_env_from_parcels_and_scenario(
    parcels: list[dict[str, Any]],
    block_compositions: dict[str, list[int]],
    block_ids: list[int],
    scenario: ScenarioSpec,
) -> GenericCountyEnv:
    perturbed = apply_slope_perturbation(parcels, scenario)
    return GenericCountyEnv(
        parcels=perturbed,
        block_compositions=block_compositions,
        block_ids=block_ids,
        total_budget=int(scenario.total_budget),
        swaps_per_step=int(scenario.swaps_per_step),
    )


def evaluate_baseline_policy_on_env(
    env: GenericCountyEnv,
    policy: str,
    scenario_id: str,
    seed: int,
) -> dict[str, Any]:
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
        action = choose_full_env_action(policy, features, mask, rng)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return _result_row(
        policy=policy,
        scenario_id=scenario_id,
        seed=seed,
        total_reward=total_reward,
        selected_blocks=selected_blocks,
        info=last_info,
        deterministic_policy=policy != "random",
    )


def evaluate_linear_weight_policy(
    env: GenericCountyEnv,
    weights: np.ndarray,
    policy_name: str,
    scenario_id: str,
    seed: int = 0,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=int(seed))
    done = False
    total_reward = 0.0
    selected_blocks: list[int] = []
    last_info: dict[str, Any] = {}
    while not done:
        features = _block_features_from_obs(obs, env.n_blocks).astype(np.float64)
        mask = features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        scores = _score_policy(features, np.asarray(weights, dtype=np.float64))
        scores[~mask] = -np.inf
        action = int(np.argmax(scores))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return _result_row(
        policy=policy_name,
        scenario_id=scenario_id,
        seed=seed,
        total_reward=total_reward,
        selected_blocks=selected_blocks,
        info=last_info,
        deterministic_policy=True,
    )


def _result_row(
    *,
    policy: str,
    scenario_id: str,
    seed: int,
    total_reward: float,
    selected_blocks: list[int],
    info: dict[str, Any],
    deterministic_policy: bool,
) -> dict[str, Any]:
    return {
        "policy": str(policy),
        "scenario_id": str(scenario_id),
        "seed": int(seed),
        "deterministic_policy": bool(deterministic_policy),
        "steps": int(info.get("step", 0)),
        "reward": round(float(total_reward), 6),
        "slope_change_pct": float(info.get("slope_change_pct", 0.0)),
        "cont_change": float(info.get("cont_change", 0.0)),
        "baimu_count_change": int(info.get("baimu_count_change", 0)),
        "baimu_area_change_ha": float(info.get("baimu_area_change_ha", 0.0)),
        "completed_swaps": int(info.get("budget_used", 0)),
        "unique_blocks": len(set(selected_blocks)),
        "selected_blocks_head": selected_blocks[:20],
    }


def summarize_policy_scenario_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        grouped.setdefault(str(row["policy"]), []).append(row)
    summaries: dict[str, dict[str, Any]] = {}
    for policy, rows in grouped.items():
        summary = summarize_runs(rows)
        scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
        deterministic = all(bool(row.get("deterministic_policy", False)) for row in rows)
        rewards = [float(row["reward"]) for row in rows]
        slopes = [float(row["slope_change_pct"]) for row in rows]
        summary.update(
            {
                "policy": policy,
                "scenario_count": len(scenario_ids),
                "scenario_ids": scenario_ids,
                "deterministic_policy": deterministic,
                "seed_replication_is_independent": not deterministic,
                "reward_worst": round(min(rewards), 6) if rewards else None,
                "slope_change_pct_worst": round(max(slopes), 6) if slopes else None,
            }
        )
        summaries[policy] = summary
    return summaries
