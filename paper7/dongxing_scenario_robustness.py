"""Scenario-based Dongxing robustness experiments for Paper 7."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
        deterministic = all(bool(row.get("deterministic_policy", False)) for row in rows)
        summary_rows = _deduplicate_deterministic_scenario_rows(rows) if deterministic else rows
        summary = summarize_runs(summary_rows)
        scenario_ids = sorted({str(row["scenario_id"]) for row in rows})
        rewards = [float(row["reward"]) for row in summary_rows]
        slopes = [float(row["slope_change_pct"]) for row in summary_rows]
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


def _deduplicate_deterministic_scenario_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for row in rows:
        selected.setdefault(str(row["scenario_id"]), row)
    return list(selected.values())


def optimize_scenario_robust_linear_policy(
    envs: list[GenericCountyEnv],
    iterations: int = 8,
    population_size: int = 32,
    elite_frac: float = 0.25,
    seed: int = 0,
    transition_model: dict[str, Any] | None = None,
    initial_observations: list[np.ndarray] | None = None,
    n_blocks: int | None = None,
    horizon: int | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not envs:
        raise ValueError("At least one scenario environment is required")
    iteration_count = int(iterations)
    if iteration_count <= 0:
        raise ValueError("iterations must be positive")
    population_count = int(population_size)
    if population_count <= 0:
        raise ValueError("population_size must be positive")
    elite_fraction = float(elite_frac)
    if not 0.0 < elite_fraction <= 1.0:
        raise ValueError("elite_frac must be in (0, 1]")

    if transition_model is not None:
        observations = list(initial_observations or [])
        if not observations:
            raise ValueError("At least one initial observation is required for surrogate optimization")
        block_count = int(n_blocks if n_blocks is not None else envs[0].n_blocks)
        planning_horizon = int(horizon if horizon is not None else max(env.max_steps for env in envs))
        optimizer = dict(
            optimize_policy_weights_cem(
                initial_observations=observations,
                n_blocks=block_count,
                model=transition_model,
                horizon=planning_horizon,
                iterations=iteration_count,
                population_size=population_count,
                elite_frac=elite_fraction,
                seed=int(seed),
            )
        )
        optimizer["optimizer"] = "cross_entropy_method_learned_scenario_surrogate"
        optimizer["scenario_selection_count"] = len(observations)
        optimizer["best_score"] = optimizer.get("best_surrogate_score")
        return np.asarray(optimizer["weights"], dtype=np.float64), optimizer

    rng = np.random.default_rng(int(seed))
    dim = K_BLOCK_GENERIC + 1
    mean = np.zeros(dim, dtype=np.float64)
    mean[0] = 1.0
    mean[-1] = 1.0
    scale = np.ones(dim, dtype=np.float64)
    elite_count = min(population_count, max(1, int(round(population_count * elite_fraction))))
    best_weights = mean.copy()
    best_score = -np.inf
    history: list[dict[str, Any]] = []

    for iteration in range(iteration_count):
        population = rng.normal(mean, scale, size=(population_count, dim))
        scores = np.asarray(
            [_mean_real_scenario_score(envs, weights) for weights in population],
            dtype=np.float64,
        )
        elite_indices = np.argsort(scores)[-elite_count:]
        elite = population[elite_indices]
        elite_scores = scores[elite_indices]
        mean = elite.mean(axis=0)
        scale = np.maximum(elite.std(axis=0), 0.05)
        if float(elite_scores[-1]) > best_score:
            best_score = float(elite_scores[-1])
            best_weights = population[int(elite_indices[-1])].copy()
        history.append(
            {
                "iteration": int(iteration),
                "population_mean_score": round(float(scores.mean()), 6),
                "elite_mean_score": round(float(elite_scores.mean()), 6),
                "best_score": round(float(best_score), 6),
            }
        )

    return best_weights, {
        "optimizer": "cross_entropy_method_real_scenario_smoke",
        "iterations": iteration_count,
        "population_size": population_count,
        "elite_frac": elite_fraction,
        "seed": int(seed),
        "weights": [round(float(value), 10) for value in best_weights.tolist()],
        "history": history,
        "best_score": round(float(best_score), 6),
    }


def _mean_real_scenario_score(envs: list[GenericCountyEnv], weights: np.ndarray) -> float:
    scores = []
    for env in envs:
        row = evaluate_linear_weight_policy(
            env=env,
            weights=np.asarray(weights, dtype=np.float64),
            policy_name="candidate",
            scenario_id="selection",
        )
        scores.append(float(row["reward"]))
    return float(np.mean(scores))


def _collect_scenario_transition_rows(
    selection_envs: list[GenericCountyEnv],
    policies: list[str],
    seeds: list[int],
    max_steps: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for env in selection_envs:
        rows.extend(
            collect_transition_rows(
                env_factory=lambda env=env: env,
                policies=policies,
                seeds=seeds,
                max_steps=max_steps,
            )
        )
    return rows


def _initial_observations_from_envs(envs: list[GenericCountyEnv]) -> list[np.ndarray]:
    observations: list[np.ndarray] = []
    for index, env in enumerate(envs):
        obs, _ = env.reset(seed=int(index))
        observations.append(np.asarray(obs, dtype=np.float32))
    return observations


def run_scenario_robustness_experiment(
    *,
    parcels: list[dict[str, Any]],
    block_compositions: dict[str, list[int]],
    block_ids: list[int],
    scenarios: list[ScenarioSpec],
    baseline_policies: list[str],
    random_seeds: list[int],
    cem_iterations: int,
    cem_population_size: int,
    output_path: Path | None,
) -> dict[str, Any]:
    scenario_ids = [str(spec.scenario_id) for spec in scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise ValueError("Scenario IDs must be unique")

    scenario_envs = {
        spec.scenario_id: build_env_from_parcels_and_scenario(
            parcels=parcels,
            block_compositions=block_compositions,
            block_ids=block_ids,
            scenario=spec,
        )
        for spec in scenarios
    }
    selection_envs = [
        scenario_envs[spec.scenario_id]
        for spec in scenarios
        if spec.split == "selection"
    ]
    if not selection_envs:
        raise ValueError("At least one selection scenario is required")
    transition_max_steps = min(100, max(env.max_steps for env in selection_envs))
    transition_rows = _collect_scenario_transition_rows(
        selection_envs=selection_envs,
        policies=list(TRANSITION_POLICIES),
        seeds=random_seeds,
        max_steps=transition_max_steps,
    )
    transition_model = fit_one_step_model(transition_rows, ridge=1e-3)
    initial_observations = _initial_observations_from_envs(selection_envs)
    weights, optimizer = optimize_scenario_robust_linear_policy(
        envs=selection_envs,
        iterations=cem_iterations,
        population_size=cem_population_size,
        elite_frac=0.25,
        seed=0,
        transition_model=transition_model,
        initial_observations=initial_observations,
        n_blocks=selection_envs[0].n_blocks,
        horizon=transition_max_steps,
    )
    optimizer.update(
        {
            "mbrl_transition_model_used": True,
            "n_training_transitions": len(transition_rows),
            "selection_scenario_ids": [
                str(spec.scenario_id) for spec in scenarios if spec.split == "selection"
            ],
            "transition_collection_policies": list(TRANSITION_POLICIES),
            "transition_collection_seeds": [int(seed) for seed in random_seeds],
            "transition_max_steps": int(transition_max_steps),
        }
    )

    runs: list[dict[str, Any]] = []
    for spec in scenarios:
        env = scenario_envs[spec.scenario_id]
        for policy in baseline_policies:
            seeds = random_seeds if policy == "random" else [0]
            for seed in seeds:
                runs.append(
                    evaluate_baseline_policy_on_env(
                        env=env,
                        policy=policy,
                        scenario_id=spec.scenario_id,
                        seed=int(seed),
                    )
                )
        runs.append(
            evaluate_linear_weight_policy(
                env=env,
                weights=weights,
                policy_name="scenario_robust_mbrl",
                scenario_id=spec.scenario_id,
            )
        )

    report = {
        "description": "Dongxing scenario-based robustness evaluation with a scenario-robust linear learned-environment planner.",
        "status": "supported_as_dongxing_scenario_robustness",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "optimizer_evaluation_environment": "learned_scenario_surrogate",
        "final_evaluation_environment": "real_scenario_environments",
        "scenario_count": len(scenarios),
        "scenarios": [asdict(spec) for spec in scenarios],
        "experiment_config": {
            "baseline_policies": list(baseline_policies),
            "random_seeds": [int(seed) for seed in random_seeds],
            "cem_iterations": int(cem_iterations),
            "cem_population_size": int(cem_population_size),
            "elite_frac": 0.25,
            "optimizer_seed": 0,
            "transition_collection_policies": list(TRANSITION_POLICIES),
            "transition_collection_seeds": [int(seed) for seed in random_seeds],
            "transition_max_steps": int(transition_max_steps),
        },
        "policy_summaries": summarize_policy_scenario_runs(runs),
        "runs": runs,
        "optimizer": optimizer,
        "mbrl_transition_model_used": True,
        "n_training_transitions": len(transition_rows),
        "planning_horizon": int(transition_max_steps),
        "deterministic_seed_repetition_avoided": True,
        "policy_transfer_tested": False,
        "claim_boundary": (
            "Scenario-based Dongxing robustness for local learned-environment planning; "
            "deterministic Dongxing seed repetitions are not treated as independent "
            "replications, and this is not direct Bishan-to-Dongxing policy transfer."
        ),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_block_package(block_dir: Path) -> tuple[dict[str, list[int]], list[int]]:
    block_compositions = json.loads((block_dir / "block_compositions.json").read_text(encoding="utf-8"))
    block_features = json.loads((block_dir / "block_features.json").read_text(encoding="utf-8"))
    block_ids = [int(item["block_id"]) for item in block_features]
    return block_compositions, block_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--policies", default="random,dynamic_slope_gap,scalarized_default,baimu_aware")
    parser.add_argument("--random-seeds", default="0,1,2")
    parser.add_argument("--cem-iterations", type=int, default=8)
    parser.add_argument("--cem-population-size", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/full_rigor/dongxing_scenario_robustness.json"),
    )
    return parser.parse_args()


def _parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv(raw)]


def main() -> None:
    args = parse_args()
    parcels = load_dongxing_parcels_for_full_env(args.dltb, args.block_dir)
    block_compositions, block_ids = load_block_package(args.block_dir)
    report = run_scenario_robustness_experiment(
        parcels=parcels,
        block_compositions=block_compositions,
        block_ids=block_ids,
        scenarios=build_default_scenario_specs(),
        baseline_policies=_parse_csv(args.policies),
        random_seeds=_parse_int_csv(args.random_seeds),
        cem_iterations=args.cem_iterations,
        cem_population_size=args.cem_population_size,
        output_path=None,
    )
    report["command_config"] = {
        "dltb": os.fspath(args.dltb),
        "block_dir": os.fspath(args.block_dir),
        "policies": _parse_csv(args.policies),
        "random_seeds": _parse_int_csv(args.random_seeds),
        "cem_iterations": int(args.cem_iterations),
        "cem_population_size": int(args.cem_population_size),
        "output": os.fspath(args.output),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "scenario_count": report["scenario_count"]}, indent=2))


if __name__ == "__main__":
    main()
