"""Dongxing multi-step learned-environment policy optimization experiment."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper7.dongxing_full_baselines import _block_features_from_obs, summarize_runs
from paper7.dongxing_full_env import build_dongxing_full_env
from paper7.dongxing_full_model_based_policy import (
    fit_one_step_model,
    predict_one_step,
    summarize_model_based_runs,
)
from paper7.dongxing_full_transition_diagnostics import (
    TRANSITION_POLICIES,
    collect_transition_rows,
)
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC, K_GLOBAL_GENERIC


def rollout_surrogate_policy(
    initial_obs: np.ndarray,
    n_blocks: int,
    model: dict[str, Any],
    weights: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Roll a linear block-scoring policy forward in the learned environment."""
    obs = np.asarray(initial_obs, dtype=np.float64).copy()
    n_blocks = int(n_blocks)
    total_reward = 0.0
    selected_actions: list[int] = []
    for _ in range(int(horizon)):
        block_features = _block_features_from_obs(obs, n_blocks).astype(np.float64)
        mask = block_features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        scores = _score_policy(block_features, np.asarray(weights, dtype=np.float64))
        scores[~mask] = -np.inf
        action = int(np.argmax(scores))
        global_features = _global_features_from_obs(obs, n_blocks)
        predicted = predict_one_step(model, block_features[action], global_features)
        obs = _apply_predicted_step(
            obs=obs,
            n_blocks=n_blocks,
            action=action,
            predicted=predicted,
        )
        total_reward += float(predicted["reward"])
        selected_actions.append(action)
    return {
        "steps": len(selected_actions),
        "predicted_reward_sum": round(float(total_reward), 6),
        "selected_actions": selected_actions,
        "final_obs": obs.astype(np.float32),
    }


def optimize_policy_weights_cem(
    initial_observations: list[np.ndarray],
    n_blocks: int,
    model: dict[str, Any],
    horizon: int,
    iterations: int,
    population_size: int,
    elite_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Optimize linear policy weights inside the learned environment with CEM."""
    if not initial_observations:
        raise ValueError("At least one initial observation is required")
    rng = np.random.default_rng(int(seed))
    dim = K_BLOCK_GENERIC + 1
    mean = np.zeros(dim, dtype=np.float64)
    mean[0] = 1.0
    mean[-1] = 1.0
    scale = np.ones(dim, dtype=np.float64)
    elite_count = max(1, int(round(int(population_size) * float(elite_frac))))
    history: list[dict[str, Any]] = []
    best_weights = mean.copy()
    best_score = -np.inf

    for iteration in range(int(iterations)):
        population = rng.normal(mean, scale, size=(int(population_size), dim))
        scores = np.asarray(
            [
                _mean_surrogate_score(
                    initial_observations=initial_observations,
                    n_blocks=n_blocks,
                    model=model,
                    weights=weights,
                    horizon=horizon,
                )
                for weights in population
            ],
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

    return {
        "optimizer": "cross_entropy_method",
        "horizon": int(horizon),
        "iterations": int(iterations),
        "population_size": int(population_size),
        "elite_frac": float(elite_frac),
        "seed": int(seed),
        "weights": [round(float(value), 10) for value in best_weights.tolist()],
        "history": history,
        "best_surrogate_score": round(float(best_score), 6),
    }


def evaluate_multistep_policy_real(
    env: GenericCountyEnv,
    model: dict[str, Any],
    weights: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the learned-environment-optimized linear policy in the real env."""
    obs, _ = env.reset(seed=int(seed))
    done = False
    total_reward = 0.0
    selected_blocks: list[int] = []
    last_info: dict[str, Any] = {}
    while not done:
        block_features = _block_features_from_obs(obs, env.n_blocks).astype(np.float64)
        mask = block_features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        scores = _score_policy(block_features, np.asarray(weights, dtype=np.float64))
        scores[~mask] = -np.inf
        action = int(np.argmax(scores))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return {
        "policy": "multistep_learned_env_optimized",
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
        "transition_model_used_for_training": True,
    }


def run_experiment(
    env_factory: Callable[[], GenericCountyEnv],
    baseline_path: Path,
    collection_policies: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    max_steps: int,
    ridge: float,
    cem_iterations: int,
    population_size: int,
    elite_frac: float,
    optimizer_seed: int,
) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    rows = collect_transition_rows(
        env_factory=env_factory,
        policies=collection_policies,
        seeds=train_seeds,
        max_steps=max_steps,
    )
    model = fit_one_step_model(rows, ridge=ridge)
    initial_observations = _collect_initial_observations(env_factory, train_seeds)
    optimizer = optimize_policy_weights_cem(
        initial_observations=initial_observations,
        n_blocks=env_factory().n_blocks,
        model=model,
        horizon=max_steps,
        iterations=cem_iterations,
        population_size=population_size,
        elite_frac=elite_frac,
        seed=optimizer_seed,
    )
    weights = np.asarray(optimizer["weights"], dtype=np.float64)
    eval_env = env_factory()
    runs = [
        evaluate_multistep_policy_real(eval_env, model, weights, seed=int(seed))
        for seed in eval_seeds
    ]
    summary, comparisons = summarize_model_based_runs(runs, baseline_report)
    return {
        "description": (
            "Dongxing multi-step learned-environment policy optimization with "
            "CEM-trained linear block policy and final full real-environment evaluation."
        ),
        "status": "supported_as_dongxing_multistep_learned_environment_policy",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "collection_policies": list(collection_policies),
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "max_steps": int(max_steps),
        "ridge": float(ridge),
        "n_training_transitions": int(len(rows)),
        "optimizer": optimizer,
        "real_environment_eval": {"summary": summary, "runs": runs},
        "baseline_path": os.fspath(baseline_path),
        "comparisons": comparisons,
        "mbrl_transition_model_used": True,
        "planning_horizon": int(max_steps),
        "multi_step_mbrl_planning_tested": True,
        "policy_transfer_tested": False,
        "claim_boundary": (
            "Local Dongxing multi-step learned-environment policy optimization; "
            "final metrics are evaluated in the full real environment. This is "
            "not direct Bishan-to-Dongxing policy transfer."
        ),
    }


def _score_policy(block_features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    feature_weights = np.asarray(weights[:K_BLOCK_GENERIC], dtype=np.float64)
    bias = float(weights[K_BLOCK_GENERIC]) if len(weights) > K_BLOCK_GENERIC else 0.0
    return block_features.astype(np.float64) @ feature_weights + bias


def _mean_surrogate_score(
    initial_observations: list[np.ndarray],
    n_blocks: int,
    model: dict[str, Any],
    weights: np.ndarray,
    horizon: int,
) -> float:
    scores = [
        rollout_surrogate_policy(
            initial_obs=obs,
            n_blocks=n_blocks,
            model=model,
            weights=weights,
            horizon=horizon,
        )["predicted_reward_sum"]
        for obs in initial_observations
    ]
    return float(np.mean(scores))


def _apply_predicted_step(
    obs: np.ndarray,
    n_blocks: int,
    action: int,
    predicted: dict[str, Any],
) -> np.ndarray:
    next_obs = np.asarray(obs, dtype=np.float64).copy()
    start = int(action) * K_BLOCK_GENERIC
    selected = np.asarray(predicted["next_selected_block_features"], dtype=np.float64)
    global_features = np.asarray(predicted["next_global_features"], dtype=np.float64)
    next_obs[start : start + K_BLOCK_GENERIC] = selected[:K_BLOCK_GENERIC]
    global_start = int(n_blocks) * K_BLOCK_GENERIC
    next_obs[global_start : global_start + K_GLOBAL_GENERIC] = global_features[:K_GLOBAL_GENERIC]
    next_obs[: int(n_blocks) * K_BLOCK_GENERIC : K_BLOCK_GENERIC] = np.maximum(
        next_obs[: int(n_blocks) * K_BLOCK_GENERIC : K_BLOCK_GENERIC],
        0.0,
    )
    return next_obs


def _global_features_from_obs(obs: np.ndarray, n_blocks: int) -> np.ndarray:
    start = int(n_blocks) * K_BLOCK_GENERIC
    return np.asarray(obs[start : start + K_GLOBAL_GENERIC], dtype=np.float64)


def _collect_initial_observations(
    env_factory: Callable[[], GenericCountyEnv],
    seeds: list[int],
) -> list[np.ndarray]:
    observations: list[np.ndarray] = []
    env = env_factory()
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        observations.append(np.asarray(obs, dtype=np.float32))
    return observations


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(raw)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--baseline", type=Path, default=Path("paper7/results/full_rigor/dongxing_full_baselines.json"))
    parser.add_argument("--collection-policies", default=",".join(TRANSITION_POLICIES))
    parser.add_argument("--train-seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seeds", default="10,11,12,13,14")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--cem-iterations", type=int, default=12)
    parser.add_argument("--population-size", type=int, default=48)
    parser.add_argument("--elite-frac", type=float, default=0.25)
    parser.add_argument("--optimizer-seed", type=int, default=0)
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/full_rigor/dongxing_multistep_mbrl_policy.json"),
    )
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
        collection_policies=_parse_csv_list(args.collection_policies),
        train_seeds=_parse_int_list(args.train_seeds),
        eval_seeds=_parse_int_list(args.eval_seeds),
        max_steps=args.max_steps,
        ridge=args.ridge,
        cem_iterations=args.cem_iterations,
        population_size=args.population_size,
        elite_frac=args.elite_frac,
        optimizer_seed=args.optimizer_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "n_training_transitions": report["n_training_transitions"],
                "n_eval": len(report["real_environment_eval"]["runs"]),
                "planning_horizon": report["planning_horizon"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
