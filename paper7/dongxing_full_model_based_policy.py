"""One-step learned model-based Dongxing full-environment policy experiment."""

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
from paper7.dongxing_full_transition_diagnostics import (
    TRANSITION_POLICIES,
    collect_transition_rows,
)
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC, K_GLOBAL_GENERIC


def fit_one_step_model(rows: list[dict[str, Any]], ridge: float = 1e-3) -> dict[str, Any]:
    if not rows:
        raise ValueError("At least one transition row is required")
    x = _design_matrix(rows)
    scaler = _fit_standardizer(x)
    x_scaled = _apply_standardizer(x, scaler)
    y_selected = np.asarray([row["next_selected_block_features"] for row in rows], dtype=np.float64)
    y_global = np.asarray([row["next_global_features"] for row in rows], dtype=np.float64)
    y_reward = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64).reshape(-1, 1)
    selected_coef = _fit_ridge(x_scaled, y_selected, ridge)
    global_coef = _fit_ridge(x_scaled, y_global, ridge)
    reward_coef = _fit_ridge(x_scaled, y_reward, ridge)
    return {
        "model_type": "one_step_ridge_transition_reward",
        "ridge": float(ridge),
        "n_training_transitions": int(len(rows)),
        "feature_dims": {
            "design": int(x.shape[1]),
            "selected_block": K_BLOCK_GENERIC,
            "global": K_GLOBAL_GENERIC,
        },
        "feature_standardization": {
            "enabled": True,
            "constant_columns": int(scaler["constant_columns"]),
            "max_abs_mean": round(float(np.max(np.abs(scaler["mean"]))), 6),
            "max_scale": round(float(np.max(scaler["scale"])), 6),
        },
        "design_mean": [round(float(value), 10) for value in scaler["mean"].tolist()],
        "design_scale": [round(float(value), 10) for value in scaler["scale"].tolist()],
        "selected_coef": _round_matrix(selected_coef),
        "global_coef": _round_matrix(global_coef),
        "reward_coef": [round(float(value), 10) for value in reward_coef.reshape(-1).tolist()],
    }


def select_model_based_action(
    obs: np.ndarray,
    n_blocks: int,
    model: dict[str, Any],
) -> int:
    scores = predict_action_rewards(obs, n_blocks, model)
    if not bool(np.isfinite(scores).any()):
        return 0
    return int(np.argmax(scores))


def predict_action_rewards(
    obs: np.ndarray,
    n_blocks: int,
    model: dict[str, Any],
) -> np.ndarray:
    block_features = _block_features_from_obs(obs, int(n_blocks)).astype(np.float64)
    global_features = _global_features_from_obs(obs, int(n_blocks)).astype(np.float64)
    mask = block_features[:, 0] > 0.0
    x = np.column_stack(
        [
            np.ones(int(n_blocks), dtype=np.float64),
            block_features,
            np.repeat(global_features.reshape(1, -1), int(n_blocks), axis=0),
        ]
    )
    mean = np.asarray(model["design_mean"], dtype=np.float64)
    scale = np.asarray(model["design_scale"], dtype=np.float64)
    x_scaled = (x - mean) / scale
    x_scaled[:, 0] = 1.0
    reward_coef = np.asarray(model["reward_coef"], dtype=np.float64).reshape(-1)
    scores = x_scaled @ reward_coef
    scores[~mask] = -np.inf
    return scores


def predict_one_step(
    model: dict[str, Any],
    selected_block_features: np.ndarray,
    global_features: np.ndarray,
) -> dict[str, Any]:
    x = np.concatenate(
        [
            np.asarray([1.0], dtype=np.float64),
            np.asarray(selected_block_features, dtype=np.float64),
            np.asarray(global_features, dtype=np.float64),
        ]
    ).reshape(1, -1)
    mean = np.asarray(model["design_mean"], dtype=np.float64)
    scale = np.asarray(model["design_scale"], dtype=np.float64)
    x_scaled = (x - mean) / scale
    x_scaled[:, 0] = 1.0
    selected = x_scaled @ np.asarray(model["selected_coef"], dtype=np.float64)
    global_pred = x_scaled @ np.asarray(model["global_coef"], dtype=np.float64)
    reward = x_scaled @ np.asarray(model["reward_coef"], dtype=np.float64).reshape(-1, 1)
    return {
        "next_selected_block_features": selected.reshape(-1),
        "next_global_features": global_pred.reshape(-1),
        "reward": float(reward.reshape(-1)[0]),
    }


def evaluate_model_based_policy(
    env: GenericCountyEnv,
    model: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=int(seed))
    done = False
    total_reward = 0.0
    total_predicted_reward = 0.0
    selected_blocks: list[int] = []
    last_info: dict[str, Any] = {}
    while not done:
        features = _block_features_from_obs(obs, env.n_blocks)
        mask = features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        action = select_model_based_action(obs, env.n_blocks, model)
        total_predicted_reward += float(predict_action_rewards(obs, env.n_blocks, model)[action])
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return {
        "policy": "one_step_model_based_reward",
        "seed": int(seed),
        "steps": int(last_info.get("step", 0)),
        "reward": round(float(total_reward), 6),
        "model_predicted_reward_sum": round(float(total_predicted_reward), 6),
        "slope_change_pct": float(last_info.get("slope_change_pct", 0.0)),
        "cont_change": float(last_info.get("cont_change", 0.0)),
        "baimu_count_change": int(last_info.get("baimu_count_change", 0)),
        "baimu_area_change_ha": float(last_info.get("baimu_area_change_ha", 0.0)),
        "completed_swaps": int(last_info.get("budget_used", 0)),
        "unique_blocks": len(set(selected_blocks)),
        "selected_blocks_head": selected_blocks[:20],
    }


def summarize_model_based_runs(
    runs: list[dict[str, Any]],
    baseline_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = summarize_runs(runs)
    comparisons: dict[str, Any] = {
        "higher_reward_is_better": True,
        "lower_slope_change_pct_is_better": True,
    }
    for policy, baseline in baseline_report.get("policy_summaries", {}).items():
        for field in (
            "reward_mean",
            "slope_change_pct_mean",
            "cont_change_mean",
            "baimu_area_change_ha_mean",
        ):
            if summary.get(field) is None or baseline.get(field) is None:
                continue
            comparisons[f"model_based_minus_{policy}_{field}"] = round(
                float(summary[field]) - float(baseline[field]),
                6,
            )
    return summary, comparisons


def run_experiment(
    env_factory: Callable[[], GenericCountyEnv],
    baseline_path: Path,
    collection_policies: list[str],
    train_seeds: list[int],
    eval_seeds: list[int],
    max_steps: int,
    ridge: float,
) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    rows = collect_transition_rows(
        env_factory=env_factory,
        policies=collection_policies,
        seeds=train_seeds,
        max_steps=max_steps,
    )
    model = fit_one_step_model(rows, ridge=ridge)
    eval_env = env_factory()
    runs = [evaluate_model_based_policy(eval_env, model, seed=int(seed)) for seed in eval_seeds]
    summary, comparisons = summarize_model_based_runs(runs, baseline_report)
    return {
        "description": (
            "Dongxing one-step learned transition/reward model-based action "
            "selection evaluated in the full real environment."
        ),
        "status": "supported_as_dongxing_full_one_step_model_based_policy",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "collection_policies": list(collection_policies),
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "max_steps": int(max_steps),
        "ridge": float(ridge),
        "n_training_transitions": int(len(rows)),
        "model": model,
        "model_based_policy": {"summary": summary, "runs": runs},
        "baseline_path": os.fspath(baseline_path),
        "comparisons": comparisons,
        "mbrl_transition_model_used": True,
        "planning_horizon": 1,
        "policy_transfer_tested": False,
        "claim_boundary": (
            "Local Dongxing one-step model-based action selection; this is not "
            "cross-county policy transfer and not multi-step MBRL planning."
        ),
    }


def _design_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    selected = np.asarray([row["selected_block_features"] for row in rows], dtype=np.float64)
    global_features = np.asarray([row["global_features"] for row in rows], dtype=np.float64)
    return np.column_stack([np.ones(len(rows)), selected, global_features])


def _fit_standardizer(x: np.ndarray) -> dict[str, Any]:
    mean = np.asarray(x, dtype=np.float64).mean(axis=0)
    scale = np.asarray(x, dtype=np.float64).std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.0
    return {
        "mean": mean,
        "scale": scale,
        "constant_columns": int(constant.sum()),
    }


def _apply_standardizer(x: np.ndarray, scaler: dict[str, Any]) -> np.ndarray:
    scaled = (np.asarray(x, dtype=np.float64) - scaler["mean"]) / scaler["scale"]
    scaled[:, 0] = 1.0
    return scaled


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _global_features_from_obs(obs: np.ndarray, n_blocks: int) -> np.ndarray:
    start = int(n_blocks) * K_BLOCK_GENERIC
    return np.asarray(obs[start : start + K_GLOBAL_GENERIC], dtype=np.float64)


def _round_matrix(values: np.ndarray) -> list[list[float]]:
    return [
        [round(float(value), 10) for value in row]
        for row in np.asarray(values, dtype=np.float64).tolist()
    ]


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
    parser.add_argument("--eval-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/full_rigor/dongxing_full_model_based_policy.json"),
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "n_training_transitions": report["n_training_transitions"],
                "n_eval": len(report["model_based_policy"]["runs"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
