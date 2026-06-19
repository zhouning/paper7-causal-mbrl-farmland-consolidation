"""Dongxing full-environment transition diagnostics for Paper 7."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper7.dongxing_full_baselines import (
    POLICIES,
    _block_features_from_obs,
    choose_full_env_action,
)
from paper7.dongxing_full_env import build_dongxing_full_env
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC, K_GLOBAL_GENERIC


TRANSITION_POLICIES = (
    "random",
    "dynamic_slope_gap",
    "area_weighted_slope_gap",
    "contiguity_aware",
    "baimu_aware",
    "scalarized_default",
)


def collect_transition_rows(
    env_factory: Callable[[], GenericCountyEnv],
    policies: list[str],
    seeds: list[int],
    max_steps: int = 100,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    env = env_factory()
    for policy in policies:
        if policy not in POLICIES:
            raise ValueError(f"Unsupported collection policy {policy!r}")
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            obs, _ = env.reset(seed=int(seed))
            for step in range(int(max_steps)):
                block_features = _block_features_from_obs(obs, env.n_blocks)
                global_features = _global_features_from_obs(obs, env.n_blocks)
                mask = block_features[:, 0] > 0.0
                if not bool(mask.any()):
                    break
                action = choose_full_env_action(policy, block_features, mask, rng)
                selected = block_features[int(action)].copy()
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_block_features = _block_features_from_obs(next_obs, env.n_blocks)
                next_global_features = _global_features_from_obs(next_obs, env.n_blocks)
                rows.append(
                    {
                        "policy": policy,
                        "seed": int(seed),
                        "step": int(step + 1),
                        "action": int(action),
                        "selected_block_features": selected,
                        "global_features": global_features.copy(),
                        "next_selected_block_features": next_block_features[int(action)].copy(),
                        "next_global_features": next_global_features.copy(),
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "slope_change_pct": float(info.get("slope_change_pct", 0.0)),
                        "cont_change": float(info.get("cont_change", 0.0)),
                        "baimu_area_change_ha": float(info.get("baimu_area_change_ha", 0.0)),
                    }
                )
                obs = next_obs
                if terminated or truncated:
                    break
    return rows


def train_ridge_transition(
    rows: list[dict[str, Any]],
    train_fraction: float = 0.8,
    ridge: float = 1e-3,
    split_seed: int = 42,
) -> dict[str, Any]:
    if len(rows) < 2:
        raise ValueError("At least two transition rows are required")
    x = _design_matrix(rows)
    y_selected = np.asarray([row["next_selected_block_features"] for row in rows], dtype=np.float64)
    y_global = np.asarray([row["next_global_features"] for row in rows], dtype=np.float64)
    y_reward = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64).reshape(-1, 1)
    n_train = max(1, min(len(rows) - 1, int(round(len(rows) * float(train_fraction)))))
    indices = np.random.default_rng(int(split_seed)).permutation(len(rows))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    x_scaled, scaler = _standardize_design(x, train_idx)

    selected_model = _fit_ridge(x_scaled[train_idx], y_selected[train_idx], ridge)
    global_model = _fit_ridge(x_scaled[train_idx], y_global[train_idx], ridge)
    reward_model = _fit_ridge(x_scaled[train_idx], y_reward[train_idx], ridge)

    pred_selected = _predict(x_scaled[val_idx], selected_model)
    pred_global = _predict(x_scaled[val_idx], global_model)
    pred_reward = _predict(x_scaled[val_idx], reward_model).reshape(-1)
    true_reward = y_reward[val_idx].reshape(-1)

    current_selected = np.asarray(
        [rows[int(i)]["selected_block_features"] for i in val_idx],
        dtype=np.float64,
    )
    current_global = np.asarray(
        [rows[int(i)]["global_features"] for i in val_idx],
        dtype=np.float64,
    )
    reward_mean = float(np.mean(y_reward[train_idx]))
    return {
        "model_type": "ridge_selected_block_global_reward",
        "ridge": float(ridge),
        "split": "seeded_random_transition_split",
        "split_seed": int(split_seed),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "feature_standardization": scaler,
        "selected_feature_mae": _mae(pred_selected, y_selected[val_idx]),
        "selected_feature_persistence_mae": _mae(current_selected, y_selected[val_idx]),
        "global_feature_mae": _mae(pred_global, y_global[val_idx]),
        "global_feature_persistence_mae": _mae(current_global, y_global[val_idx]),
        "reward_mae": _mae(pred_reward, true_reward),
        "reward_persistence_mae": _mae(np.full_like(true_reward, reward_mean), true_reward),
    }


def train_policy_holdout_diagnostics(
    rows: list[dict[str, Any]],
    holdout_policy: str,
    ridge: float = 1e-3,
) -> dict[str, Any]:
    train_rows = [row for row in rows if str(row["policy"]) != holdout_policy]
    val_rows = [row for row in rows if str(row["policy"]) == holdout_policy]
    if not train_rows or not val_rows:
        raise ValueError(f"Holdout policy {holdout_policy!r} requires train and validation rows")
    ordered = train_rows + val_rows
    x = _design_matrix(ordered)
    y_selected = np.asarray([row["next_selected_block_features"] for row in ordered], dtype=np.float64)
    y_global = np.asarray([row["next_global_features"] for row in ordered], dtype=np.float64)
    y_reward = np.asarray([float(row["reward"]) for row in ordered], dtype=np.float64).reshape(-1, 1)
    train_idx = np.arange(len(train_rows))
    val_idx = np.arange(len(train_rows), len(ordered))
    x_scaled, scaler = _standardize_design(x, train_idx)

    selected_model = _fit_ridge(x_scaled[train_idx], y_selected[train_idx], ridge)
    global_model = _fit_ridge(x_scaled[train_idx], y_global[train_idx], ridge)
    reward_model = _fit_ridge(x_scaled[train_idx], y_reward[train_idx], ridge)

    pred_selected = _predict(x_scaled[val_idx], selected_model)
    pred_global = _predict(x_scaled[val_idx], global_model)
    pred_reward = _predict(x_scaled[val_idx], reward_model).reshape(-1)
    true_reward = y_reward[val_idx].reshape(-1)
    current_selected = np.asarray(
        [ordered[int(i)]["selected_block_features"] for i in val_idx],
        dtype=np.float64,
    )
    current_global = np.asarray(
        [ordered[int(i)]["global_features"] for i in val_idx],
        dtype=np.float64,
    )
    reward_mean = float(np.mean(y_reward[train_idx]))
    return {
        "model_type": "ridge_selected_block_global_reward",
        "ridge": float(ridge),
        "split": "policy_holdout",
        "holdout_policy": str(holdout_policy),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "feature_standardization": scaler,
        "selected_feature_mae": _mae(pred_selected, y_selected[val_idx]),
        "selected_feature_persistence_mae": _mae(current_selected, y_selected[val_idx]),
        "global_feature_mae": _mae(pred_global, y_global[val_idx]),
        "global_feature_persistence_mae": _mae(current_global, y_global[val_idx]),
        "reward_mae": _mae(pred_reward, true_reward),
        "reward_persistence_mae": _mae(np.full_like(true_reward, reward_mean), true_reward),
    }


def summarize_transition_diagnostics(
    rows: list[dict[str, Any]],
    model_result: dict[str, Any],
    holdout_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    policies = sorted({str(row["policy"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    holdouts = holdout_results or []
    return {
        "description": (
            "Dongxing full real-environment transition diagnostic over selected "
            "block, global features, and scalar reward. This tests transition "
            "learnability only; no learned policy is trained in this artifact."
        ),
        "status": "supported_as_dongxing_full_transition_diagnostic",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_transitions": int(len(rows)),
        "policies": policies,
        "seeds": seeds,
        "feature_dims": {
            "selected_block": K_BLOCK_GENERIC,
            "global": K_GLOBAL_GENERIC,
        },
        "model": model_result,
        "policy_holdout_diagnostics": holdouts,
        "policy_holdout_count": len(holdouts),
        "mbrl_policy_trained": False,
        "policy_transfer_tested": False,
        "claim_boundary": (
            "Local Dongxing transition learnability diagnostic; not a policy "
            "optimization result and not cross-county transfer evidence."
        ),
    }


def _global_features_from_obs(obs: np.ndarray, n_blocks: int) -> np.ndarray:
    start = int(n_blocks) * K_BLOCK_GENERIC
    return np.asarray(obs[start : start + K_GLOBAL_GENERIC], dtype=np.float32)


def _design_matrix(rows: list[dict[str, Any]]) -> np.ndarray:
    selected = np.asarray([row["selected_block_features"] for row in rows], dtype=np.float64)
    global_features = np.asarray([row["global_features"] for row in rows], dtype=np.float64)
    return np.column_stack([np.ones(len(rows)), selected, global_features])


def _standardize_design(x: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    train = np.asarray(x[train_idx], dtype=np.float64)
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    constant = scale < 1e-12
    scale[constant] = 1.0
    scaled = (np.asarray(x, dtype=np.float64) - mean) / scale
    scaled[:, 0] = 1.0
    return scaled, {
        "enabled": True,
        "constant_columns": int(constant.sum()),
        "max_abs_mean": round(float(np.max(np.abs(mean))), 6),
        "max_scale": round(float(np.max(scale)), 6),
    }


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def _predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    return x @ coef


def _mae(pred: np.ndarray, true: np.ndarray) -> float:
    return round(float(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)).mean()), 6)


def _jsonify_rows_head(rows: list[dict[str, Any]], limit: int = 20) -> list[dict[str, Any]]:
    compact = []
    for row in rows[: int(limit)]:
        compact.append(
            {
                "policy": row["policy"],
                "seed": int(row["seed"]),
                "step": int(row["step"]),
                "action": int(row["action"]),
                "reward": round(float(row["reward"]), 6),
                "slope_change_pct": round(float(row["slope_change_pct"]), 6),
            }
        )
    return compact


def _parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(item) for item in _parse_csv_list(raw)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--policies", default=",".join(TRANSITION_POLICIES))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--holdout-policies", default="")
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/full_rigor/dongxing_transition_diagnostics.json"))
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

    rows = collect_transition_rows(
        env_factory=env_factory,
        policies=_parse_csv_list(args.policies),
        seeds=_parse_int_list(args.seeds),
        max_steps=args.max_steps,
    )
    model_result = train_ridge_transition(
        rows,
        train_fraction=args.train_fraction,
        ridge=args.ridge,
        split_seed=args.split_seed,
    )
    holdout_results = [
        train_policy_holdout_diagnostics(rows, holdout_policy=policy, ridge=args.ridge)
        for policy in _parse_csv_list(args.holdout_policies)
    ]
    report = summarize_transition_diagnostics(rows, model_result, holdout_results)
    report["command_config"] = {
        "dltb": os.fspath(args.dltb),
        "block_dir": os.fspath(args.block_dir),
        "max_steps": int(args.max_steps),
        "train_fraction": float(args.train_fraction),
        "ridge": float(args.ridge),
        "split_seed": int(args.split_seed),
        "holdout_policies": _parse_csv_list(args.holdout_policies),
        "total_budget": int(args.total_budget),
        "swaps_per_step": int(args.swaps_per_step),
    }
    report["transitions_head"] = _jsonify_rows_head(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "n_transitions": len(rows), "model": model_result}, indent=2))


if __name__ == "__main__":
    main()
