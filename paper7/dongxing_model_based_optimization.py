"""Scoring-grid optimization for Dongxing one-step model-based policies."""

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
    predict_action_rewards,
    summarize_model_based_runs,
)
from paper7.dongxing_full_transition_diagnostics import (
    TRANSITION_POLICIES,
    collect_transition_rows,
)
from paper7.generic_county_env import GenericCountyEnv


def make_candidate_grid() -> list[dict[str, Any]]:
    candidates = [
        ("reward_only", 1.0, 0.0, 0.0, 0.0, 0.0),
        ("reward_slope_bonus", 1.0, 0.75, 0.0, 0.0, 0.0),
        ("reward_slope_bonus_x2", 1.0, 1.5, 0.0, 0.0, 0.0),
        ("reward_current_farm_bonus", 1.0, 0.0, 0.75, 0.0, 0.0),
        ("reward_neighbor_bonus", 1.0, 0.0, 0.0, 0.75, 0.0),
        ("reward_slope_current", 1.0, 0.75, 0.75, 0.0, 0.0),
        ("reward_slope_neighbor", 1.0, 0.75, 0.0, 0.75, 0.0),
        ("reward_diversity_penalty", 1.0, 0.0, 0.0, 0.0, 0.10),
        ("reward_slope_diversity", 1.0, 0.75, 0.0, 0.0, 0.10),
        ("reward_slope_current_diversity", 1.0, 0.75, 0.75, 0.0, 0.10),
    ]
    return [
        {
            "name": name,
            "reward_weight": reward_weight,
            "slope_weight": slope_weight,
            "current_farm_weight": current_farm_weight,
            "neighbor_weight": neighbor_weight,
            "diversity_penalty": diversity_penalty,
        }
        for (
            name,
            reward_weight,
            slope_weight,
            current_farm_weight,
            neighbor_weight,
            diversity_penalty,
        ) in candidates
    ]


def score_actions(
    obs: np.ndarray,
    n_blocks: int,
    model: dict[str, Any],
    candidate: dict[str, Any],
    selected_counts: dict[int, int],
) -> np.ndarray:
    block_features = _block_features_from_obs(obs, int(n_blocks)).astype(np.float64)
    scores = float(candidate["reward_weight"]) * predict_action_rewards(obs, n_blocks, model)
    scores = scores + float(candidate.get("slope_weight", 0.0)) * block_features[:, 0]
    scores = scores + float(candidate.get("current_farm_weight", 0.0)) * block_features[:, 4]
    scores = scores + float(candidate.get("neighbor_weight", 0.0)) * block_features[:, 5]
    if float(candidate.get("diversity_penalty", 0.0)) > 0:
        counts = np.asarray(
            [selected_counts.get(int(action), 0) for action in range(int(n_blocks))],
            dtype=np.float64,
        )
        scores = scores - float(candidate["diversity_penalty"]) * counts
    scores[block_features[:, 0] <= 0.0] = -np.inf
    return scores


def evaluate_candidate_policy(
    env: GenericCountyEnv,
    model: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    obs, _ = env.reset(seed=int(seed))
    done = False
    total_reward = 0.0
    total_score = 0.0
    selected_blocks: list[int] = []
    selected_counts: dict[int, int] = {}
    last_info: dict[str, Any] = {}
    while not done:
        features = _block_features_from_obs(obs, env.n_blocks)
        if not bool((features[:, 0] > 0.0).any()):
            break
        scores = score_actions(obs, env.n_blocks, model, candidate, selected_counts)
        action = int(np.argmax(scores)) if bool(np.isfinite(scores).any()) else 0
        total_score += float(scores[action])
        selected_counts[action] = selected_counts.get(action, 0) + 1
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return {
        "policy": "optimized_one_step_model_based",
        "candidate": str(candidate["name"]),
        "seed": int(seed),
        "steps": int(last_info.get("step", 0)),
        "reward": round(float(total_reward), 6),
        "model_score_sum": round(float(total_score), 6),
        "slope_change_pct": float(last_info.get("slope_change_pct", 0.0)),
        "cont_change": float(last_info.get("cont_change", 0.0)),
        "baimu_count_change": int(last_info.get("baimu_count_change", 0)),
        "baimu_area_change_ha": float(last_info.get("baimu_area_change_ha", 0.0)),
        "completed_swaps": int(last_info.get("budget_used", 0)),
        "unique_blocks": len(set(selected_blocks)),
        "selected_blocks_head": selected_blocks[:20],
    }


def select_best_candidate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        raise ValueError("At least one candidate summary is required")
    return sorted(
        summaries,
        key=lambda row: (
            -float(row.get("reward_mean", -np.inf)),
            float(row.get("slope_change_pct_mean", np.inf)),
            str(row.get("candidate", "")),
        ),
    )[0]


def run_optimization_experiment(
    env_factory: Callable[[], GenericCountyEnv],
    baseline_path: Path,
    collection_policies: list[str],
    train_seeds: list[int],
    selection_seeds: list[int],
    eval_seeds: list[int],
    max_steps: int,
    ridge: float,
    candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    candidates = candidates or make_candidate_grid()
    rows = collect_transition_rows(
        env_factory=env_factory,
        policies=collection_policies,
        seeds=train_seeds,
        max_steps=max_steps,
    )
    model = fit_one_step_model(rows, ridge=ridge)

    selection_summaries = []
    selection_env = env_factory()
    for candidate in candidates:
        runs = [
            evaluate_candidate_policy(selection_env, model, candidate, seed=int(seed))
            for seed in selection_seeds
        ]
        summary = summarize_runs(runs)
        summary["candidate"] = str(candidate["name"])
        selection_summaries.append(summary)

    best_summary = select_best_candidate(selection_summaries)
    best_candidate = next(
        candidate for candidate in candidates if candidate["name"] == best_summary["candidate"]
    )
    eval_env = env_factory()
    eval_runs = [
        evaluate_candidate_policy(eval_env, model, best_candidate, seed=int(seed))
        for seed in eval_seeds
    ]
    eval_summary, comparisons = summarize_model_based_runs(eval_runs, baseline_report)
    return {
        "description": (
            "Dongxing one-step model-based scoring-grid optimization with "
            "separate selection and held-out evaluation seeds."
        ),
        "status": "supported_as_dongxing_model_based_scoring_optimization",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "collection_policies": list(collection_policies),
        "train_seeds": [int(seed) for seed in train_seeds],
        "selection_seeds": [int(seed) for seed in selection_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "max_steps": int(max_steps),
        "ridge": float(ridge),
        "n_training_transitions": int(len(rows)),
        "candidate_grid": candidates,
        "candidate_selection_summaries": selection_summaries,
        "best_candidate": best_candidate,
        "best_selection_summary": best_summary,
        "heldout_eval": {"summary": eval_summary, "runs": eval_runs},
        "baseline_path": os.fspath(baseline_path),
        "comparisons": comparisons,
        "mbrl_transition_model_used": True,
        "planning_horizon": 1,
        "selection_eval_split": True,
        "policy_transfer_tested": False,
        "claim_boundary": (
            "Local Dongxing one-step model-based scoring optimization with "
            "held-out evaluation; not cross-county transfer and not multi-step MBRL."
        ),
    }


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
    parser.add_argument("--selection-seeds", default="5,6,7,8,9")
    parser.add_argument("--eval-seeds", default="10,11,12,13,14")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/full_rigor/dongxing_model_based_optimization.json"),
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

    report = run_optimization_experiment(
        env_factory=env_factory,
        baseline_path=args.baseline,
        collection_policies=_parse_csv_list(args.collection_policies),
        train_seeds=_parse_int_list(args.train_seeds),
        selection_seeds=_parse_int_list(args.selection_seeds),
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
                "best_candidate": report["best_candidate"]["name"],
                "heldout_reward_mean": report["heldout_eval"]["summary"].get("reward_mean"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
