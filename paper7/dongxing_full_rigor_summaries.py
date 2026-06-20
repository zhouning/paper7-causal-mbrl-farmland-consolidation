"""Full-rigor Dongxing summary helpers for Paper 7."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper7.generic_county_env import K_BLOCK_GENERIC, K_GLOBAL_GENERIC


PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
FULL_RIGOR_DIR = PAPER7_DIR / "results" / "full_rigor"
REVISION_DIR = PAPER7_DIR / "results" / "revision"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_dongxing_trajectory_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": _display_path(path)}
    payload = _load_json(path)
    model = dict(payload.get("model", {}))
    n_policies = payload.get("n_policies")
    if n_policies is None:
        n_policies = len(payload.get("policies", []))
    n_seeds = payload.get("n_seeds")
    if n_seeds is None:
        n_seeds = len(payload.get("seeds", []))
    random_split_reward_mae = payload.get("random_split_reward_mae")
    if random_split_reward_mae is None:
        random_split_reward_mae = model.get("reward_mae")
    random_split_reward_persistence_mae = payload.get("random_split_reward_persistence_mae")
    if random_split_reward_persistence_mae is None:
        random_split_reward_persistence_mae = model.get("reward_persistence_mae")
    policy_holdout_count = payload.get("policy_holdout_count")
    if policy_holdout_count is None:
        policy_holdout_count = len(payload.get("policy_holdout_diagnostics", []))
    policy_holdout_reward_beats_baseline_count = payload.get(
        "policy_holdout_reward_beats_baseline_count"
    )
    if policy_holdout_reward_beats_baseline_count is None:
        policy_holdout_diagnostics = payload.get("policy_holdout_diagnostics", [])
        policy_holdout_reward_beats_baseline_count = sum(
            1
            for row in policy_holdout_diagnostics
            if float(row.get("reward_mae", float("inf")))
            < float(row.get("reward_persistence_mae", float("-inf")))
        )
    return {
        "status": payload.get("status", "supported_as_dongxing_trajectory_summary"),
        "path": _display_path(path),
        "n_transitions": payload.get("n_transitions"),
        "n_policies": n_policies,
        "n_seeds": n_seeds,
        "feature_dims": payload.get("feature_dims", {}),
        "random_split_reward_mae": random_split_reward_mae,
        "random_split_reward_persistence_mae": random_split_reward_persistence_mae,
        "policy_holdout_count": policy_holdout_count,
        "policy_holdout_reward_beats_baseline_count": policy_holdout_reward_beats_baseline_count,
        "mbrl_policy_trained": bool(payload.get("mbrl_policy_trained", False)),
        "policy_transfer_tested": bool(payload.get("policy_transfer_tested", False)),
        "interpretation": payload.get(
            "interpretation",
            "Dongxing full-environment trajectory and transition diagnostics; not transfer evidence.",
        ),
    }


def summarize_dongxing_mbrl_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": _display_path(path)}
    payload = _load_json(path)
    trajectory = payload.get("transition_diagnostics", {})
    full_policy = payload.get("full_model_based_policy", {})
    optimization = payload.get("model_based_optimization", {})
    return {
        "status": payload.get("status", "supported_as_local_dongxing_mbrl_results"),
        "path": _display_path(path),
        "transition_diagnostics": trajectory,
        "full_model_based_policy": full_policy,
        "model_based_optimization": optimization,
        "mbrl_transition_model_used": bool(payload.get("mbrl_transition_model_used", False)),
        "policy_transfer_tested": bool(payload.get("policy_transfer_tested", False)),
        "multi_step_mbrl_planning_tested": bool(
            payload.get("multi_step_mbrl_planning_tested", False)
        ),
        "interpretation": payload.get(
            "interpretation",
            "Local Dongxing one-step model-based policy and held-out scoring optimization; not transfer or multi-step planning.",
        ),
    }


def summarize_transfer_finetune_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": _display_path(path)}
    payload = _load_json(path)
    return {
        "status": payload.get(
            "status", "structurally_invalid_for_direct_policy_transfer"
        ),
        "path": _display_path(path),
        "bishan": payload.get("bishan", {}),
        "dongxing": payload.get("dongxing", {}),
        "dimension_mismatch": payload.get("dimension_mismatch", {}),
        "direct_policy_transfer_tested": bool(
            payload.get("direct_policy_transfer_tested", False)
        ),
        "fine_tuning_tested": bool(payload.get("fine_tuning_tested", False)),
        "fine_tuning_required": bool(payload.get("fine_tuning_required", False)),
        "interpretation": payload.get(
            "interpretation",
            "Bishan and Dongxing have incompatible observation/action dimensions for direct policy transfer.",
        ),
    }


def build_dongxing_trajectory_summary(
    transition_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    model = dict(transition_diagnostics.get("model", {}))
    holdouts = list(transition_diagnostics.get("policy_holdout_diagnostics", []))
    holdout_reward_wins = [
        row
        for row in holdouts
        if float(row.get("reward_mae", float("inf")))
        < float(row.get("reward_persistence_mae", float("-inf")))
    ]
    return {
        "status": "supported_as_dongxing_trajectory_summary",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_transitions": int(transition_diagnostics.get("n_transitions", 0)),
        "n_policies": len(transition_diagnostics.get("policies", [])),
        "n_seeds": len(transition_diagnostics.get("seeds", [])),
        "feature_dims": dict(transition_diagnostics.get("feature_dims", {})),
        "random_split_reward_mae": model.get("reward_mae"),
        "random_split_reward_persistence_mae": model.get("reward_persistence_mae"),
        "policy_holdout_count": len(holdouts) or int(
            transition_diagnostics.get("policy_holdout_count", 0)
        ),
        "policy_holdout_reward_beats_baseline_count": len(holdout_reward_wins)
        if holdout_reward_wins
        else int(transition_diagnostics.get("policy_holdout_reward_beats_baseline_count", 0)),
        "mbrl_policy_trained": bool(transition_diagnostics.get("mbrl_policy_trained", False)),
        "policy_transfer_tested": bool(
            transition_diagnostics.get("policy_transfer_tested", False)
        ),
        "interpretation": (
            "Dongxing full-environment transition diagnostics and trajectory summary; "
            "supports local learnability but not transfer claims."
        ),
    }


def build_dongxing_mbrl_results_summary(
    transition_diagnostics: dict[str, Any],
    full_model_based_policy: dict[str, Any],
    model_based_optimization: dict[str, Any],
) -> dict[str, Any]:
    trajectory_summary = build_dongxing_trajectory_summary(transition_diagnostics)
    policy_summary = dict(full_model_based_policy)
    optimization_summary = dict(model_based_optimization)
    return {
        "status": "supported_as_local_dongxing_mbrl_results",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "transition_diagnostics": trajectory_summary,
        "full_model_based_policy": policy_summary,
        "model_based_optimization": optimization_summary,
        "mbrl_transition_model_used": bool(
            policy_summary.get("mbrl_transition_model_used", False)
            or optimization_summary.get("mbrl_transition_model_used", False)
        ),
        "policy_transfer_tested": False,
        "multi_step_mbrl_planning_tested": False,
        "interpretation": (
            "Local Dongxing one-step model-based policy and held-out scoring "
            "optimization; not cross-county transfer and not multi-step MBRL."
        ),
    }


def build_transfer_finetune_summary(
    bishan_shape: dict[str, Any],
    dongxing_shape: dict[str, Any],
) -> dict[str, Any]:
    bishan = _normalize_shape(bishan_shape)
    dongxing = _normalize_shape(dongxing_shape)
    mismatch = {
        "bishan_observation_dim": bishan["observation_dim"],
        "dongxing_observation_dim": dongxing["observation_dim"],
        "observation_dim_match": bishan["observation_dim"] == dongxing["observation_dim"],
        "bishan_action_dim": bishan["action_dim"],
        "dongxing_action_dim": dongxing["action_dim"],
        "action_dim_match": bishan["action_dim"] == dongxing["action_dim"],
        "bishan_k_block": bishan["k_block"],
        "dongxing_k_block": dongxing["k_block"],
        "bishan_k_global": bishan["k_global"],
        "dongxing_k_global": dongxing["k_global"],
    }
    structurally_invalid = not (
        mismatch["observation_dim_match"] and mismatch["action_dim_match"]
    )
    return {
        "status": (
            "structurally_invalid_for_direct_policy_transfer"
            if structurally_invalid
            else "potentially_transferable_with_adapter"
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bishan": bishan,
        "dongxing": dongxing,
        "dimension_mismatch": mismatch,
        "direct_policy_transfer_tested": False,
        "fine_tuning_tested": False,
        "fine_tuning_required": structurally_invalid,
        "interpretation": (
            "Bishan and Dongxing use incompatible observation and action "
            "dimensions, so direct policy transfer is structurally invalid; "
            "only adapter-based representation or model transfer would be meaningful."
        ),
    }


def _normalize_shape(shape: dict[str, Any]) -> dict[str, int]:
    n_blocks = int(shape.get("n_blocks", 0))
    k_block = int(shape.get("k_block", 0))
    k_global = int(shape.get("k_global", 0))
    observation_dim = shape.get("observation_dim")
    if observation_dim is None:
        observation_dim = n_blocks * k_block + k_global
    action_dim = shape.get("action_dim")
    if action_dim is None:
        action_dim = n_blocks
    return {
        "n_blocks": n_blocks,
        "k_block": k_block,
        "k_global": k_global,
        "observation_dim": int(observation_dim),
        "action_dim": int(action_dim),
    }


def write_full_rigor_summaries(repo_root: Path = REPO_ROOT) -> dict[str, Path]:
    paper7_dir = repo_root / "paper7"
    full_rigor_dir = paper7_dir / "results" / "full_rigor"
    revision_dir = paper7_dir / "results" / "revision"

    transition_diagnostics = _load_json(full_rigor_dir / "dongxing_transition_diagnostics.json")
    full_model_based_policy = _load_json(full_rigor_dir / "dongxing_full_model_based_policy.json")
    model_based_optimization = _load_json(full_rigor_dir / "dongxing_model_based_optimization.json")
    full_env_smoke = _load_json(full_rigor_dir / "dongxing_full_env_smoke.json")

    trajectories_summary = build_dongxing_trajectory_summary(transition_diagnostics)
    mbrl_results = build_dongxing_mbrl_results_summary(
        transition_diagnostics,
        full_model_based_policy,
        model_based_optimization,
    )
    transfer_finetune = build_transfer_finetune_summary(
        {
            "n_blocks": 2600,
            "k_block": 17,
            "k_global": 12,
            "observation_dim": 2600 * 17 + 12,
            "action_dim": 2600,
        },
        {
            "n_blocks": int(full_env_smoke.get("n_blocks", 2978)),
            "k_block": K_BLOCK_GENERIC,
            "k_global": K_GLOBAL_GENERIC,
            "observation_dim": int(full_env_smoke.get("n_blocks", 2978)) * K_BLOCK_GENERIC
            + K_GLOBAL_GENERIC,
            "action_dim": int(full_env_smoke.get("n_blocks", 2978)),
        },
    )

    outputs = {
        "dongxing_trajectories_summary": full_rigor_dir / "dongxing_trajectories_summary.json",
        "dongxing_mbrl_results": full_rigor_dir / "dongxing_mbrl_results.json",
        "transfer_finetune_results": full_rigor_dir / "transfer_finetune_results.json",
    }
    _write_json(outputs["dongxing_trajectories_summary"], trajectories_summary)
    _write_json(outputs["dongxing_mbrl_results"], mbrl_results)
    _write_json(outputs["transfer_finetune_results"], transfer_finetune)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = write_full_rigor_summaries(args.repo_root)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
