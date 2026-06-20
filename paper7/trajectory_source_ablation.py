"""Trajectory-source ablation for Paper 7.

This module compares transition models trained on:
  - random-only trajectories
  - greedy-only trajectories
  - mixed random + greedy trajectories

The goal is to test whether learned transition-model fidelity depends on the
source composition of the recorded trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
DEFAULT_OUTPUT = PAPER7_DIR / "results" / "revision" / "trajectory_source_ablation.json"
DEFAULT_HORIZONS = [1, 5, 10, 25, 50, 100]
DEFAULT_HOLDOUT_SEED = 2

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def infer_policy_from_filename(path: Path) -> str:
    match = re.match(r"([A-Za-z0-9_-]+)_seed\d+$", path.stem)
    return match.group(1) if match else path.stem


def infer_seed_from_filename(path: Path) -> int | None:
    match = re.search(r"_seed(\d+)$", path.stem)
    return int(match.group(1)) if match else None


def source_name_from_policies(policies: Sequence[str] | None) -> str:
    """Return a stable source label from a policy list.

    None or an empty sequence maps to ``mixed``. Single-policy sources are
    labeled ``<policy>_only``. Multi-policy sources are sorted and joined.
    """

    if not policies:
        return "mixed"

    unique: list[str] = []
    seen: set[str] = set()
    for policy in policies:
        name = str(policy).strip()
        if not name or name in seen:
            continue
        unique.append(name)
        seen.add(name)
    if not unique:
        return "mixed"
    if len(unique) == 1:
        return f"{unique[0]}_only"
    return "_".join(unique)


class FileTrajectoryDataset:
    """Trajectory dataset loaded from an explicit list of NPZ files."""

    def __init__(self, file_paths: Sequence[Path], max_transitions_per_file: int | None = None):
        self.file_paths = [Path(path) for path in file_paths]
        if not self.file_paths:
            raise ValueError("No trajectory files were provided")

        block_features: list[np.ndarray] = []
        global_features: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[np.ndarray] = []
        next_block_features: list[np.ndarray] = []
        next_global_features: list[np.ndarray] = []
        self.policies: list[str] = []
        self.seeds: list[int | None] = []

        for path in self.file_paths:
            data = np.load(path, allow_pickle=False)
            n = len(data["actions"])
            if max_transitions_per_file is not None:
                n = min(n, int(max_transitions_per_file))

            block_features.append(data["block_features"][:n].astype(np.float32))
            global_features.append(data["global_features"][:n].astype(np.float32))
            actions.append(data["actions"][:n].astype(np.int64))
            rewards.append(data["rewards"][:n].astype(np.float32))
            next_block_features.append(data["next_block_features"][:n].astype(np.float32))
            next_global_features.append(data["next_global_features"][:n].astype(np.float32))
            self.policies.append(infer_policy_from_filename(path))
            self.seeds.append(infer_seed_from_filename(path))

            self.n_blocks = int(data["n_blocks"])
            self.k_block = int(data["k_block"])
            self.k_global = int(data["k_global"])

        self.block_features = np.concatenate(block_features, axis=0)
        self.global_features = np.concatenate(global_features, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_block_features = np.concatenate(next_block_features, axis=0)
        self.next_global_features = np.concatenate(next_global_features, axis=0)

    def __len__(self) -> int:
        return int(len(self.actions))

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        torch = _lazy_import_torch()
        return {
            "block_features": torch.tensor(self.block_features[idx]),
            "global_features": torch.tensor(self.global_features[idx]),
            "action": torch.tensor(self.actions[idx], dtype=torch.long),
            "reward": torch.tensor(self.rewards[idx], dtype=torch.float32),
            "next_block_features": torch.tensor(self.next_block_features[idx]),
            "next_global_features": torch.tensor(self.next_global_features[idx]),
        }


def _load_arrays_from_files(
    file_paths: Sequence[Path],
    max_transitions_per_file: int | None = None,
) -> dict[str, np.ndarray]:
    arrays: dict[str, list[np.ndarray]] = {
        "block_features": [],
        "global_features": [],
        "actions": [],
        "rewards": [],
        "next_block_features": [],
        "next_global_features": [],
    }
    for path in sorted(Path(p) for p in file_paths):
        data = np.load(path, allow_pickle=False)
        n = len(data["actions"])
        if max_transitions_per_file is not None:
            n = min(n, int(max_transitions_per_file))
        for key in arrays:
            arrays[key].append(data[key][:n])

    if not arrays["actions"]:
        raise ValueError("No trajectory files were provided")

    return {key: np.concatenate(values, axis=0) for key, values in arrays.items()}


def _lazy_import_torch() -> Any:
    import torch

    return torch


def _lazy_import_training_dependencies() -> tuple[Any, Any, Any]:
    from paper7.learned_env import TransitionModel
    from paper7.transition_rollout_diagnostics import choose_start_indices, rollout_model

    return TransitionModel, choose_start_indices, rollout_model


def _selected_files_for_source(
    trajectory_dir: Path,
    source_policies: Sequence[str] | None,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
) -> dict[str, list[Path]]:
    all_files = sorted(trajectory_dir.glob("*.npz"))
    wanted = None if not source_policies else {str(policy).strip() for policy in source_policies}
    eval_files = [
        path
        for path in all_files
        if infer_seed_from_filename(path) == holdout_seed
    ]

    train_files: list[Path] = []
    for path in all_files:
        policy = infer_policy_from_filename(path)
        seed = infer_seed_from_filename(path)
        if wanted is not None and policy not in wanted:
            continue
        if seed != holdout_seed:
            train_files.append(path)

    return {"train_files": train_files, "eval_files": eval_files}


def _train_transition_model_from_files(
    train_files: Sequence[Path],
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.1,
    seed: int = 42,
    max_transitions_per_file: int | None = None,
) -> tuple[Any, dict[str, list[float]], FileTrajectoryDataset]:
    torch = _lazy_import_torch()
    TransitionModel, _, _ = _lazy_import_training_dependencies()
    from torch.utils.data import DataLoader, random_split

    dataset = FileTrajectoryDataset(train_files, max_transitions_per_file=max_transitions_per_file)

    n = len(dataset)
    n_val = max(1, int(round(n * val_split))) if n > 1 else 0
    n_train = n - n_val

    generator = torch.Generator().manual_seed(int(seed))
    if n_val > 0:
        train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    else:
        train_ds = dataset
        val_ds = dataset

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    model = TransitionModel(
        n_blocks=dataset.n_blocks,
        k_block=dataset.k_block,
        k_global=dataset.k_global,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_reward_mse": [],
        "val_obs_cosine": [],
    }
    best_val_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for _epoch in range(int(epochs)):
        model.train()
        train_losses: list[float] = []
        for batch in train_loader:
            bf = batch["block_features"]
            gf = batch["global_features"]
            act = batch["action"]
            rew = batch["reward"]
            nbf = batch["next_block_features"]
            ngf = batch["next_global_features"]

            pred_nbf, pred_ngf, pred_rew = model(bf, gf, act)
            loss_block = torch.nn.functional.mse_loss(pred_nbf, nbf)
            loss_global = torch.nn.functional.mse_loss(pred_ngf, ngf)
            loss_reward = torch.nn.functional.mse_loss(pred_rew, rew)
            loss = loss_block + loss_global + 0.1 * loss_reward

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: list[float] = []
        val_rew_mse: list[float] = []
        val_cosines: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                bf = batch["block_features"]
                gf = batch["global_features"]
                act = batch["action"]
                rew = batch["reward"]
                nbf = batch["next_block_features"]
                ngf = batch["next_global_features"]

                pred_nbf, pred_ngf, pred_rew = model(bf, gf, act)
                loss_block = torch.nn.functional.mse_loss(pred_nbf, nbf)
                loss_global = torch.nn.functional.mse_loss(pred_ngf, ngf)
                loss_reward = torch.nn.functional.mse_loss(pred_rew, rew)
                loss = loss_block + loss_global + 0.1 * loss_reward
                val_losses.append(float(loss.item()))
                val_rew_mse.append(float(loss_reward.item()))

                pred_flat = torch.cat([pred_nbf.reshape(pred_nbf.size(0), -1), pred_ngf], dim=-1)
                true_flat = torch.cat([nbf.reshape(nbf.size(0), -1), ngf], dim=-1)
                val_cosines.append(float(torch.nn.functional.cosine_similarity(pred_flat, true_flat, dim=-1).mean().item()))

        train_loss = float(np.mean(train_losses)) if train_losses else math.nan
        val_loss = float(np.mean(val_losses)) if val_losses else math.nan
        val_rew = float(np.mean(val_rew_mse)) if val_rew_mse else math.nan
        val_cos = float(np.mean(val_cosines)) if val_cosines else math.nan

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_reward_mse"].append(val_rew)
        history["val_obs_cosine"].append(val_cos)

        if math.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, dataset


def _summarize_rollout_metrics(
    rollout_result: dict[str, Any],
    horizons: Sequence[int],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "horizons_requested": [int(horizon) for horizon in horizons],
    }
    for horizon in horizons:
        horizon_key = str(int(horizon))
        metrics = dict(rollout_result.get("horizons", {}).get(horizon_key, {}))
        summary[f"horizon_{horizon_key}_n_steps"] = int(metrics.get("n_steps", 0))
        for key in (
            "selected_block_mae",
            "all_block_mae",
            "global_mae",
            "reward_mae",
            "mask_agreement",
            "slope_contiguity_mae",
            "baimu_mae",
            "investment_spread_mae",
            "budget_progress_mae",
        ):
            if key in metrics:
                summary[f"horizon_{horizon_key}_{key}"] = float(metrics[key])
    return summary


def evaluate_transition_model_on_files(
    model: Any,
    file_paths: Sequence[Path],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_starts: int = 80,
    max_transitions_per_file: int | None = None,
) -> dict[str, Any]:
    _, choose_start_indices, rollout_model = _lazy_import_training_dependencies()
    arrays = _load_arrays_from_files(file_paths, max_transitions_per_file=max_transitions_per_file)
    start_indices = choose_start_indices(len(arrays["actions"]), list(horizons), int(n_starts))
    rollout_result = rollout_model(
        model=model,
        block_features=arrays["block_features"].astype(np.float32),
        global_features=arrays["global_features"].astype(np.float32),
        actions=arrays["actions"].astype(np.int64),
        rewards=arrays["rewards"].astype(np.float32),
        next_block_features=arrays["next_block_features"].astype(np.float32),
        next_global_features=arrays["next_global_features"].astype(np.float32),
        horizons=[int(horizon) for horizon in horizons],
        start_indices=start_indices,
    )
    summary = _summarize_rollout_metrics(rollout_result, horizons)
    summary["n_transitions_loaded"] = int(len(arrays["actions"]))
    summary["n_starts"] = int(len(start_indices))
    summary["file_paths"] = [display_path(Path(path)) for path in file_paths]
    return summary


def _pick_metric(evaluation: dict[str, Any], horizon: int, metric: str) -> float:
    return float(evaluation.get(f"horizon_{int(horizon)}_{metric}", math.nan))


def compare_source_reports(reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Compare source reports using lower-is-better rollout error metrics."""

    comparisons: dict[str, Any] = {}
    if not reports:
        return comparisons

    for scope in ("all", "random", "greedy"):
        for metric in ("reward_mae", "global_mae"):
            horizon_metric = f"horizon_100_{metric}"
            scored = []
            for report in reports:
                evaluation = report.get("evaluation", {}).get(scope, {})
                scored.append((float(evaluation.get(horizon_metric, math.inf)), report.get("source")))
            best_source = min(scored, key=lambda item: item[0])[1]
            comparisons[f"best_source_by_{scope}_{horizon_metric}"] = best_source

    source_lookup = {str(report.get("source")): report for report in reports}
    mixed = source_lookup.get("mixed")
    random_only = source_lookup.get("random_only")
    greedy_only = source_lookup.get("greedy_only")

    if mixed is not None and random_only is not None:
        for scope in ("all", "random", "greedy"):
            for metric in ("reward_mae", "global_mae"):
                horizon_metric = f"horizon_100_{metric}"
                mixed_value = float(
                    mixed.get("evaluation", {}).get(scope, {}).get(horizon_metric, math.nan)
                )
                random_value = float(
                    random_only.get("evaluation", {}).get(scope, {}).get(horizon_metric, math.nan)
                )
                comparisons[f"mixed_minus_random_{scope}_{horizon_metric}"] = round(
                    mixed_value - random_value, 6
                )

    if mixed is not None and greedy_only is not None:
        for scope in ("all", "random", "greedy"):
            for metric in ("reward_mae", "global_mae"):
                horizon_metric = f"horizon_100_{metric}"
                mixed_value = float(
                    mixed.get("evaluation", {}).get(scope, {}).get(horizon_metric, math.nan)
                )
                greedy_value = float(
                    greedy_only.get("evaluation", {}).get(scope, {}).get(horizon_metric, math.nan)
                )
                comparisons[f"mixed_minus_greedy_{scope}_{horizon_metric}"] = round(
                    mixed_value - greedy_value, 6
                )

    return comparisons


def _build_source_report(
    source: str,
    source_policies: Sequence[str] | None,
    trajectory_dir: Path,
    epochs: int,
    lr: float,
    batch_size: int,
    val_split: float,
    horizons: Sequence[int],
    n_starts: int,
    holdout_seed: int,
    max_transitions_per_file: int | None,
) -> dict[str, Any]:
    split = _selected_files_for_source(
        trajectory_dir=trajectory_dir,
        source_policies=source_policies,
        holdout_seed=holdout_seed,
    )
    if not split["train_files"] or not split["eval_files"]:
        return {
            "source": source,
            "status": "missing",
            "train_files": [display_path(path) for path in split["train_files"]],
            "eval_files": [display_path(path) for path in split["eval_files"]],
        }

    model, history, dataset = _train_transition_model_from_files(
        train_files=split["train_files"],
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        val_split=val_split,
        seed=42,
        max_transitions_per_file=max_transitions_per_file,
    )

    evaluation_all = evaluate_transition_model_on_files(
        model=model,
        file_paths=split["eval_files"],
        horizons=horizons,
        n_starts=n_starts,
        max_transitions_per_file=max_transitions_per_file,
    )

    eval_by_policy: dict[str, dict[str, Any]] = {}
    for policy in sorted({infer_policy_from_filename(path) for path in split["eval_files"]}):
        policy_files = [path for path in split["eval_files"] if infer_policy_from_filename(path) == policy]
        eval_by_policy[policy] = evaluate_transition_model_on_files(
            model=model,
            file_paths=policy_files,
            horizons=horizons,
            n_starts=min(n_starts, 80),
            max_transitions_per_file=max_transitions_per_file,
        )

    if "random" not in eval_by_policy:
        eval_by_policy["random"] = dict(evaluation_all)
    if "greedy" not in eval_by_policy:
        eval_by_policy["greedy"] = dict(evaluation_all)

    def _scope_summary(metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            "horizon_100_reward_mae": float(metrics.get("horizon_100_reward_mae", math.nan)),
            "horizon_100_global_mae": float(metrics.get("horizon_100_global_mae", math.nan)),
        }

    return {
        "source": source,
        "status": "supported",
        "train_policies": list(source_policies) if source_policies else None,
        "train_files": [display_path(path) for path in split["train_files"]],
        "eval_files": [display_path(path) for path in split["eval_files"]],
        "n_train_files": len(split["train_files"]),
        "n_eval_files": len(split["eval_files"]),
        "n_train_transitions": int(len(dataset)),
        "training": {
            "epochs": int(epochs),
            "learning_rate": float(lr),
            "batch_size": int(batch_size),
            "val_split": float(val_split),
            "best_val_loss": float(min(history["val_loss"])) if history["val_loss"] else math.nan,
            "final_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else math.nan,
            "final_val_reward_mse": float(history["val_reward_mse"][-1]) if history["val_reward_mse"] else math.nan,
            "final_val_obs_cosine": float(history["val_obs_cosine"][-1]) if history["val_obs_cosine"] else math.nan,
        },
        "evaluation": {
            "all": _scope_summary(evaluation_all),
            "random": _scope_summary(eval_by_policy.get("random", evaluation_all)),
            "greedy": _scope_summary(eval_by_policy.get("greedy", evaluation_all)),
        },
    }


def build_source_ablation_report(
    trajectory_dir: Path = PAPER7_DIR / "trajectories",
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.1,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    n_starts: int = 80,
    holdout_seed: int = DEFAULT_HOLDOUT_SEED,
    max_transitions_per_file: int | None = None,
) -> dict[str, Any]:
    source_specs: list[tuple[str, Sequence[str] | None]] = [
        ("random_only", ["random"]),
        ("greedy_only", ["greedy"]),
        ("mixed", None),
    ]
    source_reports = [
        _build_source_report(
            source=source_name,
            source_policies=policies,
            trajectory_dir=trajectory_dir,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            val_split=val_split,
            horizons=horizons,
            n_starts=n_starts,
            holdout_seed=holdout_seed,
            max_transitions_per_file=max_transitions_per_file,
        )
        for source_name, policies in source_specs
    ]

    comparison = compare_source_reports(source_reports)
    status = (
        "supported_as_trajectory_source_ablation"
        if all(report.get("status") == "supported" for report in source_reports)
        else "partial_or_missing"
    )
    return {
        "status": status,
        "description": (
            "Trajectory-source ablation for the learned transition model. "
            "Random-only, greedy-only, and mixed trajectory sources are "
            "trained separately and evaluated on held-out trajectories."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "trajectory_dir": display_path(trajectory_dir),
        "holdout_seed": int(holdout_seed),
        "horizons": [int(horizon) for horizon in horizons],
        "n_starts": int(n_starts),
        "source_reports": source_reports,
        "comparison": comparison,
        "interpretation": (
            "Source composition is treated as a robustness factor for the learned "
            "transition model; held-out rollout errors are compared across random, "
            "greedy, and mixed training data sources."
        ),
    }


def write_source_ablation_report(
    output: Path = DEFAULT_OUTPUT,
    **kwargs: Any,
) -> dict[str, Any]:
    report = build_source_ablation_report(**kwargs)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=PAPER7_DIR / "trajectories")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--horizons", default="1,5,10,25,50,100")
    parser.add_argument("--n-starts", type=int, default=80)
    parser.add_argument("--holdout-seed", type=int, default=DEFAULT_HOLDOUT_SEED)
    parser.add_argument("--max-transitions-per-file", type=int, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    horizons = [int(item) for item in str(args.horizons).split(",") if item.strip()]
    report = write_source_ablation_report(
        output=args.output,
        trajectory_dir=args.trajectory_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        val_split=args.val_split,
        horizons=horizons,
        n_starts=args.n_starts,
        holdout_seed=args.holdout_seed,
        max_transitions_per_file=args.max_transitions_per_file,
    )
    print(json.dumps({
        "status": report["status"],
        "output": display_path(args.output),
        "best_source_by_all_horizon_100_reward_mae": report["comparison"].get(
            "best_source_by_all_horizon_100_reward_mae"
        ),
        "best_source_by_all_horizon_100_global_mae": report["comparison"].get(
            "best_source_by_all_horizon_100_global_mae"
        ),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
