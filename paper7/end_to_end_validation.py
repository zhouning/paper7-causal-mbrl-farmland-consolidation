"""
End-to-end evidence audit for the Paper 7 CEUS revision.

This script does not retrain expensive RL models. It validates the evidence chain
from data assets and recorded trajectories through learned-environment training,
real-environment evaluation, calibration diagnostics, strong baselines, and the
external Dongxing feasibility check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
DEFAULT_OUT = PAPER7_DIR / "results" / "revision" / "end_to_end_validation.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.dongxing_full_rigor_summaries import (
    summarize_dongxing_mbrl_results,
    summarize_dongxing_trajectory_summary,
    summarize_transfer_finetune_results,
)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def display_path(path: Path) -> str:
    """Return a stable path string for repo files and temporary test files."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def describe_file(path: Path, hash_files: bool = False) -> dict[str, Any]:
    exists = path.exists()
    item: dict[str, Any] = {
        "path": display_path(path) if path.is_absolute() else str(path),
        "exists": exists,
    }
    if exists and path.is_file():
        stat = path.stat()
        item["bytes"] = stat.st_size
        item["modified_utc"] = datetime.fromtimestamp(
            stat.st_mtime, tz=timezone.utc
        ).isoformat()
        if hash_files:
            item["sha256"] = file_sha256(path)
    return item


def _extract_slope(payload: Any) -> float:
    if isinstance(payload, list):
        values = [float(row["slope_change_pct"]) for row in payload]
        return mean(values)
    if isinstance(payload, dict):
        return float(payload["slope_change_pct"])
    raise TypeError(f"Unsupported evaluation payload type: {type(payload)!r}")


def summarize_seed_evaluations(seed_dir: Path) -> dict[str, Any]:
    """Summarize paired calibrated and uncalibrated real-environment evaluations."""
    no_cal: dict[int, float] = {}
    with_cal: dict[int, float] = {}
    for path in sorted(seed_dir.glob("no_cal_eval_seed*.json")):
        seed = int(path.stem.replace("no_cal_eval_seed", ""))
        no_cal[seed] = _extract_slope(load_json(path))
    for path in sorted(seed_dir.glob("with_cal_eval_seed*.json")):
        seed = int(path.stem.replace("with_cal_eval_seed", ""))
        with_cal[seed] = _extract_slope(load_json(path))

    paired = sorted(set(no_cal).intersection(with_cal))
    no_values = [no_cal[seed] for seed in paired]
    with_values = [with_cal[seed] for seed in paired]
    no_mean = mean(no_values) if no_values else math.nan
    with_mean = mean(with_values) if with_values else math.nan
    improvement_pct = (
        (abs(with_mean) - abs(no_mean)) / abs(no_mean) * 100.0
        if no_values and no_mean != 0
        else math.nan
    )

    return {
        "status": "supported" if paired else "missing",
        "seed_dir": display_path(seed_dir),
        "n_no_cal_files": len(no_cal),
        "n_with_cal_files": len(with_cal),
        "n_paired_seeds": len(paired),
        "paired_seeds": paired,
        "balanced_pairs": len(no_cal) == len(with_cal) == len(paired),
        "no_cal_mean": round(no_mean, 6) if no_values else None,
        "no_cal_std": round(pstdev(no_values), 6) if len(no_values) > 1 else 0.0,
        "with_cal_mean": round(with_mean, 6) if with_values else None,
        "with_cal_std": round(pstdev(with_values), 6)
        if len(with_values) > 1
        else 0.0,
        "improvement_pct": round(improvement_pct, 6) if no_values else None,
    }


def summarize_alpha_grid(
    grid_path: Path, pre_specified_alpha: float = 0.185
) -> dict[str, Any]:
    """Summarize reward-scale grid results and distance from pre-specified alpha."""
    rows = load_json(grid_path)
    by_alpha: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        by_alpha[float(row["reward_scale"])].append(float(row["slope_change_pct"]))

    means = {
        alpha: mean(values)
        for alpha, values in sorted(by_alpha.items(), key=lambda item: item[0])
    }
    best_alpha = min(means, key=lambda alpha: means[alpha])
    pre_alpha = min(means, key=lambda alpha: abs(alpha - pre_specified_alpha))
    best_slope = means[best_alpha]
    pre_slope = means[pre_alpha]
    relative_gap_pct = (
        abs(abs(best_slope) - abs(pre_slope)) / abs(best_slope) * 100.0
        if best_slope != 0
        else math.nan
    )

    return {
        "status": "supported",
        "grid_path": display_path(grid_path),
        "n_runs": len(rows),
        "n_alphas": len(means),
        "best_alpha": round(best_alpha, 6),
        "best_slope_mean": round(best_slope, 6),
        "pre_specified_alpha": round(pre_specified_alpha, 6),
        "nearest_grid_alpha_to_pre_specified": round(pre_alpha, 6),
        "pre_specified_slope_mean": round(pre_slope, 6),
        "relative_gap_pct": round(relative_gap_pct, 6),
        "alpha_means": {f"{alpha:.3f}": round(value, 6) for alpha, value in means.items()},
    }


def summarize_transition_training(history_path: Path, model_path: Path) -> dict[str, Any]:
    history = load_json(history_path)
    val_loss = [float(x) for x in history.get("val_loss", [])]
    val_cos = [float(x) for x in history.get("val_obs_cosine", [])]
    val_reward = [float(x) for x in history.get("val_reward_mse", [])]
    return {
        "status": "supported" if val_loss and model_path.exists() else "missing",
        "model": describe_file(model_path),
        "history": describe_file(history_path),
        "epochs_recorded": len(val_loss),
        "best_val_loss": round(min(val_loss), 6) if val_loss else None,
        "final_val_loss": round(val_loss[-1], 6) if val_loss else None,
        "final_val_obs_cosine": round(val_cos[-1], 6) if val_cos else None,
        "final_val_reward_mse": round(val_reward[-1], 6) if val_reward else None,
    }


def summarize_rollout_diagnostics(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    h100 = payload["horizons"]["100"]
    h1 = payload["horizons"]["1"]
    summary = {
        "status": "supported",
        "path": display_path(path),
        "n_transitions_loaded": payload.get("n_transitions_loaded"),
        "n_starts": payload.get("n_starts"),
        "horizon_1_reward_mae": h1["reward_mae"],
        "horizon_100_reward_mae": h100["reward_mae"],
        "horizon_100_global_mae": h100["global_mae"],
        "horizon_100_mask_agreement": h100["mask_agreement"],
        "interpretation": (
            "supports learned-environment training as an approximate surrogate; "
            "final policy outcomes still require real-environment evaluation"
        ),
    }
    for key in (
        "slope_contiguity_mae",
        "baimu_mae",
        "investment_spread_mae",
        "budget_progress_mae",
    ):
        if key in h100:
            summary[f"horizon_100_{key}"] = h100[key]
    return summary


def summarize_policy_induced_diagnostics(path: Path) -> dict[str, Any]:
    """Summarize learned-vs-real diagnostics under trained policy actions."""
    payload = load_json(path)
    aggregate = payload["aggregate"]
    summary = {
        "status": "supported",
        "path": display_path(path),
        "support_size": int(payload.get("support_size", 0)),
        "n_policy_episodes": int(aggregate["n_episodes"]),
        "policy_induced_selected_block_mae_mean": aggregate[
            "selected_block_mae_mean_mean"
        ],
        "policy_induced_all_block_mae_mean": aggregate["all_block_mae_mean_mean"],
        "policy_induced_global_mae_mean": aggregate["global_mae_mean_mean"],
        "policy_induced_reward_mae_mean": aggregate["reward_mae_mean_mean"],
        "policy_induced_calibrated_reward_mae_mean": aggregate[
            "calibrated_reward_mae_mean_mean"
        ],
        "policy_induced_mask_agreement_mean": aggregate["mask_agreement_mean_mean"],
        "policy_induced_support_distance_mean": aggregate[
            "support_distance_mean_mean"
        ],
        "policy_induced_support_distance_q95": aggregate["support_distance_q95_mean"],
        "policy_induced_final_real_slope_pct_mean": aggregate[
            "final_real_slope_change_pct_mean"
        ],
        "interpretation": (
            "trained-policy actions induce states close to the recorded trajectory "
            "support and retain high action-mask agreement; final outcomes are "
            "still measured in the real parcel-simulation environment"
        ),
    }
    validation = payload.get("validation")
    if validation:
        summary["validation_passes_all_thresholds"] = bool(
            validation.get("passes_all_thresholds", False)
        )
        summary["validation_passes_mask_agreement_threshold"] = bool(
            validation.get("passes_mask_agreement_threshold", False)
        )
        summary["validation_passes_support_distance_threshold"] = bool(
            validation.get("passes_support_distance_threshold", False)
        )
        summary["validation_passes_reward_calibration_check"] = bool(
            validation.get("passes_reward_calibration_check", False)
        )
    return summary


def select_policy_induced_diagnostics_path(paper7_dir: Path) -> Path:
    """Prefer the expanded 15-seed diagnostic when available."""
    revision_dir = paper7_dir / "results" / "revision"
    expanded = revision_dir / "policy_induced_diagnostics_15seed.json"
    legacy = revision_dir / "policy_induced_diagnostics.json"
    return expanded if expanded.exists() else legacy


def summarize_calibration(path: Path, sensitivity_path: Path) -> dict[str, Any]:
    calibration = load_json(path)
    sensitivity = load_json(sensitivity_path)
    mixed = next(scope for scope in sensitivity["scopes"] if scope["scope"] == "mixed")
    median = next(
        item for item in mixed["thresholds"] if float(item["treatment_percentile"]) == 50.0
    )
    return {
        "status": "supported_as_observational_regularization",
        "calibration_path": display_path(path),
        "sensitivity_path": display_path(sensitivity_path),
        "empirical_att": round(float(calibration["empirical_att"]), 6),
        "predicted_att": round(float(calibration["predicted_att"]), 6),
        "calibration_factor": round(float(calibration["calibration_factor"]), 6),
        "mixed_median_common_support": median["overlap"]["common_support_share"],
        "mixed_median_trimmed_att": median["att_trimmed"]["att"],
        "mixed_median_ci_low": median["att_trimmed"]["ci_low"],
        "mixed_median_ci_high": median["att_trimmed"]["ci_high"],
        "interpretation": (
            "observational treatment-effect-informed reward regularization; "
            "not definitive causal identification"
        ),
    }


def summarize_bishan_baselines(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    by_policy = {row["policy"]: row for row in payload["summary"]}
    return {
        "status": "supported",
        "path": display_path(path),
        "budget": payload.get("budget"),
        "random_slope_mean": by_policy["random"]["slope_change_pct_mean"],
        "random_budget_completion": by_policy["random"]["budget_completion_mean"],
        "slope_gap_greedy_slope": by_policy["slope_gap_greedy"][
            "slope_change_pct_mean"
        ],
        "area_weighted_greedy_slope": by_policy["area_weighted_greedy"][
            "slope_change_pct_mean"
        ],
        "immediate_slope_delta_slope": by_policy["immediate_slope_delta"][
            "slope_change_pct_mean"
        ],
        "immediate_slope_delta_budget_completion": by_policy["immediate_slope_delta"][
            "budget_completion_mean"
        ],
        "interpretation": (
            "strong local non-learning rules do not match the calibrated learned "
            "policy under the Bishan real-environment evaluation"
        ),
    }


def summarize_reward_scaling_comparator(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    return {
        "status": "supported",
        "path": display_path(path),
        "pre_specified_rank_by_slope": payload.get("pre_specified_rank_by_slope"),
        "best_scale": payload.get("best_scale"),
        "pre_specified_scale": payload.get("pre_specified_scale"),
        "pre_vs_best_relative_gap_pct": payload.get("pre_vs_best_relative_gap_pct"),
        "pre_vs_unscaled_slope_gain_pct": payload.get("pre_vs_unscaled_slope_gain_pct"),
        "interpretation": (
            "pre-specified observational alpha is compared against ordinary "
            "heuristic reward scaling and the unscaled learned reward"
        ),
    }


def summarize_reward_weight_sensitivity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    return {
        "status": "supported_as_fixed_policy_reward_sensitivity",
        "path": display_path(path),
        "n_episodes": payload.get("n_episodes"),
        "n_weight_settings": payload.get("n_weight_settings"),
        "n_policy_weight_summaries": len(payload.get("policy_weight_summaries", [])),
        "n_policy_metric_summaries": len(payload.get("policy_metric_summaries", [])),
        "n_pareto_rows": len(payload.get("pareto_front", [])),
        "n_best_policy_by_weight": len(payload.get("best_policy_by_weight", [])),
        "policy_retraining_under_all_weights": False,
        "interpretation": (
            "fixed-policy reward-component replay; supports reward preference "
            "analysis but does not prove retrained-policy robustness under every "
            "weight setting"
        ),
    }


def summarize_planning_significance(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    effects = payload["paired_calibration_effects"]
    calibrated = payload["calibrated_policy"]
    uncalibrated = payload.get("uncalibrated_policy", {})
    return {
        "status": "supported",
        "path": display_path(path),
        "n_paired_seeds": effects["n_paired_seeds"],
        "calibrated_slope_change_pct_mean": calibrated.get("slope_change_pct_mean"),
        "uncalibrated_slope_change_pct_mean": uncalibrated.get("slope_change_pct_mean"),
        "slope_delta_with_minus_no_mean": effects[
            "slope_change_pct_delta_with_minus_no_mean"
        ],
        "contiguity_delta_with_minus_no_mean": effects.get(
            "cont_change_delta_with_minus_no_mean"
        ),
        "baimu_count_delta_with_minus_no_mean": effects.get(
            "baimu_count_change_delta_with_minus_no_mean"
        ),
        "baimu_area_delta_with_minus_no_mean": effects.get(
            "baimu_area_change_ha_delta_with_minus_no_mean"
        ),
        "action_concentration_status": payload.get("action_concentration", {}).get("status"),
        "interpretation": (
            "planning outcomes include slope, contiguity, baimu, reward, budget, "
            "and available action-concentration diagnostics"
        ),
    }


def summarize_dongxing(
    audit_path: Path, block_summary_path: Path, dynamic_path: Path
) -> dict[str, Any]:
    audit = load_json(audit_path)
    blocks = load_json(block_summary_path)
    dynamic = load_json(dynamic_path)
    return {
        "status": "supported",
        "audit_path": display_path(audit_path),
        "block_summary_path": display_path(block_summary_path),
        "dynamic_path": display_path(dynamic_path),
        "n_parcels": audit["source"]["record_count"],
        "n_blocks": blocks["n_blocks"],
        "slope_coverage_farmland": audit["slope"]["by_land_use"]["farmland"][
            "coverage"
        ],
        "slope_coverage_forest": audit["slope"]["by_land_use"]["forest"]["coverage"],
        "has_complete_slope_for_swap": blocks["has_complete_slope_for_swap"],
        "dynamic_slope_gap_pct": dynamic["strategies"]["dynamic_slope_gap"][
            "slope_change_pct"
        ],
        "dynamic_area_weighted_pct": dynamic["strategies"][
            "dynamic_area_weighted_gap"
        ]["slope_change_pct"],
        "random_dynamic_mean_pct": dynamic["random_baseline"][
            "slope_change_pct_mean"
        ],
        "random_dynamic_n": dynamic["random_baseline"]["n_seeds"],
        "has_learned_policy": False,
        "interpretation": (
            "external data, action-space, and dynamic non-RL feasibility; "
            "not learned-policy transfer"
        ),
    }


def summarize_dongxing_rl_lite(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    learned_summary = payload["learned_policy"]["summary"]
    comparisons = payload.get("comparisons", {})
    return {
        "status": payload.get("status", "missing"),
        "path": display_path(path),
        "learner_type": payload.get("learner_type"),
        "n_blocks": payload.get("n_blocks"),
        "n_parcels": payload.get("n_parcels"),
        "train_seeds": payload.get("train_seeds", []),
        "eval_seeds": payload.get("eval_seeds", []),
        "training_time_s": payload.get("training_time_s"),
        "learned_slope_change_pct_mean": learned_summary.get(
            "slope_change_pct_mean"
        ),
        "learned_completed_pairs_mean": learned_summary.get("completed_pairs_mean"),
        "learned_unique_blocks_mean": learned_summary.get("unique_blocks_mean"),
        "learned_minus_random_slope_change_pct": comparisons.get(
            "learned_minus_random_slope_change_pct"
        ),
        "learned_minus_dynamic_slope_gap_slope_change_pct": comparisons.get(
            "learned_minus_dynamic_slope_gap_slope_change_pct"
        ),
        "learned_minus_dynamic_area_weighted_gap_slope_change_pct": comparisons.get(
            "learned_minus_dynamic_area_weighted_gap_slope_change_pct"
        ),
        "policy_transfer_tested": False,
        "slope_only_rl_actionability_tested": True,
        "interpretation": (
            "external slope-only learned block-selection actionability; not full "
            "cross-county learned-policy transfer and not Bishan full-reward validation"
        ),
    }


def summarize_dongxing_full_baselines(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    return {
        "status": payload.get("status", "supported_as_full_real_environment_baseline_pilot"),
        "path": display_path(path),
        "n_runs": payload.get("n_runs"),
        "n_policies": payload.get("n_policies"),
        "policies": payload.get("policies", []),
        "seeds": payload.get("seeds", []),
        "has_full_reward_metrics": True,
        "policy_summaries": payload.get("policy_summaries", {}),
        "learned_policy_tested": False,
        "interpretation": (
            "Dongxing full real-environment baseline pilot with slope, "
            "contiguity, baimu, and scalar reward; not learned-policy transfer "
            "and not final statistical superiority evidence"
        ),
    }


def summarize_dongxing_full_learned_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    learned = payload.get("learned_policy", {})
    return {
        "status": payload.get("status", "supported_as_dongxing_full_reward_learned_policy"),
        "path": display_path(path),
        "learner_type": payload.get("learner_type"),
        "train_seeds": payload.get("train_seeds", []),
        "eval_seeds": payload.get("eval_seeds", []),
        "n_train_seeds": len(payload.get("train_seeds", [])),
        "n_eval_seeds": len(payload.get("eval_seeds", [])),
        "episodes": payload.get("episodes"),
        "training_time_s": payload.get("training_time_s"),
        "learned_summary": learned.get("summary", {}),
        "comparisons": payload.get("comparisons", {}),
        "learned_policy_tested": True,
        "transfer_tested": False,
        "mbrl_transition_model_tested": False,
        "interpretation": payload.get(
            "claim_boundary",
            "Local Dongxing full-reward learned policy; not policy transfer and not transition-model MBRL",
        ),
    }


def summarize_dongxing_transition_diagnostics(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    model = payload.get("model", {})
    holdouts = payload.get("policy_holdout_diagnostics", [])
    holdout_reward_wins = [
        row
        for row in holdouts
        if float(row.get("reward_mae", math.inf))
        < float(row.get("reward_persistence_mae", -math.inf))
    ]
    holdout_selected_wins = [
        row
        for row in holdouts
        if float(row.get("selected_feature_mae", math.inf))
        < float(row.get("selected_feature_persistence_mae", -math.inf))
    ]
    return {
        "status": payload.get("status", "supported_as_dongxing_full_transition_diagnostic"),
        "path": display_path(path),
        "n_transitions": payload.get("n_transitions"),
        "policies": payload.get("policies", []),
        "random_split_selected_feature_mae": model.get("selected_feature_mae"),
        "random_split_selected_feature_persistence_mae": model.get(
            "selected_feature_persistence_mae"
        ),
        "random_split_global_feature_mae": model.get("global_feature_mae"),
        "random_split_global_feature_persistence_mae": model.get(
            "global_feature_persistence_mae"
        ),
        "random_split_reward_mae": model.get("reward_mae"),
        "random_split_reward_persistence_mae": model.get("reward_persistence_mae"),
        "random_split_selected_beats_baseline": _metric_beats_baseline(
            model, "selected_feature_mae", "selected_feature_persistence_mae"
        ),
        "random_split_global_beats_baseline": _metric_beats_baseline(
            model, "global_feature_mae", "global_feature_persistence_mae"
        ),
        "random_split_reward_beats_baseline": _metric_beats_baseline(
            model, "reward_mae", "reward_persistence_mae"
        ),
        "policy_holdout_count": len(holdouts),
        "policy_holdout_reward_beats_baseline_count": len(holdout_reward_wins),
        "policy_holdout_selected_beats_baseline_count": len(holdout_selected_wins),
        "policy_holdout_reward_beats_baseline_policies": [
            row.get("holdout_policy") for row in holdout_reward_wins
        ],
        "policy_holdout_selected_beats_baseline_policies": [
            row.get("holdout_policy") for row in holdout_selected_wins
        ],
        "mbrl_policy_trained": False,
        "policy_transfer_tested": False,
        "interpretation": (
            "Dongxing full real-environment transition learnability diagnostic. "
            "Random transition split metrics test local learnability; policy "
            "holdout metrics expose generalization limits. This artifact does "
            "not train an MBRL policy."
        ),
    }


def _metric_beats_baseline(payload: dict[str, Any], metric: str, baseline: str) -> bool:
    if metric not in payload or baseline not in payload:
        return False
    return float(payload[metric]) < float(payload[baseline])


def summarize_dongxing_full_model_based_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    summary = payload.get("model_based_policy", {}).get("summary", {})
    return {
        "status": payload.get("status", "supported_as_dongxing_full_one_step_model_based_policy"),
        "path": display_path(path),
        "n_training_transitions": payload.get("n_training_transitions"),
        "n_eval_seeds": summary.get("n"),
        "planning_horizon": payload.get("planning_horizon"),
        "model_based_reward_mean": summary.get("reward_mean"),
        "model_based_slope_change_pct_mean": summary.get("slope_change_pct_mean"),
        "model_based_cont_change_mean": summary.get("cont_change_mean"),
        "model_based_baimu_area_change_ha_mean": summary.get(
            "baimu_area_change_ha_mean"
        ),
        "comparisons": payload.get("comparisons", {}),
        "mbrl_transition_model_used": bool(payload.get("mbrl_transition_model_used", False)),
        "policy_transfer_tested": bool(payload.get("policy_transfer_tested", False)),
        "multi_step_mbrl_planning_tested": False,
        "interpretation": payload.get(
            "claim_boundary",
            "Local one-step Dongxing model-based action selection; not transfer or multi-step MBRL",
        ),
    }


def summarize_dongxing_model_based_optimization(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    heldout_summary = payload.get("heldout_eval", {}).get("summary", {})
    comparisons = payload.get("comparisons", {})
    scalarized_delta = comparisons.get("model_based_minus_scalarized_default_reward_mean")
    baimu_delta = comparisons.get("model_based_minus_baimu_aware_reward_mean")
    return {
        "status": payload.get("status", "supported_as_dongxing_model_based_scoring_optimization"),
        "path": display_path(path),
        "n_training_transitions": payload.get("n_training_transitions"),
        "n_candidates": len(payload.get("candidate_selection_summaries", [])),
        "best_candidate": payload.get("best_candidate", {}).get("name"),
        "n_eval_seeds": heldout_summary.get("n"),
        "heldout_reward_mean": heldout_summary.get("reward_mean"),
        "heldout_slope_change_pct_mean": heldout_summary.get("slope_change_pct_mean"),
        "heldout_cont_change_mean": heldout_summary.get("cont_change_mean"),
        "heldout_baimu_area_change_ha_mean": heldout_summary.get(
            "baimu_area_change_ha_mean"
        ),
        "model_based_minus_scalarized_default_reward_mean": scalarized_delta,
        "model_based_minus_baimu_aware_reward_mean": baimu_delta,
        "beats_scalarized_default_reward": scalarized_delta is not None
        and float(scalarized_delta) > 0.0,
        "beats_baimu_aware_reward": baimu_delta is not None and float(baimu_delta) > 0.0,
        "selection_eval_split": bool(payload.get("selection_eval_split", False)),
        "planning_horizon": payload.get("planning_horizon"),
        "mbrl_transition_model_used": bool(payload.get("mbrl_transition_model_used", False)),
        "policy_transfer_tested": bool(payload.get("policy_transfer_tested", False)),
        "multi_step_mbrl_planning_tested": False,
        "interpretation": payload.get(
            "claim_boundary",
            "Held-out one-step model-based scoring optimization; not transfer or multi-step MBRL",
        ),
    }


def classify_claim_scope(evidence: dict[str, Any]) -> list[dict[str, Any]]:
    """Classify which manuscript claims are supported by which evidence level."""
    bishan = evidence.get("bishan_seed_chain", {})
    dongxing = evidence.get("dongxing_dynamic", {})
    calibration = evidence.get("calibration", {})
    rollout = evidence.get("transition_rollout", {})
    policy_shift = evidence.get("policy_induced_diagnostics", {})
    dongxing_rl = evidence.get("dongxing_rl_lite", {})
    dongxing_full = evidence.get("dongxing_full_baselines", {})
    dongxing_full_learned = evidence.get("dongxing_full_learned_policy", {})
    dongxing_transition = evidence.get("dongxing_transition_diagnostics", {})
    dongxing_trajectory = evidence.get("dongxing_trajectory_summary", {})
    dongxing_mbrl_results = evidence.get("dongxing_mbrl_results", {})
    dongxing_model_based = evidence.get("dongxing_full_model_based_policy", {})
    dongxing_model_optimization = evidence.get("dongxing_model_based_optimization", {})
    transfer_finetune = evidence.get("transfer_finetune_results", {})
    reward_sensitivity = evidence.get("reward_weight_sensitivity", {})
    has_slope_only_rl = (
        dongxing_rl.get("status") == "supported_as_slope_only_rl_actionability"
    )

    scopes = [
        {
            "id": "bishan_model_based_policy_e2e",
            "claim": (
                "Bishan model-based policies are trained in a learned environment "
                "and evaluated in the real parcel-simulation environment."
            ),
            "status": "supported"
            if bishan.get("status") == "supported"
            and bishan.get("n_paired_seeds", 0) >= 15
            else "partial_or_missing",
            "evidence_level": "end_to_end_result_chain",
            "n_paired_seeds": bishan.get("n_paired_seeds", 0),
        },
        {
            "id": "calibration_scope",
            "claim": "The calibration factor improves learned-environment policy training.",
            "status": calibration.get("status", "missing"),
            "evidence_level": "observational_regularization_plus_downstream_evaluation",
            "causal_identification_claimed": False,
            "interpretation": calibration.get("interpretation"),
        },
        {
            "id": "transition_surrogate_scope",
            "claim": "The learned transition model can be used as an approximate training surrogate.",
            "status": rollout.get("status", "missing"),
            "evidence_level": "one_step_training_metrics_plus_multistep_rollout_diagnostics",
            "standalone_simulator_for_final_outcomes": False,
            "interpretation": rollout.get("interpretation"),
        },
        {
            "id": "policy_induced_surrogate_scope",
            "claim": (
                "The trained learned-environment policy remains evaluable under "
                "synchronized learned-vs-real rollouts."
            ),
            "status": "supported"
            if policy_shift.get("status") == "supported"
            and policy_shift.get("n_policy_episodes", 0) >= 3
            else "partial_or_missing",
            "evidence_level": "trained_policy_induced_learned_vs_real_diagnostics",
            "n_policy_episodes": policy_shift.get("n_policy_episodes", 0),
            "standalone_simulator_for_final_outcomes": False,
            "interpretation": policy_shift.get("interpretation"),
        },
        {
            "id": "dongxing_external_scope",
            "claim": "Dongxing supports external-county portability.",
            "status": "supported_as_external_slope_only_actionability"
            if has_slope_only_rl
            else (
                "supported_as_external_feasibility"
                if dongxing.get("status") == "supported"
                and not dongxing.get("has_learned_policy", False)
                else "policy_transfer_supported"
            ),
            "evidence_level": "external_data_action_space_dynamic_non_rl_plus_slope_only_rl_lite"
            if has_slope_only_rl
            else "external_data_action_space_dynamic_non_rl",
            "policy_transfer_tested": False
            if has_slope_only_rl
            else bool(dongxing.get("has_learned_policy", False)),
            "slope_only_rl_actionability_tested": bool(has_slope_only_rl),
            "interpretation": dongxing_rl.get("interpretation")
            if has_slope_only_rl
            else dongxing.get(
                "interpretation",
                "external feasibility evidence; not learned-policy transfer",
            ),
        },
    ]
    if reward_sensitivity.get("status") == "supported_as_fixed_policy_reward_sensitivity":
        scopes.append(
            {
                "id": "reward_function_scope",
                "claim": (
                    "The reward function has been tested through fixed-policy "
                    "component replay across alternative weight settings."
                ),
                "status": "supported_as_fixed_policy_reward_sensitivity",
                "evidence_level": "fixed_policy_reward_component_replay",
                "policy_retraining_under_all_weights": False,
                "interpretation": reward_sensitivity.get("interpretation"),
            }
        )
    if dongxing_full.get("status") in {
        "supported_as_full_real_environment_baselines",
        "supported_as_full_real_environment_baseline_pilot",
    }:
        evidence_level = (
            "external_full_real_environment_baselines"
            if dongxing_full.get("status") == "supported_as_full_real_environment_baselines"
            else "external_full_real_environment_baseline_pilot"
        )
        scopes.append(
            {
                "id": "dongxing_full_real_environment_scope",
                "claim": (
                    "Dongxing supports full multi-objective real-environment "
                    "baseline evaluation."
                ),
                "status": dongxing_full.get("status"),
                "evidence_level": evidence_level,
                "learned_policy_tested": False,
                "interpretation": dongxing_full.get("interpretation"),
            }
        )
    if dongxing_full_learned.get("status") == "supported_as_dongxing_full_reward_learned_policy":
        scopes.append(
            {
                "id": "dongxing_full_learned_policy_scope",
                "claim": "Dongxing supports local full-reward learned-policy actionability.",
                "status": dongxing_full_learned.get("status"),
                "evidence_level": "external_full_reward_local_learned_policy",
                "n_eval_seeds": dongxing_full_learned.get("n_eval_seeds", 0),
                "policy_transfer_tested": False,
                "mbrl_transition_model_tested": False,
                "interpretation": dongxing_full_learned.get("interpretation"),
            }
        )
    if dongxing_transition.get("status") == "supported_as_dongxing_full_transition_diagnostic":
        scopes.append(
            {
                "id": "dongxing_full_transition_diagnostic_scope",
                "claim": "Dongxing supports local full-environment transition learnability diagnostics.",
                "status": dongxing_transition.get("status"),
                "evidence_level": "external_full_transition_learnability_diagnostic",
                "n_transitions": dongxing_transition.get("n_transitions"),
                "policy_holdout_reward_beats_baseline_count": dongxing_transition.get(
                    "policy_holdout_reward_beats_baseline_count"
                ),
                "policy_holdout_count": dongxing_transition.get("policy_holdout_count"),
                "mbrl_policy_trained": False,
                "policy_transfer_tested": False,
                "interpretation": dongxing_transition.get("interpretation"),
            }
        )
    if dongxing_trajectory.get("status") == "supported_as_dongxing_trajectory_summary":
        scopes.append(
            {
                "id": "dongxing_trajectory_summary_scope",
                "claim": (
                    "Dongxing full-environment trajectory evidence has been "
                    "summarized for local learnability analysis."
                ),
                "status": dongxing_trajectory.get("status"),
                "evidence_level": "local_dongxing_trajectory_summary",
                "n_transitions": dongxing_trajectory.get("n_transitions"),
                "policy_holdout_count": dongxing_trajectory.get("policy_holdout_count"),
                "policy_transfer_tested": False,
                "mbrl_policy_trained": bool(dongxing_trajectory.get("mbrl_policy_trained", False)),
                "interpretation": dongxing_trajectory.get("interpretation"),
            }
        )
    if dongxing_mbrl_results.get("status") == "supported_as_local_dongxing_mbrl_results":
        scopes.append(
            {
                "id": "dongxing_local_mbrl_results_scope",
                "claim": (
                    "Dongxing local MBRL evidence bundles transition diagnostics, "
                    "one-step model-based policy evaluation, and held-out scoring optimization."
                ),
                "status": dongxing_mbrl_results.get("status"),
                "evidence_level": "local_dongxing_mbrl_result_bundle",
                "mbrl_transition_model_used": bool(
                    dongxing_mbrl_results.get("mbrl_transition_model_used", False)
                ),
                "policy_transfer_tested": False,
                "multi_step_mbrl_planning_tested": False,
                "interpretation": dongxing_mbrl_results.get("interpretation"),
            }
        )
    if dongxing_model_based.get("status") == "supported_as_dongxing_full_one_step_model_based_policy":
        scopes.append(
            {
                "id": "dongxing_full_model_based_policy_scope",
                "claim": (
                    "Dongxing supports local one-step learned transition/reward "
                    "model-based action selection in the full real environment."
                ),
                "status": dongxing_model_based.get("status"),
                "evidence_level": "external_full_one_step_model_based_policy",
                "n_eval_seeds": dongxing_model_based.get("n_eval_seeds"),
                "planning_horizon": dongxing_model_based.get("planning_horizon"),
                "mbrl_transition_model_used": bool(
                    dongxing_model_based.get("mbrl_transition_model_used", False)
                ),
                "policy_transfer_tested": False,
                "multi_step_mbrl_planning_tested": False,
                "interpretation": dongxing_model_based.get("interpretation"),
            }
        )
    if (
        dongxing_model_optimization.get("status")
        == "supported_as_dongxing_model_based_scoring_optimization"
    ):
        scopes.append(
            {
                "id": "dongxing_model_based_optimization_scope",
                "claim": (
                    "Dongxing supports held-out one-step model-based scoring "
                    "optimization in the full real environment."
                ),
                "status": dongxing_model_optimization.get("status"),
                "evidence_level": "external_full_heldout_scoring_optimization",
                "best_candidate": dongxing_model_optimization.get("best_candidate"),
                "n_eval_seeds": dongxing_model_optimization.get("n_eval_seeds"),
                "selection_eval_split": bool(
                    dongxing_model_optimization.get("selection_eval_split", False)
                ),
                "beats_scalarized_default_reward": dongxing_model_optimization.get(
                    "beats_scalarized_default_reward"
                ),
                "beats_baimu_aware_reward": dongxing_model_optimization.get(
                    "beats_baimu_aware_reward"
                ),
                "policy_transfer_tested": False,
                "multi_step_mbrl_planning_tested": False,
                "interpretation": dongxing_model_optimization.get("interpretation"),
            }
        )
    if transfer_finetune.get("status") == "structurally_invalid_for_direct_policy_transfer":
        scopes.append(
            {
                "id": "dongxing_transfer_finetune_scope",
                "claim": (
                    "Direct Bishan-to-Dongxing policy transfer is structurally "
                    "invalid without adapter-level changes."
                ),
                "status": transfer_finetune.get("status"),
                "evidence_level": "cross_county_dimension_mismatch",
                "direct_policy_transfer_tested": bool(
                    transfer_finetune.get("direct_policy_transfer_tested", False)
                ),
                "fine_tuning_tested": bool(transfer_finetune.get("fine_tuning_tested", False)),
                "fine_tuning_required": bool(transfer_finetune.get("fine_tuning_required", False)),
                "observation_dim_match": bool(
                    transfer_finetune.get("dimension_mismatch", {}).get(
                        "observation_dim_match", False
                    )
                ),
                "action_dim_match": bool(
                    transfer_finetune.get("dimension_mismatch", {}).get("action_dim_match", False)
                ),
                "interpretation": transfer_finetune.get("interpretation"),
            }
        )
    return scopes


def build_validation_report(
    repo_root: Path = REPO_ROOT, hash_core_assets: bool = False
) -> dict[str, Any]:
    paper7_dir = repo_root / "paper7"
    evidence: dict[str, Any] = {}

    core_assets = {
        "trajectories": [
            describe_file(path, hash_files=False)
            for path in sorted((paper7_dir / "trajectories").glob("*.npz"))
        ],
        "transition_model": describe_file(
            paper7_dir / "models" / "transition_model.pt",
            hash_files=hash_core_assets,
        ),
        "training_history": describe_file(
            paper7_dir / "models" / "training_history.json",
            hash_files=hash_core_assets,
        ),
        "dongxing_slope_gpkg": describe_file(
            paper7_dir / "data" / "dongxing_DLTB_with_slope.gpkg",
            hash_files=False,
        ),
    }

    evidence["transition_training"] = summarize_transition_training(
        paper7_dir / "models" / "training_history.json",
        paper7_dir / "models" / "transition_model.pt",
    )
    evidence["bishan_seed_chain"] = summarize_seed_evaluations(
        paper7_dir / "results" / "revision" / "seeds"
    )
    evidence["alpha_grid"] = summarize_alpha_grid(
        paper7_dir / "results" / "revision" / "alpha_grid" / "grid_results.json"
    )
    evidence["reward_scaling_comparator"] = summarize_reward_scaling_comparator(
        paper7_dir / "results" / "revision" / "reward_scaling_comparator.json"
    )
    evidence["reward_weight_sensitivity"] = summarize_reward_weight_sensitivity(
        paper7_dir / "results" / "full_rigor" / "reward_weight_sensitivity.json"
    )
    evidence["planning_significance"] = summarize_planning_significance(
        paper7_dir / "results" / "revision" / "planning_significance_audit.json"
    )
    evidence["transition_rollout"] = summarize_rollout_diagnostics(
        paper7_dir / "results" / "revision" / "transition_rollout_diagnostics.json"
    )
    evidence["policy_induced_diagnostics"] = summarize_policy_induced_diagnostics(
        select_policy_induced_diagnostics_path(paper7_dir)
    )
    evidence["calibration"] = summarize_calibration(
        paper7_dir / "results" / "causal_calibration.json",
        paper7_dir / "results" / "revision" / "causal_sensitivity_diagnostics.json",
    )
    evidence["bishan_non_learning_baselines"] = summarize_bishan_baselines(
        paper7_dir / "results" / "revision" / "bishan_strong_baselines.json"
    )
    evidence["dongxing_dynamic"] = summarize_dongxing(
        paper7_dir / "results" / "dongxing_data_audit_slope.json",
        paper7_dir / "results" / "dongxing_blocks_slope" / "summary.json",
        paper7_dir / "results" / "dongxing_dynamic_baselines.json",
    )
    evidence["dongxing_rl_lite"] = summarize_dongxing_rl_lite(
        paper7_dir / "results" / "revision" / "dongxing_rl_lite.json"
    )
    evidence["dongxing_full_baselines"] = summarize_dongxing_full_baselines(
        paper7_dir / "results" / "full_rigor" / "dongxing_full_baselines.json"
    )
    evidence["dongxing_full_learned_policy"] = summarize_dongxing_full_learned_policy(
        paper7_dir / "results" / "full_rigor" / "dongxing_full_learned_policy.json"
    )
    evidence["dongxing_transition_diagnostics"] = summarize_dongxing_transition_diagnostics(
        paper7_dir / "results" / "full_rigor" / "dongxing_transition_diagnostics.json"
    )
    evidence["dongxing_trajectory_summary"] = summarize_dongxing_trajectory_summary(
        paper7_dir / "results" / "full_rigor" / "dongxing_trajectories_summary.json"
    )
    evidence["dongxing_mbrl_results"] = summarize_dongxing_mbrl_results(
        paper7_dir / "results" / "full_rigor" / "dongxing_mbrl_results.json"
    )
    evidence["transfer_finetune_results"] = summarize_transfer_finetune_results(
        paper7_dir / "results" / "full_rigor" / "transfer_finetune_results.json"
    )
    evidence["dongxing_full_model_based_policy"] = summarize_dongxing_full_model_based_policy(
        paper7_dir / "results" / "full_rigor" / "dongxing_full_model_based_policy.json"
    )
    evidence["dongxing_model_based_optimization"] = summarize_dongxing_model_based_optimization(
        paper7_dir / "results" / "full_rigor" / "dongxing_model_based_optimization.json"
    )

    return {
        "description": (
            "End-to-end evidence audit for Paper 7. Expensive RL models are not "
            "retrained; the script validates the recorded data-to-result chain and "
            "classifies each manuscript claim by evidence level."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "core_assets": core_assets,
        "evidence": evidence,
        "claim_scope": classify_claim_scope(evidence),
        "overall_status": "supported_with_bounded_external_scope",
        "external_policy_transfer_tested": False,
        "external_slope_only_rl_actionability_tested": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT),
        help="Path to write the validation JSON report.",
    )
    parser.add_argument(
        "--hash-core-assets",
        action="store_true",
        help="Hash small core assets in addition to recording file sizes.",
    )
    args = parser.parse_args()

    report = build_validation_report(hash_core_assets=args.hash_core_assets)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote end-to-end validation report to {out_path}")
    print(f"Overall status: {report['overall_status']}")
    print(
        "Dongxing scope: "
        f"{next(item for item in report['claim_scope'] if item['id'] == 'dongxing_external_scope')['status']}"
    )


if __name__ == "__main__":
    main()
