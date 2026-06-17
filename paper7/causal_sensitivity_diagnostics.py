"""Robustness diagnostics for Paper 7 observational reward calibration.

The original calibration script estimates whether selecting high-potential
blocks is associated with higher observed reward. This module adds diagnostics
that are expected in a stricter review: overlap, covariate balance, bootstrap
uncertainty, threshold sensitivity, and trimming sensitivity.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BLOCK_FEATURE_NAMES = [
    "block_farm_slope_norm",
    "block_forest_slope_norm",
    "block_slope_gap_norm",
    "block_best_swap_gain_norm",
    "block_farm_slope_std_norm",
    "block_top_farm_slope_norm",
    "block_bottom_forest_slope_norm",
    "block_farm_area_norm",
    "block_forest_area_norm",
    "block_swap_potential_norm",
    "block_swaps_in_block_norm",
    "block_compactness",
    "block_area_norm",
    "block_neighbor_invested_frac",
    "block_neighbor_farm_area_norm",
    "block_current_farm_area_norm",
    "block_invested",
]

GLOBAL_FEATURE_NAMES = [
    "budget_remaining",
    "global_slope_norm",
    "global_contiguity_norm",
    "step_frac",
    "slope_improvement",
    "contiguity_improvement",
    "baimu_count_norm",
    "baimu_area_frac",
    "blocks_invested_frac",
    "investment_entropy",
    "cross_township_baimu_ratio",
    "max_township_frac",
]

DEFAULT_COVARIATES = [
    "budget_remaining",
    "global_slope_norm",
    "global_contiguity_norm",
    "step_frac",
    "slope_improvement",
    "contiguity_improvement",
    "baimu_count_norm",
    "baimu_area_frac",
    "blocks_invested_frac",
    "investment_entropy",
    "max_township_frac",
    "block_farm_slope_norm",
    "block_forest_slope_norm",
    "block_slope_gap_norm",
    "block_farm_slope_std_norm",
    "block_farm_area_norm",
    "block_forest_area_norm",
    "block_swap_potential_norm",
    "block_compactness",
    "block_area_norm",
    "block_neighbor_invested_frac",
    "block_neighbor_farm_area_norm",
    "block_current_farm_area_norm",
    "block_invested",
    "policy_greedy",
]


def infer_policy_name(path: Path) -> str:
    """Infer trajectory policy name from a file stem such as greedy_seed0."""
    match = re.match(r"([A-Za-z0-9_-]+)_seed\d+", path.stem)
    return match.group(1) if match else path.stem


def build_rows_from_trajectory_arrays(
    block_features: np.ndarray,
    global_features: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    policy_name: str,
    treatment_percentile: float,
) -> list[dict[str, float | int | str]]:
    """Build one row per observed transition for a chosen treatment threshold."""
    rows: list[dict[str, float | int | str]] = []
    for t, action_raw in enumerate(actions):
        action = int(action_raw)
        selected_block = block_features[t, action].astype(float)
        selected_global = global_features[t].astype(float)
        threshold = float(np.percentile(block_features[t, :, 3].astype(float), treatment_percentile))
        row: dict[str, float | int | str] = {
            "policy": policy_name,
            "policy_greedy": 1.0 if policy_name.lower().startswith("greedy") else 0.0,
            "treatment": int(float(selected_block[3]) > threshold),
            "outcome": float(rewards[t]),
            "action": action,
            "treatment_percentile": float(treatment_percentile),
            "threshold_value": threshold,
        }
        for idx, name in enumerate(GLOBAL_FEATURE_NAMES):
            if idx < len(selected_global):
                row[name] = float(selected_global[idx])
        for idx, name in enumerate(BLOCK_FEATURE_NAMES):
            if idx < len(selected_block):
                row[name] = float(selected_block[idx])
        rows.append(row)
    return rows


def build_observational_dataset(
    trajectory_dir: Path,
    treatment_percentile: float,
    max_transitions_per_file: int | None = None,
) -> pd.DataFrame:
    """Load all trajectory files and build a treatment/outcome dataset."""
    rows: list[dict[str, float | int | str]] = []
    for path in sorted(trajectory_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=False)
        n = len(data["actions"])
        if max_transitions_per_file is not None:
            n = min(n, int(max_transitions_per_file))
        rows.extend(
            build_rows_from_trajectory_arrays(
                block_features=data["block_features"][:n].astype(np.float32),
                global_features=data["global_features"][:n].astype(np.float32),
                actions=data["actions"][:n],
                rewards=data["rewards"][:n].astype(np.float32),
                policy_name=infer_policy_name(path),
                treatment_percentile=treatment_percentile,
            )
        )
    return pd.DataFrame(rows)


def filter_dataset_by_policy(df: pd.DataFrame, policies: list[str] | None) -> pd.DataFrame:
    """Return rows whose policy is in the requested policy list."""
    if not policies:
        return df
    wanted = {policy.lower() for policy in policies}
    return df.loc[df["policy"].astype(str).str.lower().isin(wanted)].copy()


def add_propensity_scores(df: pd.DataFrame, covariates: list[str], random_state: int = 42) -> pd.DataFrame:
    """Estimate propensity scores using cross-fitted gradient boosting."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    result = df.copy()
    x = result[covariates].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    treatment = result["treatment"].to_numpy(dtype=int)
    if len(np.unique(treatment)) < 2:
        result["propensity"] = float(treatment.mean())
        return result

    min_class = int(np.bincount(treatment).min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        result["propensity"] = float(treatment.mean())
        return result

    model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    propensity = cross_val_predict(model, x, treatment, cv=cv, method="predict_proba")[:, 1]
    result["propensity"] = np.clip(propensity, 1e-4, 1.0 - 1e-4)
    return result


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.mean(values))
    return float(np.sum(values * weights) / total)


def _weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size <= 1:
        return 0.0
    mean = _weighted_mean(values, weights)
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.var(values, ddof=1))
    return float(np.sum(weights * (values - mean) ** 2) / total)


def _smd(treated: np.ndarray, control: np.ndarray, control_weights: np.ndarray | None = None) -> float:
    if treated.size == 0 or control.size == 0:
        return float("nan")
    treated_weights = np.ones_like(treated, dtype=float)
    if control_weights is None:
        control_weights = np.ones_like(control, dtype=float)
    treated_mean = _weighted_mean(treated, treated_weights)
    control_mean = _weighted_mean(control, control_weights)
    treated_var = _weighted_variance(treated, treated_weights)
    control_var = _weighted_variance(control, control_weights)
    pooled_sd = float(np.sqrt((treated_var + control_var) / 2.0))
    if pooled_sd < 1e-12:
        return 0.0 if abs(treated_mean - control_mean) < 1e-12 else float("inf")
    return float((treated_mean - control_mean) / pooled_sd)


def att_weights(df: pd.DataFrame) -> np.ndarray:
    """Return ATT-style weights: treated=1, controls=e/(1-e)."""
    treatment = df["treatment"].to_numpy(dtype=int)
    propensity = np.clip(df["propensity"].to_numpy(dtype=float), 1e-4, 1.0 - 1e-4)
    return np.where(treatment == 1, 1.0, propensity / (1.0 - propensity))


def standardized_mean_differences(df: pd.DataFrame, covariates: list[str]) -> dict[str, Any]:
    """Compute SMDs before and after ATT weighting."""
    treatment = df["treatment"].to_numpy(dtype=int)
    weights = att_weights(df) if "propensity" in df.columns else np.ones(len(df), dtype=float)
    result: dict[str, Any] = {}
    before_abs: list[float] = []
    after_abs: list[float] = []
    for covariate in covariates:
        values = df[covariate].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        treated = values[treatment == 1]
        control = values[treatment == 0]
        before = _smd(treated, control)
        after = _smd(treated, control, control_weights=weights[treatment == 0])
        result[covariate] = {
            "before": round(float(before), 6),
            "after_att_weighting": round(float(after), 6),
        }
        if np.isfinite(before):
            before_abs.append(abs(float(before)))
        if np.isfinite(after):
            after_abs.append(abs(float(after)))
    result["max_abs_before"] = round(float(max(before_abs)) if before_abs else float("nan"), 6)
    result["max_abs_after_att_weighting"] = round(float(max(after_abs)) if after_abs else float("nan"), 6)
    result["mean_abs_before"] = round(float(np.mean(before_abs)) if before_abs else float("nan"), 6)
    result["mean_abs_after_att_weighting"] = round(float(np.mean(after_abs)) if after_abs else float("nan"), 6)
    return result


def estimate_att_from_propensity(
    df: pd.DataFrame,
    trim_bounds: tuple[float, float] = (0.05, 0.95),
) -> dict[str, float | int]:
    """Estimate ATT using inverse-odds weights for controls."""
    low, high = trim_bounds
    propensity = df["propensity"].to_numpy(dtype=float)
    mask = (propensity >= low) & (propensity <= high)
    used = df.loc[mask].copy()
    treatment = used["treatment"].to_numpy(dtype=int)
    outcome = used["outcome"].to_numpy(dtype=float)
    if treatment.sum() == 0 or (1 - treatment).sum() == 0:
        return {
            "att": float("nan"),
            "treated_mean": float("nan"),
            "weighted_control_mean": float("nan"),
            "n_used": int(len(used)),
            "n_treated": int(treatment.sum()),
            "n_control": int((1 - treatment).sum()),
        }
    weights = att_weights(used)
    treated_mean = float(np.mean(outcome[treatment == 1]))
    control_mean = _weighted_mean(outcome[treatment == 0], weights[treatment == 0])
    return {
        "att": round(float(treated_mean - control_mean), 6),
        "treated_mean": round(treated_mean, 6),
        "weighted_control_mean": round(control_mean, 6),
        "n_used": int(len(used)),
        "n_treated": int(treatment.sum()),
        "n_control": int((1 - treatment).sum()),
    }


def bootstrap_att_ci(
    df: pd.DataFrame,
    trim_bounds: tuple[float, float],
    n_bootstrap: int = 300,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Bootstrap the ATT estimator without re-fitting propensity scores."""
    rng = np.random.default_rng(random_state)
    estimates: list[float] = []
    n = len(df)
    for _ in range(int(n_bootstrap)):
        sample = df.iloc[rng.integers(0, n, size=n)]
        estimate = estimate_att_from_propensity(sample, trim_bounds=trim_bounds)["att"]
        if np.isfinite(float(estimate)):
            estimates.append(float(estimate))
    if not estimates:
        return {"n_bootstrap": 0, "ci_low": float("nan"), "ci_high": float("nan"), "bootstrap_se": float("nan")}
    values = np.array(estimates, dtype=float)
    return {
        "n_bootstrap": int(len(values)),
        "ci_low": round(float(np.quantile(values, 0.025)), 6),
        "ci_high": round(float(np.quantile(values, 0.975)), 6),
        "bootstrap_se": round(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0, 6),
    }


def overlap_summary(df: pd.DataFrame) -> dict[str, float | int]:
    """Summarize propensity-score overlap for treated and control observations."""
    treatment = df["treatment"].to_numpy(dtype=int)
    propensity = df["propensity"].to_numpy(dtype=float)
    treated = propensity[treatment == 1]
    control = propensity[treatment == 0]
    common_low = max(float(np.min(treated)), float(np.min(control))) if len(treated) and len(control) else float("nan")
    common_high = min(float(np.max(treated)), float(np.max(control))) if len(treated) and len(control) else float("nan")
    in_common = (propensity >= common_low) & (propensity <= common_high) if np.isfinite(common_low) else np.zeros_like(propensity, dtype=bool)
    return {
        "n": int(len(df)),
        "treated": int(treatment.sum()),
        "control": int((1 - treatment).sum()),
        "treated_share": round(float(treatment.mean()), 6),
        "ps_min": round(float(np.min(propensity)), 6),
        "ps_q05": round(float(np.quantile(propensity, 0.05)), 6),
        "ps_median": round(float(np.median(propensity)), 6),
        "ps_q95": round(float(np.quantile(propensity, 0.95)), 6),
        "ps_max": round(float(np.max(propensity)), 6),
        "common_support_low": round(common_low, 6),
        "common_support_high": round(common_high, 6),
        "common_support_share": round(float(np.mean(in_common)), 6),
    }


def run_threshold_diagnostic(
    trajectory_dir: Path,
    treatment_percentile: float,
    covariates: list[str] | None = None,
    trim_bounds: tuple[float, float] = (0.05, 0.95),
    n_bootstrap: int = 300,
    max_transitions_per_file: int | None = None,
    policies: list[str] | None = None,
    random_state: int = 42,
) -> dict[str, Any]:
    """Run the full sensitivity diagnostic for one treatment threshold."""
    covariates = covariates or DEFAULT_COVARIATES
    df = build_observational_dataset(
        trajectory_dir=trajectory_dir,
        treatment_percentile=treatment_percentile,
        max_transitions_per_file=max_transitions_per_file,
    )
    df = filter_dataset_by_policy(df, policies)
    usable_covariates = [cov for cov in covariates if cov in df.columns]
    scored = add_propensity_scores(df, usable_covariates, random_state=random_state)
    trimmed = estimate_att_from_propensity(scored, trim_bounds=trim_bounds)
    untrimmed = estimate_att_from_propensity(scored, trim_bounds=(0.0, 1.0))
    ci = bootstrap_att_ci(scored, trim_bounds=trim_bounds, n_bootstrap=n_bootstrap, random_state=random_state)
    return {
        "treatment_percentile": float(treatment_percentile),
        "covariates": usable_covariates,
        "overlap": overlap_summary(scored),
        "smd": standardized_mean_differences(scored, usable_covariates),
        "att_untrimmed": untrimmed,
        "att_trimmed": {**trimmed, **ci, "trim_low": trim_bounds[0], "trim_high": trim_bounds[1]},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory-dir", type=Path, default=Path("paper7/trajectories"))
    parser.add_argument("--percentiles", default="40,50,60,70")
    parser.add_argument("--trim-low", type=float, default=0.05)
    parser.add_argument("--trim-high", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--max-transitions-per-file", type=int, default=None)
    parser.add_argument(
        "--policy-scopes",
        default="mixed,random,greedy",
        help="Comma-separated scopes to run. Use mixed for all policies, or policy names such as random and greedy.",
    )
    parser.add_argument("--output", type=Path, default=Path("paper7/results/revision/causal_sensitivity_diagnostics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    percentiles = [float(item) for item in str(args.percentiles).split(",") if item.strip()]
    scopes = [item.strip() for item in str(args.policy_scopes).split(",") if item.strip()]
    results = {
        "description": (
            "Observational treatment-effect diagnostics for reward calibration. Treatment is "
            "whether the selected block's best-swap-gain feature exceeds a within-state percentile."
        ),
        "trajectory_dir": os.fspath(args.trajectory_dir),
        "scopes": [],
    }
    for scope in scopes:
        scope_result: dict[str, Any] = {
            "scope": scope,
            "policies": None if scope.lower() == "mixed" else [scope],
            "thresholds": [],
        }
        for percentile in percentiles:
            scope_result["thresholds"].append(
                run_threshold_diagnostic(
                    trajectory_dir=args.trajectory_dir,
                    treatment_percentile=percentile,
                    trim_bounds=(args.trim_low, args.trim_high),
                    n_bootstrap=args.bootstrap,
                    max_transitions_per_file=args.max_transitions_per_file,
                    policies=None if scope.lower() == "mixed" else [scope],
                )
            )
        results["scopes"].append(scope_result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
