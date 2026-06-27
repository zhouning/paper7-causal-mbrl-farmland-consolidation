"""Build manuscript-facing claim ledgers from the Paper 7 evidence audit."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
DEFAULT_AUDIT_PATH = PAPER7_DIR / "results" / "revision" / "end_to_end_validation.json"
DEFAULT_JSON_OUT = (
    PAPER7_DIR / "results" / "full_rigor" / "manuscript_evidence_ledger.json"
)
DEFAULT_MD_OUT = (
    PAPER7_DIR / "results" / "full_rigor" / "manuscript_evidence_ledger.md"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def _path_from(mapping: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _paths(*items: str | None) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for item in items:
        if item and item not in seen:
            paths.append(item)
            seen.add(item)
    return paths


def _metric(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def _add_claim(
    claims: list[dict[str, Any]],
    *,
    claim_id: str,
    manuscript_claim: str,
    artifact_paths: list[str],
    metrics: dict[str, Any],
    statistic: str,
    claim_strength: str,
    required_boundary: str,
    manuscript_destination: str,
) -> None:
    claims.append(
        {
            "claim_id": claim_id,
            "manuscript_claim": manuscript_claim,
            "artifact_paths": artifact_paths,
            "metrics": {key: _metric(value) for key, value in metrics.items()},
            "statistic": statistic,
            "claim_strength": claim_strength,
            "required_boundary": required_boundary,
            "manuscript_destination": manuscript_destination,
        }
    )


def build_manuscript_evidence_ledger(
    audit_path: Path = DEFAULT_AUDIT_PATH,
    *,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    audit = load_json(audit_path)
    evidence = audit["evidence"]
    claims: list[dict[str, Any]] = []

    transition = evidence.get("transition_training", {})
    bishan = evidence.get("bishan_seed_chain", {})
    calibration = evidence.get("calibration", {})
    scaling = evidence.get("reward_scaling_comparator", {})
    alpha_grid = evidence.get("alpha_grid", {})
    rollout = evidence.get("transition_rollout", {})
    policy_diag = evidence.get("policy_induced_diagnostics", {})
    planning = evidence.get("planning_significance", {})
    reward_weights = evidence.get("reward_weight_sensitivity", {})
    bishan_baselines = evidence.get("bishan_non_learning_baselines", {})
    dongxing_baselines = evidence.get("dongxing_full_baselines", {})
    dongxing_learned = evidence.get("dongxing_full_learned_policy", {})
    dongxing_mbrl = evidence.get("dongxing_mbrl_results", {})
    dongxing_multistep = evidence.get("dongxing_multistep_mbrl_policy", {})
    transfer = evidence.get("transfer_finetune_results", {})
    source_ablation = evidence.get("trajectory_source_ablation", {})

    paired = bishan.get("paired_slope_test") or {}
    _add_claim(
        claims,
        claim_id="bishan_learned_environment_e2e",
        manuscript_claim=(
            "Bishan learned-environment policies are trained in the learned "
            "environment and evaluated in the real parcel-simulation environment."
        ),
        artifact_paths=_paths(
            bishan.get("seed_dir"),
            _path_from(transition.get("model", {}), "path"),
            _path_from(transition.get("history", {}), "path"),
        ),
        metrics={
            "n_paired_seeds": bishan.get("n_paired_seeds"),
            "no_cal_slope_mean": bishan.get("no_cal_mean"),
            "with_cal_slope_mean": bishan.get("with_cal_mean"),
            "improvement_pct": bishan.get("improvement_pct"),
            "transition_obs_cosine": transition.get("final_val_obs_cosine"),
            "transition_reward_mse": transition.get("final_val_reward_mse"),
        },
        statistic="paired seed summary for calibrated versus uncalibrated learned policies",
        claim_strength="supported_strong",
        required_boundary=(
            "Final policy outcomes are measured in the real environment; the "
            "learned environment is a training surrogate."
        ),
        manuscript_destination="main_results",
    )

    _add_claim(
        claims,
        claim_id="calibration_effect",
        manuscript_claim=(
            "Treatment-effect-informed reward scaling improves Bishan "
            "learned-environment policy training under paired seed evaluation."
        ),
        artifact_paths=_paths(
            calibration.get("calibration_path"),
            calibration.get("sensitivity_path"),
            bishan.get("seed_dir"),
        ),
        metrics={
            "calibration_factor": calibration.get("calibration_factor"),
            "one_sided_p": paired.get("one_sided_p"),
            "two_sided_p": paired.get("two_sided_p"),
            "improvement_pct": bishan.get("improvement_pct"),
            "calibrated_win_count": bishan.get("calibrated_slope_win_count"),
            "uncalibrated_win_count": bishan.get("uncalibrated_slope_win_count"),
        },
        statistic="exact paired sign-flip test",
        claim_strength="supported_bounded",
        required_boundary=(
            "observational reward regularization; not definitive causal "
            "identification"
        ),
        manuscript_destination="main_results",
    )

    _add_claim(
        claims,
        claim_id="reward_scaling_comparator",
        manuscript_claim=(
            "The pre-specified observational calibration factor is close to the "
            "empirical reward-scale optimum and improves over unscaled rewards."
        ),
        artifact_paths=_paths(scaling.get("path"), alpha_grid.get("grid_path")),
        metrics={
            "n_grid_runs": alpha_grid.get("n_runs"),
            "n_alphas": alpha_grid.get("n_alphas"),
            "best_scale": scaling.get("best_scale"),
            "pre_specified_scale": scaling.get("pre_specified_scale"),
            "pre_vs_best_relative_gap_pct": scaling.get(
                "pre_vs_best_relative_gap_pct"
            ),
            "pre_vs_unscaled_slope_gain_pct": scaling.get(
                "pre_vs_unscaled_slope_gain_pct"
            ),
        },
        statistic="eight-value reward-scale grid with five seeds per value",
        claim_strength="supported_strong",
        required_boundary=(
            "This supports data-driven scaling, not a proof that the global "
            "factor is optimal in every state."
        ),
        manuscript_destination="main_results",
    )

    _add_claim(
        claims,
        claim_id="transition_surrogate_diagnostics",
        manuscript_claim=(
            "Recorded-action and policy-induced diagnostics support the learned "
            "environment as an approximate training surrogate."
        ),
        artifact_paths=_paths(
            rollout.get("path"),
            policy_diag.get("path"),
            source_ablation.get("path"),
        ),
        metrics={
            "horizon_100_mask_agreement": rollout.get("horizon_100_mask_agreement"),
            "horizon_100_reward_mae": rollout.get("horizon_100_reward_mae"),
            "policy_induced_mask_agreement": policy_diag.get(
                "policy_induced_mask_agreement_mean"
            ),
            "policy_induced_support_distance": policy_diag.get(
                "policy_induced_support_distance_mean"
            ),
            "policy_induced_final_real_slope_pct": policy_diag.get(
                "policy_induced_final_real_slope_pct_mean"
            ),
            "best_trajectory_source": source_ablation.get(
                "best_source_by_all_horizon_100_reward_mae"
            ),
        },
        statistic="recorded-action rollout and synchronized learned-vs-real diagnostics",
        claim_strength="supported_bounded",
        required_boundary=(
            "Diagnostics support surrogate training but do not replace final "
            "real-environment evaluation."
        ),
        manuscript_destination="main_results",
    )

    _add_claim(
        claims,
        claim_id="planning_tradeoff_boundary",
        manuscript_claim=(
            "Calibration improves the slope objective but is not a Pareto "
            "improvement on every planning metric."
        ),
        artifact_paths=_paths(planning.get("path"), reward_weights.get("path")),
        metrics={
            "slope_delta_with_minus_no_mean": planning.get(
                "slope_delta_with_minus_no_mean"
            ),
            "contiguity_delta_with_minus_no_mean": planning.get(
                "contiguity_delta_with_minus_no_mean"
            ),
            "baimu_count_delta_with_minus_no_mean": planning.get(
                "baimu_count_delta_with_minus_no_mean"
            ),
            "baimu_area_delta_with_minus_no_mean": planning.get(
                "baimu_area_delta_with_minus_no_mean"
            ),
            "n_weight_settings": reward_weights.get("n_weight_settings"),
        },
        statistic="paired planning-metric audit and fixed-policy reward-component replay",
        claim_strength="supported_bounded",
        required_boundary=(
            "Report slope-contiguity-baimu trade-offs; do not claim a Pareto "
            "improvement across every planning metric."
        ),
        manuscript_destination="main_results_and_discussion",
    )

    _add_claim(
        claims,
        claim_id="bishan_non_learning_baselines",
        manuscript_claim=(
            "Strong Bishan non-learning rules do not match the calibrated "
            "learned policy under the stored real-environment evaluation."
        ),
        artifact_paths=_paths(bishan_baselines.get("path")),
        metrics={
            "random_slope_mean": bishan_baselines.get("random_slope_mean"),
            "slope_gap_greedy_slope": bishan_baselines.get("slope_gap_greedy_slope"),
            "area_weighted_greedy_slope": bishan_baselines.get(
                "area_weighted_greedy_slope"
            ),
            "immediate_slope_delta_slope": bishan_baselines.get(
                "immediate_slope_delta_slope"
            ),
        },
        statistic="real-environment non-learning baseline audit",
        claim_strength="supported_bounded",
        required_boundary=(
            "This is a local baseline comparison, not a universal claim over "
            "all hand-designed rules."
        ),
        manuscript_destination="main_results",
    )

    _add_claim(
        claims,
        claim_id="dongxing_local_counterpart",
        manuscript_claim=(
            "Dongxing supports a full-reward local counterpart with local "
            "learned-policy and local learned-environment evidence."
        ),
        artifact_paths=_paths(
            dongxing_baselines.get("path"),
            dongxing_learned.get("path"),
            dongxing_mbrl.get("path"),
            dongxing_multistep.get("path"),
            transfer.get("path"),
        ),
        metrics={
            "full_baseline_status": dongxing_baselines.get("status"),
            "local_learned_eval_seeds": dongxing_learned.get("n_eval_seeds"),
            "local_mbrl_status": dongxing_mbrl.get("status"),
            "multistep_planning_tested": dongxing_mbrl.get(
                "multi_step_mbrl_planning_tested"
            ),
            "multistep_reward_mean": dongxing_multistep.get("real_eval_reward_mean"),
            "multistep_slope_change_pct_mean": dongxing_multistep.get(
                "real_eval_slope_change_pct_mean"
            ),
        },
        statistic="external full-reward local counterpart result bundle",
        claim_strength="supported_bounded",
        required_boundary=(
            "Dongxing is local external-counterpart evidence, not direct "
            "Bishan-to-Dongxing policy transfer."
        ),
        manuscript_destination="main_results_and_discussion",
    )

    _add_claim(
        claims,
        claim_id="direct_transfer_boundary",
        manuscript_claim=(
            "Direct Bishan-to-Dongxing policy transfer is structurally invalid "
            "without adapter-level changes."
        ),
        artifact_paths=_paths(transfer.get("path")),
        metrics={
            "status": transfer.get("status"),
            "observation_dim_match": transfer.get("dimension_mismatch", {}).get(
                "observation_dim_match"
            ),
            "action_dim_match": transfer.get("dimension_mismatch", {}).get(
                "action_dim_match"
            ),
            "direct_policy_transfer_tested": transfer.get(
                "direct_policy_transfer_tested"
            ),
            "fine_tuning_required": transfer.get("fine_tuning_required"),
        },
        statistic="dimension-mismatch audit",
        claim_strength="structural_boundary",
        required_boundary="not direct Bishan-to-Dongxing policy transfer",
        manuscript_destination="discussion_and_limitations",
    )

    _add_claim(
        claims,
        claim_id="reward_weight_replay_boundary",
        manuscript_claim=(
            "Reward-weight sensitivity is supported as fixed-policy replay and "
            "reward-specification evidence."
        ),
        artifact_paths=_paths(reward_weights.get("path")),
        metrics={
            "n_episodes": reward_weights.get("n_episodes"),
            "n_weight_settings": reward_weights.get("n_weight_settings"),
            "policy_retraining_under_all_weights": reward_weights.get(
                "policy_retraining_under_all_weights"
            ),
            "reward_specification_exported": reward_weights.get(
                "reward_specification_exported"
            ),
        },
        statistic="fixed-policy replay across alternative reward weights",
        claim_strength="supported_bounded",
        required_boundary=(
            "fixed-policy replay; not proof that retrained policies are robust "
            "under every planning preference"
        ),
        manuscript_destination="discussion_and_supplement",
    )

    return {
        "description": "Manuscript-facing Paper 7 claim-to-evidence ledger.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "audit_path": str(audit_path),
        "overall_status": audit.get("overall_status"),
        "claims": claims,
        "required_boundaries": [
            "observational reward regularization",
            "not definitive causal identification",
            "descriptive model-free baseline comparison",
            "not direct Bishan-to-Dongxing policy transfer",
            "final real-environment evaluation",
            "fixed-policy replay",
        ],
        "forbidden_overclaims": [
            "universal generalization across counties",
            "formal superiority over all model-free RL methods",
            "direct transfer of Bishan policies to Dongxing",
            "definitive causal identification of reward effects",
            "transition model as a replacement for final real-environment evaluation",
        ],
    }


def render_ledger_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Paper 7 Manuscript Evidence Ledger",
        "",
        f"Generated UTC: `{ledger['generated_utc']}`",
        f"Audit path: `{ledger['audit_path']}`",
        f"Overall status: `{ledger['overall_status']}`",
        "",
        "## Claim Map",
        "",
        "| Claim ID | Strength | Key Metrics | Required Boundary |",
        "|---|---|---|---|",
    ]
    for row in ledger["claims"]:
        metric_text = "; ".join(
            f"{key}={value}" for key, value in row["metrics"].items()
        )
        lines.append(
            "| {claim_id} | {strength} | {metrics} | {boundary} |".format(
                claim_id=row["claim_id"],
                strength=row["claim_strength"],
                metrics=metric_text,
                boundary=row["required_boundary"],
            )
        )
    lines.extend(["", "## Artifact Paths", ""])
    for row in ledger["claims"]:
        lines.append(f"### {row['claim_id']}")
        for path in row["artifact_paths"]:
            lines.append(f"- `{path}`")
        lines.append("")
    lines.extend(["## Required Boundary Phrases", ""])
    for phrase in ledger["required_boundaries"]:
        lines.append(f"- {phrase}")
    lines.extend(["", "## Forbidden Overclaims", ""])
    for phrase in ledger["forbidden_overclaims"]:
        lines.append(f"- {phrase}")
    lines.append("")
    return "\n".join(lines)


def write_manuscript_evidence_ledger(
    *,
    audit_path: Path = DEFAULT_AUDIT_PATH,
    json_out: Path = DEFAULT_JSON_OUT,
    md_out: Path = DEFAULT_MD_OUT,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Path]:
    ledger = build_manuscript_evidence_ledger(audit_path, repo_root=repo_root)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(ledger, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md_out.write_text(render_ledger_markdown(ledger), encoding="utf-8")
    return {"json": json_out, "markdown": md_out}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    written = write_manuscript_evidence_ledger(
        audit_path=args.audit,
        json_out=args.json_out,
        md_out=args.md_out,
    )
    print(json.dumps({key: str(path) for key, path in written.items()}, indent=2))


if __name__ == "__main__":
    main()
