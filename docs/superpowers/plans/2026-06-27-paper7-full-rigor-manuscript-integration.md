# Paper 7 Full-Rigor Manuscript Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ledger-driven Paper 7 manuscript revision so every major claim traces to stored full-rigor evidence and all CEUS manuscript source copies use the same audited boundaries.

**Architecture:** Add a small evidence-ledger generator that reads the existing end-to-end audit JSON and writes machine-readable plus Markdown claim ledgers. Then add tests that verify ledger paths, audited statistics, manuscript boundary language, and source-copy consistency before applying targeted manuscript and editorial-document revisions.

**Tech Stack:** Python standard library, pytest, existing Paper 7 JSON artifacts, LaTeX source files, pdflatex.

---

## File Structure

- Create `paper7/manuscript_evidence_ledger.py`: reads `paper7/results/revision/end_to_end_validation.json` and writes manuscript-facing evidence ledgers.
- Create `tests/test_manuscript_evidence_ledger.py`: verifies the ledger generator on real artifacts and a small fixture.
- Modify `tests/test_manuscript_claim_consistency.py`: expands claim checks from the anonymous manuscript to all CEUS manuscript source copies and requires the generated ledger.
- Generate `paper7/results/full_rigor/manuscript_evidence_ledger.json`: machine-readable claim-to-evidence ledger.
- Generate `paper7/results/full_rigor/manuscript_evidence_ledger.md`: human-readable writing ledger.
- Modify `submission/ceus/01_main_document_anonymous/manuscript.tex`: adds the ledger citation and tightens wording where needed.
- Modify `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex`: syncs anonymous editable source with the main anonymous manuscript.
- Modify `submission/ceus/06_latex_source_editable/manuscript_signed.tex`: updates signed source claim wording to match audited statistics while preserving author information.
- Modify `submission/ceus/03_highlights/highlights.txt`: aligns highlights with the ledger-driven main claim.
- Modify `submission/ceus/04_cover_letter/cover_letter.txt`: aligns cover-letter claims with the full-rigor but bounded evidence chain.

---

### Task 1: Add Evidence Ledger Tests

**Files:**
- Create: `tests/test_manuscript_evidence_ledger.py`
- Read: `paper7/results/revision/end_to_end_validation.json`
- Read: `paper7/results/full_rigor/`

- [ ] **Step 1: Write the failing test**

Create `tests/test_manuscript_evidence_ledger.py` with exactly this content:

```python
import json
from pathlib import Path

import pytest

from paper7.manuscript_evidence_ledger import (
    build_manuscript_evidence_ledger,
    render_ledger_markdown,
    write_manuscript_evidence_ledger,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_AUDIT_PATH = (
    REPO_ROOT / "paper7" / "results" / "revision" / "end_to_end_validation.json"
)


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_ledger_from_real_audit_contains_required_claims():
    ledger = build_manuscript_evidence_ledger(REAL_AUDIT_PATH, repo_root=REPO_ROOT)
    rows = {row["claim_id"]: row for row in ledger["claims"]}

    expected = {
        "bishan_learned_environment_e2e",
        "calibration_effect",
        "reward_scaling_comparator",
        "transition_surrogate_diagnostics",
        "planning_tradeoff_boundary",
        "dongxing_local_counterpart",
        "direct_transfer_boundary",
        "reward_weight_replay_boundary",
    }

    assert expected.issubset(rows)
    assert rows["calibration_effect"]["metrics"]["one_sided_p"] == pytest.approx(0.011963)
    assert rows["calibration_effect"]["metrics"]["two_sided_p"] == pytest.approx(0.023926)
    assert rows["calibration_effect"]["metrics"]["improvement_pct"] == pytest.approx(
        12.993898
    )
    assert (
        rows["direct_transfer_boundary"]["claim_strength"]
        == "structural_boundary"
    )
    assert ledger["overall_status"] == "supported_with_bounded_external_scope"
    assert "observational reward regularization" in ledger["required_boundaries"]
    assert "not definitive causal identification" in ledger["required_boundaries"]


def test_real_ledger_artifact_paths_exist():
    ledger = build_manuscript_evidence_ledger(REAL_AUDIT_PATH, repo_root=REPO_ROOT)

    missing = []
    for row in ledger["claims"]:
        for rel_path in row["artifact_paths"]:
            path = REPO_ROOT / rel_path
            if not path.exists():
                missing.append(f"{row['claim_id']} -> {rel_path}")

    assert missing == []


def test_markdown_renderer_includes_claim_ids_metrics_and_boundaries():
    ledger = build_manuscript_evidence_ledger(REAL_AUDIT_PATH, repo_root=REPO_ROOT)
    markdown = render_ledger_markdown(ledger)

    assert "# Paper 7 Manuscript Evidence Ledger" in markdown
    assert "bishan_learned_environment_e2e" in markdown
    assert "calibration_effect" in markdown
    assert "one_sided_p" in markdown
    assert "not direct Bishan-to-Dongxing policy transfer" in markdown
    assert "fixed-policy replay" in markdown


def test_write_ledger_outputs_machine_and_human_readable_files(tmp_path):
    json_out = tmp_path / "manuscript_evidence_ledger.json"
    md_out = tmp_path / "manuscript_evidence_ledger.md"

    written = write_manuscript_evidence_ledger(
        audit_path=REAL_AUDIT_PATH,
        json_out=json_out,
        md_out=md_out,
        repo_root=REPO_ROOT,
    )

    assert written["json"] == json_out
    assert written["markdown"] == md_out
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    markdown = md_out.read_text(encoding="utf-8")
    assert payload["claims"]
    assert "Dongxing" in markdown


def test_fixture_ledger_keeps_claim_strengths_bounded(tmp_path):
    audit_path = tmp_path / "audit.json"
    _write_json(
        audit_path,
        {
            "overall_status": "supported_with_bounded_external_scope",
            "evidence": {
                "transition_training": {
                    "status": "supported",
                    "model": {"path": "paper7/models/transition_model.pt"},
                    "history": {"path": "paper7/models/training_history.json"},
                    "final_val_obs_cosine": 0.999794,
                    "final_val_reward_mse": 1.096993,
                },
                "bishan_seed_chain": {
                    "status": "supported",
                    "seed_dir": "paper7/results/revision/seeds",
                    "n_paired_seeds": 15,
                    "no_cal_mean": -0.975683,
                    "with_cal_mean": -1.102462,
                    "improvement_pct": 12.993898,
                    "paired_slope_test": {
                        "one_sided_p": 0.011963,
                        "two_sided_p": 0.023926,
                    },
                },
                "reward_scaling_comparator": {
                    "path": "paper7/results/revision/reward_scaling_comparator.json",
                    "best_scale": 0.2,
                    "pre_specified_scale": 0.185,
                    "pre_vs_best_relative_gap_pct": 3.12135,
                    "pre_vs_unscaled_slope_gain_pct": 22.144879,
                },
                "alpha_grid": {
                    "grid_path": "paper7/results/revision/alpha_grid/grid_results.json",
                    "n_runs": 40,
                    "n_alphas": 8,
                },
                "transition_rollout": {
                    "path": "paper7/results/revision/transition_rollout_diagnostics.json",
                    "horizon_100_mask_agreement": 0.997384,
                    "horizon_100_reward_mae": 0.234012,
                },
                "policy_induced_diagnostics": {
                    "path": "paper7/results/revision/policy_induced_diagnostics_15seed.json",
                    "n_policy_episodes": 15,
                    "policy_induced_mask_agreement_mean": 0.997629,
                    "policy_induced_support_distance_mean": 0.012601,
                },
                "calibration": {
                    "calibration_path": "paper7/results/causal_calibration.json",
                    "sensitivity_path": "paper7/results/revision/causal_sensitivity_diagnostics.json",
                    "calibration_factor": 0.185005,
                    "interpretation": "observational treatment-effect-informed reward regularization; not definitive causal identification",
                },
                "planning_significance": {
                    "path": "paper7/results/revision/planning_significance_audit.json",
                    "contiguity_delta_with_minus_no_mean": -0.001977,
                    "baimu_count_delta_with_minus_no_mean": -0.2,
                    "baimu_area_delta_with_minus_no_mean": -67.208419,
                },
                "reward_weight_sensitivity": {
                    "path": "paper7/results/full_rigor/reward_weight_sensitivity.json",
                    "n_weight_settings": 14,
                    "policy_retraining_under_all_weights": False,
                },
                "bishan_non_learning_baselines": {
                    "path": "paper7/results/revision/bishan_strong_baselines.json"
                },
                "dongxing_full_baselines": {
                    "path": "paper7/results/full_rigor/dongxing_full_baselines.json",
                    "status": "supported_as_full_real_environment_baselines",
                },
                "dongxing_full_learned_policy": {
                    "path": "paper7/results/full_rigor/dongxing_full_learned_policy.json",
                    "n_eval_seeds": 10,
                },
                "dongxing_mbrl_results": {
                    "path": "paper7/results/full_rigor/dongxing_mbrl_results.json",
                    "status": "supported_as_local_dongxing_mbrl_results",
                    "multi_step_mbrl_planning_tested": True,
                },
                "dongxing_multistep_mbrl_policy": {
                    "path": "paper7/results/full_rigor/dongxing_multistep_mbrl_policy.json",
                    "real_eval_reward_mean": 61.287306,
                    "real_eval_slope_change_pct_mean": -1.882392,
                },
                "transfer_finetune_results": {
                    "path": "paper7/results/full_rigor/transfer_finetune_results.json",
                    "status": "structurally_invalid_for_direct_policy_transfer",
                    "dimension_mismatch": {
                        "observation_dim_match": False,
                        "action_dim_match": False,
                    },
                },
            },
        },
    )

    ledger = build_manuscript_evidence_ledger(audit_path, repo_root=REPO_ROOT)
    rows = {row["claim_id"]: row for row in ledger["claims"]}

    assert rows["calibration_effect"]["claim_strength"] == "supported_bounded"
    assert rows["dongxing_local_counterpart"]["claim_strength"] == "supported_bounded"
    assert rows["reward_weight_replay_boundary"]["metrics"][
        "policy_retraining_under_all_weights"
    ] is False
```

- [ ] **Step 2: Run the new test and confirm the expected failure**

Run:

```powershell
python -m pytest tests/test_manuscript_evidence_ledger.py -q
```

Expected:

- FAIL with `ModuleNotFoundError: No module named 'paper7.manuscript_evidence_ledger'`.

### Task 2: Implement The Evidence Ledger Generator

**Files:**
- Create: `paper7/manuscript_evidence_ledger.py`
- Test: `tests/test_manuscript_evidence_ledger.py`

- [ ] **Step 1: Add the implementation**

Create `paper7/manuscript_evidence_ledger.py` with exactly this content:

```python
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
```

- [ ] **Step 2: Run the evidence ledger tests**

Run:

```powershell
python -m pytest tests/test_manuscript_evidence_ledger.py -q
```

Expected:

- PASS.

- [ ] **Step 3: Commit the generator and tests**

Run:

```powershell
git add paper7/manuscript_evidence_ledger.py tests/test_manuscript_evidence_ledger.py
git commit -m "test: add paper7 manuscript evidence ledger"
```

Expected:

- Commit succeeds.

### Task 3: Generate The Manuscript Evidence Ledger Artifacts

**Files:**
- Create: `paper7/results/full_rigor/manuscript_evidence_ledger.json`
- Create: `paper7/results/full_rigor/manuscript_evidence_ledger.md`
- Test: `tests/test_manuscript_evidence_ledger.py`

- [ ] **Step 1: Generate the ledger artifacts**

Run:

```powershell
python -m paper7.manuscript_evidence_ledger
```

Expected output includes:

```text
paper7\results\full_rigor\manuscript_evidence_ledger.json
paper7\results\full_rigor\manuscript_evidence_ledger.md
```

- [ ] **Step 2: Check the generated ledger contents**

Run:

```powershell
rg -n "calibration_effect|dongxing_local_counterpart|direct_transfer_boundary|fixed-policy replay" paper7\results\full_rigor\manuscript_evidence_ledger.md
```

Expected:

- At least one match for each searched phrase.

- [ ] **Step 3: Re-run ledger tests**

Run:

```powershell
python -m pytest tests/test_manuscript_evidence_ledger.py -q
```

Expected:

- PASS.

- [ ] **Step 4: Commit the generated ledgers**

Run:

```powershell
git add paper7/results/full_rigor/manuscript_evidence_ledger.json paper7/results/full_rigor/manuscript_evidence_ledger.md
git commit -m "docs: add paper7 manuscript evidence ledger"
```

Expected:

- Commit succeeds.

### Task 4: Harden Manuscript Claim-Consistency Tests

**Files:**
- Modify: `tests/test_manuscript_claim_consistency.py`
- Read: `paper7/results/full_rigor/manuscript_evidence_ledger.json`
- Read: `submission/ceus/01_main_document_anonymous/manuscript.tex`
- Read: `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex`
- Read: `submission/ceus/06_latex_source_editable/manuscript_signed.tex`
- Read: `submission/ceus/03_highlights/highlights.txt`
- Read: `submission/ceus/04_cover_letter/cover_letter.txt`

- [ ] **Step 1: Replace the existing manuscript consistency test**

Replace `tests/test_manuscript_claim_consistency.py` with exactly this content:

```python
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "paper7" / "results" / "revision" / "end_to_end_validation.json"
LEDGER_PATH = (
    REPO_ROOT
    / "paper7"
    / "results"
    / "full_rigor"
    / "manuscript_evidence_ledger.json"
)
MANUSCRIPT_PATHS = [
    REPO_ROOT / "submission" / "ceus" / "01_main_document_anonymous" / "manuscript.tex",
    REPO_ROOT
    / "submission"
    / "ceus"
    / "06_latex_source_editable"
    / "manuscript_anonymous_copy.tex",
    REPO_ROOT
    / "submission"
    / "ceus"
    / "06_latex_source_editable"
    / "manuscript_signed.tex",
]
HIGHLIGHTS_PATH = REPO_ROOT / "submission" / "ceus" / "03_highlights" / "highlights.txt"
COVER_LETTER_PATH = (
    REPO_ROOT / "submission" / "ceus" / "04_cover_letter" / "cover_letter.txt"
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def _read_all_manuscripts() -> dict[Path, str]:
    return {path: path.read_text(encoding="utf-8") for path in MANUSCRIPT_PATHS}


def test_manuscript_uses_audited_paired_calibration_p_values():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    paired = audit["evidence"]["bishan_seed_chain"]["paired_slope_test"]
    one_sided = f"p={paired['one_sided_p']:.3f}"
    two_sided = f"p={paired['two_sided_p']:.3f}"

    for path, manuscript in _read_all_manuscripts().items():
        compact = _compact(manuscript)
        assert "0.004" not in manuscript, path
        assert one_sided in compact, path
        assert two_sided in compact, path
        assert "Mann-Whitney" not in manuscript, path
        assert "Mann--Whitney" not in manuscript, path


def test_manuscript_keeps_review_boundaries_visible():
    required = [
        "observational reward regularization",
        "not definitive causal identification",
        "not direct bishan-to-dongxing policy transfer",
        "descriptive",
        "model-free",
        "real-environment evaluation",
        "fixed-policy",
    ]

    for path, manuscript in _read_all_manuscripts().items():
        lower = manuscript.lower()
        for phrase in required:
            assert phrase in lower, f"{phrase!r} missing from {path}"


def test_manuscript_does_not_reintroduce_forbidden_overclaims():
    forbidden_patterns = [
        r"universal generalization across counties",
        r"formal superiority over all model-free",
        r"direct transfer of bishan policies to dongxing",
        r"definitive causal identification of reward effects",
        r"replace[s]? final real-environment evaluation",
    ]

    for path, manuscript in _read_all_manuscripts().items():
        lower = manuscript.lower()
        for pattern in forbidden_patterns:
            assert re.search(pattern, lower) is None, f"{pattern!r} found in {path}"


def test_generated_ledger_supports_manuscript_claims():
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    claim_ids = {row["claim_id"] for row in ledger["claims"]}

    assert "calibration_effect" in claim_ids
    assert "dongxing_local_counterpart" in claim_ids
    assert "direct_transfer_boundary" in claim_ids
    assert "reward_weight_replay_boundary" in claim_ids
    assert "not definitive causal identification" in ledger["required_boundaries"]
    assert "not direct Bishan-to-Dongxing policy transfer" in ledger[
        "required_boundaries"
    ]


def test_highlights_and_cover_letter_use_bounded_full_rigor_framing():
    highlights = HIGHLIGHTS_PATH.read_text(encoding="utf-8").lower()
    cover = COVER_LETTER_PATH.read_text(encoding="utf-8").lower()

    assert "local counterpart" in highlights
    assert "transfer" not in highlights
    assert "descriptive comparison" in cover
    assert "direct cross-county transfer" in cover
    assert "observational" in cover
```

- [ ] **Step 2: Run the hardened claim tests and confirm the expected failure**

Run:

```powershell
python -m pytest tests/test_manuscript_claim_consistency.py -q
```

Expected:

- FAIL because `submission/ceus/06_latex_source_editable/manuscript_signed.tex` and `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex` still contain stale `p=0.004` wording.
- The cover-letter bounded-framing test may also fail until Task 5.

### Task 5: Revise Manuscript Sources, Highlights, And Cover Letter

**Files:**
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex`
- Modify: `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex`
- Modify: `submission/ceus/06_latex_source_editable/manuscript_signed.tex`
- Modify: `submission/ceus/03_highlights/highlights.txt`
- Modify: `submission/ceus/04_cover_letter/cover_letter.txt`
- Test: `tests/test_manuscript_claim_consistency.py`

- [ ] **Step 1: Update the audit paragraph in the anonymous manuscript**

In `submission/ceus/01_main_document_anonymous/manuscript.tex`, replace the paragraph beginning:

```latex
\textbf{End-to-end evidence audit.} To make the verification boundary explicit,
```

with this paragraph:

```latex
\textbf{End-to-end evidence audit and manuscript ledger.} To make the verification boundary explicit, we added an executable audit script, \path{paper7/end_to_end_validation.py}, which reads the recorded assets and result files rather than retraining the expensive RL models. The audit checks the six trajectory assets, transition-model checkpoint and training history, 15 paired calibrated/uncalibrated real-environment evaluation seeds, the 40-run reward-scaling grid, the reward-scaling comparator, recorded-action rollout diagnostics, policy-induced learned-vs-real diagnostics, the trajectory-source ablation, planning-significance audit, calibration sensitivity diagnostics, Bishan non-learning baselines, and the Dongxing full-reward counterpart chain including full baselines, local learned policy, transition diagnostics, one-step model-based action selection, held-out scoring optimization, multi-step learned-environment policy optimization, and transfer-mismatch evidence. Its output, \path{paper7/results/revision/end_to_end_validation.json}, classifies each manuscript claim by evidence level (Table~\ref{tab:e2e_scope}). We additionally export a manuscript-facing evidence ledger, \path{paper7/results/full_rigor/manuscript_evidence_ledger.json}, with a readable companion file, \path{paper7/results/full_rigor/manuscript_evidence_ledger.md}. This ledger maps each headline claim to artifact paths, key metrics, claim strength, and required boundary wording. The audit and ledger verify consistency of the stored data-to-result chain used in the paper; they do not replace full retraining when computational budgets allow.
```

- [ ] **Step 2: Add fixed-policy replay boundary wording if absent**

In the `\subsection{Limitations}` block, ensure the reward-weight sensitivity boundary appears as a separate sentence. If it is absent, add this sentence after the paragraph beginning `\textbf{Global calibration factor.}`:

```latex
\textbf{Reward-weight replay.} The reward-weight sensitivity analysis replays fixed action sequences under alternative scalar weights. It is useful for auditing planning-preference trade-offs, but it is fixed-policy replay rather than evidence that retrained policies are robust under every possible reward specification.
```

- [ ] **Step 3: Sync the anonymous editable source**

Run:

```powershell
Copy-Item -LiteralPath submission\ceus\01_main_document_anonymous\manuscript.tex -Destination submission\ceus\06_latex_source_editable\manuscript_anonymous_copy.tex
```

Expected:

- `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex` matches the anonymous manuscript source.

- [ ] **Step 4: Update the signed editable source while preserving author information**

In `submission/ceus/06_latex_source_editable/manuscript_signed.tex`, replace every stale calibration-significance phrase with the audited wording:

Replace:

```latex
a 13.0\% paired improvement ($p=0.004$).
```

with:

```latex
a 13.0\% paired improvement under an exact paired sign-flip test (one-sided $p=0.012$; two-sided $p=0.024$).
```

Replace:

```latex
improving downstream policy quality by 13.0\% ($p = 0.004$).
```

with:

```latex
improving downstream policy quality by 13.0\% under an exact paired sign-flip test (one-sided $p=0.012$; two-sided $p=0.024$).
```

Also apply the same audit paragraph replacement from Step 1 to the signed source.

- [ ] **Step 5: Revise highlights**

Replace `submission/ceus/03_highlights/highlights.txt` with exactly:

```text
* Learned environments enable CPU-based farmland planning reinforcement learning.
* Observational reward calibration reduces learned-reward exploitation.
* Policy-induced diagnostics support real-environment evaluation boundaries.
* Dongxing adds a full-reward local counterpart with learned-env evidence.
* Evidence ledger links headline claims to verified result artifacts.
```

- [ ] **Step 6: Revise cover letter claim framing**

In `submission/ceus/04_cover_letter/cover_letter.txt`, replace the numbered list item 4:

```text
4. A 15-seed real-environment evaluation showing that calibrated model-based policies improve over the uncalibrated learned-environment policy and provide descriptive comparison with GPU-trained model-free baselines.
```

with:

```text
4. A 15-seed real-environment evaluation showing that calibrated model-based policies improve over the uncalibrated learned-environment policy under an exact paired sign-flip test, with descriptive comparison to GPU-trained model-free baselines.
```

Replace list item 5:

```text
5. Policy-induced learned-vs-real diagnostics, a trajectory-source robustness ablation, and a Dongxing District full-reward local counterpart experiment covering full baselines, a local learned policy, one-step model-based action selection, held-out scoring optimization, and a multi-step learned-environment policy, with explicit limits on direct cross-county transfer claims.
```

with:

```text
5. Policy-induced learned-vs-real diagnostics, a trajectory-source robustness ablation, and a Dongxing District full-reward local counterpart experiment covering full baselines, a local learned policy, one-step model-based action selection, held-out scoring optimization, and a multi-step learned-environment policy, with explicit limits on direct cross-county transfer claims.
```

The replacement text is intentionally the same for item 5; this step verifies the phrase `direct cross-county transfer` remains visible for the claim-consistency test.

- [ ] **Step 7: Run claim consistency tests**

Run:

```powershell
python -m pytest tests/test_manuscript_claim_consistency.py -q
```

Expected:

- PASS.

- [ ] **Step 8: Commit the manuscript and editorial updates**

Run:

```powershell
git add tests/test_manuscript_claim_consistency.py submission/ceus/01_main_document_anonymous/manuscript.tex submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex submission/ceus/06_latex_source_editable/manuscript_signed.tex submission/ceus/03_highlights/highlights.txt submission/ceus/04_cover_letter/cover_letter.txt
git commit -m "docs: align paper7 manuscript with evidence ledger"
```

Expected:

- Commit succeeds.

### Task 6: Rebuild Manuscript PDFs And Source Package

**Files:**
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.pdf`
- Modify: `submission/ceus/06_latex_source_editable/manuscript_signed.pdf`
- Modify: `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.pdf`
- Modify: `submission/ceus/CEUS_paper7_latex_source_anonymous.zip`

- [ ] **Step 1: Compile the anonymous manuscript**

Run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
```

from:

```text
submission\ceus\01_main_document_anonymous
```

Expected:

- Exit code `0`.
- `manuscript.pdf` is updated.

- [ ] **Step 2: Compile the signed manuscript**

Run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error manuscript_signed.tex
```

from:

```text
submission\ceus\06_latex_source_editable
```

Expected:

- Exit code `0`.
- `manuscript_signed.pdf` is updated.

- [ ] **Step 3: Compile the anonymous editable copy**

Run:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error manuscript_anonymous_copy.tex
```

from:

```text
submission\ceus\06_latex_source_editable
```

Expected:

- Exit code `0`.
- `manuscript_anonymous_copy.pdf` is updated.

- [ ] **Step 4: Refresh the anonymous source zip**

Run:

```powershell
Compress-Archive -Path submission\ceus\01_main_document_anonymous\manuscript.tex,submission\ceus\05_figures\figure_1_pipeline.pdf -DestinationPath submission\ceus\CEUS_paper7_latex_source_anonymous.zip -Force
```

Expected:

- `submission\ceus\CEUS_paper7_latex_source_anonymous.zip` is updated.

- [ ] **Step 5: Commit rebuilt submission artifacts**

Run:

```powershell
git add submission/ceus/01_main_document_anonymous/manuscript.pdf submission/ceus/06_latex_source_editable/manuscript_signed.pdf submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.pdf submission/ceus/CEUS_paper7_latex_source_anonymous.zip
git commit -m "docs: rebuild paper7 CEUS manuscript artifacts"
```

Expected:

- Commit succeeds if the repository tracks these generated artifacts.
- If Git reports no staged PDF or zip changes, skip the commit and record that artifacts were already current.

### Task 7: Run Focused And Full Verification

**Files:**
- Test: `tests/test_manuscript_evidence_ledger.py`
- Test: `tests/test_manuscript_claim_consistency.py`
- Test: `tests/test_end_to_end_validation.py`
- Test: `tests/test_transition_rollout_diagnostics.py`
- Test: `tests/test_policy_induced_diagnostics.py`
- Test: `tests/test_reward_components.py`
- Test: `tests/test_reward_weight_sensitivity.py`
- Test: `tests/test_dongxing_full_baselines.py`
- Test: `tests/test_dongxing_full_rigor_summaries.py`
- Test: full test suite

- [ ] **Step 1: Run focused verification**

Run:

```powershell
python -m pytest tests/test_manuscript_evidence_ledger.py tests/test_manuscript_claim_consistency.py tests/test_end_to_end_validation.py tests/test_transition_rollout_diagnostics.py tests/test_policy_induced_diagnostics.py tests/test_reward_components.py tests/test_reward_weight_sensitivity.py tests/test_dongxing_full_baselines.py tests/test_dongxing_full_rigor_summaries.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Search for stale or overstated text**

Run:

```powershell
rg -n "0\\.004|Mann-Whitney|Mann--Whitney|universal generalization|formal superiority|direct transfer of Bishan policies|definitive causal identification of reward effects" submission\ceus
```

Expected:

- No matches in manuscript/editorial submission files. The generated ledger may list forbidden phrases explicitly as a guardrail, so this smoke search intentionally does not scan the ledger Markdown.

- [ ] **Step 3: Run the full pytest suite**

Run:

```powershell
python -m pytest -q
```

Expected:

- PASS.
- On the current Windows Python 3.14 environment, pytest may print a torch DLL access-violation stack after reporting passed tests. Treat exit code `0` as the pass/fail gate and report the environment warning in the final summary.

- [ ] **Step 4: Review final git diff**

Run:

```powershell
git diff --stat
git status --short --branch
git log --oneline -n 8
```

Expected:

- Working tree is clean.
- Recent commits include:
  - `test: add paper7 manuscript evidence ledger`
  - `docs: add paper7 manuscript evidence ledger`
  - `docs: align paper7 manuscript with evidence ledger`
  - `docs: rebuild paper7 CEUS manuscript artifacts` when generated artifacts changed.

---

## Self-Review

- Spec coverage: Tasks cover the evidence ledger, ledger outputs, manuscript claim tests, manuscript source synchronization, highlights, cover letter, PDF rebuild, source zip refresh, and layered verification.
- Incomplete-marker scan: no incomplete task markers or unspecified implementation steps remain.
- Type consistency: the plan consistently uses `build_manuscript_evidence_ledger`, `render_ledger_markdown`, and `write_manuscript_evidence_ledger` across tests and implementation.
- Scope check: the plan does not add a new RL algorithm, direct policy transfer, or reward-weight retraining. It implements the approved ledger-driven manuscript integration route.

