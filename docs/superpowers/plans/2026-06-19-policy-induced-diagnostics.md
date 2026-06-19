# Policy-Induced Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Paper 7 policy-induced learned-vs-real diagnostics from 3 calibrated checkpoints to all 15 calibrated policy seeds, then update the evidence audit only after validation.

**Architecture:** Add pure summary/validation helpers to the existing diagnostic script so they can be tested without running expensive environments. Reuse the existing transition model, trajectory support, and calibrated policy checkpoints for the full rollout.

**Tech Stack:** Python, pytest, NumPy, PyTorch, Stable-Baselines3/sb3-contrib, existing Paper 7 county environment.

## Execution Status

Completed on 2026-06-19. The 15-seed diagnostic was generated at `paper7/results/revision/policy_induced_diagnostics_15seed.json`, validated with all threshold checks passing, incorporated into the end-to-end evidence audit, and reflected in the anonymous CEUS manuscript only after validation. Final verification used targeted policy/audit tests, full `pytest`, two `pdflatex` passes, LaTeX log scanning, anonymous-source grep, and source-zip content matching.

---

### Task 1: Add Diagnostic Summary And Validation Tests

**Files:**
- Modify: `tests/test_policy_induced_diagnostics.py`
- Modify: `paper7/policy_induced_diagnostics.py`

- [ ] **Step 1: Write failing tests**

Create tests that import helper functions from `paper7.policy_induced_diagnostics`:

```python
import math

import pytest

from paper7.policy_induced_diagnostics import (
    summarize_policy_induced_episodes,
    validate_policy_induced_payload,
)


def _episode(seed, mask=0.998, support=0.01, raw=0.6, cal=0.11, slope=-1.0):
    return {
        "summary": {
            "seed": seed,
            "n_steps": 100,
            "global_mae_mean": 0.05 + seed * 0.001,
            "reward_mae_mean": raw,
            "calibrated_reward_mae_mean": cal,
            "mask_agreement_mean": mask,
            "support_distance_mean": support,
            "support_distance_q95": support + 0.005,
            "final_real_slope_change_pct": slope,
            "selected_block_mae_mean": 0.075,
            "all_block_mae_mean": 0.0027,
        },
        "step_metrics_head": [],
    }


def test_summary_reports_count_and_distribution_stats():
    episodes = [_episode(seed, slope=-1.0 - 0.01 * seed) for seed in range(3)]
    summary = summarize_policy_induced_episodes(episodes)
    assert summary["n_episodes"] == 3
    assert summary["mask_agreement_mean_mean"] == pytest.approx(0.998)
    assert summary["final_real_slope_change_pct_min"] == pytest.approx(-1.02)
    assert summary["final_real_slope_change_pct_max"] == pytest.approx(-1.0)
    assert summary["support_distance_q95_mean"] == pytest.approx(0.015)


def test_validation_rejects_missing_policy_seeds():
    payload = {"episodes": [_episode(seed) for seed in range(14)]}
    with pytest.raises(ValueError, match="Expected 15 episodes"):
        validate_policy_induced_payload(payload, expected_seeds=list(range(15)))


def test_validation_rejects_nan_metrics():
    episodes = [_episode(seed) for seed in range(15)]
    episodes[4]["summary"]["support_distance_mean"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_policy_induced_payload({"episodes": episodes}, expected_seeds=list(range(15)))


def test_validation_accepts_complete_15_seed_payload():
    episodes = [_episode(seed) for seed in range(15)]
    payload = {"episodes": episodes, "aggregate": summarize_policy_induced_episodes(episodes)}
    validated = validate_policy_induced_payload(payload, expected_seeds=list(range(15)))
    assert validated["n_episodes"] == 15
    assert validated["passes_mask_agreement_threshold"] is True
    assert validated["passes_support_distance_threshold"] is True
    assert validated["passes_reward_calibration_check"] is True
```

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/test_policy_induced_diagnostics.py -q`

Expected: fail because `summarize_policy_induced_episodes` and `validate_policy_induced_payload` are not exported.

- [ ] **Step 3: Implement helpers**

Add pure functions to `paper7/policy_induced_diagnostics.py`:

```python
def summarize_policy_induced_episodes(episodes):
    # compute n, mean, min, max for key summary fields
    # preserve existing aggregate key names


def validate_policy_induced_payload(payload, expected_seeds, mask_threshold=0.995, support_q95_threshold=0.05):
    # ensure expected episode count, seed coverage, finite metrics, threshold booleans
```

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m pytest tests/test_policy_induced_diagnostics.py -q`

Expected: all tests pass.

### Task 2: Expand The Diagnostic Runner To 15 Checkpoints

**Files:**
- Modify: `paper7/policy_induced_diagnostics.py`
- Output: `paper7/results/revision/policy_induced_diagnostics_15seed.json`

- [ ] **Step 1: Inspect existing CLI**

Read `paper7/policy_induced_diagnostics.py` and preserve the current default behavior unless a new argument is explicitly provided.

- [ ] **Step 2: Add seed/checkpoint arguments**

Add CLI options:

```text
--policy-dir paper7/results/revision/seeds
--label with_cal
--seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
--output paper7/results/revision/policy_induced_diagnostics_15seed.json
```

Construct checkpoint paths as `{label}_model_seed{seed}.zip`.

- [ ] **Step 3: Validate output after writing**

After writing JSON, call `validate_policy_induced_payload` with the requested seeds and include the validation result in the output under `"validation"`.

- [ ] **Step 4: Run targeted tests**

Run: `python -m pytest tests/test_policy_induced_diagnostics.py -q`

Expected: all tests pass.

### Task 3: Run The 15-Seed Diagnostic And Evidence Audit

**Files:**
- Output: `paper7/results/revision/policy_induced_diagnostics_15seed.json`
- Modify: `paper7/results/revision/end_to_end_validation.json` if the audit script supports the new file
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex` only after successful diagnostic validation
- Modify: `submission/ceus/06_latex_source_editable/manuscript_signed.tex` only after successful diagnostic validation

- [ ] **Step 1: Run full diagnostic**

Run:

```powershell
python -m paper7.policy_induced_diagnostics --policy-dir paper7/results/revision/seeds --label with_cal --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 --output paper7/results/revision/policy_induced_diagnostics_15seed.json
```

Expected: output JSON contains 15 episodes and `"validation"` indicates the threshold checks.

- [ ] **Step 2: Run preflight/evidence checks**

Run: `python scripts/paper10/preflight_submission_checks.py` only if it applies to this repository. Otherwise run:

```powershell
python -m pytest tests -q --basetmp .\.pytest-tmp-policy-induced
```

Expected: tests pass.

- [ ] **Step 3: Update manuscript only if warranted**

If validation passes, change the manuscript wording from "three calibrated policies/checkpoints" to "15 calibrated policy seeds" and update aggregate values. If validation fails, do not strengthen the claim; add or preserve limitation wording.

- [ ] **Step 4: Compile and verify CEUS manuscript if manuscript changed**

Run `pdflatex` twice in `submission/ceus/01_main_document_anonymous`.

Expected: PDF builds without undefined references or fatal errors.

- [ ] **Step 5: Commit**

Commit only the design, plan, code/tests, validated result JSON, and warranted manuscript/audit updates.
