# Paper 7 Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden Paper 7's code and experiment artifacts against the strict reviewer concerns before further manuscript revision.

**Architecture:** Keep the current experiment pipeline intact and add small auditable utilities around it. The hardening work should turn reviewer-facing claims into explicit machine-readable evidence: paired statistical tests for paired seeds, reward-spec metadata, and synchronized reproducibility documentation.

**Tech Stack:** Python standard library, existing Paper 7 modules, pytest, JSON result artifacts, Markdown documentation.

---

### Task 1: Paired Calibration Statistics

**Files:**
- Modify: `paper7/planning_significance_audit.py`
- Modify: `paper7/end_to_end_validation.py`
- Modify: `paper7/batch_revision.py`
- Test: `tests/test_planning_significance_audit.py`
- Test: `tests/test_end_to_end_validation.py`

- [ ] **Step 1: Add failing tests for paired seed statistics**

Add tests that require `paired_calibration_effects` to report:
- `paired_test`: `exact_sign_flip`
- one-sided p-value for paired slope deltas
- paired win/loss/tie counts
- an explicit interpretation boundary that the paired design invalidates independent-sample Mann-Whitney wording

- [ ] **Step 2: Run the focused tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_planning_significance_audit.py tests/test_end_to_end_validation.py -q
```

Expected: failure because the current report has paired means only.

- [ ] **Step 3: Implement exact paired sign-flip logic**

Implement in `paper7/planning_significance_audit.py` using only the standard library:
- compute observed paired delta mean
- enumerate all sign flips for up to 20 non-zero deltas
- report one-sided and two-sided p-values
- count slope wins where calibrated is lower than uncalibrated

- [ ] **Step 4: Wire paired statistics into summaries**

Expose the same paired-test fields from:
- `summarize_seed_evaluations`
- `summarize_planning_significance`
- the `calibration_scope` claim classification if useful

- [ ] **Step 5: Update `batch_revision.py` summary output**

Replace Mann-Whitney reporting with the paired exact sign-flip helper so future regenerated summaries cannot reintroduce the independent-sample test.

- [ ] **Step 6: Re-run focused tests**

Run:

```powershell
python -m pytest tests/test_planning_significance_audit.py tests/test_end_to_end_validation.py -q
```

Expected: all focused tests pass.

### Task 2: Reward Specification Export

**Files:**
- Modify: `paper7/reward_components.py`
- Modify: `paper7/reward_weight_sensitivity.py`
- Modify: `paper7/end_to_end_validation.py`
- Test: `tests/test_reward_components.py`
- Test: `tests/test_reward_weight_sensitivity.py`
- Test: `tests/test_end_to_end_validation.py`

- [ ] **Step 1: Add failing tests for reward-spec metadata**

Require a stable exported reward specification containing:
- canonical equation terms
- default weights
- sign convention for slope, contiguity, baimu area, baimu count, no-swap penalty
- a short boundary that weight replay is fixed-policy sensitivity, not retraining

- [ ] **Step 2: Run reward tests and confirm failure**

Run:

```powershell
python -m pytest tests/test_reward_components.py tests/test_reward_weight_sensitivity.py tests/test_end_to_end_validation.py -q
```

Expected: failure because no canonical reward-spec export exists.

- [ ] **Step 3: Implement reward-spec helper**

Add `reward_specification()` in `paper7/reward_components.py` and include it in reward sensitivity output.

- [ ] **Step 4: Expose reward spec in the end-to-end audit**

Add `reward_specification` to `evidence["reward_weight_sensitivity"]` and add a top-level reward-spec claim entry if the artifact exists.

- [ ] **Step 5: Re-run reward tests**

Run:

```powershell
python -m pytest tests/test_reward_components.py tests/test_reward_weight_sensitivity.py tests/test_end_to_end_validation.py -q
```

Expected: all focused tests pass.

### Task 3: Regenerate Auditable Artifacts

**Files:**
- Update generated JSON:
  - `paper7/results/revision/planning_significance_audit.json`
  - `paper7/results/revision/end_to_end_validation.json`
  - `paper7/results/full_rigor/reward_weight_sensitivity.json`
  - `paper7/results/full_rigor/reward_pareto_front.json`
- Modify: `REPRODUCIBILITY.md`

- [ ] **Step 1: Regenerate planning and reward artifacts**

Run:

```powershell
python paper7/planning_significance_audit.py
python paper7/reward_weight_sensitivity.py
python paper7/end_to_end_validation.py --out paper7/results/revision/end_to_end_validation.json
```

Expected: JSON files include paired-test and reward-spec metadata.

- [ ] **Step 2: Sync reproducibility documentation**

Update `REPRODUCIBILITY.md` to reflect:
- transition training history currently records 50 epochs
- policy-induced diagnostics use 15 calibrated policy seeds
- Dongxing now includes full-reward local counterpart evidence, not only dynamic non-RL feasibility
- stored audits verify the result chain and do not imply full fresh retraining

- [ ] **Step 3: Run final focused verification**

Run:

```powershell
python -m pytest tests/test_planning_significance_audit.py tests/test_reward_components.py tests/test_reward_weight_sensitivity.py tests/test_end_to_end_validation.py -q
```

Expected: all focused tests pass.

### Task 4: Worktree Review

**Files:**
- Inspect: `git diff --stat`
- Inspect: `git diff -- paper7 tests REPRODUCIBILITY.md docs/superpowers/plans/2026-06-25-paper7-review-hardening.md`

- [ ] **Step 1: Check generated changes are scoped**

Run:

```powershell
git status --short
git diff --stat
```

Expected: only Paper 7 hardening code, tests, result JSON, reproducibility docs, and this plan changed.

- [ ] **Step 2: Summarize residual scientific risk**

Report whether the paired p-value strengthens or weakens the previous significance wording and list the remaining experiment gaps before manuscript editing resumes.
