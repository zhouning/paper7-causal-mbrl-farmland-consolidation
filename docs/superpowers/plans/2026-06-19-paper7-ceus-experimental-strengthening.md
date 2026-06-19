# Paper 7 CEUS Experimental Strengthening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add substantive Paper 7 evidence for CEUS by generating new diagnostics, a Dongxing slope-only RL-lite experiment, validation audit updates, and manuscript/package revisions only after experiments pass.

**Architecture:** Keep experiments and manuscript edits separated. New helper scripts produce recorded JSON artifacts under `paper7/results/revision/`; `paper7/end_to_end_validation.py` classifies evidence scope; the CEUS manuscript consumes only validated artifacts.

**Tech Stack:** Python, NumPy, Gymnasium, existing Stable-Baselines/MaskablePPO stack if available, pytest, LaTeX/pdflatex.

---

### File Structure

- Create: `paper7/reward_scaling_comparator.py`  
  Summarizes alpha-grid reward scaling and compares observational `alpha=0.185` with heuristic alternatives.
- Create: `paper7/planning_significance_audit.py`  
  Summarizes planning outcomes and selected-block concentration from existing Bishan policy evaluation files.
- Create: `paper7/dongxing_rl_lite.py`  
  Defines a Dongxing slope-only Gymnasium masked environment, a lightweight learned policy fallback, training/evaluation helpers, and CLI.
- Modify: `paper7/transition_rollout_diagnostics.py`  
  Add distribution summaries and feature-group metrics without changing existing CLI defaults.
- Modify: `paper7/end_to_end_validation.py`  
  Add summaries for reward scaling comparator, planning significance audit, and Dongxing RL-lite evidence scope.
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex`  
  Update claims, methods, experiments, discussion, and limitations after result artifacts exist.
- Modify: `submission/ceus/06_latex_source_editable/manuscript_signed.tex`  
  Mirror manuscript evidence and claim-boundary changes.
- Modify: `submission/ceus/README_CEUS_submission_package.md`  
  Update evidence/package notes if manuscript/package contents change.
- Modify: `submission/ceus/CEUS_paper7_latex_source_anonymous.zip`  
  Rebuild after LaTeX verification.
- Test: `tests/test_reward_scaling_comparator.py`
- Test: `tests/test_planning_significance_audit.py`
- Test: `tests/test_dongxing_rl_lite.py`
- Test: `tests/test_transition_rollout_diagnostics.py`
- Test: `tests/test_end_to_end_validation.py`

---

### Task 1: Reward-scaling comparator

**Files:**
- Create: `paper7/reward_scaling_comparator.py`
- Create: `tests/test_reward_scaling_comparator.py`

- [ ] **Step 1: Write failing tests**

Create tests for pure helpers:

```python
from paper7.reward_scaling_comparator import (
    compare_reward_scales,
    summarize_by_scale,
)


def test_summarize_by_scale_groups_seed_rows_and_prefers_lower_slope():
    rows = [
        {"reward_scale": 0.1, "seed": 0, "slope_change_pct": -1.0, "reward_real": 10.0},
        {"reward_scale": 0.1, "seed": 1, "slope_change_pct": -1.2, "reward_real": 12.0},
        {"reward_scale": 1.0, "seed": 0, "slope_change_pct": -0.5, "reward_real": 8.0},
    ]

    summary = summarize_by_scale(rows)

    assert summary["0.100"]["n"] == 2
    assert summary["0.100"]["slope_change_pct_mean"] == -1.1
    assert summary["1.000"]["slope_change_pct_mean"] == -0.5


def test_compare_reward_scales_reports_pre_specified_gap_and_rank():
    rows = [
        {"reward_scale": 0.1, "seed": 0, "slope_change_pct": -1.0, "reward_real": 10.0},
        {"reward_scale": 0.185, "seed": 0, "slope_change_pct": -1.2, "reward_real": 11.0},
        {"reward_scale": 0.2, "seed": 0, "slope_change_pct": -1.3, "reward_real": 12.0},
        {"reward_scale": 1.0, "seed": 0, "slope_change_pct": -0.6, "reward_real": 8.0},
    ]

    comparison = compare_reward_scales(rows, pre_specified_alpha=0.185)

    assert comparison["best_scale"] == 0.2
    assert comparison["pre_specified_scale"] == 0.185
    assert comparison["pre_specified_rank_by_slope"] == 2
    assert comparison["unscaled_scale"] == 1.0
    assert comparison["pre_vs_unscaled_slope_gain_pct"] > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_reward_scaling_comparator.py -q`  
Expected: FAIL because `paper7.reward_scaling_comparator` does not exist.

- [ ] **Step 3: Implement comparator**

Implement:

```python
def summarize_by_scale(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ...

def compare_reward_scales(rows: list[dict[str, Any]], pre_specified_alpha: float = 0.185) -> dict[str, Any]:
    ...

def build_report(grid_path: Path, output_path: Path, pre_specified_alpha: float = 0.185) -> dict[str, Any]:
    ...
```

CLI default:

```powershell
python -m paper7.reward_scaling_comparator --grid paper7/results/revision/alpha_grid/grid_results.json --output paper7/results/revision/reward_scaling_comparator.json
```

- [ ] **Step 4: Verify targeted tests**

Run: `python -m pytest tests/test_reward_scaling_comparator.py -q`  
Expected: PASS.

- [ ] **Step 5: Generate artifact**

Run the CLI above.  
Expected output: `paper7/results/revision/reward_scaling_comparator.json`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/reward_scaling_comparator.py tests/test_reward_scaling_comparator.py paper7/results/revision/reward_scaling_comparator.json
git commit -m "test: add reward scaling comparator"
```

---

### Task 2: Planning significance audit

**Files:**
- Create: `paper7/planning_significance_audit.py`
- Create: `tests/test_planning_significance_audit.py`

- [ ] **Step 1: Write failing tests**

Create tests for policy summary and concentration:

```python
from paper7.planning_significance_audit import (
    concentration_metrics,
    summarize_policy_rows,
)


def test_concentration_metrics_reports_unique_share_and_top_share():
    selected = [1, 1, 2, 3]

    metrics = concentration_metrics(selected)

    assert metrics["n_actions"] == 4
    assert metrics["n_unique_blocks"] == 3
    assert metrics["unique_share"] == 0.75
    assert metrics["top1_share"] == 0.5


def test_summarize_policy_rows_handles_core_planning_fields():
    rows = [
        {
            "slope_change_pct": -1.0,
            "cont_change": 0.01,
            "baimu_count_change": 1,
            "baimu_area_change_ha": -10,
            "reward": 20,
        },
        {
            "slope_change_pct": -1.2,
            "cont_change": 0.03,
            "baimu_count_change": 3,
            "baimu_area_change_ha": -20,
            "reward": 30,
        },
    ]

    summary = summarize_policy_rows(rows)

    assert summary["n"] == 2
    assert summary["slope_change_pct_mean"] == -1.1
    assert summary["cont_change_mean"] == 0.02
    assert summary["baimu_count_change_mean"] == 2.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_planning_significance_audit.py -q`  
Expected: FAIL because `paper7.planning_significance_audit` does not exist.

- [ ] **Step 3: Implement audit script**

Implement:

```python
def concentration_metrics(selected_blocks: list[int]) -> dict[str, Any]:
    ...

def summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ...

def build_report(seed_dir: Path, baselines_path: Path, output_path: Path) -> dict[str, Any]:
    ...
```

The report should include calibrated 15-seed outcomes, uncalibrated 15-seed outcomes, reward-grid outcome context, and non-learning Bishan baseline context when available.

- [ ] **Step 4: Verify targeted tests**

Run: `python -m pytest tests/test_planning_significance_audit.py -q`  
Expected: PASS.

- [ ] **Step 5: Generate artifact**

Run:

```powershell
python -m paper7.planning_significance_audit --seed-dir paper7/results/revision/seeds --baselines paper7/results/revision/bishan_strong_baselines.json --output paper7/results/revision/planning_significance_audit.json
```

Expected output: `paper7/results/revision/planning_significance_audit.json`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/planning_significance_audit.py tests/test_planning_significance_audit.py paper7/results/revision/planning_significance_audit.json
git commit -m "test: add planning significance audit"
```

---

### Task 3: Transition diagnostic expansion

**Files:**
- Modify: `paper7/transition_rollout_diagnostics.py`
- Modify: `tests/test_transition_rollout_diagnostics.py`
- Output: `paper7/results/revision/transition_rollout_diagnostics.json`

- [ ] **Step 1: Add failing tests**

Extend tests with distribution and feature-group expectations:

```python
from paper7.transition_rollout_diagnostics import (
    summarize_feature_groups,
    summarize_step_metrics,
)


def test_summarize_step_metrics_adds_q50_and_q95_for_error_fields():
    metrics = [
        {"selected_block_mae": 1.0, "all_block_mae": 0.1, "global_mae": 2.0, "reward_abs_error": 3.0, "mask_agreement": 1.0},
        {"selected_block_mae": 3.0, "all_block_mae": 0.3, "global_mae": 4.0, "reward_abs_error": 5.0, "mask_agreement": 0.5},
    ]

    summary = summarize_step_metrics(metrics)

    assert summary["selected_block_mae_q50"] == 2.0
    assert summary["selected_block_mae_q95"] > 2.0
    assert summary["reward_mae"] == 4.0


def test_summarize_feature_groups_reports_named_global_groups():
    pred_global = [1.0, 3.0, 10.0, 10.0]
    true_global = [2.0, 1.0, 7.0, 12.0]

    groups = summarize_feature_groups(pred_global, true_global, {"first_two": [0, 1], "last_two": [2, 3]})

    assert groups["first_two_mae"] == 1.5
    assert groups["last_two_mae"] == 2.5
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_transition_rollout_diagnostics.py -q`  
Expected: FAIL until helpers are implemented or existing summary is extended.

- [ ] **Step 3: Implement diagnostic helpers**

Add:

```python
def summarize_feature_groups(pred_values, true_values, groups):
    ...
```

Extend `compute_step_metrics` and `summarize_step_metrics` to include q50, q95, and optional global group metrics while preserving existing keys used by manuscript/audit.

- [ ] **Step 4: Verify targeted tests**

Run: `python -m pytest tests/test_transition_rollout_diagnostics.py -q`  
Expected: PASS.

- [ ] **Step 5: Regenerate transition diagnostic artifact**

Run:

```powershell
python -m paper7.transition_rollout_diagnostics --output paper7/results/revision/transition_rollout_diagnostics.json
```

Expected output: updated `paper7/results/revision/transition_rollout_diagnostics.json` with additional distribution metrics.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/transition_rollout_diagnostics.py tests/test_transition_rollout_diagnostics.py paper7/results/revision/transition_rollout_diagnostics.json
git commit -m "test: expand transition rollout diagnostics"
```

---

### Task 4: Dongxing RL-lite environment and learned policy

**Files:**
- Create: `paper7/dongxing_rl_lite.py`
- Create: `tests/test_dongxing_rl_lite.py`
- Output: `paper7/results/revision/dongxing_rl_lite.json`

- [ ] **Step 1: Write failing tests**

Create toy tests:

```python
import numpy as np

from paper7.dongxing_rl_lite import (
    DongxingSlopeEnv,
    train_tabular_preference_policy,
    evaluate_preference_policy,
)


def toy_parcels():
    return [
        {"swappable_index": 0, "land_use": "farmland", "area_m2": 100.0, "slope": 10.0},
        {"swappable_index": 1, "land_use": "forest", "area_m2": 100.0, "slope": 2.0},
        {"swappable_index": 2, "land_use": "farmland", "area_m2": 100.0, "slope": 4.0},
        {"swappable_index": 3, "land_use": "forest", "area_m2": 100.0, "slope": 8.0},
    ]


def test_env_masks_only_positive_gain_blocks_and_rewards_slope_improvement():
    env = DongxingSlopeEnv(
        parcels=toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        max_steps=3,
        swaps_per_step=1,
    )

    obs, info = env.reset(seed=0)
    mask = env.action_masks()
    assert mask.tolist() == [True, False]

    obs, reward, terminated, truncated, info = env.step(0)
    assert reward > 0
    assert info["completed_pairs"] == 1
    assert info["slope_change_pct"] < 0


def test_preference_policy_learns_positive_weight_for_gain():
    env = DongxingSlopeEnv(
        parcels=toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        max_steps=3,
        swaps_per_step=1,
    )

    policy = train_tabular_preference_policy(env, seeds=[0, 1], episodes=10)
    result = evaluate_preference_policy(env, policy, seed=0)

    assert policy["weights"][0] > 0
    assert result["completed_pairs"] == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_dongxing_rl_lite.py -q`  
Expected: FAIL because `paper7.dongxing_rl_lite` does not exist.

- [ ] **Step 3: Implement environment and fallback learned policy**

Implement:

```python
class DongxingSlopeEnv(gym.Env):
    def reset(self, seed=None, options=None): ...
    def step(self, action): ...
    def action_masks(self): ...

def train_tabular_preference_policy(env, seeds: list[int], episodes: int, learning_rate: float = 0.05) -> dict[str, Any]:
    ...

def evaluate_preference_policy(env, policy: dict[str, Any], seed: int) -> dict[str, Any]:
    ...

def run_experiment(...): ...
```

The environment observation should be a compact per-block matrix flattened into a vector. At minimum include feasible gain, feasible exchange area, farm area, forest area, and already-used share. Reward should be positive when county average farmland slope decreases.

The learned fallback policy should update feature weights from observed action rewards and must be labeled `"learner_type": "tabular_preference_fallback"` in the JSON artifact. If MaskablePPO works reliably, include `"learner_type": "maskable_ppo"` and keep the fallback code for tests.

- [ ] **Step 4: Verify targeted tests**

Run: `python -m pytest tests/test_dongxing_rl_lite.py -q`  
Expected: PASS.

- [ ] **Step 5: Generate Dongxing RL-lite artifact**

Run:

```powershell
python -m paper7.dongxing_rl_lite --dltb paper7/data/dongxing_DLTB_with_slope.gpkg --block-dir paper7/results/dongxing_blocks_slope --output paper7/results/revision/dongxing_rl_lite.json --train-seeds 0,1,2,3,4 --eval-seeds 0,1,2,3,4 --episodes 200
```

Expected output: `paper7/results/revision/dongxing_rl_lite.json` with per-seed learned-policy evaluations and baseline comparisons.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/dongxing_rl_lite.py tests/test_dongxing_rl_lite.py paper7/results/revision/dongxing_rl_lite.json
git commit -m "test: add Dongxing slope-only RL-lite experiment"
```

---

### Task 5: End-to-end evidence audit update

**Files:**
- Modify: `paper7/end_to_end_validation.py`
- Modify: `tests/test_end_to_end_validation.py`
- Output: `paper7/results/revision/end_to_end_validation.json`

- [ ] **Step 1: Add failing tests**

Add tests that build temporary JSON artifacts and assert new scope classification:

```python
from paper7.end_to_end_validation import classify_claim_scope


def test_claim_scope_marks_dongxing_rl_lite_as_slope_only_not_transfer():
    evidence = {
        "dongxing_dynamic": {"status": "supported", "has_learned_policy": False},
        "dongxing_rl_lite": {
            "status": "supported_as_slope_only_rl_actionability",
            "learner_type": "tabular_preference_fallback",
        },
    }

    scopes = classify_claim_scope(evidence)
    dongxing = next(item for item in scopes if item["id"] == "dongxing_external_scope")

    assert dongxing["policy_transfer_tested"] is False
    assert dongxing["slope_only_rl_actionability_tested"] is True
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m pytest tests/test_end_to_end_validation.py -q`  
Expected: FAIL until claim scope handles `dongxing_rl_lite`.

- [ ] **Step 3: Implement audit summaries**

Add:

```python
def summarize_reward_scaling_comparator(path: Path) -> dict[str, Any]: ...
def summarize_planning_significance(path: Path) -> dict[str, Any]: ...
def summarize_dongxing_rl_lite(path: Path) -> dict[str, Any]: ...
```

Update `build_validation_report` and `classify_claim_scope` to include the new evidence with bounded wording.

- [ ] **Step 4: Verify targeted tests**

Run: `python -m pytest tests/test_end_to_end_validation.py -q`  
Expected: PASS.

- [ ] **Step 5: Regenerate evidence audit**

Run:

```powershell
python -m paper7.end_to_end_validation --out paper7/results/revision/end_to_end_validation.json
```

Expected: audit includes `reward_scaling_comparator`, `planning_significance`, and `dongxing_rl_lite`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/end_to_end_validation.py tests/test_end_to_end_validation.py paper7/results/revision/end_to_end_validation.json
git commit -m "test: extend Paper 7 evidence audit"
```

---

### Task 6: Full verification before manuscript edits

**Files:**
- No source edits unless tests expose a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_reward_scaling_comparator.py tests/test_planning_significance_audit.py tests/test_transition_rollout_diagnostics.py tests/test_dongxing_rl_lite.py tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest tests -q --basetmp .\.pytest-tmp-paper7-ceus-strengthening
```

Expected: PASS. If PyTorch prints a Windows access-violation stack after pytest success but exit code is 0, record that honestly.

- [ ] **Step 3: Inspect result artifacts**

Run:

```powershell
git status -sb
```

Expected: only intentional tracked result/source/test changes are present.

---

### Task 7: Manuscript and submission package update

**Files:**
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex`
- Modify: `submission/ceus/06_latex_source_editable/manuscript_signed.tex`
- Modify: `submission/ceus/README_CEUS_submission_package.md`
- Modify: `submission/ceus/CEUS_paper7_latex_source_anonymous.zip`

- [ ] **Step 1: Update manuscript evidence statements**

Edit only after Task 6 passes. Required changes:

- Abstract: mention the new reward-scaling comparator and Dongxing slope-only RL-lite boundary if results support it.
- Methods: add a concise Dongxing RL-lite environment paragraph.
- Experiments: add reward comparator, planning audit, transition diagnostic expansion, and Dongxing RL-lite result paragraphs.
- Discussion/Limitations: explicitly state Dongxing is slope-only actionability, not full cross-county learned-policy transfer.

- [ ] **Step 2: Mirror signed/editable manuscript**

Apply the same evidence and claim-boundary edits to `submission/ceus/06_latex_source_editable/manuscript_signed.tex`.

- [ ] **Step 3: Compile anonymous manuscript**

Run twice from `submission/ceus/01_main_document_anonymous`:

```powershell
pdflatex -interaction=nonstopmode manuscript.tex
pdflatex -interaction=nonstopmode manuscript.tex
```

Expected: `manuscript.pdf` builds successfully.

- [ ] **Step 4: Scan LaTeX log**

Run:

```powershell
rg -n "undefined|Undefined|Citation|Reference|Fatal|Emergency|Error|Rerun" manuscript.log
```

Expected: no fatal errors, undefined references, or unresolved citations.

- [ ] **Step 5: Rebuild anonymous source zip**

Recreate `submission/ceus/CEUS_paper7_latex_source_anonymous.zip` from the anonymous source folder contents required by CEUS.

- [ ] **Step 6: Final package checks**

Run:

```powershell
git status -sb
python -m paper7.end_to_end_validation --out paper7/results/revision/end_to_end_validation.json
```

Expected: audit still passes and only intentional files changed.

- [ ] **Step 7: Commit**

Run:

```powershell
git add submission/ceus paper7/results/revision/end_to_end_validation.json
git commit -m "docs: update CEUS manuscript with strengthened experiments"
```

---

### Self-review

- Spec coverage: covers low-cost audits, Dongxing RL-lite experiment, evidence audit, verification, manuscript/package update.
- Placeholder scan: no placeholder task is left; every task has concrete paths and commands.
- Type consistency: helper names are consistent across tests and implementation steps.
- Scope check: plan preserves evidence-first sequencing and does not claim full Dongxing transfer.
