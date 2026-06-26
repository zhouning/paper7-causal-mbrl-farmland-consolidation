# Paper 7 Review Claim Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Paper 7 manuscript's reviewer-facing claims match the current audited evidence chain and keep future stale p-value regressions out of the manuscript.

**Architecture:** Add a focused manuscript-consistency test that reads the CEUS manuscript and the end-to-end audit JSON. Then revise the manuscript text so headline statistics, calibration language, external-validation boundaries, and model-free baseline comparisons are stated at the audited evidence level.

**Tech Stack:** Python standard library, pytest, LaTeX manuscript text, existing Paper 7 audit JSON artifacts.

---

### Task 1: Add A Failing Manuscript-Claim Consistency Test

**Files:**
- Create: `tests/test_manuscript_claim_consistency.py`
- Read: `paper7/results/revision/end_to_end_validation.json`
- Read: `submission/ceus/01_main_document_anonymous/manuscript.tex`

- [ ] **Step 1: Write the failing test**

Create `tests/test_manuscript_claim_consistency.py` with exactly this content:

```python
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = REPO_ROOT / "paper7" / "results" / "revision" / "end_to_end_validation.json"
MANUSCRIPT_PATH = (
    REPO_ROOT
    / "submission"
    / "ceus"
    / "01_main_document_anonymous"
    / "manuscript.tex"
)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def test_manuscript_uses_audited_paired_calibration_p_values():
    audit = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8")
    compact = _compact(manuscript)

    paired = audit["evidence"]["bishan_seed_chain"]["paired_slope_test"]
    one_sided = f"p={paired['one_sided_p']:.3f}"
    two_sided = f"p={paired['two_sided_p']:.3f}"

    assert "p=0.004" not in compact
    assert one_sided in compact
    assert two_sided in compact
    assert "Mann-Whitney" not in manuscript


def test_manuscript_keeps_review_boundaries_visible():
    manuscript = MANUSCRIPT_PATH.read_text(encoding="utf-8").lower()

    assert "observational reward regularization" in manuscript
    assert "not definitive causal identification" in manuscript
    assert "not direct bishan-to-dongxing policy transfer" in manuscript
    assert "descriptive" in manuscript
    assert "model-free" in manuscript
```

- [ ] **Step 2: Run the new test and confirm the expected failure**

Run:

```powershell
python -m pytest tests/test_manuscript_claim_consistency.py -q
```

Expected:

- FAIL because `manuscript.tex` still contains `p=0.004` / `p = 0.004`.
- FAIL because the manuscript does not yet contain the rounded audited two-sided value `p=0.024`.
- The boundary test may already pass; the p-value test must fail before manuscript edits.

### Task 2: Revise Manuscript Claims And Boundaries

**Files:**
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex`
- Test: `tests/test_manuscript_claim_consistency.py`

- [ ] **Step 1: Update the abstract p-value and practical framing**

In the abstract, replace:

```latex
calibrated model-based policies achieved $-1.102\% \pm 0.100\%$ slope change versus $-0.976\% \pm 0.129\%$ without calibration, a 13.0\% paired improvement ($p=0.004$).
```

with:

```latex
calibrated model-based policies achieved $-1.102\% \pm 0.100\%$ slope change versus $-0.976\% \pm 0.129\%$ without calibration, a 13.0\% paired improvement under an exact paired sign-flip test (one-sided $p=0.012$; two-sided $p=0.024$).
```

- [ ] **Step 2: Update the contribution bullet for reward calibration**

In the second contribution bullet, replace:

```latex
The calibration corrects a 5.4$\times$ overestimation of action-quality differentials, reducing reward exploitation and improving downstream policy quality by 13.0\% ($p = 0.004$).
```

with:

```latex
The calibration corrects a 5.4$\times$ overestimation of action-quality differentials, reducing reward exploitation and improving downstream policy quality by 13.0\% under an exact paired sign-flip test (one-sided $p=0.012$; two-sided $p=0.024$).
```

- [ ] **Step 3: Update the main results paragraph that still cites Mann-Whitney**

In the paragraph beginning `\textbf{Model-based RL improves mean slope reduction`, replace:

```latex
The strongest formal test in this experiment is the methodological comparison between calibrated and uncalibrated learned environments: calibration improves model-based policy quality by 13.0\% ($p = 0.004$, Mann-Whitney $U$ test). The comparison with model-free baselines should be read as a descriptive reference under the same task rather than as a fully balanced statistical superiority claim.
```

with:

```latex
The strongest formal test in this experiment is the paired methodological comparison between calibrated and uncalibrated learned environments: calibration improves model-based policy quality by 13.0\% under an exact paired sign-flip test (one-sided $p=0.012$; two-sided $p=0.024$). The comparison with model-free baselines should be read as a descriptive reference under the same task rather than as a fully balanced statistical superiority claim.
```

- [ ] **Step 4: Update the calibration results paragraph**

Replace:

```latex
Observational calibration improves slope reduction by 13.0\% ($-1.102\%$ vs.\ $-0.976\%$), statistically significant at $p = 0.004$ (Mann-Whitney $U = 48$, one-sided). The calibrated policy achieves better slope reduction than the uncalibrated version in 10 of 15 seeds. The improvement is robust: the worst calibrated seed ($-0.946\%$) still matches the uncalibrated mean.
```

with:

```latex
Observational calibration improves slope reduction by 13.0\% ($-1.102\%$ vs.\ $-0.976\%$). Because the calibrated and uncalibrated policies are paired by seed, we use an exact paired sign-flip test rather than an independent-sample Mann--Whitney test. The paired slope delta is $-0.127$ percentage points, with one-sided $p=0.012$ and two-sided $p=0.024$; the calibrated policy achieves better slope reduction in 10 of 15 seeds. The improvement is robust in the planning sense that the worst calibrated seed ($-0.946\%$) still matches the uncalibrated mean.
```

- [ ] **Step 5: Update the conclusion bullets**

In the conclusion bullet beginning `\item \textbf{Model-based policies improve mean slope reduction`, replace the heading and first sentence with:

```latex
\item \textbf{Model-based policies provide a descriptive low-cost reference against model-free baselines.} Policies trained on the learned environment achieve $-0.976\%$ slope reduction on the real environment across 15 seeds, compared with 12-hour A100-trained MARL ($-0.84\%$) and centralized DRL ($-0.79\%$).
```

In the conclusion bullet beginning `\item \textbf{Observational calibration addresses reward exploitation`, replace:

```latex
Correcting this via a calibration factor improves policy quality by 13.0\% to $-1.102\%$ ($p = 0.004$).
```

with:

```latex
Correcting this via a calibration factor improves policy quality by 13.0\% to $-1.102\%$ under an exact paired sign-flip test (one-sided $p=0.012$; two-sided $p=0.024$).
```

- [ ] **Step 6: Run the manuscript-consistency test and confirm it passes**

Run:

```powershell
python -m pytest tests/test_manuscript_claim_consistency.py -q
```

Expected:

- PASS.

### Task 3: Run Focused Regression Tests

**Files:**
- Test: `tests/test_manuscript_claim_consistency.py`
- Test: `tests/test_planning_significance_audit.py`
- Test: `tests/test_planning_paired_statistics.py`
- Test: `tests/test_reward_components.py`
- Test: `tests/test_reward_weight_sensitivity.py`
- Test: `tests/test_end_to_end_validation.py`

- [ ] **Step 1: Run focused regression tests**

Run:

```powershell
python -m pytest tests/test_manuscript_claim_consistency.py tests/test_planning_significance_audit.py tests/test_planning_paired_statistics.py tests/test_reward_components.py tests/test_reward_weight_sensitivity.py tests/test_end_to_end_validation.py -q
```

Expected:

- PASS.

- [ ] **Step 2: Check for stale manuscript claim text**

Run:

```powershell
rg -n "p=0\\.004|p = 0\\.004|Mann-Whitney" submission\ceus\01_main_document_anonymous\manuscript.tex
```

Expected:

- No matches.

### Task 4: Full Verification And Commit

**Files:**
- Modify: `submission/ceus/01_main_document_anonymous/manuscript.tex`
- Create: `tests/test_manuscript_claim_consistency.py`
- Commit staged changes.

- [ ] **Step 1: Run the full pytest suite**

Run:

```powershell
python -m pytest -q
```

Expected:

- PASS with all tests passing.
- On the current Windows Python 3.14 environment, pytest may print a torch DLL access-violation stack after reporting passed tests. Treat exit code `0` as the pass/fail gate and report the environment warning in the final summary.

- [ ] **Step 2: Review the diff**

Run:

```powershell
git diff --stat
git diff -- tests/test_manuscript_claim_consistency.py submission/ceus/01_main_document_anonymous/manuscript.tex
```

Expected:

- Diff includes only the new manuscript-consistency test and targeted manuscript wording changes.
- No result artifact JSON files are changed.

- [ ] **Step 3: Commit the implementation**

Run:

```powershell
git add tests/test_manuscript_claim_consistency.py submission/ceus/01_main_document_anonymous/manuscript.tex
git commit -m "test: align paper7 manuscript claims with audit"
```

Expected:

- Commit succeeds.

- [ ] **Step 4: Report final state**

Run:

```powershell
git status --short --branch
git log -2 --oneline
```

Expected:

- `main` is ahead of `origin/main` by the spec and implementation commits unless the user later requests a push.
- The latest commit is `test: align paper7 manuscript claims with audit`.
