# Paper 7 Review Claim Hardening Design

## Goal

Revise Paper 7 after reviewer-style assessment so that the manuscript's headline
claims match the audited evidence chain and are framed at the right strength for
a CEUS-style submission.

## Scope

The revision focuses on claim consistency, reviewer-facing clarity, and bounded
interpretation. It does not add new experiments or change the core model-based
reinforcement learning pipeline.

## Reviewer Issues To Address

1. Statistical consistency:
   - The manuscript still contains the older calibration significance wording
     `p=0.004`.
   - The current audit reports an exact paired sign-flip one-sided p-value of
     `0.011963` and a two-sided p-value of `0.023926`.
   - Manuscript text should use rounded values such as `one-sided p=0.012` and,
     where useful, `two-sided p=0.024`.

2. Claim boundaries:
   - Treatment-effect-informed reward calibration must be described as
     observational reward regularization, not definitive causal identification.
   - Dongxing evidence must be described as a local full-reward counterpart, not
     direct Bishan-to-Dongxing policy transfer.
   - Model-free baseline comparisons are descriptive because the seed counts,
     training budgets, and training setups are not fully matched.

3. Reader value:
   - The manuscript should more clearly state the practical planning problem:
     slow parcel-level simulation and learned-reward exploitation.
   - The practical contribution should be explicit: CPU model-based training with
     final real-environment evaluation.
   - Slope-change results should be framed as county-level sequential budget
     allocation improvements, not only as algorithm scores.

## Proposed Changes

### Manuscript

Update `submission/ceus/01_main_document_anonymous/manuscript.tex` to:

- Replace stale `p=0.004` / `p = 0.004` wording with audit-consistent paired
  sign-flip p-values.
- Reword the abstract and contribution bullets to foreground:
  - the learned environment as the spatial planning training surrogate;
  - treatment-effect-informed calibration as observational regularization;
  - real-environment final evaluation;
  - CPU runtime and practical planning value.
- Add or tighten boundary language in the Results, Discussion, and Conclusions:
  - no direct cross-county policy transfer claim;
  - no definitive causal identification claim;
  - no formal superiority claim over model-free baselines.
- Add a concise planning interpretation for the Bishan slope result.

### Tests

Add a focused manuscript-consistency test before editing the manuscript. The test
should:

- read `paper7/results/revision/end_to_end_validation.json`;
- read `submission/ceus/01_main_document_anonymous/manuscript.tex`;
- fail if the manuscript contains the stale `p=0.004` wording;
- require the rounded exact paired sign-flip p-values to appear in the
  manuscript;
- require direct-transfer and observational-regularization boundary terms to
  remain visible.

The initial test is expected to fail before manuscript edits and pass after the
claim hardening changes.

## Non-Goals

- Do not add Dyna-style training in this revision.
- Do not implement adapter-based cross-county policy transfer.
- Do not add new heavy RL training runs.
- Do not rewrite the manuscript from scratch.
- Do not change stored result artifacts unless a direct inconsistency is found.

## Verification

Run, in order:

1. The new manuscript-consistency test and confirm the expected failure.
2. Apply manuscript edits.
3. Re-run the new test and confirm it passes.
4. Run relevant focused tests around end-to-end validation and reward statistics.
5. Run the full pytest suite if the focused tests are clean.

## Expected Outcome

After revision, the manuscript should make a stronger and more defensible case:

- The innovation is framed as a learned-environment MBRL system for county-scale
  farmland consolidation planning.
- The practical contribution is tied to CPU training and real-environment final
  evaluation.
- The calibration, external validation, and baseline comparisons are stated
  within the evidence boundaries verified by the audit artifacts.
