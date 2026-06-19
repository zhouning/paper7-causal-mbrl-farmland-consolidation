# Paper 7 CEUS Experimental Strengthening Design

## Purpose

Strengthen Paper 7 for CEUS submission by adding substantive experimental evidence before making manuscript changes. The work responds to the strict-review concerns that the current evidence chain is strong within Bishan but still thin on external learned-policy actionability, reward-scaling alternatives, transition-model interpretability, and planning significance.

The improvement principle is evidence first, manuscript second. No claim should be expanded until the corresponding experiment has run, produced a recorded result artifact, passed tests or audit checks, and been incorporated into the end-to-end validation report.

## Current Evidence Boundary

The repository currently supports these claims:

- Bishan model-based policies are trained in a learned environment and evaluated in the real parcel-simulation environment.
- Observational reward calibration improves learned-environment policy training across 15 paired Bishan seeds.
- A 15-seed policy-induced learned-vs-real diagnostic shows high mask agreement, close support distance, and bounded reward error under trained-policy actions.
- Dongxing currently supports external data, action-space, slope enrichment, and non-RL dynamic paired-swap feasibility.

The repository does not yet support these stronger claims:

- Full cross-county transfer of the learned Bishan policy to Dongxing.
- A full Dongxing analogue of Bishan's multi-objective slope, contiguity, and baimu reward.
- Definitive causal identification for reward calibration.
- Superiority over model-free baselines under a fully matched compute and seed protocol.

## Approved Improvement Strategy

### Phase A: Low-cost evidence audits

Add audit-style experiments that reuse existing recorded results and transition assets:

- Transition diagnostics: expand beyond whole-observation cosine similarity using selected-block error, all-block error, global-feature error, reward error, action-mask agreement, horizon curves, and selected error quantiles.
- Reward-scaling comparator: compare the pre-specified observational calibration factor `alpha=0.185` against ordinary heuristic reward scaling, the grid-search optimum, and unscaled training.
- Planning significance audit: summarize policy effects on slope, contiguity, baimu count, baimu area, reward, budget completion, and selected-block concentration so the planning interpretation is not reduced to a single slope percentage.

Outputs should be JSON files under `paper7/results/revision/` and pure helper functions with focused tests where practical.

### Phase B: Dongxing external RL-lite experiment

Build a conservative Dongxing slope-only RL experiment from the existing `DynamicSwapState`:

- Implement a Gymnasium-compatible Dongxing environment with action masks over Dongxing blocks.
- Use the existing paired-swap slope dynamics and a slope-improvement reward.
- Train a lightweight masked RL policy or, if the installed RL stack cannot support this environment reliably, run a documented lightweight policy-learning fallback that is still separate from hand-coded greedy rules.
- Evaluate learned Dongxing policies against random, dynamic slope-gap, and dynamic area-weighted non-RL baselines.
- Store per-seed results, aggregate statistics, and empirical comparisons in `paper7/results/revision/dongxing_rl_lite.json`.

This experiment must be described as external slope-only RL actionability, not full cross-county learned-policy transfer.

### Phase C: Evidence audit and manuscript update

Only after Phases A and B produce valid result artifacts:

- Update `paper7/end_to_end_validation.py` so the new evidence is classified by scope.
- Re-run relevant tests and the full test suite.
- Update the CEUS manuscript text, tables, and claim boundaries.
- Rebuild the anonymous PDF and LaTeX source zip.
- Update submission package README files if the evidence inventory changes.

## Claim Language Rules

The manuscript may say:

- The Bishan results provide the main complete learned-environment policy evidence.
- Reward calibration is an observational treatment-effect-informed regularizer.
- The pre-specified calibration factor performs near the reward-scaling grid optimum if the comparator audit confirms this.
- Dongxing provides external slope-only RL actionability if the new RL-lite policy beats or meaningfully complements non-RL baselines.

The manuscript must not say:

- Dongxing proves full cross-county learned-policy transfer.
- The calibration factor is causally identified in the strict experimental sense.
- Whole-observation cosine similarity alone validates the transition model.
- Model-based RL strictly dominates model-free RL under a fully matched training-budget protocol unless that protocol is actually run.

## Testing And Validation

The implementation should add focused unit tests before each new helper or environment behavior:

- Transition diagnostic helpers must be testable on toy arrays.
- Reward-scaling comparator summaries must be testable on synthetic grid rows.
- Dongxing RL-lite environment behavior must be testable on toy parcels and toy blocks.
- Evidence audit additions must be testable without rerunning expensive RL.

Before manuscript edits, run:

- Targeted tests for the new components.
- `python -m pytest tests -q --basetmp .\.pytest-tmp-paper7-ceus-strengthening`
- `python -m paper7.end_to_end_validation`

Before final submission-package completion, run two `pdflatex` passes in `submission/ceus/01_main_document_anonymous`, scan the LaTeX log for fatal or unresolved-reference issues, and verify the anonymous source zip contents.

## Non-goals

- No new external data download is required.
- No destructive cleanup of ignored Dongxing raster or GeoPackage assets.
- No claim expansion before result artifacts exist.
- No attempt to train an expensive full model-free cross-county baseline unless a later design explicitly approves that scope.

## Self-review

- Placeholder scan: no placeholders remain.
- Consistency check: all phases preserve the evidence-first principle and keep Dongxing scope bounded.
- Scope check: this is one implementation batch with three dependent phases: audits, Dongxing RL-lite, then manuscript update.
- Ambiguity check: if masked RL training is blocked by the installed stack, the fallback must still be a learned policy and must be labeled as a fallback in the result artifact and manuscript.
