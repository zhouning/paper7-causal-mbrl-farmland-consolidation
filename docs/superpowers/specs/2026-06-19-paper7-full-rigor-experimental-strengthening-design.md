# Paper 7 Full Rigor Experimental Strengthening Design

## Purpose

Strengthen Paper 7 according to a scientific-evidence-first standard rather than a submission-readiness standard. The next revision must not merely defend the current bounded claims. It must test whether the reward design, learned environment, and cross-county applicability remain credible when evaluated under stricter reviewer expectations.

The work therefore targets four unresolved scientific questions:

- Is the current multi-objective reward robust to reasonable weight changes, or is the reported policy quality an artifact of a chosen coefficient set?
- Can Dongxing be evaluated under a full Bishan-style multi-objective county environment rather than only a slope-only RL-lite check?
- Does model-based training work when the learned environment is trained locally on Dongxing trajectories and evaluated in the Dongxing real environment?
- Does Bishan-to-Dongxing transfer work directly, require fine-tuning, or fail under distribution shift?

No manuscript claim should be expanded until the relevant experiment has produced a stored artifact, passed tests, and been summarized in an executable evidence audit.

## Current Evidence Boundary

The current repository supports these claims:

- Bishan complete learned-environment MBRL chain: trajectory collection, transition training, model-based policy training, calibration, and real-environment evaluation.
- Bishan calibration improves slope reduction over uncalibrated learned-environment training across 15 paired seeds.
- Reward scaling comparator shows the pre-specified observational alpha is near the empirical reward-scale optimum.
- Dongxing currently supports external data ingestion, DEM-based slope enrichment, block actionability, dynamic slope-aware non-RL baselines, and slope-only learned actionability.

The current repository does not support these stronger claims:

- The reward coefficients are robust across a broad multi-objective preference range.
- Dongxing supports full slope-contiguity-baimu reward evaluation.
- Dongxing supports full local MBRL under the same evaluation discipline as Bishan.
- Bishan policies or transition models transfer successfully to Dongxing.
- A single global calibration factor is the best possible calibration mechanism.

## Recommended Experimental Route

This design approves the full-strength route rather than the earlier bounded CEUS route. The work should proceed in phases, and each phase should be publishable even if later phases fail. Negative results must be retained and interpreted, not hidden.

### Phase 1: Reward Weight Robustness And Pareto Analysis

Build reward sensitivity experiments around the existing Bishan `CountyLevelEnv` reward components:

- slope reduction
- contiguity change
- baimu area change
- baimu count change
- asymmetric baimu-area loss penalty
- invalid/no-swap penalty

The experiment should separate reward decomposition from policy training. At minimum, every evaluated episode should record per-step raw component deltas so that alternative reward weights can be replayed without rerunning the real environment.

Required analyses:

- Recompute rewards for the current 15 calibrated and 15 uncalibrated Bishan policy evaluations under alternative weight settings when component logs are available.
- Run additional real-environment heuristic baselines under the same reward decomposition: random, dynamic slope-gap, area-weighted slope-gap, contiguity-aware, baimu-aware, and scalarized greedy policies.
- Build a grid or Latin-hypercube sweep over weight families, not just one-dimensional reward scaling.
- Report Pareto fronts across final slope change, contiguity change, baimu count change, baimu area change, and total scalarized reward.
- Classify the default weights as robust, slope-biased, baimu-biased, or unstable based on their location in the Pareto distribution.

Success criterion:

- The default reward does not need to be globally optimal for every planning preference, but it must be shown to be a defensible policy preference. If it is fragile, the paper must say so and either revise the reward or report the fragility as a limitation.

### Phase 2: Generic Full County Environment For External Data

Refactor the Bishan-specific environment into a reusable full county environment that can load:

- a DLTB GeoPackage or equivalent parcel file,
- a parcel-to-block mapping,
- block compositions,
- optional block features,
- a configurable administrative unit field,
- CRS and land-use classification metadata.

The external environment must preserve the same scientific semantics as Bishan:

- parcel-label paired swaps,
- action masks over mixed farmland-forest blocks,
- area-weighted farmland slope,
- parcel adjacency-based contiguity,
- union-find baimu fang count and area,
- the same reward component definitions,
- deterministic real-environment evaluation.

The implementation should avoid copying the whole `CountyLevelEnv` into a Dongxing-only file. Shared logic should move into a generic class or small reusable utilities, while the current Bishan environment remains backward compatible for existing tests and results.

Success criterion:

- Bishan behavior remains unchanged under existing tests.
- A Dongxing full environment can reset, compute action masks, run a 100-step episode, and report slope, contiguity, baimu count, baimu area, reward components, and budget usage.

### Phase 3: Dongxing Full Multi-Objective Baselines

Before training learned models, run full Dongxing real-environment baselines:

- random valid-block selection with at least 100 seeds,
- dynamic slope-gap greedy,
- dynamic area-weighted slope-gap greedy,
- contiguity-aware greedy,
- baimu-area-aware greedy,
- scalarized greedy using the default reward weights.

All baselines must use the full Dongxing real environment and report:

- slope change percentage,
- contiguity change,
- baimu count change,
- baimu area change in hectares,
- total real reward,
- completed paired swaps,
- selected block concentration,
- per-administrative-unit investment distribution.

Success criterion:

- Establish a defensible Dongxing baseline table before claiming any learned-policy value.
- If slope-aware heuristics dominate learned methods later, that becomes a valid scientific finding.

### Phase 4: Dongxing Local Learned Environment And MBRL

Collect Dongxing trajectories from the full real environment:

- random trajectories,
- slope-greedy trajectories,
- scalarized-greedy trajectories,
- optional mixed/noisy policies for support diversity.

Train a Dongxing transition model with the same discipline as Bishan:

- train/validation split,
- one-step transition and reward metrics,
- multi-step recorded-action rollout diagnostics,
- policy-induced learned-vs-real diagnostics after policy training,
- action-mask agreement,
- reward MAE before and after optional calibration.

Train Dongxing model-based policies:

- uncalibrated learned-environment policy,
- globally calibrated learned-environment policy if a Dongxing calibration factor is estimable,
- optional state-dependent calibration pilot if common support is adequate.

Evaluate every final policy in the Dongxing real environment, not in the learned environment.

Success criterion:

- Demonstrate whether local Dongxing MBRL beats or complements the full real-environment baselines under multi-objective metrics.
- If reward exploitation or transition error prevents reliable learning, store the failure diagnostics and narrow claims accordingly.

### Phase 5: Bishan-To-Dongxing Transfer And Fine-Tuning

Test cross-county transfer explicitly rather than implying it:

- Direct policy transfer: apply Bishan-trained policy logic to Dongxing only if observation/action dimensions can be made compatible through a documented adapter.
- Transition model transfer: pretrain from Bishan transition weights and fine-tune on Dongxing trajectories when feature dimensions match.
- From-scratch Dongxing transition model: compare against transfer and fine-tuning.
- Calibration transfer: compare Bishan alpha, Dongxing alpha, unscaled reward, and any state-dependent pilot.

If direct policy transfer is structurally invalid because the action space and observation dimensions differ, the experiment must say this explicitly and shift to model-transfer or representation-transfer tests.

Success criterion:

- Classify transfer as direct-success, fine-tuning-required, unsupported, or structurally invalid.
- The paper must not claim cross-county learned-policy transfer unless this phase supports it.

### Phase 6: Evidence Audit, Tables, And Manuscript Revision

Only after experimental artifacts exist:

- Extend `paper7/end_to_end_validation.py` to classify the stronger evidence chain.
- Add result-summary JSON files under `paper7/results/full_rigor/`.
- Add focused tests for reward decomposition, generic environment behavior, Dongxing full metrics, and audit classification.
- Update manuscript text, tables, and limitations only from validated artifacts.
- Rebuild PDFs and the CEUS source package after all evidence and wording changes are complete.

## Data And Artifact Layout

New generated outputs should use a separate directory to avoid confusing prior CEUS evidence:

- `paper7/results/full_rigor/reward_weight_sensitivity.json`
- `paper7/results/full_rigor/reward_pareto_front.json`
- `paper7/results/full_rigor/dongxing_full_baselines.json`
- `paper7/results/full_rigor/dongxing_trajectories_summary.json`
- `paper7/results/full_rigor/dongxing_transition_diagnostics.json`
- `paper7/results/full_rigor/dongxing_mbrl_results.json`
- `paper7/results/full_rigor/transfer_finetune_results.json`
- `paper7/results/full_rigor/full_rigor_evidence_audit.json`

Large trajectory and model files should stay ignored by Git unless explicitly approved for Git LFS or external release storage.

## Testing And Verification

Required test coverage:

- Reward decomposition produces the same scalar reward as the current `CountyLevelEnv` formula under default weights.
- Generic county environment reproduces Bishan-compatible metrics on toy parcels and block compositions.
- Dongxing full environment computes contiguity and baimu metrics on a toy adjacency graph.
- Baseline selectors obey action masks and never select infeasible blocks.
- Evidence audit distinguishes slope-only Dongxing evidence, full Dongxing real-environment evidence, local Dongxing MBRL evidence, and transfer evidence.

Required verification before manuscript edits:

- Targeted tests for all new modules pass.
- Full `python -m pytest tests -q` passes.
- Every new result artifact has deterministic metadata: command, timestamp, input paths, seeds, environment settings, and code version.
- Audit classifies any failed experiment honestly rather than omitting it.

## Scientific Decision Rules

The following outcomes are all acceptable, but they imply different manuscript claims:

- If reward weights are robust and Dongxing local MBRL succeeds, the paper can claim stronger multi-objective external applicability.
- If reward weights are robust but Dongxing local MBRL fails, the paper should claim strong Bishan evidence and report cross-county learning limits.
- If reward weights are fragile, the paper must present a reward preference analysis and avoid implying a universal planning objective.
- If direct transfer is structurally invalid, the paper should state that county-scale action spaces require adaptation rather than direct policy transfer.
- If fine-tuning works but direct transfer fails, the paper can claim transferable learned dynamics only with local adaptation.

## Non-Goals

- Do not expand claims before experiments complete.
- Do not hide negative results.
- Do not treat Dongxing slope-only RL-lite as full validation.
- Do not commit large Dongxing GeoPackages, DEM rasters, trajectory arrays, or model checkpoints unless storage policy is separately approved.
- Do not rewrite the manuscript before the evidence audit is updated.

## Self-Review

- Placeholder scan: no placeholders remain.
- Consistency check: all phases preserve the evidence-first principle and separate real-environment evaluation from learned-environment training.
- Scope check: this is intentionally larger than the CEUS strengthening batch. It should be implemented through a detailed plan with checkpoints after each phase.
- Ambiguity check: success and failure outcomes are both mapped to allowable manuscript conclusions.
