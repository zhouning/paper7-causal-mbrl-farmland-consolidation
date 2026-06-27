# Paper 7 Reviewer-Driven Experimental Hardening Design

## Goal

Strengthen Paper 7 after reviewer-style assessment with a focused combination of
algorithmic, experimental, and manuscript improvements.

The next revision should fix the remaining reviewer risk around Dongxing
external evidence. Dongxing already supports a full-reward local counterpart,
but several deterministic Dongxing policies currently report repeated seed
evaluations with zero variance. The revision should replace that weak
replication signal with scenario-based robustness evidence and add a lightweight
scenario-robust learned-environment planner.

## Starting Context

The current repository is already strong on the Bishan primary claim:

- Bishan calibrated versus uncalibrated learned-environment policies use 15
  paired seeds.
- The audited exact paired sign-flip test reports one-sided `p=0.012` and
  two-sided `p=0.024`.
- The reward-scaling comparator shows the pre-specified observational
  calibration factor `alpha=0.185` is within 3.1 percent of the empirical
  grid-search optimum.
- The manuscript now frames calibration as observational reward regularization,
  not definitive causal identification.
- The manuscript title has already been corrected away from the old
  "Dreaming in Embedding Space" wording.

Dongxing is also available locally:

- `paper7/data/dongxing_DLTB_with_slope.gpkg`
- `paper7/data/dongxing_dem_srtmgl1.tif`
- `paper7/results/dongxing_blocks_slope/`
- `paper7/results/full_rigor/dongxing_full_baselines.json`
- `paper7/results/full_rigor/dongxing_full_learned_policy.json`
- `paper7/results/full_rigor/dongxing_full_model_based_policy.json`
- `paper7/results/full_rigor/dongxing_model_based_optimization.json`
- `paper7/results/full_rigor/dongxing_multistep_mbrl_policy.json`

The remaining problem is not data availability. It is evidentiary strength and
statistical interpretation.

## Reviewer Risks To Address

1. **Pseudo-replication in deterministic Dongxing evaluations.**
   Several deterministic policies are evaluated over multiple seeds, but
   `GenericCountyEnv.reset(seed=...)` resets to the same deterministic initial
   state. These repeated seeds are not independent stochastic replications.

2. **External robustness is underdeveloped.**
   Dongxing is currently framed as a local counterpart, which is correct, but it
   does not yet show how the policies behave under plausible DEM-derived slope
   uncertainty or planning-constraint changes.

3. **Algorithmic contribution on Dongxing can be clearer.**
   The existing Dongxing multi-step learned-environment policy is useful, but a
   reviewer can still argue that it is a single deterministic local run rather
   than a robust planning algorithm.

4. **Manuscript tables can invite over-reading.**
   `Eval seeds` wording on deterministic Dongxing rows can be mistaken for
   independent random-seed replication. The manuscript should instead report
   deterministic episodes and scenario counts separately.

## Recommended Route

Implement a scenario-based Dongxing hardening package:

1. Add scenario perturbations that create real evaluation variation.
2. Add a lightweight scenario-robust learned-environment planner.
3. Summarize policy performance across base and held-out scenarios.
4. Wire the result into the end-to-end audit, evidence ledger, tests, and
   manuscript.

This gives the paper a stronger external-evidence story without making the
unsupported claim of direct Bishan-to-Dongxing transfer.

## Non-Goals

- Do not claim direct Bishan-to-Dongxing policy transfer.
- Do not build an adapter or fine-tuning system in this cycle.
- Do not retrain the full Bishan 15-seed or reward-scaling experiments.
- Do not use deterministic Dongxing seed repetitions as independent evidence.
- Do not hide mixed or negative Dongxing results.
- Do not promote Dongxing robustness to universal cross-county generalization.

## Design

### 1. Scenario Family

Create a compact Dongxing scenario set that represents plausible planning and
data uncertainty while remaining cheap to run.

Scenario dimensions:

- **Slope perturbation:** deterministic DEM-derived slope noise or scaling,
  generated from fixed scenario seeds and clipped at zero.
- **Budget constraint:** lower, default, and higher total paired-swap budgets.
- **Execution granularity:** alternative `swaps_per_step` values.

The main output should distinguish:

- `base_scenario`: the currently reported Dongxing setting.
- `robustness_scenarios`: perturbation scenarios used for evaluation.
- `selection_scenarios`: scenarios used for robust planner optimization, if a
  held-out split is used.
- `heldout_scenarios`: scenarios not used to select planner weights.

The scenario count, not deterministic seed count, is the source of robustness
variation for deterministic policies.

### 2. Scenario-Robust Learned-Environment Planner

Add a lightweight algorithmic extension on top of the existing Dongxing
learned-environment machinery.

The robust planner should:

1. Reuse the existing one-step ridge transition/reward model utilities where
   possible.
2. Optimize a linear block-scoring policy with cross-entropy method (CEM), as
   in `dongxing_multistep_mbrl_policy.py`.
3. Train or select policy weights over a small ensemble of Dongxing scenarios
   rather than one base environment.
4. Evaluate the frozen policy in the real `GenericCountyEnv` for every held-out
   scenario.

The design target is not to invent a large new RL architecture. The target is a
reviewer-defensible robust planning variant that shows whether the learned
environment can support scenario-aware local planning.

### 3. Policy Comparison Set

Evaluate the robust planner against the current Dongxing references:

- `random`
- `dynamic_slope_gap`
- `scalarized_default`
- local learned preference policy
- one-step learned model-based policy
- held-out one-step scoring optimization
- existing multi-step learned-environment policy
- new scenario-robust learned-environment policy

For stochastic random policies, report seed variation within scenario. For
deterministic policies, report scenario variation and explicitly mark seed
variation as not meaningful.

### 4. Output Artifacts

Add one main result file:

- `paper7/results/full_rigor/dongxing_scenario_robustness.json`

The JSON should include:

- scenario definitions;
- policy list;
- per-policy, per-scenario runs;
- summary by policy across scenarios;
- worst-case reward and slope outcomes;
- comparison against `scalarized_default`, `baimu_aware`, and existing
  model-based policies;
- flags such as `deterministic_seed_repetition_avoided`,
  `scenario_count`, and `policy_transfer_tested=false`;
- a claim boundary stating that this is local Dongxing robustness evidence, not
  direct cross-county transfer.

### 5. Audit And Ledger Integration

Update the evidence chain so the manuscript cannot overstate the new result.

Modify:

- `paper7/end_to_end_validation.py`
- `paper7/dongxing_full_rigor_summaries.py`
- `paper7/manuscript_evidence_ledger.py`

Add or update evidence rows for:

- `dongxing_scenario_robustness`
- `dongxing_deterministic_evaluation_boundary`

Required boundary wording:

- "scenario-based Dongxing robustness"
- "deterministic Dongxing seed repetitions are not independent replications"
- "not direct Bishan-to-Dongxing policy transfer"

### 6. Tests

Add focused tests before implementation changes:

- scenario definitions are reproducible and have stable IDs;
- slope perturbations change slopes but preserve non-negative values and parcel
  counts;
- deterministic policies are summarized across scenarios rather than as
  independent seed replications;
- the robustness JSON includes nonzero scenario variation for at least one
  deterministic policy metric;
- the end-to-end audit includes the new robustness artifact;
- the manuscript consistency tests require scenario-based wording and reject
  pseudo-replication wording.

Existing Dongxing tests should remain in place.

### 7. Manuscript Revision

Revise the Dongxing section after the artifact exists.

Changes:

- Replace ambiguous `Eval seeds` wording for deterministic Dongxing rows with
  scenario counts and deterministic episode notes.
- Add a compact robustness table or paragraph reporting mean and worst-case
  performance across Dongxing scenarios.
- Reframe the Dongxing claim as:

  > Dongxing supports a full-reward local counterpart and scenario-based
  > robustness evidence for local learned-environment planning; it remains
  > outside the scope of direct Bishan-to-Dongxing policy transfer.

- Keep Bishan as the primary statistically tested claim.
- Keep reward-weight sensitivity as fixed-policy replay.
- Keep observational calibration as reward regularization, not causal
  identification.

### 8. Verification

Run verification in layers:

1. New scenario robustness unit tests.
2. Existing Dongxing full-rigor tests.
3. End-to-end validation tests.
4. Manuscript evidence ledger tests.
5. Manuscript claim-consistency tests.
6. Full `pytest -q`.
7. LaTeX rebuild for anonymous and signed CEUS manuscript sources if manuscript
   text changes.

The known Windows Python 3.14 / torch access-violation message after pytest is
treated as an environment warning when pytest exits with code `0`.

## Success Criteria

The work is successful when:

- Dongxing robustness is evaluated across real scenarios rather than repeated
  deterministic seeds.
- The new robust learned-environment planner has a stored, auditable result.
- The manuscript reports scenario variation and worst-case behavior.
- The evidence ledger prevents direct transfer and pseudo-replication
  overclaims.
- All focused tests and the full pytest suite pass.
- The final manuscript compiles after text revisions.

## Self-Review

- Placeholder scan: no TBD or TODO markers remain.
- Scope check: the design is focused on Dongxing robustness and one lightweight
  algorithmic extension, not open-ended transfer research.
- Internal consistency: scenario variation is the evaluation unit for
  deterministic Dongxing policies; seed variation remains valid only for
  stochastic random policies.
- Claim boundary check: the design strengthens external local evidence without
  claiming direct Bishan-to-Dongxing transfer.

