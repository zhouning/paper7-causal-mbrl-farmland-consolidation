# Policy-Induced Diagnostics Evidence Design

## Purpose

Strengthen Paper 7's experimental evidence before making further submission edits. The next batch extends the existing policy-induced learned-vs-real diagnostic from three calibrated policy checkpoints to all 15 calibrated seeds already used in the main Bishan seed experiment.

This batch tests whether trained policies that were optimized in the learned environment remain close to the recorded real-environment trajectory support when replayed synchronously against the real parcel-simulation environment.

## Scope

Included:

- Reuse the existing Bishan transition model at `paper7/models/transition_model.pt`.
- Reuse recorded trajectories in `paper7/trajectories`.
- Reuse calibrated policy checkpoints `paper7/results/revision/seeds/with_cal_model_seed0.zip` through `with_cal_model_seed14.zip`.
- Run synchronized learned-vs-real rollouts for each checkpoint.
- Report per-seed and aggregate diagnostics:
  - selected-block MAE
  - all-block MAE
  - global-feature MAE
  - raw reward MAE
  - calibrated reward MAE
  - action-mask agreement
  - support distance to recorded trajectory states
  - final real-environment slope change
- Update the evidence audit only after the diagnostic output is generated and validated.

Excluded:

- No Dongxing learned-policy transfer in this batch.
- No new transition-model architecture.
- No retraining of the 15 calibrated policies.
- No manuscript claim expansion until the diagnostic succeeds or fails clearly.

## Current Evidence Baseline

The current diagnostic file is `paper7/results/revision/policy_induced_diagnostics.json`. It covers three calibrated checkpoints and reports:

- mean action-mask agreement: 0.997491
- mean support distance: 0.011306
- q95 support distance: 0.016308
- mean global MAE: 0.052100
- mean calibrated reward MAE: 0.111302
- mean final real-environment slope change: -1.109010%

The manuscript currently describes this honestly as a three-checkpoint diagnostic. The next batch should either support replacing this with a 15-seed statement or preserve the narrower wording if the expanded result is unstable.

## Acceptance Criteria

The expanded diagnostic supports a stronger evidence claim if all of the following hold across 15 calibrated seeds:

- All 15 policy checkpoints are found and evaluated.
- Each episode reaches the same 100-step budget horizon or records a clear termination reason.
- Aggregate action-mask agreement remains at or above 0.995.
- Aggregate q95 support distance remains below 0.05.
- Aggregate calibrated reward MAE remains materially lower than raw reward MAE.
- The final real-environment slope-change distribution is consistent with the existing 15-seed calibrated policy evaluation summary.

If one or more criteria fail, the batch is still useful. The manuscript should then retain a limitation-focused interpretation, and the failure mode should be reported in the evidence audit.

## Data Flow

1. Load the trained transition model and trajectory support states.
2. For each calibrated policy checkpoint:
   - reset a learned environment and a real `CountyLevelEnv` to the same initial state convention used by the existing diagnostic;
   - select actions from the policy using learned-environment observations and action masks;
   - step both environments with the same action;
   - measure learned-vs-real state, reward, mask, and support-distance diagnostics after each step.
3. Write a JSON file with both per-step summaries and aggregate statistics.
4. Validate that the JSON includes 15 episodes and no missing metrics.
5. Update the end-to-end evidence audit classification from three-checkpoint diagnostic to 15-seed diagnostic only if validation passes.

## Testing

The batch should add focused tests for the diagnostic summarization and validation logic before changing the diagnostic script. The expensive policy rollouts can then be executed once the tests pass.

Minimum tests:

- Summary aggregation handles multiple episodes and reports min, max, mean, and q95 consistently.
- Validation rejects missing policy checkpoint entries.
- Validation rejects NaN diagnostic metrics.
- Validation accepts a complete synthetic 15-episode diagnostic payload.

## Manuscript Boundary

The manuscript may be updated only after the expanded diagnostic is generated and validated. Any text change should preserve these boundaries:

- Final policy outcomes are still measured in the real parcel-simulation environment.
- The learned environment is still an approximate training surrogate, not a stand-alone simulator for final planning outcomes.
- The expanded diagnostic addresses policy-induced distribution shift within Bishan only.
- It does not imply Dongxing learned-policy transfer.
