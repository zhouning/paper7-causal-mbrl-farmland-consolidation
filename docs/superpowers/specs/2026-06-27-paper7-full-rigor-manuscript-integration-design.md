# Paper 7 Full-Rigor Manuscript Integration Design

## Goal

Build the next Paper 7 revision around the strongest verified evidence already
in the repository, while adding only targeted diagnostics when an integration
gap blocks a defensible manuscript claim.

The revision should produce a stronger paper, not a broader but less coherent
one. The manuscript should read as an evidence-first methods paper about
treatment-effect-informed learned environments for county-scale farmland
consolidation planning.

## Current Starting Point

The repository already supports a substantial evidence chain:

- Bishan learned-environment MBRL with final real-environment evaluation.
- Fifteen paired calibrated and uncalibrated Bishan seeds.
- Exact paired sign-flip test for calibration effects:
  - one-sided `p=0.012`
  - two-sided `p=0.024`
- Reward-scaling comparator showing the pre-specified observational
  calibration factor is close to the empirical reward-scale optimum.
- Multi-step transition rollout diagnostics.
- Policy-induced learned-vs-real diagnostics for trained calibrated policies.
- Bishan strong non-learning baselines.
- Dongxing full-reward local baselines.
- Dongxing local full-reward learned-policy actionability.
- Dongxing local one-step and multi-step learned-environment evidence.
- Explicit classification that direct Bishan-to-Dongxing policy transfer is
  structurally invalid without adapter-level changes.
- Fixed-policy reward-component replay and a machine-readable reward
  specification.

The current repository status suggests that the main bottleneck is not the lack
of raw experiments. The bottleneck is that the manuscript has not yet been
rebuilt around this stronger but bounded evidence chain.

## Recommended Route

Use a three-track integration strategy.

### Track A: Evidence Audit First

Create a compact manuscript-facing evidence ledger from the existing JSON
artifacts. Each row should map one possible manuscript claim to:

- evidence artifact path;
- supported metric values;
- statistical method, if any;
- claim strength;
- required boundary wording;
- whether the result belongs in the main manuscript or supplement.

This ledger becomes the source of truth for manuscript rewriting. The manuscript
must not introduce a stronger claim than the ledger supports.

### Track B: Manuscript Re-architecture

Rebuild the paper around a clearer argument:

> In county-scale farmland consolidation planning, we show that a
> treatment-effect-informed learned environment can reduce expensive
> parcel-simulation training and mitigate learned-reward exploitation, supported
> by paired Bishan real-environment evaluation and bounded Dongxing local
> counterpart evidence; direct cross-county policy transfer remains structurally
> unsupported without adapters.

The manuscript should use this section logic:

1. **Introduction**: expensive spatial planning simulation and learned-reward
   exploitation as the practical and methodological gap.
2. **Methods**: real environment, learned environment, observational
   reward-regularization, policy training, and real-environment evaluation.
3. **Bishan results**: primary end-to-end evidence, paired calibration effect,
   reward-scaling comparator, rollout and policy-induced diagnostics.
4. **Reward and planning diagnostics**: reward-component sensitivity, planning
   metric trade-offs, strong non-learning baselines.
5. **Dongxing local counterpart**: external full-reward feasibility, local
   learned-environment evidence, and the structural reason direct policy
   transfer is not claimed.
6. **Discussion**: why the approach works, when it may fail, and what future
   adapter/fine-tuning work would need to prove.

### Track C: Targeted Gap Closure Only

Do not launch open-ended model development. Add new code or experiments only if
the evidence ledger exposes a concrete manuscript-blocking gap.

Allowed targeted additions:

- A generated manuscript evidence table from existing JSON artifacts.
- A consistency test that prevents manuscript text from drifting beyond the
  evidence ledger.
- A short diagnostic summary if an existing JSON artifact is too large or too
  hard to cite directly.
- A figure or table assembly script for existing results.
- A narrowly scoped verification run if an existing artifact is stale or
  internally inconsistent.

Non-goals:

- Do not add a new RL algorithm.
- Do not claim direct Bishan-to-Dongxing policy transfer.
- Do not retrain all reward-weight settings.
- Do not hide Dongxing negative or mixed results.
- Do not rewrite claims before the ledger is complete.

## Manuscript Claim Boundaries

The next manuscript should use the following claim boundaries.

### Supported Strong Claims

- A learned environment can serve as a practical training surrogate for Bishan
  county-scale farmland consolidation, with final outcomes evaluated in the
  real parcel-simulation environment.
- Treatment-effect-informed reward scaling improves Bishan learned-environment
  policy training under paired seed evaluation.
- The pre-specified observational calibration factor is close to the empirical
  reward-scale optimum, reducing the need for blind reward-scale search.
- Policy-induced diagnostics support the learned environment as an approximate
  surrogate, not as a standalone final simulator.
- Dongxing supports a full-reward local counterpart and local learned-environment
  evidence.

### Bounded Or Descriptive Claims

- Comparisons with model-free Bishan baselines are descriptive because training
  budgets and seed structures are not fully matched.
- Observational calibration is reward regularization informed by treatment-effect
  estimates, not definitive causal identification.
- Dongxing evidence is local external-counterpart evidence, not direct
  cross-county policy transfer.
- Reward-weight sensitivity is fixed-policy replay, not proof that retrained
  policies are robust under every planning preference.

### Claims To Avoid

- Universal generalization across counties.
- Formal superiority over all model-free RL methods.
- Direct transfer of Bishan policies to Dongxing.
- Causal identification of reward effects without qualification.
- The transition model as a replacement for final real-environment evaluation.

## Deliverables

1. `paper7/results/full_rigor/manuscript_evidence_ledger.json`
   - Machine-readable claim-to-evidence ledger.
2. `paper7/results/full_rigor/manuscript_evidence_ledger.md`
   - Human-readable ledger for manuscript writing.
3. Tests that verify:
   - all ledger artifact paths exist;
   - key audited statistics are present;
   - forbidden overclaims are absent from the manuscript;
   - required boundary language is present.
4. Revised manuscript:
   - `submission/ceus/01_main_document_anonymous/manuscript.tex`
   - `submission/ceus/06_latex_source_editable/manuscript_anonymous_copy.tex`
   - `submission/ceus/06_latex_source_editable/manuscript_signed.tex`
5. Revised highlights and cover letter after the manuscript has stabilized.
6. Rebuilt PDFs and source zip only after tests pass.

## Testing And Verification

Run verification in layers:

1. Evidence ledger tests.
2. Manuscript claim-consistency tests.
3. Focused full-rigor artifact tests:
   - end-to-end validation;
   - transition rollout diagnostics;
   - policy-induced diagnostics;
   - reward component and reward-weight sensitivity;
   - Dongxing full baselines;
   - Dongxing local MBRL result classification.
4. Full pytest suite.
5. LaTeX compile for anonymous and signed manuscripts.

The known Python 3.14 / torch access-violation message after pytest is an
environment warning when pytest exits with code `0`; the pass/fail gate remains
the process exit code.

## Success Criteria

The work is successful when:

- every major manuscript claim traces to a stored artifact;
- manuscript p-values and result numbers match the audit artifacts;
- the paper has one clear argument rather than a list of experiments;
- Dongxing is framed as bounded external-counterpart evidence;
- the manuscript explicitly states the limits of calibration, transfer, and
  reward-weight replay;
- tests prevent reintroducing stale or overstated claims;
- the final PDFs compile without blocking LaTeX errors.

## Self-Review

- Placeholder scan: no placeholder or TBD language remains.
- Scope check: the design is focused on integration and targeted diagnostics,
  not open-ended algorithm expansion.
- Consistency check: the design preserves the evidence-first rule from the
  existing full-rigor plan.
- Ambiguity check: supported, bounded, and forbidden claims are separated so an
  implementation plan can enforce them with tests.
