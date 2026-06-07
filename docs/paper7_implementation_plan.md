# Paper 7 Implementation Plan
# "Dreaming in Embedding Space: Model-Based RL for Farmland Consolidation via Causal Geospatial World Models"

## Context

The user has 6 completed papers spanning 3 research lines: DRL optimization (Paper 1-4), World Model prediction, and Causal Inference. A `dreamer_env.py` already exists at `D:/adk/data_agent/` that partially integrates World Model with DRL. The goal is to produce a 7th paper that **unifies all three lines** into a coherent model-based RL system targeting a top-tier venue (Nature Machine Intelligence / NeurIPS).

The key insight: the World Model (LatentDynamicsNet, 459K params, 80s CPU training) can serve as a Dreamer-style simulator, replacing the expensive real environment (12h A100) for DRL policy training. Causal Inference ATT estimates calibrate the World Model's scenario encoding, grounding predictions in empirical evidence.

## Critical Technical Challenge

**The domain mismatch problem**: The DRL environments (Paper 1-4) operate on **discrete parcel-level land-use states** (farmland/forest swaps within blocks), while the World Model operates on **continuous 64-dim GeoFM embeddings** at pixel level. These are fundamentally different state representations. Paper 7 must bridge this gap.

### Three possible bridge architectures (ranked by feasibility):

**Architecture A — "World Model as Auxiliary Reward" (EXISTING, extend)**
- DRL trains in the real environment (CountyLevelEnv/BlockLevelEnv)
- Every K steps, the World Model provides an auxiliary look-ahead reward
- This is what `dreamer_env.py` already implements (partially)
- Pro: Minimal changes, proven concept
- Con: Not truly "model-based RL" — still needs the real env for state transitions

**Architecture B — "World Model as Learned Environment" (NOVEL, high impact)**
- Train a **transition model** that predicts (next_obs, reward, done) from (obs, action) in the DRL's state space
- Use the LatentDynamicsNet as a **prior** or **feature extractor** to bootstrap this transition model
- DRL trains entirely in the learned environment (true Dreamer-style)
- Pro: Highest novelty, truly model-based, massive speedup potential
- Con: Requires training the transition model on trajectory data from the real env

**Architecture C — "Embedding-Space MDP" (MOST NOVEL, highest risk)**
- Reformulate the farmland optimization MDP directly in the 64-dim embedding space
- State = embedding grid, Action = spatial intervention (scenario applied to sub-region)
- World Model IS the environment — no separate transition model needed
- Pro: Maximum novelty, direct JEPA realization, cleanest theoretical story
- Con: Actions are no longer "select a block" but "apply scenario to sub-region" — very different from Paper 1-4's formulation; requires new reward definition in embedding space

### Recommended: Architecture B as primary + Architecture A as baseline

Architecture B is the sweet spot: it's a genuine model-based RL contribution that directly uses the World Model, while maintaining compatibility with the proven block-level MDP from Paper 1-4.

## Implementation Plan

### Phase 1: Data Collection — Trajectory Dataset (Week 1-2)

**Goal**: Collect state-action-reward trajectories from the real DRL environments to train the learned transition model.

**What to collect**:
- Run trained MARL and Centralized policies (from Paper 4) on all 13 townships
- Also run random and greedy policies for diversity
- For each episode step, record: `(obs_t, action_t, reward_t, obs_{t+1}, done_t, info_t)`
- Also record per-step: parcel-level land-use changes, GeoFM embeddings of affected blocks

**Files to modify/create**:
- `D:/test/collect_trajectories.py` (NEW) — run policies and save trajectory data
- Reuse: `D:/test/county_env.py`, `D:/test/county_marl_env.py`, trained model checkpoints

**Expected output**: ~50K state-action transitions across all 13 townships, multiple policies

### Phase 2: World-Model-Augmented Transition Model (Week 2-4)

**Goal**: Train a neural network that predicts (next_obs, reward) from (obs, action), bootstrapped with GeoFM embedding features from the World Model.

**Architecture (LearnedCountyEnv)**:
```
Input: [block_features (N×17) | global_features (12) | action_onehot (N) | geofm_block_embeddings (N×64)]
                                                                              ↑ from World Model
Output: [predicted_next_block_features (N×17) | predicted_next_global (12) | predicted_reward (1)]
```

The key innovation: **GeoFM block embeddings** (64-dim per block, from AlphaEarth via LatentDynamicsNet) are used as auxiliary features that encode spatial context the MDP features don't capture (land-cover history, spectral phenology, terrain microstructure).

**Files to create**:
- `D:/test/paper7/learned_env.py` (NEW) — LearnedCountyEnv that predicts state transitions
- `D:/test/paper7/train_learned_env.py` (NEW) — Train transition model on collected trajectories
- Reuse: `D:/adk/data_agent/world_model.py` — extract_embeddings(), LatentDynamicsNet

**Technical details**:
- Transition model: 3-layer MLP (input_dim → 512 → 256 → output_dim) with residual connections
- Loss: MSE on obs + MSE on reward (multi-task)
- GeoFM features: extract 64-dim embedding per block (zonal mean of parcel embeddings)
- These embeddings are **frozen** — not trained, just used as additional input features

### Phase 3: Model-Based RL Training (Week 4-6)

**Goal**: Train DRL policies using the learned environment instead of the real environment.

**Three training regimes to compare**:
1. **Real-Env Only** (baseline, = Paper 4 results): MaskablePPO on CountyLevelEnv, 500K steps, ~12h A100
2. **Learned-Env Only** (model-based): MaskablePPO on LearnedCountyEnv, same 500K steps, expected ~minutes on CPU
3. **Dyna-Style** (hybrid): Alternate between real env steps and learned env steps (ratio 1:N)

**Key metrics**:
- Policy quality: evaluate all policies on the **real** environment (final evaluation always on real env)
- Training time: wall-clock time comparison
- Sample efficiency: performance vs. number of real-env interactions

**Files to create**:
- `D:/test/paper7/train_model_based.py` (NEW) — Model-based training loop
- `D:/test/paper7/train_dyna.py` (NEW) — Dyna-style hybrid training
- Reuse: `D:/test/parcel_scoring_policy.py`, `D:/test/train_county.py` (as template)

### Phase 4: Causal Calibration Integration (Week 5-7)

**Goal**: Use Causal Inference ATT estimates to calibrate the learned environment's reward function.

**Mechanism**:
1. Run PSM/Causal Forest on historical land-use data to estimate ATT of "block investment" on slope reduction
2. Compare ATT with the learned environment's predicted reward for the same intervention
3. Compute calibration factor: α = ATT / predicted_reward
4. Adjust the learned environment's reward scaling by α

**Why this matters for the paper**: It addresses the "policy drift" problem from Paper 4 (the baimu reward trap). By grounding the learned environment's rewards in empirical causal estimates, we prevent the agent from exploiting reward model errors.

**Files to create**:
- `D:/test/paper7/causal_reward_calibration.py` (NEW) — ATT extraction + reward calibration
- Reuse: `D:/adk/data_agent/causal_inference.py` (propensity_score_matching, causal_forest_analysis)
- Reuse: `D:/adk/data_agent/causal_world_model.py` (integrate_statistical_prior pattern)

### Phase 5: Generalization via Embedding-Space Transfer (Week 6-8)

**Goal**: Demonstrate generalization without new cadastral data.

**Approach (adapted for data constraint — only Bishan available)**:
- **E6a: Intra-county leave-one-out**: Train learned env on 12 of 13 Bishan townships, evaluate on held-out township. Repeat 13 times. This tests whether the transition model generalizes across spatial structure diversity within a county.
- **E6b: Embedding-space generalization**: Extract GeoFM embeddings for 3-5 of the World Model's 17 study areas (different provinces/terrains). Show that the transition model's GeoFM-augmented features maintain predictive accuracy on embeddings from unseen regions, even though parcel-level evaluation is not possible.
- **E6c: Synthetic region generation**: Use the World Model to generate plausible future states for Bishan sub-regions under different scenarios, then test DRL policy robustness on these counterfactual environments.

**This is weaker than true cross-county transfer but still publishable** — frame as "embedding-space generalization" rather than "zero-shot policy transfer". Acknowledge the limitation and position cross-county validation as future work.

### Phase 6: Comprehensive Experiments (Week 7-10)

**Experiment matrix**:

| Experiment | Purpose | Metrics |
|-----------|---------|---------|
| E1: Model accuracy | Learned env vs real env state prediction | MSE, cosine sim on obs/reward |
| E2: Policy quality | Model-based vs real-env-trained policies | Slope%, cont, baimu (eval on real env) |
| E3: Training efficiency | Wall-clock time, GPU hours, sample efficiency | Time to threshold performance |
| E4: Ablation: GeoFM features | With vs without 64-dim embedding augmentation | Policy quality delta |
| E5: Ablation: Causal calibration | With vs without ATT reward calibration | Policy drift rate |
| E6: Generalization | Leave-one-township-out + embedding-space transfer | Slope% on held-out township |
| E7: Dyna ratio sweep | Vary real:model ratio (1:1, 1:5, 1:10, 1:50) | Pareto front: quality vs cost |
| E8: Scaling | Township vs county scale model-based RL | Convergence rate, final quality |

**Statistical rigor**: 5 seeds per condition, Mann-Whitney U tests, report mean±std

### Phase 7: Paper Writing (Week 9-12)

**Target**: Nature Machine Intelligence (primary) / NeurIPS (backup)

**Paper structure**:
1. Introduction: The gap between model-free RL (expensive, brittle) and spatial planning needs
2. Related Work: World models in RL (Dreamer, MuZero) + GeoFM + spatial optimization
3. Method: Learned environment with GeoFM embedding augmentation + causal reward calibration
4. Experiments: E1-E8 with comprehensive ablations
5. Discussion: When model-based > model-free, generalization analysis, policy drift mitigation
6. Conclusions: Unified framework connecting prediction, causation, and optimization

**Key claims to support**:
- C1: Model-based RL achieves comparable policy quality with 100x less compute
- C2: GeoFM embeddings improve transition model accuracy (ablation E4)
- C3: Causal calibration reduces policy drift (ablation E5)
- C4: The framework enables cross-region transfer (E6)

## Critical Files Inventory

### Existing (to reuse):
- `D:/test/county_env.py` — Real environment (evaluation)
- `D:/test/county_marl_env.py` — MARL environment (evaluation)
- `D:/test/parcel_scoring_policy.py` — Policy architecture
- `D:/test/block_level_env.py` — Township environment
- `D:/adk/data_agent/world_model.py` — LatentDynamicsNet, embeddings, decoder
- `D:/adk/data_agent/dreamer_env.py` — Existing partial integration (Architecture A baseline)
- `D:/adk/data_agent/causal_inference.py` — PSM, Causal Forest tools
- `D:/adk/data_agent/causal_world_model.py` — ATT calibration bridge
- `D:/test/a100/paper4_v7/` — Trained model checkpoints

### New (to create):
- `D:/test/paper7/collect_trajectories.py` — Trajectory dataset collection
- `D:/test/paper7/learned_env.py` — LearnedCountyEnv (transition model)
- `D:/test/paper7/train_learned_env.py` — Train transition model
- `D:/test/paper7/train_model_based.py` — Model-based RL training
- `D:/test/paper7/train_dyna.py` — Dyna-style hybrid training
- `D:/test/paper7/causal_reward_calibration.py` — ATT calibration
- `D:/test/paper7/transfer_experiment.py` — Cross-region experiments
- `D:/test/paper7/analyze_paper7.py` — Results aggregation and figures
- `D:/test/paper7/paper7_dreamer_farmland.tex` — Manuscript

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|:-:|:-:|-----------|
| Learned env too inaccurate for good policies | Medium | High | Dyna-style hybrid as fallback |
| GeoFM embeddings don't help transition model | Low | Medium | Ablation E4 documents this honestly |
| No cross-region data available | ~~Medium~~ **Confirmed** | Medium | Leave-one-township-out + embedding generalization test |
| Causal calibration doesn't reduce drift | Low | Low | Report null result honestly |
| Reviewers demand comparison with Dreamer/MuZero | High | Medium | Include simplified Dreamer baseline |

## Compute Budget (50-100 A100 GPU-hours)

| Phase | Task | GPU-hours | Notes |
|-------|------|:---------:|-------|
| 1 | Trajectory collection (inference only) | 2 | Run trained models on real env |
| 2 | Transition model training | 0 (CPU) | Small MLP, ~1h CPU |
| 3 | Model-based RL training | 0 (CPU) | Main speedup claim! |
| 3 | Real-env RL baselines (5 seeds) | 60 | 12h × 5 seeds (reuse Paper 4 for 4 seeds) |
| 4 | Causal inference ATT estimation | 0 (CPU) | Statistical methods |
| 5 | Generalization experiments | 5 | Inference only on held-out townships |
| 6 | Full experiment matrix evaluation | 15 | Policy eval on real env (fast) |
| - | **Total** | **~82** | Within 50-100h budget |

**Key optimization**: Reuse Paper 4's trained models (4 MARL seeds + 5 centralized seeds) as "real-env baselines". Only need to train 1 additional real-env seed for matched comparison. This saves ~100 GPU-hours.

## Verification

1. Transition model accuracy: >0.95 cosine similarity between predicted and actual next_obs
2. Model-based policy quality: within 20% of real-env-trained policy on slope reduction
3. Training speedup: >10x wall-clock reduction (target: 100x)
4. At least one ablation (E4 or E5) shows statistically significant improvement
5. Manuscript passes internal review against Nature Machine Intelligence author guidelines
