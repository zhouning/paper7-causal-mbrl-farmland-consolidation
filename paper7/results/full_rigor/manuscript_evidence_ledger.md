# Paper 7 Manuscript Evidence Ledger

Generated UTC: `2026-06-27T05:56:23.639892+00:00`
Audit path: `D:\test\paper7-causal-mbrl-farmland-consolidation\.worktrees\paper7-full-rigor-manuscript-integration\paper7\results\revision\end_to_end_validation.json`
Overall status: `supported_with_bounded_external_scope`

## Claim Map

| Claim ID | Strength | Key Metrics | Required Boundary |
|---|---|---|---|
| bishan_learned_environment_e2e | supported_strong | n_paired_seeds=15; no_cal_slope_mean=-0.975683; with_cal_slope_mean=-1.102462; improvement_pct=12.993898; transition_obs_cosine=0.999794; transition_reward_mse=1.096993 | Final policy outcomes are measured in the real environment; the learned environment is a training surrogate. |
| calibration_effect | supported_bounded | calibration_factor=0.185005; one_sided_p=0.011963; two_sided_p=0.023926; improvement_pct=12.993898; calibrated_win_count=10; uncalibrated_win_count=5 | observational reward regularization; not definitive causal identification |
| reward_scaling_comparator | supported_strong | n_grid_runs=40; n_alphas=8; best_scale=0.2; pre_specified_scale=0.185; pre_vs_best_relative_gap_pct=3.12135; pre_vs_unscaled_slope_gain_pct=22.144879 | This supports data-driven scaling, not a proof that the global factor is optimal in every state. |
| transition_surrogate_diagnostics | supported_bounded | horizon_100_mask_agreement=0.997384; horizon_100_reward_mae=0.234012; policy_induced_mask_agreement=0.997629; policy_induced_support_distance=0.012601; policy_induced_final_real_slope_pct=-1.105524; best_trajectory_source=mixed | Diagnostics support surrogate training but do not replace final real-environment evaluation. |
| planning_tradeoff_boundary | supported_bounded | slope_delta_with_minus_no_mean=-0.126779; contiguity_delta_with_minus_no_mean=-0.001977; baimu_count_delta_with_minus_no_mean=-0.2; baimu_area_delta_with_minus_no_mean=-67.208419; n_weight_settings=14 | Report slope-contiguity-baimu trade-offs; do not claim a Pareto improvement across every planning metric. |
| bishan_non_learning_baselines | supported_bounded | random_slope_mean=0.089326; slope_gap_greedy_slope=1.10799; area_weighted_greedy_slope=4.696778; immediate_slope_delta_slope=-0.15191 | This is a local baseline comparison, not a universal claim over all hand-designed rules. |
| dongxing_local_counterpart | supported_bounded | full_baseline_status=supported_as_full_real_environment_baselines; local_learned_eval_seeds=10; local_mbrl_status=supported_as_local_dongxing_mbrl_results; multistep_planning_tested=True; multistep_reward_mean=61.287306; multistep_slope_change_pct_mean=-1.882392 | Dongxing is local external-counterpart evidence, not direct Bishan-to-Dongxing policy transfer. |
| direct_transfer_boundary | structural_boundary | status=structurally_invalid_for_direct_policy_transfer; observation_dim_match=False; action_dim_match=False; direct_policy_transfer_tested=False; fine_tuning_required=True | not direct Bishan-to-Dongxing policy transfer |
| reward_weight_replay_boundary | supported_bounded | n_episodes=60; n_weight_settings=14; policy_retraining_under_all_weights=False; reward_specification_exported=True | fixed-policy replay; not proof that retrained policies are robust under every planning preference |

## Artifact Paths

### bishan_learned_environment_e2e
- `paper7\results\revision\seeds`
- `paper7\models\transition_model.pt`
- `paper7\models\training_history.json`

### calibration_effect
- `paper7\results\causal_calibration.json`
- `paper7\results\revision\causal_sensitivity_diagnostics.json`
- `paper7\results\revision\seeds`

### reward_scaling_comparator
- `paper7\results\revision\reward_scaling_comparator.json`
- `paper7\results\revision\alpha_grid\grid_results.json`

### transition_surrogate_diagnostics
- `paper7\results\revision\transition_rollout_diagnostics.json`
- `paper7\results\revision\policy_induced_diagnostics_15seed.json`
- `paper7\results\revision\trajectory_source_ablation.json`

### planning_tradeoff_boundary
- `paper7\results\revision\planning_significance_audit.json`
- `paper7\results\full_rigor\reward_weight_sensitivity.json`

### bishan_non_learning_baselines
- `paper7\results\revision\bishan_strong_baselines.json`

### dongxing_local_counterpart
- `paper7\results\full_rigor\dongxing_full_baselines.json`
- `paper7\results\full_rigor\dongxing_full_learned_policy.json`
- `paper7\results\full_rigor\dongxing_mbrl_results.json`
- `paper7\results\full_rigor\dongxing_multistep_mbrl_policy.json`
- `paper7\results\full_rigor\transfer_finetune_results.json`

### direct_transfer_boundary
- `paper7\results\full_rigor\transfer_finetune_results.json`

### reward_weight_replay_boundary
- `paper7\results\full_rigor\reward_weight_sensitivity.json`

## Required Boundary Phrases

- observational reward regularization
- not definitive causal identification
- descriptive model-free baseline comparison
- not direct Bishan-to-Dongxing policy transfer
- final real-environment evaluation
- fixed-policy replay

## Forbidden Overclaims

- universal generalization across counties
- formal superiority over all model-free RL methods
- direct transfer of Bishan policies to Dongxing
- definitive causal identification of reward effects
- transition model as a replacement for final real-environment evaluation
