"""
Paper 7 Phase 4: Causal Reward Calibration.

Estimates the Average Treatment Effect (ATT) of block investment decisions
on slope reduction using propensity score matching on trajectory data.
Then calibrates the LearnedCountyEnv's reward predictions to match.

The key insight: if the learned env systematically over/under-predicts rewards
for certain block types, the DRL agent will exploit these errors (the "reward
model exploitation" problem, seen as policy drift in Paper 4). Causal calibration
corrects this by grounding reward predictions in empirical causal estimates.

Usage:
    python paper7/causal_reward_calibration.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from county_env import K_BLOCK, K_GLOBAL_COUNTY
from learned_env import TransitionModel, LearnedCountyEnv

PAPER7_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def build_causal_dataset(trajectory_dir, policies=None):
    """Build a causal inference dataset from collected trajectories.

    For each transition (obs, action, reward, next_obs):
      - Treatment: did the agent select a "high-potential" block?
        (block with slope_gap > median = treated, else control)
      - Outcome: actual reward received
      - Confounders: global state features (budget remaining, current slope, etc.)

    Returns: DataFrame with treatment, outcome, and confounder columns.
    """
    rows = []
    for fname in sorted(os.listdir(trajectory_dir)):
        if not fname.endswith('.npz'):
            continue
        if policies and not any(fname.startswith(p) for p in policies):
            continue

        data = np.load(os.path.join(trajectory_dir, fname))
        bf = data['block_features'].astype(np.float32)  # (T, N, 17)
        gf = data['global_features']                     # (T, K_G)
        actions = data['actions']                         # (T,)
        rewards = data['rewards']                         # (T,)
        n_blocks = int(data['n_blocks'])

        for t in range(len(actions)):
            action = int(actions[t])
            selected_block = bf[t, action]  # (17,) features of selected block

            # Treatment: high slope gap (feature 3 = best_swap_gain_norm)
            median_gap = np.median(bf[t, :, 3])
            treatment = 1 if selected_block[3] > median_gap else 0

            # Confounder: global features + selected block features
            row = {
                'treatment': treatment,
                'outcome': float(rewards[t]),
                # Global confounders
                'budget_remaining': float(gf[t, 0]),
                'global_slope': float(gf[t, 1]),
                'global_cont': float(gf[t, 2]),
                'step_frac': float(gf[t, 3]),
                'slope_improvement': float(gf[t, 4]),
                # Selected block confounders
                'block_farm_slope': float(selected_block[0]),
                'block_forest_slope': float(selected_block[1]),
                'block_slope_gap': float(selected_block[2]),
                'block_swap_potential': float(selected_block[9]),
                'block_invested': float(selected_block[16]),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"Causal dataset: {len(df)} observations, "
          f"treated={df['treatment'].sum()}, control={(1-df['treatment']).sum()}")
    return df


def estimate_att_simple(df):
    """Estimate ATT using simple propensity score stratification.

    Uses sklearn GradientBoosting for propensity scores + stratified matching.
    No dependency on the external causal_inference.py module.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict

    confounders = ['budget_remaining', 'global_slope', 'global_cont', 'step_frac',
                   'slope_improvement', 'block_farm_slope', 'block_forest_slope',
                   'block_swap_potential', 'block_invested']

    X = df[confounders].values
    T = df['treatment'].values
    Y = df['outcome'].values

    # Propensity scores via gradient boosting
    ps_model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
    ps = cross_val_predict(ps_model, X, T, cv=5, method='predict_proba')[:, 1]

    # Trim extreme propensity scores
    mask = (ps > 0.05) & (ps < 0.95)
    ps_trimmed = ps[mask]
    T_trimmed = T[mask]
    Y_trimmed = Y[mask]
    print(f"  Trimmed: {mask.sum()}/{len(mask)} observations (PS in [0.05, 0.95])")

    # Stratified ATT: divide into 5 PS strata
    n_strata = 5
    strata_bounds = np.quantile(ps_trimmed, np.linspace(0, 1, n_strata + 1))
    strata_effects = []

    for i in range(n_strata):
        lo, hi = strata_bounds[i], strata_bounds[i + 1]
        in_stratum = (ps_trimmed >= lo) & (ps_trimmed <= hi)
        t_in = T_trimmed[in_stratum] == 1
        c_in = T_trimmed[in_stratum] == 0

        if t_in.sum() > 0 and c_in.sum() > 0:
            effect = Y_trimmed[in_stratum][t_in].mean() - Y_trimmed[in_stratum][c_in].mean()
            weight = t_in.sum()
            strata_effects.append((effect, weight))

    if not strata_effects:
        print("  WARNING: No valid strata, returning naive ATT")
        att = Y[T == 1].mean() - Y[T == 0].mean()
        return att, 0.0

    # Weighted ATT
    effects, weights = zip(*strata_effects)
    att = np.average(effects, weights=weights)

    # Bootstrap SE
    n_boot = 200
    boot_atts = []
    for _ in range(n_boot):
        idx = np.random.choice(len(Y_trimmed), len(Y_trimmed), replace=True)
        Yb, Tb = Y_trimmed[idx], T_trimmed[idx]
        if Tb.sum() > 0 and (1 - Tb).sum() > 0:
            boot_atts.append(Yb[Tb == 1].mean() - Yb[Tb == 0].mean())
    se = np.std(boot_atts) if boot_atts else 0.0

    return att, se


def calibrate_learned_env(transition_model_path, trajectory_dir, att, se):
    """Calibrate learned env rewards using causal ATT.

    Compares ATT (empirical) with the learned env's predicted reward
    difference between treated and control actions, then computes a
    calibration factor.
    """
    # Load transition model
    ckpt = torch.load(transition_model_path, map_location='cpu', weights_only=False)
    model = TransitionModel(
        n_blocks=int(ckpt['n_blocks']),
        k_block=int(ckpt['k_block']),
        k_global=int(ckpt['k_global']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load sample trajectories
    files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(trajectory_dir, files[0]))
    bf = data['block_features'][0].astype(np.float32)  # (N, 17)
    gf = data['global_features'][0]

    n_blocks = int(data['n_blocks'])

    # Compute learned env's predicted reward for high-potential vs low-potential blocks
    bf_t = torch.tensor(bf, dtype=torch.float32).unsqueeze(0)
    gf_t = torch.tensor(gf, dtype=torch.float32).unsqueeze(0)

    median_gap = np.median(bf[:, 3])
    high_blocks = np.where(bf[:, 3] > median_gap)[0]
    low_blocks = np.where(bf[:, 3] <= median_gap)[0]

    with torch.no_grad():
        high_rewards = []
        for b in high_blocks[:50]:  # sample up to 50
            _, _, r = model(bf_t, gf_t, torch.tensor([b]))
            high_rewards.append(r.item())

        low_rewards = []
        for b in low_blocks[:50]:
            _, _, r = model(bf_t, gf_t, torch.tensor([b]))
            low_rewards.append(r.item())

    pred_att = np.mean(high_rewards) - np.mean(low_rewards)

    # Calibration factor
    if abs(pred_att) > 1e-6:
        calibration_factor = att / pred_att
        calibration_factor = float(np.clip(calibration_factor, 0.1, 5.0))
    else:
        calibration_factor = 1.0

    return {
        'empirical_att': float(att),
        'empirical_se': float(se),
        'predicted_att': float(pred_att),
        'calibration_factor': calibration_factor,
        'high_reward_mean': float(np.mean(high_rewards)),
        'low_reward_mean': float(np.mean(low_rewards)),
        'n_high_sampled': len(high_rewards),
        'n_low_sampled': len(low_rewards),
    }


def main():
    print("=" * 60)
    print("Paper 7 Phase 4: Causal Reward Calibration")
    print("=" * 60)

    trajectory_dir = str(PAPER7_DIR / 'trajectories')
    model_path = str(PAPER7_DIR / 'models' / 'transition_model.pt')
    out_dir = str(PAPER7_DIR / 'results')

    # Step 1: Build causal dataset from trajectories
    print("\n[1/3] Building causal dataset...")
    df = build_causal_dataset(trajectory_dir)

    # Save dataset
    df_path = os.path.join(out_dir, 'causal_dataset.csv')
    df.to_csv(df_path, index=False)
    print(f"  Saved to {df_path}")

    # Step 2: Estimate ATT
    print("\n[2/3] Estimating ATT...")
    att, se = estimate_att_simple(df)
    print(f"  ATT = {att:+.4f} (SE = {se:.4f})")
    print(f"  95% CI: [{att - 1.96*se:+.4f}, {att + 1.96*se:+.4f}]")
    print(f"  Interpretation: selecting high-potential blocks yields "
          f"{att:+.4f} more reward per step on average")

    # Step 3: Calibrate learned env
    print("\n[3/3] Calibrating learned environment...")
    cal = calibrate_learned_env(model_path, trajectory_dir, att, se)
    print(f"  Empirical ATT:  {cal['empirical_att']:+.4f}")
    print(f"  Predicted ATT:  {cal['predicted_att']:+.4f}")
    print(f"  Calibration factor: {cal['calibration_factor']:.4f}")
    print(f"  (factor > 1 = learned env underestimates effect of good choices)")
    print(f"  (factor < 1 = learned env overestimates effect of good choices)")

    # Save
    cal_path = os.path.join(out_dir, 'causal_calibration.json')
    with open(cal_path, 'w') as f:
        json.dump(cal, f, indent=2)
    print(f"\n  Calibration saved to {cal_path}")

    print(f"\n{'='*60}")
    print(f"Phase 4 complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
