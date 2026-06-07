"""
Paper 7 Revision: Batch training for expanded seed experiments.

Runs 15 seeds × 2 configs (no_calibration + with_calibration)
+ alpha grid search (α = 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)

Designed to run overnight — total ~15h on CPU.

Usage:
    python paper7/batch_revision.py               # run everything
    python paper7/batch_revision.py --phase seeds  # only 15-seed experiments
    python paper7/batch_revision.py --phase grid   # only alpha grid search
"""

import os
import sys
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY
from parcel_scoring_policy import ParcelScoringPolicy
from learned_env import TransitionModel, LearnedCountyEnv

PAPER7_DIR = os.path.dirname(os.path.abspath(__file__))
REVISION_DIR = os.path.join(PAPER7_DIR, 'results', 'revision')


def load_transition_model():
    model_path = os.path.join(PAPER7_DIR, 'models', 'transition_model.pt')
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = TransitionModel(
        n_blocks=int(ckpt['n_blocks']),
        k_block=int(ckpt['k_block']),
        k_global=int(ckpt['k_global']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def create_env(tm, reward_scale=1.0):
    traj_dir = os.path.join(PAPER7_DIR, 'trajectories')
    files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(traj_dir, files[0]))
    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]

    env = LearnedCountyEnv(
        transition_model=tm,
        initial_block_features=init_bf,
        initial_global_features=init_gf,
        n_blocks=int(data['n_blocks']),
        k_block=int(data['k_block']),
        k_global=int(data['k_global']),
        max_steps=100,
    )

    if abs(reward_scale - 1.0) > 1e-6:
        original_step = env.step
        def scaled_step(action):
            obs, reward, done, trunc, info = original_step(action)
            return obs, reward * reward_scale, done, trunc, info
        env.step = scaled_step

    return env


def train_one(seed, reward_scale, label, out_dir, timesteps=100_000):
    """Train one seed, evaluate on real env, return result dict."""
    out_path = os.path.join(out_dir, f'{label}_eval_seed{seed}.json')
    if os.path.exists(out_path):
        print(f"  [{label} seed{seed}] Already done, skipping")
        with open(out_path) as f:
            return json.load(f)

    tm, ckpt = load_transition_model()
    env = create_env(tm, reward_scale=reward_scale)
    env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy, env,
        learning_rate=3e-4, n_steps=256, batch_size=128,
        n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, seed=seed, verbose=0,
        policy_kwargs=dict(
            k_parcel=int(ckpt['k_block']), k_global=int(ckpt['k_global']),
            scorer_hiddens=[128, 64], value_hiddens=[64, 32],
        ),
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    train_time = time.time() - t0

    # Save model
    model_path = os.path.join(out_dir, f'{label}_model_seed{seed}.zip')
    model.save(model_path)

    # Evaluate on real env
    real_env = CountyLevelEnv(total_budget=500, swaps_per_step=5)
    model_eval = MaskablePPO.load(model_path, env=real_env)
    obs, _ = real_env.reset()
    done = False
    total_reward = 0
    while not done:
        mask = real_env.action_masks()
        action, _ = model_eval.predict(obs, action_masks=mask, deterministic=True)
        obs, r, terminated, truncated, info = real_env.step(int(action))
        done = terminated or truncated
        total_reward += r

    result = {
        'label': label,
        'seed': seed,
        'reward_scale': reward_scale,
        'training_time_s': train_time,
        'reward_real': total_reward,
        'slope_change_pct': info.get('slope_change_pct', 0),
        'cont_change': info.get('cont_change', 0),
        'baimu_count_change': info.get('baimu_count_change', 0),
        'baimu_area_change_ha': info.get('baimu_area_change_ha', 0),
    }

    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"  [{label} seed{seed}] slope={result['slope_change_pct']:+.3f}%, "
          f"time={train_time:.0f}s")
    return result


def run_seeds(n_seeds=15):
    """Run 15 seeds × 2 configs (no_cal + with_cal α=0.185)."""
    print("=" * 60)
    print(f"Phase 1: {n_seeds}-seed experiments")
    print("=" * 60)

    out_dir = os.path.join(REVISION_DIR, 'seeds')
    os.makedirs(out_dir, exist_ok=True)

    cal_factor = 0.185  # from causal calibration

    results = []
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed}/{n_seeds-1} ---")
        # No calibration
        r = train_one(seed, 1.0, 'no_cal', out_dir)
        results.append(r)
        # With calibration
        r = train_one(seed, cal_factor, 'with_cal', out_dir)
        results.append(r)

    # Summary
    nocal = [r for r in results if r['label'] == 'no_cal']
    withcal = [r for r in results if r['label'] == 'with_cal']
    nc_slopes = [r['slope_change_pct'] for r in nocal]
    wc_slopes = [r['slope_change_pct'] for r in withcal]

    print(f"\n{'='*60}")
    print(f"15-Seed Results:")
    print(f"  No calibration:   {np.mean(nc_slopes):+.3f}% ± {np.std(nc_slopes):.3f}%")
    print(f"  With calibration: {np.mean(wc_slopes):+.3f}% ± {np.std(wc_slopes):.3f}%")

    # Statistical test
    from scipy import stats
    U, p = stats.mannwhitneyu(wc_slopes, nc_slopes, alternative='less')
    print(f"  Mann-Whitney U: U={U:.0f}, p={p:.6f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'}")
    print(f"{'='*60}")

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({
            'no_cal_mean': float(np.mean(nc_slopes)),
            'no_cal_std': float(np.std(nc_slopes)),
            'with_cal_mean': float(np.mean(wc_slopes)),
            'with_cal_std': float(np.std(wc_slopes)),
            'mann_whitney_U': float(U),
            'mann_whitney_p': float(p),
            'n_seeds': n_seeds,
        }, f, indent=2)

    return results


def run_alpha_grid():
    """Grid search over reward scaling factor α."""
    print("\n" + "=" * 60)
    print("Phase 2: Alpha grid search")
    print("=" * 60)

    out_dir = os.path.join(REVISION_DIR, 'alpha_grid')
    os.makedirs(out_dir, exist_ok=True)

    alphas = [0.1, 0.15, 0.185, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_seeds_per_alpha = 5

    results = []
    for alpha in alphas:
        print(f"\n--- α = {alpha} ---")
        for seed in range(n_seeds_per_alpha):
            label = f'alpha_{alpha:.3f}'
            r = train_one(seed, alpha, label, out_dir)
            results.append(r)

    # Summary table
    print(f"\n{'='*60}")
    print(f"Alpha Grid Results ({n_seeds_per_alpha} seeds each):")
    print(f"{'Alpha':>8} {'Mean Slope':>12} {'Std':>8}")
    print(f"{'-'*30}")
    for alpha in alphas:
        slopes = [r['slope_change_pct'] for r in results
                  if abs(r['reward_scale'] - alpha) < 0.001]
        if slopes:
            marker = ' <-- causal' if abs(alpha - 0.185) < 0.01 else ''
            print(f"{alpha:>8.3f} {np.mean(slopes):>+11.3f}% {np.std(slopes):>7.3f}%{marker}")
    print(f"{'='*60}")

    with open(os.path.join(out_dir, 'grid_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['seeds', 'grid', 'all'], default='all')
    parser.add_argument('--n_seeds', type=int, default=15)
    args = parser.parse_args()

    os.makedirs(REVISION_DIR, exist_ok=True)

    t0 = time.time()

    if args.phase in ('seeds', 'all'):
        run_seeds(args.n_seeds)

    if args.phase in ('grid', 'all'):
        run_alpha_grid()

    elapsed = time.time() - t0
    print(f"\n\nTotal elapsed: {elapsed/3600:.1f} hours")


if __name__ == '__main__':
    main()
