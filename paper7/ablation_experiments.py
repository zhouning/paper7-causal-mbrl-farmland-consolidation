"""
Paper 7 Phase 5-6: Ablation experiments.

E4: With vs without GeoFM embedding augmentation (future — needs GeoFM extraction)
E5: With vs without causal reward calibration
E7: Training data scaling (1K, 3K, 6K, 12K transitions)

Usage:
    python paper7/ablation_experiments.py
"""

import os
import sys
import json
import time
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY
from parcel_scoring_policy import ParcelScoringPolicy
from learned_env import TransitionModel, LearnedCountyEnv, TrajectoryDataset, train_transition_model

PAPER7_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


def load_transition_model(model_path):
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = TransitionModel(
        n_blocks=int(ckpt['n_blocks']),
        k_block=int(ckpt['k_block']),
        k_global=int(ckpt['k_global']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def create_learned_env(tm, traj_dir, reward_scale=1.0):
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
    # Store reward scale for calibration
    env._reward_scale = reward_scale
    original_step = env.step

    def calibrated_step(action):
        obs, reward, done, trunc, info = original_step(action)
        return obs, reward * reward_scale, done, trunc, info

    if abs(reward_scale - 1.0) > 1e-6:
        env.step = calibrated_step

    return env


def train_and_eval(env, seed, timesteps, label, out_dir):
    """Train MaskablePPO on given env, return real-env eval result."""
    env = Monitor(env)
    n_blocks = env.observation_space.shape[0] // K_BLOCK  # approximate

    model = MaskablePPO(
        ParcelScoringPolicy, env,
        learning_rate=3e-4, n_steps=256, batch_size=128,
        n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, seed=seed, verbose=0,
        policy_kwargs=dict(
            k_parcel=K_BLOCK, k_global=K_GLOBAL_COUNTY,
            scorer_hiddens=[128, 64], value_hiddens=[64, 32],
        ),
    )

    t0 = time.time()
    model.learn(total_timesteps=timesteps)
    elapsed = time.time() - t0

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
        'training_time_s': elapsed,
        'reward_real': total_reward,
        'slope_change_pct': info.get('slope_change_pct', 0),
        'cont_change': info.get('cont_change', 0),
        'baimu_count_change': info.get('baimu_count_change', 0),
    }
    print(f"  {label} seed{seed}: slope={result['slope_change_pct']:+.3f}%, "
          f"time={elapsed:.0f}s")
    return result


def run_ablation_e5_calibration():
    """E5: With vs without causal reward calibration."""
    print("\n" + "=" * 60)
    print("E5: Causal Reward Calibration Ablation")
    print("=" * 60)

    traj_dir = str(PAPER7_DIR / 'trajectories')
    model_path = str(PAPER7_DIR / 'models' / 'transition_model.pt')
    out_dir = str(PAPER7_DIR / 'results' / 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    tm, ckpt = load_transition_model(model_path)

    # Load calibration factor
    cal_path = str(PAPER7_DIR / 'results' / 'causal_calibration.json')
    with open(cal_path) as f:
        cal = json.load(f)
    cal_factor = cal['calibration_factor']
    print(f"  Calibration factor: {cal_factor:.4f}")

    results = []
    for seed in range(3):
        # Without calibration (baseline = scale 1.0)
        env_nocal = create_learned_env(tm, traj_dir, reward_scale=1.0)
        r = train_and_eval(env_nocal, seed, 100_000, 'no_calibration', out_dir)
        results.append(r)

        # With calibration
        env_cal = create_learned_env(tm, traj_dir, reward_scale=cal_factor)
        r = train_and_eval(env_cal, seed, 100_000, 'with_calibration', out_dir)
        results.append(r)

    # Summary
    nocal = [r for r in results if r['label'] == 'no_calibration']
    withcal = [r for r in results if r['label'] == 'with_calibration']

    print(f"\n  No calibration:   slope = {np.mean([r['slope_change_pct'] for r in nocal]):+.3f}% "
          f"± {np.std([r['slope_change_pct'] for r in nocal]):.3f}%")
    print(f"  With calibration: slope = {np.mean([r['slope_change_pct'] for r in withcal]):+.3f}% "
          f"± {np.std([r['slope_change_pct'] for r in withcal]):.3f}%")

    return results


def run_ablation_e7_data_scaling():
    """E7: Training data scaling — how much trajectory data is needed?"""
    print("\n" + "=" * 60)
    print("E7: Training Data Scaling Ablation")
    print("=" * 60)

    traj_dir = str(PAPER7_DIR / 'trajectories')
    out_dir = str(PAPER7_DIR / 'results' / 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    # Train transition models with different data amounts
    # Use subsets of trajectories: 1 file (2K), 2 files (4K), all 6 files (12K)
    configs = [
        ('2K', ['greedy']),    # 3 greedy files = 6K, but we'll limit epochs
        ('12K', None),         # all files = 12K
    ]

    results = []
    for label, policies in configs:
        print(f"\n  --- {label} transitions ---")

        # Train transition model
        tm, history, ds = train_transition_model(
            data_dir=traj_dir, policies=policies, epochs=30, batch_size=32)

        # Create learned env
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

        r = train_and_eval(env, seed=0, timesteps=100_000,
                           label=f'data_{label}', out_dir=out_dir)
        r['n_transitions'] = len(ds)
        r['val_cosine'] = history['val_obs_cosine'][-1]
        results.append(r)

    print(f"\n  Data scaling results:")
    for r in results:
        print(f"    {r['label']}: slope={r['slope_change_pct']:+.3f}%, "
              f"val_cos={r['val_cosine']:.6f}, n={r['n_transitions']}")

    return results


def main():
    out_dir = str(PAPER7_DIR / 'results' / 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}

    # E5: Calibration ablation
    e5 = run_ablation_e5_calibration()
    all_results['E5_calibration'] = e5

    # E7: Data scaling
    e7 = run_ablation_e7_data_scaling()
    all_results['E7_data_scaling'] = e7

    # Save all
    with open(os.path.join(out_dir, 'ablation_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("All ablation experiments complete!")
    print(f"Results saved to {out_dir}/ablation_results.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
