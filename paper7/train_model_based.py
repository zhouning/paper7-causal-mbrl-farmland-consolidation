"""
Paper 7 Phase 3: Train MaskablePPO on LearnedCountyEnv (model-based RL).

Trains DRL policies using the neural transition model instead of the real environment.
Then evaluates on the REAL CountyLevelEnv for honest comparison with Paper 4 baselines.

Three training modes:
  1. learned-only: train entirely on LearnedCountyEnv (CPU, fast)
  2. real-only: train on real CountyLevelEnv (baseline, = Paper 4)
  3. dyna: alternate real and learned env steps

Usage:
    python paper7/train_model_based.py --mode learned --seed 0 --timesteps 500000
    python paper7/train_model_based.py --mode learned --seed 0 --eval-only --model_path path/to/model.zip
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY
from parcel_scoring_policy import ParcelScoringPolicy
from learned_env import TransitionModel, LearnedCountyEnv

PAPER7_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = PAPER7_DIR / 'results'


class SimpleCallback(BaseCallback):
    """Lightweight training callback for model-based RL."""

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.episode_data = []

    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_data.append({
                    'reward': float(info['episode']['r']),
                    'length': int(info['episode']['l']),
                    'timestep': self.num_timesteps,
                })
                if len(self.episode_data) % 200 == 0:
                    recent = self.episode_data[-50:]
                    avg_r = np.mean([d['reward'] for d in recent])
                    print(f"    Step {self.num_timesteps:>7d}: "
                          f"ep={len(self.episode_data)}, avg_reward={avg_r:.2f}")
        return True

    def _on_training_end(self):
        with open(self.log_path, 'w') as f:
            json.dump(self.episode_data, f)
        print(f"  Log saved: {self.log_path} ({len(self.episode_data)} episodes)")


def load_transition_model(model_path):
    """Load trained TransitionModel from checkpoint."""
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    model = TransitionModel(
        n_blocks=int(ckpt['n_blocks']),
        k_block=int(ckpt['k_block']),
        k_global=int(ckpt['k_global']),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt


def create_learned_env(transition_model, trajectory_dir):
    """Create LearnedCountyEnv from transition model + initial state from trajectories."""
    # Load initial state from first trajectory file
    files = sorted([f for f in os.listdir(trajectory_dir) if f.endswith('.npz')])
    data = np.load(os.path.join(trajectory_dir, files[0]), allow_pickle=False)

    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]
    n_blocks = int(data['n_blocks'])
    k_block = int(data['k_block'])
    k_global = int(data['k_global'])

    env = LearnedCountyEnv(
        transition_model=transition_model,
        initial_block_features=init_bf,
        initial_global_features=init_gf,
        n_blocks=n_blocks,
        k_block=k_block,
        k_global=k_global,
        max_steps=100,
    )
    return env


def evaluate_on_real_env(model_path, n_eval=5, budget=500):
    """Evaluate a trained policy on the REAL CountyLevelEnv."""
    print("\n  Evaluating on REAL CountyLevelEnv...")
    env = CountyLevelEnv(total_budget=budget, swaps_per_step=5)
    model = MaskablePPO.load(model_path, env=env)

    results = []
    for ep in range(n_eval):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            mask = env.action_masks()
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            obs, r, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            total_reward += r

        results.append({
            'reward': total_reward,
            'slope_change_pct': info.get('slope_change_pct', 0),
            'cont_change': info.get('cont_change', 0),
            'baimu_count_change': info.get('baimu_count_change', 0),
            'baimu_area_change_ha': info.get('baimu_area_change_ha', 0),
        })
        print(f"    Eval {ep}: reward={total_reward:.1f}, "
              f"slope={info.get('slope_change_pct', 0):+.3f}%")

    # Aggregate
    mean_slope = np.mean([r['slope_change_pct'] for r in results])
    std_slope = np.std([r['slope_change_pct'] for r in results])
    mean_cont = np.mean([r['cont_change'] for r in results])
    print(f"  Real-env eval: slope={mean_slope:+.3f}%+-{std_slope:.3f}%, "
          f"cont={mean_cont:+.4f}")
    return results


def train_learned(seed, timesteps, transition_model_path, trajectory_dir, out_dir):
    """Train MaskablePPO entirely on LearnedCountyEnv."""
    print(f"\n{'='*60}")
    print(f"  Model-Based RL Training (LearnedCountyEnv)")
    print(f"  Seed: {seed}, Timesteps: {timesteps:,}")
    print(f"{'='*60}")

    # Load transition model
    tm, ckpt = load_transition_model(transition_model_path)
    n_blocks = int(ckpt['n_blocks'])
    print(f"  TransitionModel loaded: {sum(p.numel() for p in tm.parameters()):,} params")

    # Create learned env
    env = create_learned_env(tm, trajectory_dir)
    env = Monitor(env)

    obs_dim = n_blocks * int(ckpt['k_block']) + int(ckpt['k_global'])
    print(f"  Obs dim: {obs_dim}, Action dim: {n_blocks}")

    # Create MaskablePPO with same architecture as Paper 4
    model = MaskablePPO(
        ParcelScoringPolicy,
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=128,
        n_epochs=10,
        gamma=0.995,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        seed=seed,
        verbose=0,
        policy_kwargs=dict(
            k_parcel=int(ckpt['k_block']),
            k_global=int(ckpt['k_global']),
            scorer_hiddens=[128, 64],
            value_hiddens=[64, 32],
        ),
    )

    n_params = sum(p.numel() for p in model.policy.parameters())
    print(f"  Policy params: {n_params:,}")

    # Train
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f'learned_training_log_seed{seed}.json')
    model_path = os.path.join(out_dir, f'learned_model_seed{seed}.zip')

    callback = SimpleCallback(log_path=log_path)

    t0 = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - t0

    model.save(model_path)
    print(f"\n  Training complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Model saved to {model_path}")

    return model_path, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['learned', 'eval'], default='learned')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timesteps', type=int, default=500_000)
    parser.add_argument('--transition_model', default=str(PAPER7_DIR / 'models' / 'transition_model.pt'))
    parser.add_argument('--trajectory_dir', default=str(PAPER7_DIR / 'trajectories'))
    parser.add_argument('--out_dir', default=str(RESULTS_DIR))
    parser.add_argument('--model_path', default=None, help='For eval-only mode')
    parser.add_argument('--eval_episodes', type=int, default=3)
    args = parser.parse_args()

    if args.mode == 'learned':
        model_path, elapsed = train_learned(
            seed=args.seed,
            timesteps=args.timesteps,
            transition_model_path=args.transition_model,
            trajectory_dir=args.trajectory_dir,
            out_dir=args.out_dir,
        )
        # Save timing info
        timing_path = os.path.join(args.out_dir, f'learned_timing_seed{args.seed}.json')
        with open(timing_path, 'w') as f:
            json.dump({'seed': args.seed, 'timesteps': args.timesteps,
                       'elapsed_s': elapsed, 'device': 'cpu'}, f)

    elif args.mode == 'eval':
        if args.model_path is None:
            args.model_path = os.path.join(args.out_dir, f'learned_model_seed{args.seed}.zip')
        results = evaluate_on_real_env(args.model_path, n_eval=args.eval_episodes)
        eval_path = os.path.join(args.out_dir, f'learned_eval_seed{args.seed}.json')
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  Eval results saved to {eval_path}")


if __name__ == '__main__':
    main()
