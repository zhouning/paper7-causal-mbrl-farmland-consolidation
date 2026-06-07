"""
Paper 7 Phase 1: Collect state-action-reward trajectories from CountyLevelEnv.

Supports multiple policy types:
  - random: uniform random valid actions
  - greedy: select block with highest immediate slope reduction potential
  - model: load trained MaskablePPO model (requires GPU / Colab)

Outputs: paper7/trajectories/{policy}_{seed}.npz
  - obs: (T, obs_dim) float32
  - actions: (T,) int32
  - rewards: (T,) float32
  - next_obs: (T, obs_dim) float32
  - dones: (T,) bool
  - block_ids: (T,) int32 — same as actions (block index)

Usage:
    python paper7/collect_trajectories.py --policy random --seeds 5 --episodes 20
    python paper7/collect_trajectories.py --policy greedy --seeds 5 --episodes 20
    python paper7/collect_trajectories.py --policy model --model_path path/to/model.zip --seeds 1 --episodes 20
"""

import os
import sys
import argparse
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY


def collect_episode(env, policy_fn, deterministic=True):
    """Collect one episode of structured trajectory data.

    Returns list of dicts with block-level features (not flattened obs) to save space.
    Each step stores:
      - block_features: (n_blocks, K_BLOCK) float32
      - global_features: (K_GLOBAL,) float32
      - action: int (block index)
      - reward: float
      - done: bool
      - info: dict with slope_change_pct, cont_change, etc.
    """
    obs, _ = env.reset()
    n_blocks = env.n_blocks
    trajectory = []
    done = False
    while not done:
        mask = env.action_masks()
        action = policy_fn(obs, mask, deterministic)
        next_obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        # Extract structured features from flattened obs
        bf = obs[:n_blocks * K_BLOCK].reshape(n_blocks, K_BLOCK)
        gf = obs[n_blocks * K_BLOCK:]
        next_bf = next_obs[:n_blocks * K_BLOCK].reshape(n_blocks, K_BLOCK)
        next_gf = next_obs[n_blocks * K_BLOCK:]

        trajectory.append({
            'block_features': bf.astype(np.float16),  # half precision saves 50%
            'global_features': gf.astype(np.float32),
            'action': int(action),
            'reward': float(reward),
            'next_block_features': next_bf.astype(np.float16),
            'next_global_features': next_gf.astype(np.float32),
            'done': done,
        })
        obs = next_obs
    return trajectory


def random_policy(obs, mask, deterministic=False):
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return 0
    return int(np.random.choice(valid))


def greedy_policy(obs, mask, deterministic=True):
    """Select block with highest slope gap (feature index 3 in per-block features)."""
    n_blocks = (len(obs) - K_GLOBAL_COUNTY) // K_BLOCK
    block_features = obs[:n_blocks * K_BLOCK].reshape(n_blocks, K_BLOCK)
    # Feature 3 = best_swap_gain_norm (higher = more slope reduction potential)
    scores = block_features[:, 3].copy()
    scores[~mask[:n_blocks]] = -999
    return int(np.argmax(scores))


def model_policy_factory(model_path, env):
    """Create a policy function from a trained MaskablePPO model."""
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(model_path, env=env)
    def policy_fn(obs, mask, deterministic=True):
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        return int(action)
    return policy_fn


def main():
    parser = argparse.ArgumentParser(description='Collect trajectories for Paper 7')
    parser.add_argument('--policy', choices=['random', 'greedy', 'model'], default='random')
    parser.add_argument('--model_path', type=str, default=None, help='Path to trained model .zip')
    parser.add_argument('--seeds', type=int, default=5, help='Number of random seeds')
    parser.add_argument('--episodes', type=int, default=20, help='Episodes per seed')
    parser.add_argument('--budget', type=int, default=500, help='Total swap budget')
    parser.add_argument('--outdir', type=str, default=None)
    args = parser.parse_args()

    if args.outdir is None:
        args.outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trajectories')
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Collecting trajectories: policy={args.policy}, seeds={args.seeds}, "
          f"episodes={args.episodes}, budget={args.budget}")

    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        np.random.seed(seed)

        env = CountyLevelEnv(total_budget=args.budget, swaps_per_step=5)

        if args.policy == 'random':
            policy_fn = random_policy
        elif args.policy == 'greedy':
            policy_fn = greedy_policy
        elif args.policy == 'model':
            if args.model_path is None:
                raise ValueError("--model_path required for model policy")
            policy_fn = model_policy_factory(args.model_path, env)
        else:
            raise ValueError(f"Unknown policy: {args.policy}")

        all_transitions = []

        for ep in range(args.episodes):
            t0 = time.time()
            traj = collect_episode(env, policy_fn)
            elapsed = time.time() - t0
            all_transitions.extend(traj)

            total_reward = sum(t['reward'] for t in traj)
            print(f"  Ep {ep:3d}: {len(traj):3d} steps, reward={total_reward:7.2f}, "
                  f"time={elapsed:.1f}s")

        # Save as compressed numpy (structured format)
        out_path = os.path.join(args.outdir, f'{args.policy}_seed{seed}.npz')

        # Stack structured arrays
        bf = np.array([t['block_features'] for t in all_transitions], dtype=np.float16)
        gf = np.array([t['global_features'] for t in all_transitions], dtype=np.float32)
        nbf = np.array([t['next_block_features'] for t in all_transitions], dtype=np.float16)
        ngf = np.array([t['next_global_features'] for t in all_transitions], dtype=np.float32)

        np.savez_compressed(out_path,
            block_features=bf,           # (T, n_blocks, 17) float16
            global_features=gf,          # (T, K_GLOBAL) float32
            actions=np.array([t['action'] for t in all_transitions], dtype=np.int32),
            rewards=np.array([t['reward'] for t in all_transitions], dtype=np.float32),
            next_block_features=nbf,     # (T, n_blocks, 17) float16
            next_global_features=ngf,    # (T, K_GLOBAL) float32
            dones=np.array([t['done'] for t in all_transitions], dtype=bool),
            n_blocks=np.array(env.n_blocks),
            k_block=np.array(K_BLOCK),
            k_global=np.array(K_GLOBAL_COUNTY),
        )
        n_transitions = len(all_transitions)
        fsize = os.path.getsize(out_path) / 1024 / 1024
        print(f"  Saved {n_transitions} transitions to {out_path} ({fsize:.1f} MB)")

    print(f"\nDone! All trajectories saved to {args.outdir}/")


if __name__ == '__main__':
    main()
