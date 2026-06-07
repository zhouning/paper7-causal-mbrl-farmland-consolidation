"""
Paper 7 Phase 2: Train the TransitionModel on collected trajectories.

Usage:
    python paper7/train_learned_env.py
    python paper7/train_learned_env.py --epochs 200 --policies random greedy
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from learned_env import TransitionModel, TrajectoryDataset, train_transition_model, LearnedCountyEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=os.path.join(os.path.dirname(__file__), 'trajectories'))
    parser.add_argument('--policies', nargs='+', default=None, help='e.g., random greedy')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--out_dir', default=os.path.join(os.path.dirname(__file__), 'models'))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("Paper 7 Phase 2: Train Transition Model")
    print("=" * 60)

    t0 = time.time()
    model, history, dataset = train_transition_model(
        data_dir=args.data_dir,
        policies=args.policies,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - t0

    # Save model
    model_path = os.path.join(args.out_dir, 'transition_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_blocks': dataset.n_blocks,
        'k_block': dataset.k_block,
        'k_global': dataset.k_global,
        'n_transitions': len(dataset),
        'epochs': args.epochs,
        'training_time_s': elapsed,
    }, model_path)
    print(f"\nModel saved to {model_path}")

    # Save training history
    hist_path = os.path.join(args.out_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)

    # Quick sanity check: run a short episode in the learned environment
    print("\n--- Sanity Check: LearnedCountyEnv ---")
    # Use first trajectory's initial state
    data = np.load(os.path.join(args.data_dir, sorted(os.listdir(args.data_dir))[0]))
    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]

    env = LearnedCountyEnv(
        transition_model=model,
        initial_block_features=init_bf,
        initial_global_features=init_gf,
        n_blocks=dataset.n_blocks,
        k_block=dataset.k_block,
        k_global=dataset.k_global,
        max_steps=100,
    )

    obs, _ = env.reset()
    total_reward = 0
    for step in range(10):
        mask = env.action_masks()
        valid = np.where(mask)[0]
        if len(valid) == 0:
            break
        action = int(np.random.choice(valid))
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    print(f"  10 random steps in LearnedCountyEnv: total_reward={total_reward:.4f}")
    print(f"  Obs range: [{obs.min():.4f}, {obs.max():.4f}]")
    print(f"  Training time: {elapsed:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Transitions: {len(dataset):,}")
    print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"  Final val cosine: {history['val_obs_cosine'][-1]:.6f}")
    print(f"  Final val reward MSE: {history['val_reward_mse'][-1]:.6f}")
    print(f"  Time: {elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
