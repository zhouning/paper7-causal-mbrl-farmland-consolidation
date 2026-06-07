"""
Paper 7 E4: GeoFM Embedding Ablation.

Compares transition model quality and downstream policy performance
with vs without 64-dim AlphaEarth GeoFM embeddings as auxiliary features.

Usage:
    python paper7/ablation_geofm.py
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
from torch.utils.data import Dataset, DataLoader

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY
from parcel_scoring_policy import ParcelScoringPolicy
from learned_env import TransitionModel

PAPER7_DIR = os.path.dirname(os.path.abspath(__file__))


class AugmentedTrajectoryDataset(Dataset):
    """Trajectory dataset with optional GeoFM embedding augmentation."""

    def __init__(self, data_dir, geofm_path=None, policies=None):
        bfs, gfs, acts, rews, nbfs, ngfs = [], [], [], [], [], []

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.npz'):
                continue
            if policies and not any(fname.startswith(p) for p in policies):
                continue
            data = np.load(os.path.join(data_dir, fname))
            bfs.append(data['block_features'].astype(np.float32))
            gfs.append(data['global_features'])
            acts.append(data['actions'])
            rews.append(data['rewards'])
            nbfs.append(data['next_block_features'].astype(np.float32))
            ngfs.append(data['next_global_features'])
            self.n_blocks = int(data['n_blocks'])
            self.k_block = int(data['k_block'])
            self.k_global = int(data['k_global'])

        self.block_features = np.concatenate(bfs)
        self.global_features = np.concatenate(gfs)
        self.actions = np.concatenate(acts)
        self.rewards = np.concatenate(rews)
        self.next_block_features = np.concatenate(nbfs)
        self.next_global_features = np.concatenate(ngfs)

        # Load GeoFM embeddings if provided
        self.geofm = None
        self.geofm_dim = 0
        if geofm_path and os.path.exists(geofm_path):
            self.geofm = np.load(geofm_path).astype(np.float32)  # (n_blocks, 64)
            self.geofm_dim = self.geofm.shape[1]
            print(f"  GeoFM loaded: {self.geofm.shape}, will augment block features")

        print(f"  Dataset: {len(self.actions)} transitions, geofm_dim={self.geofm_dim}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        item = {
            'block_features': torch.tensor(self.block_features[idx]),
            'global_features': torch.tensor(self.global_features[idx]),
            'action': torch.tensor(self.actions[idx], dtype=torch.long),
            'reward': torch.tensor(self.rewards[idx], dtype=torch.float32),
            'next_block_features': torch.tensor(self.next_block_features[idx]),
            'next_global_features': torch.tensor(self.next_global_features[idx]),
        }
        if self.geofm is not None:
            item['geofm'] = torch.tensor(self.geofm)  # (n_blocks, 64) — same for all steps
        return item


def train_transition_model_with_geofm(dataset, epochs=50, lr=1e-3, batch_size=32):
    """Train TransitionModel, optionally with GeoFM augmentation."""
    import torch.nn.functional as F

    n = len(dataset)
    n_val = int(n * 0.1)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = TransitionModel(
        n_blocks=dataset.n_blocks,
        k_block=dataset.k_block,
        k_global=dataset.k_global,
        geofm_dim=dataset.geofm_dim,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} params (geofm_dim={dataset.geofm_dim})")

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            geofm = batch.get('geofm', None)
            pred_nbf, pred_ngf, pred_rew = model(
                batch['block_features'], batch['global_features'],
                batch['action'], geofm=geofm)
            loss = (F.mse_loss(pred_nbf, batch['next_block_features'])
                    + F.mse_loss(pred_ngf, batch['next_global_features'])
                    + 0.1 * F.mse_loss(pred_rew, batch['reward']))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_losses, val_cosines = [], []
        with torch.no_grad():
            for batch in val_loader:
                geofm = batch.get('geofm', None)
                pred_nbf, pred_ngf, pred_rew = model(
                    batch['block_features'], batch['global_features'],
                    batch['action'], geofm=geofm)
                loss = (F.mse_loss(pred_nbf, batch['next_block_features'])
                        + F.mse_loss(pred_ngf, batch['next_global_features'])
                        + 0.1 * F.mse_loss(pred_rew, batch['reward']))
                val_losses.append(loss.item())
                pred_flat = torch.cat([pred_nbf.reshape(pred_nbf.size(0), -1), pred_ngf], -1)
                true_flat = torch.cat([batch['next_block_features'].reshape(pred_nbf.size(0), -1),
                                       batch['next_global_features']], -1)
                cos = F.cosine_similarity(pred_flat, true_flat, dim=-1).mean()
                val_cosines.append(cos.item())

        vl = np.mean(val_losses)
        vc = np.mean(val_cosines)
        if vl < best_val_loss:
            best_val_loss = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: val_loss={vl:.6f}, val_cos={vc:.6f}")

    model.load_state_dict(best_state)
    return model, best_val_loss, vc


def eval_on_real_env(model_path):
    """Evaluate trained policy on real CountyLevelEnv."""
    env = CountyLevelEnv(total_budget=500, swaps_per_step=5)
    model = MaskablePPO.load(model_path, env=env)
    obs, _ = env.reset()
    done = False
    while not done:
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, r, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return {
        'slope_change_pct': info.get('slope_change_pct', 0),
        'cont_change': info.get('cont_change', 0),
        'baimu_count_change': info.get('baimu_count_change', 0),
    }


def train_rl_and_eval(tm, dataset, seed, label, out_dir):
    """Train MaskablePPO on LearnedCountyEnv backed by given transition model, eval on real."""
    from learned_env import LearnedCountyEnv

    files = sorted([f for f in os.listdir(os.path.join(PAPER7_DIR, 'trajectories'))
                    if f.endswith('.npz')])
    data = np.load(os.path.join(PAPER7_DIR, 'trajectories', files[0]))
    init_bf = data['block_features'][0].astype(np.float32)
    init_gf = data['global_features'][0]

    # Pass GeoFM embeddings if the dataset has them
    geofm = dataset.geofm if dataset.geofm is not None else None

    env = LearnedCountyEnv(
        transition_model=tm, initial_block_features=init_bf,
        initial_global_features=init_gf,
        n_blocks=dataset.n_blocks, k_block=dataset.k_block,
        k_global=dataset.k_global, max_steps=100,
        geofm_embeddings=geofm)
    env = Monitor(env)

    model = MaskablePPO(
        ParcelScoringPolicy, env, learning_rate=3e-4, n_steps=256,
        batch_size=128, n_epochs=10, gamma=0.995, gae_lambda=0.95,
        clip_range=0.2, ent_coef=0.005, seed=seed, verbose=0,
        policy_kwargs=dict(k_parcel=dataset.k_block, k_global=dataset.k_global,
                           scorer_hiddens=[128, 64], value_hiddens=[64, 32]))

    t0 = time.time()
    model.learn(total_timesteps=100_000)
    elapsed = time.time() - t0

    model_path = os.path.join(out_dir, f'{label}_model_seed{seed}.zip')
    model.save(model_path)
    print(f"    {label} seed{seed}: trained in {elapsed:.0f}s, evaluating on real env...")

    result = eval_on_real_env(model_path)
    result['label'] = label
    result['seed'] = seed
    result['training_time_s'] = elapsed
    print(f"    {label} seed{seed}: slope={result['slope_change_pct']:+.3f}%")
    return result


def main():
    print("=" * 60)
    print("E4: GeoFM Embedding Ablation")
    print("=" * 60)

    traj_dir = os.path.join(PAPER7_DIR, 'trajectories')
    geofm_path = os.path.join(PAPER7_DIR, 'data', 'block_geofm_embeddings.npy')
    out_dir = os.path.join(PAPER7_DIR, 'results', 'ablation')
    os.makedirs(out_dir, exist_ok=True)

    results = []

    for use_geofm in [False, True]:
        label = 'with_geofm' if use_geofm else 'no_geofm'
        print(f"\n--- {label} ---")

        # Load dataset
        gp = geofm_path if use_geofm else None
        ds = AugmentedTrajectoryDataset(traj_dir, geofm_path=gp)

        # Train transition model
        print(f"  Training transition model ({label})...")
        tm, val_loss, val_cos = train_transition_model_with_geofm(ds, epochs=30, batch_size=32)
        print(f"  Best val_loss={val_loss:.6f}, val_cos={val_cos:.6f}")

        # Train RL policy and evaluate on real env (1 seed for speed)
        for seed in range(2):
            r = train_rl_and_eval(tm, ds, seed, label, out_dir)
            r['tm_val_loss'] = val_loss
            r['tm_val_cosine'] = val_cos
            r['geofm_dim'] = ds.geofm_dim
            results.append(r)

    # Summary
    no_gf = [r for r in results if r['label'] == 'no_geofm']
    with_gf = [r for r in results if r['label'] == 'with_geofm']

    print(f"\n{'='*60}")
    print(f"E4 Results:")
    print(f"  No GeoFM:   slope = {np.mean([r['slope_change_pct'] for r in no_gf]):+.3f}% "
          f"± {np.std([r['slope_change_pct'] for r in no_gf]):.3f}%, "
          f"TM val_cos = {no_gf[0]['tm_val_cosine']:.6f}")
    print(f"  With GeoFM: slope = {np.mean([r['slope_change_pct'] for r in with_gf]):+.3f}% "
          f"± {np.std([r['slope_change_pct'] for r in with_gf]):.3f}%, "
          f"TM val_cos = {with_gf[0]['tm_val_cosine']:.6f}")
    print(f"{'='*60}")

    with open(os.path.join(out_dir, 'e4_geofm_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_dir}/e4_geofm_results.json")


if __name__ == '__main__':
    main()
