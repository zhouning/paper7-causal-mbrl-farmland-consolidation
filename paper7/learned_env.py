"""
Paper 7: LearnedCountyEnv — Neural transition model for model-based RL.

Predicts (next_block_features, next_global_features, reward) from
(block_features, global_features, action, geofm_embeddings).

This replaces the real CountyLevelEnv for policy training, enabling
~100x speedup (CPU minutes vs A100 hours).

Architecture:
  1. Per-block encoder: MLP encodes each block's features + GeoFM embedding
  2. Action embedding: one-hot action → dense embedding
  3. Global context: global features + aggregated block info
  4. Transition predictor: predicts residual changes to block + global features
  5. Reward head: predicts scalar reward

Usage:
    # Train
    model = TransitionModel(n_blocks=2600, k_block=17, k_global=12)
    model.fit(trajectory_data)

    # Use as Gymnasium env
    env = LearnedCountyEnv(model, initial_obs, n_blocks=2600)
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gymnasium as gym
from gymnasium import spaces


class TrajectoryDataset(Dataset):
    """Load collected trajectories for transition model training."""

    def __init__(self, data_dir, policies=None):
        """
        Args:
            data_dir: directory containing .npz trajectory files
            policies: list of policy prefixes to include (e.g., ['random', 'greedy'])
                      None = load all
        """
        self.block_features = []
        self.global_features = []
        self.actions = []
        self.rewards = []
        self.next_block_features = []
        self.next_global_features = []

        for fname in sorted(os.listdir(data_dir)):
            if not fname.endswith('.npz'):
                continue
            if policies and not any(fname.startswith(p) for p in policies):
                continue

            path = os.path.join(data_dir, fname)
            data = np.load(path, allow_pickle=False)

            self.block_features.append(data['block_features'].astype(np.float32))
            self.global_features.append(data['global_features'])
            self.actions.append(data['actions'])
            self.rewards.append(data['rewards'])
            self.next_block_features.append(data['next_block_features'].astype(np.float32))
            self.next_global_features.append(data['next_global_features'])

            self.n_blocks = int(data['n_blocks'])
            self.k_block = int(data['k_block'])
            self.k_global = int(data['k_global'])

            print(f"  Loaded {fname}: {len(data['actions'])} transitions")

        self.block_features = np.concatenate(self.block_features)
        self.global_features = np.concatenate(self.global_features)
        self.actions = np.concatenate(self.actions)
        self.rewards = np.concatenate(self.rewards)
        self.next_block_features = np.concatenate(self.next_block_features)
        self.next_global_features = np.concatenate(self.next_global_features)

        print(f"  Total: {len(self.actions)} transitions, "
              f"blocks={self.n_blocks}, k_block={self.k_block}, k_global={self.k_global}")

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        return {
            'block_features': torch.tensor(self.block_features[idx]),       # (N, 17)
            'global_features': torch.tensor(self.global_features[idx]),     # (K_G,)
            'action': torch.tensor(self.actions[idx], dtype=torch.long),    # scalar
            'reward': torch.tensor(self.rewards[idx], dtype=torch.float32), # scalar
            'next_block_features': torch.tensor(self.next_block_features[idx]),
            'next_global_features': torch.tensor(self.next_global_features[idx]),
        }


class TransitionModel(nn.Module):
    """Neural transition model for county-level farmland MDP.

    Predicts residual changes:
        next_block_features ≈ block_features + Δ_block(block_features, action, global)
        next_global_features ≈ global_features + Δ_global(block_features, action, global)
        reward ≈ f_reward(block_features, action, global)
    """

    def __init__(self, n_blocks, k_block=17, k_global=12,
                 hidden_dim=256, geofm_dim=0):
        super().__init__()
        self.n_blocks = n_blocks
        self.k_block = k_block
        self.k_global = k_global
        self.geofm_dim = geofm_dim

        # Per-block feature encoder (shared across all blocks)
        block_input_dim = k_block + geofm_dim
        self.block_encoder = nn.Sequential(
            nn.Linear(block_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Action embedding
        self.action_embed = nn.Embedding(n_blocks, 32)

        # Global context encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(k_global, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

        # Transition predictor: predicts Δ for the SELECTED block
        # Input: selected_block_encoding (32) + action_embed (32) + global (32) + neighbor_agg (32)
        self.selected_block_transition = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k_block),  # Δ_block for selected block
        )

        # Global transition predictor
        self.global_transition = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k_global),  # Δ_global
        )

        # Reward predictor
        self.reward_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, block_features, global_features, action, geofm=None):
        """
        Args:
            block_features: (B, N, K_BLOCK)
            global_features: (B, K_GLOBAL)
            action: (B,) long — selected block index
            geofm: (B, N, geofm_dim) optional GeoFM embeddings

        Returns:
            pred_next_block: (B, N, K_BLOCK)
            pred_next_global: (B, K_GLOBAL)
            pred_reward: (B,)
        """
        B, N, K = block_features.shape

        # Encode all blocks
        if geofm is not None:
            block_input = torch.cat([block_features, geofm], dim=-1)
        else:
            block_input = block_features
        block_enc = self.block_encoder(block_input)  # (B, N, 32)

        # Get selected block encoding
        action_idx = action.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, 32)
        selected_enc = block_enc.gather(1, action_idx).squeeze(1)  # (B, 32)

        # Action embedding
        action_emb = self.action_embed(action)  # (B, 32)

        # Global encoding
        global_enc = self.global_encoder(global_features)  # (B, 32)

        # Aggregate neighbor blocks (mean pool all blocks as context)
        neighbor_agg = block_enc.mean(dim=1)  # (B, 32)

        # Combined context
        context = torch.cat([selected_enc, action_emb, global_enc, neighbor_agg], dim=-1)  # (B, 128)

        # Predict deltas
        delta_selected = self.selected_block_transition(context)  # (B, K_BLOCK)
        delta_global = self.global_transition(context)  # (B, K_GLOBAL)
        pred_reward = self.reward_head(context).squeeze(-1)  # (B,)

        # Apply residual: only the selected block changes
        pred_next_block = block_features.clone()
        # Scatter delta to selected block
        for b in range(B):
            pred_next_block[b, action[b]] = block_features[b, action[b]] + delta_selected[b]

        pred_next_global = global_features + delta_global

        return pred_next_block, pred_next_global, pred_reward


def train_transition_model(data_dir, policies=None, epochs=100, lr=1e-3,
                           batch_size=64, val_split=0.1, geofm_dir=None):
    """Train the transition model on collected trajectories.

    Returns trained TransitionModel and training metrics dict.
    """
    print("Loading trajectory data...")
    dataset = TrajectoryDataset(data_dir, policies)

    # Train/val split
    n = len(dataset)
    n_val = int(n * val_split)
    n_train = n - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Build model
    model = TransitionModel(
        n_blocks=dataset.n_blocks,
        k_block=dataset.k_block,
        k_global=dataset.k_global,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"TransitionModel: {n_params:,} parameters")
    print(f"Training: {n_train} samples, Validation: {n_val} samples")

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'val_reward_mse': [], 'val_obs_cosine': []}

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            bf = batch['block_features']
            gf = batch['global_features']
            act = batch['action']
            rew = batch['reward']
            nbf = batch['next_block_features']
            ngf = batch['next_global_features']

            pred_nbf, pred_ngf, pred_rew = model(bf, gf, act)

            # Loss: MSE on block features + MSE on global + MSE on reward
            loss_block = F.mse_loss(pred_nbf, nbf)
            loss_global = F.mse_loss(pred_ngf, ngf)
            loss_reward = F.mse_loss(pred_rew, rew)
            loss = loss_block + loss_global + 0.1 * loss_reward

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses, val_rew_mse, val_cosines = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                bf = batch['block_features']
                gf = batch['global_features']
                act = batch['action']
                rew = batch['reward']
                nbf = batch['next_block_features']
                ngf = batch['next_global_features']

                pred_nbf, pred_ngf, pred_rew = model(bf, gf, act)

                loss_block = F.mse_loss(pred_nbf, nbf)
                loss_global = F.mse_loss(pred_ngf, ngf)
                loss_reward = F.mse_loss(pred_rew, rew)
                loss = loss_block + loss_global + 0.1 * loss_reward
                val_losses.append(loss.item())
                val_rew_mse.append(loss_reward.item())

                # Cosine similarity between predicted and actual next obs (flattened)
                pred_flat = torch.cat([pred_nbf.reshape(pred_nbf.size(0), -1), pred_ngf], dim=-1)
                true_flat = torch.cat([nbf.reshape(nbf.size(0), -1), ngf], dim=-1)
                cos = F.cosine_similarity(pred_flat, true_flat, dim=-1).mean()
                val_cosines.append(cos.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        val_rew = np.mean(val_rew_mse)
        val_cos = np.mean(val_cosines)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_reward_mse'].append(val_rew)
        history['val_obs_cosine'].append(val_cos)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}: train={train_loss:.6f} val={val_loss:.6f} "
                  f"rew_mse={val_rew:.6f} cos={val_cos:.6f}")

    model.load_state_dict(best_state)
    print(f"Best val loss: {best_val_loss:.6f}")

    return model, history, dataset


class LearnedCountyEnv(gym.Env):
    """Gymnasium environment backed by a trained TransitionModel.

    Drop-in replacement for CountyLevelEnv — same obs/action/reward interface
    but predictions come from the neural network instead of parcel-level simulation.
    """

    metadata = {"render_modes": []}

    def __init__(self, transition_model, initial_block_features, initial_global_features,
                 n_blocks, k_block=17, k_global=12, max_steps=100,
                 action_mask_fn=None, geofm_embeddings=None):
        super().__init__()
        self.model = transition_model
        self.model.eval()
        self.n_blocks = n_blocks
        self.k_block = k_block
        self.k_global = k_global
        self.max_steps = max_steps

        self._initial_bf = initial_block_features.copy()  # (N, K_BLOCK)
        self._initial_gf = initial_global_features.copy()  # (K_GLOBAL,)
        self._action_mask_fn = action_mask_fn

        # Optional GeoFM embeddings (static, doesn't change per step)
        self._geofm = None
        if geofm_embeddings is not None:
            self._geofm = torch.tensor(geofm_embeddings, dtype=torch.float32).unsqueeze(0)  # (1, N, 64)

        obs_dim = n_blocks * k_block + k_global
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(n_blocks)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._bf = self._initial_bf.copy()
        self._gf = self._initial_gf.copy()
        self._step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([self._bf.reshape(-1), self._gf]).astype(np.float32)

    def action_masks(self):
        """Return boolean mask of valid actions."""
        if self._action_mask_fn is not None:
            return self._action_mask_fn(self._bf)
        # Default: block is valid if it has swap potential > 0 (feature index 9)
        return self._bf[:, 9] > 0.01

    def step(self, action):
        with torch.no_grad():
            bf_t = torch.tensor(self._bf, dtype=torch.float32).unsqueeze(0)
            gf_t = torch.tensor(self._gf, dtype=torch.float32).unsqueeze(0)
            act_t = torch.tensor([action], dtype=torch.long)

            pred_nbf, pred_ngf, pred_rew = self.model(bf_t, gf_t, act_t, geofm=self._geofm)

            self._bf = pred_nbf.squeeze(0).numpy()
            self._gf = pred_ngf.squeeze(0).numpy()
            reward = pred_rew.item()

        self._step += 1
        done = self._step >= self.max_steps

        return self._get_obs(), reward, done, False, {'step': self._step}

    def render(self):
        pass

    def close(self):
        pass
