"""
Custom MaskableActorCriticPolicy with per-parcel scoring (v3).

Architecture:
  - Scorer network (shared MLP): scores each parcel independently
    Input: K per-parcel features + G global features = (K+G) per parcel
    Output: 1 logit per parcel
  - Value network (MLP): estimates state value from global features only
    Input: G global features
    Output: 1 scalar value

This design makes the model dimension-agnostic: the same trained weights
can be applied to datasets with any number of parcels N, enabling
cross-dataset transfer.
"""

from typing import Any, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.maskable.distributions import MaskableDistribution
from stable_baselines3.common.type_aliases import Schedule


class ParcelScoringPolicy(MaskableActorCriticPolicy):
    """
    Per-parcel scoring policy for land use optimization.

    Instead of a fixed-dimension action_net, uses a shared scorer MLP
    that processes each parcel's features independently, producing one
    logit per parcel.  This allows the same weights to generalize across
    datasets with different numbers of parcels.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        k_parcel: int = 6,
        k_global: int = 8,
        scorer_hiddens: Optional[list[int]] = None,
        value_hiddens: Optional[list[int]] = None,
        **kwargs,
    ):
        # Store custom params BEFORE super().__init__() because
        # the parent calls _build() inside __init__
        self.k_parcel = k_parcel
        self.k_global = k_global
        self.scorer_hiddens = scorer_hiddens or [64, 32]
        self.value_hiddens = value_hiddens or [64, 32]

        # Derive N from observation space
        obs_dim = observation_space.shape[0]
        self._n_parcels = (obs_dim - k_global) // k_parcel

        # Parent __init__ will call _build()
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        """Build scorer and value networks (replaces default mlp_extractor + action_net)."""

        # --- Scorer network: (K+G) -> hidden -> 1 ---
        scorer_layers = []
        in_dim = self.k_parcel + self.k_global
        for h in self.scorer_hiddens:
            scorer_layers.append(nn.Linear(in_dim, h))
            scorer_layers.append(nn.Tanh())
            in_dim = h
        scorer_layers.append(nn.Linear(in_dim, 1))
        self.scorer_net = nn.Sequential(*scorer_layers)

        # --- Value network: G -> hidden -> 1 ---
        value_layers = []
        in_dim = self.k_global
        for h in self.value_hiddens:
            value_layers.append(nn.Linear(in_dim, h))
            value_layers.append(nn.Tanh())
            in_dim = h
        value_layers.append(nn.Linear(in_dim, 1))
        self.value_net = nn.Sequential(*value_layers)

        # Initialize weights (orthogonal init like SB3 default)
        for module, gain in [(self.scorer_net, 0.01), (self.value_net, 1.0)]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=gain)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        # Setup optimizer
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    # ------------------------------------------------------------------
    # Observation parsing
    # ------------------------------------------------------------------

    def _parse_obs(self, obs: th.Tensor):
        """
        Split flat observation into per-parcel features and global features.

        Args:
            obs: (B, N*K + G) flat tensor

        Returns:
            per_parcel: (B, N, K) tensor
            globals: (B, G) tensor
        """
        B = obs.shape[0]
        K = self.k_parcel
        G = self.k_global
        N = self._n_parcels

        per_parcel = obs[:, :N * K].reshape(B, N, K)
        globals_ = obs[:, N * K:]
        return per_parcel, globals_

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _compute_logits(self, obs: th.Tensor):
        """
        Compute per-parcel logits using the scorer network.

        Args:
            obs: (B, N*K + G) observation tensor

        Returns:
            logits: (B, N) tensor of action logits
        """
        per_parcel, globals_ = self._parse_obs(obs)
        B, N, K = per_parcel.shape
        G = globals_.shape[1]

        # Expand globals to match parcels: (B, G) -> (B, N, G)
        globals_expanded = globals_.unsqueeze(1).expand(B, N, G)

        # Concatenate per-parcel + globals: (B, N, K+G)
        combined = th.cat([per_parcel, globals_expanded], dim=2)

        # nn.Linear supports arbitrary batch dims: (B, N, K+G) -> (B, N, 1)
        logits = self.scorer_net(combined).squeeze(-1)  # (B, N)

        return logits

    def _compute_values(self, obs: th.Tensor):
        """
        Compute state values from global features only.

        Args:
            obs: (B, N*K + G) observation tensor

        Returns:
            values: (B, 1) tensor
        """
        _, globals_ = self._parse_obs(obs)
        return self.value_net(globals_)

    # ------------------------------------------------------------------
    # Policy API overrides
    # ------------------------------------------------------------------

    def forward(
        self,
        obs: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """Forward pass: compute actions, values, and log probabilities."""
        logits = self._compute_logits(obs)
        values = self._compute_values(obs)

        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        action_masks: Optional[th.Tensor] = None,
    ) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """Evaluate actions: return values, log_prob, entropy."""
        logits = self._compute_logits(obs)
        values = self._compute_values(obs)

        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)

        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
        action_masks: Optional[np.ndarray] = None,
    ) -> th.Tensor:
        """Get action from observation."""
        logits = self._compute_logits(observation)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution.get_actions(deterministic=deterministic)

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """Get estimated values for observations."""
        return self._compute_values(obs)

    def get_distribution(self, obs, action_masks=None) -> MaskableDistribution:
        """Get the current policy distribution."""
        logits = self._compute_logits(obs)
        distribution = self.action_dist.proba_distribution(action_logits=logits)
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    # ------------------------------------------------------------------
    # Save / Load support
    # ------------------------------------------------------------------

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()
        data.update(dict(
            k_parcel=self.k_parcel,
            k_global=self.k_global,
            scorer_hiddens=self.scorer_hiddens,
            value_hiddens=self.value_hiddens,
        ))
        return data
