"""Generic full multi-objective county environment for Paper 7 external data."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from paper7.reward_components import (
    RewardComponents,
    RewardWeights,
    compute_scalar_reward,
    default_reward_weights,
)


FARMLAND = 1
FOREST = 2
K_BLOCK_GENERIC = 8
K_GLOBAL_GENERIC = 8


class GenericCountyEnv(gym.Env):
    """Parcel-block environment with slope, contiguity, baimu, and scalar reward."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        parcels: list[dict[str, Any]],
        block_compositions: dict[str, list[int]],
        block_ids: list[int] | None = None,
        total_budget: int = 500,
        swaps_per_step: int = 5,
        reward_weights: RewardWeights | None = None,
        baimu_threshold_m2: float = 66700.0,
    ) -> None:
        super().__init__()
        self.parcels = [dict(parcel) for parcel in parcels]
        self.block_compositions = {
            str(block_id): [int(index) for index in indices]
            for block_id, indices in block_compositions.items()
        }
        if block_ids is None:
            block_ids = sorted(int(block_id) for block_id in self.block_compositions)
        self.block_ids = [int(block_id) for block_id in block_ids]
        self.block_positions = {block_id: i for i, block_id in enumerate(self.block_ids)}
        self.n_blocks = len(self.block_ids)
        self.n_parcels = len(self.parcels)
        self.total_budget = int(total_budget)
        self.swaps_per_step = int(swaps_per_step)
        self.max_steps = max(1, self.total_budget // max(1, self.swaps_per_step))
        self.reward_weights = reward_weights or default_reward_weights()
        self.baimu_threshold_m2 = float(baimu_threshold_m2)

        self.initial_types = np.asarray(
            [_land_use_code(parcel["land_use"]) for parcel in self.parcels], dtype=np.int8
        )
        self.areas = np.asarray([float(parcel["area_m2"]) for parcel in self.parcels], dtype=np.float64)
        self.slopes = np.asarray([float(parcel["slope"]) for parcel in self.parcels], dtype=np.float64)
        self.geometries = [parcel.get("geometry") for parcel in self.parcels]
        self.adjacency = _build_geometry_adjacency(self.geometries)

        self.max_block_area = max(
            [sum(self.areas[index] for index in self.block_compositions[str(block_id)]) for block_id in self.block_ids]
            or [1.0]
        )
        self.initial_max_gain = 1e-8
        for block_id in self.block_ids:
            indices = self.block_compositions[str(block_id)]
            farm = [i for i in indices if self.initial_types[i] == FARMLAND]
            forest = [i for i in indices if self.initial_types[i] == FOREST]
            if farm and forest:
                self.initial_max_gain = max(
                    self.initial_max_gain,
                    float(max(self.slopes[farm]) - min(self.slopes[forest])),
                )

        self.action_space = spaces.Discrete(self.n_blocks)
        obs_dim = self.n_blocks * K_BLOCK_GENERIC + K_GLOBAL_GENERIC
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.reset()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.land_use = self.initial_types.copy()
        self.swapped = np.zeros(self.n_parcels, dtype=bool)
        self.step_count = 0
        self.budget_used = 0
        self.swaps_in_block = {int(block_id): 0 for block_id in self.block_ids}

        self._compute_metrics_full()
        self.baimu_count, self.baimu_total_area = self._count_baimu_fang()
        self.initial_slope = self.avg_farmland_slope
        self.initial_cont = self.contiguity
        self.initial_baimu_count = self.baimu_count
        self.initial_baimu_area = self.baimu_total_area
        self.initial_farm_area = self.total_farm_area

        self.prev_slope = self.initial_slope
        self.prev_cont = self.initial_cont
        self.prev_baimu_count = self.initial_baimu_count
        self.prev_baimu_area = self.initial_baimu_area
        return self._get_obs(), self._info()

    @property
    def avg_farmland_slope(self) -> float:
        return self.total_weighted_slope / max(self.total_farm_area, 1e-8)

    @property
    def contiguity(self) -> float:
        return self.total_farmland_adj / max(self.n_farmland, 1)

    def action_masks(self) -> np.ndarray:
        return np.asarray([self._block_feasible_gain(block_id) > 0 for block_id in self.block_ids], dtype=bool)

    def block_feature_matrix(self) -> np.ndarray:
        features = np.zeros((self.n_blocks, K_BLOCK_GENERIC), dtype=np.float32)
        for position, block_id in enumerate(self.block_ids):
            indices = self.block_compositions[str(block_id)]
            farm = [i for i in indices if self.land_use[i] == FARMLAND and not self.swapped[i]]
            forest = [i for i in indices if self.land_use[i] == FOREST and not self.swapped[i]]
            all_farm = [i for i in indices if self.land_use[i] == FARMLAND]
            gain = self._block_feasible_gain(block_id)
            farm_area = float(self.areas[farm].sum()) if farm else 0.0
            forest_area = float(self.areas[forest].sum()) if forest else 0.0
            current_farm_area = float(self.areas[all_farm].sum()) if all_farm else 0.0
            neighbor_context = self._neighbor_farmland_context(block_id)
            used_share = self.swaps_in_block[int(block_id)] / max(1, self.total_budget)
            features[position] = np.asarray(
                [
                    max(0.0, gain) / self.initial_max_gain,
                    min(farm_area, forest_area) / max(self.max_block_area, 1e-8),
                    farm_area / max(self.max_block_area, 1e-8),
                    forest_area / max(self.max_block_area, 1e-8),
                    current_farm_area / max(self.max_block_area, 1e-8),
                    neighbor_context,
                    used_share,
                    1.0 - self.step_count / max(1, self.max_steps),
                ],
                dtype=np.float32,
            )
        return features

    def step(self, action: int):
        action = int(action)
        block_id = self.block_ids[action] if 0 <= action < self.n_blocks else None
        completed = 0
        if block_id is not None:
            completed = self._execute_greedy_in_block(block_id, self.swaps_per_step)
        self.budget_used += completed
        if block_id is not None:
            self.swaps_in_block[int(block_id)] += completed
        self.step_count += 1

        self._compute_metrics_full()
        self.baimu_count, self.baimu_total_area = self._count_baimu_fang()
        component = RewardComponents(
            slope_delta=(self.prev_slope - self.avg_farmland_slope) / (abs(self.initial_slope) + 1e-8),
            cont_delta=(self.contiguity - self.prev_cont) / (abs(self.initial_cont) + 1e-8),
            baimu_area_delta=(self.baimu_total_area - self.prev_baimu_area)
            / (self.initial_farm_area + 1e-8),
            baimu_new_count=max(0, self.baimu_count - self.prev_baimu_count),
            completed_swaps=completed,
        )
        reward = compute_scalar_reward(component, self.reward_weights)

        self.prev_slope = self.avg_farmland_slope
        self.prev_cont = self.contiguity
        self.prev_baimu_count = self.baimu_count
        self.prev_baimu_area = self.baimu_total_area

        terminated = self.step_count >= self.max_steps
        if not terminated and not self.action_masks().any():
            terminated = True
        info = self._info()
        info.update(
            {
                "selected_block": block_id,
                "completed_swaps": int(completed),
                "reward_components": component.to_dict(),
            }
        )
        return self._get_obs(), float(reward), terminated, False, info

    def _get_obs(self) -> np.ndarray:
        return np.concatenate([self.block_feature_matrix().reshape(-1), self._global_features()]).astype(np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "step": int(self.step_count),
            "avg_slope": round(float(self.avg_farmland_slope), 6),
            "contiguity": round(float(self.contiguity), 6),
            "baimu_count": int(self.baimu_count),
            "baimu_area_ha": round(float(self.baimu_total_area / 10000.0), 6),
            "budget_used": int(self.budget_used),
            "slope_change_pct": round(
                100.0 * (self.avg_farmland_slope - self.initial_slope) / (abs(self.initial_slope) + 1e-8),
                6,
            ),
            "cont_change": round(float(self.contiguity - self.initial_cont), 6),
            "baimu_count_change": int(self.baimu_count - self.initial_baimu_count),
            "baimu_area_change_ha": round(float((self.baimu_total_area - self.initial_baimu_area) / 10000.0), 6),
        }

    def _global_features(self) -> np.ndarray:
        return np.asarray(
            [
                1.0 - self.step_count / max(1, self.max_steps),
                self.avg_farmland_slope,
                self.contiguity,
                (self.initial_slope - self.avg_farmland_slope) / (abs(self.initial_slope) + 1e-8),
                (self.contiguity - self.initial_cont) / (abs(self.initial_cont) + 1e-8),
                self.baimu_count / max(self.n_blocks, 1),
                self.baimu_total_area / max(self.total_farm_area, 1e-8),
                self.budget_used / max(1, self.total_budget),
            ],
            dtype=np.float32,
        )

    def _compute_metrics_full(self) -> None:
        farm = self.land_use == FARMLAND
        self.n_farmland = int(farm.sum())
        self.total_farm_area = float(self.areas[farm].sum())
        self.total_weighted_slope = float((self.slopes[farm] * self.areas[farm]).sum())
        self.farmland_nbr_count = np.zeros(self.n_parcels, dtype=np.int32)
        for i, neighbors in enumerate(self.adjacency):
            if len(neighbors):
                self.farmland_nbr_count[i] = int((self.land_use[neighbors] == FARMLAND).sum())
        self.total_farmland_adj = int(self.farmland_nbr_count[farm].sum())

    def _count_baimu_fang(self) -> tuple[int, float]:
        is_farm = self.land_use == FARMLAND
        parent = np.arange(self.n_parcels, dtype=np.int32)

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = int(parent[root])
            while parent[x] != root:
                parent[x], x = root, int(parent[x])
            return root

        for i, neighbors in enumerate(self.adjacency):
            if not is_farm[i]:
                continue
            for j in neighbors:
                j = int(j)
                if j <= i or not is_farm[j]:
                    continue
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
        farm_indices = np.where(is_farm)[0]
        if len(farm_indices) == 0:
            return 0, 0.0
        roots = np.asarray([find(int(i)) for i in farm_indices], dtype=np.int32)
        unique, inverse = np.unique(roots, return_inverse=True)
        component_areas = np.zeros(len(unique), dtype=np.float64)
        np.add.at(component_areas, inverse, self.areas[farm_indices])
        baimu = component_areas >= self.baimu_threshold_m2
        return int(baimu.sum()), float(component_areas[baimu].sum())

    def _execute_greedy_in_block(self, block_id: int, max_swaps: int) -> int:
        completed = 0
        indices = self.block_compositions[str(block_id)]
        for _ in range(max_swaps):
            farm = [i for i in indices if self.land_use[i] == FARMLAND and not self.swapped[i]]
            forest = [i for i in indices if self.land_use[i] == FOREST and not self.swapped[i]]
            if not farm or not forest:
                break
            best_farm = max(farm, key=lambda i: (self.slopes[i], self.areas[i]))
            best_forest = min(forest, key=lambda i: (self.slopes[i], -self.areas[i]))
            if self.slopes[best_farm] <= self.slopes[best_forest]:
                break
            self.land_use[best_farm] = FOREST
            self.land_use[best_forest] = FARMLAND
            self.swapped[best_farm] = True
            self.swapped[best_forest] = True
            completed += 1
        return completed

    def _block_feasible_gain(self, block_id: int) -> float:
        indices = self.block_compositions[str(block_id)]
        farm = [i for i in indices if self.land_use[i] == FARMLAND and not self.swapped[i]]
        forest = [i for i in indices if self.land_use[i] == FOREST and not self.swapped[i]]
        if not farm or not forest:
            return 0.0
        return float(max(self.slopes[farm]) - min(self.slopes[forest]))

    def _neighbor_farmland_context(self, block_id: int) -> float:
        indices = set(self.block_compositions[str(block_id)])
        neighbor_farm = 0
        neighbor_total = 0
        for i in indices:
            for j in self.adjacency[i]:
                if int(j) in indices:
                    continue
                neighbor_total += 1
                if self.land_use[int(j)] == FARMLAND:
                    neighbor_farm += 1
        return float(neighbor_farm / neighbor_total) if neighbor_total else 0.0


def _land_use_code(value: Any) -> int:
    text = str(value).strip().lower()
    if text in {"farmland", "farm", "1"}:
        return FARMLAND
    if text in {"forest", "woodland", "2"}:
        return FOREST
    raise ValueError(f"Unsupported land-use value {value!r}")


def _build_geometry_adjacency(geometries: list[Any]) -> list[np.ndarray]:
    try:
        from shapely.strtree import STRtree

        tree = STRtree(geometries)
        adjacency: list[set[int]] = [set() for _ in geometries]
        for i, geometry in enumerate(geometries):
            if geometry is None or geometry.is_empty:
                continue
            candidates = tree.query(geometry, predicate="intersects")
            for candidate in candidates:
                j = int(candidate)
                if i == j:
                    continue
                adjacency[i].add(j)
                adjacency[j].add(i)
        return [np.asarray(sorted(values), dtype=np.int32) for values in adjacency]
    except Exception:
        adjacency = [set() for _ in geometries]
        for i, left in enumerate(geometries):
            if left is None or left.is_empty:
                continue
            for j in range(i + 1, len(geometries)):
                right = geometries[j]
                if right is None or right.is_empty:
                    continue
                if left.intersects(right):
                    adjacency[i].add(j)
                    adjacency[j].add(i)
        return [np.asarray(sorted(values), dtype=np.int32) for values in adjacency]
