"""Reward decomposition and sensitivity helpers for Paper 7."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardWeights:
    slope_weight: float = 4000.0
    cont_weight: float = 500.0
    baimu_weight: float = 1500.0
    baimu_bonus: float = 5.0
    baimu_area_penalty: float = 2000.0
    no_swap_penalty: float = 1.0

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class RewardComponents:
    slope_delta: float
    cont_delta: float
    baimu_area_delta: float
    baimu_new_count: int
    completed_swaps: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "slope_delta": float(self.slope_delta),
            "cont_delta": float(self.cont_delta),
            "baimu_area_delta": float(self.baimu_area_delta),
            "baimu_new_count": int(self.baimu_new_count),
            "completed_swaps": int(self.completed_swaps),
        }


def default_reward_weights() -> RewardWeights:
    return RewardWeights()


def compute_scalar_reward(components: RewardComponents, weights: RewardWeights) -> float:
    reward = (
        weights.slope_weight * float(components.slope_delta)
        + weights.cont_weight * float(components.cont_delta)
        + weights.baimu_weight * float(components.baimu_area_delta)
        + weights.baimu_bonus * int(components.baimu_new_count)
    )
    if components.baimu_area_delta < 0:
        reward += weights.baimu_area_penalty * float(components.baimu_area_delta)
    if components.completed_swaps <= 0:
        reward -= weights.no_swap_penalty
    return float(reward)


def generate_weight_grid() -> list[dict[str, Any]]:
    base = default_reward_weights()
    variants = [
        ("default", base),
        ("slope_x0.5", RewardWeights(slope_weight=base.slope_weight * 0.5)),
        ("slope_x2", RewardWeights(slope_weight=base.slope_weight * 2.0)),
        ("contiguity_x0.5", RewardWeights(cont_weight=base.cont_weight * 0.5)),
        ("contiguity_x2", RewardWeights(cont_weight=base.cont_weight * 2.0)),
        ("baimu_area_x0.5", RewardWeights(baimu_weight=base.baimu_weight * 0.5)),
        ("baimu_area_x2", RewardWeights(baimu_weight=base.baimu_weight * 2.0)),
        ("baimu_count_x0.5", RewardWeights(baimu_bonus=base.baimu_bonus * 0.5)),
        ("baimu_count_x2", RewardWeights(baimu_bonus=base.baimu_bonus * 2.0)),
        ("baimu_penalty_x0.5", RewardWeights(baimu_area_penalty=base.baimu_area_penalty * 0.5)),
        ("baimu_penalty_x2", RewardWeights(baimu_area_penalty=base.baimu_area_penalty * 2.0)),
        (
            "slope_priority",
            RewardWeights(
                slope_weight=base.slope_weight * 2.0,
                cont_weight=base.cont_weight * 0.5,
                baimu_weight=base.baimu_weight * 0.5,
                baimu_bonus=base.baimu_bonus * 0.5,
            ),
        ),
        (
            "contiguity_priority",
            RewardWeights(
                slope_weight=base.slope_weight * 0.75,
                cont_weight=base.cont_weight * 3.0,
                baimu_weight=base.baimu_weight,
                baimu_bonus=base.baimu_bonus,
            ),
        ),
        (
            "baimu_priority",
            RewardWeights(
                slope_weight=base.slope_weight * 0.75,
                cont_weight=base.cont_weight,
                baimu_weight=base.baimu_weight * 3.0,
                baimu_bonus=base.baimu_bonus * 3.0,
                baimu_area_penalty=base.baimu_area_penalty * 2.0,
            ),
        ),
    ]
    return [{"name": name, "weights": weights} for name, weights in variants]


def pareto_front(rows: list[dict[str, Any]], objectives: dict[str, str]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for candidate in rows:
        dominated = False
        for other in rows:
            if other is candidate:
                continue
            if _dominates(other, candidate, objectives):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front


def _dominates(left: dict[str, Any], right: dict[str, Any], objectives: dict[str, str]) -> bool:
    at_least_one_strict = False
    for key, direction in objectives.items():
        left_value = float(left[key])
        right_value = float(right[key])
        if direction == "min":
            if left_value > right_value:
                return False
            if left_value < right_value:
                at_least_one_strict = True
        elif direction == "max":
            if left_value < right_value:
                return False
            if left_value > right_value:
                at_least_one_strict = True
        else:
            raise ValueError(f"Unknown objective direction {direction!r} for {key!r}")
    return at_least_one_strict
