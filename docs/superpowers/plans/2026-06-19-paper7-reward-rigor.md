# Paper 7 Reward Rigor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible reward decomposition, weight sensitivity, and Pareto analysis so Paper 7's reward function can be evaluated as a scientific design choice rather than an assumed coefficient set.

**Architecture:** Introduce a small reward utility module that exactly reproduces the current `CountyLevelEnv` scalar reward from raw component deltas. Add evaluation scripts that run real-environment heuristic policies while recording component logs, then replay those logs under multiple weight families and summarize Pareto trade-offs. Keep all new artifacts under `paper7/results/full_rigor/` so they do not overwrite the prior CEUS evidence chain.

**Tech Stack:** Python, NumPy, JSON, pytest, existing `CountyLevelEnv`, existing heuristic policy conventions.

---

## File Structure

- Create: `paper7/reward_components.py`  
  Owns `RewardWeights`, `RewardComponents`, scalar reward computation, reward-grid generation, and Pareto helper functions.
- Create: `paper7/reward_component_rollouts.py`  
  Runs real `CountyLevelEnv` episodes under heuristic policies and stores per-step reward component logs.
- Create: `paper7/reward_weight_sensitivity.py`  
  Replays component logs under alternative reward weights and writes sensitivity and Pareto JSON artifacts.
- Create: `tests/test_reward_components.py`  
  Unit tests for default reward equivalence, grid generation, scalarization, and Pareto logic.
- Create: `tests/test_reward_weight_sensitivity.py`  
  Unit tests for replay summaries and dominance classification on small synthetic logs.
- Modify: `paper7/end_to_end_validation.py`  
  Later adds optional audit visibility for reward-rigor artifacts without failing older CEUS checks.
- Modify: `tests/test_end_to_end_validation.py`  
  Later tests audit classification for reward sensitivity artifacts.

---

### Task 1: Reward Component Utilities

**Files:**
- Create: `paper7/reward_components.py`
- Create: `tests/test_reward_components.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_reward_components.py` with:

```python
from paper7.reward_components import (
    RewardComponents,
    RewardWeights,
    default_reward_weights,
    compute_scalar_reward,
    generate_weight_grid,
    pareto_front,
)


def test_default_reward_matches_county_env_formula_with_negative_baimu_area():
    components = RewardComponents(
        slope_delta=0.001,
        cont_delta=0.002,
        baimu_area_delta=-0.003,
        baimu_new_count=2,
        completed_swaps=5,
    )

    reward = compute_scalar_reward(components, default_reward_weights())

    expected = (
        4000.0 * 0.001
        + 500.0 * 0.002
        + 1500.0 * -0.003
        + 5.0 * 2
        + 2000.0 * -0.003
    )
    assert reward == expected


def test_invalid_or_zero_swap_action_receives_penalty():
    components = RewardComponents(
        slope_delta=0.0,
        cont_delta=0.0,
        baimu_area_delta=0.0,
        baimu_new_count=0,
        completed_swaps=0,
    )

    reward = compute_scalar_reward(components, default_reward_weights())

    assert reward == -1.0


def test_generate_weight_grid_includes_default_and_named_variants():
    grid = generate_weight_grid()
    names = {item["name"] for item in grid}

    assert "default" in names
    assert "slope_x2" in names
    assert "contiguity_x2" in names
    assert "baimu_area_x2" in names
    assert "baimu_count_x2" in names

    default = next(item for item in grid if item["name"] == "default")
    assert default["weights"].slope_weight == 4000.0


def test_pareto_front_keeps_non_dominated_rows_for_mixed_directions():
    rows = [
        {"id": "a", "slope_change_pct": -1.0, "cont_change": 0.01, "baimu_area_change_ha": 1.0},
        {"id": "b", "slope_change_pct": -1.2, "cont_change": 0.02, "baimu_area_change_ha": 2.0},
        {"id": "c", "slope_change_pct": -1.4, "cont_change": 0.005, "baimu_area_change_ha": 1.5},
    ]

    front = pareto_front(
        rows,
        objectives={
            "slope_change_pct": "min",
            "cont_change": "max",
            "baimu_area_change_ha": "max",
        },
    )

    ids = {row["id"] for row in front}
    assert ids == {"b", "c"}
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_reward_components.py -q
```

Expected: FAIL because `paper7.reward_components` does not exist.

- [ ] **Step 3: Implement reward utility module**

Create `paper7/reward_components.py` with:

```python
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
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_reward_components.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add paper7/reward_components.py tests/test_reward_components.py
git commit -m "test: add reward component utilities"
```

---

### Task 2: Real-Environment Component Rollouts

**Files:**
- Create: `paper7/reward_component_rollouts.py`
- Create: `tests/test_reward_component_rollouts.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_reward_component_rollouts.py` with:

```python
import numpy as np

from paper7.reward_component_rollouts import (
    choose_action,
    component_from_step_state,
    summarize_episode,
)


def test_component_from_step_state_matches_county_env_delta_definitions():
    component = component_from_step_state(
        prev_slope=10.0,
        cur_slope=9.0,
        initial_slope=10.0,
        prev_cont=2.0,
        cur_cont=2.2,
        initial_cont=2.0,
        prev_baimu_area=100.0,
        cur_baimu_area=120.0,
        initial_farm_area=1000.0,
        prev_baimu_count=3,
        cur_baimu_count=5,
        completed_swaps=4,
    )

    assert component.slope_delta == 0.1
    assert round(component.cont_delta, 6) == 0.1
    assert component.baimu_area_delta == 0.02
    assert component.baimu_new_count == 2
    assert component.completed_swaps == 4


def test_choose_action_obeys_mask_for_supported_heuristics():
    block_features = np.zeros((3, 17), dtype=np.float32)
    block_features[:, 3] = [0.5, 0.9, 0.4]
    block_features[:, 7] = [0.2, 0.4, 0.9]
    block_features[:, 8] = [0.2, 0.4, 0.9]
    block_features[:, 13] = [0.1, 0.8, 0.2]
    mask = np.array([True, False, True])
    rng = np.random.default_rng(0)

    assert choose_action("dynamic_slope_gap", block_features, mask, rng) == 0
    assert choose_action("area_weighted_slope_gap", block_features, mask, rng) == 2
    assert choose_action("contiguity_aware", block_features, mask, rng) == 0
    assert choose_action("baimu_aware", block_features, mask, rng) in {0, 2}
    assert choose_action("scalarized_default", block_features, mask, rng) in {0, 2}


def test_summarize_episode_reports_final_metrics_and_total_components():
    steps = [
        {
            "reward_default": 1.0,
            "slope_delta": 0.1,
            "cont_delta": 0.2,
            "baimu_area_delta": -0.01,
            "baimu_new_count": 0,
            "completed_swaps": 5,
            "slope_change_pct": -0.5,
            "cont_change": 0.01,
            "baimu_count_change": 1,
            "baimu_area_change_ha": -2.0,
        },
        {
            "reward_default": 2.0,
            "slope_delta": 0.2,
            "cont_delta": 0.1,
            "baimu_area_delta": 0.03,
            "baimu_new_count": 1,
            "completed_swaps": 5,
            "slope_change_pct": -0.8,
            "cont_change": 0.03,
            "baimu_count_change": 2,
            "baimu_area_change_ha": 4.0,
        },
    ]

    summary = summarize_episode("x", 7, steps)

    assert summary["policy"] == "x"
    assert summary["seed"] == 7
    assert summary["steps"] == 2
    assert summary["reward_default"] == 3.0
    assert summary["slope_delta_total"] == 0.3
    assert summary["final_slope_change_pct"] == -0.8
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_reward_component_rollouts.py -q
```

Expected: FAIL because `paper7.reward_component_rollouts` does not exist.

- [ ] **Step 3: Implement rollout script**

Create `paper7/reward_component_rollouts.py` with:

```python
"""Collect real-environment reward component rollouts for Paper 7."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from county_env import CountyLevelEnv, K_BLOCK, K_GLOBAL_COUNTY
from paper7.reward_components import (
    RewardComponents,
    compute_scalar_reward,
    default_reward_weights,
)


POLICIES = (
    "random",
    "dynamic_slope_gap",
    "area_weighted_slope_gap",
    "contiguity_aware",
    "baimu_aware",
    "scalarized_default",
)


def component_from_step_state(
    *,
    prev_slope: float,
    cur_slope: float,
    initial_slope: float,
    prev_cont: float,
    cur_cont: float,
    initial_cont: float,
    prev_baimu_area: float,
    cur_baimu_area: float,
    initial_farm_area: float,
    prev_baimu_count: int,
    cur_baimu_count: int,
    completed_swaps: int,
) -> RewardComponents:
    return RewardComponents(
        slope_delta=(float(prev_slope) - float(cur_slope)) / (abs(float(initial_slope)) + 1e-8),
        cont_delta=(float(cur_cont) - float(prev_cont)) / (abs(float(initial_cont)) + 1e-8),
        baimu_area_delta=(float(cur_baimu_area) - float(prev_baimu_area)) / (float(initial_farm_area) + 1e-8),
        baimu_new_count=max(0, int(cur_baimu_count) - int(prev_baimu_count)),
        completed_swaps=int(completed_swaps),
    )


def choose_action(
    policy: str,
    block_features: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return 0
    if policy == "random":
        return int(rng.choice(valid))

    scores = np.full(block_features.shape[0], -np.inf, dtype=np.float64)
    gain = block_features[:, 3].astype(np.float64)
    farm_area = block_features[:, 7].astype(np.float64)
    forest_area = block_features[:, 8].astype(np.float64)
    swap_potential = block_features[:, 9].astype(np.float64)
    neighbor_invest = block_features[:, 13].astype(np.float64)
    neighbor_farm = block_features[:, 14].astype(np.float64)
    current_farm = block_features[:, 15].astype(np.float64)

    if policy == "dynamic_slope_gap":
        raw = gain
    elif policy == "area_weighted_slope_gap":
        raw = gain * np.minimum(farm_area, forest_area)
    elif policy == "contiguity_aware":
        raw = gain + 0.25 * neighbor_farm + 0.10 * neighbor_invest
    elif policy == "baimu_aware":
        raw = gain + 0.50 * current_farm + 0.25 * swap_potential
    elif policy == "scalarized_default":
        raw = 4000.0 * gain + 500.0 * neighbor_farm + 1500.0 * current_farm + 5.0 * swap_potential
    else:
        raise ValueError(f"Unsupported policy {policy!r}")

    scores[valid] = raw[valid]
    return int(np.argmax(scores))


def run_episode(policy: str, seed: int, budget: int = 500, swaps_per_step: int = 5) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    env = CountyLevelEnv(total_budget=budget, swaps_per_step=swaps_per_step)
    obs, _ = env.reset(seed=seed)
    done = False
    steps: list[dict[str, Any]] = []

    while not done:
        prev_slope = float(env.prev_slope)
        prev_cont = float(env.prev_cont)
        prev_baimu_area = float(env.prev_baimu_area)
        prev_baimu_count = int(env.prev_baimu_count)

        block_features = obs[: env.n_blocks * K_BLOCK].reshape(env.n_blocks, K_BLOCK)
        mask = env.action_masks()
        action = choose_action(policy, block_features, mask, rng)
        next_obs, real_reward, terminated, truncated, info = env.step(action)
        component = component_from_step_state(
            prev_slope=prev_slope,
            cur_slope=float(env.avg_farmland_slope),
            initial_slope=float(env.initial_slope),
            prev_cont=prev_cont,
            cur_cont=float(env.contiguity),
            initial_cont=float(env.initial_cont),
            prev_baimu_area=prev_baimu_area,
            cur_baimu_area=float(env.baimu_total_area),
            initial_farm_area=float(env.initial_farm_area),
            prev_baimu_count=prev_baimu_count,
            cur_baimu_count=int(env.baimu_count),
            completed_swaps=int(info.get("completed_swaps", 0)),
        )
        recomputed_reward = compute_scalar_reward(component, default_reward_weights())
        step = {
            "step": int(info.get("step", len(steps) + 1)),
            "action": int(action),
            "real_reward": float(real_reward),
            "reward_default": float(recomputed_reward),
            **component.to_dict(),
            "slope_change_pct": float(info.get("slope_change_pct", 0.0)),
            "cont_change": float(info.get("cont_change", 0.0)),
            "baimu_count_change": int(info.get("baimu_count_change", 0)),
            "baimu_area_change_ha": float(info.get("baimu_area_change_ha", 0.0)),
            "budget_used": int(info.get("budget_used", 0)),
            "completed_swaps_info": int(info.get("completed_swaps", 0)),
        }
        steps.append(step)
        obs = next_obs
        done = terminated or truncated

    return {
        "policy": policy,
        "seed": int(seed),
        "budget": int(budget),
        "swaps_per_step": int(swaps_per_step),
        "summary": summarize_episode(policy, seed, steps),
        "steps": steps,
    }


def summarize_episode(policy: str, seed: int, steps: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "policy": policy,
        "seed": int(seed),
        "steps": len(steps),
        "reward_default": round(sum(float(step["reward_default"]) for step in steps), 6),
        "real_reward": round(sum(float(step.get("real_reward", 0.0)) for step in steps), 6),
        "slope_delta_total": round(sum(float(step["slope_delta"]) for step in steps), 6),
        "cont_delta_total": round(sum(float(step["cont_delta"]) for step in steps), 6),
        "baimu_area_delta_total": round(sum(float(step["baimu_area_delta"]) for step in steps), 6),
        "baimu_new_count_total": int(sum(int(step["baimu_new_count"]) for step in steps)),
        "completed_swaps_total": int(sum(int(step["completed_swaps"]) for step in steps)),
    }
    if steps:
        last = steps[-1]
        summary.update(
            {
                "final_slope_change_pct": round(float(last["slope_change_pct"]), 6),
                "final_cont_change": round(float(last["cont_change"]), 6),
                "final_baimu_count_change": int(last["baimu_count_change"]),
                "final_baimu_area_change_ha": round(float(last["baimu_area_change_ha"]), 6),
                "final_budget_used": int(last["budget_used"]),
            }
        )
    return summary


def run_suite(policies: list[str], seeds: list[int], budget: int, swaps_per_step: int) -> dict[str, Any]:
    episodes = []
    for policy in policies:
        for seed in seeds:
            episodes.append(run_episode(policy, seed, budget=budget, swaps_per_step=swaps_per_step))
    return {
        "description": "Real CountyLevelEnv reward-component rollouts for reward-weight sensitivity.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policies": policies,
        "seeds": seeds,
        "budget": int(budget),
        "swaps_per_step": int(swaps_per_step),
        "k_block": K_BLOCK,
        "k_global": K_GLOBAL_COUNTY,
        "episodes": episodes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policies", default=",".join(POLICIES))
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/full_rigor/reward_component_rollouts.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"Unsupported policies: {unknown}")
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    report = run_suite(policies, seeds, args.budget, args.swaps_per_step)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "n_episodes": len(report["episodes"])}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_reward_component_rollouts.py -q
```

Expected: PASS.

- [ ] **Step 5: Run a smoke rollout artifact**

Run:

```powershell
python -m paper7.reward_component_rollouts --policies random,dynamic_slope_gap --seeds 0 --budget 50 --output paper7/results/full_rigor/reward_component_rollouts_smoke.json
```

Expected: writes `paper7/results/full_rigor/reward_component_rollouts_smoke.json` with 2 episodes.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/reward_component_rollouts.py tests/test_reward_component_rollouts.py paper7/results/full_rigor/reward_component_rollouts_smoke.json
git commit -m "test: add reward component rollouts"
```

---

### Task 3: Reward Weight Sensitivity And Pareto Replay

**Files:**
- Create: `paper7/reward_weight_sensitivity.py`
- Create: `tests/test_reward_weight_sensitivity.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_reward_weight_sensitivity.py` with:

```python
from paper7.reward_components import RewardWeights
from paper7.reward_weight_sensitivity import (
    replay_episode_reward,
    summarize_replayed_episodes,
)


def _episode(policy, seed, slope, cont, baimu_area, baimu_count):
    return {
        "policy": policy,
        "seed": seed,
        "summary": {
            "final_slope_change_pct": slope,
            "final_cont_change": cont,
            "final_baimu_area_change_ha": baimu_area,
            "final_baimu_count_change": baimu_count,
        },
        "steps": [
            {
                "slope_delta": 0.01,
                "cont_delta": 0.02,
                "baimu_area_delta": 0.03,
                "baimu_new_count": 1,
                "completed_swaps": 5,
            }
        ],
    }


def test_replay_episode_reward_uses_requested_weights():
    episode = _episode("p", 0, -1.0, 0.01, 2.0, 1)
    reward = replay_episode_reward(episode, RewardWeights(slope_weight=1, cont_weight=1, baimu_weight=1, baimu_bonus=1))

    assert round(reward, 6) == 1.06


def test_summarize_replayed_episodes_groups_by_policy_and_weight_name():
    episodes = [
        _episode("a", 0, -1.0, 0.01, 2.0, 1),
        _episode("a", 1, -1.2, 0.02, 3.0, 2),
        _episode("b", 0, -0.5, 0.03, 5.0, 3),
    ]

    report = summarize_replayed_episodes(
        episodes,
        weight_grid=[{"name": "unit", "weights": RewardWeights(slope_weight=1, cont_weight=1, baimu_weight=1, baimu_bonus=1)}],
    )

    rows = report["policy_weight_summaries"]
    row = next(item for item in rows if item["policy"] == "a" and item["weight_name"] == "unit")
    assert row["n"] == 2
    assert row["slope_change_pct_mean"] == -1.1
    assert row["baimu_count_change_mean"] == 1.5
    assert report["pareto_front"]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_reward_weight_sensitivity.py -q
```

Expected: FAIL because `paper7.reward_weight_sensitivity` does not exist.

- [ ] **Step 3: Implement sensitivity replay script**

Create `paper7/reward_weight_sensitivity.py` with:

```python
"""Replay reward component logs under alternative reward weights."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paper7.reward_components import (
    RewardComponents,
    RewardWeights,
    compute_scalar_reward,
    generate_weight_grid,
    pareto_front,
)


def replay_episode_reward(episode: dict[str, Any], weights: RewardWeights) -> float:
    total = 0.0
    for step in episode["steps"]:
        components = RewardComponents(
            slope_delta=float(step["slope_delta"]),
            cont_delta=float(step["cont_delta"]),
            baimu_area_delta=float(step["baimu_area_delta"]),
            baimu_new_count=int(step["baimu_new_count"]),
            completed_swaps=int(step["completed_swaps"]),
        )
        total += compute_scalar_reward(components, weights)
    return float(total)


def summarize_replayed_episodes(
    episodes: list[dict[str, Any]],
    weight_grid: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if weight_grid is None:
        weight_grid = generate_weight_grid()

    rows: list[dict[str, Any]] = []
    policy_names = sorted({str(ep["policy"]) for ep in episodes})
    for item in weight_grid:
        weight_name = str(item["name"])
        weights = item["weights"]
        for policy in policy_names:
            policy_eps = [ep for ep in episodes if ep["policy"] == policy]
            replay_rewards = [replay_episode_reward(ep, weights) for ep in policy_eps]
            slopes = [float(ep["summary"]["final_slope_change_pct"]) for ep in policy_eps]
            conts = [float(ep["summary"]["final_cont_change"]) for ep in policy_eps]
            baimu_areas = [float(ep["summary"]["final_baimu_area_change_ha"]) for ep in policy_eps]
            baimu_counts = [float(ep["summary"]["final_baimu_count_change"]) for ep in policy_eps]
            rows.append(
                {
                    "policy": policy,
                    "weight_name": weight_name,
                    "weights": weights.to_dict(),
                    "n": len(policy_eps),
                    "replayed_reward_mean": _mean(replay_rewards),
                    "replayed_reward_sd": _sd(replay_rewards),
                    "slope_change_pct_mean": _mean(slopes),
                    "slope_change_pct_sd": _sd(slopes),
                    "cont_change_mean": _mean(conts),
                    "baimu_area_change_ha_mean": _mean(baimu_areas),
                    "baimu_count_change_mean": _mean(baimu_counts),
                }
            )

    front = pareto_front(
        rows,
        objectives={
            "slope_change_pct_mean": "min",
            "cont_change_mean": "max",
            "baimu_area_change_ha_mean": "max",
            "baimu_count_change_mean": "max",
        },
    )
    default_rows = [row for row in rows if row["weight_name"] == "default"]
    return {
        "description": "Reward component replay under alternative reward weights. Final planning metrics are real-environment metrics from the original rollouts.",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_episodes": len(episodes),
        "n_weight_settings": len(weight_grid),
        "policy_weight_summaries": rows,
        "pareto_front": front,
        "default_weight_rows": default_rows,
        "interpretation_boundary": "This replays scalar rewards for fixed action sequences; it does not replace retraining policies under each weight setting.",
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _sd(values: list[float]) -> float | None:
    if not values:
        return None
    mean = sum(values) / len(values)
    return round((sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5, 6)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollouts", type=Path, default=Path("paper7/results/full_rigor/reward_component_rollouts.json"))
    parser.add_argument("--output", type=Path, default=Path("paper7/results/full_rigor/reward_weight_sensitivity.json"))
    parser.add_argument("--pareto-output", type=Path, default=Path("paper7/results/full_rigor/reward_pareto_front.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts = json.loads(args.rollouts.read_text(encoding="utf-8"))
    report = summarize_replayed_episodes(rollouts["episodes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.pareto_output.write_text(json.dumps(report["pareto_front"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "pareto_output": os.fspath(args.pareto_output)}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_reward_weight_sensitivity.py -q
```

Expected: PASS.

- [ ] **Step 5: Generate smoke sensitivity artifacts**

Run:

```powershell
python -m paper7.reward_weight_sensitivity --rollouts paper7/results/full_rigor/reward_component_rollouts_smoke.json --output paper7/results/full_rigor/reward_weight_sensitivity_smoke.json --pareto-output paper7/results/full_rigor/reward_pareto_front_smoke.json
```

Expected: writes both smoke JSON artifacts.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/reward_weight_sensitivity.py tests/test_reward_weight_sensitivity.py paper7/results/full_rigor/reward_weight_sensitivity_smoke.json paper7/results/full_rigor/reward_pareto_front_smoke.json
git commit -m "test: add reward weight sensitivity replay"
```

---

### Task 4: Full Reward-Rigor Run

**Files:**
- Output: `paper7/results/full_rigor/reward_component_rollouts.json`
- Output: `paper7/results/full_rigor/reward_weight_sensitivity.json`
- Output: `paper7/results/full_rigor/reward_pareto_front.json`

- [ ] **Step 1: Run full heuristic component rollouts**

Run:

```powershell
python -m paper7.reward_component_rollouts --policies random,dynamic_slope_gap,area_weighted_slope_gap,contiguity_aware,baimu_aware,scalarized_default --seeds 0,1,2,3,4,5,6,7,8,9 --budget 500 --output paper7/results/full_rigor/reward_component_rollouts.json
```

Expected: writes 60 real-environment heuristic episodes with per-step component logs.

- [ ] **Step 2: Replay rewards under weight grid**

Run:

```powershell
python -m paper7.reward_weight_sensitivity --rollouts paper7/results/full_rigor/reward_component_rollouts.json --output paper7/results/full_rigor/reward_weight_sensitivity.json --pareto-output paper7/results/full_rigor/reward_pareto_front.json
```

Expected: writes reward sensitivity and Pareto artifacts.

- [ ] **Step 3: Inspect high-level result counts**

Run:

```powershell
Get-Content -Raw -LiteralPath paper7\results\full_rigor\reward_weight_sensitivity.json
```

Expected: JSON includes `"n_episodes": 60`, `"n_weight_settings": 14`, and non-empty `"pareto_front"`.

- [ ] **Step 4: Commit artifacts**

Run:

```powershell
git add paper7/results/full_rigor/reward_component_rollouts.json paper7/results/full_rigor/reward_weight_sensitivity.json paper7/results/full_rigor/reward_pareto_front.json
git commit -m "test: run reward rigor sensitivity experiments"
```

---

### Task 5: Evidence Audit Hook

**Files:**
- Modify: `paper7/end_to_end_validation.py`
- Modify: `tests/test_end_to_end_validation.py`

- [ ] **Step 1: Add failing test**

Add this test to `tests/test_end_to_end_validation.py`:

```python
def test_reward_rigor_scope_is_bounded_when_weight_sensitivity_exists():
    from paper7.end_to_end_validation import classify_claim_scope

    evidence = {
        "reward_weight_sensitivity": {
            "status": "supported_as_fixed_policy_reward_sensitivity",
            "n_episodes": 60,
            "n_weight_settings": 14,
        }
    }

    scopes = classify_claim_scope(evidence)
    reward_scope = next(item for item in scopes if item["id"] == "reward_function_scope")

    assert reward_scope["status"] == "supported_as_fixed_policy_reward_sensitivity"
    assert reward_scope["policy_retraining_under_all_weights"] is False
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests/test_end_to_end_validation.py -q
```

Expected: FAIL until `classify_claim_scope` emits `reward_function_scope`.

- [ ] **Step 3: Add audit summary functions**

Modify `paper7/end_to_end_validation.py` by adding:

```python
def summarize_reward_weight_sensitivity(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    data = read_json(path)
    return {
        "status": "supported_as_fixed_policy_reward_sensitivity",
        "path": str(path),
        "n_episodes": data.get("n_episodes"),
        "n_weight_settings": data.get("n_weight_settings"),
        "n_policy_weight_summaries": len(data.get("policy_weight_summaries", [])),
        "n_pareto_rows": len(data.get("pareto_front", [])),
        "policy_retraining_under_all_weights": False,
        "interpretation": "fixed-policy reward-component replay; supports reward preference analysis but does not prove retrained-policy robustness under every weight setting",
    }
```

Then add it to the evidence dictionary in `build_validation_report`:

```python
evidence["reward_weight_sensitivity"] = summarize_reward_weight_sensitivity(
    root / "paper7" / "results" / "full_rigor" / "reward_weight_sensitivity.json"
)
```

Then extend `classify_claim_scope` with:

```python
reward_sensitivity = evidence.get("reward_weight_sensitivity", {})
if reward_sensitivity.get("status") == "supported_as_fixed_policy_reward_sensitivity":
    scopes.append(
        {
            "id": "reward_function_scope",
            "claim": "The reward function has been tested through fixed-policy component replay across alternative weight settings.",
            "status": "supported_as_fixed_policy_reward_sensitivity",
            "evidence_level": "fixed_policy_reward_component_replay",
            "policy_retraining_under_all_weights": False,
            "interpretation": reward_sensitivity.get("interpretation"),
        }
    )
```

- [ ] **Step 4: Run targeted audit tests**

Run:

```powershell
python -m pytest tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 5: Regenerate audit**

Run:

```powershell
python -m paper7.end_to_end_validation --out paper7/results/full_rigor/full_rigor_evidence_audit.json
```

Expected: audit includes `reward_weight_sensitivity` and `reward_function_scope`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/end_to_end_validation.py tests/test_end_to_end_validation.py paper7/results/full_rigor/full_rigor_evidence_audit.json
git commit -m "test: audit reward sensitivity evidence"
```

---

### Task 6: Verification Checkpoint

**Files:**
- No source edits unless verification exposes a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_reward_components.py tests/test_reward_component_rollouts.py tests/test_reward_weight_sensitivity.py tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```powershell
python -m pytest tests -q --basetmp .\.pytest-tmp-paper7-reward-rigor
```

Expected: PASS.

- [ ] **Step 3: Inspect Git status**

Run:

```powershell
git status -sb
```

Expected: clean working tree after commits, or only known untracked cache/log files.

- [ ] **Step 4: Record checkpoint for the user**

Report:

- whether default reward exactly matches `CountyLevelEnv`,
- number of real-environment episodes collected,
- key Pareto result: which policies and weight settings are non-dominated,
- whether the reward analysis is fixed-policy replay or retrained-policy robustness.

## Self-Review

- Spec coverage: this plan covers Phase 1 of the full-rigor design: reward decomposition, fixed-policy sensitivity replay, Pareto analysis, audit hook, and verification. Dongxing full environment and MBRL are intentionally separate follow-up plans.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: `RewardWeights`, `RewardComponents`, `compute_scalar_reward`, `generate_weight_grid`, `pareto_front`, `component_from_step_state`, and `replay_episode_reward` names are consistent across tests and implementation steps.
