# Paper 7 Dongxing Full Learned Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a bounded learned-policy experiment in the Dongxing full multi-objective real environment and compare it against the 60-run full-reward baseline suite.

**Architecture:** Build a lightweight linear preference learner on top of `GenericCountyEnv` observations and rewards, reusing the full environment and baseline summary utilities. The experiment trains on real-environment episodes with epsilon-greedy exploration, evaluates deterministic learned policies in the real environment, compares against `dongxing_full_baselines.json`, and adds a separate audit scope that clearly states this is local Dongxing full-reward learned-policy actionability, not Bishan-to-Dongxing transfer and not transition-model MBRL.

**Tech Stack:** Python, NumPy, Gymnasium-style `GenericCountyEnv`, JSON artifacts, pytest.

---

## File Structure

- Create: `paper7/dongxing_full_learned_policy.py`  
  Full-reward Dongxing learned preference policy training, evaluation, baseline comparison, and CLI.
- Create: `tests/test_dongxing_full_learned_policy.py`  
  Toy environment tests for learning, deterministic evaluation, comparison, and claim boundary.
- Modify: `paper7/end_to_end_validation.py`  
  Add summarizer and claim scope for Dongxing full-reward learned policy.
- Modify: `tests/test_end_to_end_validation.py`  
  Add audit tests for the new learned-policy evidence scope.
- Output: `paper7/results/full_rigor/dongxing_full_learned_policy.json`  
  Real Dongxing full-reward learned-policy artifact.
- Output: `paper7/results/full_rigor/full_rigor_evidence_audit.json`  
  Updated audit including the learned-policy evidence.

---

### Task 1: Learned Policy Module

**Files:**
- Create: `paper7/dongxing_full_learned_policy.py`
- Create: `tests/test_dongxing_full_learned_policy.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dongxing_full_learned_policy.py`:

```python
import numpy as np
from shapely.geometry import box

from paper7.dongxing_full_learned_policy import (
    compare_to_full_baselines,
    evaluate_preference_policy,
    train_preference_policy,
)
from paper7.generic_county_env import GenericCountyEnv


def _toy_env() -> GenericCountyEnv:
    parcels = [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 4.0, "geometry": box(3, 0, 4, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 8.0, "geometry": box(4, 0, 5, 1)},
    ]
    return GenericCountyEnv(
        parcels=parcels,
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        total_budget=2,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )


def test_train_preference_policy_learns_positive_gain_weight():
    policy = train_preference_policy(
        env_factory=_toy_env,
        train_seeds=[0, 1],
        episodes=12,
        learning_rate=0.05,
        epsilon=0.25,
    )

    assert policy["learner_type"] == "linear_preference_full_reward"
    assert policy["weights"][0] > 0
    assert policy["training"]["episodes"] == 12


def test_evaluate_preference_policy_reports_full_metrics():
    policy = {
        "learner_type": "linear_preference_full_reward",
        "weights": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }

    result = evaluate_preference_policy(_toy_env(), policy, seed=0)

    assert result["policy"] == "learned_full_reward_preference"
    assert result["completed_swaps"] == 1
    assert result["slope_change_pct"] < 0
    assert "cont_change" in result
    assert "baimu_area_change_ha" in result


def test_compare_to_full_baselines_reports_deltas():
    learned_summary = {
        "reward_mean": 12.0,
        "slope_change_pct_mean": -1.0,
        "cont_change_mean": 0.1,
        "baimu_area_change_ha_mean": 5.0,
    }
    baselines = {
        "policy_summaries": {
            "random": {"reward_mean": 2.0, "slope_change_pct_mean": -0.2},
            "scalarized_default": {"reward_mean": 10.0, "slope_change_pct_mean": -0.8},
        }
    }

    comparisons = compare_to_full_baselines(learned_summary, baselines)

    assert comparisons["learned_minus_random_reward_mean"] == 10.0
    assert comparisons["learned_minus_scalarized_default_reward_mean"] == 2.0
    assert comparisons["learned_minus_scalarized_default_slope_change_pct_mean"] == -0.2
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_dongxing_full_learned_policy.py -q
```

Expected: FAIL because `paper7.dongxing_full_learned_policy` does not exist.

- [ ] **Step 3: Implement module**

Create `paper7/dongxing_full_learned_policy.py` with:

```python
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from paper7.dongxing_full_baselines import _block_features_from_obs, summarize_runs
from paper7.dongxing_full_env import build_dongxing_full_env
from paper7.generic_county_env import GenericCountyEnv, K_BLOCK_GENERIC

FEATURE_NAMES = (
    "feasible_slope_gain",
    "exchange_area_share",
    "available_farm_area_share",
    "available_forest_area_share",
    "current_farm_area_share",
    "neighbor_farmland_context",
    "used_share",
    "remaining_step_share",
)


def select_action(
    block_features: np.ndarray,
    weights: np.ndarray,
    mask: np.ndarray,
    rng: np.random.Generator,
    epsilon: float,
) -> int:
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    if epsilon > 0 and float(rng.random()) < float(epsilon):
        return int(rng.choice(valid))
    scores = block_features.astype(np.float64) @ weights.astype(np.float64)
    scores[~mask] = -np.inf
    scores += rng.normal(0.0, 1e-9, size=scores.shape)
    return int(np.argmax(scores))


def train_preference_policy(
    env_factory: Callable[[], GenericCountyEnv],
    train_seeds: list[int],
    episodes: int,
    learning_rate: float = 0.03,
    epsilon: float = 0.20,
) -> dict[str, Any]:
    weights = np.zeros(K_BLOCK_GENERIC, dtype=np.float64)
    history: list[dict[str, Any]] = []
    for seed in train_seeds:
        rng = np.random.default_rng(int(seed))
        for episode in range(int(episodes)):
            env = env_factory()
            obs, _ = env.reset(seed=int(seed) * 100_000 + episode)
            total_reward = 0.0
            steps = 0
            last_info: dict[str, Any] = {}
            done = False
            while not done:
                features = _block_features_from_obs(obs, env.n_blocks).astype(np.float64)
                mask = features[:, 0] > 0.0
                if not bool(mask.any()):
                    break
                action = select_action(features, weights, mask, rng, epsilon=epsilon)
                chosen = features[action].copy()
                obs, reward, terminated, truncated, info = env.step(action)
                weights += float(learning_rate) * float(reward) * chosen
                total_reward += float(reward)
                steps += 1
                last_info = info
                done = terminated or truncated
            history.append(
                {
                    "seed": int(seed),
                    "episode": int(episode),
                    "steps": int(steps),
                    "reward": round(float(total_reward), 6),
                    "slope_change_pct": float(last_info.get("slope_change_pct", 0.0)),
                    "completed_swaps": int(last_info.get("budget_used", 0)),
                }
            )
    return {
        "learner_type": "linear_preference_full_reward",
        "feature_names": list(FEATURE_NAMES),
        "weights": [round(float(value), 10) for value in weights.tolist()],
        "training": {
            "episodes": int(episodes),
            "train_seeds": [int(seed) for seed in train_seeds],
            "learning_rate": float(learning_rate),
            "epsilon": float(epsilon),
            "history_tail": history[-20:],
        },
    }


def evaluate_preference_policy(
    env: GenericCountyEnv,
    policy: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    weights = np.asarray(policy["weights"], dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    obs, _ = env.reset(seed=int(seed))
    done = False
    total_reward = 0.0
    selected_blocks: list[int] = []
    last_info: dict[str, Any] = {}
    while not done:
        features = _block_features_from_obs(obs, env.n_blocks)
        mask = features[:, 0] > 0.0
        if not bool(mask.any()):
            break
        action = select_action(features, weights, mask, rng, epsilon=0.0)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        last_info = info
        if info.get("selected_block") is not None:
            selected_blocks.append(int(info["selected_block"]))
    return {
        "policy": "learned_full_reward_preference",
        "seed": int(seed),
        "steps": int(last_info.get("step", 0)),
        "reward": round(float(total_reward), 6),
        "slope_change_pct": float(last_info.get("slope_change_pct", 0.0)),
        "cont_change": float(last_info.get("cont_change", 0.0)),
        "baimu_count_change": int(last_info.get("baimu_count_change", 0)),
        "baimu_area_change_ha": float(last_info.get("baimu_area_change_ha", 0.0)),
        "completed_swaps": int(last_info.get("budget_used", 0)),
        "unique_blocks": len(set(selected_blocks)),
        "selected_blocks_head": selected_blocks[:20],
    }


def compare_to_full_baselines(
    learned_summary: dict[str, Any],
    baseline_report: dict[str, Any],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {
        "higher_reward_is_better": True,
        "lower_slope_change_pct_is_better": True,
    }
    for policy, summary in baseline_report.get("policy_summaries", {}).items():
        for field in ("reward_mean", "slope_change_pct_mean", "cont_change_mean", "baimu_area_change_ha_mean"):
            if learned_summary.get(field) is None or summary.get(field) is None:
                continue
            comparisons[f"learned_minus_{policy}_{field}"] = round(
                float(learned_summary[field]) - float(summary[field]),
                6,
            )
    return comparisons


def run_experiment(
    env_factory: Callable[[], GenericCountyEnv],
    baseline_path: Path,
    train_seeds: list[int],
    eval_seeds: list[int],
    episodes: int,
    learning_rate: float,
    epsilon: float,
) -> dict[str, Any]:
    baseline_report = json.loads(baseline_path.read_text(encoding="utf-8-sig"))
    t0 = time.time()
    policy = train_preference_policy(
        env_factory=env_factory,
        train_seeds=train_seeds,
        episodes=episodes,
        learning_rate=learning_rate,
        epsilon=epsilon,
    )
    training_time_s = time.time() - t0
    runs = [
        evaluate_preference_policy(env_factory(), policy, seed=int(seed))
        for seed in eval_seeds
    ]
    summary = summarize_runs(runs)
    return {
        "description": "Dongxing full-reward learned preference policy evaluated in the full real environment.",
        "status": "supported_as_dongxing_full_reward_learned_policy",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "learner_type": policy["learner_type"],
        "feature_names": list(FEATURE_NAMES),
        "train_seeds": [int(seed) for seed in train_seeds],
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "episodes": int(episodes),
        "learning_rate": float(learning_rate),
        "epsilon": float(epsilon),
        "training_time_s": round(float(training_time_s), 6),
        "policy": policy,
        "learned_policy": {"summary": summary, "runs": runs},
        "baseline_path": os.fspath(baseline_path),
        "comparisons": compare_to_full_baselines(summary, baseline_report),
        "claim_boundary": (
            "Local Dongxing full-reward learned actionability; not Bishan-to-Dongxing "
            "policy transfer and not learned-transition MBRL."
        ),
    }
```

Add CLI functions:

```python
def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dltb", type=Path, default=Path("paper7/data/dongxing_DLTB_with_slope.gpkg"))
    parser.add_argument("--block-dir", type=Path, default=Path("paper7/results/dongxing_blocks_slope"))
    parser.add_argument("--baseline", type=Path, default=Path("paper7/results/full_rigor/dongxing_full_baselines.json"))
    parser.add_argument("--train-seeds", default="0,1,2,3,4")
    parser.add_argument("--eval-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--total-budget", type=int, default=500)
    parser.add_argument("--swaps-per-step", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("paper7/results/full_rigor/dongxing_full_learned_policy.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    def env_factory() -> GenericCountyEnv:
        return build_dongxing_full_env(
            dltb_path=args.dltb,
            block_dir=args.block_dir,
            total_budget=args.total_budget,
            swaps_per_step=args.swaps_per_step,
        )

    report = run_experiment(
        env_factory=env_factory,
        baseline_path=args.baseline,
        train_seeds=_parse_int_list(args.train_seeds),
        eval_seeds=_parse_int_list(args.eval_seeds),
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": os.fspath(args.output), "n_eval": len(report["learned_policy"]["runs"])}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_dongxing_full_learned_policy.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add paper7/dongxing_full_learned_policy.py tests/test_dongxing_full_learned_policy.py
git commit -m "test: add Dongxing full reward learned policy"
```

---

### Task 2: Real Dongxing Full-Reward Learned Policy Run

**Files:**
- Output: `paper7/results/full_rigor/dongxing_full_learned_policy.json`

- [ ] **Step 1: Run a small smoke**

Run:

```powershell
python -m paper7.dongxing_full_learned_policy --dltb paper7/data/dongxing_DLTB_with_slope.gpkg --block-dir paper7/results/dongxing_blocks_slope --baseline paper7/results/full_rigor/dongxing_full_baselines.json --train-seeds 0 --eval-seeds 0,1 --episodes 2 --output paper7/results/full_rigor/dongxing_full_learned_policy_smoke.json
```

Expected: writes a 2-eval smoke artifact with `status=supported_as_dongxing_full_reward_learned_policy`.

- [ ] **Step 2: Inspect smoke**

Run:

```powershell
python -c "import json; p=json.load(open('paper7/results/full_rigor/dongxing_full_learned_policy_smoke.json', encoding='utf-8')); print(p['status'], p['learned_policy']['summary']['n'])"
```

Expected: `supported_as_dongxing_full_reward_learned_policy 2`.

- [ ] **Step 3: Run full experiment**

Run:

```powershell
python -m paper7.dongxing_full_learned_policy --dltb paper7/data/dongxing_DLTB_with_slope.gpkg --block-dir paper7/results/dongxing_blocks_slope --baseline paper7/results/full_rigor/dongxing_full_baselines.json --train-seeds 0,1,2,3,4 --eval-seeds 0,1,2,3,4,5,6,7,8,9 --episodes 30 --output paper7/results/full_rigor/dongxing_full_learned_policy.json
```

Expected: writes a 10-eval learned-policy artifact. If learned policy underperforms scalarized or baimu-aware baselines, retain the result as a negative or bounded result.

- [ ] **Step 4: Inspect full result**

Run:

```powershell
python -c "import json; p=json.load(open('paper7/results/full_rigor/dongxing_full_learned_policy.json', encoding='utf-8')); print(p['status']); print(p['learned_policy']['summary']); print(p['comparisons'])"
```

Expected: shows reward, slope, contiguity, baimu metrics, and learned-minus-baseline deltas.

- [ ] **Step 5: Remove smoke artifact**

Run:

```powershell
Remove-Item -LiteralPath paper7\results\full_rigor\dongxing_full_learned_policy_smoke.json
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/results/full_rigor/dongxing_full_learned_policy.json
git commit -m "test: run Dongxing full reward learned policy"
```

---

### Task 3: Evidence Audit Hook

**Files:**
- Modify: `paper7/end_to_end_validation.py`
- Modify: `tests/test_end_to_end_validation.py`
- Output: `paper7/results/full_rigor/full_rigor_evidence_audit.json`

- [ ] **Step 1: Add failing audit tests**

Append to `tests/test_end_to_end_validation.py`:

```python
def test_summarize_dongxing_full_learned_policy_extracts_scope(tmp_path):
    from paper7.end_to_end_validation import summarize_dongxing_full_learned_policy

    path = tmp_path / "dongxing_full_learned_policy.json"
    _write_json(
        path,
        {
            "status": "supported_as_dongxing_full_reward_learned_policy",
            "learner_type": "linear_preference_full_reward",
            "train_seeds": [0, 1],
            "eval_seeds": [0, 1, 2],
            "learned_policy": {"summary": {"n": 3, "reward_mean": 12.0}},
            "comparisons": {"learned_minus_random_reward_mean": 10.0},
            "claim_boundary": "Local Dongxing full-reward learned actionability",
        },
    )

    summary = summarize_dongxing_full_learned_policy(path)

    assert summary["status"] == "supported_as_dongxing_full_reward_learned_policy"
    assert summary["learner_type"] == "linear_preference_full_reward"
    assert summary["n_eval_seeds"] == 3
    assert summary["learned_policy_tested"] is True
    assert summary["transfer_tested"] is False


def test_classify_claim_scope_marks_dongxing_full_learned_policy_as_local_not_transfer():
    scopes = classify_claim_scope(
        {
            "dongxing_full_learned_policy": {
                "status": "supported_as_dongxing_full_reward_learned_policy",
                "n_eval_seeds": 10,
                "interpretation": "local full-reward learned policy",
            }
        }
    )

    scope = next(item for item in scopes if item["id"] == "dongxing_full_learned_policy_scope")

    assert scope["status"] == "supported_as_dongxing_full_reward_learned_policy"
    assert scope["evidence_level"] == "external_full_reward_local_learned_policy"
    assert scope["policy_transfer_tested"] is False
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_end_to_end_validation.py -q -k "full_learned_policy"
```

Expected: FAIL until summarizer and scope exist.

- [ ] **Step 3: Implement audit summarizer and scope**

Add import in the test import list:

```python
summarize_dongxing_full_learned_policy,
```

Add to `paper7/end_to_end_validation.py`:

```python
def summarize_dongxing_full_learned_policy(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    learned = payload.get("learned_policy", {})
    return {
        "status": payload.get("status", "supported_as_dongxing_full_reward_learned_policy"),
        "path": display_path(path),
        "learner_type": payload.get("learner_type"),
        "train_seeds": payload.get("train_seeds", []),
        "eval_seeds": payload.get("eval_seeds", []),
        "n_train_seeds": len(payload.get("train_seeds", [])),
        "n_eval_seeds": len(payload.get("eval_seeds", [])),
        "training_time_s": payload.get("training_time_s"),
        "learned_summary": learned.get("summary", {}),
        "comparisons": payload.get("comparisons", {}),
        "learned_policy_tested": True,
        "transfer_tested": False,
        "mbrl_transition_model_tested": False,
        "interpretation": payload.get(
            "claim_boundary",
            "Local Dongxing full-reward learned policy; not policy transfer and not transition-model MBRL",
        ),
    }
```

In `build_validation_report`, add:

```python
evidence["dongxing_full_learned_policy"] = summarize_dongxing_full_learned_policy(
    paper7_dir / "results" / "full_rigor" / "dongxing_full_learned_policy.json"
)
```

In `classify_claim_scope`, add:

```python
dongxing_full_learned = evidence.get("dongxing_full_learned_policy", {})
```

Then append:

```python
if dongxing_full_learned.get("status") == "supported_as_dongxing_full_reward_learned_policy":
    scopes.append(
        {
            "id": "dongxing_full_learned_policy_scope",
            "claim": "Dongxing supports local full-reward learned-policy actionability.",
            "status": dongxing_full_learned.get("status"),
            "evidence_level": "external_full_reward_local_learned_policy",
            "n_eval_seeds": dongxing_full_learned.get("n_eval_seeds", 0),
            "policy_transfer_tested": False,
            "mbrl_transition_model_tested": False,
            "interpretation": dongxing_full_learned.get("interpretation"),
        }
    )
```

- [ ] **Step 4: Run audit tests**

Run:

```powershell
python -m pytest tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 5: Regenerate full-rigor audit**

Run:

```powershell
python -m paper7.end_to_end_validation --out paper7/results/full_rigor/full_rigor_evidence_audit.json
```

Expected: audit includes `dongxing_full_learned_policy` and `dongxing_full_learned_policy_scope`.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/end_to_end_validation.py tests/test_end_to_end_validation.py paper7/results/full_rigor/full_rigor_evidence_audit.json
git commit -m "test: audit Dongxing full learned policy evidence"
```

---

### Task 4: Verification Checkpoint

**Files:**
- No source edits unless verification exposes a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_dongxing_full_learned_policy.py tests/test_dongxing_full_baselines.py tests/test_generic_county_env.py tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```powershell
python -m pytest tests -q --basetemp .\.pytest-tmp-paper7-dongxing-full-learned
```

Expected: PASS. Record the existing PyTorch Windows access-violation tail if exit code remains 0.

- [ ] **Step 3: Inspect key result**

Run:

```powershell
python -c "import json; p=json.load(open('paper7/results/full_rigor/dongxing_full_learned_policy.json', encoding='utf-8')); print(p['learned_policy']['summary']); print(p['comparisons'])"
```

Expected: report whether learned policy beats random, scalarized_default, or baimu_aware on reward and slope. Do not reinterpret underperformance as success.

- [ ] **Step 4: Report checkpoint**

Report:

- learned policy status and claim boundary,
- train/eval seeds and number of episodes,
- learned policy reward/slope/contiguity/baimu metrics,
- comparison against 60-run full baselines,
- whether this is local learned-policy evidence or transfer/MBRL evidence.

## Self-Review

- Spec coverage: covers a bounded learned-policy experiment after full baselines; does not claim transition-model MBRL or transfer.
- Placeholder scan: no TODO/TBD placeholders remain.
- Type consistency: `train_preference_policy`, `evaluate_preference_policy`, `compare_to_full_baselines`, and `summarize_dongxing_full_learned_policy` names are used consistently.
- Scope boundary: keeps large trajectory/model files out of scope and stores only JSON summaries.
