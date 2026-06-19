# Paper 7 Dongxing Full Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate a Dongxing full multi-objective real environment that evaluates slope, contiguity, baimu fang, and the default Paper 7 reward, replacing slope-only external evidence with a stronger full-reward baseline layer.

**Architecture:** Add a generic parcel-block environment module that accepts externally supplied swappable parcels, geometries, block compositions, and reward weights. Keep it independent from the existing Bishan `CountyLevelEnv` so the current Bishan evidence chain remains stable, but reuse the same reward component utility and the same paired-swap semantics. Add Dongxing loader and full baseline runner that emit bounded full-rigor artifacts under `paper7/results/full_rigor/`.

**Tech Stack:** Python, GeoPandas, NumPy, Shapely spatial index or libpysal fallback, JSON, pytest, existing Dongxing block package.

---

## File Structure

- Create: `paper7/generic_county_env.py`  
  Generic full multi-objective parcel-block environment for external county data.
- Create: `paper7/dongxing_full_env.py`  
  Dongxing loader and CLI wrapper for `GenericCountyEnv`.
- Create: `paper7/dongxing_full_baselines.py`  
  Full-reward Dongxing baseline runner: random, slope-gap, area-weighted, contiguity-aware, baimu-aware, scalarized-default.
- Create: `tests/test_generic_county_env.py`  
  Toy geometry tests for paired swaps, action masks, contiguity, baimu fang, and reward decomposition.
- Create: `tests/test_dongxing_full_baselines.py`  
  Selector and summary tests on synthetic environment-like rows.
- Modify: `paper7/end_to_end_validation.py`  
  Add bounded audit status for Dongxing full real-environment baselines.
- Modify: `tests/test_end_to_end_validation.py`  
  Add audit classification tests.

---

### Task 1: Generic Full County Environment

**Files:**
- Create: `paper7/generic_county_env.py`
- Create: `tests/test_generic_county_env.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_generic_county_env.py` with:

```python
from shapely.geometry import box

from paper7.generic_county_env import GenericCountyEnv


def _toy_parcels():
    return [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 10.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 2.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 4.0, "geometry": box(3, 0, 4, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 8.0, "geometry": box(4, 0, 5, 1)},
    ]


def test_generic_env_masks_positive_gain_blocks_and_runs_full_reward():
    env = GenericCountyEnv(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_ids=[0, 1],
        total_budget=2,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )

    obs, info = env.reset(seed=0)
    assert env.action_masks().tolist() == [True, False]
    assert info["baimu_count"] == 1

    obs, reward, terminated, truncated, info = env.step(0)

    assert reward > 0
    assert terminated is False
    assert truncated is False
    assert info["completed_swaps"] == 1
    assert info["slope_change_pct"] < 0
    assert "reward_components" in info


def test_generic_env_counts_baimu_components_from_adjacency():
    parcels = [
        {"land_use": "farmland", "area_m2": 100.0, "slope": 1.0, "geometry": box(0, 0, 1, 1)},
        {"land_use": "farmland", "area_m2": 100.0, "slope": 1.0, "geometry": box(1, 0, 2, 1)},
        {"land_use": "farmland", "area_m2": 50.0, "slope": 1.0, "geometry": box(4, 0, 5, 1)},
        {"land_use": "forest", "area_m2": 100.0, "slope": 0.5, "geometry": box(5, 0, 6, 1)},
    ]
    env = GenericCountyEnv(
        parcels=parcels,
        block_compositions={"0": [2, 3]},
        block_ids=[0],
        total_budget=1,
        swaps_per_step=1,
        baimu_threshold_m2=150.0,
    )

    _, info = env.reset(seed=0)

    assert info["baimu_count"] == 1
    assert info["baimu_area_ha"] == 0.02
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_generic_county_env.py -q
```

Expected: FAIL because `paper7.generic_county_env` does not exist.

- [ ] **Step 3: Implement `GenericCountyEnv`**

Create `paper7/generic_county_env.py` with these public interfaces:

```python
class GenericCountyEnv(gym.Env):
    def __init__(
        self,
        parcels: list[dict[str, Any]],
        block_compositions: dict[str, list[int]],
        block_ids: list[int] | None = None,
        total_budget: int = 500,
        swaps_per_step: int = 5,
        reward_weights: RewardWeights | None = None,
        baimu_threshold_m2: float = 66700.0,
    ) -> None: ...

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None): ...
    def step(self, action: int): ...
    def action_masks(self) -> np.ndarray: ...
    def block_feature_matrix(self) -> np.ndarray: ...
```

Implementation requirements:

- Store `land_use` as integer codes compatible with `"farmland"` and `"forest"`.
- Compute area-weighted average farmland slope.
- Build parcel adjacency from provided Shapely geometries using spatial index `intersects`.
- Count baimu fang using union-find over currently farmland parcels.
- Use paired swap semantics: high-slope available farmland to forest and low-slope available forest to farmland in selected block, only if slope gap is positive.
- Track `swapped` so a parcel participates in at most one pair per episode.
- Return a compact flattened observation with per-block features and global metrics. It does not need to match Bishan `K_BLOCK=17`, but must include enough fields for baselines: gain, farm area, forest area, current farm area, neighbor farm context, baimu proximity proxy, used share.
- Compute scalar reward with `paper7.reward_components.compute_scalar_reward`.
- Include `reward_components` in `info`.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_generic_county_env.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add paper7/generic_county_env.py tests/test_generic_county_env.py
git commit -m "test: add generic full county environment"
```

---

### Task 2: Dongxing Full Environment Loader

**Files:**
- Create: `paper7/dongxing_full_env.py`
- Modify: `tests/test_generic_county_env.py`

- [ ] **Step 1: Add loader tests**

Append to `tests/test_generic_county_env.py`:

```python
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

from paper7.dongxing_full_env import build_env_from_frame_and_block_package


def test_dongxing_loader_uses_swappable_index_order(tmp_path):
    frame = gpd.GeoDataFrame(
        {
            "DLBM": ["011", "031", "011", "031"],
            "DLMC": ["farmland", "forest", "farmland", "forest"],
            "TBMJ": [100.0, 100.0, 100.0, 100.0],
            "slope_mean": [10.0, 2.0, 4.0, 8.0],
        },
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(3, 0, 4, 1), box(4, 0, 5, 1)],
        crs="EPSG:3857",
    )
    block_dir = tmp_path / "blocks"
    block_dir.mkdir()
    pd.DataFrame(
        [
            {"swappable_index": 0, "source_index": 0, "land_use": "farmland", "block_id": 0},
            {"swappable_index": 1, "source_index": 1, "land_use": "forest", "block_id": 0},
            {"swappable_index": 2, "source_index": 2, "land_use": "farmland", "block_id": 1},
            {"swappable_index": 3, "source_index": 3, "land_use": "forest", "block_id": 1},
        ]
    ).to_csv(block_dir / "parcel_block_mapping.csv", index=False)
    (block_dir / "block_compositions.json").write_text('{"0": [0, 1], "1": [2, 3]}', encoding="utf-8")
    (block_dir / "block_features.json").write_text('[{"block_id": 0}, {"block_id": 1}]', encoding="utf-8")

    env = build_env_from_frame_and_block_package(frame, block_dir, total_budget=2, swaps_per_step=1)

    assert env.n_blocks == 2
    assert env.action_masks().tolist() == [True, False]
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_generic_county_env.py -q
```

Expected: FAIL because `paper7.dongxing_full_env` does not exist.

- [ ] **Step 3: Implement loader**

Create `paper7/dongxing_full_env.py` with:

```python
def load_dongxing_parcels_for_full_env(dltb_path: Path, block_dir: Path, slope_field: str = "slope_mean") -> list[dict[str, Any]]:
    ...

def build_env_from_frame_and_block_package(frame: gpd.GeoDataFrame, block_dir: Path, total_budget: int = 500, swaps_per_step: int = 5) -> GenericCountyEnv:
    ...

def build_dongxing_full_env(dltb_path: Path, block_dir: Path, total_budget: int = 500, swaps_per_step: int = 5) -> GenericCountyEnv:
    ...
```

Important loader rules:

- Read `parcel_block_mapping.csv` sorted by `swappable_index`.
- Use `source_index` to retrieve geometry, area, slope, and land-use from the GeoDataFrame.
- Ensure each output parcel list index equals `swappable_index`.
- Read `block_compositions.json` as swappable indices.
- Read `block_features.json` only to get ordered `block_id` list.

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_generic_county_env.py -q
```

Expected: PASS.

- [ ] **Step 5: Smoke real Dongxing environment construction**

Run:

```powershell
python -m paper7.dongxing_full_env --dltb paper7/data/dongxing_DLTB_with_slope.gpkg --block-dir paper7/results/dongxing_blocks_slope --summary-output paper7/results/full_rigor/dongxing_full_env_smoke.json
```

Expected: writes a summary with `n_blocks=2978`, valid action count, initial slope, contiguity, baimu count, and baimu area.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/dongxing_full_env.py tests/test_generic_county_env.py paper7/results/full_rigor/dongxing_full_env_smoke.json
git commit -m "test: add Dongxing full environment loader"
```

---

### Task 3: Dongxing Full Multi-Objective Baselines

**Files:**
- Create: `paper7/dongxing_full_baselines.py`
- Create: `tests/test_dongxing_full_baselines.py`
- Output: `paper7/results/full_rigor/dongxing_full_baselines.json`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dongxing_full_baselines.py` with:

```python
import numpy as np

from paper7.dongxing_full_baselines import choose_full_env_action, summarize_runs


def test_choose_full_env_action_respects_mask():
    features = np.zeros((3, 8), dtype=np.float32)
    features[:, 0] = [0.5, 0.9, 0.3]
    features[:, 1] = [0.2, 0.4, 0.9]
    mask = np.array([True, False, True])
    rng = np.random.default_rng(0)

    assert choose_full_env_action("dynamic_slope_gap", features, mask, rng) == 0
    assert choose_full_env_action("area_weighted_slope_gap", features, mask, rng) == 2


def test_summarize_runs_reports_multi_objective_means():
    runs = [
        {"slope_change_pct": -1.0, "cont_change": 0.1, "baimu_count_change": 1, "baimu_area_change_ha": 2, "reward": 10},
        {"slope_change_pct": -2.0, "cont_change": 0.3, "baimu_count_change": 3, "baimu_area_change_ha": 6, "reward": 20},
    ]

    summary = summarize_runs(runs)

    assert summary["n"] == 2
    assert summary["slope_change_pct_mean"] == -1.5
    assert summary["cont_change_mean"] == 0.2
    assert summary["baimu_count_change_mean"] == 2.0
    assert summary["reward_mean"] == 15.0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```powershell
python -m pytest tests/test_dongxing_full_baselines.py -q
```

Expected: FAIL because `paper7.dongxing_full_baselines` does not exist.

- [ ] **Step 3: Implement full baseline runner**

Create `paper7/dongxing_full_baselines.py` with:

```python
POLICIES = (
    "random",
    "dynamic_slope_gap",
    "area_weighted_slope_gap",
    "contiguity_aware",
    "baimu_aware",
    "scalarized_default",
)

def choose_full_env_action(policy: str, block_features: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> int: ...
def run_policy_episode(env: GenericCountyEnv, policy: str, seed: int) -> dict[str, Any]: ...
def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]: ...
def run_suite(env_factory: Callable[[], GenericCountyEnv], policies: list[str], seeds: list[int]) -> dict[str, Any]: ...
```

Selector feature assumptions for `GenericCountyEnv.block_feature_matrix()`:

- column 0: feasible slope gain
- column 1: feasible exchange area share
- column 2: farm area share
- column 3: forest area share
- column 4: current block farmland area share
- column 5: neighbor farmland context
- column 6: used share
- column 7: remaining step share

Output fields per run:

- policy, seed, steps, reward
- slope_change_pct
- cont_change
- baimu_count_change
- baimu_area_change_ha
- completed_swaps
- unique_blocks
- selected_blocks_head

- [ ] **Step 4: Run targeted tests**

Run:

```powershell
python -m pytest tests/test_dongxing_full_baselines.py -q
```

Expected: PASS.

- [ ] **Step 5: Run real Dongxing full baselines**

Run:

```powershell
python -m paper7.dongxing_full_baselines --dltb paper7/data/dongxing_DLTB_with_slope.gpkg --block-dir paper7/results/dongxing_blocks_slope --policies random,dynamic_slope_gap,area_weighted_slope_gap,contiguity_aware,baimu_aware,scalarized_default --seeds 0,1,2,3,4,5,6,7,8,9 --output paper7/results/full_rigor/dongxing_full_baselines.json
```

Expected: writes 60 full-reward Dongxing real-environment episodes.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/dongxing_full_baselines.py tests/test_dongxing_full_baselines.py paper7/results/full_rigor/dongxing_full_baselines.json
git commit -m "test: run Dongxing full reward baselines"
```

---

### Task 4: Evidence Audit Hook

**Files:**
- Modify: `paper7/end_to_end_validation.py`
- Modify: `tests/test_end_to_end_validation.py`
- Output: `paper7/results/full_rigor/full_rigor_evidence_audit.json`

- [ ] **Step 1: Add failing audit tests**

Append to `tests/test_end_to_end_validation.py`:

```python
def test_summarize_dongxing_full_baselines_extracts_scope(tmp_path):
    from paper7.end_to_end_validation import summarize_dongxing_full_baselines

    path = tmp_path / "dongxing_full_baselines.json"
    _write_json(
        path,
        {
            "status": "supported_as_full_real_environment_baselines",
            "n_runs": 60,
            "n_policies": 6,
            "policy_summaries": {
                "random": {"slope_change_pct_mean": -0.1},
                "scalarized_default": {"slope_change_pct_mean": -0.5},
            },
        },
    )

    summary = summarize_dongxing_full_baselines(path)

    assert summary["status"] == "supported_as_full_real_environment_baselines"
    assert summary["n_runs"] == 60
    assert summary["has_full_reward_metrics"] is True
```

- [ ] **Step 2: Run test to verify failure**

Run:

```powershell
python -m pytest tests/test_end_to_end_validation.py -q
```

Expected: FAIL until summarizer exists.

- [ ] **Step 3: Implement audit summarizer and scope**

Add to `paper7/end_to_end_validation.py`:

```python
def summarize_dongxing_full_baselines(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "path": display_path(path)}
    payload = load_json(path)
    return {
        "status": payload.get("status", "supported_as_full_real_environment_baselines"),
        "path": display_path(path),
        "n_runs": payload.get("n_runs"),
        "n_policies": payload.get("n_policies"),
        "has_full_reward_metrics": True,
        "policy_summaries": payload.get("policy_summaries", {}),
        "interpretation": "Dongxing full real-environment baselines with slope, contiguity, baimu, and scalar reward; not learned-policy transfer",
    }
```

Add this evidence to `build_validation_report` and add a claim scope item:

```python
{
    "id": "dongxing_full_real_environment_scope",
    "claim": "Dongxing supports full multi-objective real-environment baseline evaluation.",
    "status": dongxing_full.get("status", "missing"),
    "evidence_level": "external_full_real_environment_baselines",
    "learned_policy_tested": False,
    "interpretation": dongxing_full.get("interpretation"),
}
```

- [ ] **Step 4: Regenerate audit**

Run:

```powershell
python -m paper7.end_to_end_validation --out paper7/results/full_rigor/full_rigor_evidence_audit.json
```

Expected: audit includes `dongxing_full_baselines`.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add paper7/end_to_end_validation.py tests/test_end_to_end_validation.py paper7/results/full_rigor/full_rigor_evidence_audit.json
git commit -m "test: audit Dongxing full baseline evidence"
```

---

### Task 5: Verification Checkpoint

**Files:**
- No source edits unless verification exposes a bug.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
python -m pytest tests/test_generic_county_env.py tests/test_dongxing_full_baselines.py tests/test_end_to_end_validation.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full tests**

Run:

```powershell
python -m pytest tests -q --basetemp .\.pytest-tmp-paper7-dongxing-full-env
```

Expected: PASS. Record any existing PyTorch Windows access-violation tail if exit code is still 0.

- [ ] **Step 3: Inspect key Dongxing full baseline result**

Run:

```powershell
Get-Content -Raw -LiteralPath paper7\results\full_rigor\dongxing_full_baselines.json
```

Expected: JSON includes full metrics for six policies, including slope, contiguity, baimu count, baimu area, reward, and completed swaps.

- [ ] **Step 4: Report checkpoint**

Report:

- whether Dongxing full environment was constructed successfully,
- initial valid action count and baimu metrics,
- which full-reward baseline was strongest on slope and scalar reward,
- whether this is still baseline evidence or learned-policy evidence.

## Self-Review

- Spec coverage: this plan covers Phase 2 and Phase 3 of the full-rigor design: generic full environment, Dongxing full loader, full real-environment baselines, and audit hook.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: `GenericCountyEnv`, `build_dongxing_full_env`, `choose_full_env_action`, `run_policy_episode`, and `summarize_dongxing_full_baselines` names are used consistently.
- Scope boundary: this plan does not train Dongxing learned environments; that remains the next plan after real-environment baselines exist.
