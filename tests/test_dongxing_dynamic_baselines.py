import pytest
import subprocess
import sys
from pathlib import Path

from paper7.dongxing_dynamic_baselines import (
    DynamicSwapState,
    run_baseline_suite,
    run_episode,
)


def _toy_parcels():
    return [
        {"swappable_index": 0, "land_use": "farmland", "area_m2": 100.0, "slope": 10.0},
        {"swappable_index": 1, "land_use": "forest", "area_m2": 100.0, "slope": 2.0},
        {"swappable_index": 2, "land_use": "farmland", "area_m2": 100.0, "slope": 4.0},
        {"swappable_index": 3, "land_use": "forest", "area_m2": 100.0, "slope": 8.0},
    ]


def _toy_blocks():
    return [
        {
            "block_id": 0,
            "avg_farm_slope": 10.0,
            "avg_forest_slope": 2.0,
            "farm_area_ha": 0.01,
            "forest_area_ha": 0.01,
        },
        {
            "block_id": 1,
            "avg_farm_slope": 4.0,
            "avg_forest_slope": 8.0,
            "farm_area_ha": 0.01,
            "forest_area_ha": 0.01,
        },
    ]


def test_dynamic_swap_state_executes_high_slope_out_low_slope_in_pair():
    state = DynamicSwapState.from_records(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
    )

    completed = state.execute_block(0, max_pairs=1)

    assert completed == 1
    assert state.land_use_label(0) == "forest"
    assert state.land_use_label(1) == "farmland"
    assert state.completed_pairs == 1
    assert state.avg_farmland_slope == pytest.approx(3.0)
    assert state.slope_change_pct == pytest.approx(-57.142857, rel=1e-6)
    assert state.pair_records[0]["slope_gap"] == 8.0


def test_dynamic_episode_skips_exhausted_or_negative_gain_blocks():
    result = run_episode(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_features=_toy_blocks(),
        strategy="dynamic_slope_gap",
        max_steps=5,
        swaps_per_step=1,
    )

    assert result["completed_pairs"] == 1
    assert result["steps"] == 1
    assert result["selected_blocks"] == [0]
    assert result["slope_change_pct"] == pytest.approx(-57.142857, rel=1e-6)


def test_dynamic_episode_can_reselect_block_until_local_pairs_are_exhausted():
    parcels = [
        {"swappable_index": 0, "land_use": "farmland", "area_m2": 100.0, "slope": 10.0},
        {"swappable_index": 1, "land_use": "farmland", "area_m2": 100.0, "slope": 9.0},
        {"swappable_index": 2, "land_use": "forest", "area_m2": 100.0, "slope": 2.0},
        {"swappable_index": 3, "land_use": "forest", "area_m2": 100.0, "slope": 3.0},
    ]
    blocks = [
        {
            "block_id": 0,
            "avg_farm_slope": 9.5,
            "avg_forest_slope": 2.5,
            "farm_area_ha": 0.02,
            "forest_area_ha": 0.02,
        }
    ]

    result = run_episode(
        parcels=parcels,
        block_compositions={"0": [0, 1, 2, 3]},
        block_features=blocks,
        strategy="dynamic_slope_gap",
        max_steps=5,
        swaps_per_step=1,
    )

    assert result["completed_pairs"] == 2
    assert result["selected_blocks"] == [0, 0]
    assert result["slope_change_pct"] < 0


def test_dynamic_baseline_suite_reports_random_distribution_and_empirical_p_values():
    suite = run_baseline_suite(
        parcels=_toy_parcels(),
        block_compositions={"0": [0, 1], "1": [2, 3]},
        block_features=_toy_blocks(),
        max_steps=5,
        swaps_per_step=1,
        random_seeds=[0, 1, 2, 3, 4],
    )

    assert suite["strategies"]["dynamic_slope_gap"]["completed_pairs"] == 1
    assert suite["random_baseline"]["n_seeds"] == 5
    assert suite["random_baseline"]["slope_change_pct_mean"] is not None
    assert suite["strategies"]["dynamic_slope_gap"]["random_p_slope_change_pct"] is not None
    assert "pair_records_head" not in suite["random_baseline"]["per_seed"][0]
    assert "selected_blocks" not in suite["random_baseline"]["per_seed"][0]


def test_dongxing_dynamic_baselines_script_help_runs_from_repo_root():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "paper7/dongxing_dynamic_baselines.py", "--help"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--block-dir" in result.stdout
