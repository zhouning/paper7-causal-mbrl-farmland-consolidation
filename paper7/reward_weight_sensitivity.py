"""Replay reward component logs under alternative reward weights."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PAPER7_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER7_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper7.reward_components import (
    RewardComponents,
    RewardWeights,
    compute_scalar_reward,
    generate_weight_grid,
    pareto_front,
    reward_specification,
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
    metric_rows: list[dict[str, Any]] = []
    policy_names = sorted({str(ep["policy"]) for ep in episodes})
    for policy in policy_names:
        policy_eps = [ep for ep in episodes if ep["policy"] == policy]
        slopes = [float(ep["summary"]["final_slope_change_pct"]) for ep in policy_eps]
        conts = [float(ep["summary"]["final_cont_change"]) for ep in policy_eps]
        baimu_areas = [float(ep["summary"]["final_baimu_area_change_ha"]) for ep in policy_eps]
        baimu_counts = [float(ep["summary"]["final_baimu_count_change"]) for ep in policy_eps]
        metric_rows.append(
            {
                "policy": policy,
                "n": len(policy_eps),
                "slope_change_pct_mean": _mean(slopes),
                "slope_change_pct_sd": _sd(slopes),
                "cont_change_mean": _mean(conts),
                "baimu_area_change_ha_mean": _mean(baimu_areas),
                "baimu_count_change_mean": _mean(baimu_counts),
            }
        )

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
        metric_rows,
        objectives={
            "slope_change_pct_mean": "min",
            "cont_change_mean": "max",
            "baimu_area_change_ha_mean": "max",
            "baimu_count_change_mean": "max",
        },
    )
    default_rows = [row for row in rows if row["weight_name"] == "default"]
    best_policy_by_weight = []
    for item in weight_grid:
        weight_name = str(item["name"])
        candidates = [row for row in rows if row["weight_name"] == weight_name]
        if candidates:
            best_policy_by_weight.append(max(candidates, key=lambda row: float(row["replayed_reward_mean"])))
    return {
        "description": (
            "Reward component replay under alternative reward weights. Final planning "
            "metrics are real-environment metrics from the original rollouts."
        ),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_episodes": len(episodes),
        "n_weight_settings": len(weight_grid),
        "policy_weight_summaries": rows,
        "policy_metric_summaries": metric_rows,
        "pareto_front": front,
        "default_weight_rows": default_rows,
        "best_policy_by_weight": best_policy_by_weight,
        "reward_specification": reward_specification(),
        "interpretation_boundary": (
            "This replays scalar rewards for fixed action sequences; it does not "
            "replace retraining policies under each weight setting."
        ),
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
    parser.add_argument(
        "--rollouts",
        type=Path,
        default=Path("paper7/results/full_rigor/reward_component_rollouts.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/full_rigor/reward_weight_sensitivity.json"),
    )
    parser.add_argument(
        "--pareto-output",
        type=Path,
        default=Path("paper7/results/full_rigor/reward_pareto_front.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rollouts = json.loads(args.rollouts.read_text(encoding="utf-8"))
    report = summarize_replayed_episodes(rollouts["episodes"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.pareto_output.write_text(json.dumps(report["pareto_front"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": os.fspath(args.output),
                "pareto_output": os.fspath(args.pareto_output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
