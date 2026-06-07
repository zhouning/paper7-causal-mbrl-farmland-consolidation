# Paper 7: Causal MBRL for Farmland Consolidation

This repository contains the complete local research package for Paper 7:

**A Causally Calibrated Learned Environment for Reinforcement Learning-Based Farmland Consolidation Planning**

The repository is organized to let reviewers inspect and reproduce the Paper 7 pipeline: county-level farmland consolidation environment, trajectory collection, learned transition model training, causal reward calibration, model-based policy training, ablation results, and the CEUS submission package.

## Repository Map

| Path | Purpose |
|---|---|
| `paper7/` | Paper 7 code, trajectories, learned model, experiment results, original manuscript files, and conversion scripts. |
| `county_env.py` | County-level real environment used for trajectory collection and final evaluation. |
| `parcel_scoring_policy.py` | Maskable PPO custom scoring policy used by model-free/model-based policies. |
| `dem_slope_analysis/output/` | Required geospatial input for the county environment. |
| `results_real/blocks/` | Required block definitions and per-township block metadata for the real environment. |
| `submission/ceus/` | CEUS submission package: anonymous manuscript, title page, highlights, cover letter, declarations, supplementary notes, and editable LaTeX sources. |
| `docs/` | Paper 7 implementation plan and pre-submission review notes. |

## Large Files

The largest trajectory and geospatial input files are distributed as GitHub Release assets instead of Git-tracked files:

- `paper7/trajectories/*.npz`
- `dem_slope_analysis/output/DLTB_with_slope.gpkg`

After cloning, download the release assets and place them at the paths listed in `DATA_ASSETS.md`.

If these files are missing, the learned-environment training and real-environment reproduction steps will not be complete. The repository keeps all code, manuscripts, result summaries, trained policy artifacts, and small supporting data in Git.

## Reproduction Overview

The main workflow is:

1. Collect trajectories from the real county environment:
   ```bash
   python paper7/collect_trajectories.py --policy random --seeds 3 --episodes 20
   python paper7/collect_trajectories.py --policy greedy --seeds 3 --episodes 20
   ```

2. Train the transition model:
   ```bash
   python paper7/train_learned_env.py --epochs 100 --policies random greedy
   ```

3. Estimate causal reward calibration:
   ```bash
   python paper7/causal_reward_calibration.py
   ```

4. Train and evaluate model-based policies:
   ```bash
   python paper7/train_model_based.py --mode learned --seed 0 --timesteps 100000
   ```

5. Run ablations:
   ```bash
   python paper7/ablation_experiments.py
   python paper7/ablation_geofm.py
   ```

See `REPRODUCIBILITY.md` for environment setup, expected inputs, and result locations.

## Current Submission Files

The CEUS-targeted manuscript package is under `submission/ceus/`.

Key files:

- `submission/ceus/01_main_document_anonymous/manuscript.pdf`
- `submission/ceus/01_main_document_anonymous/manuscript.tex`
- `submission/ceus/06_latex_source_editable/manuscript_signed.pdf`
- `submission/ceus/CEUS_paper7_latex_source_anonymous.zip`

## Notes For Reviewers

The repository intentionally includes both generated result artifacts and manuscript submission files. This makes the repository larger than a minimal code release, but preserves the complete Paper 7 research state needed for verification.

Generated Python caches are not part of the reproducibility record and are ignored by Git.
