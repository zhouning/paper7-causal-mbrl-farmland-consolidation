# Reproducibility Guide

This guide documents the files and commands needed to reproduce Paper 7.

## Hardware

The Paper 7 learned-environment pipeline was designed to run without GPU support after trajectory collection is available. The manuscript reports:

- trajectory collection: about 40 minutes
- transition model training: about 60 minutes on CPU
- model-based policy training: about 45 minutes on CPU

Model-free baselines referenced in the paper used A100 GPU runs from the shared county-level environment.

## Python Environment

Recommended Python version: 3.10 or 3.11.

Install dependencies:

```bash
pip install -r requirements.txt
```

Geospatial packages such as `geopandas`, `libpysal`, `shapely`, and `pyproj` may require platform-specific wheels or conda packages on some systems.

## Required Input Files

The real county environment depends on:

- `dem_slope_analysis/output/DLTB_with_slope.gpkg`
- `results_real/blocks/`
- `county_env.py`
- `parcel_scoring_policy.py`

The learned transition model depends on:

- `paper7/trajectories/random_seed0.npz`
- `paper7/trajectories/random_seed1.npz`
- `paper7/trajectories/random_seed2.npz`
- `paper7/trajectories/greedy_seed0.npz`
- `paper7/trajectories/greedy_seed1.npz`
- `paper7/trajectories/greedy_seed2.npz`

These large files are distributed as GitHub Release assets. See `DATA_ASSETS.md` for names, target paths, sizes, and SHA256 checksums.

## Main Reproduction Steps

Run commands from the repository root.

### 1. Optional: Recreate Trajectories

The repository includes the trajectory files used by Paper 7. To recreate them:

```bash
python paper7/collect_trajectories.py --policy random --seeds 3 --episodes 20
python paper7/collect_trajectories.py --policy greedy --seeds 3 --episodes 20
```

Outputs:

- `paper7/trajectories/random_seed*.npz`
- `paper7/trajectories/greedy_seed*.npz`

### 2. Train The Learned Transition Model

```bash
python paper7/train_learned_env.py --epochs 100 --policies random greedy
```

Outputs:

- `paper7/models/transition_model.pt`
- `paper7/models/training_history.json`

### 3. Estimate Causal Calibration

```bash
python paper7/causal_reward_calibration.py
```

Outputs are written under `paper7/results/`, including the causal dataset and calibration summaries.

### 4. Train Model-Based Policies

```bash
python paper7/train_model_based.py --mode learned --seed 0 --timesteps 100000
```

Outputs are written under `paper7/results/`.

### 5. Run Ablations

```bash
python paper7/ablation_experiments.py
python paper7/ablation_geofm.py
```

Existing ablation outputs are already included under:

- `paper7/results/revision/`
- `paper7/results/geofm/`
- `paper7/data/`

## Manuscript Rebuild

The CEUS manuscript source is in:

```bash
submission/ceus/01_main_document_anonymous/manuscript.tex
```

Build with:

```bash
cd submission/ceus/01_main_document_anonymous
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
```

The signed editable source is:

```bash
submission/ceus/06_latex_source_editable/manuscript_signed.tex
```

## Expected High-Level Results

The current manuscript reports:

- uncalibrated model-based slope reduction: `-0.976% +/- 0.129%`
- calibrated model-based slope reduction: `-1.102% +/- 0.100%`
- causal calibration gain over uncalibrated learned environment: `13.0%`, `p = 0.004`
- causally derived calibration factor: `alpha = 0.185`
- transition model validation cosine similarity: `0.9998`

See `submission/ceus/01_main_document_anonymous/manuscript.tex` and `paper7/results/` for detailed tables and result artifacts.
