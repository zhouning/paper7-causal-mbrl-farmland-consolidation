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

The Dongxing external-county checks additionally depend on Dongxing cadastral data and public DEM-derived slope assets. These files support external data/action-space checks, dynamic non-RL feasibility, local full-reward baselines, local learned-policy actionability, and local model-based planning evidence; they are still not evidence of direct Bishan-to-Dongxing learned-policy transfer.

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

The stored CEUS revision training history currently records 50 epochs. The command above is the full training recipe; the audit command below validates the stored artifact chain and does not imply fresh retraining.

### 3. Estimate Observational Reward Calibration

```bash
python paper7/causal_reward_calibration.py
```

Outputs are written under `paper7/results/`, including the observational calibration dataset and calibration summaries.

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

### 6. Audit The Stored Evidence Chain

The revision also includes a policy-induced learned-vs-real diagnostic. The stored expanded diagnostic uses 15 calibrated policy seeds, lets each policy select actions from the learned environment, replays the same actions in the real environment, and reports state, reward, mask, support-distance, and final real-outcome metrics:

```bash
python paper7/policy_induced_diagnostics.py --output paper7/results/revision/policy_induced_diagnostics.json
```

The CEUS revision includes executable audits that verify the recorded data-to-result chain without retraining all expensive policies:

```bash
python paper7/planning_significance_audit.py
python paper7/reward_weight_sensitivity.py
```

These commands refresh paired seed statistics and fixed-policy reward-component replay artifacts, including the canonical reward specification. They verify stored outputs and replay recorded action sequences; they do not perform full fresh policy retraining.

```bash
python paper7/end_to_end_validation.py --out paper7/results/revision/end_to_end_validation.json
```

Expected high-level audit status:

- `overall_status`: `supported_with_bounded_external_scope`
- Bishan learned-policy chain: 15 paired calibrated/uncalibrated real-environment evaluation seeds
- Reward-scaling grid: 40 stored runs over 8 alpha values
- Policy-induced learned-vs-real diagnostics: 15 calibrated policy seeds, with final real-environment outcomes and support-distance checks
- Dongxing: `supported_as_external_full_reward_counterpart`, still not direct learned-policy transfer

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
- paired calibrated-minus-uncalibrated slope delta: `-0.127` percentage points; exact sign-flip one-sided `p = 0.011963`
- pre-specified calibration factor: `alpha = 0.185`
- transition model validation cosine similarity: `0.9998`
- 100-step rollout diagnostics: action-mask agreement `0.997`, reward MAE `0.234`
- policy-induced diagnostics: action-mask agreement `0.998`, support distance `0.013`, final real slope change `-1.106%`
- Dongxing external counterpart: 2,978 mixed blocks from 134,369 parcels, dynamic non-RL slope reduction `0.883--1.218%`, plus local full-reward baselines and local model-based planning evidence

See `submission/ceus/01_main_document_anonymous/manuscript.tex` and `paper7/results/` for detailed tables and result artifacts.
