# Dongxing External Validation Plan

## Purpose

Use Dongxing District, Neijiang as an external-county validation case for Paper 7. The goal is to strengthen the CEUS revision with evidence that the Paper 7 geospatial preprocessing pipeline can ingest a second county-scale cadastral dataset, derive DEM-based slopes from a public source, and construct a county-scale mixed farmland-forest block action space outside Bishan.

## Data Audited

- Source root: the local Dongxing directory provided by the user (`D:\...\Neijiang Dongxing District ...`)
- Main parcel layer: recursively discovered `DLTB.shp`
- CRS: `EPSG:2359`
- Records: 134,369 parcels
- Required fields present: `DLBM`, `DLMC`, `TBMJ`
- Administrative/ownership fields present: `ZLDWDM`, `ZLDWMC`, `QSDWDM`, `QSDWMC`

## Public DEM Enrichment

The local `dem.tif` checked earlier does not cover Dongxing. A public AWS Terrain Tiles DEM was therefore downloaded for the Dongxing extent, following the same reproducible public-DEM logic used for Bishan-style terrain enrichment.

Generated DEM assets:

- Mosaic: `paper7/data/dongxing_dem_srtmgl1.tif`
- Tile cache: `paper7/data/dongxing_dem_tiles/`
- Download metadata: `paper7/results/dongxing_dem_download.json`
- Coverage audit: `paper7/results/dongxing_dem_coverage_srtmgl1.json`

DEM details:

- Provider: AWS Terrain Tiles
- Zoom: 10
- Tiles: 6 GeoTIFF tiles
- CRS: `EPSG:3857`
- Shape: 1536 x 1024
- Resolution: approximately 76.44 m
- Coverage: fully covers the Dongxing parcel layer bounds after CRS transformation

Slope enrichment output:

- Enriched parcels: `paper7/data/dongxing_DLTB_with_slope.gpkg`
- Audit: `paper7/results/dongxing_data_audit_slope.json`

Method note: parcel slopes are computed from DEM-derived slope degrees. The enrichment uses a batch rasterized zonal mean for parcels with DEM cell centers inside the parcel. For parcels too small to receive a center pixel at 76.44 m resolution, it falls back to the DEM slope value at a representative point inside the parcel. This avoids dropping small cadastral parcels while preserving mean-based estimates for larger parcels.

## Final Audit Findings

The Dongxing DLTB layer now supports both external block-input construction and a slope-swap environment:

- Farmland parcels: 45,735
- Forest parcels: 30,643
- Other parcels: 57,991
- Swappable farmland/forest parcels: 76,378
- ZLDWDM units processed: 429
- Units with at least one candidate mixed block: 424
- Candidate mixed farmland-forest blocks generated: 2,978
- DEM-derived `slope_mean` coverage: 100% for farmland, forest, and other mapped parcels
- Block-level slope completeness: 2,978/2,978 blocks have both `avg_farm_slope` and `avg_forest_slope`
- Readiness flags: `can_build_block_inputs = true`, `can_run_slope_swap_environment = true`, `can_build_external_environment = true`

Generated block outputs:

- `paper7/results/dongxing_blocks_slope/summary.json`
- `paper7/results/dongxing_blocks_slope/block_features.json`
- `paper7/results/dongxing_blocks_slope/block_compositions.json`
- `paper7/results/dongxing_blocks_slope/parcel_block_mapping.csv`

The earlier no-DEM block package is still available at `paper7/results/dongxing_blocks/` and should be treated as a pre-enrichment diagnostic output, not the final external-validation package.

## Static Opportunity Screen

A pre-RL top-100 screen was run on the final slope-enriched Dongxing block package:

- Script: `paper7/dongxing_policy_screen.py`
- Output: `paper7/results/dongxing_policy_screen_top100.json`
- Candidate blocks screened: 2,978
- Complete-slope candidate blocks: 2,978
- Random baseline: 1,000 random top-100 selections

Results:

| Screen | Mean slope gap | Positive-gap share | Exchange area (ha) | Opportunity score |
|---|---:|---:|---:|---:|
| Slope-gap ranking | 3.056336 | 1.000000 | 234.553981 | 591.253157 |
| Area-weighted ranking | 1.571083 | 1.000000 | 1,109.813117 | 1,442.349440 |
| Random baseline mean | -0.165858 | 0.443110 | 604.529579 | -188.570739 |

The deterministic screens exceeded all 1,000 random selections for mean slope gap, positive-gap share, and total opportunity score. With add-one correction, the one-sided empirical p-value is 0.000999 for each of these metrics under both deterministic screens.

Interpretation boundary: this screen shows that the Dongxing action space contains structured slope-reduction opportunities and that simple slope-aware rankings identify them much better than random block selection. It is not a dynamic RL policy evaluation, does not simulate parcel swaps through an episode, and does not test transition-model or policy transfer from Bishan to Dongxing.

## Dynamic Paired-Swap Baselines

A dynamic, non-RL paired-swap baseline was run on the final slope-enriched Dongxing block package. The experiment uses the same parcel-label paired-swap logic as the Bishan county environment: within a selected block, a high-slope available farmland parcel is converted to forest and a low-slope available forest parcel is converted to farmland, only if the parcel-level slope gap is positive. Parcel areas enter the county-level area-weighted farmland slope calculation, but the operator is a parcel-label exchange and does not require the paired parcels to have equal area.

- Script: `paper7/dongxing_dynamic_baselines.py`
- Output: `paper7/results/dongxing_dynamic_baselines.json`
- Steps: 100
- Maximum pairs per selected block: 5
- Random baseline: 200 seeded dynamic episodes

Results:

| Dynamic baseline | Completed pairs | Slope change (%) | Mean pair gap | Total pair gap | Unique blocks |
|---|---:|---:|---:|---:|---:|
| Dynamic slope-gap | 485 | -1.217577 | 9.055840 | 4,392.082609 | 100 |
| Dynamic area-weighted gap | 487 | -0.882685 | 7.538321 | 3,671.162145 | 99 |
| Random mean ± sd | 426.545 ± 13.649 | -0.357665 ± 0.086532 | 4.037575 ± 0.207955 | 1,722.523137 ± 109.533141 | 99.060 ± 0.983 |

For both deterministic dynamic baselines, the add-one one-sided empirical p-value against the 200 random episodes is 0.004975 for slope-change percentage, completed pair count, mean pair slope gap, and total pair slope gap.

Interpretation boundary: this experiment strengthens the external-county evidence because it moves beyond a static block screen and executes sequential Dongxing parcel swaps through a 100-step episode. It supports the feasibility of the Dongxing action space and shows that simple dynamic slope-aware block selection outperforms random dynamic block selection. It is still not a learned-environment experiment, not a transition-model validation, not a causal-calibration test, and not Bishan-to-Dongxing policy transfer.

## Revision Use

The current Dongxing evidence can support a manuscript subsection such as:

> External-county data and action-space validation

Bounded manuscript claim:

> To test transferability beyond the Bishan case, we applied the preprocessing and block-action-space construction pipeline to an independent county-scale cadastral dataset from Dongxing District, Neijiang. The pipeline mapped 134,369 parcels into farmland, forest, and other classes, enriched all parcels with public DEM-derived slope estimates, and constructed 2,978 mixed farmland-forest candidate blocks from 76,378 swappable parcels across 424 administrative/ownership units. All generated blocks contain continuous slope estimates for both farmland and forest components, enabling a full external slope-swap environment.

Important limitation to report:

> Dongxing slopes are derived from a public DEM mosaic at approximately 76.44 m resolution. Small cadastral parcels may be represented by a within-parcel representative-point DEM sample when no DEM cell center falls inside the parcel, so Dongxing is best used as an external pipeline, action-space, and non-RL dynamic feasibility validation case rather than as a replacement for field-measured parcel slopes or as evidence of learned-policy transfer.

Additional bounded claim now supported:

> A static top-100 opportunity screen further shows that Dongxing candidate blocks contain structured slope heterogeneity: slope-gap and area-weighted rankings selected only positive-gap blocks, while 1,000 random top-100 selections averaged a negative slope gap and 44.3% positive-gap blocks. This supports the action-space signal, not dynamic policy transfer.

Additional bounded claim supported by the dynamic run:

> Dynamic non-RL paired-swap baselines on Dongxing further show that the external action space remains actionable after sequential state updates. Over 100 decision steps, deterministic slope-aware selection reduced area-weighted farmland slope by 0.883% to 1.218%, compared with a 200-seed random dynamic baseline mean of 0.358%. This supports dynamic external-county feasibility, not learned-policy transfer.

## Commands

```powershell
$root = "D:\...\Neijiang Dongxing District ..."
python paper7\public_dem_download.py --provider aws-terrain --root $root --output paper7\data\dongxing_dem_srtmgl1.tif --metadata paper7\results\dongxing_dem_download.json --tile-dir paper7\data\dongxing_dem_tiles --zoom 10
python paper7\dongxing_slope_enrichment.py --dem paper7\data\dongxing_dem_srtmgl1.tif --root $root --coverage-json paper7\results\dongxing_dem_coverage_srtmgl1.json --output-gpkg paper7\data\dongxing_DLTB_with_slope.gpkg
python paper7\dongxing_data_audit.py --dltb paper7\data\dongxing_DLTB_with_slope.gpkg --output paper7\results\dongxing_data_audit_slope.json
python paper7\dongxing_block_inputs.py --dltb paper7\data\dongxing_DLTB_with_slope.gpkg --unit-field ZLDWDM --output-dir paper7\results\dongxing_blocks_slope
python paper7\dongxing_policy_screen.py --block-features paper7\results\dongxing_blocks_slope\block_features.json --top-k 100 --random-seeds 1000 --output paper7\results\dongxing_policy_screen_top100.json
python paper7\dongxing_dynamic_baselines.py --dltb paper7\data\dongxing_DLTB_with_slope.gpkg --block-dir paper7\results\dongxing_blocks_slope --max-steps 100 --swaps-per-step 5 --random-seeds 200 --output paper7\results\dongxing_dynamic_baselines.json
```

## Next Experiments

1. Build a generic county environment that accepts external DLTB and block package paths instead of hard-coded Bishan paths.
2. Add Dongxing connectivity and contiguity metrics so that dynamic external experiments use the full Bishan-style reward, not slope-only paired swaps.
3. Collect Dongxing trajectories, train a Dongxing transition model, and evaluate model-based policies on the Dongxing real environment.
4. If enough trajectories are available, compare Dongxing-from-scratch with Bishan-pretrained transition-model fine-tuning.
5. Put the 185 MB enriched Dongxing GeoPackage and DEM assets into a GitHub Release or external data archive rather than committing them directly to Git.
