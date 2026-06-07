# Large Reproducibility Assets

The files below are required for full Paper 7 reproduction but are too large for normal Git history. They are distributed as GitHub Release assets for the repository release `paper7-repro-assets-v1`.

Download each asset and place it at the target path shown here.

| Asset name | Target path | Size | SHA256 |
|---|---|---:|---|
| `DLTB_with_slope.gpkg` | `dem_slope_analysis/output/DLTB_with_slope.gpkg` | 153.10 MiB | `7ECCDFB11A98F4E31145E93FD270FAEA2774BED6FFAEF99C66009C4EFF9FB677` |
| `greedy_seed0.npz` | `paper7/trajectories/greedy_seed0.npz` | 264.52 MiB | `8A0E574FBF41C6E0F21371FE053814C737EDF67C072BE4D6201D6E54A2325317` |
| `greedy_seed1.npz` | `paper7/trajectories/greedy_seed1.npz` | 264.52 MiB | `8A0E574FBF41C6E0F21371FE053814C737EDF67C072BE4D6201D6E54A2325317` |
| `greedy_seed2.npz` | `paper7/trajectories/greedy_seed2.npz` | 264.52 MiB | `8A0E574FBF41C6E0F21371FE053814C737EDF67C072BE4D6201D6E54A2325317` |
| `random_seed0.npz` | `paper7/trajectories/random_seed0.npz` | 264.62 MiB | `3F5EBCE8377C9038C455A7D83E936F7A0752B0B5F3CE2CC5EDD1B6525FE51BA1` |
| `random_seed1.npz` | `paper7/trajectories/random_seed1.npz` | 264.62 MiB | `F47DEEDE36068D8B3ED92F49B561C4579AEBEA5A79366A170E7BB2FEEEB90334` |
| `random_seed2.npz` | `paper7/trajectories/random_seed2.npz` | 264.59 MiB | `8753E76223B7E795039A80B63D1C4308B9469B94846A6715D88F8D3F30B13EB5` |

## Download With GitHub CLI

From the repository root:

```bash
gh release download paper7-repro-assets-v1 --repo zhouning/paper7-causal-mbrl-farmland-consolidation --dir _release_assets
```

Then move the files:

```bash
mkdir -p paper7/trajectories
mkdir -p dem_slope_analysis/output
mv _release_assets/greedy_seed*.npz paper7/trajectories/
mv _release_assets/random_seed*.npz paper7/trajectories/
mv _release_assets/DLTB_with_slope.gpkg dem_slope_analysis/output/
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force paper7\trajectories, dem_slope_analysis\output
Move-Item _release_assets\greedy_seed*.npz paper7\trajectories\
Move-Item _release_assets\random_seed*.npz paper7\trajectories\
Move-Item _release_assets\DLTB_with_slope.gpkg dem_slope_analysis\output\
```

## Verify Checksums

On PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 paper7\trajectories\*.npz, dem_slope_analysis\output\DLTB_with_slope.gpkg
```

Compare the hashes against the table above before rerunning experiments.
