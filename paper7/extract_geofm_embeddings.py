"""
Paper 7: Extract GeoFM (AlphaEarth) 64-dim embeddings for Bishan county blocks.

For each of the ~2600 blocks, computes zonal mean of AlphaEarth embeddings
by sampling the block centroid location.

Output: paper7/data/block_geofm_embeddings.npy — shape (n_blocks, 64)

Usage:
    python paper7/extract_geofm_embeddings.py
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER7_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_block_embeddings(year=2020, scale=100):
    """Extract 64-dim AlphaEarth embeddings for all Bishan county blocks.

    Strategy: extract a grid of embeddings covering the county bbox,
    then compute per-block zonal mean by mapping block centroids to grid cells.
    """
    import ee
    ee.Initialize(project='gen-lang-client-0977577668')

    from county_env import CountyLevelEnv

    print("Loading county environment for block geometries...")
    env = CountyLevelEnv(total_budget=500, swaps_per_step=5)
    n_blocks = env.n_blocks
    print(f"  {n_blocks} blocks loaded")

    # Load parcel geometries directly from the GeoPackage
    import geopandas as gpd
    DLTB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'dem_slope_analysis', 'output', 'DLTB_with_slope.gpkg')
    TOWNSHIP_CODES = ['500227001', '500227002', '500227100', '500227101',
                      '500227102', '500227103', '500227104', '500227105',
                      '500227106', '500227107', '500227108', '500227109', '500227200']
    where_clause = " OR ".join([f"QSDWDM LIKE '{code}%'" for code in TOWNSHIP_CODES])
    print("  Loading parcel geometries from GeoPackage...")
    gdf = gpd.read_file(DLTB_PATH, where=where_clause)

    # Filter to swappable (farmland=011-013, forest=031-033)
    def _classify(dlbm):
        if isinstance(dlbm, str) and len(dlbm) >= 3:
            prefix = dlbm[:3]
            if prefix in ('011', '012', '013'): return 1
            if prefix in ('031', '032', '033'): return 2
        return 0
    gdf['type_code'] = gdf['DLBM'].apply(_classify)
    gdf_swap = gdf[gdf['type_code'].isin([1, 2])].copy()
    gdf_swap = gdf_swap.to_crs('EPSG:32648')
    print(f"  {len(gdf_swap)} swappable parcels")

    # Compute block centroids from parcel geometries
    block_centroids = []
    for b in range(n_blocks):
        parcel_indices = env.block_parcels[b]
        if len(parcel_indices) > 0:
            centroids = gdf_swap.iloc[parcel_indices].geometry.centroid
            cx = centroids.x.mean()
            cy = centroids.y.mean()
            block_centroids.append((cx, cy))
        else:
            block_centroids.append((0, 0))

    # Convert projected coords (EPSG:32648 UTM 48N) to WGS84
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:32648", "EPSG:4326", always_xy=True)
    wgs84_centroids = []
    for cx, cy in block_centroids:
        lon, lat = transformer.transform(cx, cy)
        wgs84_centroids.append((lon, lat))

    print(f"  Block centroids computed (WGS84)")
    lons = [c[0] for c in wgs84_centroids]
    lats = [c[1] for c in wgs84_centroids]
    print(f"  Lon range: [{min(lons):.4f}, {max(lons):.4f}]")
    print(f"  Lat range: [{min(lats):.4f}, {max(lats):.4f}]")

    # Extract AlphaEarth embeddings by sampling at block centroid locations
    print(f"\n  Extracting AlphaEarth embeddings at {n_blocks} centroids, year={year}")

    col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
    img = col.filterDate(f'{year}-01-01', f'{year+1}-01-01').mosaic()
    bands = [f'A{i:02d}' for i in range(64)]
    img = img.select(bands)

    # Sample in batches of 500 (GEE limit ~5000 features per sampleRegions)
    BATCH_SIZE = 500
    block_embeddings = np.zeros((n_blocks, 64), dtype=np.float32)
    total_sampled = 0

    for batch_start in range(0, n_blocks, BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, n_blocks)
        features = []
        for b in range(batch_start, batch_end):
            lon, lat = wgs84_centroids[b]
            pt = ee.Geometry.Point([lon, lat])
            features.append(ee.Feature(pt, {'block_id': b}))

        fc = ee.FeatureCollection(features)
        t0 = time.time()
        sampled = img.sampleRegions(collection=fc, scale=10, geometries=False)
        result = sampled.getInfo()
        elapsed = time.time() - t0

        for feat in result['features']:
            props = feat['properties']
            b = props['block_id']
            for i in range(64):
                block_embeddings[b, i] = props.get(f'A{i:02d}', 0.0)

        n_got = len(result['features'])
        total_sampled += n_got
        print(f"    Batch {batch_start}-{batch_end}: {n_got} points, {elapsed:.1f}s")

    print(f"  Total sampled: {total_sampled}/{n_blocks}")

    # L2 normalize
    norms = np.linalg.norm(block_embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    block_embeddings = block_embeddings / norms

    # Verify
    norms_check = np.linalg.norm(block_embeddings, axis=1)
    print(f"\n  Block embeddings: shape={block_embeddings.shape}")
    print(f"  L2 norms: mean={norms_check.mean():.4f}, min={norms_check.min():.4f}")
    print(f"  Non-zero blocks: {(norms_check > 0.1).sum()}/{n_blocks}")

    return block_embeddings, None, None


def main():
    print("=" * 60)
    print("Paper 7: Extract GeoFM Embeddings for Bishan Blocks")
    print("=" * 60)

    data_dir = os.path.join(PAPER7_DIR, 'data')
    os.makedirs(data_dir, exist_ok=True)

    embeddings, _, _ = extract_block_embeddings(year=2020)

    # Save block embeddings
    emb_path = os.path.join(data_dir, 'block_geofm_embeddings.npy')
    np.save(emb_path, embeddings)
    print(f"\nBlock embeddings saved to {emb_path} ({embeddings.nbytes/1024:.0f} KB)")

    # Save metadata
    meta = {
        'n_blocks': int(embeddings.shape[0]),
        'embedding_dim': 64,
        'year': 2020,
        'source': 'AlphaEarth GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL',
        'method': 'centroid_point_sampling',
    }
    with open(os.path.join(data_dir, 'geofm_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print("GeoFM extraction complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
