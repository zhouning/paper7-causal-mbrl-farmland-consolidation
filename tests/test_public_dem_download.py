from paper7.public_dem_download import (
    aws_terrain_tile_url,
    build_opentopography_url,
    padded_bounds,
    tiles_for_bounds,
)


def test_padded_bounds_expands_extent():
    bounds = (104.98, 29.44, 105.42, 29.85)

    padded = padded_bounds(bounds, pad_degrees=0.02)

    assert padded == (104.96, 29.42, 105.44, 29.87)


def test_build_opentopography_url_contains_dataset_and_bounds():
    url = build_opentopography_url(
        bounds=(104.96, 29.42, 105.44, 29.87),
        dem_type="SRTMGL1",
    )

    assert url.startswith("https://portal.opentopography.org/API/globaldem?")
    assert "demtype=SRTMGL1" in url
    assert "south=29.42" in url
    assert "north=29.87" in url
    assert "west=104.96" in url
    assert "east=105.44" in url
    assert "outputFormat=GTiff" in url


def test_tiles_for_bounds_returns_covering_xyz_tiles():
    tiles = tiles_for_bounds((104.95933573, 29.42342038, 105.44162589, 29.87057429), zoom=10)

    assert tiles[0] == (10, 810, 422)
    assert tiles[-1] == (10, 811, 424)
    assert len(tiles) == 6


def test_aws_terrain_tile_url_uses_public_geotiff_endpoint():
    url = aws_terrain_tile_url((10, 811, 423))

    assert url == "https://s3.amazonaws.com/elevation-tiles-prod/geotiff/10/811/423.tif"
