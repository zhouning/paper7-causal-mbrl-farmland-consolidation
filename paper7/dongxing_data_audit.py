"""Audit Dongxing District parcel data for Paper 7 external validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd


FARMLAND_CODES = {"011", "012", "013", "0101", "0102", "0103"}
FOREST_CODES = {"031", "032", "033", "0301", "0302", "0303", "0304", "0305", "0307"}
FARMLAND_NAMES = ("水田", "水浇地", "旱地", "耕地")
FOREST_NAMES = ("有林地", "乔木林地", "灌木林地", "其他林地", "林地")
REQUIRED_FIELDS = ("DLBM", "DLMC", "TBMJ")
ADMIN_FIELDS = ("ZLDWDM", "ZLDWMC", "QSDWDM", "QSDWMC")
SLOPE_CANDIDATE_FIELDS = ("slope", "SLOPE", "slope_deg", "slope_mean", "GDPDJ")


def classify_land_use(code: object, name: object) -> str:
    """Map Chinese land-use code/name fields to Paper 7's coarse classes."""
    code_text = "" if pd.isna(code) else str(code).strip()
    name_text = "" if pd.isna(name) else str(name).strip()

    if code_text in FARMLAND_CODES or code_text.startswith("01"):
        return "farmland"
    if code_text in FOREST_CODES or code_text.startswith("03"):
        return "forest"
    if any(token in name_text for token in FARMLAND_NAMES):
        return "farmland"
    if any(token in name_text for token in FOREST_NAMES):
        return "forest"
    return "other"


def audit_geodataframe(frame: gpd.GeoDataFrame, source_path: Path) -> dict[str, Any]:
    """Return a JSON-serializable audit of Dongxing DLTB parcel data."""
    missing_required = [field for field in REQUIRED_FIELDS if field not in frame.columns]
    has_required = not missing_required

    area = _area_series(frame)
    classes = frame.apply(lambda row: classify_land_use(row.get("DLBM"), row.get("DLMC")), axis=1)
    slope_field = _first_existing_field(frame, SLOPE_CANDIDATE_FIELDS)

    class_stats = {}
    for label in ("farmland", "forest", "other"):
        mask = classes == label
        class_stats[label] = {
            "count": int(mask.sum()),
            "area_m2": round(float(area[mask].sum()), 2),
            "area_share": round(float(area[mask].sum() / area.sum()), 6) if float(area.sum()) > 0 else 0.0,
        }

    administrative_units = {}
    for field in ADMIN_FIELDS:
        if field in frame.columns:
            non_null = frame[field].dropna()
            administrative_units[field] = {
                "non_null_count": int(non_null.shape[0]),
                "unique_count": int(non_null.astype(str).str.strip().replace("", pd.NA).dropna().nunique()),
            }

    slope = _slope_stats(frame, slope_field)
    slope["by_land_use"] = _slope_by_land_use(frame, classes, slope_field)

    audit = {
        "source": {
            "path": str(source_path),
            "record_count": int(frame.shape[0]),
            "crs": str(frame.crs) if frame.crs is not None else None,
            "bounds": [round(float(value), 3) for value in frame.total_bounds.tolist()]
            if frame.shape[0]
            else None,
        },
        "schema": {
            "columns": [str(column) for column in frame.columns],
            "required_fields": list(REQUIRED_FIELDS),
            "missing_required_fields": missing_required,
            "has_required_fields": has_required,
            "area_field": _area_field(frame),
        },
        "land_use": class_stats,
        "slope": slope,
        "administrative_units": administrative_units,
        "readiness": _readiness(has_required, class_stats, slope, administrative_units),
    }
    return audit


def audit_shapefile(path: Path) -> dict[str, Any]:
    frame = gpd.read_file(path, engine="pyogrio")
    return audit_geodataframe(frame, source_path=path)


def find_dltb_shapefile(root: Path) -> Path:
    matches = sorted(root.rglob("DLTB.shp"))
    if not matches:
        raise FileNotFoundError(f"No DLTB.shp found under {root}")
    return matches[0]


def write_audit(audit: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


def _area_field(frame: gpd.GeoDataFrame) -> str:
    if "TBMJ" in frame.columns:
        return "TBMJ"
    if "SHAPE_Area" in frame.columns:
        return "SHAPE_Area"
    return "geometry.area"


def _area_series(frame: gpd.GeoDataFrame) -> pd.Series:
    field = _area_field(frame)
    if field in frame.columns:
        return pd.to_numeric(frame[field], errors="coerce").fillna(0.0).astype(float)
    return frame.geometry.area.fillna(0.0).astype(float)


def _first_existing_field(frame: gpd.GeoDataFrame, candidates: tuple[str, ...]) -> str | None:
    for field in candidates:
        if field in frame.columns:
            return field
    return None


def _slope_stats(frame: gpd.GeoDataFrame, field: str | None) -> dict[str, Any]:
    if field is None:
        return {
            "field": None,
            "mode": "missing",
            "non_null_count": 0,
            "unique_values": [],
            "numeric_summary": None,
        }

    raw = frame[field].dropna()
    text = raw.astype(str).str.strip()
    text = text[text != ""]
    numeric = pd.to_numeric(text, errors="coerce").dropna()
    value_counts = text.value_counts().head(20)

    return {
        "field": field,
        "mode": "continuous_or_grade" if field != "GDPDJ" else "slope_grade_proxy",
        "non_null_count": int(text.shape[0]),
        "unique_count": int(text.nunique()),
        "unique_values": [{"value": str(index), "count": int(value)} for index, value in value_counts.items()],
        "numeric_summary": {
            "min": round(float(numeric.min()), 6),
            "mean": round(float(numeric.mean()), 6),
            "max": round(float(numeric.max()), 6),
        }
        if not numeric.empty
        else None,
    }


def _slope_by_land_use(
    frame: gpd.GeoDataFrame,
    classes: pd.Series,
    field: str | None,
) -> dict[str, dict[str, Any]]:
    if field is None:
        return {
            label: {"non_null_count": 0, "coverage": 0.0}
            for label in ("farmland", "forest", "other")
        }

    stats = {}
    values = frame[field]
    valid = values.notna() & (values.astype(str).str.strip() != "")
    for label in ("farmland", "forest", "other"):
        mask = classes == label
        total = int(mask.sum())
        non_null = int((mask & valid).sum())
        stats[label] = {
            "count": total,
            "non_null_count": non_null,
            "coverage": round(non_null / total, 6) if total else 0.0,
        }
    return stats


def _readiness(
    has_required: bool,
    class_stats: dict[str, dict[str, float | int]],
    slope: dict[str, Any],
    administrative_units: dict[str, dict[str, int]],
) -> dict[str, Any]:
    has_farmland = int(class_stats["farmland"]["count"]) > 0
    has_forest = int(class_stats["forest"]["count"]) > 0
    has_admin_units = any(stats["non_null_count"] > 0 for stats in administrative_units.values())
    slope_field = slope["field"]
    has_slope = slope_field is not None
    has_farmland_slopes = slope["by_land_use"]["farmland"]["non_null_count"] > 0 if has_slope else False
    has_forest_slopes = slope["by_land_use"]["forest"]["non_null_count"] > 0 if has_slope else False

    block_input_blockers = []
    if not has_required:
        block_input_blockers.append("missing required DLBM/DLMC/TBMJ fields")
    if not has_farmland:
        block_input_blockers.append("no mapped farmland parcels")
    if not has_forest:
        block_input_blockers.append("no mapped forest parcels")
    if not has_admin_units:
        block_input_blockers.append("no usable administrative or ownership unit field")

    environment_blockers = list(block_input_blockers)
    if not has_slope:
        environment_blockers.append("no slope or slope-grade field")
    if has_slope and not has_farmland_slopes:
        environment_blockers.append("farmland parcels lack slope values")
    if has_slope and not has_forest_slopes:
        environment_blockers.append("forest parcels lack slope values")

    return {
        "can_build_block_inputs": not block_input_blockers,
        "can_run_slope_swap_environment": not environment_blockers,
        "can_build_external_environment": not environment_blockers,
        "slope_mode": "slope_grade_proxy" if slope_field == "GDPDJ" else ("continuous_slope" if has_slope else "missing"),
        "block_input_blockers": block_input_blockers,
        "blockers": environment_blockers,
        "notes": [
            "GDPDJ is treated as a slope-grade proxy unless a DEM-derived continuous slope field is supplied.",
            "External validation should report Dongxing results separately if slope uses grade rather than continuous percent.",
            "A paired slope-swap environment requires slope values for both farmland and forest parcels.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Root directory containing Dongxing shapefiles.")
    parser.add_argument("--dltb", type=Path, help="Path to DLTB.shp. Overrides --root search.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper7/results/dongxing_data_audit.json"),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dltb is not None:
        dltb_path = args.dltb
    elif args.root is not None:
        dltb_path = find_dltb_shapefile(args.root)
    else:
        raise SystemExit("Provide --dltb or --root.")

    audit = audit_shapefile(dltb_path)
    write_audit(audit, args.output)
    print(json.dumps(audit["readiness"], ensure_ascii=False, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
