"""
IMD District-wise Rainfall Normals loader.

Loads 'district wise rainfall normal.csv', fuzzy-matches district names to
the project's district_coords.csv, and provides a lookup of monthly normals.
"""

import logging
from pathlib import Path

import pandas as pd
from thefuzz import process as fuzz_process

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DISTRICT_COORDS_FILE

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMD_NORMALS_FILE = _PROJECT_ROOT / "district wise rainfall normal.csv"
MAPPING_CACHE = _PROJECT_ROOT / "data" / "processed" / "district_normal_mapping.csv"

# Manual overrides for known mismatches (IMD uppercase -> our Title Case)
_MANUAL_MAP = {
    "BANGALORE URBAN": "Bengaluru Urban",
    "BANGALORE RURAL": "Bengaluru Rural",
    "BANGALORE": "Bengaluru Urban",
    "BOMBAY": "Mumbai",
    "CALCUTTA": "Kolkata",
    "MADRAS": "Chennai",
    "PONDICHERRY": "Puducherry",
    "SHIMOGA": "Shivamogga",
    "BELGAUM": "Belagavi",
    "BELLARY": "Ballari",
    "BIJAPUR": "Vijayapura",
    "GULBARGA": "Kalaburagi",
    "HUBLI-DHARWAD": "Hubballi-Dharwad",
    "TUMKUR": "Tumakuru",
    "MYSORE": "Mysuru",
    "MANGALORE": "Mangaluru",
    "RAIBAREILLY": "Pratapgarh",
    "N & M ANDAMAN": "North And Middle Andaman",
    "NORTH & MIDDLE ANDAMAN": "North And Middle Andaman",
    "PASHCHIM CHAMPARAN": "West Singhbhum",
}

MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def _build_mapping(our_districts: list[str], imd_df: pd.DataFrame) -> dict:
    """Build IMD district name -> our district name mapping."""
    our_upper = {d.strip().upper(): d for d in our_districts}
    imd_names = imd_df["DISTRICT"].str.strip().str.upper().unique().tolist()

    mapping = {}

    for imd_name in imd_names:
        # Check manual overrides first
        if imd_name in _MANUAL_MAP:
            mapping[imd_name] = _MANUAL_MAP[imd_name]
            continue

        # Exact match on uppercase
        title_version = imd_name.strip().title()
        if title_version in [d for d in our_districts]:
            mapping[imd_name] = title_version
            continue

        if imd_name in our_upper:
            mapping[imd_name] = our_upper[imd_name]
            continue

        # Fuzzy match
        match = fuzz_process.extractOne(
            imd_name, list(our_upper.keys()), score_cutoff=80
        )
        if match:
            mapping[imd_name] = our_upper[match[0]]
        else:
            logger.debug(f"No match for IMD district: {imd_name}")

    return mapping


def load_district_normals() -> pd.DataFrame:
    """
    Load monthly rainfall normals for districts.

    Returns:
        DataFrame with columns [district, month, normal_mm]
        where district is in Title Case (matching our system),
        month is 1-12.
    """
    if not IMD_NORMALS_FILE.exists():
        logger.warning(f"IMD normals file not found: {IMD_NORMALS_FILE}")
        return pd.DataFrame(columns=["district", "month", "normal_mm"])

    imd_df = pd.read_csv(IMD_NORMALS_FILE)

    # Load our districts
    if DISTRICT_COORDS_FILE.exists():
        coords = pd.read_csv(DISTRICT_COORDS_FILE)
        our_districts = coords["district"].str.strip().str.title().unique().tolist()
    else:
        logger.warning("district_coords.csv not found, returning raw normals")
        our_districts = []

    # Build or load mapping
    if MAPPING_CACHE.exists():
        map_df = pd.read_csv(MAPPING_CACHE)
        mapping = dict(zip(map_df["imd_name"], map_df["our_name"]))
    else:
        mapping = _build_mapping(our_districts, imd_df)
        # Save mapping cache
        MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
        map_df = pd.DataFrame([
            {"imd_name": k, "our_name": v} for k, v in mapping.items()
        ])
        map_df.to_csv(MAPPING_CACHE, index=False)
        logger.info(f"Saved district normal mapping ({len(mapping)} matches) to {MAPPING_CACHE}")

    # Map IMD names to our names
    imd_df["_imd_upper"] = imd_df["DISTRICT"].str.strip().str.upper()
    imd_df["district"] = imd_df["_imd_upper"].map(mapping)

    # Drop unmatched
    matched = imd_df.dropna(subset=["district"])

    # Melt wide to long: one row per (district, month)
    rows = []
    for _, row in matched.iterrows():
        for i, col in enumerate(MONTH_COLS, start=1):
            val = row.get(col)
            if pd.notna(val):
                rows.append({
                    "district": row["district"],
                    "month": i,
                    "normal_mm": float(val),
                })

    result = pd.DataFrame(rows)

    # If multiple IMD districts map to same our-district, average them
    if not result.empty:
        result = result.groupby(["district", "month"], as_index=False)["normal_mm"].mean()

    # Convert monthly total to daily normal (divide by days in month)
    days_in_month = {1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30,
                     7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
    result["normal_mm_daily"] = result.apply(
        lambda r: r["normal_mm"] / days_in_month.get(r["month"], 30), axis=1
    )

    n_matched = result["district"].nunique()
    logger.info(f"Loaded IMD normals for {n_matched} districts")
    print(f"  IMD normals: {n_matched} districts matched, {len(result)} district-month entries")

    return result
