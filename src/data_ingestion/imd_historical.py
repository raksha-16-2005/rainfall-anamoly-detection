"""
IMD Historical Rainfall (1901-2015) loader.

Loads subdivision-level historical rainfall, maps subdivisions to districts,
and computes long-term statistical features per subdivision+month.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DISTRICT_COORDS_FILE

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
IMD_HISTORICAL_FILE = _PROJECT_ROOT / "rainfall in india 1901-2015.csv"
SUBDIV_MAPPING_CACHE = _PROJECT_ROOT / "data" / "processed" / "district_subdivision_mapping.csv"

MONTH_COLS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]

# State -> IMD Subdivision mapping
# For states that span multiple subdivisions, we use the dominant/largest one
_STATE_TO_SUBDIVISION = {
    "Kerala": "KERALA",
    "Tamil Nadu": "TAMIL NADU",
    "Karnataka": "SOUTH INTERIOR KARNATAKA",  # default for most KA districts
    "Andhra Pradesh": "COASTAL ANDHRA PRADESH",
    "Telangana": "TELANGANA",
    "Maharashtra": "MADHYA MAHARASHTRA",  # default
    "Goa": "KONKAN & GOA",
    "Gujarat": "GUJARAT REGION",
    "Rajasthan": "EAST RAJASTHAN",  # default
    "Madhya Pradesh": "EAST MADHYA PRADESH",  # default
    "Chhattisgarh": "CHHATTISGARH",
    "Odisha": "ORISSA",
    "West Bengal": "GANGETIC WEST BENGAL",  # default
    "Bihar": "BIHAR",
    "Jharkhand": "JHARKHAND",
    "Uttar Pradesh": "EAST UTTAR PRADESH",  # default
    "Uttarakhand": "UTTARAKHAND",
    "Himachal Pradesh": "HIMACHAL PRADESH",
    "Punjab": "PUNJAB",
    "Haryana": "HARYANA DELHI & CHANDIGARH",
    "Delhi": "HARYANA DELHI & CHANDIGARH",
    "Jammu And Kashmir": "JAMMU & KASHMIR",
    "Jammu & Kashmir": "JAMMU & KASHMIR",
    "Ladakh": "JAMMU & KASHMIR",
    "Assam": "ASSAM & MEGHALAYA",
    "Meghalaya": "ASSAM & MEGHALAYA",
    "Nagaland": "NAGA MANI MIZO TRIPURA",
    "Manipur": "NAGA MANI MIZO TRIPURA",
    "Mizoram": "NAGA MANI MIZO TRIPURA",
    "Tripura": "NAGA MANI MIZO TRIPURA",
    "Arunachal Pradesh": "ARUNACHAL PRADESH",
    "Sikkim": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "Andaman And Nicobar Islands": "ANDAMAN & NICOBAR ISLANDS",
    "Andaman And Nicobar": "ANDAMAN & NICOBAR ISLANDS",
    "Lakshadweep": "LAKSHADWEEP",
    "Puducherry": "TAMIL NADU",
    "Chandigarh": "HARYANA DELHI & CHANDIGARH",
}

# District-level overrides for states with multiple subdivisions
_DISTRICT_SUBDIVISION_OVERRIDES = {
    # Karnataka
    "Mangaluru": "COASTAL KARNATAKA", "Udupi": "COASTAL KARNATAKA",
    "Dakshina Kannada": "COASTAL KARNATAKA", "Uttara Kannada": "COASTAL KARNATAKA",
    "Kalaburagi": "NORTH INTERIOR KARNATAKA", "Bidar": "NORTH INTERIOR KARNATAKA",
    "Raichur": "NORTH INTERIOR KARNATAKA", "Yadgir": "NORTH INTERIOR KARNATAKA",
    "Koppal": "NORTH INTERIOR KARNATAKA", "Ballari": "NORTH INTERIOR KARNATAKA",
    "Vijayapura": "NORTH INTERIOR KARNATAKA", "Bagalkot": "NORTH INTERIOR KARNATAKA",
    "Gadag": "NORTH INTERIOR KARNATAKA", "Haveri": "NORTH INTERIOR KARNATAKA",
    "Dharwad": "NORTH INTERIOR KARNATAKA", "Belagavi": "NORTH INTERIOR KARNATAKA",
    "Hubballi-Dharwad": "NORTH INTERIOR KARNATAKA",
    # Maharashtra
    "Mumbai": "KONKAN & GOA", "Thane": "KONKAN & GOA", "Raigad": "KONKAN & GOA",
    "Palghar": "KONKAN & GOA", "Ratnagiri": "KONKAN & GOA", "Sindhudurg": "KONKAN & GOA",
    "Nagpur": "VIDARBHA", "Wardha": "VIDARBHA", "Chandrapur": "VIDARBHA",
    "Gadchiroli": "VIDARBHA", "Gondia": "VIDARBHA", "Bhandara": "VIDARBHA",
    "Amravati": "VIDARBHA", "Akola": "VIDARBHA", "Washim": "VIDARBHA",
    "Buldhana": "VIDARBHA", "Yavatmal": "VIDARBHA",
    "Aurangabad": "MATATHWADA", "Jalna": "MATATHWADA", "Parbhani": "MATATHWADA",
    "Hingoli": "MATATHWADA", "Nanded": "MATATHWADA", "Latur": "MATATHWADA",
    "Osmanabad": "MATATHWADA", "Beed": "MATATHWADA",
    # Rajasthan
    "Jodhpur": "WEST RAJASTHAN", "Bikaner": "WEST RAJASTHAN",
    "Jaisalmer": "WEST RAJASTHAN", "Barmer": "WEST RAJASTHAN",
    "Pali": "WEST RAJASTHAN", "Nagaur": "WEST RAJASTHAN",
    "Sirohi": "WEST RAJASTHAN",
    # Andhra Pradesh
    "Kurnool": "RAYALSEEMA", "Anantapur": "RAYALSEEMA",
    "Kadapa": "RAYALSEEMA", "Chittoor": "RAYALSEEMA", "Tirupati": "RAYALSEEMA",
    # Uttar Pradesh
    "Agra": "WEST UTTAR PRADESH", "Meerut": "WEST UTTAR PRADESH",
    "Aligarh": "WEST UTTAR PRADESH", "Mathura": "WEST UTTAR PRADESH",
    "Moradabad": "WEST UTTAR PRADESH", "Bareilly": "WEST UTTAR PRADESH",
    "Saharanpur": "WEST UTTAR PRADESH", "Muzaffarnagar": "WEST UTTAR PRADESH",
    "Ghaziabad": "WEST UTTAR PRADESH", "Firozabad": "WEST UTTAR PRADESH",
    # MP
    "Indore": "WEST MADHYA PRADESH", "Ujjain": "WEST MADHYA PRADESH",
    "Ratlam": "WEST MADHYA PRADESH", "Mandsaur": "WEST MADHYA PRADESH",
    "Neemuch": "WEST MADHYA PRADESH", "Dewas": "WEST MADHYA PRADESH",
    "Gwalior": "WEST MADHYA PRADESH", "Jhansi": "WEST UTTAR PRADESH",
    # West Bengal
    "Darjeeling": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "Jalpaiguri": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    "Cooch Behar": "SUB HIMALAYAN WEST BENGAL & SIKKIM",
    # Gujarat
    "Kutch": "SAURASHTRA & KUTCH", "Bhuj": "SAURASHTRA & KUTCH",
    "Rajkot": "SAURASHTRA & KUTCH", "Jamnagar": "SAURASHTRA & KUTCH",
    "Junagadh": "SAURASHTRA & KUTCH", "Bhavnagar": "SAURASHTRA & KUTCH",
    "Amreli": "SAURASHTRA & KUTCH",
}


def _build_district_subdivision_map() -> pd.DataFrame:
    """Map each district to its IMD subdivision."""
    if not DISTRICT_COORDS_FILE.exists():
        return pd.DataFrame(columns=["district", "subdivision"])

    coords = pd.read_csv(DISTRICT_COORDS_FILE)
    rows = []
    for _, row in coords.iterrows():
        district = str(row["district"]).strip().title()
        state = str(row.get("state", "")).strip().title()

        # Check district-level override first
        if district in _DISTRICT_SUBDIVISION_OVERRIDES:
            subdiv = _DISTRICT_SUBDIVISION_OVERRIDES[district]
        elif state in _STATE_TO_SUBDIVISION:
            subdiv = _STATE_TO_SUBDIVISION[state]
        else:
            subdiv = None
            logger.debug(f"No subdivision mapping for {district} ({state})")

        if subdiv:
            rows.append({"district": district, "subdivision": subdiv})

    result = pd.DataFrame(rows)
    return result


def load_subdivision_mapping() -> pd.DataFrame:
    """Load or build district -> subdivision mapping."""
    if SUBDIV_MAPPING_CACHE.exists():
        return pd.read_csv(SUBDIV_MAPPING_CACHE)

    mapping = _build_district_subdivision_map()
    SUBDIV_MAPPING_CACHE.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(SUBDIV_MAPPING_CACHE, index=False)
    print(f"  Saved subdivision mapping ({len(mapping)} districts) to {SUBDIV_MAPPING_CACHE}")
    return mapping


def compute_historical_features() -> pd.DataFrame:
    """
    Compute per-subdivision, per-month historical statistics from 1901-2015 data.

    Returns:
        DataFrame with columns:
        [subdivision, month, hist_mean_mm, hist_std_mm, hist_p10, hist_p90, hist_trend_slope]
    """
    if not IMD_HISTORICAL_FILE.exists():
        logger.warning(f"Historical file not found: {IMD_HISTORICAL_FILE}")
        return pd.DataFrame()

    hist_df = pd.read_csv(IMD_HISTORICAL_FILE)

    rows = []
    for subdiv in hist_df["SUBDIVISION"].unique():
        sub_data = hist_df[hist_df["SUBDIVISION"] == subdiv]
        for i, col in enumerate(MONTH_COLS, start=1):
            values = sub_data[col].dropna().values
            if len(values) < 10:
                continue
            years = sub_data.loc[sub_data[col].notna(), "YEAR"].values

            # Trend: mm per decade
            if len(years) >= 20:
                coeffs = np.polyfit(years, values, 1)
                trend_slope = coeffs[0] * 10  # per decade
            else:
                trend_slope = 0.0

            rows.append({
                "subdivision": subdiv,
                "month": i,
                "hist_mean_mm": float(np.mean(values)),
                "hist_std_mm": float(np.std(values)),
                "hist_p10": float(np.percentile(values, 10)),
                "hist_p90": float(np.percentile(values, 90)),
                "hist_trend_slope": float(trend_slope),
            })

    result = pd.DataFrame(rows)
    print(f"  Historical features: {len(result)} subdivision-month entries "
          f"from {hist_df['SUBDIVISION'].nunique()} subdivisions")
    return result


def get_historical_annual_series() -> pd.DataFrame:
    """
    Return annual rainfall series for each subdivision (for dashboard charts).

    Returns:
        DataFrame with [subdivision, year, annual_mm]
    """
    if not IMD_HISTORICAL_FILE.exists():
        return pd.DataFrame(columns=["subdivision", "year", "annual_mm"])

    hist_df = pd.read_csv(IMD_HISTORICAL_FILE)
    result = hist_df[["SUBDIVISION", "YEAR", "ANNUAL"]].copy()
    result.columns = ["subdivision", "year", "annual_mm"]
    result = result.dropna(subset=["annual_mm"])
    return result
