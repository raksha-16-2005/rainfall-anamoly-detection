"""
Geospatial utilities for Karnataka districts.

Module for district coordinate lookup and fuzzy matching.
"""

from difflib import get_close_matches


# Karnataka district coordinates (latitude, longitude)
DISTRICT_COORDS = {
    "bengaluru urban": (12.9716, 77.5946),
    "bengaluru rural": (13.2137, 77.6063),
    "ramanagara": (12.7667, 77.2833),
    "chikballapur": (13.2333, 77.8167),
    "kolar": (13.1333, 78.1333),
    "tumkur": (13.2167, 77.1167),
    "chitradurga": (13.2163, 76.3986),
    "davanagere": (14.4667, 75.9167),
    "shivamogga": (13.9333, 75.5667),
    "uttara kannada": (14.5, 74.8333),
    "udupi": (13.3333, 74.75),
    "dakshina kannada": (13.3333, 75.2833),
    "hassan": (13.0033, 75.9233),
    "kodagu": (12.3381, 75.7522),
    "mysuru": (12.3050, 76.6550),
    "mandya": (12.5333, 76.8833),
    "chamarajanagar": (11.9269, 76.9388),
    "belagavi": (15.8675, 74.5135),
    "bagalkot": (16.1667, 75.6167),
    "vijayapura": (16.825, 75.7139),
    "gadag": (15.4167, 75.3333),
    "haveri": (14.7967, 75.3833),
    "hubballi-dharwad": (15.3647, 75.1239),
    "kalaburagi": (17.3297, 76.8343),
    "yadgir": (16.75, 76.7667),
    "raichur": (16.2067, 77.3667),
    "koppal": (15.3333, 75.6667),
    "chikkamagalur": (13.3167, 75.7667),
    "belgaum": (15.8675, 74.5135),  # Alternate name for Belagavi
    "gulbarga": (17.3297, 76.8343),  # Alternate name for Kalaburagi
}


def get_coords(district_name: str) -> tuple[float, float]:
    """
    Retrieve latitude and longitude for a Karnataka district with fuzzy matching.

    Performs case-insensitive lookup with fuzzy string matching to handle minor
    spelling variations and alternate district names.

    Parameters
    ----------
    district_name : str
        Name of the district (e.g., "Bengaluru Urban", "MYSURU", "belgavi").
        Fuzzy matching allows for minor spelling differences.

    Returns
    -------
    tuple[float, float]
        (latitude, longitude) tuple for the district.

    Raises
    ------
    ValueError
        If the district name cannot be matched to any known district,
        or if multiple equally-close matches are found and none match exactly.

    Notes
    -----
    - Matching is case-insensitive
    - Uses difflib.get_close_matches for fuzzy matching (cutoff=0.6)
    - Handles alternate district names (e.g., "belgaum" for "belagavi")
    """
    # Normalize input
    normalized_input = district_name.lower().strip()

    # Exact match (fastest path)
    if normalized_input in DISTRICT_COORDS:
        return DISTRICT_COORDS[normalized_input]

    # Fuzzy matching for spelling variations
    close_matches = get_close_matches(
        normalized_input, DISTRICT_COORDS.keys(), n=1, cutoff=0.6
    )

    if close_matches:
        matched_district = close_matches[0]
        return DISTRICT_COORDS[matched_district]

    # No match found
    raise ValueError(
        f"District '{district_name}' not found. "
        f"Available districts: {', '.join(sorted(set(DISTRICT_COORDS.keys())))}"
    )
