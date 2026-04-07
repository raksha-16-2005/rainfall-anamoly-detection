"""
Data loader for government rainfall data from data.gov.in.

Module for loading, validating, and preprocessing rainfall data from CSV files.
"""

import pandas as pd


def load_district_rainfall(filepath: str) -> pd.DataFrame:
    """
    Load and preprocess rainfall data from data.gov.in CSV file.

    Performs the following operations:
    1. Reads a CSV file with rainfall data
    2. Parses the Date column as datetime
    3. Handles missing values using forward-fill, then interpolation
    4. Normalizes the District column to lowercase stripped strings
    5. Returns a DataFrame sorted by District and Date

    Parameters
    ----------
    filepath : str
        Path to the CSV file containing rainfall data with columns:
        District, Date, Rainfall_mm, Normal_mm, Departure_pct

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        - District: normalized to lowercase, stripped strings
        - Date: datetime type
        - Rainfall_mm: float, missing values filled
        - Normal_mm: float
        - Departure_pct: float
        Sorted by District and Date in ascending order.

    Raises
    ------
    FileNotFoundError
        If the specified file does not exist.
    ValueError
        If required columns are missing from the CSV file.
    """
    # Read the CSV file
    df = pd.read_csv(filepath)

    # Validate required columns
    required_columns = ["District", "Date", "Rainfall_mm", "Normal_mm", "Departure_pct"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Parse Date column as datetime
    df["Date"] = pd.to_datetime(df["Date"])

    # Normalize District column to lowercase stripped strings
    df["District"] = df["District"].str.lower().str.strip()

    # Handle missing values in Rainfall_mm: forward-fill then interpolation
    df["Rainfall_mm"] = df["Rainfall_mm"].fillna(method="ffill").interpolate(method="linear")

    # Sort by District and Date
    df = df.sort_values(by=["District", "Date"]).reset_index(drop=True)

    return df
