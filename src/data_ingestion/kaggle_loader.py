"""
Kaggle dataset loader for daily rainfall data (India, 2009-2024).
Downloads, validates, and reshapes the dataset from Kaggle.
"""

import subprocess
import sys
from pathlib import Path
import pandas as pd

# Import config constants
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_RAW_DIR


def download_dataset() -> Path:
    """
    Download the Kaggle dataset using the kaggle CLI.

    Uses: kaggle datasets download -d wydoinn/daily-rainfall-data-india-2009-2024

    Returns:
        Path to the downloaded CSV file (first .csv found in DATA_RAW_DIR)

    Raises:
        RuntimeError: If kaggle CLI is not configured or download fails
    """
    dataset_id = "wydoinn/daily-rainfall-data-india-2009-2024"
    data_raw = Path(DATA_RAW_DIR)
    data_raw.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {dataset_id}")
    print(f"Destination: {data_raw.absolute()}")

    try:
        # Run kaggle CLI command
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_id,
            "--unzip",
            "-p", str(data_raw)
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Dataset downloaded successfully.")
    except FileNotFoundError:
        raise RuntimeError(
            "Kaggle CLI not found. Please install it: pip install kaggle\n"
            "Then configure credentials: place kaggle.json in ~/.kaggle/\n"
            "See: https://www.kaggle.com/settings/account"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Kaggle download failed.\n"
            f"Ensure kaggle CLI is configured with valid credentials.\n"
            f"Error: {e.stderr}"
        )

    # Find the first .csv file in DATA_RAW_DIR
    csv_files = list(data_raw.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(f"No CSV file found in {data_raw.absolute()}")

    csv_path = csv_files[0]
    print(f"Found CSV: {csv_path.name}")
    return csv_path


def load_raw(filepath: Path) -> pd.DataFrame:
    """
    Load raw CSV file with encoding fallback.

    Args:
        filepath: Path to the CSV file

    Returns:
        Raw DataFrame with debugging info printed
    """
    filepath = Path(filepath)
    print(f"Loading raw data from: {filepath}")

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        print("  UTF-8 decoding failed, trying latin-1...")
        df = pd.read_csv(filepath, encoding='latin-1')

    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")

    return df


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and normalize column names.

    Expected columns:
    - DISTRICT (or district)
    - STATE (or state)
    - YEAR (or year)
    - JAN, FEB, MAR, APR, MAY, JUN, JUL, AUG, SEP, OCT, NOV, DEC (monthly rainfall, case-insensitive)
    - ANNUAL (optional)

    All column names are normalized to uppercase.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with uppercase column names

    Raises:
        ValueError: If required columns are missing
    """
    print("Validating columns...")

    # Normalize columns to uppercase
    df.columns = [col.upper() for col in df.columns]

    # Define expected columns
    required_base = {'DISTRICT', 'STATE', 'YEAR'}
    month_cols = {'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'}
    required_all = required_base | month_cols

    # Check which required columns are present
    missing = required_all - set(df.columns)

    if missing:
        found = set(df.columns)
        raise ValueError(
            f"Missing required columns: {sorted(missing)}\n"
            f"Found columns: {sorted(found)}\n"
            f"Expected at minimum: {sorted(required_all)}"
        )

    print(f"  All required columns present: {sorted(required_base | month_cols)}")
    return df


def reshape_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape from wide format (monthly columns) to long format (one row per month).

    Input: One row per (district, year) with columns JAN, FEB, ..., DEC
    Output: One row per (district, state, year, month) with date and rainfall_mm

    Args:
        df: DataFrame with monthly columns (JAN-DEC)

    Returns:
        Reshaped DataFrame with columns: district, state, date, rainfall_mm
    """
    print("Reshaping to daily format...")

    # Define month mapping
    month_names = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4,
        'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8,
        'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }

    rows = []

    for _, row in df.iterrows():
        district = row['DISTRICT']
        state = row['STATE']
        year = int(row['YEAR'])

        for month_name, month_num in month_names.items():
            rainfall = row[month_name]

            # Handle NaN and missing values
            if pd.isna(rainfall):
                rainfall = 0
            else:
                rainfall = float(rainfall)

            # Replace negative values with 0 (common for missing data)
            if rainfall < 0:
                rainfall = 0

            # Create date for the first day of the month
            date = pd.Timestamp(year=year, month=month_num, day=1)

            rows.append({
                'district': district,
                'state': state,
                'date': date,
                'rainfall_mm': rainfall
            })

    result_df = pd.DataFrame(rows)
    print(f"  Reshaped to {len(result_df)} rows (district-month combinations)")

    return result_df


def load_kaggle_data(filepath: Path = None) -> pd.DataFrame:
    """
    Main entry point to load and process Kaggle rainfall data.

    Workflow:
    1. If filepath is None, look for CSV in DATA_RAW_DIR
    2. If no CSV found, download dataset
    3. Load raw → validate → reshape → add metadata
    4. Print summary

    Args:
        filepath: Optional path to CSV file. If None, searches DATA_RAW_DIR

    Returns:
        Processed DataFrame with columns: district, state, date, rainfall_mm, source
        Sorted by (district, date)
    """
    print("\n" + "="*70)
    print("KAGGLE DATA LOADER")
    print("="*70)

    # Step 1: Locate CSV file
    if filepath is None:
        data_raw = Path(DATA_RAW_DIR)
        csv_files = list(data_raw.glob("*.csv")) if data_raw.exists() else []

        if csv_files:
            filepath = csv_files[0]
            print(f"Found existing CSV in {data_raw}: {filepath.name}")
        else:
            print(f"No CSV found in {data_raw}, downloading...")
            filepath = download_dataset()
    else:
        filepath = Path(filepath)
        print(f"Using provided filepath: {filepath}")

    # Step 2: Load and process
    df_raw = load_raw(filepath)
    df_validated = validate_columns(df_raw)
    df_reshaped = reshape_to_daily(df_validated)

    # Step 3: Add source metadata
    df_reshaped['source'] = 'kaggle'

    # Step 4: Sort by district and date
    df_reshaped = df_reshaped.sort_values(['district', 'date']).reset_index(drop=True)

    # Step 5: Print summary
    print("\n" + "-"*70)
    print("SUMMARY")
    print("-"*70)
    print(f"Total rows: {len(df_reshaped)}")
    print(f"Unique districts: {df_reshaped['district'].nunique()}")
    print(f"Unique states: {df_reshaped['state'].nunique()}")
    print(f"Date range: {df_reshaped['date'].min().date()} to {df_reshaped['date'].max().date()}")
    print(f"Columns: {list(df_reshaped.columns)}")
    print("-"*70 + "\n")

    return df_reshaped
