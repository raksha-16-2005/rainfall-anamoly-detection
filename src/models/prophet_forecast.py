"""
Prophet-based time-series forecasting per district for the
Rainfall Anomaly Prediction System (ml_raksha).
"""

import pickle
import datetime
from pathlib import Path

import pandas as pd

try:
    from prophet import Prophet
except ImportError:
    raise ImportError(
        "prophet is not installed. Run: pip install prophet"
    )

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    MODELS_CACHE_DIR,
    MODEL_CACHE_DAYS,
    PROPHET_FORECAST_DAYS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cache_path(district: str) -> Path:
    """Return the .pkl cache path for a given district."""
    MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return MODELS_CACHE_DIR / f"prophet_{district}.pkl"


def _cache_is_valid(cache_file: Path) -> bool:
    """Return True if cache file exists and is less than MODEL_CACHE_DAYS old."""
    if not cache_file.exists():
        return False
    mtime = datetime.datetime.fromtimestamp(cache_file.stat().st_mtime)
    age_days = (datetime.datetime.now() - mtime).days
    return age_days < MODEL_CACHE_DAYS


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def train_prophet(district: str, df: pd.DataFrame) -> dict:
    """
    Train a Prophet model for a single district.

    Parameters
    ----------
    district : str
        District name used for caching and labelling.
    df : pd.DataFrame
        Must contain columns [date, rainfall_mm].
        May optionally contain a 'district' column for filtering.

    Returns
    -------
    dict
        {'model': <fitted Prophet>, 'district': district, 'trained_on': <date>}
    """
    if df.empty:
        raise ValueError(f"[Prophet] Empty DataFrame provided for district '{district}'.")

    # Filter for district if the column is present
    if "district" in df.columns:
        district_df = df[df["district"] == district].copy()
    else:
        district_df = df.copy()

    if district_df.empty:
        raise ValueError(
            f"[Prophet] No rows found for district '{district}' after filtering."
        )

    # Prepare Prophet input
    prophet_df = district_df[["date", "rainfall_mm"]].rename(
        columns={"date": "ds", "rainfall_mm": "y"}
    )
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    prophet_df = prophet_df.dropna(subset=["ds", "y"])

    if len(prophet_df) < 2:
        raise ValueError(
            f"[Prophet] Insufficient data for district '{district}' "
            f"(need at least 2 rows, got {len(prophet_df)})."
        )

    print(f"[Prophet] Training model for '{district}' on {len(prophet_df)} rows ...")

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        interval_width=0.8,
    )
    model.add_seasonality(
        name="monsoon",
        period=365.25 / 2,
        fourier_order=5,
    )
    model.add_seasonality(
        name="post_monsoon",
        period=365.25 / 4,
        fourier_order=3,
    )
    model.fit(prophet_df)

    # Cache the trained model
    cache_file = _cache_path(district)
    result = {
        "model": model,
        "district": district,
        "trained_on": datetime.date.today().isoformat(),
    }
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"[Prophet] Model cached at '{cache_file}'.")

    return result


def load_or_train(district: str, df: pd.DataFrame) -> dict:
    """
    Return a cached Prophet model if it is fresh, otherwise train a new one.

    Parameters
    ----------
    district : str
    df : pd.DataFrame

    Returns
    -------
    dict  (same shape as train_prophet return value)
    """
    cache_file = _cache_path(district)

    if _cache_is_valid(cache_file):
        print(f"[Prophet] Loading cached model for '{district}' from '{cache_file}'.")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    print(f"[Prophet] Cache missing or stale for '{district}'. Retraining ...")
    return train_prophet(district, df)


def forecast_district(
    district: str, df: pd.DataFrame, days: int = None
) -> pd.DataFrame:
    """
    Generate a rainfall forecast for a single district.

    Parameters
    ----------
    district : str
    df : pd.DataFrame
    days : int, optional
        Number of days to forecast. Defaults to PROPHET_FORECAST_DAYS.

    Returns
    -------
    pd.DataFrame
        Columns: district, ds, yhat, yhat_lower, yhat_upper
    """
    if days is None:
        days = PROPHET_FORECAST_DAYS

    model_dict = load_or_train(district, df)
    model = model_dict["model"]

    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(days).copy()
    result.insert(0, "district", district)
    result.reset_index(drop=True, inplace=True)

    print(
        f"[Prophet] Forecast for '{district}': {days} day(s) generated."
    )
    return result


def forecast_all_districts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecast all unique districts found in df.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'district' column.

    Returns
    -------
    pd.DataFrame
        Combined forecast for every district.
    """
    if df.empty:
        print("[Prophet] forecast_all_districts: received empty DataFrame.")
        return pd.DataFrame()

    if "district" not in df.columns:
        raise ValueError(
            "[Prophet] forecast_all_districts: 'district' column not found in DataFrame."
        )

    districts = df["district"].unique().tolist()
    print(
        f"[Prophet] Starting forecast for {len(districts)} district(s): {districts}"
    )

    frames = []
    for i, district in enumerate(districts, 1):
        print(f"[Prophet] Processing district {i}/{len(districts)}: '{district}'")
        try:
            result = forecast_district(district, df)
            frames.append(result)
        except Exception as exc:
            print(f"[Prophet] WARNING – skipping '{district}': {exc}")

    if not frames:
        print("[Prophet] No forecasts were generated.")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(
        f"[Prophet] forecast_all_districts complete. "
        f"Total rows: {len(combined)}."
    )
    return combined
