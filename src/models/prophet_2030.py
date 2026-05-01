"""
Prophet-based Monthly Rainfall Projection to 2030.

Combines 1901-2015 IMD historical data (subdivision-level) with 2023-2026
Open-Meteo district-level data to project monthly rainfall through 2030.

Uses monthly aggregates for robust long-range projections with confidence
intervals that widen naturally over time.
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from prophet import Prophet

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_PROCESSED_DIR, MODELS_CACHE_DIR

logger = logging.getLogger(__name__)

PROJECTION_CACHE_DIR = MODELS_CACHE_DIR / "prophet_2030"


def _build_monthly_series(district: str, classified_df: pd.DataFrame,
                          hist_annual_df: pd.DataFrame,
                          subdiv_map_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a long monthly time series for a district by combining:
    1. IMD subdivision-level historical data (1901-2015) - monthly
    2. Open-Meteo district-level data (2023-2026) - aggregated to monthly

    Returns DataFrame with columns [ds, y] for Prophet.
    """
    rows = []

    # --- Part 1: Historical subdivision data (1901-2015) ---
    subdivision = None
    if subdiv_map_df is not None and not subdiv_map_df.empty:
        match = subdiv_map_df[subdiv_map_df["district"] == district]
        if not match.empty:
            subdivision = match.iloc[0]["subdivision"]

    if subdivision is not None:
        hist_file = Path(__file__).resolve().parents[2] / "rainfall in india 1901-2015.csv"
        if hist_file.exists():
            hist = pd.read_csv(hist_file)
            sub_data = hist[hist["SUBDIVISION"] == subdivision]
            month_cols = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                          "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
            for _, row in sub_data.iterrows():
                year = int(row["YEAR"])
                for i, col in enumerate(month_cols, 1):
                    val = row.get(col)
                    if pd.notna(val):
                        # Use 15th of each month as representative date
                        rows.append({
                            "ds": pd.Timestamp(year=year, month=i, day=15),
                            "y": float(val),
                        })

    # --- Part 2: Recent Open-Meteo data (2023-2026) aggregated monthly ---
    if classified_df is not None and not classified_df.empty:
        dist_data = classified_df[classified_df["district"] == district].copy()
        if not dist_data.empty and "date" in dist_data.columns:
            dist_data["date"] = pd.to_datetime(dist_data["date"])
            dist_data["year"] = dist_data["date"].dt.year
            dist_data["month"] = dist_data["date"].dt.month
            monthly = dist_data.groupby(["year", "month"])["rainfall_mm"].sum().reset_index()
            for _, row in monthly.iterrows():
                rows.append({
                    "ds": pd.Timestamp(year=int(row["year"]), month=int(row["month"]), day=15),
                    "y": float(row["rainfall_mm"]),
                })

    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    result = pd.DataFrame(rows)
    # Deduplicate: if both sources have same month, prefer recent Open-Meteo
    result = result.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")
    result = result.sort_values("ds").reset_index(drop=True)

    return result


def project_district_2030(district: str,
                          classified_df: pd.DataFrame,
                          subdiv_map_df: pd.DataFrame,
                          use_cache: bool = True) -> pd.DataFrame:
    """
    Project monthly rainfall for a district through December 2030.

    Returns DataFrame with columns:
    [district, ds, yhat, yhat_lower, yhat_upper, type]
    where type is 'historical' for fitted values or 'projection' for future.
    """
    cache_dir = PROJECTION_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"proj_{district.replace(' ', '_')}.pkl"

    if use_cache and cache_file.exists():
        age = (datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)).days
        if age < 7:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    # Build training series
    series = _build_monthly_series(district, classified_df, None, subdiv_map_df)

    if len(series) < 24:  # need at least 2 years of monthly data
        logger.warning(f"Insufficient data for {district} ({len(series)} months)")
        return pd.DataFrame(columns=["district", "ds", "yhat", "yhat_lower", "yhat_upper", "type"])

    # Clip negative rainfall
    series["y"] = series["y"].clip(lower=0)

    # Train Prophet
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",  # rainfall is multiplicative (monsoon scaling)
        changepoint_prior_scale=0.01,  # conservative trend changes for long-range
        interval_width=0.80,  # 80% confidence interval
    )

    # Suppress Prophet's verbose output
    import logging as _logging
    _logging.getLogger("prophet").setLevel(_logging.WARNING)
    _logging.getLogger("cmdstanpy").setLevel(_logging.WARNING)

    model.fit(series)

    # Project to end of 2030
    last_date = series["ds"].max()
    end_date = pd.Timestamp("2030-12-15")
    months_ahead = (end_date.year - last_date.year) * 12 + (end_date.month - last_date.month)
    if months_ahead < 1:
        months_ahead = 1

    future = model.make_future_dataframe(periods=months_ahead, freq="MS")
    # Shift to 15th of month for consistency
    future["ds"] = future["ds"].apply(lambda d: d.replace(day=15))
    future = future.drop_duplicates(subset=["ds"])

    forecast = model.predict(future)

    # Clip negative predictions
    forecast["yhat"] = forecast["yhat"].clip(lower=0)
    forecast["yhat_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["yhat_upper"] = forecast["yhat_upper"].clip(lower=0)

    result = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    result.insert(0, "district", district)
    result["type"] = np.where(result["ds"] <= last_date, "historical", "projection")

    # Cache
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)

    return result


def project_all_districts(classified_df: pd.DataFrame,
                          subdiv_map_df: pd.DataFrame,
                          districts: list = None,
                          use_cache: bool = True) -> pd.DataFrame:
    """
    Run 2030 projection for multiple districts.

    Args:
        classified_df: Current classified rainfall data
        subdiv_map_df: District -> subdivision mapping
        districts: List of districts to project (None = all)
        use_cache: Use cached models if available

    Returns:
        Consolidated DataFrame of all projections
    """
    if districts is None:
        districts = sorted(classified_df["district"].unique().tolist())

    results = []
    total = len(districts)
    for i, district in enumerate(districts, 1):
        if i % 50 == 0 or i == 1 or i == total:
            print(f"  [{i}/{total}] Projecting {district}...")
        try:
            proj = project_district_2030(
                district, classified_df, subdiv_map_df, use_cache=use_cache
            )
            if not proj.empty:
                results.append(proj)
        except Exception as exc:
            logger.warning(f"Failed for {district}: {exc}")

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    return combined


def compute_risk_projections(projections_df: pd.DataFrame,
                             normals_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Classify projected months into risk categories based on predicted rainfall
    vs historical normals.

    Adds columns: projected_risk, departure_from_normal_pct
    """
    if projections_df.empty:
        return projections_df

    df = projections_df.copy()
    df["month"] = df["ds"].dt.month
    df["year"] = df["ds"].dt.year

    # Merge normals if available
    if normals_df is not None and not normals_df.empty:
        df = df.merge(
            normals_df[["district", "month", "normal_mm"]],
            on=["district", "month"],
            how="left",
        )
        df["departure_pct"] = (
            (df["yhat"] - df["normal_mm"]) / (df["normal_mm"] + 1e-9) * 100
        )
    else:
        df["normal_mm"] = np.nan
        df["departure_pct"] = np.nan

    # Risk classification based on upper confidence bound
    conditions = [
        df["yhat_upper"] > df["normal_mm"] * 2,  # >200% of normal
        df["yhat_upper"] > df["normal_mm"] * 1.5,  # >150% of normal
        df["yhat_lower"] < df["normal_mm"] * 0.3,  # <30% of normal (drought)
    ]
    choices = ["High Excess Risk", "Moderate Excess Risk", "Drought Risk"]
    df["projected_risk"] = np.select(conditions, choices, default="Normal")

    return df
