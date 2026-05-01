"""
Preprocessing pipeline for the Rainfall Anomaly Prediction System.

Handles missing values, normalization, rolling features, departure
percentages, source merging, and orchestrates the full pipeline.
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Make config importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import DATA_PROCESSED_DIR


# ---------------------------------------------------------------------------
# 1. Missing-value handler
# ---------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing rainfall_mm values per district.

    Strategy (applied in order):
      1. Linear interpolation for gaps up to 7 consecutive days  (limit=7)
      2. Forward-fill for remaining gaps                          (limit=3)
      3. Backward-fill for leading gaps                          (limit=3)
      4. Replace any still-remaining NaN with 0

    Args:
        df: DataFrame with at least columns [district, date, rainfall_mm].

    Returns:
        DataFrame with rainfall_mm gaps filled; original row order preserved.
    """
    if df.empty:
        return df.copy()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    filled_parts = []
    for district, grp in df.groupby("district", sort=False):
        grp = grp.sort_values("date").copy()

        # Step 1 - linear interpolation (time-aware, limit 7 periods)
        grp["rainfall_mm"] = (
            grp["rainfall_mm"]
            .interpolate(method="linear", limit=7, limit_direction="forward")
        )

        # Step 2 - forward-fill (limit 3)
        grp["rainfall_mm"] = grp["rainfall_mm"].ffill(limit=3)

        # Step 3 - backward-fill (limit 3, handles leading NaNs)
        grp["rainfall_mm"] = grp["rainfall_mm"].bfill(limit=3)

        # Step 4 - zero-fill any remainder
        grp["rainfall_mm"] = grp["rainfall_mm"].fillna(0)

        filled_parts.append(grp)

    result = pd.concat(filled_parts, ignore_index=True)
    return result


# ---------------------------------------------------------------------------
# 2. Per-district MinMax normalisation
# ---------------------------------------------------------------------------

def normalize_per_district(
    df: pd.DataFrame, feature_col: str = "rainfall_mm"
) -> tuple:
    """
    Apply MinMax scaling separately per district.

    Districts with fewer than 10 data points are skipped (normalised column
    set to NaN for those rows).

    Manual formula: (x - min) / (max - min + 1e-9)

    Args:
        df:          DataFrame containing at least [district, <feature_col>].
        feature_col: Column to normalise (default 'rainfall_mm').

    Returns:
        Tuple of:
          - df_out (pd.DataFrame): original DataFrame plus
                                   f'{feature_col}_normalized' column.
          - scalers_dict (dict):   {district: {'min': float, 'max': float}}
    """
    if df.empty:
        df_out = df.copy()
        df_out[f"{feature_col}_normalized"] = np.nan
        return df_out, {}

    df = df.copy()
    norm_col = f"{feature_col}_normalized"
    df[norm_col] = np.nan

    scalers_dict: dict = {}

    for district, grp in df.groupby("district", sort=False):
        if len(grp) < 10:
            # Too few points - leave as NaN and record nothing
            continue

        col_min = grp[feature_col].min()
        col_max = grp[feature_col].max()
        scalers_dict[district] = {"min": col_min, "max": col_max}

        normalized = (grp[feature_col] - col_min) / (col_max - col_min + 1e-9)
        df.loc[grp.index, norm_col] = normalized

    return df, scalers_dict


# ---------------------------------------------------------------------------
# 3. Rolling features
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add rolling statistical features per district.

    New columns:
      - rolling_7d_mean:  7-day rolling mean  (min_periods=3)
      - rolling_30d_mean: 30-day rolling mean (min_periods=15)
      - rolling_30d_std:  30-day rolling std  (min_periods=15)

    Args:
        df: DataFrame with at least [district, date, rainfall_mm].

    Returns:
        DataFrame with three new rolling columns appended.
    """
    if df.empty:
        df_out = df.copy()
        for col in ("rolling_7d_mean", "rolling_30d_mean", "rolling_30d_std"):
            df_out[col] = np.nan
        return df_out

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    result_parts = []
    for district, grp in df.groupby("district", sort=False):
        grp = grp.sort_values("date").copy()

        grp["rolling_7d_mean"] = (
            grp["rainfall_mm"].rolling(window=7, min_periods=3).mean()
        )
        grp["rolling_30d_mean"] = (
            grp["rainfall_mm"].rolling(window=30, min_periods=15).mean()
        )
        grp["rolling_30d_std"] = (
            grp["rainfall_mm"].rolling(window=30, min_periods=15).std()
        )

        result_parts.append(grp)

    return pd.concat(result_parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Departure percentage
# ---------------------------------------------------------------------------

def compute_departure_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage departure of rainfall_mm from a 'normal' baseline.

    Baseline selection:
      - If column 'normal_mm' exists  to  use it.
      - Otherwise                     to  use rolling_30d_mean.
        (rolling_30d_mean must already be present; call add_rolling_features
         before this function if 'normal_mm' is absent.)

    Formula:  departure_pct = (rainfall_mm - normal) / (normal + 1e-9) * 100

    Clipped to [-500, 500] to suppress extreme outliers.

    Args:
        df: DataFrame with [rainfall_mm] and either [normal_mm] or
            [rolling_30d_mean].

    Returns:
        DataFrame with 'departure_pct' column added.
    """
    if df.empty:
        df_out = df.copy()
        df_out["departure_pct"] = np.nan
        return df_out

    df = df.copy()

    if "normal_mm" in df.columns:
        baseline = df["normal_mm"]
    elif "rolling_30d_mean" in df.columns:
        baseline = df["rolling_30d_mean"]
    else:
        raise ValueError(
            "compute_departure_pct requires either 'normal_mm' or "
            "'rolling_30d_mean' column. Call add_rolling_features() first."
        )

    df["departure_pct"] = (
        (df["rainfall_mm"] - baseline) / (baseline + 1e-9) * 100
    ).clip(-500, 500)

    return df


# ---------------------------------------------------------------------------
# 5. Merge sources
# ---------------------------------------------------------------------------

def merge_sources(
    kaggle_df: pd.DataFrame, openmeteo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Outer-merge Kaggle and Open-Meteo DataFrames on (district, date).

    Pre-processing:
      - Strip whitespace and title-case district names in both frames.

    Conflict resolution for overlapping rows:
      - rainfall_mm  = mean of both sources.
      - sources_agree: True when the two values differ by less than 20 %.
      - data_source:  'both' | 'kaggle_only' | 'openmeteo_only'.

    Args:
        kaggle_df:    DataFrame with [district, date, rainfall_mm, ...].
        openmeteo_df: DataFrame with [district, date, rainfall_mm, ...].

    Returns:
        Merged DataFrame sorted by (district, date).
    """
    # --- guard: empty inputs ---
    if kaggle_df is None or kaggle_df.empty:
        if openmeteo_df is None or openmeteo_df.empty:
            return pd.DataFrame()
        result = openmeteo_df.copy()
        result["district"] = result["district"].str.strip().str.title()
        result["data_source"] = "openmeteo_only"
        result["sources_agree"] = pd.NA
        return result.sort_values(["district", "date"]).reset_index(drop=True)

    if openmeteo_df is None or openmeteo_df.empty:
        result = kaggle_df.copy()
        result["district"] = result["district"].str.strip().str.title()
        result["data_source"] = "kaggle_only"
        result["sources_agree"] = pd.NA
        return result.sort_values(["district", "date"]).reset_index(drop=True)

    # --- normalise district names ---
    kdf = kaggle_df.copy()
    odf = openmeteo_df.copy()

    kdf["district"] = kdf["district"].str.strip().str.title()
    odf["district"] = odf["district"].str.strip().str.title()

    kdf["date"] = pd.to_datetime(kdf["date"])
    odf["date"] = pd.to_datetime(odf["date"])

    # --- select merge-relevant columns ---
    k_cols = ["district", "date", "rainfall_mm"]
    if "state" in kdf.columns:
        k_cols.append("state")
    kdf_slim = kdf[k_cols].copy()
    # Drop rows where rainfall is entirely absent so they don't create
    # spurious 'unknown' data_source labels after the outer join.
    kdf_slim = kdf_slim.dropna(subset=["rainfall_mm"])

    o_cols = ["district", "date", "rainfall_mm"]
    if "state" in odf.columns:
        o_cols.append("state")
    odf_slim = odf[o_cols].copy()
    odf_slim = odf_slim.dropna(subset=["rainfall_mm"])

    # --- outer merge ---
    merged = pd.merge(
        kdf_slim.rename(columns={"rainfall_mm": "rainfall_kaggle",
                                  "state": "state_kaggle"}),
        odf_slim.rename(columns={"rainfall_mm": "rainfall_openmeteo",
                                  "state": "state_openmeteo"}),
        on=["district", "date"],
        how="outer",
    )

    # --- resolve data_source ---
    both_mask    = merged["rainfall_kaggle"].notna() & merged["rainfall_openmeteo"].notna()
    kaggle_mask  = merged["rainfall_kaggle"].notna() & merged["rainfall_openmeteo"].isna()
    omteo_mask   = merged["rainfall_kaggle"].isna()  & merged["rainfall_openmeteo"].notna()

    merged["data_source"] = "unknown"
    merged.loc[both_mask,   "data_source"] = "both"
    merged.loc[kaggle_mask, "data_source"] = "kaggle_only"
    merged.loc[omteo_mask,  "data_source"] = "openmeteo_only"

    # --- compute unified rainfall_mm ---
    merged["rainfall_mm"] = np.where(
        both_mask,
        (merged["rainfall_kaggle"] + merged["rainfall_openmeteo"]) / 2,
        np.where(
            kaggle_mask,
            merged["rainfall_kaggle"],
            merged["rainfall_openmeteo"],
        ),
    )

    # --- sources_agree (only meaningful where both exist) ---
    # |kaggle - openmeteo| / (mean + 1e-9) < 0.20
    mean_val = (merged["rainfall_kaggle"] + merged["rainfall_openmeteo"]) / 2
    rel_diff = (merged["rainfall_kaggle"] - merged["rainfall_openmeteo"]).abs() / (
        mean_val.abs() + 1e-9
    )
    merged["sources_agree"] = np.where(both_mask, rel_diff < 0.20, pd.NA)

    # --- carry state forward if available ---
    if "state_kaggle" in merged.columns and "state_openmeteo" in merged.columns:
        merged["state"] = merged["state_kaggle"].combine_first(merged["state_openmeteo"])
        merged.drop(columns=["state_kaggle", "state_openmeteo"], inplace=True)
    elif "state_kaggle" in merged.columns:
        merged.rename(columns={"state_kaggle": "state"}, inplace=True)
    elif "state_openmeteo" in merged.columns:
        merged.rename(columns={"state_openmeteo": "state"}, inplace=True)

    # --- drop helper columns ---
    merged.drop(columns=["rainfall_kaggle", "rainfall_openmeteo"], inplace=True)

    return merged.sort_values(["district", "date"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Full pipeline
# ---------------------------------------------------------------------------

def preprocess_pipeline(
    kaggle_df: pd.DataFrame = None,
    openmeteo_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Orchestrate the complete preprocessing pipeline.

    Steps:
      1. merge_sources()           (if both DataFrames provided)
      2. handle_missing_values()
      3. add_rolling_features()
      4. compute_departure_pct()
      5. normalize_per_district()  on rainfall_mm
      6. Save to DATA_PROCESSED_DIR / 'processed_rainfall.csv'

    Args:
        kaggle_df:    Kaggle DataFrame (optional).
        openmeteo_df: Open-Meteo DataFrame (optional).

    Returns:
        Fully processed DataFrame.

    Raises:
        ValueError: If neither DataFrame is provided.
    """
    # --- input validation ---
    kaggle_empty  = kaggle_df is None or (hasattr(kaggle_df,  "empty") and kaggle_df.empty)
    omteo_empty   = openmeteo_df is None or (hasattr(openmeteo_df, "empty") and openmeteo_df.empty)

    if kaggle_empty and omteo_empty:
        raise ValueError(
            "preprocess_pipeline requires at least one non-empty DataFrame "
            "(kaggle_df or openmeteo_df)."
        )

    # --- step 1: merge or select ---
    if not kaggle_empty and not omteo_empty:
        print("Step 1/5 - Merging Kaggle and Open-Meteo sources ...")
        df = merge_sources(kaggle_df, openmeteo_df)
    elif not kaggle_empty:
        print("Step 1/5 - Using Kaggle data only ...")
        df = kaggle_df.copy()
        df["district"] = df["district"].str.strip().str.title()
        df["date"] = pd.to_datetime(df["date"])
        df["data_source"] = "kaggle_only"
    else:
        print("Step 1/5 - Using Open-Meteo data only ...")
        df = openmeteo_df.copy()
        df["district"] = df["district"].str.strip().str.title()
        df["date"] = pd.to_datetime(df["date"])
        df["data_source"] = "openmeteo_only"

    if df.empty:
        print("Warning: merged DataFrame is empty - aborting pipeline.")
        return df

    # --- step 2: missing values ---
    print("Step 2/7 - Handling missing values ...")
    df = handle_missing_values(df)

    # --- step 3: attach IMD normals ---
    print("Step 3/7 - Attaching IMD climatological normals ...")
    try:
        from src.data_ingestion.imd_normals import load_district_normals
        normals = load_district_normals()
        if not normals.empty and "date" in df.columns:
            df["_month"] = pd.to_datetime(df["date"]).dt.month
            df = df.merge(
                normals[["district", "month", "normal_mm_daily"]].rename(
                    columns={"month": "_month", "normal_mm_daily": "normal_mm"}
                ),
                on=["district", "_month"],
                how="left",
            )
            df.drop(columns=["_month"], inplace=True)
            matched = df["normal_mm"].notna().sum()
            print(f"  -> {matched:,} rows got IMD normals ({matched/len(df)*100:.0f}%)")
        else:
            print("  -> Skipped (no normals data or no date column)")
    except Exception as exc:
        print(f"  -> IMD normals skipped: {exc}")

    # --- step 4: attach historical context features ---
    print("Step 4/7 - Attaching historical context features ...")
    try:
        from src.data_ingestion.imd_historical import (
            load_subdivision_mapping, compute_historical_features,
        )
        _days_map = {1:31, 2:28.25, 3:31, 4:30, 5:31, 6:30,
                     7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
        subdiv_map = load_subdivision_mapping()
        hist_feats = compute_historical_features()
        if not subdiv_map.empty and not hist_feats.empty and "date" in df.columns:
            df["_month"] = pd.to_datetime(df["date"]).dt.month
            df = df.merge(subdiv_map, on="district", how="left")
            df = df.merge(
                hist_feats.rename(columns={"month": "_month"}),
                on=["subdivision", "_month"],
                how="left",
            )
            # Compute historical departure (daily basis)
            _days_col = df["_month"].map(_days_map)
            daily_hist_mean = df["hist_mean_mm"] / _days_col
            daily_hist_std = df["hist_std_mm"] / _days_col
            df["hist_departure_pct"] = (
                (df["rainfall_mm"] - daily_hist_mean) / (daily_hist_mean + 1e-9) * 100
            ).clip(-500, 500)
            # Historical percentile rank (approximate via z-score -> CDF)
            from scipy import stats
            z = (df["rainfall_mm"] - daily_hist_mean) / (daily_hist_std + 1e-9)
            df["hist_percentile_rank"] = stats.norm.cdf(z) * 100
            df.drop(columns=["_month"], inplace=True)
            matched = df["subdivision"].notna().sum()
            print(f"  -> {matched:,} rows got historical features ({matched/len(df)*100:.0f}%)")
        else:
            df.drop(columns=["_month"], errors="ignore", inplace=True)
            print("  -> Skipped (no mapping or historical data)")
    except Exception as exc:
        print(f"  -> Historical features skipped: {exc}")

    # --- step 5: rolling features ---
    print("Step 5/7 - Adding rolling features ...")
    df = add_rolling_features(df)

    # --- step 6: departure percentage ---
    print("Step 6/7 - Computing departure percentages ...")
    df = compute_departure_pct(df)

    # --- step 7: normalise ---
    print("Step 7/7 - Normalising rainfall_mm per district ...")
    df, scalers = normalize_per_district(df, feature_col="rainfall_mm")

    # --- save to disk ---
    out_dir = Path(DATA_PROCESSED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "processed_rainfall.csv"
    df.to_csv(out_path, index=False)
    print(f"\nProcessed data saved to: {out_path.resolve()}")

    # --- summary ---
    date_min = df["date"].min()
    date_max = df["date"].max()
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE - SUMMARY")
    print("=" * 60)
    print(f"  Total rows    : {len(df):,}")
    print(f"  Districts     : {df['district'].nunique()}")
    print(f"  Date range    : {date_min.date()} to {date_max.date()}")
    print(f"  Columns       : {list(df.columns)}")
    print(f"  Scalers saved : {len(scalers)} districts")
    print("=" * 60 + "\n")

    return df
