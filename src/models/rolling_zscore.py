"""
Adaptive rolling Z-Score anomaly detection per district for the
Rainfall Anomaly Prediction System (ml_raksha).
"""

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import (
    ZSCORE_WINDOW,
    ZSCORE_MODERATE_THRESHOLD,
    ZSCORE_EXTREME_THRESHOLD,
)


# ---------------------------------------------------------------------------
# Z-score computation
# ---------------------------------------------------------------------------

def compute_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a 30-day rolling Z-score of rainfall_mm per district.

    The rolling statistics use min_periods=15 so that results are
    produced even when fewer than ZSCORE_WINDOW observations are available,
    provided at least 15 are present.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rainfall_mm'.  May contain 'district' and 'date'.

    Returns
    -------
    pd.DataFrame
        Copy of df with a 'z_score' column added.
        Rows with insufficient history (< 15 observations) will have NaN.
    """
    if df.empty:
        print("[ZScore] compute_zscore: received empty DataFrame.")
        return df

    df = df.copy()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    def _rolling_zscore(series: pd.Series) -> pd.Series:
        rolling_mean = series.rolling(
            window=ZSCORE_WINDOW, min_periods=15
        ).mean()
        rolling_std = series.rolling(
            window=ZSCORE_WINDOW, min_periods=15
        ).std()
        return (series - rolling_mean) / (rolling_std + 1e-9)

    if "district" in df.columns:
        sort_cols = ["district", "date"] if "date" in df.columns else ["district"]
        df = df.sort_values(sort_cols)
        df["z_score"] = (
            df.groupby("district")["rainfall_mm"]
            .transform(_rolling_zscore)
        )
    else:
        if "date" in df.columns:
            df = df.sort_values("date")
        df["z_score"] = _rolling_zscore(df["rainfall_mm"])

    print(
        f"[ZScore] compute_zscore complete. "
        f"Non-null z_scores: {df['z_score'].notna().sum()} / {len(df)}."
    )
    return df


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_zscore(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign a severity category based on the absolute Z-score value.

    Categories (driven by config thresholds):
    - 'Normal'   : |z_score| < ZSCORE_MODERATE_THRESHOLD
    - 'Moderate' : ZSCORE_MODERATE_THRESHOLD <= |z_score| < ZSCORE_EXTREME_THRESHOLD
    - 'Extreme'  : |z_score| >= ZSCORE_EXTREME_THRESHOLD

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'z_score' column (output of compute_zscore).

    Returns
    -------
    pd.DataFrame
        Copy of df with 'zscore_category' column added.
        Rows where z_score is NaN are labelled 'Normal' by default.
    """
    if df.empty:
        print("[ZScore] classify_zscore: received empty DataFrame.")
        return df

    if "z_score" not in df.columns:
        raise ValueError(
            "[ZScore] classify_zscore: 'z_score' column not found. "
            "Run compute_zscore() first."
        )

    df = df.copy()
    abs_z = df["z_score"].abs()

    df["zscore_category"] = "Normal"
    df.loc[
        (abs_z >= ZSCORE_MODERATE_THRESHOLD) & (abs_z < ZSCORE_EXTREME_THRESHOLD),
        "zscore_category",
    ] = "Moderate"
    df.loc[abs_z >= ZSCORE_EXTREME_THRESHOLD, "zscore_category"] = "Extreme"

    print(
        f"[ZScore] classify_zscore complete. "
        f"Normal: {(df['zscore_category'] == 'Normal').sum()}, "
        f"Moderate: {(df['zscore_category'] == 'Moderate').sum()}, "
        f"Extreme: {(df['zscore_category'] == 'Extreme').sum()}."
    )
    return df


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_zscore_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full Z-score pipeline: compute rolling Z-scores then classify them.

    After processing, prints a per-district summary showing the percentage
    of records in each severity category.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rainfall_mm'.  Should contain 'district' for
        per-district statistics.

    Returns
    -------
    pd.DataFrame
        df enriched with 'z_score' and 'zscore_category' columns.
    """
    if df.empty:
        print("[ZScore] run_zscore_analysis: received empty DataFrame.")
        return df

    print("[ZScore] Starting Z-score analysis pipeline ...")

    df = compute_zscore(df)
    df = classify_zscore(df)

    # Per-district summary
    if "district" in df.columns:
        districts = df["district"].unique().tolist()
        print(
            f"\n[ZScore] --- Per-district summary ({len(districts)} district(s)) ---"
        )
        for district in sorted(districts):
            sub = df[df["district"] == district]
            total = len(sub)
            if total == 0:
                continue
            for cat in ("Normal", "Moderate", "Extreme"):
                pct = 100.0 * (sub["zscore_category"] == cat).sum() / total
                print(f"  {district:<30}  {cat:<10}  {pct:5.1f}%")
        print("[ZScore] --- End of summary ---\n")
    else:
        total = len(df)
        if total > 0:
            print("\n[ZScore] --- Global summary ---")
            for cat in ("Normal", "Moderate", "Extreme"):
                pct = 100.0 * (df["zscore_category"] == cat).sum() / total
                print(f"  {cat:<10}  {pct:5.1f}%")
            print("[ZScore] --- End of summary ---\n")

    print("[ZScore] run_zscore_analysis complete.")
    return df
