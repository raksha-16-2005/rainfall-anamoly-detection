"""
Isolation Forest-based point anomaly detection per district for the
Rainfall Anomaly Prediction System (ml_raksha).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import ISOLATION_FOREST_CONTAMINATION


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def compute_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the DataFrame contains a 'rolling_7d_mean' column.

    If the column is already present it is returned as-is.
    Otherwise the 7-day rolling mean of rainfall_mm is computed
    per district (or globally if no district column exists).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rainfall_mm'. May contain 'district' and 'date'.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'rolling_7d_mean' column added (in-place copy).
    """
    if df.empty:
        print("[IsolationForest] compute_rolling_features: received empty DataFrame.")
        return df

    df = df.copy()

    if "rolling_7d_mean" in df.columns:
        print("[IsolationForest] 'rolling_7d_mean' already present – skipping computation.")
        return df

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if "district" in df.columns:
        df = df.sort_values(["district", "date"]) if "date" in df.columns else df.sort_values("district")
        df["rolling_7d_mean"] = (
            df.groupby("district")["rainfall_mm"]
            .transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        )
    else:
        if "date" in df.columns:
            df = df.sort_values("date")
        df["rolling_7d_mean"] = df["rainfall_mm"].rolling(window=7, min_periods=1).mean()

    print("[IsolationForest] 'rolling_7d_mean' column computed.")
    return df


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_isolation_forest(district: str, df: pd.DataFrame) -> IsolationForest:
    """
    Train an Isolation Forest model for a single district.

    Parameters
    ----------
    district : str
    df : pd.DataFrame
        Must contain 'rainfall_mm' and 'rolling_7d_mean'.
        May contain 'departure_pct'.

    Returns
    -------
    IsolationForest
        Fitted model.
    """
    if df.empty:
        raise ValueError(
            f"[IsolationForest] Empty DataFrame provided for district '{district}'."
        )

    df = compute_rolling_features(df)

    # Build feature matrix
    features = df[["rainfall_mm"]].copy()

    if "departure_pct" in df.columns:
        features["departure_pct"] = df["departure_pct"].fillna(0)
    else:
        features["departure_pct"] = 0

    features["rolling_7d_mean"] = df["rolling_7d_mean"]

    if "hist_departure_pct" in df.columns:
        features["hist_departure_pct"] = df["hist_departure_pct"].fillna(0)

    features = features.fillna(0)

    if len(features) < 2:
        raise ValueError(
            f"[IsolationForest] Insufficient data for district '{district}' "
            f"(need at least 2 rows, got {len(features)})."
        )

    print(
        f"[IsolationForest] Training model for '{district}' "
        f"on {len(features)} samples ..."
    )

    model = IsolationForest(
        contamination=ISOLATION_FOREST_CONTAMINATION,
        random_state=42,
        n_estimators=100,
    )
    model.fit(features)
    return model


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest anomaly detection for every district in df.

    Adds two columns to the returned DataFrame:
    - anomaly_flag  : -1 (anomaly) or 1 (normal), as returned by IsolationForest.predict
    - anomaly_score : decision_function value (higher = more normal)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'rainfall_mm' and 'district'.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'anomaly_flag' and 'anomaly_score' columns added.
    """
    if df.empty:
        print("[IsolationForest] detect_anomalies: received empty DataFrame.")
        return df

    if "district" not in df.columns:
        raise ValueError(
            "[IsolationForest] detect_anomalies: 'district' column not found."
        )

    df = compute_rolling_features(df.copy())

    # Initialise output columns
    df["anomaly_flag"] = 1
    df["anomaly_score"] = 0.0

    districts = df["district"].unique().tolist()
    print(f"[IsolationForest] Detecting anomalies in {len(districts)} district(s) ...")

    for district in districts:
        mask = df["district"] == district
        district_df = df[mask].copy()

        try:
            model = train_isolation_forest(district, district_df)

            # Re-build the same feature matrix used in training
            features = district_df[["rainfall_mm"]].copy()
            if "departure_pct" in district_df.columns:
                features["departure_pct"] = district_df["departure_pct"].fillna(0)
            else:
                features["departure_pct"] = 0
            features["rolling_7d_mean"] = district_df["rolling_7d_mean"]
            if "hist_departure_pct" in district_df.columns:
                features["hist_departure_pct"] = district_df["hist_departure_pct"].fillna(0)
            features = features.fillna(0)

            df.loc[mask, "anomaly_flag"] = model.predict(features)
            df.loc[mask, "anomaly_score"] = model.decision_function(features)

            n_anomalies = int((df.loc[mask, "anomaly_flag"] == -1).sum())
            print(
                f"[IsolationForest] '{district}': {n_anomalies} anomalies "
                f"detected out of {mask.sum()} records."
            )

        except Exception as exc:
            print(f"[IsolationForest] WARNING – skipping '{district}': {exc}")

    return df


# ---------------------------------------------------------------------------
# Query helpers
# ---------------------------------------------------------------------------

def get_anomalous_districts(df: pd.DataFrame, date: str = None) -> pd.DataFrame:
    """
    Return rows flagged as anomalous (anomaly_flag == -1).

    Parameters
    ----------
    df : pd.DataFrame
        Output of detect_anomalies() – must have 'anomaly_flag' column.
    date : str, optional
        If provided, further filter to rows whose 'date' column equals this value.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame of anomalous districts with their scores.
    """
    if df.empty:
        print("[IsolationForest] get_anomalous_districts: received empty DataFrame.")
        return df

    if "anomaly_flag" not in df.columns:
        raise ValueError(
            "[IsolationForest] get_anomalous_districts: "
            "'anomaly_flag' column not found. Run detect_anomalies() first."
        )

    anomalous = df[df["anomaly_flag"] == -1].copy()

    if date is not None:
        if "date" not in anomalous.columns:
            print(
                "[IsolationForest] WARNING – 'date' column not found; "
                "ignoring date filter."
            )
        else:
            anomalous["date"] = pd.to_datetime(anomalous["date"])
            filter_date = pd.to_datetime(date)
            anomalous = anomalous[anomalous["date"] == filter_date]

    print(
        f"[IsolationForest] get_anomalous_districts: "
        f"{len(anomalous)} anomalous record(s) found"
        + (f" on {date}" if date else "") + "."
    )
    return anomalous.reset_index(drop=True)
