"""
DBSCAN-based spatial clustering of anomalous districts for the
Rainfall Anomaly Prediction System (ml_raksha).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import DBSCAN_EPS_KM, DBSCAN_MIN_SAMPLES


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_clustering_features(
    anomalous_df: pd.DataFrame, coords_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge anomalous districts with their coordinates and build the
    normalised feature space used by DBSCAN.

    Parameters
    ----------
    anomalous_df : pd.DataFrame
        Must contain 'district' and 'anomaly_score'.
    coords_df : pd.DataFrame
        Must contain 'district', 'latitude', 'longitude'.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with additional columns:
        latitude, longitude, anomaly_score_normalized.
        Rows whose district has no coordinate entry are dropped with a warning.
    """
    if anomalous_df.empty:
        print("[DBSCAN] prepare_clustering_features: anomalous_df is empty.")
        return anomalous_df

    required_anomalous = {"district", "anomaly_score"}
    missing = required_anomalous - set(anomalous_df.columns)
    if missing:
        raise ValueError(
            f"[DBSCAN] prepare_clustering_features: anomalous_df is missing columns: {missing}"
        )

    required_coords = {"district", "latitude", "longitude"}
    missing_coords = required_coords - set(coords_df.columns)
    if missing_coords:
        raise ValueError(
            f"[DBSCAN] prepare_clustering_features: coords_df is missing columns: {missing_coords}"
        )

    merged = anomalous_df.merge(
        coords_df[["district", "latitude", "longitude"]],
        on="district",
        how="left",
    )

    no_coords_mask = merged["latitude"].isna() | merged["longitude"].isna()
    if no_coords_mask.any():
        missing_districts = merged.loc[no_coords_mask, "district"].unique().tolist()
        print(
            f"[DBSCAN] WARNING – {len(missing_districts)} district(s) have no coordinates "
            f"and will be dropped: {missing_districts}"
        )
        merged = merged[~no_coords_mask].copy()

    if merged.empty:
        print("[DBSCAN] prepare_clustering_features: no rows remain after coordinate merge.")
        return merged

    # Normalise anomaly_score to [0, 1]
    scaler = MinMaxScaler()
    merged["anomaly_score_normalized"] = scaler.fit_transform(
        merged[["anomaly_score"]]
    )

    print(
        f"[DBSCAN] Clustering feature matrix prepared: "
        f"{len(merged)} district-records with [latitude, longitude, anomaly_score_normalized]."
    )
    return merged


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

def cluster_anomalies(
    anomalous_df: pd.DataFrame, coords_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Apply DBSCAN spatial clustering to anomalous districts.

    Uses haversine distance on lat/lon (radians) so that DBSCAN_EPS_KM
    represents a ground-distance threshold in kilometres.

    Parameters
    ----------
    anomalous_df : pd.DataFrame
        Output of get_anomalous_districts() – must have 'district', 'anomaly_score'.
    coords_df : pd.DataFrame
        Must have 'district', 'latitude', 'longitude'.

    Returns
    -------
    pd.DataFrame
        Copy of anomalous_df with two new columns:
        - cluster_id       : int  (-1 = noise, ≥0 = cluster index)
        - is_regional_event: bool (True when cluster_id >= 0)
    """
    if anomalous_df.empty:
        print("[DBSCAN] cluster_anomalies: received empty anomalous_df.")
        return anomalous_df

    featured_df = prepare_clustering_features(anomalous_df, coords_df)

    if featured_df.empty:
        print("[DBSCAN] cluster_anomalies: feature preparation returned empty DataFrame.")
        anomalous_out = anomalous_df.copy()
        anomalous_out["cluster_id"] = -1
        anomalous_out["is_regional_event"] = False
        return anomalous_out

    # Convert degrees to radians for haversine metric
    lat_rad = np.radians(featured_df["latitude"].values)
    lon_rad = np.radians(featured_df["longitude"].values)
    coords_rad = np.column_stack([lat_rad, lon_rad])

    # Convert km threshold to radians (Earth radius ≈ 6371 km)
    eps_rad = DBSCAN_EPS_KM / 6371.0

    print(
        f"[DBSCAN] Running DBSCAN: eps={DBSCAN_EPS_KM} km ({eps_rad:.6f} rad), "
        f"min_samples={DBSCAN_MIN_SAMPLES}, metric=haversine ..."
    )

    db = DBSCAN(
        eps=eps_rad,
        min_samples=DBSCAN_MIN_SAMPLES,
        algorithm="ball_tree",
        metric="haversine",
    )
    labels = db.fit_predict(coords_rad)

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(
        f"[DBSCAN] Found {n_clusters} cluster(s), {n_noise} noise point(s) "
        f"out of {len(labels)} anomalous records."
    )

    # Attach labels back to the featured DataFrame (which may have fewer rows
    # than anomalous_df if some districts had no coordinates)
    featured_df = featured_df.copy()
    featured_df["cluster_id"] = labels
    featured_df["is_regional_event"] = labels >= 0

    # Merge cluster results back into the original anomalous_df
    cluster_cols = featured_df[["district", "cluster_id", "is_regional_event"]].copy()

    # In case of duplicate district rows keep the first assignment
    cluster_cols = cluster_cols.drop_duplicates(subset=["district"], keep="first")

    anomalous_out = anomalous_df.copy()

    # Reset any pre-existing columns to defaults first
    anomalous_out["cluster_id"] = -1
    anomalous_out["is_regional_event"] = False

    # Use index-aligned update via merge
    anomalous_out = anomalous_out.drop(
        columns=["cluster_id", "is_regional_event"], errors="ignore"
    ).merge(cluster_cols, on="district", how="left")

    anomalous_out["cluster_id"] = anomalous_out["cluster_id"].fillna(-1).astype(int)
    anomalous_out["is_regional_event"] = anomalous_out["is_regional_event"].fillna(False)

    return anomalous_out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def get_cluster_summary(clustered_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarise each identified cluster (noise points with cluster_id == -1 are excluded).

    Parameters
    ----------
    clustered_df : pd.DataFrame
        Output of cluster_anomalies() – must have 'cluster_id' and 'anomaly_score'.
        May contain 'district' and 'date'.

    Returns
    -------
    pd.DataFrame
        One row per cluster with columns:
        cluster_id, district_count, mean_anomaly_score,
        and (if 'date' is present) min_date, max_date.
    """
    if clustered_df.empty:
        print("[DBSCAN] get_cluster_summary: received empty DataFrame.")
        return pd.DataFrame()

    if "cluster_id" not in clustered_df.columns:
        raise ValueError(
            "[DBSCAN] get_cluster_summary: 'cluster_id' column not found. "
            "Run cluster_anomalies() first."
        )

    # Exclude noise
    cluster_data = clustered_df[clustered_df["cluster_id"] >= 0].copy()

    if cluster_data.empty:
        print("[DBSCAN] get_cluster_summary: no clusters found (all noise).")
        return pd.DataFrame()

    agg_dict = {"anomaly_score": "mean"}
    if "district" in cluster_data.columns:
        agg_dict["district"] = "count"
    if "date" in cluster_data.columns:
        cluster_data["date"] = pd.to_datetime(cluster_data["date"])
        agg_dict["date"] = ["min", "max"]

    summary = cluster_data.groupby("cluster_id").agg(agg_dict)

    # Flatten multi-level column names
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.reset_index()

    rename_map = {
        "anomaly_score_mean": "mean_anomaly_score",
        "district_count": "district_count",
        "date_min": "min_date",
        "date_max": "max_date",
    }
    summary = summary.rename(columns=rename_map)

    # If 'district' aggregation produced a count column named 'district'
    # rename it properly
    if "district" in summary.columns and pd.api.types.is_numeric_dtype(summary["district"]):
        summary = summary.rename(columns={"district": "district_count"})

    print(
        f"[DBSCAN] Cluster summary: {len(summary)} cluster(s) "
        f"covering {cluster_data.shape[0]} anomalous record(s)."
    )
    return summary
