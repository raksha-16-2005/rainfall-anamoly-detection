"""
Anomaly detection for rainfall data.

Module for detecting point anomalies in district-level rainfall using Isolation Forest
and spatial clustering of anomalies.
"""

import logging
import math
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from geo_utils import DISTRICT_COORDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def detect_point_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect rainfall anomalies per district using Isolation Forest.

    Applies Isolation Forest independently to each district to identify
    point anomalies in the rainfall_avg time series. Districts with
    insufficient data (< 30 rows) are marked as non-anomalous.

    Parameters
    ----------
    df : pd.DataFrame
        Rainfall DataFrame with columns:
        - district: str, district name
        - date: datetime, date of observation
        - rainfall_avg: float, average rainfall value
        - dual_source: bool, whether both data sources contributed
        Other columns are preserved in output.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns:
        - anomaly_flag: bool, True if detected as anomaly, False otherwise
        - anomaly_score: float, Isolation Forest decision function output
                         (negative = anomaly, positive = normal)
                         For districts with < 30 rows: 0.0
        Index is reset to 0, 1, 2, ...

    Notes
    -----
    - Isolation Forest contamination set to 0.05 (expect ~5% anomalies)
    - random_state=42 ensures reproducible results
    - Fits independently per district (no cross-contamination)
    - Minimum 30 rows per district required; otherwise all flagged as normal
    - anomaly_score of 0.0 indicates insufficient data for that district
    - Negative anomaly_score values indicate anomalies (follow Isolation Forest convention)

    Raises
    ------
    ValueError
        If required columns (district, rainfall_avg) are missing.
    Exception
        If unable to fit Isolation Forest model for any district.

    Examples
    --------
    >>> from anomaly_detector import detect_point_anomalies
    >>> from data_merger import merge_rainfall_sources
    >>> 
    >>> merged_df = merge_rainfall_sources(gov_df, meteo_df)
    >>> anomalies_df = detect_point_anomalies(merged_df)
    >>> print(anomalies_df.head())
    >>> 
    >>> # Filter to anomalies only
    >>> anomaly_records = anomalies_df[anomalies_df['anomaly_flag']]
    >>> print(f"Found {len(anomaly_records)} anomalous records")
    """
    try:
        # Validate required columns
        required_cols = ["district", "rainfall_avg"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(
            f"Starting anomaly detection on {len(df)} records from {df['district'].nunique()} districts"
        )

        # Initialize result columns
        df = df.copy()
        df["anomaly_flag"] = False
        df["anomaly_score"] = 0.0

        min_rows = 30

        # Process each district independently
        for district in df["district"].unique():
            district_mask = df["district"] == district
            district_data = df.loc[district_mask, "rainfall_avg"].values.reshape(-1, 1)
            district_size = len(district_data)

            if district_size < min_rows:
                logger.warning(
                    f"District '{district}': Only {district_size} rows (< {min_rows} minimum). "
                    f"Skipping anomaly detection, flagging as normal."
                )
                # Keep default False/0.0 for this district
                continue

            try:
                # Fit Isolation Forest on this district
                iso_forest = IsolationForest(
                    contamination=0.05,
                    random_state=42,
                    n_estimators=100,
                )
                iso_forest.fit(district_data)

                # Get predictions and scores
                predictions = iso_forest.predict(district_data)  # -1 for anomaly, 1 for normal
                scores = iso_forest.score_samples(district_data)  # Negative = anomaly

                # Update results for this district
                df.loc[district_mask, "anomaly_flag"] = predictions == -1
                df.loc[district_mask, "anomaly_score"] = scores

                anomaly_count = (predictions == -1).sum()
                logger.info(
                    f"District '{district}': Fitted Isolation Forest on {district_size} rows, "
                    f"detected {anomaly_count} anomalies ({100 * anomaly_count / district_size:.1f}%)"
                )

            except Exception as e:
                logger.error(
                    f"Error fitting Isolation Forest for district '{district}': {e}"
                )
                raise

        logger.info(f"Anomaly detection complete: {df['anomaly_flag'].sum()} total anomalies detected")

        # Reset index
        df = df.reset_index(drop=True)

        return df

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during anomaly detection: {type(e).__name__}: {e}")
        raise


def detect_spatial_clusters(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """
    Detect spatial clusters of rainfall anomalies on a given date.

    Filters to anomalous records on the specified date and runs DBSCAN on
    geographic coordinates to identify regional clusters. Uses the Haversine
    metric to respect Earth's curvature.

    Parameters
    ----------
    df : pd.DataFrame
        Rainfall DataFrame with anomaly detection results from detect_point_anomalies().
        Must have columns:
        - district: str, district name (must exist in DISTRICT_COORDS)
        - date: datetime, observation date
        - anomaly_flag: bool, True if point anomaly detected
    date : str
        Target date as string (e.g., "2026-04-07"). Will be converted to datetime
        and filtered using exact date match (ignoring time).

    Returns
    -------
    pd.DataFrame
        DataFrame with all records from the specified date, including columns:
        - district: str
        - date: datetime
        - spatial_cluster_id: int, DBSCAN cluster ID
                             (-1 = noise, 0+ = cluster ID)
        - in_regional_cluster: bool, True if cluster_id >= 0 (not noise)
        All other input columns preserved.
        Index is reset to 0, 1, 2, ...

    Notes
    -----
    - DBSCAN parameters: eps=1.5 (km), min_samples=3, metric='haversine'
    - Coordinates converted to radians for Haversine distance calculation
    - Only anomalous records (anomaly_flag=True) used for clustering
    - Non-anomalous records on that date get spatial_cluster_id=-1, in_regional_cluster=False
    - If no anomalies on that date, returns empty DataFrame with columns
    - District coordinates retrieved from geo_utils.DISTRICT_COORDS

    Raises
    ------
    ValueError
        If required columns (district, date, anomaly_flag) are missing.
    KeyError
        If a district is not found in DISTRICT_COORDS.
    Exception
        If unable to run DBSCAN clustering.

    Examples
    --------
    >>> from anomaly_detector import detect_point_anomalies, detect_spatial_clusters
    >>> 
    >>> anomalies_df = detect_point_anomalies(merged_df)
    >>> spatial_clusters = detect_spatial_clusters(anomalies_df, "2026-04-07")
    >>> 
    >>> # Filter to clustered anomalies only
    >>> clustered = spatial_clusters[
    ...     (spatial_clusters['anomaly_flag']) & 
    ...     (spatial_clusters['in_regional_cluster'])
    ... ]
    >>> print(f"Found {len(clustered['spatial_cluster_id'].unique())} regional clusters")
    """
    try:
        # Validate required columns
        required_cols = ["district", "date", "anomaly_flag"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert date string to datetime for filtering
        target_date = pd.to_datetime(date).date()
        df_date = df.copy()
        df_date["date_only"] = df_date["date"].dt.date

        # Filter to target date
        df_target = df_date[df_date["date_only"] == target_date].copy()

        if len(df_target) == 0:
            logger.warning(f"No records found for date {date}. Returning empty DataFrame.")
            return pd.DataFrame(columns=["district", "date", "spatial_cluster_id", "in_regional_cluster"])

        logger.info(f"Processing {len(df_target)} records for date {date}")

        # Initialize cluster columns
        df_target["spatial_cluster_id"] = -1
        df_target["in_regional_cluster"] = False

        # Filter to anomalous records
        anomalies_target = df_target[df_target["anomaly_flag"]].copy()

        if len(anomalies_target) == 0:
            logger.info(f"No anomalies detected on {date}. All records marked as non-clustered.")
        else:
            logger.info(f"Found {len(anomalies_target)} anomalous records on {date}")

            try:
                # Extract coordinates for anomalous districts
                anomaly_districts = anomalies_target["district"].values
                coords = []
                district_mapping = {}

                for i, district in enumerate(anomaly_districts):
                    if district not in DISTRICT_COORDS:
                        raise KeyError(f"District '{district}' not found in DISTRICT_COORDS")

                    lat, lon = DISTRICT_COORDS[district]
                    # Convert to radians for Haversine metric
                    lat_rad = math.radians(lat)
                    lon_rad = math.radians(lon)
                    coords.append([lat_rad, lon_rad])
                    district_mapping[i] = district

                coords = np.array(coords)

                # Run DBSCAN with Haversine metric
                # eps=1.5 km (convert to radians: 1.5 km / earth_radius_km(6371) ≈ 0.000235 rad)
                db = DBSCAN(
                    eps=1.5 / 6371,  # Convert km to radians (Earth radius ≈ 6371 km)
                    min_samples=3,
                    metric="haversine"
                )
                cluster_labels = db.fit_predict(coords)

                logger.info(
                    f"DBSCAN detected {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters, "
                    f"{(cluster_labels == -1).sum()} noise points"
                )

                # Map cluster labels back to anomalies
                for idx, (anomaly_idx, cluster_id) in enumerate(zip(
                    anomalies_target.index.values, cluster_labels
                )):
                    df_target.loc[anomaly_idx, "spatial_cluster_id"] = cluster_id
                    df_target.loc[anomaly_idx, "in_regional_cluster"] = cluster_id >= 0

            except Exception as e:
                logger.error(f"Error running DBSCAN clustering: {e}")
                raise

        # Clean up temporary column and reset index
        df_target = df_target.drop(columns=["date_only"])
        df_target = df_target.reset_index(drop=True)

        logger.info(
            f"Spatial clustering complete: {df_target['in_regional_cluster'].sum()} records in clusters"
        )

        return df_target

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except KeyError as e:
        logger.error(f"Coordinate lookup error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during spatial clustering: {type(e).__name__}: {e}")
        raise
