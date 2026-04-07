"""
Rolling Z-score anomaly detection for rainfall data.

Module for computing rolling Z-scores to detect temporal rainfall anomalies.
"""

import logging
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def compute_rolling_zscore(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Compute rolling Z-score per district for rainfall anomaly detection.

    Calculates the Z-score of each rainfall observation relative to the rolling
    mean and standard deviation within each district. This detects temporal
    anomalies where rainfall deviates from recent patterns.

    Parameters
    ----------
    df : pd.DataFrame
        Rainfall DataFrame with columns:
        - district: str, district name (or identifier)
        - date: datetime, observation date
        - rainfall_avg: float, rainfall value in mm (or normalized)
        Expected to be sorted by district, then date.
    window : int, optional
        Rolling window size in days (default: 30).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with two new columns:
        - rolling_zscore: float, Z-score relative to rolling mean/std
                         Set to 0.0 if rolling std is zero or insufficient data
        - zscore_category: str, categorization of Z-score:
                          'normal' if z < 2
                          'moderate' if 2 <= z < 3
                          'extreme' if z >= 3
        All other columns preserved. Index is reset to 0, 1, 2, ...

    Notes
    -----
    - Computes rolling window independently per district (groupby + transform)
    - min_periods=7 ensures rolling calculations begin after 7 observations per district
    - Z-score formula: z = (rainfall_avg - rolling_mean) / rolling_std
    - When rolling_std = 0 (constant rainfall) or insufficient data, z is set to 0.0
    - Input DataFrame is not modified; a copy is returned
    - Categories: normal (z < 2), moderate (2 <= z < 3), extreme (z >= 3)

    Raises
    ------
    ValueError
        If required columns (district, rainfall_avg) are missing.
    Exception
        If unable to compute rolling statistics.

    Examples
    --------
    >>> from zscore import compute_rolling_zscore
    >>> from data_merger import merge_rainfall_sources
    >>> 
    >>> merged_df = merge_rainfall_sources(gov_df, meteo_df)
    >>> zscore_df = compute_rolling_zscore(merged_df, window=30)
    >>> 
    >>> # Find extreme anomalies
    >>> extremes = zscore_df[zscore_df['zscore_category'] == 'extreme']
    >>> print(f"Found {len(extremes)} extreme rainfall anomalies")
    >>> 
    >>> # Group by category
    >>> print(zscore_df['zscore_category'].value_counts())
    """
    try:
        # Validate required columns
        required_cols = ["district", "rainfall_avg"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Computing rolling Z-score with window={window} days on {len(df)} records")

        # Create a copy to avoid modifying original
        result = df.copy()

        # Compute rolling mean per district
        result["rolling_mean"] = result.groupby("district")["rainfall_avg"].transform(
            lambda x: x.rolling(window=window, min_periods=7).mean()
        )

        # Compute rolling std per district
        result["rolling_std"] = result.groupby("district")["rainfall_avg"].transform(
            lambda x: x.rolling(window=window, min_periods=7).std()
        )

        # Compute Z-score, handling division by zero and NaN values
        # Where std is 0, NaN, or mean is NaN (insufficient data), set zscore to 0.0
        result["rolling_zscore"] = np.where(
            (result["rolling_std"] == 0) | result["rolling_std"].isna() | result["rolling_mean"].isna(),
            0.0,
            (result["rainfall_avg"] - result["rolling_mean"]) / result["rolling_std"]
        )

        # Categorize Z-score
        def categorize_zscore(z):
            if z < 2:
                return "normal"
            elif 2 <= z < 3:
                return "moderate"
            else:  # z >= 3
                return "extreme"

        result["zscore_category"] = result["rolling_zscore"].apply(categorize_zscore)

        # Drop temporary columns
        result = result.drop(columns=["rolling_mean", "rolling_std"])

        # Reset index
        result = result.reset_index(drop=True)

        # Log summary statistics
        category_counts = result["zscore_category"].value_counts().to_dict()
        logger.info(
            f"Z-score computed: {category_counts.get('normal', 0)} normal, "
            f"{category_counts.get('moderate', 0)} moderate, "
            f"{category_counts.get('extreme', 0)} extreme"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error computing rolling Z-score: {type(e).__name__}: {e}")
        raise
