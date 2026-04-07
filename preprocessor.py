"""
Data preprocessor for machine learning models.

Module for normalizing rainfall data and preparing for Prophet forecasting.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# Global storage for MinMaxScaler objects (for inverse transformation)
DISTRICT_SCALERS = {}


def preprocess_for_model(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Preprocess merged rainfall data for Prophet forecasting model.

    Groups rainfall data by district and applies min-max normalization to
    rainfall values. Creates Prophet-ready DataFrames with standardized column
    names (ds, y). Stores scalers for later inverse transformation of predictions.

    Parameters
    ----------
    df : pd.DataFrame
        Merged rainfall DataFrame with columns: District, Date, rainfall_avg.
        Typically the output from merge_rainfall_sources().

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of district name -> preprocessed DataFrame with columns:
        - ds: datetime (Prophet requirement)
        - y: float, min-max normalized rainfall_avg (0-1 range)
        - y_raw: float, original rainfall_avg value in mm
        All DataFrames sorted by ds (datetime).

    Notes
    -----
    - MinMaxScaler objects are stored in global DISTRICT_SCALERS dictionary
      for later inverse transformation
    - NaN values in rainfall_avg are dropped per district
    - Output format is ready for use with Facebook Prophet:
      ``Prophet().fit(prophet_df)``
    - To convert predictions back to mm:
      ``DISTRICT_SCALERS[district_name].inverse_transform(y_pred.reshape(-1, 1))``

    Examples
    --------
    >>> merged_df = merge_rainfall_sources(gov_df, meteo_df)
    >>> prophet_dfs = preprocess_for_model(merged_df)
    >>> bengaluru_df = prophet_dfs['bengaluru urban']
    >>> model = Prophet().fit(bengaluru_df)
    >>> forecast = model.make_future_dataframe(periods=30)
    >>> forecast = model.predict(forecast)
    """
    global DISTRICT_SCALERS
    DISTRICT_SCALERS = {}

    result = {}

    for district, group in df.groupby("District"):
        # Extract relevant columns and drop missing rainfall values
        district_df = group[["Date", "rainfall_avg"]].copy()
        district_df = district_df.dropna(subset=["rainfall_avg"])

        if len(district_df) == 0:
            # Skip districts with no valid data
            continue

        # Rename columns to Prophet format (ds=datetime, y=target)
        district_df.rename(columns={"Date": "ds", "rainfall_avg": "y_raw"}, inplace=True)

        # Apply min-max scaling (0-1 normalization)
        scaler = MinMaxScaler(feature_range=(0, 1))
        y_normalized = scaler.fit_transform(district_df[["y_raw"]])
        district_df["y"] = y_normalized.flatten()

        # Store scaler for inverse transformation of predictions
        DISTRICT_SCALERS[district] = scaler

        # Sort by date and reset index
        district_df = district_df.sort_values("ds").reset_index(drop=True)

        result[district] = district_df

    return result
