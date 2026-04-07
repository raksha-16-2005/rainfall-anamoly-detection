"""
Prophet-based forecaster for district-level rainfall predictions.

Module for forecasting rainfall using Facebook Prophet time series model.
"""

import logging
import pandas as pd
from contextlib import redirect_stdout
from io import StringIO
from prophet import Prophet
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def forecast_district(
    district_name: str, df: pd.DataFrame, forecast_days: int = 7
) -> pd.DataFrame:
    """
    Forecast daily rainfall for a district using Facebook Prophet.

    Uses Facebook Prophet to fit a time series model on historical rainfall data
    and generate probabilistic forecasts with confidence intervals. Explicitly
    configures yearly and weekly seasonality to capture rainfall patterns.

    Parameters
    ----------
    district_name : str
        Name of the district (for labeling output).
    df : pd.DataFrame
        Historical rainfall DataFrame with columns:
        - ds: datetime, daily timestamps
        - y: float, normalized rainfall values (0-1 range)
        Expected to have at least 60 data points for reliable forecasting.
    forecast_days : int, optional
        Number of days to forecast into the future (default: 7).

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns:
        - ds: datetime, forecast dates
        - yhat: float, point forecast (mean)
        - yhat_lower: float, lower confidence bound (95%)
        - yhat_upper: float, upper confidence bound (95%)
        - district: str, district name
        
        If input has fewer than 60 rows, returns an empty DataFrame.

    Warnings
    --------
    - Issues warning log if df has fewer than 60 rows (insufficient data)
    - Suppresses Prophet's verbose INFO/DEBUG output for cleaner logs

    Raises
    ------
    ValueError
        If required columns (ds, y) are missing from input DataFrame.
    Exception
        Logs any fitting or prediction errors and re-raises with context.

    Notes
    -----
    - Minimum 60 rows recommended for reliable Prophet forecasts
    - Yearly and weekly seasonalities are explicitly enabled
    - Forecast intervals are 95% confidence (0.95 scale)
    - Output values remain in normalized 0-1 range; use preprocessor.DISTRICT_SCALERS
      for denormalization to actual mm values

    Examples
    --------
    >>> import pandas as pd
    >>> from forecaster import forecast_district
    >>> from preprocessor import preprocess_for_model
    >>> 
    >>> # Assume merged_df is the output from merge_rainfall_sources()
    >>> prophet_dfs = preprocess_for_model(merged_df)
    >>> bengaluru_df = prophet_dfs['bengaluru urban']
    >>> 
    >>> forecast = forecast_district('bengaluru urban', bengaluru_df, forecast_days=14)
    >>> print(forecast.head())
    """
    try:
        # Validate required columns
        required_cols = ["ds", "y"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check minimum data requirement
        min_rows = 60
        if len(df) < min_rows:
            logger.warning(
                f"District '{district_name}': Only {len(df)} rows available. "
                f"Minimum {min_rows} rows recommended for reliable forecasting. "
                f"Returning empty forecast."
            )
            return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "district"])

        logger.info(f"Starting forecast for district: {district_name} ({len(df)} historical records)")

        # Fit Prophet model with suppressed verbose output
        # Redirect stdout to suppress Prophet's INFO/DEBUG logs
        buffer = StringIO()
        with redirect_stdout(buffer):
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,  # Daily seasonality not meaningful for rainfall
                interval_width=0.95,  # 95% confidence intervals
                changepoint_prior_scale=0.05,  # Moderate flexibility for trend changes
            )
            model.fit(df)

        logger.info(f"Prophet model fitted successfully for {district_name}")

        # Generate future dates
        future = model.make_future_dataframe(periods=forecast_days)

        # Generate predictions
        forecast = model.predict(future)

        # Extract only forecast period (drop historical)
        forecast_only = forecast.iloc[-forecast_days:].copy()

        # Select required columns and add district
        result = forecast_only[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        result["district"] = district_name

        logger.info(
            f"Forecast generated for {district_name}: {forecast_days} days, "
            f"mean yhat range: [{result['yhat'].min():.4f}, {result['yhat'].max():.4f}]"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error for {district_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error forecasting {district_name}: {type(e).__name__}: {e}")
        raise


def run_all_forecasts(
    district_data: dict[str, pd.DataFrame], forecast_days: int = 7
) -> pd.DataFrame:
    """
    Forecast rainfall for all districts and combine results.

    Iterates over all districts in the input dictionary, generates forecasts using
    forecast_district(), and concatenates results into a single DataFrame. Gracefully
    handles failures in individual districts while logging errors.

    Parameters
    ----------
    district_data : dict[str, pd.DataFrame]
        Dictionary mapping district names to DataFrames with columns ds, y.
        Typically the output from preprocess_for_model().
    forecast_days : int, optional
        Number of days to forecast for each district (default: 7).

    Returns
    -------
    pd.DataFrame
        Combined forecast results with columns:
        - ds: datetime, forecast dates
        - yhat: float, point forecast (mean)
        - yhat_lower: float, lower confidence bound (95%)
        - yhat_upper: float, upper confidence bound (95%)
        - district: str, district name
        Index is reset to 0, 1, 2, ...
        
        Returns empty DataFrame (with same columns) if all districts fail.

    Notes
    -----
    - Skips districts that fail during forecasting (logs error, continues)
    - Progress bar shows district names being processed
    - Failed districts are recorded in error logs but do not interrupt processing
    - If some districts succeed and others fail, returns forecasts for successful ones

    Examples
    --------
    >>> from preprocessor import preprocess_for_model
    >>> from forecaster import run_all_forecasts
    >>> 
    >>> prophet_dfs = preprocess_for_model(merged_df)
    >>> all_forecasts = run_all_forecasts(prophet_dfs, forecast_days=14)
    >>> print(f"Forecasts generated for {len(all_forecasts['district'].unique())} districts")
    >>> print(all_forecasts.head(10))
    """
    forecasts = []
    failed_districts = []

    logger.info(f"Starting forecasts for {len(district_data)} districts (forecast_days={forecast_days})")

    # Iterate with progress bar
    for district_name in tqdm(district_data.keys(), desc="Forecasting districts"):
        try:
            df = district_data[district_name]
            forecast = forecast_district(district_name, df, forecast_days=forecast_days)

            # Only add non-empty forecasts
            if len(forecast) > 0:
                forecasts.append(forecast)
            else:
                # Insufficient data case
                logger.warning(f"Forecast for {district_name} returned empty (likely insufficient data)")
                failed_districts.append(district_name)

        except Exception as e:
            logger.error(f"Failed to forecast {district_name}: {e}")
            failed_districts.append(district_name)

    # Combine results
    if forecasts:
        result = pd.concat(forecasts, ignore_index=True)
        logger.info(
            f"Forecasts completed: {len(result['district'].unique())} districts successful, "
            f"{len(failed_districts)} failed"
        )
        return result
    else:
        logger.warning(f"All {len(district_data)} districts failed. Returning empty forecast DataFrame.")
        return pd.DataFrame(columns=["ds", "yhat", "yhat_lower", "yhat_upper", "district"])
