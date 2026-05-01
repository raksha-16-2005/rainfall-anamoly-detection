"""
Open-Meteo API module for fetching real-time and historical precipitation data.

Handles district-level rainfall data ingestion from the Open-Meteo API
for the Rainfall Anomaly Prediction System.
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# Import constants from config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import (
    OPENMETEO_BASE_URL,
    OPENMETEO_FORECAST_URL,
    API_TIMEOUT,
    DISTRICT_COORDS_FILE,
    DATA_PROCESSED_DIR,
)

# Setup logging
logger = logging.getLogger(__name__)


def load_district_coords() -> pd.DataFrame:
    """
    Load district coordinates from CSV file.

    Returns:
        pd.DataFrame: DataFrame with columns [district, state, latitude, longitude]
    """
    try:
        df = pd.read_csv(DISTRICT_COORDS_FILE)
        logger.info(f"Loaded {len(df)} districts from {DISTRICT_COORDS_FILE}")
        return df
    except FileNotFoundError:
        logger.error(f"District coordinates file not found: {DISTRICT_COORDS_FILE}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error loading district coordinates: {e}")
        return pd.DataFrame()


def fetch_district_rainfall(
    district: str, lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch precipitation data for a district from Open-Meteo Archive API.

    Args:
        district (str): District name
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format

    Returns:
        pd.DataFrame: DataFrame with columns [district, date, rainfall_mm]
                     Returns empty DataFrame on error
    """
    print(f"Fetching {district}: {start_date} to {end_date}")

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "Asia/Kolkata",
    }

    try:
        response = requests.get(
            OPENMETEO_BASE_URL, params=params, timeout=API_TIMEOUT
        )
        response.raise_for_status()
        data = response.json()

        # Extract daily data
        daily_data = data.get("daily", {})
        times = daily_data.get("time", [])
        precipitation = daily_data.get("precipitation_sum", [])

        # Replace None with NaN
        precipitation = [p if p is not None else float('nan') for p in precipitation]

        # Create DataFrame
        df = pd.DataFrame({
            "district": district,
            "date": pd.to_datetime(times),
            "rainfall_mm": precipitation,
        })

        logger.info(f"Successfully fetched {len(df)} records for {district}")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"API error fetching {district}: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Error processing data for {district}: {e}")
        return pd.DataFrame()


def fetch_all_districts(
    start_date: str, end_date: str, use_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch precipitation data for all districts with caching support.

    Args:
        start_date (str): Start date in YYYY-MM-DD format
        end_date (str): End date in YYYY-MM-DD format
        use_cache (bool): If True, use cached files when available

    Returns:
        pd.DataFrame: Consolidated DataFrame with all districts, sorted by (district, date)
                     Includes 'source' column = 'openmeteo'
    """
    # Load district coordinates
    coords_df = load_district_coords()
    if coords_df.empty:
        logger.error("No district coordinates available")
        return pd.DataFrame()

    # Ensure processed data directory exists
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []
    districts_fetched = []

    for _, row in coords_df.iterrows():
        district = row["district"]
        lat = row["latitude"]
        lon = row["longitude"]

        # Check cache
        cache_file = DATA_PROCESSED_DIR / f"openmeteo_{district}_{start_date}_{end_date}.csv"

        if use_cache and cache_file.exists():
            logger.info(f"Loading {district} from cache")
            try:
                df = pd.read_csv(cache_file)
                df["date"] = pd.to_datetime(df["date"])
                all_data.append(df)
                continue
            except Exception as e:
                logger.warning(f"Failed to load cache for {district}: {e}")

        # Fetch from API
        df = fetch_district_rainfall(district, lat, lon, start_date, end_date)
        if not df.empty:
            # Save to cache
            try:
                df.to_csv(cache_file, index=False)
                logger.info(f"Cached data for {district}")
            except Exception as e:
                logger.warning(f"Failed to cache {district}: {e}")

            all_data.append(df)
            districts_fetched.append(district)

        # Be polite to API - small delay between requests
        time.sleep(0.1)

    # Concatenate all data
    if not all_data:
        logger.warning("No data fetched for any district")
        return pd.DataFrame()

    result = pd.concat(all_data, ignore_index=True)

    # Add source column
    result["source"] = "openmeteo"

    # Sort by district and date
    result = result.sort_values(by=["district", "date"]).reset_index(drop=True)

    # Print summary
    print(f"\n=== Summary ===")
    print(f"Total rows: {len(result)}")
    print(f"Districts fetched: {len(districts_fetched)}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"===============\n")

    logger.info(
        f"Fetched {len(result)} total rows from {len(districts_fetched)} districts"
    )
    return result


def fetch_forecast(districts: list = None) -> pd.DataFrame:
    """
    Fetch 7-day precipitation forecast for specified districts.

    Args:
        districts (list, optional): List of district names. If None, fetch for all districts.

    Returns:
        pd.DataFrame: DataFrame with columns [district, date, rainfall_mm, source]
                     source = 'openmeteo_forecast'
    """
    # Load district coordinates if needed
    if districts is None:
        coords_df = load_district_coords()
        districts = coords_df["district"].tolist()
    else:
        coords_df = load_district_coords()

    all_forecast_data = []

    for district in districts:
        # Get coordinates
        district_row = coords_df[coords_df["district"] == district]
        if district_row.empty:
            logger.warning(f"District not found: {district}")
            continue

        lat = district_row.iloc[0]["latitude"]
        lon = district_row.iloc[0]["longitude"]

        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
            "forecast_days": 7,
        }

        try:
            print(f"Fetching forecast for {district}")
            response = requests.get(
                OPENMETEO_FORECAST_URL, params=params, timeout=API_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()

            # Extract daily forecast data
            daily_data = data.get("daily", {})
            times = daily_data.get("time", [])
            precipitation = daily_data.get("precipitation_sum", [])

            # Replace None with NaN
            precipitation = [p if p is not None else float('nan') for p in precipitation]

            # Create DataFrame
            df = pd.DataFrame({
                "district": district,
                "date": pd.to_datetime(times),
                "rainfall_mm": precipitation,
                "source": "openmeteo_forecast",
            })

            all_forecast_data.append(df)
            logger.info(f"Successfully fetched forecast for {district}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                # Rate limited - wait and retry
                logger.warning(f"Rate limited for {district}, waiting 5 seconds...")
                time.sleep(5)
                try:
                    response = requests.get(
                        OPENMETEO_FORECAST_URL, params=params, timeout=API_TIMEOUT
                    )
                    response.raise_for_status()
                    data = response.json()

                    daily_data = data.get("daily", {})
                    times = daily_data.get("time", [])
                    precipitation = daily_data.get("precipitation_sum", [])

                    precipitation = [
                        p if p is not None else float('nan') for p in precipitation
                    ]

                    df = pd.DataFrame({
                        "district": district,
                        "date": pd.to_datetime(times),
                        "rainfall_mm": precipitation,
                        "source": "openmeteo_forecast",
                    })

                    all_forecast_data.append(df)
                    logger.info(f"Successfully fetched forecast for {district} on retry")
                except Exception as retry_error:
                    logger.error(f"Retry failed for {district}: {retry_error}")
            else:
                logger.error(f"HTTP error fetching forecast for {district}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"API error fetching forecast for {district}: {e}")
        except Exception as e:
            logger.error(f"Error processing forecast for {district}: {e}")

        # Be polite to API
        time.sleep(0.1)

    # Concatenate all forecast data
    if not all_forecast_data:
        logger.warning("No forecast data fetched for any district")
        return pd.DataFrame()

    result = pd.concat(all_forecast_data, ignore_index=True)
    result = result.sort_values(by=["district", "date"]).reset_index(drop=True)

    logger.info(f"Fetched forecast data for {len(all_forecast_data)} districts")
    return result


def get_recent_data(days_back: int = 90) -> pd.DataFrame:
    """
    Convenience function to fetch recent precipitation data.

    Args:
        days_back (int): Number of days to look back from today (default: 90)

    Returns:
        pd.DataFrame: DataFrame with recent precipitation data
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    logger.info(f"Fetching recent data: {start_date} to {end_date}")
    return fetch_all_districts(start_date, end_date, use_cache=True)
