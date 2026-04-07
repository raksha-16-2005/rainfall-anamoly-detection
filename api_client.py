"""
API client for Open-Meteo weather data.

Module for fetching historical precipitation data from Open-Meteo archive API.
"""

import pandas as pd
import requests
import time


def fetch_openmeteo_rainfall(
    lat: float, lon: float, start_date: str, end_date: str
) -> pd.DataFrame:
    """
    Fetch historical daily precipitation data from Open-Meteo API.

    Retrieves daily precipitation_sum for a given latitude/longitude and date range
    using the Open-Meteo archive API. Includes automatic retry logic with exponential
    backoff for robust error handling.

    Parameters
    ----------
    lat : float
        Latitude of the location (-90 to 90).
    lon : float
        Longitude of the location (-180 to 180).
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str
        End date in YYYY-MM-DD format.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Date: datetime type (daily)
        - Rainfall_mm: float, precipitation in millimeters

    Raises
    ------
    ValueError
        If the API request fails after 3 retries or if the response is invalid.
    
    Notes
    -----
    - No API key is required for Open-Meteo
    - Uses 3 retries with exponential backoff (1s, 2s, 4s) for robustness
    - Automatically handles date parsing and removes any null precipitation values
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "auto",
    }

    max_retries = 3
    retry_delay = 1  # Start with 1 second delay

    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Extract daily data from API response
            daily_data = data.get("daily", {})
            dates = daily_data.get("time", [])
            precipitation = daily_data.get("precipitation_sum", [])

            if not dates or not precipitation:
                raise ValueError("API response missing time or precipitation_sum data")

            # Create DataFrame
            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(dates),
                    "Rainfall_mm": precipitation,
                }
            )

            # Remove rows with null rainfall values
            df = df.dropna(subset=["Rainfall_mm"])

            # Sort by Date
            df = df.sort_values(by="Date").reset_index(drop=True)

            return df

        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(
                    f"Attempt {attempt + 1}/{max_retries} failed: {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise ValueError(
                    f"Failed to fetch data after {max_retries} attempts: {e}"
                ) from e
        except (KeyError, ValueError) as e:
            raise ValueError(f"Invalid API response format: {e}") from e
