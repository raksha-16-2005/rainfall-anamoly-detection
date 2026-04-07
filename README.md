# Rainfall Anomaly Detection

A Python pipeline for detecting rainfall anomalies in Karnataka using government data and satellite weather data sources.

## Project Structure

```
rainfall-anomaly-detection/
│
├── data/
│   └── rainfall_data.csv              # CSV file from data.gov.in (not included)
│
├── data_loader.py                     # Load and preprocess government rainfall data
├── api_client.py                      # Fetch data from Open-Meteo weather API
├── geo_utils.py                       # District coordinates and fuzzy matching
├── data_merger.py                     # Merge multiple data sources
├── preprocessor.py                    # Normalize and prepare data for Prophet
│
├── models/                            # Directory for saved Prophet models
├── main.py                            # Orchestrates the complete pipeline
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Download rainfall data from [data.gov.in](https://data.gov.in) and place it as `data/rainfall_data.csv`.

Expected CSV columns:
- `District`: District name
- `Date`: Date in YYYY-MM-DD format
- `Rainfall_mm`: Rainfall in millimeters
- `Normal_mm`: Normal rainfall
- `Departure_pct`: Departure from normal

### 3. Run the Pipeline

```bash
python main.py
```

## Module Details

### `data_loader.py`
- `load_district_rainfall(filepath)` → Loads and cleans government rainfall data
  - Parses dates to datetime
  - Normalizes district names (lowercase, stripped)
  - Handles missing values (forward-fill + interpolation)

### `api_client.py`
- `fetch_openmeteo_rainfall(lat, lon, start_date, end_date)` → Fetches satellite data
  - Available at: https://archive-api.open-meteo.com/v1/archive
  - Includes retry logic with exponential backoff
  - Returns daily precipitation_sum

### `geo_utils.py`
- `DISTRICT_COORDS`: Dictionary mapping Karnataka districts to (lat, lon)
- `get_coords(district_name)` → Fuzzy-matches district name and returns coordinates
  - Handles typos and alternate names

### `data_merger.py`
- `merge_rainfall_sources(gov_df, meteo_df)` → Combines government and weather data
  - Outer join on District and Date
  - Creates `rainfall_avg` (mean when both available)
  - Tracks data source with `dual_source` boolean

### `preprocessor.py`
- `preprocess_for_model(df)` → Prepares data for Prophet forecasting
  - Min-max normalization (0-1 range)
  - Creates Prophet-ready format: `ds` (datetime), `y` (normalized), `y_raw` (mm)
  - Stores `DISTRICT_SCALERS` for inverse transformation

## Usage Example

```python
from data_loader import load_district_rainfall
from api_client import fetch_openmeteo_rainfall
from geo_utils import get_coords
from data_merger import merge_rainfall_sources
from preprocessor import preprocess_for_model, DISTRICT_SCALERS

# Load government data
gov_df = load_district_rainfall("data/rainfall_data.csv")

# Fetch OpenMeteo for a district
lat, lon = get_coords("bengaluru urban")
meteo_df = fetch_openmeteo_rainfall(lat, lon, "2023-01-01", "2024-12-31")
meteo_df["District"] = "bengaluru urban"

# Merge sources
merged_df = merge_rainfall_sources(gov_df, meteo_df)

# Prepare for Prophet
prophet_dfs = preprocess_for_model(merged_df)
bengaluru_df = prophet_dfs["bengaluru urban"]

# Fit Prophet model
from prophet import Prophet
model = Prophet()
model.fit(bengaluru_df)

# Make forecast
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

# Inverse-transform predictions back to mm
import numpy as np
scaler = DISTRICT_SCALERS["bengaluru urban"]
forecast["yhat_mm"] = scaler.inverse_transform(forecast[["yhat"]])
```

## Data Sources

- **Government**: [data.gov.in](https://data.gov.in) - Ministry of Earth Sciences
- **Weather**: [Open-Meteo](https://open-meteo.com) - Free, no API key required

## Dependencies

- `pandas` - Data manipulation
- `requests` - API calls
- `scikit-learn` - Data normalization
- `prophet` - Time series forecasting
- `numpy` - Numerical operations

## Notes

- All district names are normalized to lowercase for consistent merging
- Missing rainfall values are filled using forward-fill then interpolation
- Min-max scaling preserves the scaler for inverse transformation of forecasts
- Open-Meteo API has a 3-retry mechanism with exponential backoff
