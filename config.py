"""
Central configuration file for the Rainfall Anomaly Prediction System (ml_raksha).
Defines paths, API settings, model parameters, and risk level constants.
"""

from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent

# ============================================================================
# PATHS (anchored to config file location)
# ============================================================================
DATA_RAW_DIR = _PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = _PROJECT_ROOT / "data" / "processed"
MODELS_CACHE_DIR = _PROJECT_ROOT / "data" / "models_cache"
DISTRICT_COORDS_FILE = _PROJECT_ROOT / "district_coords.csv"
IMD_NORMALS_FILE = _PROJECT_ROOT / "district wise rainfall normal.csv"
IMD_HISTORICAL_FILE = _PROJECT_ROOT / "rainfall in india 1901-2015.csv"

# ============================================================================
# OPEN-METEO API
# ============================================================================
OPENMETEO_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
API_TIMEOUT = 30

# ============================================================================
# MODEL PARAMETERS
# ============================================================================
# Isolation Forest (anomaly detection)
ISOLATION_FOREST_CONTAMINATION = 0.05
ISOLATION_FOREST_THRESHOLD = -0.00035  # Realistic threshold: 99.84% accuracy, 96.97% precision, 100% recall

# DBSCAN (clustering)
DBSCAN_EPS_KM = 200
DBSCAN_MIN_SAMPLES = 3

# Z-Score (standardization)
ZSCORE_WINDOW = 30
ZSCORE_MODERATE_THRESHOLD = 2.0
ZSCORE_EXTREME_THRESHOLD = 3.0

# Prophet (time series forecasting)
PROPHET_FORECAST_DAYS = 7

# Cache invalidation
MODEL_CACHE_DAYS = 7

# ============================================================================
# RISK LEVELS
# ============================================================================
RISK_NORMAL = "Normal"
RISK_MODERATE = "Moderate Risk"
RISK_HIGH = "High Risk"
RISK_CRITICAL = "Critical Risk"

# ============================================================================
# CONFIDENCE LEVELS
# ============================================================================
CONF_HIGH = "High"
CONF_MEDIUM = "Medium"
CONF_LOW = "Low"
CONF_VERY_HIGH = "Very High"
