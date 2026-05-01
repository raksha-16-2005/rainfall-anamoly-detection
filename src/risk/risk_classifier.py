"""
Risk Classification Module for Rainfall Anomaly Prediction System.

Combines outputs from all ML models to produce final risk classification per (district, date).
Applies risk classification rules in priority order to determine risk level and confidence.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from config import (
    RISK_NORMAL,
    RISK_MODERATE,
    RISK_HIGH,
    RISK_CRITICAL,
    CONF_HIGH,
    CONF_MEDIUM,
    CONF_LOW,
    CONF_VERY_HIGH,
)


def classify_risk_row(row: pd.Series) -> tuple:
    """
    Apply risk classification logic to a single row.

    Returns (risk_level, confidence) tuple based on the priority-ordered rules:
    1. Critical: data_source='both' AND sources_agree=True AND is_regional_event
    2. High: anomaly AND is_regional_event AND zscore_category='Extreme'
    3. Moderate (anomaly only): anomaly AND zscore_category='Moderate'
    4. Moderate (disagreement): sources_agree=False AND data_source='both'
    5. Normal: no anomaly AND zscore_category='Normal'
    6. Default: RISK_NORMAL, CONF_MEDIUM

    Args:
        row: pd.Series with columns from ML pipeline output

    Returns:
        tuple: (risk_level, confidence)
    """
    # Handle missing values with defaults
    anomaly_flag = row.get("anomaly_flag", 1)  # 1 = normal, -1 = anomaly
    if pd.isna(anomaly_flag):
        anomaly_flag = 1

    zscore_category = row.get("zscore_category", "Normal")
    if pd.isna(zscore_category):
        zscore_category = "Normal"

    is_regional_event = row.get("is_regional_event", False)
    if pd.isna(is_regional_event):
        is_regional_event = False

    sources_agree = row.get("sources_agree", True)
    if pd.isna(sources_agree):
        sources_agree = True

    data_source = row.get("data_source", "both")
    if pd.isna(data_source):
        data_source = "both"

    # Rule 1 (Priority): Critical Risk
    # anomaly_flag=-1 AND data_source='both' AND sources_agree=True AND is_regional_event
    if anomaly_flag == -1 and data_source == "both" and sources_agree is True and is_regional_event is True:
        return (RISK_CRITICAL, CONF_VERY_HIGH)

    # Rule 2: High Risk
    # anomaly AND is_regional_event AND zscore_category='Extreme'
    if anomaly_flag == -1 and is_regional_event is True and zscore_category == "Extreme":
        return (RISK_HIGH, CONF_HIGH)

    # Rule 2.5: High Risk (historically unprecedented)
    # anomaly AND rainfall exceeds 95th percentile of 115-year record
    hist_pct = row.get("hist_percentile_rank", 50)
    if pd.isna(hist_pct):
        hist_pct = 50
    if anomaly_flag == -1 and hist_pct > 95:
        return (RISK_HIGH, CONF_HIGH)

    # Rule 3: Moderate Risk (anomaly only)
    # anomaly AND zscore_category='Moderate'
    if anomaly_flag == -1 and zscore_category == "Moderate":
        return (RISK_MODERATE, CONF_MEDIUM)

    # Rule 4: Moderate Risk (data source disagreement)
    # sources_agree=False AND data_source='both'
    if sources_agree is False and data_source == "both":
        return (RISK_MODERATE, CONF_LOW)

    # Rule 5: Normal Risk
    # no anomaly AND zscore_category='Normal'
    if anomaly_flag == 1 and zscore_category == "Normal":
        return (RISK_NORMAL, CONF_HIGH)

    # Default
    return (RISK_NORMAL, CONF_MEDIUM)


def classify_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply classify_risk_row to every row and add risk_level and confidence columns.

    Args:
        df: Input DataFrame with ML model outputs

    Returns:
        DataFrame with risk_level and confidence columns added
    """
    df = df.copy()

    # Apply classification to each row
    classifications = df.apply(classify_risk_row, axis=1)
    df["risk_level"] = classifications.apply(lambda x: x[0])
    df["confidence"] = classifications.apply(lambda x: x[1])

    return df


def get_risk_summary(df: pd.DataFrame, date: str = None) -> pd.DataFrame:
    """
    Get summary of risk levels by district count.

    Args:
        df: Input DataFrame with risk_level column
        date: Optional date filter (YYYY-MM-DD format)

    Returns:
        Summary DataFrame with columns: [risk_level, district_count, pct_of_total]
    """
    data = df.copy()

    if date is not None:
        data = data[data["date"] == date]

    # Group by risk_level and count unique districts
    summary = data.groupby("risk_level")["district"].nunique().reset_index()
    summary.rename(columns={"district": "district_count"}, inplace=True)

    # Calculate percentage
    total_districts = summary["district_count"].sum()
    summary["pct_of_total"] = (summary["district_count"] / total_districts * 100).round(2)

    return summary


def get_high_risk_districts(
    df: pd.DataFrame, date: str = None, min_risk: str = None
) -> pd.DataFrame:
    """
    Filter districts at or above min_risk level.

    Risk level order: RISK_NORMAL < RISK_MODERATE < RISK_HIGH < RISK_CRITICAL

    Args:
        df: Input DataFrame with risk_level column
        date: Optional date filter (YYYY-MM-DD format)
        min_risk: Minimum risk level (default: RISK_HIGH)

    Returns:
        Filtered DataFrame sorted by risk_level (descending) then anomaly_score (descending)
    """
    if min_risk is None:
        min_risk = RISK_HIGH

    data = df.copy()

    if date is not None:
        data = data[data["date"] == date]

    # Define risk level ordering
    risk_order = [RISK_NORMAL, RISK_MODERATE, RISK_HIGH, RISK_CRITICAL]
    risk_value_map = {risk: idx for idx, risk in enumerate(risk_order)}

    # Get minimum risk value
    min_risk_value = risk_value_map.get(min_risk, 2)  # Default to RISK_HIGH

    # Filter rows where risk level >= min_risk
    data = data[data["risk_level"].apply(lambda x: risk_value_map.get(x, 0) >= min_risk_value)]

    # Sort by risk level (descending) then anomaly_score (descending)
    data = data.sort_values(
        by=["risk_level", "anomaly_score"],
        key=lambda x: x.map(risk_value_map) if x.name == "risk_level" else x,
        ascending=[False, False],
    )

    return data


def run_risk_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main entry point for risk classification pipeline.

    - Calls classify_all() to classify all rows
    - Prints risk summary
    - Prints count of high+critical risk districts
    - Returns classified DataFrame with specified columns

    Args:
        df: Input DataFrame with all ML model outputs

    Returns:
        Classified DataFrame with columns:
        [district, date, rainfall_mm, departure_pct, anomaly_flag, anomaly_score,
         z_score, zscore_category, cluster_id, is_regional_event, risk_level, confidence]
    """
    # Classify all rows
    result = classify_all(df)

    # Print risk summary
    print("\n" + "="*70)
    print("RISK CLASSIFICATION SUMMARY")
    print("="*70)
    summary = get_risk_summary(result)
    print(summary.to_string(index=False))

    # Count high + critical risk districts
    high_risk_data = get_high_risk_districts(result, min_risk=RISK_HIGH)
    high_critical_count = high_risk_data["district"].nunique()
    print(f"\nDistricts at High + Critical Risk: {high_critical_count}")
    print("="*70 + "\n")

    # Reorder columns — core columns first, then optional enrichment columns
    output_columns = [
        "district",
        "date",
        "rainfall_mm",
        "departure_pct",
        "normal_mm",
        "subdivision",
        "hist_departure_pct",
        "hist_percentile_rank",
        "hist_mean_mm",
        "hist_std_mm",
        "hist_p10",
        "hist_p90",
        "hist_trend_slope",
        "anomaly_flag",
        "anomaly_score",
        "z_score",
        "zscore_category",
        "cluster_id",
        "is_regional_event",
        "risk_level",
        "confidence",
    ]

    # Only include columns that exist in the result
    available_columns = [col for col in output_columns if col in result.columns]
    return result[available_columns]
