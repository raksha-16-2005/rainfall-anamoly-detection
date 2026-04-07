"""
Risk classification for rainfall anomalies.

Module for combining anomaly detection signals into interpretable risk levels.
"""

import logging
import math
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def classify_risk(row: dict) -> dict:
    """
    Classify rainfall anomaly risk based on multiple detection signals.

    Combines point anomalies, spatial clustering, Z-scores, dual-source agreement,
    and other signals into a composite risk assessment with confidence levels.

    Parameters
    ----------
    row : dict
        Dictionary with anomaly detection and source agreement signals:
        - anomaly_flag: bool, True if point anomaly detected via Isolation Forest
        - anomaly_score: float, Isolation Forest decision score (lower = more anomalous)
        - in_regional_cluster: bool, True if part of spatial anomaly cluster
        - zscore_category: str, 'normal' / 'moderate' / 'extreme'
        - rolling_zscore: float, rolling Z-score value
        - dual_source: bool, True if both data sources available
        - rainfall_gov: float or NaN, government source rainfall (mm)
        - rainfall_meteo: float or NaN, OpenMeteo source rainfall (mm)

    Returns
    -------
    dict
        Risk classification with keys:
        - risk_level: str, one of:
                     'Normal' - no significant anomaly
                     'Moderate Risk' - isolated anomaly or source disagreement
                     'High Risk' - regional anomaly cluster with extreme values
                     'Critical Risk' - confirmed regional anomaly (dual-source agreement)
        - confidence: str, one of:
                     'Very High' - multiple strong signals align
                     'High' - consistent multi-signal detection
                     'Medium' - single strong or multiple weak signals
                     'Low' - insufficient or conflicting signals
        - reason: str, human-readable justification for classification

    Notes
    -----
    - Classification rules applied in order; first match wins
    - Dual-source disagreement (>50 mm difference) flags data quality issues
    - Regional clusters (spatial + temporal) indicate widespread anomalies
    - Z-score extremes indicate deviations from recent history
    - Returns reasonable defaults if critical fields are missing

    Classification Rules (in priority order):
    1. Dual-source disagreement (sources conflict by >50 mm)
       → Moderate Risk, Low confidence
    2. No anomaly detected (not anomalous AND z-score < 2)
       → Normal, High confidence
    3. Isolated anomaly (anomalous but not in cluster) with moderate Z-score
       → Moderate Risk, Medium confidence
    4. Regional anomaly cluster with extreme Z-score AND dual-source confirmation
       → Critical Risk, Very High confidence
    5. Regional anomaly cluster with extreme Z-score (single or dual source)
       → High Risk, High confidence
    6. Default (catchall for edge cases)
       → Moderate Risk, Low confidence

    Examples
    --------
    >>> from risk_classifier import classify_risk
    >>> 
    >>> # Example 1: Normal conditions
    >>> row1 = {
    ...     'anomaly_flag': False,
    ...     'rolling_zscore': 0.5,
    ...     'zscore_category': 'normal',
    ...     'in_regional_cluster': False,
    ...     'dual_source': False
    ... }
    >>> classify_risk(row1)
    {'risk_level': 'Normal', 'confidence': 'High', 'reason': 'No anomaly detected'}
    
    >>> # Example 2: Critical risk (confirmed by both sources)
    >>> row2 = {
    ...     'anomaly_flag': True,
    ...     'rolling_zscore': 3.5,
    ...     'zscore_category': 'extreme',
    ...     'in_regional_cluster': True,
    ...     'dual_source': True,
    ...     'rainfall_gov': 150.0,
    ...     'rainfall_meteo': 152.0
    ... }
    >>> classify_risk(row2)
    {'risk_level': 'Critical Risk', 'confidence': 'Very High',
     'reason': 'Regional anomaly cluster with extreme Z-score, confirmed by dual sources'}
    """
    # Helper function to safely check for NaN
    def is_nan(x):
        try:
            return math.isnan(float(x)) if isinstance(x, float) else False
        except (TypeError, ValueError):
            return False

    # Rule 1: Dual-source disagreement (data quality issue)
    if row.get("dual_source", False):
        rainfall_gov = row.get("rainfall_gov")
        rainfall_meteo = row.get("rainfall_meteo")
        
        if not is_nan(rainfall_gov) and not is_nan(rainfall_meteo):
            try:
                disagreement = abs(float(rainfall_gov) - float(rainfall_meteo))
                if disagreement > 50:
                    return {
                        "risk_level": "Moderate Risk",
                        "confidence": "Low",
                        "reason": f"Dual-source disagreement ({disagreement:.1f}mm difference)"
                    }
            except (TypeError, ValueError):
                pass

    # Rule 2: No anomaly detected
    if not row.get("anomaly_flag", False):
        rolling_zscore = row.get("rolling_zscore", 0)
        if rolling_zscore < 2:
            return {
                "risk_level": "Normal",
                "confidence": "High",
                "reason": "No anomaly detected"
            }

    # Rule 3: Isolated anomaly with moderate Z-score
    if row.get("anomaly_flag", False) and not row.get("in_regional_cluster", False):
        if row.get("zscore_category") == "moderate":
            return {
                "risk_level": "Moderate Risk",
                "confidence": "Medium",
                "reason": "Isolated anomaly with moderate Z-score"
            }

    # Rule 4: Regional anomaly cluster + extreme Z-score + dual-source confirmation
    if (row.get("anomaly_flag", False) and 
        row.get("in_regional_cluster", False) and 
        row.get("zscore_category") == "extreme" and
        row.get("dual_source", False)):
        return {
            "risk_level": "Critical Risk",
            "confidence": "Very High",
            "reason": "Regional anomaly cluster with extreme Z-score, confirmed by dual sources"
        }

    # Rule 5: Regional anomaly cluster + extreme Z-score (any source combination)
    if (row.get("anomaly_flag", False) and 
        row.get("in_regional_cluster", False) and 
        row.get("zscore_category") == "extreme"):
        return {
            "risk_level": "High Risk",
            "confidence": "High",
            "reason": "Regional anomaly cluster with extreme Z-score"
        }

    # Rule 6: Default fallback (catchall)
    return {
        "risk_level": "Moderate Risk",
        "confidence": "Low",
        "reason": "Insufficient data for classification"
    }


def build_risk_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate risk classification report for all rainfall records.

    Applies classify_risk() to each row, expands the classification results
    into separate columns, and generates a summary report by risk level for
    the latest date in the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Rainfall DataFrame with all anomaly detection and source columns:
        - district: str
        - date: datetime
        - rainfall_avg: float
        - anomaly_flag: bool
        - anomaly_score: float
        - in_regional_cluster: bool
        - rolling_zscore: float
        - zscore_category: str
        - dual_source: bool
        - rainfall_gov: float or NaN
        - rainfall_meteo: float or NaN
        And other supporting columns from upstream pipeline steps.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns added:
        - risk_level: str, one of 'Normal', 'Moderate Risk', 'High Risk', 'Critical Risk'
        - confidence: str, one of 'Very High', 'High', 'Medium', 'Low'
        - reason: str, human-readable explanation for classification
        Index is reset to 0, 1, 2, ...

    Notes
    -----
    - Input DataFrame is not modified; a copy is returned
    - Risk classifications are applied independently to each row
    - Summary statistics logged for the latest date in the dataset
    - Summary shows district count grouped by risk level
    - If no date column present, summary applies to entire dataset

    Examples
    --------
    >>> from risk_classifier import build_risk_report
    >>> 
    >>> # DataFrame with all anomaly columns already added
    >>> enriched_df = detect_point_anomalies(merged_df)
    >>> enriched_df = detect_spatial_clusters(enriched_df, "2026-04-07")
    >>> enriched_df = compute_rolling_zscore(enriched_df)
    >>> 
    >>> risk_report_df = build_risk_report(enriched_df)
    >>> print(risk_report_df[['district', 'date', 'risk_level', 'confidence']].head(20))
    """
    try:
        logger.info(f"Building risk report for {len(df)} records")

        # Create a copy to avoid modifying input
        result = df.copy()

        # Apply classify_risk to each row and expand dict into columns
        risk_classifications = result.apply(
            lambda row: pd.Series(classify_risk(row.to_dict())), 
            axis=1
        )

        # Add the new columns to result
        result["risk_level"] = risk_classifications["risk_level"]
        result["confidence"] = risk_classifications["confidence"]
        result["reason"] = risk_classifications["reason"]

        # Reset index
        result = result.reset_index(drop=True)

        logger.info("Risk classifications applied to all records")

        # Generate summary statistics for latest date
        if "date" in result.columns:
            latest_date = result["date"].max()
            latest_data = result[result["date"] == latest_date]
            logger.info(f"Summary for latest date: {latest_date.date()}")
        else:
            latest_data = result
            logger.info("No date column found; summary for entire dataset")

        # Count by risk level
        risk_summary = latest_data["risk_level"].value_counts().sort_index()
        
        # Create summary message
        summary_lines = ["Risk Level Summary:"]
        summary_lines.append("-" * 60)
        for risk_level, count in risk_summary.items():
            distinct_districts = latest_data[latest_data["risk_level"] == risk_level]["district"].nunique()
            summary_lines.append(
                f"  {risk_level:20s}: {count:6d} records across {distinct_districts:3d} districts"
            )
        summary_lines.append("-" * 60)
        summary_lines.append(f"  Total: {len(latest_data)} records from {latest_data['district'].nunique()} districts")

        # Log the summary
        summary_message = "\n".join(summary_lines)
        logger.info(f"\n{summary_message}")

        return result

    except Exception as e:
        logger.error(f"Error building risk report: {type(e).__name__}: {e}")
        raise
