"""MongoDB writer module for saving predictions and alerts."""

import logging
from datetime import datetime

import pandas as pd
from pymongo.operations import UpdateOne

from db.connection import get_collection

logger = logging.getLogger(__name__)


def save_predictions(risk_df: pd.DataFrame, forecast_df: pd.DataFrame) -> None:
    """
    Upserts prediction documents to MongoDB and writes high-risk alerts.

    Performs bulk upsert of risk data enriched with forecast values into the
    'predictions' collection. Additionally, writes rows with High/Critical risk
    levels to the 'alerts' collection. Uses compound keys (district, date) for
    predictions and maintains alert history with acknowledged flag.

    Args:
        risk_df (pd.DataFrame): Risk analysis dataframe with columns:
            district, date, rainfall_avg, rainfall_gov, rainfall_meteo,
            dual_source, anomaly_flag, anomaly_score, in_regional_cluster,
            rolling_zscore, zscore_category, risk_level, confidence, reason
        forecast_df (pd.DataFrame): Forecast dataframe with columns:
            district, ds (datetime), yhat, yhat_lower, yhat_upper

    Returns:
        None

    Raises:
        ValueError: If required columns are missing from input dataframes.
        Exception: Raised and logged if database operations fail.

    Logs:
        - Total rows processed
        - Number of predictions inserted vs. modified
        - Number of alerts created
    """
    if risk_df.empty:
        logger.warning("save_predictions called with empty risk_df")
        return

    # Validate required columns
    required_risk_cols = {
        "district",
        "date",
        "rainfall_avg",
        "rainfall_gov",
        "rainfall_meteo",
        "dual_source",
        "anomaly_flag",
        "anomaly_score",
        "in_regional_cluster",
        "rolling_zscore",
        "zscore_category",
        "risk_level",
        "confidence",
        "reason",
    }
    if not required_risk_cols.issubset(risk_df.columns):
        missing = required_risk_cols - set(risk_df.columns)
        raise ValueError(f"Missing required columns in risk_df: {missing}")

    required_forecast_cols = {"district", "ds", "yhat", "yhat_lower", "yhat_upper"}
    if not required_forecast_cols.issubset(forecast_df.columns):
        missing = required_forecast_cols - set(forecast_df.columns)
        raise ValueError(f"Missing required columns in forecast_df: {missing}")

    # Prepare data for merge: convert dates to date objects for comparison
    risk_df_copy = risk_df.copy()
    risk_df_copy["date"] = pd.to_datetime(risk_df_copy["date"]).dt.date

    forecast_df_copy = forecast_df.copy()
    forecast_df_copy["date"] = pd.to_datetime(forecast_df_copy["ds"]).dt.date
    forecast_df_copy = forecast_df_copy[
        ["district", "date", "yhat", "yhat_lower", "yhat_upper"]
    ]

    # Merge forecast data with risk data on (district, date)
    merged_df = risk_df_copy.merge(
        forecast_df_copy, on=["district", "date"], how="left"
    )

    # Prepare predictions collection operations
    predictions_collection = get_collection("predictions")
    update_operations = []
    current_time = datetime.utcnow()

    for _, row in merged_df.iterrows():
        filter_doc = {"district": row["district"], "date": row["date"]}

        # Build update document with all risk columns
        update_doc = {
            "district": row["district"],
            "date": row["date"],
            "rainfall_avg": (
                float(row["rainfall_avg"]) if pd.notna(row["rainfall_avg"]) else None
            ),
            "rainfall_gov": (
                float(row["rainfall_gov"]) if pd.notna(row["rainfall_gov"]) else None
            ),
            "rainfall_meteo": (
                float(row["rainfall_meteo"])
                if pd.notna(row["rainfall_meteo"])
                else None
            ),
            "dual_source": (
                bool(row["dual_source"]) if pd.notna(row["dual_source"]) else None
            ),
            "anomaly_flag": (
                bool(row["anomaly_flag"]) if pd.notna(row["anomaly_flag"]) else None
            ),
            "anomaly_score": (
                float(row["anomaly_score"]) if pd.notna(row["anomaly_score"]) else None
            ),
            "in_regional_cluster": (
                bool(row["in_regional_cluster"])
                if pd.notna(row["in_regional_cluster"])
                else None
            ),
            "rolling_zscore": (
                float(row["rolling_zscore"])
                if pd.notna(row["rolling_zscore"])
                else None
            ),
            "zscore_category": (
                str(row["zscore_category"])
                if pd.notna(row["zscore_category"])
                else None
            ),
            "risk_level": str(row["risk_level"]),
            "confidence": (
                float(row["confidence"]) if pd.notna(row["confidence"]) else None
            ),
            "reason": str(row["reason"]) if pd.notna(row["reason"]) else None,
            "predicted_mm": float(row["yhat"]) if pd.notna(row["yhat"]) else None,
            "yhat_lower": (
                float(row["yhat_lower"]) if pd.notna(row["yhat_lower"]) else None
            ),
            "yhat_upper": (
                float(row["yhat_upper"]) if pd.notna(row["yhat_upper"]) else None
            ),
            "created_at": current_time,
        }

        update_op = UpdateOne(filter_doc, {"$set": update_doc}, upsert=True)
        update_operations.append(update_op)

    # Execute bulk write for predictions
    try:
        if update_operations:
            result = predictions_collection.bulk_write(update_operations)
            logger.info(
                f"Predictions bulk write completed: "
                f"total_rows={len(merged_df)}, "
                f"inserted={result.upserted_count}, "
                f"modified={result.modified_count}"
            )
    except Exception as e:
        logger.error(f"Error writing predictions to database: {e}")
        raise

    # Write alerts for high/critical risk
    high_risk_mask = merged_df["risk_level"].isin(["High Risk", "Critical Risk"])
    alerts_df = merged_df[high_risk_mask]

    if not alerts_df.empty:
        alerts_collection = get_collection("alerts")
        alert_operations = []

        for _, row in alerts_df.iterrows():
            filter_doc = {"district": row["district"], "date": row["date"]}

            alert_doc = {
                "district": row["district"],
                "date": row["date"],
                "risk_level": str(row["risk_level"]),
                "confidence": (
                    float(row["confidence"]) if pd.notna(row["confidence"]) else None
                ),
                "reason": str(row["reason"]) if pd.notna(row["reason"]) else None,
                "rainfall_avg": (
                    float(row["rainfall_avg"]) if pd.notna(row["rainfall_avg"]) else None
                ),
                "rolling_zscore": (
                    float(row["rolling_zscore"])
                    if pd.notna(row["rolling_zscore"])
                    else None
                ),
                "acknowledged": False,
                "created_at": current_time,
            }

            alert_op = UpdateOne(filter_doc, {"$set": alert_doc}, upsert=True)
            alert_operations.append(alert_op)

        try:
            if alert_operations:
                alert_result = alerts_collection.bulk_write(alert_operations)
                logger.info(
                    f"Alerts bulk write completed: "
                    f"total_alerts={len(alerts_df)}, "
                    f"inserted={alert_result.upserted_count}, "
                    f"modified={alert_result.modified_count}"
                )
        except Exception as e:
            logger.error(f"Error writing alerts to database: {e}")
            raise
    else:
        logger.info("No high-risk or critical-risk alerts to write")
