"""
run_pipeline.py — Main runner for ML Raksha Rainfall Anomaly Prediction System.

Fetches data from Open-Meteo (archive + forecast), runs the full ML pipeline,
and saves classified results. Optionally launches the Streamlit dashboard.
"""

import argparse
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PROCESSED_DIR, DISTRICT_COORDS_FILE
from src.data_ingestion.openmeteo_api import fetch_all_districts, fetch_forecast, load_district_coords
from src.preprocessing.preprocess import preprocess_pipeline
from src.models.isolation_forest import detect_anomalies
from src.models.rolling_zscore import run_zscore_analysis
from src.models.dbscan_clustering import cluster_anomalies
from src.risk.risk_classifier import run_risk_pipeline


def fetch_historical(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical archive data, labeled as 'kaggle' for merge_sources()."""
    print(f"\n[1/6] Fetching historical archive data ({start_date} to {end_date})...")
    df = fetch_all_districts(start_date, end_date, use_cache=True)
    if not df.empty:
        df["source"] = "kaggle"
    print(f"  -> {len(df)} rows from {df['district'].nunique() if not df.empty else 0} districts")
    return df


def fetch_realtime() -> pd.DataFrame:
    """Fetch 7-day forecast data, labeled as 'openmeteo' for merge_sources()."""
    print("\n[2/6] Fetching 7-day forecast data...")
    df = fetch_forecast()
    if not df.empty:
        df["source"] = "openmeteo"
    print(f"  -> {len(df)} rows from {df['district'].nunique() if not df.empty else 0} districts")
    return df


def fetch_recent_archive(days: int = 14) -> pd.DataFrame:
    """Fetch recent archive (overlaps with forecast for dual-source rows)."""
    end = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=days)).strftime("%Y-%m-%d")
    print(f"\n[3/6] Fetching recent archive overlap ({start} to {end})...")
    df = fetch_all_districts(start, end, use_cache=False)
    if not df.empty:
        df["source"] = "openmeteo"
    print(f"  -> {len(df)} rows (overlap window for dual-source confidence)")
    return df


def run_ml_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Run all ML models on the preprocessed data."""
    coords_df = load_district_coords()

    # Isolation Forest
    print("\n[5/6] Running ML models...")
    print("  - Isolation Forest (anomaly detection)...")
    df = detect_anomalies(df)
    n_anomalies = (df["anomaly_flag"] == -1).sum() if "anomaly_flag" in df.columns else 0
    print(f"    Flagged {n_anomalies} anomalous rows ({n_anomalies/len(df)*100:.1f}%)")

    # DBSCAN Clustering on anomalous rows
    print("  - DBSCAN (spatial clustering)...")
    if not coords_df.empty and "anomaly_flag" in df.columns:
        anomalous = df[df["anomaly_flag"] == -1].copy()
        if not anomalous.empty:
            clustered = cluster_anomalies(anomalous, coords_df)
            df = df.drop(
                columns=[c for c in ["cluster_id", "is_regional_event"] if c in df.columns],
                errors="ignore",
            )
            if "date" in clustered.columns:
                cluster_cols = clustered[["district", "date", "cluster_id", "is_regional_event"]]
                df = df.merge(cluster_cols, on=["district", "date"], how="left")
            else:
                cluster_cols = clustered[["district", "cluster_id", "is_regional_event"]].drop_duplicates(subset=["district"])
                df = df.merge(cluster_cols, on="district", how="left")
            df["cluster_id"] = df["cluster_id"].fillna(-1).astype(int)
            df["is_regional_event"] = df["is_regional_event"].fillna(False)
            n_clusters = clustered[clustered["cluster_id"] >= 0]["cluster_id"].nunique()
            print(f"    Found {n_clusters} regional clusters")
        else:
            df["cluster_id"] = -1
            df["is_regional_event"] = False
            print("    No anomalous rows to cluster")
    else:
        df["cluster_id"] = -1
        df["is_regional_event"] = False
        print("    Skipped (no coordinates or no anomaly_flag)")

    # Rolling Z-Score
    print("  - Rolling Z-Score analysis...")
    df = run_zscore_analysis(df)

    # Risk Classification
    print("  - Risk classification...")
    df = run_risk_pipeline(df)

    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Total rows:      {len(df):,}")
    print(f"Districts:       {df['district'].nunique() if 'district' in df.columns else 'N/A'}")
    if "date" in df.columns:
        print(f"Date range:      {df['date'].min()} to {df['date'].max()}")
    if "risk_level" in df.columns:
        print("\nRisk Level Distribution:")
        dist = df["risk_level"].value_counts()
        for level, count in dist.items():
            print(f"  {level:20s}: {count:>7,} ({count/len(df)*100:.1f}%)")
    if "anomaly_flag" in df.columns:
        n_anom = (df["anomaly_flag"] == -1).sum()
        print(f"\nAnomalies:       {n_anom:,} ({n_anom/len(df)*100:.1f}%)")
    if "zscore_category" in df.columns:
        print("\nZ-Score Categories:")
        for cat, count in df["zscore_category"].value_counts().items():
            print(f"  {cat:20s}: {count:>7,}")
    if "data_source" in df.columns:
        print("\nData Sources:")
        for src, count in df["data_source"].value_counts().items():
            print(f"  {src:20s}: {count:>7,}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="ML Raksha Rainfall Anomaly Pipeline")
    parser.add_argument("--launch", action="store_true", help="Launch Streamlit dashboard after pipeline")
    parser.add_argument("--start-date", default="2023-01-01", help="Historical data start date (YYYY-MM-DD)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetching, use existing processed data")
    parser.add_argument("--reprocess", action="store_true", help="Re-run preprocessing on cached raw data (no API calls)")
    args = parser.parse_args()

    # Ensure output directory exists
    Path(DATA_PROCESSED_DIR).mkdir(parents=True, exist_ok=True)

    if args.reprocess:
        # Re-run preprocessing on cached per-district CSVs without hitting the API
        print("\n[REPROCESS] Loading cached raw data from disk...")
        import glob as _glob
        cache_files = list(Path(DATA_PROCESSED_DIR).glob("openmeteo_*.csv"))
        if not cache_files:
            print("ERROR: No cached data found in data/processed/. Run without --reprocess first.")
            sys.exit(1)
        dfs = []
        for f in cache_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        historical_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        if not historical_df.empty:
            historical_df["source"] = "kaggle"
        print(f"  -> {len(historical_df):,} rows from {historical_df['district'].nunique() if not historical_df.empty else 0} districts (cached)")

        print("\n[REPROCESS] Preprocessing with IMD normals + historical features...")
        df = preprocess_pipeline(
            kaggle_df=historical_df if not historical_df.empty else None,
            openmeteo_df=None,
        )
        print(f"  -> {len(df):,} rows after preprocessing")

    elif not args.skip_fetch:
        yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Step 1: Fetch historical archive data (labeled as 'kaggle' for merge_sources)
        historical_df = fetch_historical(args.start_date, yesterday)

        # Step 2: Fetch 7-day forecast (labeled as 'openmeteo' for merge_sources)
        forecast_df = fetch_realtime()

        # Step 3: Fetch recent archive overlap (labeled as 'openmeteo' to merge with forecast)
        recent_archive_df = fetch_recent_archive(days=14)

        # Combine forecast + recent archive as the "realtime" source
        realtime_df = pd.concat([forecast_df, recent_archive_df], ignore_index=True)
        # Deduplicate: keep one row per (district, date), preferring forecast
        realtime_df = realtime_df.drop_duplicates(subset=["district", "date"], keep="first")

        # Step 4: Preprocess & merge
        print("\n[4/6] Preprocessing & merging sources...")
        df = preprocess_pipeline(
            kaggle_df=historical_df if not historical_df.empty else None,
            openmeteo_df=realtime_df if not realtime_df.empty else None,
        )
        print(f"  -> {len(df)} rows after preprocessing")
    else:
        print("\n[SKIP] Loading existing processed data...")
        processed_path = Path(DATA_PROCESSED_DIR) / "processed_rainfall.csv"
        if not processed_path.exists():
            print("ERROR: No processed data found. Run without --skip-fetch first.")
            sys.exit(1)
        df = pd.read_csv(processed_path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        print(f"  -> {len(df)} rows loaded")

    # Step 5: Run ML models
    df = run_ml_pipeline(df)

    # Step 6: Save classified output
    output_path = Path(DATA_PROCESSED_DIR) / "classified_rainfall.csv"
    df.to_csv(output_path, index=False)
    print(f"\n[6/6] Saved classified data to {output_path}")

    print_summary(df)

    # Optionally launch Streamlit
    if args.launch:
        print("\nLaunching Streamlit dashboard...")
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(Path(__file__).parent / "src" / "dashboard" / "app.py"),
        ])


if __name__ == "__main__":
    main()
