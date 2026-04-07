"""
Main orchestration script for rainfall anomaly detection project.

This module coordinates data loading, API calls, merging, and preprocessing
for the rainfall anomaly detection pipeline.
"""

import pandas as pd
from data_loader import load_district_rainfall
from api_client import fetch_openmeteo_rainfall
from geo_utils import get_coords
from data_merger import merge_rainfall_sources
from preprocessor import preprocess_for_model, DISTRICT_SCALERS


def main():
    """
    Orchestrate the complete rainfall data pipeline.
    
    This function demonstrates the workflow:
    1. Load government rainfall data from CSV
    2. Fetch OpenMeteo data for each district
    3. Merge both sources
    4. Preprocess for Prophet model
    """
    print("=" * 60)
    print("Rainfall Anomaly Detection Pipeline")
    print("=" * 60)

    # Step 1: Load government rainfall data
    print("\n[1/4] Loading government rainfall data...")
    try:
        gov_df = load_district_rainfall("data/rainfall_data.csv")
        print(f"    ✓ Loaded {len(gov_df)} records from data.gov.in")
        print(f"    Columns: {list(gov_df.columns)}")
    except FileNotFoundError:
        print("    ✗ Error: data/rainfall_data.csv not found")
        print("    Please ensure rainfall_data.csv is in the data/ directory")
        return

    # Step 2: Fetch OpenMeteo data (example for first district)
    print("\n[2/4] Fetching OpenMeteo weather data...")
    print("    (In production, fetch for all districts)")
    
    # Example: Get coordinates for a district and fetch its data
    districts_sample = gov_df["District"].unique()[:1]
    meteo_dfs = []
    
    for district in districts_sample:
        try:
            lat, lon = get_coords(district)
            # Example parameters - adjust dates as needed
            meteo_df = fetch_openmeteo_rainfall(
                lat, lon, 
                start_date="2023-01-01", 
                end_date="2024-12-31"
            )
            meteo_df["District"] = district
            meteo_dfs.append(meteo_df)
            print(f"    ✓ Fetched OpenMeteo data for {district}")
        except ValueError as e:
            print(f"    ✗ Error fetching data for {district}: {e}")
    
    if not meteo_dfs:
        print("    ✗ No OpenMeteo data fetched. Proceeding with government data only.")
        meteo_combined = pd.DataFrame(
            columns=["Date", "Rainfall_mm", "District"]
        )
    else:
        meteo_combined = pd.concat(meteo_dfs, ignore_index=True)
        print(f"    ✓ Total OpenMeteo records: {len(meteo_combined)}")

    # Step 3: Merge both sources
    print("\n[3/4] Merging rainfall sources...")
    merged_df = merge_rainfall_sources(gov_df, meteo_combined)
    print(f"    ✓ Merged dataset: {len(merged_df)} records")
    print(f"    Dual-source records: {merged_df['dual_source'].sum()}")

    # Step 4: Preprocess for Prophet
    print("\n[4/4] Preprocessing for Prophet model...")
    prophet_dfs = preprocess_for_model(merged_df)
    print(f"    ✓ Prepared {len(prophet_dfs)} districts for modeling")
    
    # Summary
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    print(f"\nPrepared districts:")
    for district in sorted(prophet_dfs.keys()):
        df = prophet_dfs[district]
        print(f"  • {district}: {len(df)} records (normalized min-max scaled)")
    
    print("\nNext steps:")
    print("  1. Fit Prophet models using prophet_dfs")
    print("  2. Use DISTRICT_SCALERS for inverse transformation of predictions")
    print("  3. Detect anomalies in forecasts vs actual values")
    
    return {
        "government_data": gov_df,
        "openmeteo_data": meteo_combined,
        "merged_data": merged_df,
        "prophet_ready": prophet_dfs,
    }


if __name__ == "__main__":
    results = main()
