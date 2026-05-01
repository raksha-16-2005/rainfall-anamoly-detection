"""
Run Prophet 2030 projections for all districts.

Usage:
    python run_projections.py                  # All 458 districts
    python run_projections.py --districts Mumbai Chennai Jaipur  # Specific ones
    python run_projections.py --sample 20      # Random 20 for quick test
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_PROCESSED_DIR
from src.models.prophet_2030 import (
    project_all_districts,
    compute_risk_projections,
)
from src.data_ingestion.imd_historical import load_subdivision_mapping
from src.data_ingestion.imd_normals import load_district_normals


def main():
    parser = argparse.ArgumentParser(description="Prophet 2030 Rainfall Projections")
    parser.add_argument("--districts", nargs="+", help="Specific districts to project")
    parser.add_argument("--sample", type=int, help="Random sample of N districts")
    parser.add_argument("--no-cache", action="store_true", help="Force retrain all models")
    args = parser.parse_args()

    # Load classified data
    classified_path = Path(DATA_PROCESSED_DIR) / "classified_rainfall.csv"
    if not classified_path.exists():
        print("ERROR: Run run_pipeline.py first to generate classified data.")
        sys.exit(1)

    print("Loading classified rainfall data...")
    classified_df = pd.read_csv(classified_path)
    classified_df["date"] = pd.to_datetime(classified_df["date"])

    # Load mappings
    subdiv_map = load_subdivision_mapping()

    # Determine districts
    if args.districts:
        districts = args.districts
    elif args.sample:
        all_districts = sorted(classified_df["district"].unique())
        import random
        districts = random.sample(all_districts, min(args.sample, len(all_districts)))
    else:
        districts = None  # all

    n = len(districts) if districts else classified_df["district"].nunique()
    print(f"\nProjecting {n} districts to 2030 using Prophet...")
    print("(This trains one Prophet model per district — may take a few minutes)\n")

    projections = project_all_districts(
        classified_df, subdiv_map,
        districts=districts,
        use_cache=not args.no_cache,
    )

    if projections.empty:
        print("ERROR: No projections generated.")
        sys.exit(1)

    # Add risk classifications
    print("\nComputing risk projections...")
    normals = load_district_normals()
    projections = compute_risk_projections(projections, normals)

    # Save
    output_path = Path(DATA_PROCESSED_DIR) / "projections_2030.csv"
    projections.to_csv(output_path, index=False)
    print(f"\nSaved projections to {output_path}")

    # Summary
    proj_only = projections[projections["type"] == "projection"]
    print(f"\n{'='*60}")
    print("PROJECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total projection rows: {len(proj_only):,}")
    print(f"Districts projected:   {proj_only['district'].nunique()}")
    print(f"Date range:            {proj_only['ds'].min()} to {proj_only['ds'].max()}")
    print(f"\nProjected Risk Distribution (2026-2030):")
    if "projected_risk" in proj_only.columns:
        for risk, count in proj_only["projected_risk"].value_counts().items():
            print(f"  {risk:25s}: {count:>6,} ({count/len(proj_only)*100:.1f}%)")

    # Show sample projections for key districts
    print(f"\nSample: Annual projected rainfall (mm) for 2027-2030:")
    key_districts = ["Mumbai", "Chennai", "Jaipur", "Shillong", "Jaisalmer"]
    for dist in key_districts:
        d = proj_only[proj_only["district"] == dist]
        if d.empty:
            continue
        d = d.copy()
        d["year"] = pd.to_datetime(d["ds"]).dt.year
        annual = d.groupby("year")["yhat"].sum()
        years_str = "  ".join(f"{y}:{v:.0f}" for y, v in annual.items() if y >= 2027)
        print(f"  {dist:20s}: {years_str}")

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
