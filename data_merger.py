"""
Data merger for combining multiple rainfall sources.

Module for merging government and meteorological rainfall data.
"""

import pandas as pd


def merge_rainfall_sources(gov_df: pd.DataFrame, meteo_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge rainfall data from two sources (data.gov.in and Open-Meteo).

    Combines government and meteorological rainfall data using an outer join
    on District and Date. Creates unified rainfall columns and tracks data
    source availability.

    Parameters
    ----------
    gov_df : pd.DataFrame
        Government rainfall DataFrame with columns: District, Date, Rainfall_mm.
        Expected from load_district_rainfall().
    meteo_df : pd.DataFrame
        Open-Meteo rainfall DataFrame with columns: Date, Rainfall_mm, District.
        Expected from fetch_openmeteo_rainfall() (called per-district).

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns:
        - District: str, district name (normalized lowercase)
        - Date: datetime
        - rainfall_gov: float, rainfall from data.gov.in (may be NaN)
        - rainfall_meteo: float, rainfall from Open-Meteo (may be NaN)
        - rainfall_avg: float, mean of both sources if both available,
                        otherwise the available value (NaN if neither available)
        - dual_source: bool, True if both sources contributed to rainfall_avg
        Sorted by District and Date in ascending order.

    Notes
    -----
    - Uses outer join to preserve all dates from both sources
    - rainfall_avg prioritizes availability over averaging when only
      one source has data
    - dual_source is False if either source is missing for a given date
    - NaN values are preserved for source-specific columns
    """
    # Rename columns to distinguish sources
    gov_df_renamed = gov_df[["District", "Date", "Rainfall_mm"]].copy()
    gov_df_renamed.rename(columns={"Rainfall_mm": "rainfall_gov"}, inplace=True)

    meteo_df_renamed = meteo_df[["Date", "Rainfall_mm"]].copy()
    meteo_df_renamed.rename(columns={"Rainfall_mm": "rainfall_meteo"}, inplace=True)

    # Check if meteo_df has District column
    if "District" not in meteo_df.columns:
        raise ValueError("meteo_df must have a 'District' column for merging")

    meteo_df_renamed["District"] = meteo_df["District"]

    # Merge on District and Date
    merged_df = pd.merge(
        gov_df_renamed,
        meteo_df_renamed,
        on=["District", "Date"],
        how="outer"
    )

    # Create rainfall_avg column
    # Use mean when both exist, otherwise use whichever is available
    merged_df["rainfall_avg"] = merged_df[["rainfall_gov", "rainfall_meteo"]].mean(axis=1)

    # Create dual_source column - True only if both sources have non-null values
    merged_df["dual_source"] = (
        merged_df["rainfall_gov"].notna() & merged_df["rainfall_meteo"].notna()
    )

    # Sort by District and Date
    merged_df = merged_df.sort_values(by=["District", "Date"]).reset_index(drop=True)

    return merged_df
