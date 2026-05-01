"""
Streamlit Dashboard for the Rainfall Anomaly Prediction System (ml_raksha).

Entry point: streamlit run src/dashboard/app.py

Displays district-level risk maps, alerts table, district deep-dive analysis,
and regional cluster information loaded from the processed ML pipeline outputs.
"""

import time
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import streamlit as st

# ---------------------------------------------------------------------------
# Path bootstrap – make project root importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    DATA_PROCESSED_DIR,
    DISTRICT_COORDS_FILE,
    RISK_NORMAL,
    RISK_MODERATE,
    RISK_HIGH,
    RISK_CRITICAL,
    ZSCORE_MODERATE_THRESHOLD,
    ZSCORE_EXTREME_THRESHOLD,
)

from src.models.isolation_forest import detect_anomalies, compute_rolling_features
from src.models.dbscan_clustering import cluster_anomalies
from src.models.rolling_zscore import run_zscore_analysis
from src.risk.risk_classifier import run_risk_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RISK_COLOR_MAP = {
    RISK_NORMAL: "green",
    RISK_MODERATE: "yellow",
    RISK_HIGH: "orange",
    RISK_CRITICAL: "red",
}

# Ordered list used for multi-select defaults and sorting
RISK_LEVELS_ORDERED = [RISK_NORMAL, RISK_MODERATE, RISK_HIGH, RISK_CRITICAL]

# Columns expected in the final processed DataFrame
EXPECTED_COLUMNS = [
    "district",
    "date",
    "rainfall_mm",
    "departure_pct",
    "anomaly_flag",
    "anomaly_score",
    "z_score",
    "zscore_category",
    "cluster_id",
    "is_regional_event",
    "risk_level",
    "confidence",
]


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def load_data() -> pd.DataFrame:
    """
    Load rainfall data with ML classifications.

    Prefers classified_rainfall.csv (pre-computed by run_pipeline.py).
    Falls back to processed_rainfall.csv + re-running models if needed.

    Returns:
        pd.DataFrame: Risk-classified DataFrame.
                      Empty DataFrame with EXPECTED_COLUMNS if no data found.
    """
    classified_path = Path(DATA_PROCESSED_DIR) / "classified_rainfall.csv"
    processed_path = Path(DATA_PROCESSED_DIR) / "processed_rainfall.csv"

    # Prefer pre-computed classified data
    if classified_path.exists():
        try:
            df = pd.read_csv(classified_path)
            if not df.empty:
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                return df
        except Exception as exc:
            st.warning(f"Failed to read classified data, falling back: {exc}")

    # Fallback: load processed data and run ML pipeline
    if not processed_path.exists():
        st.warning(
            "No processed data found. Run `python run_pipeline.py` first."
        )
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    try:
        df = pd.read_csv(processed_path)
    except Exception as exc:
        st.error(f"Failed to read processed data: {exc}")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if df.empty:
        st.warning("Processed data file is empty.")
        return pd.DataFrame(columns=EXPECTED_COLUMNS)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Run ML pipeline stages (fallback path)
    try:
        df = detect_anomalies(df)
    except Exception as exc:
        st.warning(f"Anomaly detection skipped: {exc}")

    try:
        coords_df = load_coords()
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
            else:
                df["cluster_id"] = -1
                df["is_regional_event"] = False
    except Exception as exc:
        st.warning(f"Clustering skipped: {exc}")
        if "cluster_id" not in df.columns:
            df["cluster_id"] = -1
        if "is_regional_event" not in df.columns:
            df["is_regional_event"] = False

    try:
        df = run_zscore_analysis(df)
    except Exception as exc:
        st.warning(f"Z-score analysis skipped: {exc}")

    try:
        df = run_risk_pipeline(df)
    except Exception as exc:
        st.warning(f"Risk pipeline skipped: {exc}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


@st.cache_data
def load_coords() -> pd.DataFrame:
    """
    Load district coordinates from CSV.

    Returns:
        pd.DataFrame: Columns [district, state, latitude, longitude].
                      Empty DataFrame if file not found.
    """
    coords_file = Path(DISTRICT_COORDS_FILE)
    if not coords_file.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(coords_file)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Map builder
# ---------------------------------------------------------------------------

def build_risk_map(df: pd.DataFrame, coords_df: pd.DataFrame) -> folium.Map:
    """
    Build a Folium choropleth-style map of India with per-district risk markers.

    Args:
        df:        Risk-classified DataFrame (one row per district/date).
        coords_df: District coordinates DataFrame.

    Returns:
        folium.Map centered on India.
    """
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    if df.empty or coords_df.empty:
        return india_map

    # Use the most recent date available per district
    if "date" in df.columns:
        latest = (
            df.sort_values("date")
            .groupby("district", as_index=False)
            .last()
        )
    else:
        latest = df.drop_duplicates(subset=["district"], keep="last").copy()

    # Merge with coordinates
    merged = latest.merge(
        coords_df[["district", "latitude", "longitude"]],
        on="district",
        how="inner",
    )

    for _, row in merged.iterrows():
        risk = row.get("risk_level", RISK_NORMAL)
        color = RISK_COLOR_MAP.get(risk, "gray")

        district = row.get("district", "Unknown")
        confidence = row.get("confidence", "N/A")
        rainfall = row.get("rainfall_mm", float("nan"))
        z_score = row.get("z_score", float("nan"))
        anomaly_score = row.get("anomaly_score", float("nan"))

        tooltip_text = (
            f"<b>{district}</b><br>"
            f"Risk: {risk}<br>"
            f"Confidence: {confidence}"
        )

        def _safe_fmt(val, fmt):
            try:
                return format(float(val), fmt)
            except (TypeError, ValueError):
                return "N/A"

        popup_text = (
            f"<b>{district}</b><br>"
            f"Rainfall: {_safe_fmt(rainfall, '.1f')} mm<br>"
            f"Z-Score: {_safe_fmt(z_score, '.2f')}<br>"
            f"Anomaly Score: {_safe_fmt(anomaly_score, '.3f')}"
        )

        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            tooltip=folium.Tooltip(tooltip_text),
            popup=folium.Popup(popup_text, max_width=250),
        ).add_to(india_map)

    return india_map


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> tuple:
    """
    Render sidebar controls. Shows different filters based on selected view mode.

    Returns:
        tuple: (start_date, end_date, selected_states, selected_districts,
                selected_risks, auto_refresh)
    """
    view_mode = st.sidebar.radio(
        "View Mode",
        ["Live Data (2023-2026)", "Projections (2026-2030)"],
        key="sidebar_view_mode",
        horizontal=True,
    )
    st.session_state["_view_mode"] = view_mode

    if view_mode == "Projections (2026-2030)":
        # Projection sidebar — filters rendered here, not inside the tab
        st.sidebar.markdown("---")
        st.sidebar.header("Projection Filters")

        proj_path = Path(DATA_PROCESSED_DIR) / "projections_2030.csv"
        if proj_path.exists():
            _proj = pd.read_csv(proj_path, parse_dates=["ds"])
            _proj_only = _proj[_proj["type"] == "projection"]
            _districts = sorted(_proj_only["district"].unique().tolist())
        else:
            _districts = []

        month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                     7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

        st.session_state["proj_district"] = st.sidebar.selectbox(
            "District", options=_districts, key="proj_sb_district"
        )
        st.session_state["proj_year"] = st.sidebar.selectbox(
            "Year", options=[2026, 2027, 2028, 2029, 2030], index=2, key="proj_sb_year"
        )
        st.session_state["proj_month"] = st.sidebar.selectbox(
            "Month", options=list(range(1, 13)),
            format_func=lambda m: month_map[m],
            index=6, key="proj_sb_month"
        )

        # Return dummy values — tabs 1-5 won't render in projection mode
        today = datetime.today().date()
        return (today, today, [], [], [], False)

    st.sidebar.header("Filters")

    # --- Date range ---
    today = datetime.today().date()
    default_start = today - timedelta(days=30)

    start_date = st.sidebar.date_input(
        "Start Date",
        value=default_start,
        min_value=datetime(2009, 1, 1).date(),
        max_value=today,
    )
    end_date = st.sidebar.date_input(
        "End Date",
        value=today,
        min_value=datetime(2009, 1, 1).date(),
        max_value=today,
    )

    # --- State filter ---
    available_states: list = []
    if not df.empty and "state" in df.columns:
        available_states = sorted(df["state"].dropna().unique().tolist())

    selected_states = st.sidebar.multiselect(
        "State",
        options=available_states,
        default=[],
        placeholder="All states",
    )

    # --- District filter (cascading from state) ---
    available_districts: list = []
    if not df.empty and "district" in df.columns:
        if selected_states:
            district_pool = df[df["state"].isin(selected_states)]
        else:
            district_pool = df
        available_districts = sorted(district_pool["district"].dropna().unique().tolist())

    selected_districts = st.sidebar.multiselect(
        "District",
        options=available_districts,
        default=[],
        placeholder="All districts",
    )

    # --- Risk level filter ---
    selected_risks = st.sidebar.multiselect(
        "Risk Level",
        options=RISK_LEVELS_ORDERED,
        default=RISK_LEVELS_ORDERED,
    )

    # --- Auto-refresh ---
    auto_refresh = st.sidebar.checkbox("Auto-refresh daily", value=False)

    return (
        start_date,
        end_date,
        selected_states,
        selected_districts,
        selected_risks,
        auto_refresh,
    )


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_data(
    df: pd.DataFrame,
    start_date,
    end_date,
    states: list,
    districts: list,
    risks: list,
) -> pd.DataFrame:
    """
    Apply all sidebar filters to df.

    Args:
        df:         Full DataFrame.
        start_date: datetime.date lower bound (inclusive).
        end_date:   datetime.date upper bound (inclusive).
        states:     List of selected state names (empty = all).
        districts:  List of selected district names (empty = all).
        risks:      List of selected risk levels (empty = all).

    Returns:
        Filtered pd.DataFrame.
    """
    if df.empty:
        return df

    filtered = df.copy()

    # Date filter
    if "date" in filtered.columns:
        filtered["date"] = pd.to_datetime(filtered["date"])
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        filtered = filtered[
            (filtered["date"] >= start_dt) & (filtered["date"] <= end_dt)
        ]

    # State filter
    if states and "state" in filtered.columns:
        filtered = filtered[filtered["state"].isin(states)]

    # District filter
    if districts and "district" in filtered.columns:
        filtered = filtered[filtered["district"].isin(districts)]

    # Risk filter
    if risks and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(risks)]

    return filtered.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tab 1 – Live Map
# ---------------------------------------------------------------------------

def render_tab1_map(filtered_df: pd.DataFrame, coords_df: pd.DataFrame) -> None:
    """
    Render the Live Risk Map tab.

    Args:
        filtered_df: Sidebar-filtered DataFrame.
        coords_df:   District coordinates DataFrame.
    """
    st.subheader("Live Risk Map")

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    # Metric cards
    total_districts = (
        filtered_df["district"].nunique() if "district" in filtered_df.columns else 0
    )

    high_critical = 0
    if "risk_level" in filtered_df.columns:
        high_critical = filtered_df[
            filtered_df["risk_level"].isin([RISK_HIGH, RISK_CRITICAL])
        ]["district"].nunique() if "district" in filtered_df.columns else (
            filtered_df["risk_level"].isin([RISK_HIGH, RISK_CRITICAL])
        ).sum()

    col1, col2 = st.columns(2)
    col1.metric("Districts Monitored", total_districts)
    col2.metric("High + Critical Alerts", high_critical)

    # Folium map
    risk_map = build_risk_map(filtered_df, coords_df)
    st_folium(risk_map, width=1000, height=550)


# ---------------------------------------------------------------------------
# Tab 2 – Alerts Table
# ---------------------------------------------------------------------------

def render_tab2_alerts(filtered_df: pd.DataFrame) -> None:
    """
    Render the Active Alerts tab showing High and Critical risk rows.

    Args:
        filtered_df: Sidebar-filtered DataFrame.
    """
    st.subheader("Active Alerts")

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    if "risk_level" not in filtered_df.columns:
        st.info("Risk level information not available.")
        return

    # Filter to High + Critical only
    alerts_df = filtered_df[
        filtered_df["risk_level"].isin([RISK_HIGH, RISK_CRITICAL])
    ].copy()

    if alerts_df.empty:
        st.success("No active High or Critical risk alerts for the selected period.")
        return

    # Sort by risk (Critical > High) then date
    risk_order = {RISK_CRITICAL: 3, RISK_HIGH: 2, RISK_MODERATE: 1, RISK_NORMAL: 0}
    alerts_df["_risk_sort"] = alerts_df["risk_level"].map(risk_order).fillna(0)
    alerts_df = alerts_df.sort_values(
        by=["_risk_sort", "date"], ascending=[False, False]
    ).drop(columns=["_risk_sort"])

    # Select display columns
    display_cols = [c for c in ["district", "state", "date", "rainfall_mm",
                                 "risk_level", "confidence", "z_score"]
                    if c in alerts_df.columns]

    total_alerts = len(alerts_df)
    st.metric("Total Active Alerts", total_alerts)

    st.dataframe(alerts_df[display_cols], use_container_width=True)

    # Download button
    csv_bytes = alerts_df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Alerts as CSV",
        data=csv_bytes,
        file_name="active_alerts.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------------------------
# Tab 3 – District Deep-Dive
# ---------------------------------------------------------------------------

def render_tab3_deepdive(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """
    Render the District Deep-Dive tab with historical rainfall charts and
    Prophet forecast.

    Args:
        df:          Full (unfiltered) DataFrame for historical context.
        filtered_df: Sidebar-filtered DataFrame to populate the district selector.
    """
    st.subheader("District Deep-Dive")

    if filtered_df.empty or "district" not in filtered_df.columns:
        st.info("No data available for the selected filters.")
        return

    available_districts = sorted(filtered_df["district"].dropna().unique().tolist())
    if not available_districts:
        st.info("No districts match the current filters.")
        return

    selected_district = st.selectbox(
        "Select District",
        options=available_districts,
        key="deepdive_district_select",
    )

    # Pull full history for the selected district from the unfiltered df
    if "district" in df.columns:
        district_df = df[df["district"] == selected_district].copy()
    else:
        district_df = pd.DataFrame()

    if district_df.empty:
        st.info(f"No historical data found for {selected_district}.")
        return

    if "date" in district_df.columns:
        district_df = district_df.sort_values("date")

    # --- Historical rainfall bar + rolling mean line ---
    st.markdown("#### Historical Rainfall")
    if "rainfall_mm" in district_df.columns and "date" in district_df.columns:
        fig_rain = go.Figure()

        fig_rain.add_trace(
            go.Bar(
                x=district_df["date"],
                y=district_df["rainfall_mm"],
                name="Rainfall (mm)",
                marker_color="steelblue",
                opacity=0.7,
            )
        )

        if "rolling_30d_mean" in district_df.columns:
            fig_rain.add_trace(
                go.Scatter(
                    x=district_df["date"],
                    y=district_df["rolling_30d_mean"],
                    name="30-Day Rolling Mean",
                    mode="lines",
                    line=dict(color="orange", width=2),
                )
            )

        if "normal_mm" in district_df.columns and district_df["normal_mm"].notna().any():
            fig_rain.add_trace(
                go.Scatter(
                    x=district_df["date"],
                    y=district_df["normal_mm"],
                    name="IMD Monthly Normal (daily)",
                    mode="lines",
                    line=dict(color="green", width=2, dash="dot"),
                )
            )

        fig_rain.update_layout(
            title=f"Rainfall History – {selected_district}",
            xaxis_title="Date",
            yaxis_title="Rainfall (mm)",
            legend=dict(orientation="h"),
            height=400,
        )
        st.plotly_chart(fig_rain, use_container_width=True)
    else:
        st.info("Rainfall data columns not available.")

    # --- Z-Score over time ---
    st.markdown("#### Z-Score Over Time")
    if "z_score" in district_df.columns and "date" in district_df.columns:
        fig_z = go.Figure()

        fig_z.add_trace(
            go.Scatter(
                x=district_df["date"],
                y=district_df["z_score"],
                name="Z-Score",
                mode="lines",
                line=dict(color="purple", width=1.5),
            )
        )

        # Threshold lines
        x_range = [district_df["date"].min(), district_df["date"].max()]
        for level, color, label in [
            (ZSCORE_MODERATE_THRESHOLD,  "gold",  f"+{ZSCORE_MODERATE_THRESHOLD} threshold"),
            (-ZSCORE_MODERATE_THRESHOLD, "gold",  f"-{ZSCORE_MODERATE_THRESHOLD} threshold"),
            (ZSCORE_EXTREME_THRESHOLD,   "red",   f"+{ZSCORE_EXTREME_THRESHOLD} threshold"),
            (-ZSCORE_EXTREME_THRESHOLD,  "red",   f"-{ZSCORE_EXTREME_THRESHOLD} threshold"),
        ]:
            fig_z.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[level, level],
                    name=label,
                    mode="lines",
                    line=dict(color=color, width=1, dash="dash"),
                )
            )

        fig_z.update_layout(
            title=f"Z-Score – {selected_district}",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            height=350,
        )
        st.plotly_chart(fig_z, use_container_width=True)
    else:
        st.info("Z-score data not available.")

    # --- Risk timeline ---
    st.markdown("#### Risk Timeline")
    if "risk_level" in district_df.columns and "date" in district_df.columns:
        district_df["_risk_num"] = district_df["risk_level"].map(
            {RISK_NORMAL: 0, RISK_MODERATE: 1, RISK_HIGH: 2, RISK_CRITICAL: 3}
        ).fillna(0)

        fig_risk = px.scatter(
            district_df,
            x="date",
            y="_risk_num",
            color="risk_level",
            color_discrete_map=RISK_COLOR_MAP,
            title=f"Risk Timeline – {selected_district}",
            labels={"date": "Date", "_risk_num": "Risk Level"},
            height=300,
        )
        fig_risk.update_yaxes(
            tickvals=[0, 1, 2, 3],
            ticktext=[RISK_NORMAL, RISK_MODERATE, RISK_HIGH, RISK_CRITICAL],
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    # --- Prophet forecast (optional) ---
    st.markdown("#### 7-Day Prophet Forecast")
    try:
        from src.models.prophet_forecast import forecast_district as prophet_forecast

        with st.spinner(f"Generating forecast for {selected_district}..."):
            forecast_df = prophet_forecast(selected_district, district_df)

        if not forecast_df.empty:
            fig_forecast = go.Figure()

            fig_forecast.add_trace(
                go.Scatter(
                    x=forecast_df["ds"],
                    y=forecast_df["yhat"],
                    name="Forecast",
                    mode="lines+markers",
                    line=dict(color="royalblue", width=2),
                )
            )
            fig_forecast.add_trace(
                go.Scatter(
                    x=pd.concat([forecast_df["ds"], forecast_df["ds"].iloc[::-1]]),
                    y=pd.concat([forecast_df["yhat_upper"], forecast_df["yhat_lower"].iloc[::-1]]),
                    fill="toself",
                    fillcolor="rgba(65, 105, 225, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    name="80% Confidence Interval",
                )
            )

            fig_forecast.update_layout(
                title=f"Prophet Rainfall Forecast – {selected_district}",
                xaxis_title="Date",
                yaxis_title="Rainfall (mm)",
                height=350,
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        else:
            st.info("Forecast could not be generated for this district.")

    except ImportError:
        st.info("Prophet is not installed. Install it with: pip install prophet")
    except Exception as exc:
        st.warning(f"Forecast unavailable: {exc}")


# ---------------------------------------------------------------------------
# Tab 4 – Regional Clusters
# ---------------------------------------------------------------------------

def render_tab4_clusters(filtered_df: pd.DataFrame, coords_df: pd.DataFrame) -> None:
    """
    Render the Regional Clusters tab showing DBSCAN-identified spatial events.

    Args:
        filtered_df: Sidebar-filtered DataFrame.
        coords_df:   District coordinates DataFrame.
    """
    st.subheader("Regional Clusters")

    if filtered_df.empty:
        st.info("No data available for the selected filters.")
        return

    if "is_regional_event" not in filtered_df.columns:
        st.info("Cluster information not available. Run the full ML pipeline first.")
        return

    cluster_df = filtered_df[filtered_df["is_regional_event"] == True].copy()

    if cluster_df.empty:
        st.info("No regional cluster events detected for the current filters.")
        return

    # --- Cluster map ---
    st.markdown("#### Cluster Map")

    cluster_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)

    if not coords_df.empty and "cluster_id" in cluster_df.columns:
        cluster_ids = cluster_df["cluster_id"].unique()
        # Generate a color per cluster using a fixed palette
        palette = [
            "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
            "#ff7f00", "#a65628", "#f781bf", "#999999",
        ]

        merged_clusters = cluster_df.merge(
            coords_df[["district", "latitude", "longitude"]],
            on="district",
            how="inner",
        )

        # Use latest record per district
        if "date" in merged_clusters.columns:
            merged_clusters = (
                merged_clusters.sort_values("date")
                .groupby("district", as_index=False)
                .last()
            )

        for _, row in merged_clusters.iterrows():
            cid = int(row.get("cluster_id", -1))
            color = palette[cid % len(palette)] if cid >= 0 else "gray"
            district = row.get("district", "Unknown")
            risk = row.get("risk_level", RISK_NORMAL)

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=10,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
                tooltip=folium.Tooltip(
                    f"<b>{district}</b><br>Cluster: {cid}<br>Risk: {risk}"
                ),
            ).add_to(cluster_map)

    st_folium(cluster_map, width=1000, height=500)

    # --- Cluster summary table ---
    st.markdown("#### Cluster Summary")

    if "cluster_id" in cluster_df.columns and "district" in cluster_df.columns:
        agg_dict: dict = {"district": "nunique"}
        if "date" in cluster_df.columns:
            cluster_df["date"] = pd.to_datetime(cluster_df["date"])
            agg_dict["date"] = ["min", "max"]

        summary = cluster_df[cluster_df["cluster_id"] >= 0].groupby("cluster_id").agg(agg_dict)
        summary.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in summary.columns
        ]
        summary = summary.reset_index()

        rename_map = {
            "district_nunique": "Affected Districts",
            "date_min": "First Date",
            "date_max": "Last Date",
            "cluster_id": "Cluster ID",
        }
        summary = summary.rename(columns=rename_map)

        st.dataframe(summary, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def render_tab6_projections(df: pd.DataFrame, coords_df: pd.DataFrame) -> None:
    """Render 2030 Projections tab with sidebar filters and two sections."""
    proj_path = Path(DATA_PROCESSED_DIR) / "projections_2030.csv"
    if not proj_path.exists():
        st.warning("No projection data found. Run `python run_projections.py` first.")
        return

    proj_df = pd.read_csv(proj_path, parse_dates=["ds"])
    proj_only = proj_df[proj_df["type"] == "projection"].copy()
    if proj_only.empty:
        st.warning("Projection data is empty.")
        return

    proj_only["year"] = proj_only["ds"].dt.year
    proj_only["month"] = proj_only["ds"].dt.month
    month_map = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                 7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    # Read filters from session_state (set by render_sidebar)
    proj_district = st.session_state.get("proj_district", proj_only["district"].iloc[0])
    proj_year = st.session_state.get("proj_year", 2028)
    proj_month = st.session_state.get("proj_month", 7)

    # == SECTION 1: SELECTED DISTRICT + MONTH DETAIL ==
    st.subheader(f"Projection: {proj_district} - {month_map[proj_month]} {proj_year}")

    month_row = proj_only[
        (proj_only["district"] == proj_district) &
        (proj_only["year"] == proj_year) &
        (proj_only["month"] == proj_month)
    ]

    if not month_row.empty:
        row = month_row.iloc[0]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Projected Rainfall", f"{row['yhat']:.0f} mm")
        col2.metric("Lower Bound", f"{row['yhat_lower']:.0f} mm")
        col3.metric("Upper Bound", f"{row['yhat_upper']:.0f} mm")
        risk_label = row.get("projected_risk", "N/A")
        col4.metric("Risk Level", risk_label)

        if "normal_mm" in row.index and pd.notna(row.get("normal_mm")):
            dep = row.get("departure_pct", 0)
            normal_val = row["normal_mm"]
            if dep > 50:
                st.error(f"Projected rainfall is **{dep:.0f}% above** IMD normal ({normal_val:.0f} mm).")
            elif dep > 20:
                st.warning(f"Projected rainfall is **{dep:.0f}% above** IMD normal ({normal_val:.0f} mm).")
            elif dep < -50:
                st.error(f"Projected rainfall is **{abs(dep):.0f}% below** IMD normal ({normal_val:.0f} mm) - potential drought.")
            else:
                st.success(f"Within normal range (IMD normal: {normal_val:.0f} mm, departure: {dep:+.0f}%).")

        # Monthly breakdown for the year
        st.markdown(f"#### Monthly Breakdown - {proj_district} ({proj_year})")
        year_data = proj_only[
            (proj_only["district"] == proj_district) &
            (proj_only["year"] == proj_year)
        ].copy().sort_values("month")

        if not year_data.empty:
            year_data["Month"] = year_data["month"].map(month_map)
            colors = ["red" if m == proj_month else "steelblue" for m in year_data["month"]]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=year_data["Month"], y=year_data["yhat"],
                marker_color=colors, opacity=0.8,
                error_y=dict(type="data", symmetric=False,
                    array=(year_data["yhat_upper"] - year_data["yhat"]).values,
                    arrayminus=(year_data["yhat"] - year_data["yhat_lower"]).values),
                name="Projected",
            ))
            if "normal_mm" in year_data.columns and year_data["normal_mm"].notna().any():
                fig_bar.add_trace(go.Bar(
                    x=year_data["Month"], y=year_data["normal_mm"],
                    marker_color="green", opacity=0.4, name="IMD Normal",
                ))
            fig_bar.update_layout(
                yaxis_title="Rainfall (mm)", height=400, barmode="group",
                title=f"{proj_district} - {proj_year} (selected: {month_map[proj_month]})",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Risk map
        st.markdown(f"#### All Districts - {month_map[proj_month]} {proj_year}")
        map_month = proj_only[
            (proj_only["year"] == proj_year) & (proj_only["month"] == proj_month)
        ]
        if not map_month.empty and not coords_df.empty:
            map_data = map_month.merge(coords_df, on="district", how="inner")
            if not map_data.empty:
                risk_colors = {"High Excess Risk": "red", "Moderate Excess Risk": "orange",
                               "Drought Risk": "brown", "Normal": "green"}
                m = folium.Map(location=[20.5937, 78.9629], zoom_start=5, tiles="cartodbpositron")
                for _, r in map_data.iterrows():
                    color = risk_colors.get(r.get("projected_risk", "Normal"), "gray")
                    is_sel = r["district"] == proj_district
                    folium.CircleMarker(
                        location=[r["latitude"], r["longitude"]],
                        radius=9 if is_sel else 5,
                        color="black" if is_sel else color,
                        fill=True, fill_color=color, fill_opacity=0.8 if is_sel else 0.6,
                        weight=3 if is_sel else 1,
                        popup=folium.Popup(
                            f"<b>{r['district']}</b><br>"
                            f"Projected: {r['yhat']:.0f} mm<br>"
                            f"Range: {r['yhat_lower']:.0f}-{r['yhat_upper']:.0f} mm<br>"
                            f"Risk: {r.get('projected_risk', 'N/A')}",
                            max_width=200),
                    ).add_to(m)
                st_folium(m, width=700, height=500)
                st.caption("Red=High Excess | Orange=Moderate | Brown=Drought | Green=Normal | Black border=selected")
    else:
        st.info(f"No projection for {proj_district} in {month_map[proj_month]} {proj_year}.")

    # == SECTION 2: FULL TIMELINE ==
    st.markdown("---")
    st.subheader(f"Full Projection Timeline: {proj_district}")

    dist_proj = proj_df[proj_df["district"] == proj_district].copy().sort_values("ds")
    if not dist_proj.empty:
        hist_part = dist_proj[dist_proj["type"] == "historical"].copy()
        proj_part = dist_proj[dist_proj["type"] == "projection"].copy()

        fig = go.Figure()
        recent_hist = hist_part[hist_part["ds"] >= "2015-01-01"]
        if not recent_hist.empty:
            fig.add_trace(go.Scatter(
                x=recent_hist["ds"], y=recent_hist["yhat"],
                mode="lines", name="Historical (fitted)",
                line=dict(color="steelblue", width=2),
            ))
        if not proj_part.empty:
            fig.add_trace(go.Scatter(
                x=proj_part["ds"], y=proj_part["yhat_upper"],
                mode="lines", line=dict(width=0), showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=proj_part["ds"], y=proj_part["yhat_lower"],
                mode="lines", line=dict(width=0),
                fill="tonexty", fillcolor="rgba(255,0,0,0.15)",
                name="80% Confidence Interval",
            ))
            fig.add_trace(go.Scatter(
                x=proj_part["ds"], y=proj_part["yhat"],
                mode="lines", name="Projection (2026-2030)",
                line=dict(color="red", width=2.5),
            ))
        if not hist_part.empty and not proj_part.empty:
            boundary = proj_part["ds"].min().strftime("%Y-%m-%d")
            fig.add_shape(type="line", x0=boundary, x1=boundary,
                          y0=0, y1=1, yref="paper",
                          line=dict(color="gray", width=2, dash="dash"))
            fig.add_annotation(x=boundary, y=1, yref="paper",
                               text="Projection starts", showarrow=False,
                               yshift=10, font=dict(color="gray"))
        fig.update_layout(
            title=f"Monthly Rainfall: Historical + Projection - {proj_district}",
            xaxis_title="Date", yaxis_title="Rainfall (mm/month)",
            height=500, legend=dict(orientation="h"),
            xaxis=dict(range=["2015-01-01", "2031-06-01"], dtick="M12", tickformat="%Y"),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Annual Projected Rainfall")
        proj_annual = proj_part.copy()
        proj_annual["year"] = proj_annual["ds"].dt.year
        annual_summary = proj_annual.groupby("year").agg(
            total_mm=("yhat", "sum"), lower=("yhat_lower", "sum"), upper=("yhat_upper", "sum"),
        ).reset_index()
        annual_summary.columns = ["Year", "Projected (mm)", "Lower Bound", "Upper Bound"]
        for col in ["Projected (mm)", "Lower Bound", "Upper Bound"]:
            annual_summary[col] = annual_summary[col].round(0).astype(int)
        st.dataframe(annual_summary, use_container_width=True, hide_index=True)

        with st.expander("View raw projection data"):
            show_cols = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            if "projected_risk" in proj_part.columns:
                show_cols.append("projected_risk")
            display = proj_part[show_cols].copy()
            new_names = ["Date", "Projected (mm)", "Lower", "Upper"]
            if "projected_risk" in proj_part.columns:
                new_names.append("Risk")
            display.columns = new_names
            st.dataframe(display, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Top 20 Highest Risk Districts (2027-2030)")
    future = proj_only[proj_only["year"] >= 2027]
    if "projected_risk" in future.columns:
        risk_counts = future.groupby("district")["projected_risk"].apply(
            lambda x: (x.str.contains("High|Drought", na=False)).sum()
        ).reset_index()
        risk_counts.columns = ["District", "High/Drought Risk Months"]
        risk_counts = risk_counts.sort_values("High/Drought Risk Months", ascending=False).head(20)
        if not risk_counts.empty:
            st.dataframe(risk_counts, use_container_width=True, hide_index=True)


def render_tab5_historical(df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    """Render Historical Context tab with IMD normals and long-term trends."""
    st.subheader("Historical Context")

    if filtered_df.empty or "district" not in filtered_df.columns:
        st.info("No data available for the selected filters.")
        return

    available_districts = sorted(filtered_df["district"].dropna().unique().tolist())
    if not available_districts:
        st.info("No districts match the current filters.")
        return

    selected_district = st.selectbox(
        "Select District", options=available_districts, key="hist_district_select"
    )

    district_df = df[df["district"] == selected_district].copy() if "district" in df.columns else pd.DataFrame()
    if district_df.empty:
        st.info(f"No data found for {selected_district}.")
        return
    if "date" in district_df.columns:
        district_df = district_df.sort_values("date")
        district_df["_month"] = pd.to_datetime(district_df["date"]).dt.month

    # --- 1. Monthly Normal vs Actual ---
    st.markdown("#### Monthly Rainfall: Actual vs IMD Normal")
    if "normal_mm" in district_df.columns and district_df["normal_mm"].notna().any():
        monthly = district_df.groupby("_month").agg(
            actual=("rainfall_mm", "mean"),
            normal=("normal_mm", "first"),
        ).reset_index()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly["month_name"] = monthly["_month"].map(month_names)

        fig_norm = go.Figure()
        fig_norm.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["actual"],
            name="Avg Daily Actual (mm)", marker_color="steelblue",
        ))
        fig_norm.add_trace(go.Bar(
            x=monthly["month_name"], y=monthly["normal"],
            name="IMD Normal (daily mm)", marker_color="green", opacity=0.6,
        ))
        fig_norm.update_layout(
            title=f"Monthly Comparison - {selected_district}",
            barmode="group", height=400,
            xaxis_title="Month", yaxis_title="Rainfall (mm/day)",
        )
        st.plotly_chart(fig_norm, use_container_width=True)
    else:
        st.info("IMD normal data not available for this district. "
                "The district may not be in the IMD normals dataset.")

    # --- 2. Historical Percentile Gauge ---
    if "hist_percentile_rank" in district_df.columns and district_df["hist_percentile_rank"].notna().any():
        st.markdown("#### Historical Percentile (vs 1901-2015 Record)")
        recent = district_df.tail(30)
        avg_pct = recent["hist_percentile_rank"].mean()
        col1, col2, col3 = st.columns(3)
        col1.metric("Avg Percentile (last 30 days)", f"{avg_pct:.0f}th")
        max_pct = recent["hist_percentile_rank"].max()
        col2.metric("Peak Percentile", f"{max_pct:.0f}th")
        above_90 = (recent["hist_percentile_rank"] > 90).sum()
        col3.metric("Days > 90th Percentile", f"{above_90}")

        if avg_pct > 90:
            st.error("Current rainfall is in the TOP 10% of the 115-year record for this region.")
        elif avg_pct > 75:
            st.warning("Current rainfall is ABOVE AVERAGE compared to the 115-year record.")
        else:
            st.success("Current rainfall is within normal historical range.")

    # --- 3. Long-term Subdivision Trend ---
    st.markdown("#### Long-term Annual Trend (1901-2015)")
    try:
        from src.data_ingestion.imd_historical import (
            get_historical_annual_series, load_subdivision_mapping,
        )
        subdiv_map = load_subdivision_mapping()
        if "district" in subdiv_map.columns:
            subdiv_match = subdiv_map[subdiv_map["district"] == selected_district]
            if not subdiv_match.empty:
                subdivision = subdiv_match.iloc[0]["subdivision"]
                annual = get_historical_annual_series()
                sub_annual = annual[annual["subdivision"] == subdivision].sort_values("year")

                if not sub_annual.empty:
                    import numpy as np
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=sub_annual["year"], y=sub_annual["annual_mm"],
                        mode="lines", name="Annual Rainfall",
                        line=dict(color="steelblue", width=1),
                    ))
                    # Trend line
                    coeffs = np.polyfit(sub_annual["year"], sub_annual["annual_mm"], 1)
                    trend_y = np.polyval(coeffs, sub_annual["year"])
                    fig_trend.add_trace(go.Scatter(
                        x=sub_annual["year"], y=trend_y,
                        mode="lines", name=f"Trend ({coeffs[0]*10:.1f} mm/decade)",
                        line=dict(color="red", width=2, dash="dash"),
                    ))
                    fig_trend.update_layout(
                        title=f"Annual Rainfall - {subdivision}",
                        xaxis_title="Year", yaxis_title="Rainfall (mm)",
                        height=400,
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
                    st.caption(f"District **{selected_district}** belongs to IMD subdivision: **{subdivision}**")
                else:
                    st.info(f"No historical annual data for subdivision: {subdivision}")
            else:
                st.info("No subdivision mapping found for this district.")
        else:
            st.info("Subdivision mapping not available.")
    except Exception as exc:
        st.warning(f"Historical trend unavailable: {exc}")

    # --- 4. Departure from normal over time ---
    if "hist_departure_pct" in district_df.columns and district_df["hist_departure_pct"].notna().any():
        st.markdown("#### Departure from Historical Normal")
        fig_dep = go.Figure()
        colors = ["red" if v > 0 else "blue" for v in district_df["hist_departure_pct"].fillna(0)]
        fig_dep.add_trace(go.Bar(
            x=district_df["date"], y=district_df["hist_departure_pct"],
            marker_color=colors, name="Departure %",
        ))
        fig_dep.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
        fig_dep.update_layout(
            title=f"Daily Departure from 115-Year Normal - {selected_district}",
            xaxis_title="Date", yaxis_title="Departure (%)",
            height=350,
        )
        st.plotly_chart(fig_dep, use_container_width=True)


def main() -> None:
    """
    Dashboard entry point. Configures page, loads data, renders all tabs.
    """
    st.set_page_config(
        page_title="Rainfall Anomaly Detection",
        page_icon="🌧️",
        layout="wide",
    )

    st.title("India Rainfall Anomaly Detection System")
    st.caption("Real-time district-level rainfall risk monitoring")

    # Load data with spinner
    with st.spinner("Loading data and running ML pipeline..."):
        df = load_data()
        coords_df = load_coords()

    # Render sidebar and get filter values
    (
        start_date,
        end_date,
        selected_states,
        selected_districts,
        selected_risks,
        auto_refresh,
    ) = render_sidebar(df)

    # Apply filters
    filtered_df = filter_data(
        df,
        start_date,
        end_date,
        selected_states,
        selected_districts,
        selected_risks,
    )

    # Global download button in main area
    if not filtered_df.empty:
        csv_export = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Current View as CSV",
            data=csv_export,
            file_name="rainfall_anomaly_export.csv",
            mime="text/csv",
        )
    else:
        st.info("No data matches the current filters.")

    # Tabs
    view_mode = st.session_state.get("_view_mode", "Live Data (2023-2026)")

    if view_mode == "Projections (2026-2030)":
        # Show only the projections tab
        render_tab6_projections(df, coords_df)
    else:
        # Show the 5 live data tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs(
            ["🗺️ Live Map", "🚨 Alerts", "🔍 District Analysis",
             "🌀 Regional Clusters", "📊 Historical"]
        )

        with tab1:
            render_tab1_map(filtered_df, coords_df)

        with tab2:
            render_tab2_alerts(filtered_df)

        with tab3:
            render_tab3_deepdive(df, filtered_df)

        with tab4:
            render_tab4_clusters(filtered_df, coords_df)

        with tab5:
            render_tab5_historical(df, filtered_df)

    # Auto-refresh
    if auto_refresh:
        # Use session state to track last refresh time
        if "last_refresh" not in st.session_state:
            st.session_state.last_refresh = datetime.now()

        elapsed = (datetime.now() - st.session_state.last_refresh).total_seconds()
        st.sidebar.caption(f"Last refreshed: {st.session_state.last_refresh.strftime('%H:%M:%S')}")

        if elapsed >= 86400:  # 24 hours
            st.session_state.last_refresh = datetime.now()
            st.cache_data.clear()
            st.rerun()


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
