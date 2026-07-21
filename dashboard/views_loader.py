"""
Data access layer for the Streamlit dashboard.

Data access layer for the dashboard. Kept separate from app.py so I/O and caching are isolated from UI code:
the app never reads a parquet directly, it asks this module. The only
contract this dashboard depends on is the schema of the labeled views in
data/analytics/views/ - never the notebook, never an intermediate file,
never the transform scripts. If those view schemas are stable, the
dashboard cannot break regardless of how the pipeline internals change.

All loaders are cached with st.cache_data: the views are static between
pipeline runs, so re-reading them on every widget interaction would be
pure waste. Cache invalidates automatically if the file path changes;
for a new pipeline run, the user reloads the app (or we could key on file
mtime, deliberately not done here to keep it simple).
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# Resolve views dir relative to the repo root (two levels up from this file:
# dashboard/data_loader.py -> repo root), so the app works regardless of the
# working directory Streamlit is launched from.
VIEWS_DIR = Path(__file__).resolve().parent.parent / "data" / "analytics" / "views"

TOTAL_VIEW = VIEWS_DIR / "mortality_weather_weekly_labeled.parquet"
AGE_STRATIFIED_VIEW = VIEWS_DIR / "mortality_by_age_weekly_stratified_heat.parquet"


@st.cache_data
def load_total() -> pd.DataFrame:
    """Load the region-labeled total mortality + weather weekly view.

    Grain: one row per (geo, year, week). This is the view the RR,
    regression, and trend analyses run on.

    Returns:
        DataFrame with geo/region_name/macrozone/country labels, deaths,
        population, weather columns, and any_heatwave_week.

    Raises:
        FileNotFoundError: If the view doesn't exist yet - loud on
        purpose, so the user knows to run build_analysis_view.py rather
        than seeing a confusing empty dashboard.
    """
    if not TOTAL_VIEW.exists():
        raise FileNotFoundError(
            f"Missing {TOTAL_VIEW.name}. Run `python -m src.transform.build_analysis_view` first."
        )
    df = pd.read_parquet(TOTAL_VIEW)
    df["month"] = pd.to_datetime(df["week_start_date"]).dt.month
    return df


@st.cache_data
def load_age_stratified() -> pd.DataFrame:
    """Load the age-group-stratified mortality view (stratified_heat scheme).

    Grain: one row per (geo, sex, age_group, year, week), with age_group
    in {under_65, 65-74, 75-84, 85+}. Sex is M/F only (no 'T' total), so
    callers summing across sex are not double-counting.

    Returns:
        DataFrame with labels, age_group, sex, deaths, population,
        any_heatwave_week.

    Raises:
        FileNotFoundError: If the view doesn't exist yet.
    """
    if not AGE_STRATIFIED_VIEW.exists():
        raise FileNotFoundError(
            f"Missing {AGE_STRATIFIED_VIEW.name}. Run "
            f"`python -m src.transform.build_analysis_view` first."
        )
    return pd.read_parquet(AGE_STRATIFIED_VIEW)


@st.cache_data
def region_options(_df: pd.DataFrame) -> pd.DataFrame:
    """Distinct (geo, region_name, country, macrozone) rows, for building
    selectors without scanning the full frame on every rerun.

    The leading underscore on `_df` tells st.cache_data not to hash the
    (large) DataFrame itself - we treat it as already-cached upstream and
    just derive the small lookup from it.

    Args:
        _df: The total view (or any frame carrying the four label columns).

    Returns:
        DataFrame: one row per region, sorted by country then region_name.
    """
    cols = ["geo", "region_name", "country", "macrozone"]
    return (
        _df[cols]
        .drop_duplicates()
        .sort_values(["country", "region_name"])
        .reset_index(drop=True)
    )


@st.cache_data
def region_coordinates() -> pd.DataFrame:
    """NUTS2 code -> (lat, lon), for placing regions on the map.

    Reuses src.utils.locations_loader (the single source of coordinates,
    config/locations.yaml) rather than duplicating coordinates in the
    dashboard or requiring them to be baked into the views. Returns a small
    lookup joined onto RR-by-region results at render time.

    Returns:
        DataFrame with columns [geo, lat, lon].
    """
    # Imported here (not at module top) so a missing src path fails only if
    # the map is actually used, keeping the rest of the dashboard loadable.
    from src.utils.locations_loader import load_config, get_nuts2_coordinates

    coords = get_nuts2_coordinates(load_config())
    return pd.DataFrame(
        [{"geo": c["name"], "lat": c["lat"], "lon": c["lon"]} for c in coords]
    )