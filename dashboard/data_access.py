"""
Dashboard data-access layer: Streamlit caching over the shared, pure loaders.

The actual "which parquet, at which path" knowledge lives in
src/analysis/views_loader.py, imported by BOTH the notebook and this dashboard
so neither can drift from the other. This module's only jobs are:

  1. wrap those pure loaders in @st.cache_data, so the dashboard doesn't
     re-read a static parquet on every widget interaction (the notebook,
     running once top-to-bottom, doesn't need caching and imports the pure
     loaders directly); and

  2. hold the dashboard-only helpers that have no place in the notebook or in
     src/ - region_options (builds the sidebar selectors) and
     region_coordinates (feeds the map).

Keeping the caching here rather than in src/ is deliberate: st.cache_data is a
Streamlit concern, and src/ must stay framework-agnostic so the notebook can
import from it without pulling Streamlit into a Jupyter kernel.
"""

import pandas as pd
import streamlit as st

from src.analysis.views_loader import (
    load_total as _load_total,
    load_age_stratified as _load_age_stratified,
    load_age_quinquennial as _load_age_quinquennial,
)


@st.cache_data
def load_total() -> pd.DataFrame:
    """Cached wrapper over src.analysis.views_loader.load_total()."""
    return _load_total()


@st.cache_data
def load_age_stratified() -> pd.DataFrame:
    """Cached wrapper over src.analysis.views_loader.load_age_stratified()."""
    return _load_age_stratified()


@st.cache_data
def load_age_quinquennial() -> pd.DataFrame:
    """Cached wrapper over src.analysis.views_loader.load_age_quinquennial()."""
    return _load_age_quinquennial()


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