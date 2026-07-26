"""
Analysis-view loading: the single, framework-agnostic place that knows where
the labeled analysis views live and how to read them.

This module is the shared data-access contract for BOTH consumers of the
pipeline output - the descriptive-analysis notebook and the Streamlit
dashboard. It deliberately depends on nothing but pandas: no Streamlit, no
plotting, no UI. That keeps `src/` pure (a UI framework has no business in the
pipeline layer) and lets the notebook import these loaders without pulling
Streamlit into a Jupyter kernel.

The dashboard adds its own thin caching wrapper (dashboard/data_access.py,
with @st.cache_data) on top of these functions; the notebook calls them
directly. Either way, the knowledge of "which parquet, at which path, with
what post-load fixups" lives here and nowhere else.

The only contract downstream code depends on is the schema of the labeled
views in data/analytics/views/ - never an intermediate file, never the
transform scripts. If those view schemas are stable, both consumers keep
working regardless of how the pipeline internals change.
"""

from pathlib import Path

import pandas as pd

# Resolve the views dir relative to THIS file, not the working directory, so
# the loaders work whether they're called from the repo root (Streamlit),
# from notebooks/ (Jupyter), or anywhere else. This file lives at
# src/analysis/views_loader.py, so the repo root is three levels up.
VIEWS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "analytics" / "views"

TOTAL_VIEW = VIEWS_DIR / "mortality_weather_weekly_labeled.parquet"
AGE_STRATIFIED_VIEW = VIEWS_DIR / "mortality_by_age_weekly_stratified_heat.parquet"
AGE_QUINQUENNIAL_VIEW = VIEWS_DIR / "mortality_by_age_weekly_labeled.parquet"


def load_total() -> pd.DataFrame:
    """Load the region-labeled total mortality + weather weekly view.

    Grain: one row per (geo, year, week). This is the view the RR,
    regression, and trend analyses run on.

    Returns:
        DataFrame with geo/region_name/macrozone/country labels, deaths,
        population, weather columns, any_heatwave_week, and a derived `month`
        column (added here so every caller gets it consistently).

    Raises:
        FileNotFoundError: If the view doesn't exist yet - loud on purpose,
        so the user knows to run build_analysis_view.py rather than seeing a
        confusing empty result.
    """
    if not TOTAL_VIEW.exists():
        raise FileNotFoundError(
            f"Missing {TOTAL_VIEW.name}. Run `python -m src.transform.build_analysis_view` first."
        )
    df = pd.read_parquet(TOTAL_VIEW)
    df["month"] = pd.to_datetime(df["week_start_date"]).dt.month
    return df


def load_age_stratified() -> pd.DataFrame:
    """Load the age-group-stratified mortality view (stratified_heat scheme).

    Grain: one row per (geo, sex, age_group, year, week), with age_group
    in {under_65, 65-74, 75-84, 85+}. Sex is M/F only (no 'T' total), so
    callers summing across sex are not double-counting.

    This coarse 4-bin view is for effect-modification analysis (does the heat
    effect differ by age?), NOT for ESP2013 standardization - those bins don't
    line up with the quinquennial ESP2013 weight table. Use
    load_age_quinquennial() for standardization.

    Returns:
        DataFrame with labels, age_group, sex, deaths, population,
        any_heatwave_week, population_is_missing.

    Raises:
        FileNotFoundError: If the view doesn't exist yet.
    """
    if not AGE_STRATIFIED_VIEW.exists():
        raise FileNotFoundError(
            f"Missing {AGE_STRATIFIED_VIEW.name}. Run "
            f"`python -m src.transform.build_analysis_view` first."
        )
    return pd.read_parquet(AGE_STRATIFIED_VIEW)


def load_age_quinquennial() -> pd.DataFrame:
    """Load the age view with the fine-grained quinquennial ESP2013 age bands.

    Grain: one row per (geo, sex, age, year, week), with `age` in the 18
    five-year ESP2013 bands (Y_LT5 ... Y_GE85). This is the view direct age
    standardization runs on: ESP2013 weights are defined per quinquennial band,
    so the 4-bin stratified_heat view CANNOT be ESP2013-standardized (its coarse
    bins don't line up with the weight table). Effect-modification analysis uses
    the stratified view; standardization uses this one.

    Returns:
        DataFrame with region labels, age (quinquennial), sex, deaths,
        population, any_heatwave_week, population_is_missing.

    Raises:
        FileNotFoundError: If the view doesn't exist yet - loud on purpose,
        so the user knows to run build_analysis_view.py.
    """
    if not AGE_QUINQUENNIAL_VIEW.exists():
        raise FileNotFoundError(
            f"Missing {AGE_QUINQUENNIAL_VIEW.name}. Run "
            f"`python -m src.transform.build_analysis_view` first."
        )
    return pd.read_parquet(AGE_QUINQUENNIAL_VIEW)