"""
Streamlit dashboard for the heatwave-mortality analysis.

Thin presentation layer. All epidemiological calculations come from
src/analysis/epi_metrics.py (the same functions the notebook uses); all data
access goes through dashboard/data_access.py; all reusable UI rendering lives
in dashboard/components.py. This file's job is orchestration only:
load -> read global filters -> filter -> compute -> hand to a component.

Run from the repo root:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analysis.epi_metrics import (
    relative_risk, restrict_to_warm_season, linear_fit, excess_mortality,
    age_specific_rates, direct_standardize, ESP2013_WEIGHTS,
)
from src.utils.age_bins_loader import load_config as load_age_bins_config, get_bin_labels
from dashboard.data_access import (
    load_total,
    load_age_stratified,
    load_age_quinquennial,
    region_options,
    region_coordinates,
)
from dashboard.components import (
    show_rr_headline,
    show_kpi_header,
    rr_map,
    method_expander,
    RISK_SCALE,
    RISK_RED,
)

st.set_page_config(page_title="Heatwave & Mortality", page_icon="🌡️", layout="wide")


# ==========================================================================
# UI-side aggregation helpers (no epi math here - that's in epi_metrics)
# ==========================================================================

def rr_from_frame(frame: pd.DataFrame) -> dict | None:
    """Split into heatwave/other weeks and compute RR; None if not computable."""
    exp = frame[frame["any_heatwave_week"] == True]
    nonexp = frame[frame["any_heatwave_week"] == False]
    if len(exp) == 0 or len(nonexp) == 0:
        return None
    try:
        base = relative_risk(
            exposed_deaths=exp["deaths"].sum(),
            exposed_population=exp["population"].sum(),
            unexposed_deaths=nonexp["deaths"].sum(),
            unexposed_population=nonexp["population"].sum(),
        )
        base["n_exposed_weeks"] = len(exp)
        base["n_unexposed_weeks"] = len(nonexp)
        return base
    except ValueError:
        return None

def rr_standardized_for_geo(frame_18bin: pd.DataFrame) -> dict | None:
    """Age-standardized (ESP2013) heatwave RR for a single region.

    Takes the 18-bin quinquennial frame for ONE geo, already method-prepped
    (window/lag/season applied). Collapses sex (M+F, no 'T' so no double count),
    then for each arm (heatwave vs other weeks) computes an age-specific rate as
    deaths per person-week at risk, standardizes it with ESP2013 direct
    weighting, and returns the ratio of the two standardized rates. None if
    either arm has no rows.

    Rate = total deaths in the arm / total person-weeks at risk in the arm
    (person-weeks = sum of weekly population over the arm's rows). This makes
    the two arms comparable even though they have very different week counts -
    unlike an annualized rate, which would divide by calendar years and so
    understate the arm with fewer weeks (the heatwave arm), producing a
    spuriously low RR.

    Standardization matters here because this is a cross-region comparison:
    without it, a region's RR could differ just because its population is older,
    not because heat hits harder. ESP2013 puts every region on the same age
    structure so the comparison is fair.
    """
    exp = frame_18bin[frame_18bin["any_heatwave_week"] == True]
    nonexp = frame_18bin[frame_18bin["any_heatwave_week"] == False]
    if len(exp) == 0 or len(nonexp) == 0:
        return None

    def _std_rate(arm: pd.DataFrame) -> float | None:
        # Per age band: deaths per person-week at risk. Summing population over
        # the arm's weekly rows gives person-weeks, so the denominator scales
        # with how many weeks the arm actually has - making the two arms
        # comparable. Sex is collapsed by the sum (M+F).
        by_age = arm.groupby("age", as_index=False).agg(
            deaths=("deaths", "sum"),
            person_weeks=("population", "sum"),
        )
        by_age = by_age[by_age["person_weeks"] > 0]
        if by_age.empty:
            return None
        by_age["rate"] = by_age["deaths"] / by_age["person_weeks"]

        # Direct ESP2013 standardization: weighted average of age-specific
        # rates using the standard population weights, scaled to per-100k.
        by_age["weight"] = by_age["age"].map(ESP2013_WEIGHTS)
        missing = by_age[by_age["weight"].isna()]["age"].tolist()
        if missing:
            raise ValueError(f"No ESP2013 weight for age code(s): {sorted(missing)}")
        total_weight = by_age["weight"].sum()
        std_rate = (by_age["rate"] * by_age["weight"]).sum() / total_weight * 100_000
        return std_rate

    try:
        exp_rate = _std_rate(exp)
        nonexp_rate = _std_rate(nonexp)
    except (ValueError, KeyError, IndexError):
        return None

    if not exp_rate or not nonexp_rate:
        return None

    return {
        "rr": exp_rate / nonexp_rate,
        "std_rate_exposed_per_100k": exp_rate,
        "std_rate_unexposed_per_100k": nonexp_rate,
        "n_exposed_weeks": len(exp),
    }

def apply_lag(frame: pd.DataFrame, lag_weeks: int) -> pd.DataFrame:
    """Shift the heatwave flag forward by lag_weeks within each region."""
    if lag_weeks == 0:
        return frame
    frame = frame.sort_values(["geo", "year", "week"]).copy()
    frame["any_heatwave_week"] = frame.groupby("geo")["any_heatwave_week"].shift(lag_weeks)
    return frame.dropna(subset=["any_heatwave_week"])


def rr_table_by(frame: pd.DataFrame, by_col: str) -> pd.DataFrame:
    """Compute RR for each value of by_col; return tidy DataFrame."""
    rows = []
    for key, group in frame.groupby(by_col):
        r = rr_from_frame(group)
        if r:
            rows.append({by_col: key, "RR": r["rr"], "n_exposed_weeks": r["n_exposed_weeks"]})
    return pd.DataFrame(rows)


def map_focus(coords_df: pd.DataFrame, geos: list[str]) -> tuple[dict | None, float | None]:
    """Compute a map center + zoom that frames the given set of geo codes.

    Used so selecting a country (or region) in the sidebar recenters and zooms
    the map onto it. Returns (None, None) when geos is empty or coordinates are
    missing, which tells rr_map to use its default European view.

    Zoom is picked from the geographic span of the selected points: a single
    point zooms in tight, a whole country zooms to fit. This is a heuristic
    (Web Mercator zoom isn't linear in degrees), tuned to look reasonable for
    the country spans in this dataset rather than to be geodetically exact.
    """
    if not geos:
        return None, None
    pts = coords_df[coords_df["geo"].isin(geos)]
    if pts.empty:
        return None, None
    center = {"lat": float(pts["lat"].mean()), "lon": float(pts["lon"].mean())}
    if len(pts) == 1:
        return center, 6.0
    span = max(pts["lat"].max() - pts["lat"].min(), pts["lon"].max() - pts["lon"].min())
    # Rough span-to-zoom mapping: larger span -> lower zoom.
    if span > 8:
        zoom = 4.5
    elif span > 4:
        zoom = 5.0
    elif span > 2:
        zoom = 5.5
    else:
        zoom = 6.0
    return center, zoom


# ==========================================================================
# Load data (cached) - fail friendly if views are missing
# ==========================================================================

try:
    total = load_total()
    age_stratified = load_age_stratified()
    age_quinquennial = load_age_quinquennial()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Generate the analysis views first, then reload this page.")
    st.stop()

regions = region_options(total)
try:
    coords = region_coordinates()
except Exception:
    coords = pd.DataFrame(columns=["geo", "lat", "lon"])  # map just no-ops if unavailable

age_bins_config = load_age_bins_config()
age_group_order = get_bin_labels(age_bins_config, "stratified_heat")


# ==========================================================================
# Sidebar: GLOBAL filters that every tab respects (cross-filtering)
# ==========================================================================

st.sidebar.title("Controls")
st.sidebar.caption("These filters apply across all tabs.")

st.sidebar.subheader("Geography")
country_sel = st.sidebar.selectbox(
    "Country", ["All countries"] + sorted(regions["country"].unique())
)
region_pool = regions if country_sel == "All countries" else regions[regions["country"] == country_sel]

region_sel = st.sidebar.selectbox(
    "Region",
    ["All regions"] + region_pool["region_name"].tolist(),
    help="Pick one NUTS2 region to focus every tab on it. Single-region figures are noisier.",
)
geo_sel = None
if region_sel != "All regions":
    geo_sel = region_pool[region_pool["region_name"] == region_sel]["geo"].iloc[0]

st.sidebar.subheader("Method")

# Time-window slider by real week dates (double-ended). Built from the sorted
# distinct week_start_date values so the labels are readable calendar dates,
# not raw week numbers. Applied before the season/lag logic in method_prep.
_all_weeks = sorted(pd.to_datetime(total["week_start_date"]).dt.date.unique())
week_range = st.sidebar.select_slider(
    "Time window",
    options=_all_weeks,
    value=(_all_weeks[0], _all_weeks[-1]),
    format_func=lambda d: d.strftime("%b %Y"),
    help="Restrict the analysis to a range of weeks. Narrow windows have fewer "
         "heatwave weeks, so figures get noisier.",
)

season_restricted = st.sidebar.toggle(
    "Warm season only (Mar-Nov)",
    value=True,
    help="Removes winter-baseline confounding. Off = the naive (confounded) comparison.",
)
lag_weeks = st.sidebar.slider(
    "Heatwave lag (weeks)", 0, 4, 0,
    help="Attribute deaths to a heatwave N weeks earlier. Literature suggests 1-2 weeks.",
)

st.sidebar.divider()



def geo_filter(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Apply the global country/region selection. Returns (frame, label)."""
    if geo_sel is not None:
        return frame[frame["geo"] == geo_sel], f"{region_sel} ({geo_sel})"
    if country_sel != "All countries":
        return frame[frame["country"] == country_sel], country_sel
    return frame, "All regions"


def apply_window(frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict a frame to the sidebar's selected week window.

    Separated from method_prep so tabs that must NOT apply the season
    restriction (the temperature-mortality regression, which needs the winter
    arm of the U-shaped curve) can still respect the time slider. Both this and
    method_prep read the same week_range global, so the window behaves
    identically everywhere it's applied.
    """
    start, end = week_range
    wk = pd.to_datetime(frame["week_start_date"]).dt.date
    return frame[(wk >= start) & (wk <= end)]


def method_prep(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply time window, then lag, then season restriction.

    Order matters: the time window narrows which weeks exist at all; lag then
    re-labels exposure by shifting within each region; the season restriction
    filters months last. The three controls are orthogonal (years-window vs
    exposure-shift vs months-filter) so they compose cleanly.
    """
    frame = apply_window(frame)
    frame = apply_lag(frame, lag_weeks)
    if season_restricted:
        frame = restrict_to_warm_season(frame, date_col="week_start_date")
    return frame


# ==========================================================================
# Header + KPI strip
# ==========================================================================

st.title("Heatwave & Mortality in Southern Europe")
st.caption(
    "Weekly all-cause mortality vs. temperature extremes across 59 NUTS2 regions "
    "in France, Italy & Spain (2015-2025). An exploratory, ecological analysis."
)

_valid_all = total[~total["population_is_missing"]].copy()
# Headline reflects the current time window + season/lag settings, so the KPI
# strip stays consistent with what the tabs below show (method_prep applies the
# window, lag, and season restriction together).
_headline = rr_from_frame(method_prep(_valid_all))
_years = pd.to_datetime(total["week_start_date"]).dt.year
_date_span = f"{_years.min()}-{_years.max()}"
show_kpi_header(
    n_regions=total["geo"].nunique(),
    n_deaths=int(total["deaths"].sum()),
    date_span=_date_span,
    pooled_rr=_headline["rr"] if _headline else None,
)
method_expander()

tab_overview, tab_rr, tab_age, tab_compare, tab_excess, tab_regression, tab_trend = st.tabs(
    ["Overview", "Relative Risk", "By Age", "Compare Regions", "Excess Mortality", "Temp vs Mortality", "Trend"]
)


# ==========================================================================
# TAB: Overview - the landing experience (map + headline verdict)
# ==========================================================================

with tab_overview:
    focused, label = geo_filter(_valid_all)
    prepared = method_prep(focused)
    result = rr_from_frame(prepared)

    col_map, col_verdict = st.columns([3, 2], gap="large")

    with col_map:
        st.subheader("Relative risk by region")
        map_prepared = method_prep(_valid_all)
        rr_region_map_df = rr_table_by(map_prepared, "geo")
        if not rr_region_map_df.empty:
            rr_region_map_df = rr_region_map_df.merge(
                regions[["geo", "region_name", "country"]], on="geo", how="left"
            )
            # User chose "zoom + only that country": filter the mapped points to
            # the current geography selection, and frame the view on them. The
            # color scale stays anchored at RR=1 (in rr_map) so the filtered view
            # is still color-comparable with the full-Europe view.
            if geo_sel is not None:
                map_df = rr_region_map_df[rr_region_map_df["geo"] == geo_sel]
                focus_geos = [geo_sel]
            elif country_sel != "All countries":
                map_df = rr_region_map_df[rr_region_map_df["country"] == country_sel]
                focus_geos = map_df["geo"].tolist()
            else:
                map_df = rr_region_map_df
                focus_geos = []
            center, zoom = map_focus(coords, focus_geos)
            rr_map(map_df, coords, focus_center=center, focus_zoom=zoom)
        else:
            st.info("Not enough data to map at the current settings.")

    with col_verdict:
        st.subheader("At a glance")
        if result is None:
            st.warning(f"Not enough data for **{label}** at these settings.")
        else:
            show_rr_headline(result, context_label=label)

        if geo_sel is None and country_sel == "All countries":
            st.markdown("**By country**")
            rr_country = rr_table_by(prepared, "country").sort_values("RR", ascending=False)
            if not rr_country.empty:
                fig = px.bar(
                    rr_country, x="RR", y="country", orientation="h",
                    color="RR", color_continuous_scale=RISK_SCALE, color_continuous_midpoint=1.0,
                    text=rr_country["RR"].round(3),
                )
                fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
                fig.update_layout(showlegend=False, coloraxis_showscale=False,
                                  height=220, margin=dict(l=0, r=0, t=10, b=0),
                                  yaxis_title="", xaxis_title="RR")
                st.plotly_chart(fig, width="stretch")


# ==========================================================================
# TAB: Relative Risk - pooled + regional breakdown (respects global filter)
# ==========================================================================

with tab_rr:
    st.subheader("Relative risk: heatwave vs. other weeks")

    focused, label = geo_filter(_valid_all)
    prepared = method_prep(focused)

    if geo_sel is not None:
        result = rr_from_frame(prepared)
        if result is None:
            st.warning(f"Not enough data for {label}.")
        else:
            show_rr_headline(result, context_label=label)
    else:
        result = rr_from_frame(prepared)
        if result:
            show_rr_headline(result, context_label=label)
        st.divider()
        st.markdown("**Breakdown by region** (within current selection)")
        by_region = rr_table_by(prepared, "geo").merge(
            regions[["geo", "region_name", "country"]], on="geo", how="left"
        ).sort_values("RR", ascending=False)
        if by_region.empty:
            st.info("No regions to break down at these settings.")
        else:
            top = by_region.head(25)
            fig = px.bar(
                top, x="RR", y="region_name", orientation="h",
                color="RR", color_continuous_scale=RISK_SCALE, color_continuous_midpoint=1.0,
                hover_data=["country", "n_exposed_weeks"],
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
            fig.update_layout(height=max(300, 22 * len(top)),
                              yaxis_title="", xaxis_title="Relative Risk",
                              coloraxis_showscale=False,
                              yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, width="stretch")
            st.caption("Showing up to 25 regions, highest RR first. Dashed line = no effect (RR=1).")


# ==========================================================================
# TAB: By Age - effect modification (Section 7), respects global filter
# ==========================================================================

with tab_age:
    st.subheader("Does heat affect age groups differently?")
    st.caption("Hypothesis: vulnerability rises with age, so RR should climb from under-65 to 85+.")

    valid_age = age_stratified[~age_stratified["population_is_missing"]].copy()
    by_group = (
        valid_age.groupby(
            ["geo", "country", "age_group", "year", "week", "week_start_date", "any_heatwave_week"],
            as_index=False,
        ).agg(deaths=("deaths", "sum"), population=("population", "sum"))
    )

    if geo_sel is not None:
        by_group = by_group[by_group["geo"] == geo_sel]
        age_label = f"{region_sel} ({geo_sel})"
    elif country_sel != "All countries":
        by_group = by_group[by_group["country"] == country_sel]
        age_label = country_sel
    else:
        age_label = "All regions"

    prepared_age = method_prep(by_group)

    rows = []
    for ag in age_group_order:
        r = rr_from_frame(prepared_age[prepared_age["age_group"] == ag])
        if r:
            rows.append({"age_group": ag, "RR": r["rr"],
                         "risk_exposed_per_100k": r["risk_exposed"] * 100_000,
                         "n_exposed_weeks": r["n_exposed_weeks"]})

    if not rows:
        st.warning(f"Not enough data for {age_label} at these settings.")
    else:
        rr_age = pd.DataFrame(rows)
        fig = px.bar(
            rr_age, x="age_group", y="RR", text=rr_age["RR"].round(3),
            category_orders={"age_group": age_group_order},
            color="RR", color_continuous_scale=RISK_SCALE, color_continuous_midpoint=1.0,
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(yaxis_title="Relative Risk", xaxis_title="Age group",
                          coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")

        rr_vals = rr_age.set_index("age_group").reindex(age_group_order)["RR"].dropna()
        is_monotonic = all(rr_vals.iloc[i] <= rr_vals.iloc[i + 1] for i in range(len(rr_vals) - 1))
        if is_monotonic and len(rr_vals) >= 3:
            st.success(f"**{age_label}:** RR rises with age - consistent with the vulnerability hypothesis.", icon="✅")
        else:
            st.info(f"**{age_label}:** the age gradient isn't cleanly monotonic here - possibly diluted by "
                    "all-cause mortality, small counts, or the no-lag setting. Worth noting, not a null result.", icon="ℹ️")
        st.dataframe(rr_age, width="stretch", hide_index=True)


# ==========================================================================
# TAB: Compare Regions - side-by-side RR for a user-picked set of regions
# ==========================================================================

with tab_compare:
    st.subheader("Compare regions side by side")
    st.caption("Pick any set of regions - across countries if you like - to compare their "
               "age-standardized (ESP2013) heatwave RR under the current time-window, lag, "
               "and season settings from the sidebar.")

    with st.expander("Why age-standardized here? (ESP2013)"):
        st.markdown(
            "Comparing regions is the one place standardization matters. A region's raw RR "
            "could look higher simply because its population is older and more heat-vulnerable, "
            "not because heat hits harder there. Direct standardization recomputes each region's "
            "mortality rate as if it had the **same** age structure (the European Standard "
            "Population 2013), using the 18 quinquennial age bands, so the comparison reflects "
            "the heat effect rather than demographic differences.\n\n"
            "This is deliberately **different** from the *By Age* tab, which keeps the age bands "
            "*unstandardized* on purpose - there the goal is to show how the effect *varies* by "
            "age (effect modification), not to average it away."
        )

    # This tab has its OWN region multiselect, independent of the global single
    # geography selector, so cross-country comparisons (e.g. Lombardia vs Madrid)
    # aren't blocked by the sidebar's country filter. It still respects the
    # global METHOD settings (window/lag/season) via method_prep.
    region_labels = (regions["region_name"] + "  (" + regions["country"] + ")").tolist()
    label_to_geo = dict(zip(region_labels, regions["geo"]))

    default_pick = region_labels[: min(4, len(region_labels))]
    picked = st.multiselect(
        "Regions to compare",
        options=region_labels,
        default=default_pick,
        help="Two or more regions. Single-region figures are noisier the smaller the region.",
    )

    if len(picked) < 2:
        st.info("Pick at least two regions to compare.")
    else:
        picked_geos = [label_to_geo[label] for label in picked]

        # Standardization needs the 18-bin quinquennial view, not the 4-bin one.
        aq_valid = age_quinquennial[~age_quinquennial["population_is_missing"]].copy()
        subset = aq_valid[aq_valid["geo"].isin(picked_geos)]
        prepared_cmp = method_prep(subset)

        rows = []
        for geo in picked_geos:
            r = rr_standardized_for_geo(prepared_cmp[prepared_cmp["geo"] == geo])
            meta = regions[regions["geo"] == geo].iloc[0]
            if r:
                rows.append({
                    "Region": meta["region_name"],
                    "Country": meta["country"],
                    "geo": geo,
                    "RR (ESP2013)": r["rr"],
                    "Std. rate, heatwave /100k": r["std_rate_exposed_per_100k"],
                    "Std. rate, other /100k": r["std_rate_unexposed_per_100k"],
                    "Heatwave weeks": r["n_exposed_weeks"],
                })

        if not rows:
            st.warning("None of the selected regions have enough data at the current settings.")
        else:
            cmp_df = pd.DataFrame(rows).sort_values("RR (ESP2013)", ascending=False)

            # Side-by-side RR bars, one per region, shared RR=1 reference and the
            # same RR-anchored color scale as the map, so colors mean the same thing.
            fig = px.bar(
                cmp_df, x="RR (ESP2013)", y="Region", orientation="h",
                color="RR (ESP2013)", color_continuous_scale=RISK_SCALE, color_continuous_midpoint=1.0,
                text=cmp_df["RR (ESP2013)"].round(3), hover_data=["Country", "Heatwave weeks"],
            )
            fig.add_vline(x=1.0, line_dash="dash", line_color="gray", annotation_text="RR = 1")
            fig.update_layout(
                height=max(240, 46 * len(cmp_df)),
                yaxis_title="", xaxis_title="Age-standardized Relative Risk (ESP2013)",
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),  # highest RR on top
                margin=dict(l=0, r=0, t=10, b=0),
            )
            st.plotly_chart(fig, width="stretch")

            # A one-line takeaway naming the extremes, so the comparison has a verdict.
            top = cmp_df.iloc[0]
            bottom = cmp_df.iloc[-1]
            st.caption(
                f"Highest age-standardized heatwave association: **{top['Region']}** "
                f"(RR {top['RR (ESP2013)']:.3f}); lowest: **{bottom['Region']}** "
                f"(RR {bottom['RR (ESP2013)']:.3f}). Differences across small regions can "
                "reflect noise as much as real variation."
            )

            st.dataframe(
                cmp_df.drop(columns="geo").round(3),
                width="stretch",
                hide_index=True,
            )



# ==========================================================================
# TAB: Excess Mortality - descriptive excess vs same-ISO-week baseline
# ==========================================================================

with tab_excess:
    st.subheader("Excess mortality during heatwave weeks")
    st.caption("How many more deaths occurred during heatwave weeks than in the same ISO week of "
               "years without a heatwave. A descriptive complement to Relative Risk - the "
               "absolute, more communicable figure.")

    st.warning(
        "This is **descriptive, not causal**: the excess is measured against a same-week, "
        "non-heatwave baseline and is *not* a count of heat-attributable deaths. Part may reflect "
        "other factors, and the baseline doesn't adjust for the long-term population trend.",
        icon="⚠️",
    )

    # Respects the global geography selection, but NOT the season/lag method
    # controls: excess_mortality builds its own same-ISO-week baseline, which
    # already handles seasonality by construction. Applying the warm-season
    # filter on top would just shrink the sample without changing the logic.
    # The time-window slider IS applied, so you can focus on specific years.
    valid_excess = _valid_all.copy()
    valid_excess = apply_window(valid_excess)
    valid_excess, excess_label = geo_filter(valid_excess)

    if valid_excess.empty:
        st.warning(f"No data for {excess_label} in the selected window.")
    else:
        try:
            excess = excess_mortality(valid_excess)
        except ValueError as e:
            st.error(str(e))
            excess = None

        if excess is None or excess.empty:
            st.info("Not enough non-heatwave baseline weeks to compute excess for this selection.")
        else:
            # Headline totals for the current selection.
            total_excess = excess["excess_abs"].sum()
            total_baseline = excess["baseline_deaths"].sum()
            pct = (total_excess / total_baseline * 100) if total_baseline else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Excess deaths", f"{total_excess:,.0f}",
                      delta=f"{pct:+.1f}% vs baseline", delta_color="inverse")
            c2.metric("Baseline (expected) deaths", f"{total_baseline:,.0f}")
            c3.metric("Heatwave weeks", f"{len(excess):,}")

            # Excess deaths per year - surfaces which years drove it (e.g. 2022).
            by_year = (
                excess.groupby("year")
                .agg(excess_deaths=("excess_abs", "sum"),
                     baseline=("baseline_deaths", "sum"))
                .reset_index()
            )
            by_year["excess_pct"] = by_year["excess_deaths"] / by_year["baseline"] * 100

            fig = px.bar(
                by_year, x="year", y="excess_deaths",
                color="excess_deaths", color_continuous_scale=RISK_SCALE,
                color_continuous_midpoint=0.0,
                hover_data={"excess_pct": ":.1f"},
            )
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(
                xaxis_title="Year",
                yaxis_title="Excess deaths (vs same-week baseline)",
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(f"**{excess_label}** - a tall bar marks a year whose heatwave weeks far "
                       "exceeded a normal year's equivalent weeks. Negative bars are years where "
                       "flagged weeks fell below baseline: a reminder this is a noisy descriptive "
                       "measure, not a causal signal.")

            # Per-country breakdown, only meaningful when not already filtered to one.
            if geo_sel is None and country_sel == "All countries":
                by_country = (
                    excess.groupby("country")
                    .agg(excess_deaths=("excess_abs", "sum"),
                         baseline=("baseline_deaths", "sum"))
                    .reset_index()
                )
                by_country["excess_pct"] = by_country["excess_deaths"] / by_country["baseline"] * 100
                by_country = by_country.sort_values("excess_pct", ascending=False)
                st.markdown("**By country**")
                st.dataframe(
                    by_country.rename(columns={
                        "country": "Country", "excess_deaths": "Excess deaths",
                        "baseline": "Baseline deaths", "excess_pct": "Excess %",
                    }).round(1),
                    width="stretch", hide_index=True,
                )

# ==========================================================================
# TAB: Temp vs Mortality regression - full-year on purpose, respects geo
# ==========================================================================

with tab_regression:
    st.subheader("Temperature vs. mortality rate")
    st.caption("The time-window slider applies here, but the season filter is intentionally "
               "ignored, so both the cold and heat arms of the U-shaped relationship stay "
               "visible. A single line flattens both.")

    valid_reg = total[~total["population_is_missing"]].copy()
    valid_reg["crude_rate_per_100k"] = valid_reg["deaths"] / valid_reg["population"] * 100_000
    valid_reg = apply_window(valid_reg)  # window yes, season no (needs the winter arm)
    valid_reg, reg_label = geo_filter(valid_reg)

    if len(valid_reg) < 2 or valid_reg["temperature_2m_mean"].nunique() < 2:
        st.warning(f"Not enough data to fit a line for {reg_label}.")
    else:
        fit = linear_fit(valid_reg["temperature_2m_mean"], valid_reg["crude_rate_per_100k"])
        degree = st.slider("Polynomial degree", 1, 5, 3,
                       help="1 = straight line. 2 = symmetric U. 3 = asymmetric J (steeper heat "
                            "arm). Higher = risk of fitting noise. Watch R² vs curve wiggliness.")
        x_line = np.linspace(valid_reg["temperature_2m_mean"].min(), valid_reg["temperature_2m_mean"].max(), 100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=valid_reg["temperature_2m_mean"], y=valid_reg["crude_rate_per_100k"],
            mode="markers", marker=dict(size=4, opacity=0.12, color="#4a6fa5"),
            name="week x region",
        ))
        fig.add_trace(go.Scatter(
            x=x_line, y=fit["slope"] * x_line + fit["intercept"], mode="lines",
            line=dict(color=RISK_RED, width=3),
            name=f"fit: slope={fit['slope']:.3f}, r={fit['r']:.3f}",
        ))
        if degree >= 2:
            from src.analysis.epi_metrics import polynomial_fit
            try:
                pfit = polynomial_fit(valid_reg["temperature_2m_mean"],
                                      valid_reg["crude_rate_per_100k"], degree=degree)
                fig.add_trace(go.Scatter(
                    x=pfit["x_curve"], y=pfit["y_curve"], mode="lines",
                    line=dict(color="#e67e22", width=3),
                    name=f"degree {degree} (R²={pfit['r2']:.3f})",
                ))
            except ValueError as e:
                st.caption(f"Polynomial fit unavailable: {e}")
        fig.update_layout(xaxis_title="Weekly mean temperature (C)",
                          yaxis_title="Mortality rate (/100k)",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"**{reg_label}** - a positive slope reflects the heat arm; the cold arm bends the "
                   "true curve upward at low temperatures, which a straight line can't capture (the U/J "
                   "shape a DLNM would model - out of scope here).")


# ==========================================================================
# TAB: Trend
# ==========================================================================

with tab_trend:
    st.subheader("Weekly mortality over time")

    # Respect the global geography selection: a single region shows just that
    # region's series; a country shows its macrozones; otherwise all macrozones.
    trend_src = total.copy()
    if geo_sel is not None:
        trend_src = trend_src[trend_src["geo"] == geo_sel]
        st.caption(f"Weekly deaths in {region_sel} ({geo_sel}) within the selected time window. "
                   "Regular waves are seasonality (winter peaks), not heat - so the season "
                   "filter is not applied here.")
    else:
        if country_sel != "All countries":
            trend_src = trend_src[trend_src["country"] == country_sel]
        st.caption("Raw deaths per macrozone within the selected time window. Regular waves are "
                   "seasonality (winter peaks), not heat - so the season filter is not applied here.")

    trend_src = apply_window(trend_src)  # window only: lag/season don't apply to a raw death-count series

    if trend_src.empty:
        st.warning("No data available for the current selection and time window.")
    else:
        if geo_sel is not None:
            # Single region: one series for that region.
            trend = (
                trend_src.groupby("week_start_date", as_index=False)
                .agg(deaths=("deaths", "sum"))
            )
            trend["series"] = region_sel
            y_title = "Deaths (region total)"
        else:
            # Multiple regions: break down by country-macrozone.
            trend = (
                trend_src.groupby(["country", "macrozone", "week_start_date"], as_index=False)
                .agg(deaths=("deaths", "sum"))
            )
            trend["series"] = trend["country"] + " - " + trend["macrozone"]
            y_title = "Deaths (macrozone total)"

        fig = px.line(trend.sort_values("week_start_date"),
                      x="week_start_date", y="deaths", color="series")
        fig.update_layout(xaxis_title="Week", yaxis_title=y_title,
                          legend_title="")
        st.plotly_chart(fig, width="stretch")