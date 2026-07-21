"""
Reusable UI components for the dashboard.

These are pure presentation helpers - they take already-computed numbers or
DataFrames and render Streamlit/Plotly output. They contain no epidemiological
logic (that stays in src/analysis/epi_metrics.py) and no data loading (that
stays in dashboard/views_loader.py). The point is to keep app.py readable: the
app orchestrates (load -> filter -> compute -> render), and delegates the
"how it looks" to this module.

Design conventions enforced here, so they're consistent everywhere:
- Red = higher risk / more deaths, everywhere. Never red-for-good.
- RR is always shown WITH a plain-language verdict, never as a bare number,
  so a non-expert isn't left to interpret "1.017" on their own.
- The RR=1 reference (no effect) is always visible on RR charts.
"""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Single place for the risk color scale, so every chart matches.
RISK_SCALE = "RdBu_r"  # reversed: high value -> red, low -> blue
RISK_RED = "#c0392b"


def rr_verdict(rr: float) -> tuple[str, str]:
    """Turn a relative-risk number into a plain-language verdict + severity.

    Returns (message, severity) where severity is one of "high", "elevated",
    "neutral", "protective-artifact" - used to pick color/icon. The wording is
    deliberately cautious: this is all-cause mortality in an ecological study,
    so we describe association, never causation, and flag implausible RR<1 as a
    likely artifact (the seasonal-confounding lesson from the notebook) rather
    than "heat protects".

    Args:
        rr: Relative risk (heatwave vs non-heatwave weeks).

    Returns:
        (message, severity) tuple.
    """
    pct = (rr - 1) * 100
    if rr >= 1.10:
        return (f"Heatwave weeks show **{pct:.0f}% higher** mortality — a substantial association.", "high")
    if rr >= 1.02:
        return (f"Heatwave weeks show **{pct:.1f}% higher** mortality — modest but present.", "elevated")
    if rr >= 0.98:
        return (f"Mortality in heatwave weeks is **about the same** ({pct:+.1f}%) — no clear association at this aggregation.", "neutral")
    return (
        f"Apparent **{pct:.1f}%** *lower* mortality in heatwave weeks — implausible for heat, "
        "and usually a sign of residual seasonal confounding (try the warm-season filter).",
        "protective-artifact",
    )


def show_rr_headline(result: dict, context_label: str = "") -> None:
    """Render the RR result as a metric row + a colored verdict banner.

    Args:
        result: dict from the RR helper, with keys rr, risk_exposed,
            risk_unexposed, n_exposed_weeks, n_unexposed_weeks.
        context_label: optional short label of what subset this describes
            (e.g. "Italy", "Lombardia (ITC4)"), shown in the caption.
    """
    c1, c2, c3 = st.columns(3)
    delta_pct = (result["rr"] - 1) * 100
    c1.metric(
        "Relative Risk",
        f"{result['rr']:.3f}",
        delta=f"{delta_pct:+.1f}% vs no heatwave",
        delta_color="inverse",  # higher mortality is "bad" -> red delta
    )
    c2.metric("Risk in heatwave weeks", f"{result['risk_exposed']*100_000:.1f} /100k")
    c3.metric("Risk in other weeks", f"{result['risk_unexposed']*100_000:.1f} /100k")

    message, severity = rr_verdict(result["rr"])
    if severity == "high":
        st.error(message, icon="🔴")
    elif severity == "elevated":
        st.warning(message, icon="🟠")
    elif severity == "neutral":
        st.info(message, icon="⚪")
    else:
        st.warning(message, icon="⚠️")

    caption = (
        f"{result['n_exposed_weeks']:,} heatwave region-weeks vs "
        f"{result['n_unexposed_weeks']:,} other region-weeks."
    )
    if context_label:
        caption = f"**{context_label}** — " + caption
    st.caption(caption)


def show_kpi_header(
    n_regions: int, n_deaths: int, date_span: str, pooled_rr: float | None
) -> None:
    """Top-of-page KPI strip, so the app answers "what is this?" before any
    filter is touched.

    Args:
        n_regions: Number of NUTS2 regions in the data.
        n_deaths: Total deaths covered.
        date_span: Human string like "2015-2025".
        pooled_rr: Headline season-corrected RR, or None if unavailable.
    """
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Regions", f"{n_regions}")
    k2.metric("Deaths analyzed", f"{n_deaths/1e6:.1f}M" if n_deaths >= 1e6 else f"{n_deaths:,}")
    k3.metric("Period", date_span)
    if pooled_rr is not None:
        k4.metric("Overall heatwave RR", f"{pooled_rr:.3f}",
                  delta=f"{(pooled_rr-1)*100:+.1f}%", delta_color="inverse")
    else:
        k4.metric("Overall heatwave RR", "n/a")


def rr_map(rr_by_region_df, coords_df, focus_center: dict | None = None, focus_zoom: float | None = None) -> None:
    """Scatter-on-map of RR by region: color = RR (red=higher), size = exposure.

    Uses capital-city coordinates (from locations.yaml via views_loader) rather
    than NUTS2 polygon boundaries - no external GeoJSON dependency, no
    code-matching fragility, and it reads clearly for 59 points. This is a
    deliberate engineering choice, not a fallback.

    Args:
        rr_by_region_df: DataFrame with columns geo, region_name, country, RR,
            n_exposed_weeks.
        coords_df: DataFrame with columns geo, lat, lon.
        focus_center: optional {"lat": ..., "lon": ...} to recenter the map
            (e.g. on a selected country). None = default European view.
        focus_zoom: optional zoom level to pair with focus_center.

    The diverging color scale is always anchored at RR=1 (color_continuous_midpoint),
    so a filtered view (e.g. one country) stays color-comparable with the full
    map instead of silently recalibrating its reds and blues to the subset.
    """
    merged = rr_by_region_df.merge(coords_df, on="geo", how="inner")
    if merged.empty:
        st.info("No regions with both an RR and coordinates to map.")
        return

    center = focus_center if focus_center else {"lat": 43.0, "lon": 5.0}
    zoom = focus_zoom if focus_zoom is not None else 3.2

    fig = px.scatter_map(
        merged,
        lat="lat",
        lon="lon",
        color="RR",
        size="n_exposed_weeks",
        color_continuous_scale=RISK_SCALE,
        color_continuous_midpoint=1.0,  # anchor the diverging scale at RR=1
        hover_name="region_name",
        hover_data={"country": True, "RR": ":.3f", "lat": False, "lon": False},
        size_max=22,
        zoom=zoom,
        center=center,
        height=520,
    )
    fig.update_layout(
        map_style="carto-positron",
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(title="RR"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Color = relative risk (red = higher mortality in heatwave weeks, blue = lower, "
        "anchored at RR=1 so views stay comparable). Bubble size = heatwave weeks observed. "
        "Placed at each region's capital city."
    )


def method_expander() -> None:
    """A collapsible 'about the method' block, so caveats live in the UI
    instead of being lost. Same limitations stated in the notebook."""
    with st.expander("About this analysis & its limitations"):
        st.markdown(
            """
            **What this is.** An *ecological time-series* study: it relates weekly
            all-cause mortality to temperature extremes across NUTS2 regions. It
            does **not** estimate heat-*attributable* deaths — that needs a
            counterfactual model (out of scope here).

            **Relative Risk (RR).** Mortality rate in heatwave weeks ÷ mortality
            rate in other weeks. RR > 1 means higher mortality when heatwaves occur.

            **Why the "warm season" filter matters.** Heatwaves only happen in
            warmer months, while winter has the year's highest baseline mortality
            (flu, cold) for reasons unrelated to heat. Comparing across the whole
            year makes heat look falsely *protective* (RR < 1). Restricting both
            groups to March–November removes that seasonal confounding.

            **The lag slider.** Heat effects can be delayed 1–2 weeks
            ("harvesting"). A lag of 0 (same week) likely *underestimates* the
            true effect. This is exposure re-labeling, not a full lag model.

            **Reading single regions.** Small regions have far fewer weeks, so
            their RR is noisier — treat region-level numbers as exploratory.
            """
        )