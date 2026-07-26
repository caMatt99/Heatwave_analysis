"""
Shared epidemiological calculation functions - direct standardization and
relative risk - used by both the descriptive-analysis notebook and the
Streamlit dashboard.

Design principle: these functions operate on already-prepared, already-filtered
DataFrames or plain aggregated numbers. They do NOT decide which rows to
exclude (e.g. missing population) or what counts as "exposed" (heatwave
weeks, a season window, a country, an age group) those are analysis-specific
decisions made by the caller. Keeping that logic out of this module is what
lets the same relative_risk() power the notebook's naive RR, its
season-corrected RR, its per-country RR, AND a future per-age-group RR,
without this module needing to know about seasons, countries, or age at all.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score



# ESP2013 age-standard weights (source: Eurostat 2013, via PHEindicatormethods).
# Reconciled to 18 groups: Y_GE85 = original 85-89 (1500) + 90+ (1000), matching
# the Y85-89/Y_GE90 -> Y_GE85 reconciliation already applied in
# build_weekly_dataset_by_age.py.
ESP2013_WEIGHTS = {
    "Y_LT5": 5000, "Y5-9": 5500, "Y10-14": 5500, "Y15-19": 5500,
    "Y20-24": 6000, "Y25-29": 6000, "Y30-34": 6500, "Y35-39": 7000,
    "Y40-44": 7000, "Y45-49": 7000, "Y50-54": 7000, "Y55-59": 6500,
    "Y60-64": 6000, "Y65-69": 5500, "Y70-74": 5000, "Y75-79": 4000,
    "Y80-84": 2500,
    "Y_GE85": 1500 + 1000,
}
assert sum(ESP2013_WEIGHTS.values()) == 100_000, "ESP2013 weights must sum to 100,000"


def age_specific_rates(
    df: pd.DataFrame,
    group_cols: list[str],
    age_col: str = "age",
    deaths_col: str = "deaths",
    population_col: str = "population",
    year_col: str = "year",
) -> pd.DataFrame:
    """Aggregate multi-year deaths/population into stable age-specific rates.

    Aggregating across years (rather than standardizing week by week) avoids
    unstable rates from low weekly death counts in narrow age bands.

    Args:
        df: Input DataFrame, already filtered by the caller (e.g. rows with
            missing population excluded) - this function does not filter.
        group_cols: Columns identifying the comparison unit, e.g. ["geo", "region_name"].
        age_col: Column holding the age-band code.
        deaths_col: Column holding weekly death counts.
        population_col: Column holding population (assumed constant per year
            within a group/age; averaged here as a stable per-year figure).
        year_col: Column holding the calendar year, used to compute an
            average annual rate.

    Returns:
        DataFrame: one row per group_cols + age_col, with `age_specific_rate`
        (average annual deaths / average annual population).
    """
    agg = df.groupby(list(group_cols) + [age_col], as_index=False).agg(
        deaths=(deaths_col, "sum"),
        population=(population_col, "mean"),
        n_years=(year_col, "nunique"),
    )
    agg["avg_annual_deaths"] = agg["deaths"] / agg["n_years"]
    agg["avg_annual_population"] = agg["population"]
    agg["age_specific_rate"] = agg["avg_annual_deaths"] / agg["avg_annual_population"]
    return agg


def direct_standardize(
    rates_df: pd.DataFrame,
    weights: dict[str, float],
    group_cols: list[str],
    age_col: str = "age",
    rate_col: str = "age_specific_rate",
    standard_population: int = 100_000,
) -> pd.DataFrame:
    """Direct age standardization.

    expected_deaths = age_specific_rate x standard_weight, summed across ages
    and divided by the total standard weight, scaled to `standard_population`.

    Args:
        rates_df: Output of age_specific_rates() (or any DataFrame with the
            same shape: one row per group_cols + age_col).
        weights: Age code -> standard weight, e.g. ESP2013_WEIGHTS.
        group_cols: Columns identifying the comparison unit.
        age_col: Column holding the age-band code.
        rate_col: Column holding the age-specific rate to standardize.
        standard_population: Total of the standard population (100,000 for ESP2013).

    Returns:
        DataFrame: one row per group_cols, with `standardized_rate_per_100k`.

    Raises:
        ValueError: If rates_df contains an age code with no matching weight -
        fails loudly rather than silently dropping that age group's contribution.
    """
    missing = set(rates_df[age_col].unique()) - set(weights.keys())
    if missing:
        raise ValueError(
            f"No standard weight defined for age code(s): {sorted(missing)}. "
            f"Every age code in the data must have a matching ESP2013 weight."
        )

    def _standardize_one_group(group: pd.DataFrame) -> float:
        group_weights = group[age_col].map(weights)
        expected_deaths = group[rate_col] * group_weights
        return expected_deaths.sum() / group_weights.sum() * standard_population

    result = (
        rates_df.groupby(list(group_cols))
        .apply(_standardize_one_group, include_groups=False)
        .reset_index(name="standardized_rate_per_100k")
    )
    return result


def relative_risk(
    exposed_deaths: float,
    exposed_population: float,
    unexposed_deaths: float,
    unexposed_population: float,
) -> dict[str, float]:
    """Relative risk from already-aggregated exposed/unexposed sums.

    Deliberately takes plain numbers, not a DataFrame with an exposure column:
    the caller decides what "exposed" means (a heatwave flag, a season
    restriction, a specific country, a specific age group) by choosing what to
    sum before calling this. This is what lets the same function serve the
    notebook's naive RR, its season-corrected RR, its per-country RR, and a
    future per-age-group RR without any changes here.

    Args:
        exposed_deaths: Total deaths in the exposed group.
        exposed_population: Total population(-time) at risk in the exposed group.
        unexposed_deaths: Total deaths in the unexposed group.
        unexposed_population: Total population(-time) at risk in the unexposed group.

    Returns:
        dict: {"risk_exposed": ..., "risk_unexposed": ..., "rr": ...}

    Raises:
        ValueError: If either population is zero (risk undefined), or if
        unexposed risk is zero (RR undefined) - fails loudly rather than
        returning inf/NaN silently.
    """
    if exposed_population == 0 or unexposed_population == 0:
        raise ValueError("Cannot compute risk with zero population in a group.")

    risk_exposed = exposed_deaths / exposed_population
    risk_unexposed = unexposed_deaths / unexposed_population

    if risk_unexposed == 0:
        raise ValueError("Unexposed risk is zero; RR is undefined.")

    return {
        "risk_exposed": risk_exposed,
        "risk_unexposed": risk_unexposed,
        "rr": risk_exposed / risk_unexposed,
    }


def restrict_to_warm_season(
    df: pd.DataFrame, date_col: str = "week_start_date", start_month: int = 3, end_month: int = 11
) -> pd.DataFrame:
    """Restrict a DataFrame to weeks within a seasonal window (inclusive).

    Used to control for seasonal confounding in RR calculations: comparing
    heatwave vs non-heatwave weeks only makes sense within the same seasonal
    window, since winter months have no heatwaves and a much higher baseline
    mortality unrelated to heat.

    Args:
        df: Input DataFrame with a date column.
        date_col: Column holding a date or datetime-parseable value.
        start_month: First month to include (default 3 = March).
        end_month: Last month to include (default 11 = November).

    Returns:
        DataFrame: rows where the month of date_col falls in [start_month, end_month].
    """
    months = pd.to_datetime(df[date_col]).dt.month
    return df[months.between(start_month, end_month)]


def linear_fit(x: pd.Series, y: pd.Series) -> dict[str, float]:
    """Simple linear regression (least squares) with correlation coefficient.

    Extracted so the same fit logic can be reused for the pooled temperature-
    mortality scatter (all regions) and for a drill-down view restricted to
    one country or one region (e.g. an interactive Streamlit selector),
    without duplicating the numpy calls in each place.

    Args:
        x: Independent variable (e.g. weekly mean temperature).
        y: Dependent variable (e.g. crude mortality rate).

    Returns:
        dict: {"slope": ..., "intercept": ..., "r": ...} where r is the
        Pearson correlation coefficient between x and y.

    Raises:
        ValueError: If x and y have fewer than 2 points, or all-identical
        values in x (regression undefined) - fails loudly rather than
        returning NaN silently.
    """
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 points to fit a line.")
    if x.nunique() < 2:
        raise ValueError("x has no variation (all identical values); slope is undefined.")

    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    return {"slope": float(slope), "intercept": float(intercept), "r": float(r)}


def polynomial_fit(x: pd.Series, y: pd.Series, degree: int = 3) -> dict:
    """Polynomial regression of y on x, via a scikit-learn pipeline.

    Captures the non-linear (U/J-shaped) temperature-mortality relationship that
    a straight line flattens: cold and heat both raise mortality, with a minimum
    at some comfort temperature.
    The pipeline form scales cleanly and exposes R^2 for comparing degrees.

    Args:
        x: Independent variable (weekly mean temperature).
        y: Dependent variable (crude mortality rate).
        degree: Polynomial degree. 2 = symmetric U; 3 = asymmetric J (steeper
            heat arm); higher degrees risk fitting noise.

    Returns:
        dict with:
        - "x_curve", "y_curve": sorted x grid and fitted y, ready to plot as a line
        - "r2": coefficient of determination on the fitted data
        - "degree": the degree used (echoed back, for labels)
        - "model": the fitted sklearn pipeline (for predicting new points)

    Raises:
        ValueError: If fewer than degree+1 points, or x has no variation.
    """
    if len(x) < degree + 1 or len(y) < degree + 1:
        raise ValueError(f"Need at least {degree + 1} points to fit a degree-{degree} polynomial.")
    if x.nunique() < 2:
        raise ValueError("x has no variation (all identical values); fit is undefined.")

    X = x.to_numpy().reshape(-1, 1)
    y_arr = y.to_numpy()

    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(X, y_arr)

    r2 = r2_score(y_arr, model.predict(X))

    # Smooth curve over the sorted x-range for plotting.
    x_curve = np.linspace(x.min(), x.max(), 200)
    y_curve = model.predict(x_curve.reshape(-1, 1))

    return {"x_curve": x_curve, "y_curve": y_curve, "r2": float(r2),
            "degree": degree, "model": model}

def excess_mortality(
    df: pd.DataFrame,
    exposure_col: str = "any_heatwave_week",
    deaths_col: str = "deaths",
    geo_col: str = "geo",
    week_col: str = "week",
) -> pd.DataFrame:
    """Excess mortality vs a same-ISO-week, non-exposed baseline.

    For each (geo, ISO week), the baseline is the mean deaths in that same ISO
    week across the years when that week was NOT flagged as exposed (no
    heatwave). Excess is then observed - baseline, computed only on the exposed
    rows. This deliberately keeps the exposure out of its own baseline (the
    baseline is built from non-exposed years only), so the excess isn't diluted
    by the very weeks being measured.

    This is a DESCRIPTIVE excess measure, not a causal attribution: the excess
    during heatwave weeks is not wholly heat-attributable (a late flu peak or a
    local event could contribute), and this baseline does not adjust for the
    long-term population trend. It answers "how much higher was mortality than a
    normal year's same week", not "how many deaths did heat cause".

    Args:
        df: Weekly frame, already filtered by the caller (e.g. missing
            population removed). Must contain geo, week, deaths, and the
            exposure flag columns.
        exposure_col: Boolean column marking exposed (heatwave) weeks.
        deaths_col: Weekly death count column.
        geo_col: Region key column.
        week_col: ISO week-number column.

    Returns:
        DataFrame of the exposed rows with added columns:
        - baseline_deaths: mean deaths for that geo+ISO-week in non-exposed years
        - excess_abs: observed deaths - baseline_deaths
        - excess_rel: excess_abs / baseline_deaths (NaN where baseline is 0)
        Rows whose (geo, week) has no non-exposed year to form a baseline are
        dropped (no baseline is definable for them).

    Raises:
        ValueError: If required columns are missing.
    """
    required = {exposure_col, deaths_col, geo_col, week_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"excess_mortality: missing columns {sorted(missing)}.")

    # Baseline: mean deaths per (geo, ISO week) over NON-exposed rows only.
    non_exposed = df[df[exposure_col] == False]
    baseline = (
        non_exposed.groupby([geo_col, week_col])[deaths_col]
        .mean()
        .rename("baseline_deaths")
        .reset_index()
    )

    # Excess is defined on the exposed rows.
    exposed = df[df[exposure_col] == True].merge(baseline, on=[geo_col, week_col], how="left")

    # Drop exposed weeks with no non-exposed baseline (can't define an expected value).
    exposed = exposed.dropna(subset=["baseline_deaths"]).copy()

    exposed["excess_abs"] = exposed[deaths_col] - exposed["baseline_deaths"]
    exposed["excess_rel"] = exposed["excess_abs"] / exposed["baseline_deaths"].where(
        exposed["baseline_deaths"] != 0
    )
    return exposed


