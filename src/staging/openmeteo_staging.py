"""
Staging layer for Open-Meteo raw extracts.

Turns the per-(location, year) daily JSON cache files produced by
openmeteo_client.py into two tidy tables:

    daily table   -> (location, date, <daily vars>, is_interpolated,
                      is_hot_day, is_heatwave_day)
    weekly table  -> (location, year, week, week_start_date,
                      <weekly-aggregated vars>, n_days_available,
                      heatwave_days_in_week, any_heatwave_week)

Design decisions:
- Heat waves are a daily-consecutive phenomenon and are detected on the
  DAILY series, never on the weekly aggregate - a week with an ordinary
  mean can still contain a 3-day spike, which a weekly mean would hide.
  The result is then rolled up into a weekly count/flag.
- The heat wave threshold is climatological, not a fixed global value:
  the 90th percentile of temperature_2m_max, computed per
  (location, calendar month) over the whole available history. A day
  qualifies as "hot" if it exceeds this threshold; a "heatwave_day" is
  a hot day that is part of a run of 3+ consecutive hot days.
- Only temperature columns are interpolated for isolated missing days
  (temperature is continuous and strongly autocorrelated day-to-day).
  precipitation_sum, wind_speed_10m_max and sunshine_duration are left
  as NaN when missing, flagged, and simply excluded from that day's
  weekly sum/max - fabricating a rain or sunshine value from neighbours
  is a much weaker assumption than for temperature.
- Weekly aggregation uses a different function per column (mean/max/min/
  sum as appropriate), never a single blanket aggregation.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.staging.common import save_parquet

RAW_DIR = Path("data/raw/openmeteo")
STAGING_DIR = Path("data/staging/openmeteo")

# Columns that are safe to interpolate (continuous, autocorrelated).
INTERPOLATE_COLS = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"]
MAX_INTERPOLATION_GAP_DAYS = 2

# How each daily column rolls up to a weekly value.
WEEKLY_AGG_FUNCS = {
    "temperature_2m_mean": "mean",
    "temperature_2m_max": "max",
    "temperature_2m_min": "min",
    "precipitation_sum": "sum",
    "wind_speed_10m_max": "max",
    "sunshine_duration": "sum",
}

HEATWAVE_PERCENTILE = 0.90
HEATWAVE_MIN_CONSECUTIVE_DAYS = 3
MIN_YEARS_FOR_THRESHOLD = 5  # below this, a per-month percentile is not a reliable climatological baseline

# A per-month relative percentile alone flags ~10% of days in EVERY month,
# including winter (e.g. an unusually mild January day), producing "heat
# waves" spread evenly across the whole year instead of concentrated in
# summer. Requiring temperature_2m_max to also clear this absolute floor
# filters that out. 25.0 was chosen empirically from this project's own
# data: across all 59 locations, the highest winter (DJF) 90th-percentile
# max temperature is 22.0 C and the lowest summer (JJA) 90th-percentile
# max temperature is 26.3 C - 25.0 sits cleanly in that gap, so it never
# suppresses a genuine local summer extreme while still excluding every
# location's winter noise.
HEATWAVE_ABSOLUTE_MIN_TEMP = 25.0


def _load_one_file(path: Path) -> pd.DataFrame:
    """Load one (location, year) daily JSON cache file into a tidy frame.

    Args:
        path: Path to a single cached JSON file.

    Returns:
        DataFrame: One row per day, columns ["location", "date"] plus
        every variable in payload["daily"].
    """
    with open(path) as f:
        payload = json.load(f)

    daily = payload["daily"]
    df = pd.DataFrame(daily)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df["location"] = payload["location"]

    return df


def load_all_daily(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load and concatenate every cached daily JSON file.

    Args:
        raw_dir: Directory containing the "{location}_{year}.json" cache files.

    Returns:
        DataFrame: All (location, date) rows, sorted chronologically per location.
    """
    frames = [_load_one_file(p) for p in sorted(raw_dir.glob("*.json"))]
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    return df


def interpolate_temperature_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate short gaps in temperature columns only, per location.

    Adds an `is_interpolated` column flagging any value that was filled
    in (True for at least one interpolated variable in that row).
    Gaps longer than MAX_INTERPOLATION_GAP_DAYS are left as NaN, since a
    long gap is a real data quality issue, not something to paper over.

    Args:
        df: Daily DataFrame as returned by load_all_daily().

    Returns:
        DataFrame: Same shape, temperature columns interpolated where
        the gap is short enough, plus an `is_interpolated` flag column.
    """
    df = df.copy()
    was_na = df[INTERPOLATE_COLS].isna()

    filled_parts = []
    for location, group in df.groupby("location", sort=False):
        group = group.set_index("date")
        interpolated = group[INTERPOLATE_COLS].interpolate(
            method="time", limit=MAX_INTERPOLATION_GAP_DAYS, limit_area="inside"
        )
        group[INTERPOLATE_COLS] = interpolated
        filled_parts.append(group.reset_index())

    df = pd.concat(filled_parts, ignore_index=True)
    df = df.sort_values(["location", "date"]).reset_index(drop=True)

    still_na = df[INTERPOLATE_COLS].isna()
    was_filled = was_na & ~still_na
    df["is_interpolated"] = was_filled.any(axis=1)

    return df


def compute_heatwave_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag hot days and heat wave days per location, using a
    per-(location, month) climatological threshold.

    Locations with fewer than MIN_YEARS_FOR_THRESHOLD distinct years of
    data are skipped: a percentile computed on too little history is
    not a real climatological baseline, it's just "the hottest handful
    of days in the one year we happen to have". For those locations
    `is_hot_day` / `is_heatwave_day` are left as NaN (unknown), never
    False (which would wrongly imply "checked, not a heatwave").

    Args:
        df: Daily DataFrame with a `temperature_2m_max` column.

    Returns:
        DataFrame: Same shape, plus `is_hot_day` (above BOTH the monthly
        90th percentile for that location AND the absolute
        HEATWAVE_ABSOLUTE_MIN_TEMP floor) and `is_heatwave_day`
        (part of a run of 3+ consecutive hot days). Both NaN for
        locations without enough history to compute a threshold.
    """
    df = df.copy()
    df["month"] = df["date"].dt.month

    years_available = df.groupby("location")["date"].apply(lambda s: s.dt.year.nunique())
    sufficient_history = years_available[years_available >= MIN_YEARS_FOR_THRESHOLD].index.tolist()
    insufficient = years_available[years_available < MIN_YEARS_FOR_THRESHOLD]

    if not insufficient.empty:
        logging.warning(
            f"Skipping heatwave threshold for locations with <{MIN_YEARS_FOR_THRESHOLD}y "
            f"of history: {insufficient.to_dict()}"
        )

    thresholds = (
        df[df["location"].isin(sufficient_history)]
        .groupby(["location", "month"])["temperature_2m_max"]
        .quantile(HEATWAVE_PERCENTILE)
        .rename("hot_threshold")
    )
    df = df.merge(thresholds, on=["location", "month"], how="left")
    # Nullable "boolean" dtype (not plain bool) so NA can coexist with True/False.
    # AND the relative (per-location, per-month) criterion with an absolute
    # floor - see HEATWAVE_ABSOLUTE_MIN_TEMP - so a day must be both locally
    # anomalous AND plausibly summer-hot to count, rather than just "warm
    # for whatever month it happens to be".
    exceeds_relative = df["temperature_2m_max"] > df["hot_threshold"]
    exceeds_absolute = df["temperature_2m_max"] > HEATWAVE_ABSOLUTE_MIN_TEMP
    df["is_hot_day"] = (exceeds_relative & exceeds_absolute).astype("boolean")
    df.loc[df["hot_threshold"].isna(), "is_hot_day"] = pd.NA

    heatwave_flags = []
    for location, group in df.groupby("location", sort=False):
        group = group.sort_values("date")
        if location not in sufficient_history:
            heatwave_flags.append(pd.Series(pd.NA, index=group.index, dtype="boolean"))
            continue
        run_id = (group["is_hot_day"] != group["is_hot_day"].shift()).cumsum()
        run_length = group.groupby(run_id)["is_hot_day"].transform("size")
        is_heatwave = (group["is_hot_day"] & (run_length >= HEATWAVE_MIN_CONSECUTIVE_DAYS)).astype("boolean")
        heatwave_flags.append(pd.Series(is_heatwave, index=group.index, dtype="boolean"))

    df["is_heatwave_day"] = pd.concat(heatwave_flags).sort_index()
    df = df.drop(columns=["month", "hot_threshold"])

    return df


def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Roll the daily table up to ISO-week grain, one row per
    location x ISO year x ISO week.

    Args:
        df: Daily DataFrame with weather columns and heat wave flags.

    Returns:
        DataFrame: [location, year, week, week_start_date,
        <WEEKLY_AGG_FUNCS columns>, n_days_available,
        heatwave_days_in_week, any_heatwave_week].
    """
    df = df.copy()
    iso = df["date"].dt.isocalendar()
    df["year"] = iso["year"]
    df["week"] = iso["week"]

    grouped = df.groupby(["location", "year", "week"])

    def _sum_preserving_unknown(s: pd.Series):
        """Sum a nullable-boolean Series, but return NA if the whole
        week is NA (insufficient history), instead of silently
        collapsing "unknown" into "0 found"."""
        if s.isna().all():
            return pd.NA
        return int(s.fillna(False).sum())

    weekly = grouped.agg(WEEKLY_AGG_FUNCS)
    weekly["n_days_available"] = grouped["temperature_2m_mean"].apply(lambda s: s.notna().sum())
    weekly["heatwave_days_in_week"] = grouped["is_heatwave_day"].apply(_sum_preserving_unknown)
    weekly["any_heatwave_week"] = weekly["heatwave_days_in_week"].apply(
        lambda x: pd.NA if pd.isna(x) else x > 0
    )
    weekly["week_start_date"] = grouped["date"].min().apply(
        lambda d: (d - pd.Timedelta(days=d.isocalendar().weekday - 1)).date()
    )

    weekly = weekly.reset_index()
    weekly = weekly.sort_values(["location", "year", "week"]).reset_index(drop=True)

    cols = (["location", "year", "week", "week_start_date"]
            + list(WEEKLY_AGG_FUNCS.keys())
            + ["n_days_available", "heatwave_days_in_week", "any_heatwave_week"])
    return weekly[cols]


def run() -> None:
    """Run the full Open-Meteo staging pipeline and write both the
    daily and weekly staged tables to STAGING_DIR.
    """
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    daily = load_all_daily()
    if daily.empty:
        logging.warning("No raw Open-Meteo files found, nothing to stage.")
        return

    daily = interpolate_temperature_gaps(daily)
    daily = compute_heatwave_flags(daily)
    save_parquet(STAGING_DIR / "weather_daily.parquet", daily)
    logging.info(f"Staged weather_daily: {len(daily)} rows -> {STAGING_DIR / 'weather_daily.parquet'}")

    weekly = aggregate_to_weekly(daily)
    save_parquet(STAGING_DIR / "weather_weekly.parquet", weekly)
    logging.info(f"Staged weather_weekly: {len(weekly)} rows -> {STAGING_DIR / 'weather_weekly.parquet'}")


if __name__ == "__main__":
    setup_logging()
    run()