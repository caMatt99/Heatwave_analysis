"""
Staging layer for Eurostat raw extracts.

Turns the wide, one-row-per-region parquet files produced by
eurostat_client.py into long, tidy tables at their native grain:

    mortality_total     -> (geo, year, week, week_start_date, deaths, is_missing)
    mortality_by_age    -> (geo, sex, age, year, week, week_start_date, deaths, is_missing)
    population_by_age   -> (geo, sex, age, year, population, is_missing)

Design decisions (see project discussion):
- Leading/trailing NaNs are NOT missing data - they mark the real start
  and end of a region's data coverage (e.g. France's weekly mortality
  series genuinely starts at 2013-W01, four years later than Italy or
  Spain, and Eurostat's publication lag differs by country). These are
  trimmed per-row based on that row's own first/last valid value.
- NaNs found *inside* a region's coverage window are real gaps. These
  are kept as NaN and flagged via `is_missing=True`, never filled here.
  Any imputation decision belongs to the analysis stage, not staging.
- Raw files are never modified or overwritten; staging only reads them.
"""

import logging
import re
from datetime import date
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.staging.common import save_parquet, week_label_to_date

RAW_DIR = Path("data/raw/eurostat")
STAGING_DIR = Path("data/staging/eurostat")

GEO_COL = "geo\\TIME_PERIOD"
WEEK_COL_PATTERN = re.compile(r"^\d{4}-W\d{2}$")
YEAR_COL_PATTERN = re.compile(r"^\d{4}$")


def _week_to_date(year_week: str) -> date:
    """Convert an ISO 'YYYY-Www' label to the Monday of that week.

    Thin wrapper around the shared common.week_label_to_date, kept so
    the rest of this module doesn't need to import from two places.

    Args:
        year_week: Label like "2023-W07".

    Returns:
        date: The Monday of the corresponding ISO week.
    """
    return week_label_to_date(year_week)


def _trim_coverage_and_flag_gaps(
    df: pd.DataFrame, period_cols: list[str], group_cols: list[str], value_name: str
) -> pd.DataFrame:
    """Drop leading/trailing NaNs per group (real absence of coverage),
    keep internal NaNs and flag them as missing (real data gaps).

    Args:
        df: Wide DataFrame, one row per group_cols combination, one
            column per period (week or year).
        period_cols: Names of the period columns, in chronological order.
        group_cols: Columns identifying a single time series (e.g.
            ["geo"] or ["geo", "sex", "age"]).
        value_name: Name to give the molten value column.

    Returns:
        DataFrame: Long format with columns group_cols + ["period",
        value_name, "is_missing"], trimmed to each group's own coverage
        window.
    """
    frames = []
    for _, row in df.iterrows():
        series = row[period_cols]
        first_valid = series.first_valid_index()
        last_valid = series.last_valid_index()

        if first_valid is None:
            # Entire row is empty - nothing to stage for this group.
            logging.warning(
                f"Row with no valid values at all, skipping: "
                f"{ {c: row[c] for c in group_cols} }"
            )
            continue

        start_idx = period_cols.index(first_valid)
        end_idx = period_cols.index(last_valid)
        covered_cols = period_cols[start_idx:end_idx + 1]

        long_slice = pd.DataFrame({
            **{c: row[c] for c in group_cols},
            "period": covered_cols,
            value_name: [row[c] for c in covered_cols],
        })
        long_slice["is_missing"] = long_slice[value_name].isna()
        frames.append(long_slice)

    if not frames:
        return pd.DataFrame(columns=group_cols + ["period", value_name, "is_missing"])

    return pd.concat(frames, ignore_index=True)


def stage_weekly(path: Path, value_name: str, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    """Stage a weekly Eurostat file (mortality_total or mortality_by_age).

    Args:
        path: Path to the raw parquet file.
        value_name: Name for the staged value column (e.g. "deaths").
        extra_group_cols: Extra dimension columns beyond geo, e.g.
            ["sex", "age"] for mortality_by_age. None for mortality_total.

    Returns:
        DataFrame: Long format with columns [geo, <extra_group_cols>,
        year, week, week_start_date, value_name, is_missing], sorted
        chronologically per group.
    """
    df = pd.read_parquet(path)

    week_cols = [c for c in df.columns if WEEK_COL_PATTERN.match(c)]
    week_cols = sorted(week_cols)  # lexicographic sort = chronological for YYYY-Www

    group_cols = ["geo"] + (extra_group_cols or [])
    df = df.rename(columns={GEO_COL: "geo"})

    staged = _trim_coverage_and_flag_gaps(df, week_cols, group_cols, value_name)

    staged[["year", "week"]] = staged["period"].str.split("-W", expand=True)
    staged["year"] = staged["year"].astype(int)
    staged["week"] = staged["week"].astype(int)
    staged["week_start_date"] = staged["period"].apply(_week_to_date)
    staged = staged.drop(columns=["period"])

    sort_cols = group_cols + ["year", "week"]
    staged = staged.sort_values(sort_cols).reset_index(drop=True)

    return staged[group_cols + ["year", "week", "week_start_date", value_name, "is_missing"]]


def stage_annual(path: Path, value_name: str, extra_group_cols: list[str] | None = None) -> pd.DataFrame:
    """Stage an annual Eurostat file (population_by_age).

    Args:
        path: Path to the raw parquet file.
        value_name: Name for the staged value column (e.g. "population").
        extra_group_cols: Extra dimension columns beyond geo, e.g.
            ["sex", "age"] for population_by_age.

    Returns:
        DataFrame: Long format with columns [geo, <extra_group_cols>,
        year, value_name, is_missing], sorted chronologically per group.
    """
    df = pd.read_parquet(path)

    year_cols = [c for c in df.columns if YEAR_COL_PATTERN.match(c)]
    year_cols = sorted(year_cols, key=int)

    group_cols = ["geo"] + (extra_group_cols or [])
    df = df.rename(columns={GEO_COL: "geo"})

    staged = _trim_coverage_and_flag_gaps(df, year_cols, group_cols, value_name)
    staged["year"] = staged["period"].astype(int)
    staged = staged.drop(columns=["period"])

    sort_cols = group_cols + ["year"]
    staged = staged.sort_values(sort_cols).reset_index(drop=True)

    return staged[group_cols + ["year", value_name, "is_missing"]]


def run() -> None:
    """Stage all three Eurostat raw extracts and write them to STAGING_DIR.

    Each output file covers every macrozone that has a matching raw file
    (i.e. this globs data/raw/eurostat/*_mortality_total.parquet etc.
    and concatenates across regions/countries).
    """
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    mortality_total_frames = [
        stage_weekly(p, value_name="deaths")
        for p in sorted(RAW_DIR.glob("*_mortality_total.parquet"))
    ]
    if mortality_total_frames:
        out = pd.concat(mortality_total_frames, ignore_index=True)
        save_parquet(STAGING_DIR / "mortality_total.parquet", out)
        logging.info(f"Staged mortality_total: {len(out)} rows -> {STAGING_DIR / 'mortality_total.parquet'}")

    mortality_by_age_frames = [
        stage_weekly(p, value_name="deaths", extra_group_cols=["sex", "age"])
        for p in sorted(RAW_DIR.glob("*_mortality_by_age.parquet"))
    ]
    if mortality_by_age_frames:
        out = pd.concat(mortality_by_age_frames, ignore_index=True)
        save_parquet(STAGING_DIR / "mortality_by_age.parquet", out)
        logging.info(f"Staged mortality_by_age: {len(out)} rows -> {STAGING_DIR / 'mortality_by_age.parquet'}")

    population_frames = [
        stage_annual(p, value_name="population", extra_group_cols=["sex", "age"])
        for p in sorted(RAW_DIR.glob("*_population_by_age.parquet"))
    ]
    if population_frames:
        out = pd.concat(population_frames, ignore_index=True)
        save_parquet(STAGING_DIR / "population_by_age.parquet", out)
        logging.info(f"Staged population_by_age: {len(out)} rows -> {STAGING_DIR / 'population_by_age.parquet'}")


if __name__ == "__main__":
    setup_logging()
    run()