"""
Transform layer: builds the age/sex-stratified weekly analytical
dataset, joining mortality_by_age, population_by_age, and
weather_weekly at (geo, sex, age, year, week) grain.

This is a SEPARATE table from mortality_weather_weekly (built by
build_weekly_dataset.py), not a superset merged into it: most analyses
(descriptive trends, crude rates, the temperature-mortality scatter)
don't need the age breakdown, and folding it in would multiply that
table's size ~36x for no benefit to those use cases. This table exists
specifically for age-standardization (DAPH course chapter 3) and for
examining whether heat-mortality association differs by age group.

Critical data-quality issue this module exists to handle:
mortality_by_age and population_by_age do NOT share the same age
bins at the oldest ages - mortality has separate "Y85-89" and "Y_GE90"
brackets, while population has them already combined into a single
"Y_GE85" bracket (this was flagged early on, in the AGE_CODES comment
in eurostat_client.py, and confirmed on real data: mortality has 19
age codes, population has 18, differing exactly in this way). Joining
directly on `age` would silently leave "Y85-89" and "Y_GE90" mortality
rows with no matching population - not an error, just missing
population for the oldest age groups. This module reconciles the two
by collapsing mortality's two oldest brackets into "Y_GE85" BEFORE the
join, matching population's granularity everywhere.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.staging.common import save_parquet

MORTALITY_BY_AGE_PATH = Path("data/staging/eurostat/mortality_by_age.parquet")
POPULATION_BY_AGE_PATH = Path("data/staging/eurostat/population_by_age.parquet")
WEATHER_PATH = Path("data/staging/openmeteo/weather_weekly.parquet")

OUTPUT_DIR = Path("data/analytics")
OUTPUT_PATH = OUTPUT_DIR / "mortality_by_age_weekly.parquet"
CSV_OUTPUT_PATH = OUTPUT_DIR / "mortality_by_age_weekly.csv"

# mortality_by_age's two oldest brackets get collapsed into this single
# label, to match population_by_age's oldest bracket exactly.
OLDEST_MORTALITY_BINS = ["Y85-89", "Y_GE90"]
RECONCILED_OLDEST_BIN = "Y_GE85"


def reconcile_age_bins(mortality: pd.DataFrame) -> pd.DataFrame:
    """Collapse mortality_by_age's two oldest brackets into one, to
    match population_by_age's oldest bracket.

    Args:
        mortality: Staged mortality_by_age DataFrame, with an `age`
            column using the finer-grained brackets (including
            "Y85-89" and "Y_GE90" separately).

    Returns:
        DataFrame: Same shape/columns, but with "Y85-89" and "Y_GE90"
        rows summed together into a single "Y_GE85" row per
        (geo, sex, year, week). `is_missing` becomes True for the
        combined row if EITHER of the two source rows was missing -
        summing a real count with an unknown one still makes the
        total unreliable, not just partially so.
    """
    df = mortality.copy()
    is_oldest = df["age"].isin(OLDEST_MORTALITY_BINS)

    unchanged = df[~is_oldest]
    oldest = df[is_oldest].copy()
    oldest["age"] = RECONCILED_OLDEST_BIN

    group_cols = ["geo", "sex", "age", "year", "week", "week_start_date"]
    collapsed = (
        oldest.groupby(group_cols, as_index=False)
        .agg(deaths=("deaths", "sum"), is_missing=("is_missing", "any"))
    )

    result = pd.concat([unchanged, collapsed], ignore_index=True)
    return result


def load_mortality_by_age(path: Path = MORTALITY_BY_AGE_PATH) -> pd.DataFrame:
    """Load staged mortality_by_age and reconcile its age bins.

    Args:
        path: Path to the staged mortality_by_age parquet.

    Returns:
        DataFrame: [geo, sex, age, year, week, week_start_date, deaths,
        mortality_is_missing], with age bins matching population_by_age.
    """
    df = pd.read_parquet(path)
    df = reconcile_age_bins(df)
    return df.rename(columns={"is_missing": "mortality_is_missing"})


def load_population_by_age(path: Path = POPULATION_BY_AGE_PATH) -> pd.DataFrame:
    """Load staged population_by_age as-is (already at geo/sex/age/year grain).

    Args:
        path: Path to the staged population_by_age parquet.

    Returns:
        DataFrame: [geo, sex, age, year, population, population_is_missing].
    """
    df = pd.read_parquet(path)
    return df.rename(columns={"is_missing": "population_is_missing"})


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    """Load the staged weekly weather table (same as build_weekly_dataset.py).

    Args:
        path: Path to the staged weather_weekly parquet.

    Returns:
        DataFrame: weather_weekly with `location` renamed to `geo` and
        `week_start_date` dropped (kept from mortality instead).
    """
    df = pd.read_parquet(path)
    df = df.rename(columns={"location": "geo"})
    df = df.drop(columns=["week_start_date"])
    return df


def build_weekly_dataset_by_age(
    mortality_path: Path = MORTALITY_BY_AGE_PATH,
    population_path: Path = POPULATION_BY_AGE_PATH,
    weather_path: Path = WEATHER_PATH,
) -> pd.DataFrame:
    """Join age/sex-stratified mortality, population, and weather into
    one weekly analytical dataset.

    Args:
        mortality_path: Path to staged mortality_by_age parquet.
        population_path: Path to staged population_by_age parquet.
        weather_path: Path to staged weather_weekly parquet.

    Returns:
        DataFrame: One row per (geo, sex, age, year, week), sorted by
        geo, sex, age, year, week.
    """
    mortality = load_mortality_by_age(mortality_path)
    population = load_population_by_age(population_path)
    weather = load_weather(weather_path)

    merged = mortality.merge(weather, on=["geo", "year", "week"], how="inner")
    logging.info(
        f"Joined mortality_by_age ({len(mortality)} rows) with weather "
        f"({len(weather)} rows) -> {len(merged)} rows"
    )

    merged = merged.merge(population, on=["geo", "sex", "age", "year"], how="left")
    missing_population = merged["population"].isna().sum()
    if missing_population:
        logging.warning(
            f"{missing_population} row(s) have no population figure "
            f"(outside population's 1990-2025 coverage) - left as NaN, not fabricated"
        )
    merged["population_is_missing"] = merged["population_is_missing"].fillna(True).astype(bool)

    merged = merged.sort_values(["geo", "sex", "age", "year", "week"]).reset_index(drop=True)

    return merged


def run() -> None:
    """Build the age-stratified weekly dataset and write it to both
    parquet and CSV, always overwriting - same reproducibility
    rationale as build_weekly_dataset.py.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_weekly_dataset_by_age()

    save_parquet(OUTPUT_PATH, dataset)
    dataset.to_csv(CSV_OUTPUT_PATH, index=False)

    logging.info(
        f"Built age-stratified weekly dataset: {len(dataset)} rows -> "
        f"{OUTPUT_PATH} (+ {CSV_OUTPUT_PATH.name})"
    )


if __name__ == "__main__":
    setup_logging()
    run()