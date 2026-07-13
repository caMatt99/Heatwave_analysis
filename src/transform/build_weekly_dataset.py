"""
Transform layer: builds the final weekly analytical dataset by joining
the three staged tables produced by src/staging/.

    mortality_total   (geo, year, week)  - weekly deaths, native grain
    weather_weekly     (geo, year, week)  - weekly weather + heat wave flags, native grain
    population_by_age  (geo, year)        - annual population, native grain

Design decisions:
- Population is kept at its native annual grain in staging and only
  broadcast to weekly grain HERE, at query time - never persisted as
  52 duplicated rows upstream. This is the standard "keep native grain
  in storage, reconcile at query time" pattern: one source-of-truth row
  per (geo, year), no redundancy, no risk of partial-update drift if
  Eurostat later revises a population estimate.
- mortality_total and weather_weekly are joined with an INNER join on
  (geo, year, week): weather is the narrower time range of the two
  (2015-2026 vs mortality's 2000-2026), so the result is naturally
  bounded to weeks where both sources actually have data - no need to
  artificially truncate either source beforehand.
- Population is LEFT-joined on (geo, year): a handful of boundary weeks
  (ISO year 2026, week 1) fall outside population's 1990-2025 coverage
  and legitimately get NaN population - this is left as NaN, not
  fabricated, consistent with the missing-data philosophy used
  throughout staging.
- Population is aggregated (summed across sex and age) from its native
  by-age/sex grain to a single total-population-per-region-per-year
  figure, since mortality_total has no age/sex breakdown to match
  against. If ANY underlying age/sex subgroup for a (geo, year) was
  flagged is_missing in staging, the aggregated total is flagged
  is_missing too (population_is_missing=True) rather than silently
  summing only the available subgroups and under-counting.
- This module does NOT compute derived epidemiological measures (rates,
  standardization, RR/OR, etc.) - that belongs to the analysis stage
  (DAPH course chapters 1-5), not to the transform stage. This module's
  job is strictly: get the three sources onto one row per (geo, week),
  with raw counts and population as-is.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.staging.common import save_parquet

MORTALITY_PATH = Path("data/staging/eurostat/mortality_total.parquet")
POPULATION_PATH = Path("data/staging/eurostat/population_by_age.parquet")
WEATHER_PATH = Path("data/staging/openmeteo/weather_weekly.parquet")

OUTPUT_DIR = Path("data/analytics")
OUTPUT_PATH = OUTPUT_DIR / "mortality_weather_weekly.parquet"
CSV_OUTPUT_PATH = OUTPUT_DIR / "mortality_weather_weekly.csv"


def load_mortality(path: Path = MORTALITY_PATH) -> pd.DataFrame:
    """Load the staged weekly total-mortality table.

    Args:
        path: Path to the staged mortality_total parquet.

    Returns:
        DataFrame: [geo, year, week, week_start_date, deaths,
        mortality_is_missing].
    """
    df = pd.read_parquet(path)
    return df.rename(columns={"is_missing": "mortality_is_missing"})


def load_weather(path: Path = WEATHER_PATH) -> pd.DataFrame:
    """Load the staged weekly weather table.

    Args:
        path: Path to the staged weather_weekly parquet.

    Returns:
        DataFrame: Same columns as weather_weekly, with `location`
        renamed to `geo` so it matches the Eurostat join key, and
        `week_start_date` dropped (kept from mortality instead - both
        sources compute the same Monday-of-ISO-week value, no need to
        carry it twice).
    """
    df = pd.read_parquet(path)
    df = df.rename(columns={"location": "geo"})
    df = df.drop(columns=["week_start_date"])
    return df


def build_population_by_year(path: Path = POPULATION_PATH) -> pd.DataFrame:
    """Aggregate the staged by-age/sex population table to one total
    population figure per (geo, year).

    Args:
        path: Path to the staged population_by_age parquet.

    Returns:
        DataFrame: [geo, year, population, population_is_missing].
        population_is_missing is True if ANY age/sex subgroup for that
        (geo, year) was flagged missing in staging - the total is then
        an undercount, not a true figure, and callers should not treat
        it as reliable even though a numeric value is present.
    """
    df = pd.read_parquet(path)

    aggregated = (
        df.groupby(["geo", "year"])
        .agg(
            population=("population", "sum"),
            population_is_missing=("is_missing", "any"),
        )
        .reset_index()
    )
    return aggregated


def build_weekly_dataset(
    mortality_path: Path = MORTALITY_PATH,
    weather_path: Path = WEATHER_PATH,
    population_path: Path = POPULATION_PATH,
) -> pd.DataFrame:
    """Join mortality, weather, and broadcast population into the final
    weekly analytical dataset.

    Args:
        mortality_path: Path to staged mortality_total parquet.
        weather_path: Path to staged weather_weekly parquet.
        population_path: Path to staged population_by_age parquet.

    Returns:
        DataFrame: One row per (geo, year, week), with deaths, weather
        variables and heat wave flags, and broadcast population -
        sorted by geo, year, week.
    """
    mortality = load_mortality(mortality_path)
    weather = load_weather(weather_path)
    population = build_population_by_year(population_path)

    merged = mortality.merge(weather, on=["geo", "year", "week"], how="inner")
    logging.info(
        f"Joined mortality ({len(mortality)} rows) with weather ({len(weather)} rows) "
        f"-> {len(merged)} rows"
    )

    merged = merged.merge(population, on=["geo", "year"], how="left")
    missing_population = merged["population"].isna().sum()
    if missing_population:
        logging.warning(
            f"{missing_population} row(s) have no population figure "
            f"(outside population's 1990-2025 coverage) - left as NaN, not fabricated"
        )
    # Rows with no matching population row at all (outside 1990-2025) get
    # NaN here from the left join - that's unambiguously "missing", not
    # "unknown", so mark it True explicitly rather than leaving a NaN
    # flag next to a NaN value.
    merged["population_is_missing"] = merged["population_is_missing"].fillna(True).astype(bool)

    merged = merged.sort_values(["geo", "year", "week"]).reset_index(drop=True)

    return merged


def run() -> None:
    """Build the final weekly dataset and write it to both OUTPUT_PATH
    (parquet, the working format used by the rest of the pipeline) and
    CSV_OUTPUT_PATH (for spreadsheet inspection / the dashboard).

    Both files are always overwritten from the same in-memory
    DataFrame, so they can never drift out of sync with each other -
    unlike generating the CSV as a separate manual step later, which
    silently goes stale the next time the parquet is rebuilt. This is
    safe specifically because this output is fully reproducible from
    upstream raw/staging data - nothing here is a unique source of
    truth that would be lost by overwriting it.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_weekly_dataset()

    save_parquet(OUTPUT_PATH, dataset)
    dataset.to_csv(CSV_OUTPUT_PATH, index=False)

    logging.info(
        f"Built weekly analytical dataset: {len(dataset)} rows -> "
        f"{OUTPUT_PATH} (+ {CSV_OUTPUT_PATH.name})"
    )


if __name__ == "__main__":
    setup_logging()
    run()