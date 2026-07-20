"""
Builds human-readable "analysis views" of the final analytical
datasets, by left-joining dim_region.csv onto them, and - for the
age-stratified dataset only - additional views regrouped into coarser
age bins per named scheme (config/age_bins.yaml).

This is a presentation-layer step, not a pipeline stage: the base
outputs of build_weekly_dataset.py and build_weekly_dataset_by_age.py
stay exactly as they are (geo-keyed, minimal, fully reproducible from
staging) - nothing about them changes. This module reads those
finished outputs and produces separate, additional files with
region_name/macrozone/country columns added alongside `geo` (not
replacing it - geo remains the reliable join key; the added columns
are for reading, plotting, and reporting).

Kept as a distinct step (not baked into the transform scripts
themselves) so the two "pure" analytical datasets remain reusable for
anything that doesn't care about human-readable labels or a specific
age grouping, and so this enrichment logic exists in exactly one place
instead of being duplicated across build_weekly_dataset.py and
build_weekly_dataset_by_age.py.

Age-bin regrouping operates on the labeled mortality_by_age_weekly
view, using the age codes as they exist there - i.e. AFTER
build_weekly_dataset_by_age.py has already collapsed "Y85-89" +
"Y_GE90" into "Y_GE85". See config/age_bins.yaml and
src/utils/age_bins_loader.py for the scheme definitions.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.staging.common import save_parquet
from src.utils.age_bins_loader import (
    load_config as load_age_bins_config,
    get_scheme,
    validate_scheme,
)

DIM_REGION_PATH = Path("data/analytics/dim_region.csv")

SOURCES = {
    "mortality_weather_weekly": Path("data/analytics/mortality_weather_weekly.parquet"),
    "mortality_by_age_weekly": Path("data/analytics/mortality_by_age_weekly.parquet"),
}

OUTPUT_DIR = Path("data/analytics/views")

# Only mortality_by_age_weekly has an `age` column to regroup; naming
# it explicitly here (rather than checking `"age" in df.columns`) makes
# it obvious at a glance which source the age-bin step applies to.
AGE_STRATIFIED_SOURCE = "mortality_by_age_weekly"

# Weather columns are duplicated identically across sex/age within the
# same (geo, year, week), because weather is joined on (geo, year,
# week) only in build_weekly_dataset_by_age.py - so regrouping must
# take the first value, never sum it.
WEATHER_COLS_FIRST = [
    "week_start_date",
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
    "n_days_available",
    "heatwave_days_in_week",
    "any_heatwave_week",
]


def add_region_labels(df: pd.DataFrame, dim_region: pd.DataFrame) -> pd.DataFrame:
    """Left-join region_name/macrozone/country onto any geo-keyed DataFrame.

    Args:
        df: Any DataFrame with a `geo` column (NUTS2 codes).
        dim_region: The dim_region table (geo, country, macrozone,
            region_name, capital_city).

    Returns:
        DataFrame: Same rows as df, with country/macrozone/region_name/
        capital_city columns added. `geo` is kept, not dropped.

    Raises:
        ValueError: If any geo code in df has no match in dim_region -
        loud on purpose, since a silent NaN region_name would only be
        noticed much later, in a plot with a blank label.
    """
    merged = df.merge(dim_region, on="geo", how="left")

    unmatched = merged[merged["region_name"].isna()]["geo"].unique()
    if len(unmatched) > 0:
        raise ValueError(
            f"{len(unmatched)} geo code(s) in the dataset have no match in "
            f"dim_region.csv: {sorted(unmatched)}. Re-run build_dim_region.py "
            f"if locations.yaml has changed."
        )

    return merged


def apply_age_bin_scheme(
    df: pd.DataFrame, age_bins_config: dict, scheme_name: str
) -> pd.DataFrame:
    """Regroup mortality_by_age_weekly's fine-grained `age` column into
    a coarser scheme, aggregating deaths/population accordingly.

    Args:
        df: Labeled mortality_by_age_weekly DataFrame (one row per
            geo, sex, age, year, week), with `age` using the codes
            produced by build_weekly_dataset_by_age.py's reconciliation
            (i.e. "Y_GE85", not "Y85-89"/"Y_GE90" separately).
        age_bins_config: Parsed config as returned by
            age_bins_loader.load_config().
        scheme_name: Which scheme in config/age_bins.yaml to apply.

    Returns:
        DataFrame: One row per (geo, sex, age_group, year, week), where
        age_group replaces age. deaths and population are summed
        across the original age codes folded into each age_group;
        mortality_is_missing/population_is_missing become True if
        EITHER contributing row was missing (same reasoning as the
        Y85-89/Y_GE90 collapse: a real count summed with an unknown
        one is unreliable, not just partially so). Weather columns are
        carried through unchanged (first value, since they're
        duplicated identically across ages within the same geo/year/week).

    Raises:
        ValueError: If the data contains an age code the scheme
        doesn't cover (via age_bins_loader.validate_scheme) - fails
        loudly rather than silently dropping those rows.
    """
    validate_scheme(age_bins_config, scheme_name, set(df["age"].unique()))
    mapping = get_scheme(age_bins_config, scheme_name)

    df = df.copy()
    df["age_group"] = df["age"].map(mapping)

    group_cols = [
        c for c in ["geo", "region_name", "macrozone", "country", "capital_city", "sex", "age_group", "year", "week"]
        if c in df.columns
    ]

    agg = {
        "deaths": "sum",
        "mortality_is_missing": "any",
        "population": "sum",
        "population_is_missing": "any",
    }
    agg.update({c: "first" for c in WEATHER_COLS_FIRST if c in df.columns})

    grouped = df.groupby(group_cols, as_index=False).agg(agg)
    return grouped.sort_values(["geo", "sex", "age_group", "year", "week"]).reset_index(drop=True)


def run() -> None:
    """Build a labeled view for every dataset in SOURCES, plus one
    age-bin-regrouped view per scheme in config/age_bins.yaml for
    AGE_STRATIFIED_SOURCE. Writes each as both parquet and CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dim_region = pd.read_csv(DIM_REGION_PATH)
    age_bins_config = load_age_bins_config()

    for name, path in SOURCES.items():
        df = pd.read_parquet(path)
        labeled = add_region_labels(df, dim_region)

        parquet_path = OUTPUT_DIR / f"{name}_labeled.parquet"
        csv_path = OUTPUT_DIR / f"{name}_labeled.csv"

        save_parquet(parquet_path, labeled)
        labeled.to_csv(csv_path, index=False)

        logging.info(f"Built {name}_labeled: {len(labeled)} rows -> {parquet_path} (+ .csv)")

        if name == AGE_STRATIFIED_SOURCE:
            for scheme_name in age_bins_config["schemes"]:
                binned = apply_age_bin_scheme(labeled, age_bins_config, scheme_name)

                scheme_parquet_path = OUTPUT_DIR / f"{name}_{scheme_name}.parquet"
                scheme_csv_path = OUTPUT_DIR / f"{name}_{scheme_name}.csv"

                save_parquet(scheme_parquet_path, binned)
                binned.to_csv(scheme_csv_path, index=False)

                logging.info(
                    f"Built {name}_{scheme_name}: {len(binned)} rows -> "
                    f"{scheme_parquet_path} (+ .csv)"
                )


if __name__ == "__main__":
    setup_logging()
    run()