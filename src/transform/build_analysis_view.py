"""
Builds human-readable "analysis views" of the final analytical
datasets, by left-joining dim_region.csv onto them.

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
anything that doesn't care about human-readable labels, and so this
enrichment logic exists in exactly one place instead of being
duplicated across build_weekly_dataset.py and
build_weekly_dataset_by_age.py.
"""

import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.staging.common import save_parquet

DIM_REGION_PATH = Path("data/analytics/dim_region.csv")

SOURCES = {
    "mortality_weather_weekly": Path("data/analytics/mortality_weather_weekly.parquet"),
    "mortality_by_age_weekly": Path("data/analytics/mortality_by_age_weekly.parquet"),
}

OUTPUT_DIR = Path("data/analytics/views")


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


def run() -> None:
    """Build a labeled view for every dataset in SOURCES and write
    each to OUTPUT_DIR, as both parquet and CSV.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dim_region = pd.read_csv(DIM_REGION_PATH)

    for name, path in SOURCES.items():
        df = pd.read_parquet(path)
        labeled = add_region_labels(df, dim_region)

        parquet_path = OUTPUT_DIR / f"{name}_labeled.parquet"
        csv_path = OUTPUT_DIR / f"{name}_labeled.csv"

        save_parquet(parquet_path, labeled)
        labeled.to_csv(csv_path, index=False)

        logging.info(f"Built {name}_labeled: {len(labeled)} rows -> {parquet_path} (+ .csv)")


if __name__ == "__main__":
    setup_logging()
    run()