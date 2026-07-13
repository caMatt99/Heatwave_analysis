"""
Shared utilities used by both eurostat_staging.py and openmeteo_staging.py.

Kept deliberately small: only functions with genuine duplicate logic
across both staging modules live here. Source-specific logic (melting,
coverage trimming, heat wave detection, weekly aggregation) stays in
its own module.
"""

from datetime import date
from pathlib import Path

import pandas as pd


def save_parquet(path: Path, df: pd.DataFrame) -> None:
    """Write a DataFrame to a temp file, then rename it into place.

    Used by every staging module so a crash mid-write never leaves a
    truncated/corrupt parquet file at the final destination.

    Args:
        path: Final destination path.
        df: DataFrame to persist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    df.to_parquet(tmp_path)
    tmp_path.rename(path)


def iso_week_to_date(year: int, week: int) -> date:
    """Return the Monday of a given ISO 8601 (year, week).

    This is the single source of truth for "ISO week -> calendar date"
    used across both staging modules, so Eurostat's "YYYY-Www" labels
    and Open-Meteo's daily dates always resolve to the same convention
    (Monday as day 1 of the week) and stay joinable on week_start_date.

    Args:
        year: ISO year (note: can differ from the calendar year for
            dates near Dec 31 / Jan 1).
        week: ISO week number (1-53).

    Returns:
        date: The Monday of that ISO week.
    """
    return date.fromisocalendar(int(year), int(week), 1)


def week_label_to_date(year_week: str) -> date:
    """Convert a Eurostat-style 'YYYY-Www' label to the Monday of that week.

    Args:
        year_week: Label like "2023-W07".

    Returns:
        date: The Monday of the corresponding ISO week.
    """
    year, week = year_week.split("-W")
    return iso_week_to_date(int(year), int(week))