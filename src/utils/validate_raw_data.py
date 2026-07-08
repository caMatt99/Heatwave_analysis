"""
Validation script for raw cached data, run before building the
transform layer. Checks that every expected file exists, is readable,
and has a sane shape - catching data problems here is much cheaper
than discovering them halfway through a transform script.
"""

import sys
import json
import logging
from pathlib import Path

import pandas as pd

from src.utils.logging_config import setup_logging
from src.utils.locations_loader import (
    load_config,
    get_macrozones_nuts2,
    get_locations_coordinates,
    get_date_range,
)

RAW_EUROSTAT_DIR = Path("data/raw/eurostat")
RAW_OPENMETEO_DIR = Path("data/raw/openmeteo")

EUROSTAT_DATASETS = {
    "mortality_total": {"geo_col": "geo\\TIME_PERIOD", "expect_age": False, "expect_sex": ["T"]},
    "mortality_by_age": {"geo_col": "geo\\TIME_PERIOD", "expect_age": True, "expect_sex": ["M", "F"]},
    "population_by_age": {"geo_col": "geo\\TIME_PERIOD", "expect_age": True, "expect_sex": ["M", "F"]},
}


def validate_eurostat_file(path: Path, geo_codes: list[str], expect_age: bool,
                            expect_sex: list[str]) -> list[str]:
    """Validate one cached Eurostat parquet file.

    Args:
        path: Path to the parquet file.
        geo_codes: NUTS2 codes expected to appear in this file.
        expect_age: Whether an "age" column should be present.
        expect_sex: Sex codes expected to appear.

    Returns:
        list[str]: Problems found (empty list means the file is fine).
    """
    problems = []

    if not path.exists():
        return [f"missing file: {path}"]

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return [f"unreadable file {path}: {e}"]

    if df.empty:
        problems.append(f"{path.name}: file is empty")
        return problems

    geo_col = "geo\\TIME_PERIOD"
    if geo_col not in df.columns:
        problems.append(f"{path.name}: missing expected column '{geo_col}'")
    else:
        found_geo = set(df[geo_col].unique())
        missing_geo = set(geo_codes) - found_geo
        if missing_geo:
            problems.append(f"{path.name}: missing NUTS2 codes {sorted(missing_geo)}")

    if expect_age and "age" not in df.columns:
        problems.append(f"{path.name}: expected an 'age' column, not found")

    if "sex" in df.columns:
        found_sex = set(df["sex"].unique())
        missing_sex = set(expect_sex) - found_sex
        if missing_sex:
            problems.append(f"{path.name}: missing sex codes {sorted(missing_sex)}")

    duplicate_key_cols = [c for c in ["geo\\TIME_PERIOD", "age", "sex"] if c in df.columns]
    if duplicate_key_cols and df.duplicated(subset=duplicate_key_cols).any():
        n_dupes = df.duplicated(subset=duplicate_key_cols).sum()
        problems.append(f"{path.name}: {n_dupes} duplicate rows on {duplicate_key_cols}")

    return problems


def summarize_eurostat_file(path: Path) -> dict:
    """Extract summary stats from one cached Eurostat file.

    Args:
        path: Path to the parquet file.

    Returns:
        dict: Summary stats (rows, geo codes count, week range), or
        an "error" key if the file couldn't be read.
    """
    if not path.exists():
        return {"error": "file not found"}

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        return {"error": str(e)}

    geo_col = "geo\\TIME_PERIOD"
    week_columns = [c for c in df.columns if c not in ("freq", "sex", "unit", "age", geo_col)]

    return {
        "rows": len(df),
        "geo_codes": df[geo_col].nunique() if geo_col in df.columns else None,
        "period_start": week_columns[0] if week_columns else None,
        "period_end": week_columns[-1] if week_columns else None,
    }


def summarize_openmeteo_file(path: Path) -> dict:
    """Extract summary stats from one cached Open-Meteo file.

    Args:
        path: Path to the JSON file.

    Returns:
        dict: Summary stats (day count, date range), or an "error"
        key if the file couldn't be read.
    """
    if not path.exists():
        return {"error": "file not found"}

    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as e:
        return {"error": str(e)}

    times = payload.get("daily", {}).get("time", [])
    return {
        "days": len(times),
        "date_start": times[0] if times else None,
        "date_end": times[-1] if times else None,
    }

def validate_all_eurostat(macrozones: dict[str, list[str]]) -> list[str]:
    """Validate every Eurostat file for every macrozone and dataset.

    Args:
        macrozones: Mapping of macrozone name -> list of NUTS2 codes.

    Returns:
        list[str]: All problems found across every file.
    """
    all_problems = []

    for suffix, spec in EUROSTAT_DATASETS.items():
        for name, geo_codes in macrozones.items():
            path = RAW_EUROSTAT_DIR / f"{name}_{suffix}.parquet"
            problems = validate_eurostat_file(
                path, geo_codes, spec["expect_age"], spec["expect_sex"]
            )
            all_problems.extend(problems)

    return all_problems


def validate_openmeteo_file(path: Path) -> list[str]:
    """Validate one cached Open-Meteo JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        list[str]: Problems found (empty list means the file is fine).
    """
    if not path.exists():
        return [f"missing file: {path}"]

    try:
        with open(path) as f:
            payload = json.load(f)
    except Exception as e:
        return [f"unreadable file {path}: {e}"]

    problems = []
    required_keys = ["temperature_2m_mean", "temperature_2m_max", "temperature_2m_min", "time"]

    if "daily" not in payload:
        return [f"{path.name}: missing 'daily' key"]

    for key in required_keys:
        if key not in payload["daily"]:
            problems.append(f"{path.name}: missing 'daily.{key}'")

    n_days = len(payload["daily"].get("time", []))
    if n_days < 300:  # a full year should have ~365; flag suspiciously short years
        problems.append(f"{path.name}: only {n_days} days found, expected ~365")

    return problems


def validate_all_openmeteo(locations: list[dict], years: list[int]) -> list[str]:
    """Validate every Open-Meteo file for every location and year.

    Args:
        locations: List of dicts with key "name".
        years: Calendar years expected to have been fetched.

    Returns:
        list[str]: All problems found across every file.
    """
    all_problems = []

    for loc in locations:
        for year in years:
            path = RAW_OPENMETEO_DIR / f"{loc['name']}_{year}.json"
            all_problems.extend(validate_openmeteo_file(path))

    return all_problems


def run_validation() -> bool:
    """Run all validation checks, print a summary, and log problems.

    Returns:
        bool: True if no problems were found, False otherwise.
    """
    config = load_config()
    macrozones = get_macrozones_nuts2(config)
    locations = get_locations_coordinates(config)
    start_year, end_year = get_date_range(config)
    years = list(range(start_year, end_year + 1))

    logging.info("Validating Eurostat files...")
    eurostat_problems = validate_all_eurostat(macrozones)

    print("\n=== EUROSTAT SUMMARY ===")
    for suffix in EUROSTAT_DATASETS:
        print(f"\n--- {suffix} ---")
        for name in macrozones:
            path = RAW_EUROSTAT_DIR / f"{name}_{suffix}.parquet"
            summary = summarize_eurostat_file(path)
            if "error" in summary:
                print(f"  {name:12s} ERROR: {summary['error']}")
            else:
                print(f"  {name:12s} rows={summary['rows']:4d}  "
                      f"geo_codes={summary['geo_codes']:2d}  "
                      f"period={summary['period_start']} -> {summary['period_end']}")

    logging.info("Validating Open-Meteo files...")
    openmeteo_problems = validate_all_openmeteo(locations, years)

    print("\n=== OPEN-METEO SUMMARY ===")
    for loc in locations:
        total_days = 0
        year_range = (None, None)
        for year in years:
            path = RAW_OPENMETEO_DIR / f"{loc['name']}_{year}.json"
            summary = summarize_openmeteo_file(path)
            if "error" not in summary:
                total_days += summary["days"]
                if year_range[0] is None:
                    year_range = (summary["date_start"], summary["date_end"])
                else:
                    year_range = (year_range[0], summary["date_end"])
        print(f"  {loc['name']:12s} total_days={total_days:4d}  "
              f"range={year_range[0]} -> {year_range[1]}")

    all_problems = eurostat_problems + openmeteo_problems

    print()
    if all_problems:
        logging.error(f"Validation found {len(all_problems)} problem(s):")
        for problem in all_problems:
            logging.error(f"  - {problem}")
        return False

    logging.info("All raw data files passed validation.")
    return True

if __name__ == "__main__":
    setup_logging()

    if not run_validation():
        sys.exit(1)