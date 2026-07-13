"""
Extraction client for Eurostat datasets used in this project:
- DEMO_R_MWK2_TS: weekly deaths, total, by NUTS2 region
- DEMO_R_MWK2_05: weekly deaths, by 5-year age group, sex and NUTS2 region
- DEMO_R_PJANGROUP: population on 1 January, by age group, sex and NUTS2 region

All three share the same fetch/retry/cache pattern - only the dataset
code and filter_pars change between them.
"""

import sys
import time
import logging
from pathlib import Path

import eurostat
from requests.exceptions import HTTPError

from src.utils.logging_config import setup_logging
from src.utils.locations_loader import load_config, get_macrozones_nuts2

RAW_DIR = Path("data/raw/eurostat")
MAX_RETRIES = 3

AGE_CODES = {
    "mortality": ["Y_LT5", "Y5-9", "Y10-14", "Y15-19", "Y20-24", "Y25-29",
                  "Y30-34", "Y35-39", "Y40-44", "Y45-49", "Y50-54", "Y55-59",
                  "Y60-64", "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y85-89", "Y_GE90"],
    "population": ["Y_LT5", "Y5-9", "Y10-14", "Y15-19", "Y20-24", "Y25-29",
                   "Y30-34", "Y35-39", "Y40-44", "Y45-49", "Y50-54", "Y55-59",
                   "Y60-64", "Y65-69", "Y70-74", "Y75-79", "Y80-84", "Y_GE85"],
}


def _save_parquet(path: Path, df) -> None:
    """Write a DataFrame to a temp file, then rename it into place."""
    tmp_path = path.with_suffix(".tmp")
    df.to_parquet(tmp_path)
    tmp_path.rename(path)


def fetch_eurostat_data(dataset_code: str, filter_pars: dict, cache_name: str) -> Path:
    """Fetch and cache one query against any Eurostat dataset.

    Retries transient failures. An HTTP 4xx (genuinely invalid request)
    is not retried, since it would fail identically every time.

    Args:
        dataset_code: Eurostat dataset identifier (e.g. "DEMO_R_MWK2_TS").
        filter_pars: Query filters, e.g. {"geo": [...], "sex": [...], "age": [...]}.
        cache_name: Identifier used in the cache filename.

    Returns:
        Path to the cached parquet file.

    Raises:
        ValueError: If the response is empty on every retry.
        Exception: Any other error, re-raised after exhausting retries
            (or immediately, for non-retryable errors).
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DIR / f"{cache_name}.parquet"

    if target_path.exists():
        logging.debug(f"Cache hit: {cache_name}, skipping API call")
        return target_path

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = eurostat.get_data_df(dataset_code, filter_pars=filter_pars)

            if df is None or df.empty:
                raise ValueError("empty response from Eurostat API")

            _save_parquet(target_path, df)
            logging.info(f"Saved {cache_name} -> {target_path}")
            return target_path

        except HTTPError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                logging.error(f"Non-retryable client error for {cache_name}: {e}")
                raise
            last_error = e
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {cache_name}: {e}")
            time.sleep(2 ** attempt)

        except Exception as e:
            last_error = e
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {cache_name}: {e}")
            time.sleep(2 ** attempt)

    logging.error(f"Error fetching {cache_name}: {last_error}")
    raise last_error


def fetch_all(macrozones: dict[str, list[str]], dataset_code: str,
              cache_suffix: str, extra_filters: dict | None = None) -> dict[str, Path]:
    """Fetch every macrozone for one dataset, skipping failures instead
    of aborting.

    Args:
        macrozones: Mapping of macrozone name -> list of NUTS2 codes.
        dataset_code: Eurostat dataset identifier.
        cache_suffix: Appended to each macrozone name in the cache
            filename, to distinguish datasets (e.g. "mortality_total",
            "mortality_by_age", "population_by_age").
        extra_filters: Additional filter_pars beyond "geo" (e.g. sex, age).

    Returns:
        dict: Mapping of macrozone name -> cached file path, for every
        successful fetch.
    """
    results = {}
    failures = []
    cache_hits = 0
    downloaded = 0
    extra_filters = extra_filters or {}

    for name, geo_codes in macrozones.items():
        cache_name = f"{name}_{cache_suffix}"
        target_path = RAW_DIR / f"{cache_name}.parquet"
        was_cached = target_path.exists()

        filter_pars = {"geo": geo_codes, **extra_filters}
        try:
            results[name] = fetch_eurostat_data(dataset_code, filter_pars, cache_name)
            if was_cached:
                cache_hits += 1
            else:
                downloaded += 1
        except Exception as e:
            logging.error(f"Skipping {name} after exhausting retries: {e}")
            failures.append(name)
        time.sleep(1)

    logging.info(
        f"{cache_suffix}: {cache_hits} cache hit(s), {downloaded} downloaded, "
        f"{len(failures)} failed ({len(macrozones)} total)"
    )
    if failures:
        logging.error(f"{len(failures)} macrozone(s) failed: {failures}")

    return results


if __name__ == "__main__":
    setup_logging()

    config = load_config()
    macrozones = get_macrozones_nuts2(config)

    logging.info("Starting Eurostat extraction (weekly deaths, total)")
    mortality_total = fetch_all(
        macrozones, "DEMO_R_MWK2_TS", "mortality_total",
        extra_filters={"sex": ["T"]},
    )
    mortality_total_ok = len(mortality_total) == len(macrozones)

    logging.info("Starting Eurostat extraction (weekly deaths, by age)")
    mortality_by_age = fetch_all(
        macrozones, "DEMO_R_MWK2_05", "mortality_by_age",
        extra_filters={"sex": ["M", "F"], "age": AGE_CODES["mortality"]},
    )
    mortality_by_age_ok = len(mortality_by_age) == len(macrozones)

    logging.info("Starting Eurostat extraction (population, by age)")
    population_by_age = fetch_all(
        macrozones, "DEMO_R_PJANGROUP", "population_by_age",
        extra_filters={"sex": ["M", "F"], "age": AGE_CODES["population"]},
    )
    population_by_age_ok = len(population_by_age) == len(macrozones)

    expected = len(macrozones)
    if (len(mortality_total) < expected
            or len(mortality_by_age) < expected
            or len(population_by_age) < expected):
        sys.exit(1)