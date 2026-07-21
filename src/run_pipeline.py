"""
Entry point for the extraction stage of the pipeline.
Runs both extraction clients (Eurostat weekly deaths/population,
Open-Meteo daily weather) against the same locations config, so a
single command produces all the raw data the transform stage will need.
"""

import sys
import logging

from src.utils.logging_config import setup_logging
from src.utils.locations_loader import (
    load_config,
    get_macrozones_nuts2,
    get_nuts2_coordinates,
    get_date_range,
)
from src.extract import eurostat_client, openmeteo_client


def run_extraction() -> bool:
    """Run all extraction steps using the shared locations config.

    Returns:
        bool: True if every dataset/macrozone/chunk was fetched
        successfully, False if anything is missing.
    """
    config = load_config()
    macrozones = get_macrozones_nuts2(config)

    logging.info("Starting Eurostat extraction (weekly deaths, total)")
    mortality_total = eurostat_client.fetch_all(
        macrozones, "DEMO_R_MWK2_TS", "mortality_total",
        extra_filters={"sex": ["T"]},
    )
    mortality_total_ok = len(mortality_total) == len(macrozones)

    logging.info("Starting Eurostat extraction (weekly deaths, by age)")
    mortality_by_age = eurostat_client.fetch_all(
        macrozones, "DEMO_R_MWK2_05", "mortality_by_age",
        extra_filters={
            "sex": ["M", "F"],
            "age": eurostat_client.AGE_CODES["mortality"],
        },
    )
    mortality_by_age_ok = len(mortality_by_age) == len(macrozones)

    logging.info("Starting Eurostat extraction (population, by age)")
    population_by_age = eurostat_client.fetch_all(
        macrozones, "DEMO_R_PJANGROUP", "population_by_age",
        extra_filters={
            "sex": ["M", "F"],
            "age": eurostat_client.AGE_CODES["population"],
        },
    )
    population_by_age_ok = len(population_by_age) == len(macrozones)

    logging.info("Starting Open-Meteo extraction (daily weather)")
    locations = get_nuts2_coordinates(config)
    start_year, end_year = get_date_range(config)
    years = list(range(start_year, end_year + 1))
    weather_results = openmeteo_client.fetch_all(locations, years)
    weather_ok = len(weather_results) == len(locations) * len(years)

    if not mortality_total_ok:
        logging.error("Total mortality extraction incomplete")
    if not mortality_by_age_ok:
        logging.error("Mortality-by-age extraction incomplete")
    if not population_by_age_ok:
        logging.error("Population-by-age extraction incomplete")
    if not weather_ok:
        logging.error("Weather extraction incomplete")

    return (
        mortality_total_ok
        and mortality_by_age_ok
        and population_by_age_ok
        and weather_ok
    )


if __name__ == "__main__":
    setup_logging()

    success = run_extraction()

    if not success:
        sys.exit(1)

    logging.info("Extraction stage completed successfully")