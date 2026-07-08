"""
Extraction client for the Open-Meteo Historical Weather API.
Fetches daily weather data one (location, year) chunk at a time and
caches each chunk as raw JSON.
"""

import sys
import json
import time
import logging
import requests
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.utils.locations_loader import (
    load_config,
    get_macrozones_nuts2,
    get_date_range,
)

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = Path("data/raw/openmeteo")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
]


def _build_params(lat: float, lon: float, year: int) -> dict:
    """Build the query params for one location/year request.

    Args:
        lat: Latitude of the location.
        lon: Longitude of the location.
        year: Calendar year to fetch.

    Returns:
        dict: Query parameters ready for requests.get().
    """
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }


def _save_json(path: Path, data: dict) -> None:
    """Write JSON to a temp file, then rename it into place.

    Args:
        path: Final destination path.
        data: JSON-serializable payload to persist.
    """
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
        f.flush()
    tmp_path.rename(path)


def fetch_weather_year(location_name: str, lat: float, lon: float, year: int) -> Path:
    """Fetch and cache one year of daily weather for one location.

    Retries transient failures (timeouts, connection errors, malformed
    payloads). An HTTP 4xx (e.g. a date range the archive doesn't
    cover) is not retried, since the same request would fail
    identically every time.

    Args:
        location_name: Identifier used in the cache filename (e.g. "IT_nord").
        lat: Latitude of the location.
        lon: Longitude of the location.
        year: Calendar year to fetch.

    Returns:
        Path to the cached JSON file.

    Raises:
        requests.exceptions.RequestException: If the API call fails
            (immediately for a 4xx, or after exhausting retries otherwise).
        ValueError: If the response is malformed on every retry.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DIR / f"{location_name}_{year}.json"

    if target_path.exists():
        logging.info(f"Cache hit: {location_name} {year}, skipping API call")
        return target_path

    params = _build_params(lat, lon, year)
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(BASE_URL, params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()

            if "daily" not in payload or "time" not in payload["daily"]:
                raise ValueError("malformed response: missing 'daily.time'")

            payload["location"] = location_name
            payload["year"] = year

            _save_json(target_path, payload)
            logging.info(f"Saved {location_name} {year} -> {target_path}")
            return target_path

        except requests.exceptions.HTTPError as e:
            if e.response is not None and 400 <= e.response.status_code < 500:
                logging.error(f"Non-retryable client error for {location_name} {year}: {e}")
                raise
            last_error = e
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {location_name} {year}: {e}")
            time.sleep(2 ** attempt)

        except (requests.exceptions.RequestException, ValueError) as e:
            last_error = e
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {location_name} {year}: {e}")
            time.sleep(2 ** attempt)

    logging.error(f"Error fetching weather for {location_name} {year}: {last_error}")
    raise last_error


def fetch_all(locations: list[dict], years: list[int]) -> dict[str, Path]:
    """Fetch every (location, year) combination.

    Catches per-item failures here, not inside fetch_weather_year, so
    one bad chunk doesn't abort the rest of the batch.

    Args:
        locations: List of dicts with keys "name", "lat", "lon".
        years: List of calendar years to fetch for every location.

    Returns:
        dict: Mapping of "{location}_{year}" -> cached file path, for
        every successful fetch.
    """
    results = {}
    failures = []

    for loc in locations:
        for year in years:
            key = f"{loc['name']}_{year}"
            try:
                results[key] = fetch_weather_year(loc["name"], loc["lat"], loc["lon"], year)
            except (requests.exceptions.RequestException, ValueError) as e:
                logging.error(f"Skipping {key} after exhausting retries: {e}")
                failures.append(key)
            time.sleep(1)

    if failures:
        logging.error(f"{len(failures)} chunk(s) failed: {failures}")

    return results


if __name__ == "__main__":
    setup_logging()

    config = load_config()
    locations = get_macrozones_nuts2(config)
    start_year, end_year = get_date_range(config)
    years = list(range(start_year, end_year + 1))

    results = fetch_all(locations, years)
    expected = len(locations) * len(years)

    if len(results) < expected:
        sys.exit(1)