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
    get_nuts2_coordinates,
    get_date_range,
)

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
RAW_DIR = Path("data/raw/openmeteo")
MAX_RETRIES = 3
TIMEOUT_SECONDS = 60

# If the API returns 429 this many times in a row (across different
# location/year requests, not just retries of the same one), stop the
# whole batch instead of burning through the rest of the queue - every
# remaining request would fail identically until the quota window resets.
MAX_CONSECUTIVE_RATE_LIMITS = 3
RATE_LIMIT_BACKOFF_SECONDS = 60  # fallback if the API gives no Retry-After

DAILY_VARIABLES = [
    "temperature_2m_mean",
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "sunshine_duration",
]


class RateLimitStop(Exception):
    """Raised to abort the whole batch when the rate limit keeps firing."""
    pass


def _build_params(lat: float, lon: float, year: int) -> dict:
    """Build the query params for one location/year request."""
    return {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
    }


def _save_json(path: Path, data: dict) -> None:
    """Write JSON to a temp file, then rename it into place."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f)
        f.flush()
    tmp_path.rename(path)


def fetch_weather_year(location_name: str, lat: float, lon: float, year: int) -> Path:
    """Fetch and cache one year of daily weather for one location.

    Retries transient failures (timeouts, connection errors, malformed
    payloads, and 429 rate-limit responses with backoff). A genuine 4xx
    client error (e.g. a malformed request) is not retried, since the
    same request would fail identically every time.

    Args:
        location_name: Identifier used in the cache filename (e.g. "ITC1").
        lat: Latitude of the location.
        lon: Longitude of the location.
        year: Calendar year to fetch.

    Returns:
        Path to the cached JSON file.

    Raises:
        requests.exceptions.RequestException: If the API call fails
            (immediately for a genuine 4xx, or after exhausting retries).
        ValueError: If the response is malformed on every retry.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target_path = RAW_DIR / f"{location_name}_{year}.json"

    if target_path.exists():
        logging.debug(f"Cache hit: {location_name} {year}, skipping API call")
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
            status = e.response.status_code if e.response is not None else None

            if status == 429:
                retry_after = e.response.headers.get("Retry-After")
                wait_seconds = float(retry_after) if retry_after else RATE_LIMIT_BACKOFF_SECONDS
                logging.warning(
                    f"Rate limited (429) for {location_name} {year}, "
                    f"waiting {wait_seconds}s before retry {attempt}/{MAX_RETRIES}"
                )
                last_error = e
                time.sleep(wait_seconds)
                continue

            if status is not None and 400 <= status < 500:
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
    one bad chunk doesn't abort the rest of the batch - EXCEPT for
    rate-limit (429) errors: if MAX_CONSECUTIVE_RATE_LIMITS different
    (location, year) requests in a row all come back rate-limited, the
    whole batch is stopped. At that point every other request would
    fail identically too - continuing would just burn through the rest
    of the queue for nothing and spam the logs.

    Args:
        locations: List of dicts with keys "name", "lat", "lon".
        years: List of calendar years to fetch for every location.

    Returns:
        dict: Mapping of "{location}_{year}" -> cached file path, for
        every successful fetch.

    Raises:
        RateLimitStop: If the rate limit keeps firing across
            MAX_CONSECUTIVE_RATE_LIMITS consecutive distinct requests.
    """
    results = {}
    failures = []
    cache_hits = 0
    downloaded = 0
    consecutive_rate_limits = 0

    for loc in locations:
        for year in years:
            key = f"{loc['name']}_{year}"

            # Skip cache hits without touching the rate-limit counter.
            target_path = RAW_DIR / f"{loc['name']}_{year}.json"
            if target_path.exists():
                results[key] = fetch_weather_year(loc["name"], loc["lat"], loc["lon"], year)
                cache_hits += 1
                continue

            try:
                results[key] = fetch_weather_year(loc["name"], loc["lat"], loc["lon"], year)
                downloaded += 1
                consecutive_rate_limits = 0
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    consecutive_rate_limits += 1
                    logging.error(f"Skipping {key} after exhausting retries (rate limited): {e}")
                    failures.append(key)
                    if consecutive_rate_limits >= MAX_CONSECUTIVE_RATE_LIMITS:
                        logging.error(
                            f"Rate limit hit on {consecutive_rate_limits} consecutive "
                            f"requests - stopping the batch. Already fetched: {len(results)}. "
                            f"Re-run this same command later: cached files are skipped "
                            f"automatically, only the missing ones will be requested."
                        )
                        raise RateLimitStop(
                            f"Stopped after {consecutive_rate_limits} consecutive 429s. "
                            f"{len(results)} locations/years already cached and safe."
                        )
                    continue
                logging.error(f"Skipping {key} after exhausting retries: {e}")
                failures.append(key)
                consecutive_rate_limits = 0
            except (requests.exceptions.RequestException, ValueError) as e:
                logging.error(f"Skipping {key} after exhausting retries: {e}")
                failures.append(key)
                consecutive_rate_limits = 0
            time.sleep(1)

    logging.info(
        f"Weather extraction: {cache_hits} cache hit(s), {downloaded} downloaded, "
        f"{len(failures)} failed ({cache_hits + downloaded + len(failures)} total)"
    )
    if failures:
        logging.error(f"{len(failures)} chunk(s) failed: {failures}")

    return results


if __name__ == "__main__":
    setup_logging()

    config = load_config()
    locations = get_nuts2_coordinates(config)
    start_year, end_year = get_date_range(config)
    years = list(range(start_year, end_year + 1))

    try:
        results = fetch_all(locations, years)
    except RateLimitStop as e:
        logging.error(str(e))
        sys.exit(2)  # distinct exit code: "stopped due to rate limit", not a hard failure

    expected = len(locations) * len(years)
    if len(results) < expected:
        sys.exit(1)