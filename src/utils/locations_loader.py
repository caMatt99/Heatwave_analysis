"""
Loader for config/locations.yaml - the single source of truth for the
9 macrozones (3 countries x 3 zones) used across the pipeline.
Both eurostat_client.py and openmeteo_client.py read through this
module instead of parsing the YAML themselves or hardcoding data.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path("config/locations.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and parse the locations config file.

    Args:
        path: Path to the YAML config file.

    Returns:
        dict: Parsed YAML content.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the file isn't valid YAML.
    """
    with open(path) as f:
        return yaml.safe_load(f)


def get_macrozones_nuts2(config: dict) -> dict[str, list[str]]:
    """Build the macrozone-name -> NUTS2-codes mapping used by
    eurostat_client.fetch_all().

    Args:
        config: Parsed config as returned by load_config().

    Returns:
        dict: e.g. {"IT_nord": ["ITC1", ...], "IT_centro": [...], ...}
    """
    result = {}
    for country in config["countries"].values():
        for zone in country["macrozones"].values():
            result[zone["name"]] = zone["nuts2"]
    return result


def get_nuts2_coordinates(config: dict) -> list[dict]:
    """Build the per-NUTS2 location list used by openmeteo_client.fetch_all().

    Unlike get_locations_coordinates (macrozone-level), this returns
    one entry per NUTS2 region, so each region gets its own weather
    series from its own capital city, instead of sharing a single
    macrozone centroid.

    Args:
        config: Parsed config as returned by load_config().

    Returns:
        list[dict]: One entry per NUTS2 code, each with "name" (the
        NUTS2 code, used as cache filename identifier), "lat", "lon".
    """
    result = []
    for country in config["countries"].values():
        for zone in country["macrozones"].values():
            for nuts2_code, region in zone["regions"].items():
                result.append({
                    "name": nuts2_code,
                    "lat": region["lat"],
                    "lon": region["lon"],
                })
    return result

def get_date_range(config: dict) -> tuple[int, int]:
    """Return the (start_year, end_year) range from config.

    Args:
        config: Parsed config as returned by load_config().

    Returns:
        tuple: (start_year, end_year), both inclusive.
    """
    date_range = config["date_range"]
    return date_range["start_year"], date_range["end_year"]