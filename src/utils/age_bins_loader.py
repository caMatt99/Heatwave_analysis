"""
Loader for config/age_bins.yaml - named age-bin grouping schemes used
to collapse the fine-grained Eurostat age brackets in
mortality_by_age_weekly into coarser, analysis-specific groups.

Kept separate from locations_loader.py on purpose: the two configs
change for unrelated reasons (geography is structural and stable; age
schemes are analytical and may be revised per-analysis) and have
different, smaller consumer sets (build_analysis_views.py and the
Streamlit dashboard, not the extract/staging layers).

IMPORTANT: schemes here operate on age codes as they appear in
mortality_by_age_weekly.parquet, i.e. AFTER build_weekly_dataset_by_age.py
has already collapsed "Y85-89" + "Y_GE90" into "Y_GE85". This module
does not touch anything upstream of that reconciliation step.
"""

from pathlib import Path

import yaml

CONFIG_PATH = Path("config/age_bins.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict:
    """Load and parse the age bins config file.

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


def get_scheme(config: dict, scheme_name: str) -> dict[str, str]:
    """Build the age-code -> bin-label mapping for one named scheme.

    Args:
        config: Parsed config as returned by load_config().
        scheme_name: Key under `schemes` in age_bins.yaml, e.g.
            "standardization" or "stratified_heat".

    Returns:
        dict: e.g. {"Y_LT5": "0-64", "Y5-9": "0-64", ..., "Y_GE85": "85+"}
        Flattened for direct use with DataFrame.map()/replace().

    Raises:
        KeyError: If scheme_name isn't defined in the config.
    """
    try:
        scheme = config["schemes"][scheme_name]
    except KeyError:
        available = list(config.get("schemes", {}).keys())
        raise KeyError(
            f"Unknown age bin scheme '{scheme_name}'. Available: {available}"
        )

    mapping = {}
    for bin_def in scheme["bins"]:
        label = bin_def["label"]
        for age_code in bin_def["ranges"]:
            mapping[age_code] = label
    return mapping


def get_bin_labels(config: dict, scheme_name: str) -> list[str]:
    """Return a scheme's bin labels in definition order (for consistent
    plotting/sorting order in the dashboard, rather than relying on
    whatever order groupby happens to produce).

    Args:
        config: Parsed config as returned by load_config().
        scheme_name: Key under `schemes` in age_bins.yaml.

    Returns:
        list[str]: Bin labels in the order they're defined in the YAML.
    """
    scheme = config["schemes"][scheme_name]
    return [bin_def["label"] for bin_def in scheme["bins"]]


def validate_scheme(config: dict, scheme_name: str, age_codes_in_data: set[str]) -> None:
    """Fail loudly if the data contains age codes the scheme doesn't
    account for, rather than silently dropping those rows on join/map.

    Args:
        config: Parsed config as returned by load_config().
        scheme_name: Key under `schemes` in age_bins.yaml.
        age_codes_in_data: Set of distinct `age` values actually present
            in mortality_by_age_weekly (or the DataFrame being binned).

    Raises:
        ValueError: If any age code in the data isn't covered by the
            scheme's ranges.
    """
    mapping = get_scheme(config, scheme_name)
    missing = age_codes_in_data - set(mapping.keys())
    if missing:
        raise ValueError(
            f"Age bin scheme '{scheme_name}' does not cover age codes "
            f"present in the data: {sorted(missing)}. Update "
            f"config/age_bins.yaml to include them."
        )