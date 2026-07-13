"""
Builds dim_region.csv: a small human-readable lookup table mapping
each NUTS2 code used in the pipeline to its official region name,
macrozone, country, and capital city.

This is NOT part of the analytical pipeline itself - mortality_total,
weather_weekly, population_by_age, and the joined weekly dataset all
keep NUTS2 codes as their key, by design (stable, unambiguous,
interoperable with other Eurostat data). This table exists purely for
the presentation layer (dashboard, plots, reports): join it in at the
very last step, when data is shown to a human, not upstream in the
pipeline.

Official NUTS2 region names are not present in locations.yaml (which
only has NUTS2 codes and capital-city coordinates), so they're listed
here explicitly, verified against Eurostat's NUTS nomenclature.
"""

import csv
import logging
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.utils.locations_loader import load_config

OUTPUT_PATH = Path("data/analytics/dim_region.csv")

REGION_NAMES = {
    # Italy - nord
    "ITC1": "Piemonte",
    "ITC2": "Valle d'Aosta",
    "ITC3": "Liguria",
    "ITC4": "Lombardia",
    "ITH1": "Provincia Autonoma di Bolzano/Bozen",
    "ITH2": "Provincia Autonoma di Trento",
    "ITH3": "Veneto",
    "ITH4": "Friuli-Venezia Giulia",
    "ITH5": "Emilia-Romagna",
    # Italy - centro
    "ITI1": "Toscana",
    "ITI2": "Umbria",
    "ITI3": "Marche",
    "ITI4": "Lazio",
    "ITG2": "Sardegna",

    # Italy - sud
    "ITF1": "Abruzzo",
    "ITF2": "Molise",
    "ITF3": "Campania",
    "ITF4": "Puglia",
    "ITF5": "Basilicata",
    "ITF6": "Calabria",
    "ITG1": "Sicilia",
    # France - nord
    "FRE1": "Nord-Pas-de-Calais",
    "FRE2": "Picardie",
    "FRD1": "Basse-Normandie",
    "FRD2": "Haute-Normandie",
    "FR10": "Île-de-France",
    "FRF1": "Alsace",
    "FRF2": "Champagne-Ardenne",
    "FRF3": "Lorraine",
    "FRH0": "Bretagne",
    "FRB0": "Centre-Val de Loire",
    # France - centro
    "FRG0": "Pays de la Loire",
    "FRC1": "Bourgogne",
    "FRC2": "Franche-Comté",
    "FRI3": "Poitou-Charentes",
    "FRI2": "Limousin",
    "FRK1": "Auvergne",
    "FRK2": "Rhône-Alpes",
    "FRI1": "Aquitaine",
    # France - sud
    "FRJ1": "Languedoc-Roussillon",
    "FRJ2": "Midi-Pyrénées",
    "FRL0": "Provence-Alpes-Côte d'Azur",
    "FRM0": "Corse",
    # Spain - nord
    "ES11": "Galicia",
    "ES12": "Principado de Asturias",
    "ES13": "Cantabria",
    "ES21": "País Vasco",
    "ES22": "Comunidad Foral de Navarra",
    "ES23": "La Rioja",
    # Spain - centro
    "ES41": "Castilla y León",
    "ES30": "Comunidad de Madrid",
    "ES42": "Castilla-La Mancha",
    "ES43": "Extremadura",
    "ES24": "Aragón",
    # Spain - sud
    "ES51": "Cataluña",
    "ES52": "Comunidad Valenciana",
    "ES53": "Illes Balears",
    "ES62": "Región de Murcia",
    "ES61": "Andalucía",
}


def build_dim_region(config: dict) -> list[dict]:
    """Build the dim_region rows from locations.yaml + REGION_NAMES.

    Args:
        config: Parsed config as returned by locations_loader.load_config().

    Returns:
        list[dict]: One row per NUTS2 code, each with geo, country,
        macrozone, region_name, capital_city.

    Raises:
        KeyError: If a NUTS2 code in locations.yaml has no entry in
            REGION_NAMES - this is deliberately loud rather than
            silently falling back to the code itself, since a missing
            name would otherwise go unnoticed until someone spots a
            raw code in the dashboard.
    """
    rows = []
    for country_key, country in config["countries"].items():
        for macrozone_key, zone in country["macrozones"].items():
            for nuts2_code, region in zone["regions"].items():
                if nuts2_code not in REGION_NAMES:
                    raise KeyError(
                        f"No official region name found for {nuts2_code} "
                        f"({country_key}/{macrozone_key}) - add it to REGION_NAMES"
                    )
                rows.append({
                    "geo": nuts2_code,
                    "country": country["code"],
                    "macrozone": macrozone_key,
                    "region_name": REGION_NAMES[nuts2_code],
                    "capital_city": region["city"],
                })
    return rows


def run() -> None:
    """Build dim_region and write it to OUTPUT_PATH as CSV."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    config = load_config()
    rows = build_dim_region(config)

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["geo", "country", "macrozone", "region_name", "capital_city"])
        writer.writeheader()
        writer.writerows(rows)

    logging.info(f"Built dim_region: {len(rows)} rows -> {OUTPUT_PATH}")


if __name__ == "__main__":
    setup_logging()
    run()