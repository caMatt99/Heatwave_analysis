# Heatwave & Mortality Pipeline – Data Engineering Project

## Overview

This project builds an end-to-end data pipeline and dashboard for a Data Analysis for Public Health (DAPH) course project. It analyzes the relationship between temperature and weekly all-cause mortality across **59 NUTS2 regions** spanning **France**, **Italy**, and **Spain**. Each region also belongs to one of three geographic macrozones (`nord`, `centro`, `sud`) per country, used as an optional grouping dimension for aggregated comparisons — the actual unit of analysis is the individual NUTS2 region, not the macrozone.

The pipeline extracts and stages two data sources:

1. **Eurostat** (`DEMO_R_MWK2_TS`, `DEMO_R_MWK2_05`, `DEMO_R_PJANGROUP`) — weekly all-cause mortality (total and by age/sex) and annual population by age/sex, all at NUTS2 resolution.
2. **Open-Meteo** (historical weather archive) — daily temperature, precipitation, wind, and sunshine data, one series per NUTS2 region (sourced from that region's capital city).

The analytical approach applies epidemiological measures from DAPH course chapters 1–5 (measures of occurrence, standardization, association measures, bias/confounding) — explicitly **not** advanced statistical models such as DLNM. This is an ecological time-series study using all-cause mortality; heat-attributable deaths require a counterfactual model and cannot be read directly from the data.

The final deliverable is a Streamlit dashboard visualizing climate effects on mortality.

---

## System Requirements

- Python 3.10 or later
- `pandas`, `pyarrow`, `pyyaml`, `requests`, `eurostat`

---

## Installation

```bash
python -m venv heatwave_venv
source heatwave_venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Run the extraction stage

Fetches raw data from Eurostat (mortality, population) and Open-Meteo (daily weather) into `data/raw/`, using `config/locations.yaml` as the single source of truth for NUTS2 regions and coordinates.

```bash
python -m src.run_pipeline
```

Extraction is cache-aware: already-downloaded files are skipped on re-run, so an interrupted run (e.g. due to an Open-Meteo rate limit) can simply be re-launched to resume from where it left off.

### 2. Run the staging stage

Transforms the raw extracts into clean, long-format tables at their native grain, applying coverage trimming, missing-data flags, and (for weather) heat wave detection.

```bash
python -m src.staging.eurostat_staging
python -m src.staging.openmeteo_staging
```

### 3. Build the analytical datasets

Joins the staged Eurostat and Open-Meteo tables into the final weekly analytical datasets, broadcasting annual population figures to weekly grain at query time.

```bash
python -m src.transform.build_weekly_dataset          # total mortality, (geo, year, week) grain
python -m src.transform.build_weekly_dataset_by_age    # age/sex-stratified, (geo, sex, age, year, week) grain
```

### 4. Build the region lookup table

Generates a small human-readable lookup (NUTS2 code → region name, macrozone, country, capital city) from `config/locations.yaml`.

```bash
python -m src.transform.build_dim_region
```

### 5. Build human-readable analysis views

Left-joins the region lookup onto both analytical datasets, adding `region_name`/`macrozone`/`country` columns alongside the underlying `geo` key — for plotting, reporting, and the dashboard. The base analytical datasets themselves are left untouched (geo-keyed only), so anything that doesn't need human-readable labels keeps using them directly.

```bash
python -m src.transform.build_analysis_view
```

---

## Script Descriptions

### Extraction (`src/extract/`)

| Script | Description |
|---|---|
| `eurostat_client.py` | Fetches the three Eurostat datasets (mortality total, mortality by age, population by age) via the `eurostat` package, one call per macrozone, with retry/backoff, atomic cache writes, and a summary log line (cache hits / downloaded / failed) instead of one line per chunk |
| `openmeteo_client.py` | Fetches daily weather via the Open-Meteo archive API, one call per NUTS2 region per year, with retry/backoff, explicit 429 rate-limit handling (respects `Retry-After`, stops the whole batch after repeated consecutive rate limits instead of exhausting the queue), atomic cache writes, and the same summary log line |

### Staging (`src/staging/`)

| Script | Description |
|---|---|
| `common.py` | Shared utilities used by both staging modules: atomic parquet writes, ISO-week-to-date conversion |
| `eurostat_staging.py` | Melts the wide Eurostat parquet files into long tables (`mortality_total`, `mortality_by_age`, `population_by_age`); trims each series to its real coverage window (leading/trailing gaps = no data collected, not missing) and flags any remaining internal gap as `is_missing` |
| `openmeteo_staging.py` | Loads daily weather JSON into a tidy daily table; interpolates short gaps in temperature only; detects heat wave days via a per-region, per-month climatological threshold combined with an absolute temperature floor (empirically calibrated so summer extremes are never suppressed and winter noise is always filtered), skipped with an explicit `NA` for regions with insufficient history; aggregates to ISO week with a distinct aggregation function per variable |

### Transform (`src/transform/`)

| Script | Description |
|---|---|
| `build_weekly_dataset.py` | Joins staged `mortality_total` and `weather_weekly` (inner, on geo/year/week), then left-joins annual `population_by_age` aggregated to a single per-region-per-year total, broadcast to weekly grain at query time. Writes `mortality_weather_weekly.parquet`/`.csv` |
| `build_weekly_dataset_by_age.py` | Same join logic, but keeps the age/sex breakdown. Reconciles the age-bin mismatch between mortality (which splits 85+ into `Y85-89`/`Y_GE90`) and population (single `Y_GE85` bin) by collapsing mortality's two oldest brackets before joining. Writes `mortality_by_age_weekly.parquet`/`.csv`, for age-standardization and age-specific heat-vulnerability analysis |
| `build_dim_region.py` | Builds `dim_region.csv`: NUTS2 code → official region name, macrozone, country, capital city, sourced from `locations.yaml` plus a verified name lookup. Fails loudly if any NUTS2 code has no matching region name |
| `build_analysis_view.py` | Left-joins `dim_region` onto both analytical datasets, writing labeled copies to `data/analytics/views/` — `geo` is kept, not replaced, so the underlying join key stays available alongside the human-readable columns |

### Configuration (`src/utils/`, `config/`)

| File | Description |
|---|---|
| `locations.yaml` | Single source of truth: 9 macrozones × NUTS2 region codes, each with its capital-city coordinates (used individually for weather extraction), plus excluded regions and the extraction date range |
| `locations_loader.py` | Loader functions building the macrozone→NUTS2 mapping (for Eurostat) and the per-NUTS2 coordinate list (for Open-Meteo) from `locations.yaml` |
| `logging_config.py` | Centralized logging setup used by every script in the pipeline |
| `parquet_to_csv.py` | Standalone one-off utility (not part of the pipeline) for converting any parquet file to CSV on demand |

---

## Repository Structure

```
Heatwave_analysis/
├── config/
│   └── locations.yaml
├── src/
│   ├── extract/
│   │   ├── eurostat_client.py
│   │   └── openmeteo_client.py
│   ├── staging/
│   │   ├── common.py
│   │   ├── eurostat_staging.py
│   │   └── openmeteo_staging.py
│   ├── transform/
│   │   ├── build_weekly_dataset.py
│   │   ├── build_weekly_dataset_by_age.py
│   │   ├── build_dim_region.py
│   │   └── build_analysis_view.py
│   ├── utils/
│   │   ├── locations_loader.py
│   │   ├── logging_config.py
│   │   └── parquet_to_csv.py
│   └── run_pipeline.py
├── data/
│   ├── raw/
│   │   ├── eurostat/
│   │   └── openmeteo/
│   ├── staging/
│   │   ├── eurostat/
│   │   └── openmeteo/
│   └── analytics/
│       ├── dim_region.csv
│       ├── mortality_weather_weekly.parquet
│       ├── mortality_weather_weekly.csv
│       ├── mortality_by_age_weekly.parquet
│       ├── mortality_by_age_weekly.csv
│       └── views/
│           ├── mortality_weather_weekly_labeled.parquet
│           ├── mortality_weather_weekly_labeled.csv
│           ├── mortality_by_age_weekly_labeled.parquet
│           └── mortality_by_age_weekly_labeled.csv
├── requirements.txt
├── LICENSE
└── README.md
```

