# Heatwave & Mortality Pipeline – Data Engineering Project

## Overview

This project builds an end-to-end data pipeline and dashboard for a Data Analysis for Public Health (DAPH) course project. It analyzes the relationship between temperature and weekly all-cause mortality across nine macrozones: **France**, **Italy**, and **Spain**, each divided into three geographic areas (`nord`, `centro`, `sud`) based on NUTS2 regional groupings.

The pipeline extracts and stages two data sources:

1. **Eurostat** (`DEMO_R_MWK2_TS`, `DEMO_R_MWK2_05`, `DEMO_R_PJANGROUP`) — weekly all-cause mortality and annual population, both at NUTS2 resolution.
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

### 3. Build the analytical dataset *(planned)*

Joins the staged Eurostat and Open-Meteo tables into the final weekly analytical dataset, broadcasting annual population figures to weekly grain at query time.

```bash
python -m src.transform.build_weekly_dataset
```

---

## Script Descriptions

### Extraction (`src/extract/`)

| Script | Description |
|---|---|
| `eurostat_client.py` | Fetches the three Eurostat datasets (mortality total, mortality by age, population by age) via the `eurostat` package, one call per macrozone, with retry/backoff and atomic cache writes |
| `openmeteo_client.py` | Fetches daily weather via the Open-Meteo archive API, one call per NUTS2 region per year, with retry/backoff, explicit 429 rate-limit handling (respects `Retry-After`, stops the whole batch after repeated consecutive rate limits instead of exhausting the queue), and atomic cache writes |

### Staging (`src/staging/`)

| Script | Description |
|---|---|
| `common.py` | Shared utilities used by both staging modules: atomic parquet writes, ISO-week-to-date conversion |
| `eurostat_staging.py` | Melts the wide Eurostat parquet files into long tables (`mortality_total`, `mortality_by_age`, `population_by_age`); trims each series to its real coverage window (leading/trailing gaps = no data collected, not missing) and flags any remaining internal gap as `is_missing` |
| `openmeteo_staging.py` | Loads daily weather JSON into a tidy daily table; interpolates short gaps in temperature only; detects heat wave days via a per-region, per-month climatological threshold (skipped, with an explicit `NA`, for regions with insufficient history); aggregates to ISO week with a distinct aggregation function per variable |

### Configuration (`src/utils/`, `config/`)

| File | Description |
|---|---|
| `locations.yaml` | Single source of truth: 9 macrozones × NUTS2 region codes × per-region capital-city coordinates, plus excluded regions and the extraction date range |
| `locations_loader.py` | Loader functions building the macrozone→NUTS2 mapping (for Eurostat) and the per-NUTS2 coordinate list (for Open-Meteo) from `locations.yaml` |
| `logging_config.py` | Centralized logging setup used by every script in the pipeline |

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
│   │   └── build_weekly_dataset.py      # planned
│   ├── utils/
│   │   ├── locations_loader.py
│   │   └── logging_config.py
│   └── run_pipeline.py
├── data/
│   ├── raw/
│   │   ├── eurostat/
│   │   └── openmeteo/
│   ├── staging/
│   │   ├── eurostat/
│   │   └── openmeteo/
│   └── analytics/                       # planned
├── requirements.txt
├── LICENSE
└── README.md
```

---

