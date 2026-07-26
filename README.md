# Heatwave & Mortality Pipeline – Data Engineering Project

## Overview

This project builds an end-to-end data pipeline, epidemiological analysis, and interactive dashboard for a Data Analysis for Public Health (DAPH) course project. It analyzes the relationship between temperature and weekly all cause mortality across **59 NUTS2 regions** spanning **France**, **Italy**, and **Spain**. Each region also belongs to one of three geographic macrozones (`nord`, `centro`, `sud`) per country, used as an optional grouping dimension for aggregated comparisons the actual unit of analysis is the individual NUTS2 region. 

The pipeline extracts and stages two data sources:

1. **Eurostat** (`DEMO_R_MWK2_TS`, `DEMO_R_MWK2_05`, `DEMO_R_PJANGROUP`) weekly all cause mortality (total and by age/sex) and annual population by age/sex, all at NUTS2 resolution.
2. **Open-Meteo** (historical weather archive) daily temperature, precipitation, wind, and sunshine data, one series per NUTS2 region (sourced from that region's capital city).

The analytical approach applies epidemiological measures such as: measures of occurrence, standardization, association measures, bias/confounding. This is an ecological time-series study using all-cause mortality; heat-attributable deaths require a counterfactual model and cannot be read directly from the data.

The project has three consumers of the pipeline's output, all reading the same analysis views: a Jupyter **notebook** (static descriptive analysis), a shared **analysis module** (`epi_metrics.py`), and an interactive **Streamlit dashboard**.

---

## System Requirements

- Python 3.10 or later
- Pipeline: `pandas`, `pyarrow`, `pyyaml`, `requests`, `eurostat`, `numpy`
- Analysis: `scikit-learn` (polynomial temperature-mortality regression)
- Dashboard: `streamlit`, `plotly`

All are pinned in `requirements.txt`.

---

## Installation

```bash
python -m venv heatwave_venv
source heatwave_venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

The project has two phases: build the data (steps 1–5, the pipeline), then consume it (the notebook and/or the dashboard). The pipeline produces the analysis views in `data/analytics/views/`, which are the single contract every downstream consumer depends on.

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

Generates a small human readable lookup (NUTS2 code → region name, macrozone, country, capital city) from `config/locations.yaml`.

```bash
python -m src.transform.build_dim_region
```

### 5. Build human-readable analysis views

Left-joins the region lookup onto both analytical datasets, adding `region_name`/`macrozone`/`country` columns alongside the underlying `geo` key for plotting, reporting, and the dashboard. The base analytical datasets themselves are left untouched (geo-keyed only), so anything that doesn't need human-readable labels keeps using them directly.

For the age-stratified dataset specifically, this step also produces one additional regrouped view per named scheme in `config/age_bins.yaml`  currently `stratified_heat`, which collapses the fine-grained Eurostat age brackets into `under_65` / `65-74` / `75-84` / `85+` for effect-modification analysis (does age change how strongly heat is associated with mortality?). This is **not** used for the notebook's age-standardization, which relies on the fine-grained quinquennial bins and the full ESP2013 weight table as is a coarser scheme here would not line up with those weights, so `age_bins.yaml` intentionally has no "standardization" scheme.

```bash
python -m src.transform.build_analysis_view
```

### 6. Explore the analysis notebook

`notebooks/01_Descriptive_analysis.ipynb` walks through the epidemiological analysis on the labeled views: weekly mortality trends, crude and ESP2013-standardized rates, heatwave relative risk (with the seasonal-confounding correction), a temperature-mortality regression (linear plus a polynomial fit that captures the U/J-shaped cold and heat arms), descriptive excess mortality against a same-week baseline, age structure as a confounder, and effect modification by age group. It imports the shared calculation functions from `src/analysis/epi_metrics.py` rather than redefining them, so the numbers match the dashboard exactly.

```bash
jupyter notebook notebooks/01_Descriptive_analysis.ipynb
```

### 7. Launch the interactive dashboard

The Streamlit dashboard is the interactive counterpart to the notebook: same calculations, but parameterized by user controlled filters (time window, heatwave lag, warm-season restriction, geography). Run it from the repo root, after the views exist (step 5).

```bash
streamlit run dashboard/app.py
```

If a view is missing, the app says which build step to run rather than failing silently. On macOS, if the `streamlit` command resolves to a system Python instead of the venv, launch it as a module: `python -m streamlit run dashboard/app.py`.

---

## Script Descriptions

### Extraction (`src/extract/`)

| Script | Description |
|---|---|
| `eurostat_client.py` | Fetches the three Eurostat datasets (mortality total, mortality by age, population by age) via the `eurostat` package, one call per macrozone, with retry/backoff, atomic cache writes, and a summary log line (cache hits / downloaded / failed) instead of one line per chunk |
| `openmeteo_client.py` | Fetches daily weather via the Open-Meteo archive API, one call per NUTS2 region per year, with retry/backoff, explicit 429 rate limit handling (respects `Retry-After`, stops the whole batch after repeated consecutive rate limits instead of exhausting the queue), atomic cache writes, and the same summary log line |

### Staging (`src/staging/`)

| Script | Description |
|---|---|
| `common.py` | Shared utilities used by both staging modules: atomic parquet writes, ISO week to date conversion |
| `eurostat_staging.py` | Melts the wide Eurostat parquet files into long tables (`mortality_total`, `mortality_by_age`, `population_by_age`); trims each series to its real coverage window (leading/trailing gaps = no data collected, not missing) and flags any remaining internal gap as `is_missing` |
| `openmeteo_staging.py` | Loads daily weather JSON into a tidy daily table; interpolates short gaps in temperature only; detects heat wave days via a per region, per month climatological threshold combined with an absolute temperature floor (empirically calibrated so summer extremes are never suppressed and winter noise is always filtered), skipped with an explicit `NA` for regions with insufficient history; aggregates to ISO week with a distinct aggregation function per variable |

### Transform (`src/transform/`)

| Script | Description |
|---|---|
| `build_weekly_dataset.py` | Joins staged `mortality_total` and `weather_weekly` (inner, on geo/year/week), then left-joins annual `population_by_age` aggregated to a single per region per year total, broadcast to weekly grain at query time. Writes `mortality_weather_weekly.parquet`/`.csv` |
| `build_weekly_dataset_by_age.py` | Same join logic, but keeps the age/sex breakdown. Reconciles the age-bin mismatch between mortality (which splits 85+ into `Y85-89`/`Y_GE90`) and population (single `Y_GE85` bin) by collapsing mortality's two oldest brackets before joining. Writes `mortality_by_age_weekly.parquet`/`.csv`, for age standardization and age specific heat vulnerability analysis |
| `build_dim_region.py` | Builds `dim_region.csv`: NUTS2 code → official region name, macrozone, country, capital city, sourced from `locations.yaml` plus a verified name lookup. Fails loudly if any NUTS2 code has no matching region name |
| `build_analysis_view.py` | Left-joins `dim_region` onto both analytical datasets, writing labeled copies to `data/analytics/views/` — `geo` is kept, not replaced, so the underlying join key stays available alongside the human readable columns. For `mortality_by_age_weekly`, also regroups the labeled view into coarser age bins per scheme in `age_bins.yaml` (currently `stratified_heat`), aggregating `deaths`/`population` (summed) and their `is_missing` flags (`any`) across the collapsed age codes, while carrying weather columns through unchanged (they're duplicated identically across ages within the same geo/year/week, so they're taken as-is rather than summed) |

### Analysis (`src/analysis/`)

| Script | Description |
|---|---|
| `epi_metrics.py` | Shared epidemiological calculation functions imported by **both** the notebook and the dashboard, so a formula lives in exactly one tested place and the two consumers can never diverge. Includes the ESP2013 weight table, `age_specific_rates()` and `direct_standardize()` (age standardization), `relative_risk()` (from already-aggregated exposed/unexposed sums, so the caller defines what "exposed" means — heatwave weeks, a season, a country, an age group), `restrict_to_warm_season()` (seasonal-confounding control), `linear_fit()` (the straight-line temperature-mortality regression), `polynomial_fit()` (a scikit-learn polynomial regression capturing the non-linear U/J-shaped temperature-mortality curve, with R² for comparing degrees), and `excess_mortality()` (descriptive excess deaths against a same-ISO-week, non-heatwave baseline built from non-exposed years only, so the exposure never contaminates its own baseline). All functions are pure in the sense that they take already-filtered inputs and never decide which rows to exclude, leaving that analysis-specific choice to the caller |

### Dashboard (`dashboard/`)

The dashboard is a thin presentation layer over the pipeline: it contains no epidemiological logic (that stays in `epi_metrics.py`) and reads only the analysis views (never the notebook or intermediate files). It depends on `src/` in one direction only  `src/` knows nothing about the dashboard.

| File | Description |
|---|---|
| `app.py` | The Streamlit app. Orchestrates load → global filters → per-tab compute → render. Seven tabs: **Overview** (KPI header, RR map, headline verdict), **Relative Risk** (pooled + regional breakdown), **By Age** (effect modification, section 7 of the notebook made interactive), **Compare Regions** (side-by-side RR for a user-picked set of regions, across countries), **Excess Mortality** (descriptive excess deaths vs a same-ISO-week baseline, per year and per country, flagged as descriptive not causal), **Temp vs Mortality** (regression with the winter arm kept visible, offering a straight-line fit plus an adjustable-degree polynomial fit for the U/J curve), and **Trend** (weekly mortality over time, showing the selected single region alone or a country's macrozones). A sidebar holds the global controls every tab respects: geography, a week-range time window, a 0–4 week heatwave lag, and the warm season toggle |
| `views_loader.py` | Cached data-access layer (`st.cache_data`). Loads the labeled views and, for the map, the NUTS2 coordinates via `src.utils.locations_loader` (so coordinates stay sourced from `locations.yaml`, not duplicated). Fails with an explicit "run build_analysis_view first" message if a view is missing |
| `components.py` | Reusable UI rendering: the KPI header, the plain-language RR verdict (which flags an implausible RR < 1 as likely seasonal confounding rather than "heat protects"), the region map (scatter-on-map coloured by RR, anchored at RR=1 so filtered views stay comparable), and the collapsible method/limitations panel. Enforces consistent conventions — red = higher risk everywhere, RR = 1 reference line always visible |

### Configuration & utilities (`src/utils/`, `config/`)

| File | Description |
|---|---|
| `locations.yaml` | Single source of truth: 9 macrozones × NUTS2 region codes, each with its capital-city coordinates (used individually for weather extraction), plus excluded regions and the extraction date range |
| `locations_loader.py` | Loader functions building the macrozone→NUTS2 mapping (for Eurostat) and the per-NUTS2 coordinate list (for Open-Meteo, and for the dashboard map) from `locations.yaml` |
| `age_bins.yaml` | Named age-bin grouping schemes used to collapse the fine-grained Eurostat age brackets in `mortality_by_age_weekly` into coarser, analysis-specific groups. Kept separate from `locations.yaml` since it changes independently (analytical choice, not geography) and has a different, smaller consumer set (`build_analysis_view.py`, the Streamlit dashboard). Schemes operate on age codes as they exist **after** `build_weekly_dataset_by_age.py`'s reconciliation (`Y_GE85`, not `Y85-89`/`Y_GE90`) |
| `age_bins_loader.py` | Loader functions mirroring `locations_loader.py`'s style: `load_config()` reads the YAML, `get_scheme()` flattens a named scheme into an age-code→bin-label mapping, `get_bin_labels()` returns bin labels in definition order (for consistent plot/UI ordering), `validate_scheme()` fails loudly if the data contains an age code the chosen scheme doesn't cover |
| `logging_config.py` | Centralized logging setup used by every script in the pipeline |
| `validate_raw_data.py` | Standalone validation utility for sanity-checking the raw extracts before staging |

---

## Repository Structure

```
Heatwave_analysis/
├── config/
│   ├── locations.yaml
│   └── age_bins.yaml
├── src/
│   ├── run_pipeline.py
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
│   ├── analysis/
│   │   └── epi_metrics.py
│   └── utils/
│       ├── locations_loader.py
│       ├── age_bins_loader.py
│       ├── logging_config.py
│       └── validate_raw_data.py
├── notebooks/
│   └── 01_Descriptive_analysis.ipynb
├── dashboard/
│   ├── app.py
│   ├── views_loader.py
│   └── components.py
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
│           ├── mortality_by_age_weekly_labeled.csv
│           ├── mortality_by_age_weekly_stratified_heat.parquet
│           └── mortality_by_age_weekly_stratified_heat.csv
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Design Principles

A few conventions held throughout the project:

- **One direction of dependency.** `src/` produces data; `data/analytics/views/` is the contract; `notebooks/` and `dashboard/` consume it. The dashboard imports from `src/`, never the reverse.
- **Shared logic in one place.** Every epidemiological formula lives in `epi_metrics.py`, imported by both the notebook and the dashboard, so they can't drift apart.
- **Native grain in storage, reconcile at query time.** Population is stored annually and broadcast to weekly grain only when joined; age bins are regrouped in the view layer, never destructively upstream.
- **Fail loudly, not silently.** Missing regions, uncovered age codes, and zero-population groups raise explicit errors rather than producing quietly wrong numbers.
- **Separate technical fixes from analytical choices.** The `Y85-89`/`Y_GE90` → `Y_GE85` collapse is a source-compatibility fix upstream; the coarse age bins for effect-modification are an analytical choice in the view layer. They are kept distinct.