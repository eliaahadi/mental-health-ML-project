
# Global Mental Health Indicators — Portfolio Project

A beginner‑to‑advanced coding project scaffold to explore global mental health indicators, visualize country‑level trends, analyze gender disparities, and build forecasting and clustering models.

## Tracks

**Beginner**
- Load a global mental‑health CSV (from Kaggle or elsewhere) into `data/raw/`.
- Standardize columns via `config/columns.yaml` and run the cleaning pipeline.
- Make a choropleth map and line charts of prevalence by country over time.
- (Optional) Launch the Streamlit app for interactive exploration.

**Intermediate**
- Compute gender gaps and explore socio‑economic correlates (e.g., GDP per capita).
- Join with external indicators (place CSVs in `data/external/`). Examples: DALYs, suicide rate.
- Train a simple regression model to predict prevalence from socio‑economic features.

**Advanced**
- **Prevalence forecasting** with SARIMAX per country/indicator.
- **DALYs correlation**: correlate prevalence with burden (DALYs per 100k).
- **Country clustering**: cluster countries by mental‑health profile.

> ⚠️ This repo does not download data for you. Place your CSVs in `data/raw/` and `data/external/` as noted below.

## Quickstart

```bash
# 1) Create and activate a virtual env (optional)
python -m venv .venv && source .venv/bin/activate

# 2) Install deps
pip install -r requirements.txt

# 3) Configure column mappings (edit config/columns.yaml)
# 4) Place mental-health CSV(s) into data/raw/ and optional external CSV(s) into data/external/

# 5) Run pipeline steps via CLI
python -m mh.cli ingest --input data/raw/your_file.csv --output data/processed/standardized.parquet --config config/columns.yaml
python -m mh.cli clean --input data/processed/standardized.parquet --output data/processed/clean.parquet
python -m mh.cli gender-gap --input data/processed/clean.parquet --output reports/gender_gap.csv
python -m mh.cli visualize --input data/processed/clean.parquet --out_html reports/overview.html
python -m mh.cli forecast --input data/processed/clean.parquet --country "United States" --indicator prevalence_total --steps 5 --out_png reports/forecast_USA.png
python -m mh.cli cluster --input data/processed/clean.parquet --year latest --k 5 --out_csv reports/clusters.csv
```

## Data expectations

Your raw dataset should include at least:
- Country name
- Year
- One or more mental‑health prevalence indicators (e.g., total, depression, anxiety)
- (Optional) Sex (male/female) for gender‑gap analysis

Map your raw column names to canonical names via `config/columns.yaml`. Example:

```yaml
country: Country
year: Year
sex: Sex
prevalence_total: Prevalence_Total
prevalence_male: Prevalence_Male
prevalence_female: Prevalence_Female
prevalence_depression: Depression_Prevalence
prevalence_anxiety: Anxiety_Prevalence
```

## External data (optional but recommended)

Save external indicator CSVs (e.g., DALYs per 100k, suicide rate) in `data/external/`. Then run joins with:

```bash
python -m mh.cli join-external --base data/processed/clean.parquet --external data/external/dalys.csv --key country_iso3 year
```

## Streamlit app

```bash
streamlit run app/streamlit_app.py
```

The app expects `data/processed/clean.parquet` and shows:
- An interactive choropleth map by country/year
- Country trend lines for selected indicators
- (Optional) Gender gap panel if sex‑stratified columns exist

## Project layout

```
mh/                 # Python package
  cli.py            # Typer CLI entry point
  config.py
  data.py           # ingest/clean/join helpers
  plots.py          # choropleth and line charts (Plotly)
  models.py         # forecasting & clustering
  utils.py          # ISO-3 mapping, checks

app/
  streamlit_app.py  # interactive dashboard

config/
  columns.yaml      # Map raw -> canonical column names
  params.yaml       # Hyperparameters, defaults

data/
  raw/              # Place raw datasets here (gitignored)
  processed/        # Outputs (parquet)
  external/         # DALYs, suicide rate, GDP, etc.

reports/
  # HTML/CSV/PNG outputs
```

## Notes

- Country codes are standardized to ISO‑3 using `country_converter`.
- Visualizations use Plotly.
- Forecasting uses `statsmodels` SARIMAX as a simple baseline.
- Clustering uses K‑Means on scaled features; adjust in `params.yaml`.
# mental-health-ML-project
