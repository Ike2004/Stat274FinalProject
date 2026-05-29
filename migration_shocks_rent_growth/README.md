# Migration Shocks and Rent Growth in Supply-Constrained Housing Markets

This folder is a reproducible project scaffold for a county-year causal analysis of pandemic-era migration shocks and rent growth.

## Research design status

The current recommended design is:

- Outcome: annual county rent growth from Zillow ZORI, measured as `d_log_rent`.
- Treatment intensity: pandemic net in-migration as a share of prior population.
- Baseline estimand: high-inflow counties after 2020 relative to neutral counties, with county and year fixed effects.
- Main heterogeneity: interaction with pre-pandemic housing supply constraints, preferably zoning restrictiveness.
- Robustness: continuous migration shock, alternative migration bins, ACS rent, alternative supply measures, pre-trend/placebo checks.

The most important caveat is that treatment bins should be defined using a clearly pre-specified pandemic migration window, not using the full panel outcome period in a way that mechanically selects places with rent booms.

## Folder layout

- `src/01_download_data.py`: downloads or stages public datasets where stable links can be discovered.
- `src/02_build_panel.py`: converts raw ZORI and migration data into a county-year panel.
- `src/03_estimate_models.py`: runs baseline DiD, zoning heterogeneity, and robustness specifications.
- `paper/draft.qmd`: paper draft with placeholders for tables/figures.
- `data/raw/`: downloaded source data.
- `data/processed/`: analysis-ready panel.
- `output/tables/`: model tables.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python src/01_download_data.py
python src/02_build_panel.py
python src/03_estimate_models.py
```

Some datasets, especially zoning/WRLURI files, may need manual download depending on source availability and license page behavior. Put a county-level zoning file at `data/raw/zoning/zoning_county.csv` with columns:

```text
fips,zoning_index
```

The scripts will run without zoning but will skip the main zoning heterogeneity table.

## Data source notes

- Zillow ZORI: Zillow Research housing data page.
- IRS SOI migration: IRS county-to-county migration files.
- Census BPS: county annual building permit data.
- ACS: demographic and housing controls, ideally pulled via Census API or downloaded as ACS 5-year county files.
- WRLURI/Wharton: preferred zoning restrictiveness source, usually requires using the published survey file and aggregating local jurisdictions to counties.

