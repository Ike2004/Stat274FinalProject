# Migration Shocks and Rent Growth in Supply-Constrained Housing Markets

This folder is a reproducible project scaffold for a county-year analysis of early-pandemic migration-demand shocks and subsequent rent growth.

## Research design status

The current main design is:

- Outcome: annual county rent growth from Zillow ZORI, measured as `d_log_rent`.
- Treatment: early-pandemic abnormal migration shock, defined as `NetMigRate_2021 - 0.5 * (NetMigRate_2019 + NetMigRate_2020)`.
- Main estimand: whether counties in higher shock quintiles experienced faster subsequent rent growth in 2022-2023 relative to Q3 middle-shock counties.
- Main model: lagged five-bin DiD with county and year fixed effects.
- Central diagnostic: event-study for Q5 shock counties relative to Q3 counties, omitting 2019.
- Supply heterogeneity: interaction between Q5 shock, post period, and low 2020 residential permits per capita.
- Interpretation: reduced-form migration-demand shock, not the pure causal effect of population growth.

The most important caveat is that IRS migration shocks may reflect population inflow, migrant composition, income, remote-work status, and housing preferences. The paper therefore avoids interpreting the estimates as the pure effect of population counts.

## Folder layout

- `src/01_download_data.py`: downloads or stages public datasets where stable links can be discovered.
- `src/02_build_panel.py`: converts raw ZORI and migration data into a county-year panel.
- `src/03_estimate_models.py`: constructs the early-pandemic shock, runs lagged five-bin DiD, event-study, supply heterogeneity, and robustness specifications.
- `src/04_make_figures.py`: generates the revised shock distribution, raw trend, event-study, and binscatter figures.
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

The current supply proxy is 2020 residential permits per capita from the Census Building Permits Survey. Direct zoning data are not used in the main specification.

## Data source notes

- Zillow ZORI: Zillow Research housing data page.
- IRS SOI migration: IRS county-to-county migration files.
- Census BPS: county annual building permit data.
- ACS: demographic and housing controls, ideally pulled via Census API or downloaded as ACS 5-year county files.
- WRLURI/Wharton: preferred zoning restrictiveness source, usually requires using the published survey file and aggregating local jurisdictions to counties.
