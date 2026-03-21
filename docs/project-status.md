# Project Status

Last updated: 2026-03-21

## Scope

This repository is a local A-share daily cross-sectional research system centered on:

- raw market data refresh into DuckDB
- panel export for research-ready data
- feature generation and label construction
- Ridge and LightGBM model training
- portfolio construction and backtest diagnostics
- factor research workflows such as rolling research, factor-chain screening, and family labs

## Current baseline

- default CLI entrypoint: `aqt`
- source package: `src/aqt/`
- raw store: `stock_data.duckdb`
- default panel path: `data/daily_bars.parquet`
- default output root: `outputs/`
- unified factor research chain: `factor_registry -> factor_evaluation -> factor_whitelist -> split-level Ridge gate -> LightGBM`
- split outputs now distinguish:
  - `selected_features.csv`: pre-Ridge candidate set
  - `lgbm_selected_features.csv`: Ridge-filtered feature set actually used by LightGBM
- current execution comparison baseline: `outputs/2025-horizon-matrix`

## Main workflows

- `aqt run`: default rolling training and backtest flow
- `aqt research-run`: rolling train/valid/test evaluation
- `aqt single-factor-run`: unified factor evaluation and whitelist generation
- `aqt factor-chain-run`: whitelist-driven Ridge and LightGBM feature chain
- `aqt family-lab`: grouped factor family experiments
- `aqt update-raw`: Tushare incremental refresh into DuckDB
- `aqt daily-research`: one-shot daily refresh plus research run

## Data snapshot

DuckDB tables observed on 2026-03-21:

- `daily_kline`: 15,727,978 rows
- `daily_basic`: 11,534,329 rows
- `stock_basic`: 5,810 rows
- `index_weight`: 74,002 rows
- `index_daily`: 1,503 rows
- `fina_indicator_raw_v2`: 160,405 rows
- `fina_indicator_clean`: 87,613 rows
- `margin_detail`: 1,191,525 rows
- `top_inst`: 232,189 rows
- `stock_zt_pool`: 112 rows
- `financial_news`: 15 rows

## Latest tracked git baseline

- branch: `main`
- initial import commit: `21df251`
- collaboration baseline commit: `9da8080`

## Current research direction

- 2024 single-factor screening currently favors long-horizon trend and price-position signals over the first batch of fundamental factors
- the corrected 2025 execution matrix does not support replacing the weekly baseline with `20d_biweekly`
- `10d_weekly` is currently the strongest observed default in the corrected matrix, while `20d_biweekly` remains more regime-sensitive and materially stronger in `H2 2025` than in `H1 2025`

## Working conventions

- keep `main` in a runnable state
- develop new work in `feat/*`, `fix/*`, or `research/*`
- do not version local data, databases, or generated outputs
- use `make help` for common commands
- treat `factor_evaluation.csv` as the only source of truth for single-factor quality decisions

## Next recommended maintenance

- keep this file updated when workflow assumptions or major milestones change
- add a short note here when a new research direction becomes the active focus
- consider recording a weekly summary if the project starts moving faster
