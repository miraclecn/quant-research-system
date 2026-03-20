# A-Share Daily Research Starter

This repository is a minimal local research system for A-share daily cross-sectional strategies.

Current default research setup:

- Universe: CSI 1000 style pool with liquidity filters
- Frequency: daily bars
- Rebalance: weekly
- Labels: 10-day forward return as primary, 5-day as comparison
- Models: Ridge baseline, LightGBM main model
- Portfolio: equal-weight top N names

## Expected input data

The workflow accepts either:

- a daily panel file at `data/daily_bars.parquet` or `data/daily_bars.csv`
- a local DuckDB file at `stock_data.duckdb`

If no explicit `--input` is passed, the loader will try `data/daily_bars.parquet`, `data/daily_bars.csv`, then `stock_data.duckdb`.

Expected columns:

- `date`: trading date
- `symbol`: stock code
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`
- `turnover_rate` optional
- `float_mv` optional
- `industry` optional
- `is_st` optional
- `listed_days` optional
- `is_paused` optional
- `is_limit_up` optional
- `is_limit_down` optional
- `in_universe` optional, precomputed membership flag for your stock pool

If optional columns are missing, the pipeline will fill reasonable defaults.

## Quick start

1. Install dependencies:

```bash
pip install -e .
```

2. Prepare one of these inputs:

- Put your panel data in `data/daily_bars.parquet`, or
- Keep `stock_data.duckdb` in the project root. The built-in loader will join `daily_kline`, `daily_basic`, `stock_basic`, and when available `index_weight`.

3. Run the full workflow:

```bash
aqt run
```

Optional overrides:

```bash
aqt run --input data/daily_bars.csv --train-days 504 --top-n 20 --rebalance-weekday 4
aqt run --input stock_data.duckdb --start-date 2020-01-01 --end-date 2025-12-31
```

Optional research comparison:

```bash
aqt research-run --input data/daily_bars.parquet --research-start 2020-01-01 --research-end 2026-03-19 --train-months 36 --valid-months 12 --test-months 12 --step-months 12 --neutralize-scores
```

This keeps the default raw-score workflow unchanged and adds an extra `lgbm_neutralized` result using cross-sectional industry and size neutralization.

Unified factor research workflow:

```bash
aqt single-factor-run --input data/daily_bars.parquet --output-dir outputs/factor-eval --research-start 2020-01-01 --research-end 2025-12-31 --single-factor-top-k 20
aqt factor-chain-run --input data/daily_bars.parquet --output-dir outputs/factor-chain --research-start 2020-01-01 --research-end 2025-12-31 --train-months 36 --valid-months 12 --test-months 12 --step-months 12 --factor-top-k 20
```

This workflow is now organized as:

- `factor_registry.csv`: candidate factor catalog with family, expression, and params
- `factor_evaluation.csv`: unified single-factor evaluation table
- `factor_whitelist.csv`: factors passing the single-factor gate
- `ridge_screen.csv`: whitelist factors re-screened by rolling Ridge stability
- `lgbm_feature_pool.csv`: final feature pool handed to LightGBM research

Key evaluation knobs can now be tuned from the CLI, for example:

```bash
aqt single-factor-run \
  --input data/daily_bars.parquet \
  --output-dir outputs/factor-eval \
  --research-start 2020-01-01 \
  --research-end 2025-12-31 \
  --factor-min-abs-rank-ic-ir 0.12 \
  --factor-min-bucket-spearman 0.25 \
  --factor-min-positive-top-bucket-excess-ratio 0.6
```

Recommended workflow when keeping DuckDB as the raw store:

```bash
export TUSHARE_TOKEN=...
aqt update-raw --input stock_data.duckdb --lookback-days 7
aqt update-index-weight --input stock_data.duckdb --index-code 000852.SH --start-date 2020-01-01 --end-date 2026-03-19
aqt export-panel --input stock_data.duckdb --output data/daily_bars.parquet --start-date 2020-01-01 --end-date 2025-12-31
aqt run --input data/daily_bars.parquet
```

One-shot daily workflow:

```bash
export TUSHARE_TOKEN=...
aqt daily-research --input stock_data.duckdb --output data/daily_bars.parquet --output-dir outputs/daily --lookback-days 7 --start-date 2020-01-01 --end-date 2026-03-19
```

Scheduled daily workflow:

```bash
export TUSHARE_TOKEN=...
bash scripts/daily_research.sh
bash scripts/check_daily_outputs.sh
```

Example `crontab -e` entry for weekdays at `16:45`:

```cron
45 16 * * 1-5 cd /home/nan/lightgbm && TUSHARE_TOKEN=your_token_here /bin/bash scripts/daily_research.sh >> /home/nan/lightgbm/outputs/daily/cron.log 2>&1
```

Quick health check after the run:

```bash
bash scripts/check_daily_outputs.sh
```

4. Outputs:

- `outputs/metrics.json`
- `outputs/run_config.json`
- `outputs/backtest_equity.csv` for the default LightGBM strategy
- `outputs/backtest_equity_ridge.csv`
- `outputs/backtest_equity_lgbm.csv`
- `outputs/latest_signals.csv` for the default LightGBM strategy
- `outputs/latest_signals_ridge.csv`
- `outputs/latest_signals_lgbm.csv`
- `outputs/predictions.parquet`

For `research-run`, the top-level output directory also includes:

- `run_config.json` with the exact run settings
- `research_metrics.json` with nested split metrics
- `split_summary.csv` with one flat row per split, phase, and model

For `single-factor-run`, the top-level output directory also includes:

- `factor_registry.csv`
- `factor_evaluation.csv`
- `factor_whitelist.csv`
- `single_factor_annual_report.csv`
- `single_factor_stability.csv`

For `factor-chain-run`, the top-level output directory also includes:

- `factor_registry.csv`
- `factor_evaluation.csv`
- `factor_whitelist.csv`
- `ridge_screen.csv`
- `lgbm_feature_pool.csv`
- split-level `selected_features.csv`, `ridge_coefficients.csv`, and `research_metrics.json`

## Notes

- This starter is intentionally simple and meant for fast iteration.
- It assumes signal generation at day `t` close and trade execution at day `t+1` open.
- The weekly rebalance date is chosen as the target weekday when present, otherwise the last available trading day of that week.
- The backtest includes basic transaction costs and equal-weight portfolio construction.
- When using `stock_data.duckdb`, the loader converts Tushare-style `vol` and `amount` units into shares and RMB.
- The default index membership code is `000852.SH` for CSI 1000. When `index_weight` is available, the loader uses true monthly constituent history; otherwise it falls back to the old `CSI 1000 style` proxy pool.
- `aqt update-raw` uses Tushare to incrementally refresh `daily_kline`, `daily_basic`, and `stock_basic`. Pass `--tushare-token` or set `TUSHARE_TOKEN`.
- `aqt update-index-weight` backfills or refreshes monthly index constituents and weights, for example `000852.SH` for CSI 1000.
- `aqt update-index-daily` backfills or refreshes official index daily bars for benchmark comparison.
- `aqt daily-research` runs `update-raw -> update-index-weight -> update-index-daily -> export-panel -> run` in one command.
- `scripts/daily_research.sh` wraps the same flow for cron-style scheduling and defaults to the current research settings.
- `scripts/check_daily_outputs.sh` verifies that daily artifacts exist, are non-empty, and that the cron log does not obviously contain failures.
- `aqt prune-db` produces a keep/drop recommendation so the DuckDB file can converge toward a raw-only store.
- `aqt prune-db --execute` applies that drop list to the database. Run a filesystem backup first if you want rollback.
- `aqt export-panel` lets you materialize a research-ready `parquet` file and keep `duckdb` as the raw source of truth.
- `aqt single-factor-run` is now the unified factor evaluation entrypoint and should be treated as the only source of truth for factor quality gates.
- `aqt factor-chain-run` now consumes `factor_whitelist.csv` from the unified single-factor evaluation flow instead of using an independent ad hoc factor scoring rule.
- You should tighten the execution model before going live, especially around limit-up, limit-down, suspensions, and order sizing.
