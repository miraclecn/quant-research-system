# Research Notes

Last updated: 2026-03-21

## Current high-level read

- the default multi-factor `lgbm` run is currently better than the `ridge` baseline
- some quick or narrow-window runs look strong, but drawdown is still large
- reversal-style single factors currently look more promising than the SMA trend family
- factor-chain results are not yet stable enough to treat as a better default than the raw baseline
- the factor research workflow is being unified so that single-factor evaluation becomes the only upstream gate for Ridge and LightGBM

## Workflow note

The repository now moves toward one factor research system:

1. build `factor_registry.csv`
2. run unified single-factor evaluation into `factor_evaluation.csv`
3. gate factors into `factor_whitelist.csv`
4. re-screen whitelist factors with Ridge into `ridge_screen.csv`
5. pass the surviving pool into LightGBM via `lgbm_feature_pool.csv`

## Baseline model snapshot

From `outputs/metrics.json`:

- `ridge`: annual return about `-4.6%`, Sharpe about `-0.17`, max drawdown about `-56.3%`
- `lgbm`: annual return about `18.6%`, Sharpe about `0.63`, max drawdown about `-37.4%`

From `outputs/final-run/metrics.json`:

- `ridge`: annual return about `-10.9%`, Sharpe about `-0.46`
- `lgbm`: annual return about `10.2%`, Sharpe about `0.41`

From `outputs/daily-quick/metrics.json`:

- `ridge`: annual return about `-1.5%`, Sharpe about `-0.05`
- `lgbm`: annual return about `44.0%`, Sharpe about `1.28`

Interpretation:

- `lgbm` is the current practical default
- fast smoke or quick runs may overstate quality, so decisions should rely more on the broader baseline and rolling research outputs

## Factor family read

From `outputs/family-sma/single_factor_annual_report.csv`:

- many top SMA-family signals are sign-flipped, which means the useful effect may be closer to mean reversion than pure trend following
- the examples inspected show modest or mixed monotonicity

From `outputs/family-reversal/single_factor_annual_report.csv`:

- `ret_20`, `ret_reversal_5_20`, `ret_10`, `ret_5`, and `intraday_ret_1` all show usable yearly examples
- several reversal-family factors show stronger top-bucket excess and better long-short Sharpe than the SMA examples inspected

Interpretation:

- reversal is the stronger family to prioritize for the next screening round
- SMA-derived features may still matter, but more as contrarian or interaction inputs than as standalone trend signals

## Factor-chain read

From `outputs/factor-chain-36-v3/split_summary.csv`:

- the inspected split shows selected features concentrated in turnover, volatility, amplitude, and long-horizon positioning features
- the valid period is weak and the test period is outright negative for both `ridge` and `lgbm`
- this suggests the screened feature subset is not yet robust across regime changes

Interpretation:

- factor-chain is useful as a research workflow, not yet a stable production default
- feature screening should be judged on out-of-sample stability, not just train-period IC

## 2024 factor to 2025 execution matrix

From `outputs/single-factor-2024-formal/factor_whitelist.csv`:

- the 2024 whitelist is dominated by long-horizon trend and price-position factors such as `trend_55_120`, `close_to_ma_120`, `channel_pos_120`, and `distance_to_low_120`
- no first-batch fundamental factor entered the 2024 whitelist

From `outputs/2025-horizon-matrix/summary.csv`:

- after correcting the annualization logic for biweekly returns, `20d_biweekly` is still positive but no longer dominates in the extreme way first observed
- `10d_weekly` and `20d_weekly` are both strong for `lgbm`
- the corrected matrix now reads roughly:
  - `10d_weekly`: annual return `88.8%`, Sharpe `2.33`
  - `20d_weekly`: annual return `68.4%`, Sharpe `2.16`
  - `20d_biweekly`: annual return `40.5%`, Sharpe `1.40`

Interpretation:

- the first `20d_biweekly` jump was materially overstated by a metrics bug
- the corrected results do not support the claim that `20d_biweekly` is decisively better than weekly execution
- at the moment `10d_weekly` remains the strongest observed default in this matrix

From `outputs/2025-horizon-matrix/20d_biweekly/period_split_summary.csv`:

- the `20d_biweekly` uplift is not evenly distributed through the year
- `lgbm` in `H1 2025` is only mildly positive, with annualized return about `2.7%` and Sharpe about `0.09`
- `lgbm` in `H2 2025` is much stronger, with annualized return about `85.7%` and Sharpe about `3.30`

Interpretation:

- `20d_biweekly` is not a bad configuration, but its edge is not stable enough to replace the weekly baseline
- any move toward this configuration should include at least one more stability pass, such as split-half or rolling-year validation

## Current working default

If the goal is to keep one baseline flow active today:

1. keep `aqt run` with raw `lgbm` as the default benchmark
2. use family and factor-chain workflows as research branches, not as the main production candidate yet
3. keep tracking drawdown, benchmark excess, and size bias before treating any uplift as real

## What to update next

- add notes here whenever a new experiment becomes the active direction
- record which output directory is considered the current decision baseline
- when a new factor family becomes interesting, summarize it here in 5 to 10 lines instead of relying on memory
