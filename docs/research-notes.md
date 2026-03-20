# Research Notes

Last updated: 2026-03-20

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

## Current working default

If the goal is to keep one baseline flow active today:

1. keep `aqt run` with raw `lgbm` as the default benchmark
2. use family and factor-chain workflows as research branches, not as the main production candidate yet
3. keep tracking drawdown, benchmark excess, and size bias before treating any uplift as real

## What to update next

- add notes here whenever a new experiment becomes the active direction
- record which output directory is considered the current decision baseline
- when a new factor family becomes interesting, summarize it here in 5 to 10 lines instead of relying on memory
