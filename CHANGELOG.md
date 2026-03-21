# Changelog

## 2026-03-20

### Added

- initialized local git repository and pushed `main` to GitHub
- added repository ignore rules for local environments, data, outputs, and databases
- added [`CONTRIBUTING.md`](CONTRIBUTING.md) with branch and commit conventions
- added [`Makefile`](Makefile) with common workflow entrypoints
- added [`docs/project-status.md`](docs/project-status.md) to summarize current scope and baseline
- added [`scripts/status_snapshot.sh`](scripts/status_snapshot.sh) to print git and artifact status in one command
- added [`docs/research-notes.md`](docs/research-notes.md) to capture current research readouts
- added [`scripts/generate_summary.sh`](scripts/generate_summary.sh) to generate a shareable markdown status summary

### Notes

- current remote repository: `git@github.com:miraclecn/quant-research-system.git`
- current collaboration baseline commit sequence:
  - `21df251` initial project import
  - `9da8080` add git collaboration guide
  - `c7c3a25` add workflow entrypoints and status doc

## Update rule

When a change materially affects workflow, data assumptions, research direction, or repository structure, add a short entry here.

## 2026-03-21

### Added

- added `scripts/run_2025_horizon_matrix.py` to compare fixed-factor execution setups across `10d_weekly`, `20d_weekly`, and `20d_biweekly`
- added `outputs/2025-horizon-matrix/20d_biweekly/period_split_summary.csv` and `.json` for quick split-half stability checks

### Changed

- added LightGBM device configuration plumbing in `src/aqt/config.py`, `src/aqt/cli.py`, and `src/aqt/models.py`
- corrected the matrix workflow so `20d_biweekly` reuses `20d` predictions but filters to actual biweekly rebalance dates instead of accidentally reusing weekly execution

### Notes

- the current Python environment still cannot run LightGBM GPU training: a real fit attempt fails with `No OpenCL device found`
- an annualization bug in `src/aqt/backtest.py` was later identified: biweekly portfolios had been annualized as if they were weekly
- after correcting that bug, `20d_biweekly` remains usable but no longer dominates the weekly baselines
