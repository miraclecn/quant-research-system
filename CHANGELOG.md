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
