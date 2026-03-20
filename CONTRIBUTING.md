# Collaboration Guide

## Branches

- `main`: stable baseline that should stay runnable
- feature branches: use `feat/<topic>`, `fix/<topic>`, or `research/<topic>`

## Recommended flow

1. Start from the latest `main`
2. Create a focused branch for one task
3. Commit in small steps with clear messages
4. Merge back to `main` only after a local sanity check

Example:

```bash
git checkout main
git pull
git checkout -b feat/add-neutralized-report
```

## Commit messages

Prefer short imperative messages:

- `add factor-chain split summary export`
- `fix index benchmark merge on empty dates`
- `update daily research script defaults`

## What should be versioned

Track:

- source code under `src/`
- scripts under `scripts/`
- project docs and config files

Do not track:

- local virtual environments
- `node_modules/`
- large raw databases
- generated research outputs under `outputs/`
- local panels under `data/`

## Useful commands

```bash
make help
git status
git log --oneline --decorate -n 20
git diff
git checkout -b feat/<topic>
git add .
git commit -m "describe change"
git push -u origin <branch>
```
