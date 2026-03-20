#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_DIR="${1:-$ROOT_DIR/outputs}"
DEFAULT_METRICS="${OUTPUT_DIR}/metrics.json"
DAILY_OUTPUT_DIR="${2:-$ROOT_DIR/outputs/daily}"

echo "== git branch =="
git status --short --branch
echo

echo "== recent commits =="
git log --oneline --decorate -n 5
echo

echo "== tracked docs =="
for path in README.md CONTRIBUTING.md CHANGELOG.md docs/project-status.md; do
  if [[ -f "$path" ]]; then
    printf '%s\n' "$path"
  fi
done
echo

echo "== key local assets =="
for path in stock_data.duckdb data/daily_bars.parquet outputs/metrics.json outputs/latest_signals.csv; do
  if [[ -e "$path" ]]; then
    size="$(du -sh "$path" 2>/dev/null | awk '{print $1}')"
    printf '%s\t%s\n' "$path" "$size"
  fi
done
echo

echo "== default metrics =="
if [[ -f "$DEFAULT_METRICS" ]]; then
  ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

path = Path("outputs/metrics.json")
data = json.loads(path.read_text())
for model_name in ("ridge", "lgbm", "lgbm_neutralized"):
    value = data.get(model_name)
    if isinstance(value, dict):
        print(
            f"{model_name}\tannual_return_est={value.get('annual_return_est')}\t"
            f"sharpe_est={value.get('sharpe_est')}\tmax_drawdown={value.get('max_drawdown')}"
        )
PY
else
  echo "outputs/metrics.json missing"
fi
echo

echo "== daily output health =="
if [[ -d "$DAILY_OUTPUT_DIR" ]]; then
  bash scripts/check_daily_outputs.sh "$DAILY_OUTPUT_DIR" || true
else
  echo "${DAILY_OUTPUT_DIR} missing"
fi
