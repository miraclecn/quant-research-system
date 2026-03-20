#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_PATH="${1:-$ROOT_DIR/docs/generated-summary.md}"
DATE_TAG="$(date '+%Y-%m-%d %H:%M:%S %z')"

mkdir -p "$(dirname "$OUT_PATH")"

{
  echo "# Generated Summary"
  echo
  echo "Generated at: ${DATE_TAG}"
  echo
  echo "## Git"
  echo
  git status --short --branch
  echo
  git log --oneline --decorate -n 5
  echo
  echo "## Core Assets"
  echo
  for path in stock_data.duckdb data/daily_bars.parquet outputs/metrics.json outputs/latest_signals.csv; do
    if [[ -e "$path" ]]; then
      size="$(du -sh "$path" 2>/dev/null | awk '{print $1}')"
      printf -- "- %s (%s)\n" "$path" "$size"
    fi
  done
  echo
  echo "## Metrics"
  echo
  if [[ -f outputs/metrics.json ]]; then
    ./.venv/bin/python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("outputs/metrics.json").read_text())
for model_name in ("ridge", "lgbm", "lgbm_neutralized"):
    value = data.get(model_name)
    if isinstance(value, dict):
        print(
            f"- {model_name}: annual_return_est={value.get('annual_return_est')}, "
            f"sharpe_est={value.get('sharpe_est')}, max_drawdown={value.get('max_drawdown')}"
        )
PY
  else
    echo "- outputs/metrics.json missing"
  fi
  echo
  echo "## Daily Output Health"
  echo
  if [[ -d outputs/daily ]]; then
    bash scripts/check_daily_outputs.sh outputs/daily || true
  else
    echo "- outputs/daily missing"
  fi
} > "$OUT_PATH"

printf 'wrote %s\n' "$OUT_PATH"
