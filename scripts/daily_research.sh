#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -z "${TUSHARE_TOKEN:-}" ]]; then
  echo "TUSHARE_TOKEN is not set" >&2
  exit 1
fi

OUTPUT_PANEL="${OUTPUT_PANEL:-data/daily_bars.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/daily}"
LOOKBACK_DAYS="${LOOKBACK_DAYS:-7}"
TRAIN_DAYS="${TRAIN_DAYS:-504}"
TOP_N="${TOP_N:-12}"
REBALANCE_WEEKDAY="${REBALANCE_WEEKDAY:-4}"

mkdir -p "$OUTPUT_DIR"

"$ROOT_DIR/.venv/bin/aqt" daily-research \
  --input "$ROOT_DIR/stock_data.duckdb" \
  --output "$ROOT_DIR/$OUTPUT_PANEL" \
  --output-dir "$ROOT_DIR/$OUTPUT_DIR" \
  --lookback-days "$LOOKBACK_DAYS" \
  --train-days "$TRAIN_DAYS" \
  --top-n "$TOP_N" \
  --rebalance-weekday "$REBALANCE_WEEKDAY"
