#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/outputs/daily}"
LOG_FILE="${2:-$OUTPUT_DIR/cron.log}"

METRICS_FILE="$OUTPUT_DIR/metrics.json"
SIGNALS_FILE="$OUTPUT_DIR/latest_signals.csv"
PREDICTIONS_FILE="$OUTPUT_DIR/predictions.parquet"

status="ok"
reason=""

if [[ ! -f "$METRICS_FILE" ]]; then
  status="error"
  reason="metrics.json missing"
elif [[ ! -f "$SIGNALS_FILE" ]]; then
  status="error"
  reason="latest_signals.csv missing"
elif [[ ! -f "$PREDICTIONS_FILE" ]]; then
  status="error"
  reason="predictions.parquet missing"
elif [[ ! -s "$PREDICTIONS_FILE" ]]; then
  status="error"
  reason="predictions.parquet empty"
fi

if [[ "$status" == "ok" && -f "$LOG_FILE" ]] && grep -qiE "traceback|error|exception" "$LOG_FILE"; then
  status="warn"
  reason="cron.log contains error keywords"
fi

if [[ "$status" != "ok" && "$status" != "warn" ]]; then
  echo "status=$status"
  echo "reason=$reason"
  exit 1
fi

metrics_mtime="$(date -r "$METRICS_FILE" '+%Y-%m-%d %H:%M:%S' 2>/dev/null || true)"
signals_rows="$(tail -n +2 "$SIGNALS_FILE" | wc -l | tr -d ' ')"
predictions_size_mb="$(du -m "$PREDICTIONS_FILE" | awk '{print $1}')"

echo "status=$status"
if [[ -n "$reason" ]]; then
  echo "reason=$reason"
fi
echo "metrics_updated=$metrics_mtime"
echo "latest_signals_rows=$signals_rows"
echo "predictions_size_mb=$predictions_size_mb"
