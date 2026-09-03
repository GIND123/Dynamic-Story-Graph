#!/usr/bin/env bash
# CPU-only: replay every cached proposal stream through the policy ladder,
# score all three planes, and build the report. Free to re-run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=".venv/bin/python"

score () {  # score <corpus> <run-dir-name> <out-name>
  local corpus="$1" run="$2" out="$3"
  [ -d "artifacts/proposals/$run" ] || { echo "skip $run (no proposals)"; return 0; }
  echo "=== $out ==="
  $PY -m dsg study --corpus "$corpus" \
      --proposals "artifacts/proposals/$run" --out "artifacts/results/$out"
  $PY -m dsg report --results "artifacts/results/$out" --out "artifacts/report/$out"
}

score pdnc    pdnc-qwen7b-w3200     pdnc-qwen7b
score litbank litbank-qwen7b-w1200  litbank-qwen7b
score pdnc    pdnc-qwen3b-w3200     pdnc-qwen3b
score pdnc    pdnc-qwen1.5b-w3200   pdnc-qwen1.5b
score pdnc    pdnc-qwen14b-w3200    pdnc-qwen14b
echo "analysis complete"
