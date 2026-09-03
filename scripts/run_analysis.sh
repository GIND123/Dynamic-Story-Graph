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

# The in-flight runs of 2026-09-03 were named before a fix to run-id ordering
# and landed in "-w0" directories. Normalise them to the setting actually used
# so paths match what the extractor writes from now on.
for pair in "pdnc-qwen7b-w0:pdnc-qwen7b-w3200" "litbank-qwen7b-w0:litbank-qwen7b-w1200" \
            "pdnc-qwen3b-w0:pdnc-qwen3b-w3200" "pdnc-qwen1.5b-w0:pdnc-qwen1.5b-w3200" \
            "pdnc-qwen14b-w0:pdnc-qwen14b-w3200"; do
  src="artifacts/proposals/${pair%%:*}"; dst="artifacts/proposals/${pair##*:}"
  [ -d "$src" ] && [ ! -d "$dst" ] && mv "$src" "$dst" && echo "renamed $src -> $dst"
done

score pdnc    pdnc-qwen7b-w3200     pdnc-qwen7b
score litbank litbank-qwen7b-w1200  litbank-qwen7b
score pdnc    pdnc-qwen3b-w3200     pdnc-qwen3b
score pdnc    pdnc-qwen1.5b-w3200   pdnc-qwen1.5b
score pdnc    pdnc-qwen14b-w3200    pdnc-qwen14b
echo "analysis complete"

