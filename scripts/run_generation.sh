#!/usr/bin/env bash
# The generation experiment: can a story get longer without breaking its graph?
#
# --detach is mandatory. Without it Modal cancels the remote function the moment
# the local client disconnects, and an hour of GPU time is billed for nothing.
# Results are also written to the dsg-results volume, so a run whose client died
# can be recovered:
#     modal run dsg/infra/modal_generate.py::fetch --run-id <run-id>
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL=".venv/bin/modal"; PY=".venv/bin/python"
LOG_DIR="artifacts/logs"; mkdir -p "$LOG_DIR"

STORIES="${STORIES:-30}"
CHAPTERS="${CHAPTERS:-20}"
MODEL="${MODEL:-qwen7b}"
RUN_ID="gen-${MODEL}-s${STORIES}-c${CHAPTERS}"

echo "=== $RUN_ID ==="
$MODAL run --detach dsg/infra/modal_generate.py::main \
    --model "$MODEL" --stories "$STORIES" --chapters "$CHAPTERS" --run-id "$RUN_ID" \
    > "$LOG_DIR/$RUN_ID.log" 2>&1 || {
      echo "client died; attempting recovery from the volume"
      $MODAL run dsg/infra/modal_generate.py::fetch --run-id "$RUN_ID"
    }
grep -E '^\[gen\] done|^\[local\] wrote' "$LOG_DIR/$RUN_ID.log" || true

$PY -m dsg gen-report \
    --generation "artifacts/generation/$RUN_ID/generation.json" \
    --out "artifacts/report/$RUN_ID"
echo "generation analysis complete"
