#!/usr/bin/env bash
# The generation experiment: can a story get longer without breaking its graph?
#
# Deliberately NOT detached. On an unreliable connection, dying with the client
# is what we want: the GPU stops billing. Every chapter is checkpointed to the
# dsg-results volume, so re-running resumes rather than restarting, and a
# finished run can always be pulled down with:
#     modal run dsg/infra/modal_generate.py::fetch --run-id <run-id>
#     scripts/modal_control.sh status | stop | fetch
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL=".venv/bin/modal"; PY=".venv/bin/python"
LOG_DIR="artifacts/logs"; mkdir -p "$LOG_DIR"

STORIES="${STORIES:-30}"
CHAPTERS="${CHAPTERS:-20}"
MODEL="${MODEL:-qwen7b}"
RUN_ID="gen-${MODEL}-s${STORIES}-c${CHAPTERS}"

echo "=== $RUN_ID ==="
$MODAL run dsg/infra/modal_generate.py::main \
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
