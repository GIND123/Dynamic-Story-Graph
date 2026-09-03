#!/usr/bin/env bash
# Every GPU run in the study. Each is independent and resumable: proposals are
# written per (corpus, model) under artifacts/proposals/, and nothing downstream
# needs the GPU again.
#
# Rough cost at current Modal rates: ~$3-4 total for all five runs.
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL=".venv/bin/modal"
LOG_DIR="artifacts/logs"; mkdir -p "$LOG_DIR"

run () {  # run <corpus> <model> [extra args...]
  local corpus="$1" model="$2"; shift 2
  local tag="${corpus}-${model}"
  echo "=== $tag ==="
  $MODAL run dsg/infra/modal_extract.py \
      --corpus "$corpus" --model "$model" "$@" \
      > "$LOG_DIR/extract-$tag.log" 2>&1
  grep -E '^\[dsg\] done|^\[local\] wrote|^\[local\] \{' "$LOG_DIR/extract-$tag.log" || true
}

# Main study: the full 28-novel corpus at the primary model size.
run pdnc qwen7b

# Short-context control on LitBank's ~2K-token excerpts.
run litbank qwen7b

# Model-scale ablation: does the representation matter more as the model shrinks?
run pdnc qwen3b
run pdnc qwen1.5b
run pdnc qwen14b

echo "all extractions complete"
