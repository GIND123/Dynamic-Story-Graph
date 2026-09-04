#!/usr/bin/env bash
# The whole system, end to end. Safe to re-run: every stage resumes from its
# checkpoint on the dsg-results volume, and every stage mirrors to the Hub.
#
#   scripts/run_pipeline.sh            # all stages
#   STAGE=train scripts/run_pipeline.sh  # one stage
#
# Nothing is detached. If the connection drops the GPU stops billing, and the
# next run picks up where this one left off.
set -uo pipefail
cd "$(dirname "$0")/.."
MODAL=".venv/bin/modal"; PY=".venv/bin/python"
LOG="artifacts/logs"; mkdir -p "$LOG"
set -a; [ -f .env ] && . ./.env; set +a

BOOKS="${BOOKS:-150}"
CHAPTERS_TRAIN="${CHAPTERS_TRAIN:-24}"
STORIES="${STORIES:-30}"
CHAPTERS_GEN="${CHAPTERS_GEN:-20}"
MODEL="${MODEL:-qwen7b}"
TRAIN_MODEL="${TRAIN_MODEL:-qwen3b}"
DATA_RUN="${DATA_RUN:-dsg-traindata-v1}"
TRAIN_RUN="${TRAIN_RUN:-dsg-writer-${TRAIN_MODEL}}"
GEN_RUN="${GEN_RUN:-gen-${MODEL}-s${STORIES}-c${CHAPTERS_GEN}}"
LORA_REPO="${LORA_REPO:-GOVINDFROM/dsg-writer-${TRAIN_MODEL}}"
STAGE="${STAGE:-all}"

stage () { [ "$STAGE" = "all" ] || [ "$STAGE" = "$1" ]; }

if stage data; then
  echo "=== 1/4 dataset ($BOOKS novels x $CHAPTERS_TRAIN chapters) ==="
  $MODAL run dsg/infra/modal_traindata.py::main \
      --books "$BOOKS" --shards 10 --chapters "$CHAPTERS_TRAIN" \
      --model "$MODEL" --run-id "$DATA_RUN" --checkpoint-every 2 \
      2>&1 | tee "$LOG/$DATA_RUN.log" | grep -E '^\[data\]|^\[local\]|Error'
fi

if stage train; then
  echo "=== 2/4 train LoRA ($TRAIN_MODEL) ==="
  $MODAL run dsg/infra/modal_train.py::main \
      --model "$TRAIN_MODEL" --epochs 2 --run-id "$TRAIN_RUN" --save-every 50 \
      2>&1 | tee "$LOG/$TRAIN_RUN.log" | grep -E '^\[train\]|^\[local\]|Error'
fi

if stage generate; then
  echo "=== 3/4 generate ($STORIES stories x $CHAPTERS_GEN chapters) ==="
  $MODAL run dsg/infra/modal_generate.py::main \
      --model "$MODEL" --stories "$STORIES" --chapters "$CHAPTERS_GEN" \
      --run-id "$GEN_RUN" --lora-repo "$LORA_REPO" \
      2>&1 | tee "$LOG/$GEN_RUN.log" | grep -E '^\[gen\]|^\[local\]|Error'
fi

if stage report; then
  echo "=== 4/4 report + backup ==="
  GEN_JSON="artifacts/generation/$GEN_RUN/generation.json"
  if [ ! -f "$GEN_JSON" ]; then
    echo "  no local generation.json; recovering from the volume"
    $MODAL run dsg/infra/modal_generate.py::fetch --run-id "$GEN_RUN" || true
  fi
  [ -f "$GEN_JSON" ] && $PY -m dsg gen-report --generation "$GEN_JSON" \
      --out "artifacts/report/$GEN_RUN"
  $PY -m dsg push --note "pipeline $GEN_RUN"
fi

echo "pipeline stage '$STAGE' complete"
