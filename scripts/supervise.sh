#!/usr/bin/env bash
# Run a Modal job across an unreliable connection.
#
#   scripts/supervise.sh <log-file> <done-pattern> <command...>
#
# A dropped connection kills the Modal client, which terminates the container --
# that is deliberate, because it stops the billing. But it also means a long job
# needs someone to start it again. This is that someone: it retries, waiting for
# the network to come back first, and because every stage checkpoints, each
# attempt resumes rather than restarting.
#
# It stops when the log matches <done-pattern>, or after MAX_ATTEMPTS.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG="$1"; DONE_RE="$2"; shift 2
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
NET_WAIT="${NET_WAIT:-900}"     # seconds to wait for the network per attempt
BACKOFF="${BACKOFF:-20}"

net_up () { curl -sI --max-time 8 https://api.modal.com >/dev/null 2>&1; }

wait_for_net () {
  local waited=0
  until net_up; do
    if [ "$waited" -ge "$NET_WAIT" ]; then
      echo "[supervise] network still down after ${waited}s; giving up this attempt"
      return 1
    fi
    echo "[supervise] network down; retrying in ${BACKOFF}s (waited ${waited}s)"
    sleep "$BACKOFF"; waited=$((waited + BACKOFF))
  done
  return 0
}

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  if grep -qE "$DONE_RE" "$LOG" 2>/dev/null; then
    echo "[supervise] already complete"; exit 0
  fi
  wait_for_net || continue
  echo "[supervise] attempt $attempt/$MAX_ATTEMPTS: $*"
  "$@" >> "$LOG" 2>&1
  status=$?
  if grep -qE "$DONE_RE" "$LOG" 2>/dev/null; then
    echo "[supervise] complete on attempt $attempt"; exit 0
  fi
  echo "[supervise] attempt $attempt ended (exit $status); checkpoint keeps the progress"
  sleep "$BACKOFF"
done

echo "[supervise] exhausted $MAX_ATTEMPTS attempts; run scripts/modal_control.sh status" >&2
exit 1
