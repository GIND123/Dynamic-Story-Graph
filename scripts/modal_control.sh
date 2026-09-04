#!/usr/bin/env bash
# Operate the GPU jobs on a flaky connection.
#
#   scripts/modal_control.sh status   what is running and what it is costing
#   scripts/modal_control.sh stop     terminate every running app, now
#   scripts/modal_control.sh fetch    pull finished results down from the volume
#
# The design assumption is that the connection drops without warning. Runs are
# therefore started WITHOUT --detach, so losing the client kills the container
# and stops the billing, and every job checkpoints to the dsg-results volume as
# it goes, so the next run resumes instead of restarting.
set -uo pipefail
cd "$(dirname "$0")/.."
MODAL=".venv/bin/modal"

case "${1:-status}" in
  status)
    echo "=== running apps ==="
    $MODAL app list 2>/dev/null | grep -E "ephemeral|running" || echo "  (none running)"
    echo
    echo "=== checkpoints on the dsg-results volume ==="
    $MODAL volume ls dsg-results 2>/dev/null || echo "  (volume unreachable)"
    ;;

  stop)
    echo "stopping every running app..."
    ids=$($MODAL app list 2>/dev/null | grep -Eo 'ap-[A-Za-z0-9]+' | sort -u)
    if [ -z "$ids" ]; then
      echo "  nothing running"
      exit 0
    fi
    for id in $ids; do
      echo "  stopping $id"
      $MODAL app stop "$id" --yes 2>&1 | tail -1
    done
    echo "done -- no GPU should now be billing"
    ;;

  fetch)
    mkdir -p artifacts/volume
    for name in $($MODAL volume ls dsg-results 2>/dev/null | grep -Eo '[A-Za-z0-9._-]+\.json'); do
      echo "  fetching $name"
      $MODAL volume get dsg-results "$name" "artifacts/volume/$name" --force 2>&1 | tail -1
    done
    ls -la artifacts/volume 2>/dev/null | tail -n +2
    ;;

  *)
    echo "usage: $0 {status|stop|fetch}" >&2
    exit 2
    ;;
esac
