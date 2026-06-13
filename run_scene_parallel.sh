#!/usr/bin/env bash
# Run all Scene benchmark splits in parallel (one Python process per run).
#
# Usage:
#   ./run_scene_parallel.sh           # use cache where available
#   ./run_scene_parallel.sh --refresh # retrain all runs
#
# Each run writes its own cache (scene_run0.pkl … scene_run4.pkl).
# After all finish, results are merged and printed.

set -euo pipefail
cd "$(dirname "$0")"

REFRESH=()
if [[ "${1:-}" == "--refresh" ]]; then
  REFRESH=(--refresh)
fi

N_RUNS=5
LOG_DIR="data/cache/parallel_logs"
mkdir -p "$LOG_DIR"

echo "=== Scene parallel benchmark ($N_RUNS runs) ==="
python run_scene.py "${REFRESH[@]}" --run 0 &
pid0=$!
python run_scene.py "${REFRESH[@]}" --run 1 &
pid1=$!
python run_scene.py "${REFRESH[@]}" --run 2 &
pid2=$!
python run_scene.py "${REFRESH[@]}" --run 3 &
pid3=$!
python run_scene.py "${REFRESH[@]}" --run 4 &
pid4=$!

PIDS=($pid0 $pid1 $pid2 $pid3 $pid4)
FAILED=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then
    echo "Run $i failed (see above)" >&2
    FAILED=1
  fi
done

if [[ "$FAILED" -ne 0 ]]; then
  echo "One or more runs failed. Completed runs are still cached; rerun to resume." >&2
  exit 1
fi

echo ""
python run_scene.py --merge-only
echo "Done."
