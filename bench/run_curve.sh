#!/usr/bin/env bash
# Полная кривая «потоки → время тика» для отчёта. Идёт около получаса, поэтому в фоне
# и с пониженным приоритетом: на той же машине может жить что-то поважнее замера.
#
#   nohup nice -n 10 bash bench/run_curve.sh > results/raw/curve.log 2>&1 &
#
# Переменные: AUDIO (моно 8 кГц), STREAMS (список точек), POINT_SEC (секунд на точку).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-/srv/stt-lab/.venv/bin/python}"
AUDIO="${AUDIO:-samples/tone-long.8k.wav}"
STREAMS="${STREAMS:-1 2 3 4 5 6 8 10}"
POINT_SEC="${POINT_SEC:-60}"
export PYTHONUNBUFFERED=1
mkdir -p results/raw

run() { echo; echo "== $(date +%H:%M) $1"; shift; "$PY" bench/bench.py --audio "$AUDIO" --seconds "$POINT_SEC" "$@"; }

# shellcheck disable=SC2086 — список точек намеренно разворачивается в аргументы
run "batch, 3 потока ORT"   --streams $STREAMS --mode batch   --threads 3 --tag curve-batch-ort3
run "threads, 6 рабочих"    --streams $STREAMS --mode threads --workers 6 --tag curve-threads-w6
run "threads, 3 рабочих"    --streams $STREAMS --mode threads --workers 3 --tag curve-threads-w3
run "kenlm, threads, 6 рабочих" --streams 2 4 6 --mode threads --workers 6 --decoder kenlm --tag curve-kenlm-w6
echo; echo "CURVE_DONE $(date -Is)"
