#!/usr/bin/env bash
# Серия по набору записей с эталоном: черновик T-one (greedy и KenLM, 8 кГц) и чистовик
# GigaAM-v3 e2e (fp32, 16 кГц) плюс Parakeet v3 (int8, 16 кГц) для каскада латиницей.
# Вход — samples/<имя>.<дорожка>.{8k,16k}.wav; по умолчанию дорожка mix (обе стороны вместе).
# Как и run_batch_set.sh, перед каждым прогоном ждёт, пока на узле нет живого звонка.
#
#   nohup bash bench/run_etalon_set.sh call-a call-b … > results/raw/etalon-set.log 2>&1 &
#   TRACKS="host client" bash bench/run_etalon_set.sh call-a     # по дорожкам
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=${STT_LAB_ROOT:-/srv/stt-lab}
V="$ROOT/.venv/bin/python"
OX="$ROOT/.venv-onnxasr/bin/python"
PARAKEET=${PARAKEET:-models/onnx/parakeet-v3}
TRACKS=${TRACKS:-mix}
export PYTHONUNBUFFERED=1

relays() { ss -uanp 2>/dev/null | grep turnserver | awk '{print $4}' | awk -F: '$NF>=49152' | wc -l; }
kbps() { local a b; a=$(awk '/:/ && !/lo:/{s+=$2+$10} END{print s}' /proc/net/dev); sleep 10
         b=$(awk '/:/ && !/lo:/{s+=$2+$10} END{print s}' /proc/net/dev); echo $(( (b - a) / 10240 )); }
busy() { [ "$(relays)" -gt 0 ] && [ "$(kbps)" -gt "${BUSY_KBPS:-50}" ]; }
wait_quiet() {
  while busy; do echo "$(date '+%F %T') идёт звонок — жду"; sleep 300; done
}
stage() { wait_quiet; echo; echo "=== $(date '+%F %T') $*"; }

for name in "$@"; do
  for t in $TRACKS; do
    f8="samples/$name.$t.8k.wav"; f16="samples/$name.$t.16k.wav"
    [ -f "$f8" ] && [ -f "$f16" ] || { echo "нет $f8 / $f16" >&2; continue; }
    stage "$name.$t T-one greedy"
    nice -n 19 "$V" quality/run_tone.py --decoder greedy "$f8"
    stage "$name.$t T-one KenLM"
    nice -n 19 "$V" quality/run_tone.py --decoder kenlm "$f8"
    stage "$name.$t GigaAM-v3 e2e fp32 16k"
    nice -n 19 "$OX" quality/run_onnxasr.py --model gigaam-v3-e2e-rnnt --quant fp32 --tag etalon "$f16" 2>&1 | grep -vi warning
    stage "$name.$t Parakeet v3 int8 16k"
    nice -n 19 "$OX" quality/run_onnxasr.py --model nemo-parakeet-tdt-0.6b-v3 --quant int8 --path "$PARAKEET" --tag etalon "$f16" 2>&1 | grep -vi warning
  done
done
echo; echo "ETALON_DONE $(date -Is)"
