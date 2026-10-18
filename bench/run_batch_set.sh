#!/usr/bin/env bash
# Серия пакетных прогонов для второго прохода каскада: GigaAM-v3 и Parakeet v3 через onnx-asr,
# fp32/int8, вход 16 кГц (полная полоса) и 8 кГц (то, что видит T-one).
# Перед каждым прогоном ждёт, пока у coturn на этой машине нет релейных сокетов:
# узел обслуживает звонки, тяжёлый CPU нельзя пускать поверх живой встречи.
#
#   nohup bash bench/run_batch_set.sh запись > results/raw/batch-запись.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?имя записи без .host/.client}
PY=${PY:-/srv/stt-lab/.venv-onnxasr/bin/python}
PARAKEET=${PARAKEET:-models/onnx/parakeet-v3}

relays() { ss -uanp 2>/dev/null | grep turnserver | awk '{print $4}' | awk -F: '$NF>=49152' | wc -l; }
# Пустые аллокации (проверки ретранслятора) держат сокеты часами без медиа; звонок — это сокеты И трафик.
kbps() { local a b; a=$(awk '/:/ && !/lo:/{s+=$2+$10} END{print s}' /proc/net/dev); sleep 10
         b=$(awk '/:/ && !/lo:/{s+=$2+$10} END{print s}' /proc/net/dev); echo $(( (b - a) / 10240 )); }
busy() { [ "$(relays)" -gt 0 ] && [ "$(kbps)" -gt "${BUSY_KBPS:-50}" ]; }
wait_quiet() {
  local n
  while busy; do
    echo "$(date '+%F %T') идёт звонок (сокетов $(relays), трафик выше ${BUSY_KBPS:-50} КБ/с) — жду"; sleep 300
  done
}
run() {  # модель, квантование, частота, (опц.) путь
  wait_quiet
  echo "=== $(date '+%F %T') $1 $2 ${3}k"
  nice -n 19 "$PY" quality/run_onnxasr.py --model "$1" --quant "$2" ${4:+--path "$4"} --tag set \
    "samples/$NAME.host.${3}k.wav" "samples/$NAME.client.${3}k.wav" 2>&1 | grep -v -i warning
}

run gigaam-v3-e2e-rnnt fp32 16
run gigaam-v3-e2e-rnnt int8 16
run gigaam-v3-e2e-rnnt fp32 8
run nemo-parakeet-tdt-0.6b-v3 int8 16 "$PARAKEET"
run nemo-parakeet-tdt-0.6b-v3 fp32 16 "$PARAKEET"
run gigaam-v3-e2e-rnnt int8 8
echo "=== $(date '+%F %T') готово"
