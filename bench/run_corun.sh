#!/usr/bin/env bash
# Можно ли гонять чистовик (GigaAM, пакетно) на той же машине, где идут живые субтитры (T-one)?
# Три сценария по LEN секунд звука, в каждом две встречи = четыре живых потока через сервис:
#   solo        — только сервис;
#   corun-all   — плюс GigaAM int8 на nice 19, ORT на все ядра;
#   corun-t2    — плюс GigaAM int8 на nice 19, ORT на 2 потока.
# Пакетный прогон крутится по кругу, пока идут встречи; его строки в results/batch-<хост>.csv
# с пометкой сценария дают RTF чистовика под нагрузкой. Отставание фраз печатает client_demo.
#
#   nohup bash bench/run_corun.sh запись > results/raw/corun-запись.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
NAME=${1:?имя записи}
LEN=${LEN:-600}
ROOT="${STT_LAB_ROOT:-/srv/stt-lab}"
V="$ROOT/.venv/bin/python"
B="$ROOT/.venv-onnxasr/bin/python"
export PYTHONUNBUFFERED=1 STT_SHARED_SECRET=lab-corun STT_MAX_STREAMS=4 STT_MODEL_PATH="$PWD/models/T-one-fp32/model.onnx"
TMP=$(mktemp -d)
for r in host client; do
  ffmpeg -hide_banner -loglevel error -y -t "$LEN" -i "samples/$NAME.$r.8k.wav" "$TMP/$r.8k.wav"
  ffmpeg -hide_banner -loglevel error -y -t "$LEN" -i "samples/$NAME.$r.16k.wav" "$TMP/$NAME-cut.$r.16k.wav"
done

relays() { ss -uanp 2>/dev/null | grep turnserver | awk '{print $4}' | awk -F: '$NF>=49152' | wc -l; }
# Пустые аллокации (проверки ретранслятора) держат сокеты часами без медиа; звонок — это сокеты И трафик.
kbps() { local a b; a=$(awk '/:/ && !/lo:/{s+=$2+$10} END{print s}' /proc/net/dev); sleep 10
         b=$(awk '/:/ && !/lo:/{s+=$2+$10} END{print s}' /proc/net/dev); echo $(( (b - a) / 10240 )); }
busy() { [ "$(relays)" -gt 0 ] && [ "$(kbps)" -gt "${BUSY_KBPS:-50}" ]; }
wait_quiet() { while busy; do echo "$(date '+%F %T') идёт звонок (сокетов $(relays)) — жду"; sleep 300; done; }

scenario() {  # имя, потоки ORT для чистовика (пусто — без чистовика)
  wait_quiet
  echo; echo "== $(date '+%F %T') сценарий $1"
  "$ROOT/.venv/bin/uvicorn" service.app:main --factory --host 127.0.0.1 --port 8090 > "results/raw/corun-service-$1.log" 2>&1 &
  local svc=$! bat=""
  for _ in $(seq 1 90); do curl -sf 127.0.0.1:8090/health >/dev/null && break; sleep 1; done
  if [ -n "${2:-}" ]; then
    ( while :; do nice -n 19 "$B" quality/run_onnxasr.py --model gigaam-v3-e2e-rnnt --quant int8 --threads "$2" --tag "$1" \
        "$TMP/$NAME-cut.host.16k.wav" "$TMP/$NAME-cut.client.16k.wav" 2>&1 | grep -E "RTF|загрузка"; done ) &
    bat=$!
    sleep 25  # чистовик успевает загрузить модель и выйти на полную нагрузку
  fi
  "$V" service/client_demo.py --host "$TMP/host.8k.wav" --client "$TMP/client.8k.wav" --quiet | sed 's/^/[встреча 1] /' &
  local m1=$!
  "$V" service/client_demo.py --host "$TMP/host.8k.wav" --client "$TMP/client.8k.wav" --quiet | sed 's/^/[встреча 2] /' &
  local m2=$!
  wait "$m1" "$m2"
  [ -n "$bat" ] && { pkill -P "$bat"; kill "$bat"; pkill -f "run_onnxasr.py.*--tag $1"; } 2>/dev/null
  kill "$svc" 2>/dev/null; wait "$svc" 2>/dev/null
  sleep 10
}

scenario solo
scenario corun-all 0
scenario corun-t2 2
rm -rf "$TMP"
echo; echo "CORUN_DONE $(date -Is)"
