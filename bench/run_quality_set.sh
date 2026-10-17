#!/usr/bin/env bash
# Полная серия по одной записи: все движки на двух каналах. Долго (часы), запускать в фоне:
#   nohup bash bench/run_quality_set.sh <имя> > results/raw/quality-<имя>.log 2>&1 &
# Ждёт samples/<имя>.host.8k.wav и samples/<имя>.client.8k.wav (из bench/prepare_samples.sh).
#
# Порядок: сначала сервис в реальном времени один на машине (честные задержки фраз),
# потом офлайн-прогоны; KenLM и Vosk ru-0.42 не запускаются одновременно — вместе
# они не влезают в память.
set -uo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?имя записи}"
ROOT="${STT_LAB_ROOT:-/srv/stt-lab}"
V="$ROOT/.venv/bin/python"
H="samples/$NAME.host.8k.wav"; C="samples/$NAME.client.8k.wav"
[ -f "$H" ] && [ -f "$C" ] || { echo "нет $H / $C" >&2; exit 1; }
export PYTHONUNBUFFERED=1
stage() { echo; echo "== $(date +%H:%M) $*"; }

stage "A. сервис stt-streaming, greedy, реальное время"
export STT_SHARED_SECRET=lab-quality STT_MAX_STREAMS=4 STT_MODEL_PATH="$PWD/models/T-one-fp32/model.onnx"
[ -f "$STT_MODEL_PATH" ] || "$V" bench/to_fp32.py >/dev/null
"$ROOT/.venv/bin/uvicorn" service.app:main --factory --host 127.0.0.1 --port 8090 > "results/raw/service-$NAME.log" 2>&1 &
SVC=$!
for _ in $(seq 1 60); do curl -sf 127.0.0.1:8090/health >/dev/null && break; sleep 1; done
"$V" service/client_demo.py --host "$H" --client "$C" --save-as "$NAME" --tag tone-service-greedy --quiet
kill "$SVC" 2>/dev/null; wait "$SVC" 2>/dev/null

stage "B1. T-one greedy офлайн, 8 кГц напрямую"
"$V" quality/run_tone.py --decoder greedy "$H" "$C"
stage "B2. T-one KenLM офлайн"
"$V" quality/run_tone.py --decoder kenlm "$H" "$C"

stage "C. GigaAM-v3 e2e_rnnt (пунктуация)"
nice -n 5 "$ROOT/.venv-gigaam/bin/python" quality/run_gigaam.py --model v3_e2e_rnnt "$H" "$C" 2>&1 | grep -vE "Warning|warn"

stage "D. Vosk ru-0.42"
nice -n 5 "$ROOT/.venv-vosk/bin/python" quality/run_vosk.py --model vosk-model-ru-0.42 "$H" "$C" 2>&1 | grep -v LOG

echo; echo "QUALITY_DONE $(date -Is)"
