#!/usr/bin/env bash
# Движки для сравнения качества: GigaAM-v3 и Vosk. Каждый в своём venv:
# GigaAM требует onnxruntime 1.23, а стенд T-one работает на 1.31.
#
#   bash deploy/install-extras.sh            # оба
#   bash deploy/install-extras.sh gigaam     # только один
set -euo pipefail
ROOT="${STT_LAB_ROOT:-/srv/stt-lab}"
WHAT="${1:-all}"

if [ "$WHAT" = all ] || [ "$WHAT" = gigaam ]; then
  echo "== GigaAM-v3 → $ROOT/.venv-gigaam"
  [ -d "$ROOT/.venv-gigaam" ] || python3 -m venv "$ROOT/.venv-gigaam"
  P="$ROOT/.venv-gigaam/bin/pip"
  "$P" install -q --upgrade pip wheel
  # torch только CPU — иначе pip тянет CUDA на гигабайты
  "$P" install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  "$P" install -q "gigaam @ git+https://github.com/salute-developers/GigaAM" soundfile psutil
  "$ROOT/.venv-gigaam/bin/python" -c "import gigaam, torch; print('gigaam ok, torch', torch.__version__, 'threads', torch.get_num_threads())"
fi

if [ "$WHAT" = all ] || [ "$WHAT" = vosk ]; then
  echo "== Vosk → $ROOT/.venv-vosk"
  [ -d "$ROOT/.venv-vosk" ] || python3 -m venv "$ROOT/.venv-vosk"
  P="$ROOT/.venv-vosk/bin/pip"
  "$P" install -q --upgrade pip wheel
  "$P" install -q vosk soundfile psutil
  mkdir -p "$ROOT/models"
  for m in vosk-model-small-ru-0.22 vosk-model-ru-0.42; do
    if [ ! -d "$ROOT/models/$m" ]; then
      echo "  скачиваю $m"
      curl -sSL -o "/tmp/$m.zip" "https://alphacephei.com/vosk/models/$m.zip"
      # unzip на машине может не быть — распаковываем питоном
      python3 -m zipfile -e "/tmp/$m.zip" "$ROOT/models/" && rm -f "/tmp/$m.zip"
      [ -d "$ROOT/models/$m/conf" ] || { echo "модель $m не распаковалась" >&2; exit 1; }
    fi
  done
  "$ROOT/.venv-vosk/bin/python" -c "import vosk; print('vosk ok')"
fi
echo "EXTRAS_DONE $(date -Is)"
