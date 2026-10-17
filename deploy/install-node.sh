#!/usr/bin/env bash
# Подготовка машины под стенд stt-lab: системные пакеты, venv, T-one и веса.
# Запуск от root. Службы НЕ перезапускаются (needrestart в режиме списка):
# на машине могут работать другие сервисы, и их рестарт не нужен.
#
#   bash deploy/install-node.sh            # пакеты, venv, model.onnx
#   STT_LAB_KENLM=1 bash deploy/install-node.sh   # плюс kenlm.bin (5,4 ГБ) в фоне
set -euo pipefail

ROOT="${STT_LAB_ROOT:-/srv/stt-lab}"
export DEBIAN_FRONTEND=noninteractive NEEDRESTART_MODE=l

echo "== $(date -Is) пакеты"
apt-get update -q
apt-get install -y -q python3-venv python3-dev python3-pip git ffmpeg build-essential cmake

echo "== venv в $ROOT"
mkdir -p "$ROOT"/{samples,results}
[ -d "$ROOT/.venv" ] || python3 -m venv "$ROOT/.venv"
PIP="$ROOT/.venv/bin/pip"
PY="$ROOT/.venv/bin/python"
"$PIP" install -q --upgrade pip wheel
# T-one ставится из GitHub: пакета на PyPI нет. kenlm собирается из исходников,
# поэтому выше build-essential.
"$PIP" install -q "tone @ git+https://github.com/voicekit-team/T-one@main" \
  psutil soundfile jiwer huggingface-hub

echo "== веса T-one (кэш Hugging Face, тот же, что использует пайплайн)"
"$PY" - <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("t-tech/T-one", allow_patterns=["*.json", "model.onnx", "README.md"])
print("weights:", path)
PY

echo "== окружение"
"$PY" - <<'PY'
import platform, onnxruntime, numpy
print("python", platform.python_version(), "| onnxruntime", onnxruntime.__version__, "| numpy", numpy.__version__)
PY
echo "STAGE1_DONE $(date -Is)"

if [ "${STT_LAB_KENLM:-0}" = "1" ]; then
  nohup "$PY" -c "from huggingface_hub import hf_hub_download; print(hf_hub_download('t-tech/T-one', 'kenlm.bin'))" \
    > "$ROOT/kenlm-download.log" 2>&1 &
  echo "KENLM_DOWNLOAD_STARTED (журнал $ROOT/kenlm-download.log)"
fi
