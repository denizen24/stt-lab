#!/usr/bin/env bash
# Второй проход (часть 2): onnx-asr в своём venv и веса Parakeet TDT 0.6B v3.
# Этим venv пользуются quality/run_onnxasr.py и серии bench/run_batch_set.sh,
# bench/run_etalon_set.sh, bench/run_corun.sh ($STT_LAB_ROOT/.venv-onnxasr).
#
# GigaAM-v3 отдельно не качается: onnx-asr сам берёт его в кэш Hugging Face при первом
# запуске. Parakeet кладётся в каталог (по умолчанию models/onnx/parakeet-v3 в корне клона,
# как ждут серийные скрипты): onnxruntime ≥ 1.24 не читает внешние веса fp32 (.onnx.data)
# через симлинки кэша. Около 3 ГБ (fp32 и int8).
#
# Повторный запуск безопасен: venv не пересоздаётся, уже скачанные файлы не качаются заново.
#
#   bash deploy/install-onnxasr.sh                  # venv и Parakeet
#   bash deploy/install-onnxasr.sh venv             # только venv
#   PARAKEET=/путь bash deploy/install-onnxasr.sh   # Parakeet в другой каталог (тот же PARAKEET — сериям)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="${STT_LAB_ROOT:-/srv/stt-lab}"
PARAKEET=${PARAKEET:-models/onnx/parakeet-v3}
WHAT="${1:-all}"
VENV="$ROOT/.venv-onnxasr"
OX="$VENV/bin/python"

case "$WHAT" in all|venv) ;; *) echo "неизвестный аргумент: $WHAT (all или venv)" >&2; exit 2 ;; esac

echo "== onnx-asr → $VENV"
mkdir -p "$ROOT"
[ -d "$VENV" ] || python3 -m venv "$VENV"
"$VENV/bin/pip" install -q --upgrade pip wheel
"$VENV/bin/pip" install -q "onnx-asr[cpu,hub]" soundfile psutil
"$OX" -c "import onnx_asr, onnxruntime, soundfile, psutil; print('onnx-asr ok, onnxruntime', onnxruntime.__version__)"

if [ "$WHAT" = all ]; then
  echo "== Parakeet TDT 0.6B v3 → $PARAKEET"
  mkdir -p "$PARAKEET"
  PARAKEET_DIR="$PARAKEET" "$OX" - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download("istupakov/parakeet-tdt-0.6b-v3-onnx", local_dir=os.environ["PARAKEET_DIR"])
print("parakeet:", path)
PY
fi
echo "ONNXASR_DONE $(date -Is)"
