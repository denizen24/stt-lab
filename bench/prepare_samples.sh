#!/usr/bin/env bash
# Приводит аудио к формату модели: моно, 16 бит, 8 кГц, wav → samples/<имя>.8k.wav
# RATE=16000 — то же на 16 кГц (samples/<имя>.16k.wav): полная полоса для пакетных моделей.
# Без аргументов берёт образцы из самого пакета T-one (они уже в составе установки).
#
#   bash bench/prepare_samples.sh                 # образцы T-one
#   bash bench/prepare_samples.sh запись.webm …   # свои файлы (стерео разложится на каналы)
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -z "${PY:-}" ]; then
  for p in .venv/bin/python /srv/stt-lab/.venv/bin/python python3; do
    command -v "$p" >/dev/null 2>&1 && PY=$p && break
  done
fi
mkdir -p samples
RATE=${RATE:-8000}
SUF=$((RATE / 1000))k

convert() {  # вход, имя выхода, (опц.) карта каналов
  local src=$1 name=$2 map=${3:-}
  ffmpeg -hide_banner -loglevel error -y -i "$src" ${map:+-map_channel $map} -ac 1 -ar "$RATE" -sample_fmt s16 "samples/$name.$SUF.wav"
  echo "samples/$name.$SUF.wav  $(ffprobe -v error -show_entries format=duration -of csv=p=0 "samples/$name.$SUF.wav" | cut -c1-6) с"
}

if [ $# -eq 0 ]; then
  dir=$("$PY" -c "import tone, pathlib; print(pathlib.Path(tone.__file__).parent / 'demo' / 'audio_examples')")
  convert "$dir/audio_short.flac" tone-short
  convert "$dir/audio_long.flac" tone-long
  exit 0
fi

for f in "$@"; do
  base=$(basename "${f%.*}")
  ch=$(ffprobe -v error -select_streams a:0 -show_entries stream=channels -of csv=p=0 "$f")
  if [ "$ch" -ge 2 ]; then
    # Две дорожки: ведущий в левом канале, клиент в правом.
    convert "$f" "$base.host" 0.0.0 && convert "$f" "$base.client" 0.0.1
  else
    convert "$f" "$base"
  fi
done
