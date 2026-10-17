#!/usr/bin/env bash
# Копирует код стенда на машину замеров в /srv/stt-lab/src (без .git, аудио и сырых тиков),
# а результаты прогонов забирает обратно в results/.
#
#   bash deploy/sync-node.sh            # туда
#   bash deploy/sync-node.sh --pull     # результаты обратно
set -euo pipefail
cd "$(dirname "$0")/.."
HOST="${STT_LAB_HOST:-node-b}"  # SSH-алиас машины замеров
REMOTE="${STT_LAB_ROOT:-/srv/stt-lab}"

if [ "${1:-}" = "--pull" ]; then
  rsync -az --exclude 'raw/' "$HOST:$REMOTE/src/results/" results/
  echo "results/ обновлены с $HOST"
  exit 0
fi

rsync -az --delete \
  --exclude '.git' --exclude 'samples/*' --exclude 'results/' --exclude 'models/' --exclude '.venv' --exclude '__pycache__' \
  ./ "$HOST:$REMOTE/src/"
echo "код на $HOST:$REMOTE/src"
