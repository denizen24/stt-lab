#!/usr/bin/env python3
"""Vosk как baseline: потоковое распознавание на CPU, модели ru-0.42 (1,8 ГБ) и small-ru-0.22 (45 МБ).

Аудио подаётся кусками по 300 мс в 16 кГц, как в стриминге; время считается
по всему файлу. Пишет results/quality/<имя>.vosk-<модель>.txt.

Запускать из venv Vosk: /srv/stt-lab/.venv-vosk/bin/python quality/run_vosk.py --model vosk-model-ru-0.42 samples/*.8k.wav
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"
SR = 16000
CHUNK = int(SR * 0.3)


def to_16k(path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp()) / (path.stem + ".16k.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), "-sample_fmt", "s16", str(tmp)], check=True)
    return tmp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", default="vosk-model-ru-0.42")
    ap.add_argument("--models-dir", type=Path, default=Path("/srv/stt-lab/models"))
    args = ap.parse_args()
    from vosk import KaldiRecognizer, Model, SetLogLevel  # noqa: PLC0415

    SetLogLevel(-1)
    t = time.perf_counter()
    model = Model(str(args.models_dir / args.model))
    print(f"{args.model}: загрузка {time.perf_counter() - t:.1f} с · RSS {psutil.Process().memory_info().rss / 2**20:.0f} МБ")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        data, _ = sf.read(str(to_16k(f)), dtype="int16")
        rec = KaldiRecognizer(model, SR)
        texts, chunk_ms = [], []
        t_all = time.perf_counter()
        for i in range(0, len(data), CHUNK):
            t = time.perf_counter()
            if rec.AcceptWaveform(data[i : i + CHUNK].tobytes()):
                texts.append(json.loads(rec.Result()).get("text", ""))
            chunk_ms.append((time.perf_counter() - t) * 1000)
        texts.append(json.loads(rec.FinalResult()).get("text", ""))
        dt = time.perf_counter() - t_all
        sec = len(data) / SR
        name = f.name.replace(".8k.wav", "").replace(".wav", "")
        short = args.model.replace("vosk-model-", "")
        (OUT / f"{name}.vosk-{short}.txt").write_text(" ".join(x for x in texts if x) + "\n", encoding="utf-8")
        chunk_ms.sort()
        print(f"{name}: {sec:.0f} с за {dt:.1f} с (RTF {dt / sec:.2f}) · чанк 300 мс p50 {chunk_ms[len(chunk_ms) // 2]:.0f} / p95 {chunk_ms[int(len(chunk_ms) * 0.95) - 1]:.0f} мс · RSS {psutil.Process().memory_info().rss / 2**20:.0f} МБ")
        print("  " + " ".join(texts)[:300])


if __name__ == "__main__":
    main()
