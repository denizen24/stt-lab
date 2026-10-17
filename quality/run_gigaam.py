#!/usr/bin/env python3
"""Пакетная расшифровка через GigaAM-v3 (Сбер, MIT) — верхняя планка качества и чистовой текст.

GigaAM принимает отрезки до 25 с, а длинные записи в оригинале режет через pyannote
(нужен токен Hugging Face). Здесь режем сами: по тишине (энергия ниже порога
дольше 0,3 с), куски не длиннее 25 с. Модель `v3_e2e_rnnt` ставит пунктуацию и
заглавные, `v3_rnnt` — нет. Аудио приводится к 16 кГц через ffmpeg.

Запускать из venv GigaAM: /srv/stt-lab/.venv-gigaam/bin/python quality/run_gigaam.py --model v3_e2e_rnnt samples/*.8k.wav
Пишет results/quality/<имя>.gigaam-<модель>.txt и время на час аудио.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"
SR = 16000


def to_16k(path: Path) -> Path:
    tmp = Path(tempfile.mkdtemp()) / (path.stem + ".16k.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(path), "-ac", "1", "-ar", str(SR), str(tmp)], check=True)
    return tmp


def segments(audio: np.ndarray, max_sec: float = 25.0, win: float = 0.05, min_sil: float = 0.3) -> list[tuple[int, int]]:
    """Границы кусков в сэмплах: режем в длинных тишинах, не длиннее max_sec."""
    n = int(SR * win)
    frames = audio[: len(audio) // n * n].reshape(-1, n)
    rms = np.sqrt((frames.astype(np.float32) ** 2).mean(axis=1))
    thr = max(rms.mean() * 0.15, 1e-4)
    silent = rms < thr
    cuts, start, last_sil = [], 0, None
    run = 0
    for i, s in enumerate(silent):
        run = run + 1 if s else 0
        if run * win >= min_sil:
            last_sil = i - run // 2
        if (i - start) * win >= max_sec:
            cut = last_sil if last_sil and last_sil > start else i
            cuts.append((start * n, cut * n))
            start, last_sil = cut, None
    cuts.append((start * n, len(audio)))
    return [(a, b) for a, b in cuts if b - a > n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", default="v3_e2e_rnnt", help="v3_e2e_rnnt (с пунктуацией), v3_rnnt, v3_ctc, v3_e2e_ctc")
    args = ap.parse_args()
    import gigaam  # noqa: PLC0415 — импорт тяжёлый, пусть падает уже после разбора аргументов

    t = time.perf_counter()
    model = gigaam.load_model(args.model)
    print(f"gigaam {args.model}: загрузка {time.perf_counter() - t:.1f} с · RSS {psutil.Process().memory_info().rss / 2**20:.0f} МБ")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        wav16 = to_16k(f)
        audio, _ = sf.read(str(wav16), dtype="float32")
        pieces = []
        t = time.perf_counter()
        for a, b in segments(audio):
            piece = wav16.with_name(f"seg-{a}.wav")
            sf.write(str(piece), audio[a:b], SR)
            r = model.transcribe(str(piece))  # строка в старых версиях, TranscriptionResult в новых
            text = r if isinstance(r, str) else (getattr(r, "transcription", None) or getattr(r, "text", None) or str(r))
            pieces.append(text.strip())
        dt = time.perf_counter() - t
        sec = len(audio) / SR
        name = f.name.replace(".8k.wav", "").replace(".wav", "")
        (OUT / f"{name}.gigaam-{args.model}.txt").write_text(" ".join(p for p in pieces if p) + "\n", encoding="utf-8")
        print(f"{name}: {sec:.0f} с, {len(pieces)} кусков, за {dt:.1f} с (RTF {dt / sec:.2f}, ≈{dt / sec * 60:.0f} мин на час аудио) · RSS {psutil.Process().memory_info().rss / 2**20:.0f} МБ")
        print("  " + " ".join(pieces)[:300])


if __name__ == "__main__":
    main()
