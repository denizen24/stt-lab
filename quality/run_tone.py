#!/usr/bin/env python3
"""Расшифровка файлов T-one офлайн — тем же стриминговым пайплайном, но без пауз реального времени.

Пишет results/quality/<имя>.tone-<декодер>.txt (сплошной текст для WER) и рядом
.phrases.txt с таймкодами фраз. Длительность работы и RTF печатаются: это ещё
одна точка про скорость машины.

  python quality/run_tone.py --decoder greedy samples/*.8k.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from tone import StreamingCTCPipeline
from tone.decoder import DecoderType

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--decoder", choices=["greedy", "kenlm"], default="greedy")
    args = ap.parse_args()
    kind = DecoderType.GREEDY if args.decoder == "greedy" else DecoderType.BEAM_SEARCH
    t = time.perf_counter()
    pipe = StreamingCTCPipeline.from_hugging_face(decoder_type=kind)
    print(f"модель с декодером {args.decoder} загружена за {time.perf_counter() - t:.1f} с")
    OUT.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        audio, sr = sf.read(str(f), dtype="int16")
        if sr != 8000:
            print(f"{f}: {sr} Гц, пропускаю — нужен 8k.wav из bench/prepare_samples.sh")
            continue
        audio = (audio[:, 0] if audio.ndim == 2 else audio).astype(np.int32)
        t = time.perf_counter()
        phrases = pipe.forward_offline(audio)
        dt = time.perf_counter() - t
        name = f.name.replace(".8k.wav", "")
        (OUT / f"{name}.tone-{args.decoder}.txt").write_text(" ".join(p.text for p in phrases) + "\n", encoding="utf-8")
        (OUT / f"{name}.tone-{args.decoder}.phrases.txt").write_text(
            "".join(f"[{p.start_time:7.2f}-{p.end_time:7.2f}] {p.text}\n" for p in phrases), encoding="utf-8")
        print(f"{name}: {len(audio) / 8000:.0f} с аудио, {len(phrases)} фраз, за {dt:.1f} с (RTF {dt / (len(audio) / 8000):.2f})")


if __name__ == "__main__":
    main()
