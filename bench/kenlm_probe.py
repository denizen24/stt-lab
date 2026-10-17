#!/usr/bin/env python3
"""Цена декодера KenLM beam search: сколько стоит каждая фраза и сколько памяти ест модель языка.

Декодер вызывается в конце фразы (по паузе), поэтому его стоимость — не ровная
нагрузка, а всплеск. Скрипт оборачивает декодер таймером, прогоняет запись
офлайн и печатает по каждой фразе: миллисекунды декода, длину фразы в секундах
и текст. Для сравнения тот же файл печатается greedy-декодером.

  python bench/kenlm_probe.py --audio samples/tone-long.8k.wav
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

from tone import StreamingCTCPipeline
from tone.decoder import DecoderType


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True)
    args = ap.parse_args()
    audio, _ = sf.read(str(args.audio), dtype="int16")
    audio = audio.astype(np.int32)
    proc = psutil.Process()

    t = time.perf_counter()
    greedy = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.GREEDY)
    print(f"greedy: загрузка {time.perf_counter() - t:.1f} с · RSS {proc.memory_info().rss / 2**20:.0f} МБ")
    greedy_phrases = greedy.forward_offline(audio)

    t = time.perf_counter()
    beam = StreamingCTCPipeline.from_hugging_face(decoder_type=DecoderType.BEAM_SEARCH)
    print(f"kenlm:  загрузка {time.perf_counter() - t:.1f} с · RSS {proc.memory_info().rss / 2**20:.0f} МБ")

    log: list[tuple[float, float]] = []
    original = beam.decoder.forward

    def timed(logprobs):  # noqa: ANN001, ANN202
        t0 = time.perf_counter()
        text = original(logprobs)
        log.append(((time.perf_counter() - t0) * 1000, len(logprobs) * 0.03))
        return text

    beam.decoder.forward = timed  # type: ignore[method-assign]
    t = time.perf_counter()
    beam_phrases = beam.forward_offline(audio)
    total = time.perf_counter() - t
    decode_total = sum(ms for ms, _ in log)
    print(
        f"офлайн с KenLM: {len(audio) / 8000:.1f} с аудио за {total:.1f} с, из них декод {decode_total / 1000:.1f} с · "
        f"RSS после {proc.memory_info().rss / 2**20:.0f} МБ"
    )
    print(f"{'декод, мс':>10} {'фраза, с':>9}  {'мс на 1 с речи':>14}  текст (kenlm | greedy)")
    for (ms, sec), ph, gp in zip(log, beam_phrases, greedy_phrases):
        mark = "" if ph.text == gp.text else "  ← отличается"
        print(f"{ms:10.1f} {sec:9.1f}  {ms / max(sec, 0.01):14.0f}  {ph.text} | {gp.text}{mark}")


if __name__ == "__main__":
    main()
