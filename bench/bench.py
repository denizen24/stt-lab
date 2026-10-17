#!/usr/bin/env python3
"""Нагрузочный замер T-one: N потоков реального времени в одном процессе.

Воспроизводит устройство будущего сервиса: одна загруженная модель, у каждого
потока своё состояние, раз в 300 мс все активные потоки проходят через
акустическую модель одним батчем, затем сплиттер и декодер по одному потоку.

На каждый тик пишутся три времени:
  acoustic_ms — один вызов onnxruntime на батч из N чанков;
  decode_ms   — сплиттер плюс декодер по всем потокам (у KenLM это всплески
                в конце фраз, а не ровная нагрузка);
  tick_ms     — сумма, то, что должно укладываться в 300 мс.
И отставание от реального времени: насколько поздно начался тик относительно
расписания. Если оно растёт — машина не тянет столько потоков.

Пример:
  python bench/bench.py --audio samples/long.8k.wav --streams 2 10 20 --decoder greedy
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"

TICK_SEC = 0.3


def load_audio_8k(path: Path) -> np.ndarray:
    """Моно 8 кГц int32 в диапазоне int16 — то, чего ждёт модель."""
    data, sr = sf.read(str(path), dtype="int16", always_2d=True)
    if sr != 8000:
        sys.exit(f"{path}: частота {sr} Гц, нужна 8000. Прогоните bench/prepare_samples.sh")
    mono = data[:, 0] if data.shape[1] == 1 else data.mean(axis=1).astype(np.int16)
    return mono.astype(np.int32)


def build_pipeline(decoder: str, threads: int | None):
    """Собирает пайплайн T-one, при необходимости ограничивая потоки onnxruntime."""
    import onnxruntime as ort
    from tone.decoder import BeamSearchCTCDecoder, GreedyCTCDecoder
    from tone.logprob_splitter import StreamingLogprobSplitter
    from tone.onnx_wrapper import StreamingCTCModel
    from tone.pipeline import StreamingCTCPipeline

    model_path = StreamingCTCModel.download_from_hugging_face()
    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    model = StreamingCTCModel(ort.InferenceSession(model_path, sess_options=opts))
    dec = GreedyCTCDecoder() if decoder == "greedy" else BeamSearchCTCDecoder.from_hugging_face()
    return StreamingCTCPipeline(model, StreamingLogprobSplitter(), dec)


class Stream:
    """Один участник звонка: своя позиция в аудио и своё состояние сплиттера."""

    def __init__(self, audio: np.ndarray, offset: int, chunk: int) -> None:
        self.audio = audio
        self.pos = offset
        self.chunk = chunk
        self.splitter_state = None
        self.phrases = 0

    def next_chunk(self) -> np.ndarray:
        end = self.pos + self.chunk
        if end <= len(self.audio):
            piece = self.audio[self.pos : end]
        else:  # аудио зациклено, чтобы прогон был любой длины
            piece = np.concatenate((self.audio[self.pos :], self.audio[: end - len(self.audio)]))
        self.pos = end % len(self.audio)
        return piece


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    k = (len(values) - 1) * p
    ordered = sorted(values)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def step_stream(pipeline, s: "Stream") -> tuple[float, float, int]:
    """Режим threads: один поток целиком — акустика батчем 1, сплиттер, декодер."""
    t0 = time.perf_counter()
    logprobs, s.model_state = pipeline.model.forward(s.next_chunk()[None, :, None], s.model_state)
    t1 = time.perf_counter()
    found, s.splitter_state = pipeline.logprob_splitter.forward(np.asarray(logprobs[0], dtype=np.float32), s.splitter_state)
    for ph in found:
        pipeline.decoder.forward(ph.logprobs)
    s.phrases += len(found)
    return (t1 - t0) * 1000, (time.perf_counter() - t1) * 1000, len(found)


def run_one(pipeline, audio: np.ndarray, n_streams: int, seconds: float, warmup: float, mode: str, workers: int) -> dict:
    from tone.onnx_wrapper import StreamingCTCModel

    chunk = StreamingCTCModel.AUDIO_CHUNK_SAMPLES
    # Потоки стартуют с разных мест записи, чтобы концы фраз не совпадали.
    step = len(audio) // max(n_streams, 1)
    streams = [Stream(audio, (i * step) % len(audio), chunk) for i in range(n_streams)]
    for s in streams:
        s.model_state = None
    model_state = None
    pool = ThreadPoolExecutor(max_workers=workers) if mode == "threads" else None
    proc = psutil.Process()
    proc.cpu_percent(None)
    rss_max = 0
    ticks: list[dict] = []
    n_ticks = int((seconds + warmup) / TICK_SEC)
    t_start = time.perf_counter()
    for k in range(n_ticks):
        deadline = t_start + k * TICK_SEC
        now = time.perf_counter()
        lag_ms = max(0.0, (now - deadline) * 1000)
        if now < deadline:
            time.sleep(deadline - now)
        t0 = time.perf_counter()
        phrases_now = 0
        if pool is None:
            batch = np.stack([s.next_chunk() for s in streams])[:, :, None]
            logprobs, model_state = pipeline.model.forward(batch, model_state)
            t1 = time.perf_counter()
            for i, s in enumerate(streams):
                lp = np.asarray(logprobs[i], dtype=np.float32)
                found, s.splitter_state = pipeline.logprob_splitter.forward(lp, s.splitter_state)
                for ph in found:
                    pipeline.decoder.forward(ph.logprobs)
                    phrases_now += 1
                s.phrases += len(found)
            t2 = time.perf_counter()
            acoustic_ms, decode_ms = (t1 - t0) * 1000, (t2 - t1) * 1000
        else:
            # Все потоки идут параллельно; onnxruntime отпускает GIL на время run,
            # а вот сплиттер и декодер — чистый Python и выполняются по очереди.
            parts = list(pool.map(lambda s: step_stream(pipeline, s), streams))
            t2 = time.perf_counter()
            acoustic_ms = max(p[0] for p in parts)  # критический путь, а не сумма
            decode_ms = max(p[1] for p in parts)
            phrases_now = sum(p[2] for p in parts)
        rss_max = max(rss_max, proc.memory_info().rss)
        if k * TICK_SEC >= warmup:
            ticks.append(
                {
                    "acoustic_ms": acoustic_ms,
                    "decode_ms": decode_ms,
                    "tick_ms": (t2 - t0) * 1000,
                    "lag_ms": lag_ms,
                    "phrases": phrases_now,
                }
            )
    if pool is not None:
        pool.shutdown()
    cpu_pct = proc.cpu_percent(None)  # средняя загрузка процесса за прогон, 100 = одно ядро
    tick = [t["tick_ms"] for t in ticks]
    acoustic = [t["acoustic_ms"] for t in ticks]
    decode = [t["decode_ms"] for t in ticks]
    lag = [t["lag_ms"] for t in ticks]
    return {
        "streams": n_streams,
        "ticks": len(ticks),
        "tick_p50_ms": round(statistics.median(tick), 1),
        "tick_p95_ms": round(percentile(tick, 0.95), 1),
        "tick_max_ms": round(max(tick), 1),
        "acoustic_p50_ms": round(statistics.median(acoustic), 1),
        "acoustic_p95_ms": round(percentile(acoustic, 0.95), 1),
        "decode_p50_ms": round(statistics.median(decode), 1),
        "decode_p95_ms": round(percentile(decode, 0.95), 1),
        "decode_max_ms": round(max(decode), 1),
        "lag_max_ms": round(max(lag), 1),
        "lag_end_ms": round(lag[-1], 1),
        "phrases": sum(t["phrases"] for t in ticks),
        "cpu_pct_of_core": round(cpu_pct, 0),
        "cpu_pct_of_machine": round(cpu_pct / (os.cpu_count() or 1), 0),
        "rss_mb": round(rss_max / 1024 / 1024),
        "_ticks": ticks,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True, help="моно 8 кГц wav/flac")
    ap.add_argument("--streams", type=int, nargs="+", default=[2, 10, 20, 40, 64])
    ap.add_argument("--decoder", choices=["greedy", "kenlm"], default="greedy")
    ap.add_argument("--seconds", type=float, default=60, help="длительность замера после прогрева")
    ap.add_argument("--warmup", type=float, default=6, help="секунды прогрева, в статистику не входят")
    ap.add_argument("--threads", type=int, default=None, help="intra_op потоки onnxruntime (по умолчанию все ядра)")
    ap.add_argument("--mode", choices=["batch", "threads"], default="batch",
                    help="batch: один вызов модели на все потоки; threads: поток на участника, intra_op=1")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="рабочих потоков в режиме threads")
    ap.add_argument("--tag", default="", help="пометка прогона для таблицы")
    args = ap.parse_args()
    if args.mode == "threads" and args.threads is None:
        args.threads = 1

    audio = load_audio_8k(args.audio)
    print(f"аудио {args.audio.name}: {len(audio) / 8000:.1f} с · декодер {args.decoder} · режим {args.mode} · потоки ORT {args.threads or 'все'}" + (f" · рабочих {args.workers}" if args.mode == "threads" else ""))
    t = time.perf_counter()
    pipeline = build_pipeline(args.decoder, args.threads)
    print(f"модель загружена за {time.perf_counter() - t:.1f} с, RSS {psutil.Process().memory_info().rss / 2**20:.0f} МБ")

    host = socket.gethostname()
    stamp = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M")
    RAW.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS / f"bench-{host}.csv"
    fields = [
        "stamp", "host", "tag", "mode", "decoder", "ort_threads", "audio", "streams", "ticks",
        "tick_p50_ms", "tick_p95_ms", "tick_max_ms", "acoustic_p50_ms", "acoustic_p95_ms",
        "decode_p50_ms", "decode_p95_ms", "decode_max_ms", "lag_max_ms", "lag_end_ms",
        "phrases", "cpu_pct_of_core", "cpu_pct_of_machine", "rss_mb",
    ]
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if new_file:
            writer.writeheader()
        for n in args.streams:
            print(f"\n== {n} потоков, {args.seconds:.0f} с")
            r = run_one(pipeline, audio, n, args.seconds, args.warmup, args.mode, args.workers)
            ticks = r.pop("_ticks")
            row = {"stamp": stamp, "host": host, "tag": args.tag, "mode": args.mode, "decoder": args.decoder,
                   "ort_threads": args.threads or "all", "audio": args.audio.name, **r}
            writer.writerow(row)
            fh.flush()
            (RAW / f"{stamp}-{args.mode}-{args.decoder}-{n}.json").write_text(json.dumps({"row": row, "ticks": ticks}), encoding="utf-8")
            verdict = "держит" if r["tick_p95_ms"] < TICK_SEC * 1000 and r["lag_end_ms"] < 1000 else "НЕ ДЕРЖИТ"
            print(
                f"tick p50/p95/max {r['tick_p50_ms']}/{r['tick_p95_ms']}/{r['tick_max_ms']} мс · "
                f"акустика p95 {r['acoustic_p95_ms']} · декод p95/max {r['decode_p95_ms']}/{r['decode_max_ms']} · "
                f"отставание max {r['lag_max_ms']} мс · CPU {r['cpu_pct_of_machine']}% машины · "
                f"RSS {r['rss_mb']} МБ · фраз {r['phrases']} → {verdict}"
            )
    print(f"\nтаблица: {csv_path}")


if __name__ == "__main__":
    main()
