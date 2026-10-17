#!/usr/bin/env python3
"""Зонд акустической модели T-one: сколько стоит один чанк 300 мс на этой машине.

Зависит только от onnxruntime, numpy, huggingface-hub и soundfile — пакет tone
не нужен, поэтому зонд можно запустить на любой машине за минуту и сравнить
процессоры между собой. Один поток, чанки по порядку, состояние передаётся
как в стриминге.

  python bench/acoustic.py --audio samples/tone-short.8k.wav --threads 1 2 4 --profile
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
from huggingface_hub import hf_hub_download

CHUNK = 2400
STATE = 219729
PAD = 2400


def chunks_of(path: Path) -> list[np.ndarray]:
    data, sr = sf.read(str(path), dtype="int16", always_2d=True)
    assert sr == 8000, f"нужно 8 кГц, а в файле {sr}"
    audio = np.pad(data[:, 0].astype(np.int32), (PAD, PAD))
    audio = np.pad(audio, (0, -len(audio) % CHUNK))
    return np.split(audio, len(audio) // CHUNK)


def session(model: str, threads: int, spin: bool, profile: bool) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    opts.enable_profiling = profile
    return ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])


def run(sess: ort.InferenceSession, chunks: list[np.ndarray]) -> list[float]:
    dtype = np.float16 if "float16" in sess.get_inputs()[1].type else np.float32  # состояние того типа, какого ждёт граф
    state = np.zeros((1, STATE), dtype=dtype)
    times = []
    for c in chunks:
        t = time.perf_counter()
        _, state = sess.run(None, {"signal": c[None, :, None], "state": state})
        times.append((time.perf_counter() - t) * 1000)
    return times


def top_ops(profile_file: str, n: int = 8) -> list[tuple[str, float, int]]:
    events = json.loads(Path(profile_file).read_text())
    by_op: dict[str, list[float]] = defaultdict(list)
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            by_op[e["args"].get("op_name", "?")].append(e["dur"] / 1000)
    total = sum(sum(v) for v in by_op.values()) or 1
    rows = sorted(((op, sum(v) / total * 100, len(v)) for op, v in by_op.items()), key=lambda r: -r[1])
    return rows[:n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--no-spin", action="store_true", help="выключить spin-wait потоков onnxruntime")
    ap.add_argument("--profile", action="store_true", help="показать, какие операции съедают время")
    ap.add_argument("--repeat", type=int, default=2, help="проходов по файлу; первый — прогрев")
    ap.add_argument("--model", type=Path, default=None, help="свой .onnx вместо model.onnx из Hugging Face (например, квантованный)")
    args = ap.parse_args()

    model = str(args.model) if args.model else hf_hub_download("t-tech/T-one", "model.onnx")
    chunks = chunks_of(args.audio)
    print(f"onnxruntime {ort.__version__} · {args.audio.name}: {len(chunks)} чанков по 300 мс · spin {'off' if args.no_spin else 'on'}")
    for th in args.threads:
        t = time.perf_counter()
        sess = session(model, th, not args.no_spin, args.profile)
        load = time.perf_counter() - t
        times: list[float] = []
        for i in range(args.repeat):
            times = run(sess, chunks)  # остаётся последний проход
        rtf = sum(times) / (len(chunks) * 300)
        print(
            f"threads={th}: загрузка {load:.1f} с · чанк p50 {statistics.median(times):.1f} мс · "
            f"p95 {sorted(times)[int(len(times) * 0.95) - 1]:.1f} мс · max {max(times):.1f} мс · RTF {rtf:.3f}"
        )
        if args.profile:
            prof = sess.end_profiling()
            for op, pct, cnt in top_ops(prof):
                print(f"    {op:<22} {pct:5.1f}%  ({cnt} вызовов)")
            Path(prof).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
