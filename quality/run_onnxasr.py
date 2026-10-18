#!/usr/bin/env python3
"""Пакетная расшифровка через onnx-asr: GigaAM-v3 и Parakeet TDT 0.6B v3, fp32 или int8.

Второй проход каскада (чистовик после звонка) и вторая модель для правок латиницей.
Нарезка та же, что в run_gigaam.py (по тишине, куски до 25 с), чтобы сравнение
torch/onnx и fp32/int8 шло на одинаковых кусках, а каскад (quality/cascade.py)
мог сопоставлять модели кусок к куску.

Запускать из своего venv: /srv/stt-lab/.venv-onnxasr/bin/python quality/run_onnxasr.py \
    --model gigaam-v3-e2e-rnnt --quant int8 samples/*.16k.wav

Пишет results/quality/<имя>.<метка>.txt (сплошной текст для score.py), рядом
<имя>.<метка>.segments.jsonl (границы куска и текст) и строку замера в
results/batch-<хост>.csv: загрузка, RTF, минут на час аудио, пиковая память.
Метка: короткое имя модели, -int8 при квантовании, -8k если вход был 8 кГц.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

from run_gigaam import SR, segments

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"
SHORT = {
    "gigaam-v3-e2e-rnnt": "onnx-gigaam-e2e-rnnt",
    "gigaam-v3-rnnt": "onnx-gigaam-rnnt",
    "gigaam-v3-e2e-ctc": "onnx-gigaam-e2e-ctc",
    "nemo-parakeet-tdt-0.6b-v3": "parakeet-v3",
}


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # Linux: КБ


def load(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio[:, 0]
    if sr != SR:  # 8 кГц дорожки T-one поднимаем до 16 кГц: так увидим цену узкой полосы
        import subprocess  # noqa: PLC0415

        raw = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            check=True, capture_output=True).stdout
        audio = np.frombuffer(raw, dtype=np.float32).copy()
    return audio, sr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", default="gigaam-v3-e2e-rnnt", choices=sorted(SHORT))
    ap.add_argument("--quant", choices=["fp32", "int8"], default="fp32")
    ap.add_argument("--threads", type=int, default=0, help="intra_op потоков ORT, 0 — по умолчанию (все ядра)")
    ap.add_argument("--path", type=Path, help="каталог с файлами модели; иначе из кэша HF. Нужен для fp32 Parakeet: "
                    "onnxruntime ≥1.24 не читает внешние веса (.onnx.data) через симлинки кэша")
    ap.add_argument("--tag", default="", help="произвольная пометка строки в CSV (например, corun)")
    args = ap.parse_args()
    import onnx_asr  # noqa: PLC0415
    import onnxruntime as rt  # noqa: PLC0415

    so = rt.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
        so.inter_op_num_threads = 1
    t = time.perf_counter()
    model = onnx_asr.load_model(args.model, args.path, quantization=None if args.quant == "fp32" else "int8", sess_options=so)
    load_s = time.perf_counter() - t
    label = SHORT[args.model] + ("-int8" if args.quant == "int8" else "")
    print(f"{label}: загрузка {load_s:.1f} с · пик RSS {peak_rss_mb():.0f} МБ")
    OUT.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    log = ROOT / "results" / f"batch-{host}.csv"
    new = not log.exists()
    with log.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["ts", "file", "engine", "src_sr", "threads", "tag", "audio_s", "pieces", "load_s", "work_s", "rtf", "min_per_hour", "peak_rss_mb", "cpu_s"])
        for f in args.files:
            audio, src_sr = load(f)
            name = f.name.split(".8k.wav")[0].split(".16k.wav")[0].removesuffix(".wav")
            lab = label + ("-8k" if src_sr == 8000 else "")
            rows, pieces = [], []
            cpu0, t = os.times(), time.perf_counter()
            for a, b in segments(audio):
                text = model.recognize(audio[a:b], sample_rate=SR).strip()
                rows.append({"start": round(a / SR, 2), "end": round(b / SR, 2), "text": text})
                pieces.append(text)
            dt = time.perf_counter() - t
            cpu1 = os.times()
            cpu = (cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)
            sec = len(audio) / SR
            (OUT / f"{name}.{lab}.txt").write_text(" ".join(p for p in pieces if p) + "\n", encoding="utf-8")
            with (OUT / f"{name}.{lab}.segments.jsonl").open("w", encoding="utf-8") as sj:
                sj.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds"), name, lab, src_sr, args.threads or os.cpu_count(), args.tag,
                        round(sec, 1), len(rows), round(load_s, 1), round(dt, 1), round(dt / sec, 3), round(dt / sec * 60, 1), round(peak_rss_mb()), round(cpu, 1)])
            fh.flush()
            print(f"{name} [{lab}]: {sec:.0f} с, {len(rows)} кусков, за {dt:.1f} с (RTF {dt / sec:.3f}, ≈{dt / sec * 60:.1f} мин на час) · "
                  f"CPU {cpu:.0f} с · пик RSS {peak_rss_mb():.0f} МБ")
            print("  " + " ".join(pieces)[:200])


if __name__ == "__main__":
    main()
