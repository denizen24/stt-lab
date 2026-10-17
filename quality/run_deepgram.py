#!/usr/bin/env python3
"""Контрольная расшифровка через Deepgram (pre-recorded) тех же файлов, что идут в T-one.

Ключ берётся из переменной DEEPGRAM_API_KEY или из файла .env (--env), в вывод
не попадает. Результат: текст по каналам в results/quality/<имя>.deepgram-<модель>.txt
и сырой JSON в results/raw/. Стерео отправляется с multichannel=true — Deepgram
тарифицирует такой файл по каждому каналу, то есть вдвое.

  python quality/run_deepgram.py --env /path/.env --model nova-3 samples/tone-long.8k.wav
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"
RAW = ROOT / "results" / "raw"


def api_key(env_file: Path | None) -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "")
    if not key and env_file:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("DEEPGRAM_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"')
    if not key:
        sys.exit("нет DEEPGRAM_API_KEY: задайте переменную или --env")
    return key


def transcribe(path: Path, model: str, key: str, language: str) -> dict:
    info = sf.info(str(path))
    params = {"model": model, "language": language, "punctuate": "true", "smart_format": "false"}
    if info.channels > 1:
        params["multichannel"] = "true"
    url = "https://api.deepgram.com/v1/listen?" + "&".join(f"{k}={v}" for k, v in params.items())
    mime = {"WAV": "audio/wav", "FLAC": "audio/flac"}.get(info.format, "application/octet-stream")
    req = urllib.request.Request(url, data=path.read_bytes(), method="POST",
                                 headers={"Authorization": f"Token {key}", "Content-Type": mime})
    t = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        sys.exit(f"Deepgram {e.code}: {e.read()[:300].decode(errors='replace')}")
    body["_elapsed_sec"] = round(time.perf_counter() - t, 1)
    body["_audio_sec"] = round(info.duration, 1)
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--model", default="nova-2")
    ap.add_argument("--language", default="ru")
    ap.add_argument("--env", type=Path, default=None, help=".env с DEEPGRAM_API_KEY")
    args = ap.parse_args()
    key = api_key(args.env)
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    for f in args.files:
        body = transcribe(f, args.model, key, args.language)
        name = f.name.replace(".8k.wav", "").replace(".wav", "")
        (RAW / f"{name}.deepgram-{args.model}.json").write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
        channels = body["results"]["channels"]
        for i, ch in enumerate(channels):
            alt = ch["alternatives"][0]
            suffix = f".{['host', 'client'][i] if len(channels) == 2 else f'ch{i}'}" if len(channels) > 1 else ""
            out = OUT / f"{name}{suffix}.deepgram-{args.model}.txt"
            out.write_text(alt["transcript"].strip() + "\n", encoding="utf-8")
            meta = body.get("metadata", {})
            print(f"{out.name}: {len(alt.get('words', []))} слов · уверенность {alt.get('confidence', 0):.2f} · "
                  f"аудио {body['_audio_sec']} с за {body['_elapsed_sec']} с · модель {meta.get('models', ['?'])[0][:8]}…")
            print("  " + alt["transcript"].strip()[:400])


if __name__ == "__main__":
    main()
