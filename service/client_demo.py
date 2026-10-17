#!/usr/bin/env python3
"""Пробный клиент: открывает встречу и гонит две дорожки из файлов в реальном времени, как браузер.

Файлы любого формата: ffmpeg переводит их в PCM s16le mono 48 кГц, дальше кадры
по 20 мс уходят в WebSocket. Печатает фразы с ролью и отставанием от живого края.

  python service/client_demo.py --host samples/x.host.8k.wav --client samples/x.client.8k.wav
  python service/client_demo.py --host a.wav --client b.wav --speed 4   # быстрее реального времени
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import httpx
import websockets

FRAME_MS = 20
FRAME_BYTES = 48000 * FRAME_MS // 1000 * 2
ROOT = Path(__file__).resolve().parent.parent
GOT: dict[str, list[dict]] = {"host": [], "client": []}


def pcm48(path: Path) -> bytes:
    return subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
                           "-ar", "48000", "-"], check=True, capture_output=True).stdout


async def send_track(url: str, role: str, data: bytes, speed: float, headers: dict, quiet: bool = False) -> None:
    async with websockets.connect(f"{url}/stream?role={role}", additional_headers=headers, max_size=None) as ws:
        async def reader() -> None:
            async for msg in ws:
                ev = json.loads(msg)
                GOT[role].append(ev)
                if not quiet:
                    print(f"[{role:6}] {ev['start_ms'] / 1000:7.2f}-{ev['end_ms'] / 1000:7.2f}  lag {ev['lag_ms']:5d} мс  {ev['text']}")

        task = asyncio.create_task(reader())
        t0 = time.perf_counter()
        for i in range(0, len(data), FRAME_BYTES):
            await ws.send(data[i : i + FRAME_BYTES])
            due = t0 + (i // FRAME_BYTES + 1) * FRAME_MS / 1000 / speed
            delay = due - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        await asyncio.sleep(1.0)
        await ws.close()
        await task


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default=os.environ.get("STT_URL", "http://127.0.0.1:8090"))
    ap.add_argument("--host", type=Path, required=True)
    ap.add_argument("--client", type=Path, required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--save-as", default=None, help="имя записи: тексты по ролям → results/quality/<имя>.<роль>.<tag>.txt")
    ap.add_argument("--tag", default="tone-service-greedy")
    ap.add_argument("--quiet", action="store_true", help="не печатать фразы (текст не должен попадать в журналы)")
    args = ap.parse_args()
    headers = {"X-STT-Secret": os.environ.get("STT_SHARED_SECRET", "")}
    async with httpx.AsyncClient(base_url=args.base, headers=headers, timeout=30) as http:
        cap = (await http.get("/v1/capacity")).json()
        print("ёмкость:", cap)
        r = await http.post("/v1/sessions")
        r.raise_for_status()
        sid = r.json()["session_id"]
        ws_base = args.base.replace("http", "ws", 1) + f"/v1/sessions/{sid}"
        t = time.perf_counter()
        await asyncio.gather(send_track(ws_base, "host", pcm48(args.host), args.speed, headers, args.quiet),
                             send_track(ws_base, "client", pcm48(args.client), args.speed, headers, args.quiet))
        tail = (await http.delete(f"/v1/sessions/{sid}")).json()["phrases"]
        for ev in tail:
            GOT[ev["role"]].append(ev)
            if not args.quiet:
                print(f"[{ev['role']:6}] {ev['start_ms'] / 1000:7.2f}-{ev['end_ms'] / 1000:7.2f}  (хвост)        {ev['text']}")
        print(f"сессия закрыта за {time.perf_counter() - t:.1f} с; метрики:\n" + (await http.get("/metrics")).text)
        for role, evs in GOT.items():
            lags = sorted(e["lag_ms"] for e in evs if "lag_ms" in e)
            if lags:
                print(f"{role}: фраз {len(evs)}, отставание p50 {lags[len(lags) // 2]} мс, p95 {lags[int(len(lags) * 0.95) - 1]} мс, max {lags[-1]} мс")
            if args.save_as:
                out = ROOT / "results" / "quality"
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{args.save_as}.{role}.{args.tag}.txt").write_text(" ".join(e["text"] for e in evs) + "\n", encoding="utf-8")
                (out / f"{args.save_as}.{role}.{args.tag}.phrases.txt").write_text(
                    "".join(f"[{e['start_ms'] / 1000:7.2f}-{e['end_ms'] / 1000:7.2f}] lag {e.get('lag_ms', 0):5d}  {e['text']}\n" for e in evs), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
