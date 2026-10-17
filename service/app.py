"""stt-streaming: онлайн-распознавание русской речи для звонка один на один.

Один процесс, одна модель, две дорожки на встречу. Контракт:

  GET  /health                       — процесс жив, модель загружена
  GET  /v1/capacity                  — активные дорожки, лимит, принимает ли новые сессии
  POST /v1/sessions                  — создать встречу (резервирует две дорожки); 429 при перегрузе
  WS   /v1/sessions/{id}/stream?role=host|client
        → клиент шлёт бинарный PCM s16le, mono, 48 кГц; ресемплинг делает сервис
        ← сервер шлёт JSON {"type":"phrase","role","text","start_ms","end_ms","lag_ms"}
           Частичных гипотез нет: текст приходит фразой по паузе.
  DELETE /v1/sessions/{id}          — finalize обеих дорожек, хвост фраз в ответе
  GET  /metrics                      — текст в духе Prometheus, без содержимого речи

Авторизация: заголовок X-STT-Secret равен STT_SHARED_SECRET (если задан).
В логах нет ни текста фраз, ни аудио: только идентификаторы, счётчики и время.
"""

from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import psutil
from fastapi import Depends, FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import PlainTextResponse

from service.engine import Engine, StreamState
from service.resample import Downsampler

log = logging.getLogger("stt-streaming")
ROLES = ("host", "client")


@dataclass
class Track:
    role: str
    state: StreamState
    down: Downsampler = field(default_factory=Downsampler)
    received_ms: int = 0  # сколько аудио (после ресемплинга) принято
    phrases: int = 0
    connected: bool = False
    closed: bool = False


@dataclass
class Session:
    id: str
    created: float
    tracks: dict[str, Track]
    tail: list[dict] = field(default_factory=list)


class Hub:
    """Учёт сессий и дорожек, лимит и пул рабочих потоков."""

    def __init__(self, engine: Engine, max_streams: int, workers: int) -> None:
        self.engine = engine
        self.max_streams = max_streams
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stt")
        self.sessions: dict[str, Session] = {}
        self.rejected = 0
        self.sessions_total = 0

    @property
    def active_streams(self) -> int:
        return sum(1 for s in self.sessions.values() for t in s.tracks.values() if not t.closed)

    def accepting(self) -> bool:
        return self.active_streams + len(ROLES) <= self.max_streams

    def create(self) -> Session:
        if not self.accepting():
            self.rejected += 1
            raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "нет свободных дорожек")
        sid = uuid.uuid4().hex
        s = Session(sid, time.time(), {r: Track(r, self.engine.new_stream()) for r in ROLES})
        self.sessions[sid] = s
        self.sessions_total += 1
        log.info("session %s created, active_streams=%d", sid[:8], self.active_streams)
        return s

    async def feed(self, track: Track, pcm48: bytes) -> list[dict]:
        samples = np.frombuffer(pcm48, dtype="<i2")
        pcm8 = track.down.push(samples)
        if pcm8.size == 0:
            return []
        track.received_ms += int(len(pcm8) * 1000 / 8000)
        loop = asyncio.get_running_loop()
        phrases = await loop.run_in_executor(self.pool, track.state.push, pcm8)
        return [self._event(track, p) for p in phrases]

    async def finalize(self, track: Track) -> list[dict]:
        if track.closed:
            return []
        track.closed = True
        loop = asyncio.get_running_loop()
        phrases = await loop.run_in_executor(self.pool, track.state.finalize)
        return [self._event(track, p) for p in phrases]

    def _event(self, track: Track, p) -> dict:  # noqa: ANN001
        track.phrases += 1
        return {"type": "phrase", "role": track.role, "text": p.text, "start_ms": p.start_ms, "end_ms": p.end_ms,
                "lag_ms": max(0, track.received_ms - p.end_ms)}

    async def close(self, sid: str) -> list[dict]:
        s = self.sessions.get(sid)
        if s is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "нет такой сессии")
        tail = list(s.tail)
        for t in s.tracks.values():
            tail.extend(await self.finalize(t))
        del self.sessions[sid]
        log.info("session %s closed after %.0fs, phrases host=%d client=%d", sid[:8], time.time() - s.created,
                 s.tracks["host"].phrases, s.tracks["client"].phrases)
        return tail


def create_app(engine: Engine, *, max_streams: int | None = None, workers: int | None = None,
               secret: str | None = None) -> FastAPI:
    max_streams = max_streams or int(os.environ.get("STT_MAX_STREAMS", "4"))
    workers = workers or int(os.environ.get("STT_WORKERS", str(os.cpu_count() or 2)))
    secret = secret if secret is not None else os.environ.get("STT_SHARED_SECRET", "")
    hub = Hub(engine, max_streams, workers)
    app = FastAPI(title="stt-streaming", version="0.1.0", docs_url=None, redoc_url=None)
    app.state.hub = hub

    def check_secret(request: Request) -> None:
        if secret and not secrets.compare_digest(request.headers.get("x-stt-secret", ""), secret):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "нужен X-STT-Secret")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "engine": engine.name, "active_streams": hub.active_streams}

    @app.get("/v1/capacity", dependencies=[Depends(check_secret)])
    def capacity() -> dict:
        return {"active_streams": hub.active_streams, "max_streams": hub.max_streams, "accepting": hub.accepting()}

    @app.post("/v1/sessions", status_code=201, dependencies=[Depends(check_secret)])
    def create_session() -> dict:
        s = hub.create()
        return {"session_id": s.id, "roles": list(ROLES), "partials": False, "input": "pcm_s16le_mono_48000"}

    @app.delete("/v1/sessions/{sid}", dependencies=[Depends(check_secret)])
    async def delete_session(sid: str) -> dict:
        return {"phrases": await hub.close(sid)}

    @app.websocket("/v1/sessions/{sid}/stream")
    async def stream(ws: WebSocket, sid: str, role: str = Query(...)) -> None:
        if secret and not secrets.compare_digest(ws.headers.get("x-stt-secret", ""), secret):
            await ws.close(code=4401)
            return
        s = hub.sessions.get(sid)
        if s is None or role not in ROLES or s.tracks[role].connected:
            await ws.close(code=4404)
            return
        track = s.tracks[role]
        track.connected = True
        await ws.accept()
        log.info("session %s %s connected", sid[:8], role)
        try:
            while True:
                data = await ws.receive_bytes()
                for ev in await hub.feed(track, data):
                    await ws.send_json(ev)
        except WebSocketDisconnect:
            pass
        finally:
            # Дорожка закрылась — досчитываем хвост; если сессия ещё жива, хвост отдадим в DELETE.
            tail = await hub.finalize(track)
            if sid in hub.sessions:
                hub.sessions[sid].tail.extend(tail)
            log.info("session %s %s disconnected, received=%dms phrases=%d", sid[:8], role, track.received_ms, track.phrases)

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        rss = psutil.Process().memory_info().rss / 2**20
        p95 = getattr(engine, "tick_p95_ms", lambda: 0.0)()
        return (
            f"stt_active_streams {hub.active_streams}\n"
            f"stt_max_streams {hub.max_streams}\n"
            f"stt_sessions_total {hub.sessions_total}\n"
            f"stt_rejected_total {hub.rejected}\n"
            f"stt_tick_p95_ms {p95}\n"
            f"stt_rss_mb {rss:.0f}\n"
        )

    return app


def main() -> FastAPI:
    """Точка входа для uvicorn: `uvicorn service.app:main --factory`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from service.engine import ToneEngine

    t = time.perf_counter()
    engine = ToneEngine(decoder=os.environ.get("STT_DECODER", "greedy"))
    log.info("engine %s loaded in %.1fs: model=%s decoder=%s", engine.name, time.perf_counter() - t,
             engine.model_path, engine.decoder)
    return create_app(engine)
