"""Контракт сервиса на подменном движке: модель для этих проверок не нужна."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from service.app import create_app
from service.engine import Phrase
from service.resample import FACTOR, Downsampler

SECRET = "test-secret"
H = {"X-STT-Secret": SECRET}


class FakeStream:
    """Выдаёт фразу на каждые 3 чанка по 300 мс и одну в finalize."""

    def __init__(self) -> None:
        self.samples = 0
        self.chunks = 0

    def push(self, pcm8k: np.ndarray) -> list[Phrase]:
        self.samples += len(pcm8k)
        out = []
        while self.samples >= 2400:
            self.samples -= 2400
            self.chunks += 1
            if self.chunks % 3 == 0:
                out.append(Phrase("фраза", (self.chunks - 3) * 300, self.chunks * 300))
        return out

    def finalize(self) -> list[Phrase]:
        return [Phrase("хвост", self.chunks * 300, self.chunks * 300 + 100)]


class FakeEngine:
    name = "fake"

    def new_stream(self) -> FakeStream:
        return FakeStream()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app(FakeEngine(), max_streams=4, workers=2, secret=SECRET))


def pcm48(ms: int) -> bytes:
    return (np.zeros(48000 * ms // 1000, dtype="<i2")).tobytes()


def test_health_is_open_and_everything_else_needs_the_secret(client: TestClient) -> None:
    assert client.get("/health").json()["ok"] is True
    assert client.get("/v1/capacity").status_code == 401
    assert client.get("/v1/capacity", headers=H).json() == {"active_streams": 0, "max_streams": 4, "accepting": True}


def test_a_meeting_gets_phrases_on_both_tracks_and_a_tail_on_delete(client: TestClient) -> None:
    sid = client.post("/v1/sessions", headers=H).json()["session_id"]
    got = {}
    for role in ("host", "client"):
        with client.websocket_connect(f"/v1/sessions/{sid}/stream?role={role}", headers=H) as ws:
            for _ in range(5):  # 5 × 200 мс = 1 с → 3 чанка → одна фраза
                ws.send_bytes(pcm48(200))
            got[role] = ws.receive_json()
    assert got["host"]["type"] == "phrase" and got["host"]["role"] == "host"
    assert got["client"]["text"] == "фраза"
    assert 0 <= got["host"]["lag_ms"] <= 1000
    tail = client.delete(f"/v1/sessions/{sid}", headers=H).json()["phrases"]
    assert [t["text"] for t in tail] == ["хвост", "хвост"]
    assert client.get("/v1/capacity", headers=H).json()["active_streams"] == 0


def test_capacity_limit_rejects_the_third_meeting_without_creating_it(client: TestClient) -> None:
    a = client.post("/v1/sessions", headers=H).json()["session_id"]
    b = client.post("/v1/sessions", headers=H).json()["session_id"]
    assert client.get("/v1/capacity", headers=H).json() == {"active_streams": 4, "max_streams": 4, "accepting": False}
    assert client.post("/v1/sessions", headers=H).status_code == 429
    assert "stt_rejected_total 1" in client.get("/metrics").text
    client.delete(f"/v1/sessions/{a}", headers=H)
    assert client.get("/v1/capacity", headers=H).json()["accepting"] is True
    client.delete(f"/v1/sessions/{b}", headers=H)


def test_unknown_session_or_role_closes_the_socket(client: TestClient) -> None:
    sid = client.post("/v1/sessions", headers=H).json()["session_id"]
    with pytest.raises(Exception):  # noqa: B017, PT011 — starlette поднимает WebSocketDisconnect с кодом 4404
        with client.websocket_connect(f"/v1/sessions/{sid}/stream?role=guest", headers=H) as ws:
            ws.receive_json()
    assert client.delete(f"/v1/sessions/{sid}", headers=H).status_code == 200
    assert client.delete(f"/v1/sessions/{sid}", headers=H).status_code == 404


def test_downsampler_keeps_sample_count_and_passes_a_voice_band_tone() -> None:
    d = Downsampler()
    t = np.arange(48000) / 48000
    tone = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
    out = np.concatenate([d.push(tone[i : i + 960]) for i in range(0, len(tone), 960)])
    assert len(out) == len(tone) // FACTOR
    assert out[2000:].std() > 6000  # сигнал прошёл, а не затух
    hiss = Downsampler()
    tone_hi = (np.sin(2 * np.pi * 20000 * t) * 10000).astype(np.int16)
    out_hi = np.concatenate([hiss.push(tone_hi[i : i + 960]) for i in range(0, len(tone_hi), 960)])
    assert out_hi[2000:].std() < 300  # а 20 кГц фильтр срезал, иначе он бы отразился в полосу речи
