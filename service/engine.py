"""Движок распознавания: одна загруженная модель T-one, у каждой дорожки своё состояние.

По замерам стенда модель не выигрывает от батчинга и от многопоточного ORT, зато
хорошо живёт как «один поток ORT на дорожку»: поэтому сессия создаётся с
intra_op_num_threads=1, а параллельность даёт пул рабочих потоков в приложении.
Акустика идёт через onnxruntime и отпускает GIL; сплиттер и декодер — Python.

Движок ничего не знает о сети: ему отдают int16 8 кГц, он возвращает фразы.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

CHUNK = 2400  # 300 мс при 8 кГц
PAD = 2400  # модель обучена с паддингом 300 мс по краям, см. tone.pipeline


@dataclass
class Phrase:
    text: str
    start_ms: int
    end_ms: int


class StreamState(Protocol):
    def push(self, pcm8k: np.ndarray) -> list[Phrase]: ...
    def finalize(self) -> list[Phrase]: ...


class Engine(Protocol):
    name: str

    def new_stream(self) -> StreamState: ...


class _ToneStream:
    def __init__(self, engine: "ToneEngine") -> None:
        self._e = engine
        self._buf = np.array([], dtype=np.int32)
        self._model_state = None
        self._split_state = None
        self._started = False
        self._done = False

    def _forward(self, chunk: np.ndarray, *, is_last: bool = False) -> list[Phrase]:
        from tone.pipeline import StreamingCTCPipeline

        pipe: StreamingCTCPipeline = self._e.pipeline
        state = None if self._model_state is None else (self._model_state, self._split_state)
        t = time.perf_counter()
        phrases, (self._model_state, self._split_state) = pipe.forward(chunk, state, is_last=is_last)
        self._e.observe_tick((time.perf_counter() - t) * 1000)
        # Пайплайн сам вычитает паддинг из таймкодов, поэтому его времена — уже «аудио без паддинга».
        return [Phrase(p.text, int(p.start_time * 1000), int(p.end_time * 1000)) for p in phrases if p.text]

    def push(self, pcm8k: np.ndarray) -> list[Phrase]:
        if self._done:
            return []
        if not self._started:
            self._buf = np.concatenate((np.zeros(PAD, dtype=np.int32), pcm8k.astype(np.int32)))
            self._started = True
        else:
            self._buf = np.concatenate((self._buf, pcm8k.astype(np.int32)))
        out: list[Phrase] = []
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            out.extend(self._forward(chunk))
        return out

    def finalize(self) -> list[Phrase]:
        if self._done:
            return []
        self._done = True
        if not self._started:
            return []
        tail = np.concatenate((self._buf, np.zeros(PAD, dtype=np.int32)))
        tail = np.pad(tail, (0, -len(tail) % CHUNK))
        out: list[Phrase] = []
        chunks = np.split(tail, len(tail) // CHUNK) if len(tail) else []
        for i, chunk in enumerate(chunks):
            out.extend(self._forward(chunk, is_last=i == len(chunks) - 1))
        if not chunks:
            out.extend(self._forward(np.zeros(CHUNK, dtype=np.int32), is_last=True))
        return out


class ToneEngine:
    """Настоящий движок. Параметры — из окружения: STT_MODEL_PATH, STT_DECODER."""

    name = "t-one"

    def __init__(self, model_path: str | None = None, decoder: str = "greedy") -> None:
        import onnxruntime as ort
        from tone.decoder import BeamSearchCTCDecoder, GreedyCTCDecoder
        from tone.logprob_splitter import StreamingLogprobSplitter
        from tone.onnx_wrapper import StreamingCTCModel
        from tone.pipeline import StreamingCTCPipeline

        path = model_path or os.environ.get("STT_MODEL_PATH") or StreamingCTCModel.download_from_hugging_face()
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        state_dtype = np.float16 if "float16" in sess.get_inputs()[1].type else np.float32

        class _Model(StreamingCTCModel):
            def forward(self, audio_chunk, state=None):  # noqa: ANN001
                if state is None:
                    state = np.zeros((audio_chunk.shape[0], self.STATE_SIZE), dtype=state_dtype)
                return self._ort_sess.run(None, {"signal": audio_chunk, "state": state})

        dec = GreedyCTCDecoder() if decoder == "greedy" else BeamSearchCTCDecoder.from_hugging_face()
        self.pipeline = StreamingCTCPipeline(_Model(sess), StreamingLogprobSplitter(), dec)
        self.model_path = Path(path).name
        self.decoder = decoder
        self._ticks: list[float] = []

    def new_stream(self) -> _ToneStream:
        return _ToneStream(self)

    def observe_tick(self, ms: float) -> None:
        self._ticks.append(ms)
        if len(self._ticks) > 2000:
            del self._ticks[:1000]

    def tick_p95_ms(self) -> float:
        if not self._ticks:
            return 0.0
        s = sorted(self._ticks[-1000:])
        return round(s[int(len(s) * 0.95) - 1] if len(s) > 1 else s[0], 1)
