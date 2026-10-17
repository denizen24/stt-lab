"""Потоковый ресемплер 48 кГц → 8 кГц для одной аудиодорожки.

Браузер отдаёт PCM 48 кГц, модель ждёт 8 кГц. Понижаем в 6 раз: сначала
фильтр низких частот (FIR, срез 3,6 кГц — граница телефонной полосы, на
которой обучена T-one), потом берём каждый шестой отсчёт. Фильтр хранит хвост
между вызовами, поэтому стыки кусков не щёлкают и не теряют отсчётов.
"""

from __future__ import annotations

import numpy as np

FACTOR = 6
IN_RATE = 48000
OUT_RATE = 8000


def _lowpass_taps(numtaps: int = 97, cutoff_hz: float = 3600.0) -> np.ndarray:
    """Оконный FIR (Блэкман) без scipy: зависимостей у сервиса и так хватает."""
    n = np.arange(numtaps) - (numtaps - 1) / 2
    fc = cutoff_hz / IN_RATE
    h = 2 * fc * np.sinc(2 * fc * n)
    window = np.blackman(numtaps)
    h *= window
    return (h / h.sum()).astype(np.float32)


TAPS = _lowpass_taps()


class Downsampler:
    """Принимает int16 48 кГц любыми порциями, отдаёт int16 8 кГц."""

    def __init__(self) -> None:
        self._tail = np.zeros(len(TAPS) - 1, dtype=np.float32)
        self._phase = 0  # сколько отсчётов пропустить до следующего взятого

    def push(self, pcm48: np.ndarray) -> np.ndarray:
        if pcm48.size == 0:
            return np.zeros(0, dtype=np.int16)
        x = np.concatenate((self._tail, pcm48.astype(np.float32)))
        y = np.convolve(x, TAPS, mode="valid")  # len(y) == len(pcm48)
        self._tail = x[-(len(TAPS) - 1):]
        out = y[self._phase :: FACTOR]
        self._phase = (self._phase - len(y)) % FACTOR
        return np.clip(np.round(out), -32768, 32767).astype(np.int16)
