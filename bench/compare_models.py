#!/usr/bin/env python3
"""Сравнивает варианты акустической модели T-one: скорость чанка и текст на одной записи.

Первый файл в списке — эталон. Для каждого: время загрузки, p50/p95 чанка
300 мс в один поток, RTF; затем расшифровка длинной записи greedy-декодером
и проверка, совпали ли фразы с эталоном дословно. Тип состояния стрима
(float16 или float32) берётся из самой модели.

  python bench/compare_models.py --short samples/tone-short.8k.wav --long samples/tone-long.8k.wav \
      "$(python -c 'from huggingface_hub import hf_hub_download as d; print(d("t-tech/T-one","model.onnx"))')" \
      models/T-one-fp32/model.onnx
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acoustic import chunks_of  # noqa: E402

from tone.decoder import GreedyCTCDecoder  # noqa: E402
from tone.logprob_splitter import StreamingLogprobSplitter  # noqa: E402
from tone.onnx_wrapper import StreamingCTCModel  # noqa: E402
from tone.pipeline import StreamingCTCPipeline  # noqa: E402


class AnyPrecisionModel(StreamingCTCModel):
    """Та же акустическая модель, но состояние того типа, какого требует граф."""

    def __init__(self, path: str, threads: int | None = None) -> None:
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        super().__init__(ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"]))
        self.state_dtype = np.float16 if "float16" in self._ort_sess.get_inputs()[1].type else np.float32

    def forward(self, audio_chunk, state=None):  # noqa: ANN001, D102
        if state is None:
            state = np.zeros((audio_chunk.shape[0], self.STATE_SIZE), dtype=self.state_dtype)
        return self._ort_sess.run(None, {"signal": audio_chunk, "state": state})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="пути к .onnx, первый — эталон")
    ap.add_argument("--short", type=Path, required=True)
    ap.add_argument("--long", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    chunks = chunks_of(args.short)
    audio, _ = sf.read(str(args.long), dtype="int16")
    audio = audio.astype(np.int32)
    reference: list[str] | None = None
    proc = psutil.Process()
    for path in args.models:
        rss0 = proc.memory_info().rss
        t = time.perf_counter()
        model = AnyPrecisionModel(path, args.threads)
        load = time.perf_counter() - t
        state = None
        for c in chunks:  # прогрев
            _, state = model.forward(c[None, :, None], state)
        state, times = None, []
        for c in chunks:
            t = time.perf_counter()
            _, state = model.forward(c[None, :, None], state)
            times.append((time.perf_counter() - t) * 1000)
        rtf = sum(times) / (len(chunks) * 300)
        pipe = StreamingCTCPipeline(model, StreamingLogprobSplitter(), GreedyCTCDecoder())
        t = time.perf_counter()
        phrases = pipe.forward_offline(audio)
        offline = time.perf_counter() - t
        texts = [p.text for p in phrases]
        verdict = "эталон" if reference is None else ("текст совпал дословно" if texts == reference else "ТЕКСТ ОТЛИЧАЕТСЯ")
        reference = reference or texts
        print(
            f"{Path(path).parent.name}/{Path(path).name} · {Path(path).stat().st_size / 2**20:.0f} МБ · состояние {np.dtype(model.state_dtype).name}\n"
            f"  загрузка {load:.1f} с · +RSS {(proc.memory_info().rss - rss0) / 2**20:.0f} МБ · "
            f"чанк p50 {statistics.median(times):.1f} / p95 {sorted(times)[int(len(times) * 0.95) - 1]:.1f} мс · "
            f"RTF {rtf:.3f} ({args.threads} поток ORT) · офлайн {len(audio) / 8000:.0f} с за {offline:.1f} с → {verdict}"
        )
        if verdict == "ТЕКСТ ОТЛИЧАЕТСЯ":
            for a, b in zip(reference, texts):
                if a != b:
                    print(f"    эталон: {a}\n    здесь:  {b}")


if __name__ == "__main__":
    main()
