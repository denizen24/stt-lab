#!/usr/bin/env python3
"""Динамическое int8-квантование акустической модели T-one и честная проверка, что оно даёт.

На процессорах без AVX2 матричные операции в fp32 медленные; int8 иногда
выигрывает в 1,5–2 раза, но может испортить распознавание. Поэтому скрипт
меряет и скорость (чанк 300 мс, один поток), и текст: одна и та же запись
прогоняется через исходную и квантованную модель с greedy-декодером,
фразы печатаются рядом.

  python bench/quantize.py --short samples/tone-short.8k.wav --long samples/tone-long.8k.wav
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acoustic import chunks_of, run, session  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def transcribe(model_path: str, audio: np.ndarray) -> tuple[list, float]:
    from tone.decoder import GreedyCTCDecoder
    from tone.logprob_splitter import StreamingLogprobSplitter
    from tone.onnx_wrapper import StreamingCTCModel
    from tone.pipeline import StreamingCTCPipeline

    pipe = StreamingCTCPipeline(StreamingCTCModel.from_local(model_path), StreamingLogprobSplitter(), GreedyCTCDecoder())
    t = time.perf_counter()
    phrases = pipe.forward_offline(audio)
    return phrases, time.perf_counter() - t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--short", type=Path, required=True, help="короткий файл для замера скорости")
    ap.add_argument("--long", type=Path, required=True, help="длинный файл для сравнения текста")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "T-one-int8" / "model.onnx")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    src = hf_hub_download("t-tech/T-one", "model.onnx")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        import onnx
        from onnxruntime.quantization import QuantType, quantize_dynamic

        # Внутри модели часть ветвей считается в float16, а динамическое квантование
        # умеет только float32 — такие MatMul исключаем, остальные переводим в int8.
        m = onnx.shape_inference.infer_shapes(onnx.load(src))
        f16 = {vi.name for vi in list(m.graph.value_info) + list(m.graph.input)
               if vi.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16}
        f16 |= {t.name for t in m.graph.initializer if t.data_type == onnx.TensorProto.FLOAT16}
        matmuls = [n for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
        exclude = [n.name for n in matmuls if any(i in f16 for i in n.input)]
        print(f"MatMul всего {len(matmuls)}, из них в float16 и остаются как есть: {len(exclude)}")
        t = time.perf_counter()
        quantize_dynamic(src, str(args.out), weight_type=QuantType.QUInt8,
                         op_types_to_quantize=["MatMul", "Gemm"], nodes_to_exclude=exclude)
        print(f"квантование: {time.perf_counter() - t:.0f} с")
    print(f"fp32 {Path(src).stat().st_size / 2**20:.0f} МБ → int8 {args.out.stat().st_size / 2**20:.0f} МБ")

    chunks = chunks_of(args.short)
    for name, path in (("fp32", src), ("int8", str(args.out))):
        sess = session(path, args.threads, True, False)
        run(sess, chunks)  # прогрев
        times = run(sess, chunks)
        print(f"{name}: чанк p50 {statistics.median(times):.1f} мс · max {max(times):.1f} мс · RTF {sum(times) / (len(chunks) * 300):.3f} ({args.threads} потока ORT)")

    audio, _ = sf.read(str(args.long), dtype="int16")
    audio = audio.astype(np.int32)
    results = {}
    for name, path in (("fp32", src), ("int8", str(args.out))):
        phrases, dt = transcribe(path, audio)
        results[name] = phrases
        print(f"\n{name}: офлайн {len(audio) / 8000:.1f} с за {dt:.1f} с")
        for ph in phrases:
            print(f"  [{ph.start_time:6.2f}-{ph.end_time:6.2f}] {ph.text}")
    same = [a.text for a in results["fp32"]] == [b.text for b in results["int8"]]
    print("\nтексты совпадают дословно" if same else "\nтексты РАЗЛИЧАЮТСЯ — качество int8 надо мерить по WER на наборе")


if __name__ == "__main__":
    main()
