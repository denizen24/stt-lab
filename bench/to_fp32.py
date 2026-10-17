#!/usr/bin/env python3
"""Переводит акустическую модель T-one из float16 в float32 целиком.

Зачем: веса и почти все матричные операции модели хранятся в float16. На x86
без AVX512-FP16 у onnxruntime нет быстрого ядра для таких MatMul, и они
считаются медленным обобщённым путём. В float32 те же операции идут через
обычный sgemm. Качество не меняется: float32 шире float16.

Меняются: инициализаторы, константы в атрибутах, целевой тип у Cast,
типы входов и выходов графа (в том числе состояния стрима — поэтому
состояние такой модели тоже надо подавать как float32).

  python bench/to_fp32.py --out models/T-one-fp32/model.onnx
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import AttributeProto, TensorProto, numpy_helper

ROOT = Path(__file__).resolve().parent.parent


def to_fp32_tensor(t: onnx.TensorProto) -> bool:
    if t.data_type != TensorProto.FLOAT16:
        return False
    arr = numpy_helper.to_array(t).astype(np.float32)
    t.CopyFrom(numpy_helper.from_array(arr, t.name))
    return True


def convert(src: str, dst: Path) -> dict[str, int]:
    m = onnx.load(src)
    g = m.graph
    counts = {"инициализаторов": 0, "констант": 0, "Cast": 0, "входов/выходов": 0}
    for t in g.initializer:
        counts["инициализаторов"] += to_fp32_tensor(t)
    for node in g.node:
        for a in node.attribute:
            if node.op_type == "Cast" and a.name == "to" and a.i == TensorProto.FLOAT16:
                a.i = TensorProto.FLOAT
                counts["Cast"] += 1
            elif a.type == AttributeProto.TENSOR:
                counts["констант"] += to_fp32_tensor(a.t)
    for vi in list(g.input) + list(g.output):
        if vi.type.tensor_type.elem_type == TensorProto.FLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
            counts["входов/выходов"] += 1
    del g.value_info[:]  # старые типы промежуточных тензоров только мешают, выводим заново
    m = onnx.shape_inference.infer_shapes(m)
    onnx.checker.check_model(m)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None, help="исходная model.onnx; по умолчанию из Hugging Face")
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "T-one-fp32" / "model.onnx")
    args = ap.parse_args()
    src = args.src or hf_hub_download("t-tech/T-one", "model.onnx")
    t = time.perf_counter()
    counts = convert(src, args.out)
    print(f"{src} → {args.out} за {time.perf_counter() - t:.0f} с")
    print("переведено в float32:", ", ".join(f"{v} {k}" for k, v in counts.items()))
    print(f"размер {Path(src).stat().st_size / 2**20:.0f} → {args.out.stat().st_size / 2**20:.0f} МБ")


if __name__ == "__main__":
    main()
