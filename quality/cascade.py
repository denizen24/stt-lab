#!/usr/bin/env python3
"""Каскад правок: чистовик GigaAM, из второй модели (Parakeet) берём только латиницу.

Схема из статьи pahaz (habr 1095512): обе модели распознают одни и те же куски
(run_onnxasr.py пишет .segments.jsonl с одинаковой нарезкой). В каждом куске слова
выравниваются; если на месте расхождения у второй модели слово на латинице и оно по
звучанию похоже на фрагмент GigaAM (похожесть ключей ≥ порога, по умолчанию 0,4),
фрагмент GigaAM заменяется. --any снимает требование латиницы (контроль: у автора
на встречах так становилось хуже, 14,5% → 17,4%).

  python quality/cascade.py запись.client onnx-gigaam-e2e-rnnt parakeet-v3-int8
Пишет results/quality/<запись>.cascade[-any][-dict].txt и список замен рядом в .log.
"""

from __future__ import annotations

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

from metrics import LATIN, to_sound
from score import normalize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"


def key(words: list[str]) -> str:
    return normalize(to_sound(" ".join(words))).replace(" ", "")


def bare(w: str) -> str:
    return re.sub(r"[^\wё-]", "", w.lower())


def merge(g_text: str, p_text: str, threshold: float, any_word: bool, vocab: set[str] | None = None) -> tuple[str, list[str]]:
    g, p = g_text.split(), p_text.split()
    sm = SequenceMatcher(a=[bare(w) for w in g], b=[bare(w) for w in p], autojunk=False)
    out, log = [], []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "replace" or not (any_word or any(LATIN.search(w) for w in p[j1:j2])):
            out += g[i1:i2]
            continue
        if any_word:  # контроль: весь расходящийся фрагмент целиком
            cands = [(i1, i2, p[j1:j2])]
        else:  # каждое латинское слово ищет свой фрагмент GigaAM (1–4 слова) внутри расхождения
            cands, cur = [], i1
            for w in p[j1:j2]:
                if not LATIN.search(w) or cur >= i2:
                    continue
                if vocab is not None and bare(w).strip("-.") not in vocab:
                    continue  # личный словарь: латиницу берём, только если она в нём есть
                a, b = max(((a, b) for a in range(cur, i2) for b in range(a + 1, min(a + 4, i2) + 1)),
                           key=lambda ab: SequenceMatcher(a=key(g[ab[0]:ab[1]]), b=key([w]), autojunk=False).ratio())
                cands.append((a, b, [w]))
                cur = b
        cur = i1
        for a, b, words in cands:
            sim = SequenceMatcher(a=key(g[a:b]), b=key(words), autojunk=False).ratio()
            if sim < threshold:
                continue
            repl = [re.sub(r"[.,!?…]+$", "", w) for w in words]
            mark = re.search(r"[.,!?…]+$", g[b - 1])  # пунктуацию оставляем от GigaAM
            repl[-1] += mark.group(0) if mark else ""
            out += g[cur:a] + repl
            cur = b
            log.append(f"{' '.join(g[a:b])!r} → {' '.join(words)!r} (похожесть {sim:.2f})")
        out += g[cur:i2]
    return " ".join(out), log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("base", help="метка чистовика, например onnx-gigaam-e2e-rnnt")
    ap.add_argument("second", help="метка второй модели, например parakeet-v3-int8")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--any", action="store_true", help="заменять любые слова, не только латиницу")
    ap.add_argument("--dict", type=Path, help="личный словарь (слово в строке): латиница из второй модели только из него. "
                    "Без словаря «нас» → NAS и «нет» → .NET проходят порог похожести")
    args = ap.parse_args()

    def segs(label: str) -> list[dict]:
        return [json.loads(ln) for ln in (OUT / f"{args.name}.{label}.segments.jsonl").read_text(encoding="utf-8").splitlines()]

    vocab = None
    if args.dict:
        vocab = {bare(w).strip("-.") for w in args.dict.read_text(encoding="utf-8").split() if not w.startswith("#")}
    base, second = segs(args.base), segs(args.second)
    if [(s["start"], s["end"]) for s in base] != [(s["start"], s["end"]) for s in second]:
        raise SystemExit("нарезка у моделей разная — прогоните обе через run_onnxasr.py на одном файле")
    texts, logs = [], []
    for b, s in zip(base, second):
        t, lg = merge(b["text"], s["text"], args.threshold, args.any, vocab)
        texts.append(t)
        logs += [f"[{b['start']:7.1f}] {x}" for x in lg]
    suffix = "cascade" + ("-any" if args.any else "") + ("-dict" if vocab is not None else "")
    (OUT / f"{args.name}.{suffix}.txt").write_text(" ".join(x for x in texts if x) + "\n", encoding="utf-8")
    (OUT / f"{args.name}.{suffix}.log").write_text("\n".join(logs) + "\n", encoding="utf-8")
    print(f"{args.name}: {len(logs)} замен ({args.base} + {args.second}, порог {args.threshold}{', любые слова' if args.any else ''})")
    for x in logs[:15]:
        print("  " + x)


if __name__ == "__main__":
    main()
