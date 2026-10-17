#!/usr/bin/env python3
"""Случайная выборка спорных мест для короткой проверки на слух вместо полной вычитки.

Берёт расхождения движков с эталоном (как diff_review.py), случайно выбирает N мест
на дорожку и печатает таблицу с таймкодом, контекстом, вариантами и пустой колонкой
«кто прав». Проверяющий слушает запись в этих местах и пишет в колонку: `эталон`,
имя движка или `никто`. Доля мест, где эталон ошибся, даёт поправку к WER всех движков.

  python quality/sample_review.py --record <запись> --roles host client \
      --engines tone-kenlm gigaam-v3_e2e_rnnt --per-role 20 --seed 1
  → results/quality/<запись>.sample.md
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diff_review import diffs, mmss, word_times  # noqa: E402
from score import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results" / "quality"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--record", required=True)
    ap.add_argument("--roles", nargs="+", default=["host", "client"])
    ap.add_argument("--engines", nargs="+", required=True)
    ap.add_argument("--per-role", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seg_file = Q / f"{args.record}.deepgram-ref.segments.txt"

    out = [f"# Выборка спорных мест: {args.record}", "",
           f"По {args.per_role} случайных мест на дорожку (seed {args.seed}). Слушать запись в указанное время и писать в колонку "
           "«кто прав»: `эталон`, имя движка или `никто`. Если слов в эталоне нет («∅»), а движок их слышит, и это голос "
           "второго участника — писать `пролезание`.", ""]
    for role in args.roles:
        name = f"{args.record}.{role}"
        ref = normalize((ROOT / "samples" / "ref" / f"{name}.txt").read_text(encoding="utf-8")).split()
        times = word_times(seg_file, role) if seg_file.exists() else []
        by_pos: dict[int, dict] = {}
        for eng in args.engines:
            p = Q / f"{name}.{eng}.txt"
            if not p.exists():
                continue
            for pos, ctx, was, now in diffs(ref, normalize(p.read_text(encoding="utf-8")).split()):
                d = by_pos.setdefault(pos, {"ctx": ctx, "was": was, "variants": {}})
                d["variants"][eng] = now
        picked = sorted(rng.sample(sorted(by_pos), min(args.per_role, len(by_pos))))
        out += [f"## Дорожка {role}: {len(picked)} из {len(by_pos)} спорных мест", "",
                "| № | время | эталон говорит | " + " | ".join(args.engines) + " | контекст | кто прав |",
                "|---|---|---|" + "---|" * len(args.engines) + "---|---|"]
        for i, pos in enumerate(picked, 1):
            d = by_pos[pos]
            t = mmss(times[min(pos, len(times) - 1)]) if times else "—"
            cells = " | ".join(d["variants"].get(e, "= эталон") for e in args.engines)
            out.append(f"| {i} | {t} | {d['was']} | {cells} | {d['ctx']} |  |")
        out.append("")
    path = Q / f"{args.record}.sample.md"
    path.write_text("\n".join(out), encoding="utf-8")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
