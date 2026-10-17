#!/usr/bin/env python3
"""Список расхождений между движками — чтобы вычитывать не всю запись, а спорные места.

Берёт эталон (или псевдоэталон) и гипотезы одной записи, выравнивает по словам
после той же нормализации, что в score.py, и печатает каждое расхождение в
контексте с таймкодом. Таймкод берётся из файла сегментов с метками времени
(например, расшифровка Deepgram: `[start-end] Роль: текст`), если он есть рядом.

Пометка ⚠ ставится, когда один и тот же вариант против эталона дали движки
РАЗНЫХ семейств (T-one и GigaAM, GigaAM и Vosk…): три варианта T-one — это
один движок, их согласие ничего не доказывает. Такие места переслушивать первыми:
возможно, ошибается эталон.

«∅» в эталоне и слова у движка — вставка: часто это голос второго участника,
долетевший в микрофон (пролезание канала), а не ошибка распознавания.

  python quality/diff_review.py --name <запись>.client --engines tone-service-greedy gigaam-v3_e2e_rnnt
  → results/quality/<name>.review.md
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score import normalize  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "results" / "quality"
SEG = re.compile(r"^\[\s*([\d.]+)-\s*([\d.]+)\]\s*([^:]+):\s*(.*)$")
ROLE_LABELS = {"host": "Ведущий", "client": "Клиент"}


def family(engine: str) -> str:
    return engine.split("-")[0]


def word_times(segments_file: Path, role: str) -> list[float]:
    """Время начала для каждого слова эталона (по сегментам нужной роли)."""
    times: list[float] = []
    for line in segments_file.read_text(encoding="utf-8").splitlines():
        m = SEG.match(line)
        if not m or m.group(3).strip() != ROLE_LABELS.get(role, role):
            continue
        start, end, words = float(m.group(1)), float(m.group(2)), normalize(m.group(4)).split()
        for i in range(len(words)):
            times.append(start + (end - start) * i / max(len(words), 1))
    return times


def mmss(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def diffs(ref: list[str], hyp: list[str], ctx: int = 4) -> list[tuple[int, str, str, str]]:
    out = []
    sm = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        before = " ".join(ref[max(0, i1 - ctx) : i1])
        after = " ".join(ref[i2 : i2 + ctx])
        out.append((i1, f"{before} […] {after}", " ".join(ref[i1:i2]) or "∅", " ".join(hyp[j1:j2]) or "∅"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="имя записи с ролью, как в samples/ref")
    ap.add_argument("--engines", nargs="+", required=True)
    ap.add_argument("--segments", type=Path, default=None, help="файл сегментов с таймкодами; по умолчанию <запись>.deepgram-ref.segments.txt")
    ap.add_argument("--max", type=int, default=600)
    args = ap.parse_args()

    rec, _, role = args.name.rpartition(".")
    ref_path = ROOT / "samples" / "ref" / f"{args.name}.txt"
    ref = normalize(ref_path.read_text(encoding="utf-8")).split()
    seg_file = args.segments or (Q / f"{rec}.deepgram-ref.segments.txt")
    times = word_times(seg_file, role) if seg_file.exists() else []
    if times and len(times) != len(ref):
        print(f"внимание: слов в сегментах {len(times)}, в эталоне {len(ref)} — таймкоды приблизительные")

    per_engine: dict[str, list[tuple[int, str, str, str]]] = {}
    for eng in args.engines:
        p = Q / f"{args.name}.{eng}.txt"
        if not p.exists():
            print(f"нет {p.name}, пропускаю")
            continue
        per_engine[eng] = diffs(ref, normalize(p.read_text(encoding="utf-8")).split())

    by_pos: dict[int, list[tuple[str, str, str, str]]] = {}
    for eng, items in per_engine.items():
        for pos, ctx, was, now in items:
            by_pos.setdefault(pos, []).append((eng, ctx, was, now))

    hot = inserts = 0
    body: list[str] = []
    for pos in sorted(by_pos)[: args.max]:
        items = by_pos[pos]
        fams: dict[str, set[str]] = {}
        for eng, _, _, now in items:
            fams.setdefault(now, set()).add(family(eng))
        agreed = max((len(f) for f in fams.values()), default=0) >= 2
        hot += agreed
        ctx, was = items[0][1], items[0][2]
        inserts += was == "∅"
        t = mmss(times[min(pos, len(times) - 1)]) if times else "—"
        mark = "⚠ " if agreed else ""
        kind = " (вставка)" if was == "∅" else ""
        body.append(f"- {mark}**{t}** слово {pos}{kind}: «{was}» · контекст: {ctx}")
        for eng, _, _, now in items:
            body.append(f"    - {eng}: «{now}»")

    lines = [
        f"# Расхождения: {args.name}", "",
        f"Эталон: `{ref_path.relative_to(ROOT)}`, {len(ref)} слов. Движки: {', '.join(per_engine)}.",
        f"Всего спорных мест: {len(by_pos)}, из них ⚠ (согласие разных семейств против эталона): {hot}, вставок: {inserts}.", "",
        "⚠ — переслушать первыми. «∅» + слова у движка — вставка: часто голос второго участника в микрофоне.", "",
        *body, "",
    ]
    out = Q / f"{args.name}.review.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{out.relative_to(ROOT)}: спорных мест {len(by_pos)}, ⚠ {hot}, вставок {inserts}")
    for eng, items in per_engine.items():
        print(f"  {eng}: расхождений {len(items)}")


if __name__ == "__main__":
    main()
