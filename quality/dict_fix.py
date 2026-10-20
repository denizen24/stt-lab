#!/usr/bin/env python3
"""Личный словарь поверх чистовика, без второй модели.

Каскад с Parakeet переносит в текст то, что вторая модель услышала сама, а термин
пользователя она обычно тоже не знает («OpenSpec» → «OpenSpeak»). Здесь наоборот:
термин берётся из словаря пользователя, его латиница переводится в кириллический ключ
по звучанию (как WERx в metrics.py), и слово чистовика заменяется, если его ключ
начинается с ключа термина, а хвост не длиннее --tail букв (падежное окончание:
«опенспеку» → «OpenSpec»). Окончание при замене теряется — для поиска и итогов это
лучше, чем кириллица, для чтения спорно.

  python quality/dict_fix.py запись.mix onnx-gigaam-e2e-rnnt OpenSpec [термин …]
Пишет results/quality/<запись>.dictfix.txt и список замен в .dictfix.log.
"""
import argparse
import re
from pathlib import Path

from metrics import latin_key

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "quality"


def bare(word: str) -> str:
    return re.sub(r"[^а-яёa-z]", "", word.lower()).replace("ё", "е")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name")
    ap.add_argument("base")
    ap.add_argument("terms", nargs="+")
    ap.add_argument("--tail", type=int, default=3)
    args = ap.parse_args()

    keys = [(t, bare(latin_key(t))) for t in args.terms]
    words = (OUT / f"{args.name}.{args.base}.txt").read_text(encoding="utf-8").split()
    log = []
    for i, w in enumerate(words):
        b = bare(w)
        for term, k in keys:
            if b != bare(term) and b.startswith(k) and len(b) - len(k) <= args.tail:
                punct = re.search(r"[.,!?;:—…]+$", w)
                words[i] = term + (punct.group(0) if punct else "")
                log.append(f"{w!r} → {words[i]!r}")
                break
    (OUT / f"{args.name}.dictfix.txt").write_text(" ".join(words) + "\n", encoding="utf-8")
    (OUT / f"{args.name}.dictfix.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    print(f"{args.name}: замен {len(log)}; ключи: " + ", ".join(f"{t}={k}" for t, k in keys))


if __name__ == "__main__":
    main()
