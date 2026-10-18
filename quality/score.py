#!/usr/bin/env python3
"""WER и CER распознанного текста против эталона, одинаково для всех движков.

Перед сравнением оба текста приводятся к одному виду: строчные буквы, ё → е,
числа прописью (num2words, если установлен), только буквы и пробелы. Пунктуацию
и регистр мы не оцениваем — T-one их не ставит, а задача сравнения — слова.

  python quality/score.py --ref samples/ref/запись.host.txt --hyp results/quality/запись.host.tone-greedy.txt
  python quality/score.py --table results/quality   # все пары <имя>.<движок>.txt против samples/ref/<имя>.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jiwer

ROOT = Path(__file__).resolve().parent.parent
WORD = re.compile(r"[^0-9a-zа-я ]+")
DIGITS = re.compile(r"\d+")
ORDINAL = re.compile(r"(\d+)-([а-я]{1,3})\b")  # «1-ый», «2-ой», «0-ой»: так пишет порядковые Deepgram

try:
    from num2words import num2words
except ImportError:  # без num2words числа цифрами остаются как есть и считаются отдельными словами
    num2words = None


def _digits_to_words(m: re.Match) -> str:
    n = int(m.group(0))
    return num2words(n, lang="ru") if num2words and n < 10**12 else m.group(0)


def _ordinal_to_words(m: re.Match) -> str:
    """«1-ому» → «первому»: порядковое из num2words без окончания плюс наращение из текста."""
    n, suffix = int(m.group(1)), m.group(2)
    if not num2words or n >= 10**6:
        return m.group(0)
    word = "нулевой" if n == 0 else num2words(n, lang="ru", to="ordinal")
    return word[:-2] + suffix if len(suffix) >= 2 else word


def normalize(text: str) -> str:
    """Строчные, ё → е, числа прописью, только буквы, цифры и пробелы.

    Движки с нормализацией текста (Deepgram, GigaAM e2e) пишут «90», потоковые — «девяносто»;
    без перевода чисел в слова это считалось бы ошибкой распознавания.
    """
    text = text.lower().replace("ё", "е")
    text = ORDINAL.sub(_ordinal_to_words, text)
    text = DIGITS.sub(_digits_to_words, text)
    text = WORD.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def score(ref: str, hyp: str) -> dict[str, float]:
    r, h = normalize(ref), normalize(hyp)
    if not r:
        raise ValueError("пустой эталон")
    out = jiwer.process_words(r, h or "∅")
    return {
        "wer": round(out.wer * 100, 1),
        "cer": round(jiwer.cer(r, h or "∅") * 100, 1),
        "sub": out.substitutions,
        "del": out.deletions,
        "ins": out.insertions,
        "ref_words": len(r.split()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path)
    ap.add_argument("--hyp", type=Path)
    ap.add_argument("--table", type=Path, help="каталог с <имя>.<движок>.txt; эталоны ищутся в samples/ref/<имя>.txt")
    args = ap.parse_args()

    if args.table:
        rows = []
        # Идём от эталонов: имя записи может содержать точки (например, «…-abcd1234.host»),
        # как и имя движка («vosk-small-ru-0.22»), поэтому делить имя файла по точке нельзя.
        for ref in sorted((ROOT / "samples" / "ref").glob("*.txt")):
            name = ref.stem
            for hyp in sorted(args.table.glob(f"{name}.*.txt")):
                engine = hyp.name[len(name) + 1 : -len(".txt")]
                if engine.endswith(".phrases"):
                    continue
                s = score(ref.read_text(encoding="utf-8"), hyp.read_text(encoding="utf-8"))
                rows.append((name, engine, s))
        if not rows:
            raise SystemExit("нет пар гипотеза/эталон")
        print(f"{'запись':<28} {'движок':<22} {'WER %':>6} {'CER %':>6} {'зам':>4} {'проп':>4} {'вст':>4} {'слов':>5}")
        for name, engine, s in rows:
            print(f"{name:<28} {engine:<22} {s['wer']:6.1f} {s['cer']:6.1f} {s['sub']:4d} {s['del']:4d} {s['ins']:4d} {s['ref_words']:5d}")
        return

    if not (args.ref and args.hyp):
        ap.error("нужны --ref и --hyp либо --table")
    s = score(args.ref.read_text(encoding="utf-8"), args.hyp.read_text(encoding="utf-8"))
    print(f"WER {s['wer']}% · CER {s['cer']}% · замен {s['sub']}, пропусков {s['del']}, вставок {s['ins']} на {s['ref_words']} слов")


if __name__ == "__main__":
    main()
