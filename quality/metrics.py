#!/usr/bin/env python3
"""Метрики сверх WER: то, что средний WER прячет.

  WERx      — WER «по звучанию»: латиница в обоих текстах сводится к кириллическому
              фонетическому ключу (Cloud-MD → «клауд мд»), так что «клауд» против «Cloud»
              не считается ошибкой, а «SPN» против «VPN» — считается.
  числа     — доля слов-числительных эталона (после перевода цифр в слова), которые
              движок записал не так: «двадцать пять» против «двадцать» — ошибка.
  термины   — то же для слов из списка терминов (samples/ref/<запись>.terms.txt,
              по слову или фразе в строке; имена, методики, названия).
  пункт. F1 — совпадение знаков после слова (конец фразы . ! ? / запятая) с эталоном,
              только на словах, распознанных верно. Для движков без пунктуации пусто.

Режим --draft-final: черновик и чистовик против эталона, по каждому слову эталона —
верно ли оно в черновике и в чистовике. Даёт «исправлено / испорчено / не тронуто»,
то есть что увидит пользователь, когда чистовик заменит черновик.

  python quality/metrics.py --table results/quality
  python quality/metrics.py --draft-final запись.client tone-kenlm onnx-gigaam-e2e-rnnt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jiwer

from score import normalize

ROOT = Path(__file__).resolve().parent.parent
REF = ROOT / "samples" / "ref"
LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*")
# Корни русских числительных в любых падежах: один/одного/одной, двадцать/двадцати, сотни…
NUMERAL = re.compile(
    r"^(ноль|нол|нул|од(ин|на|но|ного|ной|ному|ним|ном|ну|ни|них)|дв(а|е|ух|ум|умя)|тр(и|ех|ем|емя)|четыр|пят|шест|сем|восем|девят|"
    r"десят|одиннадцат|двенадцат|тринадцат|четырнадцат|пятнадцат|шестнадцат|семнадцат|восемнадцат|девятнадцат|"
    r"двадцат|тридцат|сорок|пятьдесят|шестьдесят|семьдесят|восемьдесят|девяност|сто|ст(а|ам|ами|ах)$|двест|трист|четырест|"
    r"сот|тысяч|миллион|миллиард|полтор|перв|втор|трет|четверт|пол$)")
END = re.compile(r"[.!?…]")

# Латиница → кириллица по звучанию: сначала буквосочетания, потом буквы.
DIGRAPHS = [("sch", "ш"), ("tion", "шн"), ("sh", "ш"), ("ch", "ч"), ("th", "т"), ("ph", "ф"), ("ck", "к"), ("qu", "кв"),
            ("oo", "у"), ("ee", "и"), ("ea", "и"), ("ou", "ау"), ("ow", "оу"), ("ai", "эй"), ("ay", "эй"), ("oa", "о"),
            ("au", "о"), ("ie", "и"), ("ui", "у"), ("kn", "н"), ("wh", "в"), ("ng", "нг")]
LETTERS = dict(zip("abcdefghijklmnopqrstuvwxyz", ["а", "б", "к", "д", "е", "ф", "г", "х", "и", "дж", "к", "л", "м", "н", "о", "п", "к",
                                                 "р", "с", "т", "у", "в", "в", "кс", "и", "з"]))
SPELLED = dict(zip("abcdefghijklmnopqrstuvwxyz", ["эй", "би", "си", "ди", "и", "эф", "джи", "эйч", "ай", "джей", "кей", "эл", "эм",
                                                 "эн", "о", "пи", "кью", "ар", "эс", "ти", "ю", "ви", "дабл ю", "экс", "уай", "зет"]))


def latin_key(word: str) -> str:
    """Кириллический ключ латинского слова. Короткие сплошь согласные (VPN, MD) — побуквенно."""
    if "-" in word:  # Cloud-MD → «клауд эм ди»
        return " ".join(latin_key(p) for p in word.split("-") if p)
    w = word.lower()
    if len(w) <= 4 and not re.search(r"[aeiouy]", w):
        return " ".join(SPELLED[c] for c in w if c in SPELLED)
    for a, b in DIGRAPHS:
        w = w.replace(a, b)
    return "".join(LETTERS.get(c, c) for c in w)


def to_sound(text: str) -> str:
    return LATIN.sub(lambda m: latin_key(m.group(0)), text)


def ref_tokens(text: str) -> tuple[list[str], list[str]]:
    """Нормализованные слова и знак после каждого («end», «comma» или «»)."""
    words, marks = [], []
    for tok in text.split():
        sub = normalize(tok).split()
        if not sub:
            continue
        tail = re.sub(r"[\wё-]+", "", tok.lower())[-1:]
        mark = "end" if END.search(tail) else ("comma" if tail in {",", ";", ":"} else "")
        words += sub
        marks += [""] * (len(sub) - 1) + [mark]
    return words, marks


def correct_flags(ref_words: list[str], hyp_words: list[str]) -> tuple[list[bool], dict[int, int]]:
    """Для каждого слова эталона: распознано ли оно верно; и пара «индекс эталона → индекс гипотезы»."""
    out = jiwer.process_words(" ".join(ref_words), " ".join(hyp_words) or "∅")
    ok, pair = [False] * len(ref_words), {}
    for ch in out.alignments[0]:
        if ch.type == "equal":
            for k in range(ch.ref_end_idx - ch.ref_start_idx):
                ok[ch.ref_start_idx + k] = True
                pair[ch.ref_start_idx + k] = ch.hyp_start_idx + k
    return ok, pair


def class_error(ref_words: list[str], ok: list[bool], mask: list[bool]) -> tuple[int, int]:
    n = sum(mask)
    return sum(1 for m, o in zip(mask, ok) if m and not o), n


def terms_mask(ref_words: list[str], terms: list[str]) -> list[bool]:
    mask = [False] * len(ref_words)
    for t in terms:
        tw = normalize(to_sound(t)).split()
        for i in range(len(ref_words) - len(tw) + 1):
            if tw and ref_words[i : i + len(tw)] == tw:
                for k in range(len(tw)):
                    mask[i + k] = True
    return mask


def punct_f1(ref_marks: list[str], hyp_marks: list[str], pair: dict[int, int], kind: str) -> float | None:
    tp = fp = fn = 0
    for i, j in pair.items():
        r, h = ref_marks[i] == kind, hyp_marks[j] == kind
        tp += r and h
        fp += h and not r
        fn += r and not h
    return None if tp + fp + fn == 0 else round(200 * tp / (2 * tp + fp + fn), 1)


def evaluate(ref_text: str, hyp_text: str, terms: list[str]) -> dict:
    rw, rm = ref_tokens(ref_text)
    hw, hm = ref_tokens(hyp_text)
    ok, pair = correct_flags(rw, hw)
    wer = jiwer.wer(" ".join(rw), " ".join(hw) or "∅") * 100
    xr, xh = normalize(to_sound(ref_text)), normalize(to_sound(hyp_text))
    num_err, num_n = class_error(rw, ok, [bool(NUMERAL.match(w)) for w in rw])
    term_err, term_n = class_error(rw, ok, terms_mask(rw, terms)) if terms else (0, 0)
    has_punct = any(hm)
    return {
        "wer": round(wer, 1),
        "werx": round(jiwer.wer(xr, xh or "∅") * 100, 1),
        "num": f"{num_err}/{num_n}",
        "terms": f"{term_err}/{term_n}" if term_n else "—",
        "f1_end": punct_f1(rm, hm, pair, "end") if has_punct else None,
        "f1_comma": punct_f1(rm, hm, pair, "comma") if has_punct else None,
    }


def table(directory: Path) -> None:
    print(f"{'запись':<30} {'движок':<30} {'WER':>5} {'WERx':>5} {'числа':>7} {'термины':>8} {'F1 .?!':>6} {'F1 ,':>5}")
    for ref in sorted(REF.glob("*.txt")):
        if ref.name.endswith(".terms.txt"):
            continue
        name = ref.stem
        tf = REF / f"{name}.terms.txt"
        terms = [ln.strip() for ln in tf.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")] if tf.exists() else []
        for hyp in sorted(directory.glob(f"{name}.*.txt")):
            engine = hyp.name[len(name) + 1 : -len(".txt")]
            if engine.endswith((".phrases", ".segments")):
                continue
            m = evaluate(ref.read_text(encoding="utf-8"), hyp.read_text(encoding="utf-8"), terms)
            f = lambda v: "—" if v is None else f"{v:.0f}"  # noqa: E731
            print(f"{name[-30:]:<30} {engine:<30} {m['wer']:5.1f} {m['werx']:5.1f} {m['num']:>7} {m['terms']:>8} {f(m['f1_end']):>6} {f(m['f1_comma']):>5}")


def draft_final(name: str, draft: str, final: str, directory: Path) -> None:
    rw, _ = ref_tokens((REF / f"{name}.txt").read_text(encoding="utf-8"))
    dw, _ = ref_tokens((directory / f"{name}.{draft}.txt").read_text(encoding="utf-8"))
    fw, _ = ref_tokens((directory / f"{name}.{final}.txt").read_text(encoding="utf-8"))
    d_ok, _ = correct_flags(rw, dw)
    f_ok, _ = correct_flags(rw, fw)
    fixed = sum(1 for a, b in zip(d_ok, f_ok) if not a and b)
    broken = sum(1 for a, b in zip(d_ok, f_ok) if a and not b)
    both_ok = sum(1 for a, b in zip(d_ok, f_ok) if a and b)
    both_bad = len(rw) - fixed - broken - both_ok
    change = jiwer.wer(" ".join(fw), " ".join(dw) or "∅") * 100  # сколько правок пользователь увидит при замене
    n = len(rw)
    print(f"{name}: черновик {draft} → чистовик {final}, {n} слов эталона")
    print(f"  верно в обоих     {both_ok:5d} ({100 * both_ok / n:.1f}%)")
    print(f"  исправлено        {fixed:5d} ({100 * fixed / n:.1f}%)")
    print(f"  испорчено         {broken:5d} ({100 * broken / n:.1f}%)")
    print(f"  неверно в обоих   {both_bad:5d} ({100 * both_bad / n:.1f}%)")
    print(f"  расхождение черновика с чистовиком (WER черновика против чистовика): {change:.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", type=Path)
    ap.add_argument("--draft-final", nargs=3, metavar=("ЗАПИСЬ", "ЧЕРНОВИК", "ЧИСТОВИК"))
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "quality")
    args = ap.parse_args()
    if args.table:
        table(args.table)
    elif args.draft_final:
        draft_final(*args.draft_final, args.dir)
    else:
        ap.error("нужен --table или --draft-final")


if __name__ == "__main__":
    main()
