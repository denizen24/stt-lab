"""Эталон по дорожкам из общего эталона с сегментами.

Вход — JSON с сегментами `{start, end, speaker, text}` (секунды от начала записи) и две
дорожки записи `samples/<имя>.host.16k.wav`, `samples/<имя>.client.16k.wav`. Каждый сегмент
уходит в ту дорожку, где он громче: имена спикеров в групповых встречах часто расставлены
эвристикой, а громкость дорожки говорит, чей это микрофон. Сегменты, где обе дорожки
почти одинаковы (разница меньше --margin дБ), идут в обе и попадают в отчёт.

Спорные пометки эталона `[?вариант1/вариант2]` заменяются первым вариантом,
`[неразборчиво]` выбрасывается — число тех и других печатается.

    python quality/refs_from_segments.py эталон.json call-a [--margin 6] [--host-speaker Имя]
    → samples/ref/call-a.host.txt, samples/ref/call-a.client.txt
"""
import argparse
import audioop  # есть в 3.12; в 3.13 убран — тогда заменить на numpy
import json
import math
import re
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DISPUTED = re.compile(r"\[\?([^/\]]*)(?:/[^\]]*)?\]")
UNCLEAR = re.compile(r"\[неразборчиво\]", re.I)


def db(path: Path, start: float, end: float) -> float:
    with wave.open(str(path)) as w:
        rate = w.getframerate()
        w.setpos(max(0, min(int(start * rate), w.getnframes())))
        frames = w.readframes(max(0, int((end - start) * rate)))
    if not frames:
        return -120.0
    return 20 * math.log10(audioop.rms(frames, 2) + 1)


def clean(text: str) -> tuple:
    disputed = len(DISPUTED.findall(text))
    unclear = len(UNCLEAR.findall(text))
    text = UNCLEAR.sub(" ", DISPUTED.sub(lambda m: m.group(1), text))
    return re.sub(r"\s+", " ", text).strip(), disputed, unclear


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("segments", type=Path)
    ap.add_argument("name")
    ap.add_argument("--margin", type=float, default=6.0, help="дБ, ниже которых сегмент считается общим")
    ap.add_argument("--host-speaker", default="", help="имя ведущего в эталоне — только для сверки")
    args = ap.parse_args()

    segs = json.loads(args.segments.read_text(encoding="utf-8"))["segments"]
    host_wav = ROOT / "samples" / f"{args.name}.host.16k.wav"
    client_wav = ROOT / "samples" / f"{args.name}.client.16k.wav"
    out = {"host": [], "client": []}
    both = disputed = unclear = mismatch = 0
    for s in segs:
        text, d, u = clean(s["text"])
        disputed += d; unclear += u
        # Служебные пометки вроде «[СТЫК ЗАПИСЕЙ: …]» — сегмент нулевой длины, не речь.
        if not text or s["end"] <= s["start"]:
            continue
        h, c = db(host_wav, s["start"], s["end"]), db(client_wav, s["start"], s["end"])
        if abs(h - c) < args.margin:
            both += 1
            out["host"].append(text); out["client"].append(text)
            continue
        role = "host" if h > c else "client"
        out[role].append(text)
        if args.host_speaker and (s["speaker"] == args.host_speaker) != (role == "host"):
            mismatch += 1

    ref = ROOT / "samples" / "ref"
    for role, lines in out.items():
        (ref / f"{args.name}.{role}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    words = {r: sum(len(t.split()) for t in v) for r, v in out.items()}
    print(f"{args.name}: сегментов {len(segs)}, слов ведущий {words['host']} / клиент {words['client']}, "
          f"в обе дорожки {both}, спорных [?] {disputed}, [неразборчиво] {unclear}"
          + (f", расходится с именем спикера {mismatch}" if args.host_speaker else ""))


if __name__ == "__main__":
    main()
