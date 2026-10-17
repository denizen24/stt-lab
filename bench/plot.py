#!/usr/bin/env python3
"""График «потоки → p95 времени тика» по results/bench-<host>.csv.

Берёт прогоны с тегами curve-*, рисует по линии на конфигурацию и порог 300 мс:
выше него поток уже не успевает за реальным временем.

  python bench/plot.py --host node-b
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LABELS = {
    "curve-batch-ort3": "batch: один вызов на все потоки, 3 потока ORT",
    "curve-threads-w6": "threads: поток на участника, 6 рабочих",
    "curve-threads-w3": "threads: поток на участника, 3 рабочих",
    "curve-kenlm-w6": "threads, 6 рабочих, декодер KenLM",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True)
    ap.add_argument("--metric", default="tick_p95_ms", choices=["tick_p95_ms", "tick_p50_ms", "acoustic_p95_ms"])
    args = ap.parse_args()

    rows = list(csv.DictReader((ROOT / "results" / f"bench-{args.host}.csv").open(encoding="utf-8")))
    series: dict[str, dict[int, float]] = defaultdict(dict)
    env = {}
    for r in rows:
        if r["tag"].startswith("curve-"):
            series[r["tag"]][int(r["streams"])] = float(r[args.metric])  # последний прогон точки побеждает
            env = r
    if not series:
        raise SystemExit("нет прогонов с тегами curve-*")

    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150)
    for tag, pts in series.items():
        xs = sorted(pts)
        ax.plot(xs, [pts[x] for x in xs], marker="o", label=LABELS.get(tag, tag))
    ax.axhline(300, color="#a65d14", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1], 305, "300 мс — потолок реального времени", ha="right", va="bottom", fontsize=8, color="#a65d14")
    ax.set_xlabel("одновременных аудиопотоков (встреча = 2 потока)")
    ax.set_ylabel({"tick_p95_ms": "p95 времени тика, мс", "tick_p50_ms": "p50 времени тика, мс", "acoustic_p95_ms": "p95 акустики, мс"}[args.metric])
    ax.set_title(f"T-one на {args.host}: {env.get('audio', '')}, greedy (KenLM подписан отдельно)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = ROOT / "results" / f"curve-{args.host}-{args.metric}.png"
    fig.tight_layout()
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
