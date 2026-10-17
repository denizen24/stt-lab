#!/usr/bin/env python3
"""Описание машины, на которой сняты замеры.

Пишет results/env-<hostname>.md: процессор и его векторные расширения, ядра,
память, ОС, версии Python, onnxruntime, numpy и T-one. Без этого файла цифры
прогона нельзя ни сравнить, ни опубликовать.
"""

from __future__ import annotations

import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def cpu_info() -> dict[str, str]:
    text = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace")
    model = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    flags = re.search(r"^flags\s*:\s*(.+)$", text, re.M)
    have = set(flags.group(1).split()) if flags else set()
    simd = [f for f in ("sse4_2", "avx", "avx2", "fma", "avx512f", "avx512_vnni", "amx_tile") if f in have]
    return {
        "model": model.group(1).strip() if model else platform.processor() or "?",
        "logical_cpus": str(os.cpu_count()),
        "simd": " ".join(simd) or "нет AVX",
    }


def mem_total_gb() -> str:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return f"{int(line.split()[1]) / 1024 / 1024:.1f}"
    return "?"


def pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 — пакета может не быть, это не ошибка стенда
        return "не установлен"


def ort_details() -> str:
    try:
        import onnxruntime as ort

        return f"{ort.__version__} · providers: {', '.join(ort.get_available_providers())}"
    except Exception:  # noqa: BLE001
        return "не установлен"


def os_release() -> str:
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return platform.platform()


def virtualization() -> str:
    try:
        return subprocess.run(["systemd-detect-virt"], capture_output=True, text=True, check=False).stdout.strip() or "?"
    except OSError:
        return "?"


def main() -> None:
    host = socket.gethostname()
    cpu = cpu_info()
    lines = [
        f"# Машина стенда: {host}",
        "",
        f"Снято: {datetime.now(timezone.utc).astimezone().isoformat(timespec='minutes')}",
        "",
        "| Параметр | Значение |",
        "|---|---|",
        f"| Процессор | {cpu['model']} |",
        f"| Логических ядер | {cpu['logical_cpus']} |",
        f"| Векторные расширения | {cpu['simd']} |",
        f"| Память | {mem_total_gb()} ГБ |",
        f"| Виртуализация | {virtualization()} |",
        f"| ОС | {os_release()}, ядро {platform.release()} |",
        f"| Python | {platform.python_version()} |",
        f"| onnxruntime | {ort_details()} |",
        f"| numpy | {pkg_version('numpy')} |",
        f"| T-one (пакет tone) | {pkg_version('tone')} |",
        f"| pyctcdecode / kenlm | {pkg_version('pyctcdecode')} / {pkg_version('kenlm')} |",
        "",
        "Отсутствие AVX2 и FMA означает, что onnxruntime работает по медленному пути: "
        "цифры такой машины — нижняя граница, а не ориентир для современного CPU.",
        "",
    ]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"env-{host}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    sys.stdout.write("\n".join(lines))
    sys.stdout.write(f"→ {out}\n")


if __name__ == "__main__":
    main()
