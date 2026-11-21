"""Proof of life for the watchdog (supervisor.py).

Every log line and every assistant step calls beat(). While the assistant is in the middle of a pass ("busy") the
program writes the time of the last sign of progress to data/heartbeat.json; if that stops for too long the
supervisor knows the program is stuck (not just idle) and restarts it."""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

_last = time.time()
_busy = False


def beat() -> None:
    global _last
    _last = time.time()


def set_busy(busy: bool) -> None:
    global _busy
    _busy = bool(busy)
    beat()


def snapshot() -> dict:
    return {"ts": _last, "busy": _busy, "pid": os.getpid(), "written": time.time()}


def write(path: Path) -> None:
    try:
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(snapshot()), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        pass


def start_writer(path: Path, every: float = 15.0) -> threading.Thread:
    path.parent.mkdir(parents=True, exist_ok=True)

    def loop() -> None:
        while True:
            write(path)
            time.sleep(every)
    t = threading.Thread(target=loop, daemon=True, name="featureclip-heartbeat")
    t.start()
    return t


def read(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
