"""Watchdog: keeps the featureclip UI + assistant running.

    python -m featureclip.supervisor            (start_watchdog.bat does this)

It starts `python -m featureclip ui --no-browser` and restarts it when
  * the program exits or crashes,
  * the web page stops answering (several checks in a row), or
  * the assistant is in the middle of a pass but nothing has happened for `--stale-minutes` (a hung step).
Interrupted links are put back in the queue by the assistant itself after the restart, and it tells you why it was restarted.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from . import heartbeat


def should_restart(hb: dict | None, now: float, stale_minutes: float) -> str:
    """'' = fine, otherwise the reason. Only a BUSY program with no progress counts as stuck (idle is fine)."""
    if not hb or not hb.get("busy"):
        return ""
    idle = now - float(hb.get("ts") or now)
    if idle > stale_minutes * 60:
        return f"assistant made no progress for {idle / 60:.0f} minutes (stuck)"
    return ""


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.kill()
        proc.wait(timeout=20)
    except Exception:
        pass


class Supervisor:
    def __init__(self, cmd: list[str], data_dir: Path, port: int = 8765, stale_minutes: float = 20.0,
                 health_fails: int = 4, grace_seconds: float = 120.0, check_url: bool = True):
        self.cmd, self.data_dir, self.port = cmd, Path(data_dir), port
        self.stale_minutes, self.health_fails, self.grace, self.check_url = stale_minutes, health_fails, grace_seconds, check_url
        self.proc: subprocess.Popen | None = None
        self.started = 0.0
        self.fails = 0
        self.restarts: list[float] = []

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] watchdog: {msg}"
        print(line, flush=True)
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.data_dir / "watchdog.log", "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def start(self) -> None:
        self.proc = subprocess.Popen(self.cmd)
        self.started = time.time()
        self.fails = 0
        self.log(f"program started (pid {self.proc.pid})")

    def restart(self, reason: str) -> None:
        self.log(f"restarting: {reason}")
        if self.proc is not None and self.proc.poll() is None:
            _kill_tree(self.proc)
        try:
            (self.data_dir / "watchdog_last.json").write_text(json.dumps({"ts": time.time(), "reason": reason}), encoding="utf-8")
        except Exception:
            pass
        now = time.time()
        self.restarts = [t for t in self.restarts if now - t < 600] + [now]
        if len(self.restarts) > 5:                       # crash loop: do not hammer, wait and try again
            self.log("more than 5 restarts in 10 minutes: waiting 5 minutes before the next try")
            time.sleep(300)
        else:
            time.sleep(5)
        self.start()

    def healthy(self) -> bool:
        if not self.check_url:
            return True
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/state", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    def step(self, now: float | None = None) -> str:
        """One check. Returns the restart reason ('' when nothing was done)."""
        now = now or time.time()
        if self.proc is None:
            self.start()
            return ""
        rc = self.proc.poll()
        if rc is not None:
            reason = f"the program stopped (exit code {rc})"
            self.restart(reason)
            return reason
        if now - self.started < self.grace:
            return ""
        if self.check_url:
            self.fails = 0 if self.healthy() else self.fails + 1
            if self.fails >= self.health_fails:
                reason = "the web page does not answer"
                self.restart(reason)
                return reason
        why = should_restart(heartbeat.read(self.data_dir / "heartbeat.json"), now, self.stale_minutes)
        if why:
            self.restart(why)
            return why
        return ""

    def run_forever(self, every: float = 20.0) -> None:
        try:
            while True:
                self.step()
                time.sleep(every)
        except KeyboardInterrupt:
            self.log("stopped by you")
            if self.proc is not None and self.proc.poll() is None:
                _kill_tree(self.proc)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="keep featureclip running (restart on crash or hang)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--stale-minutes", type=float, default=20.0)
    a = ap.parse_args(argv)
    cmd = [sys.executable, "-m", "featureclip", "--root", a.root]
    if a.config:
        cmd += ["--config", a.config]
    cmd += ["ui", "--no-browser", "--port", str(a.port)]
    Supervisor(cmd, Path(a.root) / "data", a.port, a.stale_minutes).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
