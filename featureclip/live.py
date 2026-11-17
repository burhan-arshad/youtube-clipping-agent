"""Live mode: record a live stream in short pieces and push every finished piece through
scan -> judge -> build. Finished clips wait in the review queue (nothing is uploaded here).

Two things run at the same time and never block each other:

  RECORDER thread  ffmpeg stream copy (almost no CPU). Keeps the stream recording, reconnects at once if it
                   drops, and puts every finished piece into a queue.
  WORKER (caller)  takes pieces from the queue one by one: scan -> judge -> build. If the AI free-tier limit is
                   hit, the Gemini client switches to the next model; if every model is exhausted the worker
                   waits and retries while the recorder keeps recording, so no piece is lost.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

from .pipeline import log
from .sources import resolve_source


def stream_urls(url: str, max_height: int = 720) -> list[str]:
    """Direct media URL(s) of a live stream, via yt-dlp (video and audio may be separate)."""
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd += ["-g", "--no-playlist", "-f", f"b[height<={max_height}]/bv*[height<={max_height}]+ba/b", url]
    p = subprocess.run(cmd, capture_output=True, text=True)
    urls = [u for u in p.stdout.split() if u.startswith("http")]
    if p.returncode != 0 or not urls:
        raise RuntimeError("yt-dlp could not open the live stream (not live yet? wrong URL?): "
                           + (p.stderr.strip().splitlines() or ["?"])[-1])
    return urls


def is_live(url: str) -> bool | None:
    """True/False from yt-dlp, None if it cannot tell."""
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    try:
        p = subprocess.run(cmd + ["--print", "%(live_status)s", "--no-playlist", url], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    out = p.stdout.strip().splitlines()
    if p.returncode == 0 and out:
        return out[-1].strip().lower() in ("is_live", "is_upcoming", "true")
    return None


def start_recorder(inputs: list[str], out_dir: Path, seconds: int, run: int = 0) -> subprocess.Popen:
    """inputs: stream URLs or one local file (read at real speed, for testing).
    `run` is part of the file name so a reconnect never overwrites earlier pieces."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    for i in inputs:
        if Path(i).is_file():
            cmd += ["-re"]
            i = str(Path(i).resolve())
        cmd += ["-i", i]
    cmd += ["-map", "0:v:0", "-map", f"{len(inputs) - 1}:a:0?", "-c", "copy",
            "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1",
            "-segment_format", "mpegts", str(out_dir / f"seg_r{run:02d}_%05d.ts")]
    return subprocess.Popen(cmd, cwd=str(out_dir))


def watch(pipe, source_id: str, url: str | None = None, segment_seconds: int = 600,
          local_test_file: str | None = None, poll: float = 5.0, max_height: int = 720,
          retry_seconds: int = 20, wait_minutes: int = 180, stop=None, ai_retry_seconds: int = 90,
          ai_max_retries: int = 40) -> int:
    """Blocks until the stream ends (or `stop` is set / Ctrl+C). Returns the number of pieces processed."""
    src = resolve_source(source_id, pipe.sources)
    source_id = src.id
    out_dir = pipe.work / "live" / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    url = url or src.url
    if not url and not local_test_file:
        raise ValueError("no stream URL: pass --url or set url in sources.yaml")

    stop_evt = stop if stop is not None else threading.Event()
    q: "queue.Queue[Path]" = queue.Queue()
    state = {"ended": False, "recorder": None}
    existing = {p.name for p in out_dir.glob("seg_*.ts")}
    stop_at = time.time() + wait_minutes * 60

    # ------------------------------------------------------------ recorder thread (no database access here)
    def recorder_loop() -> None:
        rec, run, seen = None, 0, set(existing)
        try:
            while not stop_evt.is_set():
                if rec is None:
                    try:
                        inputs = [local_test_file] if local_test_file else stream_urls(url, max_height)
                        rec = start_recorder(inputs, out_dir, segment_seconds, run)
                        state["recorder"] = rec
                        run += 1
                        log("recording started" + (" (test file)" if local_test_file else ""))
                    except Exception as e:
                        if time.time() > stop_at:
                            log("gave up waiting for the stream")
                            return
                        log(f"waiting for stream: {e}  (retry in {retry_seconds}s)")
                        stop_evt.wait(retry_seconds)
                        continue
                segs = sorted(p for p in out_dir.glob("seg_*.ts") if p.name not in seen)
                finished = rec.poll() is not None
                for s in (segs if finished else segs[:-1]):      # the newest piece is still being written
                    seen.add(s.name)
                    q.put(s)
                if finished:
                    # ffmpeg ending cleanly can also mean an expired stream link while the stream is still live
                    clean_but_live = (rec.returncode == 0 and not local_test_file and not stop_evt.is_set()
                                      and time.time() < stop_at and is_live(url) is True)
                    if (rec.returncode not in (0, None) and not local_test_file and time.time() < stop_at) or clean_but_live:
                        log("recording interrupted, reconnecting ...")
                        rec = None
                        stop_evt.wait(min(retry_seconds, 5))     # reconnect fast: the worker may be busy for minutes
                        continue
                    return
                stop_evt.wait(poll)
        finally:
            if rec is not None and rec.poll() is None:
                rec.terminate()
                try:
                    rec.wait(timeout=20)
                except Exception:
                    rec.kill()
            for s in sorted(p for p in out_dir.glob("seg_*.ts") if p.name not in seen):
                seen.add(s.name)                                   # last (partial) piece after Stop
                q.put(s)
            state["ended"] = True

    # ------------------------------------------------------------ worker (runs in the calling thread)
    handled = 0

    def wait_interruptible(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            time.sleep(1.0)

    def drain(vid: int) -> None:
        """judge + build until nothing is left. AI limit exhausted -> wait and try again, never skip."""
        from .llm import LLMError
        st = pipe.store
        for attempt in range(ai_max_retries + 1):
            try:
                if st.clips(vid, "candidate"):
                    pipe.judge(vid)
                if st.clips(vid, "judged"):
                    pipe.build(vid)
            except LLMError as e:
                log(f"  AI unavailable: {e}")
            left = len(st.clips(vid, "candidate")) + len(st.clips(vid, "judged"))
            if not left:
                return
            if attempt >= ai_max_retries:
                log(f"  giving up on {left} window(s) of this piece after {ai_max_retries} retries")
                return
            log(f"  all AI models are busy / out of quota: {left} window(s) waiting, retry in {ai_retry_seconds}s "
                f"(recording continues, {q.qsize()} more piece(s) queued)")
            wait_interruptible(ai_retry_seconds)

    def process(seg: Path) -> None:
        nonlocal handled
        log(f"live piece {seg.name}: processing ({q.qsize()} more waiting)")
        try:
            mp4 = seg.with_suffix(".mp4")      # clean timestamps so seeking works everywhere
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(seg), "-c", "copy",
                                "-movflags", "+faststart", str(mp4)])
            vid = pipe.scan(source_id, path=mp4 if r.returncode == 0 and mp4.is_file() else seg)
            drain(vid)
        except Exception as e:           # one bad piece must not stop the live run
            log(f"  piece failed: {type(e).__name__}: {e}")
        handled += 1

    t = threading.Thread(target=recorder_loop, name="live-recorder", daemon=True)
    t.start()
    try:
        while True:
            try:
                seg = q.get(timeout=2.0)
            except queue.Empty:
                if state["ended"] and q.empty():
                    break
                continue
            process(seg)
    except KeyboardInterrupt:
        log("Ctrl+C: stopping the recorder, then processing what is already recorded (Ctrl+C again = quit now)")
        stop_evt.set()
        t.join(timeout=30)
        while not q.empty():
            process(q.get())
    finally:
        stop_evt.set()
        rec = state["recorder"]
        if rec is not None and rec.poll() is None:
            rec.terminate()
    log(f"live done: {handled} piece(s) processed.")
    return handled
