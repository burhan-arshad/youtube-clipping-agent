"""Live mode: record a live stream in ~10 minute pieces and push every finished piece through
scan -> judge -> build. Finished clips wait in the review queue (nothing is uploaded here).

  recorder (ffmpeg, stream copy, almost no CPU)  ->  work/live/<source>/seg_00001.ts, seg_00002.ts ...
  processor (this loop)                          ->  handles piece N as soon as piece N+1 has started
"""
from __future__ import annotations

import shutil
import subprocess
import sys
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


def start_recorder(inputs: list[str], out_dir: Path, seconds: int) -> subprocess.Popen:
    """inputs: stream URLs or one local file (read at real speed, for testing)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    for i in inputs:
        if Path(i).is_file():
            cmd += ["-re"]
            i = str(Path(i).resolve())
        cmd += ["-i", i]
    cmd += ["-map", "0:v:0", "-map", f"{len(inputs) - 1}:a:0?", "-c", "copy",
            "-f", "segment", "-segment_time", str(seconds), "-reset_timestamps", "1",
            "-segment_format", "mpegts", str(out_dir / "seg_%05d.ts")]
    return subprocess.Popen(cmd, cwd=str(out_dir))


def watch(pipe, source_id: str, url: str | None = None, segment_seconds: int = 600,
          local_test_file: str | None = None, poll: float = 5.0, max_height: int = 720,
          retry_seconds: int = 20, wait_minutes: int = 180) -> int:
    """Blocks until the stream ends (or Ctrl+C). Returns the number of pieces processed."""
    src = resolve_source(source_id, pipe.sources)
    source_id = src.id
    out_dir = pipe.work / "live" / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    url = url or src.url
    if not url and not local_test_file:
        raise ValueError("no stream URL: pass --url or set url in sources.yaml")

    rec, done, stop_at = None, set(), time.time() + wait_minutes * 60
    existing = {p.name for p in out_dir.glob("seg_*.ts")}
    handled = 0

    def process(seg: Path) -> None:
        nonlocal handled
        log(f"live piece {seg.name}: processing")
        try:
            mp4 = seg.with_suffix(".mp4")      # clean timestamps so seeking works everywhere
            r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(seg), "-c", "copy",
                                "-movflags", "+faststart", str(mp4)])
            pipe.run(source_id, path=mp4 if r.returncode == 0 and mp4.is_file() else seg)
        except Exception as e:           # one bad piece must not stop the live run
            log(f"  piece failed: {type(e).__name__}: {e}")
        done.add(seg.name)
        handled += 1

    try:
        while True:
            if rec is None:
                try:
                    inputs = [local_test_file] if local_test_file else stream_urls(url, max_height)
                    rec = start_recorder(inputs, out_dir, segment_seconds)
                    log("recording started" + (" (test file)" if local_test_file else ""))
                except Exception as e:
                    if time.time() > stop_at:
                        log("gave up waiting for the stream")
                        break
                    log(f"waiting for stream: {e}  (retry in {retry_seconds}s)")
                    time.sleep(retry_seconds)
                    continue
            segs = sorted(p for p in out_dir.glob("seg_*.ts") if p.name not in existing)
            finished = rec.poll() is not None
            ready = segs if finished else segs[:-1]      # the newest piece is still being written
            todo = [s for s in ready if s.name not in done]
            if todo:
                process(todo[0])
                continue
            if finished:
                if rec.returncode not in (0, None) and not local_test_file and time.time() < stop_at:
                    log("recording interrupted, trying to reconnect ...")
                    rec = None
                    time.sleep(retry_seconds)
                    continue
                break
            time.sleep(poll)
    except KeyboardInterrupt:
        log("Ctrl+C: stopping the recorder, processing the last piece")
        if rec and rec.poll() is None:
            rec.terminate()
            rec.wait(timeout=20)
        for s in sorted(p for p in out_dir.glob("seg_*.ts") if p.name not in existing and p.name not in done):
            process(s)
    finally:
        if rec and rec.poll() is None:
            rec.terminate()
    log(f"live done: {handled} piece(s) processed. Next: python -m featureclip list --status ready")
    return handled
