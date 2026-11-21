"""Thin, Windows-safe wrappers around ffmpeg / ffprobe."""
from __future__ import annotations

import json
import re
import shutil
import threading
import subprocess
from pathlib import Path


class FFError(RuntimeError):
    pass


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


class Stopped(RuntimeError):
    """The owner pressed Stop: the running job is cancelled at the next step."""


_stop = threading.local()


def set_stop(ev) -> None:
    """Called by the job runner: the Stop button of the job sets this event."""
    _stop.ev = ev


def check_stop() -> None:
    ev = getattr(_stop, "ev", None)
    if ev is not None and ev.is_set():
        raise Stopped("stopped by you")


def run(cmd: list[str], cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    check_stop()
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ev = getattr(_stop, "ev", None)
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if ev is not None and ev.is_set():          # Stop pressed: kill the running ffmpeg at once
                proc.kill()
                proc.communicate()
                raise Stopped("stopped by you")
    done = subprocess.CompletedProcess(cmd, proc.returncode, out, err)
    if check and done.returncode != 0:
        tail = err.decode("utf-8", "replace")[-1500:]
        raise FFError(f"command failed ({done.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return done


def probe(path: str | Path) -> dict:
    """Return duration, width, height, fps, has_audio."""
    p = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    info = json.loads(p.stdout.decode("utf-8", "replace"))
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    dur = float(info.get("format", {}).get("duration") or (v or {}).get("duration") or 0)
    fps = 0.0
    if v and v.get("avg_frame_rate") and "/" in v["avg_frame_rate"]:
        n, d = v["avg_frame_rate"].split("/")
        fps = float(n) / float(d) if float(d) else 0.0
    return {
        "duration": dur,
        "width": int((v or {}).get("width") or 0),
        "height": int((v or {}).get("height") or 0),
        "fps": fps,
        "has_video": v is not None,
        "has_audio": a is not None,
    }


def av_start_offset(path: str | Path) -> float | None:
    """audio start_time minus video start_time in seconds (None if unknown). A big value means the recording itself is out of sync."""
    try:
        p = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_entries", "stream=codec_type,start_time", str(path)])
        st = {s["codec_type"]: float(s["start_time"]) for s in json.loads(p.stdout.decode("utf-8", "replace")).get("streams", [])
              if s.get("codec_type") in ("video", "audio") and s.get("start_time") not in (None, "N/A")}
        return st["audio"] - st["video"] if "audio" in st and "video" in st else None
    except Exception:
        return None


def extract_audio_wav(src: str | Path, dst: str | Path, start: float | None = None,
                      dur: float | None = None, sr: int = 16000) -> Path:
    cmd = ["ffmpeg", "-y", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(dst)]
    run(cmd)
    return Path(dst)


def grab_frame(src: str | Path, t: float, dst: str | Path, width: int = 640, crop=None) -> Path:
    """One frame at t. If t is at/after the end (or the seek fails) step back until a frame exists."""
    dst = Path(dst)
    for back in (0.0, 0.5, 1.5, 3.0, 6.0):
        dst.unlink(missing_ok=True)
        run([
            "ffmpeg", "-y", "-v", "error", "-ss", f"{max(t - back, 0):.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", (f"crop={crop[2]}:{crop[3]}:{crop[0]}:{crop[1]}," if crop else "") + f"scale={width}:-2",
            "-q:v", "4", str(dst),
        ], check=False)
        if dst.is_file() and dst.stat().st_size > 0:
            return dst
    raise RuntimeError(f"could not read a frame near {t:.1f}s of {src}")


_LUFS_RE = re.compile(r"I:\s+(-?\d+\.?\d*)\s+LUFS")


def measure_loudness(path: str | Path) -> float | None:
    p = run(["ffmpeg", "-nostats", "-v", "info", "-i", str(path), "-af", "ebur128=peak=true",
             "-f", "null", "-"], check=False)
    text = p.stderr.decode("utf-8", "replace")
    # the last "I:" in the summary block is the integrated loudness
    matches = _LUFS_RE.findall(text)
    return float(matches[-1]) if matches else None
