"""Thin, Windows-safe wrappers around ffmpeg / ffprobe."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


class FFError(RuntimeError):
    pass


def have(tool: str) -> bool:
    return shutil.which(tool) is not None


def run(cmd: list[str], cwd: str | Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if check and proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace")[-1500:]
        raise FFError(f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n{tail}")
    return proc


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


def grab_frame(src: str | Path, t: float, dst: str | Path, width: int = 640) -> Path:
    """One frame at t. If t is at/after the end (or the seek fails) step back until a frame exists."""
    dst = Path(dst)
    for back in (0.0, 0.5, 1.5, 3.0, 6.0):
        dst.unlink(missing_ok=True)
        run([
            "ffmpeg", "-y", "-v", "error", "-ss", f"{max(t - back, 0):.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "4", str(dst),
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
