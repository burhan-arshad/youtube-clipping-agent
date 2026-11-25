"""Get a video file onto disk (local file, or download with yt-dlp)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .ffutil import probe
from .sources import Source


def _kill_tree(proc) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()


class LiveStreamError(RuntimeError):
    """The link turned out to be a running live stream: it must be recorded live, not downloaded."""


def ingest(src: Source, workdir: Path, cfg=None, stop=None) -> Path:
    """Return the path of the video file for this source (download it if it is a link)."""
    if src.path:
        p = Path(src.path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"source '{src.id}': file not found: {p}")
        return p
    if not src.url:
        raise ValueError(f"source '{src.id}' has neither path nor url")
    dest_dir = workdir / "downloads" / src.id
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = [f for f in dest_dir.glob("video.*") if f.suffix not in (".part", ".ytdl")]
    if existing:
        return existing[0]
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd += [
        # <=1080p keeps the download and the CPU work reasonable on a laptop
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "-o", str(dest_dir / "video.%(ext)s"),
        "--no-playlist", "--retries", "10", "--sleep-requests", "1", "--newline",
        "--remote-components", "ejs:github",   # lets yt-dlp solve the YouTube JS challenge (needs deno/node)
    ]
    browser = (cfg.get("ingest.cookies_from_browser") if cfg else None)
    cfile = (cfg.get("ingest.cookies_file") if cfg else None)
    if cfile:     # cookies.txt exported from the browser (see README)
        cmd += ["--cookies", str(cfile)]
    elif browser:   # e.g. "chrome" / "edge" / "firefox": helps when YouTube answers 429 / "confirm you're not a bot"
        cmd += ["--cookies-from-browser", str(browser)]
    cmd += [src.url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    last_step = -1
    for line in proc.stdout:                       # show progress without flooding the log
        if stop is not None and stop.is_set():
            _kill_tree(proc)
            for part in dest_dir.glob("video.*"):
                part.unlink(missing_ok=True)       # never keep a half download
            raise RuntimeError("stopped by you")
        line = line.rstrip()
        if "yt_live_broadcast" in line or "[download] Downloading live" in line:
            _kill_tree(proc)                       # a running live stream never "finishes": stop at once
            for part in dest_dir.glob("video.*"):
                part.unlink(missing_ok=True)
            raise LiveStreamError("this link is a running LIVE stream")
        if line.startswith("[download]") and "%" in line:
            try:
                step = int(float(line.split("%")[0].split()[-1]) // 10)
            except ValueError:
                step = -2
            if step == last_step:
                continue
            last_step = step
        if line:
            print(line, flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(
            "yt-dlp could not download the video. Try: 1) pip install -U yt-dlp  2) wait a few minutes (YouTube "
            "429 = too many requests)  3) in config.yaml add  ingest: {cookies_from_browser: chrome}  "
            "(use the browser where you are logged in to YouTube)")
    files = [f for f in dest_dir.glob("video.*") if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("yt-dlp finished but no file was found")
    return files[0]


def _ytdlp_base(cfg) -> list[str]:
    exe = shutil.which("yt-dlp")
    cmd = [exe] if exe else [sys.executable, "-m", "yt_dlp"]
    cmd += ["--no-playlist", "--remote-components", "ejs:github"]
    cfile = cfg.get("ingest.cookies_file") if cfg else None
    browser = cfg.get("ingest.cookies_from_browser") if cfg else None
    if cfile:
        cmd += ["--cookies", str(cfile)]
    elif browser:
        cmd += ["--cookies-from-browser", str(browser)]
    return cmd


def remote_duration(url: str, cfg=None) -> float | None:
    """Length of an online video in seconds WITHOUT downloading it (None if unknown)."""
    try:
        r = subprocess.run(_ytdlp_base(cfg) + ["--skip-download", "--print", "duration", url],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=90)
        for line in reversed((r.stdout or "").splitlines()):
            try:
                return float(line.strip())
            except ValueError:
                continue
    except Exception:
        pass
    return None


def ingest_part(src: Source, workdir: Path, cfg, start: float, end: float, k: int, stop=None) -> Path:
    """Download only the seconds start..end of an online video (one 'part'), so a 6-hour stream never has to fit on disk."""
    dest = workdir / "downloads" / src.id / f"part_{k:03d}"
    dest.mkdir(parents=True, exist_ok=True)
    existing = [f for f in dest.glob("video.*") if f.suffix not in (".part", ".ytdl")]
    if existing:
        return existing[0]
    cmd = _ytdlp_base(cfg) + [
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b", "--merge-output-format", "mp4",
        "--download-sections", f"*{int(start)}-{int(end)}", "--retries", "10", "--newline",
        "-o", str(dest / "video.%(ext)s"), src.url]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    last = -1
    for line in proc.stdout:
        if stop is not None and stop.is_set():
            _kill_tree(proc)
            shutil.rmtree(dest, ignore_errors=True)
            raise RuntimeError("stopped by you")
        line = line.rstrip()
        if line.startswith("[download]") and "%" in line:
            try:
                step = int(float(line.split("%")[0].split()[-1]) // 20)
            except ValueError:
                step = -2
            if step == last:
                continue
            last = step
        if line:
            print(line, flush=True)
    proc.wait()
    files = [f for f in dest.glob("video.*") if f.suffix not in (".part", ".ytdl")]
    if proc.returncode != 0 or not files:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError("yt-dlp could not download this part of the video (pip install -U yt-dlp, or try again later)")
    return files[0]


def describe(path: Path) -> dict:
    info = probe(path)
    if not info["has_video"]:
        raise ValueError(f"{path} has no video stream")
    return info
