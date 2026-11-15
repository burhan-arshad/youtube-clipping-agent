"""Get a video file onto disk (local file, or download with yt-dlp)."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .ffutil import probe
from .sources import Source


def ingest(src: Source, workdir: Path, cfg=None) -> Path:
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
        "--no-playlist", "--retries", "10", "--sleep-requests", "1",
        "--remote-components", "ejs:github",   # lets yt-dlp solve the YouTube JS challenge (needs deno/node)
    ]
    browser = (cfg.get("ingest.cookies_from_browser") if cfg else None)
    cfile = (cfg.get("ingest.cookies_file") if cfg else None)
    if cfile:     # cookies.txt exported from the browser (see README)
        cmd += ["--cookies", str(cfile)]
    elif browser:   # e.g. "chrome" / "edge" / "firefox": helps when YouTube answers 429 / "confirm you're not a bot"
        cmd += ["--cookies-from-browser", str(browser)]
    cmd += [src.url]
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(
            "yt-dlp could not download the video. Try: 1) pip install -U yt-dlp  2) wait a few minutes (YouTube "
            "429 = too many requests)  3) in config.yaml add  ingest: {cookies_from_browser: chrome}  "
            "(use the browser where you are logged in to YouTube)")
    files = [f for f in dest_dir.glob("video.*") if f.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("yt-dlp finished but no file was found")
    return files[0]


def describe(path: Path) -> dict:
    info = probe(path)
    if not info["has_video"]:
        raise ValueError(f"{path} has no video stream")
    return info
