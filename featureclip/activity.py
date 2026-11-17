"""Turns the raw log of the running job into something a person understands at a glance:
which step is running, what exactly it is doing, how far it is, and a short list of recent events."""
from __future__ import annotations

import re

STEPS = [("source", "Link"), ("record", "Record / download"), ("scan", "Scan video"), ("judge", "Pick moments"),
         ("build", "Make Short"), ("review", "Quality check"), ("publish", "Schedule")]
ORDER = {k: i for i, (k, _) in enumerate(STEPS)}

# (regex, stage, headline template using groups)   - first match on the NEWEST line wins
MARKERS = [
    (r"AI unavailable|all AI models are busy|out of quota", "judge", "Waiting for the AI free quota (recording continues)"),
    (r"waiting for stream", "record", "Waiting for the live stream to start"),
    (r"recording interrupted", "record", "Connection lost, reconnecting to the live stream"),
    (r"building clip (\d+) \((.*)\)", "build", "Making Short #{0}: {1}"),
    (r"-> (ready|dropped|error)", "build", "Short finished ({0}), moving on"),
    (r"clip \d+ @ \d+s: (ACCEPT|DROP\w*|REJECT\w*) '?(.*?)'? \(", "judge", "AI judged a moment: {0} {1}"),
    (r"judge done: (\d+) clips? kept", "judge", "AI picked {0} moment(s)"),
    (r"judging up to (\d+)", "judge", "AI is checking {0} possible moments"),
    (r"layout: (.*)", "scan", "Found the gameplay area"),
    (r"\[layout\]", "scan", "Working out where the gameplay is on screen"),
    (r"candidates: (\d+) windows", "scan", "{0} possible moments found, preparing the AI check"),
    (r"signals:", "scan", "Analysing sound and motion (CPU, can take a few minutes)"),
    (r"live piece (\S+): processing", "scan", "Live piece {0} is being processed"),
    (r"live done", "record", "Live finished"),
    (r"recording started", "record", "Recording the live stream"),
    (r"video (\d+): (.*?)\s+([\d.]+) min", "scan", "Video loaded ({2} min), starting the scan"),
    (r"upload", "publish", "Uploading"),
]
NOISE = ("Opening '", "[https", "[hls", "[youtube]", "[info]", "[in#", "[out#", "Press [q]", "Stream mapping", "Output #")


def _clock(t: str) -> str:
    m = re.match(r"(\d+):(\d+):(\d+)", t or "")
    if not m:
        return t or ""
    h, mi, s = map(int, m.groups())
    return f"{h}h {mi:02d}m" if h else f"{mi}m {s:02d}s"


def from_lines(lines: list[str]) -> dict:
    """lines: the newest log lines of the running job. Returns {stage, headline, progress, speed_warning, recent}."""
    tail = [l for l in lines[-500:] if l.strip()]
    res = {"stage": None, "headline": "", "progress": None, "warning": "", "recent": []}
    # 1) newest meaningful event
    for line in reversed(tail[-120:]):
        if line.startswith("frame=") or line.startswith("[download]"):
            continue
        for rx, stage, tpl in MARKERS:
            m = re.search(rx, line)
            if m:
                res["stage"], res["headline"] = stage, tpl.format(*m.groups())
                break
        if res["stage"]:
            break
    # 2) progress lines (downloads / yt-dlp's ffmpeg) are newer information when they come last
    last = tail[-1] if tail else ""
    for line in reversed(tail[-25:]):
        d = re.match(r"\[download\]\s+([\d.]+)%", line)
        if d:
            res["progress"] = float(d.group(1))
            if not res["stage"] or last.startswith("[download]") or last.startswith("frame="):
                res["stage"], res["headline"] = "record", f"Downloading video: {d.group(1)}%"
            break
    fr = next((l for l in reversed(tail[-25:]) if l.startswith("frame=")), None)
    if fr and (not res["stage"] or last.startswith("frame=")):
        t = re.search(r"time=(\S+)", fr)
        sp = re.search(r"speed=\s*([\d.]+)x", fr)
        res["stage"] = "record"
        res["headline"] = f"Downloading / recording: {_clock(t.group(1) if t else '')} of video so far"
        if sp and float(sp.group(1)) < 0.95:
            res["warning"] = (f"Speed is only {sp.group(1)}x, slower than real time. For a LIVE stream this means the "
                              "download can never catch up; the link is probably a live stream added as a normal video.")
    # 3) recent events for the feed
    for line in tail:
        if not re.match(r"\[\d\d:\d\d:\d\d\]", line) or any(n in line for n in NOISE):
            continue
        res["recent"].append({"t": line[1:9], "text": line[11:].strip()[:170]})
    res["recent"] = res["recent"][-12:]
    return res
