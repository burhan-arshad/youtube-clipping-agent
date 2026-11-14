"""Automatic quality gates on the final mp4. A clip that fails is never uploaded."""
from __future__ import annotations

import re
from pathlib import Path

from .ffutil import measure_loudness, probe, run

_SIL = re.compile(r"silence_duration:\s*([\d.]+)")
_BLK = re.compile(r"black_duration:([\d.]+)")
_FRZ = re.compile(r"freeze_duration:\s*([\d.]+)")


def _sum(rx: re.Pattern, text: str) -> float:
    return sum(float(m) for m in rx.findall(text))


def run_qa(path: str | Path, cfg) -> dict:
    info = probe(path)
    problems: list[str] = []
    dur = info["duration"]
    if dur < float(cfg.get("qa.min_duration", 8)):
        problems.append(f"too short ({dur:.1f}s)")
    if dur > float(cfg.get("qa.max_duration", 59)):
        problems.append(f"too long for a Short ({dur:.1f}s)")
    if (info["width"], info["height"]) != (int(cfg.get("video.width")), int(cfg.get("video.height"))):
        problems.append(f"wrong size {info['width']}x{info['height']}")
    if not info["has_audio"]:
        problems.append("no audio track")

    p = run(["ffmpeg", "-nostats", "-i", str(path),
             "-vf", "blackdetect=d=1.0:pix_th=0.10,freezedetect=n=-60dB:d=2.5",
             "-af", "silencedetect=n=-45dB:d=3.0", "-f", "null", "-"], check=False)
    log = p.stderr.decode("utf-8", "replace")
    black, frozen, silent = _sum(_BLK, log), _sum(_FRZ, log), _sum(_SIL, log)
    if dur > 0:
        if black / dur > 0.4:
            problems.append(f"mostly black ({black:.1f}s)")
        if frozen / dur > 0.6:
            problems.append(f"mostly frozen ({frozen:.1f}s)")
        if silent / dur > 0.7:
            problems.append(f"mostly silent ({silent:.1f}s)")
    lufs = measure_loudness(path)
    if lufs is not None and not (-20.0 <= lufs <= -9.0):
        problems.append(f"loudness {lufs:.1f} LUFS out of range")
    return {"ok": not problems, "problems": problems, "duration": round(dur, 2),
            "width": info["width"], "height": info["height"], "lufs": lufs,
            "black_s": round(black, 1), "frozen_s": round(frozen, 1), "silent_s": round(silent, 1)}
