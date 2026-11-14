"""Cheap per-second signals (no AI): audio loudness bursts, visual motion, scene cuts.

Everything streams through ffmpeg pipes, so a 2-hour VOD never has to sit in RAM.
"""
from __future__ import annotations

import subprocess

import numpy as np


def _robust_z(x: np.ndarray, window: int = 300) -> np.ndarray:
    """z-score against a rolling median/MAD baseline (quiet and loud streams both work)."""
    n = len(x)
    if n == 0:
        return x
    z = np.zeros(n, dtype=np.float64)
    half = window // 2
    for i in range(0, n, 30):  # recompute baseline every 30 s, cheap and smooth enough
        lo, hi = max(0, i - half), min(n, i + half)
        seg = x[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med)) * 1.4826 + 1e-6
        z[i:i + 30] = (x[i:i + 30] - med) / mad
    return z


def audio_signal(video: str, duration: float) -> np.ndarray:
    """Per-second loudness (dB) -> robust z-score + 'burst' (jump above the previous 8 s)."""
    n = int(np.ceil(duration))
    sr = 16000
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vn", "-ac", "1", "-ar", str(sr),
           "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    db = []
    chunk_bytes = sr * 2  # one second
    while True:
        buf = proc.stdout.read(chunk_bytes)
        if not buf:
            break
        if len(buf) < 4:
            break
        a = np.frombuffer(buf[: len(buf) // 2 * 2], dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(a * a)) + 1e-9)
        db.append(20 * np.log10(rms))
    proc.stdout.close()
    proc.wait()
    arr = np.array(db[:n] + [db[-1] if db else -90.0] * max(0, n - len(db)), dtype=np.float64)
    if n == 0:
        return arr
    z = _robust_z(arr)
    burst = np.zeros(n)
    for i in range(n):
        prev = arr[max(0, i - 8):i]
        base = np.median(prev) if len(prev) else arr[i]
        burst[i] = arr[i] - base
    # a loud second after quiet is worth more than steady loud audio
    return np.clip(z, 0, 8) * 0.6 + np.clip(burst / 6.0, 0, 6) * 0.4


def visual_signals(video: str, duration: float, sample_fps: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Per-second (motion, cuts). motion = mean abs frame difference; cuts = big jumps."""
    n = int(np.ceil(duration))
    w, h = 96, 54
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-an",
           "-vf", f"fps={sample_fps},scale={w}:{h},format=gray",
           "-f", "rawvideo", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    fsize = w * h
    prev = None
    diffs: list[float] = []
    while True:
        buf = proc.stdout.read(fsize)
        if len(buf) < fsize:
            break
        cur = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(cur - prev))))
        prev = cur
    proc.stdout.close()
    proc.wait()
    d = np.array(diffs, dtype=np.float64)
    if len(d) == 0:
        return np.zeros(n), np.zeros(n)
    # robust threshold for "this is a cut / big visual change"
    med = np.median(d)
    mad = np.median(np.abs(d - med)) * 1.4826 + 1e-6
    cut_flags = (d > med + 6 * mad) & (d > 12.0)
    motion = np.zeros(n)
    cuts = np.zeros(n)
    for i, (val, flag) in enumerate(zip(d, cut_flags)):
        sec = min(n - 1, int((i + 1) / sample_fps))
        motion[sec] = max(motion[sec], val)
        if flag:
            cuts[sec] += 1
    return _robust_z(motion), np.clip(cuts, 0, 3)


def compute_signals(video: str, duration: float, sample_fps: int = 2) -> dict[str, np.ndarray]:
    audio = audio_signal(video, duration)
    motion, cuts = visual_signals(video, duration, sample_fps)
    n = max(len(audio), len(motion), len(cuts))

    def pad(a: np.ndarray) -> np.ndarray:
        return np.pad(a, (0, n - len(a)))

    return {"audio": pad(audio), "motion": pad(motion), "cuts": pad(cuts)}
