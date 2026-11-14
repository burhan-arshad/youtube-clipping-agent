"""Turn per-second signals into a ranked list of candidate windows."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Candidate:
    start: float
    end: float
    peak: float       # second of the strongest moment
    score: float


def _smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if len(x) < k:
        return x
    return np.convolve(x, np.ones(k) / k, mode="same")


def combine(signals: dict[str, np.ndarray], w_audio: float, w_motion: float, w_cut: float) -> np.ndarray:
    audio = np.clip(signals["audio"], 0, 8)
    motion = np.clip(signals["motion"], 0, 8)
    cuts = signals["cuts"]
    return _smooth(w_audio * audio + w_motion * motion + w_cut * cuts, 3)


def pick_candidates(
    combined: np.ndarray,
    duration: float,
    window: float = 30.0,
    pre_roll: float = 10.0,
    top_k: int = 30,
    min_gap: float = 25.0,
    skip_start: float = 30.0,
    min_score: float = 0.5,
) -> list[Candidate]:
    """Greedy non-maximum suppression: best peak first, then peaks at least min_gap away."""
    n = len(combined)
    order = np.argsort(-combined)
    chosen: list[int] = []
    for idx in order:
        t = int(idx)
        if t < skip_start or t >= duration - 3:
            continue
        if combined[t] < min_score:
            break
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
        if len(chosen) >= top_k:
            break
    out: list[Candidate] = []
    for t in chosen:
        start = max(0.0, t - pre_roll)
        end = min(duration, start + window)
        start = max(0.0, end - window)  # keep full window near the end of the video
        seg = combined[int(start):int(end)]
        score = float(combined[t] + 0.3 * (seg.mean() if len(seg) else 0.0))
        out.append(Candidate(start=round(start, 2), end=round(end, 2), peak=float(t), score=round(score, 3)))
    out.sort(key=lambda c: -c.score)
    return out
