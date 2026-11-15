"""Background music. No downloads, no copyright: the tracks are synthesised here (chords, bass, pluck, soft beat).
If the folder `music/` contains your own licensed .mp3/.wav files they are used instead."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np

SR = 44100

# (name, bpm, root MIDI note, chord roots as semitone offsets, minor?)
STYLES = [
    ("night-drive", 96, 57, [0, -4, -9, -2], True),      # Am  F   C   G   -> moody, cinematic
    ("neon-city", 104, 55, [0, -2, -4, -5], True),        # Gm  F   Eb  D
    ("rise", 100, 53, [0, 5, 3, 7], True),               # Fm  Bbm Ab  C
]


def _midi(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def _env(n: int, a: float, d: float, sr: int = SR) -> np.ndarray:
    t = np.arange(n) / sr
    return np.minimum(t / max(a, 1e-4), 1.0) * np.exp(-t / max(d, 1e-4))


def _lowpass(x: np.ndarray, k: int) -> np.ndarray:
    ker = np.ones(k) / k
    return np.convolve(x, ker, mode="same")


def render_track(style_idx: int = 0, seconds: float = 48.0, seed: int = 1) -> np.ndarray:
    name, bpm, root, prog, minor = STYLES[style_idx % len(STYLES)]
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar = beat * 4
    n_total = int(seconds * SR)
    out = np.zeros(n_total, dtype=np.float64)
    third = 3 if minor else 4
    bars = int(seconds / bar) + 1
    for b in range(bars):
        r = root + prog[b % len(prog)]
        t0 = int(b * bar * SR)
        if t0 >= n_total:
            break
        seg = min(int(bar * SR), n_total - t0)
        t = np.arange(seg) / SR
        # --- pad: three detuned saw-ish voices, slow attack, soft lowpass
        pad = np.zeros(seg)
        for iv in (0, third, 7):
            f = _midi(r + 12 + iv)
            for det in (-0.004, 0.0, 0.004):
                ph = 2 * np.pi * f * (1 + det) * t
                pad += np.sin(ph) + 0.4 * np.sin(2 * ph) + 0.2 * np.sin(3 * ph)
        pad = _lowpass(pad, 40) * np.minimum(t / 0.5, 1) * np.minimum((bar - t) / 0.4, 1) * 0.045
        out[t0:t0 + seg] += pad
        # --- bass: root, one note per beat pair
        for k in range(8):
            s = int((k * beat / 2) * SR)
            if s >= seg:
                break
            ln = min(int(beat / 2 * SR), seg - s)
            tt = np.arange(ln) / SR
            note = r - 12 + (0 if k % 4 != 3 else 7)
            out[t0 + s:t0 + s + ln] += 0.16 * np.sin(2 * np.pi * _midi(note) * tt) * _env(ln, 0.01, 0.35)
        # --- pluck arpeggio on 8ths
        arp = [0, 7, 12, third + 12, 7, 12, third + 12, 19]
        for k in range(8):
            s = int((k * beat / 2) * SR)
            if s >= seg:
                break
            ln = min(int(beat * SR), seg - s)
            tt = np.arange(ln) / SR
            f = _midi(r + 12 + arp[k])
            p = (np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(4 * np.pi * f * tt)) * _env(ln, 0.004, 0.22)
            vel = 0.075 if k % 2 == 0 else 0.05
            out[t0 + s:t0 + s + ln] += vel * p
        # --- drums: soft kick on 1 and 3, closed hat on offbeats, clap-ish noise on 2 and 4
        for k in range(4):
            s = int(k * beat * SR)
            if s >= seg:
                break
            if k in (0, 2):
                ln = min(int(0.28 * SR), seg - s)
                tt = np.arange(ln) / SR
                kick = np.sin(2 * np.pi * (48 + 90 * np.exp(-tt * 28)) * tt) * np.exp(-tt * 9)
                out[t0 + s:t0 + s + ln] += 0.30 * kick
            else:
                ln = min(int(0.12 * SR), seg - s)
                nz = _lowpass(rng.standard_normal(ln), 3) * _env(ln, 0.002, 0.05)
                out[t0 + s:t0 + s + ln] += 0.05 * nz
            h = s + int(beat / 2 * SR)
            ln = min(int(0.05 * SR), max(0, seg - h))
            if ln > 0:
                hat = rng.standard_normal(ln)
                hat = hat - _lowpass(hat, 6)
                out[t0 + h:t0 + h + ln] += 0.035 * hat * _env(ln, 0.001, 0.02)
    # gentle limiter + loop-friendly fades, normalise to about -18 dBFS RMS
    out = np.tanh(out * 1.4)
    fade = int(0.05 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    rms = np.sqrt(np.mean(out ** 2)) + 1e-9
    out = out * (10 ** (-18 / 20) / rms)
    return np.clip(out, -1, 1).astype(np.float32)


def pick_track(cfg, workdir: Path, key: str, root: Path) -> Path | None:
    """A music file for this clip: the user's own folder first, otherwise a generated track (cached)."""
    if not cfg.get("audio.music", True):
        return None
    mdir = Path(cfg.get("audio.music_dir", "music"))
    mdir = mdir if mdir.is_absolute() else root / mdir
    own = sorted([p for ext in ("*.mp3", "*.wav", "*.m4a", "*.ogg", "*.flac") for p in mdir.glob(ext)]) \
        if mdir.is_dir() else []
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16)
    if own:
        return random.Random(h).choice(own)
    import soundfile as sf
    idx = h % len(STYLES)
    gen = workdir / "music"
    gen.mkdir(parents=True, exist_ok=True)
    p = gen / f"{STYLES[idx][0]}.wav"
    if not p.is_file():
        sf.write(str(p), render_track(idx, 48.0, seed=idx + 7), SR)
    return p
