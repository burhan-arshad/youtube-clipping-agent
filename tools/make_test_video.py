"""Make a synthetic 'gameplay + streamer overlay' test video (no real footage needed).

Layout (1920x1080):
  * moving, colourful 'game world' filling the whole frame
  * a static HUD strip along the bottom and a fake 'webcam' box in the bottom-right corner
  * a blue-ish static border (like a stream overlay) around everything
Events (known ground truth, used by the tests):
  * t = 40 s and t = 85 s: loud burst + sudden scene change ("something cool")
Usage: python tools/make_test_video.py out.mp4 [duration_seconds]
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np


def make(out: str, duration: int = 120, w: int = 1920, h: int = 1080, fps: int = 30,
         events: tuple = (40, 85)) -> None:
    k = w / 1920.0                      # scale the overlay geometry for small test videos
    S = lambda v: max(1, int(round(v * k)))  # noqa: E731
    events = list(events)
    # --- audio: low noise, loud bursts at events
    sr = 44100
    t = np.arange(int(duration * sr)) / sr
    rng = np.random.default_rng(1)
    audio = 0.01 * rng.standard_normal(len(t))
    audio += 0.02 * np.sin(2 * np.pi * 220 * t)
    for e in events:
        m = (t >= e) & (t < e + 4)
        audio[m] += 0.5 * np.sin(2 * np.pi * 660 * t[m]) * np.exp(-(t[m] - e) * 0.6)
    audio16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    import wave
    wav_path = out + ".wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio16.tobytes())

    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-i", wav_path, "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "26", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    yy, xx = np.mgrid[0:h, 0:w]
    xx = xx / k
    yy = yy / k
    palette_a = np.array([30, 120, 60], dtype=np.float32)
    palette_b = np.array([200, 80, 20], dtype=np.float32)
    for f in range(duration * fps):
        sec = f / fps
        phase = sec * 0.9
        pal = palette_b if any(e <= sec < e + 6 for e in events) else palette_a
        base = (np.sin(xx / 90.0 + phase * 3) + np.cos(yy / 70.0 - phase * 2)) * 0.5 + 0.5
        frame = (base[..., None] * pal[None, None, :]).astype(np.float32)
        # a moving 'car' in the middle area
        cx = int(S(960 + 260 * np.sin(phase * 1.3)))
        cy = int(S(560 + 120 * np.cos(phase * 0.9)))
        frame[cy - S(40):cy + S(40), cx - S(70):cx + S(70)] = [240, 240, 240]
        if any(e <= sec < e + 3 for e in events):  # big visual change during an event
            frame[S(200):S(700), S(700):S(1250)] = [0, 200, 255]
        frame = frame.clip(0, 255).astype(np.uint8)
        # static overlays
        b = S(12)
        frame[:b, :] = [160, 90, 20]
        frame[-b:, :] = [160, 90, 20]
        frame[:, :b] = [160, 90, 20]
        frame[:, -b:] = [160, 90, 20]
        frame[h - S(90):h - b, b:w - b] = [30, 30, 30]           # HUD strip (static)
        # fake webcam box (moves a bit like a person), bottom-right
        frame[h - S(420):h - S(100), w - S(540):w - S(100)] = [90, 90, 90]
        px = int(w - S(320) + S(40) * np.sin(sec * 2.0))
        frame[h - S(330):h - S(190), px - S(50):px + S(50)] = [180, 150, 130]
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


if __name__ == "__main__":
    make(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 120)
