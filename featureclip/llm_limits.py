"""Free-tier rate limits per model (RPM = requests/minute, TPM = tokens/minute, RPD = requests/day).

Copied from the owner's Google AI Studio "Rate limits by model" page on 2026-11-16. Google changes these
often, so they are shown for reference only; the real numbers are always on aistudio.google.com."""
from __future__ import annotations

import re

# (display name, category, RPM, TPM, RPD)  -- None means unlimited
LIMITS: list[tuple[str, str, int | None, int | None, int | None]] = [
    ("Gemini 3.5 Flash Lite", "Text-out", 15, 250_000, 500),
    ("Gemini 3.1 Flash Lite", "Text-out", 15, 250_000, 500),
    ("Gemini 3 Flash", "Text-out", 5, 250_000, 20),
    ("Gemini 3.6 Flash", "Text-out", 5, 250_000, 20),
    ("Gemini 3.8 Flash", "Text-out", 5, 250_000, 20),
    ("Gemini 4 Hydrogen", "Text-out", 5, 250_000, 20),
    ("Gemini 2.5 Flash", "Text-out", 5, 250_000, 20),
    ("Gemini 2.5 Flash Lite", "Text-out", 10, 250_000, 20),
    ("Gemma 4 26B", "Other", 30, 16_000, 14_400),
    ("Gemma 4 31B", "Other", 30, 16_000, 14_400),
    ("Gemini Robotics ER 2 Preview", "Other", 5, 250_000, 20),
    ("Gemini 3.8 Flash TTS", "Multi-modal", 3, 10_000, 10),
    ("Gemini 3.8 Flash Lite TTS", "Multi-modal", 3, 10_000, 10),
    ("Gemini 2.5 Flash TTS", "Multi-modal", 3, 10_000, 10),
    ("Gemini 3.1 Flash TTS", "Multi-modal", 3, 10_000, 10),
    ("Gemini Embedding 1", "Other", 100, 30_000, 1_000),
    ("Gemini Embedding 2", "Other", 100, 30_000, 1_000),
]


def _key(name: str) -> str:
    return re.sub(r"[^a-z0-9.]+", "-", name.lower()).strip("-")


_BY_KEY = {_key(n): (rpm, tpm, rpd) for n, _c, rpm, tpm, rpd in LIMITS}


def limits_for(model_id: str) -> dict | None:
    """Look up a model id like 'gemini-3.5-flash-lite' (also tolerates '-preview' / '-001' suffixes)."""
    k = re.sub(r"-(preview|exp|latest)[-\w]*$|-\d{3}$", "", model_id.lower())
    k = {"gemma-4-31b-it": "gemma-4-31b", "gemma-4-26b-it": "gemma-4-26b"}.get(k, k)
    hit = _BY_KEY.get(k)
    return {"rpm": hit[0], "tpm": hit[1], "rpd": hit[2]} if hit else None


def peak_for(model_id: str) -> dict | None:
    return PEAKS.get(re.sub(r"-(preview|exp|latest)[-\w]*$|-\d{3}$", "", model_id.lower()))


def as_rows() -> list[dict]:
    return [{"model": n, "category": c, "rpm": rpm, "tpm": tpm, "rpd": rpd} for n, c, rpm, tpm, rpd in LIMITS]


# Peak usage per model over the last 28 days, copied from the same AI Studio page (16 Nov 2026).
PEAKS = {
    "gemini-3.5-flash-lite": {"rpm": 16, "tpm": 147_520, "rpd": 434},
    "gemini-3.6-flash": {"rpm": 2, "tpm": 10_060, "rpd": 23},
}
