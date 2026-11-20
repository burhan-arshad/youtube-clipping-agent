"""Two kinds of clips, two YouTube channels.

voiceover  -> hybrid / fill layout with the AI voice-over      -> channel login in `upload.token` (token.json)
streamer   -> webcam on top, the streamer's own voice          -> channel login in `channels.streamer.token`

Each clip remembers what it is (judgement.streamer_clip), so the assistant always knows where it belongs."""
from __future__ import annotations

import json
from pathlib import Path

KEYS = ("voiceover", "streamer")
LABELS = {"voiceover": "Voice-over channel", "streamer": "Streamer-clips channel"}


def key_of_clip(c) -> str:
    try:
        j = json.loads(c["judgement"] or "{}")
    except Exception:
        j = {}
    return "streamer" if j.get("streamer_clip") else "voiceover"


def token_path(cfg, key: str) -> Path:
    raw = cfg.get("channels.streamer.token", "token_streamer.json") if key == "streamer" else cfg.get("upload.token", "token.json")
    p = Path(str(raw))
    return p if p.is_absolute() else cfg.root / p


def logged_in(cfg, key: str) -> bool:
    return token_path(cfg, key).is_file()


def overrides(cfg, key: str) -> dict:
    ov = {}
    if key == "streamer":
        ov["upload.token"] = str(cfg.get("channels.streamer.token", "token_streamer.json"))
    name = str(cfg.get(f"channels.{key}.name", "") or "").strip()
    if name:
        ov["branding.name"] = name
    return ov


def use(cfg, key: str):
    """with channels.use(cfg, 'streamer'): ... every YouTube call inside uses that channel's login."""
    return cfg.use(overrides(cfg, key))


def call(cfg, clip, fn, *args):
    """fn(cfg, *args) with the login of the channel this clip belongs to."""
    with use(cfg, key_of_clip(clip)):
        return fn(cfg, *args)


def pause_key(key: str) -> str:
    return "upload_pause_until" if key != "streamer" else "upload_pause_until:streamer"
