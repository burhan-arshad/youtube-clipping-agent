"""Configuration: defaults < config.yaml < environment variables (.env)."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "paths": {
        "workdir": "work",
        "output": "output",
        "db": "data/featureclip.db",
        "models": "~/.featureclip/models",
    },
    "game": {
        # Used in prompts and in titles. Change it and the tool works for any game.
        "name": "GTA 6",
        "topic_hint": "open-world action game; show gameplay features, mechanics, details",
    },
    "video": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "crf": 21,
        "preset": "veryfast",
        "clip_min": 15.0,
        "clip_max": 40.0,
    },
    "layout": {
        # fill  = gameplay fills the whole 9:16 screen (streamer, facecam, overlays cropped away)
        # hybrid = 4:5 gameplay block over a blurred copy of itself (more of the scene visible)
        "mode": "fill",
        "sample_frames": 24,
        "face_detect": True,
    },
    "detect": {
        "sample_fps": 2,
        "window": 30.0,         # candidate window length in seconds
        "pre_roll": 10.0,       # seconds before the peak
        "top_k": 30,            # candidates per video that go to Whisper + LLM
        "min_gap": 25.0,        # minimum distance between two candidate peaks
        "w_audio": 1.0,
        "w_motion": 0.8,
        "w_cut": 0.6,
        "skip_start": 30.0,     # ignore the first seconds (intros, "hello chat")
    },
    "whisper": {
        "model": "base.en",     # use "base" or "small" for non-English streamers
        "device": "cpu",
        "compute_type": "int8",
        "threads": 4,
        "language": "en",
    },
    "llm": {
        "provider": "gemini",   # gemini | openai (any OpenAI-compatible API: Groq, OpenRouter...)
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
        "gemini_base_url": "https://generativelanguage.googleapis.com",
        "openai_base_url": "https://api.groq.com/openai/v1",
        "openai_models": [],
        "vision": True,
        "frames": 4,
        "frame_width": 640,
        "timeout": 90,
        "max_retries": 2,
        "daily_limit": {},      # e.g. {"gemini-2.5-flash": 20}
    },
    "judge": {
        "min_confidence": 60,
        "min_interest": 55,
        "max_clips_per_video": 8,
        "reject_categories": [
            "cutscene", "story_spoiler", "glitch_exploit", "mod_or_cheat",
            "menu_or_lobby", "nothing_notable",
        ],
    },
    "narration": {
        "engine": "kokoro",     # kokoro | dummy (tests only)
        "voice": "af_heart",
        "speed": 1.0,
        "lang": "en-us",
        "words_per_second": 2.4,
        "lead_in": 0.4,         # seconds of silence before the voice starts
        "max_speedup": 1.15,
        "language": "English",
    },
    "audio": {
        # off  = remove the game/streamer audio completely (safest: no streamer voice, no game music)
        # duck = keep it very quiet under the voice-over
        "game_audio": "off",
        "duck_db": -22,
        "target_lufs": -14,
    },
    "captions": {
        "enabled": True,
        "font": "Arial",
        "size": 84,
        "words_per_chunk": 3,
        "margin_bottom": 330,
        "hook_seconds": 2.6,
        # auto = use Whisper to time the words of OUR voice-over (falls back to an estimate)
        "align": "auto",
    },
    "qa": {
        "min_duration": 8.0,
        "max_duration": 59.0,
    },
    "upload": {
        "privacy": "private",             # private | unlisted | public  (start with private!)
        "max_per_day": 3,                 # daily upload cap (slow, varied posting looks more human)
        "category_id": "20",              # 20 = Gaming
        "tags": ["gaming", "shorts"],
        "synthetic_disclosure": True,     # YouTube "altered or synthetic content" flag (AI voice)
        "made_for_kids": False,
        "title_suffix": " #Shorts",
        "description_footer": "",
        "client_secret": "client_secret.json",
        "token": "token.json",
    },
    "review": {
        # False = a human approves every clip before upload (recommended while launching).
        "auto_approve": False,
    },
}


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_dotenv(path: str | Path = ".env") -> None:
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class Config:
    def __init__(self, data: dict, root: Path):
        self.data = data
        self.root = root

    def __getitem__(self, key: str):
        return self.data[key]

    def get(self, dotted: str, default=None):
        cur: Any = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def path(self, dotted: str) -> Path:
        raw = str(self.get(dotted))
        p = Path(os.path.expanduser(raw))
        if not p.is_absolute():
            p = self.root / p
        return p

    @property
    def gemini_key(self) -> str:
        return os.environ.get("GEMINI_API_KEY", "")

    @property
    def openai_key(self) -> str:
        return os.environ.get("OPENAI_COMPAT_API_KEY", "")


def load_config(path: str | Path | None = None, root: str | Path | None = None) -> Config:
    root_p = Path(root) if root else Path.cwd()
    load_dotenv(root_p / ".env")
    cfg_path = Path(path) if path else root_p / "config.yaml"
    user: dict = {}
    if cfg_path.is_file():
        user = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    data = deep_merge(DEFAULTS, user)
    env_models = os.environ.get("GEMINI_MODELS")
    if env_models:
        data["llm"]["models"] = [m.strip() for m in env_models.split(",") if m.strip()]
    return Config(data, root_p)
