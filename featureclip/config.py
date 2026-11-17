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
        # Things worth looking for. The AI may not know a brand-new game, so tell it what players care about.
        # Edit freely in Settings; one idea per item.
        "hints": [
            "vehicle damage and deformation (dents, broken glass, falling parts, smoke)",
            "car and bike physics (suspension, drifting, tire smoke, water splashes, crashes)",
            "NPC behaviour (reactions to the player, phones and filming, fleeing, talking, fighting)",
            "police behaviour (wanted level, chases, roadblocks, arrests)",
            "animals and wildlife (alligators, birds, fish, dogs)",
            "weather, lighting, heat haze, reflections, water, wet surfaces",
            "crowds, traffic density, city life, beaches, boats, jet skis",
            "character animation details (footsteps in sand, clothes, hands, faces, swimming)",
            "weapons: reload, recoil, aiming, cover, shooting reactions, destruction of objects",
            "interactions: shops, phones, doors, interiors, mini-features, interface elements",
        ],
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
    "schedule": {
        "autotune": "off",                # off | suggest | auto  (learns the best publish times from YouTube views)
        "pool": ["17:00", "19:00", "21:00", "23:00", "01:00", "03:00", "05:00", "07:00"],
        # 3 publish slots per day in Pakistan time (PKT): UK/EU evening + US afternoon/evening. See schedule.py.
        "enabled": True,
        "slots": ["21:00", "01:00", "05:00"],
        "max_ahead_days": 3,
    },
    "assistant": {
        "enabled": False,
        "mode": "auto",                 # auto = approve + schedule by itself; ask = only review, I approve
        "review_strictness": "normal",  # normal | strict
        "max_rebuilds": 2,              # how many times it re-makes a clip that failed review
        "buffer_days": 2,               # notify me when less than this many days of clips are left
        "check_every_seconds": 60,
    },
    "notify": {"ntfy_topic": "", "telegram_chat_id": "", "discord_webhook": "", "email_to": ""},
    "branding": {
        # Shown under the heading on every Short (e.g. "@DetailRadar"). Empty = nothing is shown.
        "handle": "",
        "name": "",              # channel name written in the description ("Channel: ...")
    },
    "layout": {
        # fill  = gameplay fills the whole 9:16 screen (streamer, facecam, overlays cropped away)
        # hybrid = 4:5 gameplay block over a blurred copy of itself (more of the scene visible)
        "mode": "fill",
        "sample_frames": 24,
        "face_detect": True,
        "ai": True,                 # let the vision model (Gemini) find webcam / chat / text to cut away
        "facecam": "auto",          # auto | top-left | top-right | bottom-left | bottom-right | none
        "facecam_size": [0.27, 0.38],   # width, height of the corner area as a fraction of the frame
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
        "models": ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"],   # free tier: 500 requests/day each
        "gemini_base_url": "https://generativelanguage.googleapis.com",
        "openai_base_url": "https://api.groq.com/openai/v1",
        "openai_models": [],
        "vision": True,
        "frames": 8,             # frames the judge sees per window (more = it finds WHEN the feature happens)
        "frame_width": 512,
        "timeout": 90,
        "max_retries": 2,
        "daily_limit": {},      # e.g. {"gemini-2.5-flash": 20}
    },
    "judge": {
        "strictness": "normal",      # loose | normal | strict (sets the two numbers below)
        "min_confidence": None,
        "min_interest": None,
        "require_evidence": True,
        "min_script_match": 70,
        "allow_cutscenes": True,     # trailers / cinematic footage may be used (plot retelling is still not allowed)
        "max_clips_per_video": 8,
        "reject_categories": [
            "story_spoiler", "glitch_exploit", "mod_or_cheat",
            "menu_or_lobby", "nothing_notable",
        ],
    },
    "narration": {
        "engine": "gemini",     # gemini (expressive, free quota) | elevenlabs (optional) | kokoro (offline) | dummy (tests)
        "gemini_voice": "Fenrir",         # any prebuilt name or an Extended Voice Library id (Settings > Voice lab)
        "gemini_style": "a real person narrating gameplay live: warm, curious, genuinely excited at the big moment, natural pace",
        "gemini_models": ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview"],
        "eleven_voice_id": "",
        "eleven_model": "eleven_turbo_v2_5",
        "voice": "bm_fable",               # the offline voice (Kokoro), also used when the cloud voice fails
        "speed": 1.0,
        "lang": "en-gb",
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
        "music": True,                  # royalty-free background music (made by this program, no copyright)
        "music_db": -22,                # music level before it is ducked under the voice
        "music_dir": "music",           # optional: put your own licensed .mp3/.wav files here, they are used first
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
        "analytics": False,               # True = also read watch-time/retention (asks you to sign in again once)
        "manage_videos": False,           # True = lets safe mode pull scheduled videos back to private (sign in again)
    },
    "insights": {"min_samples": 3, "min_age_hours": 12, "refresh_hours": 3},
    "cleanup": {
        "enabled": True,
        "raw_hours": 12,                  # delete recordings / downloads this old once their clips are made
        "keep_output_days": 7,            # delete finished mp4 this many days after upload
        "min_free_gb": 4,                 # below this free space, clean up much more aggressively
    },
    "safe": {"auto": True, "fail_streak": 3, "reject_streak": 8},
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
