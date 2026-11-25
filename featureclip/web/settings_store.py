"""What the Settings page can change, and how it is saved (config.yaml for options, .env for secrets)."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

# type: text | password | number | select | bool | folder | file
FIELDS: list[dict] = [
    # ---- AI
    {"group": "AI (Gemini)", "key": "env:GEMINI_API_KEY", "label": "Gemini API key", "type": "password",
     "help": "Free key from aistudio.google.com/apikey. It is stored only on this computer (.env file)."},
    {"group": "AI (Gemini)", "key": "env:GEMINI_MODELS", "label": "Gemini models (optional)", "type": "text",
     "help": "Leave empty for automatic selection, or list models separated by commas: gemini-3.5-flash,gemini-3.5-flash-lite"},
    {"group": "AI (Gemini)", "key": "judge.strictness", "label": "Clip selection strictness", "type": "select",
     "options": ["loose", "normal", "strict"],
     "help": "loose = more clips (some less accurate), strict = fewer but better clips"},
    {"group": "AI (Gemini)", "key": "judge.quality_floor", "label": "Skip boring clips: minimum interest (0-100)", "type": "number",
     "help": "Clips below this interest are skipped (the best 2 of each recording piece always stay). 55 = only clips with something to watch."},
    {"group": "AI (Gemini)", "key": "judge.max_clips_per_video", "label": "Max clips per video", "type": "number"},
    {"group": "AI (Gemini)", "key": "game.name", "label": "Game name", "type": "text",
     "help": "Used by the AI in hooks, titles and scripts (e.g. GTA 6). Change it to clip any other game."},
    {"group": "AI (Gemini)", "key": "game.topic_hint", "label": "What counts as a feature", "type": "text",
     "help": "One line that tells the AI what to look for, e.g. 'open-world action game; gameplay features, mechanics, details'"},
    {"group": "Channel branding", "key": "branding.name", "label": "Channel name (in the description)", "type": "text",
     "help": "Written as 'Channel: <name>' at the end of every description. No links."},
    {"group": "Channel branding", "key": "branding.handle", "label": "Channel handle / name on video", "type": "text",
     "help": "Shown under the heading on every Short so nobody can reuse your clips unnoticed (e.g. @DetailRadar). Leave empty to hide."},
    {"group": "Channel branding", "key": "channels.voiceover.name", "label": "Voice-over channel name", "type": "text",
     "help": "Name written as 'Channel: ...' in voice-over descriptions. Empty = the name above."},
    {"group": "Safety & disk", "key": "cleanup.max_work_gb", "label": "Work folder: maximum size (GB)", "type": "number",
     "help": "Recordings and temporary files never grow beyond this; old files are deleted first."},
    {"group": "Safety & disk", "key": "cleanup.stop_free_gb", "label": "Pause live recording below this free space (GB)", "type": "number"},
    {"group": "Autopilot", "key": "autopilot.enabled", "label": "Autopilot: one link per day from the links file", "type": "bool"},
    {"group": "Autopilot", "key": "autopilot.links_file", "label": "Links file", "type": "text",
     "help": "One video link per line. Optional: link | streamer name | facecam corner."},
    {"group": "Autopilot", "key": "autopilot.run_at", "label": "Start time each day (HH:MM)", "type": "text"},
    {"group": "Autopilot", "key": "autopilot.links_per_day", "label": "Links per day", "type": "number"},
    {"group": "Autopilot", "key": "autopilot.voiceover_clips", "label": "Voice-over clips per link", "type": "number"},
    {"group": "Autopilot", "key": "autopilot.warn_links_left", "label": "Remind me when this many links are left", "type": "number"},
    {"group": "Safety & disk", "key": "backup.enabled", "label": "Daily backup of database and settings", "type": "bool"},
    {"group": "Safety & disk", "key": "backup.keep", "label": "Backups to keep", "type": "number"},
    # ---- folders
    {"group": "Folders", "key": "paths.output", "label": "Output folder (finished clips)", "type": "folder",
     "help": "Final Shorts (mp4) are saved here"},
    {"group": "Folders", "key": "paths.workdir", "label": "Downloads / working folder", "type": "folder",
     "help": "Downloaded videos and temporary files live here (can use a lot of disk space)"},
    {"group": "Folders", "key": "audio.music_dir", "label": "Your own music folder (optional)", "type": "folder",
     "help": "Put licensed mp3/wav files here. If empty, the built-in royalty-free generated music is used"},
    # ---- video look
    {"group": "Video look", "key": "layout.mode", "label": "Layout", "type": "select",
     "options": ["hybrid", "fill"],
     "help": "hybrid = large gameplay block over a blurred background, fill = full vertical screen (narrow crop)"},
    {"group": "Video look", "key": "layout.ai", "label": "Detect webcam / chat / text overlays with AI", "type": "bool"},
    {"group": "Video look", "key": "layout.facecam", "label": "Facecam position", "type": "select",
     "options": ["auto", "top-left", "top-right", "bottom-left", "bottom-right", "none"],
     "help": "auto = AI finds it (recommended). Change only if the AI gets it wrong"},
    {"group": "Finding features", "key": "game.hints", "label": "What to look for in this game", "type": "list",
     "help": "Comma separated ideas the AI should watch for (it may not know a brand-new game). Edit to teach it."},
    {"group": "Video look", "key": "video.clip_min", "label": "Minimum clip length (seconds)", "type": "number"},
    {"group": "Video look", "key": "video.clip_max", "label": "Maximum clip length (seconds)", "type": "number"},
    {"group": "Video look", "key": "captions.enabled", "label": "Captions (on-screen words)", "type": "bool"},
    # ---- voice and music
    {"group": "Voice and music", "key": "narration.engine", "label": "Voice engine", "type": "select",
     "options": ["gemini", "elevenlabs", "kokoro"],
     "help": "gemini = emotional, uses your Gemini key (about 10 voice requests a day per model). kokoro = offline fallback."},
    {"group": "Voice and music", "key": "narration.gemini_voice", "label": "Gemini voice (name or library id)", "type": "text",
     "help": "Try voices in the Voice lab below. Library voices are used by far fewer channels."},
    {"group": "Voice and music", "key": "narration.gemini_style", "label": "How it should sound", "type": "text",
     "help": "Short delivery instruction, e.g. 'excited gaming commentator, natural, with real emotion'"},
    {"group": "Voice and music", "key": "env:ELEVENLABS_API_KEY", "label": "ElevenLabs API key (optional)", "type": "password",
     "help": "Free plan: 10k characters/month, non-commercial and may not allow API use. Only a bonus."},
    {"group": "Voice and music", "key": "narration.eleven_voice_id", "label": "ElevenLabs voice id (optional)", "type": "text"},
    {"group": "Voice and music", "key": "narration.voice", "label": "Offline fallback voice (Kokoro)", "type": "select",
     "options": ["bm_fable", "bm_george", "bm_lewis", "am_adam", "am_michael", "am_puck", "af_heart", "af_bella",
                 "af_nicole", "af_sarah", "bf_emma", "bf_isabella"],
     "help": "bm_/bf_ = British, am_/af_ = American; m = male, f = female"},
    {"group": "Voice and music", "key": "narration.punch_up", "label": "Script punch-up pass (funnier, more emotional)", "type": "bool",
     "help": "One extra AI request per clip. Switch off if the free quota runs out too fast."},
    {"group": "Voice and music", "key": "audio.music", "label": "Background music", "type": "bool"},
    {"group": "Voice and music", "key": "audio.music_db", "label": "Music volume (dB, -30 quiet ... -15 loud)",
     "type": "number"},
    # ---- download
    {"group": "Download (YouTube 429 / bot error)", "key": "ingest.cookies_file", "label": "cookies.txt file",
     "type": "file", "help": "Choose a cookies.txt file if YouTube says 'Sign in to confirm you are not a bot'"},
    {"group": "Download (YouTube 429 / bot error)", "key": "ingest.cookies_from_browser",
     "label": "Or read cookies from a browser", "type": "select", "options": ["", "chrome", "edge", "firefox"],
     "help": "The browser must be fully closed. The cookies.txt method is more reliable"},
    {"group": "Download (YouTube 429 / bot error)", "key": "ingest.part_minutes", "label": "Download long videos in parts of (minutes)",
     "type": "number", "help": "A 5-hour stream is downloaded 30 minutes at a time: clips are made, the part is deleted, then the next part comes. 0 = download everything at once"},
    {"group": "Download (YouTube 429 / bot error)", "key": "ingest.max_minutes", "label": "Use at most this many minutes of one video",
     "type": "number", "help": "0 = no limit (it stops anyway as soon as the wanted number of clips is made)"},
    # ---- upload
    {"group": "YouTube upload", "key": "upload.privacy", "label": "Upload privacy", "type": "select",
     "options": ["private", "unlisted", "public"], "help": "Keep it private at first"},
    {"group": "YouTube upload", "key": "upload.max_per_day", "label": "Max uploads per day",
     "type": "number"},
    {"group": "YouTube upload", "key": "upload.title_suffix", "label": "Title suffix", "type": "text"},
    {"group": "YouTube upload", "key": "upload.synthetic_disclosure",
     "label": "Mark as 'altered / synthetic content' (AI voice)", "type": "bool"},
    {"group": "YouTube upload", "key": "review.auto_approve", "label": "Auto-approve clips (skip manual review)",
     "type": "bool", "help": "Keep this off until you fully trust the output"},
    # ---- assistant / schedule / notifications
    {"group": "Assistant", "key": "assistant.enabled", "label": "Assistant switched on", "type": "bool",
     "help": "The assistant reviews clips, approves them and schedules uploads by itself."},
    {"group": "Assistant", "key": "assistant.mode", "label": "Assistant mode", "type": "select",
     "options": ["auto", "ask"], "help": "auto = approve and schedule good clips by itself; ask = it only reviews, you approve."},
    {"group": "Assistant", "key": "assistant.review_strictness", "label": "Review strictness", "type": "select",
     "options": ["normal", "strict"], "help": "strict = the quality score must be 80+ (normal = 65+)"},
    {"group": "Assistant", "key": "assistant.max_rebuilds", "label": "Re-make a failed clip (times)", "type": "number"},
    {"group": "Assistant", "key": "assistant.piece_minutes", "label": "Live recording piece length (minutes)", "type": "number",
     "help": "Smaller = the first clips appear sooner (5 is a good value). Applies when the assistant starts the next live recording."},
    {"group": "Assistant", "key": "learn.use_playbook", "label": "Let the writers learn from my YouTube results (channel playbook)", "type": "bool",
     "help": "Needs about 6 published Shorts with stats. For retention numbers also switch on upload.analytics and sign in again."},
    {"group": "Assistant", "key": "assistant.stock_target", "label": "Enough clips in stock (live pieces are skipped above this; 0 = never skip)", "type": "number"},
    {"group": "Assistant", "key": "assistant.weekly_report", "label": "Send a weekly report", "type": "bool"},
    {"group": "Assistant", "key": "assistant.buffer_days", "label": "Warn me when fewer than N days of clips are left", "type": "number"},
    {"group": "Schedule", "key": "schedule.enabled", "label": "Schedule uploads (publish at the slot times)", "type": "bool",
     "help": "Videos are uploaded early and YouTube publishes them at the slot time, even if this PC is off."},
    {"group": "Schedule", "key": "schedule.slots", "label": "Publish times in Pakistan time (PKT)", "type": "list",
     "help": "Comma separated, 24h. Default 21:00, 01:00, 05:00 = UK 4pm/8pm/12am, Europe +1h, US East 11am/3pm/7pm."},
    {"group": "Schedule", "key": "schedule.max_ahead_days", "label": "Schedule this many days ahead", "type": "number"},
    {"group": "Schedule", "key": "schedule.autotune", "label": "Learn the best publish times", "type": "select",
     "options": ["off", "suggest", "auto"],
     "help": "off = fixed slots. suggest = tells you a better slot. auto = swaps one slot per week when results prove it (needs 3+ videos per slot)."},
    {"group": "Schedule", "key": "schedule.pool", "label": "Times it may try (PKT)", "type": "list",
     "help": "Candidate publish times for learning, e.g. 17:00,19:00,21:00,23:00,01:00,03:00,05:00,07:00"},
    {"group": "Safety & disk", "key": "safe.auto", "label": "Switch safe mode on by itself when things go wrong", "type": "bool",
     "help": "After repeated upload failures or many rejected clips in a row, uploads pause and you get a notification."},
    {"group": "Safety & disk", "key": "upload.manage_videos", "label": "Let safe mode pull scheduled videos back to private", "type": "bool",
     "help": "Needs one extra YouTube permission: sign in again in the YouTube tab after switching this on."},
    {"group": "Safety & disk", "key": "upload.analytics", "label": "Read watch time / retention from YouTube", "type": "bool",
     "help": "Needs the YouTube Analytics API enabled in Google Cloud and a new sign-in. Without it you still get views, likes and comments."},
    {"group": "Safety & disk", "key": "cleanup.enabled", "label": "Delete old recordings automatically", "type": "bool"},
    {"group": "Safety & disk", "key": "cleanup.raw_hours", "label": "Delete recordings/downloads after (hours)", "type": "number"},
    {"group": "Safety & disk", "key": "cleanup.keep_output_days", "label": "Delete finished mp4 files after upload + (days)", "type": "number"},
    {"group": "Safety & disk", "key": "cleanup.min_free_gb", "label": "Clean harder when free disk is below (GB)", "type": "number"},
    {"group": "Notifications", "key": "notify.ntfy_topic", "label": "ntfy.sh topic (phone push)", "type": "text",
     "help": "Install the free ntfy app, subscribe to a hard-to-guess topic name, and type the same name here."},
    {"group": "Notifications", "key": "notify.email_to", "label": "Gmail address for email alerts", "type": "text",
     "help": "Alerts are sent from and to this Gmail. Needs the app password below."},
    {"group": "Notifications", "key": "env:NOTIFY_EMAIL_PASSWORD", "label": "Gmail app password", "type": "password",
     "help": "Google Account > Security > 2-Step Verification > App passwords. Not your normal password."},
    {"group": "Notifications", "key": "env:TELEGRAM_BOT_TOKEN", "label": "Telegram bot token (optional, blocked in Pakistan)", "type": "password",
     "help": "Create a bot with @BotFather and paste its token."},
    {"group": "Notifications", "key": "notify.telegram_chat_id", "label": "Telegram chat id (optional)", "type": "text"},
]

SECRET_ENV = {"env:GEMINI_API_KEY", "env:ELEVENLABS_API_KEY", "env:TELEGRAM_BOT_TOKEN", "env:NOTIFY_EMAIL_PASSWORD"}


def _get(d: dict, dotted: str, default=None):
    cur: Any = d
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def _set(d: dict, dotted: str, value) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        if not isinstance(cur.get(p), dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _unset(d: dict, dotted: str) -> None:
    parts = dotted.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.get(p)
        if not isinstance(cur, dict):
            return
    cur.pop(parts[-1], None)
    # drop empty parents
    if not cur and len(parts) > 1:
        _unset(d, ".".join(parts[:-1]))


def config_file(root: Path, config_path: str | None) -> Path:
    return Path(config_path) if config_path else root / "config.yaml"


def read_user_config(root: Path, config_path: str | None = None) -> dict:
    p = config_file(root, config_path)
    if p.is_file():
        return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return {}


def read_env(root: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    p = root / ".env"
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)", line)
            if m and not line.lstrip().startswith("#"):
                out[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return out


def write_env(root: Path, name: str, value: str) -> None:
    p = root / ".env"
    lines = p.read_text(encoding="utf-8").splitlines() if p.is_file() else []
    out, done = [], False
    for line in lines:
        if re.match(rf"\s*{re.escape(name)}\s*=", line):
            if not done and value:
                out.append(f"{name}={value}")
            done = True
        else:
            out.append(line)
    if not done and value:
        out.append(f"{name}={value}")
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    if value:
        os.environ[name] = value
    else:
        os.environ.pop(name, None)


def describe(cfg, root: Path) -> list[dict]:
    """Schema + current values for the Settings page (secrets are masked)."""
    env = read_env(root)
    out = []
    for f in FIELDS:
        item = dict(f)
        k = f["key"]
        if k.startswith("env:"):
            v = env.get(k[4:], os.environ.get(k[4:], ""))
            if k in SECRET_ENV:
                item["value"] = ""
                item["is_set"] = bool(v)
                item["masked"] = (v[:4] + "…" + v[-3:]) if len(v) > 10 else ("•" * len(v))
            else:
                item["value"] = v
        else:
            v = cfg.get(k)
            if isinstance(v, (list, tuple)):
                v = ", ".join(str(x) for x in v)
            item["value"] = "" if v is None else v
        out.append(item)
    return out


def save(root: Path, values: dict, config_path: str | None = None) -> list[str]:
    """Write changed settings. Returns the list of keys that were saved."""
    by_key = {f["key"]: f for f in FIELDS}
    user = read_user_config(root, config_path)
    saved = []
    for k, v in values.items():
        f = by_key.get(k)
        if not f:
            continue
        if k.startswith("env:"):
            name = k[4:]
            if k in SECRET_ENV and (v is None or str(v).strip() == ""):
                continue                     # empty secret box = keep the old key
            write_env(root, name, str(v or "").strip())
            saved.append(k)
            continue
        t = f["type"]
        if t == "bool":
            v = bool(v)
        elif t == "number":
            if v in ("", None):
                _unset(user, k)
                saved.append(k)
                continue
            fv = float(v)
            v = int(fv) if fv == int(fv) else fv
        elif t == "list":
            v = [x for x in re.split(r"[,;\s]+", str(v or "")) if x]
            if not v:
                _unset(user, k)
                saved.append(k)
                continue
        else:
            v = "" if v is None else str(v).strip()
            if v == "" and t in ("text", "folder", "file", "select"):
                _unset(user, k)
                saved.append(k)
                continue
        _set(user, k, v)
        saved.append(k)
    p = config_file(root, config_path)
    text = yaml.safe_dump(user, sort_keys=False, allow_unicode=True)
    if p.is_file():                                   # safe save: keep the last good copy, write to a temp file, then swap
        try:
            p.with_suffix(".yaml.bak").write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        except Exception:
            pass
    tmp = p.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return saved
