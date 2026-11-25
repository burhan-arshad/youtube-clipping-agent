# featureclip

**A free, local-first tool that turns streamers' gameplay videos into YouTube Shorts and publishes them for you.**

It finds the clip-worthy moments, builds vertical (9:16) Shorts, checks their quality, writes the titles and
descriptions, and schedules them on YouTube. It can run on autopilot: put video links in a text file and it makes and
schedules clips every day, learns from the results, restarts itself after a crash or reboot, and tells you when it needs you.

> Built for a normal Windows laptop: **8 GB RAM, no GPU**, no paid services. Speech-to-text, voice and video editing
> run locally on the CPU. The only online AI is Google's **free** Gemini API.

---

## Contents
1. [What it makes](#what-it-makes)
2. [How it works](#how-it-works)
3. [Features](#features)
4. [Install (Windows)](#install-windows)
5. [Quick start](#quick-start)
6. [Autopilot](#autopilot)
7. [The web UI (tabs)](#the-web-ui-tabs)
8. [Upgrading from the old two-channel version](#upgrading-from-the-old-two-channel-version)
9. [Configuration](#configuration)
10. [Command line](#command-line)
11. [Reliability and safety](#reliability-and-safety)
12. [Data, backups and disk space](#data-backups-and-disk-space)
13. [Troubleshooting](#troubleshooting)
14. [Tests](#tests)
15. [Project layout](#project-layout)
16. [Known limits](#known-limits)
17. [Copyright and platform rules](#copyright-and-platform-rules-read-this)
18. [License](#license)

---

## What it makes

| Clip type | Look | Audio | Goes to |
|---|---|---|---|
| **Voice-over** | gameplay filling the 9:16 screen (or a 4:5 block over a blurred copy), burned-in captions | AI-written short script, spoken by a local voice (Kokoro) or Gemini voice, soft background music | your YouTube channel |

(The earlier "streamer reaction" clip type and second channel were discontinued: the tool now has one job and one channel.)

## How it works

```
 video link / links.txt / live stream
        │
        ▼
  ┌───────────┐   audio loudness bursts, motion, scene cuts (cheap, no AI)
  │   scan    │──► ranked candidate windows
  └─────┬─────┘
        ▼
  ┌───────────┐   Whisper (local) hears the speech, Gemini looks at frames
  │   judge   │──► "is this clip-worthy?" + category
  └─────┬─────┘
        ▼
  ┌───────────┐   find gameplay area / webcam, crop 9:16, script + voice (voice-over mode),
  │   build   │   captions, loudness, music, one ffmpeg render, automatic QA gates
  └─────┬─────┘
        ▼
  ┌───────────┐   AI reviewer re-checks title, description, thumbnail and the video itself;
  │  review   │   good clips are approved automatically (or wait for you in "ask" mode)
  └─────┬─────┘
        ▼
  ┌───────────┐   next free publish slot of that clip's channel (PKT slots), resumable upload,
  │  publish  │   daily caps, YouTube limit handling
  └─────┬─────┘
        ▼
  stats ──► learning (per-channel "playbook") ──► better prompts next time
```

Every stage is resumable: signals, transcripts and finished clips are cached, so a crash or a restart never loses work.

## Features

**Making clips**
- Signal scan (audio, motion, scene cuts), Whisper transcription, Gemini vision judge with hard rules
  (cutscenes, story spoilers, glitches/exploits, mods/cheats and menus are always rejected).
- Automatic layout detection (gameplay area, webcam corner, chat/overlays) with a visual preview and manual crop.
- Voice-over script writer with a code-side checker (numbers/dates the AI was not given are removed).
- Local voices (Kokoro), optional Gemini voices, synthesised copyright-free background music, animated captions.
- Live mode: records a live stream in short pieces and clips every finished piece.
- QA gates on the final mp4 (a clip that fails is never uploaded).

**Publishing**
- One YouTube channel: login, publish slots, daily cap and limit handling.
- Schedule on the next free slot, or **upload now** (public immediately), for all approved clips.
- Publishing slots are written in Pakistan time (PKT) and converted for YouTube; launch mode for stream days.
- Title/description/tags/thumbnail written by the AI, with credit to the streamer in the description.

**Automation**
- **Autopilot:** a links file, N voice-over clips per day from one video, quality check, scheduling.
- **Assistant:** works the queue, reviews, approves, publishes, reads YouTube stats, retrains, cleans the disk, makes backups.
- Notifications: in-app bell, phone push (ntfy.sh), Telegram, Gmail. Errors, finished links and "links running out" reminders.
- Watchdog: restarts the program after a crash or a hang; starts on Windows login.
- Safe mode: one switch that stops every upload and new job.

**Learning**
- Reads views/likes (and optionally watch time) from YouTube, keeps a channel playbook, and feeds the
  lessons into the script/title prompts. Prefers the clip categories and hook/title styles that got more views, and
  suggests (or applies) better publish times. Analytics can be downloaded as CSV (Analytics tab).

**Operations**
- Web UI with Settings in one place, Memory (disk) tab, live activity log with Stop buttons, analytics, daily backups.
- Free-tier friendly Gemini client: model fallback chain, per-model rate limits, cool-downs, quota display.
  An OpenAI-compatible provider (Groq, OpenRouter, a paid OpenAI key…) can be used instead.

## Install (Windows)

1. **Python 3.11 or 3.12** from python.org (tick *Add to PATH*).
2. **ffmpeg** (full build): `winget install Gyan.FFmpeg`, then open a new terminal.
3. In the project folder:
   ```bat
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m featureclip init
   ```
   (`setup_windows.bat` does all of this with one double-click.)
4. Put your free Gemini key in `.env`: `GEMINI_API_KEY=...` (get one at https://aistudio.google.com/apikey).
5. `python -m featureclip doctor --ping --whisper` and fix every FAIL line.
   The first run downloads the Whisper model (~140 MB) and the Kokoro voice (~120 MB), once.

Linux/macOS also work from the command line (same steps with your shell's venv activation); the `.bat` helpers are Windows only.

## Quick start

```bat
python -m featureclip ui          :: opens http://127.0.0.1:8765
```
1. **Settings → AI (Gemini):** paste your key.
2. **YouTube tab:** choose `client_secret.json` (see below), then log in once.
3. **New video:** paste a video link, enter how many clips, press **Make clips**.
4. **Clips:** review the result. Or switch the **Assistant** on and let it approve and schedule.

### YouTube setup (one time)
Google Cloud Console → new project → enable **YouTube Data API v3** → OAuth consent screen (External, add your own email as a
test user) → create an **OAuth client ID of type Desktop app** → download the JSON and pick it in the YouTube tab.
Each channel has its own *YouTube login* button; choose the Google account/channel for that clip type.

> Until Google audits your API project, YouTube may keep API uploads **private** and ignore the schedule. The assistant
> tells you when that happens; you can then publish from YouTube Studio.

## Autopilot

Put one video link per line in `links.txt` (recorded videos only):

```text
# link | streamer name (optional) | webcam corner (optional)
https://www.youtube.com/watch?v=XXXXXXXXXXX | Ducky Bhai
https://www.youtube.com/watch?v=YYYYYYYYYYY
```

Turn on **Settings → Autopilot → enabled** and switch the **Assistant** on. Then, every day at `run_at` (default 09:00):

1. the next link is taken (one per day by default; if the laptop was off, it catches up when it starts);
2. the video is downloaded once and turned into 5 voice-over clips;
3. each clip goes through the AI quality check; good ones are approved and scheduled on the next free slot;
4. you get a notification with the result. If a link produces nothing it is retried once, then skipped with an error notice;
5. when few links are left (default 3), when the last link is taken, and every day once none are left, you get a reminder.

Live-stream links are skipped (record those from the Assistant/New video tabs). Training from YouTube results continues in the background.

**Starting by itself after a reboot:** run `install_autostart.bat` once. It registers a Windows task that starts the
program **hidden** (`start_background.vbs` -> `start_ui_hidden.bat`, the watchdog) at every login, also on battery. It starts
when you log in, not before. The laptop must be on and awake: in sleep nothing runs, after waking it continues by itself.
Schedule a power plan with sleep set to *Never* if you want the daily run at its exact time. Logs: `data/watchdog.log`.

## The web UI (tabs)

| Tab | What is there |
|---|---|
| 🤖 **Assistant** | animated robot showing what the assistant is doing right now, steps, live log and decisions, on/off, Safe mode, Launch mode, Autopilot status and link box, the link queue (with Stop) |
| 🎬 **New video** | make clips from one video or a live stream, layout preview and crop editor |
| 🎞 **Clips** | every clip with video, title, description, thumbnail, AI review; approve, reject, rebuild, regenerate text |
| ▶️ **YouTube** | login, **Upload now** and **Schedule on available slots**; slots and scheduled videos; one-time Google setup |
| 📜 **Activity / Log** | every job with its live log, ⏹ Stop per job, **Stop everything** |
| 📈 **Analytics** | results by category and publish time, learning playbook, CSV downloads, Gemini usage, free quota |
| 🧠 **Memory** | disk usage; delete **clip files** or **work files** (the data stays); smart cleanup; backups |
| ⚙️ **Settings** | every setting in one place, in sections, one Save button; tools: self-test, Gemini test, system check, voice lab |

## Upgrading from the old two-channel version

The first time the new version starts it **backs up** your database and then removes the old streamer-clip leftovers
(streamer clips, their files, their queue entries, old streamer settings). Voice-over clips are never touched. You can also
run it yourself: `python -m featureclip remove-streamer` (shows what would go) and add `--yes` to really delete.
Then delete `token_streamer.json` and the `streamer:` / `channels:` blocks in `config.yaml` by hand (they are ignored).
Clips already uploaded to the old streamer channel stay on YouTube; delete them in YouTube Studio if you want.

## Configuration

Settings live in `config.yaml` (created by `init`; edited through the Settings tab, a copy is kept as `config.yaml.bak`).
Secrets live in `.env` and never in `config.yaml`. Defaults and comments: `featureclip/config.py`.

| Key | Default | Meaning |
|---|---|---|
| `assistant.enabled` / `assistant.mode` | off / `auto` | `auto` approves and schedules by itself, `ask` only reviews |
| `autopilot.enabled` | off | work through the links file |
| `autopilot.run_at`, `links_per_day` | `09:00`, `1` | when and how many links per day |
| `autopilot.voiceover_clips` | `5` | clips per link |
| `autopilot.warn_links_left` | `3` | reminder threshold |
| `schedule.slots` | `21:00, 01:00, 05:00` | publish times, **Pakistan time** |
| `schedule.max_ahead_days` | `3` | how far ahead slots are filled |
| `upload.privacy` | `private` | `private` / `unlisted` / `public` (start with private) |
| `upload.max_per_day` | `3` | daily upload cap when not scheduling |
| `ingest.part_minutes` | `30` | long online videos are downloaded this many minutes at a time; each part is deleted after its clips are made (`0` = whole video) |
| `ingest.max_minutes` | `0` | use at most this many minutes of one video (`0` = no limit) |
| `layout.mode` | `fill` | `fill` or `hybrid` |
| `cleanup.max_work_gb` | `20` | hard limit of the work folder |
| `backup.every_hours`, `backup.keep` | `24`, `7` | automatic backups |
| `llm.provider` | `gemini` | or an OpenAI-compatible provider for a paid key |

`.env`: `GEMINI_API_KEY`, optional `GEMINI_MODELS`, `OPENAI_COMPAT_API_KEY`, `TELEGRAM_BOT_TOKEN`, `NOTIFY_EMAIL_PASSWORD`.

## Command line

```bat
python -m featureclip <command> [--root PATH] [--config FILE]
```

| Command | Purpose |
|---|---|
| `init` | create `config.yaml` and `.env` from the examples |
| `doctor [--ping] [--whisper]` | check the installation (`--ping` tests the Gemini key) |
| `selftest` | run the whole pipeline on a synthetic video (no key, no footage needed) |
| `ui` | start the web UI at http://127.0.0.1:8765 |
| `run <video link or file>` | scan → judge → build for one video |
| `scan`, `judge`, `build` | the same, one stage at a time |
| `watch <id> --url <live url>` | record a live stream in pieces and clip each piece |
| `layout <video> --out layout.png` | draw the detected gameplay area on a frame |
| `list`, `approve`, `upload [--dry-run]` | review and publish from the terminal |
| `daily <video>` | run a source then upload approved clips (Task Scheduler friendly) |
| `rejudge`, `retry` | redo judging or rebuild a clip with the newest rules |
| `backup`, `restore <zip>` | save / restore the database and settings (close the program before restoring) |
| `tidy [--yes]` | list (or delete) files the program does not need; `python tidy.py` works without installing anything |
| `python -m featureclip.supervisor --root .` | run the UI + assistant under the watchdog |

Windows helpers: `start_background.vbs` (hidden start), `start_ui_hidden.bat` / `start_watchdog.bat` (watchdog), `start_live.bat`, `install_autostart.bat`,
`remove_autostart.bat`, `tidy.bat`.

## Reliability and safety

- **Watchdog** restarts the app on a crash, an unresponsive UI, or a stuck job (no heartbeat for 20 minutes) and tells you.
- **Safe mode** stops all uploads and new work with one click; it can also pull scheduled videos back to private.
- **Stop** buttons kill the running ffmpeg/job at the next step; finished data is kept.
- **Disk protection:** a size cap for the work folder, automatic cleanup of used files, live recording pauses when the disk is almost full.
- **YouTube limits:** upload-limit and quota errors pause that channel for a few hours instead of failing clips.
- **Hard content rules** in code (see *Features*), and an AI review of every clip before it can be approved.
- **Settings** are saved atomically; a backup of the database and settings is made every 24 hours.
- **Secrets:** `.env`, `client_secret.json`, `token*.json` and `cookies.txt` stay on your machine. They are in `.gitignore`; never commit them.

## Data, backups and disk space

| Location | Content |
|---|---|
| `data/featureclip.db` | clips, titles, stats, learning, queue, autopilot state (SQLite) |
| `data/backups/` | zip backups (database + settings) |
| `work/` | downloads, live recordings, caches (can be deleted, re-created when needed) |
| `output/` | finished clips and thumbnails (the **Memory** tab can delete the files and keep the data) |
| `config.yaml`, `.env`, `links.txt` | your settings, secrets, autopilot links |

Restore: close the program, then `python -m featureclip restore data\backups\<file>.zip`.

## Troubleshooting

| Problem | What to do |
|---|---|
| `ModuleNotFoundError: yaml` (or any module) | you are not in the environment where `pip install -r requirements.txt` was run; activate it first |
| Gemini 429 / quota | the tool switches models by itself; see Analytics → Free quota. Quota resets at midnight Pacific time |
| YouTube keeps uploads private | your Google project is not audited yet; publish from YouTube Studio (the assistant notifies you) |
| Disk too small for a long stream | Settings → Download → *Download long videos in parts of* (default 30 min): one part on disk at a time |
| yt-dlp bot / 429 errors | put a `cookies.txt` in the project folder or set the browser cookie option in Settings → Download |
| UI shows old settings | press **Save all settings**; settings are only written on Save |
| Disk full | Memory tab → delete work files / clip files |

## Tests

```bat
pip install -r requirements.txt
python -m pytest -q --ignore=tests/test_live.py
```
About 170 tests; they need no internet and no API key (a fake Gemini server and a fake YouTube client are used). The
full-pipeline tests render real video with ffmpeg and take several minutes. `tests/test_live.py` needs a real live stream.

## Project layout

```
featureclip/
  pipeline.py      the stages (scan, judge, build, review, upload), resumable
  agent.py         the assistant loop (queue, review, publish, learn, housekeeping)
  autopilot.py     links file, one link per day, reminders
  judge.py, signals.py, candidates.py, layout.py, ai_layout.py   finding and judging moments
  transcribe.py, narrate.py, tts.py, music.py, captions.py, compose.py, qa.py   building the clip
  metadata.py, uploader.py, schedule.py, launch.py, quota.py                 publishing
  learn.py, insights.py, styles.py        learning from YouTube results
  llm.py, llm_limits.py                   Gemini (and OpenAI-compatible) client, fallbacks, limits
  live.py          live recording in pieces
  supervisor.py, heartbeat.py, safe.py, cleanup.py, backup.py, memory.py, notify.py   reliability
  web/             FastAPI app, job manager, settings store, single-page UI (static/index.html)
tests/             pytest suite
tools/             make_test_video.py (synthetic test stream)
config.example.yaml, .env.example, links.txt.example
```

## Known limits

- Tested on synthetic streams and mocked services; behaviour on very long real streams, real YouTube logins and a real
  Windows hang is only as good as the first real runs. Read the first days' logs.
- Autopilot handles **recorded** videos (finished streams are downloaded in parts); running live links are skipped.
- Gemini's free tier has small daily limits for the smarter models; the tool uses the high-limit Flash-Lite models first.
- Facecam/gameplay detection can be wrong for unusual layouts; use the layout preview and the crop fields.
- Twitch chat signals are not supported.
- Windows is the supported platform for the helper scripts.

## Copyright and platform rules (read this)

- A voice-over does **not** make footage copyright-free. The footage belongs to the game publisher and the streamer.
  Use it only with the streamer's permission, which is necessary but not sufficient.
- Publishers' fan-content policies are often **non-commercial**. Monetizing game footage can lead to claims, takedowns,
  demonetization or strikes. Read the current policy yourself and consider asking the publisher for explicit permission.
- YouTube's reused/inauthentic-content rules require real added value in each Short. Do not post many near-identical clips;
  keep the daily cap low and vary the content.
- The description always credits the streamer. Get their permission.
- This tool deliberately contains no tricks to avoid Content ID or copy detection.

## License

No license has been chosen yet. Add a `LICENSE` file (for example MIT) before making the repository public if you want others to use it.
