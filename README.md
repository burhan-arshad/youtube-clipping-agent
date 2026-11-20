# featureclip

**Free, CPU-only tool that turns gaming streams into YouTube Shorts and publishes them for you.**
It watches live streams or recorded videos, finds the moments worth clipping, builds vertical 9:16 Shorts,
writes the title and description, and uploads them on a schedule to one or two YouTube channels.
Built for a laptop with 8 GB RAM and no GPU. The only online AI it needs is Google's free Gemini API.

---

## Two kinds of clips

| | **Voice-over clips** | **Streamer clips** |
|---|---|---|
| Picks | game features, funny, chaotic, epic and discovery moments | strong reactions only: laughing, screaming, shock |
| Look | gameplay cropped to 9:16 (hybrid or fill layout), captions | webcam on top, gameplay below |
| Sound | AI voice-over (Gemini TTS, Kokoro offline fallback) + generated copyright-free music | the streamer's own voice and the game's original audio |
| Text | script, title and description about the game moment | title and description about the streamer's reaction, with credit by name |
| Goes to | channel 1 | channel 2 |

You choose the type and how many clips you want every time you give it a link. Every clip remembers its type, so
the assistant uploads it to the right channel, with that channel's own login, publish slots and daily limit.

## How it works

```
source (live piece or video)
   -> scan      audio loudness + motion peaks -> candidate windows
   -> judge     Gemini looks at frames (+ face close-ups for streamer clips) and the Whisper transcript
   -> build     voice-over lane: script -> voice -> music -> crop -> captions
                streamer lane:   find webcam -> stack webcam + game -> original audio -> captions
   -> review    AI quality, wording and policy checks, then you approve (or the assistant does)
   -> upload    next free publish slot per channel, or "Upload now"
   -> learn     YouTube stats feed a small playbook that nudges what gets picked and how it is written
```

Everything except the Gemini calls and the YouTube upload runs locally: ffmpeg, faster-whisper, Kokoro,
OpenCV and a SQLite database.

## Features

- **Web UI** on `127.0.0.1:8765`: new video / live, clip review, YouTube login, settings, assistant dashboard.
- **Assistant (agent)**: keeps a queue of links, records live streams in 5-minute pieces, builds and reviews clips,
  uploads on a schedule, pauses safely when a limit is hit, and sends alerts (in-app, ntfy, Telegram).
- **Gemini failover**: several models, automatic switch on quota or errors, waits for the daily reset
  (Pacific midnight, about 13:00 Pakistan time) instead of dropping work.
- **Multi-channel**: one YouTube login per clip type, separate schedule and upload-limit pause per channel.
- **Launch mode**: first uploads go out one hour apart, then the normal slots take over.
- **Quality gates**: boring clips are dropped, wrong-game claims are filtered, titles and descriptions are validated,
  clips are ranked so the best get the earliest slots.
- **Credits**: descriptions credit the streamer by name, never with links.
- **Safe mode**: stops every upload with one switch; nothing is lost.

## Install (Windows)

1. Python 3.11 or 3.12 (tick "Add to PATH").
2. ffmpeg (full build): `winget install Gyan.FFmpeg`, then open a new terminal.
3. In the project folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m featureclip init
   ```
   (`setup_windows.bat` does the same.)
4. Put your free key in `.env`: `GEMINI_API_KEY=...` (from https://aistudio.google.com/apikey).
5. `python -m featureclip doctor --ping --whisper` and fix every FAIL line.
   The first run downloads the Whisper model (~140 MB) and the Kokoro voice (~120 MB) once.

## Run

```
python -m featureclip ui            # web UI (or start_ui.bat)
python -m featureclip run <source>  # one source, end to end
python -m featureclip watch <live-url>
python -m featureclip selftest      # whole pipeline on a synthetic video, no key or footage needed
```

In the UI, open **Assistant**, paste a link, pick the clip type and how many clips you want, and add it to the queue.

### YouTube setup (one time)

1. Google Cloud Console: new project, enable **YouTube Data API v3**, OAuth consent screen (External, add your
   accounts as test users), create an **OAuth client ID of type Desktop app**, save it as `client_secret.json` here.
2. UI, **YouTube** tab: log in once per channel (voice-over channel, streamer-clips channel).
3. Until Google audits your API project, YouTube may keep API uploads private and ignore the schedule; the tool
   shows what YouTube actually applied. Channel avatar, banner and name are set in YouTube Studio.

### Useful settings

| Setting | Meaning |
|---|---|
| `layout.mode` | default look: `hybrid`, `fill` or `streamer` (a source's chosen type overrides it) |
| `layout.facecam` | webcam corner (`top-right`, ...) or `auto` |
| `streamer.top_ratio`, `streamer.clip_min/max`, `streamer.min_reaction` | streamer look, length and reaction strictness |
| `schedule.slots`, `schedule.launch_gap_minutes` | publish times and launch spacing |
| `assistant.piece_minutes`, `assistant.stock_target` | live piece length, how many clips to keep ready |
| `channels.streamer.name`, `branding.name` | channel names used in descriptions |

## Project layout

```
featureclip/            pipeline, judge, narrate, compose, layout, live, uploader, agent, channels, web/
featureclip/web/        FastAPI backend + single-page UI (vanilla JS)
tests/                  pytest suite (real ffmpeg renders on synthetic video)
config.example.yaml     all settings with comments
sources.example.yaml    example sources
```

Run the tests: `python -m pytest tests -q --ignore=tests/test_live.py`.

## Status and limits

- Built and tested on synthetic video and real live runs on a Windows laptop. The streamer mode and the two-channel
  upload path have a full automated test suite but are new; check the first clips before trusting them.
- The webcam in streamer clips is a small part of the stream, so it is enlarged and can look soft.
- Free quotas apply: Gemini models have daily request limits and YouTube limits uploads per channel per day.
- Never commit `.env`, `client_secret.json`, `token*.json` or the `data/` folder.

## Responsible use

Use only footage you have the right to use, such as your own streams or content whose creator has agreed to clips.
Credit creators, follow YouTube's policies and the game publisher's content rules, and check them before you
rely on the channel for income. The tool credits the streamer by name and adds no links.
