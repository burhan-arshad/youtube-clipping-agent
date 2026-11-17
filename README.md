# featureclip

Turns gameplay recordings you have **permission** to use into vertical (9:16) YouTube Shorts:

```
recording -> find exciting moments -> AI checks "is a real game feature shown?" -> crop to gameplay only
          -> AI writes + speaks a short explanation -> captions -> quality checks -> review -> upload
```

Everything runs on a normal laptop (8 GB RAM, **no GPU**) and costs nothing: audio/motion analysis,
Whisper (speech-to-text), Kokoro (voice) and ffmpeg run locally on the CPU; the only online AI call is
Google's **free** Gemini API.

## What it does for you
| Step | What happens | Where it runs |
|---|---|---|
| scan | permission check, audio-loudness + motion + scene-change analysis, ~30 candidate windows | CPU, local |
| judge | Whisper hears what the streamer said, Gemini looks at 4 frames and decides *feature or not* | Whisper local, Gemini free API |
| build | streamer/facecam/overlays cropped away, 9:16 crop that follows the action, script, voice-over, captions, QA | CPU, local |
| review | you look at the clips and `approve` the good ones | you |
| upload | YouTube Data API, **private** by default, daily cap | Google API |

Hard rules in code: cutscenes, story spoilers, glitches/exploits, mods/cheats and menus are always rejected;
numbers/dates the AI was not given are removed from the script; game/streamer audio is removed by default
(no streamer voice, no licensed in-game music); 

## Install (Windows)
1. Python 3.11 or 3.12 from python.org (tick "Add to PATH").
2. ffmpeg (full build): `winget install Gyan.FFmpeg`, then open a new terminal.
3. In this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   python -m featureclip init
   ```
4. Put your free key in `.env` (`GEMINI_API_KEY=...`, from https://aistudio.google.com/apikey).
5. `python -m featureclip doctor --ping --whisper` and fix every FAIL line.
   (The first run downloads the Whisper model ~140 MB and the Kokoro voice ~120 MB, once.)

Check the Gemini model names in `config.yaml` against Google AI Studio; `doctor --ping` tells you if they work.

## Use
```
# 1. pass a video link or file directly (or name it in sources.yaml)
# 2. try the layout detector on a recording (draws green = gameplay, red = facecam, yellow = crop)
python -m featureclip layout path\to\vod.mp4 --out layout.png
# 3. run everything
python -m featureclip run my_source_id
# 4. look at output\*.mp4, then
python -m featureclip list
python -m featureclip approve 3 5          # or: approve --all
python -m featureclip upload --dry-run
python -m featureclip upload
```
Step by step instead of `run`: `scan` -> `judge <video_id>` -> `build <video_id>`.
A crash never loses work: signals, transcripts and finished clips are cached; just run the command again.
`retry <clip_id>` re-builds a clip that ended in `error`.

No GPU, 8 GB RAM: timings are estimates, not measured on your laptop. On my test machine building one clip took
about 40 s; analysing a long 1080p60 VOD is dominated by video decoding and may take 10-30 minutes, plus a few
minutes of Whisper on the candidates. Close other heavy programs while it runs.

### YouTube upload setup (one time)
Google Cloud Console -> new project -> enable "YouTube Data API v3" -> OAuth consent screen -> create an
**OAuth client ID of type Desktop app** -> download as `client_secret.json` into this folder. The first upload
opens a browser to log in. Note: until Google audits the API project, YouTube may force API uploads to
**private**; `upload` prints the privacy status YouTube really applied.

### Daily automation (Windows Task Scheduler)
Action: `.venv\Scripts\python.exe -m featureclip --root C:\path\to\featureclip daily my_source_id`
(`daily` runs the source, then uploads clips you already approved, up to `upload.max_per_day`).
Keep `review.auto_approve: false` until you trust the output.

## Copyright, Rockstar and YouTube rules (read this)
* A voice-over does **not** make footage copyright-free. The footage is the game publisher's (and the
  streamer's recording). Written permission from the streamer is necessary but not sufficient.
* The Rockstar policy summaries I found describe fan use as **non-commercial**. Monetizing GTA footage is therefore
  a real risk (claims, takedowns, demonetization, strikes). Read the current policy on Rockstar's support site
  yourself, and consider asking Take-Two/Rockstar for explicit permission before relying on this for income.
* YouTube's reused-content / inauthentic-content rules: each Short must add real value (accurate explanation,
  good editing). Do not post dozens of near-identical clips a day. Keep `max_per_day` low and vary the content.
* This tool deliberately does **not** include tricks to dodge Content ID or copy detection.

## Layout notes
* `layout.mode: fill` fills the 9:16 screen with the gameplay crop; `hybrid` shows a 4:5 block over a blurred copy.
* Facecam detection uses OpenCV face detection; if a streamer's facecam is not detected, set
  `crop: [x, y, w, h]` for that source in `sources.yaml` (use `layout` to see the numbers).
* LIVE: `python -m featureclip watch SOURCE_ID --url <live url>` records the stream in 10-minute pieces and
  clips every finished piece (clips wait in the review queue; nothing is uploaded automatically).
  `--test-file video.mp4` simulates a live stream. Clips appear ~10-15 minutes after the moment happened.
* Not supported yet: Twitch chat signals.

## Tests
`pip install pytest` then `python -m pytest -q` (about 3 minutes; needs no internet and no API key).
Tests cover signal detection, layout/crop, the LLM client (fake Gemini server), judge rules, script validation,
captions, the permission gate, the full pipeline on a synthetic stream, and the upload loop (fake YouTube client).
Optional: `FEATURECLIP_MODELS=<folder with kokoro/ model files>` also runs the real Kokoro voice test.

## Project layout
```
featureclip/   signals, candidates, layout, transcribe, llm, judge, narrate, captions, compose, qa, uploader, pipeline, cli
tests/         pytest suite        tools/make_test_video.py   synthetic test stream
docs/outreach_message.md           config.example.yaml / sources.example.yaml / .env.example
```

## Browser UI (naya)
`pip install -r requirements.txt` then `start_ui.bat` (or `python -m featureclip ui`). In the browser
http://127.0.0.1:8765 opens: API key, folders, video link, clip review and YouTube upload all from one page.
`sources.yaml` is optional.
