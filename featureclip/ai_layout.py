"""AI layout detection: show a few screenshots of the stream to the vision model (Gemini) and ask where the
game picture is and which stream elements (webcam, chat, text, logos, alerts) must be cut away.
The answer is validated in code; if anything looks wrong the caller falls back to the classic detector."""
from __future__ import annotations

from pathlib import Path

from .ffutil import grab_frame
from .layout import Box, Layout, _even

SYSTEM = "You are a precise video layout analyst. Reply with JSON only."

PROMPT = """These {n} screenshots come from ONE live stream / recording of a video game. The stream layout is
the same in all of them (the game picture changes, the overlays stay).

Find, as boxes in the format [ymin, xmin, ymax, xmax] with values from 0 to 1000 (0,0 = top-left, 1000,1000 = bottom-right):

1. "gameplay": the FULL extent of the game picture, as LARGE as possible. Include the parts of the game picture
   that are covered by a webcam, chat or other overlays: those overlays are cut away later by software, so do
   NOT shrink the gameplay rectangle to avoid them. Only leave out areas that show no game at all: black bars,
   solid-colour borders, side panels or backgrounds that are not the game. The game's own HUD (minimap, health
   bar, speedometer, game subtitles) belongs to the game.
2. "overlays": every element added by the streamer or the stream software that lies INSIDE or touches that
   gameplay rectangle: webcam / facecam, live chat, donation or subscriber alerts, channel logo or watermark,
   banners, tickers, counters, social-media handles, "LIVE" badges, subtitles written by the streamer.
   Give each a TIGHT box (only the element itself, no game picture around it) and a kind:
   facecam | chat | text | logo | alert | other.

Rules:
- If the whole frame is the game and nothing was added, use gameplay [0,0,1000,1000] and overlays [].
- A webcam can be small and in a corner of the game picture: it is still a "facecam" overlay.
- Do not list the game's own HUD, characters or in-game signs as overlays.
- Be exact with the edges; do not leave a margin around the boxes.
- The goal: the biggest possible clean game picture is kept, so the gameplay box must be big and the overlay
  boxes must be small and tight.

Reply with exactly this JSON:
{{"gameplay": [ymin, xmin, ymax, xmax],
  "overlays": [{{"kind": "facecam", "box": [ymin, xmin, ymax, xmax]}}],
  "layout_type": "fullscreen_game | game_with_overlay | game_with_sidebar | other",
  "notes": "one short sentence"}}"""


REVIEW = """This is one of the screenshots, with a measuring grid added by us: cyan lines every 100 units, the
numbers on the edges are the 0-1000 coordinates (x along the top, y along the left side). A first analysis drew
its boxes on the picture: GREEN = gameplay, RED (numbered) = overlays. Its answer was:
{first}

Check and CORRECT it. Use the grid to read exact numbers.
1. "gameplay" must be as LARGE as possible: the whole game picture, INCLUDING the parts hidden behind a webcam,
   chat or other overlays (software removes the overlays later). Leave out only areas that show no game at all
   (black bars, solid borders, panels with other content).
2. Every overlay box must be TIGHT around that element only. If a box also covers game picture, shrink it.
3. Add overlays that were missed (chat, counters, texts, logos, alerts, webcam). Delete boxes that are not overlays.

Reply with the corrected JSON in exactly the same format:
{{"gameplay": [ymin, xmin, ymax, xmax],
  "overlays": [{{"kind": "facecam", "box": [ymin, xmin, ymax, xmax]}}],
  "layout_type": "fullscreen_game | game_with_overlay | game_with_sidebar | other",
  "notes": "one short sentence"}}"""


def _annotate(jpg: bytes, raw: dict) -> bytes:
    """Draw the measuring grid and the first answer's boxes on a screenshot."""
    import cv2
    import numpy as np
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    for k in range(1, 10):
        x, y = int(w * k / 10), int(h * k / 10)
        cv2.line(img, (x, 0), (x, h), (255, 255, 0), 1)
        cv2.line(img, (0, y), (w, y), (255, 255, 0), 1)
        cv2.putText(img, str(k * 100), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(img, str(k * 100), (2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    def draw(v, color, label=""):
        b = _box(v)
        if b:
            p0, p1 = (int(b[1] * w), int(b[0] * h)), (int(b[3] * w) - 1, int(b[2] * h) - 1)
            cv2.rectangle(img, p0, p1, color, 2)
            if label:
                cv2.putText(img, label, (p0[0] + 4, p0[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    draw(raw.get("gameplay"), (0, 220, 0), "game")
    for i, o in enumerate(raw.get("overlays") or [], 1):
        if isinstance(o, dict):
            draw(o.get("box"), (0, 0, 255), f"{i} {o.get('kind', '')}")
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()


def _box(v):
    try:
        y0, x0, y1, x1 = [float(t) for t in v]
    except (TypeError, ValueError):
        return None
    y0, x0, y1, x1 = (min(max(t, 0.0), 1000.0) for t in (y0, x0, y1, x1))
    if y1 - y0 < 8 or x1 - x0 < 8:
        return None
    return y0 / 1000, x0 / 1000, y1 / 1000, x1 / 1000


def parse(raw: dict, fw: int, fh: int, pad: float = 0.012) -> tuple[Box, list[Box], dict] | None:
    """-> (gameplay rect, overlay boxes (only those that matter), info) in pixels, or None if the answer is unusable."""
    g = _box(raw.get("gameplay"))
    if g is None:
        return None
    gy0, gx0, gy1, gx1 = g
    if (gy1 - gy0) * (gx1 - gx0) < 0.2:          # the game picture is at least 20% of the frame
        return None
    ins = 0.004                                   # trim a hair so compression halos stay out
    rect = (gx0 + ins, gy0 + ins, gx1 - ins, gy1 - ins)
    overlays, kinds = [], []
    for o in raw.get("overlays") or []:
        b = _box(o.get("box") if isinstance(o, dict) else None)
        if b is None:
            continue
        y0, x0, y1, x1 = b
        if (y1 - y0) * (x1 - x0) > 0.45:          # an "overlay" covering half of the screen is a mistake
            continue
        x0, y0, x1, y1 = max(x0 - pad, 0), max(y0 - pad, 0), min(x1 + pad, 1), min(y1 + pad, 1)
        # keep only overlays that actually overlap the gameplay rectangle
        if x1 <= rect[0] or x0 >= rect[2] or y1 <= rect[1] or y0 >= rect[3]:
            continue
        overlays.append((x0, y0, x1, y1))
        kinds.append(str(o.get("kind", "other")))

    def px(x0, y0, x1, y1) -> Box:
        return (_even(x0 * fw), _even(y0 * fh), _even((x1 - x0) * fw), _even((y1 - y0) * fh))

    return px(*rect), [px(*b) for b in overlays], {"kinds": kinds, "type": raw.get("layout_type", ""),
                                                  "notes": raw.get("notes", "")}


def detect(llm, video: str | Path, duration: float, frame_w: int, frame_h: int,
           tmp: Path, n_frames: int = 3) -> Layout | None:
    """Ask the vision model. Returns None when no LLM is available or the answer is not usable."""
    try:
        if not llm.available():
            return None
        tmp.mkdir(parents=True, exist_ok=True)
        fracs = (0.30, 0.50, 0.70) if n_frames >= 3 else (0.5,)
        images = []
        for i, f in enumerate(fracs):
            p = tmp / f"layout_{i}.jpg"
            grab_frame(video, duration * f, p, 960)
            images.append(p.read_bytes())
        raw = llm.json(PROMPT.format(n=len(images)), system=SYSTEM, images=images, temperature=0.1)
        out = parse(raw, frame_w, frame_h)
        if out is not None:                       # second look: the model checks and corrects its own boxes
            try:
                import json
                raw2 = llm.json(REVIEW.format(first=json.dumps(raw)), system=SYSTEM,
                                images=[_annotate(images[len(images) // 2], raw)], temperature=0.1)
                out2 = parse(raw2, frame_w, frame_h)
                if out2 is not None:
                    out = out2
            except Exception as e:
                print(f"[layout] AI review step skipped ({type(e).__name__}: {e})")
    except Exception as e:                        # never block the pipeline because of the layout helper
        print(f"[layout] AI layout failed ({type(e).__name__}: {e}); using the classic detector")
        return None
    if out is None:
        print("[layout] AI layout answer was not usable; using the classic detector")
        return None
    rect, avoid, info = out
    print(f"[layout] AI says: {info['type'] or '?'} - {info['notes']} | overlays: {info['kinds'] or 'none'}")
    return Layout(frame_w, frame_h, rect, avoid, "ai")
