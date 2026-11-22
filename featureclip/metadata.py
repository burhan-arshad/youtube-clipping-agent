"""YouTube metadata (title, description, tags) written by the LLM, plus a generated thumbnail."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .narrate import mentions_game

TITLE_RULES = """- 40 to 70 characters, must contain the game name "{game}"
- a curiosity gap or a clear promise that is TRUE for what is on screen (no fake claims, no "leaked", no "confirmed")
- no emoji, no hashtags, no ALL CAPS words except short names (GTA, NPC, AI)
- do not start with the game name every time; vary the structure"""


def channel_name(cfg) -> str:
    return str(cfg.get("branding.name", "") or "").strip() or str(cfg.get("branding.handle", "") or "").strip()


def credit_lines(cfg, streamer: str = "") -> list[str]:
    """Footer of the description: names only, never links (links in Shorts descriptions are not wanted)."""
    lines = []
    if streamer:
        lines.append(f"Footage from {streamer}'s stream.")
    lines.append("Voice-over: AI-generated.")
    foot = str(cfg.get("upload.description_footer", "") or "").strip()
    if foot:
        lines.append(foot)
    name = channel_name(cfg)
    if name:
        lines.append(f"Channel: {name}")
    return lines


def thumb_line(meta: dict | None, hook: str, title: str) -> str:
    """The few words printed on the thumbnail: the AI's thumb_text, else the hook cut at a word boundary."""
    t = (meta or {}).get("thumb_text") or ""
    if not t:
        t = re.sub(r"\s*#\w+", "", hook or title or "")
        t = re.sub(r"^[^:]{2,12}:\s*", "", t).strip()
        if len(t) > 34:
            cut = t[:34].rsplit(" ", 1)[0]
            t = cut.rstrip(",;:-")
    return t.strip() or "WATCH THIS"


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", str(t or "")).strip().strip('"').strip()


def fallback(judgement: dict, hook_title: str, game: str, streamer: str, credit_url: str, cfg) -> dict:
    from .uploader import build_description, build_title
    title = build_title(hook_title or judgement.get("feature_name", "Gameplay detail"),
                        judgement.get("feature_name", ""), norm_suffix(cfg.get("upload.title_suffix", "")))
    return {"title": title, "description": build_description(judgement, streamer, credit_url, cfg),
            "tags": default_tags(game), "thumb_text": ""}


def default_tags(game: str) -> list[str]:
    g = game or "gaming"
    return [g, f"{g} gameplay", f"{g} details", f"{g} features", "open world", "game details", "gaming shorts"]


def norm_suffix(suffix) -> str:
    """Title suffix with exactly one leading space ('#Shorts' -> ' #Shorts'); '' stays ''."""
    s = str(suffix or "").strip()
    return (" " + s) if s else ""


_TAIL_TAGS = re.compile(r"(\s*#\w+)+\s*$")


PLAYBOOK = ""            # learned from YouTube results (learn.py); set by the pipeline before a clip is built


def validate(meta: dict, game: str) -> list[str]:
    p = []
    t = meta.get("title", "")
    if not (15 <= len(t) <= 100):
        p.append(f"title length {len(t)} (want 40-70)")
    if game and not mentions_game(t, game):
        p.append(f"title must contain '{game}'")
    if "#" in _TAIL_TAGS.sub("", t):      # hashtags at the very end are the suffix added on purpose
        p.append("no hashtags in the title")
    d = meta.get("description", "")
    if len(d) < 80:
        p.append("description too short")
    if len(re.findall(r"#\w+", d)) > 12:
        p.append("too many hashtags")
    return p


def sanitize(meta: dict, game: str, suffix: str, judgement: dict, cfg, streamer: str = "") -> dict:
    """Repair title/description rule problems without the AI, so a good video is never lost to wording."""
    suffix = norm_suffix(suffix)
    title = _clean(meta.get("title", ""))
    title = re.sub(r"\s*#\w+", "", title).strip(" -:|")
    if game and not mentions_game(title, game):
        title = f"{game}: {title}" if title else f"{game}: {judgement.get('feature_name', 'gameplay detail')}"
    room = 95 - len(suffix)
    if len(title) > room:
        title = title[:room].rsplit(" ", 1)[0].rstrip(",;:-")
    if len(title) < 20:
        title = f"{title} - {judgement.get('feature_name', '')}".strip(" -")[:room]
    desc = str(meta.get("description", "") or "").strip()
    if len(desc) < 80:
        desc = build_description_plain(judgement, streamer, cfg)
    tags_line = re.findall(r"#\w+", desc)
    if len(tags_line) > 12:
        keep = 0
        def _cut(m):
            nonlocal keep
            keep += 1
            return m.group(0) if keep <= 12 else ""
        desc = re.sub(r"#\w+", _cut, desc)
    return {**meta, "title": (title + suffix)[:100], "description": desc[:4900]}


def build_description_plain(judgement: dict, streamer: str, cfg) -> str:
    from .uploader import build_description
    return build_description(judgement, streamer, "", cfg)


def generate(llm, cfg, judgement: dict, script: str, hook_title: str, streamer: str = "", credit_url: str = "",
             title_style: str = "") -> dict:
    """Returns {"title","description","tags"}. Falls back to the plain builder if the AI cannot deliver."""
    game = str(cfg.get("game.name", "") or "")
    handle = str(cfg.get("branding.handle", "") or "").strip()
    fb = fallback(judgement, hook_title, game, streamer, credit_url, cfg)
    if not llm or not llm.available():
        return fb
    facts = (f"Feature: {judgement.get('feature_name', '')}\nWhat happens: {judgement.get('what_happens', '')}\n"
             f"Visible evidence: {'; '.join(judgement.get('visible_evidence', []) or [])}\n"
             f"Voice-over: {script}")
    prompt = f"""You write YouTube Shorts metadata for a {game} channel that explains small game details.
{PLAYBOOK}

FACTS (only use these, never invent anything):
{facts}

Write:
1. "title" - {TITLE_RULES.format(game=game)}{(" Title style for this video: " + title_style + ".") if title_style else ""}
2. "description" - plain text, 5 parts separated by blank lines:
   a) one strong sentence (max 140 characters) saying what the viewer will see
   b) 2 short sentences giving context from the facts
   c) one question that makes viewers comment
   d) 3 to 5 hashtags (include #{re.sub(r'[^A-Za-z0-9]', '', game)} and #Shorts), nothing else on that line
   Do not add credits, links or disclaimers; the program adds them.
3. "tags" - 8 to 12 search keywords (lowercase, no #), mixing the game name, the feature and generic gaming terms.
4. "thumb_text" - 2 to 4 punchy words for the thumbnail picture (what the viewer will see or feel), no game name,
   no hashtags, e.g. "REAL CRASH DAMAGE" or "WAIT FOR IT". Never more than 28 characters.

Return JSON only: {{"title": "...", "description": "...", "tags": ["..."], "thumb_text": "..."}}"""
    meta = None
    feedback = ""
    for _ in range(3):
        try:
            raw = llm.json(prompt + feedback, temperature=0.8)
        except Exception:
            break
        cand = {"title": _clean(raw.get("title")), "description": str(raw.get("description", "")).strip(),
                "thumb_text": re.sub(r"[#\n]", "", _clean(raw.get("thumb_text")))[:28],
                "tags": [re.sub(r"[#\n]", "", _clean(t)).lower() for t in (raw.get("tags") or []) if _clean(t)][:15]}
        problems = validate(cand, game)
        if not problems:
            meta = cand
            break
        feedback = "\n\nYour previous answer had problems, fix them: " + "; ".join(problems)
    if not meta:
        return fb
    tail = credit_lines(cfg, streamer)
    body = meta["description"]
    m = re.search(r"\n\s*(#\w+(?:\s+#\w+)*)\s*$", body)
    tags_line = m.group(1) if m else ""
    if m:
        body = body[: m.start()].rstrip()
    desc = body + "\n\n" + "\n".join(tail) + ("\n\n" + tags_line if tags_line else "")
    suffix = norm_suffix(cfg.get("upload.title_suffix", ""))
    title = _TAIL_TAGS.sub("", meta["title"]).strip()
    if len(title) + len(suffix) <= 100:
        title += suffix
    tags = meta["tags"] or default_tags(game)
    while sum(len(t) + 1 for t in tags) > 450:
        tags.pop()
    return {"title": title[:100], "description": desc[:4900], "tags": tags, "thumb_text": meta.get("thumb_text", "")}


# ------------------------------------------------------------------ thumbnail
_FONTS = ["C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/arialbd.ttf",
          "/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]


def _font(size: int):
    from PIL import ImageFont
    for f in _FONTS:
        try:
            return ImageFont.truetype(f, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _score(img: np.ndarray) -> float:
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(g, cv2.CV_64F).var()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = float(hsv[..., 1].mean())
    bright = float(g.mean())
    penalty = 0.0 if 50 < bright < 200 else 400.0          # skip too dark / blown-out frames
    return min(sharp, 1500) + sat * 3 - penalty


def pick_frame(video: str, start: float, dur: float, crop, tmp: Path, n: int = 8) -> Path:
    """Best looking frame of the clip (sharp, colourful, well exposed), cropped to the gameplay area."""
    import cv2
    from .ffutil import grab_frame
    tmp.mkdir(parents=True, exist_ok=True)
    best, best_s = None, -1e9
    for i in range(n):
        t = start + dur * (0.12 + 0.76 * i / max(n - 1, 1))
        f = tmp / f"th_{i}.jpg"
        try:
            grab_frame(video, t, f, 1280, crop)
            img = cv2.imread(str(f))
            if img is None:
                continue
            s = _score(img)
            if s > best_s:
                best, best_s = f, s
        except Exception:
            continue
    if best is None:
        raise RuntimeError("no frame could be extracted for the thumbnail")
    return best


def make_thumbnail(frame: Path, out: Path, title: str, cfg, size=(1280, 720)) -> Path:
    """Gameplay frame + bold title + game badge. 1280x720 JPEG, well under YouTube's 2 MB limit."""
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
    W, H = size
    img = Image.open(frame).convert("RGB")
    bg = img.resize((W, H), Image.LANCZOS).filter(ImageFilter.GaussianBlur(18))
    bg = ImageEnhance.Brightness(bg).enhance(0.55)
    scale = min(W / img.width, H / img.height)
    fg = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    fg = ImageEnhance.Color(ImageEnhance.Contrast(fg).enhance(1.12)).enhance(1.25)
    bg.paste(fg, ((W - fg.width) // 2, (H - fg.height) // 2))
    shade = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(shade)
    for y in range(H):                                     # dark gradient at the bottom for the text
        d.line([(0, y), (W, y)], fill=int(190 * max(0, (y - H * 0.45) / (H * 0.55)) ** 1.4))
    bg = Image.composite(Image.new("RGB", (W, H), (8, 6, 20)), bg, shade)
    draw = ImageDraw.Draw(bg)
    title = re.sub(r"\s*#\w+", "", title or "").strip()
    text = re.sub(r"^[^:]{2,12}:\s*", "", title)[:60] or title                # drop a "GTA 6:" prefix, the badge shows it
    words, lines, cur = text.upper().split(), [], ""
    f = _font(92)
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=f) > W - 120 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    lines.append(cur)
    while len(lines) > 3:
        lines = [" ".join(lines[:2])] + lines[2:]
    f = _font(92 if len(lines) <= 2 else 76)
    y = H - 60 - len(lines) * (f.size + 6)
    for ln in lines:
        x = 60
        for dx, dy in [(-4, -4), (4, -4), (-4, 4), (4, 4), (0, 6)]:
            draw.text((x + dx, y + dy), ln, font=f, fill=(0, 0, 0))
        draw.text((x, y), ln, font=f, fill=(255, 226, 64))
        y += f.size + 6
    game = str(cfg.get("game.name", "") or "").upper()
    if game:
        bf = _font(54)
        tw = draw.textlength(game, font=bf)
        draw.rounded_rectangle([40, 36, 40 + tw + 56, 36 + 82], radius=16, fill=(232, 48, 127))
        draw.text((68, 36 + 8), game, font=bf, fill=(255, 255, 255))
    handle = str(cfg.get("branding.handle", "") or "").strip()
    if handle:
        hf = _font(40)
        draw.text((W - 40 - draw.textlength(handle, font=hf), 44), handle, font=hf, fill=(255, 255, 255),
                  stroke_width=3, stroke_fill=(0, 0, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    q = 90
    while True:
        bg.save(out, "JPEG", quality=q)
        if out.stat().st_size < 1_900_000 or q <= 60:
            return out
        q -= 10
