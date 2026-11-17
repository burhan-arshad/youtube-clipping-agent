"""Burned-in captions as an ASS file: 2-3 words at a time, the spoken word highlighted,
plus a short on-screen hook at the top for the first seconds."""
from __future__ import annotations

from pathlib import Path


def _t(sec: float) -> str:
    sec = max(0.0, sec)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _esc(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")").replace("\n", " ")


def fit_hook(text: str, width: int, base_size: int, max_lines: int = 3, max_height: int | None = None) -> tuple[str, int]:
    """Break the hook into lines and pick a font size so that every line fits inside the frame."""
    words = text.split()
    usable = width - 160                        # side margins
    char_w = 0.64                               # width of a bold capital relative to the font size
    size = base_size
    while True:
        per_line = max(6, int(usable / (char_w * size)))
        lines, cur = [], ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > per_line:
                lines.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            lines.append(cur)
        longest = max((len(x) for x in lines), default=0)
        fits_h = max_height is None or len(lines) * size * 1.18 <= max_height
        if (len(lines) <= max_lines and longest <= per_line and fits_h) or size <= 40:
            return "\\N".join(lines), size
        size = int(size * 0.9)


def build_ass(words: list[tuple[float, float, str]], hook: str, cfg, offset: float = 0.0,
              title_box: tuple[int, int, int] | None = None, duration: float | None = None, handle: str = "") -> str:
    w, h = int(cfg.get("video.width")), int(cfg.get("video.height"))
    font = cfg.get("captions.font")
    size = int(cfg.get("captions.size"))
    margin_v = int(cfg.get("captions.margin_bottom"))
    per = max(1, int(cfg.get("captions.words_per_chunk", 3)))
    hook_secs = float(cfg.get("captions.hook_seconds", 2.6))

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 0",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,{margin_v},1",
        f"Style: Hook,{font},{int(size * 1.15)},&H0000E6FF,&H0000E6FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,3,8,60,60,110,1",
        f"Style: Handle,{font},{int(size * 0.62)},&H0000E6FF,&H0000E6FF,&H00000000,&H96000000,-1,0,0,0,100,100,2,0,1,4,2,5,60,60,0,1",
        f"Style: Title,{font},{int(size * 1.2)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,7,3,5,60,60,0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hook.strip() and title_box and duration:
        # heading in the blurred strip above the gameplay, visible for the whole clip
        cy, box_h, hy = title_box
        ttxt, tsize = fit_hook(_esc(hook.strip()), w, int(size * 1.2), max_lines=3, max_height=box_h)
        lines.append(f"Dialogue: 1,{_t(0)},{_t(duration)},Title,,0,0,0,,"
                     f"{{\\an5\\pos({w // 2},{cy})\\fs{tsize}\\fad(250,0)}}{ttxt}")
        if handle.strip():
            lines.append(f"Dialogue: 1,{_t(0)},{_t(duration)},Handle,,0,0,0,,"
                         f"{{\\an5\\pos({w // 2},{hy})}}{_esc(handle.strip())}")
    elif hook.strip():
        htxt, hsize = fit_hook(_esc(hook.strip().upper()), w, int(size * 1.15))
        lines.append(f"Dialogue: 1,{_t(offset)},{_t(offset + hook_secs)},Hook,,0,0,0,,"
                     f"{{\\fs{hsize}\\fad(150,200)}}{htxt}")
    if handle.strip() and not (hook.strip() and title_box and duration):
        end = duration or (words[-1][1] + offset + 1 if words else 10)
        lines.append(f"Dialogue: 1,{_t(0)},{_t(end)},Handle,,0,0,0,,{{\\an8\\pos({w // 2},170)}}{_esc(handle.strip())}")
    chunks = [words[i:i + per] for i in range(0, len(words), per)]
    for ci, chunk in enumerate(chunks):
        for wi, (s, e, _w) in enumerate(chunk):
            # the active word is yellow; the event lasts until the next word starts
            end = chunk[wi + 1][0] if wi + 1 < len(chunk) else (
                chunks[ci + 1][0][0] if ci + 1 < len(chunks) else e)
            end = max(end, s + 0.12)
            parts = []
            for j, (_, _, ww) in enumerate(chunk):
                txt = _esc(ww)
                parts.append(f"{{\\c&H00E6FF&}}{txt}{{\\c&HFFFFFF&}}" if j == wi else txt)
            lines.append(f"Dialogue: 0,{_t(offset + s)},{_t(offset + end)},Cap,,0,0,0,,{' '.join(parts)}")
    return "\n".join(lines) + "\n"


def write_ass(path: Path, words, hook: str, cfg, offset: float = 0.0, title_box=None, duration=None,
              handle: str = "") -> Path:
    path.write_text(build_ass(words, hook, cfg, offset, title_box, duration, handle), encoding="utf-8")
    return path
