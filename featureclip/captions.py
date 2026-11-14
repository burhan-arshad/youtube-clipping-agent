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


def build_ass(words: list[tuple[float, float, str]], hook: str, cfg, offset: float = 0.0) -> str:
    w, h = int(cfg.get("video.width")), int(cfg.get("video.height"))
    font = cfg.get("captions.font")
    size = int(cfg.get("captions.size"))
    margin_v = int(cfg.get("captions.margin_bottom"))
    per = max(1, int(cfg.get("captions.words_per_chunk", 3)))
    hook_secs = float(cfg.get("captions.hook_seconds", 2.6))

    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {w}", f"PlayResY: {h}", "WrapStyle: 2",
        "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Cap,{font},{size},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,2,2,60,60,{margin_v},1",
        f"Style: Hook,{font},{int(size * 1.15)},&H0000E6FF,&H0000E6FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,8,3,8,60,60,230,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    if hook.strip():
        lines.append(f"Dialogue: 1,{_t(offset)},{_t(offset + hook_secs)},Hook,,0,0,0,,"
                     f"{{\\fad(150,200)}}{_esc(hook.strip().upper())}")
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


def write_ass(path: Path, words, hook: str, cfg, offset: float = 0.0) -> Path:
    path.write_text(build_ass(words, hook, cfg, offset), encoding="utf-8")
    return path
