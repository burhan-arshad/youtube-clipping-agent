"""A footage source: a video link or a local file."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Source:
    id: str
    streamer: str
    url: str = ""                       # video URL (yt-dlp) ...
    path: str = ""                      # ... or a local file
    credit_url: str = ""                # shown in the video description
    language: str = "en"
    facecam: str = ""                   # top-left | top-right | bottom-left | bottom-right | none
    crop: list[int] | None = None       # optional manual gameplay rectangle [x, y, w, h]
    notes: str = ""
    extra: dict = field(default_factory=dict)


def resolve_source(name: str, sources: dict) -> "Source":
    """A known source id, or a URL, or a local video path."""
    import hashlib
    from pathlib import Path as _P
    if name in sources:
        return sources[name]
    h = hashlib.sha1(name.encode()).hexdigest()[:8]
    if name.lower().startswith(("http://", "https://")):
        src = Source(id=f"web-{h}", streamer="", url=name)
    elif _P(name.strip('"')).expanduser().is_file():
        p = _P(name.strip('"')).expanduser()
        src = Source(id=f"file-{h}", streamer="", path=str(p))
    else:
        raise KeyError(f"'{name}' is not a known source, not a link (https://...) and not an existing file")
    sources[src.id] = src
    return src
