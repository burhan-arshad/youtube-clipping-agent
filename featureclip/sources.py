"""sources.yaml: optional named footage sources (url or local path). You may also pass a URL or file path directly."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
    """A name from sources.yaml, or a URL, or a local video path."""
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
        raise KeyError(f"'{name}' is not a source name from sources.yaml, not a link (https://...) and not an existing file")
    sources[src.id] = src
    return src


def load_sources(path: str | Path) -> dict[str, Source]:
    p = Path(path)
    if not p.is_file():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, Source] = {}
    for item in raw.get("sources", []):
        known = {k: item[k] for k in item if k in Source.__dataclass_fields__ and k != "extra"}
        src = Source(**known)
        out[src.id] = src
    return out
