"""sources.yaml: where footage comes from, and the PERMISSION GATE.

A source can only be processed if permission is 'granted' and a proof note is
filled in. This is deliberate: the tool refuses to touch footage you have no
written OK for.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Source:
    id: str
    streamer: str
    permission: str = "pending"         # granted | pending | denied
    permission_proof: str = ""          # link / screenshot name / DM date: your evidence
    url: str = ""                       # video URL (yt-dlp) ...
    path: str = ""                      # ... or a local file
    credit_url: str = ""                # shown in the video description
    language: str = "en"
    crop: list[int] | None = None       # optional manual gameplay rectangle [x, y, w, h]
    notes: str = ""
    extra: dict = field(default_factory=dict)

    def check_permission(self) -> None:
        if self.permission != "granted":
            raise PermissionError(
                f"source '{self.id}' ({self.streamer}): permission is '{self.permission}'. "
                "Get an explicit OK first, then set permission: granted in sources.yaml."
            )
        if not self.permission_proof.strip():
            raise PermissionError(
                f"source '{self.id}': permission_proof is empty. "
                "Write where the OK is (DM date, e-mail, link) so you can show it later."
            )


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
