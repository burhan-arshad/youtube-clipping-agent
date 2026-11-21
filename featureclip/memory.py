"""Memory tab: how much disk the clips and work files use, and deleting those FILES while the data stays.

The database (clips, titles, stats, learning, queue, settings) is never touched here, so the assistant keeps everything
it learned. Only the video files go."""
from __future__ import annotations

import shutil
from pathlib import Path

from .cleanup import _size, free_gb

DONE = ("uploaded", "rejected", "dropped", "error")


def _dir_info(p: Path) -> dict:
    files = [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else []
    return {"path": str(p), "files": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 1)}


def usage(cfg, store) -> dict:
    work, out = cfg.path("paths.workdir"), cfg.path("paths.output")
    db = cfg.path("paths.db")
    bk = cfg.root / "data" / "backups"
    clips = store.clips()
    with_file = sum(1 for c in clips if c["path"] and Path(c["path"]).is_file())
    return {"clips": _dir_info(out), "work": _dir_info(work), "backups": _dir_info(bk),
            "db_mb": round((_size(db) if db.is_file() else 0) / 1e6, 1),
            "clip_rows": len(clips), "clip_files": with_file,
            "waiting": sum(1 for c in clips if c["status"] in ("ready", "approved") and c["path"] and Path(c["path"]).is_file()),
            "free_gb": round(free_gb(work if work.exists() else cfg.root), 1)}


def _wipe(p: Path) -> tuple[int, float]:
    n, size = 0, 0
    if not p.is_dir():
        return 0, 0.0
    for f in list(p.iterdir()):
        n += sum(1 for x in f.rglob("*") if x.is_file()) if f.is_dir() else 1
        size += _size(f)
        shutil.rmtree(f, ignore_errors=True) if f.is_dir() else f.unlink(missing_ok=True)
    return n, size / 1e6


def delete(cfg, store, what: str, only_done: bool = True) -> dict:
    """what = 'clips' (the finished mp4 + thumbnails) or 'work' (downloads, recordings, caches)."""
    if what == "work":
        n, mb = _wipe(cfg.path("paths.workdir"))
        return {"files": n, "freed_mb": round(mb, 1)}
    if what != "clips":
        raise ValueError("what must be 'clips' or 'work'")
    out = cfg.path("paths.output")
    n, size = 0, 0
    for c in store.clips():
        if only_done and c["status"] not in DONE:
            continue
        if not c["path"]:
            continue
        p = Path(c["path"])
        for f in (list(p.parent.glob(p.stem + "*")) if p.parent.is_dir() else []):
            if f.is_file():
                size += f.stat().st_size
                f.unlink(missing_ok=True)
                n += 1
        fields = {"path": ""}
        if c["status"] in ("ready", "approved"):          # its file is gone: it must be rebuilt before it can go to YouTube
            fields.update(status="rejected", error="file deleted in the Memory tab - press Rebuild to make it again")
        store.update_clip(c["id"], **fields)
    if not only_done:
        n2, mb2 = _wipe(out)
        n, size = n + n2, size + mb2 * 1e6
    return {"files": n, "freed_mb": round(size / 1e6, 1)}
