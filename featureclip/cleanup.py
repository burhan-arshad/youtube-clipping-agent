"""Disk cleanup so a laptop never fills up. Deletes only things that can be rebuilt or are already used:

  * live recordings and downloaded sources, once all clips of that video are made (and old enough)
  * per-clip temporary folders of clips that are finished, rejected or dropped
  * finished mp4/jpg files some days after they were uploaded (the YouTube copy is the master)
Never touches: clips waiting to be built, clips waiting for review/upload, files modified in the last 3 minutes.
"""
from __future__ import annotations

import shutil
import time
from pathlib import Path

PENDING_VIDEO = ("candidate", "judged", "rendered")


def free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(path).free / 1e9
    except Exception:
        return 999.0


def _size(p: Path) -> int:
    try:
        if p.is_file():
            return p.stat().st_size
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return 0


def _rm(p: Path) -> None:
    if p.is_dir():
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            p.unlink()
        except Exception:
            pass


def run(cfg, store, dry_run: bool = False) -> dict:
    work, out = cfg.path("paths.workdir"), cfg.path("paths.output")
    now = time.time()
    low = free_gb(work) < float(cfg.get("cleanup.min_free_gb", 4))
    raw_age = (0.25 if low else float(cfg.get("cleanup.raw_hours", 12))) * 3600
    keep_out = (1 if low else float(cfg.get("cleanup.keep_output_days", 7))) * 86400
    todo: list[tuple[Path, str]] = []

    def old(p: Path, age: float) -> bool:
        try:
            return now - p.stat().st_mtime > age
        except Exception:
            return False

    videos = {v["id"]: v for v in store.videos()}
    busy_paths: set[str] = set()
    clips_by_video: dict[int, list] = {}
    for c in store.clips():
        clips_by_video.setdefault(c["video_id"], []).append(c)
    for vid, cs in clips_by_video.items():
        for c in cs:
            unreviewed_ready = c["status"] == "ready" and not c["review"]
            if c["status"] in PENDING_VIDEO or unreviewed_ready:
                busy_paths.add(str(videos[vid]["path"]))
    busy_stems = {Path(x).with_suffix("") for x in busy_paths}
    registered = {Path(v["path"]).with_suffix("") for v in videos.values()}
    # raw recordings / downloads that live inside the work folder (a source file chosen by you is never touched)
    for v in videos.values():
        p = Path(v["path"])
        if p.is_file() and work in p.parents and p.with_suffix("") not in busy_stems and old(p, raw_age) \
                and "downloads" not in p.parts:
            todo.append((p, "raw video"))
    live = work / "live"
    if live.is_dir():
        for p in live.rglob("seg_*"):
            if not p.is_file() or p.with_suffix("") in busy_stems:
                continue
            # pieces that were never processed may still wait in the worker queue: keep them at least 6 h
            limit = raw_age if p.with_suffix("") in registered else max(raw_age, 6 * 3600)
            if old(p, limit):
                todo.append((p, "live segment"))
    dl = work / "downloads"
    if dl.is_dir():
        for d in dl.iterdir():
            if d.is_dir() and old(d, raw_age):
                vids = [x for x in videos.values() if str(x["path"]).startswith(str(d))]
                if not any(str(x["path"]) in busy_paths for x in vids):
                    todo.append((d, "downloaded video"))
    # per-clip temp folders
    for vid, cs in clips_by_video.items():
        for c in cs:
            if c["status"] in ("uploaded", "rejected", "dropped", "error"):
                d = work / f"video_{vid}" / f"clip_{c['id']}"
                if d.is_dir() and old(d, 3600):
                    todo.append((d, "clip temp"))
    # finished output files
    for c in store.clips():
        if c["status"] == "uploaded" and c["path"]:
            t = c["scheduled_at"] or c["updated"]
            if now - t > keep_out and now > t:
                for f in (Path(c["path"]), Path(c["path"]).with_suffix(".jpg")):
                    if f.is_file():
                        todo.append((f, "old output"))
        elif c["status"] in ("rejected", "dropped") and c["path"] and now - c["updated"] > 2 * 86400:
            f = Path(c["path"])
            if f.is_file():
                todo.append((f, "rejected output"))
    seen, freed, items = set(), 0, []
    for p, why in todo:
        if str(p) in seen or not p.exists():
            continue
        seen.add(str(p))
        sz = _size(p)
        freed += sz
        items.append({"path": str(p), "why": why, "mb": round(sz / 1e6, 1)})
        if not dry_run:
            _rm(p)
    after = free_gb(work)
    return {"freed_mb": round(freed / 1e6, 1), "files": len(items), "items": items[:50], "dry_run": dry_run,
            "free_gb": round(after, 1), "low_disk": low}
