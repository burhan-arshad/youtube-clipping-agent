"""One-time clean-up of the discontinued "streamer clips" channel.

Removes: every clip (rows + mp4/thumbnail/unsynced files) that was made in streamer mode, the queue entries that asked for
streamer clips, and the old streamer-only settings stored in the database. The voice-over clips are never touched.
A backup of the database/settings is made first. Safe to run again (it then finds nothing)."""
from __future__ import annotations

import json
from pathlib import Path

SUFFIX = "-streamer"          # streamer-mode sources were stored as "<id>-streamer"
OLD_KV = ("playbook:streamer", "upload_pause_until:streamer")


def find(cfg, store) -> dict:
    vids = [v for v in store.videos() if str(v["source_id"]).endswith(SUFFIX)]
    vid_ids = {v["id"] for v in vids}
    clips = [c for c in store.clips() if c["video_id"] in vid_ids]
    files = []
    for c in clips:
        if c["path"]:
            p = Path(c["path"])
            if p.parent.is_dir():
                files += [f for f in p.parent.glob(p.stem + "*") if f.is_file()]
    queue = []
    for s in store.agent_sources():
        try:
            mode = json.loads(s.get("options") or "{}").get("mode")
        except ValueError:
            mode = None
        if mode == "streamer" or "[autopilot streamer]" in str(s.get("name", "")):
            queue.append(s["id"])
    kv = [k for k in OLD_KV if store.kv_get(k, "")]
    return {"videos": [v["id"] for v in vids], "clips": [c["id"] for c in clips], "files": files, "queue": queue, "kv": kv,
            "on_youtube": sum(1 for c in clips if c["youtube_id"])}


def remove(cfg, store, apply: bool = False) -> dict:
    f = find(cfg, store)
    r = {"clips": len(f["clips"]), "videos": len(f["videos"]), "files": len(f["files"]), "queue": len(f["queue"]),
         "kv": len(f["kv"]), "on_youtube": f["on_youtube"], "applied": False,
         "mb": round(sum(p.stat().st_size for p in f["files"] if p.exists()) / 1e6, 1)}
    if not apply or not (f["clips"] or f["videos"] or f["queue"] or f["kv"]):
        return r
    from . import backup
    try:
        backup.run(cfg)
    except Exception:
        pass
    for p in f["files"]:
        p.unlink(missing_ok=True)
    db = store.db
    for cid in f["clips"]:
        db.execute("DELETE FROM clip_stats WHERE clip_id=?", (cid,))
        db.execute("DELETE FROM clips WHERE id=?", (cid,))
    for vid in f["videos"]:
        db.execute("DELETE FROM clips WHERE video_id=?", (vid,))
        db.execute("DELETE FROM videos WHERE id=?", (vid,))
    for sid in f["queue"]:
        db.execute("DELETE FROM agent_sources WHERE id=?", (sid,))
    for k in f["kv"]:
        db.execute("DELETE FROM kv WHERE key=?", (k,))
    db.commit()
    r["applied"] = True
    return r


def auto(cfg, store) -> dict | None:
    """Run once, silently, the first time the new version starts."""
    if store.kv_get("streamer_removed", ""):
        return None
    r = remove(cfg, store, apply=True)
    store.kv_set("streamer_removed", "1")
    return r
