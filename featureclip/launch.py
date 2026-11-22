"""Launch mode: the first clips of a real stream day go out close together, then the normal slots continue.

  clip 1 -> public NOW, clip 2 -> +gap minutes, clip 3 -> +2*gap minutes (default gap 60), after that the usual
  schedule.slots (21:00 / 01:00 / 05:00 PKT ...). It can also move the videos that are ALREADY scheduled (test
  uploads) into those launch times. State lives in the database (kv "launch"), so a restart does not lose it."""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

from . import schedule, uploader

LEAD_MINUTES = 12            # YouTube wants a publish time a little in the future


def _load(store) -> dict | None:
    try:
        d = json.loads(store.kv_get("launch", "") or "null")
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def status(store, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    d = _load(store)
    if not d:
        return {"active": False}
    left = [i for i in range(int(d["count"])) if i not in d.get("used", [])]
    active = bool(left) and now < d["t0"] + (d["count"] + 24) * d["gap"] * 60
    return {"active": active, "count": d["count"], "gap": d["gap"], "left": len(left), "t0": d["t0"]}


def start(store, count: int = 3, gap: int = 60, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    count, gap = max(1, min(int(count), 10)), max(5, min(int(gap), 240))
    store.kv_set("launch", json.dumps({"t0": now, "count": count, "gap": gap, "used": []}))
    return status(store, now)


def stop(store) -> None:
    store.kv_set("launch", "")


def next_time(store, now: float | None = None):
    """(index, datetime | None). None = publish right now. Returns None when launch mode is not active."""
    now = now if now is not None else time.time()
    if not status(store, now)["active"]:
        return None
    d = _load(store)
    idx = next(i for i in range(int(d["count"])) if i not in d.get("used", []))
    if idx == 0:
        return idx, None
    t = max(d["t0"] + idx * d["gap"] * 60, now + LEAD_MINUTES * 60)
    return idx, datetime.fromtimestamp(t, timezone.utc)


def mark_used(store, idx: int) -> None:
    d = _load(store)
    if d and idx not in d.get("used", []):
        d.setdefault("used", []).append(idx)
        store.kv_set("launch", json.dumps(d))


from .safe import _set  # noqa: E402


def reschedule_existing(cfg, store, setter=None, now: float | None = None) -> dict:
    """Move already scheduled (not yet public) uploads: the first ones into the launch times, the rest into the
    normal slots. Returns {moved, failed}."""
    now = now if now is not None else time.time()
    if setter is None and not cfg.get("upload.manage_videos", False):
        raise RuntimeError("To move videos that are already scheduled, turn on 'Let safe mode pull scheduled videos back' "
                           "in Settings, restart, and sign in again in the YouTube tab. Then press the launch button again.")
    setter = setter or uploader.set_visibility
    held = json.loads(store.kv_get("safe_held", "[]") or "[]")
    todo = sorted((c for c in store.clips(status="uploaded")
                   if c["youtube_id"] and c["scheduled_at"] and c["scheduled_at"] > now and c["id"] not in held),
                  key=lambda c: c["scheduled_at"])
    slots = schedule.parse_slots(cfg.get("schedule.slots", schedule.DEFAULT_SLOTS)) or schedule.parse_slots(schedule.DEFAULT_SLOTS)
    nowdt = datetime.fromtimestamp(now, timezone.utc)
    taken: list[datetime] = []
    moved, failed = 0, []
    for c in todo:
        nxt = next_time(store, now)
        try:
            if nxt is not None and nxt[1] is None:                 # launch slot 0: public right away
                _set(setter, cfg, c, None, True)
                store.update_clip(c["id"], scheduled_at=now, publish_state="public now (launch)")
                mark_used(store, nxt[0])
            else:
                when = nxt[1] if nxt is not None else (schedule.next_slots(nowdt, 1, slots, taken) or [None])[0]
                if when is None:
                    continue
                _set(setter, cfg, c, schedule.iso_utc(when))
                store.update_clip(c["id"], scheduled_at=when.timestamp(), publish_state=f"scheduled {schedule.fmt_pkt(when)}")
                taken.append(when)
                if nxt is not None:
                    mark_used(store, nxt[0])
            moved += 1
        except Exception as e:
            failed.append({"clip": c["id"], "error": str(e)[:200]})
    return {"moved": moved, "failed": failed}
