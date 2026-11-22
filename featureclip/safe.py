"""Safe mode: one switch that stops everything that could publish or start new work.

ON  -> no uploads (manual or assistant), the assistant starts no new links and approves nothing;
       reviewing, building and everything already on YouTube stay as they are.
"Pull back" (optional, needs upload.manage_videos) also turns videos that are scheduled but not yet public
back to plain private, so nothing goes out while you investigate. "Release" schedules them again.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from . import schedule, uploader


def is_on(store) -> bool:
    return store.kv_get("safe_mode", "") == "1"


def info(store) -> dict:
    return {"on": is_on(store), "reason": store.kv_get("safe_reason", ""),
            "since": float(store.kv_get("safe_since", "0") or 0), "held": json.loads(store.kv_get("safe_held", "[]") or "[]")}


def turn_on(store, reason: str = "switched on by you") -> None:
    store.kv_set("safe_mode", "1")
    store.kv_set("safe_reason", reason)
    store.kv_set("safe_since", str(time.time()))


def turn_off(store) -> None:
    store.kv_set("safe_mode", "")
    store.kv_set("safe_reason", "")
    store.kv_set("fail_streak", "0")
    store.kv_set("reject_streak", "0")


def _set(setter, cfg, clip, *args):
    """set_visibility for one clip."""
    return setter(cfg, clip["youtube_id"], *args)


def pull_back(cfg, store, setter=None) -> dict:
    """Make every not-yet-public scheduled video plain private. Returns {pulled, failed}."""
    if not cfg.get("upload.manage_videos", False) and setter is None:
        raise RuntimeError("Turn on 'Let safe mode pull scheduled videos back' in Settings and sign in again "
                           "(YouTube tab), then try again.")
    setter = setter or uploader.set_visibility
    now = time.time()
    held = json.loads(store.kv_get("safe_held", "[]") or "[]")
    pulled, failed = 0, []
    for c in store.clips(status="uploaded"):
        if c["youtube_id"] and c["scheduled_at"] and c["scheduled_at"] > now and c["id"] not in held:
            try:
                _set(setter, cfg, c, None)
                held.append(c["id"])
                store.update_clip(c["id"], publish_state="held private (safe mode)")
                pulled += 1
            except Exception as e:
                failed.append({"clip": c["id"], "error": str(e)[:200]})
    store.kv_set("safe_held", json.dumps(held))
    return {"pulled": pulled, "failed": failed}


def release(cfg, store, setter=None) -> dict:
    """Turn safe mode off and schedule the held videos again at the next free slots."""
    setter = setter or uploader.set_visibility
    held = json.loads(store.kv_get("safe_held", "[]") or "[]")
    slots = schedule.parse_slots(cfg.get("schedule.slots", schedule.DEFAULT_SLOTS)) or schedule.parse_slots(schedule.DEFAULT_SLOTS)
    now = datetime.now(timezone.utc)
    taken = [datetime.fromtimestamp(c["scheduled_at"], timezone.utc) for c in store.clips(status="uploaded")
             if c["scheduled_at"] and c["id"] not in held and c["scheduled_at"] > now.timestamp()]
    done, failed, keep = 0, [], []
    for cid in held:
        c = store.clip(cid)
        nxt = schedule.next_slots(now, 1, slots, taken)
        if not nxt:
            keep.append(cid)
            continue
        try:
            _set(setter, cfg, c, schedule.iso_utc(nxt[0]))
            taken.append(nxt[0])
            store.update_clip(cid, scheduled_at=nxt[0].timestamp(), publish_state=f"scheduled {schedule.fmt_pkt(nxt[0])}")
            done += 1
        except Exception as e:
            keep.append(cid)
            failed.append({"clip": cid, "error": str(e)[:200]})
    store.kv_set("safe_held", json.dumps(keep))
    turn_off(store)
    return {"rescheduled": done, "failed": failed}
