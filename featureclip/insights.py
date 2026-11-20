"""What the assistant learns: quality score history, YouTube results per category and per publish slot,
and slot suggestions. Everything here is plain arithmetic on the database (no AI calls)."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from . import schedule


def _j(text, default):
    try:
        return json.loads(text) if text else default
    except Exception:
        return default


def category_of(c) -> str:
    return str(_j(c["judgement"], {}).get("category") or "unknown")


def score_history(store) -> list[dict]:
    out = []
    for c in store.clips():
        rv = _j(c["review"], None)
        if not rv or "score" not in rv:
            continue
        out.append({"id": c["id"], "ts": rv.get("ts") or c["updated"], "score": round(float(rv["score"]), 1),
                    "ok": bool(rv.get("ok")), "category": category_of(c), "status": c["status"],
                    "title": (c["title"] or "")[:70]})
    return sorted(out, key=lambda r: r["ts"])


def publish_ts(c) -> float | None:
    """When the clip became (or becomes) public. Uploaded without a schedule -> the upload time."""
    if c["status"] != "uploaded":
        return None
    return c["scheduled_at"] or c["updated"]


def slot_key(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).astimezone(schedule.PKT).strftime("%H:%M")


def clip_rows(store, min_age_hours: float = 12.0) -> list[dict]:
    """One row per published clip that has a stats snapshot and is old enough to judge."""
    stats = store.latest_stats()
    now = time.time()
    rows = []
    for c in store.clips(status="uploaded"):
        t = publish_ts(c)
        st = stats.get(c["id"])
        if not t or not st or t > now:
            continue
        age_h = (now - t) / 3600
        if age_h < min_age_hours:
            continue
        views = st["views"] or 0
        sty = _j(c["judgement"], {}).get("styles") or {}
        rows.append({"hook_style": sty.get("hook"), "title_style": sty.get("title"), "id": c["id"], "title": (c["title"] or "")[:70], "category": category_of(c),
                     "slot": slot_key(t), "age_h": round(age_h, 1), "views": views, "likes": st["likes"] or 0,
                     "comments": st["comments"] or 0, "avg_pct": st["avg_view_pct"],
                     "rate": views / max(6.0, min(age_h, 72.0)),          # views per hour, first 3 days at most
                     "score": (_j(c["review"], {}) or {}).get("score")})
    return rows


def _group(rows: list[dict], key: str) -> list[dict]:
    g: dict[str, list[dict]] = {}
    for r in rows:
        g.setdefault(r[key], []).append(r)
    out = []
    for k, rs in g.items():
        pcts = [r["avg_pct"] for r in rs if r["avg_pct"] is not None]
        out.append({"key": k, "n": len(rs), "avg_rate": round(sum(r["rate"] for r in rs) / len(rs), 2),
                    "avg_views": round(sum(r["views"] for r in rs) / len(rs), 1),
                    "avg_likes": round(sum(r["likes"] for r in rs) / len(rs), 1),
                    "avg_pct": round(sum(pcts) / len(pcts), 1) if pcts else None})
    return sorted(out, key=lambda x: -x["avg_rate"])


def report(store, cfg) -> dict:
    mins = float(cfg.get("insights.min_age_hours", 12))
    rows = clip_rows(store, mins)
    hist = score_history(store)
    cats_score: dict[str, list[float]] = {}
    for h in hist:
        cats_score.setdefault(h["category"], []).append(h["score"])
    by_cat = {g["key"]: g for g in _group(rows, "category")}
    categories = []
    for k in sorted(set(cats_score) | set(by_cat)):
        sc = cats_score.get(k, [])
        g = by_cat.get(k, {})
        categories.append({"category": k, "clips": len(sc), "avg_score": round(sum(sc) / len(sc), 1) if sc else None,
                           "published": g.get("n", 0), "avg_views": g.get("avg_views"), "avg_rate": g.get("avg_rate"),
                           "avg_pct": g.get("avg_pct")})
    categories.sort(key=lambda r: -(r["avg_rate"] or 0) - (r["avg_score"] or 0) / 1000)
    slots = _group(rows, "slot")
    slots.sort(key=lambda s: s["key"])
    return {"history": hist[-120:], "categories": categories, "slots": slots,
            "published_with_stats": len(rows), "top": sorted(rows, key=lambda r: -r["rate"])[:5],
            "min_samples": int(cfg.get("insights.min_samples", 3)), "min_age_hours": mins,
            "suggestion": suggest_slots(store, cfg, rows)}


def _gap_ok(slots: list[str]) -> bool:
    mins = sorted(int(s[:2]) * 60 + int(s[3:]) for s in slots)
    if len(mins) < 2:
        return True
    gaps = [(mins[(i + 1) % len(mins)] - m) % 1440 for i, m in enumerate(mins)]
    return min(gaps) >= 180


def suggest_slots(store, cfg, rows: list[dict] | None = None) -> dict:
    """Swap at most one slot per week: first try untested times from the pool (explore), then move the weakest
    slot to a pool time that clearly performed better (exploit). Returns {change, slots, reason}."""
    cur = ["%02d:%02d" % hm for hm in schedule.parse_slots(cfg.get("schedule.slots", schedule.DEFAULT_SLOTS))]
    pool = ["%02d:%02d" % hm for hm in schedule.parse_slots(cfg.get("schedule.pool", []))] or cur
    need = int(cfg.get("insights.min_samples", 3))
    rows = rows if rows is not None else clip_rows(store, float(cfg.get("insights.min_age_hours", 12)))
    stat = {s["key"]: s for s in _group(rows, "slot")}
    n = lambda t: stat.get(t, {}).get("n", 0)
    rate = lambda t: stat.get(t, {}).get("avg_rate", 0.0)
    last = float(store.kv_get("slot_change_ts", "0") or 0)
    res = {"change": False, "slots": cur, "reason": "", "ts_last_change": last}
    if time.time() - last < 7 * 86400:
        res["reason"] = "A slot was changed less than 7 days ago; waiting for more results."
        return res
    tested = [t for t in cur if n(t) >= need]
    # 1) explore: some slot in use has too few samples -> keep, let it collect data first
    if len(tested) < len(cur):
        # a pool time nobody has tried yet is worth testing only after current slots have data
        res["reason"] = f"Collecting data: every slot needs {need}+ published videos with stats before it can be judged."
        return res
    weakest = min(cur, key=rate)
    untried = [t for t in pool if t not in cur and n(t) < need]
    tried_better = [t for t in pool if t not in cur and n(t) >= need and rate(t) > rate(weakest) * 1.25]
    def swapped(t):
        return sorted(cur[:cur.index(weakest)] + [t] + cur[cur.index(weakest) + 1:])
    options = [(t, f"{t} got {rate(t):.1f} views/hour vs {rate(weakest):.1f} for {weakest}.")
               for t in sorted(tried_better, key=rate, reverse=True)]
    options += [(t, f"Testing {t} (no data yet) instead of the weakest slot {weakest} ({rate(weakest):.1f} views/hour).")
                for t in untried]
    if not options:
        res["reason"] = "All slots look fine; nothing better found."
        return res
    pick = next(((t, w) for t, w in options if _gap_ok(swapped(t))), None)
    if not pick:
        res["reason"] = "Every better time would be less than 3 hours from another slot."
        return res
    cand, why = pick
    new = swapped(cand)
    res.update(change=True, slots=new, reason=why, drop=weakest, add=cand)
    return res


def collect(store, cfg, fetch=None) -> int:
    """Download current stats of every published clip (<=30 days old). Returns clips updated."""
    from . import uploader
    now = time.time()
    clips = []
    for c in store.clips(status="uploaded"):
        t = publish_ts(c)
        if c["youtube_id"] and t and t <= now and now - t < 30 * 86400:
            clips.append(c)
    if not clips:
        return 0
    fetch = fetch or uploader.fetch_stats
    from . import channels
    data: dict = {}
    for key in channels.KEYS:                          # each channel is read with its own login
        part = [c["youtube_id"] for c in clips if channels.key_of_clip(c) == key]
        if not part or not channels.logged_in(cfg, key) and fetch is uploader.fetch_stats:
            continue
        try:
            with channels.use(cfg, key):
                data.update(fetch(cfg, part))
        except Exception:
            if key == "voiceover":
                raise
    n = 0
    for c in clips:
        d = data.get(c["youtube_id"])
        if d:
            store.add_stats(c["id"], d["views"], d["likes"], d["comments"], d.get("avg_pct"), d.get("avg_sec"))
            n += 1
    store.kv_set("stats_ts", str(now))
    return n


def apply_slots(root, config_path, slots: list[str], store) -> None:
    from .web import settings_store as ss
    ss.save(root, {"schedule.slots": ",".join(slots)}, config_path)
    store.kv_set("slot_change_ts", str(time.time()))
