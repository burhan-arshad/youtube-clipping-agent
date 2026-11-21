"""Autopilot: a text file of video links, worked through one link per day without you.

links.txt (one link per line, '#' = comment; optional "| streamer name | facecam corner"):
    https://www.youtube.com/watch?v=XXXX | Ducky Bhai | top-right
    https://www.youtube.com/watch?v=YYYY

Every day at `autopilot.run_at` (or as soon as the laptop is on after that time) the next link is taken and the SAME video
is made twice: N voice-over clips (voice-over channel) and M streamer reaction clips (streamer channel). The assistant then
does what it always does: AI quality check, auto-approve, next free publish slot, stats, learning. You are told about
the result of every link, about errors, and when the links are running out.

Recorded videos only: a live stream link is skipped with a message (use the Assistant tab for live recording)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

KEY = "autopilot_state"
BUILT = ("ready", "approved", "uploaded", "rejected", "judged", "rendered")


# ---------------------------------------------------------------- the links file
def link_key(url: str) -> str:
    return hashlib.sha1(url.strip().rstrip("/").encode()).hexdigest()[:10]


def parse_links(text: str) -> list[dict]:
    out, seen = [], set()
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("﻿")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        url = parts[0]
        if not url.lower().startswith(("http://", "https://")):
            continue
        k = link_key(url)
        if k in seen:
            continue
        seen.add(k)
        out.append({"key": k, "url": url, "streamer": parts[1] if len(parts) > 1 else "",
                    "facecam": parts[2] if len(parts) > 2 else ""})
    return out


def links_path(cfg) -> Path:
    p = Path(str(cfg.get("autopilot.links_file", "links.txt")))
    return p if p.is_absolute() else cfg.root / p


def read_links(cfg) -> list[dict]:
    try:
        return parse_links(links_path(cfg).read_text(encoding="utf-8"))
    except Exception:
        return []


def add_links(cfg, text: str) -> int:
    """Append new links (only the ones not in the file yet). Returns how many were added."""
    have = {l["key"] for l in read_links(cfg)}
    new = [l for l in parse_links(text) if l["key"] not in have]
    if not new:
        return 0
    f = links_path(cfg)
    f.parent.mkdir(parents=True, exist_ok=True)
    prev = f.read_text(encoding="utf-8") if f.is_file() else "# one video link per line: https://... | streamer name | facecam corner (optional)\n"
    lines = []
    for l in parse_links(text):
        if l["key"] in have:
            continue
        lines.append(" | ".join([l["url"]] + ([l["streamer"]] if l["streamer"] or l["facecam"] else []) + ([l["facecam"]] if l["facecam"] else [])))
    f.write_text(prev.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return len(new)


# ---------------------------------------------------------------- state
def load_state(store) -> dict:
    try:
        st = json.loads(store.kv_get(KEY, "") or "null") or {}
    except Exception:
        st = {}
    st.setdefault("links", {})
    st.setdefault("started", {})
    return st


def save_state(store, st: dict) -> None:
    store.kv_set(KEY, json.dumps(st))


def _run_at(cfg) -> tuple[int, int]:
    try:
        h, m = str(cfg.get("autopilot.run_at", "09:00")).split(":")
        return int(h), int(m)
    except Exception:
        return 9, 0


def due_today(cfg, st: dict, now: float | None = None) -> bool:
    lt = time.localtime(now or time.time())
    today = time.strftime("%Y-%m-%d", lt)
    if (lt.tm_hour, lt.tm_min) < _run_at(cfg):
        return False
    return int(st["started"].get(today, 0)) < int(cfg.get("autopilot.links_per_day", 1))


def fetch_streamer_name(url: str, cfg) -> str:
    """The channel that published the video (used for the credit line) when the file does not say."""
    try:
        from .live import ytdlp_extra
        cmd = [sys.executable, "-m", "yt_dlp", "--no-playlist", "--skip-download", "--print", "%(channel)s", *ytdlp_extra(cfg), url]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        name = (p.stdout or "").strip().splitlines()[-1].strip() if p.returncode == 0 and p.stdout.strip() else ""
        return "" if name in ("NA", "None") else name
    except Exception:
        return ""


# ---------------------------------------------------------------- the daily step
def _clips_made(store, source_id: str) -> int:
    vids = {v["id"] for v in store.videos() if v["source_id"] == source_id}
    return sum(1 for c in store.clips() if c["video_id"] in vids and c["status"] in BUILT)


def tick(cfg, store, notifier, now: float | None = None, is_live_fn=None, streamer_fn=None) -> None:
    if not cfg.get("autopilot.enabled", False):
        return
    from .web.jobs import source_for
    now = now or time.time()
    today = time.strftime("%Y-%m-%d", time.localtime(now))
    st = load_state(store)
    links = read_links(cfg)
    for l in links:                                         # new lines in the file join the list
        st["links"].setdefault(l["key"], {"url": l["url"], "status": "pending", "attempts": 0})
        st["links"][l["key"]].update(streamer=l["streamer"], facecam=l["facecam"])
    order = [l["key"] for l in links]
    n_vo, n_st = int(cfg.get("autopilot.voiceover_clips", 5)), int(cfg.get("autopilot.streamer_clips", 5))

    # 1. finish the link that is running
    for k, d in st["links"].items():
        if d["status"] != "running":
            continue
        srcs = {s["id"]: s for s in store.agent_sources()}
        a, b = srcs.get(d.get("vo_sid")), srcs.get(d.get("st_sid"))
        if a is None or b is None:                          # removed from the queue by hand
            d["status"] = "failed"
            d["error"] = "its queue entries were removed"
            continue
        if a["status"] not in ("done", "error") or b["status"] not in ("done", "error"):
            continue
        vo = _clips_made(store, source_for(d["url"], mode="voiceover").id)
        sc = _clips_made(store, source_for(d["url"], mode="streamer").id)
        d.update(voiceover=vo, streamer_n=sc, finished=now)
        err = "; ".join(x for x in ((a.get("note") or ""), (b.get("note") or "")) if x)[:300]
        if vo + sc == 0:
            if int(d.get("attempts", 0)) < 2:
                d["status"] = "pending"
                notifier.send("warn", "Autopilot: no clips from a link, will try again",
                              f"{d['url'][:70]}\n{err or 'nothing worth clipping was found'}")
            else:
                d["status"] = "failed"
                d["error"] = err or "nothing worth clipping was found"
                notifier.send("error", "Autopilot: link failed twice, skipped", f"{d['url'][:70]}\n{d['error']}")
        else:
            d["status"] = "done"
            short = []
            if vo < n_vo:
                short.append(f"voice-over {vo}/{n_vo}")
            if sc < n_st:
                short.append(f"streamer {sc}/{n_st}")
            notifier.send("ok" if not short else "warn", "Autopilot: link finished",
                          f"{d['url'][:70]}\nVoice-over clips: {vo}, streamer clips: {sc}. They go through the quality check and get "
                          f"publish slots automatically." + (f"\nFewer than wanted: {', '.join(short)}." if short else ""))

    # 2. start the next link (once per day, after run_at; also when the laptop was off at that time)
    running = any(d["status"] == "running" for d in st["links"].values())
    pending = [k for k in order if st["links"][k]["status"] == "pending"]
    if not running and pending and due_today(cfg, st, now):
        k = pending[0]
        d = st["links"][k]
        live = (is_live_fn or _is_live)(d["url"], cfg)
        if live:
            d["status"] = "skipped"
            d["error"] = "live stream"
            notifier.send("warn", "Autopilot: link skipped (live stream)",
                          f"{d['url'][:70]}\nAutopilot only works on recorded videos. Record live streams from the Assistant tab.")
        else:
            who = d.get("streamer") or (streamer_fn or fetch_streamer_name)(d["url"], cfg)
            d["streamer"] = who
            base = {"facecam": d.get("facecam", ""), "autopilot": k}
            d["vo_sid"] = store.add_agent_source(d["url"], "video", f"[autopilot voice-over] {d['url'][-40:]}", who, "",
                                                 {**base, "mode": "voiceover", "clips": n_vo})
            d["st_sid"] = store.add_agent_source(d["url"], "video", f"[autopilot streamer] {d['url'][-40:]}", who, "",
                                                 {**base, "mode": "streamer", "clips": n_st})
            d["status"], d["started"], d["attempts"] = "running", now, int(d.get("attempts", 0)) + 1
            st["started"][today] = int(st["started"].get(today, 0)) + 1
            pending = [x for x in pending if x != k]
            notifier.send("info", "Autopilot started a link",
                          f"{d['url'][:70]}\nStreamer: {who or 'unknown'} | {n_vo} voice-over + {n_st} streamer clips")
        # keep going next tick for the next link of the day

    # 3. running out of links
    left = len([k for k in order if st["links"][k]["status"] == "pending"])
    running = any(d["status"] == "running" for d in st["links"].values())
    thr = int(cfg.get("autopilot.warn_links_left", 3))
    if left == 0 and running:
        if st.get("warned_last") != "last":
            st["warned_last"] = "last"
            notifier.send("warn", "Autopilot: this is the LAST link", "No links are left after the current one. Add new links to "
                          f"{links_path(cfg).name} (or in the Assistant tab) to keep the channels fed.")
    elif left == 0 and not running and st.get("empty_day") != today and (time.localtime(now).tm_hour, time.localtime(now).tm_min) >= _run_at(cfg):
        st["empty_day"] = today
        notifier.send("warn", "Autopilot: no links left", f"Nothing to do today. Add links to {links_path(cfg).name} (or in the Assistant tab).")
    elif 0 < left <= thr and st.get("warned_left") != left:
        st["warned_left"] = left
        notifier.send("info", f"Autopilot: only {left} link(s) left", "Add more links soon so the channels do not run dry.")
    if left > thr:
        st["warned_left"], st["warned_last"] = None, None
    save_state(store, st)


def _is_live(url: str, cfg) -> bool:
    try:
        from .live import is_live
        return bool(is_live(url, cfg)) or "/live" in url.lower()
    except Exception:
        return False


# ---------------------------------------------------------------- for the UI
def status(cfg, store) -> dict:
    st = load_state(store)
    links = read_links(cfg)
    items = []
    for l in links:
        d = st["links"].get(l["key"], {"status": "pending"})
        items.append({"key": l["key"], "url": l["url"], "streamer": d.get("streamer") or l["streamer"], "status": d.get("status", "pending"),
                      "voiceover": d.get("voiceover"), "streamer_n": d.get("streamer_n"), "error": d.get("error", "")})
    count = lambda s: sum(1 for i in items if i["status"] == s)
    h, m = _run_at(cfg)
    return {"enabled": bool(cfg.get("autopilot.enabled", False)), "assistant_on": bool(cfg.get("assistant.enabled", False)),
            "file": str(links_path(cfg)), "run_at": f"{h:02d}:{m:02d}", "total": len(items), "pending": count("pending"),
            "running": count("running"), "done": count("done"), "failed": count("failed"), "skipped": count("skipped"),
            "voiceover_clips": int(cfg.get("autopilot.voiceover_clips", 5)), "streamer_clips": int(cfg.get("autopilot.streamer_clips", 5)),
            "items": items[-12:] if len(items) > 12 else items}


def retry(store, key: str) -> bool:
    st = load_state(store)
    d = st["links"].get(key)
    if not d or d["status"] in ("running",):
        return False
    d.update(status="pending", attempts=0, error="")
    save_state(store, st)
    return True
