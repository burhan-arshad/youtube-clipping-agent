"""The assistant: does by itself what a human editor would do.

  1. takes the next link from the source queue and starts the recording / clip job
  2. reviews every finished clip in detail (video frames, title, description, thumbnail, length, duplicates)
  3. fixes what can be fixed (new title/description/thumbnail, or one rebuild); rejects what cannot
  4. approves good clips and uploads them to YouTube with a publish time (3 slots per day, Pakistan time)
  5. tells the owner (bell icon, ntfy.sh, Telegram) when the clip stock is running low or something needs attention

It runs in its own thread, so it keeps working while a live stream is being recorded.
"""
from __future__ import annotations

import collections
import json
import threading
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import metadata, schedule
from .config import load_config
from .ffutil import grab_frame, probe
from .llm import LLM, LLMError
from .notify import Notifier
from .store import Store

REVIEW_PROMPT = """You are the final quality reviewer of a YouTube Shorts channel about {game}. The images are frames
of the finished vertical video in time order (the last image is the thumbnail, if there is one).

Title: {title}
Description:
{description}
Voice-over script: {script}
Feature the clip claims to show: {feature}

Check strictly and report every problem:
- video: a real person (streamer/webcam face), chat, donation/follower alert, streamer logo or name visible
- video: black bars, a second window, a menu, a loading screen, or mostly empty/boring gameplay
- video: the heading text at the top is cut off, unreadable or overlapping; captions cut off or unreadable
- script: the voice-over says something that is not visible in the frames
- metadata: the title or description promises something that is not shown, is clickbait that is not true, claims leaks or
  official information, or has spelling mistakes
- thumbnail: unreadable text, misleading, unrelated to the video
- policy: very graphic gore, sexual content, slurs/hate, real-world violence, or anything YouTube would age-restrict or demonetize

Return JSON only:
{{"score": 0-100 overall quality, "issues": [{{"type": "video|script|metadata|thumbnail|policy", "text": "short, specific"}}]}}
An empty issues list means you found nothing wrong."""


def check_basics(c, cfg) -> list[dict]:
    """Deterministic checks. Each item: {name, ok, detail}."""
    out = []
    path = c["path"]
    ok_file = bool(path) and Path(path).is_file()
    out.append({"name": "video file exists", "ok": ok_file, "detail": ""})
    if ok_file:
        try:
            info = probe(path)
            d = info["duration"]
            out.append({"name": "length 8-59 s (Shorts)", "ok": 8 <= d <= 59.5, "detail": f"{d:.1f}s"})
            vertical = info["height"] >= info["width"] * 1.7
            out.append({"name": "vertical 9:16", "ok": vertical, "detail": f"{info['width']}x{info['height']}"})
            out.append({"name": "has audio", "ok": bool(info["has_audio"]), "detail": ""})
        except Exception as e:
            out.append({"name": "video readable", "ok": False, "detail": str(e)[:100]})
    game = str(cfg.get("game.name", "") or "")
    probs = metadata.validate({"title": c["title"] or "", "description": c["description"] or ""}, game)
    out.append({"name": "title + description rules", "ok": not probs, "detail": "; ".join(probs)})
    th = c["thumb"]
    out.append({"name": "thumbnail < 2 MB", "ok": bool(th) and Path(th).is_file() and Path(th).stat().st_size < 2_000_000,
                "detail": "" if th else "missing"})
    handle = str(cfg.get("branding.handle", "") or "").strip()
    if handle:
        out.append({"name": "channel handle set", "ok": True, "detail": handle})
    return out


def review_clip(llm, cfg, c, tmp: Path, store=None) -> dict:
    """Full review of one clip. Returns {ok, score, issues, checks}. Raises LLMError if the AI is unavailable."""
    checks = check_basics(c, cfg)
    issues = []
    for ch in checks:
        if not ch["ok"]:
            t = "metadata" if "title" in ch["name"] else ("thumbnail" if "thumbnail" in ch["name"] else "video")
            issues.append({"type": t, "text": f"{ch['name']}: {ch['detail'] or 'failed'}"})
    score = 100
    if c["path"] and Path(c["path"]).is_file():
        dur = probe(c["path"])["duration"]
        tmp.mkdir(parents=True, exist_ok=True)
        imgs = []
        for i, f in enumerate((0.08, 0.28, 0.5, 0.72, 0.92)):
            dst = tmp / f"rv_{i}.jpg"
            try:
                grab_frame(c["path"], dur * f, dst, 540)
                imgs.append(dst.read_bytes())
            except Exception:
                pass
        if c["thumb"] and Path(c["thumb"]).is_file():
            from PIL import Image
            im = Image.open(c["thumb"]).convert("RGB")
            im.thumbnail((640, 640))
            p = tmp / "rv_thumb.jpg"
            im.save(p, quality=80)
            imgs.append(p.read_bytes())
        j = json.loads(c["judgement"]) if c["judgement"] else {}
        prompt = REVIEW_PROMPT.format(game=cfg.get("game.name", "the game"), title=c["title"], description=c["description"],
                                      script=c["script"], feature=j.get("feature_name", ""))
        raw = llm.json(prompt, images=imgs or None, temperature=0.1)
        try:
            score = float(raw.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        for it in raw.get("issues") or []:
            if isinstance(it, dict):
                t = str(it.get("type", "video")).lower()
                issues.append({"type": t if t in ("video", "script", "metadata", "thumbnail", "policy") else "video",
                               "text": str(it.get("text", ""))[:200]})
    need = 80 if str(cfg.get("assistant.review_strictness", "normal")) == "strict" else 65
    ok = not issues and score >= need
    if issues and score >= need + 15 and all(i["type"] == "thumbnail" for i in issues):
        ok = True                          # a weak thumbnail alone must not block a good Short (Shorts ignore it anyway)
    return {"ok": ok, "score": score, "issues": issues, "checks": checks, "ts": time.time()}


from . import cleanup, insights, safe  # noqa: E402


class Assistant:
    def __init__(self, root: Path, config_path, jobs):
        self.root, self.config_path, self.jobs = Path(root), config_path, jobs
        self.state = {"state": "off", "message": "Assistant is off", "last_tick": None, "tick_error": ""}
        self.events = collections.deque(maxlen=60)          # what the assistant decided, newest last
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True, name="featureclip-assistant")
        self._busy = threading.Lock()

    def start(self) -> None:
        try:
            self.recover()
        except Exception:
            traceback.print_exc()
        self._t.start()

    def recover(self) -> None:
        """After a crash / PC restart no job is running any more: put interrupted links back in the queue."""
        cfg = load_config(self.config_path, root=self.root)
        store = Store(cfg.path("paths.db"))
        try:
            resumed = 0
            for s in store.agent_sources("running"):
                opts = json.loads(s["options"] or "{}")
                opts["restarts"] = int(opts.get("restarts", 0)) + 1
                opts.pop("job", None)
                if opts["restarts"] <= 3:
                    store.update_agent_source(s["id"], status="queued", options=json.dumps(opts),
                                              note="restarted after the program was closed")
                    resumed += 1
                else:
                    store.update_agent_source(s["id"], status="error", finished=time.time(),
                                              note="interrupted 3 times; add the link again if it is still wanted")
            if resumed and cfg.get("assistant.enabled", False):
                Notifier(cfg, store).send("info", "Assistant resumed after a restart",
                                          f"{resumed} interrupted link(s) were put back in the queue.")
        finally:
            store.close()

    def _event(self, level: str, text: str) -> None:
        self.events.append({"ts": time.time(), "level": level, "text": text[:220]})

    def _phase(self, key: str | None, text: str = "") -> None:
        """What the assistant itself is doing right now (shown live in the dashboard)."""
        cur = self.state.get("phase") or {}
        since = cur.get("since", time.time()) if cur.get("key") == key else time.time()
        self.state["phase"] = {"key": key, "text": text, "since": since} if key else None

    def wake(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            every = 60
            try:
                cfg = load_config(self.config_path, root=self.root)
                every = max(10, int(cfg.get("assistant.check_every_seconds", 60)))
                self.tick(cfg)
            except Exception as e:
                self.state["tick_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            self._wake.wait(every)
            self._wake.clear()

    # ------------------------------------------------------------------ one pass
    def tick(self, cfg=None) -> None:
        cfg = cfg or load_config(self.config_path, root=self.root)
        if not cfg.get("assistant.enabled", False):
            self.state.update(state="off", message="Assistant is off (switch it on in the Assistant tab)")
            return
        if not self._busy.acquire(blocking=False):
            return
        store = Store(cfg.path("paths.db"))
        try:
            self.state["tick_error"] = ""
            notifier = Notifier(cfg, store)
            llm = LLM(cfg, store)
            from .web.jobs import load_all_sources
            from .pipeline import Pipeline
            pipe = Pipeline(cfg, store, load_all_sources(self.root))
            self.state.update(state="working", message="Checking sources, reviewing clips, publishing ...")
            safe_on = safe.is_on(store)
            self._phase("source", "Looking at the link queue")
            self._sources(cfg, store, notifier, safe_on)
            self._phase("scan", "Finishing unfinished clips")
            self._resume(cfg, store, pipe)
            self._phase("review", "Checking clips")
            self._review(cfg, store, llm, pipe, notifier, safe_on)
            if not safe_on:
                self._phase("publish", "Scheduling approved clips")
                self._publish(cfg, store, pipe, notifier)
            self._phase("learn", "Reading YouTube results")
            self._learn(cfg, store, notifier)
            self._phase("clean", "Cleaning old files")
            self._housekeeping(cfg, store, notifier)
            self._phase(None)
            self._stock(cfg, store, notifier)
            msg = self._summary(store)
            if safe_on:
                msg = "SAFE MODE ON (" + (safe.info(store)["reason"] or "no reason given") + "). " + msg
            self.state.update(state="idle", message=msg, last_tick=time.time())
        finally:
            self._phase(None)
            store.close()
            self._busy.release()

    def _summary(self, store) -> str:
        n = {s: len(store.clips(status=s)) for s in ("ready", "approved")}
        future = sum(1 for c in store.clips(status="uploaded") if c["scheduled_at"] and c["scheduled_at"] > time.time())
        q = len(store.agent_sources("queued"))
        run = len(store.agent_sources("running"))
        return f"{n['ready']} in review, {n['approved']} approved, {future} scheduled, {q} link(s) queued, {run} running"

    # ------------------------------------------------------------------ 1. sources
    def _sources(self, cfg, store, notifier, safe_on=False) -> None:
        for s in store.agent_sources("running"):
            opts = json.loads(s["options"] or "{}")
            job = self.jobs.get(opts.get("job", -1))
            if job is None or job.status in ("done", "error", "stopped"):
                ok = job is not None and job.status in ("done", "stopped")
                store.update_agent_source(s["id"], status="done" if ok else "error", finished=time.time(),
                                          note=(job.error if job and job.error else ""))
                self._event("ok" if ok else "bad", f"Source finished: {s['name'][:50]}" + ("" if ok else " (failed)"))
                n = len([c for c in store.clips(status="ready")])
                notifier.send("ok" if ok else "error", f"Source finished: {s['name'][:60]}",
                              f"{n} clip(s) waiting for review." if ok else f"Failed: {(job.error if job else 'job lost')}")
        if self.jobs.running() is not None or safe_on:
            return
        queued = store.agent_sources("queued")
        if not queued:
            return
        s = queued[0]
        if s["kind"] == "auto":                       # live or recorded? ask YouTube instead of guessing from the link
            from .live import is_live
            self._phase("source", "Checking whether the link is a live stream")
            kind = "live" if (is_live(s["url"]) or "/live" in s["url"].lower()) else "video"
            store.update_agent_source(s["id"], kind=kind)
            s = dict(s, kind=kind)
            self._event("info", f"Link is a {'LIVE stream' if kind == 'live' else 'recorded video'}: {s['url'][:70]}")
        opts = json.loads(s["options"] or "{}")
        args = dict(video=s["url"], streamer=s["streamer"] or "", credit_url=s["credit_url"] or "",
                    facecam=opts.get("facecam", ""), crop=opts.get("crop"))
        if s["kind"] == "live":
            job = self.jobs.submit("watch", f"LIVE (assistant): {s['url'][:60]}", minutes=opts.get("minutes", 5), **args)
        else:
            job = self.jobs.submit("run", f"Make clips (assistant): {s['url'][:60]}", **args)
        opts["job"] = job.id
        store.update_agent_source(s["id"], status="running", started=time.time(), options=json.dumps(opts))
        notifier.send("info", "Assistant started a source", f"{s['kind']}: {s['url'][:80]}")
        self._event("ok", f"Started {s['kind']}: {s['url'][:70]}")

    # ------------------------------------------------------------------ 2+3. review and fix
    def _review(self, cfg, store, llm, pipe, notifier, safe_on=False) -> None:
        mode = "ask" if safe_on else str(cfg.get("assistant.mode", "auto"))
        max_rebuilds = int(cfg.get("assistant.max_rebuilds", 1))
        for c in store.clips(status="ready"):
            if c["review"]:
                continue
            tmp = pipe.cdir(c["video_id"], c["id"]) / "review"
            self._phase("review", f"Checking clip {c['id']}: {(c['title'] or '')[:60]}")
            try:
                rv = review_clip(llm, cfg, c, tmp, store)
            except LLMError as e:
                self.state["message"] = f"Waiting for the AI quota ({str(e)[:60]}); will retry"
                return                                    # keeps the clip for the next pass, never skips it
            attempts = int(store.kv_get(f"rv:{c['id']}", "0"))
            rv["attempt"] = attempts + 1
            if rv["ok"]:
                store.update_clip(c["id"], review=rv)
                self._event("ok", f"Clip {c['id']} passed the check (score {round(rv['score'])})"
                            + (": approved" if mode == "auto" else ": waiting for your approval"))
                if mode == "auto":
                    store.update_clip(c["id"], status="approved")
                    store.kv_set("reject_streak", "0")
                continue
            types = {i["type"] for i in rv["issues"]}
            summary = "; ".join(i["text"] for i in rv["issues"])[:300]
            store.kv_set(f"rv:{c['id']}", str(attempts + 1))
            self._event("warn", f"Clip {c['id']} has problems: {summary[:120]}")
            if types <= {"metadata", "thumbnail"} and attempts < 2 and c["judgement"]:
                self._event("info", f"Clip {c['id']}: writing a new title/description/thumbnail")
                self._fix_metadata(cfg, store, llm, pipe, c)
                store.update_clip(c["id"], review=None)       # review again next pass
                continue
            if types & {"video", "script"} and "policy" not in types and c["rebuilds"] < max_rebuilds and mode == "auto":
                store.update_clip(c["id"], status="judged", rebuilds=c["rebuilds"] + 1, error=None, script=None,
                                  review=None)
                self._phase("build", f"Re-making clip {c['id']} to fix: {summary[:80]}")
                self._event("info", f"Clip {c['id']}: re-making the video")
                try:
                    pipe.build_clip(c["id"])
                except LLMError:
                    return
                except Exception as e:
                    store.update_clip(c["id"], status="error", error=f"rebuild: {e}")
                continue
            store.update_clip(c["id"], review=rv)
            if mode == "auto":
                store.update_clip(c["id"], status="rejected", error="assistant review: " + summary)
                self._event("bad", f"Clip {c['id']} rejected")
                notifier.send("warn", f"Assistant rejected clip {c['id']}", f"{c['title'] or ''}\n{summary}")
                streak = int(store.kv_get("reject_streak", "0") or 0) + 1
                store.kv_set("reject_streak", str(streak))
                if cfg.get("safe.auto", True) and streak >= int(cfg.get("safe.reject_streak", 8)):
                    safe.turn_on(store, f"{streak} clips in a row were rejected: quality problem")
                    notifier.send("error", "Safe mode switched on", f"{streak} rejected clips in a row. Check the source or settings, then switch safe mode off.")

    def _fix_metadata(self, cfg, store, llm, pipe, c) -> None:
        j = json.loads(c["judgement"])
        video = store.video(c["video_id"])
        from .sources import resolve_source
        src = pipe.sources.get(video["source_id"])
        meta = metadata.generate(llm, cfg, j, c["script"] or "", "", src.streamer if src else "",
                                 src.credit_url if src else "")
        fields = {"title": meta["title"], "description": meta["description"], "tags": meta["tags"]}
        try:
            probe_d = probe(c["path"])["duration"]
            frame = metadata.pick_frame(c["path"], 0, probe_d, None, pipe.cdir(c["video_id"], c["id"]) / "thumb2")
            fields["thumb"] = str(metadata.make_thumbnail(frame, Path(c["path"]).with_suffix(".jpg"), metadata.thumb_line(meta, "", meta["title"]), cfg))
        except Exception:
            pass
        store.update_clip(c["id"], **fields)

    # ------------------------------------------------------------------ 4. publish
    def _publish(self, cfg, store, pipe, notifier) -> None:
        if str(cfg.get("assistant.mode", "auto")) != "auto":
            return
        if not store.clips(status="approved"):
            return
        tok = Path(str(cfg.get("upload.token", "token.json")))
        if not (tok if tok.is_absolute() else cfg.root / tok).is_file():
            if not store.kv_get("warned_login"):
                notifier.send("warn", "YouTube is not connected",
                              "Open the YouTube tab, upload client_secret.json and log in. Approved clips are waiting.")
                store.kv_set("warned_login", "1")
            return
        store.kv_set("warned_login", "")
        pipe.upload()
        for r in getattr(pipe, "upload_results", []):
            c = store.clip(r["clip"])
            self._event("ok" if r["ok"] else "bad", (f"Uploaded clip {r['clip']}: {r['state']}" if r["ok"]
                                                        else f"Upload failed for clip {r['clip']}"))
            if r["ok"]:
                if r.get("scheduled") or "public" in r["state"]:
                    notifier.send("ok", "Video scheduled", f"{c['title']}\n{r['state']}")
                else:
                    notifier.send("warn", "Uploaded, but YouTube kept it private", f"{c['title']}\n{r['state']}")
            else:
                notifier.send("error", "Upload failed", f"{c['title']}\n{r.get('error', '')}")
        for r in getattr(pipe, "upload_results", []):
            streak = 0 if r["ok"] else int(store.kv_get("fail_streak", "0") or 0) + 1
            store.kv_set("fail_streak", str(streak))
            if streak and cfg.get("safe.auto", True) and streak >= int(cfg.get("safe.fail_streak", 3)):
                safe.turn_on(store, f"{streak} uploads failed in a row")
                notifier.send("error", "Safe mode switched on", f"{streak} uploads failed in a row: {r.get('error', '')[:150]}")
                break

    # ------------------------------------------------------------------ 5. stock
    def _stock(self, cfg, store, notifier) -> None:
        slots = len(schedule.parse_slots(cfg.get("schedule.slots", schedule.DEFAULT_SLOTS))) or 3
        need = int(cfg.get("assistant.buffer_days", 2)) * slots
        now = time.time()
        future = sum(1 for c in store.clips(status="uploaded") if c["scheduled_at"] and c["scheduled_at"] > now)
        stock = future + len(store.clips(status="approved")) + len(store.clips(status="ready"))
        busy = store.agent_sources("queued") or store.agent_sources("running")
        if stock < need and not busy:
            last = float(store.kv_get("low_stock_ts", "0") or 0)
            if now - last > 6 * 3600:
                store.kv_set("low_stock_ts", str(now))
                if stock == 0:
                    notifier.send("warn", "Clips are finished", "No clips left and no link in the queue. Send a new live or video link.")
                else:
                    notifier.send("warn", "Clip stock is running low",
                                  f"Only {stock} clip(s) left (need {need} for {cfg.get('assistant.buffer_days', 2)} days). Add a new link.")


    # ------------------------------------------------------------------ resume after a crash
    def _resume(self, cfg, store, pipe) -> None:
        """Clips left half-way (program closed while judging/building) are finished here, one video per pass."""
        if self.jobs.running() is not None:
            return
        for v in store.videos():
            vid = v["id"]
            judged = store.clips(vid, "judged")
            cands = store.clips(vid, "candidate")
            never_judged = bool(cands) and len(store.clips(vid)) == len(cands)
            if not (judged or never_judged):
                continue
            if not Path(v["path"]).is_file():
                for c in judged + (cands if never_judged else []):
                    store.update_clip(c["id"], status="dropped", error="source file was cleaned up before this clip was made")
                continue
            try:
                self.state["message"] = f"Finishing unfinished clips of video {vid} ..."
                if never_judged:
                    pipe.judge(vid)
                if store.clips(vid, "judged"):
                    pipe.build(vid)
            except LLMError:
                pass                                    # AI quota: try again on the next pass
            except Exception:
                traceback.print_exc()
            return

    # ------------------------------------------------------------------ learn from YouTube results
    def _learn(self, cfg, store, notifier) -> None:
        tok = Path(str(cfg.get("upload.token", "token.json")))
        if not (tok if tok.is_absolute() else cfg.root / tok).is_file():
            return
        every = float(cfg.get("insights.refresh_hours", 3)) * 3600
        last = float(store.kv_get("stats_try_ts", "0") or 0)
        if time.time() - last < every:
            return
        store.kv_set("stats_try_ts", str(time.time()))
        try:
            insights.collect(store, cfg)
        except Exception as e:
            self.state["tick_error"] = f"stats: {type(e).__name__}: {str(e)[:120]}"
            return
        mode = str(cfg.get("schedule.autotune", "off"))
        if mode == "off":
            return
        sg = insights.suggest_slots(store, cfg)
        if not sg["change"]:
            return
        key = f"{sg['drop']}>{sg['add']}"
        if mode == "auto":
            insights.apply_slots(self.root, self.config_path, sg["slots"], store)
            notifier.send("info", "Publish times updated", f"{sg['drop']} -> {sg['add']}. {sg['reason']}")
        elif store.kv_get("slot_suggest_sent") != key:
            store.kv_set("slot_suggest_sent", key)
            notifier.send("info", "Better publish time found", f"{sg['reason']} Open the Assistant tab to apply it.")

    # ------------------------------------------------------------------ disk cleanup
    def _housekeeping(self, cfg, store, notifier) -> None:
        if not cfg.get("cleanup.enabled", True):
            return
        low = cleanup.free_gb(cfg.path("paths.workdir")) < float(cfg.get("cleanup.min_free_gb", 4))
        last = float(store.kv_get("cleanup_ts", "0") or 0)
        if time.time() - last < (600 if low else 1800):
            return
        store.kv_set("cleanup_ts", str(time.time()))
        r = cleanup.run(cfg, store)
        store.kv_set("cleanup_last", json.dumps({k: r[k] for k in ("freed_mb", "files", "free_gb", "low_disk")} | {"ts": time.time()}))
        if r["low_disk"] and r["free_gb"] < float(cfg.get("cleanup.min_free_gb", 4)):
            if time.time() - float(store.kv_get("lowdisk_ts", "0") or 0) > 6 * 3600:
                store.kv_set("lowdisk_ts", str(time.time()))
                notifier.send("warn", "Disk is almost full", f"Only {r['free_gb']} GB free even after cleaning. Free some space.")
