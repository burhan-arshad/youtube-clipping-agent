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
from pathlib import Path

from . import metadata, schedule
from .config import load_config
from .ffutil import grab_frame, probe
from .llm import LLM, LLMError
from .notify import Notifier
from .store import Store

REVIEW_PROMPT = """You are the final quality reviewer of a YouTube Shorts channel about {game}. The images are frames
of the finished vertical video in time order{times_note} (the last image is the thumbnail, if there is one).

GAME NAME: the footage is from "{game}" (possibly newer than your knowledge). The correct name is exactly "{game}".
Never claim the title/description/script "refer to the wrong game", never suggest another game or version, and never say
the game does not exist.

Title: {title}
Description:
{description}
Voice-over (each line with the second it is spoken):
{script}
Feature the clip claims to show: {feature}
{timeline_block}
THE LAYOUT IS INTENDED, do not report it: the gameplay sits in a window over a blurred copy of the same footage,
with the heading text above and the channel handle below, and burned-in captions. That is not "letterboxing",
"a second window" or "overlapping branding".

Report only real problems. Give every issue a severity:
- "blocker": a real person (streamer/webcam face), chat, donation alert, streamer name/logo is visible | the voice-over
  states something that the frames/timeline clearly contradict or show only LATER than it is spoken | the title or
  description promises something that is not shown or claims leaks/official information | a loading screen, menu, or
  almost nothing happening | very graphic gore, sexual content, slurs, real-world violence (type "policy").
- "minor": anything cosmetic or uncertain: caption or heading slightly cut, a weak thumbnail, a typo, wording you would
  phrase differently, something you cannot be sure of from only a few frames.
If you are not sure, it is "minor". Do not invent problems.

Return JSON only:
{{"score": 0-100 overall quality, "issues": [{{"type": "video|script|metadata|thumbnail|policy", "severity": "blocker|minor", "text": "short, specific"}}]}}
An empty issues list means you found nothing wrong."""


def _wrong_game_claim(text: str, game: str) -> bool:
    """True for reviewer remarks like 'refers to GTA 5 but the footage is GTA 6' (hallucination about the game name)."""
    import re
    t = text.lower()
    if not game or re.search(r"\b(5|v)\b", game.lower()):
        return False                                       # only for newer-than-V games (e.g. GTA 6)
    if re.search(r"gta\s*(5|v)\b|grand theft auto\s*(5|v)\b", t) and re.search(r"gta\s*(6|vi)|grand theft auto\s*(6|vi)|footage|wrong game|incorrect", t):
        return True
    return bool(re.search(r"(does not|doesn't|not) (yet )?exist|not (been )?released|unreleased|fan.?made|leak", t)
                and game.lower() in t)


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
    # wording is repaired in code before review; it must never stop a good video, so it is informational only
    out.append({"name": "title + description rules", "ok": True, "detail": "; ".join(probs)})
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
            issues.append({"type": t, "severity": "blocker", "text": f"{ch['name']}: {ch['detail'] or 'failed'}"})
    score = 100
    if c["path"] and Path(c["path"]).is_file():
        dur = probe(c["path"])["duration"]
        tmp.mkdir(parents=True, exist_ok=True)
        imgs = []
        fr_times = []
        for i, f in enumerate((0.05, 0.18, 0.31, 0.44, 0.57, 0.7, 0.83, 0.95)):
            dst = tmp / f"rv_{i}.jpg"
            try:
                grab_frame(c["path"], dur * f, dst, 480)
                imgs.append(dst.read_bytes())
                fr_times.append(dur * f)
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
        beats = j.get("beats") if j.get("beats_script") == c["script"] else None
        script_txt = ("\n".join(f"  at {b['at']:.1f}s: {b['text']}" for b in beats) if beats else (c["script"] or ""))
        tl = j.get("timeline_clip") or []
        timeline_block = ("What the picture shows, from a separate frame-by-frame analysis (trust this):\n"
                          + "\n".join(f"  {t:.1f}s: {w}" for t, w in tl) + "\n") if tl else ""
        times_note = " (frames at " + ", ".join(f"{t:.0f}s" for t in fr_times) + ")" if fr_times else ""
        prompt = REVIEW_PROMPT.format(game=cfg.get("game.name", "the game"), title=c["title"], description=c["description"],
                                      script=script_txt, feature=j.get("feature_name", ""),
                                      timeline_block=timeline_block, times_note=times_note)
        raw = llm.json(prompt, images=imgs or None, temperature=0.1)
        try:
            score = float(raw.get("score", 0))
        except (TypeError, ValueError):
            score = 0.0
        game_name = str(cfg.get("game.name", "") or "")
        for it in raw.get("issues") or []:
            if isinstance(it, dict):
                if _wrong_game_claim(str(it.get("text", "")), game_name):
                    continue                              # the reviewer's knowledge predates the game: ignore it
                t = str(it.get("type", "video")).lower()
                t = t if t in ("video", "script", "metadata", "thumbnail", "policy") else "video"
                sev = "blocker" if (t == "policy" or (str(it.get("severity", "")).lower() == "blocker" and t != "thumbnail")) else "minor"
                issues.append({"type": t, "severity": sev, "text": str(it.get("text", ""))[:200]})
    need = 80 if str(cfg.get("assistant.review_strictness", "normal")) == "strict" else 65
    blockers = [i for i in issues if i.get("severity") == "blocker"]
    ok = not blockers and score >= need      # minor remarks are kept as notes but never stop a good Short
    return {"ok": ok, "score": score, "issues": issues, "blockers": len(blockers), "checks": checks, "ts": time.time()}


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
            try:
                from . import extras
                extras.first_comments(cfg, store)
                extras.weekly_report(cfg, store, notifier)
            except Exception:
                traceback.print_exc()
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
            kind = "live" if (is_live(s["url"], cfg) or "/live" in s["url"].lower()) else "video"
            store.update_agent_source(s["id"], kind=kind)
            s = dict(s, kind=kind)
            self._event("info", f"Link is a {'LIVE stream' if kind == 'live' else 'recorded video'}: {s['url'][:70]}")
        opts = json.loads(s["options"] or "{}")
        args = dict(video=s["url"], streamer=s["streamer"] or "", credit_url=s["credit_url"] or "",
                    facecam=opts.get("facecam", ""), crop=opts.get("crop"))
        if s["kind"] == "live":
            job = self.jobs.submit("watch", f"LIVE (assistant): {s['url'][:60]}", minutes=float(cfg.get("assistant.piece_minutes", 5) or 5), **args)
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
            c = self._repair_wording(cfg, store, pipe, c)
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
            blockers = [i for i in rv["issues"] if i.get("severity") == "blocker"]
            types = {i["type"] for i in blockers}
            summary = "; ".join(i["text"] for i in (blockers or rv["issues"]))[:300]
            store.kv_set(f"rv:{c['id']}", str(attempts + 1))
            self._event("warn", f"Clip {c['id']}: {len(blockers)} serious / {len(rv['issues']) - len(blockers)} minor remark(s): {summary[:110]}")
            if "policy" in types:
                store.update_clip(c["id"], review=rv)
                if mode == "auto":
                    store.update_clip(c["id"], status="rejected", error="assistant review: " + summary)
                    self._event("bad", f"Clip {c['id']} rejected (policy)")
                    notifier.send("warn", f"Assistant rejected clip {c['id']}", f"{c['title'] or ''}\n{summary}")
                continue
            if types & {"video", "script", "metadata"} and c["rebuilds"] < max_rebuilds and mode == "auto" and c["judgement"]:
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
            if types <= {"metadata"} and attempts < 2 and c["judgement"]:
                self._event("info", f"Clip {c['id']}: writing a new title/description/thumbnail")
                self._fix_metadata(cfg, store, llm, pipe, c)
                store.update_clip(c["id"], review=None)       # review again next pass
                continue
            # Serious problems that could not be fixed automatically: do NOT throw the clip away.
            # It stays in "Clips" waiting for you (Approve anyway / Rebuild / Reject).
            rv["needs_you"] = True
            store.update_clip(c["id"], review=rv)
            self._event("bad", f"Clip {c['id']} needs your decision: {summary[:100]}")
            notifier.send("warn", f"Clip {c['id']} needs your decision",
                          f"{c['title'] or ''}\n{summary}\nOpen Clips: approve it anyway, rebuild it or reject it.")

    def _repair_wording(self, cfg, store, pipe, c):
        """Title/description rule problems are fixed in code (no AI, no rejection)."""
        game = str(cfg.get("game.name", "") or "")
        probs = metadata.validate({"title": c["title"] or "", "description": c["description"] or ""}, game)
        if not probs or not c["judgement"]:
            return c
        j = json.loads(c["judgement"])
        video = store.video(c["video_id"])
        src = pipe.sources.get(video["source_id"]) if video else None
        fixed = metadata.sanitize({"title": c["title"] or "", "description": c["description"] or ""}, game,
                                  metadata.norm_suffix(cfg.get("upload.title_suffix", "")), j, cfg, src.streamer if src else "")
        store.update_clip(c["id"], title=fixed["title"], description=fixed["description"])
        self._event("info", f"Clip {c['id']}: wording fixed automatically ({'; '.join(probs)[:80]})")
        return store.clip(c["id"])

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
            if r.get("limit"):
                key = store.kv_get("upload_pause_until", "")
                if store.kv_get("limit_notified") != key:
                    store.kv_set("limit_notified", key)
                    self._event("warn", "YouTube upload limit reached: uploads paused, clips are kept")
                    notifier.send("warn", "YouTube upload limit reached",
                                  "YouTube allows only a few uploads per day for this channel. The assistant waits and "
                                  "tries again later. Verify your phone number at youtube.com/verify to raise the limit.")
                continue
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
            if r.get("limit"):
                continue                                  # a daily limit is not a failure of the system
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
        try:
            from . import learn
            learn.refresh(store, cfg)                  # retrain the channel playbook on the new numbers
        except Exception:
            pass
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
