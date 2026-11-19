"""FastAPI app: everything the command line can do, from a browser page on your own computer."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ..agent import Assistant
from .. import activity, cleanup, insights, launch, learn, metadata, safe, schedule as sched
from ..config import load_config
from ..pipeline import Pipeline
from .. import extras
from ..quota import day_start, next_quota_reset, quota_day, reset_text
from ..llm_limits import as_rows as limit_rows, limits_for, peak_for
from ..store import Store
from . import settings_store as ss
from .jobs import JobManager, load_all_sources

STATIC = Path(__file__).resolve().parent / "static"


def create_app(root: str | Path = ".", config_path: str | None = None) -> FastAPI:
    root = Path(root).resolve()
    app = FastAPI(title="featureclip", docs_url=None, redoc_url=None)
    jm = JobManager(root, config_path)
    assistant = Assistant(root, config_path, jm)
    assistant.start()
    try:                                  # after every start (e.g. after copying an update) check that everything loads
        _c0 = load_config(config_path, root=root)
        _s0 = Store(_c0.path("paths.db"))
        extras.run_startup_check(_c0, _s0)
        _s0.close()
    except Exception:
        pass

    def cfg():
        return load_config(config_path, root=root)

    def store(c=None):
        c = c or cfg()
        return Store(c.path("paths.db"))

    # ------------------------------------------------------------------ page
    @app.get("/", response_class=HTMLResponse)
    def index():
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"),
                            headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ state / health
    @app.get("/api/state")
    def state():
        c = cfg()
        st = store(c)
        counts: dict[str, int] = {}
        for r in st.clips():
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        secret = c.root / c.get("upload.client_secret", "client_secret.json")
        token = c.root / c.get("upload.token", "token.json")
        run = jm.running()
        return {
            "has_key": bool(c.gemini_key or c.openai_key),
            "ffmpeg": bool(shutil.which("ffmpeg") and shutil.which("ffprobe")),
            "client_secret": secret.is_file(), "youtube_logged_in": token.is_file(),
            "counts": counts, "running": run.info(0)["title"] if run else None,
            "output": str(c.path("paths.output")), "workdir": str(c.path("paths.workdir")),
            "uploads_today": Pipeline(c, st, {}).uploads_today(), "max_per_day": c.get("upload.max_per_day"),
            "privacy": c.get("upload.privacy"), "root": str(root),
        }

    @app.get("/api/doctor")
    def doctor():
        c = cfg()
        checks = []

        def add(name, ok, hint=""):
            checks.append({"name": name, "ok": bool(ok), "hint": hint})
        add("ffmpeg", shutil.which("ffmpeg"), "winget install Gyan.FFmpeg  (then reopen the terminal)")
        add("ffprobe", shutil.which("ffprobe"))
        for mod in ("numpy", "cv2", "faster_whisper", "kokoro_onnx", "yt_dlp", "googleapiclient"):
            try:
                __import__(mod)
                add(mod, True)
            except Exception as e:
                add(mod, False, f"pip install -r requirements.txt ({type(e).__name__})")
        try:
            import PIL  # noqa: F401
            add("Pillow (thumbnails)", True)
        except Exception:
            add("Pillow (thumbnails)", False, "pip install pillow")
        try:
            from .. import metadata as _m
            from PIL import ImageFont as _IF
            def _has(_f):
                try:
                    _IF.truetype(_f, 20)
                    return True
                except Exception:
                    return False
            add("Thumbnail font (bold)", any(_has(f) for f in _m._FONTS), "Install Arial/Impact or the thumbnail will use a plain font")
        except Exception:
            pass
        add("Gemini API key", c.gemini_key, "Add the key in Settings")
        add("YouTube login", (c.root / c.get("upload.token", "token.json")).is_file(), "Open the YouTube tab and sign in")
        from .. import schedule as _s
        add("Publish slots valid", bool(_s.parse_slots(c.get("schedule.slots", _s.DEFAULT_SLOTS))), "Settings > Schedule: use HH:MM, comma separated")
        add("Phone/email alerts set", bool(c.get("notify.ntfy_topic") or c.get("notify.discord_webhook") or c.get("notify.email_to")),
            "Settings > Notifications: ntfy topic, Discord webhook or Gmail (Telegram is blocked in Pakistan)")
        kd = c.path("paths.models") / "kokoro"
        add("Voice model (Kokoro)", all((kd / n).is_file() for n in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin")),
            "downloaded automatically the first time a voice-over is made (~120 MB)")
        return checks

    @app.post("/api/test-llm")
    def test_llm():
        from ..llm import LLM
        c = cfg()
        try:
            r = LLM(c, store(c)).json('Reply with exactly this JSON: {"ok": true}')
            return {"ok": bool(r.get("ok")), "message": "Gemini is working"}
        except Exception as e:
            return {"ok": False, "message": str(e)[:600]}

    @app.get("/api/usage")
    def usage():
        """Live view of the requests this app sent to Gemini: per model requests, fails, per-minute rate, cooldowns."""
        c = cfg()
        st = store(c)
        now = time.time()
        midnight = day_start()   # quota day starts at midnight Pacific time (13:00 PKT in winter)
        calls = st.calls_since(midnight - 6 * 86400)
        cool = st.cooldowns()
        today = [r for r in calls if r["ts"] >= midnight]
        models: dict[str, dict] = {}

        def m(name):
            lim = limits_for(name) or {}
            return models.setdefault(name, {"model": name, "ok": 0, "fail": 0, "rpm_now": 0, "rpm_peak": 0,
                                            "last_error": "", "last_at": None, "limit_rpd": lim.get("rpd"), "used_today": st.usage_today(name),
                                            "limit_rpm": lim.get("rpm"), "limit_tpm": lim.get("tpm"),
                                            "google_peak": peak_for(name), "cooldown": None, "per_min": {}})
        for name in c.get("llm.models", []) or []:
            m(name)
        for r in today:
            d = m(r["model"])
            d["ok" if r["ok"] else "fail"] += 1
            if now - r["ts"] <= 60:
                d["rpm_now"] += 1
            d["per_min"][int(r["ts"] // 60)] = d["per_min"].get(int(r["ts"] // 60), 0) + 1
            if not r["ok"]:
                d["last_error"], d["last_at"] = r["err"] or "error", r["ts"]
        for name, cd in cool.items():
            m(name)["cooldown"] = {"until": cd["until"], "reason": cd["reason"], "left": int(cd["until"] - now)}
        for d in models.values():
            d["rpm_peak"] = max(d["per_min"].values(), default=0)
            del d["per_min"]
        days: dict[str, int] = {}
        for r in calls:
            if r["ok"]:
                k = quota_day(r["ts"])
                days[k] = days.get(k, 0) + 1
        recent = [{"ts": r["ts"], "model": r["model"], "ok": bool(r["ok"]), "err": r["err"] or ""}
                  for r in calls[-25:]][::-1]
        key = c.gemini_key or ""
        rows = sorted(models.values(), key=lambda x: (-(x["ok"] + x["fail"]), x["model"]))
        return {"has_key": bool(key), "key_hint": (key[:4] + "…" + key[-3:]) if len(key) > 10 else "",
                "now": now, "ok_today": sum(d["ok"] for d in rows), "fail_today": sum(d["fail"] for d in rows),
                "rpm_now": sum(d["rpm_now"] for d in rows), "models": rows, "recent": recent,
                "days": [{"day": d, "calls": n} for d, n in sorted(days.items())],
                "configured_models": list(c.get("llm.models", [])), "limits": limit_rows(),
                "reset_text": reset_text(), "reset_at": next_quota_reset(), "quota_day": quota_day()}

    # ------------------------------------------------------------------ settings
    @app.get("/api/settings")
    def get_settings():
        return ss.describe(cfg(), root)

    @app.post("/api/settings")
    def save_settings(values: dict = Body(...)):
        saved = ss.save(root, values, config_path)
        return {"saved": saved}

    @app.post("/api/pick")
    def pick(kind: str = Body(..., embed=True), initial: str = Body("", embed=True)):
        """Native folder / file chooser on this computer (tkinter)."""
        fn = "askdirectory" if kind == "folder" else "askopenfilename"
        code = ("import tkinter as tk\nfrom tkinter import filedialog as fd\n"
                "r=tk.Tk(); r.withdraw(); r.attributes('-topmost', True)\n"
                f"p=fd.{fn}(initialdir={initial!r} or None)\nprint(p or '')\n")
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
        except Exception as e:
            raise HTTPException(501, f"Could not open the folder chooser: {e}")
        if out.returncode != 0:
            raise HTTPException(501, "Folder chooser is not available here; type the path instead.")
        return {"path": out.stdout.strip().replace("/", os.sep) if os.sep == "\\" else out.stdout.strip()}

    @app.post("/api/open-folder")
    def open_folder(path: str = Body(..., embed=True)):
        p = Path(path)
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(p))                       # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            raise HTTPException(501, str(e))
        return {"ok": True}

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs")
    def new_job(body: dict = Body(...)):
        kind = body.get("kind")
        c = cfg()
        if kind == "run":
            v = (body.get("video") or "").strip()
            if not v:
                raise HTTPException(400, "Enter a video link or a file path")
            if not (c.gemini_key or c.openai_key):
                raise HTTPException(400, "Add your Gemini API key in Settings first")
            j = jm.submit("run", f"Make clips: {v[:70]}", video=v, streamer=body.get("streamer", ""),
                          credit_url=body.get("credit_url", ""), facecam=body.get("facecam", ""), crop=body.get("crop"))
        elif kind == "layout":
            v = (body.get("video") or "").strip()
            if not v:
                raise HTTPException(400, "Enter a video link or a file path")
            j = jm.submit("layout", ("Live preview: " if body.get("live") else "Layout check: ") + v[:70], video=v,
                          live=bool(body.get("live")), refresh=bool(body.get("refresh")),
                          facecam=body.get("facecam", ""), crop=body.get("crop"))
        elif kind == "watch":
            v = (body.get("video") or "").strip()
            if not v:
                raise HTTPException(400, "Enter the live stream link")
            if not (c.gemini_key or c.openai_key):
                raise HTTPException(400, "Add your Gemini API key in Settings first")
            j = jm.submit("watch", f"LIVE: {v[:70]}", video=v, minutes=body.get("minutes", 10),
                          streamer=body.get("streamer", ""), credit_url=body.get("credit_url", ""),
                          facecam=body.get("facecam", ""), crop=body.get("crop"))
        elif kind == "rejudge":
            j = jm.submit("rejudge", f"Re-judge: video {body['video_id']}", video_id=int(body["video_id"]))
        elif kind == "build":
            j = jm.submit("build", f"Rebuild: video {body['video_id']}", video_id=int(body["video_id"]))
        elif kind == "upload":
            j = jm.submit("upload", "YouTube upload" + (" (dry run)" if body.get("dry_run") else ""),
                          dry_run=bool(body.get("dry_run")), limit=body.get("limit"))
        elif kind == "youtube_login":
            j = jm.submit("youtube_login", "YouTube login")
        else:
            raise HTTPException(400, f"unknown job kind {kind!r}")
        return {"id": j.id}

    @app.get("/api/jobs")
    def list_jobs():
        return [{k: v for k, v in j.info(0).items() if k not in ("lines",)} for j in reversed(jm.jobs[-40:])]

    @app.get("/api/jobs/{jid}")
    def job_info(jid: int, since: int = 0):
        j = jm.get(jid)
        if not j:
            raise HTTPException(404, "no such job")
        return j.info(since)

    @app.post("/api/jobs/{jid}/stop")
    def stop_job(jid: int):
        return {"ok": jm.stop(jid)}

    @app.get("/media/job/{jid}/image")
    def job_image(jid: int):
        j = jm.get(jid)
        p = Path((j.result or {}).get("image", "")) if j else None
        if not p or not p.is_file():
            raise HTTPException(404)
        return FileResponse(p, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------------ clips
    def clip_dict(r, videos) -> dict:
        try:
            j = json.loads(r["judgement"]) if r["judgement"] else {}
        except Exception:
            j = {}
        try:
            qa = json.loads(r["qa"]) if r["qa"] else {}
        except Exception:
            qa = {}
        v = videos.get(r["video_id"])
        reason = r["error"] or ("" if j.get("accept", True) else j.get("decision_reason", ""))
        path = r["path"] or ""
        return {
            "id": r["id"], "video_id": r["video_id"], "status": r["status"], "start": r["start"], "end": r["end"],
            "title": r["title"] or "", "description": r["description"] or "", "script": r["script"] or "",
            "feature": j.get("feature_name", ""), "category": j.get("category", ""),
            "confidence": j.get("confidence"), "interest": j.get("interest"),
            "evidence": j.get("visible_evidence", []), "what": j.get("what_happens", ""),
            "reason": reason, "hint": extras.humanize_error(reason), "qa_ok": qa.get("ok"), "has_file": bool(path and Path(path).is_file()),
            "youtube_id": r["youtube_id"] if "youtube_id" in r.keys() else None,
            "video_name": Path(v["path"]).parent.name if v else "", "updated": r["updated"],
            "tags": json.loads(r["tags"]) if r["tags"] else [], "has_thumb": bool(r["thumb"] and Path(r["thumb"]).is_file()),
            "review": json.loads(r["review"]) if r["review"] else None,
            "scheduled_at": r["scheduled_at"], "publish_state": r["publish_state"] or "",
        }

    @app.get("/api/clips")
    def clips(status: str = Query("all")):
        st = store()
        videos = {v["id"]: v for v in st.videos()}
        rows = st.clips()
        want = {"all": None, "ready": ["ready"], "approved": ["approved"], "uploaded": ["uploaded"],
                "dropped": ["dropped", "rejected", "error"]}.get(status)
        out = [clip_dict(r, videos) for r in rows if r["status"] != "candidate"
               and (want is None or r["status"] in want)]
        out.sort(key=lambda d: d["updated"] or 0, reverse=True)
        return out[:400]

    @app.get("/api/videos")
    def videos():
        st = store()
        out = []
        for v in st.videos():
            n = {}
            for c in st.clips(v["id"]):
                n[c["status"]] = n.get(c["status"], 0) + 1
            out.append({"id": v["id"], "source": v["source_id"], "name": Path(v["path"]).parent.name,
                        "duration": v["duration"], "counts": n})
        return list(reversed(out))

    @app.get("/media/clip/{cid}")
    def clip_file(cid: int):
        r = store().clip(cid)
        if not r or not r["path"] or not Path(r["path"]).is_file():
            raise HTTPException(404)
        return FileResponse(r["path"], media_type="video/mp4", filename=Path(r["path"]).name)

    def pipe():
        c = cfg()
        return Pipeline(c, Store(c.path("paths.db")), load_all_sources(root))

    @app.post("/api/clips/approve")
    def approve(ids: list[int] = Body(..., embed=True)):
        pipe().approve(ids)
        return {"ok": True}

    @app.post("/api/clips/approve-all")
    def approve_all():
        p = pipe()
        ids = [r["id"] for r in p.store.clips(status="ready")]
        p.approve(ids)
        return {"approved": len(ids)}

    @app.post("/api/clips/reject")
    def reject(ids: list[int] = Body(..., embed=True)):
        pipe().reject(ids)
        return {"ok": True}

    @app.post("/api/clips/{cid}/rebuild")
    def rebuild(cid: int):
        st = store()
        r = st.clip(cid)
        if not r:
            raise HTTPException(404)
        st.update_clip(cid, status="judged", error=None, script=None)
        j = jm.submit("build", f"Rebuild: clip {cid}", video_id=r["video_id"])
        return {"id": j.id}

    @app.get("/media/clip/{cid}/thumb")
    def clip_thumb(cid: int):
        r = store().clip(cid)
        if not r or not r["thumb"] or not Path(r["thumb"]).is_file():
            raise HTTPException(404)
        return FileResponse(r["thumb"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/clips/{cid}/regen-meta")
    def regen_meta(cid: int):
        """New title, description, tags and thumbnail for this clip."""
        c = cfg()
        st = store(c)
        r = st.clip(cid)
        if not r["path"] or not r["judgement"]:
            raise HTTPException(400, "This clip has no video yet")
        from ..llm import LLM, LLMError
        from ..ffutil import probe
        p = pipe()
        video = st.video(r["video_id"])
        src = load_all_sources(root).get(video["source_id"])
        try:
            meta = metadata.generate(LLM(c, st), c, json.loads(r["judgement"]), r["script"] or "", "",
                                     src.streamer if src else "", src.credit_url if src else "")
        except LLMError as e:
            raise HTTPException(503, f"AI not available right now: {e}")
        fields = {"title": meta["title"], "description": meta["description"], "tags": meta["tags"], "review": None}
        try:
            jj = json.loads(r["judgement"])
            if jj.pop("manual_meta", None):
                fields["judgement"] = jj              # AI wrote them again: a later rebuild may replace them too
        except Exception:
            pass
        try:
            frame = metadata.pick_frame(r["path"], 0, probe(r["path"])["duration"], None, p.cdir(r["video_id"], cid) / "thumb2")
            fields["thumb"] = str(metadata.make_thumbnail(frame, Path(r["path"]).with_suffix(".jpg"), metadata.thumb_line(meta, "", meta["title"]), c))
        except Exception:
            pass
        st.update_clip(cid, **fields)
        return {"ok": True, "title": meta["title"]}

    @app.post("/api/clips/{cid}/review")
    def rereview(cid: int):
        store().update_clip(cid, review=None)
        store().kv_set(f"rv:{cid}", "0")
        assistant.wake()
        return {"ok": True}

    @app.post("/api/clips/{cid}/publish")
    def publish_one(cid: int):
        st = store()
        if st.clip(cid)["status"] not in ("ready", "approved"):
            raise HTTPException(400, "Only ready or approved clips can be uploaded")
        st.update_clip(cid, status="approved")
        j = jm.submit("upload", f"Upload clip {cid}", dry_run=False, limit=1, only=[cid])
        return {"id": j.id}

    @app.patch("/api/clips/{cid}")
    def edit(cid: int, body: dict = Body(...)):
        fields = {k: str(body[k]) for k in ("title", "description") if k in body}
        if "tags" in body:
            fields["tags"] = [t.strip() for t in (body["tags"] if isinstance(body["tags"], list) else str(body["tags"]).split(",")) if t.strip()]
        st = store()
        try:
            r = st.clip(cid)
        except KeyError:
            raise HTTPException(404, "clip not found")
        if fields:
            st.update_clip(cid, **fields)
            if r["judgement"]:                       # remember that the owner wrote these: a rebuild must keep them
                j = json.loads(r["judgement"])
                j["manual_meta"] = True
                st.update_clip(cid, judgement=j)
        return {"ok": True}

    @app.post("/api/clips/{cid}/revoice")
    def revoice(cid: int, body: dict = Body(...)):
        """New voice-over from the script text you edited: only the voice, captions and video are made again
        (no AI call for the script, title or description)."""
        from ..narrate import target_words, validate_script
        c = cfg()
        st = store(c)
        try:
            r = st.clip(cid)
        except KeyError:
            raise HTTPException(404, "clip not found")
        if not r["judgement"]:
            raise HTTPException(404, "clip not found")
        if r["status"] not in ("ready", "approved", "dropped", "error", "judged"):
            raise HTTPException(400, "This clip was already uploaded")
        script = " ".join(str(body.get("script", "")).split())
        j = json.loads(r["judgement"])
        dur = float(j.get("end_offset", 0)) - float(j.get("start_offset", 0)) or 30.0
        lo, hi = target_words(dur, c)
        n = len(script.split())
        if n < 6:
            raise HTTPException(400, "Script bahut chhota hai (kam az kam 6 alfaz).")
        if n > hi * 1.4:
            raise HTTPException(400, f"Script bahut lamba hai ({n} alfaz). Is clip ke liye {lo}-{hi} alfaz theek hain.")
        bad = [p for p in validate_script(script, " ".join([j.get("feature_name", ""), j.get("what_happens", ""), " ".join(j.get("visible_evidence", []))]
                                          + [w for _t, w in j.get("timeline_clip", [])]), 0, 10_000, "", False)
               if p.startswith(("contains", "number"))]
        if bad:
            raise HTTPException(400, "; ".join(bad))
        j.pop("beats", None)
        j.pop("beats_script", None)
        st.update_clip(cid, status="judged", script=script, judgement=j, error=None, review=None)
        st.kv_set(f"rv:{cid}", "0")
        job = jm.submit("build", f"New voice: clip {cid}", video_id=r["video_id"])
        return {"id": job.id}

    # ------------------------------------------------------------------ schedule / assistant / notifications
    @app.get("/api/schedule")
    def schedule_info():
        from datetime import datetime, timezone
        c = cfg()
        slots = sched.parse_slots(c.get("schedule.slots", sched.DEFAULT_SLOTS)) or sched.parse_slots(sched.DEFAULT_SLOTS)
        st = store(c)
        taken = [datetime.fromtimestamp(r["scheduled_at"], timezone.utc) for r in st.clips(status="uploaded") if r["scheduled_at"]]
        now = datetime.now(timezone.utc)
        free = sched.next_slots(now, 9, slots, taken)
        up = [{"id": r["id"], "title": r["title"], "at": r["scheduled_at"], "at_text": sched.fmt_pkt(datetime.fromtimestamp(r["scheduled_at"], timezone.utc)),
               "state": r["publish_state"] or "", "youtube_id": r["youtube_id"]}
              for r in st.clips(status="uploaded") if r["scheduled_at"] and r["scheduled_at"] > now.timestamp() - 86400]
        up.sort(key=lambda d: d["at"])
        return {"enabled": bool(c.get("schedule.enabled", True)),
                "slots": [{"pkt": f"{h:02d}:{m:02d}", "audience": sched.audience_times((h, m))} for h, m in slots],
                "free": [{"iso": t.isoformat(), "text": sched.fmt_pkt(t)} for t in free], "scheduled": up,
                "note": "Winter clocks. UK/EU/US change one hour in spring and autumn, so re-check the slots then."}

    @app.get("/api/assistant")
    def assistant_state():
        c = cfg()
        st = store(c)
        srcs = st.agent_sources()
        return {"enabled": bool(c.get("assistant.enabled", False)), "mode": c.get("assistant.mode", "auto"),
                "state": assistant.state, "sources": list(reversed(srcs))[:40],
                "counts": {s: len(st.clips(status=s)) for s in ("ready", "approved", "uploaded", "rejected")},
                "unread": sum(1 for n in st.notifications(50) if not n["read"]),
                "safe": safe.info(st), "launch": launch.status(st),
                "disk": {"free_gb": round(cleanup.free_gb(c.path("paths.workdir")), 1),
                         "last": json.loads(st.kv_get("cleanup_last", "null") or "null")}}

    @app.get("/api/insights")
    def insights_report():
        c = cfg()
        st = store(c)
        r = insights.report(st, c)
        r["stats_ts"] = float(st.kv_get("stats_ts", "0") or 0)
        r["autotune"] = c.get("schedule.autotune", "off")
        r["analytics"] = bool(c.get("upload.analytics", False))
        return r

    @app.post("/api/insights/refresh")
    def insights_refresh():
        c = cfg()
        try:
            n = insights.collect(store(c), c)
        except Exception as e:
            raise HTTPException(400, f"{type(e).__name__}: {str(e)[:300]}")
        try:
            learn.refresh(store(c), c)
        except Exception:
            pass
        return {"updated": n}

    @app.get("/api/learn")
    def learn_report():
        c = cfg()
        st = store(c)
        r = learn.load(st) or learn.build(st, c)
        r["analytics_on"] = bool(c.get("upload.analytics", False))
        r["in_use"] = bool(c.get("learn.use_playbook", True)) and bool(r.get("playbook"))
        return r

    @app.post("/api/learn/refresh")
    def learn_refresh():
        c = cfg()
        return learn.refresh(store(c), c)

    @app.post("/api/insights/apply-slots")
    def insights_apply():
        c = cfg()
        st = store(c)
        sg = insights.suggest_slots(st, c)
        if not sg["change"]:
            raise HTTPException(400, sg["reason"] or "No change suggested")
        insights.apply_slots(root, config_path, sg["slots"], st)
        return {"slots": sg["slots"]}

    @app.get("/api/voices")
    def voices(library: bool = False, lang: str = "en"):
        from .. import tts
        c = cfg()
        out = {"prebuilt": [{"id": k, "style": v} for k, v in tts.PREBUILT.items()], "library": [], "error": ""}
        if library:
            try:
                out["library"] = tts.library_voices(c, lang)
            except Exception as e:
                out["error"] = str(e)[:300]
        out["current"] = str(c.get("narration.gemini_voice", ""))
        return out

    @app.post("/api/voice-test")
    def voice_test(body: dict = Body(...)):
        """Speak a sample with a voice/style so you can listen before choosing (uses 1 Gemini voice request)."""
        import base64
        import io

        import soundfile as sf

        from .. import tts
        c = cfg()
        text = str(body.get("text") or "Whoa, look at that. The whole back end just folded like real metal. <short pause> Did you notice that?")[:400]
        try:
            audio, sr = tts.gemini_speak(c, text, voice=str(body.get("voice") or "").strip() or None,
                                         style=str(body.get("style") if body.get("style") is not None else ""))
        except Exception as e:
            raise HTTPException(400, str(e)[:400])
        buf = io.BytesIO()
        sf.write(buf, audio, sr, format="WAV")
        return {"audio": "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode(), "seconds": round(len(audio) / sr, 1)}

    @app.post("/api/assistant/recheck-rejected")
    def recheck_rejected():
        """Give clips that the (old, too strict) review rejected a second chance with the new review rules."""
        st = store()
        n = 0
        for x in st.clips(status="rejected"):
            if x["path"] and Path(x["path"]).is_file() and str(x["error"] or "").startswith("assistant review"):
                st.update_clip(x["id"], status="ready", review=None, error=None, rebuilds=0)
                st.kv_set(f"rv:{x['id']}", "0")
                n += 1
        assistant.wake()
        return {"requeued": n}

    @app.get("/api/launch")
    def launch_info():
        return launch.status(store())

    @app.post("/api/launch")
    def launch_set(body: dict = Body(default={})):
        c = cfg()
        st = store(c)
        if body.get("off"):
            launch.stop(st)
            return launch.status(st)
        gap = int(body.get("gap") or c.get("schedule.launch_gap_minutes", 60))
        out = launch.start(st, int(body.get("count") or 3), gap)
        out["moved"], out["failed"] = 0, []
        if body.get("reset_existing", True):
            try:
                r = launch.reschedule_existing(c, st)
                out["moved"], out["failed"] = r["moved"], r["failed"]
            except Exception as e:
                out["failed"] = [{"clip": 0, "error": str(e)[:200]}]
        assistant.wake()
        return out

    @app.post("/api/safe")
    def safe_set(body: dict = Body(...)):
        st = store()
        if body.get("on"):
            safe.turn_on(st, str(body.get("reason") or "switched on by you"))
            assistant.wake()
            return safe.info(st)
        safe.turn_off(st)
        assistant.wake()
        return safe.info(st)

    @app.post("/api/safe/pull-back")
    def safe_pull():
        c = cfg()
        try:
            return safe.pull_back(c, store(c))
        except Exception as e:
            raise HTTPException(400, str(e)[:400])

    @app.post("/api/safe/release")
    def safe_release():
        c = cfg()
        try:
            return safe.release(c, store(c))
        except Exception as e:
            raise HTTPException(400, str(e)[:400])

    @app.post("/api/cleanup")
    def cleanup_now(body: dict = Body(default={})):
        c = cfg()
        return cleanup.run(c, store(c), dry_run=bool(body.get("dry_run")))

    @app.post("/api/assistant/sources")
    def assistant_add(body: dict = Body(...)):
        url = (body.get("url") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            raise HTTPException(400, "Paste a full link (https://...)")
        kind = body.get("kind") or "auto"
        if kind == "auto":
            kind = "live" if any(w in url.lower() for w in ("/live", "live_stream")) else "auto"   # the assistant asks YouTube
        opts = {"minutes": body.get("minutes", 5), "facecam": body.get("facecam", ""), "crop": body.get("crop")}
        sid = store().add_agent_source(url, kind, body.get("name", ""), body.get("streamer", ""),
                                       body.get("credit_url", ""), opts)
        assistant.wake()
        return {"id": sid, "kind": kind}

    @app.delete("/api/assistant/sources/{sid}")
    def assistant_del(sid: int):
        st = store()
        row = next((s for s in st.agent_sources() if s["id"] == sid), None)
        if row and row["status"] == "running":
            opts = json.loads(row["options"] or "{}")
            if opts.get("job"):
                jm.stop(opts["job"])
        st.delete_agent_source(sid)
        return {"ok": True}

    @app.post("/api/assistant/sources/{sid}/retry")
    def assistant_retry(sid: int):
        store().update_agent_source(sid, status="queued", note="")
        assistant.wake()
        return {"ok": True}

    def quota_summary(c, st, now) -> dict:
        """Gemini models with today's calls / free limit, cooldowns and the reset time (for the dashboard meter)."""
        cd = st.cooldowns()
        rows = []
        for m in list(c.get("llm.models", []) or []):
            lim = (limits_for(m) or {}).get("rpd")
            used = st.usage_today(m)
            rows.append({"model": m, "used": used, "limit": lim, "left": (max(0, lim - used) if lim else None),
                         "cooling": ({"reason": cd[m]["reason"], "left": int(cd[m]["until"] - now)} if m in cd else None)})
        tts_rows = []
        for m in list(c.get("narration.gemini_models", []) or []):
            lim = (limits_for(m) or {}).get("rpd")
            used = st.usage_today(m)
            k = "tts:" + m
            tts_rows.append({"model": m, "used": used, "limit": lim, "left": (max(0, lim - used) if lim else None),
                             "cooling": ({"reason": cd[k]["reason"], "left": int(cd[k]["until"] - now)} if k in cd else None)})
        usable = [r for r in rows if not r["cooling"] and (r["left"] is None or r["left"] > 0)]
        return {"models": rows, "tts": tts_rows, "usable": len(usable), "total": len(rows),
                "reset_text": reset_text(now), "reset_in": int(next_quota_reset(now) - now),
                "total_left": sum(r["left"] for r in rows if r["left"] is not None and not r["cooling"])}

    @app.get("/api/selftest")
    def selftest_api(online: bool = Query(False)):
        c = cfg()
        res = extras.selftest(c, store(c), root, online=online)
        return {"checks": res, "ok": not any(x["level"] == "bad" for x in res),
                "bad": sum(1 for x in res if x["level"] == "bad"), "warn": sum(1 for x in res if x["level"] == "warn"), "ts": time.time()}

    @app.get("/api/startup-check")
    def startup_check():
        try:
            return json.loads(store().kv_get("startup_check", "") or "null") or {"bad": [], "warn": [], "ts": None}
        except Exception:
            return {"bad": [], "warn": [], "ts": None}

    @app.get("/api/assistant/live")
    def assistant_live():
        """Everything the 'Right now' card shows: current step, what exactly is happening, progress, recent events."""
        c = cfg()
        st = store(c)
        now = time.time()
        job = jm.running()
        ph = assistant.state.get("phase")
        out = {"enabled": bool(c.get("assistant.enabled", False)), "safe": safe.is_on(st), "now": now,
               "mode": "idle", "headline": "", "detail": "", "stage": None, "progress": None, "warning": "",
               "job": None, "source": None, "assistant": None, "feed": [], "events": list(assistant.events)[-15:][::-1]}
        if job:
            act = activity.from_lines(list(job.lines[-500:]))
            src = next((s for s in st.agent_sources("running") if json.loads(s["options"] or "{}").get("job") == job.id), None)
            out.update(mode="job", stage=act["stage"], headline=act["headline"] or "Starting ...", progress=act["progress"],
                       warning=act["warning"], feed=act["recent"][::-1],
                       job={"id": job.id, "title": job.title, "kind": job.kind, "since": job.started or job.created},
                       source=({"id": src["id"], "kind": src["kind"], "url": src["url"]} if src else None))
            pieces = sum(1 for l in job.lines if "live piece" in l and ": processing" in l)
            out["pieces"] = pieces
        if ph and ph.get("key"):
            out["assistant"] = {"key": ph["key"], "text": ph["text"], "since": ph["since"]}
            if not job:
                out.update(mode="assistant", stage=ph["key"], headline=ph["text"])
        active = {out["stage"]} if out["stage"] else set()
        if ph and ph.get("key") in ("review", "publish"):
            active.add(ph["key"])
        order = {k: i for i, (k, _) in enumerate(activity.STEPS)}
        top = max((order[k] for k in active if k in order), default=-1)
        out["steps"] = [{"key": k, "label": lb,
                         "state": "active" if k in active else ("done" if (job and order[k] < top and out["mode"] == "job") else "todo")}
                        for k, lb in activity.STEPS]
        future = [x for x in st.clips(status="uploaded") if x["scheduled_at"] and x["scheduled_at"] > now]
        nxt = min((x["scheduled_at"] for x in future), default=None)
        def _needs(x):
            try:
                return bool(json.loads(x["review"] or "null").get("needs_you"))
            except Exception:
                return False
        needs_you = sum(1 for x in st.clips(status="ready") if _needs(x))
        out["counters"] = {"needs_you": needs_you, "in_review": len(st.clips(status="ready")) - needs_you, "approved": len(st.clips(status="approved")),
                           "scheduled": len(future), "uploaded": len(st.clips(status="uploaded")),
                           "rejected": len(st.clips(status="rejected")),
                           "next_publish": sched.fmt_pkt(__import__("datetime").datetime.fromtimestamp(nxt, __import__("datetime").timezone.utc)) if nxt else ""}
        out["quota"] = quota_summary(c, st, now)
        if out["mode"] == "idle":
            queued = st.agent_sources("queued")
            if not out["enabled"]:
                out["headline"], out["detail"] = "Assistant is switched off", "Switch it on, then add a live or video link."
            elif out["safe"]:
                out["headline"], out["detail"] = "Safe mode is on", "Nothing is uploaded or started until you switch it off."
            elif queued:
                out["headline"], out["detail"] = "A link is waiting to start", f"{len(queued)} link(s) in the queue."
            elif not (out["counters"]["in_review"] or out["counters"]["approved"] or out["counters"]["scheduled"]):
                out["headline"], out["detail"] = "Waiting for a new link", "No clips left. Add a live or video link."
            else:
                out["headline"] = "Resting: everything is up to date"
                out["detail"] = ("Next video goes public " + out["counters"]["next_publish"]) if out["counters"]["next_publish"] else "Checking again soon."
            last = assistant.state.get("last_tick")
            every = int(c.get("assistant.check_every_seconds", 60))
            out["next_check_in"] = max(0, int(last + every - now)) if last else None
        return out

    @app.post("/api/assistant/sources/{sid}/stop")
    def assistant_stop(sid: int, body: dict = Body(default={})):
        st = store()
        src = next((s for s in st.agent_sources() if s["id"] == sid), None)
        if not src:
            raise HTTPException(404, "not found")
        opts = json.loads(src["options"] or "{}")
        if opts.get("job") is not None:
            jm.stop(opts["job"])
        if body.get("restart"):                    # forget the guess, ask YouTube again (live or recorded?)
            opts.pop("job", None)
            st.update_agent_source(sid, kind="auto", status="queued", options=json.dumps(opts), note="restarted by you")
            assistant.wake()
        else:
            st.update_agent_source(sid, status="error", finished=time.time(), note="stopped by you")
        return {"ok": True}

    @app.post("/api/assistant/run-now")
    def assistant_run():
        assistant.wake()
        return {"ok": True}

    @app.get("/api/notifications")
    def notifications():
        st = store()
        return {"items": st.notifications(60), "unread": sum(1 for n in st.notifications(60) if not n["read"])}

    @app.post("/api/notifications/read")
    def notifications_read():
        store().mark_notifications_read()
        return {"ok": True}

    @app.post("/api/notifications/test")
    def notifications_test():
        from ..notify import Notifier
        c = cfg()
        Notifier(c, store(c)).send("info", "Test notification", "If you can read this, notifications work.")
        return {"ok": True}

    # ------------------------------------------------------------------ youtube
    @app.post("/api/youtube/client-secret")
    def client_secret(content: str = Body(..., embed=True)):
        try:
            data = json.loads(content)
        except Exception:
            raise HTTPException(400, "This file is not valid JSON")
        if not isinstance(data, dict) or not ({"installed", "web"} & set(data)):
            raise HTTPException(400, "This does not look like a client_secret.json. The OAuth client type must be 'Desktop app'.")
        c = cfg()
        (c.root / c.get("upload.client_secret", "client_secret.json")).write_text(content, encoding="utf-8")
        return {"ok": True}

    @app.post("/api/youtube/logout")
    def yt_logout():
        c = cfg()
        t = c.root / c.get("upload.token", "token.json")
        t.unlink(missing_ok=True)
        return {"ok": True}

    @app.exception_handler(Exception)
    async def boom(_, exc):
        return JSONResponse({"detail": f"{type(exc).__name__}: {exc}"}, status_code=500)

    return app


def serve(root: str = ".", config_path: str | None = None, host: str = "127.0.0.1", port: int = 8765,
          open_browser: bool = True) -> None:
    import threading
    import webbrowser

    import logging

    import uvicorn

    class _QuietResets(logging.Filter):
        """Windows prints a scary traceback when the browser closes a connection; it is harmless."""
        def filter(self, record):
            ei = record.exc_info
            return not (ei and isinstance(ei[1], (ConnectionResetError, ConnectionAbortedError)))
    logging.getLogger("asyncio").addFilter(_QuietResets())
    app = create_app(root, config_path)
    url = f"http://{host}:{port}"
    print(f"featureclip UI: {url}   (press Ctrl+C in this window to stop)")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
