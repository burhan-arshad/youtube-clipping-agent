"""The whole flow, stage by stage. Every stage is resumable: results are cached in the
work folder and in SQLite, so a crash or Ctrl+C never loses finished work.

  scan   : get video -> signals -> candidate windows
  judge  : Whisper on candidates -> LLM (frames + words) -> accept/drop
  build  : crop to gameplay -> script -> voice-over -> captions -> render -> QA -> ready
  upload : only clips that a human approved (or auto_approve=true)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import launch, learn, metadata, safe, schedule, captions, candidates, compose, judge, layout as lay, music, narrate, qa, signals, uploader
from .ffutil import extract_audio_wav, probe
from .ingest import describe, ingest
from .llm import LLM, LLMError
from .sources import Source, resolve_source
from .store import Store
from .transcribe import Transcript, transcribe_file


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def slugify(s: str, n: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:n] or "clip"


class Pipeline:
    def __init__(self, cfg, store: Store, sources: dict[str, Source], llm: LLM | None = None):
        self.cfg, self.store, self.sources = cfg, store, sources
        self.llm = llm or LLM(cfg, store)
        self.stop_event = None                  # set by the UI so a long download can be cancelled
        self.work = cfg.path("paths.workdir")
        self.out = cfg.path("paths.output")
        self.work.mkdir(parents=True, exist_ok=True)
        self.out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ helpers
    def vdir(self, video_id: int) -> Path:
        d = self.work / f"video_{video_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cdir(self, video_id: int, clip_id: int) -> Path:
        d = self.vdir(video_id) / f"clip_{clip_id}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ------------------------------------------------------------ 1. scan
    def scan(self, source_id: str, path=None) -> int:
        """path: use this local file instead of the source's url/path (live segments)."""
        src = resolve_source(source_id, self.sources)
        source_id = src.id
        path = Path(path) if path else ingest(src, self.work, self.cfg, stop=self.stop_event)
        info = describe(path)
        vid = self.store.add_video(src.id, str(path), info)
        vdir = self.vdir(vid)
        if self.store.refresh_video(vid, info):
            (vdir / "signals.npz").unlink(missing_ok=True)       # same file name, new recording: old analysis is wrong
            (vdir / "layout.json").unlink(missing_ok=True)
            log("this file name was used by an older recording: old analysis cleared")
        log(f"video {vid}: {path.name}  {info['duration'] / 60:.1f} min  {info['width']}x{info['height']}")

        sig_file = vdir / "signals.npz"
        if sig_file.is_file():
            data = np.load(sig_file)
            sig = {k: data[k] for k in data.files}
            log("signals: loaded from cache")
        else:
            log("signals: analysing audio + motion (CPU, a few minutes for a long VOD) ...")
            sig = signals.compute_signals(str(path), info["duration"], int(self.cfg.get("detect.sample_fps")))
            np.savez(sig_file, **sig)
        d = self.cfg.get
        combined = candidates.combine(sig, d("detect.w_audio"), d("detect.w_motion"), d("detect.w_cut"))
        cands = candidates.pick_candidates(
            combined, info["duration"], window=d("detect.window"), pre_roll=d("detect.pre_roll"),
            top_k=int(d("detect.top_k")), min_gap=d("detect.min_gap"), skip_start=d("detect.skip_start"))
        self.store.delete_candidates(vid)
        existing = {(round(c["start"], 1)) for c in self.store.clips(vid)}
        added = 0
        for c in cands:
            if round(c.start, 1) in existing:
                continue
            self.store.add_clip(vid, c.start, c.end, c.score)
            added += 1
        self.store.set_video_stage(vid, "scanned")
        log(f"candidates: {added} windows queued")
        return vid

    # ------------------------------------------------------------ 2. judge
    def _transcript_for(self, video: str, clip) -> Transcript:
        cd = self.cdir(clip["video_id"], clip["id"])
        cache = cd / "transcript.json"
        if cache.is_file():
            t = json.loads(cache.read_text(encoding="utf-8"))
            return Transcript(text=t["text"], available=t.get("available", True))
        wav = extract_audio_wav(video, cd / "window.wav", clip["start"], clip["end"] - clip["start"])
        tr = transcribe_file(wav, self.cfg)
        cache.write_text(json.dumps({"text": tr.text, "available": tr.available, "error": tr.error}),
                         encoding="utf-8")
        if wav.exists():
            wav.unlink()
        return tr

    def judge(self, video_id: int) -> int:
        if not self.llm.available():
            raise LLMError("no LLM key: put GEMINI_API_KEY in .env (free key from Google AI Studio)")
        video = self.store.video(video_id)
        path = video["path"]
        max_clips = int(self.cfg.get("judge.max_clips_per_video"))
        todo = sorted(self.store.clips(video_id, "candidate"), key=lambda c: -c["signal_score"])
        cap = int(self.cfg.get("judge.max_windows_per_video", 7))
        for c in todo[cap:]:                                   # weaker windows are skipped to save the free AI quota
            self.store.update_clip(c["id"], status="dropped", error="not judged: weaker window (saves AI quota)")
        todo = todo[:cap]
        accepted: list[tuple[int, dict]] = [
            (c["id"], json.loads(c["judgement"])) for c in self.store.clips(video_id, "judged")]
        log(f"judging up to {len(todo)} candidates (stops early once enough good ones are found)")
        crop = None       # the judge should see what the viewer will see: gameplay only, no webcam/chat
        try:
            L = self._layout(video_id, self.sources.get(video["source_id"]))
            asp = 9 / 16 if self.cfg.get("layout.mode") == "fill" else (1.0, 1.7)
            crop = lay.choose_crop(L, None, asp)
        except Exception as e:
            log(f"  (layout for the judge failed: {e}; the judge sees the full frame)")
        for c in todo:
            if len(accepted) >= int(max_clips * 1.5):
                break
            cd = self.cdir(video_id, c["id"])
            try:
                tr = self._transcript_for(path, c)
                j = judge.judge_window(self.llm, self.cfg, path, c["start"], c["end"], tr.text,
                                       c["signal_score"], cd / "frames", crop)
            except LLMError as e:
                log(f"  clip {c['id']}: LLM unavailable -> stopping judge ({e})")
                break
            except Exception as e:  # one bad window must not stop the run
                # not an AI-quota problem (e.g. a frame could not be read): drop it, never retry it forever
                self.store.update_clip(c["id"], status="dropped", error=f"judge: {type(e).__name__}: {e}")
                log(f"  clip {c['id']}: skipped, could not be judged ({e})")
                continue
            if j["accept"]:
                self.store.update_clip(c["id"], status="judged", judgement=j)
                accepted.append((c["id"], j))
                log(f"  clip {c['id']} @ {c['start']:.0f}s: ACCEPT {j['feature_name']!r} "
                    f"(conf {j['confidence']:.0f}, interest {j['interest']:.0f})")
            else:
                self.store.update_clip(c["id"], status="dropped", judgement=j, error=j["decision_reason"])
                log(f"  clip {c['id']} @ {c['start']:.0f}s: drop ({j['decision_reason']})")
        from . import styles as _styles
        keep = set(judge.dedupe(accepted, max_clips, _styles.category_boost(self.store, self.cfg)))
        # same feature already made from an earlier video/live segment of this source -> skip
        seen = self._seen_features(video["source_id"], video_id)
        for cid, j in accepted:
            if cid in keep and judge.feature_key(j["feature_name"]) in seen:
                keep.discard(cid)
                log(f"  clip {cid}: feature {j['feature_name']!r} already made before -> skipped")
        # the same moment found twice (overlapping windows, or an earlier clip that looks the same)
        spans = {c["id"]: (c["start"] + (json.loads(c["judgement"]).get("start_offset") or 0),
                           c["start"] + (json.loads(c["judgement"]).get("end_offset") or 0))
                 for c in self.store.clips(video_id) if c["judgement"]}
        earlier = [json.loads(c["judgement"]) for c in self.store.clips()
                   if c["video_id"] != video_id and c["status"] in ("ready", "approved", "uploaded", "judged") and c["judgement"]]
        kept_js: list[tuple[int, dict]] = []
        for cid, j in sorted(accepted, key=lambda t: -(t[1]["interest"] + t[1]["confidence"])):
            if cid not in keep:
                continue
            if any(judge.same_moment(j, k, spans.get(cid), spans.get(kc)) for kc, k in kept_js) \
                    or any(judge.same_moment(j, e) for e in earlier):
                keep.discard(cid)
                log(f"  clip {cid}: same moment as another clip -> skipped")
                continue
            kept_js.append((cid, j))
        # quality gate: boring clips ("a man walks along a road") do not get views. Keep the best few of each piece and
        # drop the rest when their interest is under the floor.
        floor, keep_min = float(self.cfg.get("judge.quality_floor", 55)), int(self.cfg.get("judge.keep_min", 2))
        ranked = [t for t in sorted(accepted, key=lambda t: -(t[1]["interest"] + 0.3 * t[1]["confidence"])) if t[0] in keep]
        for pos, (cid, j) in enumerate(ranked):
            if pos >= keep_min and j["interest"] < floor:
                keep.discard(cid)
                log(f"  clip {cid}: interest {j['interest']:.0f} is under the quality floor {floor:.0f} -> skipped")
        for cid, _ in accepted:
            if cid not in keep:
                self.store.update_clip(cid, status="dropped", error="duplicate feature, low interest or over the per-video limit")
        self.store.set_video_stage(video_id, "judged")
        log(f"judge done: {len(keep)} clips kept")
        return len(keep)

    def _seen_features(self, source_id: str, except_video: int) -> set[str]:
        seen = set()
        vids = {v["id"] for v in self.store.videos() if v["source_id"] == source_id and v["id"] != except_video}
        for c in self.store.clips():
            if c["video_id"] in vids and c["status"] in ("ready", "approved", "uploaded", "judged") and c["judgement"]:
                seen.add(judge.feature_key(json.loads(c["judgement"])["feature_name"]))
        return seen

    # ------------------------------------------------------------ 3. build
    def _layout(self, video_id: int, src: Source | None) -> lay.Layout:
        vdir = self.vdir(video_id)
        cache = vdir / "layout.json"
        video = self.store.video(video_id)
        if cache.is_file():
            d = json.loads(cache.read_text(encoding="utf-8"))
            return lay.Layout(d["frame"][0], d["frame"][1], tuple(d["rect"]),
                              [tuple(b) for b in d["avoid"]], d["origin"])
        sid = video["source_id"]
        shared = self.work / "layouts" / f"{sid}.json"          # one layout for all pieces of a live stream
        reuse = bool(self.cfg.get("layout.reuse_per_source", True))
        if reuse and shared.is_file():
            try:
                d = json.loads(shared.read_text(encoding="utf-8"))
                if d["L"]["frame"] == [video["width"], video["height"]] and d["uses"] < int(self.cfg.get("layout.reuse_pieces", 6)):
                    d["uses"] += 1
                    shared.write_text(json.dumps(d), encoding="utf-8")
                    L = lay.Layout(d["L"]["frame"][0], d["L"]["frame"][1], tuple(d["L"]["rect"]),
                                   [tuple(b) for b in d["L"]["avoid"]], d["L"]["origin"])
                    cache.write_text(json.dumps(L.to_dict()), encoding="utf-8")
                    log(f"layout: reused from the previous piece {L.rect} ({L.origin})")
                    return L
            except Exception:
                pass
        L = lay.detect_layout(video["path"], video["duration"], video["width"], video["height"],
                              n_frames=int(self.cfg.get("layout.sample_frames")),
                              face_detect=bool(self.cfg.get("layout.face_detect")),
                              manual=(src.crop if src else None),
                              facecam=((src.facecam if src and src.facecam else None)
                                       or self.cfg.get("layout.facecam", "auto")),
                              facecam_size=tuple(self.cfg.get("layout.facecam_size", [0.27, 0.38])),
                              crop_frac=((src.extra or {}).get("crop_frac") if src else None) or self.cfg.get("layout.crop_frac"),
                              llm=(self.llm if self.cfg.get("layout.ai", True) else None), tmp=vdir / "layout_tmp")
        cache.write_text(json.dumps(L.to_dict()), encoding="utf-8")
        if reuse:
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_text(json.dumps({"L": L.to_dict(), "uses": 1}), encoding="utf-8")
        log(f"layout: gameplay area {L.rect} ({L.origin}), avoid boxes: {L.avoid or 'none'}")
        return L

    def build_clip(self, clip_id: int) -> str:
        c = self.store.clip(clip_id)
        video = self.store.video(c["video_id"])
        src = self.sources.get(video["source_id"])
        j = json.loads(c["judgement"])
        cd = self.cdir(video["id"], clip_id)
        g = self.cfg.get
        lead = float(g("narration.lead_in"))
        clip_min, clip_max = float(g("video.clip_min")), float(g("video.clip_max"))

        start = c["start"] + j["start_offset"]
        dur = j["end_offset"] - j["start_offset"]
        dur = min(dur, video["duration"] - start)
        if dur < 2.0:                          # the window lies outside the file (stale record): never build a negative clip
            self.store.update_clip(clip_id, status="dropped", error="clip window is outside the video file")
            return "dropped"
        tr = self._transcript_for(video["path"], c)

        L = self._layout(video["id"], src)
        aspect = 9 / 16 if g("layout.mode") == "fill" else (1.0, 1.7)
        frames = lay.sample_frames(video["path"], list(np.linspace(start, start + dur, 8)), 640)
        energy = lay.energy_map(frames) if len(frames) >= 2 else None
        crop = lay.choose_crop(L, energy, aspect)

        script = c["script"]
        hook = ""
        beats = j.get("beats") if (script and j.get("beats_script") == script) else None
        if not script:
            tl_imgs, tl_times = judge.grab_keyframes(video["path"], start, start + dur, 8, 512, cd / "kf", crop)
            timeline = narrate.describe_timeline(self.llm, self.cfg, j, tl_imgs, tl_times, dur)
            if timeline:
                j["timeline_clip"] = [[t, w] for t, w in timeline["events"]]
                j["moment"] = timeline["moment"]
                log("  scene timeline: " + "; ".join(f"{t:.0f}s {w[:40]}" for t, w in timeline["events"][:4]) + " ...")
            min_match = float(g("judge.min_script_match", 70))
            feedback, s, check = "", None, None
            best = None                                   # (match, script_dict, check): the best attempt so far
            from . import styles as _styles
            narrate.STYLE_WEIGHTS = _styles.weights(self.store, self.cfg)
            narrate.PLAYBOOK = metadata.PLAYBOOK = learn.playbook_text(self.store, self.cfg)
            for attempt in range(int(g("judge.script_attempts", 2))):
                s = narrate.write_script(self.llm, self.cfg, j, tr.text, dur, images=tl_imgs, feedback=feedback,
                                         timeline=timeline)
                check = narrate.verify_script(self.llm, self.cfg, j, s["script"], tl_imgs, times=tl_times,
                                              beats=s.get("beats"))
                if check["overlay"]:
                    self.store.update_clip(clip_id, status="dropped",
                                           error="streamer / webcam / chat visible in the final picture")
                    return "dropped"
                if best is None or check["match"] > best[0]:
                    best = (check["match"], s, check)
                if check["match"] >= min_match:
                    break
                feedback = f"the checker said the script does not match the pictures or their timing ({check['problems']})"
                log(f"  script match {check['match']:.0f}/100, rewriting ...")
            else:
                floor = float(g("judge.script_floor", 35))
                if best and best[0] >= floor:             # not perfect, but usable: keep the best try instead of losing the clip
                    _, s, check = best
                    log(f"  script match only {check['match']:.0f}/100: using the best try anyway (floor {floor:.0f})")
                else:
                    self.store.update_clip(clip_id, status="dropped",
                                           error=f"script does not match the video ({check['match']:.0f}/100: {check['problems']})")
                    return "dropped"
            script, hook = s["script"], s.get("title") or s.get("on_screen_hook", "")
            beats = s.get("beats")
            j["beats"], j["beats_script"] = beats, script
            j["styles"] = s.get("styles") or {}
            j["hook"] = hook
            self.store.update_clip(clip_id, script=script, judgement=j)
        hook = hook or j.get("hook") or j.get("hook_title", "")

        wav = cd / "narration.wav"
        n = narrate.synthesize(self.cfg, script, wav, max_duration=dur - lead - 0.8, beats=beats, lead=lead,
                               tone=narrate.tone_for(j, self.cfg))
        need = n.duration + lead + 0.8
        if need > dur:  # extend the clip a little (if the video has footage left and limits allow)
            new_dur = min(need, clip_max + 4, video["duration"] - start)
            if n.duration + lead > new_dur - 0.2:
                raise ValueError(f"voice-over ({n.duration:.1f}s) does not fit the clip ({dur:.1f}s)")
            dur = new_dur
        elif need < dur - 1.0:  # the voice-over is shorter than the clip: end the clip right after it
            dur = max(need, float(g("video.clip_floor", 10)))

        words = n.words
        if not n.aligned and g("captions.align", "auto") in ("auto", "whisper"):
            tw = transcribe_file(wav, self.cfg, words=True)
            if tw.available and len(tw.words) == len(script.split()):
                words = [(a, b, orig) for (a, b, _), orig in zip(tw.words, script.split())]
        ass = None
        if g("captions.enabled"):
            box = compose.title_area(crop, g("layout.mode", "fill"), int(g("video.width")), int(g("video.height")))
            ass = captions.write_ass(cd / "captions.ass", words, hook, self.cfg, offset=lead,
                                     title_box=box, duration=dur, handle=str(g("branding.handle", "") or ""))

        final = cd / "final.mp4"
        try:
            track = music.pick_track(self.cfg, self.work, f"{video['id']}-{clip_id}", self.cfg.root)
        except Exception as e:
            log(f"  (no background music: {e})")
            track = None
        compose.render(video["path"], start, dur, crop, wav, ass, final, self.cfg,
                       has_game_audio=probe(video["path"])["has_audio"], music_path=track)
        result = qa.run_qa(final, self.cfg)
        self.store.update_clip(clip_id, qa=result)
        if not result["ok"]:
            self.store.update_clip(clip_id, status="dropped", error="QA: " + "; ".join(result["problems"]))
            return "dropped"
        name = f"{video['id']:03d}_{clip_id:04d}_{slugify(j['feature_name'])}.mp4"
        dest = self.out / name
        shutil.copy2(final, dest)
        ts = (j.get("styles") or {}).get("title")
        if j.get("manual_meta"):                   # the owner edited title/description by hand: keep them
            try:
                tg = json.loads(c["tags"]) if c["tags"] else []
            except Exception:
                tg = []
            meta = {"title": c["title"] or "", "description": c["description"] or "", "tags": tg, "thumb_text": ""}
        else:
            meta = metadata.generate(self.llm, self.cfg, j, script, hook, src.streamer if src else "",
                                     src.credit_url if src else "",
                                     title_style=narrate.TITLE_STYLES[ts] if isinstance(ts, int) and 0 <= ts < len(narrate.TITLE_STYLES) else "")
        thumb = None
        try:
            frame = metadata.pick_frame(video["path"], start, dur, crop, cd / "thumb")
            thumb = str(metadata.make_thumbnail(frame, dest.with_suffix(".jpg"), metadata.thumb_line(meta, hook, meta["title"]), self.cfg))
        except Exception as e:
            log(f"  (no thumbnail: {e})")
        status = "approved" if g("review.auto_approve") else "ready"
        self.store.update_clip(clip_id, status=status, path=str(dest), title=meta["title"],
                               description=meta["description"], tags=meta["tags"], thumb=thumb, error=None)
        return status

    def build(self, video_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.store.clips(video_id, "judged"):
            log(f"building clip {c['id']} ({json.loads(c['judgement'])['feature_name']}) ...")
            try:
                st = self.build_clip(c["id"])
            except LLMError as e:
                log(f"  LLM unavailable, stopping build: {e}")
                break
            except Exception as e:
                self.store.update_clip(c["id"], status="error", error=f"build: {type(e).__name__}: {e}")
                log(f"  error: {e}")
                st = "error"
            counts[st] = counts.get(st, 0) + 1
            why = self.store.clip(c["id"])["error"] if st in ("dropped", "error") else None
            log(f"  -> {st}" + (f" ({why})" if why else ""))
        self.store.set_video_stage(video_id, "built")
        return counts

    # ------------------------------------------------------------ whole run
    def run(self, source_id: str, path=None) -> int:
        vid = self.scan(source_id, path)
        self.judge(vid)
        self.build(vid)
        return vid

    # ------------------------------------------------------------ review + upload
    def approve(self, ids: list[int]) -> None:
        for i in ids:
            if self.store.clip(i)["status"] != "ready":
                log(f"clip {i}: status is {self.store.clip(i)['status']}, only 'ready' clips can be approved")
                continue
            self.store.update_clip(i, status="approved")
            log(f"clip {i}: approved")

    def reject(self, ids: list[int]) -> None:
        for i in ids:
            self.store.update_clip(i, status="rejected")
            log(f"clip {i}: rejected")

    def uploads_today(self) -> int:
        start_of_day = time.mktime(time.strptime(time.strftime("%Y-%m-%d"), "%Y-%m-%d"))
        return sum(1 for c in self.store.clips(status="uploaded") if c["updated"] >= start_of_day)

    def scheduled_times(self) -> list[datetime]:
        """Publish times that are already taken by uploaded clips (future ones only matter)."""
        out = []
        for c in self.store.clips(status="uploaded"):
            if c["scheduled_at"]:
                out.append(datetime.fromtimestamp(c["scheduled_at"], timezone.utc))
        return out

    def upload(self, limit: int | None = None, dry_run: bool = False, only: list[int] | None = None) -> int:
        """Upload approved clips. With schedule.enabled each clip gets the next free publish slot (PKT times from
        schedule.slots, e.g. 3 per day) and YouTube publishes it by itself at that time."""
        self.upload_results = []
        if safe.is_on(self.store):
            log("SAFE MODE is on: nothing is uploaded (switch it off in the Assistant tab)")
            return 0
        until = float(self.store.kv_get("upload_pause_until", "0") or 0)
        if not dry_run and time.time() < until:
            log("YouTube's upload limit was reached: uploads wait until "
                + datetime.fromtimestamp(until, schedule.PKT).strftime("%H:%M PKT") + " (approved clips are kept)")
            return 0
        sched = bool(self.cfg.get("schedule.enabled", True))
        slots = schedule.parse_slots(self.cfg.get("schedule.slots", schedule.DEFAULT_SLOTS)) or schedule.parse_slots(schedule.DEFAULT_SLOTS)
        now = datetime.now(timezone.utc)
        taken = self.scheduled_times()
        if sched:
            ahead = int(self.cfg.get("schedule.max_ahead_days", 3)) * len(slots)
            room = max(0, ahead - sum(1 for t in taken if t > now))
        else:
            room = max(0, int(self.cfg.get("upload.max_per_day", 3)) - self.uploads_today())
        if limit is not None:
            room = min(room, limit)
        done = 0
        def _quality(c) -> float:                      # the best clips get the earliest publish slots
            try:
                j = json.loads(c["judgement"] or "{}")
                return float(j.get("interest", 0)) + float(j.get("confidence", 0)) * 0.3
            except Exception:
                return 0.0
        for c in sorted(self.store.clips(status="approved"), key=lambda c: -_quality(c)):
            if only is not None and c["id"] not in only:
                continue
            if done >= room:
                log("upload limit reached (" + ("schedule is full for the next days; raise Settings > schedule days ahead"
                    if sched else "daily cap upload.max_per_day") + ")")
                break
            when, public_now, launch_idx = None, False, None
            if sched:
                lt = launch.next_time(self.store, now.timestamp())
                if lt is not None:                           # launch mode: now, +1 h, +2 h ... then the normal slots
                    launch_idx = lt[0]
                    if lt[1] is None:
                        public_now = True
                    else:
                        when = lt[1]
                if when is None and not public_now:
                    nxt = schedule.next_slots(now, 1, slots, taken)
                    if not nxt:
                        log("no free publish slot found")
                        break
                    when = nxt[0]
            tags = json.loads(c["tags"]) if c["tags"] else None
            body = uploader.build_body(c["title"], c["description"], self.cfg, tags,
                                       schedule.iso_utc(when) if when else None, public=public_now)
            if dry_run:
                log(f"[dry-run] would upload clip {c['id']}: {c['title']!r} "
                    + (f"publish at {schedule.fmt_pkt(when)}" if when else f"privacy={body['status']['privacyStatus']}"))
                if when:
                    taken.append(when)
                done += 1
                continue
            try:
                r = uploader.upload_full(c["path"], body, self.cfg, c["thumb"])
            except Exception as e:
                if "uploadLimitExceeded" in str(e) or "exceeded the number of videos" in str(e):
                    hours = float(self.cfg.get("upload.limit_pause_hours", 6))
                    self.store.kv_set("upload_pause_until", str(time.time() + hours * 3600))
                    self.store.update_clip(c["id"], error="YouTube daily upload limit reached; will retry later")
                    log(f"YouTube says the channel reached its upload limit: stopping uploads for {hours:g} h "
                        f"(approved clips stay in the queue, nothing is lost)")
                    self.upload_results.append({"clip": c["id"], "ok": False, "limit": True, "error": str(e)[:300]})
                    break                                    # no point trying the other clips: they would fail the same way
                self.store.update_clip(c["id"], error=f"upload: {type(e).__name__}: {e}")
                log(f"clip {c['id']}: upload failed: {e}")
                self.upload_results.append({"clip": c["id"], "ok": False, "error": str(e)[:300]})
                continue
            state = "public now" if str(r["privacy"]).startswith("public") else (
                f"scheduled {schedule.fmt_pkt(when)}" if (when and r["publish_at"]) else
                ("private - schedule was ignored by YouTube (API project not audited): publish it in YouTube Studio"
                 if when else str(r["privacy"])))
            if when:
                taken.append(when)
            if launch_idx is not None:
                launch.mark_used(self.store, launch_idx)
            self.store.update_clip(c["id"], status="uploaded", youtube_id=r["id"], error=None,
                                   scheduled_at=(when.timestamp() if when else (now.timestamp() if public_now else None)),
                                   publish_state=state)
            log(f"clip {c['id']}: uploaded https://youtube.com/shorts/{r['id']}  ({state})"
                + ("" if r["thumb_ok"] is None else (" thumbnail set" if r["thumb_ok"] else f" thumbnail not accepted ({r['thumb_error'][:60]})")))
            self.upload_results.append({"clip": c["id"], "ok": True, "id": r["id"], "state": state,
                                        "scheduled": bool(when and r["publish_at"]), "thumb_ok": r["thumb_ok"]})
            done += 1
        return done
