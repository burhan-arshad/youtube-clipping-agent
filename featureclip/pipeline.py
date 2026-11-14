"""The whole flow, stage by stage. Every stage is resumable: results are cached in the
work folder and in SQLite, so a crash or Ctrl+C never loses finished work.

  scan   : permission check -> get video -> signals -> candidate windows
  judge  : Whisper on candidates -> LLM (frames + words) -> accept/drop
  build  : crop to gameplay -> script -> voice-over -> captions -> render -> QA -> ready
  upload : only clips that a human approved (or auto_approve=true)
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from . import captions, candidates, compose, judge, layout as lay, narrate, qa, signals, uploader
from .ffutil import extract_audio_wav, probe
from .ingest import describe, ingest
from .llm import LLM, LLMError
from .sources import Source
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
        if source_id not in self.sources:
            raise KeyError(f"unknown source '{source_id}' (see sources.yaml)")
        src = self.sources[source_id]
        src.check_permission()
        path = Path(path) if path else ingest(src, self.work)   # ingest also checks permission
        info = describe(path)
        vid = self.store.add_video(src.id, str(path), info)
        vdir = self.vdir(vid)
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
        accepted: list[tuple[int, dict]] = [
            (c["id"], json.loads(c["judgement"])) for c in self.store.clips(video_id, "judged")]
        log(f"judging up to {len(todo)} candidates (stops early once enough good ones are found)")
        for c in todo:
            if len(accepted) >= int(max_clips * 1.5):
                break
            cd = self.cdir(video_id, c["id"])
            try:
                tr = self._transcript_for(path, c)
                j = judge.judge_window(self.llm, self.cfg, path, c["start"], c["end"], tr.text,
                                       c["signal_score"], cd / "frames")
            except LLMError as e:
                log(f"  clip {c['id']}: LLM unavailable -> stopping judge ({e})")
                break
            except Exception as e:  # one bad window must not stop the run
                self.store.update_clip(c["id"], error=f"judge: {type(e).__name__}: {e}")
                log(f"  clip {c['id']}: judge error {e}")
                continue
            if j["accept"]:
                self.store.update_clip(c["id"], status="judged", judgement=j)
                accepted.append((c["id"], j))
                log(f"  clip {c['id']} @ {c['start']:.0f}s: ACCEPT {j['feature_name']!r} "
                    f"(conf {j['confidence']:.0f}, interest {j['interest']:.0f})")
            else:
                self.store.update_clip(c["id"], status="dropped", judgement=j, error=j["decision_reason"])
                log(f"  clip {c['id']} @ {c['start']:.0f}s: drop ({j['decision_reason']})")
        keep = set(judge.dedupe(accepted, max_clips))
        # same feature already made from an earlier video/live segment of this source -> skip
        seen = self._seen_features(video["source_id"], video_id)
        for cid, j in accepted:
            if cid in keep and judge.feature_key(j["feature_name"]) in seen:
                keep.discard(cid)
                log(f"  clip {cid}: feature {j['feature_name']!r} already made before -> skipped")
        for cid, _ in accepted:
            if cid not in keep:
                self.store.update_clip(cid, status="dropped", error="duplicate feature or over the per-video limit")
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
        L = lay.detect_layout(video["path"], video["duration"], video["width"], video["height"],
                              n_frames=int(self.cfg.get("layout.sample_frames")),
                              face_detect=bool(self.cfg.get("layout.face_detect")),
                              manual=(src.crop if src else None))
        cache.write_text(json.dumps(L.to_dict()), encoding="utf-8")
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
        tr = self._transcript_for(video["path"], c)

        L = self._layout(video["id"], src)
        aspect = 9 / 16 if g("layout.mode") == "fill" else 4 / 5
        frames = lay.sample_frames(video["path"], list(np.linspace(start, start + dur, 8)), 640)
        energy = lay.energy_map(frames) if len(frames) >= 2 else None
        crop = lay.choose_crop(L, energy, aspect)

        script = c["script"]
        hook = ""
        if not script:
            s = narrate.write_script(self.llm, self.cfg, j, tr.text, dur)
            script, hook = s["script"], s["on_screen_hook"]
            self.store.update_clip(clip_id, script=script)
        hook = hook or j.get("hook_title", "")

        wav = cd / "narration.wav"
        n = narrate.synthesize(self.cfg, script, wav, max_duration=dur - lead - 0.8)
        need = n.duration + lead + 0.8
        if need > dur:  # extend the clip a little (if the video has footage left and limits allow)
            new_dur = min(need, clip_max + 4, video["duration"] - start)
            if n.duration + lead > new_dur - 0.2:
                raise ValueError(f"voice-over ({n.duration:.1f}s) does not fit the clip ({dur:.1f}s)")
            dur = new_dur

        words = n.words
        if g("captions.align", "auto") in ("auto", "whisper"):
            tw = transcribe_file(wav, self.cfg, words=True)
            if tw.available and len(tw.words) == len(script.split()):
                words = [(a, b, orig) for (a, b, _), orig in zip(tw.words, script.split())]
        ass = None
        if g("captions.enabled"):
            ass = captions.write_ass(cd / "captions.ass", words, hook, self.cfg, offset=lead)

        final = cd / "final.mp4"
        compose.render(video["path"], start, dur, crop, wav, ass, final, self.cfg,
                       has_game_audio=probe(video["path"])["has_audio"])
        result = qa.run_qa(final, self.cfg)
        self.store.update_clip(clip_id, qa=result)
        if not result["ok"]:
            self.store.update_clip(clip_id, status="dropped", error="QA: " + "; ".join(result["problems"]))
            return "dropped"
        name = f"{video['id']:03d}_{clip_id:04d}_{slugify(j['feature_name'])}.mp4"
        dest = self.out / name
        shutil.copy2(final, dest)
        title = uploader.build_title(hook or j["feature_name"], j["feature_name"], g("upload.title_suffix", ""))
        desc = uploader.build_description(j, src.streamer if src else "", src.credit_url if src else "", self.cfg)
        status = "approved" if g("review.auto_approve") else "ready"
        self.store.update_clip(clip_id, status=status, path=str(dest), title=title, description=desc, error=None)
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
            log(f"  -> {st}")
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

    def upload(self, limit: int | None = None, dry_run: bool = False) -> int:
        cap = int(self.cfg.get("upload.max_per_day", 3))
        room = max(0, cap - self.uploads_today())
        if limit is not None:
            room = min(room, limit)
        done = 0
        for c in self.store.clips(status="approved"):
            if done >= room:
                log(f"daily upload cap reached ({cap}/day, set upload.max_per_day)")
                break
            body = uploader.build_body(c["title"], c["description"], self.cfg)
            if dry_run:
                log(f"[dry-run] would upload clip {c['id']}: {c['title']!r} privacy={body['status']['privacyStatus']}")
                done += 1
                continue
            try:
                vid, privacy = uploader.upload_video(c["path"], body, self.cfg)
            except Exception as e:
                self.store.update_clip(c["id"], error=f"upload: {type(e).__name__}: {e}")
                log(f"clip {c['id']}: upload failed: {e}")
                continue
            self.store.update_clip(c["id"], status="uploaded", youtube_id=vid, error=None)
            log(f"clip {c['id']}: uploaded https://youtube.com/shorts/{vid}  (privacy: {privacy})")
            done += 1
        return done
