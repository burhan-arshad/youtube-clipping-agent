"""Background jobs for the web UI: one worker thread, live log capture, stop for live recording."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import time
import traceback
from argparse import Namespace
from pathlib import Path

from ..config import load_config
from ..pipeline import Pipeline
from ..sources import Source, load_sources
from ..store import Store


class _Router(io.TextIOBase):
    """Sends print() output of a job thread to that job's log; everything else goes to the console."""

    def __init__(self, orig):
        self.orig = orig
        self.local = threading.local()

    def write(self, s):
        job = getattr(self.local, "job", None)
        if job is not None:
            job.feed(s)
        try:
            self.orig.write(s)
        except Exception:
            pass
        return len(s)

    def flush(self):
        try:
            self.orig.flush()
        except Exception:
            pass

    def isatty(self):
        return False


_router_out: _Router | None = None


def install_router() -> None:
    global _router_out
    if _router_out is None:
        _router_out = _Router(sys.stdout)
        sys.stdout = _router_out
        sys.stderr = _Router(sys.stderr)
        sys.stderr.local = _router_out.local      # same thread -> same job


class Job:
    def __init__(self, jid: int, kind: str, title: str, args: dict):
        self.id, self.kind, self.title, self.args = jid, kind, title, args
        self.status = "queued"            # queued | running | done | error | stopped
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.lines: list[str] = []
        self.error = ""
        self.result: dict = {}
        self.stop_event = threading.Event()
        self._buf = ""
        self._lock = threading.Lock()

    def feed(self, s: str) -> None:
        with self._lock:
            self._buf += s
            parts = self._buf.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            self._buf = parts.pop()
            for p in parts:
                p = p.rstrip()
                if p:
                    self.lines.append(p)
            if len(self.lines) > 4000:
                del self.lines[:1000]

    def say(self, msg: str) -> None:
        self.feed(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    def info(self, since: int = 0) -> dict:
        with self._lock:
            lines = self.lines[since:]
            total = len(self.lines)
        return {"id": self.id, "kind": self.kind, "title": self.title, "status": self.status,
                "created": self.created, "started": self.started, "finished": self.finished,
                "error": self.error, "result": self.result, "lines": lines, "total": total}


def web_sources_file(root: Path) -> Path:
    return root / "data" / "web_sources.json"


def load_all_sources(root: Path) -> dict[str, Source]:
    sources = load_sources(root / "sources.yaml")
    f = web_sources_file(root)
    if f.is_file():
        try:
            for sid, d in json.loads(f.read_text(encoding="utf-8")).items():
                sources[sid] = Source(**{k: v for k, v in d.items() if k in Source.__dataclass_fields__})
        except Exception:
            pass
    return sources


def remember_source(root: Path, src: Source) -> None:
    f = web_sources_file(root)
    f.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if f.is_file():
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    data[src.id] = {"id": src.id, "streamer": src.streamer, "url": src.url, "path": src.path,
                    "credit_url": src.credit_url, "facecam": src.facecam, "crop": src.crop, "extra": src.extra}
    f.write_text(json.dumps(data, indent=1), encoding="utf-8")


def parse_crop_frac(v) -> list[float] | None:
    """[x, y, w, h] in percent (0-100) -> fractions, or None. Accepts a list or 'x,y,w,h'."""
    if not v:
        return None
    try:
        vals = [float(x) for x in (v if isinstance(v, (list, tuple)) else str(v).split(","))]
    except ValueError:
        return None
    if len(vals) != 4 or vals[2] < 15 or vals[3] < 15 or any(x < 0 for x in vals) or vals[0] + vals[2] > 100.5 or vals[1] + vals[3] > 100.5:
        return None
    return [round(x / 100.0, 4) for x in vals]


def source_for(url_or_path: str, streamer: str = "", credit_url: str = "", facecam: str = "",
               crop: list[float] | None = None) -> Source:
    v = url_or_path.strip().strip('"')
    h = hashlib.sha1(v.encode()).hexdigest()[:8]
    fc = "" if (facecam or "auto") == "auto" else facecam
    extra = {"crop_frac": parse_crop_frac(crop)} if parse_crop_frac(crop) else {}
    if v.lower().startswith(("http://", "https://")):
        return Source(id=f"web-{h}", streamer=streamer, url=v, credit_url=credit_url, facecam=fc, extra=extra)
    return Source(id=f"file-{h}", streamer=streamer, path=v, credit_url=credit_url, facecam=fc, extra=extra)


def layout_preview(root: Path, config_path, video: str, facecam: str, crop, out: Path, say) -> None:
    """Run the AI layout detection on a video file and draw what will be kept / cut. `crop` = [x,y,w,h] percent."""
    import cv2
    from .. import layout as lay
    from ..ffutil import probe
    from ..llm import LLM
    cfg = load_config(config_path, root=root)
    store = Store(cfg.path("paths.db"))
    info = probe(video)
    frac = parse_crop_frac(crop)
    L = lay.detect_layout(video, info["duration"], info["width"], info["height"],
                          n_frames=int(cfg.get("layout.sample_frames")), face_detect=bool(cfg.get("layout.face_detect")),
                          facecam=(facecam or cfg.get("layout.facecam", "auto")),
                          facecam_size=tuple(cfg.get("layout.facecam_size", [0.27, 0.38])),
                          crop_frac=frac or cfg.get("layout.crop_frac"),
                          llm=(LLM(cfg, store) if cfg.get("layout.ai", True) else None),
                          tmp=cfg.path("paths.workdir") / "layout_tmp")
    frames = lay.sample_frames(video, [info["duration"] * f for f in (0.3, 0.4, 0.5, 0.6)], 640)
    aspect = 9 / 16 if cfg.get("layout.mode") == "fill" else (1.0, 1.7)
    crop_box = lay.choose_crop(L, lay.energy_map(frames) if len(frames) > 1 else None, aspect)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), lay.draw_debug(frames[len(frames) // 2], L, crop_box))
    say(f"layout: {L.origin}; gameplay area {L.rect}; cut (webcam/chat/text): {L.avoid or 'none'}")


class JobManager:
    def __init__(self, root: Path, config_path: str | None = None):
        self.root, self.config_path = root, config_path
        self.jobs: list[Job] = []
        self._next = 1
        self._cv = threading.Condition()
        install_router()
        self._worker = threading.Thread(target=self._loop, daemon=True, name="featureclip-worker")
        self._worker.start()

    # ---- public
    def submit(self, kind: str, title: str, **args) -> Job:
        with self._cv:
            job = Job(self._next, kind, title, args)
            self._next += 1
            self.jobs.append(job)
            self._cv.notify()
        return job

    def get(self, jid: int) -> Job | None:
        return next((j for j in self.jobs if j.id == jid), None)

    def stop(self, jid: int) -> bool:
        j = self.get(jid)
        if not j:
            return False
        j.stop_event.set()
        if j.status == "queued":
            j.status = "stopped"
        return True

    def running(self) -> Job | None:
        return next((j for j in self.jobs if j.status == "running"), None)

    # ---- worker
    def _loop(self):
        while True:
            with self._cv:
                job = next((j for j in self.jobs if j.status == "queued"), None)
                while job is None:
                    self._cv.wait()
                    job = next((j for j in self.jobs if j.status == "queued"), None)
                job.status = "running"
                job.started = time.time()
            assert _router_out is not None
            _router_out.local.job = job
            try:
                getattr(self, f"_do_{job.kind}")(job)
                if job.status == "running":
                    job.status = "stopped" if job.stop_event.is_set() else "done"
            except Exception as e:
                job.status = "stopped" if job.stop_event.is_set() else "error"
                job.error = f"{type(e).__name__}: {e}"
                job.say("ERROR: " + job.error)
                traceback.print_exc()
            finally:
                job.finished = time.time()
                _router_out.local.job = None

    def _pipeline(self):
        cfg = load_config(self.config_path, root=self.root)
        store = Store(cfg.path("paths.db"))
        sources = load_all_sources(self.root)
        return cfg, store, sources, Pipeline(cfg, store, sources)

    # ---- job kinds
    def _do_run(self, job: Job):
        a = job.args
        src = source_for(a["video"], a.get("streamer", ""), a.get("credit_url", ""), a.get("facecam", ""), a.get("crop"))
        cfg, store, sources, p = self._pipeline()
        sources[src.id] = src
        p.stop_event = job.stop_event
        remember_source(self.root, src)
        job.say(f"video: {a['video']}")
        if str(a["video"]).lower().startswith("http"):
            from ..live import is_live
            if is_live(a["video"], cfg):           # a running live stream cannot be downloaded like a normal video
                job.say("This link is a LIVE stream -> recording it live instead of downloading it from the start.")
                return self._do_watch(job)
        from ..ingest import LiveStreamError
        try:
            vid = p.run(src.id)
        except LiveStreamError:               # the live check could not tell, but the download shows it is live
            job.say("This link is a LIVE stream -> recording it live instead of downloading it from the start.")
            return self._do_watch(job)
        counts: dict[str, int] = {}
        for c in store.clips(vid):
            counts[c["status"]] = counts.get(c["status"], 0) + 1
        job.result = {"video_id": vid, "counts": counts}
        job.say(f"Finished. Clips: {counts or 'none'}")

    def _do_rejudge(self, job: Job):
        from ..cli import cmd_rejudge
        vid = int(job.args["video_id"])
        cmd_rejudge(Namespace(config=self.config_path, root=str(self.root), video_id=vid))

    def _do_build(self, job: Job):
        cfg, store, sources, p = self._pipeline()
        p.build(int(job.args["video_id"]))

    def _do_layout(self, job: Job):
        """Preview for a recorded video (link or file) or for a LIVE stream (8 second sample)."""
        from ..cli import cmd_layout  # noqa: F401  (kept for the CLI)
        a = job.args
        out = self.root / "data" / "ui" / f"layout_{job.id}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        video = a["video"].strip().strip('"')
        cfg = load_config(self.config_path, root=self.root)
        if a.get("live"):
            sample = self.root / "data" / "ui" / f"live_sample_{hashlib.sha1(video.encode()).hexdigest()[:8]}.mp4"
            if a.get("refresh") or not sample.is_file():
                from ..live import stream_urls
                import subprocess
                job.say("Grabbing 8 seconds from the live stream ...")
                inputs = stream_urls(video, 720, load_config(self.config_path, root=self.root))
                cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
                for i in inputs:
                    cmd += ["-i", i]
                cmd += ["-t", "8", "-map", "0:v:0", "-map", f"{len(inputs) - 1}:a:0?", "-c", "copy", str(sample)]
                r = subprocess.run(cmd, timeout=90)
                if r.returncode != 0 or not sample.is_file():
                    raise RuntimeError("Could not read the live stream (is it live right now? cookies needed?)")
            path = str(sample)
        elif video.lower().startswith(("http://", "https://")):
            from ..ingest import ingest
            job.say("Downloading the video first (it is reused by 'Make clips') ...")
            path = str(ingest(source_for(video), cfg.path("paths.workdir"), cfg))
        else:
            path = video
        if not Path(path).is_file():
            raise RuntimeError(f"File not found: {path}")
        layout_preview(self.root, self.config_path, path, a.get("facecam") or "", a.get("crop"), out, job.say)
        if not out.is_file():
            raise RuntimeError("Layout image was not created (see the log above)")
        job.result = {"image": str(out)}

    def _do_watch(self, job: Job):
        from ..live import watch
        a = job.args
        src = source_for(a["video"], a.get("streamer", ""), a.get("credit_url", ""), a.get("facecam", ""), a.get("crop"))
        cfg, store, sources, p = self._pipeline()
        sources[src.id] = src
        remember_source(self.root, src)
        local = Path(str(a["video"])).expanduser()
        is_file = local.is_file()          # a video file instead of a link = simulated live test (plays at real speed)
        if is_file:
            job.say("Test mode: the file is played at real speed as if it were a live stream.")
        watch(p, src.id, url=None if is_file else a["video"], local_test_file=str(local) if is_file else None,
              segment_seconds=int(float(a.get("minutes", 10)) * 60),
              stop=job.stop_event, wait_minutes=int(a.get("wait_minutes", 180)))

    def _do_upload(self, job: Job):
        cfg, store, sources, p = self._pipeline()
        n = p.upload(limit=job.args.get("limit"), dry_run=bool(job.args.get("dry_run")), only=job.args.get("only"))
        job.result = {"uploaded": n}

    def _do_youtube_login(self, job: Job):
        from .. import uploader
        cfg = load_config(self.config_path, root=self.root)
        job.say("A browser window will open: choose the Google account that owns your YouTube channel and click Allow ...")
        uploader.get_credentials(cfg)
        job.say("YouTube login complete.")
