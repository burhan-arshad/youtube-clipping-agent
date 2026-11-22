"""Command line: python -m featureclip <command>"""
from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
from pathlib import Path

from .config import load_config
from .ffutil import have, run
from .pipeline import Pipeline, log
from .store import Store

ROOT_FILES = ["config.example.yaml", ".env.example"]


def _setup(args):
    cfg = load_config(args.config, root=args.root)
    store = Store(cfg.path("paths.db"))
    return cfg, store, {}, Pipeline(cfg, store, {})


def cmd_init(args) -> int:
    root = Path(args.root)
    here = Path(__file__).resolve().parent.parent
    for name in ROOT_FILES:
        src = here / name
        dst = root / name.replace(".example", "")
        if src.is_file() and not dst.exists():
            shutil.copy(src, dst)
            print(f"created {dst}")
    print("Next: edit .env (GEMINI_API_KEY), then: python -m featureclip doctor")
    return 0


def cmd_doctor(args) -> int:
    cfg = load_config(args.config, root=args.root)
    ok = True

    def line(good: bool, msg: str, hint: str = "") -> None:
        nonlocal ok
        print(("  OK   " if good else "  FAIL ") + msg + ("" if good or not hint else f"\n         -> {hint}"))
        ok = ok and good

    print("Tools")
    for t in ("ffmpeg", "ffprobe"):
        line(have(t), t, "install ffmpeg and add it to PATH (winget install Gyan.FFmpeg)")
    if have("ffmpeg"):
        filters = run(["ffmpeg", "-hide_banner", "-filters"]).stdout.decode()
        for f in ("ass", "loudnorm", "sidechaincompress", "blackdetect", "freezedetect"):
            line(f" {f} " in filters, f"ffmpeg filter '{f}'", "use a full ffmpeg build (gyan.dev 'full')")
    print("Python packages")
    for mod in ("numpy", "cv2", "yaml", "requests", "faster_whisper", "kokoro_onnx", "soundfile",
                "yt_dlp", "googleapiclient", "google_auth_oauthlib"):
        try:
            importlib.import_module(mod)
            line(True, mod)
        except Exception as e:
            line(False, mod, f"pip install -r requirements.txt  ({type(e).__name__})")
    print("Keys / files")
    line(bool(cfg.gemini_key) or bool(cfg.openai_key), "LLM key (GEMINI_API_KEY in .env)",
         "free key: https://aistudio.google.com/apikey")
    kdir = cfg.path("paths.models") / "kokoro"
    have_k = all((kdir / n).is_file() for n in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"))
    print(("  OK   " if have_k else "  INFO ") + "Kokoro voice model " +
          ("present" if have_k else "will be downloaded (~120 MB) on first voice-over"))
    sec = cfg.root / cfg.get("upload.client_secret")
    print(("  OK   " if sec.is_file() else "  INFO ") + f"YouTube client secret ({sec.name})" +
          ("" if sec.is_file() else " - only needed for upload"))
    if args.ping:
        from .llm import LLM
        try:
            r = LLM(cfg).json('Reply with exactly this JSON: {"ok": true}')
            line(bool(r.get("ok")), "LLM ping")
        except Exception as e:
            line(False, "LLM ping", str(e))
    if args.whisper:
        from .transcribe import _model
        try:
            _model(cfg)
            line(True, f"Whisper model '{cfg.get('whisper.model')}' loads")
        except Exception as e:
            line(False, "Whisper model", f"{type(e).__name__}: {e}")
    print("\nAll good." if ok else "\nFix the FAIL lines above, then run doctor again.")
    return 0 if ok else 1


def cmd_make_test_video(args) -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from make_test_video import make
    make(args.out, args.seconds)
    print(f"written {args.out}")
    return 0


SELFTEST_SCRIPT = ("Wait, look at what this car just did. The front folds inward on impact. "
                   "The glass cracks across the windshield. Every crash looks different. "
                   "The metal bends in a new way each time. It is a small detail that makes "
                   "the whole world feel real. Which detail did you spot first?")


class _CannedLLM:
    """Stands in for Gemini in `selftest` so no API key is needed."""
    n = 0

    def available(self):
        return True

    def json(self, prompt, system="", images=None, temperature=0.4):
        if "Allowed categories" in prompt:
            _CannedLLM.n += 1
            if _CannedLLM.n <= 1:
                return {"visible_evidence": ["the bonnet folds in", "glass cracks"],
                        "real_person_or_overlay_visible": False, "feature_shown": True, "category": "vehicle", "confidence": 90, "interest": 80,
                        "feature_name": "Realistic car damage", "start_offset": 6, "end_offset": 30,
                        "what_happens": "A car crumples on impact.", "hook_title": "Car damage is wild",
                        "reason": "visible"}
            return {"feature_shown": False, "category": "nothing_notable", "confidence": 90, "interest": 5}
        if "strict quality checker" in prompt:
            return {"match": 90, "real_person_or_overlay_visible": False, "problems": ""}
        return {"script": SELFTEST_SCRIPT, "on_screen_hook": "WILD CAR DAMAGE"}


def cmd_selftest(args) -> int:
    """Whole pipeline on a synthetic stream with the REAL voice, no API key, no real footage."""
    from . import pipeline as pl
    from .ffutil import probe
    from .sources import Source
    from .transcribe import Transcript
    real = load_config(args.config, root=args.root)
    root = Path(args.root).resolve() / "selftest"
    root.mkdir(exist_ok=True)
    cfg = load_config(root=root)
    cfg.data["paths"]["models"] = str(real.path("paths.models"))
    cfg.data["detect"]["skip_start"] = 10
    cfg.data["video"]["clip_min"], cfg.data["video"]["clip_max"] = 15, 30
    cfg.data["captions"]["align"] = "script"
    cfg.data["layout"]["ai"] = False
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from make_test_video import make
    vid_path = root / "stream.mp4"
    if not vid_path.is_file():
        print("1/4 making a 70 s test stream ...")
        make(str(vid_path), duration=70, w=640, h=360, fps=15, events=(25, 50))
    pl.transcribe_file = lambda wav, c, words=False, offset=0.0: (
        Transcript(text="wait look at this", available=True) if not words else Transcript(available=False))
    store = Store(root / "selftest.db")
    store.close()
    (root / "selftest.db").unlink(missing_ok=True)
    store = Store(root / "selftest.db")
    src = Source(id="selftest", streamer="TestStreamer", path=str(vid_path), credit_url="https://example.test")
    p = pl.Pipeline(cfg, store, {"selftest": src}, _CannedLLM())
    print("2/4 running the pipeline (first time it downloads the voice model ~120 MB) ...")
    vid = p.run("selftest")
    ready = store.clips(vid, "ready")
    print("3/4 checking the result ...")
    problems = []
    if not ready:
        problems.append("no clip reached 'ready': " + "; ".join(
            f"#{c['id']} {c['status']} {c['error']}" for c in store.clips(vid)))
    for c in ready:
        info = probe(c["path"])
        if (info["width"], info["height"]) != (1080, 1920):
            problems.append(f"wrong size {info['width']}x{info['height']}")
        if not info["has_audio"]:
            problems.append("no audio")
        print(f"   clip {c['path']}  {info['duration']:.0f}s  QA={json.loads(c['qa'])['ok']}")
    if problems:
        print("\nSELFTEST FAILED:\n - " + "\n - ".join(problems))
        return 1
    print(f"4/4 OK. Open this file and check: picture is vertical gameplay only (no blue border / "
          f"webcam box / HUD strip), voice speaks, captions are visible:\n   {ready[0]['path']}")
    return 0


def cmd_watch(args) -> int:
    from .live import watch
    _, _, _, p = _setup(args)
    watch(p, args.source, url=args.url, segment_seconds=int(args.minutes * 60),
          local_test_file=args.test_file, wait_minutes=args.wait_minutes)
    return 0


def cmd_scan(args) -> int:
    *_, p = _setup(args)
    p.scan(args.source)
    return 0


def cmd_judge(args) -> int:
    *_, p = _setup(args)
    p.judge(args.video_id)
    return 0


def cmd_build(args) -> int:
    *_, p = _setup(args)
    print(p.build(args.video_id))
    return 0


def cmd_run(args) -> int:
    *_, p = _setup(args)
    p.run(args.source)
    return 0


def cmd_list(args) -> int:
    _, store, _, _ = _setup(args)
    rows = store.clips(status=args.status)
    if not rows:
        print("no clips")
    for c in rows:
        j = json.loads(c["judgement"]) if c["judgement"] else {}
        print(f"#{c['id']:<4} video {c['video_id']:<3} {c['status']:<9} "
              f"{c['start']:>7.0f}s  {j.get('feature_name', '-'):<34} {c['path'] or ''}"
              + (f"   [{c['error']}]" if c["error"] else ""))
    return 0


def cmd_layout(args) -> int:
    import cv2
    from . import layout as lay
    from .ffutil import probe
    from .llm import LLM
    if args.video.lower().startswith(("http://", "https://")):
        from .ingest import ingest
        from .sources import resolve_source
        cfg0 = load_config(args.config, root=args.root)
        src = resolve_source(args.video, {})
        print("downloading the video first (once; it is reused by `run`) ...")
        args.video = str(ingest(src, cfg0.path("paths.workdir"), cfg0))
    if not Path(args.video).is_file():
        print(f"File not found: {args.video}\n"
              "Write the REAL path of a video (e.g. D:\\videos\\gta.mp4), not the word video.mp4.\n"
              "Tip: drag the video file into the terminal window and its path is typed for you.")
        return 1
    info = probe(args.video)
    cfg = load_config(args.config, root=args.root)
    L = lay.detect_layout(args.video, info["duration"], info["width"], info["height"],
                          n_frames=int(cfg.get("layout.sample_frames")),
                          face_detect=bool(cfg.get("layout.face_detect")),
                          facecam=args.facecam or cfg.get("layout.facecam", "auto"),
                          facecam_size=tuple(cfg.get("layout.facecam_size", [0.27, 0.38])),
                          crop_frac=cfg.get("layout.crop_frac"),
                          llm=(LLM(cfg) if cfg.get("layout.ai", True) else None),
                          tmp=cfg.path("paths.workdir") / "layout_tmp")
    frames = lay.sample_frames(args.video, [info["duration"] * f for f in (0.3, 0.4, 0.5, 0.6)], 640)
    aspect = 9 / 16 if cfg.get("layout.mode") == "fill" else (1.0, 1.7)
    crop = lay.choose_crop(L, lay.energy_map(frames) if len(frames) > 1 else None, aspect)
    img = lay.draw_debug(frames[len(frames) // 2], L, crop)
    out = Path(args.out)
    cv2.imwrite(str(out), img)
    print(f"gameplay area (green): {L.rect}  facecam avoid (red): {L.avoid or 'none'}  "
          f"crop (yellow): {crop}\nsaved {out}\n"
          "If the green box is wrong, set crop: [x, y, w, h] when you add the link.")
    return 0


def cmd_approve(args) -> int:
    _, store, _, p = _setup(args)
    ids = [c["id"] for c in store.clips(status="ready")] if args.all else args.ids
    p.approve(ids)
    return 0


def cmd_reject(args) -> int:
    *_, p = _setup(args)
    p.reject(args.ids)
    return 0


def cmd_retry(args) -> int:
    _, store, _, _ = _setup(args)
    for i in args.ids:
        store.update_clip(i, status="judged", error=None, script=None)
        log(f"clip {i}: reset to 'judged' (run build again)")
    return 0


def cmd_rejudge(args) -> int:
    """Judge + build an already scanned video again (after the rules or the layout code improved)."""
    _, store, _, p = _setup(args)
    n = 0
    for c in store.clips(args.video_id):
        if c["status"] in ("dropped", "judged", "error", "candidate"):
            store.update_clip(c["id"], status="candidate", judgement=None, error=None, script=None)
            n += 1
    (p.vdir(args.video_id) / "layout.json").unlink(missing_ok=True)
    log(f"video {args.video_id}: {n} windows are judged again")
    p.judge(args.video_id)
    p.build(args.video_id)
    return 0


def cmd_ui(args) -> int:
    from .web.app import serve
    serve(args.root, args.config, port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_backup(args) -> int:
    from . import backup
    cfg = load_config(args.config, root=args.root)
    f = backup.run(cfg)
    print(f"backup saved: {f}")
    return 0


def cmd_restore(args) -> int:
    from . import backup
    cfg = load_config(args.config, root=args.root)
    done = backup.restore(cfg, args.file)
    print("restored:\n  " + "\n  ".join(done) + "\n(start the program again)")
    return 0


def cmd_upload(args) -> int:
    *_, p = _setup(args)
    p.upload(limit=args.limit, dry_run=args.dry_run)
    return 0


def cmd_daily(args) -> int:
    _, _, _, p = _setup(args)
    for s in args.sources:
        try:
            p.run(s)
        except PermissionError as e:
            log(f"SKIPPED {s}: {e}")
    p.upload()
    return 0


TIDY = ["start_ui.bat", "layout.png", "sources.example.yaml", "docs/outreach_message.md", "streamer_channel_kit.zip", "featureclip/bounds.py", "featureclip/syncfix.py", "featureclip/channels.py",
        "tests/test_streamer.py", "tests/test_update34.py", "token_streamer.json",
        ".pytest_cache"]


def cmd_tidy(args) -> int:
    """List (or with --yes delete) files the program does not need: old helpers, caches, leftover update zips."""
    root = Path(args.root).resolve()
    found = [root / n for n in TIDY if (root / n).exists()]
    found += [p for p in root.glob("update*.zip")]
    found += [p for p in root.rglob("__pycache__") if ".venv" not in p.parts and "node_modules" not in p.parts]
    if not found:
        print("Nothing to tidy.")
        return 0
    for p in found:
        print(("delete " if args.yes else "would delete ") + str(p.relative_to(root)))
        if args.yes:
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    if not args.yes:
        print("\nNothing was deleted. Run again with --yes to delete these.")
    return 0


def cmd_remove_streamer(args) -> int:
    """Delete the leftovers of the discontinued streamer channel (dry run unless --yes). Voice-over clips are never touched."""
    from . import legacy
    cfg, store, _, _ = _setup(args)
    r = legacy.remove(cfg, store, apply=args.yes)
    print(f"streamer clips: {r['clips']} ({r['files']} files, {r['mb']} MB), queue entries: {r['queue']}, old settings: {r['kv']}")
    if r["on_youtube"]:
        print(f"{r['on_youtube']} of them were uploaded to the streamer channel (they stay on YouTube; delete them in YouTube Studio).")
    print("Done. A backup was made first." if r["applied"] else "Nothing was deleted. Run again with --yes to delete.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="featureclip", description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--root", default=".", help="project folder holding config.yaml / .env")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create config.yaml, .env from the examples").set_defaults(fn=cmd_init)
    d = sub.add_parser("doctor", help="check that everything is installed")
    d.add_argument("--ping", action="store_true", help="make one tiny LLM call")
    d.add_argument("--whisper", action="store_true", help="try loading the Whisper model")
    d.set_defaults(fn=cmd_doctor)
    t = sub.add_parser("make-test-video", help="synthetic video to try the pipeline")
    t.add_argument("out")
    t.add_argument("--seconds", type=int, default=90)
    t.set_defaults(fn=cmd_make_test_video)
    for name, fn, arg, hlp in [
        ("scan", cmd_scan, "source", "signals + candidate windows (source = name, link or video file)"),
        ("judge", cmd_judge, "video_id", "Whisper + LLM decide which windows show a feature"),
        ("build", cmd_build, "video_id", "voice-over, captions, vertical render, QA"),
        ("run", cmd_run, "source", "scan + judge + build"),
    ]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument(arg, type=int if arg == "video_id" else str)
        s.set_defaults(fn=fn)
    sub.add_parser("selftest", help="test the whole pipeline on a synthetic video (no key, no footage)"
                   ).set_defaults(fn=cmd_selftest)
    w = sub.add_parser("watch", help="LIVE: record a live stream in pieces and clip each finished piece")
    w.add_argument("source")
    w.add_argument("--url", default=None, help="live URL (the stream link)")
    w.add_argument("--minutes", type=float, default=10, help="length of each recorded piece")
    w.add_argument("--wait-minutes", type=int, default=180, help="keep retrying this long if not live yet")
    w.add_argument("--test-file", default=None, help="simulate a live stream from a local video file")
    w.set_defaults(fn=cmd_watch)
    l = sub.add_parser("list", help="show clips")
    l.add_argument("--status", default=None)
    l.set_defaults(fn=cmd_list)
    lo = sub.add_parser("layout", help="draw the detected gameplay area / crop on a frame")
    lo.add_argument("video")
    lo.add_argument("--out", default="layout_debug.png")
    lo.add_argument("--facecam", default=None, help="top-left | top-right | bottom-left | bottom-right | none")
    lo.set_defaults(fn=cmd_layout)
    a = sub.add_parser("approve", help="approve ready clips for upload")
    a.add_argument("ids", type=int, nargs="*")
    a.add_argument("--all", action="store_true")
    a.set_defaults(fn=cmd_approve)
    for name, fn in (("reject", cmd_reject), ("retry", cmd_retry)):
        s = sub.add_parser(name)
        s.add_argument("ids", type=int, nargs="+")
        s.set_defaults(fn=fn)
    ui = sub.add_parser("ui", help="open the web page (settings, run, review, upload)")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument("--no-browser", action="store_true")
    ui.set_defaults(fn=cmd_ui)
    bk = sub.add_parser("backup", help="save database + settings into data/backups now")
    bk.set_defaults(fn=cmd_backup)
    ty = sub.add_parser("tidy", help="list unneeded files (add --yes to delete them)")
    ty.add_argument("--yes", action="store_true")
    ty.set_defaults(fn=cmd_tidy)
    rm = sub.add_parser("remove-streamer", help="delete old streamer clips/settings (add --yes to really delete)")
    rm.add_argument("--yes", action="store_true")
    rm.set_defaults(fn=cmd_remove_streamer)
    rs = sub.add_parser("restore", help="put a backup zip back (close the program first)")
    rs.add_argument("file")
    rs.set_defaults(fn=cmd_restore)
    rj = sub.add_parser("rejudge", help="judge + build a scanned video again with the newest rules")
    rj.add_argument("video_id", type=int)
    rj.set_defaults(fn=cmd_rejudge)
    u = sub.add_parser("upload", help="upload approved clips to YouTube")
    u.add_argument("--limit", type=int, default=None)
    u.add_argument("--dry-run", action="store_true")
    u.set_defaults(fn=cmd_upload)
    dl = sub.add_parser("daily", help="run sources then upload approved clips (for Task Scheduler)")
    dl.add_argument("sources", nargs="+")
    dl.set_defaults(fn=cmd_daily)

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
