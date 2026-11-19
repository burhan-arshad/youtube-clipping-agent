"""End to end on a synthetic stream. The LLM is faked, Whisper is faked and the voice is a
dummy tone (so the test needs no internet, no API key and no 100 MB models).
The real Kokoro voice is covered by test_real_kokoro (skipped if the model is absent)."""
import json
import os
import shutil
from pathlib import Path

import pytest

from featureclip import pipeline as pl
from featureclip.ffutil import probe
from featureclip.sources import Source
from featureclip.store import Store
from featureclip.transcribe import Transcript

SCRIPT = ("Wait, look at what this car just did in GTA 6. The front folds inward on impact. "
          "The glass cracks across the windshield. Every crash looks different. "
          "The metal bends in a new way each time. It is a small detail that makes "
          "the whole world feel real. Which detail did you spot first?")


class ScriptedLLM:
    """Pretends to be Gemini: accepts the first two windows it sees, rejects the rest."""

    def __init__(self):
        self.judged = 0
        self.script_calls = 0

    def available(self):
        return True

    def json(self, prompt, system="", images=None, temperature=0.4):
        if "Allowed categories" in prompt:
            self.judged += 1
            assert images, "vision frames must be sent to the judge"
            if self.judged <= 2:
                return {"feature_shown": True, "category": "vehicle", "confidence": 90, "interest": 80,
                        "feature_name": f"Realistic car damage {self.judged}",
                        "visible_evidence": ["the bonnet folds inward", "glass cracks on the screen"],
                        "real_person_or_overlay_visible": False, "start_offset": 6,
                        "end_offset": 30, "what_happens": "A car crumples on impact.",
                        "hook_title": "Car damage is wild", "reason": "visible"}
            return {"feature_shown": False, "category": "nothing_notable", "confidence": 90, "interest": 5}
        if "strict quality checker" in prompt:
            return {"match": 90, "real_person_or_overlay_visible": False, "problems": ""}
        if "screenshots" in prompt:           # the layout helper: give an unusable answer -> classic detector
            return {}
        if "YouTube Shorts metadata" in prompt or "For EACH frame say what is visible" in prompt:   # metadata helper: unusable -> plain builder
            return {}
        self.script_calls += 1
        return {"script": SCRIPT, "on_screen_hook": "WILD CAR DAMAGE"}


@pytest.fixture
def env(cfg, test_video, monkeypatch):
    monkeypatch.setattr(pl, "transcribe_file",
                        lambda wav, c, words=False, offset=0.0: Transcript(text="wait look at this", available=True)
                        if not words else Transcript(available=False))
    cfg.data["video"]["clip_min"] = 15
    cfg.data["video"]["clip_max"] = 30
    store = Store(cfg.path("paths.db"))
    src = Source(id="synthetic", streamer="TestStreamer",
                 path=str(test_video),
                 credit_url="https://example.test/@t")
    llm = ScriptedLLM()
    return cfg, store, {"synthetic": src}, llm, pl.Pipeline(cfg, store, {"synthetic": src}, llm)


def test_full_run(env):
    cfg, store, sources, llm, p = env
    vid = p.run("synthetic")

    clips = store.clips(vid)
    by_status = {}
    for c in clips:
        by_status.setdefault(c["status"], []).append(c)
    ready = by_status.get("ready", [])
    assert len(ready) == 2, {k: len(v) for k, v in by_status.items()}
    assert llm.script_calls == 2

    for c in ready:
        out = Path(c["path"])
        assert out.is_file() and out.parent == cfg.path("paths.output")
        info = probe(out)
        assert (info["width"], info["height"]) == (1080, 1920)      # vertical Short
        assert info["has_audio"] and 8 <= info["duration"] <= 59
        assert json.loads(c["qa"])["ok"]
        assert c["title"].endswith("#Shorts") and "TestStreamer" in c["description"]
        assert c["script"] == SCRIPT
        j = json.loads(c["judgement"])
        assert j["category"] == "vehicle" and j["accept"]

    # the two clips are different features around the two events (25 s and 50 s)
    centers = sorted(c["start"] for c in ready)
    assert centers[0] < 30 < centers[1]


def test_review_gate_and_upload_cap(env, monkeypatch):
    cfg, store, sources, llm, p = env
    vid = p.run("synthetic")
    ready = store.clips(vid, "ready")
    assert ready and not store.clips(status="approved")             # nothing is auto-approved
    assert p.upload(dry_run=True) == 0                              # nothing approved -> nothing to upload
    p.approve([ready[0]["id"]])
    assert [c["id"] for c in store.clips(status="approved")] == [ready[0]["id"]]
    assert p.upload(dry_run=True) == 1
    p.reject([ready[1]["id"]])
    assert store.clip(ready[1]["id"])["status"] == "rejected"

    uploaded = []
    monkeypatch.setattr(pl.uploader, "upload_full",
                        lambda path, body, c, thumb=None: (uploaded.append(body) or
                            {"id": "abc123", "privacy": "private", "publish_at": body["status"].get("publishAt"),
                             "thumb_ok": None, "thumb_error": ""}))
    assert p.upload() == 1
    assert store.clip(ready[0]["id"])["status"] == "uploaded"
    assert store.clip(ready[0]["id"])["youtube_id"] == "abc123"
    assert uploaded[0]["status"]["privacyStatus"] == "private"
    assert p.upload() == 0                                          # already uploaded


def test_daily_cap(env, monkeypatch):
    cfg, store, sources, llm, p = env
    cfg.data["upload"]["max_per_day"] = 1
    cfg.data.setdefault("schedule", {})["enabled"] = False
    p.run("synthetic")
    p.approve([c["id"] for c in store.clips(status="ready")])
    monkeypatch.setattr(pl.uploader, "upload_full",
                        lambda path, body, c, thumb=None: {"id": "vid", "privacy": "private", "publish_at": None,
                                                           "thumb_ok": None, "thumb_error": ""})
    assert p.upload() == 1
    assert p.upload() == 0                                          # cap reached for today


def test_resume_does_not_redo_work(env, monkeypatch):
    cfg, store, sources, llm, p = env
    vid = p.run("synthetic")
    calls = {"n": 0}
    monkeypatch.setattr(pl.signals, "compute_signals",
                        lambda *a, **k: calls.__setitem__("n", calls["n"] + 1) or {})
    p.scan("synthetic")                                             # signals come from the cache
    assert calls["n"] == 0
    assert len(store.clips(vid, "ready")) == 2                      # earlier clips untouched


def test_game_audio_is_removed_by_default(env):
    cfg, store, sources, llm, p = env
    vid = p.run("synthetic")
    from featureclip.ffutil import run
    c = store.clips(vid, "ready")[0]
    # the narration is a quiet 180 Hz dummy tone; the source's 660 Hz burst must not be audible
    proc = run(["ffmpeg", "-v", "error", "-i", c["path"], "-vn", "-ac", "1", "-ar", "8000",
                "-f", "f32le", "-"])
    import numpy as np
    a = np.frombuffer(proc.stdout, dtype=np.float32)
    spec = np.abs(np.fft.rfft(a))
    freqs = np.fft.rfftfreq(len(a), 1 / 8000)
    band = lambda f: spec[(freqs > f - 15) & (freqs < f + 15)].sum()  # noqa: E731
    assert band(180) > 5 * band(660)


@pytest.mark.skipif(not os.environ.get("FEATURECLIP_MODELS"),
                    reason="set FEATURECLIP_MODELS to a folder containing kokoro/ model files")
def test_real_kokoro(cfg, tmp_path):
    from featureclip import narrate
    cfg.data["paths"]["models"] = os.environ["FEATURECLIP_MODELS"]
    cfg.data["narration"]["engine"] = "kokoro"
    n = narrate.synthesize(cfg, SCRIPT, tmp_path / "v.wav", max_duration=30)
    assert 8 < n.duration < 30 and len(n.words) == len(SCRIPT.split())


def test_launch_mode_now_then_hourly_then_normal_slots(env, monkeypatch):
    from featureclip import launch
    cfg, store, sources, llm, p = env
    p.run("synthetic")
    ids = [c["id"] for c in store.clips(status="ready")]
    p.approve(ids)
    launch.start(store, count=2, gap=60)
    bodies = []
    monkeypatch.setattr(pl.uploader, "upload_full",
                        lambda path, body, c, thumb=None: (bodies.append(body) or
                            {"id": f"v{len(bodies)}", "privacy": body["status"]["privacyStatus"],
                             "publish_at": body["status"].get("publishAt"), "thumb_ok": None, "thumb_error": ""}))
    n = p.upload()
    assert n == len(ids) and n >= 2
    assert bodies[0]["status"]["privacyStatus"] == "public" and "publishAt" not in bodies[0]["status"]
    assert bodies[1]["status"]["privacyStatus"] == "private" and "publishAt" in bodies[1]["status"]
    from datetime import datetime, timezone
    t1 = datetime.strptime(bodies[1]["status"]["publishAt"], "%Y-%m-%dT%H:%M:%S.000Z").replace(tzinfo=timezone.utc)
    assert 50 * 60 <= (t1 - datetime.now(timezone.utc)).total_seconds() <= 62 * 60       # about one hour later
    assert launch.status(store)["active"] is False                                       # launch finished


def test_launch_reschedules_existing_videos(env):
    from featureclip import launch
    cfg, store, sources, llm, p = env
    import time
    vid = store.add_video("x", "/tmp/x.mp4", {"duration": 60, "width": 1280, "height": 720, "fps": 30})
    now = time.time()
    cids = []
    for i in range(4):
        cid = store.add_clip(vid, 0, 20, 1.0)
        store.update_clip(cid, status="uploaded", youtube_id=f"y{i}", scheduled_at=now + 86400 + i * 3600)
        cids.append(cid)
    calls = []
    launch.start(store, 3, 60, now)
    r = launch.reschedule_existing(cfg, store, setter=lambda c, v, at, public=False: calls.append((v, at, public)), now=now)
    assert r["moved"] == 4 and not r["failed"]
    assert calls[0] == ("y0", None, True)                         # first one public right now
    assert calls[1][1] and calls[2][1] and calls[1][2] is False   # next two scheduled (1 h and 2 h)
    assert calls[3][1] and calls[3][1] != calls[2][1]             # fourth one goes to a normal slot (its time depends on the clock)


def test_upload_limit_pauses_uploads_without_failing_clips(env, monkeypatch):
    cfg, store, sources, llm, p = env
    p.run("synthetic")
    ids = [c["id"] for c in store.clips(status="ready")]
    p.approve(ids)
    calls = []

    def boom(path, body, c, thumb=None):
        calls.append(1)
        raise RuntimeError('<HttpError 400 "The user has exceeded the number of videos they may upload." reason uploadLimitExceeded>')
    monkeypatch.setattr(pl.uploader, "upload_full", boom)
    assert p.upload() == 0
    assert len(calls) == 1                                           # stops at the first limit error
    assert p.upload_results[0]["limit"] is True
    assert all(store.clip(i)["status"] == "approved" for i in ids)   # nothing lost, nothing marked failed
    assert p.upload() == 0 and len(calls) == 1                       # paused: no new attempt (saves API quota)
    from featureclip import extras
    assert "upload" in extras.humanize_error("uploadLimitExceeded").lower()


def test_quality_gate_skips_boring_clips_but_keeps_best_two(env):
    cfg, store, sources, llm, p = env
    from featureclip import judge as jd
    assert float(cfg.get("judge.quality_floor")) == 55 and int(cfg.get("judge.keep_min")) == 2
