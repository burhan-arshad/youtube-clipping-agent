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

SCRIPT = ("Wait, look at what this car just did. The front folds inward on impact. "
          "The glass cracks across the windshield. Every crash looks different. "
          "It is a small detail that makes the world feel real.")


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
                        "feature_name": f"Realistic car damage {self.judged}", "start_offset": 6,
                        "end_offset": 30, "what_happens": "A car crumples on impact.",
                        "hook_title": "Car damage is wild", "reason": "visible"}
            return {"feature_shown": False, "category": "nothing_notable", "confidence": 90, "interest": 5}
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
    src = Source(id="synthetic", streamer="TestStreamer", permission="granted",
                 permission_proof="synthetic test video", path=str(test_video),
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
    monkeypatch.setattr(pl.uploader, "upload_video",
                        lambda path, body, c: (uploaded.append(body) or ("abc123", "private")))
    assert p.upload() == 1
    assert store.clip(ready[0]["id"])["status"] == "uploaded"
    assert store.clip(ready[0]["id"])["youtube_id"] == "abc123"
    assert uploaded[0]["status"]["privacyStatus"] == "private"
    assert p.upload() == 0                                          # already uploaded


def test_daily_cap(env, monkeypatch):
    cfg, store, sources, llm, p = env
    cfg.data["upload"]["max_per_day"] = 1
    p.run("synthetic")
    p.approve([c["id"] for c in store.clips(status="ready")])
    monkeypatch.setattr(pl.uploader, "upload_video", lambda path, body, c: ("vid", "private"))
    assert p.upload() == 1
    assert p.upload() == 0                                          # cap reached for today


def test_pending_permission_is_refused(env):
    cfg, store, sources, llm, p = env
    sources["synthetic"].permission = "pending"
    with pytest.raises(PermissionError):
        p.scan("synthetic")
    assert store.videos() == []                                     # nothing was even registered


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
