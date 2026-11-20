"""Streamer layout mode: webcam on top, gameplay below, the streamer's own audio, no voice-over."""
import json
from pathlib import Path

from featureclip import compose, metadata, pipeline as pl
from featureclip.ffutil import probe
from featureclip.layout import Layout, find_cam
from featureclip.sources import Source
from featureclip.store import Store
from featureclip.transcribe import Transcript

from test_pipeline import ScriptedLLM


def test_layout_cam_roundtrip():
    L = Layout(640, 360, (0, 0, 400, 300), [], "ai", (440, 10, 200, 150))
    assert tuple(L.cam) == (440, 10, 200, 150)
    assert tuple(Layout.from_dict(L.to_dict()).cam) == (440, 10, 200, 150)


def test_find_cam_configured_corner(test_video):
    L = Layout(640, 360, (0, 0, 640, 360))
    cam = find_cam(L, str(test_video), 70, facecam="top-right")
    assert cam is not None and cam[0] > 320 and cam[1] < 180


def test_render_streamer_real_ffmpeg(cfg, test_video, tmp_path):
    out = compose.render_streamer(str(test_video), 5, 6, (440, 0, 200, 150), (0, 0, 640, 360), None,
                                  tmp_path / "s.mp4", cfg, has_audio=True)
    info = probe(out)
    assert (info["width"], info["height"]) == (1080, 1920)
    assert info["has_audio"] and 5 <= info["duration"] <= 7


def test_credit_streamer_mode(cfg):
    cfg.data["layout"]["mode"] = "streamer"
    s = "\n".join(metadata.credit_lines(cfg, "Ducky Bhai"))
    assert "Ducky Bhai" in s and "http" not in s


class StreamLLM(ScriptedLLM):
    def json(self, prompt, *a, **k):
        r = super().json(prompt, *a, **k)
        if "STREAMER-CLIP MODE" in prompt and isinstance(r, dict) and r.get("feature_shown"):
            r.update(category="funny_moment", reaction=80)
        return r


def _env(cfg, test_video, monkeypatch, facecam):
    monkeypatch.setattr(pl, "transcribe_file",
                        lambda wav, c, words=False, offset=0.0: Transcript(available=False))
    cfg.data["video"]["clip_min"] = 15
    cfg.data["video"]["clip_max"] = 30
    cfg.data["layout"]["mode"] = "streamer"
    cfg.data["layout"]["facecam"] = facecam
    store = Store(cfg.path("paths.db"))
    src = Source(id="synthetic", streamer="TestStreamer", path=str(test_video), credit_url="")
    llm = StreamLLM()
    return store, pl.Pipeline(cfg, store, {"synthetic": src}, llm), llm


def test_pipeline_streamer(cfg, test_video, monkeypatch):
    store, p, llm = _env(cfg, test_video, monkeypatch, "top-right")
    vid = p.run("synthetic")
    ready = store.clips(vid, "ready")
    assert ready
    assert llm.script_calls == 0                       # no AI script in streamer mode
    for c in ready:
        info = probe(Path(c["path"]))
        assert (info["width"], info["height"]) == (1080, 1920) and info["has_audio"]
        assert not c["script"]
        assert "TestStreamer" in c["description"]
        assert json.loads(c["judgement"]).get("streamer_clip")


def test_streamer_fallback_without_cam(cfg, test_video, monkeypatch):
    store, p, llm = _env(cfg, test_video, monkeypatch, "none")
    monkeypatch.setattr(pl.Pipeline, "_cam", lambda self, *a, **k: None)
    vid = p.run("synthetic")
    assert store.clips(vid, "ready")
    assert cfg.get("layout.mode") == "streamer"        # mode restored after the fallback


def test_upload_now_is_public_and_ignores_schedule(cfg, test_video, monkeypatch):
    store, p, llm = _env(cfg, test_video, monkeypatch, "top-right")
    cfg.data["layout"]["mode"] = "hybrid"
    cfg.data["upload"]["max_per_day"] = 0
    vid = p.run("synthetic")
    c = store.clips(vid, "ready")[0]
    p.approve([c["id"]])
    sent = []
    monkeypatch.setattr(pl.uploader, "upload_full",
                        lambda path, body, cf, thumb=None: (sent.append(body) or
                            {"id": "x1", "privacy": "public", "publish_at": None, "thumb_ok": None, "thumb_error": ""}))
    assert p.upload(only=[c["id"]], public_now=True) == 1
    assert sent[0]["status"]["privacyStatus"] == "public" and "publishAt" not in sent[0]["status"]
    assert store.clip(c["id"])["status"] == "uploaded"


# ---------------------------------------------------------------- multi channel
def test_channel_of_clip_and_overrides(cfg):
    from featureclip import channels
    assert channels.key_of_clip({"judgement": json.dumps({"streamer_clip": True})}) == "streamer"
    assert channels.key_of_clip({"judgement": json.dumps({})}) == "voiceover"
    assert channels.key_of_clip({"judgement": None}) == "voiceover"
    base = cfg.get("upload.token")
    with channels.use(cfg, "streamer"):
        assert cfg.get("upload.token") == "token_streamer.json"
    assert cfg.get("upload.token") == base


def test_source_mode_scopes_settings(cfg, test_video, monkeypatch):
    from featureclip.web.jobs import source_for
    s = source_for(str(test_video), mode="streamer", clips=3)
    assert s.extra["mode"] == "streamer" and s.extra["clips"] == 3
    ov = pl.source_overrides(cfg, s)
    assert ov["layout.mode"] == "streamer" and ov["video.clip_min"] == 20
    cfg.data["layout"]["mode"] = "streamer"
    v = source_for(str(test_video), mode="voiceover")
    assert pl.source_overrides(cfg, v)["layout.mode"] == "hybrid"


def test_wanted_clip_count_limits_judge(cfg, test_video, monkeypatch):
    store, p, llm = _env(cfg, test_video, monkeypatch, "top-right")
    p.sources["synthetic"].extra = {"mode": "streamer", "clips": 1, "since": 0}
    vid = p.run("synthetic")
    assert len(store.clips(vid, "ready")) <= 1


def test_uploads_go_to_their_own_channel(cfg, test_video, monkeypatch):
    from featureclip import channels
    store, p, llm = _env(cfg, test_video, monkeypatch, "top-right")
    vid = p.run("synthetic")
    c = store.clips(vid, "ready")[0]
    j = json.loads(c["judgement"]); j["streamer_clip"] = True
    store.update_clip(c["id"], judgement=json.dumps(j), status="approved")
    used = []
    monkeypatch.setattr(channels, "logged_in", lambda cfg_, key: True)
    monkeypatch.setattr(pl.uploader, "upload_full",
                        lambda path, body, cf, thumb=None: (used.append(cf.get("upload.token")) or
                            {"id": "y1", "privacy": "private", "publish_at": "x", "thumb_ok": None, "thumb_error": ""}))
    assert p.upload() == 1
    assert used == ["token_streamer.json"]


def test_login_and_mode_reach_the_job(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from featureclip.web import jobs as J
    from featureclip.web.app import create_app
    seen = []
    class Fake:
        id = 1
    monkeypatch.setattr(J.JobManager, "submit", lambda self, kind, title, **kw: (seen.append((kind, kw)) or Fake()))
    c = TestClient(create_app(tmp_path))
    assert c.post("/api/jobs", json={"kind": "youtube_login", "channel": "streamer"}).status_code == 200
    assert seen[-1][1]["channel"] == "streamer"
    assert c.post("/api/jobs", json={"kind": "youtube_login", "channel": "x"}).status_code == 400
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTEST12345678901234")
    assert c.post("/api/jobs", json={"kind": "run", "video": "a.mp4", "mode": "streamer", "clips": 4}).status_code == 200
    assert seen[-1][1]["mode"] == "streamer" and seen[-1][1]["clips"] == 4
    assert c.post("/api/jobs", json={"kind": "run", "video": "a.mp4", "mode": "bad"}).status_code == 400


def test_streamer_accepts_only_strong_reactions(cfg):
    from featureclip import judge
    cfg.data["layout"]["mode"] = "streamer"
    base = {"feature_shown": True, "category": "funny_moment", "feature_name": "x", "what_happens": "y",
            "confidence": 90, "interest": 80, "visible_evidence": ["a", "b"], "reaction": 80,
            "real_person_or_overlay_visible": True}
    assert judge.decide(dict(base), cfg)[0]
    assert not judge.decide({**base, "reaction": 30}, cfg)[0]
    assert not judge.decide({**base, "reaction": None}, cfg)[0]
    assert not judge.decide({**base, "category": "new_feature"}, cfg)[0]       # a feature without a reaction


def test_streamer_judge_sees_face_closeups(cfg, test_video, tmp_path):
    from featureclip import judge
    cfg.data["layout"]["mode"] = "streamer"
    seen = {}
    class L:
        def json(self, prompt, system="", images=None, **k):
            seen["n"], seen["prompt"] = len(images or []), prompt
            return {"feature_shown": True, "category": "funny_moment", "feature_name": "scream", "what_happens": "x",
                    "confidence": 90, "interest": 80, "visible_evidence": ["a", "b"], "reaction": 80}
    j = judge.judge_window(L(), cfg, str(test_video), 5, 35, "oh my god!", 60.0, tmp_path, None, (440, 0, 200, 150))
    assert j["accept"], j["decision_reason"]
    assert seen["n"] == int(cfg.get("llm.frames", 4)) + 2 and "close-ups of the streamer's face" in seen["prompt"]
