"""Tests for the 'final polish' update: quota days, model fallbacks, punch-up, style learning, pre-filter,
duplicates, captions, first comment, weekly report, self-test, clip editor."""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pytest

from featureclip import extras, judge, narrate, quota, styles
from featureclip.config import load_config
from featureclip.store import Store


@pytest.fixture()
def cfg0(tmp_path):
    return load_config(root=tmp_path)


@pytest.fixture()
def st(tmp_path):
    return Store(tmp_path / "t.db")


# ------------------------------------------------------------------ quota / models
def test_quota_day_is_pacific_and_reset_in_pkt():
    # 2026-11-18 07:30 UTC = 2026-11-17 23:30 PST -> still the 17th; reset in 30 min (= 13:00 PKT)
    t = datetime(2026, 11, 18, 7, 30, tzinfo=timezone.utc).timestamp()
    assert quota.quota_day(t) == "2026-11-17"
    r = quota.next_quota_reset(t)
    assert 25 * 60 < r - t < 35 * 60
    assert quota.reset_text(t) == "13:00 PKT"
    assert quota.quota_day(t + 3600) == "2026-11-18"


def test_old_config_models_get_fallbacks_and_big_limits_first(tmp_path):
    (tmp_path / "config.yaml").write_text("llm:\n  models: ['gemini-3.5-flash', 'gemini-3.5-flash-lite']\n")
    c = load_config(root=tmp_path)
    m = c.get("llm.models")
    assert m[0] in ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite")      # 500/day models first
    assert "gemini-3.5-flash" in m and len(m) >= 8 and m[-1].startswith("gemma")
    (tmp_path / "config.yaml").write_text("llm:\n  models_locked: true\n  models: ['a', 'b']\n")
    assert load_config(root=tmp_path).get("llm.models") == ["a", "b"]


def test_llm_budget_and_quality_order(cfg0, st, monkeypatch):
    from featureclip.llm import LLM
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    llm = LLM(cfg0, st)
    assert llm.ordered_models(True)[0] == "gemini-3.6-flash"
    assert llm.ordered_models(False)[0] == "gemini-3.5-flash-lite"
    assert not llm._over_budget("gemini-3.6-flash")
    for _ in range(20):
        st.bump_usage("gemini-3.6-flash")            # 20 of 20 per day (limit 96 % = 19)
    assert llm._over_budget("gemini-3.6-flash")


def test_gemma_is_text_only(cfg0, st, monkeypatch):
    from featureclip.llm import LLM
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    llm = LLM(cfg0, st)
    llm.cfg.data["llm"]["models"] = ["gemma-4-31b-it"]
    calls = []
    llm._call = lambda *a, **k: calls.append(a) or '{"ok": true}'
    with pytest.raises(Exception):
        llm.json("x", images=[b"img"])               # no usable model for image calls
    assert not calls
    assert llm.json("x") == {"ok": True} and calls


# ------------------------------------------------------------------ narrate
class FakeLLM:
    def __init__(self, answer):
        self.answer = answer
        self.kw = None

    def json(self, prompt, system="", images=None, temperature=0.4, quality=False):
        self.kw = quality
        return self.answer


J = {"feature_name": "Gator in store", "what_happens": "An alligator walks inside a store", "visible_evidence": ["gator", "store door"],
     "category": "funny_moment"}
BEATS = [{"at": 0.5, "text": "Wait, GTA 6 has gators shopping?"}, {"at": 4.0, "text": "Look, it just walks in."},
         {"at": 8.0, "text": "Nobody stops it at the door."}, {"at": 12.0, "text": "Would you shop here?"}]
CFG_RES = {"script": " ".join(b["text"] for b in BEATS), "beats": BEATS}


def test_punch_up_uses_new_text_but_keeps_timing(cfg0):
    new = [{"at": 9.9, "text": "Okay, GTA 6 has gators shopping? No way!"}, {"at": 1, "text": "It just strolls right in."},
           {"at": 2, "text": "Nobody stops it at the door, honestly."}, {"at": 3, "text": "Would you shop here?"}]
    llm = FakeLLM({"beats": new})
    out = narrate.punch_up(llm, cfg0, J, dict(CFG_RES), None, 16.0)
    assert llm.kw is True                           # asked for a quality model
    assert [b["at"] for b in out["beats"]] == [0.5, 4.0, 8.0, 12.0]
    assert "No way" in out["script"] and out.get("punched")


def test_punch_up_keeps_original_when_invalid(cfg0):
    bad_count = {"beats": [{"at": 0, "text": "only one"}]}
    assert narrate.punch_up(FakeLLM(bad_count), cfg0, J, dict(CFG_RES), None, 16.0) == CFG_RES
    emoji = {"beats": [dict(b, text=b["text"] + " 😂") for b in BEATS]}
    assert narrate.punch_up(FakeLLM(emoji), cfg0, J, dict(CFG_RES), None, 16.0) == CFG_RES
    invented = {"beats": [dict(b, text=b["text"] + " It cost 500 dollars.") for b in BEATS]}
    assert narrate.punch_up(FakeLLM(invented), cfg0, J, dict(CFG_RES), None, 16.0) == CFG_RES

    class Boom:
        def json(self, *a, **k):
            raise RuntimeError("quota")
    assert narrate.punch_up(Boom(), cfg0, J, dict(CFG_RES), None, 16.0) == CFG_RES
    cfg0.data["narration"]["punch_up"] = False
    assert narrate.punch_up(FakeLLM(emoji), cfg0, J, dict(CFG_RES), None, 16.0) == CFG_RES


def test_style_pick_follows_weights_but_still_explores():
    import random
    narrate.STYLE_WEIGHTS = {"hook": {0: 1.6, 1: 0.6}}
    try:
        rng = random.Random(1)
        seen = []
        for _ in range(200):
            narrate._RECENT["hook"].clear()
            narrate._pick("hook", narrate.HOOK_STYLES, rng)
            seen.append(narrate._LAST_PICK["hook"])
        assert seen.count(0) > seen.count(1) and len(set(seen)) > 3
    finally:
        narrate.STYLE_WEIGHTS = {}


def test_styles_weights_need_enough_data(cfg0, st):
    assert styles.weights(st, cfg0) == {"hook": {}, "title": {}} and styles.category_boost(st, cfg0) == {}
    rows = [{"rate": 10.0, "category": "funny_moment", "hook_style": 0, "title_style": 1}] * 6 + \
           [{"rate": 2.0, "category": "feature_shown", "hook_style": 1, "title_style": 1}] * 6
    m = styles._multipliers(rows, lambda r: r["category"], 3)
    assert m["funny_moment"] > 1 > m["feature_shown"]


# ------------------------------------------------------------------ judge
def _jpg(arr):
    import cv2
    return cv2.imencode(".jpg", arr)[1].tobytes()


def test_motion_level_static_vs_moving():
    flat = [_jpg(np.full((90, 160, 3), 80, np.uint8)) for _ in range(4)]
    rng = np.random.default_rng(0)
    moving = [_jpg(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8)) for _ in range(4)]
    assert judge.motion_level(flat) < 1.0 < judge.motion_level(moving)
    assert judge.motion_level([b"broken", b"x"]) is None


def test_static_window_is_skipped_without_ai(cfg0, tmp_path, monkeypatch):
    class NoAI:
        def json(self, *a, **k):
            raise AssertionError("AI must not be called for a frozen picture")
    flat = [_jpg(np.full((90, 160, 3), 80, np.uint8)) for _ in range(4)]
    monkeypatch.setattr(judge, "grab_keyframes", lambda *a, **k: (flat, [1, 2, 3, 4]))
    j = judge.judge_window(NoAI(), cfg0, "v.mp4", 0, 30, "", 1.0, tmp_path)
    assert j["accept"] is False and "frozen" in j["decision_reason"]


def test_same_moment_detection():
    a = {"frame_hash": 0b1010101010, "feature_name": "alligator in store"}
    b = {"frame_hash": 0b1010101010 ^ 0b11, "feature_name": "something else"}
    c = {"frame_hash": (1 << 60) - 1, "feature_name": "car crash"}
    assert judge.same_moment(a, b) and not judge.same_moment(a, c)
    assert judge.same_moment(a, c, (0, 20), (5, 25)) and not judge.same_moment(a, c, (0, 20), (40, 60))


def test_dhash_and_hamming():
    img = np.tile(np.arange(160, dtype=np.uint8), (90, 1))
    h = judge.dhash(_jpg(np.dstack([img] * 3)))
    assert isinstance(h, int) and judge.hamming(h, h) == 0 and judge.dhash(b"junk") is None


def test_clip_starts_close_to_the_action():
    raw = {"feature_shown": True, "key_moment_offset": 20, "start_offset": 0, "end_offset": 30}
    j = judge.normalize(raw, 30.0, 15.0, 40.0)
    assert j["start_offset"] >= 14.9 and j["end_offset"] - j["start_offset"] >= 14.9


def test_dedupe_uses_category_boost():
    items = [(1, {"feature_name": "a b", "interest": 60, "confidence": 60, "category": "feature_shown"}),
             (2, {"feature_name": "c d", "interest": 55, "confidence": 60, "category": "funny_moment"})]
    assert judge.dedupe(items, 1) == [1]
    assert judge.dedupe(items, 1, {"funny_moment": 1.5}) == [2]


# ------------------------------------------------------------------ captions
def test_caption_keywords_get_their_own_colour(cfg0):
    from featureclip import captions
    assert captions.is_key("CRASH!") and captions.is_key("500") and captions.is_key("boom") and not captions.is_key("the")
    ass = captions.build_ass([(0, .4, "This"), (.4, .8, "CRASH"), (.8, 1.2, "is"), (1.2, 1.6, "wild")], "", cfg0)
    assert "&H2D5BFF&" in ass and "\\t(0,90" in ass


# ------------------------------------------------------------------ extras
def test_humanize_error_known_and_unknown():
    assert "quota" in extras.humanize_error("gemini-3.5-flash: 429 daily quota").lower()
    assert "busy" in extras.humanize_error("_Transient: HTTP 503").lower()
    assert extras.humanize_error("something weird") == ""


def test_selftest_runs_and_flags_missing_key(cfg0, st, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    res = extras.selftest(cfg0, st)
    names = {c["name"]: c for c in res}
    assert names["Program files"]["level"] == "ok"
    assert names["Gemini API key"]["level"] == "bad" and names["Title rules"]["level"] == "ok"


# ------------------------------------------------------------------ web
@pytest.fixture()
def client(tmp_path):
    from fastapi.testclient import TestClient
    from featureclip.web.app import create_app
    return TestClient(create_app(tmp_path))


def test_selftest_startup_and_quota_endpoints(client):
    r = client.get("/api/selftest").json()
    assert "checks" in r and isinstance(r["ok"], bool)
    assert "bad" in client.get("/api/startup-check").json()
    live = client.get("/api/assistant/live").json()
    q = live["quota"]
    assert q["total"] >= 8 and q["reset_text"].endswith("PKT") and q["models"][0]["limit"] == 500
    u = client.get("/api/usage").json()
    assert u["reset_text"].endswith("PKT") and "used_today" in u["models"][0]


def test_revoice_validation(client, tmp_path):
    assert client.post("/api/clips/99/revoice", json={"script": "hello there my friend"}).status_code == 404
    c = load_config(root=tmp_path)
    s = Store(c.path("paths.db"))
    vid = s.add_video("src", "/x.mp4", {"duration": 100.0, "width": 1920, "height": 1080})
    cid = s.add_clip(vid, 0.0, 30.0, 1.0)
    s.update_clip(cid, status="ready", judgement={"feature_name": "x", "what_happens": "y", "visible_evidence": [],
                                                   "start_offset": 0, "end_offset": 20})
    assert client.post(f"/api/clips/{cid}/revoice", json={"script": "too short"}).status_code == 400
    long = " ".join(["word"] * 200)
    assert client.post(f"/api/clips/{cid}/revoice", json={"script": long}).status_code == 400
    assert client.post(f"/api/clips/{cid}/revoice", json={"script": "Check this out #fyp and like and subscribe now"}).status_code == 400
    ok = client.post(f"/api/clips/{cid}/revoice", json={"script": "Wait, this alligator just walked into a shop. Would you serve him?"})
    assert ok.status_code == 200 and "id" in ok.json()
    cl = s.clip(cid)
    assert cl["script"].startswith("Wait, this alligator") and cl["status"] == "judged"


def test_slow_model_goes_last_and_timeout_moves_on(cfg0, st, monkeypatch):
    import requests
    from featureclip.llm import LLM
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    llm = LLM(cfg0, st)
    calls = []

    def fake(model, prompt, system, images, temperature):
        calls.append(model)
        if len(calls) == 1:
            raise requests.Timeout("slow")
        return '{"ok": 1}'
    llm._call = fake
    assert llm.json("x") == {"ok": 1}
    assert len(calls) == 2 and calls[0] != calls[1]            # timeout -> next model at once, no second try
    assert llm.ordered_models()[-1] == calls[0]                # the slow one is demoted


def _fake_published(st, n, now):
    """n published clips: even ids are 'funny' with many views + high retention, odd ids 'epic' with few."""
    vid = st.add_video("x", "/tmp/x.mp4", {"duration": 60, "width": 1280, "height": 720, "fps": 30})
    for i in range(n):
        cid = st.add_clip(vid, 0, 20, 1.0)
        funny = i % 2 == 0
        j = {"category": "funny_moment" if funny else "epic_stunt", "start_offset": 0, "end_offset": 22 if funny else 38,
             "styles": {"hook": 1 if funny else 2, "title": 0}}
        st.update_clip(cid, status="uploaded", youtube_id=f"y{i}", scheduled_at=now - 86400 * 3, judgement=json.dumps(j),
                       title=("Wait, a gator shops?" if funny else "Impossible jump in GTA 6 landed"))
        st.add_stats(cid, 3000 if funny else 200, 150 if funny else 5, 20 if funny else 1, 85 if funny else 40, 20)


def test_learn_needs_data_then_finds_winners(cfg0, st):
    from featureclip import learn
    now = time.time()
    _fake_published(st, 3, now)
    rep = learn.refresh(st, cfg0)
    assert rep["clips"] == 3 and rep["playbook"] == "" and learn.playbook_text(st, cfg0) == ""
    _fake_published(st, 9, now)
    rep = learn.refresh(st, cfg0)
    assert rep["clips"] == 12 and rep["analytics"] is True
    top = {(f["feature"], f["value"]): f["lift"] for f in rep["features"]}
    assert top[("category", "funny_moment")] > 5 > -5 > top[("category", "epic_stunt")]
    txt = learn.playbook_text(st, cfg0)
    assert "funny_moment" in txt and "gator" in txt and len(txt) <= 700
    cfg0.data["learn"]["use_playbook"] = False
    assert learn.playbook_text(st, cfg0) == ""


def test_learn_endpoint_and_prompt_injection(client):
    r = client.get("/api/learn")
    assert r.status_code == 200 and r.json()["clips"] == 0
    assert client.post("/api/learn/refresh").status_code == 200
    from featureclip import narrate, metadata
    assert hasattr(narrate, "PLAYBOOK") and hasattr(metadata, "PLAYBOOK")


def test_live_link_downloaded_as_video_is_stopped_and_switched(cfg0, tmp_path, monkeypatch):
    """yt-dlp starts recording a running live stream as one endless file: ingest must notice and stop it."""
    from featureclip import ingest as ing
    from featureclip.sources import Source

    class FakeProc:
        def __init__(self, *a, **k):
            self.stdout = iter(["[youtube] x: Downloading webpage\n",
                                "[https @ 0x1] Opening 'https://rr1.googlevideo.com/videoplayback/source/yt_live_broadcast/live/1/seg.ts' for reading\n"])
        def wait(self, *a, **k): return 0
        def kill(self): pass
        pid = 1
    killed = []
    monkeypatch.setattr(ing.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ing, "_kill_tree", lambda p: killed.append(1))
    src = Source(id="s1", url="https://www.youtube.com/watch?v=abc", streamer="")
    with pytest.raises(ing.LiveStreamError):
        ing.ingest(src, tmp_path, cfg0)
    assert killed


def test_reused_file_name_refreshes_old_video_row(st):
    """Live pieces are named seg_r00_00000 on every recording: a new 10 min file must not inherit a 30 s record."""
    vid = st.add_video("s", "/x/seg_r00_00000.mp4", {"duration": 34.0, "width": 1280, "height": 720})
    old = st.add_clip(vid, 10, 40, 1.0)
    done = st.add_clip(vid, 20, 50, 1.0)
    st.update_clip(done, status="uploaded")
    assert st.add_video("s", "/x/seg_r00_00000.mp4", {"duration": 600.0, "width": 1280, "height": 720}) == vid
    assert st.refresh_video(vid, {"duration": 600.0, "width": 1280, "height": 720}) is True
    assert st.video(vid)["duration"] == 600.0
    assert [c["id"] for c in st.clips(vid)] == [done]            # the stale candidate is gone
    assert st.clip(done)["status"] == "uploaded"                   # finished work is never touched
    assert st.refresh_video(vid, {"duration": 600.4, "width": 1280, "height": 720}) is False


def test_live_helpers_pass_cookies(cfg0):
    from featureclip import live
    cfg0.data.setdefault("ingest", {})["cookies_file"] = "C:/c.txt"
    assert live.ytdlp_extra(cfg0)[-2:] == ["--cookies", "C:/c.txt"]
    assert "--remote-components" in live.ytdlp_extra(None)
