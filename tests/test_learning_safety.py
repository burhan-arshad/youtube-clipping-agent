import json
import os
import time
from pathlib import Path

from featureclip import cleanup, insights, safe, schedule
from featureclip.agent import Assistant
from featureclip.config import load_config
from featureclip.store import Store


def _env(tmp_path):
    cfg = load_config(None, root=tmp_path)
    return cfg, Store(cfg.path("paths.db"))


def _published(st, n, slot, category, views, age_h=48, vid_id=1, path="/x/v.mp4"):
    """n published clips at a PKT slot, each with one stats snapshot."""
    h, m = map(int, slot.split(":"))
    for i in range(n):
        v = st.add_video("s", f"{path}{slot}{category}{i}", {"duration": 70, "width": 640, "height": 360})
        cid = st.add_clip(v, 1, 20, 1.0)
        # a time that falls on `slot` in PKT, `age_h` hours ago (aligned to the slot)
        now = time.time() - age_h * 3600
        day = schedule.datetime.fromtimestamp(now, schedule.PKT).replace(hour=h, minute=m, second=0, microsecond=0)
        st.update_clip(cid, status="uploaded", youtube_id=f"yt{cid}", scheduled_at=day.timestamp(),
                       judgement={"category": category},
                       review={"ok": True, "score": 80 + i, "ts": time.time()})
        st.add_stats(cid, views, views // 10, 1)


def test_score_history_and_categories(tmp_path):
    cfg, st = _env(tmp_path)
    _published(st, 3, "21:00", "vehicle", 600)
    _published(st, 3, "01:00", "weather", 60)
    r = insights.report(st, cfg)
    assert len(r["history"]) == 6 and r["history"][0]["score"] >= 80
    cats = {c["category"]: c for c in r["categories"]}
    assert cats["vehicle"]["avg_views"] > cats["weather"]["avg_views"]
    assert r["categories"][0]["category"] == "vehicle"              # best first
    assert {s["key"] for s in r["slots"]} == {"21:00", "01:00"}


def test_stats_collect_uses_fetcher_and_skips_future(tmp_path):
    cfg, st = _env(tmp_path)
    v = st.add_video("s", "/x/a.mp4", {"duration": 70, "width": 640, "height": 360})
    a, b = st.add_clip(v, 1, 20, 1.0), st.add_clip(v, 30, 50, 1.0)
    st.update_clip(a, status="uploaded", youtube_id="AAA", scheduled_at=time.time() - 7200)
    st.update_clip(b, status="uploaded", youtube_id="BBB", scheduled_at=time.time() + 7200)   # not public yet
    asked = []
    n = insights.collect(st, cfg, fetch=lambda c, ids: asked.append(ids) or {"AAA": {"views": 5, "likes": 1, "comments": 0}})
    assert n == 1 and asked == [["AAA"]]
    assert st.latest_stats()[a]["views"] == 5


def test_slot_suggestion_waits_then_explores_then_exploits(tmp_path):
    cfg, st = _env(tmp_path)
    # no data -> waiting
    s0 = insights.suggest_slots(st, cfg)
    assert not s0["change"] and "Collecting data" in s0["reason"]
    # all three slots tested, 21:00 is weak -> explores an untried pool time in its place
    _published(st, 3, "21:00", "a", 30)
    _published(st, 3, "01:00", "a", 900)
    _published(st, 3, "05:00", "a", 800)
    s1 = insights.suggest_slots(st, cfg)
    assert s1["change"] and s1["drop"] == "21:00" and s1["add"] in cfg.get("schedule.pool")
    assert len(s1["slots"]) == 3 and insights._gap_ok(s1["slots"])
    # a pool time with data that clearly wins is chosen first
    _published(st, 3, "19:00", "a", 3000)
    s2 = insights.suggest_slots(st, cfg)
    assert s2["add"] == "19:00"
    # just changed -> locked for a week
    insights.apply_slots(tmp_path, None, s2["slots"], st)
    assert not insights.suggest_slots(st, cfg)["change"]
    assert "19:00" in (tmp_path / "config.yaml").read_text()


def test_cleanup_deletes_only_safe_things(tmp_path):
    cfg, st = _env(tmp_path)
    work = cfg.path("paths.workdir")
    live = work / "live" / "s"
    live.mkdir(parents=True)
    old_t = time.time() - 30 * 3600

    def mk(p, age=old_t, size=1000):
        p.write_bytes(b"x" * size)
        os.utime(p, (age, age))
        return p
    done = mk(live / "seg_r00_00000.mp4")           # all clips of it are made -> deletable
    pend = mk(live / "seg_r00_00001.mp4")           # still has a 'judged' clip -> keep
    fresh = mk(live / "seg_r00_00002.mp4", age=time.time())                # being written -> keep
    unseen = mk(live / "seg_r00_00003.ts", age=time.time() - 8 * 3600)     # unprocessed but only 8 h old -> keep (<12 h)
    v1 = st.add_video("s", str(done), {"duration": 70, "width": 640, "height": 360})
    v2 = st.add_video("s", str(pend), {"duration": 70, "width": 640, "height": 360})
    c1 = st.add_clip(v1, 1, 20, 1.0)
    st.update_clip(c1, status="uploaded", review={"ok": True, "score": 90})
    c2 = st.add_clip(v2, 1, 20, 1.0)
    st.update_clip(c2, status="judged")
    out = cfg.path("paths.output")
    out.mkdir(parents=True, exist_ok=True)
    outfile = mk(out / "clip1.mp4", age=old_t)
    st.update_clip(c1, path=str(outfile), scheduled_at=time.time() - 10 * 86400)
    keepme = mk(out / "clip2.mp4", age=old_t)
    st.update_clip(c2, path=str(keepme))
    dry = cleanup.run(cfg, st, dry_run=True)
    assert done.exists() and dry["files"] >= 2
    r = cleanup.run(cfg, st)
    assert not done.exists() and not outfile.exists()
    assert pend.exists() and fresh.exists() and unseen.exists() and keepme.exists()
    assert r["freed_mb"] >= 0


def test_safe_mode_blocks_upload_and_pulls_back(tmp_path, monkeypatch):
    from featureclip import pipeline as pl
    cfg, st = _env(tmp_path)
    v = st.add_video("s", "/x/a.mp4", {"duration": 70, "width": 640, "height": 360})
    c = st.add_clip(v, 1, 20, 1.0)
    st.update_clip(c, status="approved", title="T #Shorts", description="D")
    p = pl.Pipeline(cfg, st, {})
    safe.turn_on(st, "test")
    assert p.upload() == 0 and st.clip(c)["status"] == "approved"
    # scheduled video gets pulled back (fake YouTube) and later re-scheduled
    d = st.add_clip(v, 30, 50, 1.0)
    st.update_clip(d, status="uploaded", youtube_id="YT1", scheduled_at=time.time() + 86400)
    calls = []
    r = safe.pull_back(cfg, st, setter=lambda cfg_, vid, at: calls.append((vid, at)))
    assert r["pulled"] == 1 and calls == [("YT1", None)]
    assert safe.info(st)["held"] == [d] and "held" in st.clip(d)["publish_state"]
    r2 = safe.release(cfg, st, setter=lambda cfg_, vid, at: calls.append((vid, at)))
    assert r2["rescheduled"] == 1 and calls[-1][1].endswith("Z") and not safe.is_on(st)
    assert st.clip(d)["scheduled_at"] > time.time()


def test_pull_back_needs_permission_without_setter(tmp_path):
    cfg, st = _env(tmp_path)
    try:
        safe.pull_back(cfg, st)
        assert False
    except RuntimeError as e:
        assert "Settings" in str(e)


def test_recover_requeues_interrupted_links_and_resume_drops_missing_source(tmp_path):
    cfg, st = _env(tmp_path)
    sid = st.add_agent_source("https://youtube.com/watch?v=x", "live")
    st.update_agent_source(sid, status="running", options=json.dumps({"job": 7}))
    a = Assistant(tmp_path, None, jobs=None)
    a.recover()
    s = st.agent_sources()[0]
    assert s["status"] == "queued" and json.loads(s["options"])["restarts"] == 1
    for _ in range(3):
        st.update_agent_source(sid, status="running")
        a.recover()
    assert st.agent_sources()[0]["status"] == "error"               # gave up after 3 restarts
    # unfinished 'judged' clip whose source file was cleaned -> dropped, not retried forever
    v = st.add_video("s", str(tmp_path / "gone.mp4"), {"duration": 70, "width": 640, "height": 360})
    c = st.add_clip(v, 1, 20, 1.0)
    st.update_clip(c, status="judged")

    class Jobs:
        def running(self):
            return None
    a.jobs = Jobs()
    a._resume(cfg, st, pipe=None)
    assert st.clip(c)["status"] == "dropped"
