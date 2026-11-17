import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from featureclip import agent, metadata, schedule
from featureclip.config import load_config
from featureclip.store import Store


def test_slots_and_next_free_times():
    slots = schedule.parse_slots(schedule.DEFAULT_SLOTS)
    now = datetime(2026, 11, 16, 15, 0, tzinfo=timezone.utc)          # 20:00 PKT
    first = schedule.next_slots(now, 4, slots)
    assert [schedule.fmt_pkt(t) for t in first[:3]] == ["Mon 16 Nov, 21:00 PKT", "Tue 17 Nov, 01:00 PKT", "Tue 17 Nov, 05:00 PKT"]
    taken = schedule.next_slots(now, 2, slots)
    again = schedule.next_slots(now, 2, slots, taken)
    assert again[0] not in taken and again[0] > taken[-1]
    assert schedule.iso_utc(first[0]) == "2026-11-16T16:00:00.000Z"
    assert schedule.parse_slots("25:00, bad, 7:30") == [(7, 30)]


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.prompts = list(replies), []

    def available(self):
        return True

    def json(self, prompt, **kw):
        self.prompts.append(prompt)
        return self.replies.pop(0)


J = {"feature_name": "Police arrest", "what_happens": "Cops surround the player and arrest him.",
     "visible_evidence": ["officers aim guns", "player raises hands"]}


def test_metadata_generate_validates_and_falls_back(tmp_path):
    cfg = load_config(None, root=tmp_path)
    good = {"title": "GTA 6 cops can arrest you mid-chase", "tags": ["gta 6", "police"],
            "description": "Watch GTA 6 cops surround the player.\n\nThe arrest looks different from GTA 5.\n\nDid you notice this?\n\n#GTA6 #Shorts"}
    m = metadata.generate(FakeLLM([good]), cfg, J, "script", "", "Ducky", "https://yt/@d")
    assert m["title"].startswith("GTA 6") and "Footage from Ducky" in m["description"] and "http" not in m["description"]
    assert "AI-generated" in m["description"] and m["description"].rstrip().endswith("#GTA6 #Shorts")
    bad = {"title": "cops arrest", "description": "x", "tags": []}
    m2 = metadata.generate(FakeLLM([bad, bad, bad]), cfg, J, "script", "Cops arrest you", "", "")
    assert "GTA 6" in m2["tags"][0] or m2["tags"]                    # fallback builder, still usable
    assert m2["description"]


def make_clip(tmp_path, test_video, **over):
    cfg = load_config(None, root=tmp_path)
    st = Store(cfg.path("paths.db"))
    vid = st.add_video("s", str(test_video), {"duration": 70, "width": 640, "height": 360})
    cid = st.add_clip(vid, 10, 30, 1.0)
    st.update_clip(cid, status="ready", title="GTA 6 cops can arrest you mid-chase", script="x",
                   description="Watch GTA 6 cops surround the player.\n\nMore text here to be long enough for the rule.\n\nDid you notice?\n\n#GTA6 #Shorts",
                   judgement=J, path=str(test_video), **over)
    return cfg, st, st.clip(cid)


def test_review_flags_non_vertical_and_ai_issues(tmp_path, test_video):
    cfg, st, c = make_clip(tmp_path, test_video)
    llm = FakeLLM([{"score": 90, "issues": [{"type": "video", "text": "streamer face visible"}]}])
    rv = agent.review_clip(llm, cfg, c, tmp_path / "rv")
    assert not rv["ok"]
    texts = " | ".join(i["text"] for i in rv["issues"])
    assert "streamer face visible" in texts and "vertical 9:16" in texts and "thumbnail" in texts


def test_review_passes_clean_clip(tmp_path):
    import subprocess
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=540x960:rate=15:d=12",
                    "-f", "lavfi", "-i", "sine=d=12", "-shortest", str(vid)], check=True)
    cfg, st, c = make_clip(tmp_path, vid)
    from featureclip import metadata as md
    fr = md.pick_frame(str(vid), 0, 12, None, tmp_path / "f")
    th = md.make_thumbnail(fr, tmp_path / "t.jpg", "Cops arrest", cfg)
    st.update_clip(c["id"], thumb=str(th))
    rv = agent.review_clip(FakeLLM([{"score": 88, "issues": []}]), cfg, st.clip(c["id"]), tmp_path / "rv")
    assert rv["ok"] and rv["score"] == 88 and all(ch["ok"] for ch in rv["checks"])


class FakeJobs:
    def __init__(self):
        self.submitted, self._running = [], None

    def running(self):
        return self._running

    def get(self, jid):
        return None

    def submit(self, kind, title, **args):
        self.submitted.append((kind, args))
        from types import SimpleNamespace
        return SimpleNamespace(id=len(self.submitted), status="running", error="")


def test_assistant_starts_queued_source_and_warns_when_stock_is_low(tmp_path):
    cfg = load_config(None, root=tmp_path)
    (tmp_path / "config.yaml").write_text("assistant:\n  enabled: true\n  buffer_days: 2\n")
    st = Store(cfg.path("paths.db"))
    st.add_agent_source("https://youtube.com/live/abc", "live", streamer="Ducky", options={"minutes": 5})
    jobs = FakeJobs()
    a = agent.Assistant(tmp_path, None, jobs)
    a.tick()
    assert jobs.submitted and jobs.submitted[0][0] == "watch" and jobs.submitted[0][1]["streamer"] == "Ducky"
    assert st.agent_sources()[0]["status"] == "running"
    # job vanished -> source is closed; no link left + no clips -> "clips are finished" notification
    a.tick()
    titles = [n["title"] for n in Store(cfg.path("paths.db")).notifications()]
    assert "Clips are finished" in titles


class StubPipe:
    def __init__(self, tmp):
        self.tmp, self.sources = tmp, {}

    def cdir(self, v, c):
        return self.tmp / f"c{c}"


def _clean_clip(tmp_path):
    import subprocess
    vid = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=540x960:rate=15:d=12",
                    "-f", "lavfi", "-i", "sine=d=12", "-shortest", str(vid)], check=True)
    cfg, st, c = make_clip(tmp_path, vid)
    fr = metadata.pick_frame(str(vid), 0, 12, None, tmp_path / "f")
    st.update_clip(c["id"], thumb=str(metadata.make_thumbnail(fr, tmp_path / "t.jpg", "Cops", cfg)))
    return cfg, st, c["id"]


def test_auto_mode_approves_good_and_rejects_policy_problems(tmp_path):
    from featureclip.notify import Notifier
    cfg, st, cid = _clean_clip(tmp_path)
    a = agent.Assistant(tmp_path, None, FakeJobs())
    note = Notifier(cfg, st)
    a._review(cfg, st, FakeLLM([{"score": 90, "issues": []}]), StubPipe(tmp_path), note)
    assert st.clip(cid)["status"] == "approved" and json.loads(st.clip(cid)["review"])["ok"]
    # second clip with a policy issue is rejected, owner is told
    st.update_clip(cid, status="ready", review=None)
    a._review(cfg, st, FakeLLM([{"score": 90, "issues": [{"type": "policy", "text": "graphic gore"}]}]), StubPipe(tmp_path), note)
    c = st.clip(cid)
    assert c["status"] == "rejected" and "graphic gore" in c["error"]
    assert any("rejected" in n["title"] for n in st.notifications())


def test_ask_mode_never_approves_by_itself(tmp_path):
    from featureclip.notify import Notifier
    cfg, st, cid = _clean_clip(tmp_path)
    (tmp_path / "config.yaml").write_text("assistant:\n  mode: ask\n")
    cfg = load_config(None, root=tmp_path)
    agent.Assistant(tmp_path, None, FakeJobs())._review(cfg, st, FakeLLM([{"score": 95, "issues": []}]), StubPipe(tmp_path), Notifier(cfg, st))
    assert st.clip(cid)["status"] == "ready" and json.loads(st.clip(cid)["review"])["ok"]


def test_ai_unavailable_keeps_clip_for_next_pass(tmp_path):
    from featureclip.llm import LLMError
    from featureclip.notify import Notifier
    cfg, st, cid = _clean_clip(tmp_path)

    class Down(FakeLLM):
        def json(self, *a, **k):
            raise LLMError("quota")
    agent.Assistant(tmp_path, None, FakeJobs())._review(cfg, st, Down([]), StubPipe(tmp_path), Notifier(cfg, st))
    c = st.clip(cid)
    assert c["status"] == "ready" and not c["review"]


def test_minor_remarks_never_block_a_good_clip(tmp_path):
    from featureclip.notify import Notifier
    cfg, st, cid = _clean_clip(tmp_path)
    a = agent.Assistant(tmp_path, None, FakeJobs())
    issues = [{"type": "video", "severity": "minor", "text": "caption slightly cut"},
              {"type": "thumbnail", "severity": "blocker", "text": "weak thumbnail"}]      # thumbnail can never be a blocker
    a._review(cfg, st, FakeLLM([{"score": 88, "issues": issues}]), StubPipe(tmp_path), Notifier(cfg, st))
    c = st.clip(cid)
    rv = json.loads(c["review"])
    assert c["status"] == "approved" and rv["ok"] and rv["blockers"] == 0 and len(rv["issues"]) == 2


def test_wording_problems_are_repaired_in_code_not_rejected(tmp_path):
    from featureclip.notify import Notifier
    cfg, st, cid = _clean_clip(tmp_path)
    st.update_clip(cid, title="Cars #crash hard in the city #Shorts", description="short")
    a = agent.Assistant(tmp_path, None, FakeJobs())
    a._review(cfg, st, FakeLLM([{"score": 90, "issues": []}]), StubPipe(tmp_path), Notifier(cfg, st))
    c = st.clip(cid)
    assert c["status"] == "approved" and "#crash" not in c["title"] and c["title"].endswith("#Shorts")
    assert "GTA 6" in c["title"] and len(c["description"]) >= 80 and "http" not in c["description"]


def test_serious_problem_after_rebuilds_waits_for_the_owner(tmp_path):
    from featureclip.notify import Notifier
    cfg, st, cid = _clean_clip(tmp_path)
    st.update_clip(cid, rebuilds=2)                         # rebuild budget already used
    a = agent.Assistant(tmp_path, None, FakeJobs())
    bad = {"score": 70, "issues": [{"type": "script", "severity": "blocker", "text": "voice talks about a chase not shown"}]}
    a._review(cfg, st, FakeLLM([bad]), StubPipe(tmp_path), Notifier(cfg, st))
    c = st.clip(cid)
    assert c["status"] == "ready" and json.loads(c["review"])["needs_you"] is True      # kept, not rejected
    assert any("needs your decision" in n["title"] for n in st.notifications(10))
