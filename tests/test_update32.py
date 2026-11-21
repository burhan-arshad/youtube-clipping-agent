import json, time
from pathlib import Path
from featureclip import autopilot
from featureclip.config import load_config
from featureclip.store import Store
from featureclip.web.jobs import source_for


class N:
    def __init__(self): self.msgs = []
    def send(self, level, title, body=""): self.msgs.append((level, title, body))


def mk(tmp_path, links, **ap):
    (tmp_path / "links.txt").write_text(links, encoding="utf-8")
    cfg = load_config(None, root=tmp_path) if False else load_config(tmp_path / "config.yaml", root=tmp_path)
    cfg.data["autopilot"].update(enabled=True, run_at="00:00", **ap)
    return cfg, Store(tmp_path / "t.db")


def test_parse_links():
    r = autopilot.parse_links("# c\n\nhttps://a.com/1 | Duck | top-right\nnotalink\nhttps://a.com/1\nhttps://b.com/2\n")
    assert [x["url"] for x in r] == ["https://a.com/1", "https://b.com/2"]
    assert r[0]["streamer"] == "Duck" and r[0]["facecam"] == "top-right"


def test_source_ids_differ_but_download_shared():
    a, b = source_for("https://a.com/1", mode="voiceover"), source_for("https://a.com/1", mode="streamer")
    assert a.id != b.id and b.id.removesuffix("-streamer") == a.id


def test_daily_start_finish_and_reminders(tmp_path):
    cfg, st = mk(tmp_path, "https://a.com/1 | Duck\nhttps://a.com/2 | Duck\n", warn_links_left=1)
    n = N()
    autopilot.tick(cfg, st, n, is_live_fn=lambda u, c: False)
    q = st.agent_sources()
    assert len(q) == 2 and {json.loads(s["options"])["mode"] for s in q} == {"voiceover", "streamer"}
    autopilot.tick(cfg, st, n, is_live_fn=lambda u, c: False)          # same day: no second link
    assert len(st.agent_sources()) == 2
    for s in q:
        st.update_agent_source(s["id"], status="done")
    autopilot.tick(cfg, st, n, is_live_fn=lambda u, c: False)
    d = list(autopilot.load_state(st)["links"].values())[0]
    assert d["status"] == "pending" and d["attempts"] == 1             # 0 clips -> retry later
    assert any("try again" in m[1] for m in n.msgs)


def test_failed_twice_and_live_skipped(tmp_path):
    cfg, st = mk(tmp_path, "https://a.com/live\n")
    n = N()
    autopilot.tick(cfg, st, n, is_live_fn=lambda u, c: True)
    assert list(autopilot.load_state(st)["links"].values())[0]["status"] == "skipped"
    assert any("live" in m[1] for m in n.msgs) and any("no links left" in m[1].lower() for m in n.msgs)


def test_add_links_appends_once(tmp_path):
    cfg, st = mk(tmp_path, "https://a.com/1\n")
    assert autopilot.add_links(cfg, "https://a.com/1\nhttps://a.com/2 | X") == 1
    assert len(autopilot.read_links(cfg)) == 2


def test_success_summary_and_next_day_not_before_run_at(tmp_path):
    cfg, st = mk(tmp_path, "https://a.com/1 | Duck\n")
    n = N()
    autopilot.tick(cfg, st, n, is_live_fn=lambda u, c: False)
    for s in st.agent_sources():
        st.update_agent_source(s["id"], status="done")
    sid = source_for("https://a.com/1", mode="voiceover").id
    vid = st.add_video(sid, "x.mp4", {"duration": 100.0})
    for _ in range(3):
        c = st.add_clip(vid, 0.0, 30.0, 1.0)
        st.update_clip(c, status="ready")
    autopilot.tick(cfg, st, n, is_live_fn=lambda u, c: False)
    assert list(autopilot.load_state(st)["links"].values())[0]["status"] == "done"
    assert any(m[1] == "Autopilot: link finished" and "Fewer" in m[2] for m in n.msgs)
    cfg.data["autopilot"]["run_at"] = "23:59"
    assert not autopilot.due_today(cfg, {"started": {}}, now=time.mktime(time.strptime("2026-01-01 10:00", "%Y-%m-%d %H:%M")))
