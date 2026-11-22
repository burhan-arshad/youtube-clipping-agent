"""The streamer channel is gone: legacy clean-up keeps voice-over clips, the app imports and the UI has no streamer mode."""
import json
from pathlib import Path

from featureclip import legacy
from featureclip.config import load_config
from featureclip.store import Store


def _setup(tmp_path):
    cfg = load_config(tmp_path / "config.yaml", root=tmp_path)
    st = Store(tmp_path / "t.db")
    out = tmp_path / "out"
    out.mkdir()
    ids = {}
    for name, sid in (("vo", "web-aaaa1111"), ("st", "web-aaaa1111-streamer")):
        v = st.add_video(sid, str(tmp_path / f"{name}.mp4"), {"duration": 60})
        c = st.add_clip(v, 1, 20, 0.5)
        f = out / f"{name}.mp4"
        f.write_bytes(b"x")
        (out / f"{name}.unsynced.mp4").write_bytes(b"x")
        st.update_clip(c, status="ready", path=str(f))
        ids[name] = c
    st.add_agent_source("https://a.com/1", "video", "[autopilot streamer] x", "D", "", {"mode": "streamer", "clips": 5})
    st.add_agent_source("https://a.com/1", "video", "[autopilot] x", "D", "", {"clips": 5})
    st.kv_set("playbook:streamer", "{}")
    return cfg, st, ids, out


def test_dry_run_changes_nothing(tmp_path):
    cfg, st, ids, out = _setup(tmp_path)
    r = legacy.remove(cfg, st, apply=False)
    assert r["clips"] == 1 and r["queue"] == 1 and not r["applied"]
    assert len(st.clips()) == 2 and (out / "st.mp4").exists()


def test_remove_keeps_voiceover(tmp_path):
    cfg, st, ids, out = _setup(tmp_path)
    r = legacy.remove(cfg, st, apply=True)
    assert r["applied"]
    assert [c["id"] for c in st.clips()] == [ids["vo"]]
    assert (out / "vo.mp4").exists() and (out / "vo.unsynced.mp4").exists()
    assert not (out / "st.mp4").exists() and not (out / "st.unsynced.mp4").exists()
    assert len(st.agent_sources()) == 1 and st.kv_get("playbook:streamer") == ""
    assert legacy.remove(cfg, st, apply=True)["clips"] == 0          # second run: nothing left
    assert legacy.auto(cfg, st) is not None and legacy.auto(cfg, st) is None


def test_ui_has_no_streamer_mode(tmp_path):
    html = (Path(__file__).parent.parent / "featureclip/web/static/index.html").read_text(encoding="utf-8")
    for w in ("mkMode", "asMode", "fixSync", "S.channels", "Streamer clips"):
        assert w not in html
