"""Long online videos are processed part by part and each part is deleted afterwards."""
from pathlib import Path
from types import SimpleNamespace

from featureclip import ingest as ing
from featureclip.config import load_config
from featureclip.pipeline import Pipeline
from featureclip.sources import Source
from featureclip.store import Store


def mk(tmp_path, total, clips=2):
    cfg = load_config(tmp_path / "config.yaml", root=tmp_path)
    cfg.data["ingest"] = {"part_minutes": 10, "max_minutes": 0}
    st = Store(tmp_path / "t.db")
    src = Source(id="web-x", streamer="D", url="https://y.com/v", extra={"clips": clips, "since": 0})
    p = Pipeline(cfg, st, {src.id: src})
    return cfg, st, p, src


def test_parts_downloaded_one_at_a_time_and_deleted(tmp_path, monkeypatch):
    cfg, st, p, src = mk(tmp_path, 3600)
    calls = []
    monkeypatch.setattr(ing, "remote_duration", lambda url, cfg=None: 3600.0)

    def fake_part(src, work, cfg, a, b, k, stop=None):
        d = Path(work) / "downloads" / src.id / f"part_{k:03d}"
        d.mkdir(parents=True, exist_ok=True)
        f = d / "video.mp4"
        f.write_bytes(b"x")
        calls.append((a, b, k))
        return f
    monkeypatch.setattr(ing, "ingest_part", fake_part)
    made = {"n": 0}

    def fake_scan(sid, path=None):
        return st.add_video(sid, str(path), {"duration": 600})
    p.scan = fake_scan
    p.judge = lambda vid: None

    def fake_build(vid):
        made["n"] += 1
        c = st.add_clip(vid, 40, 70, 0.5)
        st.update_clip(c, status="ready")
    p.build = fake_build
    p.run_parts("web-x")
    assert calls == [(0, 600, 0), (600, 1200, 1)]                    # stops after 2 clips are made
    assert not list((Path(p.work) / "downloads" / "web-x").glob("part_*"))   # parts are gone
    assert st.kv_get("parts_done:web-x") == "0"


def test_analytics_export_csv(tmp_path):
    from fastapi.testclient import TestClient
    from featureclip.web.app import create_app
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")
    c = TestClient(create_app(tmp_path, tmp_path / "config.yaml"))
    for kind, head in (("clips", "clip_id,title"), ("features", "feature,value")):
        r = c.get(f"/api/analytics/export?kind={kind}")
        assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
        assert r.text.lstrip("﻿").startswith(head)
