import threading
from pathlib import Path

import pytest

from featureclip import ffutil, memory
from featureclip.cli import main as cli_main
from featureclip.config import load_config
from featureclip.store import Store


def test_stop_kills_running_command():
    ev = threading.Event()
    ffutil.set_stop(ev)
    try:
        threading.Timer(0.6, ev.set).start()
        with pytest.raises(ffutil.Stopped):
            ffutil.run(["python", "-c", "import time; time.sleep(30)"])
        with pytest.raises(ffutil.Stopped):                 # once stopped, the next command does not even start
            ffutil.run(["python", "-c", "pass"])
    finally:
        ffutil.set_stop(None)
    assert ffutil.run(["python", "-c", "print(1)"]).stdout.strip() == b"1"


def _env(tmp_path):
    cfg = load_config(tmp_path / "config.yaml", root=tmp_path)
    st = Store(tmp_path / "t.db")
    out = cfg.path("paths.output")
    out.mkdir(parents=True, exist_ok=True)
    vid = st.add_video("s", "x.mp4", {"duration": 10})
    ids = {}
    for name, status in (("up", "uploaded"), ("wait", "approved")):
        c = st.add_clip(vid, 0, 10, 1.0)
        f = out / f"{name}.mp4"
        f.write_bytes(b"x" * 1000)
        (out / f"{name}.jpg").write_bytes(b"y" * 10)
        st.update_clip(c, status=status, path=str(f), title=name)
        ids[name] = c
    return cfg, st, ids, out


def test_memory_delete_clips_keeps_data_and_waiting_clips(tmp_path):
    cfg, st, ids, out = _env(tmp_path)
    u = memory.usage(cfg, st)
    assert u["clip_rows"] == 2 and u["clip_files"] == 2 and u["waiting"] == 1
    r = memory.delete(cfg, st, "clips", only_done=True)
    assert r["files"] == 2 and not (out / "up.mp4").exists() and (out / "wait.mp4").exists()
    assert st.clip(ids["up"])["title"] == "up" and st.clip(ids["up"])["path"] == ""       # data stays
    assert st.clip(ids["wait"])["status"] == "approved"
    memory.delete(cfg, st, "clips", only_done=False)
    assert not list(out.iterdir()) and st.clip(ids["wait"])["status"] == "rejected" and st.clip(ids["wait"])["title"] == "wait"


def test_memory_delete_work(tmp_path):
    cfg, st, ids, out = _env(tmp_path)
    w = cfg.path("paths.workdir")
    (w / "downloads" / "a").mkdir(parents=True)
    (w / "downloads" / "a" / "video.mp4").write_bytes(b"z" * 500)
    assert memory.delete(cfg, st, "work")["files"] == 1 and not any(w.iterdir())
    assert len(st.videos()) == 1


def test_tidy_lists_then_deletes(tmp_path, capsys):
    (tmp_path / "start_ui.bat").write_text("x")
    (tmp_path / "layout.png").write_bytes(b"x")
    (tmp_path / "pkg" / "__pycache__").mkdir(parents=True)
    assert cli_main(["--root", str(tmp_path), "tidy"]) == 0
    assert (tmp_path / "layout.png").exists() and "would delete" in capsys.readouterr().out
    cli_main(["--root", str(tmp_path), "tidy", "--yes"])
    assert not (tmp_path / "layout.png").exists() and not (tmp_path / "start_ui.bat").exists() and not (tmp_path / "pkg" / "__pycache__").exists()


def test_settings_save_keeps_backup_copy(tmp_path):
    from featureclip.web import settings_store as ss
    (tmp_path / "config.yaml").write_text("game:\n  name: A\n", encoding="utf-8")
    ss.save(tmp_path, {"game.name": "B"})
    assert "B" in (tmp_path / "config.yaml").read_text() and "A" in (tmp_path / "config.yaml.bak").read_text()


def test_api_memory_stop_all_and_channel_upload(tmp_path):
    from fastapi.testclient import TestClient
    from featureclip.web.app import create_app
    c = TestClient(create_app(tmp_path))
    assert "free_gb" in c.get("/api/memory").json()
    assert c.post("/api/memory/delete", json={"what": "nope"}).status_code == 400
    assert c.post("/api/stop-all").json()["stopped"] == 0
    r = c.post("/api/jobs", json={"kind": "upload", "channel": "streamer", "public_now": True, "dry_run": True})
    assert r.status_code == 200
