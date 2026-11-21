"""Watchdog, backup, disk limit, clean sentence bounds, per-channel playbook."""
import json
import os
import sqlite3
import sys
import time
import zipfile
from pathlib import Path

from featureclip import backup, bounds, cleanup, heartbeat, learn, supervisor
from featureclip.store import Store


# ---------------------------------------------------------------- bounds
W = [(10.0, 10.4, "no"), (10.5, 10.9, "way"), (11.0, 11.4, "that"), (11.5, 12.0, "happened."),
     (14.0, 14.3, "bro"), (14.4, 14.9, "look"), (15.0, 15.6, "at"), (15.7, 16.3, "this!"), (20.0, 20.5, "later")]


def test_bounds_moves_start_back_and_end_forward():
    s, e, info = bounds.refine(W, 10.8, 15.2, video_len=100)
    assert abs(s - (10.0 - 0.35)) < 0.01                 # back to the start of "no way that happened."
    assert abs(e - (16.3 + 0.5)) < 0.01                  # forward to the end of "bro look at this!"
    assert info["start_moved"] > 0 and info["end_moved"] > 0


def test_bounds_keeps_pauses_and_limits():
    s, e, _ = bounds.refine(W, 12.5, 13.5, video_len=100)             # both in silence: unchanged
    assert (s, e) == (12.5, 13.5)
    s, e, _ = bounds.refine(W, 10.8, 15.2, video_len=16.5)            # never past the end of the video
    assert e <= 16.5
    s, e, _ = bounds.refine(W, 10.8, 15.2, video_len=100, max_len=5.0)
    assert e - s <= 5.6
    assert bounds.refine([], 1, 5) == (1, 5, {"start_moved": 0.0, "end_moved": 0.0})


# ---------------------------------------------------------------- watchdog
def test_should_restart_only_when_busy_and_silent():
    now = time.time()
    assert supervisor.should_restart({"busy": False, "ts": now - 9999}, now, 20) == ""
    assert supervisor.should_restart({"busy": True, "ts": now - 60}, now, 20) == ""
    assert "no progress" in supervisor.should_restart({"busy": True, "ts": now - 25 * 60}, now, 20)
    assert supervisor.should_restart(None, now, 20) == ""


def test_supervisor_restarts_a_dead_program(tmp_path, monkeypatch):
    sup = supervisor.Supervisor([sys.executable, "-c", "import sys; sys.exit(3)"], tmp_path, check_url=False, grace_seconds=0)
    monkeypatch.setattr(supervisor.time, "sleep", lambda s: None)
    sup.start()
    sup.proc.wait()
    assert "exit code 3" in sup.step()
    assert sup.proc.poll() is None or sup.proc.poll() == 3
    sup.proc.wait()
    assert (tmp_path / "watchdog_last.json").is_file() and "exit code" in (tmp_path / "watchdog.log").read_text()


def test_supervisor_kills_a_stuck_program(tmp_path, monkeypatch):
    sup = supervisor.Supervisor([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, check_url=False,
                                grace_seconds=0, stale_minutes=1)
    monkeypatch.setattr(supervisor.time, "sleep", lambda s: None)
    sup.start()
    old = sup.proc
    (tmp_path / "heartbeat.json").write_text(json.dumps({"busy": True, "ts": time.time() - 600}))
    why = sup.step()
    assert "stuck" in why and old.poll() is not None and sup.proc is not old
    sup._kill = supervisor._kill_tree(sup.proc)


def test_heartbeat_file(tmp_path):
    heartbeat.set_busy(True)
    heartbeat.write(tmp_path / "hb.json")
    hb = heartbeat.read(tmp_path / "hb.json")
    assert hb["busy"] is True and abs(hb["ts"] - time.time()) < 5
    heartbeat.set_busy(False)


# ---------------------------------------------------------------- backup
def test_backup_and_restore_roundtrip(cfg):
    st = Store(cfg.path("paths.db"))
    st.kv_set("hello", "world")
    st.close()
    (cfg.root / "config.yaml").write_text("x: 1\n")
    (cfg.root / "client_secret.json").write_text("{}")
    f = backup.run(cfg)
    names = zipfile.ZipFile(f).namelist()
    assert any(n.startswith("db/") for n in names) and "config.yaml" in names
    assert "client_secret.json" not in names                       # secrets stay out by default
    st = Store(cfg.path("paths.db"))
    st.kv_set("hello", "changed")
    st.close()
    backup.restore(cfg, f)
    st = Store(cfg.path("paths.db"))
    assert st.kv_get("hello") == "world"
    st.close()


def test_backup_prunes_and_is_due(cfg):
    cfg.data.setdefault("backup", {})["keep"] = 2
    d = backup.backup_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        (d / f"featureclip_2026010{i}_0000.zip").write_bytes(b"x")
    backup.prune(cfg)
    assert len(list(d.glob("featureclip_*.zip"))) == 2
    assert backup.due(cfg, time.time() - 25 * 3600) and not backup.due(cfg, time.time() - 3600)


# ---------------------------------------------------------------- disk
def test_disk_cap_makes_cleanup_aggressive(cfg, tmp_path):
    cfg.data["cleanup"]["max_work_gb"] = 0.0000001          # any file is "over the cap"
    work = cfg.path("paths.workdir")
    (work / "live").mkdir(parents=True, exist_ok=True)
    seg = work / "live" / "seg_r00_00000.ts"
    seg.write_bytes(b"x" * 5000)
    old = time.time() - 7 * 3600
    os.utime(seg, (old, old))
    st = Store(cfg.path("paths.db"))
    r = cleanup.run(cfg, st)
    st.close()
    assert r["over_cap"] and not seg.exists()


def test_disk_critical(cfg, tmp_path):
    cfg.data["cleanup"]["stop_free_gb"] = 10 ** 9
    assert cleanup.disk_critical(cfg, tmp_path)
    cfg.data["cleanup"]["stop_free_gb"] = 0
    assert not cleanup.disk_critical(cfg, tmp_path)


# ---------------------------------------------------------------- per channel playbook
def test_playbook_is_per_channel(cfg):
    st = Store(cfg.path("paths.db"))
    rep_v = learn.refresh(st, cfg)
    assert rep_v["channel"] == "voiceover"
    assert learn.load(st, "streamer")["channel"] == "streamer" and learn.load(st, "voiceover")["channel"] == "voiceover"
    st.kv_set("playbook:streamer", json.dumps({"playbook": "STREAMER ONLY"}))
    assert learn.playbook_text(st, cfg, "streamer") == "STREAMER ONLY"
    assert "STREAMER" not in learn.playbook_text(st, cfg, "voiceover")
    st.close()
