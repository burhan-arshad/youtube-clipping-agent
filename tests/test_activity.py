import json
import threading

from fastapi.testclient import TestClient

from featureclip import activity, ingest
from featureclip.web.app import create_app


def test_activity_download_percent_and_slow_live_warning():
    r = activity.from_lines(["[download] Destination: x", "[download]  43.2% of 100MiB at 5MiB/s"])
    assert r["stage"] == "record" and r["progress"] == 43.2 and "43.2%" in r["headline"]
    slow = ["frame=15599 fps= 22 q=-1.0 size=  191232KiB time=00:04:20.08 bitrate=6023.3kbits/s speed=0.363x"] * 3
    r = activity.from_lines(slow)
    assert "4m 20s" in r["headline"] and "LIVE" in r["warning"]
    fast = ["frame=1 fps=30 time=00:00:10.00 speed=1.01x"]
    assert activity.from_lines(fast)["warning"] == ""


def test_activity_stage_follows_newest_line_and_hides_noise():
    lines = ["[10:00:00] signals: analysing audio + motion (CPU)", "[10:01:00] judging up to 8 candidates (stops early)",
             "[10:01:05]   clip 1 @ 40s: ACCEPT 'Car damage' (conf 90, interest 80)",
             "[10:01:30] building clip 1 (Car damage) ...", "[https @ 0x1] Opening 'https://x' for reading"]
    r = activity.from_lines(lines)
    assert r["stage"] == "build" and "Car damage" in r["headline"]
    assert all("Opening" not in e["text"] for e in r["recent"]) and len(r["recent"]) == 4
    assert activity.from_lines(["[10:02:00]   AI unavailable: quota"])["stage"] == "judge"


def test_live_endpoint_idle_and_stop_restart(tmp_path):
    c = TestClient(create_app(tmp_path))
    L = c.get("/api/assistant/live").json()
    assert L["mode"] == "idle" and "switched off" in L["headline"] and len(L["steps"]) == 7
    r = c.post("/api/assistant/sources", json={"url": "https://www.youtube.com/watch?v=abc", "kind": "auto"})
    assert r.json()["kind"] == "auto"                                  # decided later by asking YouTube, not by the URL shape
    sid = r.json()["id"]
    assert c.post(f"/api/assistant/sources/{sid}/stop", json={"restart": True}).status_code == 200
    src = c.get("/api/assistant").json()["sources"][0]
    assert src["status"] == "queued" and src["kind"] == "auto"
    c.post(f"/api/assistant/sources/{sid}/stop", json={})
    assert c.get("/api/assistant").json()["sources"][0]["status"] == "error"


def test_ingest_download_can_be_cancelled(tmp_path, monkeypatch):
    from featureclip.sources import Source

    class P:
        def __init__(self, *a, **k):
            self.stdout = iter(["[download]   1.0% of 1MiB\n"] * 50)
            self.pid = 1
            self.killed = False

        def wait(self, timeout=None):
            return 0

        def terminate(self):
            self.killed = True
    monkeypatch.setattr(ingest.subprocess, "Popen", P)
    stop = threading.Event()
    stop.set()
    try:
        ingest.ingest(Source(id="x", streamer="s", url="https://youtube.com/watch?v=1"), tmp_path, None, stop=stop)
        assert False
    except RuntimeError as e:
        assert "stopped" in str(e)
