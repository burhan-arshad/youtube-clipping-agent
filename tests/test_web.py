import json
from pathlib import Path

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from featureclip.web.app import create_app


@pytest.fixture()
def client(tmp_path):
    return TestClient(create_app(tmp_path))


def test_index_and_state(client):
    r = client.get("/")
    assert r.status_code == 200 and "featureclip" in r.text
    s = client.get("/api/state").json()
    assert {"has_key", "ffmpeg", "counts", "max_per_day"} <= set(s)


def test_settings_roundtrip_and_secret_masked(client, tmp_path):
    client.post("/api/settings", json={"env:GEMINI_API_KEY": "AIzaSyTESTKEY1234567890", "judge.strictness": "strict"})
    fields = {f["key"]: f for f in client.get("/api/settings").json()}
    k = fields["env:GEMINI_API_KEY"]
    assert k["is_set"] and k["value"] == "" and "TESTKEY" not in json.dumps(k)
    assert fields["judge.strictness"]["value"] == "strict"
    assert "AIzaSyTESTKEY" in (tmp_path / ".env").read_text()
    # empty secret box keeps the old key
    client.post("/api/settings", json={"env:GEMINI_API_KEY": ""})
    assert "AIzaSyTESTKEY" in (tmp_path / ".env").read_text()


def test_job_validation(client, monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    assert client.post("/api/jobs", json={"kind": "run", "video": ""}).status_code == 400
    assert client.post("/api/jobs", json={"kind": "run", "video": "x.mp4"}).status_code == 400  # no key
    assert client.post("/api/jobs", json={"kind": "nope"}).status_code == 400


def test_clips_empty_and_secret_validation(client):
    assert client.get("/api/clips?status=ready").json() == []
    assert client.post("/api/clips/approve-all").json() == {"approved": 0}
    assert client.post("/api/youtube/client-secret", json={"content": "not json"}).status_code == 400
    ok = client.post("/api/youtube/client-secret", json={"content": json.dumps({"installed": {}})})
    assert ok.status_code == 200
    assert client.get("/api/state").json()["client_secret"] is True


def test_layout_job_runs_and_reports_error(client, tmp_path):
    import time
    r = client.post("/api/jobs", json={"kind": "layout", "video": str(tmp_path / "missing.mp4")}).json()
    for _ in range(50):
        j = client.get(f"/api/jobs/{r['id']}").json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(0.1)
    assert j["status"] == "error"


def test_usage_endpoint(client):
    u = client.get("/api/usage").json()
    assert u["ok_today"] == 0 and u["fail_today"] == 0 and "limits" in u and "models" in u


def test_usage_counts_calls_and_cooldowns(client, tmp_path):
    from featureclip.config import load_config
    from featureclip.store import Store
    st = Store(load_config(None, root=tmp_path).path("paths.db"))
    st.log_call("gemini-3.5-flash-lite", True)
    st.log_call("gemini-3.5-flash-lite", True)
    st.log_call("gemini-3.6-flash", False, "", "429 daily quota")
    st.set_cooldown("gemini-3.6-flash", __import__("time").time() + 3000, "daily quota used up")
    u = client.get("/api/usage").json()
    by = {m["model"]: m for m in u["models"]}
    assert by["gemini-3.5-flash-lite"]["ok"] == 2 and by["gemini-3.5-flash-lite"]["rpm_now"] == 2
    assert by["gemini-3.5-flash-lite"]["limit_rpd"] == 500 and by["gemini-3.5-flash-lite"]["google_peak"]["rpd"] == 434
    assert by["gemini-3.6-flash"]["fail"] == 1 and by["gemini-3.6-flash"]["cooldown"]["left"] > 2000
    assert u["recent"][0]["model"] == "gemini-3.6-flash"


def test_assistant_queue_and_schedule_and_notifications(client):
    r = client.post("/api/assistant/sources", json={"url": "https://youtube.com/watch?v=abc"})
    assert r.status_code == 200 and r.json()["kind"] == "auto"   # the assistant asks YouTube if it is live
    assert client.post("/api/assistant/sources", json={"url": "https://youtube.com/live/xyz"}).json()["kind"] == "live"
    assert client.post("/api/assistant/sources", json={"url": "nope"}).status_code == 400
    a = client.get("/api/assistant").json()
    assert len(a["sources"]) == 2 and a["enabled"] is False
    sid = a["sources"][0]["id"]
    assert client.delete(f"/api/assistant/sources/{sid}").status_code == 200
    s = client.get("/api/schedule").json()
    assert [x["pkt"] for x in s["slots"]] == ["01:00", "05:00", "21:00"] and len(s["free"]) == 9
    assert "UK" in s["slots"][0]["audience"]
    client.post("/api/notifications/test")
    n = client.get("/api/notifications").json()
    assert n["unread"] == 1 and n["items"][0]["title"] == "Test notification"
    client.post("/api/notifications/read")
    assert client.get("/api/notifications").json()["unread"] == 0


def test_assistant_settings_roundtrip(client, tmp_path):
    client.post("/api/settings", json={"assistant.enabled": True, "schedule.slots": "20:00, 02:00", "notify.ntfy_topic": "my-topic-123"})
    f = {x["key"]: x for x in client.get("/api/settings").json()}
    assert f["assistant.enabled"]["value"] is True
    assert f["schedule.slots"]["value"] == "20:00, 02:00"
    assert client.get("/api/assistant").json()["enabled"] is True
