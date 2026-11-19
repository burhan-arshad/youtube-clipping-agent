import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from featureclip.llm import LLM, LLMError, extract_json


class Fake(BaseHTTPRequestHandler):
    script: list = []          # [(status, body_text)] per request, consumed in order
    seen: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append({"path": self.path, "key": self.headers.get("x-goog-api-key"), "body": body})
        status, text = Fake.script.pop(0)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        if status == 200:
            self.wfile.write(json.dumps({"candidates": [{"content": {"parts": [{"text": text}]}}]}).encode())
        else:
            self.wfile.write(b"{}")

    models_list: list = []

    def do_GET(self):  # noqa: N802
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"models": [
            {"name": f"models/{n}", "supportedGenerationMethods": ["generateContent"]} for n in Fake.models_list
        ]}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    Fake.script, Fake.seen = [], []
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


@pytest.fixture
def llm(cfg, server, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    cfg.data["llm"]["gemini_base_url"] = server
    cfg.data["llm"]["models"] = ["m-main", "m-backup"]
    return LLM(cfg, sleep=lambda s: None)


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": {"b": [1,2]}}\n```') == {"a": {"b": [1, 2]}}
    assert extract_json('Sure! Here it is: {"a": "x}y"} hope it helps') == {"a": "x}y"}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_request_shape_and_images(llm):
    Fake.script = [(200, '{"ok": true}')]
    out = llm.json("hello", system="be nice", images=[b"\xff\xd8fake"])
    assert out == {"ok": True}
    req = Fake.seen[0]
    assert req["path"] == "/v1beta/models/m-main:generateContent"
    assert req["key"] == "test-key"                       # key in a header, never in the URL
    parts = req["body"]["contents"][0]["parts"]
    assert parts[0]["text"] == "hello" and parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    assert req["body"]["systemInstruction"]["parts"][0]["text"] .endswith("be nice")
    assert req["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_failover_on_rate_limit(llm):
    Fake.script = [(429, ""), (200, '{"from": "backup"}')]
    assert llm.json("x") == {"from": "backup"}
    assert [r["path"] for r in Fake.seen] == [
        "/v1beta/models/m-main:generateContent", "/v1beta/models/m-backup:generateContent"]


def test_bad_model_name_is_skipped_for_the_session(llm):
    Fake.script = [(404, ""), (200, '{"n": 1}'), (200, '{"n": 2}')]
    assert llm.json("x") == {"n": 1}
    assert llm.json("x") == {"n": 2}
    assert [r["path"].split("/")[-1].split(":")[0] for r in Fake.seen] == ["m-main", "m-backup", "m-backup"]


def test_broken_json_retries_then_works(llm):
    Fake.script = [(200, "not json at all"), (200, '{"fixed": true}')]
    assert llm.json("x") == {"fixed": True}


def test_everything_fails_raises(llm):
    Fake.script = [(500, "")] * 8
    with pytest.raises(LLMError):
        llm.json("x")


def test_no_key(cfg, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert not LLM(cfg).available()
    with pytest.raises(LLMError):
        LLM(cfg).json("x")


def test_all_models_gone_triggers_discovery(llm):
    Fake.models_list = ["gemini-9-flash-lite", "gemini-9-flash", "gemini-9-flash-image", "gemini-9-pro"]
    Fake.script = [(404, ""), (404, ""), (200, '{"found": true}')]
    assert llm.json("x") == {"found": True}
    assert Fake.seen[-1]["path"] == "/v1beta/models/gemini-9-flash:generateContent"   # best flash, no image/pro


def test_daily_quota_cooldown_until_reset_and_title_suffix():
    import time
    from featureclip.llm import next_quota_reset
    from featureclip import metadata
    now = time.time()
    r = next_quota_reset(now)
    assert now < r <= now + 24 * 3600 + 10
    assert metadata.norm_suffix("#Shorts") == " #Shorts" and metadata.norm_suffix("") == ""
    assert metadata.validate({"title": "GTA 6 gator walks into a store#Shorts", "description": "x" * 90}, "GTA 6") == []
