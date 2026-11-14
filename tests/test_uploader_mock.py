"""The upload loop with a fake YouTube client (no network, no Google account needed)."""
import googleapiclient.discovery
import httplib2
import pytest
from googleapiclient.errors import HttpError

from featureclip import uploader


class FakeRequest:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)

    def next_chunk(self):
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return None, o


class FakeYT:
    def __init__(self, outcomes, privacy="private"):
        self.outcomes, self.privacy, self.inserted = outcomes, privacy, None

    def videos(self):
        return self

    def insert(self, part, body, media_body):
        self.inserted = {"part": part, "body": body}
        return FakeRequest(self.outcomes)

    def list(self, part, id):
        class R:
            def execute(s):
                return {"items": [{"status": {"privacyStatus": self_outer.privacy}}]}
        self_outer = self
        return R()


@pytest.fixture
def mp4(tmp_path):
    p = tmp_path / "a.mp4"
    p.write_bytes(b"\x00" * 2048)
    return p


def patch(monkeypatch, yt):
    monkeypatch.setattr(uploader, "get_credentials", lambda cfg: None)
    monkeypatch.setattr(googleapiclient.discovery, "build", lambda *a, **k: yt)
    monkeypatch.setattr(uploader.time, "sleep", lambda s: None)


def test_upload_returns_real_privacy(cfg, mp4, monkeypatch):
    yt = FakeYT([{"id": "vid1"}], privacy="private")
    patch(monkeypatch, yt)
    body = uploader.build_body("T", "D", cfg)
    body["status"]["privacyStatus"] = "public"           # we asked for public ...
    vid, privacy = uploader.upload_video(mp4, body, cfg)
    assert vid == "vid1"
    assert privacy == "private"                          # ... YouTube says private (unverified API project)
    assert yt.inserted["part"] == "snippet,status"


def test_upload_retries_server_errors(cfg, mp4, monkeypatch):
    err = HttpError(httplib2.Response({"status": 503}), b"busy")
    yt = FakeYT([err, err, {"id": "vid2"}])
    patch(monkeypatch, yt)
    vid, _ = uploader.upload_video(mp4, uploader.build_body("T", "D", cfg), cfg)
    assert vid == "vid2"


def test_upload_does_not_hide_client_errors(cfg, mp4, monkeypatch):
    err = HttpError(httplib2.Response({"status": 403}), b"quota")
    patch(monkeypatch, FakeYT([err]))
    with pytest.raises(HttpError):
        uploader.upload_video(mp4, uploader.build_body("T", "D", cfg), cfg)
