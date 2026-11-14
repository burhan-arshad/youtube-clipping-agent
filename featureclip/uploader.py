"""YouTube upload (Data API v3, resumable). Private by default.

NOTE: if your Google Cloud project has not passed YouTube's API audit, YouTube
forces API uploads to 'private' no matter what you ask for. We read the real
privacy status back after every upload so you see what actually happened.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


def build_title(hook: str, feature: str, suffix: str) -> str:
    base = (hook or feature or "Gameplay feature").strip()
    base = re.sub(r"\s+", " ", base)
    room = 100 - len(suffix)
    if len(base) > room:
        base = base[: room - 1].rstrip() + "…"
    return base + suffix


def build_description(judgement: dict, streamer: str, credit_url: str, cfg) -> str:
    lines = [judgement.get("what_happens", "").strip(), ""]
    if streamer:
        credit = f"Gameplay footage: {streamer}"
        if credit_url:
            credit += f" ({credit_url})"
        lines.append(credit + " - used with permission.")
    lines.append("Voice-over: AI-generated.")
    footer = cfg.get("upload.description_footer", "").strip()
    if footer:
        lines += ["", footer]
    lines += ["", "#Shorts #" + re.sub(r"[^A-Za-z0-9]", "", cfg.get("game.name", "gaming"))]
    return "\n".join(lines).strip()[:4900]


def build_body(title: str, description: str, cfg) -> dict:
    up = lambda k, d=None: cfg.get(f"upload.{k}", d)  # noqa: E731
    status = {
        "privacyStatus": up("privacy", "private"),
        "selfDeclaredMadeForKids": bool(up("made_for_kids", False)),
    }
    if up("synthetic_disclosure", True):
        # YouTube's "altered or synthetic content" disclosure (our voice-over is AI).
        status["containsSyntheticMedia"] = True
    return {
        "snippet": {
            "title": title,
            "description": description,
            "tags": list(up("tags", [])),
            "categoryId": str(up("category_id", "20")),
        },
        "status": status,
    }


def get_credentials(cfg):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    secret = Path(cfg.get("upload.client_secret"))
    token = Path(cfg.get("upload.token"))
    if not secret.is_absolute():
        secret = cfg.root / secret
    if not token.is_absolute():
        token = cfg.root / token
    creds = None
    if token.is_file():
        creds = Credentials.from_authorized_user_file(str(token), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not secret.is_file():
            raise FileNotFoundError(
                f"{secret} not found. Create an OAuth 'Desktop app' client in Google Cloud Console "
                "(YouTube Data API v3 enabled) and save the JSON as client_secret.json.")
        flow = InstalledAppFlow.from_client_secrets_file(str(secret), SCOPES)
        creds = flow.run_local_server(port=0)
    token.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_video(path: str | Path, body: dict, cfg) -> tuple[str, str]:
    """Upload and return (video_id, real_privacy_status)."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    yt = build("youtube", "v3", credentials=get_credentials(cfg), cache_discovery=False)
    media = MediaFileUpload(str(path), mimetype="video/mp4", resumable=True, chunksize=8 * 1024 * 1024)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    response, retries = None, 0
    while response is None:
        try:
            _status, response = req.next_chunk()
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 5:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    vid = response["id"]
    privacy = body["status"]["privacyStatus"]
    try:
        info = yt.videos().list(part="status", id=vid).execute()
        privacy = info["items"][0]["status"]["privacyStatus"]
    except Exception:
        privacy = f"{privacy} (could not verify)"
    return vid, privacy
