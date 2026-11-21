"""YouTube upload (Data API v3, resumable). Private by default.

NOTE: if your Google Cloud project has not passed YouTube's API audit, YouTube
forces API uploads to 'private' no matter what you ask for. We read the real
privacy status back after every upload so you see what actually happened.
"""
from __future__ import annotations

import re
import json
import time
from pathlib import Path

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]
# Needed for watch-time / retention numbers (YouTube Analytics API). Added only when upload.analytics is on;
# the first run after switching it on asks you to sign in again.
ANALYTICS_SCOPE = "https://www.googleapis.com/auth/yt-analytics.readonly"
# Needed to change a video's visibility/schedule after upload (safe mode "pull back").
# youtube.upload alone cannot do videos.update; "youtube" covers it. Added only when upload.manage_videos is on.
MANAGE_SCOPE = "https://www.googleapis.com/auth/youtube"


def scopes_for(cfg) -> list[str]:
    sc = list(SCOPES)
    if cfg.get("upload.analytics", False):
        sc.append(ANALYTICS_SCOPE)
    if cfg.get("upload.manage_videos", False):
        sc.append(MANAGE_SCOPE)
    return sc


def build_title(hook: str, feature: str, suffix: str) -> str:
    from .metadata import norm_suffix
    suffix = norm_suffix(suffix)
    base = (hook or feature or "Gameplay feature").strip()
    base = re.sub(r"\s*#\w+", "", re.sub(r"\s+", " ", base)).strip()
    room = 100 - len(suffix)
    if len(base) > room:
        base = base[: room - 1].rstrip() + "…"
    return base + suffix


def build_description(judgement: dict, streamer: str, credit_url: str, cfg) -> str:
    """Plain-builder description (used when the AI cannot write one). Names only, no links."""
    from .metadata import credit_lines
    lines = [judgement.get("what_happens", "").strip(), ""] + credit_lines(cfg, streamer, judgement.get("streamer_clip"))
    lines += ["", "#Shorts #" + re.sub(r"[^A-Za-z0-9]", "", cfg.get("game.name", "gaming"))]
    return "\n".join(lines).strip()[:4900]


def build_body(title: str, description: str, cfg, tags: list[str] | None = None, publish_at: str | None = None,
               public: bool = False) -> dict:
    """publish_at: ISO UTC time. YouTube only accepts a schedule on a private video, so the video is uploaded
    private and YouTube flips it to public at that time (the laptop can be off by then)."""
    up = lambda k, d=None: cfg.get(f"upload.{k}", d)  # noqa: E731
    status = {
        "privacyStatus": up("privacy", "private"),
        "selfDeclaredMadeForKids": bool(up("made_for_kids", False)),
    }
    if public and not publish_at:
        status["privacyStatus"] = "public"
    if publish_at:
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at
    if up("synthetic_disclosure", True):
        # YouTube's "altered or synthetic content" disclosure (our voice-over is AI).
        status["containsSyntheticMedia"] = True
    return {
        "snippet": {
            "title": title,
            "description": description,
            "tags": list(tags if tags else up("tags", [])),
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
    scopes = scopes_for(cfg)
    creds = None
    if token.is_file():
        try:
            granted = set(json.loads(token.read_text(encoding="utf-8")).get("scopes") or [])
        except Exception:
            granted = set()
        if granted.issuperset(scopes):
            creds = Credentials.from_authorized_user_file(str(token), scopes)
        # else: a needed permission is missing -> sign in again below
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not secret.is_file():
            raise FileNotFoundError(
                f"{secret} not found. Create an OAuth 'Desktop app' client in Google Cloud Console "
                "(YouTube Data API v3 enabled) and save the JSON as client_secret.json.")
        flow = InstalledAppFlow.from_client_secrets_file(str(secret), scopes)
        creds = flow.run_local_server(port=0)
    token.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_full(path: str | Path, body: dict, cfg, thumb: str | Path | None = None) -> dict:
    """Upload, optionally set a custom thumbnail, and read back what YouTube really stored.
    Returns {id, privacy, publish_at, thumb_ok, thumb_error}."""
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
    out = {"id": vid, "privacy": body["status"]["privacyStatus"], "publish_at": body["status"].get("publishAt"),
           "thumb_ok": None, "thumb_error": ""}
    if thumb and Path(thumb).is_file():
        try:                                   # YouTube may refuse custom thumbnails for Shorts; that is not fatal
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(thumb), mimetype="image/jpeg")).execute()
            out["thumb_ok"] = True
        except Exception as e:
            out["thumb_ok"], out["thumb_error"] = False, str(e)[:200]
            small = _small_thumb(thumb)        # one more try with a lighter picture (size/format problems)
            if small and "forbidden" not in str(e).lower() and "403" not in str(e):
                try:
                    yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(str(small), mimetype="image/jpeg")).execute()
                    out["thumb_ok"], out["thumb_error"] = True, ""
                except Exception as e2:
                    out["thumb_error"] = str(e2)[:200]
    try:
        info = yt.videos().list(part="status", id=vid).execute()["items"][0]["status"]
        out["privacy"] = info["privacyStatus"]
        out["publish_at"] = info.get("publishAt")      # None = YouTube dropped the schedule (unaudited project)
    except Exception:
        out["privacy"] = f"{out['privacy']} (could not verify)"
    return out


def _small_thumb(thumb) -> Path | None:
    """A 1280x720, quality-80 copy of the thumbnail (YouTube wants <2 MB, 16:9)."""
    try:
        from PIL import Image
        p = Path(thumb)
        im = Image.open(p).convert("RGB")
        im.thumbnail((1280, 720))
        out = p.with_name(p.stem + "_small.jpg")
        im.save(out, "JPEG", quality=80)
        return out
    except Exception:
        return None


def upload_video(path: str | Path, body: dict, cfg) -> tuple[str, str]:
    """Upload and return (video_id, real_privacy_status)."""
    r = upload_full(path, body, cfg)
    return r["id"], r["privacy"]


def _service(cfg):
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=get_credentials(cfg), cache_discovery=False)


def fetch_stats(cfg, video_ids: list[str]) -> dict[str, dict]:
    """views / likes / comments for videos (50 per request, 1 quota unit). With upload.analytics on, also the
    average % viewed and average view seconds from the YouTube Analytics API (silently skipped if not allowed)."""
    out: dict[str, dict] = {}
    yt = _service(cfg)
    for i in range(0, len(video_ids), 50):
        chunk = video_ids[i:i + 50]
        for it in yt.videos().list(part="statistics,status", id=",".join(chunk)).execute().get("items", []):
            st = it.get("statistics", {})
            out[it["id"]] = {"views": int(st.get("viewCount", 0)), "likes": int(st.get("likeCount", 0)),
                             "comments": int(st.get("commentCount", 0)), "avg_pct": None, "avg_sec": None,
                             "privacy": it.get("status", {}).get("privacyStatus", "")}
    if out and cfg.get("upload.analytics", False):
        try:
            from googleapiclient.discovery import build
            ya = build("youtubeAnalytics", "v2", credentials=get_credentials(cfg), cache_discovery=False)
            today = time.strftime("%Y-%m-%d")
            for vid in out:
                r = ya.reports().query(ids="channel==MINE", startDate="2020-01-01", endDate=today,
                                       metrics="averageViewPercentage,averageViewDuration",
                                       filters=f"video=={vid}").execute()
                row = (r.get("rows") or [[None, None]])[0]
                out[vid]["avg_pct"], out[vid]["avg_sec"] = row[0], row[1]
        except Exception:
            pass
    return out


def set_visibility(cfg, video_id: str, publish_at: str | None = None, public: bool = False) -> None:
    """publish_at=None -> plain private (no schedule). publish_at=ISO time -> private + scheduled. public=True -> public now."""
    yt = _service(cfg)
    status = {"privacyStatus": "public" if public else "private", "selfDeclaredMadeForKids": False}
    if publish_at:
        status["publishAt"] = publish_at
    yt.videos().update(part="status", body={"id": video_id, "status": status}).execute()


def post_comment(cfg, video_id: str, text: str) -> str:
    """Post a top-level comment as the channel owner. Returns the comment id. (The API cannot pin comments.)"""
    yt = _service(cfg)
    r = yt.commentThreads().insert(part="snippet", body={"snippet": {"videoId": video_id, "topLevelComment": {
        "snippet": {"textOriginal": text[:9000]}}}}).execute()
    return r.get("id", "")
