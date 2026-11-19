"""Google's free-tier quota days run on Pacific time: they reset at midnight in Los Angeles
(13:00 in Pakistan in winter / PST, 12:00 in summer / PDT). Everything that counts "today's" calls uses this."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone


def _tz():
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo("America/Los_Angeles")
    except Exception:
        return timezone(timedelta(hours=-8))        # no tz database (some Windows installs): assume winter time


def quota_day(now: float | None = None) -> str:
    """Date string of the current quota day, e.g. '2026-11-17' (Pacific time)."""
    return datetime.fromtimestamp(now if now is not None else time.time(), _tz()).strftime("%Y-%m-%d")


def day_start(now: float | None = None) -> float:
    """Unix time when the current quota day started."""
    d = datetime.fromtimestamp(now if now is not None else time.time(), _tz())
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def next_quota_reset(now: float | None = None) -> float:
    now = now if now is not None else time.time()
    d = datetime.fromtimestamp(now, _tz())
    return (d + timedelta(days=1)).replace(hour=0, minute=0, second=5, microsecond=0).timestamp()


def reset_text(now: float | None = None) -> str:
    """'13:00 PKT' style text for the next reset, in Pakistan time (UTC+5)."""
    t = next_quota_reset(now)
    return datetime.fromtimestamp(t, timezone(timedelta(hours=5))).strftime("%H:%M") + " PKT"
