"""Publishing slots. Times are written in Pakistan time (PKT, UTC+5, no daylight saving) and converted to UTC
for YouTube's `publishAt`.

Default 3 slots per day, chosen for the target audience (UK, Europe, USA) in winter time:
  21:00 PKT = 16:00 UK / 17:00 CET / 11:00 US-East / 08:00 US-West   (after school, commute home)
  01:00 PKT = 20:00 UK / 21:00 CET / 15:00 US-East / 12:00 US-West   (UK + Europe prime time, US afternoon)
  05:00 PKT = 00:00 UK / 01:00 CET / 19:00 US-East / 16:00 US-West   (US evening prime time)
In summer the UK/EU/US clocks move one hour, so re-check the slots in March.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

DEFAULT_SLOTS = ["21:00", "01:00", "05:00"]
try:
    PKT = ZoneInfo("Asia/Karachi")
except Exception:                      # Windows without tzdata: Pakistan has no daylight saving, so a fixed UTC+5 is exact
    PKT = timezone(timedelta(hours=5), "PKT")


def parse_slots(value) -> list[tuple[int, int]]:
    """'21:00, 01:00' or a list -> [(21, 0), (1, 0)] (sorted, unique, invalid items ignored)."""
    items = value if isinstance(value, (list, tuple)) else re.split(r"[,\s;]+", str(value or ""))
    out = set()
    for it in items:
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(it))
        if m and int(m.group(1)) < 24 and int(m.group(2)) < 60:
            out.add((int(m.group(1)), int(m.group(2))))
    return sorted(out)


def audience_times(slot_pkt: tuple[int, int]) -> dict[str, str]:
    """What a PKT slot means for the target audiences (today's real UTC offsets, so DST is handled)."""
    now = datetime.now(timezone.utc)
    base = datetime(now.year, now.month, now.day, slot_pkt[0], slot_pkt[1], tzinfo=PKT)
    zones = {"UK": "Europe/London", "Europe (CET)": "Europe/Paris", "US East": "America/New_York",
             "US West": "America/Los_Angeles"}
    out = {}
    for n, z in zones.items():
        try:
            out[n] = base.astimezone(ZoneInfo(z)).strftime("%H:%M")
        except Exception:                      # no tz database: skip that audience instead of crashing
            pass
    return out


def next_slots(now: datetime, n: int, slots: list[tuple[int, int]], taken: list[datetime] | None = None,
               min_lead_minutes: int = 20, horizon_days: int = 30) -> list[datetime]:
    """The next `n` free publish times (UTC, timezone-aware) at or after now + lead, skipping `taken`."""
    if not slots or n <= 0:
        return []
    now = now.astimezone(timezone.utc)
    earliest = now + timedelta(minutes=min_lead_minutes)
    taken_utc = {t.astimezone(timezone.utc).replace(second=0, microsecond=0) for t in (taken or [])}
    day0 = now.astimezone(PKT).date()
    out: list[datetime] = []
    for d in range(-1, horizon_days):
        day = day0 + timedelta(days=d)
        for h, m in slots:
            t = datetime(day.year, day.month, day.day, h, m, tzinfo=PKT).astimezone(timezone.utc)
            if t >= earliest and t not in taken_utc:
                out.append(t)
                if len(out) == n:
                    return out
    return out


def iso_utc(t: datetime) -> str:
    return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def fmt_pkt(t: datetime) -> str:
    return t.astimezone(PKT).strftime("%a %d %b, %H:%M PKT")
