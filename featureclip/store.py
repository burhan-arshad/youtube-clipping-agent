"""SQLite job store. Clip status machine:

candidate -> judged -> rendered -> ready -> approved -> uploaded
                 \\-> dropped (judge/QA said no)    ready -> rejected (human said no)
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id TEXT NOT NULL,
    path TEXT NOT NULL,
    duration REAL,
    width INTEGER, height INTEGER,
    stage TEXT NOT NULL DEFAULT 'new',
    created REAL
);
CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    start REAL, end REAL,
    signal_score REAL,
    status TEXT NOT NULL DEFAULT 'candidate',
    judgement TEXT,          -- json from the LLM judge
    script TEXT,             -- voice-over text
    title TEXT,
    description TEXT,
    path TEXT,               -- final mp4
    qa TEXT,                 -- json
    youtube_id TEXT,
    error TEXT,
    created REAL, updated REAL,
    FOREIGN KEY(video_id) REFERENCES videos(id)
);
CREATE TABLE IF NOT EXISTS llm_usage (
    day TEXT, model TEXT, calls INTEGER DEFAULT 0,
    PRIMARY KEY(day, model)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    ts REAL, model TEXT, ok INTEGER, kind TEXT, err TEXT
);
CREATE INDEX IF NOT EXISTS llm_calls_ts ON llm_calls(ts);
CREATE TABLE IF NOT EXISTS llm_cooldown (
    model TEXT PRIMARY KEY, until REAL, reason TEXT
);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, level TEXT, title TEXT, body TEXT, read INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT, url TEXT, kind TEXT, name TEXT, status TEXT DEFAULT 'queued',
    streamer TEXT, credit_url TEXT, note TEXT, added REAL, started REAL, finished REAL, options TEXT
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS clip_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT, clip_id INTEGER NOT NULL, ts REAL NOT NULL,
    views INTEGER, likes INTEGER, comments INTEGER, avg_view_pct REAL, avg_view_sec REAL
);
CREATE INDEX IF NOT EXISTS idx_clip_stats ON clip_stats(clip_id, ts);
CREATE TABLE IF NOT EXISTS llm_stats (
    day TEXT, model TEXT, rate_limited INTEGER DEFAULT 0, errors INTEGER DEFAULT 0,
    last_error TEXT, last_at REAL,
    PRIMARY KEY(day, model)
);
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # several threads (live worker, assistant, web requests) each open their own connection
        self.db = sqlite3.connect(str(self.path), timeout=30)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:
            pass
        self.db.executescript(SCHEMA)
        have = {r["name"] for r in self.db.execute("PRAGMA table_info(clips)")}
        for col, typ in (("thumb", "TEXT"), ("tags", "TEXT"), ("scheduled_at", "REAL"), ("review", "TEXT"),
                         ("publish_state", "TEXT"), ("rebuilds", "INTEGER DEFAULT 0")):
            if col not in have:
                self.db.execute(f"ALTER TABLE clips ADD COLUMN {col} {typ}")
        self.db.commit()

    def close(self) -> None:
        self.db.close()

    # ---- videos
    def add_video(self, source_id: str, path: str, info: dict) -> int:
        row = self.db.execute("SELECT id FROM videos WHERE source_id=? AND path=?",
                              (source_id, path)).fetchone()
        if row:
            return int(row["id"])
        cur = self.db.execute(
            "INSERT INTO videos(source_id,path,duration,width,height,created) VALUES(?,?,?,?,?,?)",
            (source_id, path, info.get("duration"), info.get("width"), info.get("height"), time.time()),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def video(self, video_id: int) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM videos WHERE id=?", (video_id,)).fetchone()
        if not row:
            raise KeyError(f"video {video_id} not found")
        return row

    def set_video_stage(self, video_id: int, stage: str) -> None:
        self.db.execute("UPDATE videos SET stage=? WHERE id=?", (stage, video_id))
        self.db.commit()

    def videos(self) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT * FROM videos ORDER BY id"))

    # ---- clips
    def add_clip(self, video_id: int, start: float, end: float, signal_score: float) -> int:
        now = time.time()
        cur = self.db.execute(
            "INSERT INTO clips(video_id,start,end,signal_score,created,updated) VALUES(?,?,?,?,?,?)",
            (video_id, start, end, signal_score, now, now),
        )
        self.db.commit()
        return int(cur.lastrowid)

    def clip(self, clip_id: int) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        if not row:
            raise KeyError(f"clip {clip_id} not found")
        return row

    def clips(self, video_id: int | None = None, status: str | None = None) -> list[sqlite3.Row]:
        q, args = "SELECT * FROM clips WHERE 1=1", []
        if video_id is not None:
            q += " AND video_id=?"
            args.append(video_id)
        if status is not None:
            q += " AND status=?"
            args.append(status)
        return list(self.db.execute(q + " ORDER BY id", args))

    def update_clip(self, clip_id: int, **fields: Any) -> None:
        if not fields:
            return
        for k in ("judgement", "qa", "review", "tags"):
            if k in fields and not isinstance(fields[k], (str, type(None))):
                fields[k] = json.dumps(fields[k], ensure_ascii=False)
        fields["updated"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE clips SET {cols} WHERE id=?", (*fields.values(), clip_id))
        self.db.commit()

    def delete_candidates(self, video_id: int) -> None:
        self.db.execute("DELETE FROM clips WHERE video_id=? AND status='candidate'", (video_id,))
        self.db.commit()

    # ---- llm usage (free-tier budgets)
    def usage_today(self, model: str) -> int:
        day = time.strftime("%Y-%m-%d")
        row = self.db.execute("SELECT calls FROM llm_usage WHERE day=? AND model=?",
                              (day, model)).fetchone()
        return int(row["calls"]) if row else 0

    def bump_usage(self, model: str) -> None:
        day = time.strftime("%Y-%m-%d")
        self.db.execute(
            "INSERT INTO llm_usage(day,model,calls) VALUES(?,?,1) "
            "ON CONFLICT(day,model) DO UPDATE SET calls=calls+1", (day, model))
        self.db.commit()

    def bump_stat(self, model: str, kind: str, err: str = "") -> None:
        """kind: 'rate_limited' or 'errors' (free-tier usage page)."""
        if kind not in ("rate_limited", "errors"):
            return
        day = time.strftime("%Y-%m-%d")
        self.db.execute(
            f"INSERT INTO llm_stats(day,model,{kind},last_error,last_at) VALUES(?,?,1,?,?) "
            f"ON CONFLICT(day,model) DO UPDATE SET {kind}={kind}+1, last_error=excluded.last_error, "
            f"last_at=excluded.last_at", (day, model, err[:300], time.time()))
        self.db.commit()

    def usage_report(self, days: int = 7) -> dict:
        since = time.strftime("%Y-%m-%d", time.localtime(time.time() - (days - 1) * 86400))
        calls = self.db.execute("SELECT day, model, calls FROM llm_usage WHERE day>=? ORDER BY day", (since,)).fetchall()
        stats = self.db.execute("SELECT day, model, rate_limited, errors, last_error, last_at FROM llm_stats "
                                "WHERE day>=?", (since,)).fetchall()
        return {"calls": [dict(r) for r in calls], "stats": [dict(r) for r in stats]}

    # ---- per-call log (live usage page)
    def log_call(self, model: str, ok: bool, kind: str = "", err: str = "") -> None:
        self.db.execute("INSERT INTO llm_calls(ts,model,ok,kind,err) VALUES(?,?,?,?,?)",
                        (time.time(), model, 1 if ok else 0, kind, err[:300]))
        self.db.execute("DELETE FROM llm_calls WHERE ts<?", (time.time() - 8 * 86400,))
        self.db.commit()

    def calls_since(self, since: float) -> list[sqlite3.Row]:
        return list(self.db.execute("SELECT ts, model, ok, kind, err FROM llm_calls WHERE ts>=? ORDER BY ts", (since,)))

    def set_cooldown(self, model: str, until: float, reason: str) -> None:
        self.db.execute("INSERT INTO llm_cooldown(model,until,reason) VALUES(?,?,?) "
                        "ON CONFLICT(model) DO UPDATE SET until=excluded.until, reason=excluded.reason",
                        (model, until, reason[:200]))
        self.db.commit()

    def cooldowns(self) -> dict[str, dict]:
        now = time.time()
        return {r["model"]: {"until": r["until"], "reason": r["reason"]}
                for r in self.db.execute("SELECT * FROM llm_cooldown") if r["until"] > now}

    # ---- notifications
    def add_notification(self, level: str, title: str, body: str = "") -> int:
        cur = self.db.execute("INSERT INTO notifications(ts,level,title,body) VALUES(?,?,?,?)",
                              (time.time(), level, title, body))
        self.db.commit()
        return int(cur.lastrowid)

    def notifications(self, limit: int = 50) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM notifications ORDER BY id DESC LIMIT ?", (limit,))]

    def mark_notifications_read(self) -> None:
        self.db.execute("UPDATE notifications SET read=1")
        self.db.commit()

    # ---- assistant source queue
    def add_agent_source(self, url: str, kind: str, name: str = "", streamer: str = "", credit_url: str = "",
                         options: dict | None = None) -> int:
        cur = self.db.execute(
            "INSERT INTO agent_sources(url,kind,name,streamer,credit_url,added,options) VALUES(?,?,?,?,?,?,?)",
            (url, kind, name or url, streamer, credit_url, time.time(), json.dumps(options or {})))
        self.db.commit()
        return int(cur.lastrowid)

    def agent_sources(self, status: str | None = None) -> list[dict]:
        q = "SELECT * FROM agent_sources" + (" WHERE status=?" if status else "") + " ORDER BY id"
        return [dict(r) for r in self.db.execute(q, (status,) if status else ())]

    def update_agent_source(self, sid: int, **fields: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in fields)
        self.db.execute(f"UPDATE agent_sources SET {cols} WHERE id=?", (*fields.values(), sid))
        self.db.commit()

    def delete_agent_source(self, sid: int) -> None:
        self.db.execute("DELETE FROM agent_sources WHERE id=?", (sid,))
        self.db.commit()

    # ---- small key/value (assistant state)
    def kv_get(self, key: str, default: str = "") -> str:
        row = self.db.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def kv_set(self, key: str, value: str) -> None:
        self.db.execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                        (key, value))
        self.db.commit()

    # ---- YouTube stats snapshots
    def add_stats(self, clip_id: int, views, likes, comments, avg_pct=None, avg_sec=None) -> None:
        self.db.execute("INSERT INTO clip_stats(clip_id,ts,views,likes,comments,avg_view_pct,avg_view_sec) VALUES(?,?,?,?,?,?,?)",
                        (clip_id, time.time(), views, likes, comments, avg_pct, avg_sec))
        self.db.commit()

    def latest_stats(self) -> dict[int, dict]:
        """Newest snapshot per clip."""
        out = {}
        for r in self.db.execute("SELECT * FROM clip_stats ORDER BY ts"):
            out[r["clip_id"]] = dict(r)
        return out

    def stats_history(self, clip_id: int) -> list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM clip_stats WHERE clip_id=? ORDER BY ts", (clip_id,))]

    def video_by_path(self) -> dict[str, int]:
        return {r["path"]: r["id"] for r in self.db.execute("SELECT id, path FROM videos")}
