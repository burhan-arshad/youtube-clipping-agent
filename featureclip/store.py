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
"""


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path))
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
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
        for k in ("judgement", "qa"):
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
