"""Daily backup of everything that cannot be rebuilt: the database (clips, stats, queue, learning), settings and sources.

  data/backups/featureclip_YYYYmmdd_HHMM.zip     (the newest `backup.keep` are kept)

Secrets (.env, client_secret.json, token*.json) are NOT included unless backup.include_secrets is on.
Restore (program must be closed):  python -m featureclip restore data\\backups\\featureclip_....zip
"""
from __future__ import annotations

import sqlite3
import tempfile
import time
import zipfile
from pathlib import Path

PLAIN = ("config.yaml", "data/web_sources.json")
SECRETS = ("client_secret.json", ".env")


def backup_dir(cfg) -> Path:
    return cfg.root / "data" / "backups"


def _secret_files(cfg) -> list[str]:
    names = list(SECRETS)
    for p in cfg.root.glob("token*.json"):
        names.append(p.name)
    return names


def run(cfg) -> Path:
    d = backup_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    out = d / f"featureclip_{time.strftime('%Y%m%d_%H%M')}.zip"
    db = Path(cfg.path("paths.db"))
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        if db.is_file():                                  # consistent copy even while the program is writing
            snap = Path(tmp) / "db.sqlite"
            src, dst = sqlite3.connect(str(db), timeout=30), sqlite3.connect(str(snap))
            try:
                src.backup(dst)
            finally:
                dst.close()
                src.close()
            z.write(snap, "db/" + db.name)
        names = list(PLAIN) + (_secret_files(cfg) if cfg.get("backup.include_secrets", False) else [])
        for n in names:
            f = cfg.root / n
            if f.is_file():
                z.write(f, n)
    prune(cfg)
    return out


def prune(cfg) -> None:
    keep = max(1, int(cfg.get("backup.keep", 7)))
    files = sorted(backup_dir(cfg).glob("featureclip_*.zip"))
    for f in files[:-keep]:
        try:
            f.unlink()
        except Exception:
            pass


def listing(cfg) -> list[dict]:
    out = []
    for f in sorted(backup_dir(cfg).glob("featureclip_*.zip"), reverse=True):
        out.append({"name": f.name, "mb": round(f.stat().st_size / 1e6, 2), "ts": f.stat().st_mtime})
    return out


def due(cfg, last_ts: float, now: float | None = None) -> bool:
    return (now or time.time()) - last_ts >= float(cfg.get("backup.every_hours", 24)) * 3600


def restore(cfg, zip_path: str | Path) -> list[str]:
    """Put the files of a backup back. Run only while the program is closed."""
    zp = Path(zip_path)
    if not zp.is_file():
        raise FileNotFoundError(str(zp))
    done = []
    db = Path(cfg.path("paths.db"))
    with zipfile.ZipFile(zp) as z:
        for n in z.namelist():
            if n.startswith("db/"):
                db.parent.mkdir(parents=True, exist_ok=True)
                for ext in ("-wal", "-shm"):
                    Path(str(db) + ext).unlink(missing_ok=True)
                db.write_bytes(z.read(n))
                done.append(str(db))
            elif ".." not in n and not n.startswith("/"):
                dest = cfg.root / n
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(z.read(n))
                done.append(str(dest))
    return done
