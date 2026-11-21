"""The channel playbook: trains the system on YouTube results.

Every published Short gets a performance score (0-100, ranked against the channel's own Shorts) from views per hour,
average % viewed (needs upload.analytics), likes and comments. Each clip's "features" (category, hook style, title
style, length, publish time, title shape) are compared: a feature whose clips beat the channel average gets a positive
lift, one that falls behind gets a negative lift (small samples are shrunk towards the average, so 2 lucky clips do
not decide anything). The result is
  * a short PLAYBOOK text that the script writer and the metadata writer get in their prompt, and
  * a report for the Assistant tab ("what the system learned").
No AI calls, plain arithmetic on the database. With too little data the playbook is empty and nothing changes."""
from __future__ import annotations

import json
import time

from . import channels, insights

MIN_CLIPS = 6          # published clips with stats before anything is learned
SHRINK = 3.0           # a feature with n clips counts n / (n + SHRINK) of its own average
MIN_LIFT = 5.0         # score points a feature must be above/below the channel average to be mentioned


def _pct_rank(values: list[float | None]) -> list[float | None]:
    present = sorted(v for v in values if v is not None)
    if len(present) < 2:
        return [50.0 if v is not None else None for v in values]
    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        below = sum(1 for p in present if p < v) + 0.5 * (sum(1 for p in present if p == v) - 1)
        out.append(100.0 * below / (len(present) - 1))
    return out


def _bucket(x: float, edges: list[tuple[float, str]], last: str) -> str:
    for lim, name in edges:
        if x < lim:
            return name
    return last


def _rows(store, cfg, key: str | None = None) -> list[dict]:
    base = {r["id"]: r for r in insights.clip_rows(store, float(cfg.get("insights.min_age_hours", 12)))}
    rows = []
    for c in store.clips(status="uploaded"):
        if key is not None and channels.key_of_clip(c) != key:        # each channel learns from its own audience only
            continue
        r = base.get(c["id"])
        if not r:
            continue
        j = insights._j(c["judgement"], {})
        title = c["title"] or ""
        dur = (j.get("end_offset") or 0) - (j.get("start_offset") or 0)
        views = max(1, r["views"])
        feats = {
            "category": r["category"],
            "hook style": f"#{r['hook_style']}" if r.get("hook_style") is not None else None,
            "title style": f"#{r['title_style']}" if r.get("title_style") is not None else None,
            "publish time": r["slot"] + " PKT",
            "clip length": (_bucket(dur, [(15, "under 15 s"), (25, "15-25 s"), (35, "25-35 s")], "35 s or more")
                            if dur > 0 else None),
            "title length": _bucket(len(title), [(40, "short title (<40)"), (62, "medium title (40-61)")], "long title (62+)"),
            "title has ?": "yes" if "?" in title else "no",
        }
        rows.append({"id": c["id"], "title": title, "feats": {k: v for k, v in feats.items() if v},
                     "rate": r["rate"], "pct": r["avg_pct"], "like": r["likes"] / views, "comm": r["comments"] / views,
                     "views": r["views"]})
    return rows


def _score(rows: list[dict]) -> None:
    parts = {"rate": 0.55, "pct": 0.30, "like": 0.10, "comm": 0.05}
    ranks = {k: _pct_rank([r[k] for r in rows]) for k in parts}
    for i, r in enumerate(rows):
        tot = wsum = 0.0
        for k, w in parts.items():
            v = ranks[k][i]
            if v is not None:
                tot += w * v
                wsum += w
        r["score"] = round(tot / wsum, 1) if wsum else 50.0


def build(store, cfg, key: str | None = "voiceover") -> dict:
    rows = _rows(store, cfg, key)
    analytics = any(r["pct"] is not None for r in rows)
    out = {"ts": time.time(), "channel": key, "clips": len(rows), "need": MIN_CLIPS, "analytics": analytics, "features": [],
           "best": [], "worst": [], "playbook": ""}
    if len(rows) < MIN_CLIPS:
        return out
    _score(rows)
    overall = sum(r["score"] for r in rows) / len(rows)
    groups: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        for k, v in r["feats"].items():
            groups.setdefault((k, v), []).append(r["score"])
    need = max(2, int(cfg.get("insights.min_samples", 3)) - 1)
    feats = []
    for (k, v), sc in groups.items():
        if len(sc) < need:
            continue
        shrunk = (sum(sc) + SHRINK * overall) / (len(sc) + SHRINK)
        feats.append({"feature": k, "value": v, "n": len(sc), "avg": round(sum(sc) / len(sc), 1),
                      "lift": round(shrunk - overall, 1)})
    feats.sort(key=lambda f: -abs(f["lift"]))
    out["features"] = feats
    out["overall"] = round(overall, 1)
    ranked = sorted(rows, key=lambda r: -r["score"])
    out["best"] = [{"id": r["id"], "title": r["title"][:80], "score": r["score"], "views": r["views"]} for r in ranked[:3]]
    out["worst"] = [{"id": r["id"], "title": r["title"][:80], "score": r["score"], "views": r["views"]} for r in ranked[-3:][::-1]]
    out["playbook"] = _playbook(out)
    return out


def _playbook(rep: dict) -> str:
    good = [f for f in rep["features"] if f["lift"] >= MIN_LIFT][:4]
    bad = [f for f in rep["features"] if f["lift"] <= -MIN_LIFT][:3]
    if not good and not bad and not rep["best"]:
        return ""
    lines = [f"CHANNEL PLAYBOOK (learned from {rep['clips']} published Shorts; use it as a tie-breaker, never against the footage):"]
    if good:
        lines.append("- Worked better than average: " + "; ".join(f"{f['feature']} = {f['value']} (+{f['lift']:.0f})" for f in good))
    if bad:
        lines.append("- Did worse than average: " + "; ".join(f"{f['feature']} = {f['value']} ({f['lift']:.0f})" for f in bad))
    if rep["best"]:
        lines.append("- Best titles so far (copy the energy, not the words): " + " | ".join(b["title"] for b in rep["best"][:2]))
    if rep["worst"] and rep["clips"] >= 10:
        lines.append("- Weakest titles (avoid that style): " + " | ".join(b["title"] for b in rep["worst"][:1]))
    return "\n".join(lines)[:700]


def _kv(key: str) -> str:
    return "playbook" if key == "voiceover" else f"playbook:{key}"      # the voice-over key keeps its old name


def refresh(store, cfg, key: str | None = None) -> dict:
    """Retrain one channel (or both). Returns the voice-over report for old callers."""
    out = {}
    for k in ([key] if key else list(channels.KEYS)):
        out[k] = build(store, cfg, k)
        store.kv_set(_kv(k), json.dumps(out[k]))
    return out.get("voiceover") or next(iter(out.values()))


def load(store, key: str = "voiceover") -> dict:
    try:
        return json.loads(store.kv_get(_kv(key), "") or "null") or {}
    except Exception:
        return {}


def playbook_text(store, cfg, key: str = "voiceover") -> str:
    """Text for the writers' prompts ('' while there is too little data or the feature is switched off)."""
    if not cfg.get("learn.use_playbook", True):
        return ""
    try:
        return str(load(store, key).get("playbook") or "")
    except Exception:
        return ""
