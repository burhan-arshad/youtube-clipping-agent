"""Learning from YouTube results: which hook / title style and which clip category get more views.
Pure arithmetic on the database (no AI calls). Until enough clips have stats, every weight is neutral (1.0)."""
from __future__ import annotations

from . import insights


def _multipliers(rows: list[dict], key, need: int) -> dict:
    """{value: multiplier 0.6-1.6} = how much better/worse than the average a value's views per hour are."""
    if len(rows) < need * 3:
        return {}
    overall = sum(r["rate"] for r in rows) / len(rows)
    if overall <= 0:
        return {}
    groups: dict = {}
    for r in rows:
        v = key(r)
        if v is not None:
            groups.setdefault(v, []).append(r["rate"])
    return {v: round(max(0.6, min(1.6, (sum(rs) / len(rs)) / overall)), 2)
            for v, rs in groups.items() if len(rs) >= need}


def weights(store, cfg) -> dict:
    """{"hook": {style_index: weight}, "title": {...}} for narrate.STYLE_WEIGHTS."""
    try:
        need = int(cfg.get("insights.min_samples", 3))
        rows = insights.clip_rows(store, float(cfg.get("insights.min_age_hours", 12)))
        return {"hook": _multipliers(rows, lambda r: r.get("hook_style"), need),
                "title": _multipliers(rows, lambda r: r.get("title_style"), need)}
    except Exception:
        return {}


def category_boost(store, cfg) -> dict:
    """{category: multiplier} used to rank clips of the same piece: categories that did better are made first."""
    try:
        need = int(cfg.get("insights.min_samples", 3))
        rows = insights.clip_rows(store, float(cfg.get("insights.min_age_hours", 12)))
        return _multipliers(rows, lambda r: r.get("category"), need)
    except Exception:
        return {}
