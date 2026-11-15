"""LLM judge: does this window show a concrete gameplay *feature*? What exactly?

The model sees a few keyframes + what the streamer said + the signal strength.
Hard rules are applied in code afterwards (the model proposes, code decides).
"""
from __future__ import annotations

from pathlib import Path

from .ffutil import grab_frame

CATEGORIES = [
    "new_feature", "gameplay_mechanic", "hidden_detail", "physics", "visual_detail",
    "npc_behavior", "world_interaction", "vehicle", "weapon", "ui_ux",
    # always rejected:
    "cutscene", "story_spoiler", "glitch_exploit", "mod_or_cheat", "menu_or_lobby", "nothing_notable",
]

SYSTEM = (
    "You are a careful video editor for a YouTube Shorts channel about a video game. "
    "You only state what is visible in the frames or said in the transcript. "
    "You never invent names, numbers, dates, prices or release facts. Reply with JSON only."
)


def build_prompt(game: str, topic_hint: str, transcript: str, window_len: float,
                 frame_times: list[float], signal_score: float, clip_min: float, clip_max: float) -> str:
    cats = ", ".join(CATEGORIES)
    times = ", ".join(f"{t:.0f}s" for t in frame_times)
    tr = transcript.strip() or "(no transcript available)"
    return f"""Game: {game} ({topic_hint}).
You see {len(frame_times)} frames taken at {times} from a {window_len:.0f}-second window of a recording.
The frames already show only the game picture. The footage can be live gameplay, a trailer or a first-reaction
stream: whatever it is, your job is to find the BEST thing in this window for a "{game} features" Short.
What the streamer said in this window (automatic transcript, may contain mistakes):
\"\"\"{tr}\"\"\"
Automatic excitement score for the window (louder/more action = higher): {signal_score:.1f}

Pick ONE concrete feature, mechanic or detail that a game fan would find cool and that one short voice-over can
explain. Be generous, not strict: lighting/graphics/weather detail, character or crowd animation, NPC behaviour,
physics, vehicles and driving, weapons, world detail, the map, interface, a new mechanic: all of these count.
Name it honestly from what is VISIBLE; never invent facts.
Say feature_shown=false ONLY if the frames show a menu, loading screen, black/blank screen, or nothing but a
static image or a talking face.
Not allowed: glitches/exploits, mods/cheats. Cinematic scenes and trailers are fine when they clearly show a
visual or gameplay feature, but do not retell the story (no plot spoilers).

How to score: "confidence" = how sure you are about what is shown (0-100). "interest" = how cool this is for a
general gaming audience: a normal good-looking moment is 45-60, something special 70+, something amazing 85+.

Allowed categories: {cats}

Return JSON with exactly these keys:
{{
 "feature_shown": true/false,
 "category": one of the allowed categories,
 "feature_name": "3-6 words naming the feature, e.g. 'Realistic car damage'",
 "what_happens": "1-2 sentences describing only what is visible or said",
 "confidence": 0-100,
 "interest": 0-100,
 "hook_title": "max 7 words, curiosity-driven, no clickbait lies, no emoji",
 "start_offset": seconds from window start where the clip should begin,
 "end_offset": seconds from window start where it should end (clip length {clip_min:.0f}-{clip_max:.0f}s),
 "reason": "one short sentence"
}}"""


def grab_keyframes(video: str, start: float, end: float, n: int, width: int, tmp: Path,
                   crop=None) -> tuple[list[bytes], list[float]]:
    tmp.mkdir(parents=True, exist_ok=True)
    times = [start + (end - start) * (i + 0.5) / n for i in range(n)]
    imgs: list[bytes] = []
    for i, t in enumerate(times):
        p = tmp / f"kf_{i}.jpg"
        grab_frame(video, t, p, width, crop)
        imgs.append(p.read_bytes())
    return imgs, [t - start for t in times]


def normalize(raw: dict, window_len: float, clip_min: float, clip_max: float) -> dict:
    """Clean up the model's JSON: types, ranges, sane clip bounds."""
    def num(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    cat = str(raw.get("category", "nothing_notable")).strip().lower()
    if cat not in CATEGORIES:
        cat = "nothing_notable"
    so = max(0.0, min(num(raw.get("start_offset"), 0.0), window_len))
    eo = max(0.0, min(num(raw.get("end_offset"), window_len), window_len))
    length = eo - so
    if length < clip_min or length > clip_max or eo <= so:
        so, eo = 0.0, min(window_len, clip_max)
        if eo - so < clip_min:           # window itself shorter than the minimum
            so = max(0.0, eo - clip_min)
    return {
        "feature_shown": bool(raw.get("feature_shown")) is True,
        "category": cat,
        "feature_name": str(raw.get("feature_name", "")).strip()[:80],
        "what_happens": str(raw.get("what_happens", "")).strip()[:600],
        "confidence": max(0.0, min(100.0, num(raw.get("confidence")))),
        "interest": max(0.0, min(100.0, num(raw.get("interest")))),
        "hook_title": str(raw.get("hook_title", "")).strip().strip('"')[:70],
        "start_offset": round(so, 2),
        "end_offset": round(eo, 2),
        "reason": str(raw.get("reason", "")).strip()[:300],
    }


def decide(j: dict, cfg) -> tuple[bool, str]:
    """Hard rules, applied in code no matter what the model says."""
    if not j["feature_shown"]:
        return False, "no feature shown"
    rejected = set(cfg.get("judge.reject_categories", []))
    if cfg.get("judge.allow_cutscenes", True):       # trailers / cinematic footage are fine
        rejected.discard("cutscene")
    if j["category"] in rejected:
        return False, f"rejected category: {j['category']}"
    if not j["feature_name"] or not j["what_happens"]:
        return False, "missing feature description"
    if j["confidence"] < float(cfg.get("judge.min_confidence", 60)):
        return False, f"confidence {j['confidence']:.0f} too low"
    if j["interest"] < float(cfg.get("judge.min_interest", 55)):
        return False, f"interest {j['interest']:.0f} too low"
    return True, "ok"


def judge_window(llm, cfg, video: str, start: float, end: float, transcript: str,
                 signal_score: float, tmp: Path, crop=None) -> dict:
    window_len = end - start
    images, times = [], []
    if cfg.get("llm.vision", True):
        images, times = grab_keyframes(video, start, end, int(cfg.get("llm.frames", 4)),
                                       int(cfg.get("llm.frame_width", 640)), tmp, crop)
    else:
        times = [window_len / 2]
    prompt = build_prompt(
        cfg.get("game.name"), cfg.get("game.topic_hint"), transcript, window_len, times,
        signal_score, float(cfg.get("video.clip_min")), float(cfg.get("video.clip_max")),
    )
    raw = llm.json(prompt, system=SYSTEM, images=images or None)
    j = normalize(raw, window_len, float(cfg.get("video.clip_min")), float(cfg.get("video.clip_max")))
    ok, why = decide(j, cfg)
    j["accept"], j["decision_reason"] = ok, why
    return j


def feature_key(name: str) -> str:
    return " ".join(sorted(set(name.lower().split())))


def dedupe(items: list[tuple[int, dict]], max_clips: int) -> list[int]:
    """items = [(clip_id, judgement)] of accepted clips -> ids to keep (best per feature, top N)."""
    best: dict[str, tuple[int, dict]] = {}
    for cid, j in items:
        key = feature_key(j["feature_name"])
        score = j["interest"] + j["confidence"]
        if key not in best or score > best[key][1]["interest"] + best[key][1]["confidence"]:
            best[key] = (cid, j)
    ranked = sorted(best.values(), key=lambda t: -(t[1]["interest"] + t[1]["confidence"]))
    return [cid for cid, _ in ranked[:max_clips]]
