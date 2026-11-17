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
                 frame_times: list[float], signal_score: float, clip_min: float, clip_max: float,
                 hints: list[str] | None = None) -> str:
    cats = ", ".join(CATEGORIES)
    times = ", ".join(f"{t:.0f}s" for t in frame_times)
    tr = transcript.strip() or "(no transcript available)"
    hint_txt = "\n".join(f"- {h}" for h in (hints or []))
    hint_block = (f"\nThings viewers of this game care about (look for these, but only report what you really see):\n{hint_txt}\n"
                  if hint_txt else "")
    return f"""Game: {game} ({topic_hint}).
You see {len(frame_times)} frames taken at {times} from a {window_len:.0f}-second window of a recording.
The frames are already cropped to the game picture. Decide whether this window shows ONE specific, visible
{game} feature that a short voice-over can explain, and if so, name it.

{hint_block}
The frames are numbered in time order (frame 1 is the earliest). Study how the picture CHANGES from frame to frame:
the feature is usually something that happens between two frames (a crash, a reaction, an animal appearing).
The streamer's words below are only a weak hint (they often talk about unrelated things). Trust the FRAMES.
\"\"\"{tr}\"\"\"
Automatic excitement score for the window (louder/more action = higher): {signal_score:.1f}

IS a feature (you can point at it in the frames): vehicle damage or physics, NPC reactions or behaviour, crowd
density, weather or lighting changes, water, reflections or other graphics detail, character animation details,
weapon handling, an interaction or mechanic, an interface element, a notable area or landmark of the map, animals.
IS NOT a feature (set feature_shown=false): ordinary driving, walking or standing with nothing special, plain
scenery, people only talking, menus, loading or black screens, a static image, a webcam or reaction, chat, text or
logos, a loud moment whose cause you cannot see.
Not allowed at all: glitches/exploits, mods/cheats. Cinematic scenes and trailers are fine when they clearly show a
visual or gameplay feature, but do not retell the story (no plot spoilers).

Proof rule: "visible_evidence" must list 2-3 short, concrete things you actually SEE in the frames that prove the
feature (for example "the car bonnet crumples after hitting the wall"). If you cannot write two concrete items,
set feature_shown=false.
"timeline": one entry per frame in order, so that the exact moment can be found later.
"key_moment_offset": the second (counted from the window start) at which the feature actually HAPPENS or becomes
visible, estimated from the frame times above (for a crash: the second of the impact, not the driving before it).
"real_person_or_overlay_visible": true if ANY frame shows a real person (webcam, streamer, a photographed face),
a chat window, a donation/subscriber alert, or text/logos added by the streamer. Game characters do not count.

Scoring: "confidence" = how sure you are about what is shown (0-100). "interest" = how cool it is for a general
gaming audience: ordinary 20-40, decent 45-60, clearly cool 65-80, amazing 85+. Most windows score under 50.

Allowed categories: {cats}

Return JSON with exactly these keys:
{{
 "feature_shown": true/false,
 "category": one of the allowed categories,
 "feature_name": "3-6 words naming the feature, e.g. 'Realistic car damage'",
 "what_happens": "1-2 sentences describing only what is visible",
 "visible_evidence": ["concrete thing 1", "concrete thing 2"],
 "timeline": [{{"frame": 1, "what": "what is visible in frame 1 (max 15 words)"}}, ...one per frame],
 "key_moment_offset": seconds from window start when the feature happens,
 "real_person_or_overlay_visible": true/false,
 "confidence": 0-100,
 "interest": 0-100,
 "hook_title": "max 7 words, curiosity-driven, no clickbait lies, no emoji",
 "start_offset": seconds from window start where the clip should begin (2-4 seconds BEFORE the key moment, so the viewer sees it coming),
 "end_offset": seconds from window start where it should end (at least 4 seconds after the key moment; clip length {clip_min:.0f}-{clip_max:.0f}s),
 "reason": "one short sentence"
}}"""


def grab_keyframes(video: str, start: float, end: float, n: int, width: int, tmp: Path,
                   crop=None) -> tuple[list[bytes], list[float]]:
    """n frames evenly spread over [start, end]. A frame that cannot be read (e.g. the last seconds of a file that is
    still being finished) is retried a little earlier and finally skipped; at least 2 must work."""
    tmp.mkdir(parents=True, exist_ok=True)
    wanted = [start + (end - start) * (i + 0.5) / n for i in range(n)]
    imgs: list[bytes] = []
    times: list[float] = []
    last_err = None
    for i, t in enumerate(wanted):
        p = tmp / f"kf_{i}.jpg"
        for back in (0.0, 1.5, 3.0):
            try:
                grab_frame(video, max(start, t - back), p, width, crop)
                imgs.append(p.read_bytes())
                times.append(max(start, t - back) - start)
                break
            except Exception as e:           # noqa: BLE001
                last_err = e
    if len(imgs) < min(2, n):
        raise RuntimeError(f"could not read frames near {wanted[0]:.0f}-{wanted[-1]:.0f}s of {Path(video).name}: {last_err}")
    return imgs, times


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
    km = raw.get("key_moment_offset")
    km = None if km in (None, "") else max(0.0, min(num(km, 0.0), window_len))
    if km is not None:                           # the viewer must see it coming and see the result
        so = min(so, max(0.0, km - 2.5))
        eo = max(eo, min(window_len, km + 4.5))
        if eo - so > clip_max:
            so = max(0.0, eo - clip_max)
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
        "visible_evidence": [str(x).strip()[:160] for x in (raw.get("visible_evidence") or [])
                             if isinstance(x, (str, int, float)) and str(x).strip()][:4],
        "real_person_or_overlay_visible": bool(raw.get("real_person_or_overlay_visible")) is True,
        "hook_title": str(raw.get("hook_title", "")).strip().strip('"')[:70],
        "key_moment_offset": None if km is None else round(km, 2),
        "timeline": [str(t.get("what", ""))[:140] if isinstance(t, dict) else str(t)[:140]
                     for t in (raw.get("timeline") or [])][:12],
        "start_offset": round(so, 2),
        "end_offset": round(eo, 2),
        "reason": str(raw.get("reason", "")).strip()[:300],
    }


PRESETS = {"loose": (45, 40), "normal": (58, 52), "strict": (70, 65)}


def thresholds(cfg) -> tuple[float, float]:
    """(min confidence, min interest): explicit numbers win, otherwise the strictness preset."""
    base = PRESETS.get(str(cfg.get("judge.strictness", "normal")).lower(), PRESETS["normal"])
    c, i = cfg.get("judge.min_confidence"), cfg.get("judge.min_interest")
    return float(c if c is not None else base[0]), float(i if i is not None else base[1])


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
    if j.get("real_person_or_overlay_visible"):
        return False, "streamer / webcam / chat / overlay visible in the picture"
    if cfg.get("judge.require_evidence", True) and len(j.get("visible_evidence", [])) < 2:
        return False, "no concrete visible evidence"
    min_conf, min_int = thresholds(cfg)
    if j["confidence"] < min_conf:
        return False, f"confidence {j['confidence']:.0f} too low (need {min_conf:.0f})"
    if j["interest"] < min_int:
        return False, f"interest {j['interest']:.0f} too low (need {min_int:.0f})"
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
        hints=list(cfg.get("game.hints", []) or []),
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
