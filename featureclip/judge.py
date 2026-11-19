"""LLM judge: does this window hold a clip-worthy moment (feature, funny, chaos, epic, discovery, drama)? What exactly?

The model sees a few keyframes + what the streamer said + the signal strength.
Hard rules are applied in code afterwards (the model proposes, code decides).
"""
from __future__ import annotations

from pathlib import Path

from .ffutil import grab_frame

CATEGORIES = [
    "new_feature", "gameplay_mechanic", "hidden_detail", "physics", "visual_detail",
    "npc_behavior", "world_interaction", "vehicle", "weapon", "ui_ux",
    # general highlight moments (not only features)
    "funny_moment", "chaotic_action", "epic_stunt", "interesting_discovery", "dramatic_moment", "gameplay_highlight",
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
    return f"""Game: {game} ({topic_hint}).{hint_block}
You see {len(frame_times)} frames taken at {times} from a {window_len:.0f}-second window of a recording.
The frames are already cropped to the game picture. Decide whether this window contains ONE clip-worthy
moment that people would enjoy as a 15-40 second Short, and if so, name it. A clip-worthy moment can be ANY of:
 - a gameplay FEATURE or detail (graphics, physics, damage, NPC behaviour, animals, weather, an interaction)
 - a FUNNY moment (a fail, absurd NPC reaction, ragdoll/physics comedy, a ridiculous situation, awkward luck)
 - CHAOTIC ACTION (a chase, a pile-up, an explosion, a shootout, police going wild)
 - an EPIC or skilful moment (a stunt jump, a close escape, a lucky shot, a clutch save)
 - an interesting DISCOVERY (a hidden place or detail, a surprise, something rarely seen)
 - a DRAMATIC moment (tension, a near miss, a big reveal in the gameplay itself)

The frames are numbered in time order (frame 1 is the earliest). Study how the picture CHANGES from frame to frame:
the feature is usually something that happens between two frames (a crash, a reaction, an animal appearing).
The streamer's words below are only a weak hint (they often talk about unrelated things). Trust the FRAMES.
\"\"\"{tr}\"\"\"
Automatic excitement score for the window (louder/more action = higher): {signal_score:.1f}

The streamer's reaction (laughing, shouting, swearing, going quiet) often marks a funny or exciting moment: when the
words or the excitement score suggest one, look in the frames for what caused it.
NOT clip-worthy (set feature_shown=false): ordinary driving, walking or standing with nothing special happening,
plain scenery, people only talking, menus, loading or black screens, a static image, a webcam or reaction, chat, text or
logos, a loud moment whose cause you cannot see.
Not allowed at all: mods/cheats. Cinematic scenes are fine when they show something interesting, but do not retell the
story (no plot spoilers). Glitches that look funny are fine to report as funny_moment.
When in doubt between "boring" and "decent", choose decent: it is better to offer a clip than to miss one.

Proof rule: "visible_evidence" must list 2-3 short, concrete things you actually SEE in the frames that prove the
moment (for example "the car launches off the ramp and lands on its roof"). If you cannot write two concrete items,
set feature_shown=false. "feature_shown" means "this window has a clip-worthy moment of any kind above".
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
 "feature_name": "3-7 words naming the moment, e.g. 'Realistic car damage' or 'Car launches off a ramp'",
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


def motion_level(images: list[bytes]) -> float | None:
    """Average picture change between neighbouring frames (0-255 scale). ~0 = a frozen picture / menu / loading
    screen. None when it cannot be measured (then the AI decides)."""
    try:
        import cv2
        import numpy as np
        grays = []
        for b in images:
            im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_GRAYSCALE)
            if im is None:
                return None
            grays.append(cv2.resize(im, (64, 36)).astype("float32"))
        if len(grays) < 2:
            return None
        return float(np.mean([np.abs(a - b).mean() for a, b in zip(grays, grays[1:])]))
    except Exception:
        return None


def dhash(img_bytes: bytes) -> int | None:
    """64-bit difference hash of a frame: two frames of the same scene give hashes with few differing bits."""
    try:
        import cv2
        import numpy as np
        im = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        sm = cv2.resize(im, (9, 8)).astype("int16")
        bits = (sm[:, 1:] > sm[:, :-1]).flatten()
        return int("".join("1" if b else "0" for b in bits), 2)
    except Exception:
        return None


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


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
        if km - so > 4.0 and eo - km >= clip_min * 0.3:
            so = max(so, min(km - 4.0, eo - clip_min))     # never open with a long slow build-up (but keep the minimum length)
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


PRESETS = {"loose": (45, 38), "normal": (58, 52), "strict": (70, 65)}


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
    mo = motion_level(images) if images else None
    if mo is not None and mo < float(cfg.get("judge.min_motion", 1.0)):
        j = normalize({}, window_len, float(cfg.get("video.clip_min")), float(cfg.get("video.clip_max")))
        j.update(accept=False, decision_reason="picture is almost frozen (menu / loading / standing still), skipped without AI",
                 motion=round(mo, 2))
        return j
    prompt = build_prompt(
        cfg.get("game.name"), cfg.get("game.topic_hint"), transcript, window_len, times,
        signal_score, float(cfg.get("video.clip_min")), float(cfg.get("video.clip_max")),
        hints=list(cfg.get("game.hints", []) or []),
    )
    raw = llm.json(prompt, system=SYSTEM, images=images or None)
    j = normalize(raw, window_len, float(cfg.get("video.clip_min")), float(cfg.get("video.clip_max")))
    ok, why = decide(j, cfg)
    j["accept"], j["decision_reason"] = ok, why
    if mo is not None:
        j["motion"] = round(mo, 2)
    if images:
        j["frame_hash"] = dhash(images[len(images) // 2])        # used to spot the same moment made twice
    return j


def feature_key(name: str) -> str:
    return " ".join(sorted(set(name.lower().split())))


def dedupe(items: list[tuple[int, dict]], max_clips: int, boost: dict | None = None) -> list[int]:
    """items = [(clip_id, judgement)] of accepted clips -> ids to keep (best per feature, top N).
    boost = {category: multiplier} from what did well on YouTube (empty until there is data)."""
    boost = boost or {}

    def sc(j: dict) -> float:
        return (j["interest"] + j["confidence"]) * float(boost.get(j.get("category"), 1.0))
    best: dict[str, tuple[int, dict]] = {}
    for cid, j in items:
        key = feature_key(j["feature_name"])
        if key not in best or sc(j) > sc(best[key][1]):
            best[key] = (cid, j)
    ranked = sorted(best.values(), key=lambda t: -sc(t[1]))
    return [cid for cid, _ in ranked[:max_clips]]


def same_moment(a: dict, b: dict, a_abs: tuple[float, float] | None = None, b_abs: tuple[float, float] | None = None) -> bool:
    """True when two accepted clips show the same moment: their time ranges in the same video overlap a lot, or the
    middle frames look almost identical and they describe the same thing."""
    if a_abs and b_abs:
        ov = min(a_abs[1], b_abs[1]) - max(a_abs[0], b_abs[0])
        if ov > 0.5 * min(a_abs[1] - a_abs[0], b_abs[1] - b_abs[0]):
            return True
    ha, hb = a.get("frame_hash"), b.get("frame_hash")
    if isinstance(ha, int) and isinstance(hb, int):
        d = hamming(ha, hb)
        if d <= 4:
            return True
        if d <= 10 and set(feature_key(a.get("feature_name", "")).split()) & set(feature_key(b.get("feature_name", "")).split()):
            return True
    return False
