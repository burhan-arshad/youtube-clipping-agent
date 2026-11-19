"""Voice-over: the LLM writes a short script (checked by code), Kokoro speaks it locally."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ffutil import run

KOKORO_URLS = {
    "kokoro-v1.0.int8.onnx":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx",
    "voices-v1.0.bin":
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
}

FORBIDDEN = re.compile(
    r"(like and subscribe|subscribe|link in (the )?(bio|description)|smash that|hit the bell|"
    r"follow for more|#\w+)", re.I)

SYSTEM = (
    "You write tight, accurate voice-over scripts for YouTube Shorts about video games. "
    "You only use facts you are given. Reply with JSON only."
)


# ------------------------------------------------------------------ script

# Different openings / endings / titles so that not every Short sounds the same.
HOOK_STYLES = [
    "a curious question that names the game, e.g. \"Did you know {game} lets you do this?\"",
    "a surprise statement, e.g. \"{game} just did something I did not expect.\"",
    "a challenge to the viewer, e.g. \"Watch this {game} detail closely.\"",
    "a bold claim about the feature, e.g. \"This might be the smartest thing in {game}.\"",
    "a 'most players will miss this' opener, e.g. \"Here is a {game} detail most people will miss.\"",
    "a direct reaction, e.g. \"Wait, {game} can really do this?\"",
    "a mini-story opener, e.g. \"I did not expect this to happen in {game}.\"",
]
ENDING_STYLES = [
    "ask: \"Did you notice this before?\"",
    "ask: \"What is your opinion on this feature?\"",
    "ask: \"Would you want this in every open-world game?\"",
    "ask: \"Is this a small detail or a big deal?\"",
    "ask: \"Which detail surprised you most?\"",
    "ask: \"Do you think Rockstar nailed this one?\"",
    "ask: \"Be honest, did you expect this level of detail?\"",
    "ask: \"How would you rate this feature out of ten?\"",
]
TITLE_STYLES = [
    "a curiosity gap",
    "a bold claim",
    "a short question",
    "a surprising fact",
    "'Nobody talks about' / 'Most people missed' style",
    "an exclamation about the detail",
]
_RECENT: dict[str, list[int]] = {"hook": [], "end": [], "title": []}
# set by the pipeline from what worked on YouTube: {"hook": {style_index: weight}, "title": {...}}; empty = all equal
STYLE_WEIGHTS: dict[str, dict[int, float]] = {}
PLAYBOOK = ""            # learned from YouTube results (learn.py); set by the pipeline before a clip is built
_LAST_PICK: dict[str, int] = {}


def _pick(kind: str, options: list[str], rng) -> str:
    """Choice that avoids the last few picks, so consecutive clips differ. Styles that got more views are picked more
    often, but 25 % of the time any style is tried, so the channel keeps learning."""
    recent = _RECENT[kind]
    pool = [i for i in range(len(options)) if i not in recent[-3:]] or list(range(len(options)))
    w = STYLE_WEIGHTS.get(kind) or {}
    if w and rng.random() > 0.25:
        i = rng.choices(pool, weights=[max(0.05, float(w.get(k, 1.0))) for k in pool])[0]
    else:
        i = rng.choice(pool)
    recent.append(i)
    del recent[:-6]
    _LAST_PICK[kind] = i
    return options[i]


def _norm_game(t: str) -> str:
    return re.sub(r"[^a-z0-9]", "", t.lower()).replace("gtavi", "gta6")


def mentions_game(text: str, game: str) -> bool:
    return bool(game) and _norm_game(game) in _norm_game(text)


def _sentences(script: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?])\s+", script.strip()) if x]

def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’-]+", s))


def target_words(clip_dur: float, cfg) -> tuple[int, int]:
    lead = float(cfg.get("narration.lead_in", 0.4))
    speak = max(4.0, clip_dur - lead - 1.0)
    wps = float(cfg.get("narration.words_per_second", 2.5))
    hi = int(speak * wps)
    lo = int(hi * 0.85)
    return lo, hi


def validate_script(script: str, facts: str, lo: int, hi: int, game: str = "", need_question: bool = False) -> list[str]:
    problems = []
    n = word_count(script)
    if n < lo:
        problems.append(f"too short: {n} words, need {lo}-{hi}")
    if n > hi:
        problems.append(f"too long: {n} words, need {lo}-{hi}")
    sents = _sentences(script)
    if game and sents and not mentions_game(" ".join(sents[:2]), game):
        problems.append(f"the first sentence must mention the game name '{game}'")
    if need_question and sents and not sents[-1].rstrip().endswith("?"):
        problems.append("the last sentence must be a question to the viewer (end with ?)")
    if FORBIDDEN.search(script):
        problems.append("contains a call-to-action / hashtag; remove it")
    if re.search(r"[\U0001F300-\U0001FAFF☀-➿]", script):
        problems.append("contains emoji; remove it")
    facts_nums = set(re.findall(r"\d+(?:[.,]\d+)?", facts)) | set(re.findall(r"\d+(?:[.,]\d+)?", game))
    for num in re.findall(r"\d+(?:[.,]\d+)?", script):
        if num not in facts_nums:
            problems.append(f"number '{num}' is not in the given facts; remove or rephrase")
    return problems


def _fit_length(script: str, lo: int, hi: int) -> str | None:
    """Trim middle sentences (keeps the hook and the closing question) when the script is too long."""
    sents = re.split(r"(?<=[.!?])\s+", script.strip())
    while word_count(" ".join(sents)) > hi and len(sents) > 3:
        sents.pop(len(sents) // 2)
    out = " ".join(sents)
    return out if lo * 0.9 <= word_count(out) <= hi else None


def describe_timeline(llm, cfg, judgement: dict, images, times: list[float], clip_dur: float) -> dict | None:
    """Look at frames of the FINAL clip and say what is visible when, and when the feature happens.
    Returns {"events": [(t, what)], "moment": float|None} or None if the AI could not do it."""
    if not images or len(images) != len(times):
        return None
    tl = ", ".join(f"{t:.1f}s" for t in times)
    prompt = f"""These {len(images)} frames come from a {clip_dur:.0f}-second gameplay clip of {cfg.get('game.name')}, in time
order, at {tl} from the start of the clip. The clip is meant to show: {judgement.get('feature_name', '')}.
For EACH frame say what is visible (max 14 words, only what you see). Then give "moment": the second (from the clip
start) at which the feature actually happens or first becomes visible (for a crash: the impact), or null if it is
visible from the start. Be exact: compare neighbouring frames to see when things change.
Return JSON only: {{"frames": [{{"t": seconds, "what": "..."}}, ...], "moment": number or null}}"""
    from .llm import LLMError
    try:
        raw = llm.json(prompt, system=SYSTEM, images=images, temperature=0.1)
    except LLMError:
        raise
    except Exception:
        return None
    ev = []
    for f in raw.get("frames") or []:
        try:
            ev.append((round(float(f["t"]), 1), str(f["what"]).strip()[:120]))
        except Exception:
            continue
    if len(ev) < 2:
        return None
    try:
        mom = None if raw.get("moment") in (None, "", "null") else max(0.0, min(float(raw["moment"]), clip_dur))
    except (TypeError, ValueError):
        mom = None
    return {"events": ev, "moment": mom}


KIND_OF = {"funny_moment": "funny", "chaotic_action": "chaos", "epic_stunt": "epic", "interesting_discovery": "discovery",
           "dramatic_moment": "dramatic"}
WRITER_STYLE = {
    "funny": "This is a FUNNY moment. Use comic timing: set it up straight-faced, then land the punchline right when it happens, "
             "like a friend laughing at what just happened. Playful, a little cheeky, never mean. One quick joke or comparison is enough.",
    "chaos": "This is CHAOTIC ACTION. High energy, short punchy lines, escalating excitement, like a hyped commentator. Name each thing "
             "that goes wrong in order.",
    "epic": "This is an EPIC / skilful moment. Build tension, then react with real awe at the payoff. Make the viewer feel how hard or "
            "lucky it was.",
    "discovery": "This is an interesting DISCOVERY. Sound curious and a bit amazed ('wait, look at this'), point at the detail, "
                 "and say why it is cool or surprising.",
    "dramatic": "This is a DRAMATIC moment. Lower, tense delivery, short lines, a pause before the reveal, then let the result land.",
    "feature": "This shows a game FEATURE. Be informative and enthusiastic: say exactly what the game does here and the one detail "
               "casual viewers would miss.",
}
EXAMPLES = {
    "funny": 'Example of the RHYTHM only (never copy the words): "Okay, nothing suspicious here." ... "Wait. Is that a gator... shopping?" '
             '... "Nobody told the cashier. Honestly, same." ... "Would you check out next to him?"',
    "chaos": 'Example of the RHYTHM only: "Three cars. One red light. Zero chance." ... "Boom! That one flipped, that one is on fire!" '
             '... "And the cops just joined the party." ... "Could you survive this?"',
    "epic": 'Example of the RHYTHM only: "This jump looks impossible." ... "Speed is up... he is committing..." ... "He stuck it! '
            'Right on the edge!" ... "Could you land that?"',
    "discovery": 'Example of the RHYTHM only: "Look at the water, not the boat." ... "See how it ripples around the legs?" '
                 '... "Small detail, but it makes everything feel alive." ... "What else have you spotted?"',
    "dramatic": 'Example of the RHYTHM only: "Everything is quiet." ... "Too quiet." ... "And there it is." ... "Would you have run?"',
    "feature": 'Example of the RHYTHM only: "Watch what the car does next." ... "The bumper actually crumples, panel by panel." '
               '... "Most games just swap in a dented model. This one bends." ... "Do you want this in every game?"',
}
TONE = {
    "funny": "playful and amused, with natural laughter in the voice, comic timing and a cheeky punchline",
    "chaos": "fast, hyped and breathless, rising excitement like a live commentator",
    "epic": "awed and intense, building up then exploding with excitement at the payoff",
    "discovery": "curious and amazed, a bit hushed, then delighted",
    "dramatic": "tense and low, slower, with a pause before the reveal",
    "feature": "enthusiastic and clear, like a friend showing off something cool",
}


def moment_kind(j: dict) -> str:
    return KIND_OF.get(str(j.get("category", "")), "feature")


def tone_for(j: dict, cfg) -> str:
    """Delivery instruction for the voice: the channel style plus how THIS moment should sound."""
    base = str(cfg.get("narration.gemini_style", "") or "").strip()
    return (base + "; " if base else "") + "for this clip: " + TONE[moment_kind(j)]


def est_seconds(text: str, cfg) -> float:
    from .tts import strip_tags
    return word_count(strip_tags(text)) / float(cfg.get("narration.words_per_second", 2.4)) + 0.2


def repair_beats(beats: list[dict], cfg) -> list[dict]:
    """Instead of sending the AI back to fix timing, move a beat later if the line before it is still speaking."""
    out, prev_end = [], 0.0
    for b in sorted(beats, key=lambda x: x["at"]):
        at = max(b["at"], prev_end + 0.1)
        out.append({"at": round(at, 2), "text": b["text"]})
        prev_end = at + est_seconds(b["text"], cfg)
    return out


def validate_beats(beats: list[dict], clip_dur: float, cfg, lead: float = 0.4) -> list[str]:
    """Only things the code cannot repair. (Overlaps are repaired; a voice that is a bit long is sped up or the clip extended.)"""
    p = []
    if not (2 <= len(beats) <= 10):
        p.append(f"use 4 to 7 beats (you wrote {len(beats)})")
    return p


def _parse_beats(raw) -> list[dict]:
    out = []
    for b in raw or []:
        try:
            t = re.sub(r"\s+", " ", str(b["text"])).strip()
            if t:
                out.append({"at": max(0.0, float(b["at"])), "text": t})
        except Exception:
            continue
    out.sort(key=lambda x: x["at"])
    return out


def _ask(llm, prompt: str, quality: bool = False, **kw) -> dict:
    """llm.json that prefers the smarter models for creative writing (falls back for simple/fake clients)."""
    if quality:
        try:
            return llm.json(prompt, quality=True, **kw)
        except TypeError:
            pass
    return llm.json(prompt, **kw)


def write_script(llm, cfg, judgement: dict, transcript: str, clip_dur: float, images=None,
                 feedback: str = "", timeline: dict | None = None) -> dict:
    """Writes the script (see _write_script_inner), then an optional punch-up pass and remembers the styles used."""
    res = _write_script_inner(llm, cfg, judgement, transcript, clip_dur, images, feedback, timeline)
    styles = {k: v for k, v in _LAST_PICK.items()}
    if not feedback and res.get("beats"):                       # only on the first try: later tries are fixes
        res = punch_up(llm, cfg, judgement, res, timeline, clip_dur)
    res["styles"] = styles
    return res


def punch_up(llm, cfg, judgement: dict, result: dict, timeline: dict | None, clip_dur: float) -> dict:
    """Second pass: same facts and timing, but funnier / more emotional / more vivid. If anything about the new
    version breaks a rule, the original is kept, so this can only make a script better, never worse."""
    if not cfg.get("narration.punch_up", True):
        return result
    beats = result.get("beats") or []
    if not (2 <= len(beats) <= 10):
        return result
    game = str(cfg.get("game.name", "") or "")
    lo, hi = target_words(clip_dur, cfg)
    kind = moment_kind(judgement)
    tl = ""
    if timeline and timeline.get("events"):
        tl = "What the viewer sees and when:\n" + "\n".join(f"  {t:.1f}s: {w}" for t, w in timeline["events"]) + "\n"
    evidence = "; ".join(judgement.get("visible_evidence", []))
    facts = f"{judgement.get('feature_name', '')}. {judgement.get('what_happens', '')} {evidence} " + " ".join(
        w for _t, w in (timeline or {}).get("events", []))
    cur = "\n".join(f'  {{"at": {b["at"]}, "text": {json.dumps(b["text"])}}}' for b in beats)
    prompt = f"""Here is a timed voice-over for a {clip_dur:.0f}-second Short about {game}.
{tl}
Current beats:
{cur}

Rewrite it so it sounds like a REAL person reacting live: more emotion, sharper wording, a funnier or more vivid
comparison where it fits, one small human reaction (a laugh, "no way", a gasp as words). {WRITER_STYLE[kind]}
HARD RULES: keep exactly the same number of beats and the same "at" times; keep every fact (no new facts, numbers or
names); a line may only describe what is already visible at its "at" time; keep the hook naming "{game}" and the
closing question; total {lo} to {hi} words; no emoji, no hashtags, no "I"/"we" about the streamer, no
like-and-subscribe.
Return JSON only: {{"beats": [{{"at": 0.5, "text": "..."}}, ...]}}"""
    try:
        raw = _ask(llm, prompt, quality=True, system=SYSTEM, temperature=0.8)
    except Exception:
        return result                                              # quota / network: keep the first version
    texts = []
    for b in raw.get("beats") or []:                               # keep the order the model wrote them in
        try:
            t = re.sub(r"\s+", " ", str(b["text"])).strip()
        except Exception:
            t = ""
        if t:
            texts.append(t)
    if len(texts) != len(beats):
        return result
    new = [{"at": o["at"], "text": t} for o, t in zip(beats, texts)]       # timing stays exactly as verified
    from .tts import strip_tags
    if not any("<" in b["text"] for b in beats):
        new = [{"at": b["at"], "text": strip_tags(b["text"])} for b in new]
    script = " ".join(strip_tags(b["text"]) for b in new)
    if validate_script(script, facts, int(lo * 0.7), int(hi * 1.3), game, False):
        return result
    return {**result, "script": script, "beats": new, "punched": True}


def _write_script_inner(llm, cfg, judgement: dict, transcript: str, clip_dur: float, images=None,
                 feedback: str = "", timeline: dict | None = None) -> dict:
    """transcript is ignored on purpose: streamers talk about unrelated things, the script must come from the picture."""
    lo, hi = target_words(clip_dur, cfg)
    evidence = "; ".join(judgement.get("visible_evidence", []))
    facts = f"{judgement['feature_name']}. {judgement['what_happens']} {evidence}"
    problems: list[str] = []
    last: dict = {}
    candidates: list[tuple[int, dict]] = []
    game = str(cfg.get("game.name", "") or "")
    import random
    rng = random.Random()
    hook_style = _pick("hook", HOOK_STYLES, rng).replace("{game}", game or "the game")
    end_style = _pick("end", ENDING_STYLES, rng)
    title_style = _pick("title", TITLE_STYLES, rng)
    if timeline and timeline.get("events"):
        return _write_beats(llm, cfg, judgement, clip_dur, images, feedback, timeline, facts, lo, hi,
                            hook_style, end_style, title_style)
    for attempt in range(4):
        fix = ""
        if problems or feedback:
            fix = "\nYour previous script had problems, fix them: " + "; ".join(problems + ([feedback] if feedback else []))
        prompt = f"""Write the voice-over for a {clip_dur:.0f}-second YouTube Short about {cfg.get('game.name')}.
The viewer sees only the gameplay clip (never the streamer). Language: {cfg.get('narration.language', 'English')}.

Feature: {judgement['feature_name']}
What is visible: {judgement['what_happens']}
Concrete things seen in the clip: {evidence or '(see the images)'}
{"The attached images are frames of the clip in time order. Describe ONLY what you can see in them." if images else ""}

Structure (it will be read aloud, so write it like a confident, excited friend talking):
1. HOOK (first sentence, at most 14 words): it MUST name the game "{game}" so viewers know what they are watching.
   Style for this clip: {hook_style}. It must be true to what is visible. Do not reuse the example word for word.
2. BODY (2-4 short sentences): what exactly happens, one concrete visible detail per sentence, and why a fan
   should care. Vary the rhythm. Plain words, no filler, no repeating the hook.
3. ENDING (last sentence, at most 14 words, must end with a question mark): a short question that invites
   comments. Style for this clip: {end_style}. You may rephrase it.
Rules:
- The script must fill the clip: it has to run for the whole {clip_dur:.0f} seconds, so write {lo} to {hi} words.
- Use ONLY the facts above. No invented numbers, dates, names, comparisons or release info.
- Do not say "I" or "we", do not mention the streamer, no emoji, no hashtags, no "like and subscribe".
- Short sentences.{fix}

Also write a "title": a catchy on-screen heading for the top of the video and for YouTube, 4 to 9 words,
mentioning "{game}", style: {title_style}. Truthful, no emoji, no hashtags, no ALL CAPS shouting.

Return JSON: {{"script": "...", "title": "..."}}"""
        raw = _ask(llm, prompt, quality=True, system=SYSTEM, images=images or None, temperature=0.6)
        script = re.sub(r"\s+", " ", str(raw.get("script", ""))).strip()
        title = re.sub(r"\s+", " ", str(raw.get("title") or raw.get("on_screen_hook") or "")).strip().strip('"')
        if title and game and not mentions_game(title, game) and len(title) + len(game) + 2 <= 70:
            title = f"{game}: {title}"
        last = {"script": script, "title": title[:70], "on_screen_hook": title[:70]}
        problems = validate_script(script, facts, lo, hi, game, True)
        if not problems:
            return last
        n = word_count(script)
        if n > hi:
            problems.append(f"it has {n} words, cut it to at most {hi} (remove a whole sentence)")
        elif n < lo:
            problems.append(f"it has {n} words, add more detail to reach at least {lo}")
        if all(p.startswith(("too long", "too short", "it has")) for p in problems):
            candidates.append((abs(n - (lo + hi) // 2), dict(last)))
    # Last resort: a script that is only a little off in length is repaired instead of failing the clip.
    for _, cand in sorted(candidates, key=lambda c: c[0]):
        fixed = _fit_length(cand["script"], lo, hi)
        if fixed and not validate_script(fixed, facts, lo, hi):
            return {**cand, "script": fixed}
    raise ValueError("could not get a valid script: " + "; ".join(problems))


def _write_beats(llm, cfg, judgement, clip_dur, images, feedback, timeline, facts, lo, hi,
                 hook_style, end_style, title_style) -> dict:
    """Script as timed beats: every line is tied to a second of the clip, so the voice never talks about something
    the picture has not shown yet."""
    game = str(cfg.get("game.name", "") or "")
    lead = float(cfg.get("narration.lead_in", 0.4))
    ev = timeline["events"]
    mom = timeline.get("moment")
    tl_txt = "\n".join(f"  {t:>5.1f}s  {w}" for t, w in ev)
    tags_ok = str(cfg.get("narration.engine", "kokoro")) == "gemini"
    _k = moment_kind(judgement)
    style_block = WRITER_STYLE[_k] + "\n" + EXAMPLES.get(_k, "")
    tag_rule = ('- You may use at most 2 delivery tags in the whole script, only where a real person would: <short pause> '
                'before a reveal, <breath> before something big, <laugh> after something funny. Put capitals on at most 2 '
                'words for stress.' if tags_ok else "- No tags, no capital-letter words.")
    problems: list[str] = []
    last: dict = {}
    for attempt in range(4):
        fix = ""
        if problems or feedback:
            fix = "\nYour previous version had problems, fix every one: " + "; ".join(problems + ([feedback] if feedback else []))
        prompt = f"""Write a timed voice-over for a {clip_dur:.0f}-second YouTube Short about {game}. The viewer sees only the
gameplay (never the streamer). Language: {cfg.get('narration.language', 'English')}.

Feature: {judgement['feature_name']}
What happens: {judgement['what_happens']}

WHAT THE VIEWER SEES AND WHEN (seconds from the start of the clip):
{tl_txt}
{f"The feature itself happens at about {mom:.1f}s." if mom is not None else ""}

Write BEATS. Each beat is {{"at": second the line starts, "text": 1-2 short spoken sentences}}.
THE GOLDEN RULE: a line may only describe what the viewer can already see at its "at" time. Before the feature happens
you can only set the scene, build curiosity and tell people what to watch for ("Watch the back of that car...").
Right at the moment ({f"{mom:.1f}s" if mom is not None else "when it appears"}) react to it, then explain it and why it matters.
Never say something already happened before the picture shows it.

{style_block}
{PLAYBOOK}
Voice and style: a real person, hyped but not fake, like a friend who just saw something wild. Be specific and
vivid (what exactly bends, breaks, moves, reacts), add one detail a casual viewer would miss, and show emotion
(surprise, respect, amusement). Vary sentence length. Show, do not tell: name the exact thing the viewer sees instead of \"this is crazy\". No generic filler like "amazing" without saying why.
Structure: 1 HOOK beat at 0.3s-1s (max 14 words, names "{game}", style: {hook_style}; it may tease, not reveal),
2-4 BODY beats (build-up, reaction at the moment, explanation, one extra detail), 1 ENDING beat (a question to the viewer,
max 14 words, style: {end_style}).
VALUE: the viewer must get something from these 20 seconds: a detail only a careful player notices, what it
implies (cause and effect, what it says about the world or the story), or a "would you survive / would you do this"
stake. The HOOK must open a curiosity gap in the first 2 seconds (a question, a contradiction, "watch the ...").
Never narrate walking, menus or idle talking as if it were exciting; stay with what really happens.
Timing: speaking speed is about {cfg.get('narration.words_per_second', 2.4)} words per second, so a 10-word line takes about 4 seconds; the next
beat must start after the previous one has finished. Together all lines have {lo} to {hi} words and the last one ends
before {clip_dur - 0.8:.1f}s.
Rules: use ONLY the facts and pictures above, no invented numbers/names/dates/release info; no "I"/"we"; do not mention the
streamer; no emoji, no hashtags, no like-and-subscribe.
{tag_rule}{fix}

Also write "title": an on-screen heading (4 to 9 words, mentions "{game}", style: {title_style}), truthful, no emoji, no hashtags.
Return JSON: {{"beats": [{{"at": 0.5, "text": "..."}}, ...], "title": "..."}}"""
        raw = _ask(llm, prompt, quality=True, system=SYSTEM, images=images or None, temperature=0.7)
        beats = repair_beats(_parse_beats(raw.get("beats")), cfg)
        title = re.sub(r"\s+", " ", str(raw.get("title") or "")).strip().strip('"')
        if title and game and not mentions_game(title, game) and len(title) + len(game) + 2 <= 70:
            title = f"{game}: {title}"
        from .tts import strip_tags
        script = " ".join(strip_tags(b["text"]) for b in beats)
        last = {"script": script, "title": title[:70], "on_screen_hook": title[:70], "beats": beats}
        # a little shorter or longer than the target is fine: the voice is sped up a bit or the clip is extended
        problems = validate_script(script, facts, int(lo * 0.7), int(hi * 1.3), game, False) + validate_beats(beats, clip_dur, cfg, lead)
        if not tags_ok:
            beats = [{"at": b["at"], "text": strip_tags(b["text"])} for b in beats]
            last["beats"] = beats
        if not problems:
            return last
    # last try still has small problems: use it anyway unless it breaks a real rule (numbers, call-to-action, emoji)
    hard = [p for p in problems if p.startswith(("number", "contains")) or ("too short" in p and word_count(last.get("script", "")) < 15)]
    if last and last.get("script") and not hard:
        return last
    raise ValueError("could not get a valid script: " + "; ".join(problems))


def verify_script(llm, cfg, judgement: dict, script: str, images, times: list[float] | None = None,
                  beats: list[dict] | None = None) -> dict:
    """Second look: does the voice-over match the pictures (and their timing), and is a streamer / webcam /
    overlay visible?"""
    timing = ""
    if times and len(times) == len(images or []):
        timing += "The frames are at: " + ", ".join(f"{t:.1f}s" for t in times) + " from the clip start.\n"
    if beats:
        timing += ("Each line is SPOKEN at the time shown. A line that describes something the frames do not show yet at\n"
                   "that time is wrong (for example talking about a crash before it happens). Lines:\n"
                   + "\n".join(f"  at {b['at']:.1f}s: {b['text']}" for b in beats) + "\n")
    prompt = f"""You are a strict quality checker for a gaming Short. The images are frames of the clip in time order.
{timing}Feature the clip claims to show: {judgement['feature_name']}
Voice-over text:
\"\"\"{script}\"\"\"

Does the voice-over match what is visible? Every concrete claim must be visible in the frames; the feature must be
really shown. Vague hype that is not wrong is fine, specific claims that are not visible are not.
Also say whether any frame shows a real person (webcam/streamer), a chat window, a donation/subscriber alert or
text/logos added by the streamer (game characters do not count).
Return JSON: {{"match": 0-100, "real_person_or_overlay_visible": true/false, "problems": "what is wrong, or empty"}}"""
    raw = llm.json(prompt, system=SYSTEM, images=images or None, temperature=0.1)
    try:
        match = float(raw.get("match", 0))
    except (TypeError, ValueError):
        match = 0.0
    return {"match": match, "overlay": bool(raw.get("real_person_or_overlay_visible")),
            "problems": str(raw.get("problems", "")).strip()[:300]}


# ------------------------------------------------------------------ TTS

@dataclass
class Narration:
    wav: Path
    duration: float
    words: list[tuple[float, float, str]]   # (start, end, word); measured for cloud voices, estimated otherwise
    aligned: bool = False                    # True when `words` were measured from the audio itself


def _download(url: str, dest: Path) -> None:
    import requests
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if tmp.stat().st_size < 1_000_000:
        tmp.unlink()
        raise RuntimeError(f"download of {url} looks broken")
    tmp.replace(dest)


def kokoro_files(cfg) -> tuple[Path, Path]:
    d = cfg.path("paths.models") / "kokoro"
    paths = []
    for name, url in KOKORO_URLS.items():
        p = d / name
        if not p.is_file():
            print(f"[tts] downloading {name} (one time) ...")
            _download(url, p)
        paths.append(p)
    return paths[0], paths[1]


_KOKORO = None


def _kokoro(cfg):
    global _KOKORO
    if _KOKORO is None:
        from kokoro_onnx import Kokoro
        model, voices = kokoro_files(cfg)
        _KOKORO = Kokoro(str(model), str(voices))
    return _KOKORO


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _speak(cfg, sentence: str) -> tuple[np.ndarray, int]:
    engine = cfg.get("narration.engine", "kokoro")
    if engine == "dummy":  # tests only: a quiet tone, length proportional to the text
        sr = 24000
        dur = max(0.4, 0.06 * len(sentence))
        t = np.arange(int(dur * sr)) / sr
        return (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32), sr
    k = _kokoro(cfg)
    samples, sr = k.create(sentence, voice=cfg.get("narration.voice"),
                           speed=float(cfg.get("narration.speed", 1.0)),
                           lang=cfg.get("narration.lang", "en-us"))
    return np.asarray(samples, dtype=np.float32), int(sr)


def estimate_word_times(sentence_spans: list[tuple[str, float, float]]) -> list[tuple[float, float, str]]:
    out = []
    for text, s, e in sentence_spans:
        words = text.split()
        if not words:
            continue
        weights = np.array([len(w) + 2 for w in words], dtype=float)
        edges = np.concatenate([[0], np.cumsum(weights)]) / weights.sum()
        for i, w in enumerate(words):
            out.append((s + (e - s) * edges[i], s + (e - s) * edges[i + 1], w))
    return out


def apply_speed(wav_in: Path, wav_out: Path, speed: float) -> None:
    run(["ffmpeg", "-y", "-v", "error", "-i", str(wav_in), "-filter:a", f"atempo={speed:.4f}",
         str(wav_out)])


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf
    sf.write(str(path), audio, sr)


def _finish(audio: np.ndarray, sr: int, spans: list[tuple[str, float, float]], out_wav: Path,
            max_duration: float | None, cfg, aligned_words=None) -> Narration:
    """Write the wav; if it is longer than max_duration speed it up a little (up to narration.max_speedup)."""
    raw = out_wav.with_name(out_wav.stem + "_raw.wav")
    _write_wav(raw, audio, sr)
    total = len(audio) / sr
    speed = 1.0
    if max_duration and total > max_duration:
        speed = min(float(cfg.get("narration.max_speedup", 1.15)), total / max_duration)
    if speed > 1.001:
        apply_speed(raw, out_wav, speed)
        spans = [(s, a / speed, b / speed) for s, a, b in spans]
        total /= speed
        if aligned_words:
            aligned_words = [(a / speed, b / speed, w) for a, b, w in aligned_words]
    else:
        raw.replace(out_wav)
    if raw.exists():
        raw.unlink()
    return Narration(wav=out_wav, duration=total, words=aligned_words or estimate_word_times(spans),
                     aligned=bool(aligned_words))


def _place(chunks: list[tuple[np.ndarray, float]], sr: int, lead: float, gap: float = 0.18):
    """chunks = [(samples, wanted_start_in_clip_seconds)] -> (audio, start_seconds_in_audio per chunk).
    A chunk starts at its wanted time (minus the lead-in the renderer adds) but never before the previous one ended."""
    out, starts, cursor = [], [], 0.0
    for samples, want in chunks:
        t = max(cursor, (want - lead) if want is not None else cursor)
        if t > cursor:
            out.append(np.zeros(int((t - cursor) * sr), dtype=np.float32))
        starts.append(t)
        out.append(samples)
        cursor = t + len(samples) / sr + gap
        out.append(np.zeros(int(gap * sr), dtype=np.float32))
    return (np.concatenate(out) if out else np.zeros(1, dtype=np.float32)), starts


def _synth_local(cfg, text: str, out_wav: Path, max_duration, beats, lead) -> Narration:
    """Offline voice, one sentence at a time (so each line can be placed on its moment)."""
    from .tts import strip_tags
    groups = ([(strip_tags(b["text"]), float(b["at"])) for b in beats] if beats
              else [(strip_tags(text), None)])
    sentences_per = [split_sentences(g) for g, _ in groups]
    if not any(sentences_per):
        raise ValueError("empty script")
    chunks, spans_src, sr_ref = [], [], None
    for (g, at), sents in zip(groups, sentences_per):
        first = True
        for sent in sents:
            samples, sr = _speak(cfg, sent)
            sr_ref = sr_ref or sr
            chunks.append((samples, at if first else None))
            spans_src.append(sent)
            first = False
    audio, starts = _place(chunks, sr_ref, lead)
    spans = [(sent, st, st + len(smp) / sr_ref) for sent, st, (smp, _) in zip(spans_src, starts, chunks)]
    return _finish(audio, sr_ref, spans, out_wav, max_duration, cfg)


def _norm_tok(w: str) -> str:
    return re.sub(r"[^a-z0-9]", "", w.lower())


def align_words(script_words: list[str], heard: list[tuple[float, float, str]]):
    """Match the words of the script to the words Whisper heard -> (start, end) for every script word, or None."""
    import difflib
    if not heard or not script_words:
        return None
    a = [_norm_tok(w) for w in script_words]
    b = [_norm_tok(w[2]) for w in heard]
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    pos: list[tuple[float, float] | None] = [None] * len(a)
    hits = 0
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            pos[blk.a + k] = (heard[blk.b + k][0], heard[blk.b + k][1])
            hits += 1
    if hits < max(3, int(0.6 * len(a))):
        return None
    i = 0
    while i < len(pos):                       # fill the gaps between matched words
        if pos[i] is None:
            j = i
            while j < len(pos) and pos[j] is None:
                j += 1
            left = pos[i - 1][1] if i > 0 else 0.0
            right = pos[j][0] if j < len(pos) else heard[-1][1]
            n = j - i
            for k in range(n):
                pos[i + k] = (left + (right - left) * k / n, left + (right - left) * (k + 1) / n)
            i = j
        else:
            i += 1
    return pos


def _synth_cloud(cfg, engine: str, text: str, out_wav: Path, max_duration, beats, lead, tone: str | None = None) -> Narration:
    from . import tts
    from .transcribe import transcribe_file
    spoken = " ".join(b["text"] for b in beats) if beats else text
    if engine == "gemini":
        audio, sr = tts.gemini_speak(cfg, spoken, style=tone)
    else:
        audio, sr = tts.eleven_speak(cfg, tts.strip_tags(spoken))
    clean = tts.strip_tags(spoken)
    words = clean.split()
    raw = out_wav.with_name(out_wav.stem + "_cloud.wav")
    _write_wav(raw, audio, sr)
    heard = transcribe_file(raw, cfg, words=True)
    pos = align_words(words, heard.words) if heard.available and heard.words else None
    try:
        raw.unlink()
    except Exception:
        pass
    total = len(audio) / sr
    if pos is None:                                   # no timings: estimate and keep the natural pacing
        spans = [(clean, 0.0, total)]
        return _finish(audio, sr, spans, out_wav, max_duration, cfg)
    if not beats:
        wl = [(a, b, w) for (a, b), w in zip(pos, words)]
        return _finish(audio, sr, [(clean, 0.0, total)], out_wav, max_duration, cfg, aligned_words=wl)
    # cut the single recording into the beats and put every beat on its moment
    counts = [len(tts.strip_tags(b["text"]).split()) for b in beats]
    idx, k = [], 0
    for c in counts:
        idx.append((k, k + c))
        k += c
    cuts = []
    for n, (i0, i1) in enumerate(idx):
        if i1 <= i0:
            cuts.append((0.0, 0.0))
            continue
        t0 = pos[i0][0] - 0.06
        t1 = pos[i1 - 1][1] + 0.15
        cuts.append((max(0.0, t0), min(total, t1)))
    for n in range(len(cuts) - 1):                    # never overlap: split the pause between two beats in the middle
        if cuts[n][1] > cuts[n + 1][0]:
            mid = (cuts[n][1] + cuts[n + 1][0]) / 2
            cuts[n] = (cuts[n][0], mid)
            cuts[n + 1] = (mid, cuts[n + 1][1])
    chunks = []
    for (t0, t1), b in zip(cuts, beats):
        seg = audio[int(t0 * sr): int(t1 * sr)].astype(np.float32)
        chunks.append((seg, float(b["at"])))
    out, starts = _place(chunks, sr, lead, gap=0.05)
    new_words: list[tuple[float, float, str]] = []
    for (i0, i1), (t0, _t1), st in zip(idx, cuts, starts):
        for w_i in range(i0, i1):
            a, b = pos[w_i]
            new_words.append((max(0.0, a - t0) + st, max(0.0, b - t0) + st, words[w_i]))
    return _finish(out, sr, [(clean, 0.0, len(out) / sr)], out_wav, max_duration, cfg, aligned_words=new_words)


def synthesize(cfg, text: str, out_wav: Path, max_duration: float | None = None,
               beats: list[dict] | None = None, lead: float = 0.0, tone: str | None = None) -> Narration:
    """Speak the script. With `beats` ([{at, text}], `at` = second of the clip) every line is placed on its moment.
    Cloud voices (Gemini / ElevenLabs) fall back to the offline voice if they fail, so a Short is never lost."""
    engine = str(cfg.get("narration.engine", "kokoro"))
    if engine in ("gemini", "elevenlabs"):
        try:
            return _synth_cloud(cfg, engine, text, out_wav, max_duration, beats, lead, tone)
        except Exception as e:                           # noqa: BLE001
            print(f"[tts] {engine} voice failed ({str(e)[:160]}); using the offline voice for this Short")
    return _synth_local(cfg, text, out_wav, max_duration, beats, lead)
