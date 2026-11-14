"""Voice-over: the LLM writes a short script (checked by code), Kokoro speaks it locally."""
from __future__ import annotations

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

def word_count(s: str) -> int:
    return len(re.findall(r"[A-Za-z0-9'’-]+", s))


def target_words(clip_dur: float, cfg) -> tuple[int, int]:
    lead = float(cfg.get("narration.lead_in", 0.4))
    speak = max(4.0, clip_dur - lead - 1.0)
    wps = float(cfg.get("narration.words_per_second", 2.4))
    hi = int(speak * wps)
    lo = int(hi * 0.6)
    return lo, hi


def validate_script(script: str, facts: str, lo: int, hi: int) -> list[str]:
    problems = []
    n = word_count(script)
    if n < lo:
        problems.append(f"too short: {n} words, need {lo}-{hi}")
    if n > hi:
        problems.append(f"too long: {n} words, need {lo}-{hi}")
    if FORBIDDEN.search(script):
        problems.append("contains a call-to-action / hashtag; remove it")
    if re.search(r"[\U0001F300-\U0001FAFF☀-➿]", script):
        problems.append("contains emoji; remove it")
    facts_nums = set(re.findall(r"\d+(?:[.,]\d+)?", facts))
    for num in re.findall(r"\d+(?:[.,]\d+)?", script):
        if num not in facts_nums:
            problems.append(f"number '{num}' is not in the given facts; remove or rephrase")
    return problems


def write_script(llm, cfg, judgement: dict, transcript: str, clip_dur: float) -> dict:
    lo, hi = target_words(clip_dur, cfg)
    facts = f"{judgement['feature_name']}. {judgement['what_happens']} {transcript}"
    problems: list[str] = []
    last: dict = {}
    for attempt in range(3):
        fix = ""
        if problems:
            fix = "\nYour previous script had problems, fix them: " + "; ".join(problems)
        prompt = f"""Write the voice-over for a {clip_dur:.0f}-second YouTube Short about {cfg.get('game.name')}.
The viewer sees only the gameplay clip (never the streamer). Language: {cfg.get('narration.language', 'English')}.

Feature: {judgement['feature_name']}
What is visible: {judgement['what_happens']}
Streamer's words (hints only, do not quote them): {transcript.strip() or '(none)'}

Rules:
- The first sentence is a hook of at most 12 words that makes people keep watching.
- Then explain what the feature is and why it is cool, in plain spoken language.
- Use ONLY the facts above. No invented numbers, dates, names, comparisons or release info.
- Do not say "I" or "we", do not mention the streamer, no emoji, no hashtags, no "like and subscribe".
- Between {lo} and {hi} words. Short sentences. It will be read aloud.{fix}

Return JSON: {{"script": "...", "on_screen_hook": "max 6 words, ALL CAPS"}}"""
        raw = llm.json(prompt, system=SYSTEM, temperature=0.6)
        script = re.sub(r"\s+", " ", str(raw.get("script", ""))).strip()
        last = {"script": script, "on_screen_hook": str(raw.get("on_screen_hook", "")).strip()[:60]}
        problems = validate_script(script, facts, lo, hi)
        if not problems:
            return last
    raise ValueError("could not get a valid script: " + "; ".join(problems))


# ------------------------------------------------------------------ TTS

@dataclass
class Narration:
    wav: Path
    duration: float
    words: list[tuple[float, float, str]]   # estimated (start, end, word)


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


def synthesize(cfg, text: str, out_wav: Path, max_duration: float | None = None) -> Narration:
    """Speak `text`; if it is longer than max_duration, speed it up (up to max_speedup)."""
    import soundfile as sf
    sentences = split_sentences(text)
    if not sentences:
        raise ValueError("empty script")
    gap = 0.18
    chunks, spans, t = [], [], 0.0
    sr_ref = None
    for s in sentences:
        samples, sr = _speak(cfg, s)
        sr_ref = sr_ref or sr
        dur = len(samples) / sr
        spans.append((s, t, t + dur))
        chunks.append(samples)
        chunks.append(np.zeros(int(gap * sr), dtype=np.float32))
        t += dur + gap
    audio = np.concatenate(chunks)
    raw = out_wav.with_name(out_wav.stem + "_raw.wav")
    sf.write(str(raw), audio, sr_ref)
    total = len(audio) / sr_ref
    speed = 1.0
    if max_duration and total > max_duration:
        speed = min(float(cfg.get("narration.max_speedup", 1.15)), total / max_duration)
    if speed > 1.001:
        apply_speed(raw, out_wav, speed)
        spans = [(s, a / speed, b / speed) for s, a, b in spans]
        total /= speed
    else:
        raw.replace(out_wav)
    if raw.exists():
        raw.unlink()
    return Narration(wav=out_wav, duration=total, words=estimate_word_times(spans))
