"""Local speech-to-text (faster-whisper, CPU). Used for:
  * what the streamer says inside a candidate window  (context for the LLM judge)
  * word timings of our own voice-over               (caption timing)
If faster-whisper or its model cannot be loaded, we degrade gracefully (empty transcript)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

_MODEL = None
_MODEL_KEY = None


@dataclass
class Transcript:
    text: str = ""
    words: list[tuple[float, float, str]] = field(default_factory=list)   # (start, end, word)
    segments: list[tuple[float, float, str]] = field(default_factory=list)
    available: bool = True
    error: str = ""


def _model(cfg):
    global _MODEL, _MODEL_KEY
    key = (cfg.get("whisper.model"), cfg.get("whisper.device"), cfg.get("whisper.compute_type"))
    if _MODEL is None or _MODEL_KEY != key:
        from faster_whisper import WhisperModel  # imported lazily (heavy)
        _MODEL = WhisperModel(
            cfg.get("whisper.model"), device=cfg.get("whisper.device"),
            compute_type=cfg.get("whisper.compute_type"),
            cpu_threads=int(cfg.get("whisper.threads", 4)),
        )
        _MODEL_KEY = key
    return _MODEL


def transcribe_file(wav: str | Path, cfg, words: bool = False, offset: float = 0.0) -> Transcript:
    try:
        model = _model(cfg)
        lang = cfg.get("whisper.language") or None
        segs, _info = model.transcribe(
            str(wav), language=lang, beam_size=1, vad_filter=True,
            word_timestamps=words, condition_on_previous_text=False,
        )
        out = Transcript()
        texts = []
        for s in segs:
            texts.append(s.text.strip())
            out.segments.append((offset + s.start, offset + s.end, s.text.strip()))
            if words and getattr(s, "words", None):
                for w in s.words:
                    out.words.append((offset + w.start, offset + w.end, w.word.strip()))
        out.text = " ".join(t for t in texts if t)
        return out
    except Exception as e:  # model missing / no internet for first download / etc.
        return Transcript(available=False, error=f"{type(e).__name__}: {e}")


_KEYWORDS = re.compile(
    r"\b(wait|no way|look at|did you see|whoa|wow|oh my god|omg|holy|insane|crazy|new|detail|"
    r"feature|can you|you can now|they added|never seen|so cool|incredible|unreal)\b", re.I)


def keyword_hits(text: str) -> int:
    return len(_KEYWORDS.findall(text or ""))
