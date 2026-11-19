"""Voices. Order of preference is set in Settings (narration.engine):

  gemini      Gemini TTS with your normal Gemini key. Free tier ~10 requests/day per model, so ONE request is made
              per Short (the whole script). Expressive: you give a delivery style and the script may contain
              tags like <short pause> or <laugh>. Hundreds of "Extended Voice Library" voices exist, so you can
              pick one that few channels use.
  elevenlabs  Optional. Needs ELEVENLABS_API_KEY. The free plan is small (10k characters a month), is meant for
              non-commercial use and may not allow API use at all, so it is only a bonus.
  kokoro      Offline, unlimited, always the last fallback.
"""
from __future__ import annotations

import base64
import io
import os
import re
import time

import numpy as np
import requests

from .llm_limits import limits_for
from .quota import next_quota_reset

GEMINI_MODELS = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-flash-tts"]

PREBUILT = {
    "Zephyr": "Bright", "Puck": "Upbeat", "Charon": "Informative", "Kore": "Firm", "Fenrir": "Excitable",
    "Leda": "Youthful", "Orus": "Firm", "Aoede": "Breezy", "Callirrhoe": "Easy-going", "Autonoe": "Bright",
    "Enceladus": "Breathy", "Iapetus": "Clear", "Umbriel": "Easy-going", "Algieba": "Smooth", "Despina": "Smooth",
    "Erinome": "Clear", "Algenib": "Gravelly", "Rasalgethi": "Informative", "Laomedeia": "Upbeat",
    "Achernar": "Soft", "Alnilam": "Firm", "Schedar": "Even", "Gacrux": "Mature", "Pulcherrima": "Forward",
    "Achird": "Friendly", "Zubenelgenubi": "Casual", "Vindemiatrix": "Gentle", "Sadachbia": "Lively",
    "Sadaltager": "Knowledgeable", "Sulafat": "Warm",
}
TAG = re.compile(r"<\s*(short pause|long pause|breath|sigh|laugh|cough)\s*>", re.I)


class TTSError(RuntimeError):
    pass


def strip_tags(text: str) -> str:
    """Spoken text without delivery tags (what captions show, and what offline voices read)."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]{1,20}>", " ", text or "")).strip()


def _store(cfg):
    try:
        from .store import Store
        return Store(cfg.path("paths.db"))
    except Exception:
        return None


def _decode_audio(b64: str, mime: str) -> tuple[np.ndarray, int]:
    raw = base64.b64decode(b64)
    if raw[:4] == b"RIFF":
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)
    m = re.search(r"rate=(\d+)", mime or "")
    sr = int(m.group(1)) if m else 24000
    return (np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0), sr


def gemini_speak(cfg, text: str, voice: str | None = None, style: str | None = None,
                 post=requests.post) -> tuple[np.ndarray, int]:
    """One request -> the whole text spoken. Tries the models in order; remembers daily-quota cooldowns."""
    key = cfg.gemini_key
    if not key:
        raise TTSError("no Gemini key")
    voice = voice or str(cfg.get("narration.gemini_voice", "Fenrir"))
    style = style if style is not None else str(cfg.get("narration.gemini_style", "") or "")
    models = list(cfg.get("narration.gemini_models", GEMINI_MODELS) or GEMINI_MODELS)
    base = str(cfg.get("llm.gemini_base_url", "https://generativelanguage.googleapis.com")).rstrip("/")
    st = _store(cfg)
    cool = st.cooldowns() if st else {}
    errors = []
    try:
        for model in models:
            lim = (limits_for(model) or {}).get("rpd")
            if lim and st and st.usage_today(model) >= int(lim * 0.96):
                errors.append(f"{model}: daily limit reached")
                continue
            if f"tts:{model}" in cool:
                errors.append(f"{model}: cooling down")
                continue
            part = {"text": text}
            if style:
                part["speech_metadata"] = {"style": style}
            for schema in ("voice", "prebuilt"):
                vc = {"voice": voice} if schema == "voice" else {"prebuiltVoiceConfig": {"voiceName": voice}}
                body = {"contents": [{"role": "user", "parts": [part]}],
                        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": vc}}}
                try:
                    r = post(f"{base}/v1beta/models/{model}:generateContent", json=body,
                             headers={"x-goog-api-key": key, "Content-Type": "application/json"}, timeout=120)
                except requests.RequestException as e:
                    errors.append(f"{model}: {type(e).__name__}")
                    break
                if r.status_code == 429:
                    daily = "PerDay" in r.text or "per day" in r.text.lower()
                    if st:
                        st.set_cooldown(f"tts:{model}", (next_quota_reset() if daily else time.time() + 60), "daily quota used up" if daily else "rate limited")
                        st.log_call(model, False, "tts", "429")
                    errors.append(f"{model}: quota")
                    break
                if r.status_code == 400 and schema == "voice" and "voice" in r.text.lower():
                    continue                                   # older schema name, try the other one
                if r.status_code >= 400:
                    if st:
                        st.log_call(model, False, "tts", f"HTTP {r.status_code}")
                    errors.append(f"{model}: HTTP {r.status_code} {r.text[:120]}")
                    break
                try:
                    part_out = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
                    audio = _decode_audio(part_out["data"], part_out.get("mimeType", ""))
                except Exception as e:
                    errors.append(f"{model}: bad answer ({type(e).__name__})")
                    break
                if st:
                    st.log_call(model, True, "tts", "")
                    st.bump_usage(model)
                if len(audio[0]) < audio[1] * 0.3:
                    errors.append(f"{model}: empty audio")
                    break
                return audio
    finally:
        if st:
            try:
                st.close()
            except Exception:
                pass
    raise TTSError("Gemini voice failed: " + "; ".join(errors))


def eleven_speak(cfg, text: str, post=requests.post) -> tuple[np.ndarray, int]:
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    voice = str(cfg.get("narration.eleven_voice_id", "") or "").strip()
    if not key or not voice:
        raise TTSError("ElevenLabs needs ELEVENLABS_API_KEY and narration.eleven_voice_id")
    r = post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=pcm_24000",
             headers={"xi-api-key": key, "Content-Type": "application/json"}, timeout=120,
             json={"text": text, "model_id": str(cfg.get("narration.eleven_model", "eleven_turbo_v2_5")),
                   "voice_settings": {"stability": 0.35, "similarity_boost": 0.8, "style": 0.5, "use_speaker_boost": True}})
    if r.status_code >= 400:
        raise TTSError(f"ElevenLabs HTTP {r.status_code}: {r.text[:160]}")
    raw = r.content
    return (np.frombuffer(raw[: len(raw) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0), 24000


def library_voices(cfg, language: str = "en", get=requests.get, limit: int = 200) -> list[dict]:
    """Gemini 'Extended Voice Library' (hundreds of voices) filtered by language prefix."""
    key = cfg.gemini_key
    if not key:
        raise TTSError("no Gemini key")
    base = str(cfg.get("llm.gemini_base_url", "https://generativelanguage.googleapis.com")).rstrip("/")
    out, token = [], ""
    while len(out) < limit:
        r = get(f"{base}/v1beta/voices", headers={"x-goog-api-key": key}, timeout=30,
                params={"pageSize": 100, **({"pageToken": token} if token else {})})
        if r.status_code >= 400:
            raise TTSError(f"HTTP {r.status_code}: {r.text[:160]}")
        j = r.json()
        for v in j.get("voices", []):
            g = lambda *ks: next((v[k] for k in ks if k in v), "")  # noqa: E731
            lang = str(g("languageCode", "language_code"))
            if language and not lang.lower().startswith(language.lower()):
                continue
            out.append({"id": g("id", "name"), "name": g("displayName", "display_name"), "lang": lang,
                        "accent": g("accent"), "gender": g("gender"), "pitch": g("pitch"),
                        "description": str(g("description"))[:140]})
        token = j.get("nextPageToken") or j.get("next_page_token") or ""
        if not token:
            break
    return out[:limit]
