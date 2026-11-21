"""Small helpers: the self-test (Health check) and plain-language error messages."""
from __future__ import annotations

import importlib
import json
import pkgutil
import re
import shutil
import time
from pathlib import Path

from . import quota

# ------------------------------------------------------------------ plain-language errors (#23)
_RULES: list[tuple[str, str]] = [
    (r"429.*(daily|per ?day)|daily quota", "Gemini ka aaj ka free quota khatam. Assistant khud doosre model par chala jata hai; "
                                           "sab khatam hon to {reset} par dobara chalega."),
    (r"429|rate limit|resource.?exhausted", "Gemini ne thori der ke liye roka (bohat tezi se requests). Assistant ek minute baad khud dobara try karta hai."),
    (r"\b50[0234]\b|overloaded|unavailable|_Transient", "Google ka server abhi busy hai. Kuch seconds mein khud dobara try hota hai ya doosra model use hota hai."),
    (r"no llm key|api key|gemini_api_key|\b401\b|\b403\b.*key", "Gemini API key nahi mili ya galat hai. .env file mein GEMINI_API_KEY check karein."),
    (r"uploadlimitexceeded|exceeded the number of videos", "YouTube ne is channel ke daily upload ki hadd laga di hai (naye channel par kam hoti hai). Assistant khud ruk kar 6 ghante baad dobara koshish karega; clips safe hain. youtube.com/verify par phone verify karne se hadd barh jati hai."),
    (r"quotaexceeded|youtube.*quota", "YouTube API ka daily quota ho gaya. Kal dobara upload hoga; kuch karne ki zaroorat nahi."),
    (r"invalid_grant|token.*(expired|revoked)|refresh", "YouTube login expire ho gaya. YouTube tab mein dobara sign in karein."),
    (r"no space left|disk", "Disk bhar gaya hai. Purani recordings delete karein ya Settings > Safety & disk mein cleanup chalayein."),
    (r"ffmpeg|ffprobe", "ffmpeg ka masla. Check karein ke ffmpeg install hai (winget install Gyan.FFmpeg) aur terminal dobara kholein."),
    (r"live|stream.*(ended|offline)|not currently live", "Stream abhi live nahi hai ya khatam ho gayi. Link dobara check karein."),
    (r"thumbnail", "Thumbnail YouTube ne qubool nahi kiya (Shorts par aam baat hai). Video phir bhi upload ho jati hai."),
    (r"timeout|timed out|connection|network|dns", "Internet ka masla ya server slow. Connection check karein; app khud dobara try karegi."),
    (r"script does not match|match \d+", "AI ko lagta tha script aur video match nahi karte, is liye ye clip chhor di gayi. Normal hai, agli clip banegi."),
]


def humanize_error(text: str) -> str:
    """Short Roman-Urdu explanation of a technical error, '' when there is nothing useful to add."""
    t = str(text or "")
    for pat, msg in _RULES:
        if re.search(pat, t, re.I):
            return msg.format(reset=quota.reset_text())
    return ""


# ------------------------------------------------------------------ self-test (#1, #5)
def _chk(out: list, name: str, level: str, detail: str = "", fix: str = "") -> None:
    out.append({"name": name, "level": level, "ok": level != "bad", "detail": detail, "fix": fix})


def selftest(cfg, store, root: Path | None = None, online: bool = False) -> list[dict]:
    """Everything that must work tomorrow, in one list. level: ok | warn | bad. `online` also asks Google whether
    the Gemini key works (the models list call does not use any generation quota)."""
    out: list[dict] = []
    # 1. all program modules load (catches a half-copied update)
    import featureclip
    bad = []
    for m in pkgutil.walk_packages(featureclip.__path__, "featureclip."):
        if m.name.endswith((".__main__",)) or ".web.static" in m.name:
            continue
        try:
            importlib.import_module(m.name)
        except Exception as e:                       # noqa: BLE001
            bad.append(f"{m.name}: {type(e).__name__}: {e}")
    _chk(out, "Program files", "bad" if bad else "ok", "; ".join(bad)[:300] or "all modules load",
         "Update zip poora extract nahi hua? Dobara extract karke app restart karein." if bad else "")
    # 2. tools
    for tool in ("ffmpeg", "ffprobe"):
        _chk(out, tool, "ok" if shutil.which(tool) else "bad", "", "winget install Gyan.FFmpeg" if not shutil.which(tool) else "")
    for mod, label in (("numpy", "numpy"), ("cv2", "OpenCV"), ("faster_whisper", "Whisper"), ("yt_dlp", "yt-dlp"),
                       ("googleapiclient", "YouTube library"), ("PIL", "Pillow")):
        try:
            importlib.import_module(mod)
            _chk(out, label, "ok")
        except Exception:
            _chk(out, label, "bad", "", "pip install -r requirements.txt")
    # 3. database read/write
    try:
        store.kv_set("selftest_probe", str(time.time()))
        _chk(out, "Database", "ok" if store.kv_get("selftest_probe") else "bad")
    except Exception as e:                            # noqa: BLE001
        _chk(out, "Database", "bad", str(e)[:150])
    # 4. disk
    try:
        from . import cleanup
        free = cleanup.free_gb(cfg.path("paths.workdir"))
        low = float(cfg.get("cleanup.min_free_gb", 4))
        _chk(out, "Free disk space", "ok" if free >= low * 1.5 else ("warn" if free >= low else "bad"), f"{free:.1f} GB free",
             "Purani recordings delete karein (Settings > Safety & disk)." if free < low * 1.5 else "")
    except Exception as e:                            # noqa: BLE001
        _chk(out, "Free disk space", "warn", str(e)[:100])
    # 5. game name and settings
    game = str(cfg.get("game.name", "") or "")
    _chk(out, "Game name", "ok" if game else "bad", game or "empty", "Settings > Finding features > Game name" if not game else "")
    from . import schedule
    slots = schedule.parse_slots(cfg.get("schedule.slots", schedule.DEFAULT_SLOTS))
    _chk(out, "Publish slots", "ok" if slots else "bad", ", ".join("%02d:%02d" % s for s in slots), "HH:MM comma separated" if not slots else "")
    # 6. Gemini key + model list
    if not cfg.gemini_key:
        _chk(out, "Gemini API key", "bad", "missing", "Add GEMINI_API_KEY to the .env file")
    elif online:
        try:
            import requests
            base = str(cfg.get("llm.gemini_base_url", "https://generativelanguage.googleapis.com"))
            r = requests.get(f"{base}/v1beta/models", params={"pageSize": 1}, headers={"x-goog-api-key": cfg.gemini_key}, timeout=20)
            _chk(out, "Gemini API key", "ok" if r.status_code == 200 else "bad", f"Google answered HTTP {r.status_code}",
                 "Key galat/expired: Google AI Studio se nayi key banayein." if r.status_code != 200 else "")
        except Exception as e:                        # noqa: BLE001
            _chk(out, "Gemini API key", "warn", f"could not reach Google: {type(e).__name__}", "Internet check karein")
    else:
        _chk(out, "Gemini API key", "ok", "set (use the full test to check it with Google)")
    # 7. models + quota
    try:
        from .llm import LLM
        llm = LLM(cfg, store)
        cd = store.cooldowns()
        usable = [m for m in llm.models() if m not in cd and not llm._over_budget(m)]
        txt = f"{len(usable)} of {len(llm.models())} models usable now"
        if cd:
            txt += "; cooling down: " + ", ".join(f"{m} ({v['reason']})" for m, v in list(cd.items())[:4])
        _chk(out, "Gemini models", "ok" if usable else "warn", txt,
             f"Sab models ka quota khatam: {quota.reset_text()} par reset hoga." if not usable else "")
    except Exception as e:                            # noqa: BLE001
        _chk(out, "Gemini models", "warn", str(e)[:150])
    # 8. voice
    eng = str(cfg.get("narration.engine", "kokoro"))
    kd = cfg.path("paths.models") / "kokoro"
    kok = all((kd / n).is_file() for n in ("kokoro-v1.0.int8.onnx", "voices-v1.0.bin"))
    _chk(out, "Offline voice (Kokoro)", "ok" if kok else "warn", "ready" if kok else "not downloaded yet",
         "Pehli voice-over ke waqt ~120 MB khud download hogi (internet chahiye). Stream se pehle ek test clip banayein." if not kok else "")
    _chk(out, "Voice engine", "ok", eng + (" (cloud voice with offline fallback)" if eng != "kokoro" else ""))
    # 9. YouTube login
    tok = Path(str(cfg.get("upload.token", "token.json")))
    tok = tok if tok.is_absolute() else cfg.root / tok
    sec = Path(str(cfg.get("upload.client_secret", "client_secret.json")))
    sec = sec if sec.is_absolute() else cfg.root / sec
    if tok.is_file():
        try:
            from .uploader import scopes_for
            granted = set(json.loads(tok.read_text(encoding="utf-8")).get("scopes") or [])
            miss = set(scopes_for(cfg)) - granted
            _chk(out, "YouTube login", "warn" if miss else "ok", "extra permission needed" if miss else "signed in",
                 "Ek nayi permission chahiye: YouTube tab mein dobara sign in karein." if miss else "")
        except Exception:
            _chk(out, "YouTube login", "ok", "signed in")
    else:
        _chk(out, "YouTube login", "bad", "not signed in", "YouTube tab: client_secret.json upload karein aur sign in karein"
             if not sec.is_file() else "YouTube tab mein sign in karein")
    # 10. alerts
    chans = [n for n, k in (("ntfy", "notify.ntfy_topic"), ("Gmail", "notify.email_to")) if cfg.get(k)]
    _chk(out, "Phone alerts", "ok" if chans else "warn", ", ".join(chans) or "none", "Settings > Notifications (ntfy/Discord/Gmail)" if not chans else "")
    # 11. quick logic checks of the rules that broke before
    try:
        from . import metadata
        t = "GTA 6 gator walks into a store" + metadata.norm_suffix(cfg.get("upload.title_suffix", ""))
        probs = metadata.validate({"title": t, "description": "x" * 90}, game)
        _chk(out, "Title rules", "ok" if not probs else "warn", t, "; ".join(probs))
    except Exception as e:                            # noqa: BLE001
        _chk(out, "Title rules", "bad", str(e)[:150])
    return out
