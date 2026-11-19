"""Free-tier friendly LLM client: Gemini (default) or any OpenAI-compatible API.

* a *chain* of models: if one is rate-limited (429), overloaded (5xx) or returns
  broken JSON, the next one is used automatically
* optional daily call budget per model (stored in the SQLite db)
* images (keyframes) are sent inline for vision judging
"""
from __future__ import annotations

import base64
import json
import re
import time
from typing import Any

import requests


class LLMError(RuntimeError):
    pass


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t, flags=re.I)
    try:
        v = json.loads(t)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(t)):
            c = t[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        v = json.loads(t[start:i + 1])
                        if isinstance(v, dict):
                            return v
                    except json.JSONDecodeError:
                        break
                    break
        start = t.find("{", start + 1)
    raise ValueError("no JSON object found in model reply")


from .llm_limits import limits_for  # noqa: E402
from .quota import next_quota_reset, reset_text  # noqa: E402,F401


class LLM:
    def __init__(self, cfg, store=None, sleep=time.sleep, now=time.time):
        self.cfg = cfg
        self.store = store
        self._sleep = sleep
        self._now = now
        self._cool: dict[str, float] = {}      # model -> time when usable again
        self._dead: set[str] = set()           # models that returned 400/404 this session
        self._extra: list[str] = []            # models found by auto-discovery
        self._discovered = False
        self.provider = cfg.get("llm.provider", "gemini")
        self.timeout = int(cfg.get("llm.timeout", 60))
        self._slow: dict[str, float] = {}          # model -> time until which it is tried last (it was too slow)
        self.gemini_base = cfg.get("llm.gemini_base_url", "https://generativelanguage.googleapis.com")

    # ---- availability
    def models(self) -> list[str]:
        if self.provider == "openai":
            return list(self.cfg.get("llm.openai_models", []))
        return list(self.cfg.get("llm.models", [])) + self._extra

    def discover(self) -> bool:
        """Ask Google which Gemini models this key can use today (model names change often)."""
        self._discovered = True
        if self.provider != "gemini":
            return False
        try:
            r = requests.get(f"{self.gemini_base}/v1beta/models", params={"pageSize": 200}, headers={"x-goog-api-key": self.key()},
                             timeout=30)
            r.raise_for_status()
            found = []
            for m in r.json().get("models", []):
                name = m.get("name", "").split("/")[-1]
                if ("generateContent" in m.get("supportedGenerationMethods", []) and "flash" in name
                        and not any(x in name for x in ("image", "tts", "live", "audio", "embedding",
                                                        "thinking", "robotics", "computer", "learn"))
                        and name not in self.models() and name not in self._dead):
                    found.append(name)

            def rank(n: str):
                import re
                v = re.search(r"(\d+(?:\.\d+)?)", n)
                return (-(float(v.group(1)) if v else 0), "lite" in n, "preview" in n or "exp" in n, n)
            self._extra = sorted(found, key=rank)[:4]
            if self._extra:
                print(f"[llm] configured models are gone; using discovered models: {self._extra}")
            return bool(self._extra)
        except Exception:
            return False

    def _log(self, model: str, ok: bool, err: str = "") -> None:
        try:
            if self.store:
                self.store.log_call(model, ok, "", err)
        except Exception:
            pass

    def _persist_cooldown(self, model: str, reason: str, secs: float | None = None) -> None:
        try:
            if self.store:
                self.store.set_cooldown(model, self._now() + (secs if secs is not None else self._cool.get(model, 0) - self._now()), reason)
        except Exception:
            pass

    def _stat(self, model: str, kind: str, err: str) -> None:
        try:
            if self.store:
                self.store.bump_stat(model, kind, err)
        except Exception:
            pass

    def key(self) -> str:
        return self.cfg.openai_key if self.provider == "openai" else self.cfg.gemini_key

    def available(self) -> bool:
        return bool(self.key()) and bool(self.models())

    # ---- public
    def ordered_models(self, quality: bool = False) -> list[str]:
        """All models in fallback order. quality=True puts the smarter (low daily limit) models first."""
        base = self.models()
        now = self._now()
        slow = [m for m in base if self._slow.get(m, 0) > now]
        base = [m for m in base if m not in slow] + slow         # slow models only as a last resort
        if not quality or self.provider != "gemini":
            return base
        q = [m for m in (self.cfg.get("llm.quality_models", []) or []) if m in base and m not in slow]
        return q + [m for m in base if m not in q]

    def _over_budget(self, model: str) -> bool:
        """True when this app already used (almost) the whole free daily limit of the model."""
        if not self.store:
            return False
        limit = (self.cfg.get("llm.daily_limit", {}) or {}).get(model)
        if not limit:
            lim = limits_for(model) or {}
            rpd = lim.get("rpd")
            limit = int(rpd * float(self.cfg.get("llm.daily_limit_margin", 0.96))) if rpd else None
        return bool(limit) and self.store.usage_today(model) >= int(limit)

    def json(self, prompt: str, system: str = "", images: list[bytes] | None = None,
             temperature: float = 0.4, quality: bool = False) -> dict:
        if not self.available():
            raise LLMError("no LLM key configured (set GEMINI_API_KEY in .env)")
        game = str(self.cfg.get("game.name", "") or "")
        if game:
            system = (f'FACT: the footage is from the game "{game}" (it may be newer than your knowledge). Always call it '
                      f'exactly "{game}"; never rename it, never replace it with another game or version, and never say '
                      f'it does not exist. ' + system)
        last_err = "unknown"
        waits = 0
        while True:
            tried_any = False
            soonest = None
            for model in self.ordered_models(quality):
                if model in self._dead:
                    continue
                if images and model.lower().startswith("gemma"):
                    continue                    # Gemma is used for text-only calls
                if self._over_budget(model):
                    continue
                cool_until = self._cool.get(model, 0)
                try:
                    if self.store:
                        shared = self.store.cooldowns().get(model)
                        if shared:
                            cool_until = max(cool_until, shared["until"])
                except Exception:
                    pass
                if cool_until > self._now():
                    soonest = cool_until if soonest is None else min(soonest, cool_until)
                    continue
                tried_any = True
                for attempt in range(int(self.cfg.get("llm.max_retries", 2))):
                    try:
                        t0 = self._now()
                        text = self._call(model, prompt, system, images, temperature)
                        if self._now() - t0 > float(self.cfg.get("llm.slow_seconds", 40)):
                            self._slow[model] = self._now() + 900   # works but slow: other models go first for a while
                        if self.store:
                            self.store.bump_usage(model)
                            self._log(model, True)
                        return extract_json(text)
                    except _RateLimited as e:
                        self._stat(model, "rate_limited", "HTTP 429 (daily quota used up)" if e.daily else "HTTP 429 (free-tier rate limit)")
                        until = next_quota_reset(self._now()) if e.daily else self._now() + 60
                        self._cool[model] = until                                       # next model is tried at once
                        self._log(model, False, "429 daily quota" if e.daily else "429 rate limit")
                        self._persist_cooldown(model, f"daily quota used up (resets {reset_text()})" if e.daily else "rate limit (per minute)", until - self._now())
                        last_err = f"{model}: rate limited"
                        break
                    except _Dead as e:
                        self._stat(model, "errors", str(e))
                        self._log(model, False, str(e)[:120])
                        self._persist_cooldown(model, "model not available: " + str(e)[:80], 3600)
                        self._dead.add(model)
                        last_err = f"{model}: {e}"
                        break
                    except requests.Timeout:
                        last_err = f"{model}: no answer within {self.timeout}s"
                        self._stat(model, "errors", last_err)
                        self._log(model, False, last_err)
                        self._slow[model] = self._now() + 900
                        self._cool[model] = self._now() + 120        # too slow: go to the next model at once, retry later
                        self._persist_cooldown(model, "too slow (timeout)", 120)
                        break
                    except (_Transient, ValueError, requests.RequestException) as e:
                        last_err = f"{model}: {type(e).__name__}: {e}"
                        self._stat(model, "errors", last_err)
                        self._log(model, False, last_err[:120])
                        if attempt + 1 < int(self.cfg.get("llm.max_retries", 2)):
                            self._sleep(2.0 * (attempt + 1))        # busy server: wait a little, then retry once
                            continue
                        self._cool[model] = self._now() + 45        # still busy: let the other models work for a while
                        self._persist_cooldown(model, "server busy (HTTP 5xx)", 45)
                        break
            if not self._discovered and self._dead and self.discover():
                continue                       # new models found -> try them
            if tried_any:
                # every usable model failed this round
                raise LLMError(f"all models failed; last error: {last_err}")
            if soonest is not None and waits < 3:
                self._sleep(max(0.0, min(65.0, soonest - self._now())))
                waits += 1
                continue
            raise LLMError(f"no usable model left (budget/cooldown/dead). last: {last_err}")

    # ---- providers
    def _call(self, model: str, prompt: str, system: str, images, temperature: float) -> str:
        if self.provider == "openai":
            return self._call_openai(model, prompt, system, images, temperature)
        return self._call_gemini(model, prompt, system, images, temperature)

    def _call_gemini(self, model, prompt, system, images, temperature) -> str:
        parts: list[dict[str, Any]] = [{"text": prompt}]
        for img in images or []:
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": base64.b64encode(img).decode()}})
        gemma = model.lower().startswith("gemma")
        if gemma and system:                    # Gemma has no separate system instruction: fold it into the prompt
            parts[0]["text"] = system + "\n\n" + prompt + "\n\nAnswer with JSON only."
        gen: dict[str, Any] = {"temperature": temperature}
        if not gemma:
            gen["responseMimeType"] = "application/json"
        body: dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
        if system and not gemma:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        url = f"{self.gemini_base}/v1beta/models/{model}:generateContent"
        r = requests.post(url, json=body, timeout=self.timeout,
                          headers={"x-goog-api-key": self.key(), "Content-Type": "application/json"})
        self._check(r)
        data = r.json()
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise _Transient(f"unexpected reply shape (blocked or empty): {str(data)[:200]}")

    def _call_openai(self, model, prompt, system, images, temperature) -> str:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for img in images or []:
            content.append({"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}})
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": content})
        base = self.cfg.get("llm.openai_base_url").rstrip("/")
        r = requests.post(f"{base}/chat/completions", timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key()}"},
                          json={"model": model, "messages": msgs, "temperature": temperature,
                                "response_format": {"type": "json_object"}})
        self._check(r)
        try:
            return r.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise _Transient("unexpected reply shape")

    @staticmethod
    def _check(r: requests.Response) -> None:
        if r.status_code == 429:
            raise _RateLimited("PerDay" in r.text or "per day" in r.text.lower())
        if r.status_code in (400, 401, 403, 404):
            raise _Dead(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 500:
            raise _Transient(f"HTTP {r.status_code}")
        r.raise_for_status()


class _RateLimited(Exception):
    def __init__(self, daily: bool = False):
        super().__init__("rate limited")
        self.daily = daily


class _Dead(Exception):
    pass


class _Transient(Exception):
    pass
