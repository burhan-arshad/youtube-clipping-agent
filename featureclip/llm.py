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
        self.timeout = int(cfg.get("llm.timeout", 90))
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

    def key(self) -> str:
        return self.cfg.openai_key if self.provider == "openai" else self.cfg.gemini_key

    def available(self) -> bool:
        return bool(self.key()) and bool(self.models())

    # ---- public
    def json(self, prompt: str, system: str = "", images: list[bytes] | None = None,
             temperature: float = 0.4) -> dict:
        if not self.available():
            raise LLMError("no LLM key configured (set GEMINI_API_KEY in .env)")
        last_err = "unknown"
        waits = 0
        while True:
            tried_any = False
            soonest = None
            for model in self.models():
                if model in self._dead:
                    continue
                limit = (self.cfg.get("llm.daily_limit", {}) or {}).get(model)
                if limit and self.store and self.store.usage_today(model) >= int(limit):
                    continue
                cool_until = self._cool.get(model, 0)
                if cool_until > self._now():
                    soonest = cool_until if soonest is None else min(soonest, cool_until)
                    continue
                tried_any = True
                for attempt in range(int(self.cfg.get("llm.max_retries", 2))):
                    try:
                        text = self._call(model, prompt, system, images, temperature)
                        if self.store:
                            self.store.bump_usage(model)
                        return extract_json(text)
                    except _RateLimited:
                        self._cool[model] = self._now() + 60
                        last_err = f"{model}: rate limited"
                        break
                    except _Dead as e:
                        self._dead.add(model)
                        last_err = f"{model}: {e}"
                        break
                    except (_Transient, ValueError, requests.RequestException) as e:
                        last_err = f"{model}: {type(e).__name__}: {e}"
                        self._sleep(1.0)
                        continue
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
        body: dict[str, Any] = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        if system:
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
            raise _RateLimited()
        if r.status_code in (400, 401, 403, 404):
            raise _Dead(f"HTTP {r.status_code}: {r.text[:200]}")
        if r.status_code >= 500:
            raise _Transient(f"HTTP {r.status_code}")
        r.raise_for_status()


class _RateLimited(Exception):
    pass


class _Dead(Exception):
    pass


class _Transient(Exception):
    pass
