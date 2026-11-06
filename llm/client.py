"""LLM clients: the real Gemini client, and a fake one for tests.

Setup (the key lives in your .env file, never in chat, never in code):
    GEMINI_API_KEY=your_key
    GEMINI_MODEL=<a model id your key can use>      (see below)
    LLM_MODE=off                                    (optional: turn the LLM off, use plain summaries)

Which model? Model names change often and differ by account, so this project does not guess.
Run this to see exactly which model ids YOUR key can use, then copy one into GEMINI_MODEL:
    python -m llm.client --list-models
and this to send one tiny test message:
    python -m llm.client --test

Honest limits
-------------
  - This code could not be run against the real Gemini API while it was written (no network and no
    key in the build sandbox). The request/response shapes follow Google's documented REST format
    (POST .../v1beta/models/<model>:generateContent, key in the x-goog-api-key header) and are
    tested against recorded-style fake responses. Run `--test` once to confirm it works for you.
  - It uses only the standard library (urllib), so there is nothing extra to install.
  - The API key is sent only in a header, never in the URL, and is masked in every error message.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PLACEHOLDER_KEYS = {"", "paste_your_key_here", "your_key", "changeme"}


class LLMError(Exception):
    """Anything that stops the LLM from answering. The message is safe to show (key masked)."""


# ----------------------------------------------------------------------------
# Fake client (tests and offline demos)
# ----------------------------------------------------------------------------
class FakeLLM:
    """Returns scripted replies in order. A reply can be a string, an Exception (raised), or a
    function (system, user) -> string. Records every call in `.calls`."""
    name = "fake"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.responses:
            raise LLMError("FakeLLM has no more scripted replies.")
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r(system, user) if callable(r) else r


# ----------------------------------------------------------------------------
# Gemini over plain HTTPS
# ----------------------------------------------------------------------------
def _urllib_transport(method: str, url: str, headers: dict, body: bytes | None, timeout: float):
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310 (fixed https host)
            return resp.status, resp.read()
    except urllib.error.HTTPError as err:
        return err.code, err.read()
    except (urllib.error.URLError, TimeoutError, OSError) as err:
        raise LLMError(f"Could not reach the Gemini API ({err.__class__.__name__}). "
                       f"Check the internet connection.") from None


class GeminiClient:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout: float = 30.0, temperature: float = 0.2,
                 max_output_tokens: int = 1024, base_url: str = BASE_URL, transport=None):
        if not api_key or api_key.strip().lower() in PLACEHOLDER_KEYS:
            raise LLMError("GEMINI_API_KEY is missing. Put your key in the .env file.")
        if not model:
            raise LLMError("GEMINI_MODEL is not set. Run  python -m llm.client --list-models  and copy "
                           "one model id into your .env file as GEMINI_MODEL=...")
        self._key = api_key.strip()
        self.model = model.strip().removeprefix("models/")
        self.timeout, self.temperature, self.max_output_tokens = timeout, temperature, max_output_tokens
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _urllib_transport

    # -- helpers
    def _mask(self, text: str) -> str:
        return str(text).replace(self._key, "***")

    def _headers(self) -> dict:
        return {"Content-Type": "application/json", "x-goog-api-key": self._key}

    def _error_text(self, data: bytes) -> str:
        try:
            msg = json.loads(data.decode("utf-8", "replace")).get("error", {}).get("message", "")
        except (ValueError, AttributeError):
            msg = ""
        return self._mask(msg or data[:200].decode("utf-8", "replace"))[:300]

    def _raise_for(self, status: int, data: bytes):
        detail = self._error_text(data)
        if status in (400, 401, 403):
            hint = "Check GEMINI_API_KEY and that the key is allowed to use this model."
        elif status == 404:
            hint = ("The model was not found. Run  python -m llm.client --list-models  and put a listed "
                    "id in GEMINI_MODEL.")
        elif status == 429:
            hint = "Rate limit or free-tier quota reached. Wait a bit and try again."
        elif status >= 500:
            hint = "Google's service had a problem. Try again later."
        else:
            hint = ""
        raise LLMError(f"Gemini returned HTTP {status}. {hint} {detail}".strip())

    # -- calls
    def generate(self, system: str, user: str) -> str:
        url = f"{self.base_url}/models/{self.model}:generateContent"
        body = json.dumps({
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": self.temperature, "maxOutputTokens": self.max_output_tokens},
        }).encode("utf-8")
        status, data = self._transport("POST", url, self._headers(), body, self.timeout)
        if status != 200:
            self._raise_for(status, data)
        try:
            obj = json.loads(data.decode("utf-8"))
        except ValueError:
            raise LLMError("Gemini returned a reply that is not valid JSON.") from None
        cands = obj.get("candidates") or []
        if not cands:
            block = (obj.get("promptFeedback") or {}).get("blockReason")
            raise LLMError(f"Gemini returned no answer" + (f" (blocked: {block})." if block else "."))
        parts = (cands[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought")).strip()
        if not text:
            raise LLMError(f"Gemini returned an empty answer (finish reason: {cands[0].get('finishReason')}).")
        return text

    def list_models(self) -> list[str]:
        """Model ids this key can use for generateContent."""
        out, token = [], None
        for _ in range(5):
            url = f"{self.base_url}/models?pageSize=100" + (f"&pageToken={token}" if token else "")
            status, data = self._transport("GET", url, self._headers(), None, self.timeout)
            if status != 200:
                self._raise_for(status, data)
            obj = json.loads(data.decode("utf-8"))
            for m in obj.get("models", []):
                if "generateContent" in (m.get("supportedGenerationMethods") or []):
                    out.append(str(m.get("name", "")).removeprefix("models/"))
            token = obj.get("nextPageToken")
            if not token:
                break
        return sorted(out)


# ----------------------------------------------------------------------------
# Setup from the environment
# ----------------------------------------------------------------------------
def _env(env=None):
    if env is not None:
        return env
    try:                                                    # read .env if python-dotenv is installed
        from dotenv import load_dotenv
        from pathlib import Path
        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except ImportError:
        pass
    return os.environ


def llm_status(env=None) -> dict:
    """Plain facts for a health screen: is the LLM configured, and if not, why not. Never shows the key."""
    e = _env(env)
    if str(e.get("LLM_MODE", "")).lower() == "off":
        return {"configured": False, "reason": "LLM_MODE=off: plain summaries are used.", "model": None}
    key = str(e.get("GEMINI_API_KEY", "")).strip()
    if key.lower() in PLACEHOLDER_KEYS:
        return {"configured": False, "reason": "No GEMINI_API_KEY set: plain summaries are used.", "model": None}
    model = str(e.get("GEMINI_MODEL", "")).strip()
    if not model:
        return {"configured": False, "model": None,
                "reason": "GEMINI_MODEL is not set. Run python -m llm.client --list-models and set it."}
    return {"configured": True, "reason": "ok", "model": model.removeprefix("models/")}


def get_llm(env=None, transport=None):
    """A GeminiClient if everything is configured, otherwise None (the app then uses plain summaries)."""
    e = _env(env)
    st = llm_status(e)
    if not st["configured"]:
        return None
    return GeminiClient(str(e.get("GEMINI_API_KEY")).strip(), st["model"], transport=transport)


if __name__ == "__main__":
    args = sys.argv[1:]
    status = llm_status()
    print("LLM status:", status["reason"] if not status["configured"] else f"configured, model {status['model']}")
    try:
        key = str(_env().get("GEMINI_API_KEY", "")).strip()
        if "--list-models" in args:
            if key.lower() in PLACEHOLDER_KEYS:
                raise LLMError("GEMINI_API_KEY is missing. Put your key in the .env file.")
            client = GeminiClient(key, "placeholder-model")
            ids = client.list_models()
            print("Models your key can use with generateContent:" if ids else "No models were listed.")
            for i in ids:
                print("  ", i)
            print("\nCopy one id into your .env file as  GEMINI_MODEL=<id>")
        elif "--test" in args:
            client = get_llm()
            if client is None:
                raise LLMError(status["reason"])
            print("Reply:", client.generate("Reply with exactly one word.", "Say OK."))
        else:
            print("Usage: python -m llm.client [--list-models | --test]")
    except LLMError as err:
        print("Problem:", err)
        sys.exit(1)
