"""Client for the three Bodhan models used by the pipeline. Every ASR, MT and TTS call goes through here.

- Indic-Transcribe: POST /v1/audio/transcriptions   (multipart, <=30 s of audio, returns {"text"})
- Indic-Translate:  POST /v1/chat/completions       (OpenAI-compatible; dedicated /translate is currently 404)
- Indic-Speak:      POST /v1/audio/speech           (JSON, returns PCM16 24 kHz mono WAV)

Retries with exponential backoff on 429/5xx, a global concurrency limit, an on-disk cache so
reruns do not repeat paid calls, and per-model cost metering.
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx

from . import config
from .logs import log


class BodhanError(RuntimeError):
    def __init__(self, msg: str, status: int | None = None, code: str | None = None):
        super().__init__(msg)
        self.status, self.code = status, code


def hint(status: int | None, model: str) -> str:
    """Plain-English next step for the errors people actually hit."""
    if status in (401, 403):
        return f" | Hint: the key was rejected. Check it is correct and was created for {model} (each Bodhan key serves one model)."
    if status == 404:
        return (f" | Hint: route not found for this key. Most often the key was not created for {model} "
                "(each Bodhan key serves one model), or BODHAN_BASE_URL is wrong.")
    if status == 429:
        return " | Hint: rate limited or out of credit. Check your balance in the Bodhan console, lower concurrency and wait."
    return ""


class Meter:
    """Counts calls, retries, latency and cost (INR) per model. Thread safe."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {m: {"calls": 0, "cache_hits": 0, "retries": 0, "seconds": 0.0, "cost_inr": 0.0}
                     for m in ("transcribe", "translate", "speak")}

    def add(self, model: str, **kw):
        with self.lock:
            for k, v in kw.items():
                self.data[model][k] += v

    def snapshot(self) -> dict:
        with self.lock:
            out = {m: {k: round(v, 4) for k, v in d.items()} for m, d in self.data.items()}
        out["total_cost_inr"] = round(sum(d["cost_inr"] for d in out.values() if isinstance(d, dict)), 4)
        return out


class BodhanClient:
    def __init__(self, keys: config.Keys, cache_dir: Path | None, max_concurrency: int = 4,
                 meter: Meter | None = None, base_url: str | None = None, max_attempts: int = 5,
                 backoff: float = 1.5, http: httpx.Client | None = None):
        self.keys, self.cache_dir = keys, cache_dir
        self.sem = threading.BoundedSemaphore(max_concurrency)
        self.meter = meter or Meter()
        self.base = (base_url or config.BASE_URL).rstrip("/")
        self.max_attempts, self.backoff = max_attempts, backoff
        self.http = http or httpx.Client(timeout=httpx.Timeout(120, connect=15))
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def with_meter(self, meter: "Meter") -> "BodhanClient":
        """Same connection, cache and concurrency limit, but costs are counted on a separate meter (per language)."""
        c = object.__new__(BodhanClient)
        c.__dict__.update(self.__dict__)
        c.meter = meter
        return c

    # ---- cache ----
    def _ck(self, kind: str, *parts) -> str:
        return hashlib.sha256(json.dumps([kind, *parts], ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def _cget(self, key: str, ext: str):
        if not self.cache_dir:
            return None
        p = self.cache_dir / f"{key}.{ext}"
        return p.read_bytes() if p.exists() else None

    def _cput(self, key: str, ext: str, data: bytes):
        if self.cache_dir:
            (self.cache_dir / f"{key}.{ext}").write_bytes(data)

    # ---- http with retries ----
    def _request(self, model: str, label: str, fn):
        last = None
        for attempt in range(1, self.max_attempts + 1):
            t0 = time.time()
            try:
                with self.sem:
                    r = fn()
                self.meter.add(model, calls=1, seconds=time.time() - t0)
                if r.status_code == 200:
                    log("api_call", api=label, status=200, seconds=round(time.time() - t0, 2))
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = float(r.headers.get("retry-after", 0) or 0) or self.backoff * 2 ** (attempt - 1)
                    last = BodhanError(f"{label}: HTTP {r.status_code}{hint(r.status_code, 'indic-' + model)}", r.status_code)
                    self.meter.add(model, retries=1)
                    log("api_retry", api=label, status=r.status_code, attempt=attempt, wait=round(wait, 1))
                    time.sleep(wait)
                    continue
                code = None
                try:
                    code = r.json().get("error", {}).get("code")
                except Exception:
                    pass
                log("api_error", api=label, status=r.status_code, body=r.text[:300], request_id=r.headers.get("x-request-id", ""))
                raise BodhanError(f"{label}: HTTP {r.status_code}: {r.text[:300]}{hint(r.status_code, 'indic-' + model)}", r.status_code, code)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = BodhanError(f"{label}: {type(e).__name__}")
                self.meter.add(model, retries=1)
                log("api_retry", api=label, error=type(e).__name__, attempt=attempt)
                time.sleep(self.backoff * 2 ** (attempt - 1))
        raise last or BodhanError(f"{label}: failed")

    def _auth(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}"}

    # ---- Indic-Transcribe ----
    def transcribe(self, wav_path: Path, lang: str, seconds: float) -> str:
        data = Path(wav_path).read_bytes()
        key = self._ck("stt", hashlib.sha256(data).hexdigest(), lang)
        hit = self._cget(key, "json")
        if hit:
            self.meter.add("transcribe", cache_hits=1)
            return json.loads(hit)["text"]
        r = self._request("transcribe", "transcribe", lambda: self.http.post(
            f"{self.base}/v1/audio/transcriptions", headers=self._auth(self.keys.transcribe),
            files={"file": (Path(wav_path).name, data, "audio/wav")},
            data={"model": "indic-transcribe", "language": lang}))
        text = (r.json().get("text") or "").strip()
        self.meter.add("transcribe", cost_inr=seconds / 60 * config.PRICE_TRANSCRIBE_PER_MIN)
        self._cput(key, "json", json.dumps({"text": text}, ensure_ascii=False).encode())
        return text

    # ---- Indic-Translate ----
    # The dedicated POST /translate endpoint advertised in the docs currently returns 404.
    # The live path is the OpenAI-compatible chat completions endpoint.
    def translate(self, text: str, lang: str, script: str = "native") -> dict:
        key = self._ck("mt", text, lang, script)
        hit = self._cget(key, "json")
        if hit:
            self.meter.add("translate", cache_hits=1)
            return json.loads(hit)

        body = {
            "model": "indic-translate",
            "messages": [{"role": "user", "content": text}],
            "target_language_code": lang,
        }
        if script != "native":
            body["target_script"] = script

        r = self._request(
            "translate",
            "translate",
            lambda: self.http.post(
                f"{self.base}/v1/chat/completions",
                headers=self._auth(self.keys.translate),
                json=body,
            ),
        )
        j = r.json()

        # chat-completions response shape
        content = ""
        try:
            content = (j["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            # defensive: fall back to older /translate shape if it ever returns
            content = (j.get("translation") or "").strip()

        tokens = (
            (j.get("usage") or {}).get("completion_tokens")
            or (j.get("usage") or {}).get("output_tokens")
            or max(1, len(content) // 2)
        )
        truncated = bool(j.get("truncated", False))

        out = {"text": content, "truncated": truncated, "tokens": tokens}
        self.meter.add("translate", cost_inr=tokens / 10000 * config.PRICE_TRANSLATE_PER_10K_TOKENS)
        if not out["truncated"]:  # never cache a cut-off translation
            self._cput(key, "json", json.dumps(out, ensure_ascii=False).encode())
        return out

    # ---- Indic-Speak ----
    def speak(self, text: str, voice: str, lang: str, style: str | None = None, nonce: int = 0) -> bytes:
        instr = {"lang": lang}
        if style:
            instr["style"] = style
        key = self._ck("tts", text, voice, json.dumps(instr, sort_keys=True), nonce)
        hit = self._cget(key, "wav")
        if hit:
            self.meter.add("speak", cache_hits=1)
            return hit
        r = self._request("speak", "speak", lambda: self.http.post(
            f"{self.base}/v1/audio/speech", headers=self._auth(self.keys.speak),
            json={"model": "indic-speak", "input": text, "voice": voice,
                  "instructions": json.dumps(instr)}))
        self.meter.add("speak", cost_inr=len(text) / 10000 * config.PRICE_SPEAK_PER_10K_CHARS)
        self._cput(key, "wav", r.content)
        return r.content
