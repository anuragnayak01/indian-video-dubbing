"""Client for the three Bodhan models used by the pipeline. Every ASR, MT and TTS call goes through here.

- Indic-Transcribe: POST /v1/audio/transcriptions   (multipart, <=30 s of audio, returns {"text"})
- Indic-OCR:        POST /v1/chat/completions (model indic-ocr, image as base64 data URI, returns blocks with bbox)
- Indic-Translate:  POST /v1/chat/completions first (live path); POST /translate only as fallback
- Indic-Speak:      POST /v1/audio/speech           (JSON, returns PCM16 24 kHz mono WAV)

Retries with exponential backoff on 429/5xx, a global concurrency limit, an on-disk cache so
reruns do not repeat paid calls, and per-model cost metering.
"""
import base64
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
        return (" | Hint: Bodhan is limiting requests for this key (the text before this hint is Bodhan's own reply). If your balance is "
                "fine, the key is being rate limited: set DUB_CONCURRENCY=1 and BODHAN_MIN_INTERVAL=2 on Render and rerun. "
                "The app also slows itself after a 429.")
    return ""


class Meter:
    """Counts calls, retries, latency and cost (INR) per model. Thread safe."""

    def __init__(self):
        self.lock = threading.Lock()
        self.data = {m: {"calls": 0, "cache_hits": 0, "retries": 0, "seconds": 0.0, "cost_inr": 0.0}
                     for m in ("transcribe", "translate", "speak", "ocr")}

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
        # Pacing: minimum seconds between request starts per model.
        # Default 1.5s (was 0) so free-tier / low RPM keys do not immediately 429.
        # Override with BODHAN_MIN_INTERVAL=0 for full speed when your quota allows it.
        self.base_pace = float(os.environ.get("BODHAN_MIN_INTERVAL", "1.5") or 0)
        self.pace, self.next_at, self.plock = {}, {}, threading.Lock()
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
    def _pace_wait(self, model: str):
        iv = self.pace.get(model, self.base_pace)
        if iv <= 0:
            return
        with self.plock:
            now = time.monotonic()
            at = max(now, self.next_at.get(model, 0.0))
            self.next_at[model] = at + iv
        if at > now:
            time.sleep(at - now)

    def _slow_down(self, model: str):
        if self.backoff <= 0:
            return
        with self.plock:
            self.pace[model] = min(15.0, max(self.pace.get(model, self.base_pace) * 2, self.backoff))

    def _relax(self, model: str):
        with self.plock:
            cur = self.pace.get(model)
            if cur:
                self.pace[model] = cur * 0.8 if cur * 0.8 > max(self.base_pace, 0.2) else self.base_pace

    def _request(self, model: str, label: str, fn):
        last = None
        # 429 gets twice as many attempts as other failures: it is the one that clears if we wait.
        limit = self.max_attempts * 2 if self.max_attempts > 1 else 1
        for attempt in range(1, limit + 1):
            if attempt > self.max_attempts and not (last and last.status == 429):
                break
            t0 = time.time()
            try:
                self._pace_wait(model)
                with self.sem:
                    r = fn()
                self.meter.add(model, calls=1, seconds=time.time() - t0)
                if r.status_code == 200:
                    self._relax(model)
                    log("api_call", api=label, status=200, seconds=round(time.time() - t0, 2))
                    return r
                if r.status_code in (429, 500, 502, 503, 504):
                    wait = min(60.0, float(r.headers.get("retry-after", 0) or 0) or min(30.0, self.backoff * 2 ** (attempt - 1)))
                    body = r.text[:200].replace("\n", " ")
                    detail = f": {body}" if body and r.status_code == 429 else ""
                    last = BodhanError(f"{label}: HTTP {r.status_code}{detail}{hint(r.status_code, 'indic-' + model)}", r.status_code)
                    self.meter.add(model, retries=1)
                    if r.status_code == 429:
                        self._slow_down(model)
                    log("api_retry", api=label, status=r.status_code, attempt=attempt, wait=round(wait, 1), body=body,
                        retry_after=r.headers.get("retry-after", ""), ratelimit=r.headers.get("x-ratelimit-remaining", ""),
                        request_id=r.headers.get("x-request-id", ""))
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
                time.sleep(min(30.0, self.backoff * 2 ** (attempt - 1)))
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
    # Live path first: /v1/chat/completions (the dedicated /translate currently returns 404).
    # Order can be overridden with BODHAN_TRANSLATE_PATH=/v1/chat/completions or /translate.
    TRANSLATE_ROUTES = ("/v1/chat/completions", "/translate")

    @staticmethod
    def _translate_body(path: str, text: str, lang: str, script: str) -> dict:
        if path.endswith("/chat/completions"):
            body = {
                "model": "indic-translate",
                "messages": [{"role": "user", "content": text}],
                "target_language_code": lang,
            }
            if script != "native":
                body["target_script"] = script
            return body
        body = {"text": text, "target_language": lang}
        if script != "native":
            body["script"] = script
        return body

    def _post_translate(self, text: str, lang: str, script: str):
        paths = [os.environ["BODHAN_TRANSLATE_PATH"]] if os.environ.get("BODHAN_TRANSLATE_PATH") else list(self.TRANSLATE_ROUTES)
        start = getattr(self, "_tpath", None)
        if start in paths:
            paths.remove(start)
            paths.insert(0, start)
        err = None
        for path in paths:
            body = self._translate_body(path, text, lang, script)
            try:
                r = self._request("translate", "translate", lambda path=path, body=body: self.http.post(
                    f"{self.base}{path}", headers=self._auth(self.keys.translate), json=body))
                self._tpath = path
                return r
            except BodhanError as e:
                if e.status != 404:
                    raise
                log("translate_path_not_found", path=path, detail=str(e)[:200])
                err = e
        raise err

    def translate(self, text: str, lang: str, script: str = "native") -> dict:
        key = self._ck("mt", text, lang, script)
        hit = self._cget(key, "json")
        if hit:
            self.meter.add("translate", cache_hits=1)
            return json.loads(hit)
        j = self._post_translate(text, lang, script).json()
        try:   # chat-completions shape, else the /translate shape
            content = (j["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            content = (j.get("translation") or "").strip()
        use = j.get("usage") or {}
        tokens = use.get("completion_tokens") or use.get("output_tokens") or max(1, len(content) // 2)
        out = {"text": content, "truncated": bool(j.get("truncated")), "tokens": tokens}
        self.meter.add("translate", cost_inr=tokens / 10000 * config.PRICE_TRANSLATE_PER_10K_TOKENS)
        if not out["truncated"]:  # never cache a cut-off translation
            self._cput(key, "json", json.dumps(out, ensure_ascii=False).encode())
        return out

    # ---- Indic-OCR ----
    def ocr(self, png: bytes, max_tokens: int = 2048) -> dict:
        """One whole image in, reading-ordered blocks out: {"text", "blocks": [{order,label,type,bbox,conf,text}]}.
        POST /v1/chat/completions, model indic-ocr, image as a base64 data URI (PNG or JPEG only)."""
        if not self.keys.ocr:
            raise BodhanError("ocr: BODHAN_OCR_KEY is not set (create a key for Indic-OCR in the Bodhan console)")
        key = self._ck("ocr", hashlib.sha256(png).hexdigest())
        hit = self._cget(key, "json")
        if hit:
            self.meter.add("ocr", cache_hits=1)
            return json.loads(hit)
        uri = "data:image/png;base64," + base64.b64encode(png).decode()

        def call(tokens):
            body = {"model": "indic-ocr", "max_tokens": tokens,
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}}]}]}
            return self._request("ocr", "ocr", lambda: self.http.post(
                f"{self.base}/v1/chat/completions", headers=self._auth(self.keys.ocr), json=body))

        try:
            r = call(max_tokens)
        except BodhanError as e:
            if e.status != 502:
                raise
            log("ocr_retry_more_tokens", detail=str(e)[:160])      # a block was longer than max_tokens (e.g. a big table)
            r = call(max_tokens * 2)
        j = r.json()
        try:
            text = (j["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError):
            text = ""
        blocks = [{"order": int(b.get("order", i)), "label": str(b.get("label", "")), "type": str(b.get("type", "")),
                   "bbox": [float(v) for v in (b.get("bbox_xyxy") or [0, 0, 0, 0])][:4], "conf": float(b.get("conf", 0) or 0),
                   "text": str(b.get("text") or "")} for i, b in enumerate(j.get("blocks") or [])]
        out = {"text": text, "blocks": blocks}
        self.meter.add("ocr", cost_inr=config.PRICE_OCR_PER_IMAGE)
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
