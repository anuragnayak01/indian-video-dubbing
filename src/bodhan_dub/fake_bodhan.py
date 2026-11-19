"""An in-process fake of the Bodhan hosted API (httpx MockTransport) for offline tests.

It is a test double only: the pipeline never falls back to it. TTS returns a tone whose pitch encodes the text and whose
length depends on the text length. STT recognises those tones by pitch (pitch survives time-stretching) and returns the
text, so the real back-transcription loop runs. Doubled audio comes back as the text twice, which is how a "repeat"
failure is simulated. Anything that is not one of our tones is "source speech" and returns `stt_text`.
"""
import io
import json
import zlib

import httpx
import numpy as np
import soundfile as sf

SR = 24000


def freq_for(text: str) -> float:
    return 400.0 + (zlib.crc32(text.encode()) % 1500) * 2.0


def tone(seconds: float, f: float) -> bytes:
    t = np.arange(int(SR * seconds)) / SR
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * f * t)).astype("float32"), SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class FakeBodhan:
    def __init__(self, stt_text=None, chars_per_sec: float = 14.0, repeat_once: set | None = None, truncate_once: set | None = None, fail_langs: set | None = None):
        self.calls = {"stt": 0, "mt": 0, "tts": 0}
        self.stt_text = stt_text or (lambda n: f"this is spoken sentence number {n}.")
        self.cps = chars_per_sec
        self.repeat_once, self.truncate_once = repeat_once or set(), truncate_once or set()
        self.done: set[str] = set()
        self.table: dict[float, tuple[str, float]] = {}   # freq -> (text, nominal seconds)
        self.n_source = 0
        self.fail_langs = fail_langs or set()
        self.mt_log: list[str] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self.handle), base_url="https://api.bodhan.ai")

    def handle(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path

        # Live path: OpenAI-compatible chat completions (dedicated /translate currently returns 404)
        if path == "/v1/chat/completions":
            self.calls["mt"] += 1
            body = json.loads(req.content)
            if body.get("model") != "indic-translate":
                return httpx.Response(400, json={"error": {"message": "unknown model"}})
            text = body["messages"][0]["content"]
            lang = body.get("target_language_code") or body.get("target_language") or "hi"
            self.mt_log.append(text)
            if lang in self.fail_langs:
                return httpx.Response(400, json={"error": {"message": "unsupported"}})
            translated = f"[{lang}] {text}"
            return httpx.Response(200, json={
                "id": "fake-mt",
                "object": "chat.completion",
                "model": "indic-translate",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": translated},
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": max(1, len(text) // 4),
                    "completion_tokens": max(1, len(translated) // 3),
                    "total_tokens": max(1, len(text) // 4) + max(1, len(translated) // 3),
                },
            })

        # Keep old /translate for any residual callers / docs examples
        if path == "/translate":
            self.calls["mt"] += 1
            body = json.loads(req.content)
            self.mt_log.append(body["text"])
            if body["target_language"] in self.fail_langs:
                return httpx.Response(400, json={"error": {"message": "unsupported"}})
            return httpx.Response(200, json={"translation": f"[{body['target_language']}] " + body["text"], "truncated": False,
                                             "usage": {"output_tokens": max(1, len(body["text"]) // 3)}})

        if path == "/v1/audio/speech":
            self.calls["tts"] += 1
            text = json.loads(req.content)["input"]
            nominal = max(0.4, len(text) / self.cps)
            f = freq_for(text)
            self.table[f] = (text, nominal)
            dur = nominal
            if text in self.repeat_once and ("r", text) not in self.done:
                self.done.add(("r", text))
                dur = nominal * 2
            if text in self.truncate_once and ("t", text) not in self.done:
                self.done.add(("t", text))
                dur = nominal * 0.35
            return httpx.Response(200, content=tone(dur, f), headers={"content-type": "audio/wav"})

        if path == "/v1/audio/transcriptions":
            self.calls["stt"] += 1
            return httpx.Response(200, json={"text": self._stt(req.content)})

        return httpx.Response(404, json={"error": {"message": "not found"}})

    def _stt(self, content: bytes) -> str:
        a = content.find(b"RIFF")
        b = content.rfind(b"\r\n--")
        try:
            x, sr = sf.read(io.BytesIO(content[a:b]), dtype="float32")
        except Exception:
            return ""
        if len(x) == 0 or float(np.sqrt(np.mean(x ** 2))) < 0.02:
            return ""  # silence or music: nothing said
        dur = len(x) / sr
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
        peak = float(np.fft.rfftfreq(len(x), 1 / sr)[int(spec.argmax())])
        if self.table:
            f = min(self.table, key=lambda k: abs(k - peak))
            if abs(f - peak) < 6:
                text, nominal = self.table[f]
                if dur > 1.6 * nominal / 1.2:      # clip is about twice as long as it should be: spoken twice
                    return f"{text} {text}"
                if dur < 0.6 * nominal / 1.2 * 1.0 and dur < nominal * 0.5:  # cut short: only the first words
                    return " ".join(text.split()[: max(1, len(text.split()) // 3)])
                return text
        self.n_source += 1
        return self.stt_text(self.n_source)
