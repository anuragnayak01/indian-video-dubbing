"""The workflow's "brain": an OpenAI-compatible LLM used only for judgement on the English side.

Allowed here: grouping fragments into sentences, building a glossary, shortening English, judging a
dubbed line. Never allowed: writing Indian-language text, transcribing or synthesizing speech. Output that
contains Indic script is rejected. Every call has a deterministic fallback so the pipeline runs without an LLM.
"""
import json
import os
import re

from . import segmentation
from .logs import log

INDIC = re.compile(r"[ऀ-෿؀-ۿ᱐-᱿]")


def contains_indic(s: str) -> bool:
    return bool(INDIC.search(s or ""))


def _json(text: str):
    m = re.search(r"[\[{].*[\]}]", text, re.S)
    return json.loads(m.group(0)) if m else None


class Brain:
    def __init__(self, client=None, model: str | None = None):
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.client = client
        if client is None and os.environ.get("LLM_BASE_URL") and os.environ.get("LLM_API_KEY") and self.model:
            from openai import OpenAI
            self.client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])
        self.calls = 0
        self.last_error = ""

    @property
    def available(self) -> bool:
        return self.client is not None

    def _ask(self, system: str, user: str):
        self.calls += 1
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            r = self.client.chat.completions.create(model=self.model, temperature=0, messages=msgs)
        except Exception:  # some providers reject sampling parameters; retry with defaults
            r = self.client.chat.completions.create(model=self.model, messages=msgs)
        return r.choices[0].message.content or ""

    # ---- decision 1: sentence grouping ----
    def group_sentences(self, texts: list[str], chunks: list[dict]) -> tuple[list[list[int]], str]:
        fallback = segmentation.heuristic_groups(texts, chunks)
        if not self.available or len(texts) < 2:
            return fallback, "heuristic"
        try:
            lines = "\n".join(f"{i}: [{chunks[i]['start']:.1f}-{chunks[i]['end']:.1f}s] {t}" for i, t in enumerate(texts))
            out = _json(self._ask(
                "You regroup consecutive speech fragments from an English transcript into complete sentences so each "
                "group can be translated on its own. Keep order. Never split a fragment. Do not exceed 20 seconds per "
                'group. Reply only with JSON: {"groups": [[0,1],[2],...]}.', lines))
            groups = out["groups"] if isinstance(out, dict) else out
            if segmentation.valid_groups(groups, len(texts), chunks):
                return groups, "llm"
            self.last_error = "the LLM returned an invalid grouping"
            log("llm_rejected", decision="group_sentences", reason="invalid partition")
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {str(e)[:160]}"
            log("llm_failed", decision="group_sentences", error=str(e)[:150])
        return fallback, "heuristic"

    # ---- decision 2: glossary ----
    def build_glossary(self, full_text: str) -> list[dict]:
        if not self.available:
            return []
        try:
            out = _json(self._ask(
                "List terms in this English lesson transcript that must be translated the same way every time: "
                "person and place names, acronyms, technical or mathematical terms, units. Copy each term exactly "
                "as written. pin=true for names and technical terms that should keep one fixed rendering. At most 40. "
                'Reply only with JSON: {"terms": [{"term": "...", "kind": "name|acronym|term|unit", "pin": true}]}.',
                full_text[:12000]))
            terms = out["terms"] if isinstance(out, dict) else out
            low = full_text.lower()
            return [t for t in terms if isinstance(t, dict) and t.get("term") and t["term"].lower() in low][:40]
        except Exception as e:
            log("llm_failed", decision="build_glossary", error=str(e)[:150])
            return []

    # ---- decision 3: shorten English when the dub is too long ----
    def condense(self, english: str, ratio: float, keep_terms: list[str]) -> str | None:
        if not self.available:
            return None
        try:
            t = self._ask(
                f"Rewrite the English sentence to about {int(ratio * 100)}% of its length. Keep the meaning, all numbers "
                f"and these terms unchanged: {keep_terms}. Plain spoken English only. Reply with the sentence only.",
                english).strip().strip('"')
            if t and len(t) < len(english) and not contains_indic(t):
                return t
        except Exception as e:
            log("llm_failed", decision="condense", error=str(e)[:150])
        return None

    # ---- decision 4: judge a borderline dubbed line (reads target text, never writes it) ----
    def judge(self, english: str, translation: str, back_transcript: str) -> bool | None:
        if not self.available:
            return None
        try:
            out = _json(self._ask(
                "You check a dubbed line. Compare the English source, the intended translation and what the speech "
                "synthesizer actually said (a transcript of it). Say ok=false only if content is missing, repeated or "
                'changed in meaning. Reply only with JSON: {"ok": true|false}.',
                json.dumps({"english": english, "translation": translation, "spoken": back_transcript},
                           ensure_ascii=False)))
            return bool(out["ok"])
        except Exception as e:
            log("llm_failed", decision="judge", error=str(e)[:150])
            return None
