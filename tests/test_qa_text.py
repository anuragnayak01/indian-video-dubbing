from bodhan_dub import qa, textutil
from bodhan_dub.llm import Brain, contains_indic


def test_cer_basics():
    assert qa.cer("नमस्ते दुनिया", "नमस्ते दुनिया") == 0.0
    assert 0 < qa.cer("hello world", "hello word") < 0.2


def test_repeat_and_omission_detection():
    ref = "आज हम त्रिभुज का क्षेत्रफल सीखेंगे"
    assert qa.assess(ref, ref)["ok"]
    rep = qa.assess(ref, ref + " " + ref)
    assert "repeat" in rep["flags"]
    om = qa.assess(ref, "आज हम")
    assert "omission" in om["flags"]


def test_legit_repeated_phrase_is_not_flagged():
    ref = "बार बार बार बार कहो बार बार बार बार"
    assert "repeat" not in qa.assess(ref, ref)["flags"]


def test_numbers_preserved_across_scripts():
    assert qa.numbers_preserved("the sum is 3,000 and 45", "योग 3000 और ४५ है") == []
    assert qa.numbers_preserved("take 12 apples", "सेब लो") == ["12"]


def test_split_for_tts_keeps_words_whole_and_short():
    text = "यह पहला वाक्य है। " * 30
    parts = textutil.split_for_tts(text, 120)
    assert all(len(p) <= 125 for p in parts)
    assert " ".join(parts).split() == text.split()


def test_glossary_user_translation_substitution():
    gl = [{"term": "Pythagoras theorem", "pin": True, "translations": {"hi": "पाइथागोरस प्रमेय"}}]
    t, used = textutil.apply_user_translations("We prove the Pythagoras theorem today", gl, "hi")
    assert "पाइथागोरस प्रमेय" in t and used == ["Pythagoras theorem"]
    t2, used2 = textutil.apply_user_translations("We prove the Pythagoras theorem today", gl, "ta")
    assert used2 == [] and t2 == "We prove the Pythagoras theorem today"


def test_llm_output_with_indic_script_is_rejected():
    from conftest import FakeLLM
    llm = FakeLLM()
    brain = Brain(client=llm, model="m")
    llm.shorten = False
    assert contains_indic("नमस्ते") and not contains_indic("hello")
    assert brain.condense("A long English sentence here", 0.5, []) is None   # reply was not shorter


def test_invalid_llm_grouping_falls_back_to_heuristic():
    from conftest import FakeLLM
    chunks = [{"start": 0, "end": 2}, {"start": 3, "end": 5}]
    groups, how = Brain(client=FakeLLM(bad_groups=True), model="m").group_sentences(["one.", "two."], chunks)
    assert how == "heuristic" and groups == [[0], [1]]


def test_voice_selection_uses_the_languages_own_voices_and_overrides():
    from bodhan_dub.pipeline import pick_voice
    assert pick_voice({}, "ta", "S1", "female") == "Anitha"
    assert pick_voice({}, "ta", "S2", "male") == "Arun"
    assert pick_voice({}, "hi", "S2", "female", ["S1", "S2"]) == "Suhani"       # two women: second Hindi voice
    assert pick_voice({"female": "X"}, "hi", "S1", "female") == "X"
    assert pick_voice({"speakers": {"S1": "Zed"}, "female": "X"}, "hi", "S1", "female") == "Zed"
    assert pick_voice({}, "xx", "S1", "male") == "Amit"


def test_translate_falls_back_to_v1_path_on_404():
    import httpx

    from bodhan_dub import config
    from bodhan_dub.bodhan import BodhanClient
    seen = []

    def handler(req):
        seen.append(req.url.path)
        if req.url.path == "/translate":
            return httpx.Response(404, json={"detail": "Not Found"})
        return httpx.Response(200, json={"translation": "नमस्ते", "truncated": False, "usage": {"completion_tokens": 3}})

    keys = config.Keys("a", "b", "c")
    c = BodhanClient(keys, None, http=httpx.Client(transport=httpx.MockTransport(handler)), backoff=0)
    assert c.translate("hello", "hi")["text"] == "नमस्ते"
    assert seen == ["/translate", "/v1/translate"]
    c.translate("again", "hi")
    assert seen[-1] == "/v1/translate" and seen.count("/translate") == 1     # remembers the working path


def test_preflight_fails_fast_with_hint_on_wrong_key():
    import httpx
    import pytest

    from bodhan_dub import config
    from bodhan_dub.bodhan import BodhanClient, BodhanError
    from bodhan_dub.pipeline import preflight
    calls = []

    def handler(req):
        calls.append(req.url.path)
        return httpx.Response(404, json={"detail": "Not Found"})

    c = BodhanClient(config.Keys("a", "b", "c"), None, http=httpx.Client(transport=httpx.MockTransport(handler)), backoff=0)
    with pytest.raises(BodhanError) as e:
        preflight(c, "hi")
    assert "indic-translate" in str(e.value) and "each Bodhan key serves one model" in str(e.value)
    assert set(calls) == {"/translate", "/v1/translate"}      # nothing else was called, no transcription spend


def test_console_log_line_is_readable_and_flags_errors(capsys, tmp_path):
    from bodhan_dub import logs
    logs.setup(tmp_path / "r.jsonl", echo=True, tag="abc123")
    logs.log("api_error", api="translate", status=404, body='{"detail":"Not Found"}')
    logs.log("stage", stage="vad", spans=3)
    out = capsys.readouterr().out.splitlines()
    assert "ERROR [abc123] api_error api=translate status=404" in out[0]
    assert "INFO  [abc123] stage stage=vad spans=3" in out[1]
    logs.setup(None, echo=False)
