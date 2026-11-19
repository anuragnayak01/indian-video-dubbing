"""Web UI backend: validation, upload -> dub -> edit one line -> re-run, using the offline fake API."""
import importlib
import time

import pytest
from fake_bodhan import FakeBodhan
from fastapi.testclient import TestClient

from bodhan_dub import config
from bodhan_dub.bodhan import BodhanClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("BODHAN_DUB_WEB_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("BODHAN_API_KEY", "x")
    for k in ("BODHAN_TRANSCRIBE_KEY", "BODHAN_TRANSLATE_KEY", "BODHAN_SPEAK_KEY", "LLM_BASE_URL"):
        monkeypatch.delenv(k, raising=False)
    from bodhan_dub.web import server
    server = importlib.reload(server)
    fake = FakeBodhan()
    monkeypatch.setattr(server, "BodhanClient", lambda keys, cache, max_concurrency=4: BodhanClient(
        keys, cache, max_concurrency=1, http=fake.client, base_url="https://api.bodhan.ai", backoff=0.01))
    return TestClient(server.app), server


def wait(c, pid, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/projects/{pid}").json()
        if j["job"]["state"] not in ("queued", "running"):
            return j
        time.sleep(0.4)
    raise TimeoutError


def test_validation(client, video):
    c, _ = client
    assert c.get("/api/config").json()["missing_keys"] == []
    assert c.get("/api/projects/..%2f..").status_code == 404
    with open(video, "rb") as f:
        assert c.post("/api/projects", files={"file": ("a.mp4", f)}, data={"langs": "xx"}).status_code == 400
    assert c.post("/api/projects", files={"file": ("a.txt", b"hi")}, data={"langs": "hi"}).status_code == 400


def test_upload_edit_rerun(client, video):
    c, _ = client
    with open(video, "rb") as f:
        r = c.post("/api/projects", files={"file": ("lesson.mp4", f)}, data={"langs": "hi,ta", "name": "Lesson"})
    pid = r.json()["id"]
    j = wait(c, pid)
    assert j["job"]["state"] == "done" and set(j["report"]["languages"]) == {"hi", "ta"}
    d = c.get(f"/api/projects/{pid}/lang/hi").json()
    assert len(d["segments"]) == 3 and d["ai_dubbed"]
    assert c.get(f"/api/projects/{pid}/file/video", params={"lang": "hi"}).status_code == 200
    assert c.get(f"/api/projects/{pid}/file/clip", params={"lang": "hi", "idx": 0}).status_code == 200
    assert c.get(f"/api/projects/{pid}/file/video", params={"lang": "zz"}).status_code == 404

    cost_before = j["report"]["languages"]["ta"]["cost"]["total_cost_inr"]
    assert c.patch(f"/api/projects/{pid}/lang/hi/segments/1",
                   json={"text": "मेरा अनुवाद", "speed": 5, "voice": "Amit", "evil": "x"}).status_code == 200
    j2 = wait(c, pid)
    s1 = c.get(f"/api/projects/{pid}/lang/hi").json()["segments"][1]
    assert s1["translation"] == "मेरा अनुवाद" and s1["voice"] == "Amit"
    assert s1["stretch"] == config.STRETCH_MAX           # speed 5 was clamped to the allowed range
    assert c.get(f"/api/projects/{pid}/lang/hi").json()["overrides"]["1"].keys() == {"text", "speed", "voice"}
    assert j2["report"]["languages"]["ta"]["cost"]["total_cost_inr"] == cost_before   # other language untouched, still reported

    assert c.put(f"/api/projects/{pid}/glossary", json={"terms": [{"term": "sentence", "translations": {"hi": "वाक्य"}}]}).status_code == 200
    assert wait(c, pid)["user_glossary"][0]["translations"] == {"hi": "वाक्य"}
    assert c.delete(f"/api/projects/{pid}").status_code == 200


def test_token_and_cors(tmp_path, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    from bodhan_dub.web import server
    monkeypatch.setenv("DUB_ACCESS_TOKEN", "s3cret")
    monkeypatch.setenv("DUB_CORS_ORIGINS", "https://app.vercel.app")
    monkeypatch.setenv("BODHAN_DUB_WEB_DATA", str(tmp_path))
    s = importlib.reload(server)
    try:
        c = TestClient(s.app)
        assert c.get("/api/health").json()["auth"] is True
        assert c.get("/api/projects").status_code == 401
        assert c.get("/api/projects", headers={"X-Access-Token": "s3cret"}).status_code == 200
        assert c.post("/api/projects/abc/rerun").status_code == 401
        pre = c.options("/api/projects", headers={"Origin": "https://app.vercel.app", "Access-Control-Request-Method": "POST",
                                                  "Access-Control-Request-Headers": "x-access-token"})
        assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "https://app.vercel.app"
    finally:
        monkeypatch.delenv("DUB_ACCESS_TOKEN")
        monkeypatch.delenv("DUB_CORS_ORIGINS")
        importlib.reload(server)


def test_diagnose_reports_keys_and_statuses(client, monkeypatch):
    import httpx
    client, server = client
    monkeypatch.setenv("BODHAN_TRANSLATE_KEY", "same-key-1234")
    monkeypatch.setenv("BODHAN_TRANSCRIBE_KEY", "same-key-1234")
    monkeypatch.setattr(httpx, "post", lambda *a, **k: httpx.Response(404, json={"detail": "Not Found"}, headers={"x-request-id": "r1"}))
    monkeypatch.setattr(server, "BodhanClient", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("speak: HTTP 404")))
    j = client.post("/api/diagnose").json()
    assert j["translate_key_same_as_transcribe"] is True and j["keys"]["translate"]["last4"] == "1234"
    assert [t["status"] for t in j["tests"][:2]] == [404, 404] and "error" in j["tests"][2]


def test_diagnose_get_needs_query_token_and_post_needs_header(tmp_path, monkeypatch):
    import importlib

    from fastapi.testclient import TestClient

    from bodhan_dub.web import server
    monkeypatch.setenv("DUB_ACCESS_TOKEN", "tok")
    monkeypatch.setenv("BODHAN_DUB_WEB_DATA", str(tmp_path))
    s = importlib.reload(server)
    try:
        c = TestClient(s.app)
        assert c.get("/api/diagnose").status_code == 401
        assert c.post("/api/diagnose").status_code == 401
        assert c.get("/api/diagnose?token=bad").status_code == 401
    finally:
        monkeypatch.delenv("DUB_ACCESS_TOKEN")
        importlib.reload(server)
