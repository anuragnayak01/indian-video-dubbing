"""Thin web UI over the same pipeline. No dubbing logic lives here: it uploads, starts `run_pipeline`, reads the
logs/report/segments the pipeline wrote, and saves edits as per-segment overrides that the next (cached) run applies.
"""
import hmac
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import yaml
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

from .. import config, logs, textutil
from ..bodhan import BodhanClient
from ..llm import Brain
from ..pipeline import Options, run_pipeline

DATA = Path(os.environ.get("BODHAN_DUB_WEB_DATA", "./webdata")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
HERE = Path(__file__).parent
app = FastAPI(title="Bodhan Dub Studio")
ACCESS_TOKEN = os.environ.get("DUB_ACCESS_TOKEN", "")          # set it whenever the server is public
CORS_ORIGINS = [o.strip().rstrip("/") for o in os.environ.get("DUB_CORS_ORIGINS", "").split(",") if o.strip()]


@app.middleware("http")
async def require_token(request: Request, call_next):
    """Everything that spends Bodhan credit or lists projects needs the token. Media GETs (/file/) are authorised by
    the unguessable project id so <video>/<audio> tags work cross-origin without headers."""
    p = request.url.path
    open_ = request.method in ("OPTIONS", "HEAD") or p in ("/", "/config.js", "/api/health") or (
        request.method == "GET" and p.startswith("/api/projects/") and "/file/" in p)
    if ACCESS_TOKEN and p.startswith("/api") and not open_:
        if not hmac.compare_digest(request.headers.get("x-access-token", ""), ACCESS_TOKEN):
            return JSONResponse({"detail": "Access token required"}, status_code=401)
    return await call_next(request)


if CORS_ORIGINS:      # added last so it is the outermost layer and answers preflights first
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                       allow_headers=["Content-Type", "X-Access-Token"], max_age=600)
JOB_LOCK = threading.Lock()      # the logger is process-wide, so jobs run one at a time
JOBS: dict[str, dict] = {}
OVERRIDE_FIELDS = {"text", "english", "voice", "style", "speed", "max_stretch"}


def pdir(pid: str) -> Path:
    if not pid.isalnum():
        raise HTTPException(404)
    d = DATA / pid
    if not d.exists():
        raise HTTPException(404, "project not found")
    return d


def check_lang(lang: str) -> str:
    if lang not in config.LANGUAGES:
        raise HTTPException(404, "unknown language")
    return lang


def meta(pid: str) -> dict:
    return json.loads((pdir(pid) / "meta.json").read_text())


def save_meta(pid: str, m: dict) -> None:
    textutil.atomic_write(DATA / pid / "meta.json", json.dumps(m, ensure_ascii=False, indent=1))


def start_job(pid: str, langs: list[str], merge: bool) -> None:
    m = meta(pid)
    out = DATA / pid / "out"
    JOBS[pid] = {"state": "queued", "langs": langs, "error": "", "log_offset": 0, "started": time.time()}

    def work():
        with JOB_LOCK:
            job = JOBS[pid]
            job["state"] = "running"
            log_file = out / "run.log.jsonl"
            job["log_offset"] = log_file.stat().st_size if log_file.exists() else 0
            try:
                keys = config.Keys.from_env()
                if keys.missing():
                    raise RuntimeError("Missing environment variables: " + ", ".join(keys.missing()))
                out.mkdir(exist_ok=True)
                logs.setup(log_file, echo=False)
                client = BodhanClient(keys, out / "work" / "cache", max_concurrency=m.get("concurrency", 4))
                gl, vo = DATA / pid / "glossary.yaml", DATA / pid / "voices.yaml"
                opt = Options(input=DATA / pid / m["input"], langs=langs, out=out, glossary=gl if gl.exists() else None,
                              voices=vo if vo.exists() else None, code_mixed=m.get("code_mixed", False),
                              concurrency=m.get("concurrency", 4), style=m.get("style") or None, merge=merge)
                rc = run_pipeline(opt, client, Brain())
                job["state"] = "done" if rc == 0 else "finished_with_errors"
            except Exception as e:  # surface any failure to the UI instead of dying silently
                job["state"], job["error"] = "error", str(e)[:400]
            job["finished"] = time.time()

    threading.Thread(target=work, daemon=True).start()


def read_log(pid: str) -> list[dict]:
    f = DATA / pid / "out" / "run.log.jsonl"
    job = JOBS.get(pid)
    if not f.exists() or not job:
        return []
    with open(f, "rb") as fh:
        fh.seek(job["log_offset"])
        raw = fh.read().decode("utf-8", "ignore")
    out = []
    for line in raw.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def progress(pid: str) -> dict:
    ev = read_log(pid)
    stages = [e["stage"] for e in ev if e.get("event") == "stage"]
    langs: dict[str, dict] = {}
    for e in ev:
        if "lang" not in e:
            continue
        L = langs.setdefault(e["lang"], {"attempts": 0, "idx": set(), "stage": "translate+speak", "total": None, "retries": 0})
        if e["event"] == "segment_attempt":
            L["idx"].add(e["idx"])
            L["attempts"] += 1
        elif e["event"] == "lang_stage":
            L["stage"] = e["stage"]
            if e["stage"] == "segments":
                L["total"] = e.get("total")
        elif e["event"] in ("condensed", "segment_failed"):
            L["retries"] += 1
        elif e["event"] == "language_failed":
            L["stage"] = "failed"
    for L in langs.values():
        L["done"] = len(L.pop("idx"))
    retries = [e for e in ev if e.get("event") == "api_retry"]
    last = retries[-1] if retries else {}
    last_retry = f'{last.get("api", "")}: ' + (f'HTTP {last["status"]}' if last.get("status") else str(last.get("error", ""))) if last else ""
    return {"source_stages": stages, "languages": langs, "api_retries": len(retries), "last_retry": last_retry}


# ------------------------------------------------------------------ API
@app.get("/api/health")
def health():
    return {"ok": True, "auth": bool(ACCESS_TOKEN)}


@app.get("/config.js")
def config_js():
    return Response("window.DUB_API='';", media_type="application/javascript")


@app.get("/api/config")
def get_config():
    keys = config.Keys.from_env()
    return {"languages": [{"code": k, "label": v} for k, v in config.LANGUAGES.items()], "voices": config.VOICES,
            "styles": config.STYLES, "missing_keys": keys.missing(), "llm": Brain().available,
            "stretch": [config.STRETCH_MIN, config.STRETCH_MAX], "default_langs": ["hi", "ta", "bn"]}


@app.get("/api/projects")
def list_projects():
    out = []
    for d in sorted(DATA.iterdir(), key=lambda p: -p.stat().st_mtime):
        if (d / "meta.json").exists():
            m = json.loads((d / "meta.json").read_text())
            out.append({**m, "id": d.name, "job": JOBS.get(d.name, {}).get("state", "idle")})
    return out


@app.post("/api/projects")
async def create_project(file: UploadFile = File(...), langs: str = Form("hi,ta,bn"), name: str = Form(""),
                         code_mixed: bool = Form(False), style: str = Form("")):
    lang_list = [x for x in (s.strip().lower() for s in langs.split(",")) if x]
    bad = [x for x in lang_list if x not in config.LANGUAGES]
    if not lang_list or bad:
        raise HTTPException(400, f"Choose at least one supported language. Unknown: {bad}")
    if style and style not in config.STYLES:
        raise HTTPException(400, "Unknown style")
    ext = Path(file.filename or "v.mp4").suffix.lower()
    if ext not in (".mp4", ".mov", ".mkv", ".webm"):
        raise HTTPException(400, "Upload an MP4, MOV, MKV or WebM video")
    pid = uuid.uuid4().hex[:10]
    d = DATA / pid
    d.mkdir()
    with open(d / f"input{ext}", "wb") as fh:
        while chunk := await file.read(1 << 20):
            fh.write(chunk)
    m = {"name": name or Path(file.filename or "video").stem, "input": f"input{ext}", "langs": lang_list, "code_mixed": code_mixed,
         "style": style, "created": time.time(), "concurrency": 4}
    save_meta(pid, m)
    start_job(pid, lang_list, merge=False)
    return {"id": pid, **m}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    d = pdir(pid)
    out = {"id": pid, **meta(pid), "job": JOBS.get(pid, {"state": "idle"}), "progress": progress(pid)}
    for key, fn in (("report", "out/report.json"), ("transcript", "out/transcript.en.json")):
        f = d / fn
        out[key] = json.loads(f.read_text()) if f.exists() else None
    gl, vo = d / "glossary.yaml", d / "voices.yaml"
    out["user_glossary"] = (yaml.safe_load(gl.read_text()) or {}).get("terms", []) if gl.exists() else []
    out["voices_cfg"] = yaml.safe_load(vo.read_text()) if vo.exists() else {}
    return out


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    shutil.rmtree(pdir(pid))
    JOBS.pop(pid, None)
    return {"ok": True}


@app.get("/api/projects/{pid}/lang/{lang}")
def get_lang(pid: str, lang: str):
    f = pdir(pid) / "out" / check_lang(lang) / f"segments.{lang}.json"
    if not f.exists():
        raise HTTPException(404, "not ready")
    ov = pdir(pid) / "out" / lang / "overrides.json"
    return {**json.loads(f.read_text()), "overrides": json.loads(ov.read_text()) if ov.exists() else {}}


def busy(pid: str) -> bool:
    return JOBS.get(pid, {}).get("state") in ("queued", "running")


@app.patch("/api/projects/{pid}/lang/{lang}/segments/{idx}")
def edit_segment(pid: str, lang: str, idx: int, body: dict = Body(...)):
    if busy(pid):
        raise HTTPException(409, "A run is in progress; wait for it to finish")
    d = pdir(pid) / "out" / check_lang(lang)
    clean = {k: v for k, v in body.items() if k in OVERRIDE_FIELDS and v not in (None, "")}
    if "speed" in clean:
        clean["speed"] = min(config.STRETCH_MAX, max(config.STRETCH_MIN, float(clean["speed"])))
    if "style" in clean and clean["style"] not in config.STYLES:
        raise HTTPException(400, "Unknown style")
    f = d / "overrides.json"
    ov = json.loads(f.read_text()) if f.exists() else {}
    if body.get("reset"):
        ov.pop(str(idx), None)
    else:
        ov[str(idx)] = clean
    textutil.atomic_write(f, json.dumps(ov, ensure_ascii=False, indent=1))
    start_job(pid, [lang], merge=True)
    return {"ok": True}


@app.put("/api/projects/{pid}/glossary")
def put_glossary(pid: str, body: dict = Body(...)):
    if busy(pid):
        raise HTTPException(409, "A run is in progress; wait for it to finish")
    terms = [{"term": t["term"].strip(), "pin": bool(t.get("pin", True)), "kind": t.get("kind", "term"),
              "translations": {k: v for k, v in (t.get("translations") or {}).items() if v and k in config.LANGUAGES}}
             for t in body.get("terms", []) if str(t.get("term", "")).strip()]
    (pdir(pid) / "glossary.yaml").write_text(yaml.safe_dump({"terms": terms}, allow_unicode=True))
    start_job(pid, meta(pid)["langs"], merge=True)
    return {"ok": True, "terms": len(terms)}


@app.put("/api/projects/{pid}/voices")
def put_voices(pid: str, body: dict = Body(...)):
    if busy(pid):
        raise HTTPException(409, "A run is in progress; wait for it to finish")
    spk = {k: v for k, v in (body.get("speakers") or {}).items() if v}
    (pdir(pid) / "voices.yaml").write_text(yaml.safe_dump({"speakers": spk}, allow_unicode=True))
    start_job(pid, meta(pid)["langs"], merge=True)
    return {"ok": True}


@app.post("/api/projects/{pid}/rerun")
def rerun(pid: str):
    if busy(pid):
        raise HTTPException(409, "A run is in progress")
    start_job(pid, meta(pid)["langs"], merge=True)
    return {"ok": True}


@app.get("/api/projects/{pid}/file/{kind}")
def get_file(pid: str, kind: str, lang: str = "", idx: int = -1):
    d = pdir(pid)
    m = meta(pid)
    paths = {"original": d / m["input"], "report": d / "out" / "report.json", "transcript": d / "out" / "transcript.en.json",
             "log": d / "out" / "run.log.jsonl"}
    if kind in ("video", "srt", "segments", "clip", "mix"):
        check_lang(lang)
        paths.update({"video": d / "out" / lang / f"video.{lang}.mp4", "srt": d / "out" / lang / f"subtitles.{lang}.srt",
                      "segments": d / "out" / lang / f"segments.{lang}.json",
                      "clip": d / "out" / "work" / lang / f"seg{idx}.wav", "mix": d / "out" / "work" / lang / "mix.wav"})
    f = paths.get(kind)
    if not f or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, headers={"Cache-Control": "no-store"})


@app.get("/")
def index():
    return FileResponse(HERE / "index.html")
