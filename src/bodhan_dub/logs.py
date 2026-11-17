"""Structured JSON-lines logging, one record per stage or per segment event."""
import json
import threading
import time
from pathlib import Path

_lock = threading.Lock()
_path: Path | None = None
_echo = True


def setup(path: Path | None, echo: bool = True) -> None:
    global _path, _echo
    _path, _echo = path, echo
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)


def log(event: str, **fields) -> None:
    rec = {"t": round(time.time(), 3), "event": event, **fields}
    line = json.dumps(rec, ensure_ascii=False, default=str)
    with _lock:
        if _path:
            with open(_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        if _echo:
            short = {k: v for k, v in rec.items() if k != "t"}
            print(json.dumps(short, ensure_ascii=False, default=str)[:300], flush=True)
