"""In-memory store for converted result files, fetched by one-time token.

Why: cross-origin iframes don't reliably trigger blob-URL downloads
(Chrome ignores `a.click()` after user activation expires and may
ignore the `download` attribute entirely depending on policy). Handing
the frontend a real backend URL — served with Content-Disposition:
attachment — sidesteps every iframe download quirk.

Each entry expires after RESULT_TTL_SECONDS even if never fetched.
Tokens are 24 random bytes (~192 bits), so they're effectively
unguessable and safe to share within the TTL window.
"""
from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass

RESULT_TTL_SECONDS = 600  # 10 minutes


@dataclass
class Entry:
    data: bytes
    mime: str
    filename: str
    expires_at: float


_lock = threading.Lock()
_store: dict[str, Entry] = {}


def put(data: bytes, mime: str, filename: str) -> str:
    _gc_expired()
    token = secrets.token_urlsafe(24)
    with _lock:
        _store[token] = Entry(
            data=data,
            mime=mime,
            filename=filename,
            expires_at=time.time() + RESULT_TTL_SECONDS,
        )
    return token


def get(token: str) -> Entry | None:
    _gc_expired()
    with _lock:
        entry = _store.get(token)
    if entry is None or entry.expires_at < time.time():
        return None
    return entry


def _gc_expired() -> None:
    now = time.time()
    with _lock:
        expired = [t for t, e in _store.items() if e.expires_at < now]
        for t in expired:
            _store.pop(t, None)
