"""FastAPI app for PDF conversion + OCR.

Copyright (C) 2026  pymu contributors
Licensed under the GNU Affero General Public License v3.0 or later.
See LICENSE and NOTICE.md.

Endpoints:
  POST /api/convert  — merge any number of files (images / PDFs / office) into
                       one PDF, with per-file rotation and an optional OCR
                       text layer.
  POST /api/ocr      — add OCR text layer to a single existing PDF.
  GET  /api/info     — public service info (license, source URL).
  GET  /             — serves the frontend (static files).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles

from backend.services import converter, job_queue, ocr, results, spacing

# Make our `pymu.*` loggers visible under uvicorn (which doesn't auto-attach
# a handler to user loggers).
_pymu_logger = logging.getLogger("pymu")
if not _pymu_logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _pymu_logger.addHandler(_h)
_pymu_logger.setLevel(logging.INFO)
_pymu_logger.propagate = False

log = _pymu_logger


def _warmup_ocr_in_background() -> None:
    def _run():
        try:
            log.info("OCR warmup: loading PaddleOCR model…")
            ocr.warmup()
            log.info("OCR warmup: ready.")
            log.info("Spacing warmup: probing Ollama…")
            spacing.warmup()
            log.info("Spacing warmup: done.")
        except Exception:
            log.exception("warmup failed (will retry on first request)")

    threading.Thread(target=_run, name="ocr-warmup", daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Kick off model load right away so the first /api/convert with OCR
    # doesn't pay the multi-second (or first-time multi-minute) cold start.
    # Runs in a background thread so the server still becomes ready quickly.
    _warmup_ocr_in_background()
    job_queue.QUEUE.start_worker(job_queue.process)
    from backend.services import stats
    stats.start_daily_summary_worker()
    yield


app = FastAPI(title="PyMu PDF Converter", lifespan=lifespan)

# Embed guard: this service is meant to be loaded as an iframe from the blog.
# Direct access and embeds from other origins are rejected.
_BLOG_HOST = "infra-oldman.tistory.com"
_ALLOWED_HOSTS = frozenset({_BLOG_HOST, "hangsil.myvnc.com"})
_CSP_FRAME_ANCESTORS = f"frame-ancestors https://{_BLOG_HOST}"


def _host_allowed(url_value: str) -> bool:
    if not url_value:
        return False
    try:
        parsed = urlparse(url_value)
    except ValueError:
        return False
    return parsed.scheme in ("http", "https") and parsed.hostname in _ALLOWED_HOSTS


@app.middleware("http")
async def _embed_guard(request: Request, call_next):
    path = request.url.path
    # /api/health stays open for uptime monitors.
    # /api/result/* is gated by an unguessable token already; let it through
    # so users opening the download in a new tab (no referer) still work.
    if path == "/api/health" or path.startswith("/api/result/"):
        return await call_next(request)

    origin = request.headers.get("origin", "")
    referer = request.headers.get("referer", "")
    if not (_host_allowed(origin) or _host_allowed(referer)):
        return Response("forbidden", status_code=403)

    response = await call_next(request)
    if "text/html" in response.headers.get("content-type", ""):
        response.headers["Content-Security-Policy"] = _CSP_FRAME_ANCESTORS
    return response


ALLOWED_ROTATIONS = {0, 90, 180, 270}
MAX_TOTAL_BYTES = 200 * 1024 * 1024  # 200 MB total per request


@app.post("/api/convert")
async def convert(
    request: Request,
    files: list[UploadFile] = File(...),
    meta: str = Form("[]"),
    ocr_enabled: bool = Form(True),
    llm_enhance: bool = Form(True),
):
    if not files:
        raise HTTPException(400, "no files uploaded")
    try:
        meta_list = json.loads(meta) if meta else []
    except json.JSONDecodeError:
        raise HTTPException(400, "meta is not valid JSON")
    if not isinstance(meta_list, list):
        raise HTTPException(400, "meta must be a JSON array")

    items: list[converter.InputItem] = []
    total = 0
    for i, up in enumerate(files):
        ext = Path(up.filename or "").suffix.lower()
        if ext not in converter.SUPPORTED_EXTS:
            raise HTTPException(400, f"unsupported file type: {up.filename}")
        data = await up.read()
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise HTTPException(413, "total upload size too large")
        m = meta_list[i] if i < len(meta_list) and isinstance(meta_list[i], dict) else {}
        rotation = int(m.get("rotation", 0)) % 360
        if rotation not in ALLOWED_ROTATIONS:
            raise HTTPException(400, f"rotation must be one of {sorted(ALLOWED_ROTATIONS)}")
        merge = bool(m.get("merge", True))
        items.append(converter.InputItem(
            filename=up.filename or f"file-{i}",
            data=data,
            rotation=rotation,
            merge=merge,
        ))

    job = job_queue.Job(
        kind="convert",
        payload={
            "items": items,
            "ocr_enabled": ocr_enabled,
            "llm_enhance": llm_enhance,
        },
        client_ip=(request.client.host if request.client else ""),
    )
    job_queue.QUEUE.submit(job)
    return StreamingResponse(
        _job_event_stream(request, job),
        media_type="application/x-ndjson",
    )


@app.post("/api/ocr")
async def ocr_pdf(
    request: Request,
    file: UploadFile = File(...),
    llm_enhance: bool = Form(True),
):
    ext = Path(file.filename or "").suffix.lower()
    if ext != ".pdf":
        raise HTTPException(400, "only .pdf accepted")
    data = await file.read()
    if len(data) > MAX_TOTAL_BYTES:
        raise HTTPException(413, "file too large")

    base = (file.filename or "ocr.pdf").rsplit("/", 1)[-1]
    out_name = base[:-4] + ".ocr.pdf" if base.lower().endswith(".pdf") else base + ".ocr.pdf"
    job = job_queue.Job(
        kind="ocr",
        payload={"pdf_bytes": data, "filename": out_name, "llm_enhance": llm_enhance},
        client_ip=(request.client.host if request.client else ""),
    )
    job_queue.QUEUE.submit(job)
    return StreamingResponse(
        _job_event_stream(request, job),
        media_type="application/x-ndjson",
    )


async def _job_event_stream(request: Request, job: "job_queue.Job"):
    """Drain Job.events queue, emit NDJSON lines, watch client disconnect.

    Also emits a 'queued' event every 2 seconds while the job is still in the
    waiting queue. Heartbeat keeps the HTTP connection alive even if the user
    waits a long time.
    """
    pos_task = asyncio.create_task(_emit_queue_position(job))
    discon_task = asyncio.create_task(_watch_disconnect(request, job))
    loop = asyncio.get_event_loop()
    try:
        # Initial heartbeat so the browser sees the stream is live.
        yield (json.dumps({"type": "queued", "position": job_queue.QUEUE.position(job) + 1, "ahead": job_queue.QUEUE.position(job)}, ensure_ascii=False) + "\n").encode("utf-8")
        while True:
            evt = await loop.run_in_executor(None, lambda: job.events.get())
            yield (json.dumps(evt, ensure_ascii=False) + "\n").encode("utf-8")
            if evt.get("type") in ("done", "cancelled", "error"):
                break
    finally:
        pos_task.cancel()
        discon_task.cancel()
        # Safety: ensure the job is marked cancelled if we exited early.
        if not job.cancelled:
            try:
                job_queue.QUEUE.cancel(job)
            except Exception:
                pass


async def _emit_queue_position(job: "job_queue.Job"):
    """Periodically push the current queue position into job.events while
    the job is still waiting. Stops once the job has started running."""
    while True:
        pos = job_queue.QUEUE.position(job)
        if pos == -1:
            return  # 처리 시작됨 (또는 cancel/done)
        job.events.put({
            "type": "queued",
            "position": pos + 1,
            "ahead": pos,
        })
        await asyncio.sleep(2)


async def _watch_disconnect(request: Request, job: "job_queue.Job"):
    while True:
        try:
            if await request.is_disconnected():
                job_queue.QUEUE.cancel(job)
                return
        except Exception:
            return
        await asyncio.sleep(1)


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/info")
def info():
    """Public service info used by the frontend (license/source links)."""
    return {
        "license": "AGPL-3.0",
        "source_url": os.environ.get("PYMU_SOURCE_URL", ""),
    }


@app.get("/api/result/{token}")
def fetch_result(token: str):
    """Return a converted file by its one-time token.

    Tokens expire after results.RESULT_TTL_SECONDS; trying to fetch a
    missing or expired token returns 404.
    """
    from urllib.parse import quote
    entry = results.get(token)
    if entry is None:
        raise HTTPException(404, "result not found or expired")
    ascii_name = entry.filename.encode("ascii", errors="replace").decode("ascii").replace('"', "")
    disposition = f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(entry.filename)}"
    return Response(
        content=entry.data,
        media_type=entry.mime,
        headers={
            "Content-Disposition": disposition,
            "X-Content-Type-Options": "nosniff",
            "Cache-Control": "no-store",
        },
    )


# Mount the frontend last so /api/* takes precedence.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")
