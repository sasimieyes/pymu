"""Single-worker FIFO job queue for /api/convert and /api/ocr.

Why: paddle + ollama 모두 GPU 메모리를 공유하고, 동시 다중 변환은 OOM 또는
sequential serialization 으로 어차피 큰 이득 없음. 한 번에 하나씩 처리하면서
대기자에게 큐 위치를 streaming 으로 알려주는 게 훨씬 깔끔하다.

설계:
- Job: 한 변환 요청 (kind = 'convert' | 'ocr', payload, events queue, cancelled).
- JobQueue: threading.Condition 으로 단일 워커가 대기, FIFO 순서대로 처리.
- 워커 thread (백그라운드 daemon) 는 모듈 import 시 자동 시작.
- API handler 가 Job 을 submit 하고, 같은 Job 의 events 큐에서 progress/done
  이벤트를 꺼내 streaming 응답으로 흘려보냄.
- 클라이언트 disconnect 가 감지되면 handler 가 queue.cancel(job) 호출. 워커는
  페이지 단위 콜백에서 job.cancelled 체크하고 즉시 raise CancelledError.
"""
from __future__ import annotations

import logging
import queue as _queue
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

log = logging.getLogger("pymu.queue")


class CancelledError(Exception):
    """Raised inside a job worker when the client disconnected."""


@dataclass
class Job:
    kind: str  # 'convert' | 'ocr'
    payload: dict
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    cancelled: bool = False
    events: "_queue.Queue[dict]" = field(default_factory=_queue.Queue)


class JobQueue:
    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._waiting: list[Job] = []
        self._running: Optional[Job] = None
        self._worker_started = False

    # ── API handler 측 ───────────────────────────────────────────────

    def submit(self, job: Job) -> None:
        with self._cv:
            self._waiting.append(job)
            self._cv.notify_all()

    def cancel(self, job: Job) -> None:
        """Mark a job cancelled. Safe to call from any thread."""
        with self._cv:
            job.cancelled = True
            if job in self._waiting:
                self._waiting.remove(job)
                self._cv.notify_all()

    def position(self, job: Job) -> int:
        """Return 0-based position in the waiting queue.
        - 0 ... N-1: still waiting (0 = next up)
        - -1: already running, finished, or cancelled-and-removed.
        """
        with self._cv:
            try:
                return self._waiting.index(job)
            except ValueError:
                return -1

    # ── 워커 thread 측 ────────────────────────────────────────────────

    def _acquire_next(self) -> Job:
        with self._cv:
            while True:
                # 큐 앞에서부터 cancelled 가 아닌 첫 job 꺼냄
                for j in list(self._waiting):
                    if j.cancelled:
                        self._waiting.remove(j)
                        continue
                    self._waiting.remove(j)
                    self._running = j
                    return j
                self._cv.wait()

    def _worker_loop(self, processor: Callable[[Job], None]) -> None:
        log.info("job queue worker started")
        while True:
            try:
                job = self._acquire_next()
            except Exception:
                log.exception("acquire_next failed")
                continue
            try:
                if job.cancelled:
                    job.events.put({"type": "cancelled"})
                else:
                    processor(job)
            except CancelledError:
                log.info("job %s cancelled", job.id)
                job.events.put({"type": "cancelled"})
            except Exception as e:  # noqa: BLE001
                log.exception("job %s failed", job.id)
                job.events.put({"type": "error", "message": f"내부 오류: {e}"})
            finally:
                with self._cv:
                    self._running = None

    def start_worker(self, processor: Callable[[Job], None]) -> None:
        """Start the background worker thread once."""
        with self._cv:
            if self._worker_started:
                return
            self._worker_started = True
        t = threading.Thread(
            target=self._worker_loop, args=(processor,), name="job-worker", daemon=True,
        )
        t.start()


# ── 모듈 전역 큐 ────────────────────────────────────────────────────

QUEUE = JobQueue()


# ── Job processor (lazy imports to avoid cycles) ────────────────────

def process(job: Job) -> None:
    """Dispatch a Job to the appropriate handler."""
    if job.kind == "convert":
        _process_convert(job)
    elif job.kind == "ocr":
        _process_ocr(job)
    else:
        raise ValueError(f"unknown job kind: {job.kind!r}")


def _check_cancelled(job: Job) -> None:
    if job.cancelled:
        raise CancelledError()


def _process_convert(job: Job) -> None:
    import base64
    import io
    import zipfile
    from pathlib import Path

    from backend.services import converter, ocr, office, spacing

    items = job.payload["items"]
    ocr_enabled = bool(job.payload.get("ocr_enabled", True))
    llm_enhance = bool(job.payload.get("llm_enhance", True))

    # 진행률 분배: 변환 자체가 차지하는 시작 구간 + OCR/LLM 이 차지하는 메인
    # 구간을 각 그룹에 균등 분배.
    if ocr_enabled and llm_enhance:
        pre_lo, main_hi = 5, 95
    elif ocr_enabled:
        pre_lo, main_hi = 10, 95
    else:
        pre_lo, main_hi = 5, 95

    def emit_progress(percent: int, label: str) -> None:
        job.events.put({"type": "progress", "percent": percent, "label": label})

    _check_cancelled(job)
    emit_progress(2, "파일 분석 중…")

    # 그룹 분리: merge=True 모임 → 1 그룹 (병합), merge=False → 각자 단독.
    merge_group = [it for it in items if getattr(it, "merge", True)]
    solo_items = [it for it in items if not getattr(it, "merge", True)]
    groups: list[tuple[str, list]] = []  # (suggested_filename, items)
    if merge_group:
        groups.append(("merged.pdf", merge_group))
    for it in solo_items:
        base = Path(it.filename or "file").stem or "file"
        groups.append((f"{base}.pdf", [it]))
    if not groups:
        # 안전: items 가 비어있다 — 호출 측에서 막혀야 하지만, 안전망.
        job.events.put({"type": "error", "message": "처리할 파일이 없습니다."})
        return

    n_groups = len(groups)

    # 각 그룹의 진행률 lo/hi 를 미리 계산.
    def group_range(idx: int) -> tuple[int, int]:
        lo = pre_lo + (main_hi - pre_lo) * idx / n_groups
        hi = pre_lo + (main_hi - pre_lo) * (idx + 1) / n_groups
        return int(lo), int(hi)

    results: list[tuple[str, bytes]] = []
    used_llm = False

    for idx, (suggested_name, group_items) in enumerate(groups):
        _check_cancelled(job)
        lo, hi = group_range(idx)
        group_label = f"그룹 {idx + 1}/{n_groups}" if n_groups > 1 else ""
        emit_progress(lo, f"{group_label} PDF 병합 중…".strip())

        try:
            pdf_bytes = converter.items_to_pdf(group_items)
        except office.OfficeConvertError as e:
            job.events.put({"type": "error", "message": str(e)})
            return
        except ValueError as e:
            job.events.put({"type": "error", "message": str(e)})
            return

        if ocr_enabled:
            def make_cb(group_lo: int, group_hi: int, group_idx: int):
                def cb(done: int, total_pages: int) -> None:
                    _check_cancelled(job)
                    if total_pages <= 0:
                        return
                    pct = group_lo + (group_hi - group_lo) * done / total_pages
                    label_base = (
                        f"OCR + 정확도 향상 페이지 {done}/{total_pages}"
                        if llm_enhance
                        else f"OCR 페이지 {done}/{total_pages}"
                    )
                    label = (
                        f"{label_base} (그룹 {group_idx + 1}/{n_groups})"
                        if n_groups > 1 else label_base
                    )
                    emit_progress(int(pct), label)
                return cb

            pdf_bytes = ocr.add_ocr_layer(
                pdf_bytes,
                progress_cb=make_cb(lo, hi, idx),
                use_llm=llm_enhance,
                is_cancelled=lambda: job.cancelled,
            )
            if llm_enhance:
                used_llm = True

        results.append((suggested_name, pdf_bytes))

    if used_llm:
        try:
            spacing.release_model()
        except Exception:
            log.debug("spacing release failed", exc_info=True)

    _check_cancelled(job)
    emit_progress(96, "PDF 직렬화…")

    if len(results) == 1:
        filename, pdf_bytes = results[0]
        payload_b64 = base64.b64encode(pdf_bytes).decode("ascii")
        mime = "application/pdf"
    else:
        # 다중 결과 → zip
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            seen: dict[str, int] = {}
            for fn, pdf in results:
                # 동일 이름 충돌 방지 (e.g., 두 단독 그룹이 같은 stem)
                base = fn
                if base in seen:
                    seen[base] += 1
                    stem = base[:-4] if base.lower().endswith(".pdf") else base
                    base = f"{stem}-{seen[base]}.pdf"
                else:
                    seen[base] = 1
                zf.writestr(base, pdf)
        zip_bytes = buf.getvalue()
        filename = "converted.zip"
        payload_b64 = base64.b64encode(zip_bytes).decode("ascii")
        mime = "application/zip"

    job.events.put({"type": "progress", "percent": 100, "label": "완료"})
    job.events.put({
        "type": "done",
        "filename": filename,
        "data": payload_b64,
        "mime": mime,
    })


def _process_ocr(job: Job) -> None:
    import base64
    from backend.services import ocr, spacing

    pdf_bytes = job.payload["pdf_bytes"]
    filename = job.payload.get("filename", "ocr.pdf")
    llm_enhance = bool(job.payload.get("llm_enhance", True))

    def emit_progress(percent: int, label: str) -> None:
        job.events.put({"type": "progress", "percent": percent, "label": label})

    _check_cancelled(job)
    emit_progress(2, "시작…")

    full_lo, full_hi = (5, 95) if llm_enhance else (5, 95)

    def cb(done: int, total_pages: int) -> None:
        _check_cancelled(job)
        if total_pages <= 0:
            return
        pct = full_lo + (full_hi - full_lo) * done / total_pages
        label = (
            f"OCR + 정확도 향상 페이지 {done}/{total_pages}"
            if llm_enhance
            else f"OCR 페이지 {done}/{total_pages}"
        )
        emit_progress(int(pct), label)

    pdf_bytes = ocr.add_ocr_layer(
        pdf_bytes,
        progress_cb=cb,
        use_llm=llm_enhance,
        is_cancelled=lambda: job.cancelled,
    )
    if llm_enhance:
        try:
            spacing.release_model()
        except Exception:
            log.debug("spacing release failed", exc_info=True)

    _check_cancelled(job)
    emit_progress(96, "PDF 직렬화…")
    payload_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    job.events.put({"type": "progress", "percent": 100, "label": "완료"})
    job.events.put({
        "type": "done", "filename": filename, "data": payload_b64,
        "mime": "application/pdf",
    })
