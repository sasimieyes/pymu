"""Per-conversion JSONL log + daily summary worker.

Two outputs:
1. JSONL — every job emits one line at terminal state (done/cancelled/error)
   to `<PYMU_LOG_DIR>/conversions.jsonl`. Each line is self-contained JSON.
2. Daily summary — a background timer fires at local midnight and emits a
   single high-level INFO log line aggregating yesterday's JSONL lines.

Both are best-effort and never raise. Disk write contention is guarded by a
process-local Lock — the queue worker is single-threaded so the only other
writers are the daily worker and tests.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger("pymu.stats")

# Default log dir matches pymu-svc.xml <logpath> for one-stop ops view.
_DEFAULT_LOG_DIR = Path("D:/project/logs/pymu")
_LOG_DIR = Path(os.environ.get("PYMU_LOG_DIR", str(_DEFAULT_LOG_DIR)))
_JSONL_PATH = _LOG_DIR / "conversions.jsonl"
_WRITE_LOCK = threading.Lock()
_WORKER_STARTED = False


def _ensure_dir() -> None:
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        log.debug("log dir create failed (non-fatal)", exc_info=True)


def log_conversion(record: dict[str, Any]) -> None:
    """Append one JSON record (one line) to conversions.jsonl. Never raises."""
    try:
        record.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _WRITE_LOCK:
            _ensure_dir()
            with _JSONL_PATH.open("a", encoding="utf-8") as f:
                f.write(line)
    except Exception:
        log.warning("conversion log write failed", exc_info=True)


def _read_records_for(date_str: str) -> list[dict[str, Any]]:
    """Read JSONL records whose `ts` starts with date_str (YYYY-MM-DD)."""
    if not _JSONL_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        with _JSONL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                ts = rec.get("ts", "")
                if isinstance(ts, str) and ts.startswith(date_str):
                    out.append(rec)
    except Exception:
        log.warning("conversion log read failed", exc_info=True)
    return out


def _summarize_yesterday() -> None:
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    recs = _read_records_for(yesterday)
    if not recs:
        log.info("daily summary %s — no conversions", yesterday)
        return
    total = len(recs)
    done = sum(1 for r in recs if r.get("status") == "done")
    cancelled = sum(1 for r in recs if r.get("status") == "cancelled")
    error = sum(1 for r in recs if r.get("status") == "error")
    ocr_n = sum(1 for r in recs if r.get("ocr"))
    llm_n = sum(1 for r in recs if r.get("llm"))
    pages = [r.get("pages") for r in recs if isinstance(r.get("pages"), (int, float))]
    took = [r.get("took_s") for r in recs if isinstance(r.get("took_s"), (int, float))]
    in_bytes = sum((r.get("in_bytes") or 0) for r in recs)
    out_bytes = sum((r.get("out_bytes") or 0) for r in recs)
    avg_pages = (sum(pages) / len(pages)) if pages else 0
    avg_took = (sum(took) / len(took)) if took else 0
    log.info(
        "daily summary %s — total=%d done=%d cancelled=%d error=%d ocr=%d llm=%d "
        "avg_pages=%.1f avg_took=%.1fs in=%.2fMB out=%.2fMB",
        yesterday, total, done, cancelled, error, ocr_n, llm_n,
        avg_pages, avg_took, in_bytes / 1e6, out_bytes / 1e6,
    )


def _seconds_until_next_local_midnight() -> float:
    now = datetime.now()
    nxt = datetime.combine(now.date() + timedelta(days=1), dtime(0, 0, 0))
    return max(1.0, (nxt - now).total_seconds())


def _schedule_next() -> None:
    delay = _seconds_until_next_local_midnight()
    timer = threading.Timer(delay, _daily_tick)
    timer.daemon = True
    timer.name = "pymu-daily-summary"
    timer.start()


def _daily_tick() -> None:
    try:
        _summarize_yesterday()
    except Exception:
        log.exception("daily summary tick failed")
    finally:
        _schedule_next()


def start_daily_summary_worker() -> None:
    """Start the background daily summary worker once per process."""
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    _WORKER_STARTED = True
    _ensure_dir()
    _schedule_next()
    log.info("daily summary worker scheduled (next fire at local midnight)")
