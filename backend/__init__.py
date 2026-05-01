"""Cap BLAS / OpenMP thread pools so PaddleOCR & numpy don't peg every core.

These env vars must be set *before* numpy / paddle are imported anywhere
in the process. Putting the logic in the package __init__ makes that
happen as soon as uvicorn imports `backend.main`.

Override default with the PYMU_OCR_THREADS env var:
    PYMU_OCR_THREADS=4   # cap to 4 threads
    PYMU_OCR_THREADS=0   # use all cores (disable cap)
Default: 70% of os.cpu_count() rounded down, minimum 1.
"""
from __future__ import annotations

import os

_DEFAULT_FRACTION = 0.7


def _resolve_thread_cap() -> int | None:
    """Return number of threads to allow, or None for "no cap (use all)"."""
    raw = os.environ.get("PYMU_OCR_THREADS")
    if raw is not None:
        try:
            n = int(raw)
        except ValueError:
            n = -1
        if n == 0:
            return None  # explicit opt-out
        if n > 0:
            return n
    cpu = os.cpu_count() or 1
    return max(1, int(cpu * _DEFAULT_FRACTION))


THREAD_CAP = _resolve_thread_cap()

if THREAD_CAP is not None:
    _val = str(THREAD_CAP)
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        # setdefault so the user's own export still wins.
        os.environ.setdefault(var, _val)
