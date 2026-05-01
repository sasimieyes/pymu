"""Add an invisible OCR text layer to PDFs using PaddleOCR + PyMuPDF.

Strategy: OCR every page in full, then drop OCR results whose bbox lies
inside an existing vector-text region. Net effect: only text that lives
inside images (or on truly scanned pages) ends up as new invisible
glyphs. Vector text already on the page is untouched, so we never
duplicate.

The OCR text is rendered with render_mode=3 (invisible) so PDFs stay
visually identical but become searchable / copy-pasteable.

This module is intentionally minimal: PaddleOCR 3.x is invoked with
`lang="korean"` (which auto-selects `korean_PP-OCRv5_mobile_rec`),
text-line orientation classification on, and the doc-orientation /
unwarping stages disabled (we feed already-rectified PDF page renders).
No image preprocessing, fixed page-render zoom. We'll layer
optimizations back in one at a time and measure each change.
"""
from __future__ import annotations

import io
import logging
import os
import threading
from typing import Iterable

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from backend import THREAD_CAP
from backend.services.spacing import apply_correction_batch

# paddle GPU 메모리 풀: 기본 big-bucket 정책은 한 번 확장된 풀을 안 줄여
# 누적 점유처럼 보임. auto_growth 로 필요할 때만 확장하고 8 GB 의 50% 까지로 상한.
# 반드시 paddle / paddleocr 가 처음 import 되기 전에 설정해야 효과 있음.
os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
os.environ.setdefault("FLAGS_fraction_of_gpu_memory_to_use", "0.5")

log = logging.getLogger("pymu.ocr")

# Paddle is heavy to import; do it lazily and once.
_OCR_LOCK = threading.Lock()
_OCR_INSTANCE = None

# Default render zoom for normal PDF pages (2.0 ≈ 144 DPI). Simple,
# predictable baseline; revisit once we have a measurement to compare
# against.
_RENDER_ZOOM = 2.0

# Hard cap on the OCR-stage pixmap area, in pixels. Image-from-photo
# inputs become PDF pages whose rect equals the raw pixel dims (e.g.
# 3500×2625 pt), so rendering at zoom 2.0 would yield a 36 MP pixmap
# and push PaddleOCR's detector into multi-GB territory. We dynamically
# lower the zoom to keep any single page's pixmap at or below this cap.
# Normal text-PDF pages stay well under it and render at full _RENDER_ZOOM.
_MAX_OCR_PIXELS = 9_000_000  # ~9 MP, e.g. 3464×2598 or 3674×2449

# An OCR'd box is treated as "already covered" if at least this fraction of
# its area lies inside any existing vector-text bbox.
_TEXT_OVERLAP_THRESHOLD = 0.7


def _get_ocr():
    """Lazily build a singleton PaddleOCR instance (Korean + English)."""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is None:
        with _OCR_LOCK:
            if _OCR_INSTANCE is None:
                from paddleocr import PaddleOCR
                # Pin every sub-model explicitly. paddleocr 3.x's `lang=`
                # auto-mapping is bypassed as soon as you override one model
                # name, so partial overrides silently fall back to the default
                # multilingual server rec (we lost korean specialization that
                # way). The three names below are the official PP-OCRv5 mobile
                # variants — much lighter than the *_server_* defaults that
                # `lang="korean"` would pick, and the korean rec keeps Hangul
                # accuracy.
                # GPU 메모리 여유로 검출 정확도 우선 — server_det 채택. mobile 대비 작은 영역 분리 + 한글 자모 분리 정확도 향상.
                kwargs = dict(
                    text_detection_model_name="PP-OCRv5_server_det",
                    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
                    textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
                    # 박스 확장 비율 ↑: 한글 자모 잘림 방지 + 인라인 영문 토큰 분리 개선 (기본 1.5).
                    text_det_unclip_ratio=1.8,
                    use_textline_orientation=True,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    # paddle GPU 추론 (페이지당 ~0.05s). 시스템 RAM 대신 VRAM 사용으로
                    # 큰 이미지 페이지에서의 메모리 폭주를 방지한다.
                    # NOTE: NUM_PARALLEL=2 + paddle CPU 시나리오는 측정 결과 효과
                    # 없음 — gemma e2b 의 GPU 토큰 throughput (~33 tok/s) 자체가
                    # 한계이고, 큰 prompt 에선 단일 GPU slot 으로 직렬 처리됨.
                    # paddle CPU 는 OCR 5초/페이지 손실만 추가. 가장 단순한 GPU+단일
                    # LLM 조합이 현재 최선 (정확도 우선).
                    device="gpu",
                    # paddlepaddle 3.3 + PP-OCRv5 PIR models hit an unimplemented
                    # oneDNN attribute path on Windows; force the plain CPU kernel.
                    enable_mkldnn=False,
                )
                if THREAD_CAP is not None:
                    kwargs["cpu_threads"] = THREAD_CAP
                    log.info("PaddleOCR cpu_threads=%d", THREAD_CAP)
                _OCR_INSTANCE = PaddleOCR(**kwargs)
    return _OCR_INSTANCE


def warmup() -> None:
    """Trigger model load AND prime any lazy first-call init."""
    ocr_inst = _get_ocr()
    try:
        dummy = np.zeros((64, 256, 3), dtype=np.uint8)
        ocr_inst.predict(dummy)
    except Exception:
        log.warning("warmup dummy inference failed (non-fatal)", exc_info=True)


def _quad_bbox(quad) -> tuple[float, float, float, float]:
    """Convert a 4-point polygon to an axis-aligned (x0, y0, x1, y1) bbox."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return min(xs), min(ys), max(xs), max(ys)


def _ocr_image(
    img: Image.Image,
) -> list[tuple[tuple[float, float, float, float], str]]:
    """Run OCR on a PIL image and return raw [(bbox, text), ...] in image pixels.

    Per-line text correction is no longer done here; the caller batches lines
    by page and runs a single LLM call against the whole page (see
    add_ocr_layer below).
    """
    ocr = _get_ocr()
    arr = np.array(img.convert("RGB"))
    raw = ocr.predict(arr)
    if not raw:
        return []
    # OCRResult is a dict-like (UserDict). Use item access, not getattr.
    res = raw[0]
    polys = res.get("rec_polys") or []
    texts = res.get("rec_texts") or []
    results = []
    for quad, text in zip(polys, texts):
        if not text:
            continue
        results.append((_quad_bbox(quad), text))
    return results


def _estimate_text_width(text: str, fontsize: float) -> float:
    """Rough width estimate for mixed Korean/Latin text at `fontsize`.

    CJK glyphs are full-width (~1.0 em); Latin/digits/punctuation average
    around 0.55 em in Source Han style fonts. We use this instead of
    `fitz.get_text_length()` because the latter can be unreliable for
    PyMuPDF's built-in CJK font aliases ("korea") and silently
    underestimates Hangul, leading to clipped invisible text.
    """
    cjk = 0
    other = 0
    for c in text:
        cp = ord(c)
        if (0xAC00 <= cp <= 0xD7AF       # Hangul syllables
                or 0x1100 <= cp <= 0x11FF    # Hangul jamo
                or 0x3130 <= cp <= 0x318F    # Hangul compatibility jamo
                or 0x4E00 <= cp <= 0x9FFF    # CJK unified ideographs
                or 0x3040 <= cp <= 0x30FF    # Hiragana/Katakana
                or 0xFF00 <= cp <= 0xFFEF):  # full-width ASCII / half-width kana
            cjk += 1
        else:
            other += 1
    return fontsize * (cjk * 1.0 + other * 0.55)


def _fit_fontsize(text: str, max_w: float, max_h: float) -> float:
    """Largest fontsize where `text` fits in (max_w, max_h)."""
    if not text or max_w <= 0 or max_h <= 0:
        return 1.0
    size = max(1.0, max_h * 0.9)
    width_estimate = _estimate_text_width(text, size)
    if width_estimate <= max_w:
        return size
    return max(1.0, size * max_w / width_estimate)


def _split_to_word_boxes(
    rect: fitz.Rect, text: str
) -> Iterable[tuple[fitz.Rect, str]]:
    """Distribute `text` across `rect` per-word so PDF text selection
    aligns with visible word boundaries.
    """
    words = text.split()
    if len(words) <= 1:
        yield rect, text
        return
    units = sum(len(w) for w in words) + (len(words) - 1)
    if units <= 0:
        yield rect, text
        return
    unit_w = rect.width / units
    x = rect.x0
    for i, word in enumerate(words):
        word_w = unit_w * len(word)
        yield fitz.Rect(x, rect.y0, x + word_w, rect.y1), word
        x += word_w
        if i < len(words) - 1:
            x += unit_w  # inter-word gap


def _insert_invisible_text(
    page: fitz.Page,
    boxes: Iterable[tuple[tuple[float, float, float, float], str]],
    src_w: int,
    src_h: int,
    target_rect: fitz.Rect,
) -> None:
    """Place invisible glyphs onto the page.

    Uses PyMuPDF's built-in CJK font ("korea") so Hangul, kana and Latin
    OCR results all get glyph coverage. Helvetica would silently drop
    non-ASCII chars and the invisible layer would be unsearchable.
    """
    sx = target_rect.width / src_w
    sy = target_rect.height / src_h

    for (x0, y0, x1, y1), text in boxes:
        line_rect = fitz.Rect(
            target_rect.x0 + x0 * sx,
            target_rect.y0 + y0 * sy,
            target_rect.x0 + x1 * sx,
            target_rect.y0 + y1 * sy,
        )
        if line_rect.is_empty or line_rect.width < 1 or line_rect.height < 1:
            continue
        for sub_rect, sub_text in _split_to_word_boxes(line_rect, text):
            if not sub_text or sub_rect.width < 1 or sub_rect.height < 1:
                continue
            # invisible glyph 의 selection 영역을 OCR 박스와 일치시키려면
            # baseline 을 박스 하단(y1) 에 두고 글자를 위로 올린다.
            # insert_textbox 는 baseline 을 자동 배치해서 vertical 미세 오프셋이
            # 발생, 사용자가 selection 잡을 때 보이는 글자와 어긋나 보임.
            # 한글은 descender 가 거의 없어 baseline≈box bottom 가정이 적합하고,
            # 영문 descender 가 있는 글자도 OCR 박스가 visual extent 를 이미 포함
            # 하므로 시각적 차이 미미.
            fontsize = _fit_fontsize(sub_text, sub_rect.width, sub_rect.height)
            page.insert_text(
                (sub_rect.x0, sub_rect.y1),
                sub_text,
                fontsize=fontsize,
                fontname="korea",  # CJK + Latin coverage
                render_mode=3,     # invisible, but selectable/searchable
            )


def _ocr_zoom_for(page: fitz.Page) -> float:
    """Pick a render zoom that keeps the OCR-stage pixmap under _MAX_OCR_PIXELS.

    For ordinary text PDFs (e.g. letter/A4 pages) zoom stays at _RENDER_ZOOM.
    For image-derived "pages" whose rect already spans thousands of points,
    the zoom is dynamically lowered so the rasterized pixmap area stays
    bounded — preventing PaddleOCR from being handed 30+ MP arrays.
    """
    rect = page.rect
    pixels_at_default = (rect.width * _RENDER_ZOOM) * (rect.height * _RENDER_ZOOM)
    if pixels_at_default <= _MAX_OCR_PIXELS or pixels_at_default <= 0:
        return _RENDER_ZOOM
    # Uniformly scale zoom so the resulting pixmap area equals the cap.
    return _RENDER_ZOOM * (_MAX_OCR_PIXELS / pixels_at_default) ** 0.5


def _existing_text_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Bounding rects (page coords) of every word with non-empty text."""
    rects: list[fitz.Rect] = []
    for w in page.get_text("words"):
        # words tuple: (x0, y0, x1, y1, "text", block_no, line_no, word_no)
        if len(w) < 5 or not w[4] or not w[4].strip():
            continue
        rects.append(fitz.Rect(w[:4]))
    return rects


def _overlaps_existing_text(box: fitz.Rect, text_rects: list[fitz.Rect]) -> bool:
    """True if ≥THRESHOLD of `box`'s area sits inside any text rect."""
    box_area = box.get_area()
    if box_area <= 0:
        return False
    for tr in text_rects:
        inter = box & tr
        if inter.is_empty:
            continue
        if inter.get_area() / box_area >= _TEXT_OVERLAP_THRESHOLD:
            return True
    return False


def add_ocr_layer(
    pdf_bytes: bytes,
    progress_cb=None,
    use_llm: bool = True,
    is_cancelled=None,
) -> bytes:
    """Add an invisible OCR text layer for content not already in the text layer.

    Each page is rendered at a fixed zoom and run through PaddleOCR. Any
    OCR box that overlaps an existing vector-text word bbox is dropped,
    so only glyphs from images / scans get added.

    `progress_cb`, if provided, is called as `progress_cb(page_done, total)`
    after each page so callers can drive a progress UI.
    `use_llm` controls whether OCR text is passed through the LLM correction step.
    `is_cancelled`, if provided, is a zero-arg callable returning True when the
    caller wants to abort. Checked at the start of every page; on True the
    function raises backend.services.job_queue.CancelledError.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        total = len(doc)
        for page_idx, page in enumerate(doc):
            if is_cancelled is not None and is_cancelled():
                from backend.services.job_queue import CancelledError
                raise CancelledError()

            text_rects = _existing_text_rects(page)

            zoom = _ocr_zoom_for(page)
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom), alpha=False,
            )
            full = Image.open(io.BytesIO(pix.tobytes("png")))
            ocr_boxes = _ocr_image(full)
            if not ocr_boxes:
                log.info("page %d: OCR found 0 boxes", page_idx)
                if progress_cb is not None:
                    try:
                        progress_cb(page_idx + 1, total)
                    except Exception:
                        log.warning("progress_cb raised; continuing", exc_info=True)
                continue

            sx_page = page.rect.width / full.width
            sy_page = page.rect.height / full.height

            accepted: list[tuple[tuple[float, float, float, float], str]] = []
            skipped = 0
            for (x0, y0, x1, y1), text in ocr_boxes:
                page_box = fitz.Rect(
                    x0 * sx_page, y0 * sy_page, x1 * sx_page, y1 * sy_page,
                )
                if _overlaps_existing_text(page_box, text_rects):
                    skipped += 1
                    continue
                accepted.append(((x0, y0, x1, y1), text))

            log.info(
                "page %d: OCR=%d, existing_text_words=%d, skipped=%d, kept=%d",
                page_idx, len(ocr_boxes), len(text_rects), skipped, len(accepted),
            )
            if accepted:
                # 페이지의 모든 라인 텍스트를 한 번에 LLM batch 로 교정.
                # apply_correction_batch 는 실패 시 per-line fallback 처리.
                texts_for_llm = [t for (_b, t) in accepted]
                corrected = apply_correction_batch(texts_for_llm, use_llm=use_llm)
                accepted = [
                    (bbox, corrected[i]) for i, (bbox, _t) in enumerate(accepted)
                ]
                _insert_invisible_text(
                    page,
                    accepted,
                    src_w=full.width,
                    src_h=full.height,
                    target_rect=page.rect,
                )
            if progress_cb is not None:
                try:
                    progress_cb(page_idx + 1, total)
                except Exception:
                    log.warning("progress_cb raised; continuing", exc_info=True)
        return doc.tobytes(garbage=3, deflate=True)
    finally:
        doc.close()
        # server_det + 큰 입력으로 paddle 메모리 풀이 GB 단위로 부풀음.
        # 모델 가중치는 유지하고 비어있는 풀 청크만 OS 로 반환.
        try:
            import paddle
            paddle.device.cuda.empty_cache()
        except Exception:
            log.debug("paddle empty_cache failed (non-fatal)", exc_info=True)
