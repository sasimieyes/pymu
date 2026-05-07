"""Convert various input files into a single merged PDF using PyMuPDF."""
from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageOps

from backend.services import office

# Smartphone photos can be 4000+ px on a side. If we hand that straight
# to PyMuPDF as page dimensions, rendering later blows past PIL's
# decompression-bomb limit. Cap the long edge to a value that's still
# plenty for OCR and on-screen viewing.
_MAX_IMAGE_LONG_EDGE_PX = 3500

# 결과 PDF 모든 페이지 통일 사이즈. PyMuPDF 의 단위는 pt (1 pt = 1/72 인치).
# A4 portrait (210 × 297 mm) ≈ 595 × 842 pt. 이미지·기존 PDF 페이지를 이 안에
# 비율 유지하면서 중앙 정렬한다. 결과적으로 모든 페이지가 같은 사이즈로 보여
# 뷰어/뷰어 줌 레벨이 페이지 사이에 점프하지 않는다.
_TARGET_PAGE_W_PT = 595.0
_TARGET_PAGE_H_PT = 842.0
# 이미지/페이지 가장자리 여백 (pt). 너무 빡빡하지 않게.
_PAGE_PADDING_PT = 12.0


def _fit_centered(src_w: float, src_h: float, dst_w: float, dst_h: float, padding: float) -> fitz.Rect:
    """Return a fitz.Rect inside dst that preserves aspect ratio of src and is
    centered with `padding` margin around the available area. Falls back to the
    full dst rect if src dims are non-positive.
    """
    if src_w <= 0 or src_h <= 0:
        return fitz.Rect(padding, padding, dst_w - padding, dst_h - padding)
    avail_w = max(1.0, dst_w - 2 * padding)
    avail_h = max(1.0, dst_h - 2 * padding)
    scale = min(avail_w / src_w, avail_h / src_h)
    new_w = src_w * scale
    new_h = src_h * scale
    x0 = (dst_w - new_w) / 2.0
    y0 = (dst_h - new_h) / 2.0
    return fitz.Rect(x0, y0, x0 + new_w, y0 + new_h)

# Defuse PIL's bomb check on the trusted side (our own server-rendered
# pixmaps); we already cap inputs above and bound total upload size in
# the API layer.
Image.MAX_IMAGE_PIXELS = None

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif"}
PDF_EXTS = {".pdf"}
OFFICE_EXTS = office.OFFICE_EXTS
SUPPORTED_EXTS = IMAGE_EXTS | PDF_EXTS | OFFICE_EXTS


@dataclass
class InputItem:
    """One input file to be merged into the output PDF."""
    filename: str
    data: bytes
    rotation: int = 0  # 0/90/180/270 — applied to every page in this item
    # True 면 다른 merge=True 아이템들과 한 PDF 로 병합. False 면 단독 PDF
    # 로 빠져 별도 다운로드 대상이 된다 (`_process_convert` 에서 그룹 분리).
    merge: bool = True

    @property
    def ext(self) -> str:
        return Path(self.filename).suffix.lower()


# JPEG quality used when we re-encode images during merge. q=90 keeps
# Korean-text legibility intact for OCR and on-screen viewing while making
# the output 3-4x smaller and the merge step ~3.7x faster than the previous
# PNG-re-encode path (measured on 4000x3000 photo input).
_MERGE_JPEG_QUALITY = 90


def _image_bytes_to_pdf(data: bytes, rotation: int) -> fitz.Document:
    """Convert image bytes to a single A4-portrait PDF page.

    The image is downscaled to a sane pixel budget, then placed centered
    on a fixed-size page so every output page has the same dimensions —
    no more zoom-jumping between pages of different aspect ratios in the
    PDF viewer.

    Two paths:
    1. Fast pass-through: JPEG input with no rotation, normal EXIF
       orientation, RGB mode, and already within the size budget — feed
       the original bytes straight into PyMuPDF's insert_image. ~150x
       faster than re-encoding for the common "phone photo / small scan"
       case.
    2. Slow path: any input that needs orientation fix, rotation, mode
       conversion, or downscaling. Decode with PIL, normalize, then
       re-encode as JPEG (not PNG — measured 3.7x faster and 4x smaller).
    """
    img = Image.open(io.BytesIO(data))
    fmt = (img.format or "").upper()
    try:
        exif = img.getexif()
    except Exception:
        exif = None
    orientation = exif.get(0x0112, 1) if exif else 1

    # Fast path: JPEG that already meets every constraint.
    if (
        rotation == 0
        and fmt == "JPEG"
        and orientation == 1
        and img.mode == "RGB"
        and max(img.width, img.height) <= _MAX_IMAGE_LONG_EDGE_PX
    ):
        w, h = img.size
        img.close()
        doc = fitz.open()
        page = doc.new_page(width=_TARGET_PAGE_W_PT, height=_TARGET_PAGE_H_PT)
        target = _fit_centered(
            w, h, _TARGET_PAGE_W_PT, _TARGET_PAGE_H_PT, _PAGE_PADDING_PT,
        )
        page.insert_image(target, stream=data)
        return doc

    # Slow path: decode, normalize, re-encode.
    # Apply EXIF orientation so phone photos taken in portrait don't come
    # out sideways. Must happen before user-specified rotation.
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        # Covers RGBA / LA / P / L / CMYK / etc. JPEG encoder rejects
        # anything but RGB / L / CMYK, and we want consistent color.
        img = img.convert("RGB")
    if rotation:
        img = img.rotate(-rotation, expand=True)  # PIL rotates CCW; we want CW

    # Downscale very large photos so the resulting PDF page (and any
    # downstream rasterization) stays within sane pixel budgets.
    longest = max(img.width, img.height)
    if longest > _MAX_IMAGE_LONG_EDGE_PX:
        scale = _MAX_IMAGE_LONG_EDGE_PX / longest
        new_size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_MERGE_JPEG_QUALITY, optimize=False)

    doc = fitz.open()
    page = doc.new_page(width=_TARGET_PAGE_W_PT, height=_TARGET_PAGE_H_PT)
    target = _fit_centered(
        img.width, img.height,
        _TARGET_PAGE_W_PT, _TARGET_PAGE_H_PT,
        _PAGE_PADDING_PT,
    )
    page.insert_image(target, stream=buf.getvalue())
    return doc


def _pdf_with_rotation(data: bytes, rotation: int) -> fitz.Document:
    """Re-render every page of a PDF onto A4-portrait pages, applying
    `rotation` (additive) and centering with letterbox so all pages of
    the merged output share one consistent size. Vector text is preserved
    via `show_pdf_page` (no rasterization)."""
    src = fitz.open(stream=data, filetype="pdf")
    out = fitz.open()
    for src_page in src:
        if rotation:
            src_page.set_rotation((src_page.rotation + rotation) % 360)
        # Effective on-screen size after rotation: rotated 90/270 swaps w/h.
        sw, sh = src_page.rect.width, src_page.rect.height
        if src_page.rotation in (90, 270):
            sw, sh = sh, sw
        new_page = out.new_page(
            width=_TARGET_PAGE_W_PT, height=_TARGET_PAGE_H_PT,
        )
        target = _fit_centered(
            sw, sh,
            _TARGET_PAGE_W_PT, _TARGET_PAGE_H_PT,
            _PAGE_PADDING_PT,
        )
        new_page.show_pdf_page(target, src, src_page.number)
    src.close()
    return out


def items_to_pdf(items: list[InputItem]) -> bytes:
    """Merge a list of input items into a single PDF, in order."""
    if not items:
        raise ValueError("no input items")

    out = fitz.open()
    try:
        for item in items:
            if item.ext in IMAGE_EXTS:
                src = _image_bytes_to_pdf(item.data, item.rotation)
            elif item.ext in PDF_EXTS:
                src = _pdf_with_rotation(item.data, item.rotation)
            elif item.ext in OFFICE_EXTS:
                pdf_bytes = office.office_to_pdf(item.filename, item.data)
                src = _pdf_with_rotation(pdf_bytes, item.rotation)
            else:
                raise ValueError(f"unsupported extension: {item.ext}")
            try:
                out.insert_pdf(src)
            finally:
                src.close()
        # garbage=3 prunes orphaned objects produced by repeated insert_pdf
        # calls (each input gets its own page tree fragment merged in). No
        # deflate here — output is JPEG-heavy and re-deflating would just
        # spend CPU for ~0% gain.
        return out.tobytes(garbage=3)
    finally:
        out.close()
