"""Convert office documents (DOCX, XLSX, PPTX, HWP, etc.) to PDF.

Implementation: shell out to LibreOffice headless. This handles a wide
range of formats reliably and preserves layout/fonts far better than any
pure-Python alternative.

For .hwp / .hwpx (Korean Hancom Office) the H2Orestart extension
(ebandal/H2Orestart) must be installed system-wide so the LibreOffice
profile owned by the service account can pick it up:

    "C:/Program Files/LibreOffice/program/unopkg.exe" add --shared --suppress-license H2Orestart.oxt

Requires LibreOffice (`soffice`) on the system. Set `SOFFICE_PATH` to
override autodetection.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("pymu.office")

OFFICE_EXTS = {
    ".doc", ".docx", ".odt", ".rtf",
    ".hwp", ".hwpx",
    ".xls", ".xlsx", ".ods", ".csv",
    ".ppt", ".pptx", ".odp",
    ".txt", ".html", ".htm",
}

CONVERT_TIMEOUT_SEC = 180


class OfficeConvertError(RuntimeError):
    pass


def _find_soffice() -> str | None:
    """Locate the LibreOffice CLI. Returns None if not installed."""
    explicit = os.environ.get("SOFFICE_PATH")
    if explicit and Path(explicit).exists():
        return explicit

    found = shutil.which("soffice") or shutil.which("soffice.exe")
    if found:
        return found

    candidates = [
        Path.home() / "scoop" / "apps" / "libreoffice" / "current" / "program" / "soffice.exe",
        Path("C:/Program Files/LibreOffice/program/soffice.exe"),
        Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
        Path("/usr/bin/soffice"),
        Path("/usr/local/bin/soffice"),
        Path("/Applications/LibreOffice.app/Contents/MacOS/soffice"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def is_available() -> bool:
    return _find_soffice() is not None


def office_to_pdf(filename: str, data: bytes) -> bytes:
    """Convert an office document to PDF bytes via LibreOffice headless."""
    soffice = _find_soffice()
    if soffice is None:
        raise OfficeConvertError(
            "LibreOffice가 설치되어 있지 않습니다. "
            "`scoop install libreoffice` 또는 https://www.libreoffice.org 에서 설치하세요."
        )

    with tempfile.TemporaryDirectory(prefix="pymu_office_") as tmp:
        tmp_path = Path(tmp)
        # Use the provided filename (sanitized) so the output PDF picks up its stem.
        safe_name = Path(filename).name or "input"
        in_path = tmp_path / safe_name
        in_path.write_bytes(data)

        out_dir = tmp_path / "out"
        out_dir.mkdir()

        # Use a per-call user profile dir so concurrent calls don't collide
        # with each other or with a running desktop LibreOffice.
        profile_dir = (tmp_path / "profile").as_posix()

        cmd = [
            soffice,
            "--headless",
            "--norestore",
            "--nologo",
            "--nofirststartwizard",
            f"-env:UserInstallation=file:///{profile_dir}",
            "--convert-to", "pdf",
            "--outdir", str(out_dir),
            str(in_path),
        ]
        log.info("running soffice convert: %s", in_path.name)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                timeout=CONVERT_TIMEOUT_SEC,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise OfficeConvertError(f"LibreOffice 변환 시간 초과: {filename}") from e

        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or b"").decode("utf-8", "replace").strip()
            raise OfficeConvertError(f"LibreOffice 변환 실패: {err or 'unknown error'}")

        out_pdf = out_dir / (in_path.stem + ".pdf")
        if not out_pdf.exists():
            # Fallback: pick any PDF in the outdir.
            pdfs = list(out_dir.glob("*.pdf"))
            if not pdfs:
                raise OfficeConvertError(f"출력 PDF가 생성되지 않았습니다: {filename}")
            out_pdf = pdfs[0]
        return out_pdf.read_bytes()
