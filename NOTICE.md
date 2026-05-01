# Third-Party Notices

This project (the "Software") is licensed under the GNU Affero General Public
License v3.0 (AGPL-3.0). See `LICENSE` for the full text.

The Software incorporates or depends on the following third-party components.
Each component is the property of its respective copyright holder(s) and is
licensed under the terms summarized below. Full license texts are available at
the linked URLs.

---

## Python dependencies

### PyMuPDF — `fitz`
- **License**: GNU AGPL-3.0 (or commercial license from Artifex)
- **Project**: https://github.com/pymupdf/PyMuPDF
- **License text**: https://www.gnu.org/licenses/agpl-3.0.txt
- **Note**: PyMuPDF's AGPL is the reason this project itself is published under AGPL-3.0.

### PaddleOCR
- **License**: Apache License 2.0
- **Project**: https://github.com/PaddlePaddle/PaddleOCR
- **License text**: https://www.apache.org/licenses/LICENSE-2.0

### PaddlePaddle (paddlepaddle)
- **License**: Apache License 2.0
- **Project**: https://github.com/PaddlePaddle/Paddle
- **License text**: https://www.apache.org/licenses/LICENSE-2.0

### FastAPI
- **License**: MIT License
- **Project**: https://github.com/tiangolo/fastapi
- **License text**: https://github.com/tiangolo/fastapi/blob/master/LICENSE

### Starlette (transitive via FastAPI)
- **License**: BSD 3-Clause
- **Project**: https://github.com/encode/starlette

### Uvicorn
- **License**: BSD 3-Clause
- **Project**: https://github.com/encode/uvicorn

### python-multipart
- **License**: Apache License 2.0
- **Project**: https://github.com/Kludex/python-multipart

### Pillow
- **License**: MIT-CMU (Historical Permission Notice and Disclaimer)
- **Project**: https://github.com/python-pillow/Pillow
- **License text**: https://github.com/python-pillow/Pillow/blob/main/LICENSE

### NumPy
- **License**: BSD 3-Clause
- **Project**: https://github.com/numpy/numpy

### setuptools
- **License**: MIT License
- **Project**: https://github.com/pypa/setuptools

### Pydantic (transitive via FastAPI)
- **License**: MIT License
- **Project**: https://github.com/pydantic/pydantic

---

## Frontend dependencies

### SortableJS (loaded from CDN)
- **License**: MIT License
- **Project**: https://github.com/SortableJS/Sortable
- **CDN**: https://cdn.jsdelivr.net/npm/sortablejs

---

## External runtime tools

The following tools are invoked by the Software at runtime via subprocess but
are **not** redistributed with this project. Users install them themselves.

### LibreOffice (`soffice`)
- **License**: Mozilla Public License 2.0
- **Project**: https://www.libreoffice.org/
- **Note**: Invoked via subprocess for office document conversion. Not redistributed.

### H2Orestart (LibreOffice extension for HWP/HWPX)
- **License**: GNU GPL-3.0
- **Project**: https://github.com/ebandal/H2Orestart
- **Note**: Optional. Loaded by LibreOffice when present. Not redistributed.

### Ollama
- **License**: MIT License
- **Project**: https://github.com/ollama/ollama
- **Note**: Optional. Used as a local HTTP server for LLM inference (`localhost:11434`). Not redistributed.

### Gemma 4 (model weights)
- **License**: Gemma Terms of Use (https://ai.google.dev/gemma/terms)
- **Project**: https://huggingface.co/google/gemma-4-E2B-it
- **Note**: Optional. Pulled via `ollama pull` by the operator. Not redistributed.

### WinSW (Windows service wrapper)
- **License**: MIT License
- **Project**: https://github.com/winsw/winsw
- **Note**: Optional. Used to install the FastAPI server as a Windows service. Not redistributed.

### PaddlePaddle GPU build
- **License**: Apache License 2.0
- **Project**: https://github.com/PaddlePaddle/Paddle
- **Note**: Replaces CPU `paddlepaddle` for GPU inference. CUDA runtime DLLs are auto-installed by `paddlepaddle-gpu` wheel.

---

## Apache 2.0 / BSD attribution

For Apache-2.0 and BSD-licensed components above:

- Copyright notices and license texts are preserved at the project links above.
- No NOTICE files from upstream Apache projects have been modified.
- This project is not endorsed by, or affiliated with, any of the listed
  projects or their copyright holders.

---

## Source code

This Software is open source under AGPL-3.0. The source code is available at:

> **<REPLACE_WITH_REPO_URL>**

If you interact with this Software over a network, you have the right to
obtain its complete corresponding source code under the terms of the AGPL.
