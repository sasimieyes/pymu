"""Korean OCR post-correction via local Ollama (Gemma 4 E2B).

PaddleOCR's korean rec model returns line-level text without spaces
("현대에는컴퓨팅...") and occasional typos ("수십역"→"수십억"). We
post-process each OCR line through a local Gemma 4 E2B served by
Ollama on localhost:11434.

The LLM call is per-line with a strict prompt that instructs the model
to preserve every character — Latin/digits/punctuation must not be
dropped — and only insert spaces or fix obvious typos.

If Ollama is unreachable or times out the OCR pipeline keeps running:
we fall back to a regex pass that inserts a space at Hangul↔ASCII-letter
or Hangul↔digit boundaries. That covers the easy cases ("안녕하세요Hello"
→ "안녕하세요 Hello") but won't fix Hangul-Hangul spacing on its own.
"""
from __future__ import annotations

import logging
import os
import re
import threading

import httpx

log = logging.getLogger("pymu.spacing")

_OLLAMA_URL = os.environ.get("PYMU_OLLAMA_URL", "http://localhost:11434")
_OLLAMA_MODEL = os.environ.get("PYMU_SPACING_MODEL", "gemma4:e2b-it-q4_K_M")
# 첫 호출은 모델 로드(디스크→VRAM)까지 합쳐 20초를 넘길 수 있어 90s 로 둠.
# 정상 추론은 1~5초라 timeout 이 길어도 응답이 빠르게 돌아오면 영향 없음.
_OLLAMA_TIMEOUT = float(os.environ.get("PYMU_OLLAMA_TIMEOUT", "90"))

_PROMPT = (
    "다음은 OCR로 추출한 한국어 텍스트 한 줄입니다. 띄어쓰기와 명백한 오탈자만 교정해서 결과만 출력하세요.\n"
    "- 원문의 모든 글자(영어, 숫자, 구두점, 기호 포함)를 그대로 유지하세요. 영문/숫자/기호를 절대 빼거나 다른 글자로 바꾸지 마세요.\n"
    "- 줄을 합치거나 분리하지 마세요. 한 줄로 출력하세요.\n"
    "- 추가 설명, 인용 부호, 번역, 주석 없이 교정된 텍스트 한 줄만 출력하세요.\n\n"
    "원문: {text}\n"
    "교정:"
)

# Hangul + ASCII letter/digit boundary regex (both directions).
_HANGUL = r"가-힣ㄱ-ㆎ"
_RE_HAN_TO_LATIN = re.compile(rf"([{_HANGUL}])([A-Za-z0-9])")
_RE_LATIN_TO_HAN = re.compile(rf"([A-Za-z0-9])([{_HANGUL}])")

# httpx client cached per-thread so connection reuse works across many lines.
_TLS = threading.local()


def _client() -> httpx.Client:
    c = getattr(_TLS, "client", None)
    if c is None:
        c = httpx.Client(timeout=_OLLAMA_TIMEOUT)
        _TLS.client = c
    return c


def _boundary_only(text: str) -> str:
    """Insert a space between Hangul and adjacent ASCII letter/digit."""
    text = _RE_HAN_TO_LATIN.sub(r"\1 \2", text)
    text = _RE_LATIN_TO_HAN.sub(r"\1 \2", text)
    return text


def _llm_correct(text: str) -> str:
    """Single-line correction via Ollama. Raises httpx errors on failure."""
    r = _client().post(
        f"{_OLLAMA_URL}/api/chat",
        json={
            "model": _OLLAMA_MODEL,
            "messages": [{"role": "user", "content": _PROMPT.format(text=text)}],
            "stream": False,
            "think": False,
            # 한 변환 도중에는 페이지 간 reload 없이 유지하도록 30 초.
            # 변환 끝나고 release_model() 이 호출되면 즉시 unload.
            "keep_alive": "30s",
            "options": {
                "temperature": 0,
                "num_predict": 400,
                "num_ctx": 2048,
                "use_mmap": True,
            },
        },
    )
    r.raise_for_status()
    out = (r.json().get("message") or {}).get("content", "").strip()
    # Strip occasional decorations the model adds despite the strict prompt.
    out = out.strip("\"'`")
    if "\n" in out:
        out = out.split("\n", 1)[0].strip()
    return out or text


def apply_correction(text: str, use_llm: bool = True) -> str:
    """Return spacing-/typo-corrected text. Always returns a string —
    never raises. If `use_llm` is False, only the cheap Hangul-ASCII
    boundary regex runs. If the LLM call fails (Ollama down, timeout),
    we degrade to boundary regex too.
    """
    if not text or not text.strip():
        return text
    # Pure-Latin / digit-only lines — LLM is overkill; boundary regex is also
    # a no-op on them, so just return as-is.
    if not re.search(rf"[{_HANGUL}]", text):
        return text
    if not use_llm:
        return _boundary_only(text)
    try:
        return _llm_correct(text)
    except Exception:
        log.warning("LLM spacing failed, falling back to boundary regex", exc_info=True)
        return _boundary_only(text)


# Batch LLM correction — 페이지의 모든 line 을 numbered list 로 묶어 한 번에
# 호출. 호출당 first-token-latency 를 page 1번으로 줄이고, LLM 이 줄 사이
# 컨텍스트를 봐서 줄 끝/시작 띄어쓰기, 마침표 직후 띄어쓰기, 줄바꿈으로
# 분리된 단어 등을 더 자연스럽게 처리할 수 있게 한다.
_BATCH_PROMPT = (
    "다음은 OCR로 추출한 한국어 텍스트의 여러 줄입니다. 각 줄의 띄어쓰기와 명백한 오탈자만 교정해서 같은 형식으로 출력하세요.\n"
    "- 출력 형식: 입력과 동일한 [번호] 접두사를 그대로 유지하고, 그 뒤에 교정된 텍스트 한 줄.\n"
    "- 줄 수, 줄 순서, 번호 모두 입력과 똑같이 유지하세요. 줄을 합치거나 분리하지 마세요.\n"
    "- 원문의 모든 글자(영어, 숫자, 구두점, 기호 포함)를 그대로 유지하세요. 영문/숫자/기호를 절대 빼거나 다른 글자로 바꾸지 마세요.\n"
    "- 추가 설명, 인용 부호, 번역, 주석 없이 [번호] 줄들만 출력하세요.\n\n"
    "입력:\n{items}\n\n"
    "출력:"
)

# LLM 이 출력하는 numbered line 의 다양한 형식 ([1], 1., 1:, 1)) 모두 매칭.
_RE_NUMBERED_LINE = re.compile(r"^\s*[\[\(]?(\d+)[\]\)\.\:]\s*(.*)$")

# 한 번 LLM 호출에 보낼 최대 라인 수. gemma4:e2b 같은 작은 모델은 30+줄 numbered
# list 를 안정적으로 출력하지 못해 mapping mismatch 가 잦았다. 10 줄 chunk 면
# 작은 모델도 안정적이고, 호출 수 (페이지당 50→5) 도 충분히 줄어든다.
_BATCH_CHUNK_SIZE = 10
# 같은 ollama 모델에 chunk 호출을 동시에 던질 worker 수. ollama 의
# OLLAMA_NUM_PARALLEL 환경변수와 같이 늘려야 GPU 가 실제로 batch 처리.
# RTX 4070 Laptop (8GB) + paddle (1.5G) + gemma e2b (2G/slot) 환경에서
# slot 2개면 paddle 포함 5.5G 사용으로 안전.
_BATCH_PARALLEL = 2


def _llm_correct_batch_partial(texts: list[str]) -> dict[int, str]:
    """Like _llm_correct_batch but returns a dict {0-based index: text} of
    the lines the LLM successfully returned. Missing lines are simply absent
    from the dict; caller falls back per-line for those.

    Raises httpx errors on transport failure.
    """
    items = "\n".join(f"[{i + 1}] {t}" for i, t in enumerate(texts))
    prompt = _BATCH_PROMPT.format(items=items)
    # 한국어 1글자 = gemma tokenizer 에서 보통 2~3 토큰. 출력은 입력과 비슷한
    # 길이 (약간 더 길 수도 — 띄어쓰기 추가). 충분히 잡아주지 않으면 LLM 이
    # 중간에 잘려 mapping mismatch 발생 → fallback. 8192 까지 허용 (gemma4
    # context 128K 라 여유).
    approx_tokens = sum(len(t) for t in texts) * 3 + 400
    n_predict = min(8192, max(800, approx_tokens))
    r = _client().post(
        f"{_OLLAMA_URL}/api/chat",
        json={
            "model": _OLLAMA_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "keep_alive": "30s",
            "options": {
                "temperature": 0,
                "num_predict": n_predict,
                # KV cache 작게 — 우리 chunk 가 짧아 큰 컨텍스트 불필요.
                # 추론 메모리/속도 모두 이득.
                "num_ctx": 2048,
                "use_mmap": True,
            },
        },
    )
    r.raise_for_status()
    raw = (r.json().get("message") or {}).get("content", "")
    parsed: dict[int, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        m = _RE_NUMBERED_LINE.match(line)
        if not m:
            continue
        n = int(m.group(1))
        if 1 <= n <= len(texts):
            parsed[n - 1] = m.group(2).strip().strip("\"'`")
    if len(parsed) < len(texts):
        log.info(
            "batch chunk partial: got %d of %d lines (will per-line fallback the rest)",
            len(parsed), len(texts),
        )
    return parsed


def apply_correction_batch(texts: list[str], use_llm: bool = True) -> list[str]:
    """Correct a list of OCR lines in chunks of `_BATCH_CHUNK_SIZE`,
    dispatching up to `_BATCH_PARALLEL` chunks concurrently against ollama.

    - 한글이 없는 라인은 통과.
    - use_llm=False 면 Hangul-bearing 라인에 boundary regex 만 적용.
    - 그렇지 않으면 chunks 를 ThreadPoolExecutor 로 병렬 LLM 호출. 각 chunk
      응답 mapping 이 부분적이면 받은 라인만 사용하고 빠진 라인은 per-line
      apply_correction 으로 fallback.
    - 항상 `len(texts)` 와 같은 길이 list 반환. 절대 raise 하지 않음.
    """
    from concurrent.futures import ThreadPoolExecutor

    if not texts:
        return list(texts)
    if not use_llm:
        return [
            _boundary_only(t) if (t and re.search(rf"[{_HANGUL}]", t)) else t
            for t in texts
        ]

    # Hangul-bearing 라인 인덱스만 LLM 으로.
    targets_idx = [
        i for i, t in enumerate(texts)
        if t and t.strip() and re.search(rf"[{_HANGUL}]", t)
    ]
    if not targets_idx:
        return list(texts)

    # Chunks 미리 분할.
    chunks: list[list[int]] = [
        targets_idx[i : i + _BATCH_CHUNK_SIZE]
        for i in range(0, len(targets_idx), _BATCH_CHUNK_SIZE)
    ]

    def _call_chunk(chunk_idx_list: list[int]) -> dict[int, str]:
        chunk_texts = [texts[i] for i in chunk_idx_list]
        try:
            return _llm_correct_batch_partial(chunk_texts)
        except Exception as e:
            log.warning(
                "batch chunk LLM call failed (%s); per-line fallback for this chunk", e,
            )
            return {}

    # 병렬 호출. max_workers=1 이면 직렬 동작과 동일 (안전 fallback).
    workers = max(1, min(_BATCH_PARALLEL, len(chunks)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="spacing-batch") as ex:
        results_by_chunk = list(ex.map(_call_chunk, chunks))

    result = list(texts)

    for chunk_idx_list, corrected in zip(chunks, results_by_chunk):
        for j, src_idx in enumerate(chunk_idx_list):
            if j in corrected and corrected[j]:
                result[src_idx] = corrected[j]
            else:
                # per-line apply_correction (LLM 다시 한 번 호출, 라인별로).
                result[src_idx] = apply_correction(texts[src_idx], use_llm=True)
    return result


def healthcheck() -> bool:
    """True if Ollama is up and the spacing model is registered."""
    try:
        r = _client().get(f"{_OLLAMA_URL}/api/tags", timeout=3.0)
        r.raise_for_status()
        names = [m.get("name") for m in r.json().get("models", [])]
        ok = _OLLAMA_MODEL in names
        if not ok:
            log.warning("Ollama up but model %r not in tags %r", _OLLAMA_MODEL, names)
        return ok
    except Exception:
        log.info("Ollama healthcheck failed (will use boundary fallback)", exc_info=True)
        return False


def warmup() -> None:
    """Probe Ollama and prime an inference. Non-fatal if unreachable."""
    if not healthcheck():
        return
    try:
        apply_correction("안녕하세요반갑습니다")
        log.info("spacing warmup ok (model=%s)", _OLLAMA_MODEL)
    except Exception:
        log.warning("spacing warmup inference failed (non-fatal)", exc_info=True)


def release_model() -> None:
    """Force-unload the spacing LLM from VRAM via keep_alive=0.

    Called by the /api/convert worker once OCR + LLM correction is done so
    the ~2 GB GPU memory is freed for other work. Non-fatal if Ollama is
    unreachable — the model would auto-unload after the keep_alive on the
    last inference call expires anyway.
    """
    try:
        _client().post(
            f"{_OLLAMA_URL}/api/chat",
            json={
                "model": _OLLAMA_MODEL,
                "messages": [],
                "stream": False,
                "keep_alive": 0,
            },
            timeout=5.0,
        )
        log.info("spacing model unloaded (keep_alive=0)")
    except Exception:
        log.debug("spacing release failed (non-fatal)", exc_info=True)
