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
#
# Prompt 구성: system + few-shot + user-data-only.
# - 규칙은 system 메시지로 분리 → 작은 모델(e2b)에서 형식 일탈/지시 누락↓
# - 1쌍의 few-shot (한글 띄어쓰기 / Latin 분리 / 자모 오탈자 교정 모두 cover)
# - system + few-shot 부분은 모든 호출에 동일 → ollama 의 KV cache prefix
#   reuse 로 prompt eval 비용이 cached call 에서 ~3배 빨라짐 (실측). 페이지가
#   누적될수록 추가 부담 없이 안정성·정확도만 확보.
_BATCH_SYS_PROMPT = (
    "당신은 OCR로 추출된 한국어 텍스트의 띄어쓰기와 오탈자만 보정하는 도구입니다. 다음 규칙을 절대 어기지 마세요.\n\n"
    "[허용되는 변경]\n"
    "1. 한글 단어 사이의 공백 추가/삭제\n"
    "2. 한글과 영문/숫자/기호 사이의 공백 추가/삭제\n"
    "3. 단일 한글 자모 오탈자 (예: \"역\"→\"억\", \"지\"→\"치\" 처럼 인접 자모 오인식)\n\n"
    "[절대 금지]\n"
    "1. 영문 단어, 숫자, 구두점(. , : ; \" ' ( ) [ ])과 기호를 추가/삭제/변경하지 마세요.\n"
    "2. 단어를 다른 단어로 의역/대체하지 마세요. 의미가 어색해도 자모 오인식이 아니면 그대로 두세요.\n"
    "3. 줄을 합치거나 분리하지 마세요. 입력 줄 수와 출력 줄 수는 반드시 같아야 합니다.\n"
    "4. 입력에 없던 콤마, 마침표, 따옴표를 새로 넣지 마세요.\n\n"
    "[출력 형식]\n"
    "- 입력의 각 줄은 \"[N] 본문\" 형태입니다. 출력도 동일하게 \"[N] 교정본문\" 형태로, 같은 N을 사용해 같은 순서로 출력하세요.\n"
    "- 교정할 부분이 없으면 원문을 그대로 출력하세요. 절대 줄을 생략하지 마세요.\n"
    "- \"[N] 본문\" 줄 외에 어떤 텍스트도 출력하지 마세요. 헤더, 빈 줄, 설명, \"출력:\" 같은 라벨, 코드블록, 인용부호 모두 금지."
)
_BATCH_FEWSHOT_USER = (
    "입력:\n"
    "[1] 안녕하세요반갑습니다\n"
    "[2] Pythonprogramming은인기있다\n"
    "[3] 수십역원규모이다\n\n"
    "출력:"
)
_BATCH_FEWSHOT_ASSISTANT = (
    "[1] 안녕하세요 반갑습니다\n"
    "[2] Python programming은 인기있다\n"
    "[3] 수십억 원 규모이다"
)

# LLM 이 출력하는 numbered line 의 다양한 형식 ([1], 1., 1:, 1)) 모두 매칭.
_RE_NUMBERED_LINE = re.compile(r"^\s*[\[\(]?(\d+)[\]\)\.\:]\s*(.*)$")

# 한 번 LLM 호출에 보낼 최대 라인 수. system + few-shot 분리한 v2 프롬프트
# 기준 25줄까지 mapping 100%, ASCII 보존 100% 실측. 안전 마진 두고 20 으로 둠.
# 페이지당 호출 수가 (50줄 기준) 5→3 으로 줄어 wall time ~11% 단축.
_BATCH_CHUNK_SIZE = 20
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
    messages = [
        {"role": "system", "content": _BATCH_SYS_PROMPT},
        {"role": "user", "content": _BATCH_FEWSHOT_USER},
        {"role": "assistant", "content": _BATCH_FEWSHOT_ASSISTANT},
        {"role": "user", "content": f"입력:\n{items}\n\n출력:"},
    ]
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
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "30s",
            "options": {
                "temperature": 0,
                "num_predict": n_predict,
                # system + few-shot 추가로 prefix 가 ~650 tok. chunk=20 의
                # 입력+출력까지 합치면 1.5~2K. 4096 이면 안전 마진 충분하고,
                # 동일 num_ctx 유지해야 ollama prefix KV cache 가 호출 간 재사용됨.
                "num_ctx": 4096,
                "use_mmap": True,
                # 모델이 응답 끝낸 뒤 다음 입력 라벨을 hallucinate 하는 케이스
                # 방지. 측정상 트리거 빈도는 낮지만 안전장치.
                "stop": ["입력:", "원문:", "참고:"],
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
