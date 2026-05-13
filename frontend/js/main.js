// PDF 변환기 — 프론트엔드
// SortableJS로 파일 순서 편집, 회전 버튼, OCR 토글, 백엔드 호출.

(function () {
  "use strict";

  const ACCEPT = /\.(pdf|jpe?g|png|bmp|tiff?|webp|gif|docx?|hwpx?|odt|rtf|xlsx?|ods|csv|pptx?|odp|txt|html?)$/i;
  const state = {
    files: [],   // [{id, file, rotation}]
    nextId: 1,
  };

  // ── PDF.js (썸네일 렌더링용) ──────────────────────────
  // ESM 빌드를 동적 import 해서 worker 까지 자동 셋업.
  // 실패 시 PDF 썸네일은 generic 아이콘으로 떨어짐.
  let _pdfjsPromise = null;
  function _pdfjs() {
    if (_pdfjsPromise) return _pdfjsPromise;
    _pdfjsPromise = import("https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.min.mjs")
      .then((m) => {
        m.GlobalWorkerOptions.workerSrc =
          "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.10.38/pdf.worker.min.mjs";
        return m;
      })
      .catch(() => null);
    return _pdfjsPromise;
  }

  const THUMB_W = 96;  // px (CSS 도 같은 값)
  const THUMB_H = 124; // px (A4 비율 약간 닮게)

  async function makeThumbnail(file) {
    const lower = (file.name || "").toLowerCase();
    if (/\.(jpe?g|png|bmp|tiff?|webp|gif)$/i.test(lower)) {
      return await _imageThumb(file);
    }
    if (/\.pdf$/i.test(lower)) {
      return await _pdfThumb(file);
    }
    return null;  // 오피스 등 → 일반 아이콘
  }

  async function _imageThumb(file) {
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        try {
          const canvas = document.createElement("canvas");
          canvas.width = THUMB_W;
          canvas.height = THUMB_H;
          const ctx = canvas.getContext("2d");
          ctx.fillStyle = "#fff";
          ctx.fillRect(0, 0, THUMB_W, THUMB_H);
          // 비율 유지 + 중앙 정렬
          const scale = Math.min(THUMB_W / img.width, THUMB_H / img.height);
          const w = img.width * scale;
          const h = img.height * scale;
          ctx.drawImage(img, (THUMB_W - w) / 2, (THUMB_H - h) / 2, w, h);
          URL.revokeObjectURL(url);
          resolve(canvas.toDataURL("image/jpeg", 0.7));
        } catch (e) {
          URL.revokeObjectURL(url);
          resolve(null);
        }
      };
      img.onerror = () => { URL.revokeObjectURL(url); resolve(null); };
      img.src = url;
    });
  }

  async function _pdfThumb(file) {
    const pdfjs = await _pdfjs();
    if (!pdfjs) return null;
    try {
      const buf = await file.arrayBuffer();
      const pdf = await pdfjs.getDocument({ data: buf }).promise;
      const page = await pdf.getPage(1);
      const baseViewport = page.getViewport({ scale: 1.0 });
      const scale = Math.min(THUMB_W / baseViewport.width, THUMB_H / baseViewport.height);
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement("canvas");
      canvas.width = THUMB_W;
      canvas.height = THUMB_H;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, THUMB_W, THUMB_H);
      // 페이지를 캔버스 중앙에 그림
      const offsetX = (THUMB_W - viewport.width) / 2;
      const offsetY = (THUMB_H - viewport.height) / 2;
      await page.render({
        canvasContext: ctx,
        viewport,
        transform: [1, 0, 0, 1, offsetX, offsetY],
      }).promise;
      try { pdf.destroy(); } catch (e) { /* ignore */ }
      return canvas.toDataURL("image/jpeg", 0.7);
    } catch (e) {
      return null;
    }
  }

  // ─── 변환 탭: 드롭존 + 파일 목록 ─────────────────
  const dz = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const listEl = document.getElementById("file-list");
  const convertBtn = document.getElementById("convert-btn");
  const ocrToggle = document.getElementById("ocr-toggle");
  const llmToggle = document.getElementById("llm-toggle");

  // OCR 꺼지면 LLM 토글도 의미 없음 → disable
  function updateLlmToggleState() {
    llmToggle.disabled = !ocrToggle.checked;
    if (!ocrToggle.checked) {
      llmToggle.parentElement.classList.add("disabled");
    } else {
      llmToggle.parentElement.classList.remove("disabled");
    }
  }
  ocrToggle.addEventListener("change", updateLlmToggleState);
  updateLlmToggleState();

  dz.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => addFiles(e.target.files));
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag-over"); })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag-over"); })
  );
  dz.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

  function addFiles(fileList) {
    const skipped = [];
    const newItems = [];
    for (const f of fileList) {
      if (!ACCEPT.test(f.name)) { skipped.push(f.name); continue; }
      const item = { id: state.nextId++, file: f, rotation: 0, thumbnail: null, merge: true };
      state.files.push(item);
      newItems.push(item);
    }
    render();
    if (skipped.length) setStatus(`지원하지 않는 형식: ${skipped.join(", ")}`, "error");
    else setStatus("");
    fileInput.value = "";
    // 새로 추가된 파일들의 썸네일을 백그라운드로 생성. 끝나면 그 아이템만 in-place
    // 갱신 후 부분 re-render.
    for (const item of newItems) {
      makeThumbnail(item.file).then((thumb) => {
        if (!state.files.includes(item)) return;  // 사이에 제거됐을 수도
        item.thumbnail = thumb;
        render();
      }).catch(() => { /* ignore */ });
    }
  }

  function render() {
    listEl.innerHTML = "";
    for (const item of state.files) {
      const li = document.createElement("li");
      li.className = "file-item";
      li.dataset.id = String(item.id);
      const thumbHtml = item.thumbnail
        ? `<img class="thumb" src="${item.thumbnail}" alt="" style="transform: rotate(${item.rotation}deg)" />`
        : `<span class="thumb thumb-placeholder" style="transform: rotate(${item.rotation}deg)">${_iconForFile(item.file.name)}</span>`;
      li.innerHTML = `
        <span class="handle" title="드래그해서 순서 변경">⋮⋮</span>
        ${thumbHtml}
        <div class="info">
          <div class="name" title="${escapeHtml(item.file.name)}">${escapeHtml(item.file.name)}</div>
          <div class="meta">${formatSize(item.file.size)} · ${item.rotation}°</div>
        </div>
        <label class="merge-toggle" title="체크하면 다른 체크된 파일들과 한 PDF로 병합. 체크 해제하면 단독 PDF로 별도 다운로드.">
          <input type="checkbox" data-act="merge" ${item.merge ? "checked" : ""} />
          <span>병합</span>
        </label>
        <button class="rotate-btn" data-act="rotate" title="90° 회전">↻ 회전</button>
        <button class="remove-btn" data-act="remove" title="제거">✕</button>
      `;
      listEl.appendChild(li);
    }
    convertBtn.disabled = state.files.length === 0;
  }

  function _iconForFile(name) {
    const lower = (name || "").toLowerCase();
    if (/\.pdf$/i.test(lower)) return "📄";
    if (/\.(docx?|hwpx?|odt|rtf|txt|html?)$/i.test(lower)) return "📝";
    if (/\.(xlsx?|ods|csv)$/i.test(lower)) return "📊";
    if (/\.(pptx?|odp)$/i.test(lower)) return "📊";
    if (/\.(jpe?g|png|bmp|tiff?|webp|gif)$/i.test(lower)) return "🖼";
    return "📎";
  }

  listEl.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-act]");
    if (!btn) return;
    const li = e.target.closest(".file-item");
    const id = Number(li.dataset.id);
    const item = state.files.find((f) => f.id === id);
    if (!item) return;
    if (btn.dataset.act === "rotate") {
      item.rotation = (item.rotation + 90) % 360;
    } else if (btn.dataset.act === "remove") {
      state.files = state.files.filter((f) => f.id !== id);
    }
    render();
  });

  // SortableJS — 드래그로 순서 변경
  // 체크박스 (병합 토글) 변경
  listEl.addEventListener("change", (e) => {
    const cb = e.target;
    if (!(cb instanceof HTMLInputElement) || cb.dataset.act !== "merge") return;
    const li = e.target.closest(".file-item");
    if (!li) return;
    const id = Number(li.dataset.id);
    const item = state.files.find((f) => f.id === id);
    if (!item) return;
    item.merge = cb.checked;
    // 메타에만 영향 — 썸네일/회전 표시는 그대로라 re-render 불필요.
  });

  // eslint-disable-next-line no-undef
  Sortable.create(listEl, {
    animation: 150,
    handle: ".handle",
    ghostClass: "sortable-ghost",
    onEnd: () => {
      const newOrder = Array.from(listEl.children).map((li) => Number(li.dataset.id));
      state.files.sort((a, b) => newOrder.indexOf(a.id) - newOrder.indexOf(b.id));
    },
  });

  convertBtn.addEventListener("click", async () => {
    if (!state.files.length) return;
    const fd = new FormData();
    const meta = state.files.map((f) => ({
      rotation: f.rotation,
      merge: f.merge !== false,
    }));
    for (const f of state.files) fd.append("files", f.file, f.file.name);
    fd.append("meta", JSON.stringify(meta));
    fd.append("ocr_enabled", ocrToggle.checked ? "true" : "false");
    fd.append("llm_enhance", (ocrToggle.checked && llmToggle.checked) ? "true" : "false");

    setStatusProgress(0, "시작 중…");
    convertBtn.disabled = true;
    try {
      const res = await fetch("/api/convert", { method: "POST", body: fd });
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(errText || `HTTP ${res.status}`);
      }
      // NDJSON streaming: 한 줄씩 파싱, 마지막 done 이벤트의 base64 PDF 를 다운로드
      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";
      let lastError = null;
      let donePayload = null;
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        let nl;
        while ((nl = buf.indexOf("\n")) !== -1) {
          const line = buf.slice(0, nl).trim();
          buf = buf.slice(nl + 1);
          if (!line) continue;
          let evt;
          try { evt = JSON.parse(line); }
          catch { continue; }
          if (evt.type === "queued") {
            setStatusQueued(evt.position || 1, evt.ahead || 0);
          } else if (evt.type === "progress") {
            setStatusProgress(evt.percent || 0, evt.label || "");
          } else if (evt.type === "done") {
            donePayload = evt;
          } else if (evt.type === "cancelled") {
            lastError = "취소되었습니다";
          } else if (evt.type === "error") {
            lastError = evt.message || "알 수 없는 오류";
          }
        }
      }
      if (lastError) throw new Error(lastError);
      if (!donePayload) throw new Error("응답이 끊김");
      const fallback = donePayload.mime === "application/zip" ? "converted.zip" : "merged.pdf";
      showDownloadReady(donePayload.download_url, donePayload.filename || fallback);
    } catch (err) {
      setStatus(`실패: ${err.message || err}`, "error");
    } finally {
      convertBtn.disabled = state.files.length === 0;
    }
  });

  // ─── 유틸 ──────────────────────────────────────
  function setStatus(msg, kind) {
    const el = document.getElementById("status");
    el.className = "status" + (kind ? ` ${kind}` : "");
    if (kind === "busy") {
      el.innerHTML = '<span class="status-text"></span><span class="loader-bar" role="progressbar" aria-label="진행 중"></span>';
      el.querySelector(".status-text").textContent = msg || "";
    } else {
      el.textContent = msg || "";
    }
  }

  // setStatus 의 busy 상태에서 determinate 진행률 바를 표시한다.
  function setStatusProgress(percent, label) {
    const el = document.getElementById("status");
    el.className = "status busy";
    const pct = Math.max(0, Math.min(100, Math.round(percent)));
    el.innerHTML =
      '<span class="status-text"></span>' +
      '<span class="loader-bar determinate" role="progressbar" aria-valuemin="0" aria-valuemax="100"' +
      ' aria-valuenow="' + pct + '" style="--progress: ' + pct + '%"></span>';
    el.querySelector(".status-text").textContent = label
      ? (pct + "% — " + label)
      : (pct + "%");
  }

  // 큐 대기 중 상태 — 인디터미닛 슬라이드 + "대기 중 (N번째)" 라벨.
  function setStatusQueued(position, ahead) {
    const el = document.getElementById("status");
    el.className = "status busy";
    const label = ahead && ahead > 0
      ? `대기 중 (${position}번째 / 앞에 ${ahead}명)`
      : "곧 시작합니다…";
    el.innerHTML =
      '<span class="status-text"></span>' +
      '<span class="loader-bar" role="progressbar" aria-label="대기 중"></span>';
    el.querySelector(".status-text").textContent = label;
  }

  function formatSize(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
  }

  function showDownloadReady(downloadUrl, filename) {
    const el = document.getElementById("status");
    el.className = "status ok";
    el.innerHTML = "";

    const msg = document.createElement("span");
    msg.className = "status-text";
    msg.textContent = "완료. 아래 버튼을 눌러 다운로드하세요. ";
    el.appendChild(msg);

    const absoluteUrl = new URL(downloadUrl, window.location.origin).toString();
    console.log("[pymu] download ready:", absoluteUrl, filename);

    const a = document.createElement("a");
    a.href = absoluteUrl;
    a.download = filename;
    // Top-frame navigation is the fallback if the explicit click handler
    // below doesn't run for some reason. Server's Content-Disposition
    // turns the navigation into a download regardless of target.
    a.target = "_top";
    a.rel = "noopener";
    a.className = "primary download-link";
    a.textContent = `⬇ ${filename}`;

    // Downloading from inside the iframe is blocked by two layers:
    //   1. Chrome's cross-origin iframe download policy (silent drop).
    //   2. Tistory wraps embedded iframes in a sandbox without
    //      allow-top-navigation, so writing window.top.location throws
    //      a SecurityError.
    // Path that works: postMessage the URL up to the parent (blog),
    // which is the top frame and unsandboxed. The parent navigates
    // itself with window.location.href; the server's
    // Content-Disposition: attachment intercepts the navigation as a
    // download, so the blog page itself stays put.
    a.addEventListener("click", function (ev) {
      ev.preventDefault();
      console.log("[pymu] download clicked:", absoluteUrl);

      // Best effort: try top-level navigation first for non-sandboxed
      // embeds. Falls through to postMessage on SecurityError.
      try {
        if (window.top && window.top !== window) {
          window.top.location.href = absoluteUrl;
          return;
        }
      } catch (_) {
        // Sandboxed — fall through to postMessage.
      }

      if (window.parent && window.parent !== window) {
        try {
          window.parent.postMessage(
            { type: "pymu:download", url: absoluteUrl, filename: filename },
            "*"
          );
          return;
        } catch (err) {
          console.warn("[pymu] postMessage failed:", err);
        }
      }

      // Last resort (not embedded, or postMessage threw): self-navigate.
      window.location.href = absoluteUrl;
    });

    el.appendChild(a);
  }
})();
