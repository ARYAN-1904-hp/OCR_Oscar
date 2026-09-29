(() => {
  const form = document.getElementById("upload-form");
  const fileInput = document.getElementById("file-input");
  const folderInput = document.getElementById("folder-input");
  const folderBtn = document.getElementById("folder-btn");
  const dropzone = document.getElementById("dropzone");
  const fileName = document.getElementById("file-name");
  const processBtn = document.getElementById("process-btn");
  const statusEl = document.getElementById("status");
  const resultsPlaceholder = document.getElementById("results-placeholder");
  const resultsText = document.getElementById("results-text");
  const resultsJson = document.getElementById("results-json");
  const copyBtn = document.getElementById("copy-btn");
  const downloadTxtBtn = document.getElementById("download-txt-btn");
  const downloadJsonBtn = document.getElementById("download-json-btn");
  const downloadHtmlBtn = document.getElementById("download-html-btn");
  const storageHint = document.getElementById("storage-hint");
  const tabButtons = Array.from(document.querySelectorAll(".tab"));
  const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
  const processingToggle = document.getElementById("processing-toggle");
  const processingPanel = document.getElementById("processing-panel");
  const pipelineSteps = Array.from(
    document.querySelectorAll("#pipeline-steps .pipeline-step")
  );
  const pipelineMeta = document.getElementById("pipeline-meta");
  const pipelineSuccess = document.getElementById("pipeline-success");
  const pipelineTime = document.getElementById("pipeline-time");

  // Existing FastAPI routes: POST /ocr (1 file), POST /ocr-batch (2–1000 files)
  const API_BASE = (window.OCR_API_BASE || "http://127.0.0.1:8000").replace(
    /\/$/,
    ""
  );
  const API_UPLOAD_URL = `${API_BASE}/ocr`;
  const API_BATCH_URL = `${API_BASE}/ocr-batch`;
  const MAX_FILES = 1000;
  // Folder selection ignores the input's `accept` attribute, so filter here.
  const ALLOWED_EXTENSIONS = new Set(
    (fileInput.getAttribute("accept") || "")
      .split(",")
      .map((ext) => ext.trim().toLowerCase())
      .filter(Boolean)
  );

  const PIPELINE_STEP_IDS = [
    "upload",
    "validate",
    "ocr",
    "layout",
    "post",
    "handler",
    "txt",
    "json",
  ];

  let activeTab = "text";
  let sourceFileBase = "ocr_output";
  let lastPayload = {
    text: "",
    json: "",
  };
  let lastLayoutHtmlUrl = "";
  let pipelineTimer = null;
  let pipelineStartedAt = 0;

  function setDownloadEnabled(enabled) {
    copyBtn.disabled = !enabled;
    downloadTxtBtn.disabled = !enabled;
    downloadJsonBtn.disabled = !enabled;
    if (downloadHtmlBtn) {
      downloadHtmlBtn.disabled = !enabled || !lastLayoutHtmlUrl;
    }
  }

  function downloadFromUrl(url, filename) {
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  }

  function setStorageHint(data) {
    if (!storageHint) {
      return;
    }
    const items = Array.isArray(data && data.results)
      ? data.results
      : data
        ? [data]
        : [];
    const lines = items
      .map((item) => {
        const storage = item && item.storage;
        if (!storage) {
          return "";
        }
        return (
          `${item.filename || storage.document_id}: ` +
          `original → ${storage.storage_path} | ` +
          `outputs → ${storage.output_path} ` +
          `(includes layout.html) | ` +
          `queue=${storage.queue_status || "completed"} | ` +
          `metadata+queue JSON updated`
        );
      })
      .filter(Boolean);

    if (!lines.length) {
      storageHint.hidden = true;
      storageHint.textContent = "";
      return;
    }
    storageHint.hidden = false;
    storageHint.textContent = lines.join("\n");
  }

  function downloadBlob(filename, content, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function safeBaseName(filename) {
    const base = (filename || "ocr_output").replace(/\.[^.]+$/, "");
    return base.replace(/[^\w.\-]+/g, "_") || "ocr_output";
  }

  function setStatus(message, kind) {
    statusEl.textContent = message;
    statusEl.classList.remove("is-busy", "is-error");
    if (kind) {
      statusEl.classList.add(kind);
    }
  }

  function setProcessingExpanded(expanded) {
    if (!processingToggle || !processingPanel) {
      return;
    }
    processingToggle.setAttribute("aria-expanded", expanded ? "true" : "false");
    processingPanel.setAttribute("aria-hidden", expanded ? "false" : "true");
    processingPanel.classList.toggle("is-open", expanded);
  }

  function clearPipelineTimer() {
    if (pipelineTimer) {
      clearInterval(pipelineTimer);
      pipelineTimer = null;
    }
  }

  function resetPipeline() {
    clearPipelineTimer();
    pipelineSteps.forEach((step) => {
      step.classList.remove("is-active", "is-done", "is-error");
      step.classList.add("is-pending");
    });
    if (pipelineMeta) {
      pipelineMeta.hidden = true;
    }
    if (pipelineSuccess) {
      pipelineSuccess.hidden = true;
    }
    if (pipelineTime) {
      pipelineTime.textContent = "—";
    }
  }

  function setPipelineProgress(activeIndex) {
    pipelineSteps.forEach((step, index) => {
      step.classList.remove("is-pending", "is-active", "is-done", "is-error");
      if (index < activeIndex) {
        step.classList.add("is-done");
      } else if (index === activeIndex) {
        step.classList.add("is-active");
      } else {
        step.classList.add("is-pending");
      }
    });
  }

  function startPipelineProgress() {
    resetPipeline();
    setProcessingExpanded(true);
    pipelineStartedAt = performance.now();
    let activeIndex = 0;
    setPipelineProgress(activeIndex);

    pipelineTimer = setInterval(() => {
      if (activeIndex < PIPELINE_STEP_IDS.length - 1) {
        activeIndex += 1;
        setPipelineProgress(activeIndex);
      }
    }, 900);
  }

  function completePipelineSuccess() {
    clearPipelineTimer();
    pipelineSteps.forEach((step) => {
      step.classList.remove("is-pending", "is-active", "is-error");
      step.classList.add("is-done");
    });
    const seconds = Math.max(
      0.1,
      (performance.now() - pipelineStartedAt) / 1000
    );
    if (pipelineTime) {
      pipelineTime.textContent = `${seconds.toFixed(1)} sec`;
    }
    if (pipelineMeta) {
      pipelineMeta.hidden = false;
    }
    if (pipelineSuccess) {
      pipelineSuccess.hidden = false;
    }
    setProcessingExpanded(true);
  }

  function completePipelineError() {
    clearPipelineTimer();
    let markedError = false;
    pipelineSteps.forEach((step) => {
      if (step.classList.contains("is-active") && !markedError) {
        step.classList.remove("is-active", "is-pending", "is-done");
        step.classList.add("is-error");
        markedError = true;
      } else if (
        !step.classList.contains("is-done") &&
        !step.classList.contains("is-error")
      ) {
        step.classList.remove("is-active");
        step.classList.add("is-pending");
      }
    });
    if (!markedError && pipelineSteps.length) {
      const first = pipelineSteps[0];
      first.classList.remove("is-pending", "is-active", "is-done");
      first.classList.add("is-error");
    }
    const seconds = Math.max(
      0.1,
      (performance.now() - pipelineStartedAt) / 1000
    );
    if (pipelineTime) {
      pipelineTime.textContent = `${seconds.toFixed(1)} sec`;
    }
    if (pipelineMeta) {
      pipelineMeta.hidden = false;
    }
    if (pipelineSuccess) {
      pipelineSuccess.hidden = true;
    }
    setProcessingExpanded(true);
  }

  function switchTab(tabName) {
    activeTab = tabName;

    tabButtons.forEach((btn) => {
      const selected = btn.dataset.tab === tabName;
      btn.classList.toggle("is-active", selected);
      btn.setAttribute("aria-selected", selected ? "true" : "false");
    });

    tabPanels.forEach((panel) => {
      const selected = panel.dataset.panel === tabName;
      panel.classList.toggle("is-active", selected);
      panel.hidden = !selected;
    });
  }

  function extractTextFromResponse(data) {
    if (typeof data.plain_text === "string" && data.plain_text.trim()) {
      return data.plain_text;
    }
    if (typeof data.formatted_text === "string" && data.formatted_text.trim()) {
      return data.formatted_text;
    }
    if (typeof data.text === "string" && data.text.trim()) {
      return data.text;
    }
    return "";
  }

  function getSelectedFiles() {
    return Array.from((fileInput.files && fileInput.files) || []);
  }

  function setSelectedFiles(fileList) {
    const dt = new DataTransfer();
    Array.from(fileList || [])
      .slice(0, MAX_FILES)
      .forEach((file) => dt.items.add(file));
    fileInput.files = dt.files;
  }

  function isAllowedFile(file) {
    const name = (file && file.name) || "";
    const dot = name.lastIndexOf(".");
    if (dot <= 0) {
      return false;
    }
    return ALLOWED_EXTENSIONS.has(name.slice(dot).toLowerCase());
  }

  // Filter to supported types, cap at MAX_FILES, and report what was skipped.
  function applySelection(fileList) {
    const all = Array.from(fileList || []);
    const supported = all.filter(isAllowedFile);
    const skipped = all.length - supported.length;
    setSelectedFiles(supported);
    updateFileLabel();

    const notes = [];
    if (skipped) {
      notes.push(`Skipped ${skipped} unsupported file${skipped === 1 ? "" : "s"}.`);
    }
    if (supported.length > MAX_FILES) {
      notes.push(
        `Found ${supported.length} documents; only the first ${MAX_FILES} were kept.`
      );
    }
    if (!supported.length && all.length) {
      setStatus("No supported documents found in the selection.", "is-error");
    } else if (notes.length) {
      setStatus(notes.join(" "), supported.length > MAX_FILES ? "is-error" : "");
    }
  }

  function readAllDirectoryEntries(reader) {
    // readEntries returns results in chunks (~100); call until empty.
    return new Promise((resolve, reject) => {
      const entries = [];
      const readChunk = () => {
        reader.readEntries((chunk) => {
          if (!chunk.length) {
            resolve(entries);
            return;
          }
          entries.push(...chunk);
          readChunk();
        }, reject);
      };
      readChunk();
    });
  }

  async function collectFilesFromEntry(entry, out) {
    if (!entry) {
      return;
    }
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) =>
        entry.file(resolve, reject)
      );
      out.push(file);
    } else if (entry.isDirectory) {
      const children = await readAllDirectoryEntries(entry.createReader());
      for (const child of children) {
        await collectFilesFromEntry(child, out);
      }
    }
  }

  async function filesFromDataTransfer(dataTransfer) {
    const items = Array.from((dataTransfer && dataTransfer.items) || []);
    const entries = items
      .filter((item) => item.kind === "file")
      .map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null));
    if (!entries.length || entries.some((entry) => !entry)) {
      return Array.from((dataTransfer && dataTransfer.files) || []);
    }
    const out = [];
    for (const entry of entries) {
      await collectFilesFromEntry(entry, out);
    }
    return out;
  }

  function updateFileLabel() {
    const files = getSelectedFiles();
    if (!files.length) {
      fileName.textContent = "No documents selected";
      fileName.classList.remove("has-file");
      processBtn.disabled = true;
      processBtn.textContent = "Process Documents";
      return;
    }

    if (files.length === 1) {
      fileName.textContent = files[0].name;
      processBtn.textContent = "Process Document";
    } else {
      const shown = files.slice(0, 5).map((f) => f.name).join(", ");
      const more = files.length > 5 ? ` and ${files.length - 5} more` : "";
      fileName.textContent = `${files.length} files selected: ${shown}${more}`;
      processBtn.textContent = `Process ${files.length} Documents`;
    }
    fileName.classList.add("has-file");
    processBtn.disabled = false;
  }

  function showResults(data, fileCount) {
    let text = "";
    let jsonSource = data;
    lastLayoutHtmlUrl = "";

    if (Array.isArray(data.results)) {
      text = data.results
        .map((item, index) => {
          const name = item.filename || `Document ${index + 1}`;
          const body = item.error
            ? `ERROR: ${item.error}`
            : extractTextFromResponse(item) || "No extracted text returned.";
          return `===== ${name} =====\n${body}`;
        })
        .join("\n\n");
      jsonSource = data;
      sourceFileBase =
        fileCount > 1
          ? `batch_${fileCount}_docs`
          : safeBaseName(
              (data.results[0] && data.results[0].filename) || "ocr_output"
            );
      const first = data.results[0];
      if (first && first.layout_html_url) {
        lastLayoutHtmlUrl = first.layout_html_url.startsWith("http")
          ? first.layout_html_url
          : `${API_BASE}${first.layout_html_url}`;
      }
    } else {
      text =
        `===== ${data.filename || "Document 1"} =====\n` +
        (extractTextFromResponse(data) || "No extracted text returned.");
      jsonSource = data.raw_json || data;
      sourceFileBase = safeBaseName(data.filename || "ocr_output");
      if (data.layout_html_url) {
        lastLayoutHtmlUrl = data.layout_html_url.startsWith("http")
          ? data.layout_html_url
          : `${API_BASE}${data.layout_html_url}`;
      }
    }

    const jsonText = JSON.stringify(jsonSource, null, 2);
    resultsText.textContent = text;
    resultsJson.textContent = jsonText;

    lastPayload = {
      text: resultsText.textContent,
      json: jsonText,
    };

    setStorageHint(data);
    resultsPlaceholder.hidden = true;
    setDownloadEnabled(true);
    switchTab(activeTab || "text");
  }

  function clearResults() {
    resultsText.textContent = "";
    resultsJson.textContent = "";
    lastPayload = { text: "", json: "" };
    lastLayoutHtmlUrl = "";
    setStorageHint(null);
    setDownloadEnabled(false);
    resultsPlaceholder.hidden = false;
    tabPanels.forEach((panel) => {
      panel.hidden = true;
      panel.classList.remove("is-active");
    });
    activeTab = "text";
    tabButtons.forEach((btn) => {
      const selected = btn.dataset.tab === "text";
      btn.classList.toggle("is-active", selected);
      btn.setAttribute("aria-selected", selected ? "true" : "false");
    });
  }

  function formatErrorDetail(detail) {
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      return detail
        .map((item) => item.msg || JSON.stringify(item))
        .join("; ");
    }
    if (detail && typeof detail === "object") {
      return detail.message || JSON.stringify(detail);
    }
    return "Request failed";
  }

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (resultsPlaceholder.hidden) {
        switchTab(btn.dataset.tab);
      } else {
        switchTab(btn.dataset.tab);
        tabPanels.forEach((panel) => {
          panel.hidden = true;
          panel.classList.remove("is-active");
        });
        resultsPlaceholder.hidden = false;
      }
    });
  });

  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.add("is-dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      dropzone.classList.remove("is-dragover");
    });
  });

  dropzone.addEventListener("drop", async (event) => {
    const dataTransfer = event.dataTransfer;
    if (!dataTransfer || !dataTransfer.items || !dataTransfer.items.length) {
      return;
    }
    setStatus("Reading dropped files…", "is-busy");
    try {
      const files = await filesFromDataTransfer(dataTransfer);
      setStatus("", "");
      applySelection(files);
    } catch (err) {
      setStatus(`Could not read dropped folder: ${err.message || err}`, "is-error");
    }
  });

  folderBtn.addEventListener("click", () => {
    folderInput.value = "";
    folderInput.click();
  });

  folderInput.addEventListener("change", () => {
    setStatus("", "");
    applySelection(folderInput.files);
  });

  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    setStatus("", "");
    applySelection(getSelectedFiles());
  });

  if (processingToggle && processingPanel) {
    processingToggle.addEventListener("click", () => {
      const expanded =
        processingToggle.getAttribute("aria-expanded") === "true";
      setProcessingExpanded(!expanded);
    });
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const files = getSelectedFiles();
    if (!files.length) {
      setStatus("Select at least one document before processing.", "is-error");
      return;
    }
    if (files.length > MAX_FILES) {
      setStatus(`Maximum ${MAX_FILES} files allowed per batch.`, "is-error");
      return;
    }

    const isBatch = files.length > 1;
    const body = new FormData();
    if (isBatch) {
      files.forEach((file) => body.append("files", file));
    } else {
      body.append("file", files[0]);
    }

    processBtn.disabled = true;
    clearResults();
    startPipelineProgress();
    setStatus(
      isBatch
        ? `Processing ${files.length} documents…`
        : "Processing document…",
      "is-busy"
    );

    try {
      const response = await fetch(isBatch ? API_BATCH_URL : API_UPLOAD_URL, {
        method: "POST",
        body,
      });

      if (!response.ok) {
        let detail = `Request failed (${response.status})`;
        try {
          const err = await response.json();
          detail = formatErrorDetail(err.detail || err.message || detail);
        } catch (_) {
          /* ignore */
        }
        throw new Error(detail);
      }

      const data = await response.json();
      showResults(data, files.length);
      completePipelineSuccess();
      // Hand back a folder (zip) with one <filename>.txt per image.
      const collection = data && data.text_collection;
      const zipUrl = collection && collection.zip_url;
      if (zipUrl) {
        downloadFromUrl(`${API_BASE}${zipUrl}`, `${collection.batch_id}.zip`);
      }
      const failed = Array.isArray(data.results)
        ? data.results.filter((item) => item && item.error).length
        : 0;
      const storedId =
        (data && data.document_id) ||
        (data &&
          data.results &&
          data.results[0] &&
          data.results[0].document_id) ||
        "";
      setStatus(
        isBatch
          ? `OCR completed for ${files.length - failed} of ${files.length} documents` +
              (failed ? ` (${failed} failed)` : "") +
              (zipUrl
                ? ". Downloaded a folder (zip) with one TXT file per document."
                : ".")
          : storedId
            ? `OCR completed. Saved layout.html in handler_checker/storage/outputs/${storedId}/`
            : "OCR Processing Completed Successfully"
      );
    } catch (error) {
      completePipelineError();
      const message = error.message || "Processing failed.";
      const hint =
        message.includes("Failed to fetch") || message.includes("NetworkError")
          ? " Start the API with: uvicorn app.main:app --reload --port 8000"
          : "";
      setStatus(message + hint, "is-error");
    } finally {
      processBtn.disabled = getSelectedFiles().length === 0;
    }
  });

  copyBtn.addEventListener("click", async () => {
    const text = activeTab === "json" ? lastPayload.json : lastPayload.text;

    if (!text) {
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setStatus("Copied active tab to clipboard.");
    } catch (_) {
      setStatus("Could not copy text.", "is-error");
    }
  });

  downloadTxtBtn.addEventListener("click", () => {
    if (!lastPayload.text) {
      return;
    }
    downloadBlob(
      `${sourceFileBase}_formatted_text.txt`,
      lastPayload.text,
      "text/plain;charset=utf-8"
    );
    setStatus("Downloaded TXT file.");
  });

  downloadJsonBtn.addEventListener("click", () => {
    if (!lastPayload.json) {
      return;
    }
    downloadBlob(
      `${sourceFileBase}_raw_output.json`,
      lastPayload.json,
      "application/json;charset=utf-8"
    );
    setStatus("Downloaded JSON file.");
  });

  if (downloadHtmlBtn) {
    downloadHtmlBtn.addEventListener("click", () => {
      if (!lastLayoutHtmlUrl) {
        setStatus("layout.html not available for this result.", "is-error");
        return;
      }
      downloadFromUrl(lastLayoutHtmlUrl, `${sourceFileBase}_layout.html`);
      setStatus("Downloaded layout.html (also saved in storage/outputs).");
    });
  }

})();
