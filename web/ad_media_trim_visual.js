import { app } from "../../scripts/app.js";

const TARGET_NODE = "AD_media_trim_visual";

function getWidget(node, name) {
  return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetValue(widget, value) {
  if (!widget) return;
  widget.value = value;
  if (typeof widget.callback === "function") {
    widget.callback(value);
  }
}

async function resolveMedia(path) {
  const resp = await fetch("/apt_preset/media_trim/resolve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  const data = await resp.json();
  if (!resp.ok || !data?.ok) {
    throw new Error(data?.error || `HTTP ${resp.status}`);
  }
  return data;
}

function formatSec(sec) {
  const v = Math.max(0, Number(sec || 0));
  const m = Math.floor(v / 60);
  const s = (v % 60).toFixed(2).padStart(5, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function looksLikeMediaPath(v) {
  if (typeof v !== "string") return false;
  const s = v.trim().toLowerCase();
  return (
    s.startsWith("file://") ||
    s.includes("\\") ||
    s.includes("/") ||
    /\.(mp4|mov|mkv|webm|avi|mp3|wav|flac|m4a|aac|ogg)(\?|#|$)/.test(s)
  );
}

function collectPathCandidatesFromNode(node) {
  const out = [];
  if (!node) return out;
  (node.widgets || []).forEach((w) => {
    const v = w?.value;
    if (looksLikeMediaPath(v)) out.push(String(v));
  });
  const props = node.properties || {};
  Object.values(props).forEach((v) => {
    if (looksLikeMediaPath(v)) out.push(String(v));
  });
  return out;
}

function getUpstreamNodeByInput(node, inputName) {
  const idx = (node.inputs || []).findIndex((i) => i?.name === inputName);
  if (idx < 0) return null;
  const linkId = node.inputs[idx]?.link;
  if (linkId == null || !app.graph?.links) return null;
  const info = app.graph.links[linkId];
  if (!info?.origin_id) return null;
  return app.graph.getNodeById?.(info.origin_id) || app.graph._nodes?.find((n) => n.id === info.origin_id) || null;
}

function resolvePathFromVideoPort(node) {
  const visited = new Set();
  const queue = [];
  const first = getUpstreamNodeByInput(node, "video");
  if (first) queue.push(first);

  while (queue.length) {
    const cur = queue.shift();
    if (!cur || visited.has(cur.id)) continue;
    visited.add(cur.id);
    const candidates = collectPathCandidatesFromNode(cur);
    if (candidates.length) return candidates[0];
    (cur.inputs || []).forEach((inp) => {
      const linkId = inp?.link;
      if (linkId == null || !app.graph?.links) return;
      const info = app.graph.links[linkId];
      const up = info?.origin_id
        ? app.graph.getNodeById?.(info.origin_id) || app.graph._nodes?.find((n) => n.id === info.origin_id)
        : null;
      if (up && !visited.has(up.id)) queue.push(up);
    });
  }
  return "";
}

function openTrimModal(node) {
  const pathWidget = getWidget(node, "media_path");
  const startWidget = getWidget(node, "start_sec");
  const endWidget = getWidget(node, "end_sec");
  const markersWidget = getWidget(node, "markers_json");
  const manualPath = String(pathWidget?.value || "").trim();
  const portPath = resolvePathFromVideoPort(node);
  const sourcePath = manualPath || portPath;

  if (!sourcePath) {
    alert("未找到可预览媒体：请连接可回溯到本地文件路径的 video 端口，或填写 media_path。");
    return;
  }

  const overlay = document.createElement("div");
  overlay.className = "apt-trim-overlay";
  overlay.innerHTML = `
    <div class="apt-trim-modal">
      <div class="apt-trim-head">
        <div class="apt-trim-title">音视频切割</div>
        <button class="apt-trim-close">✕</button>
      </div>
      <div class="apt-trim-body">
        <div class="apt-trim-status">加载媒体中...</div>
        <div class="apt-trim-player-wrap"></div>
        <div class="apt-wave-wrap">
          <canvas class="apt-wave-canvas" width="860" height="140"></canvas>
        </div>
        <div class="apt-playhead-label">Playhead: 00:00.00</div>
        <div class="apt-trim-controls" style="display:none;">
          <div class="apt-trim-row">
            <span>开始</span>
            <input class="apt-trim-start" type="range" min="0" max="1" step="0.01" value="0" />
            <span class="apt-trim-start-label">00:00.00</span>
          </div>
          <div class="apt-trim-row">
            <span>结束</span>
            <input class="apt-trim-end" type="range" min="0" max="1" step="0.01" value="1" />
            <span class="apt-trim-end-label">00:00.00</span>
          </div>
          <div class="apt-trim-row apt-trim-actions">
            <button class="apt-set-start">开始=当前位置</button>
            <button class="apt-set-end">结束=当前位置</button>
            <button class="apt-add-marker">添加标记</button>
            <button class="apt-clear-markers">清空标记</button>
            <button class="apt-preview-loop">循环预览</button>
            <button class="apt-apply">应用到节点</button>
          </div>
          <div class="apt-trim-row">
            <span>标记</span>
            <div class="apt-marker-list"></div>
          </div>
        </div>
      </div>
    </div>
  `;
  document.body.appendChild(overlay);

  const closeModal = () => {
    const media = overlay.querySelector("video, audio");
    if (media) {
      media.pause();
      media.removeAttribute("src");
    }
    overlay.remove();
  };

  overlay.querySelector(".apt-trim-close")?.addEventListener("click", closeModal);
  overlay.addEventListener("click", (e) => {
    if (e.target === overlay) closeModal();
  });

  const statusEl = overlay.querySelector(".apt-trim-status");
  const playerWrap = overlay.querySelector(".apt-trim-player-wrap");
  const controlsEl = overlay.querySelector(".apt-trim-controls");
  const startRange = overlay.querySelector(".apt-trim-start");
  const endRange = overlay.querySelector(".apt-trim-end");
  const startLabel = overlay.querySelector(".apt-trim-start-label");
  const endLabel = overlay.querySelector(".apt-trim-end-label");
  const markerListEl = overlay.querySelector(".apt-marker-list");
  const waveCanvas = overlay.querySelector(".apt-wave-canvas");
  const waveCtx = waveCanvas.getContext("2d");
  const playheadLabel = overlay.querySelector(".apt-playhead-label");

  let duration = 0;
  let previewLocked = false;
  let peaks = [];
  let markers = [];
  let mediaEl = null;

  const parseMarkersWidget = () => {
    try {
      const v = JSON.parse(String(markersWidget?.value || "[]"));
      return Array.isArray(v) ? v.map((x) => Number(x)).filter((x) => Number.isFinite(x) && x >= 0) : [];
    } catch {
      return [];
    }
  };
  markers = parseMarkersWidget();

  const normalizeMarkers = () => {
    const seen = new Set();
    markers = markers
      .map((v) => Math.max(0, Math.min(duration, Number(v) || 0)))
      .filter((v) => Number.isFinite(v))
      .filter((v) => {
        const k = Math.round(v * 1000);
        if (seen.has(k)) return false;
        seen.add(k);
        return v > 0.0001 && v < duration - 0.0001;
      })
      .sort((a, b) => a - b);
  };

  const drawWave = () => {
    if (!waveCtx) return;
    const w = waveCanvas.width;
    const h = waveCanvas.height;
    waveCtx.clearRect(0, 0, w, h);
    waveCtx.fillStyle = "#131313";
    waveCtx.fillRect(0, 0, w, h);
    waveCtx.strokeStyle = "#3ea6ff";
    waveCtx.lineWidth = 1;
    if (peaks.length > 0) {
      const centerY = h / 2;
      for (let i = 0; i < peaks.length; i++) {
        const x = (i / (peaks.length - 1 || 1)) * w;
        const amp = Math.max(0, Math.min(1, Number(peaks[i]) || 0));
        const bar = amp * (h * 0.45);
        waveCtx.beginPath();
        waveCtx.moveTo(x, centerY - bar);
        waveCtx.lineTo(x, centerY + bar);
        waveCtx.stroke();
      }
    }

    const startX = (Number(startRange.value || 0) / Math.max(duration, 0.001)) * w;
    const endX = (Number(endRange.value || 0) / Math.max(duration, 0.001)) * w;
    waveCtx.fillStyle = "rgba(255, 196, 0, 0.18)";
    waveCtx.fillRect(startX, 0, Math.max(1, endX - startX), h);

    waveCtx.strokeStyle = "#ff5f5f";
    waveCtx.lineWidth = 2;
    markers.forEach((m) => {
      const x = (m / Math.max(duration, 0.001)) * w;
      waveCtx.beginPath();
      waveCtx.moveTo(x, 0);
      waveCtx.lineTo(x, h);
      waveCtx.stroke();
    });

    if (mediaEl && Number.isFinite(mediaEl.currentTime)) {
      const px = (mediaEl.currentTime / Math.max(duration, 0.001)) * w;
      waveCtx.strokeStyle = "#8ef58e";
      waveCtx.lineWidth = 2;
      waveCtx.beginPath();
      waveCtx.moveTo(px, 0);
      waveCtx.lineTo(px, h);
      waveCtx.stroke();
      if (playheadLabel) {
        playheadLabel.textContent = `播放头: ${formatSec(mediaEl.currentTime)}`;
      }
    }
  };

  const renderMarkerList = () => {
    markerListEl.innerHTML = "";
    if (!markers.length) {
      markerListEl.textContent = "暂无标记";
      return;
    }
    markers.forEach((m, idx) => {
      const chip = document.createElement("button");
      chip.className = "apt-marker-chip";
      chip.textContent = `${idx + 1}: ${formatSec(m)}`;
      chip.title = "点击删除该标记";
      chip.addEventListener("click", () => {
        markers.splice(idx, 1);
        drawWave();
        renderMarkerList();
      });
      markerListEl.appendChild(chip);
    });
  };

  const renderLabels = () => {
    const s = Number(startRange.value || 0);
    const e = Number(endRange.value || 0);
    startLabel.textContent = formatSec(s);
    endLabel.textContent = formatSec(e);
  };

  const clampRanges = () => {
    let s = Number(startRange.value || 0);
    let e = Number(endRange.value || 0);
    if (s >= e) {
      if (previewLocked === "start") {
        e = Math.min(duration, s + 0.01);
      } else {
        s = Math.max(0, e - 0.01);
      }
    }
    startRange.value = String(s);
    endRange.value = String(e);
    renderLabels();
    drawWave();
  };

  (async () => {
    try {
      const data = await resolveMedia(sourcePath);
      duration = Math.max(0.01, Number(data.duration || 0));
      peaks = Array.isArray(data.peaks) ? data.peaks : [];
      normalizeMarkers();
      statusEl.textContent = `时长: ${formatSec(duration)}，拖动滑块选择切割区间`;

      mediaEl = document.createElement(data.media_type === "video" ? "video" : "audio");
      mediaEl.controls = true;
      mediaEl.preload = "metadata";
      if (data.media_type === "video") {
        mediaEl.style.maxHeight = "320px";
      }
      mediaEl.src = data.media_url;
      playerWrap.appendChild(mediaEl);
      mediaEl.addEventListener("timeupdate", () => drawWave());
      mediaEl.addEventListener("seeked", () => drawWave());

      startRange.max = String(duration);
      endRange.max = String(duration);

      const initStart = Math.max(0, Number(startWidget?.value || 0));
      const initEndRaw = Number(endWidget?.value || 0);
      const initEnd = initEndRaw > initStart ? initEndRaw : duration;
      startRange.value = String(Math.min(initStart, duration - 0.01));
      endRange.value = String(Math.min(Math.max(initEnd, Number(startRange.value) + 0.01), duration));
      renderLabels();
      controlsEl.style.display = "";
      drawWave();
      renderMarkerList();

      startRange.addEventListener("input", () => {
        previewLocked = "start";
        clampRanges();
      });
      endRange.addEventListener("input", () => {
        previewLocked = "end";
        clampRanges();
      });

      overlay.querySelector(".apt-set-start")?.addEventListener("click", () => {
        startRange.value = String(Math.max(0, Math.min(mediaEl.currentTime || 0, duration)));
        previewLocked = "start";
        clampRanges();
      });
      overlay.querySelector(".apt-set-end")?.addEventListener("click", () => {
        endRange.value = String(Math.max(0, Math.min(mediaEl.currentTime || 0, duration)));
        previewLocked = "end";
        clampRanges();
      });
      overlay.querySelector(".apt-add-marker")?.addEventListener("click", () => {
        const t = Math.max(0, Math.min(mediaEl.currentTime || 0, duration));
        markers.push(t);
        normalizeMarkers();
        drawWave();
        renderMarkerList();
      });
      overlay.querySelector(".apt-clear-markers")?.addEventListener("click", () => {
        markers = [];
        drawWave();
        renderMarkerList();
      });
      overlay.querySelector(".apt-preview-loop")?.addEventListener("click", async () => {
        const s = Number(startRange.value || 0);
        const e = Number(endRange.value || 0);
        mediaEl.currentTime = s;
        await mediaEl.play();
        const loopFn = () => {
          if (mediaEl.currentTime >= e) mediaEl.currentTime = s;
        };
        mediaEl.addEventListener("timeupdate", loopFn);
        setTimeout(() => mediaEl.removeEventListener("timeupdate", loopFn), 12000);
      });
      overlay.querySelector(".apt-apply")?.addEventListener("click", () => {
        const s = Number(startRange.value || 0);
        const e = Number(endRange.value || 0);
        normalizeMarkers();
        setWidgetValue(startWidget, Number(s.toFixed(3)));
        setWidgetValue(endWidget, Number(e.toFixed(3)));
        setWidgetValue(markersWidget, JSON.stringify(markers.map((v) => Number(v.toFixed(3)))));
        node.setDirtyCanvas(true, true);
        closeModal();
      });

      waveCanvas.addEventListener("click", (ev) => {
        if (!duration) return;
        const rect = waveCanvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const t = Math.max(0, Math.min(duration, (x / rect.width) * duration));
        markers.push(t);
        normalizeMarkers();
        renderMarkerList();
        drawWave();
      });
    } catch (err) {
      statusEl.textContent = `加载失败: ${err?.message || err}`;
    }
  })();
}

function ensureTrimButton(node) {
  if (node.constructor?.nodeData?.name !== TARGET_NODE) return;
  const exists = (node.widgets || []).find((w) => w.name === "Open Trim UI");
  if (exists) return;
  node.addWidget("button", "Open Trim UI", "open", () => openTrimModal(node));
  if (node.computeSize) node.setSize(node.computeSize());
}

app.registerExtension({
  name: "AptPreset.MediaTrimVisual",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== TARGET_NODE) return;
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
      ensureTrimButton(this);
      return r;
    };
  },
  async setup() {
    (app.graph?._nodes || []).forEach((n) => ensureTrimButton(n));
  },
});

const style = document.createElement("style");
style.textContent = `
.apt-trim-overlay{position:fixed;inset:0;background:rgba(0,0,0,.55);display:flex;align-items:center;justify-content:center;z-index:10020}
.apt-trim-modal{width:min(920px,92vw);max-height:90vh;overflow:auto;background:#1e1e1e;color:#ddd;border:1px solid #444;border-radius:10px;padding:12px}
.apt-trim-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.apt-trim-title{font-size:16px;font-weight:600}
.apt-trim-close{background:#333;color:#eee;border:1px solid #555;border-radius:6px;padding:4px 8px;cursor:pointer}
.apt-trim-status{font-size:13px;color:#9ecbff;margin-bottom:8px}
.apt-trim-player-wrap{display:flex;justify-content:center;align-items:center;margin:8px 0}
.apt-trim-player-wrap video,.apt-trim-player-wrap audio{width:min(860px,88vw);background:#000;border-radius:8px}
.apt-wave-wrap{margin:8px 0}
.apt-wave-canvas{width:min(860px,88vw);height:140px;background:#111;border:1px solid #3a3a3a;border-radius:6px}
.apt-playhead-label{font-size:12px;color:#8ef58e;margin-top:4px}
.apt-trim-row{display:flex;align-items:center;gap:8px;margin:8px 0}
.apt-trim-row span{min-width:52px}
.apt-trim-row input[type="range"]{flex:1}
.apt-trim-actions{justify-content:flex-end}
.apt-trim-actions button{background:#2b2b2b;color:#eee;border:1px solid #555;border-radius:6px;padding:6px 10px;cursor:pointer}
.apt-marker-list{display:flex;flex-wrap:wrap;gap:6px}
.apt-marker-chip{background:#292929;color:#ddd;border:1px solid #555;border-radius:999px;padding:2px 8px;cursor:pointer}
`;
document.head.appendChild(style);
