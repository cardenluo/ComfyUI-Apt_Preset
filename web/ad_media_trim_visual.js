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

function formatMediaRate(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "—";
  return Number.isInteger(number) ? String(number) : number.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function formatMediaInfo(data) {
  const lines = [];
  const video = data?.video;
  if (video) {
    const values = [
      `宽 ${Number(video.width) || "—"}`,
      `高 ${Number(video.height) || "—"}`,
      `FPS ${formatMediaRate(video.fps)}`,
      `Length ${Number(video.length) || "—"}`,
    ];
    if (video.codec) values.push(`编码 ${String(video.codec).toUpperCase()}`);
    lines.push(`视频：${values.join(" · ")}`);
  }
  const audio = data?.audio;
  if (audio) {
    const values = [
      `采样率 ${Number(audio.sample_rate) ? `${Number(audio.sample_rate)} Hz` : "—"}`,
      `声道 ${Number(audio.channels) || "—"}${audio.channel_layout ? ` (${audio.channel_layout})` : ""}`,
    ];
    if (audio.codec) values.push(`编码 ${String(audio.codec).toUpperCase()}`);
    if (Number(audio.bit_rate) > 0) values.push(`码率 ${Math.round(Number(audio.bit_rate) / 1000)} kbps`);
    lines.push(`音频：${values.join(" · ")}`);
  }
  return lines.join("\n");
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

function resolvePathFromMediaPorts(node) {
  const visited = new Set();
  const queue = [];
  ["video", "audio"].forEach((inputName) => {
    const upstream = getUpstreamNodeByInput(node, inputName);
    if (upstream) queue.push(upstream);
  });

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
  const markersWidget = getWidget(node, "markers_json");
  const modeWidget = getWidget(node, "split_mode");
  const sourcePath = resolvePathFromMediaPorts(node);

  if (!sourcePath) {
    alert("未找到可预览媒体：请连接可回溯到本地文件路径的 video 或 audio 端口。");
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
        <div class="apt-trim-media-info" hidden></div>
        <div class="apt-trim-player-wrap"></div>
        <div class="apt-wave-wrap">
          <canvas class="apt-wave-canvas" width="860" height="140"></canvas>
        </div>
        <div class="apt-playhead-label">Playhead: 00:00.00</div>
        <div class="apt-trim-controls" style="display:none;">
          <div class="apt-trim-row apt-trim-actions">
            <button class="apt-add-marker">添加标记</button>
            <button class="apt-clear-markers">清空标记</button>
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
    cancelAnimationFrame(animationFrame);
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
  const mediaInfoEl = overlay.querySelector(".apt-trim-media-info");
  const playerWrap = overlay.querySelector(".apt-trim-player-wrap");
  const controlsEl = overlay.querySelector(".apt-trim-controls");
  const markerListEl = overlay.querySelector(".apt-marker-list");
  const waveCanvas = overlay.querySelector(".apt-wave-canvas");
  const waveCtx = waveCanvas.getContext("2d");
  const playheadLabel = overlay.querySelector(".apt-playhead-label");

  let duration = 0;
  let peaks = [];
  let markers = [];
  let mediaEl = null;
  let draggedMarker = -1;
  let suppressClick = false;
  let animationFrame = 0;

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
      .map((v) => Number(Math.max(0, Math.min(duration, Number(v) || 0)).toFixed(2)))
      .filter((v) => Number.isFinite(v))
      .filter((v) => {
        const k = Math.round(v * 100);
        if (seen.has(k)) return false;
        seen.add(k);
        return v >= 0.01 && v <= duration - 0.01;
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
      chip.textContent = `${idx + 1}: ${formatSec(m)}  ×`;
      chip.title = "删除该标记";
      chip.addEventListener("click", () => {
        markers.splice(idx, 1);
        drawWave();
        renderMarkerList();
      });
      markerListEl.appendChild(chip);
    });
  };

  const timeFromPointer = (ev) => {
    const rect = waveCanvas.getBoundingClientRect();
    const ratio = Math.max(0, Math.min(1, (ev.clientX - rect.left) / Math.max(1, rect.width)));
    return Number((ratio * duration).toFixed(2));
  };

  const markerFromPointer = (ev) => {
    const rect = waveCanvas.getBoundingClientRect();
    const tolerance = duration * 10 / Math.max(1, rect.width);
    const time = timeFromPointer(ev);
    let best = -1;
    let distance = Infinity;
    markers.forEach((marker, index) => {
      const current = Math.abs(marker - time);
      if (current <= tolerance && current < distance) {
        best = index;
        distance = current;
      }
    });
    return best;
  };

  const animatePlayhead = () => {
    drawWave();
    if (mediaEl && !mediaEl.paused && !mediaEl.ended) {
      animationFrame = requestAnimationFrame(animatePlayhead);
    }
  };

  (async () => {
    try {
      const data = await resolveMedia(sourcePath);
      duration = Math.max(0.01, Number(data.duration || 0));
      peaks = Array.isArray(data.peaks) ? data.peaks : [];
      normalizeMarkers();
      const updateMediaInfo = () => {
        const text = formatMediaInfo(data);
        mediaInfoEl.textContent = text;
        mediaInfoEl.hidden = !text;
      };
      updateMediaInfo();
      statusEl.textContent = `时长: ${formatSec(duration)}；拖动红色标记线，双击或右键删除`;

      mediaEl = document.createElement(data.media_type === "video" ? "video" : "audio");
      mediaEl.controls = true;
      mediaEl.preload = "metadata";
      if (data.media_type === "video") {
        mediaEl.style.maxHeight = "320px";
      }
      mediaEl.src = data.media_url;
      playerWrap.appendChild(mediaEl);
      mediaEl.addEventListener("loadedmetadata", () => {
        if (data.media_type === "video") {
          data.video ||= {};
          if (!Number(data.video.width)) data.video.width = Number(mediaEl.videoWidth) || 0;
          if (!Number(data.video.height)) data.video.height = Number(mediaEl.videoHeight) || 0;
          if (!Number(data.video.length) && Number(data.video.fps) > 0) {
            data.video.length = Math.round(duration * Number(data.video.fps));
          }
        }
        updateMediaInfo();
      });
      mediaEl.addEventListener("timeupdate", () => drawWave());
      mediaEl.addEventListener("seeked", () => drawWave());
      mediaEl.addEventListener("play", () => {
        cancelAnimationFrame(animationFrame);
        animatePlayhead();
      });
      mediaEl.addEventListener("pause", () => drawWave());
      controlsEl.style.display = "";
      drawWave();
      renderMarkerList();
      overlay.querySelector(".apt-add-marker")?.addEventListener("click", () => {
        const t = Number(Math.max(0, Math.min(mediaEl.currentTime || 0, duration)).toFixed(2));
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
      overlay.querySelector(".apt-apply")?.addEventListener("click", () => {
        normalizeMarkers();
        setWidgetValue(markersWidget, JSON.stringify(markers.map((v) => Number(v.toFixed(2)))));
        setWidgetValue(modeWidget, "Open Trim UI");
        node.setDirtyCanvas(true, true);
        closeModal();
      });

      waveCanvas.addEventListener("click", (ev) => {
        if (!duration) return;
        if (suppressClick) {
          suppressClick = false;
          return;
        }
        mediaEl.currentTime = timeFromPointer(ev);
        drawWave();
      });
      waveCanvas.addEventListener("pointerdown", (ev) => {
        draggedMarker = markerFromPointer(ev);
        if (draggedMarker < 0) return;
        suppressClick = true;
        waveCanvas.setPointerCapture?.(ev.pointerId);
        waveCanvas.style.cursor = "ew-resize";
        ev.preventDefault();
      });
      waveCanvas.addEventListener("pointermove", (ev) => {
        if (draggedMarker < 0) {
          waveCanvas.style.cursor = markerFromPointer(ev) >= 0 ? "ew-resize" : "pointer";
          return;
        }
        markers[draggedMarker] = timeFromPointer(ev);
        drawWave();
        renderMarkerList();
      });
      const finishDrag = () => {
        if (draggedMarker < 0) return;
        draggedMarker = -1;
        normalizeMarkers();
        renderMarkerList();
        drawWave();
      };
      waveCanvas.addEventListener("pointerup", finishDrag);
      waveCanvas.addEventListener("pointercancel", finishDrag);
      const removeMarkerAtPointer = (ev) => {
        const index = markerFromPointer(ev);
        if (index < 0) return;
        ev.preventDefault();
        markers.splice(index, 1);
        renderMarkerList();
        drawWave();
      };
      waveCanvas.addEventListener("dblclick", removeMarkerAtPointer);
      waveCanvas.addEventListener("contextmenu", removeMarkerAtPointer);
    } catch (err) {
      statusEl.textContent = `加载失败: ${err?.message || err}`;
    }
  })();
}

function setNodeWidgetVisible(widget, visible) {
  if (!widget) return;
  if (!widget.__aptTrimOriginal) {
    widget.__aptTrimOriginal = {
      type: widget.type,
      computeSize: widget.computeSize,
      hidden: widget.hidden,
    };
  }
  if (visible) {
    widget.type = widget.__aptTrimOriginal.type;
    widget.hidden = widget.__aptTrimOriginal.hidden ?? false;
    widget.computeSize = widget.__aptTrimOriginal.computeSize;
  } else {
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
  }
  widget.options ||= {};
  widget.options.hidden = !visible;
  if (widget.inputEl) widget.inputEl.style.display = visible ? "" : "none";
  if (widget.element) widget.element.style.display = visible ? "" : "none";
}

function updateModeWidgets(node) {
  const mode = String(getWidget(node, "split_mode")?.value || "Open Trim UI");
  setNodeWidgetVisible(getWidget(node, "time"), mode === "按时间分割");
  setNodeWidgetVisible(getWidget(node, "number"), mode === "按数量分割");
  setNodeWidgetVisible(getWidget(node, "markers_json"), false);
  setNodeWidgetVisible(getWidget(node, "Open Trim UI"), mode === "Open Trim UI");
  const size = node.computeSize?.();
  if (size) node.setSize(size);
  node.setDirtyCanvas?.(true, true);
}

function ensureTrimButton(node) {
  if (node.constructor?.nodeData?.name !== TARGET_NODE) return;
  let button = getWidget(node, "Open Trim UI");
  if (!button) {
    button = node.addWidget("button", "Open Trim UI", "open", () => openTrimModal(node));
  }
  const modeWidget = getWidget(node, "split_mode");
  if (modeWidget && !modeWidget.__aptTrimBound) {
    modeWidget.__aptTrimBound = true;
    const callback = modeWidget.callback;
    modeWidget.callback = function () {
      const result = callback?.apply(this, arguments);
      updateModeWidgets(node);
      return result;
    };
  }
  updateModeWidgets(node);
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
    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
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
.apt-trim-media-info{display:block!important;flex:0 0 59px!important;align-self:stretch;height:59px!important;min-height:0!important;max-height:59px!important;box-sizing:border-box;margin:-2px 0 8px;padding:7px 9px;overflow:hidden;border:1px solid #3a3a3a;border-radius:6px;background:#181818;color:#cfd6df;font:12px/15px ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap}
.apt-trim-player-wrap{display:flex;justify-content:center;align-items:center;margin:8px 0}
.apt-trim-player-wrap video,.apt-trim-player-wrap audio{width:min(860px,88vw);box-sizing:border-box;background:#000;border-radius:8px}
.apt-wave-wrap{display:flex;justify-content:center;align-items:center;margin:8px 0}
.apt-wave-canvas{width:calc(min(860px,88vw) - 32px);height:140px;box-sizing:border-box;background:#111;border:1px solid #3a3a3a;border-radius:6px}
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
