import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function getImageListWidget(node) {
    return node?.widgets?.find((w) => w.name === "image_list");
}

function getThumbSizeWidget(node) {
    return node?.widgets?.find((w) => w.name === "thumb_size");
}

function getThumbSize(node) {
    const w = getThumbSizeWidget(node);
    const v = Number(w?.value);
    if (v === 64 || v === 128 || v === 384) return v;
    return 64;
}

function setThumbSize(node, size) {
    const w = getThumbSizeWidget(node);
    if (!w) return;
    w.value = size;
    w.callback?.(w.value);
}

function clampInt(v, min, max) {
    v = Math.floor(Number(v));
    if (Number.isNaN(v)) v = min;
    if (v < min) v = min;
    if (v > max) v = max;
    return v;
}

function buildVNCCSPrompt(data) {
    const azimuth = clampInt(data?.azimuth ?? 0, 0, 360) % 360;
    const elevation = clampInt(data?.elevation ?? 0, -30, 60);
    const distance = data?.distance ?? "medium shot";
    const include_trigger = data?.include_trigger !== false;

    const azimuthMap = {
        0: "front view",
        45: "front-right quarter view",
        90: "right side view",
        135: "back-right quarter view",
        180: "back view",
        225: "back-left quarter view",
        270: "left side view",
        315: "front-left quarter view",
    };

    const closestAzimuth = azimuth > 337.5 ? 0 : Object.keys(azimuthMap).map((k) => Number(k)).reduce((best, k) => {
        return Math.abs(k - azimuth) < Math.abs(best - azimuth) ? k : best;
    }, 0);

    const elevationMap = {
        "-30": "low-angle shot",
        "0": "eye-level shot",
        "30": "elevated shot",
        "60": "high-angle shot",
    };

    const closestElevation = Object.keys(elevationMap).map((k) => Number(k)).reduce((best, k) => {
        return Math.abs(k - elevation) < Math.abs(best - elevation) ? k : best;
    }, 0);

    const parts = [];
    if (include_trigger) parts.push("<sks>");
    parts.push(azimuthMap[closestAzimuth]);
    parts.push(elevationMap[String(closestElevation)]);
    parts.push(distance);
    return parts.join(" ");
}

function createVNCCSVisualUI(node) {
    const w = getCameraDataWidget(node);
    if (!w) return null;

    w.type = "hidden";
    w.computeSize = () => [0, -4];

    const container = document.createElement("div");
    container.style.cssText =
        "width:100%;padding:8px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:5px 0;pointer-events:auto;";

    const row = document.createElement("div");
    row.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:8px;";

    const mkField = (labelText) => {
        const wrap = document.createElement("div");
        wrap.style.cssText = "display:flex;flex-direction:column;gap:4px;";
        const label = document.createElement("div");
        label.textContent = labelText;
        label.style.cssText = "font-size:12px;opacity:0.9;";
        wrap.appendChild(label);
        return { wrap };
    };

    const azF = mkField("水平角度(azimuth)");
    const elF = mkField("垂直角度(elevation)");
    const distF = mkField("远近(distance)");
    const trigF = mkField("触发词");

    const az = document.createElement("input");
    az.type = "range";
    az.min = "0";
    az.max = "360";
    az.step = "45";

    const el = document.createElement("input");
    el.type = "range";
    el.min = "-30";
    el.max = "60";
    el.step = "30";

    const dist = document.createElement("select");
    for (const v of ["close-up", "medium shot", "wide shot"]) {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = v;
        dist.appendChild(opt);
    }

    const trig = document.createElement("input");
    trig.type = "checkbox";

    const azVal = document.createElement("div");
    azVal.style.cssText = "font-size:12px;opacity:0.8;";
    const elVal = document.createElement("div");
    elVal.style.cssText = "font-size:12px;opacity:0.8;";

    const promptOut = document.createElement("input");
    promptOut.type = "text";
    promptOut.readOnly = true;
    promptOut.style.cssText =
        "width:100%;padding:8px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;";

    azF.wrap.appendChild(az);
    azF.wrap.appendChild(azVal);
    elF.wrap.appendChild(el);
    elF.wrap.appendChild(elVal);
    distF.wrap.appendChild(dist);
    trigF.wrap.appendChild(trig);

    row.appendChild(azF.wrap);
    row.appendChild(elF.wrap);
    row.appendChild(distF.wrap);
    row.appendChild(trigF.wrap);

    const write = () => {
        const data = {
            azimuth: clampInt(az.value, 0, 360),
            elevation: clampInt(el.value, -30, 60),
            distance: dist.value,
            include_trigger: !!trig.checked,
        };
        w.value = JSON.stringify(data);
        w.callback?.(w.value);
        azVal.textContent = String(data.azimuth);
        elVal.textContent = String(data.elevation);
        promptOut.value = buildVNCCSPrompt(data);
    };

    const read = () => {
        let data;
        try {
            data = JSON.parse(w.value || "{}");
        } catch {
            data = {};
        }
        az.value = String(clampInt(data?.azimuth ?? 0, 0, 360));
        el.value = String(clampInt(data?.elevation ?? 0, -30, 60));
        dist.value = data?.distance ?? "medium shot";
        trig.checked = data?.include_trigger !== false;
        write();
    };

    az.addEventListener("input", write);
    el.addEventListener("input", write);
    dist.addEventListener("change", write);
    trig.addEventListener("change", write);

    container.appendChild(row);
    container.appendChild(promptOut);

    return { container, read };
}

function parseImageList(text) {
    return (text || "")
        .split("\n")
        .map((s) => s.trim())
        .filter((s) => !!s);
}

function setImageList(node, names) {
    const w = getImageListWidget(node);
    if (!w) return;
    w.value = (names || []).join("\n");
    w.callback?.(w.value);
}

function getMaxImagesValue(node) {
    const w = node?.widgets?.find((x) => x.name === "max_images");
    const v = w?.value;
    return typeof v === "number" ? v : 0;
}

function normalizeSelectedIndicesValue(value, namesLen) {
    if (!Array.isArray(value)) return [];
    const ints = value.filter((v) => Number.isInteger(v));
    const oneBased = !ints.includes(0) && ints.some((v) => v >= 1);
    const idx0 = oneBased ? ints.map((v) => v - 1) : ints;
    return idx0.filter((v) => v >= 0 && v < namesLen);
}

function serializeSelectedIndicesValue(indices0) {
    const out = Array.isArray(indices0) ? indices0.map((v) => v + 1) : [];
    return JSON.stringify(out);
}

// 获取要输出的图片索引（核心独立逻辑）
function getOutputImageIndex(node) {
    const names = parseImageList(getImageListWidget(node)?.value);
    const selectedIndicesWidget = getWidgetByName(node, "selected_indices");
    let selectedIndices = [];
    
    // 解析手动选中的索引
    if (selectedIndicesWidget && typeof selectedIndicesWidget.value === "string" && selectedIndicesWidget.value.trim()) {
        try {
            const parsed = JSON.parse(selectedIndicesWidget.value);
            if (Array.isArray(parsed)) {
                selectedIndices = normalizeSelectedIndicesValue(parsed, names.length);
            }
        } catch {}
    }
    
    // 独立逻辑：无手动选择时返回最后一张索引，有选择时返回选中的（保持原有逻辑）
    if (selectedIndices.length === 0 && names.length > 0) {
        return names.length - 1; // 输出最后一张
    }
    return selectedIndices.length > 0 ? selectedIndices[0] : -1;
}

function deepClone(obj) {
    if (typeof structuredClone === "function") return structuredClone(obj);
    return JSON.parse(JSON.stringify(obj));
}

function getWidgetByName(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function getCameraDataWidget(node) {
    return getWidgetByName(node, "camera_data");
}

async function queueCurrent(node) {
    const prompt = await app.graphToPrompt();
    // 替换为新的索引获取逻辑
    const outputIndex = getOutputImageIndex(node);
    if (outputIndex >= 0) {
        const nodeId = String(node.id);
        prompt.output[nodeId].inputs = prompt.output[nodeId].inputs || {};
        prompt.output[nodeId].inputs.mode = "single";
        prompt.output[nodeId].inputs.index = outputIndex;
    }
    await api.queuePrompt(-1, prompt);
}

async function queueAllSequential(node) {
    const names0 = parseImageList(getImageListWidget(node)?.value);
    if (!names0 || names0.length === 0) return;

    const maxImages = getMaxImagesValue(node);
    const names = maxImages && maxImages > 0 ? names0.slice(0, maxImages) : names0;
    if (names.length === 0) return;

    const wMode = getWidgetByName(node, "mode");
    const wIndex = getWidgetByName(node, "index");
    if (!wMode || !wIndex) {
        const basePrompt = await app.graphToPrompt();
        const nodeId = String(node.id);
        for (let i = 0; i < names.length; i++) {
            const prompt = deepClone(basePrompt);
            const apiNode = prompt.output?.[nodeId];
            if (!apiNode) continue;
            apiNode.inputs = apiNode.inputs || {};
            apiNode.inputs.mode = "single";
            apiNode.inputs.index = i;
            await api.queuePrompt(-1, prompt);
        }
        return;
    }

    const prevMode = wMode.value;
    const prevIndex = wIndex.value;
    try {
        wMode.value = "single";
        wMode.callback?.(wMode.value);
        for (let i = 0; i < names.length; i++) {
            wIndex.value = i;
            wIndex.callback?.(wIndex.value);
            await queueCurrent(node);
        }
    } finally {
        wMode.value = prevMode;
        wMode.callback?.(wMode.value);
        wIndex.value = prevIndex;
        wIndex.callback?.(wIndex.value);
    }
}

function getViewUrl(filename, size = 64) {
    return api.apiURL(`/Apt_Preset_IO_LoadImgList_thumb?filename=${encodeURIComponent(filename)}&size=${encodeURIComponent(size)}`);
}

function getOriginalViewPath(filename) {
    return `/view?filename=${encodeURIComponent(filename)}`;
}

function getOriginalViewPathForDrop(filename) {
    return `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`;
}

function getOriginalViewUrl(filename) {
    return api.apiURL(getOriginalViewPath(filename));
}

function getOriginalViewUrlForDrop(filename) {
    return api.apiURL(getOriginalViewPathForDrop(filename));
}

function guessMimeType(filename) {
    const name = String(filename || "").toLowerCase();
    if (name.endsWith(".png")) return "image/png";
    if (name.endsWith(".jpg") || name.endsWith(".jpeg")) return "image/jpeg";
    if (name.endsWith(".webp")) return "image/webp";
    if (name.endsWith(".gif")) return "image/gif";
    if (name.endsWith(".bmp")) return "image/bmp";
    if (name.endsWith(".tif") || name.endsWith(".tiff")) return "image/tiff";
    return "application/octet-stream";
}

function getDownloadFilename(annotatedName) {
    let base = String(annotatedName || "");
    base = base.split(/[\\/]/).pop() || base;
    base = base.replace(/\s*\[[^\]]+\]\s*$/, "");
    return base || "image";
}

const _APT_LOADIMAGE_DRAG_MIME = "application/x-apt-preset-loadimage";
let _canvasLoadImageDropInstalled = false;

function _graphPosFromEvent(e) {
    const c = app?.canvas;
    const canvas = c?.canvas;
    const ds = c?.ds;
    const rect = canvas?.getBoundingClientRect?.();
    if (!rect) return null;
    const scale = typeof ds?.scale === "number" && ds.scale !== 0 ? ds.scale : 1;
    const off = Array.isArray(ds?.offset) ? ds.offset : [0, 0];
    const x = (e.clientX - rect.left) / scale - (off[0] || 0);
    const y = (e.clientY - rect.top) / scale - (off[1] || 0);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
    return [x, y];
}

function _setLoadImageNodeFilename(node, filename) {
    const w = node?.widgets?.find((w) => w?.name === "image");
    if (!w) return false;
    w.value = filename;
    w.callback?.(filename);
    return true;
}

async function _uploadOriginalToInputByAnnotatedName(annotatedName) {
    const name = String(annotatedName ?? "").trim();
    if (!name) return null;
    let resp = null;
    for (const p of [getOriginalViewPath(name), getOriginalViewPathForDrop(name), `/view?filename=${encodeURIComponent(name)}&type=output&subfolder=`, `/view?filename=${encodeURIComponent(name)}&type=temp&subfolder=`]) {
        const r = await api.fetchApi(p, { cache: "no-store" });
        if (r.ok) {
            resp = r;
            break;
        }
    }
    if (!resp) throw new Error("view image failed");
    const blob = await resp.blob();
    const downloadName = getDownloadFilename(name);
    const mime = blob?.type && blob.type !== "" ? blob.type : guessMimeType(downloadName);
    const file = new File([blob], downloadName, { type: mime });
    return await uploadOneImage(file);
}

function ensureCanvasLoadImageDrop() {
    if (_canvasLoadImageDropInstalled) return;
    const canvasEl = app?.canvas?.canvas;
    if (!canvasEl) {
        setTimeout(ensureCanvasLoadImageDrop, 300);
        return;
    }
    _canvasLoadImageDropInstalled = true;

    canvasEl.addEventListener(
        "dragover",
        (e) => {
            const dt = e?.dataTransfer;
            if (!dt) return;
            if (!Array.from(dt.types || []).includes(_APT_LOADIMAGE_DRAG_MIME)) return;
            e.preventDefault();
            dt.dropEffect = "copy";
        },
        { capture: true }
    );

    canvasEl.addEventListener(
        "drop",
        async (e) => {
            const dt = e?.dataTransfer;
            if (!dt) return;
            if (!Array.from(dt.types || []).includes(_APT_LOADIMAGE_DRAG_MIME)) return;
            e.preventDefault();
            e.stopPropagation();

            const annotatedName = dt.getData(_APT_LOADIMAGE_DRAG_MIME);
            const pos = _graphPosFromEvent(e);
            const uploaded = await _uploadOriginalToInputByAnnotatedName(annotatedName);
            if (!uploaded) return;

            const n = LiteGraph.createNode("LoadImage");
            if (!n) return;
            app.graph.add(n);
            if (pos) n.pos = pos;
            _setLoadImageNodeFilename(n, uploaded);
            app.canvas.selectNode(n, false);
            app.graph.setDirtyCanvas(true);
        },
        { capture: true }
    );
}

function isFilesDragEvent(e) {
    const dt = e?.dataTransfer;
    if (!dt) return false;
    if (dt.files && dt.files.length > 0) return true;
    return Array.from(dt.types || []).includes("Files");
}

const _batchLoadImagesDomUIs = new Set();

function _isPointInRect(x, y, rect) {
    return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

function _getUIUnderPointer(e) {
    const x = e?.clientX;
    const y = e?.clientY;
    if (typeof x !== "number" || typeof y !== "number") return null;

    for (const entry of _batchLoadImagesDomUIs) {
        const rect = entry?.container?.getBoundingClientRect?.();
        if (!rect) continue;
        if (_isPointInRect(x, y, rect)) return entry;
    }
    return null;
}

function _setDraggingUI(activeEntry) {
    for (const entry of _batchLoadImagesDomUIs) {
        entry?.setDragging?.(entry === activeEntry);
    }
}

let _globalDragDropInstalled = false;
function ensureGlobalDragDropPrevention() {
    if (_globalDragDropInstalled) return;
    _globalDragDropInstalled = true;

    window.addEventListener(
        "dragover",
        (e) => {
            if (!isFilesDragEvent(e)) return;
            e.preventDefault();
            _setDraggingUI(_getUIUnderPointer(e));
        },
        { capture: true }
    );

    window.addEventListener(
        "drop",
        async (e) => {
            if (!isFilesDragEvent(e)) return;
            e.preventDefault();

            const hit = _getUIUnderPointer(e);
            _setDraggingUI(null);
            if (!hit) return;

            const files = Array.from(e.dataTransfer?.files || []);
            if (files.length === 0) return;
            await uploadFilesSequential(hit.node, files, { replace: false });
            hit.redraw?.();
        },
        { capture: true }
    );

    window.addEventListener(
        "dragleave",
        (e) => {
            if (!isFilesDragEvent(e)) return;
            _setDraggingUI(null);
        },
        { capture: true }
    );
}

async function uploadOneImage(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");

    const resp = await api.fetchApi("/upload/image", {
        method: "POST",
        body,
    });

    if (!resp.ok) {
        throw new Error(await resp.text());
    }

    const json = await resp.json();
    return json?.name;
}

async function uploadFilesSequential(node, files, { replace = false } = {}) {
    const w = getImageListWidget(node);
    if (!w) return [];

    const existing = replace ? [] : parseImageList(w.value);
    const uploaded = [];

    for (const file of files) {
        if (!file) continue;
        if (file?.type && !file.type.startsWith("image/")) continue;
        const name = await uploadOneImage(file);
        if (name) uploaded.push(name);
    }

    const merged = existing.concat(uploaded);
    setImageList(node, merged);
    return uploaded;
}

function openMultiSelect(node, { replace = false } = {}) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/jpeg";
    input.multiple = true;
    input.style.display = "none";
    document.body.appendChild(input);

    input.onchange = async (e) => {
        try {
            const files = Array.from(e.target.files || []);
            await uploadFilesSequential(node, files, { replace });
        } finally {
            document.body.removeChild(input);
        }
    };

    input.click();
}

function openFolderSelect(node, { replace = false } = {}) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/png,image/jpeg";
    input.multiple = true;
    input.webkitdirectory = true;
    input.directory = true;
    input.style.display = "none";
    document.body.appendChild(input);

    input.onchange = async (e) => {
        try {
            let files = Array.from(e.target.files || []);
            const allowExt = new Set([".png", ".jpg", ".jpeg"]);
            files = files.filter((f) => {
                const name = (f?.name || "").toLowerCase();
                for (const ext of allowExt) {
                    if (name.endsWith(ext)) return true;
                }
                return false;
            });
            files.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
            await uploadFilesSequential(node, files, { replace });
        } finally {
            document.body.removeChild(input);
        }
    };

    input.click();
}

function createBrowserUI(node) {
    const container = document.createElement("div");
    container.style.cssText =
        "width:100%;height:100%;min-width:0;box-sizing:border-box;overflow:hidden;padding:6px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:5px 0;pointer-events:none;display:flex;flex-direction:row;gap:6px;";

    const sidebar = document.createElement("div");
    sidebar.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:56px;width:56px;pointer-events:auto;";

    const mkBtn = (label) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
            "padding:3px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;cursor:pointer;font-size:10px;width:100%;text-align:center;";
        return b;
    };

    const replaceBtn = mkBtn("🖼️");
    const folderBtn = mkBtn("📂");
    const selectAllBtn = mkBtn("全选");
    const sortBtn = mkBtn("顺序");
    const deleteSelectedBtn = mkBtn("删除");
    const clearBtn = mkBtn("清空");
    const hideBtn = mkBtn("隐藏");
    const sizeBtn = mkBtn("尺寸");
    const sizeSlider = document.createElement("input");
    sizeSlider.type = "range";
    sizeSlider.min = "0";
    sizeSlider.max = "2";
    sizeSlider.step = "1";
    sizeSlider.style.cssText = "width:100%;margin:2px 0 0 0;cursor:pointer;";
    const sizeVal = document.createElement("div");
    sizeVal.style.cssText = "font-size:10px;opacity:0.85;text-align:center;line-height:1.2;padding-bottom:2px;";

    sidebar.appendChild(replaceBtn);
    sidebar.appendChild(folderBtn);
    sidebar.appendChild(selectAllBtn);
    sidebar.appendChild(sortBtn);
    sidebar.appendChild(deleteSelectedBtn);
    sidebar.appendChild(clearBtn);
    sidebar.appendChild(hideBtn);
    sidebar.appendChild(sizeBtn);
    sidebar.appendChild(sizeSlider);
    sidebar.appendChild(sizeVal);

    const selectedIndices = [];
    let didHydrateSelectionFromWidget = false;
    let previewsHidden = false;
    let nextSortIsAsc = true;
    const SIZE_PRESETS = [64, 128, 384];

    const syncThumbUIFromWidget = () => {
        const current = getThumbSize(node);
        const idx = SIZE_PRESETS.indexOf(current);
        sizeSlider.value = String(idx >= 0 ? idx : 0);
        sizeVal.textContent = String(current);
    };

    const applyThumbSizeFromSlider = () => {
        const idx = Math.max(0, Math.min(2, Math.floor(Number(sizeSlider.value))));
        const size = SIZE_PRESETS[idx] ?? 64;
        setThumbSize(node, size);
        sizeVal.textContent = String(size);
        redraw();
    };

    sizeBtn.onclick = () => applyThumbSizeFromSlider();
    sizeSlider.addEventListener("input", applyThumbSizeFromSlider);
    syncThumbUIFromWidget();
    {
        const wThumb = getThumbSizeWidget(node);
        const v = Number(wThumb?.value);
        if (wThumb && v !== 64 && v !== 128 && v !== 384) {
            setThumbSize(node, 64);
            syncThumbUIFromWidget();
        }
    }

    selectAllBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        if (selectedIndices.length === names.length) {
            selectedIndices.length = 0;
        } else {
            selectedIndices.length = 0;
            for (let i = 0; i < names.length; i++) {
                selectedIndices.push(i);
            }
        }
        redraw();
    };

    sortBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        const sortedNames = [...names].sort((a, b) => (nextSortIsAsc ? a.localeCompare(b) : b.localeCompare(a)));
        setImageList(node, sortedNames);
        selectedIndices.length = 0;
        nextSortIsAsc = !nextSortIsAsc;
        sortBtn.textContent = nextSortIsAsc ? "顺序" : "逆序";
        redraw();
    };

    deleteSelectedBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        const sortedIndices = [...selectedIndices].sort((a, b) => b - a);
        let newNames = [...names];
        for (const idx of sortedIndices) {
            newNames = newNames.slice(0, idx).concat(newNames.slice(idx + 1));
        }
        setImageList(node, newNames);
        selectedIndices.length = 0;
        redraw();
    };

    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;pointer-events:auto;min-width:0;min-height:0;";

    const jumpSelect = document.createElement("select");
    jumpSelect.style.cssText =
        "width:100%;margin:0 0 6px 0;padding:4px 6px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;font-size:11px;outline:none;";
    jumpSelect.addEventListener("change", () => {
        const v = Number(jumpSelect.value);
        if (!Number.isFinite(v)) return;
        const idx = Math.max(0, Math.floor(v));
        selectedIndices.length = 0;
        selectedIndices.push(idx);
        redraw();
    });

    let _jumpCacheLen = -1;
    let _jumpCacheFirst = "";
    let _jumpCacheLast = "";
    const _refreshJumpSelect = (names) => {
        const first = names.length ? String(names[0] ?? "") : "";
        const last = names.length ? String(names[names.length - 1] ?? "") : "";
        if (_jumpCacheLen === names.length && _jumpCacheFirst === first && _jumpCacheLast === last) return;
        _jumpCacheLen = names.length;
        _jumpCacheFirst = first;
        _jumpCacheLast = last;
        jumpSelect.innerHTML = "";
        const frag = document.createDocumentFragment();
        for (let i = 0; i < names.length; i++) {
            const opt = document.createElement("option");
            opt.value = String(i);
            opt.textContent = `#${i} ${names[i]}`;
            frag.appendChild(opt);
        }
        jumpSelect.appendChild(frag);
    };

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--thumb-size,64px),1fr));gap:4px;flex:1;min-width:0;min-height:0;overflow-y:auto;background:var(--comfy-input-bg);padding:4px;border-radius:4px;";

    const hiddenOverlay = document.createElement("div");
    hiddenOverlay.style.cssText =
        "flex:1;display:none;align-items:center;justify-content:center;background:var(--comfy-input-bg);border-radius:4px;color:var(--input-text);font-size:12px;opacity:0.75;";

    const redraw = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        grid.innerHTML = "";
        const thumbSize = getThumbSize(node);
        grid.style.setProperty("--thumb-size", `${thumbSize}px`);
        _refreshJumpSelect(names);

        if (!didHydrateSelectionFromWidget) {
            didHydrateSelectionFromWidget = true;
            const wSel = getWidgetByName(node, "selected_indices");
            if (wSel && typeof wSel.value === "string" && wSel.value.trim()) {
                try {
                    const parsed = JSON.parse(wSel.value);
                    if (Array.isArray(parsed)) {
                        selectedIndices.length = 0;
                        const norm = normalizeSelectedIndicesValue(parsed, names.length);
                        for (const v of norm) {
                            selectedIndices.push(v);
                        }
                    }
                } catch {}
            }
        }

        // 移除默认选中最后一张的逻辑，仅保留无效索引清理
        if (selectedIndices.length > 0) {
            for (let i = selectedIndices.length - 1; i >= 0; i--) {
                const idx = selectedIndices[i];
                if (typeof idx !== "number" || idx < 0 || idx >= names.length) {
                    selectedIndices.splice(i, 1);
                }
            }
        }
        if (names.length > 0) {
            const primary = selectedIndices.length ? selectedIndices[0] : 0;
            const v = Number.isFinite(Number(primary)) ? Math.max(0, Math.min(names.length - 1, Math.floor(Number(primary)))) : 0;
            if (jumpSelect.value !== String(v)) {
                jumpSelect.value = String(v);
            }
        }

        if (previewsHidden) {
            grid.style.display = "none";
            hiddenOverlay.style.display = "flex";
            hiddenOverlay.textContent = `预览已隐藏（${names.length}）`;

            const selectedIndicesWidget = getWidgetByName(node, "selected_indices");
            if (selectedIndicesWidget) {
                selectedIndicesWidget.value = serializeSelectedIndicesValue(selectedIndices);
                selectedIndicesWidget.callback?.(selectedIndicesWidget.value);
            }
            app.graph.setDirtyCanvas(true);
            return;
        }

        grid.style.display = "grid";
        hiddenOverlay.style.display = "none";

        const frag = document.createDocumentFragment();
        names.forEach((name, idx) => {
            const indexInSelection = selectedIndices.indexOf(idx);
            const isSelected = indexInSelection > -1;
            
            const cell = document.createElement("div");
            cell.style.cssText = `display:flex;flex-direction:column;cursor:pointer;${isSelected ? 'opacity:0.7;' : ''}`;
            
            cell.onclick = (e) => {
                e.preventDefault();
                const index = selectedIndices.indexOf(idx);
                if (index > -1) {
                    selectedIndices.splice(index, 1);
                } else {
                    selectedIndices.push(idx);
                }
                redraw();
            };

            const thumb = document.createElement("div");
            thumb.style.cssText =
                `position:relative;aspect-ratio:1;border-radius:4px;overflow:hidden;border:2px solid ${isSelected ? '#4a6' : 'var(--border-color)'};;background:transparent;`;
            thumb.draggable = true;
            thumb.addEventListener("dragstart", (e) => {
                const dt = e?.dataTransfer;
                if (!dt) return;
                const originalUrl = getOriginalViewUrlForDrop(name);
                const downloadName = getDownloadFilename(name);
                const mime = guessMimeType(downloadName);
                try {
                    dt.clearData();
                } catch {}
                dt.effectAllowed = "copy";
                dt.setData(_APT_LOADIMAGE_DRAG_MIME, String(name ?? ""));
                dt.setData("text/uri-list", originalUrl);
                dt.setData("text/plain", originalUrl);
                dt.setData("text/html", `<img src="${originalUrl}">`);
                dt.setData("DownloadURL", `${mime}:${downloadName}:${originalUrl}`);
                dt.setData("text/x-moz-url", `${originalUrl}\n${downloadName}`);
            });

            const badgeScale = Math.max(0.25, Math.min(2.4, Math.sqrt(thumbSize / 64))) * 0.5;
            const badgePadY = Math.max(1, Math.round(3 * badgeScale));
            const badgePadX = Math.max(3, Math.round(6 * badgeScale));
            const badgeFont = Math.max(6, Math.round(14 * badgeScale));
            const badgeRadius = Math.max(2, Math.round(3 * badgeScale));
            const cornerSize = Math.max(8, Math.round(18 * badgeScale));
            const cornerFont = Math.max(5, Math.round(12 * badgeScale));

            // 仅手动选中时显示标记，默认无标记
            if (isSelected) {
                const checkmark = document.createElement("div");
                checkmark.textContent = "✓";
                checkmark.style.cssText =
                    `position:absolute;top:1px;left:1px;width:${cornerSize}px;height:${cornerSize}px;background:rgba(74,170,102,0.9);color:#fff;border-radius:${badgeRadius}px;display:flex;align-items:center;justify-content:center;font-size:${cornerFont}px;z-index:2;`;
                thumb.appendChild(checkmark);
                
                const orderNumber = document.createElement("div");
                orderNumber.textContent = `#${indexInSelection + 1}`;
                orderNumber.style.cssText =
                    `position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#000000;color:#ffffff;border-radius:${badgeRadius}px;padding:${badgePadY}px ${badgePadX}px;font-size:${badgeFont}px;font-weight:bold;z-index:2;`;
                thumb.appendChild(orderNumber);
            }

            const img = document.createElement("img");
            img.src = getViewUrl(name, thumbSize);
            img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;background:transparent;";
            img.draggable = false;

            const del = document.createElement("button");
            del.textContent = "×";
            del.title = "删除";
            del.style.cssText =
                "position:absolute;top:1px;right:1px;width:10px;height:10px;background:rgba(255,0,0,0.75);color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:8px;display:flex;align-items:center;justify-content:center;z-index:1;";
            del.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                const next = names.slice(0, idx).concat(names.slice(idx + 1));
                setImageList(node, next);
                selectedIndices.length = 0;
                redraw();
            };

            const label = document.createElement("div");
            label.textContent = name;
            label.title = name;
            label.style.cssText =
                "font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:0.9;line-height:1.2;margin-top:2px;";

            thumb.appendChild(img);
            thumb.appendChild(del);
            cell.appendChild(thumb);
            cell.appendChild(label);
            frag.appendChild(cell);
        });

        grid.appendChild(frag);
        
        // 同步选中索引到控件（无选择时为空数组）
        const selectedIndicesWidget = getWidgetByName(node, "selected_indices");
        if (selectedIndicesWidget) {
            selectedIndicesWidget.value = serializeSelectedIndicesValue(selectedIndices);
            selectedIndicesWidget.callback?.(selectedIndicesWidget.value);
        }
        
        app.graph.setDirtyCanvas(true);
    };

    const handleDropFiles = async (files, { replace = false } = {}) => {
        if (!files || files.length === 0) return;
        await uploadFilesSequential(node, files, { replace });
        redraw();
    };

    mainContent.addEventListener("dragover", (e) => {
        if (!isFilesDragEvent(e)) return;
        e.preventDefault();
        e.stopPropagation();
    });

    mainContent.addEventListener("drop", async (e) => {
        if (!isFilesDragEvent(e)) return;
        e.preventDefault();
        e.stopPropagation();
        const files = Array.from(e.dataTransfer?.files || []);
        await handleDropFiles(files, { replace: false });
    });

    const setDragging = (on) => {
        container.style.border = on ? "2px dashed #4a6" : "1px solid var(--border-color)";
    };

    replaceBtn.onclick = async () => {
        openMultiSelect(node, { replace: false });
    };
    folderBtn.onclick = async () => {
        openFolderSelect(node, { replace: false });
    };
    clearBtn.onclick = () => {
        setImageList(node, []);
        selectedIndices.length = 0;
        redraw();
    };
    hideBtn.onclick = () => {
        previewsHidden = !previewsHidden;
        hideBtn.textContent = previewsHidden ? "显示" : "隐藏";
        redraw();
    };

    mainContent.appendChild(grid);
    mainContent.appendChild(hiddenOverlay);
    mainContent.insertBefore(jumpSelect, grid);
    container.appendChild(sidebar);
    container.appendChild(mainContent);

    return { container, redraw, setDragging };
}

app.registerExtension({
    name: "IO_LoadImgList.Extension",
    async setup() {
        ensureCanvasLoadImageDrop();
        api.addEventListener("IO_LoadImgList_append", function (event) {
            const nodeId = parseInt(event.detail.node);
            const node = app.graph.nodes.find((n) => n.id === nodeId);
            if (!node) return;

            const namesRaw = event.detail.names;
            const names = Array.isArray(namesRaw) ? namesRaw : typeof namesRaw === "string" ? [namesRaw] : [];
            if (names.length === 0) return;

            const wList = getImageListWidget(node);
            if (!wList) return;

            const existing = parseImageList(wList.value);
            const existingSet = new Set(existing.map((x) => String(x).toLowerCase()));
            const merged = existing.slice();
            for (const n of names) {
                const v = String(n || "").trim();
                if (!v) continue;
                const key = v.toLowerCase();
                if (existingSet.has(key)) continue;
                existingSet.add(key);
                merged.push(v);
            }
            if (merged.length === existing.length) return;

            wList.value = merged.join("\n");
            wList.callback?.(wList.value);

            const wSel = getWidgetByName(node, "selected_indices");
            if (wSel) {
                wSel.value = "[]";
                wSel.callback?.(wSel.value);
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_LoadImgList") return;

        ensureGlobalDragDropPrevention();

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const imageListWidget = getImageListWidget(this);
            if (imageListWidget) {
                imageListWidget.type = "textarea";
                imageListWidget.computeSize = () => [300, 48];
            }
            const thumbSizeWidget = getThumbSizeWidget(this);
            if (thumbSizeWidget) {
                const v = Number(thumbSizeWidget.value);
                if (v !== 64 && v !== 128 && v !== 384) {
                    thumbSizeWidget.value = 64;
                    thumbSizeWidget.callback?.(thumbSizeWidget.value);
                }
            }

            const ui = createBrowserUI(this);
            this._batchLoadImagesUI = ui;
            const minW = 420;
            const minH = 360;
            if (!this.size || this.size[0] < minW || this.size[1] < minH) {
                this.setSize([Math.max(this.size?.[0] || 0, minW), Math.max(this.size?.[1] || 0, minH)]);
            }
            // Set minimum node size to prevent button overflow
            this.minWidth = Math.max(this.minWidth || 0, 420);
            this.minHeight = Math.max(this.minHeight || 0, 360);

            this.addDOMWidget("batch_load_images", "customwidget", ui.container);

            _batchLoadImagesDomUIs.add({ node: this, container: ui.container, redraw: ui.redraw, setDragging: ui.setDragging });

            const prevOnRemoved = this.onRemoved;
            this.onRemoved = function () {
                for (const entry of _batchLoadImagesDomUIs) {
                    if (entry?.node === this) {
                        _batchLoadImagesDomUIs.delete(entry);
                        break;
                    }
                }
                return prevOnRemoved?.apply(this, arguments);
            };

            if (imageListWidget) {
                const origCallback = imageListWidget.callback;
                let lastValue = imageListWidget.value;
                imageListWidget.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw();
                };
            }
            if (thumbSizeWidget) {
                const origCallback = thumbSizeWidget.callback;
                let lastValue = thumbSizeWidget.value;
                thumbSizeWidget.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw();
                };
            }

            ui.redraw();

            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure?.apply(this, arguments);
            const imageListWidget = getImageListWidget(this);
            if (imageListWidget) {
                imageListWidget.type = "textarea";
                imageListWidget.computeSize = () => [300, 48];
            }
            const thumbSizeWidget = getThumbSizeWidget(this);
            if (thumbSizeWidget) {
                const v = Number(thumbSizeWidget.value);
                if (v !== 64 && v !== 128 && v !== 384) {
                    thumbSizeWidget.value = 64;
                    thumbSizeWidget.callback?.(thumbSizeWidget.value);
                }
            }
            this._batchLoadImagesUI?.redraw?.();
            return r;
        };
    },
});

app.registerExtension({
    name: "apt.VisualPositionControl.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "apt_VisualPositionControl") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const ui = createVNCCSVisualUI(this);
            if (ui) {
                this.addDOMWidget("vnccs_visual", "customwidget", ui.container);
                this.setSize([420, 220]);
                ui.read();
            }

            return r;
        };
    },
});
