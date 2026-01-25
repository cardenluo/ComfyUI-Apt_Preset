import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function getImageListWidget(node) {
    return node?.widgets?.find((w) => w.name === "image_list");
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
        // Fallback: modify prompt JSON directly.
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

function getViewUrl(filename) {
    const previewParam = app.getPreviewFormatParam?.() || "";
    const randParam = app.getRandParam?.() || "";
    return api.apiURL(`/view?filename=${encodeURIComponent(filename)}&type=input${previewParam}${randParam}`);
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
        // skip non-images
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
            // keep stable ordering
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
        "width:100%;padding:8px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:5px 0;pointer-events:auto;display:flex;flex-direction:row;gap:8px;";

    // Left sidebar with buttons - width reduced by half
    const sidebar = document.createElement("div");
    sidebar.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:34px;width:34px;";

    const mkBtn = (label) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
            "padding:4px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;cursor:pointer;font-size:10px;width:100%;text-align:center;";
        return b;
    };

    const replaceBtn = mkBtn("🖼️");
    const folderBtn = mkBtn("📂");
    const selectAllBtn = mkBtn("全选");
    const sortAscBtn = mkBtn("顺序");
    const sortDescBtn = mkBtn("逆序");
    const deleteSelectedBtn = mkBtn("删除");
    const clearBtn = mkBtn("清空");

    // 添加按钮
    sidebar.appendChild(replaceBtn);
    sidebar.appendChild(folderBtn);
    sidebar.appendChild(selectAllBtn);
    sidebar.appendChild(sortAscBtn);
    sidebar.appendChild(sortDescBtn);
    sidebar.appendChild(deleteSelectedBtn);
    sidebar.appendChild(clearBtn);

    // 选中图片的索引数组，用于保持选择顺序
    const selectedIndices = [];

    // 全选/取消全选
    selectAllBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        if (selectedIndices.length === names.length) {
            // 如果已经全选，取消全选
            selectedIndices.length = 0;
        } else {
            // 否则全选，按照索引顺序添加
            selectedIndices.length = 0;
            for (let i = 0; i < names.length; i++) {
                selectedIndices.push(i);
            }
        }
        redraw();
    };

    // 按名称顺序排序
    sortAscBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        const sortedNames = [...names].sort((a, b) => a.localeCompare(b));
        setImageList(node, sortedNames);
        selectedIndices.length = 0;
        redraw();
    };

    // 按名称逆序排序
    sortDescBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        const sortedNames = [...names].sort((a, b) => b.localeCompare(a));
        setImageList(node, sortedNames);
        selectedIndices.length = 0;
        redraw();
    };

    // 删除选中图片
    deleteSelectedBtn.onclick = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        // 按索引从大到小删除，避免索引偏移
        const sortedIndices = [...selectedIndices].sort((a, b) => b - a);
        let newNames = [...names];
        for (const idx of sortedIndices) {
            newNames = newNames.slice(0, idx).concat(newNames.slice(idx + 1));
        }
        setImageList(node, newNames);
        selectedIndices.length = 0;
        redraw();
    };

    // Main content area
    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;";

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:6px;flex:1;overflow-y:auto;background:var(--comfy-input-bg);padding:6px;border-radius:4px;";

    const redraw = () => {
        const names = parseImageList(getImageListWidget(node)?.value);
        grid.innerHTML = "";

        const frag = document.createDocumentFragment();
        names.forEach((name, idx) => {
            const indexInSelection = selectedIndices.indexOf(idx);
            const isSelected = indexInSelection > -1;
            
            const cell = document.createElement("div");
            cell.style.cssText = `display:flex;flex-direction:column;gap:3px;cursor:pointer;${isSelected ? 'opacity:0.7;' : ''}`;
            
            // 点击切换选择状态
            cell.onclick = (e) => {
                e.preventDefault();
                const index = selectedIndices.indexOf(idx);
                if (index > -1) {
                    // 如果已选中，移除
                    selectedIndices.splice(index, 1);
                } else {
                    // 如果未选中，添加到数组末尾
                    selectedIndices.push(idx);
                }
                redraw();
            };

            const thumb = document.createElement("div");
            thumb.style.cssText =
                `position:relative;aspect-ratio:1;border-radius:4px;overflow:hidden;border:2px solid ${isSelected ? '#4a6' : 'var(--border-color)'};;background:#000;`;

            // 选中指示器
            if (isSelected) {
                // 左上角勾选标记
                const checkmark = document.createElement("div");
                checkmark.textContent = "✓";
                checkmark.style.cssText =
                    "position:absolute;top:2px;left:2px;width:20px;height:20px;background:rgba(74,170,102,0.9);color:#fff;border-radius:3px;display:flex;align-items:center;justify-content:center;font-size:14px;";
                thumb.appendChild(checkmark);
                
                // 中央顺序编号
                const orderNumber = document.createElement("div");
                orderNumber.textContent = `#${indexInSelection + 1}`;
                orderNumber.style.cssText =
                    "position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);background:#000000;color:#ffffff;border-radius:4px;padding:4px 8px;font-size:16px;font-weight:bold;z-index:1;";
                thumb.appendChild(orderNumber);
            }

            const img = document.createElement("img");
            img.src = getViewUrl(name);
            img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";

            const del = document.createElement("button");
            del.textContent = "×";
            del.title = "删除";
            del.style.cssText =
                "position:absolute;top:2px;right:2px;width:20px;height:20px;background:rgba(255,0,0,0.75);color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:16px;line-height:1;z-index:1;";
            del.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                const next = names.slice(0, idx).concat(names.slice(idx + 1));
                setImageList(node, next);
                selectedIndices.length = 0; // 清空选择，因为索引可能变化
                redraw();
            };

            const label = document.createElement("div");
            label.textContent = name;
            label.title = name;
            label.style.cssText =
                "font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:0.9;";

            thumb.appendChild(img);
            thumb.appendChild(del);
            cell.appendChild(thumb);
            cell.appendChild(label);
            frag.appendChild(cell);
        });

        grid.appendChild(frag);
        
        // 更新选中索引到后端
        const selectedIndicesWidget = getWidgetByName(node, "selected_indices");
        if (selectedIndicesWidget) {
            selectedIndicesWidget.value = JSON.stringify(selectedIndices);
            selectedIndicesWidget.callback?.(selectedIndicesWidget.value);
        }
        
        app.graph.setDirtyCanvas(true);
    };

    const handleDropFiles = async (files, { replace = false } = {}) => {
        if (!files || files.length === 0) return;
        await uploadFilesSequential(node, files, { replace });
        redraw();
    };

    // Most reliable: handle drop on our DOM panel.
    container.addEventListener("dragover", (e) => {
        if (!isFilesDragEvent(e)) return;
        e.preventDefault();
        e.stopPropagation();
    });

    container.addEventListener("drop", async (e) => {
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
        openMultiSelect(node, { replace: true });
    };
    folderBtn.onclick = async () => {
        openFolderSelect(node, { replace: true });
    };
    clearBtn.onclick = () => {
        setImageList(node, []);
        redraw();
    };

    mainContent.appendChild(grid);
    container.appendChild(sidebar);
    container.appendChild(mainContent);

    return { container, redraw, setDragging };
}

app.registerExtension({
    name: "IO_LoadImgList.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_LoadImgList") return;

        ensureGlobalDragDropPrevention();

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const imageListWidget = getImageListWidget(this);
            if (imageListWidget) {
                // Hide the giant textbox; we manage it through the DOM UI.
                imageListWidget.type = "hidden";
                imageListWidget.computeSize = () => [0, -4];
            }

            // Create file-browser like UI
            const ui = createBrowserUI(this);
            this._batchLoadImagesUI = ui;
            this.addDOMWidget("batch_load_images", "customwidget", ui.container);
            this.setSize([420, 320]);

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

            // Keep the DOM gallery in sync if something else changes the widget.
            if (imageListWidget) {
                const origCallback = imageListWidget.callback;
                imageListWidget.callback = function (value) {
                    origCallback?.call(this, value);
                    ui.redraw();
                };
            }

            ui.redraw();

            return r;
        };

        const origOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (output) {
            origOnExecuted?.apply(this, arguments);
            this._batchLoadImagesUI?.redraw?.();
        };
    },
});

app.registerExtension({
    name: "VNCCS.VisualPositionControl.Extension",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "VNCCS_VisualPositionControl") return;

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
