import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const MEDIA_NODE_CONFIG = {
    IO_LoadVideoBatch: {
        listWidget: "video_list",
        indexWidget: "index",
        cardSizeWidget: "card_size",
        eventName: "IO_LoadVideoBatch_set",
        mediaType: "video",
        title: "视频",
    },
    IO_LoadAudioBatch: {
        listWidget: "audio_list",
        indexWidget: "index",
        cardSizeWidget: "card_size",
        eventName: "IO_LoadAudioBatch_set",
        mediaType: "audio",
        title: "音频",
    },
};

function getWidgetByName(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function parseNameList(text) {
    return (text || "")
        .split("\n")
        .map((s) => String(s || "").replace(/\r/g, "").trim())
        .filter((s) => s !== "");
}

async function uploadOneMedia(file, mediaType) {
    const body = new FormData();
    body.append("media", file, file.name);
    const resp = await api.fetchApi(`/Apt_Preset_IO_LoadMedia_upload?media_type=${encodeURIComponent(mediaType)}`, {
        method: "POST",
        body,
    });
    if (!resp.ok) throw new Error(await resp.text());
    const json = await resp.json();
    const items = Array.isArray(json?.items) ? json.items : [];
    return items.length > 0 ? String(items[0]) : "";
}

async function uploadMediaFilesSequential(files, mediaType) {
    const uploaded = [];
    for (const file of files || []) {
        if (!file) continue;
        const name = await uploadOneMedia(file, mediaType);
        if (name) uploaded.push(name);
    }
    return uploaded;
}

function setNameList(node, cfg, names) {
    const w = getWidgetByName(node, cfg.listWidget);
    if (!w) return;
    const next = Array.isArray(names) ? names : [];
    w.value = next.join("\n");
    w.callback?.(w.value);
}

function getCardSize(node, cfg) {
    const w = getWidgetByName(node, cfg.cardSizeWidget);
    const v = Number(w?.value);
    return Number.isFinite(v) ? Math.floor(v) : 120;
}

function setCardSize(node, cfg, size) {
    const w = getWidgetByName(node, cfg.cardSizeWidget);
    if (!w) return;
    const v = Number(size);
    w.value = Number.isFinite(v) ? Math.floor(v) : 120;
    w.callback?.(w.value);
}

function getIndex(node, cfg) {
    const w = getWidgetByName(node, cfg.indexWidget);
    const v = Number(w?.value);
    return Number.isFinite(v) ? Math.floor(v) : 0;
}

function setIndex(node, cfg, idx) {
    const w = getWidgetByName(node, cfg.indexWidget);
    if (!w) return;
    const v = Number(idx);
    w.value = Number.isFinite(v) ? Math.floor(v) : 0;
    w.callback?.(w.value);
}

function getMediaPreviewUrl(path, mediaType) {
    return api.apiURL(`/Apt_Preset_IO_LoadMedia_preview?path=${encodeURIComponent(path)}&media=${encodeURIComponent(mediaType)}`);
}

function createMediaBatchUI(node, cfg) {
    const container = document.createElement("div");
    container.style.cssText =
        "width:100%;height:100%;min-width:0;box-sizing:border-box;overflow:hidden;padding:8px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:5px 0;display:flex;flex-direction:row;gap:8px;z-index:10;";
    container.style.userSelect = "none";
    container.style.webkitUserSelect = "none";

    const sidebar = document.createElement("div");
    sidebar.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:56px;width:56px;pointer-events:auto;";

    const mkBtn = (label) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
            "padding:4px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;cursor:pointer;font-size:10px;width:100%;text-align:center;";
        return b;
    };

    const singleBtn = mkBtn(cfg.mediaType === "video" ? "视频" : "音频");
    const folderBtn = mkBtn("📂");
    const sortBtn = mkBtn("顺序");
    const deleteBtn = mkBtn("删除");
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

    sidebar.appendChild(singleBtn);
    sidebar.appendChild(folderBtn);
    sidebar.appendChild(sortBtn);
    sidebar.appendChild(deleteBtn);
    sidebar.appendChild(clearBtn);
    sidebar.appendChild(hideBtn);
    sidebar.appendChild(sizeBtn);
    sidebar.appendChild(sizeSlider);
    sidebar.appendChild(sizeVal);

    let previewsHidden = false;
    let nextSortIsAsc = true;
    const SIZE_PRESETS = [64, 128, 384];
    const LONG_PRESS_MS = 260;
    const DRAG_CANCEL_MOVE_PX = 8;

    const reorderNames = (arr, fromIdx, toIdx) => {
        if (!Array.isArray(arr)) return [];
        const len = arr.length;
        if (len <= 1) return arr.slice();
        const from = Math.max(0, Math.min(len - 1, Math.floor(Number(fromIdx))));
        const to = Math.max(0, Math.min(len - 1, Math.floor(Number(toIdx))));
        if (from === to) return arr.slice();
        const next = arr.slice();
        const [moved] = next.splice(from, 1);
        next.splice(to, 0, moved);
        return next;
    };

    let longPressTimer = null;
    let pressState = null;
    let dragState = null;
    let suppressNextClick = false;

    const clearLongPressTimer = () => {
        if (!longPressTimer) return;
        clearTimeout(longPressTimer);
        longPressTimer = null;
    };

    const clearDragVisual = () => {
        if (!dragState?.sourceCell) return;
        dragState.sourceCell.style.opacity = "";
        dragState.sourceCell.style.outline = "";
        dragState.sourceCell.style.outlineOffset = "";
    };

    const beginDragFromPress = (pointerId) => {
        if (!pressState || pressState.pointerId !== pointerId) return;
        dragState = {
            pointerId: pressState.pointerId,
            currentIdx: pressState.idx,
            sourceCell: pressState.sourceCell,
        };
        suppressNextClick = true;
        container.style.cursor = "grabbing";
        if (dragState.sourceCell) {
            dragState.sourceCell.style.opacity = "0.65";
            dragState.sourceCell.style.outline = "1px dashed #4a6";
            dragState.sourceCell.style.outlineOffset = "1px";
        }
    };

    const finishPointerOps = () => {
        clearLongPressTimer();
        pressState = null;
        clearDragVisual();
        dragState = null;
        container.style.cursor = "";
    };

    const syncSizeUIFromWidget = () => {
        const current = getCardSize(node, cfg);
        let idx = SIZE_PRESETS.indexOf(current);
        if (idx < 0) {
            setCardSize(node, cfg, 64);
            idx = 0;
        }
        sizeSlider.value = String(idx);
        sizeVal.textContent = String(getCardSize(node, cfg));
    };

    const applyCardSizeFromSlider = () => {
        const idx = Math.max(0, Math.min(SIZE_PRESETS.length - 1, Math.floor(Number(sizeSlider.value))));
        const size = SIZE_PRESETS[idx] ?? 64;
        setCardSize(node, cfg, size);
        sizeVal.textContent = String(size);
        redraw(true);
    };

    sizeBtn.onclick = () => applyCardSizeFromSlider();
    sizeSlider.addEventListener("input", applyCardSizeFromSlider);
    syncSizeUIFromWidget();

    sortBtn.onclick = () => {
        const names = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
        const sorted = [...names].sort((a, b) => (nextSortIsAsc ? a.localeCompare(b) : b.localeCompare(a)));
        setNameList(node, cfg, sorted);
        nextSortIsAsc = !nextSortIsAsc;
        sortBtn.textContent = nextSortIsAsc ? "顺序" : "逆序";
        redraw(true);
    };

    deleteBtn.onclick = () => {
        const names = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
        if (names.length === 0) return;
        const idx = getIndex(node, cfg);
        const idx0 = Math.max(0, Math.min(names.length - 1, idx));
        const next = names.slice(0, idx0).concat(names.slice(idx0 + 1));
        setNameList(node, cfg, next);
        if (next.length === 0) setIndex(node, cfg, 0);
        else setIndex(node, cfg, Math.max(0, Math.min(next.length - 1, idx)));
        redraw(true);
    };

    clearBtn.onclick = () => {
        setNameList(node, cfg, []);
        setIndex(node, cfg, 0);
        redraw(true);
    };

    hideBtn.onclick = () => {
        previewsHidden = !previewsHidden;
        hideBtn.textContent = previewsHidden ? "显示" : "隐藏";
        redraw();
    };

    const fileAccept = cfg.mediaType === "video"
        ? ".mp4,.mov,.mkv,.webm,.avi,.m4v,video/*"
        : ".wav,.mp3,.flac,.ogg,.m4a,.aac,audio/*";

    singleBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = fileAccept;
        input.multiple = true;
        input.style.display = "none";
        document.body.appendChild(input);

        input.onchange = async (e) => {
            try {
                const files = Array.from(e?.target?.files || []);
                if (files.length === 0) return;
                const uploaded = await uploadMediaFilesSequential(files, cfg.mediaType);
                if (uploaded.length === 0) return;
                const all = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
                all.push(...uploaded);
                setNameList(node, cfg, all);
                setIndex(node, cfg, Math.max(0, all.length - 1));
                redraw(true);
            } finally {
                try {
                    document.body.removeChild(input);
                } catch {}
            }
        };
        input.click();
    };

    folderBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = fileAccept;
        input.multiple = true;
        input.webkitdirectory = true;
        input.directory = true;
        input.style.display = "none";
        document.body.appendChild(input);

        input.onchange = async (e) => {
            try {
                let files = Array.from(e?.target?.files || []);
                files.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
                const uploaded = await uploadMediaFilesSequential(files, cfg.mediaType);
                if (uploaded.length === 0) return;
                const all = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
                all.push(...uploaded);
                setNameList(node, cfg, all);
                setIndex(node, cfg, Math.max(0, all.length - 1));
                redraw(true);
            } finally {
                try {
                    document.body.removeChild(input);
                } catch {}
            }
        };
        input.click();
    };

    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;pointer-events:auto;min-width:0;min-height:0;";

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--card-size,220px),1fr));gap:6px;flex:1;min-width:0;min-height:0;overflow-y:auto;background:var(--comfy-input-bg);padding:6px;border-radius:4px;";

    const hiddenOverlay = document.createElement("div");
    hiddenOverlay.style.cssText =
        "flex:1;display:none;align-items:center;justify-content:center;background:var(--comfy-input-bg);border-radius:4px;color:var(--input-text);font-size:12px;opacity:0.75;";

    let lastNames = null;
    let lastCardSize = null;

    const redraw = (forceFull = false) => {
        const names = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
        const cardSize = getCardSize(node, cfg);
        grid.style.setProperty("--card-size", `${cardSize}px`);

        if (previewsHidden) {
            grid.style.display = "none";
            hiddenOverlay.style.display = "flex";
            hiddenOverlay.textContent = `预览已隐藏（${names.length}）`;
            app.graph.setDirtyCanvas(true);
            return;
        }

        grid.style.display = "grid";
        hiddenOverlay.style.display = "none";
        const idx = getIndex(node, cfg);

        const namesUnchanged = lastNames && names.length === lastNames.length && names.every((n, i) => n === lastNames[i]);
        const sizeUnchanged = lastCardSize === cardSize;
        if (!forceFull && namesUnchanged && sizeUnchanged) {
            const cards = grid.querySelectorAll("[data-io-media-card]");
            cards.forEach((cell, idx0) => {
                const card = cell.querySelector(":scope > div");
                if (card) {
                    const isSelected = idx0 === idx;
                    card.style.borderColor = isSelected ? "#4a6" : "var(--border-color)";
                }
            });
            app.graph.setDirtyCanvas(true);
            return;
        }

        lastNames = [...names];
        lastCardSize = cardSize;
        grid.innerHTML = "";

        const frag = document.createDocumentFragment();
        names.forEach((name, idx0) => {
            const isSelected = idx0 === idx;
            const cell = document.createElement("div");
            cell.style.cssText = "display:flex;flex-direction:column;cursor:pointer;";
            cell.dataset.ioMediaCard = "1";
            cell.dataset.ioMediaIndex = String(idx0);

            cell.onclick = (e) => {
                if (suppressNextClick) {
                    e.preventDefault();
                    e.stopPropagation();
                    suppressNextClick = false;
                    return;
                }
                e.preventDefault();
                setIndex(node, cfg, idx0);
                redraw(false);
            };

            cell.onpointerdown = (e) => {
                if (e.button !== 0) return;
                const tag = String(e.target?.tagName || "").toUpperCase();
                if (["BUTTON", "INPUT"].includes(tag)) return;
                if (e.target?.closest?.("button")) return;

                finishPointerOps();
                pressState = {
                    pointerId: e.pointerId,
                    startX: e.clientX,
                    startY: e.clientY,
                    idx: idx0,
                    sourceCell: cell,
                };

                longPressTimer = setTimeout(() => {
                    beginDragFromPress(e.pointerId);
                }, LONG_PRESS_MS);
            };

            const card = document.createElement("div");
            card.style.cssText = `position:relative;border-radius:6px;border:2px solid ${
                isSelected ? "#4a6" : "var(--border-color)"
            };background:var(--comfy-menu-bg);padding:2px;display:flex;flex-direction:column;gap:2px;min-height:${Math.max(
                100,
                Math.floor(cardSize)
            )}px;max-height:${Math.max(120, Math.floor(cardSize) + 40)}px;overflow:hidden;`;

            if (cfg.mediaType === "video") {
                const video = document.createElement("video");
                video.src = getMediaPreviewUrl(name, cfg.mediaType);
                video.controls = true;
                video.preload = "metadata";
                video.muted = true;
                video.style.cssText = "width:100%;max-height:100%;object-fit:cover;display:block;background:#000;border-radius:4px;";
                card.appendChild(video);
            } else {
                const audio = document.createElement("audio");
                audio.src = getMediaPreviewUrl(name, cfg.mediaType);
                audio.controls = true;
                audio.preload = "metadata";
                audio.style.cssText = "width:100%;display:block;background:#111;border-radius:4px;";
                card.appendChild(audio);
            }

            const badge = document.createElement("div");
            badge.textContent = `#${idx0}`;
            badge.style.cssText =
                "position:absolute;top:3px;left:3px;font-size:10px;opacity:0.9;background:rgba(0,0,0,0.7);color:#fff;border-radius:999px;padding:2px 6px;max-width:50px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;z-index:2;";

            const del = document.createElement("button");
            del.textContent = "×";
            del.title = "删除";
            del.style.cssText =
                "position:absolute;top:3px;right:3px;width:18px;height:18px;background:rgba(255,0,0,0.75);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:11px;display:flex;align-items:center;justify-content:center;z-index:2;";
            del.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                const all = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
                const next = all.slice(0, idx0).concat(all.slice(idx0 + 1));
                setNameList(node, cfg, next);
                if (next.length === 0) setIndex(node, cfg, 0);
                else setIndex(node, cfg, Math.max(0, Math.min(next.length - 1, idx)));
                redraw(true);
            };

            const label = document.createElement("div");
            label.textContent = name;
            label.title = name;
            label.style.cssText =
                "font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:0.9;line-height:1.2;";

            card.appendChild(badge);
            card.appendChild(del);
            card.appendChild(label);
            cell.appendChild(card);
            frag.appendChild(cell);
        });

        grid.appendChild(frag);
        app.graph.setDirtyCanvas(true);
    };

    const handlePointerMove = (e) => {
        if (pressState && !dragState && e.pointerId === pressState.pointerId) {
            const dx = e.clientX - pressState.startX;
            const dy = e.clientY - pressState.startY;
            if (Math.hypot(dx, dy) > DRAG_CANCEL_MOVE_PX) {
                clearLongPressTimer();
                beginDragFromPress(e.pointerId);
            }
            return;
        }
        if (!dragState || e.pointerId !== dragState.pointerId) return;
        e.preventDefault();
        e.stopPropagation();

        const elem = document.elementFromPoint(e.clientX, e.clientY);
        const targetCell = elem?.closest?.("[data-io-media-card]");
        if (!targetCell) return;
        const targetIdx = Number(targetCell.dataset.ioMediaIndex);
        if (!Number.isFinite(targetIdx) || targetIdx === dragState.currentIdx) return;

        const names = parseNameList(getWidgetByName(node, cfg.listWidget)?.value);
        if (names.length <= 1) return;
        if (dragState.currentIdx < 0 || dragState.currentIdx >= names.length) return;
        if (targetIdx < 0 || targetIdx >= names.length) return;

        const reordered = reorderNames(names, dragState.currentIdx, targetIdx);
        setNameList(node, cfg, reordered);
        setIndex(node, cfg, targetIdx);
        dragState.currentIdx = targetIdx;
        redraw(true);
    };

    const handlePointerUpOrCancel = (e) => {
        if (pressState && e.pointerId !== pressState.pointerId && (!dragState || e.pointerId !== dragState.pointerId)) return;
        finishPointerOps();
    };

    window.addEventListener("pointermove", handlePointerMove, true);
    window.addEventListener("pointerup", handlePointerUpOrCancel, true);
    window.addEventListener("pointercancel", handlePointerUpOrCancel, true);

    mainContent.appendChild(grid);
    mainContent.appendChild(hiddenOverlay);
    container.appendChild(sidebar);
    container.appendChild(mainContent);
    return { container, redraw };
}

function applyEventToNode(eventDetail, cfg) {
    const nodeId = parseInt(eventDetail.node);
    const nodes = app.graph?._nodes || app.graph?.nodes || [];
    const node = nodes.find((n) => n.id === nodeId);
    if (!node) return;

    const itemsRaw = eventDetail.items;
    const items = Array.isArray(itemsRaw) ? itemsRaw : [];
    const wList = getWidgetByName(node, cfg.listWidget);
    if (wList) {
        const next = items
            .map((x) => String(x ?? "").replace(/\r/g, "").trim())
            .filter((x) => x !== "");
        const nextStr = next.join("\n");
        if (wList.value !== nextStr) {
            wList.value = nextStr;
            wList.callback?.(wList.value);
        }
    }

    const wIndex = getWidgetByName(node, cfg.indexWidget);
    if (wIndex) {
        const idx = Number(eventDetail.index);
        const v = Number.isFinite(idx) ? Math.floor(idx) : wIndex.value;
        if (wIndex.value !== v) {
            wIndex.value = v;
            wIndex.callback?.(wIndex.value);
        }
    }

    const wSize = getWidgetByName(node, cfg.cardSizeWidget);
    if (wSize) {
        const sz = Number(eventDetail.card_size);
        const v = Number.isFinite(sz) ? Math.floor(sz) : wSize.value;
        if (wSize.value !== v) {
            wSize.value = v;
            wSize.callback?.(wSize.value);
        }
    }

    node._ioLoadMediaBatchUI?.redraw?.(true);
}

app.registerExtension({
    name: "IO_LoadMediaBatch.Extension",
    async setup() {
        Object.values(MEDIA_NODE_CONFIG).forEach((cfg) => {
            api.addEventListener(cfg.eventName, (event) => applyEventToNode(event.detail || {}, cfg));
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const cfg = MEDIA_NODE_CONFIG[nodeData.name];
        if (!cfg) return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);
            const listWidget = getWidgetByName(this, cfg.listWidget);
            if (listWidget) {
                listWidget.type = "textarea";
                listWidget.computeSize = () => [300, 48];
            }
            const sizeWidget = getWidgetByName(this, cfg.cardSizeWidget);
            if (sizeWidget) {
                sizeWidget.type = "number";
                sizeWidget.computeSize = () => [120, 24];
            }

            const ui = createMediaBatchUI(this, cfg);
            this._ioLoadMediaBatchUI = ui;

            const minW = 420;
            const minH = 360;
            if (!this.size || this.size[0] < minW || this.size[1] < minH) {
                this.setSize([Math.max(this.size?.[0] || 0, minW), Math.max(this.size?.[1] || 0, minH)]);
            }
            this.minWidth = Math.max(this.minWidth || 0, 420);
            this.minHeight = Math.max(this.minHeight || 0, 360);
            this.addDOMWidget(`io_load_media_batch_${cfg.mediaType}`, "customwidget", ui.container);

            const wIndex = getWidgetByName(this, cfg.indexWidget);
            const wList = getWidgetByName(this, cfg.listWidget);
            const wSize = getWidgetByName(this, cfg.cardSizeWidget);

            if (wIndex) {
                const origCallback = wIndex.callback;
                let lastValue = wIndex.value;
                wIndex.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw(false);
                };
            }
            if (wList) {
                const origCallback = wList.callback;
                let lastValue = wList.value;
                wList.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw(true);
                };
            }
            if (wSize) {
                const origCallback = wSize.callback;
                let lastValue = wSize.value;
                wSize.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw(true);
                };
            }

            ui.redraw(true);
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure?.apply(this, arguments);
            const listWidget = getWidgetByName(this, cfg.listWidget);
            if (listWidget) {
                listWidget.type = "textarea";
                listWidget.computeSize = () => [300, 48];
            }
            const sizeWidget = getWidgetByName(this, cfg.cardSizeWidget);
            if (sizeWidget) {
                sizeWidget.type = "number";
                sizeWidget.computeSize = () => [120, 24];
            }
            this._ioLoadMediaBatchUI?.redraw?.(true);
            return r;
        };
    },
});
