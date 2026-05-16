import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function getWidgetByName(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function getImageListWidget(node) {
    return getWidgetByName(node, "image_list");
}

function getCardSizeWidget(node) {
    return getWidgetByName(node, "card_size");
}

function getIndexWidget(node) {
    return getWidgetByName(node, "index");
}

function parseNameList(text) {
    return (text || "")
        .split("\n")
        .map((s) => String(s || "").replace(/\r/g, "").trim())
        .filter((s) => s !== "");
}

function setNameList(node, names) {
    const w = getImageListWidget(node);
    if (!w) return;
    const next = Array.isArray(names) ? names : [];
    w.value = next.join("\n");
    w.callback?.(w.value);
}

function getCardSize(node) {
    const w = getCardSizeWidget(node);
    const v = Number(w?.value);
    return Number.isFinite(v) ? Math.floor(v) : 120;
}

function setCardSize(node, size) {
    const w = getCardSizeWidget(node);
    if (!w) return;
    const v = Number(size);
    w.value = Number.isFinite(v) ? Math.floor(v) : 120;
    w.callback?.(w.value);
}

function getIndex(node) {
    const w = getIndexWidget(node);
    const v = Number(w?.value);
    return Number.isFinite(v) ? Math.floor(v) : 0;
}

function setIndex(node, idx) {
    const w = getIndexWidget(node);
    if (!w) return;
    const v = Number(idx);
    w.value = Number.isFinite(v) ? Math.floor(v) : 0;
    w.callback?.(w.value);
}

function getViewUrl(filename, size = 120) {
    return api.apiURL(`/Apt_Preset_IO_LoadImgList_thumb?filename=${encodeURIComponent(filename)}&size=${encodeURIComponent(size)}`);
}

async function uploadOneImage(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!resp.ok) throw new Error(await resp.text());
    const json = await resp.json();
    return json?.name;
}

async function uploadFilesSequential(files) {
    const uploaded = [];
    for (const file of files || []) {
        if (!file) continue;
        if (file?.type && !String(file.type).startsWith("image/")) continue;
        const name = await uploadOneImage(file);
        if (name) uploaded.push(name);
    }
    return uploaded;
}

function _sanitizeSingleLineText(raw) {
    return String(raw ?? "")
        .replace(/\r/g, "")
        .replace(/\n/g, " ")
        .trim();
}

function _isInsideContainerPoint(container, x, y) {
    const rect = container.getBoundingClientRect?.();
    if (!rect) return false;
    return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
}

function _insertIntoInput(el, text) {
    try {
        if (!el) return false;
        const t = String(text ?? "");
        if (t === "") return true;
        const tag = String(el.tagName || "").toLowerCase();
        if (tag === "textarea") {
            el.focus?.();
            const start = typeof el.selectionStart === "number" ? el.selectionStart : el.value.length;
            const end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
            el.value = String(el.value ?? "").slice(0, start) + t + String(el.value ?? "").slice(end);
            const nextPos = start + t.length;
            el.selectionStart = nextPos;
            el.selectionEnd = nextPos;
            el.dispatchEvent?.(new Event("input", { bubbles: true }));
            return true;
        }
        if (tag === "input") {
            const type = String(el.type || "text").toLowerCase();
            const ok = new Set(["text", "search", "url", "email", "tel", "password", "number"]);
            if (!ok.has(type)) return false;
            el.focus?.();
            const start = typeof el.selectionStart === "number" ? el.selectionStart : el.value.length;
            const end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
            el.value = String(el.value ?? "").slice(0, start) + t + String(el.value ?? "").slice(end);
            const nextPos = start + t.length;
            try {
                el.selectionStart = nextPos;
                el.selectionEnd = nextPos;
            } catch {}
            el.dispatchEvent?.(new Event("input", { bubbles: true }));
            return true;
        }
        if (el.isContentEditable) {
            el.focus?.();
            try {
                document.execCommand("insertText", false, t);
                return true;
            } catch {
                return false;
            }
        }
        return false;
    } catch {
        return false;
    }
}

async function _copyImageToClipboard(filename) {
    const name = String(filename ?? "").trim();
    if (!name) return false;
    try {
        const url = `/view?filename=${encodeURIComponent(name)}&type=input&subfolder=`;
        const resp = await api.fetchApi(url, { cache: "no-store" });
        if (!resp.ok) return false;
        const blob = await resp.blob();
        if (!blob) return false;
        if (!window.ClipboardItem || !navigator.clipboard?.write) return false;
        const mime = blob.type && blob.type !== "" ? blob.type : "image/png";
        await navigator.clipboard.write([new ClipboardItem({ [mime]: blob })]);
        return true;
    } catch {
        return false;
    }
}

async function _copyToClipboard(text) {
    const t = String(text ?? "");
    if (t === "") return;
    try {
        await navigator.clipboard.writeText(t);
    } catch {}
}

function createImgBatchUI(node) {
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

    const singleBtn = mkBtn("图片");
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

    const syncSizeUIFromWidget = () => {
        const current = getCardSize(node);
        let idx = SIZE_PRESETS.indexOf(current);
        if (idx < 0) {
            setCardSize(node, 64);
            idx = 0;
        }
        sizeSlider.value = String(idx);
        sizeVal.textContent = String(getCardSize(node));
    };

    const applyCardSizeFromSlider = () => {
        const idx = Math.max(0, Math.min(SIZE_PRESETS.length - 1, Math.floor(Number(sizeSlider.value))));
        const size = SIZE_PRESETS[idx] ?? 64;
        setCardSize(node, size);
        sizeVal.textContent = String(size);
        redraw(true); // 尺寸变化需要完整重绘
    };

    sizeBtn.onclick = () => applyCardSizeFromSlider();
    sizeSlider.addEventListener("input", applyCardSizeFromSlider);
    syncSizeUIFromWidget();

    sortBtn.onclick = () => {
        const names = parseNameList(getImageListWidget(node)?.value);
        const sorted = [...names].sort((a, b) => (nextSortIsAsc ? a.localeCompare(b) : b.localeCompare(a)));
        setNameList(node, sorted);
        nextSortIsAsc = !nextSortIsAsc;
        sortBtn.textContent = nextSortIsAsc ? "顺序" : "逆序";
        redraw(true); // 强制完整重绘
    };

    deleteBtn.onclick = () => {
        const names = parseNameList(getImageListWidget(node)?.value);
        if (names.length === 0) return;
        const idx = getIndex(node);
        const idx0 = Math.max(0, Math.min(names.length - 1, idx));
        const next = names.slice(0, idx0).concat(names.slice(idx0 + 1));
        setNameList(node, next);
        if (next.length === 0) setIndex(node, 0);
        else setIndex(node, Math.max(0, Math.min(next.length - 1, idx)));
        redraw(true); // 强制完整重绘
    };

    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;pointer-events:auto;min-width:0;min-height:0;";
    mainContent.style.userSelect = "none";
    mainContent.style.webkitUserSelect = "none";

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--card-size,220px),1fr));gap:6px;flex:1;min-width:0;min-height:0;overflow-y:auto;background:var(--comfy-input-bg);padding:6px;border-radius:4px;";
    grid.style.userSelect = "none";
    grid.style.webkitUserSelect = "none";
    grid.style.touchAction = "none";

    const hiddenOverlay = document.createElement("div");
    hiddenOverlay.style.cssText =
        "flex:1;display:none;align-items:center;justify-content:center;background:var(--comfy-input-bg);border-radius:4px;color:var(--input-text);font-size:12px;opacity:0.75;";

    // 缓存上次的图片列表，用于增量更新
    let lastNames = null;
    let lastCardSize = null;

    const redraw = (forceFull = false) => {
        const names = parseNameList(getImageListWidget(node)?.value);
        const cardSize = getCardSize(node);
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

        const idx = getIndex(node);

        // 检查是否可以增量更新（只更新选中状态）
        const namesUnchanged = lastNames && names.length === lastNames.length &&
            names.every((n, i) => n === lastNames[i]);
        const sizeUnchanged = lastCardSize === cardSize;

        if (!forceFull && namesUnchanged && sizeUnchanged) {
            // 增量更新：只更新选中状态的样式
            const cards = grid.querySelectorAll("[data-io-img-card]");
            cards.forEach((cell, idx0) => {
                const card = cell.querySelector(":scope > div");
                if (card) {
                    const isSelected = idx0 === idx;
                    card.style.borderColor = isSelected ? "#4a6" : "var(--border-color)";
                }
                cell.dataset.ioImgIndex0 = String(idx0);
            });
            app.graph.setDirtyCanvas(true);
            return;
        }

        // 完整重绘
        lastNames = [...names];
        lastCardSize = cardSize;
        grid.innerHTML = "";

        const frag = document.createDocumentFragment();
        names.forEach((name, idx0) => {
            const isSelected = idx0 === idx;

            const cell = document.createElement("div");
            cell.style.cssText = `display:flex;flex-direction:column;cursor:pointer;`;
            cell.dataset.ioImgCard = "1";
            cell.dataset.ioImgIndex0 = String(idx0);

            cell.onclick = (e) => {
                e.preventDefault();
                setIndex(node, idx0);
                redraw(false); // 增量更新
            };

            const card = document.createElement("div");
            card.style.cssText = `position:relative;border-radius:6px;border:2px solid ${
                isSelected ? "#4a6" : "var(--border-color)"
            };background:var(--comfy-menu-bg);padding:2px;display:flex;flex-direction:column;gap:2px;min-height:${Math.max(
                100,
                Math.floor(cardSize)
            )}px;max-height:${Math.max(100, Math.floor(cardSize))}px;overflow:hidden;`;

            const img = document.createElement("img");
            img.src = getViewUrl(name, cardSize);
            img.style.cssText = "width:100%;aspect-ratio:1/1;object-fit:cover;display:block;background:transparent;border-radius:4px;min-height:0;";
            img.draggable = false;

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
                const all = parseNameList(getImageListWidget(node)?.value);
                const next = all.slice(0, idx0).concat(all.slice(idx0 + 1));
                setNameList(node, next);
                if (next.length === 0) setIndex(node, 0);
                else setIndex(node, Math.max(0, Math.min(next.length - 1, idx)));
                redraw(true); // 强制完整重绘
            };

            const label = document.createElement("div");
            label.textContent = name;
            label.title = name;
            label.style.cssText =
                "font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;opacity:0.9;line-height:1.2;";

            card.appendChild(img);
            card.appendChild(badge);
            card.appendChild(del);
            card.appendChild(label);
            cell.appendChild(card);
            frag.appendChild(cell);
        });

        grid.appendChild(frag);
        app.graph.setDirtyCanvas(true);
    };

    const dragState = {
        active: false,
        pointerId: null,
        startX: 0,
        startY: 0,
        fromIndex0: 0,
        toIndex0: null,
        startName: "",
        lastApplyAt: 0,
        holdTimer: null,
    };

    const clearHoldTimer = () => {
        if (dragState.holdTimer != null) {
            clearTimeout(dragState.holdTimer);
            dragState.holdTimer = null;
        }
    };

    const _targetIndexFromPoint = (x, y) => {
        const el = document.elementFromPoint(x, y);
        const card = el?.closest?.("[data-io-img-card]");
        if (!card) return null;
        const v = Number(card.dataset.ioImgIndex0);
        return Number.isFinite(v) ? Math.floor(v) : null;
    };

    const _swap = (from0, to0) => {
        const all = parseNameList(getImageListWidget(node)?.value);
        if (from0 < 0 || from0 >= all.length) return;
        if (to0 < 0) to0 = 0;
        if (to0 >= all.length) to0 = all.length - 1;
        if (from0 === to0) return;
        const next = all.slice();
        const t = next[from0];
        next[from0] = next[to0];
        next[to0] = t;
        setNameList(node, next);
        setIndex(node, to0);
        redraw(true); // 强制完整重绘
    };

    const _beginDrag = (e, idx0) => {
        dragState.active = true;
        dragState.pointerId = e.pointerId;
        dragState.fromIndex0 = idx0;
        dragState.toIndex0 = null;
        {
            const all = parseNameList(getImageListWidget(node)?.value);
            dragState.startName = idx0 >= 0 && idx0 < all.length ? String(all[idx0] ?? "") : "";
        }
        setIndex(node, idx0);
        redraw(false); // 增量更新选中状态
        try {
            grid.setPointerCapture?.(e.pointerId);
        } catch {}
    };

    const _endDrag = () => {
        dragState.active = false;
        dragState.pointerId = null;
        dragState.toIndex0 = null;
        clearHoldTimer();
    };

    grid.addEventListener(
        "pointerdown",
        (e) => {
            if (e.button != null && e.button !== 0) return;
            if (e.target?.closest?.("button")) return;
            const card = e.target?.closest?.("[data-io-img-card]");
            if (!card) return;
            const idx0 = Number(card.dataset.ioImgIndex0);
            if (!Number.isFinite(idx0)) return;

            e.preventDefault();
            e.stopPropagation();

            dragState.startX = e.clientX;
            dragState.startY = e.clientY;
            dragState.pointerId = e.pointerId;
            dragState.fromIndex0 = Math.floor(idx0);
            dragState.toIndex0 = null;

            clearHoldTimer();
            dragState.holdTimer = setTimeout(() => {
                if (dragState.active) return;
                _beginDrag(e, dragState.fromIndex0);
            }, 220);
        },
        { capture: true }
    );

    grid.addEventListener(
        "pointermove",
        (e) => {
            if (dragState.pointerId == null || e.pointerId !== dragState.pointerId) return;
            const dx = e.clientX - dragState.startX;
            const dy = e.clientY - dragState.startY;
            const moved = Math.abs(dx) + Math.abs(dy) > 6;

            if (!dragState.active) {
                if (!moved) return;
                clearHoldTimer();
                _beginDrag(e, dragState.fromIndex0);
            }

            e.preventDefault();
            e.stopPropagation();

            const now = performance.now();
            if (now - dragState.lastApplyAt < 50) return;
            dragState.lastApplyAt = now;

            const to0 = _targetIndexFromPoint(e.clientX, e.clientY);
            if (to0 == null) return;
            if (to0 === dragState.fromIndex0) return;
            dragState.toIndex0 = to0;
            setIndex(node, to0);
            redraw(false); // 增量更新选中状态
        },
        { capture: true }
    );

    const _onUpOrCancel = async (e) => {
        if (dragState.pointerId == null) return;
        if (e && e.pointerId != null && e.pointerId !== dragState.pointerId) return;
        if (dragState.active) {
            try {
                grid.releasePointerCapture?.(dragState.pointerId);
            } catch {}

            const x = e?.clientX;
            const y = e?.clientY;
            if (typeof x === "number" && typeof y === "number") {
                if (_isInsideContainerPoint(grid, x, y)) {
                    if (dragState.toIndex0 != null) {
                        _swap(dragState.fromIndex0, dragState.toIndex0);
                    } else {
                        setIndex(node, dragState.fromIndex0);
                        redraw(false); // 增量更新
                    }
                } else {
                    const text = _sanitizeSingleLineText(dragState.startName);
                    void _copyToClipboard(text);
                    try {
                        await _copyImageToClipboard(dragState.startName);
                    } catch {}
                    
                    // Try to paste into input element at cursor position
                    const el = document.elementFromPoint(x, y);
                    const target = el?.closest?.("textarea,input,[contenteditable='true']") || el;
                    if (target) {
                        _insertIntoInput(target, text);
                    }
                }
            }
        }
        _endDrag();
    };

    grid.addEventListener("pointerup", _onUpOrCancel, { capture: true });
    grid.addEventListener("pointercancel", _onUpOrCancel, { capture: true });

    singleBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.multiple = true;
        input.style.display = "none";
        document.body.appendChild(input);

        input.onchange = async (e) => {
            try {
                const files = Array.from(e?.target?.files || []);
                if (files.length === 0) return;
                const uploaded = await uploadFilesSequential(files);
                if (uploaded.length === 0) return;
                const all = parseNameList(getImageListWidget(node)?.value);
                all.push(...uploaded);
                setNameList(node, all);
                setIndex(node, all.length);
                redraw(true); // 强制完整重绘
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
        input.accept = "image/*";
        input.multiple = true;
        input.webkitdirectory = true;
        input.directory = true;
        input.style.display = "none";
        document.body.appendChild(input);

        input.onchange = async (e) => {
            try {
                let files = Array.from(e.target.files || []);
                files = files.filter((f) => String(f?.type || "").startsWith("image/"));
                files.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
                const uploaded = await uploadFilesSequential(files);
                if (uploaded.length === 0) return;
                const all = parseNameList(getImageListWidget(node)?.value);
                all.push(...uploaded);
                setNameList(node, all);
                setIndex(node, all.length);
                redraw(true); // 强制完整重绘
            } finally {
                try {
                    document.body.removeChild(input);
                } catch {}
            }
        };

        input.click();
    };

    clearBtn.onclick = () => {
        setNameList(node, []);
        setIndex(node, 0);
        redraw(true); // 强制完整重绘
    };

    hideBtn.onclick = () => {
        previewsHidden = !previewsHidden;
        hideBtn.textContent = previewsHidden ? "显示" : "隐藏";
        redraw(); // 隐藏/显示可以增量更新
    };

    mainContent.appendChild(grid);
    mainContent.appendChild(hiddenOverlay);
    container.appendChild(sidebar);
    container.appendChild(mainContent);

    return { container, redraw };
}

app.registerExtension({
    name: "IO_LoadImgBatch.Extension",
    async setup() {
        api.addEventListener("IO_LoadImgBatch_set", (event) => {
            const nodeId = parseInt(event.detail.node);
            const nodes = app.graph?._nodes || app.graph?.nodes || [];
            const node = nodes.find((n) => n.id === nodeId);
            if (!node) return;

            const itemsRaw = event.detail.items;
            const items = Array.isArray(itemsRaw) ? itemsRaw : [];
            const wList = getImageListWidget(node);
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

            const wIndex = getIndexWidget(node);
            if (wIndex) {
                const idx = Number(event.detail.index);
                const v = Number.isFinite(idx) ? Math.floor(idx) : wIndex.value;
                if (wIndex.value !== v) {
                    wIndex.value = v;
                    wIndex.callback?.(wIndex.value);
                }
            }

            const wSize = getCardSizeWidget(node);
            if (wSize) {
                const sz = Number(event.detail.card_size);
                const v = Number.isFinite(sz) ? Math.floor(sz) : wSize.value;
                if (wSize.value !== v) {
                    wSize.value = v;
                    wSize.callback?.(wSize.value);
                }
            }

            node._ioLoadImgBatchUI?.redraw?.(true); // 后端推送数据需要完整重绘
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_LoadImgBatch") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const listWidget = getImageListWidget(this);
            if (listWidget) {
                listWidget.type = "textarea";
                listWidget.computeSize = () => [300, 48];
            }
            const sizeWidget = getCardSizeWidget(this);
            if (sizeWidget) {
                sizeWidget.type = "number";
                sizeWidget.computeSize = () => [120, 24];
            }

            const ui = createImgBatchUI(this);
            this._ioLoadImgBatchUI = ui;

            const minW = 420;
            const minH = 360;
            if (!this.size || this.size[0] < minW || this.size[1] < minH) {
                this.setSize([Math.max(this.size?.[0] || 0, minW), Math.max(this.size?.[1] || 0, minH)]);
            }
            // Set minimum node size to prevent button overflow
            this.minWidth = Math.max(this.minWidth || 0, 420);
            this.minHeight = Math.max(this.minHeight || 0, 360);

            this.addDOMWidget("io_load_img_batch", "customwidget", ui.container);

            const wIndex = getIndexWidget(this);
            const wList = getImageListWidget(this);
            const wSize = getCardSizeWidget(this);

            // index 变化只需增量更新
            if (wIndex) {
                const origCallback = wIndex.callback;
                let lastValue = wIndex.value;
                wIndex.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw(false); // 增量更新
                };
            }

            // list 变化需要完整重绘
            if (wList) {
                const origCallback = wList.callback;
                let lastValue = wList.value;
                wList.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw(true); // 强制完整重绘
                };
            }

            // size 变化需要完整重绘
            if (wSize) {
                const origCallback = wSize.callback;
                let lastValue = wSize.value;
                wSize.callback = function (value) {
                    origCallback?.call(this, value);
                    if (value === lastValue) return;
                    lastValue = value;
                    ui.redraw(true); // 强制完整重绘
                };
            }

            ui.redraw(true); // 首次创建需要完整重绘
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure?.apply(this, arguments);
            const listWidget = getImageListWidget(this);
            if (listWidget) {
                listWidget.type = "textarea";
                listWidget.computeSize = () => [300, 48];
            }
            const sizeWidget = getCardSizeWidget(this);
            if (sizeWidget) {
                sizeWidget.type = "number";
                sizeWidget.computeSize = () => [120, 24];
            }
            this._ioLoadImgBatchUI?.redraw?.(true); // 从工作流加载需要完整重绘
            return r;
        };
    },
});

