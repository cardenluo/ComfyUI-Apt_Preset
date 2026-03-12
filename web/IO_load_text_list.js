import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getWidgetByName(node, name) {
    return (node?.widgets || []).find((w) => w?.name === name) ?? null;
}

function getTextListWidget(node) {
    return getWidgetByName(node, "text_list");
}

function getCardSizeWidget(node) {
    return getWidgetByName(node, "card_size");
}

function getSplitRuleWidget(node) {
    return getWidgetByName(node, "file_split_rule");
}

function getIndexWidget(node) {
    return getWidgetByName(node, "index");
}

function safeJsonParse(raw, fallback) {
    try {
        return JSON.parse(raw);
    } catch {
        return fallback;
    }
}

function normalizeNewlines(s) {
    return String(s ?? "").replace(/\r\n/g, "\n").replace(/\r/g, "\n");
}

function parseTextItems(raw) {
    const s = String(raw ?? "");
    const trimmed = s.trim();
    if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
        const v = safeJsonParse(s, null);
        if (Array.isArray(v)) return v.map((x) => (x == null ? "" : String(x)));
    }
    return normalizeNewlines(s)
        .split("\n")
        .map((x) => x.trim())
        .filter((x) => x.length > 0);
}

function getTextItems(node) {
    const w = getTextListWidget(node);
    return parseTextItems(w?.value ?? "[]");
}

function setTextItems(node, items) {
    const w = getTextListWidget(node);
    if (!w) return;
    const arr = Array.isArray(items) ? items.map((x) => (x == null ? "" : String(x))) : [];
    w.value = JSON.stringify(arr);
    w.callback?.(w.value);
}

function unescapeRule(rule) {
    const s = String(rule ?? "");
    return s.replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\r/g, "\r");
}

function splitByRule(text, ruleRaw) {
    const rule = unescapeRule(ruleRaw);
    const t = normalizeNewlines(text);
    if (!rule) {
        return t
            .split("\n")
            .map((x) => x.trim())
            .filter((x) => x.length > 0);
    }
    return t
        .split(rule)
        .map((x) => x.trim())
        .filter((x) => x.length > 0);
}

function clampIndex1Based(value, total) {
    const t = Number(total);
    if (!Number.isFinite(t) || t <= 0) return 0;
    const v = Number(value);
    if (!Number.isFinite(v)) return 0;
    const i = Math.floor(v);
    if (i < 1) return 0;
    if (i > t) return 0;
    return i;
}

function getCardSize(node) {
    const w = getCardSizeWidget(node);
    const v = Number(w?.value);
    return Number.isFinite(v) && v > 0 ? v : 100;
}

function setCardSize(node, size) {
    const w = getCardSizeWidget(node);
    if (!w) return;
    const v = Number(size);
    w.value = Number.isFinite(v) ? v : 100;
    w.callback?.(w.value);
}

function getSplitRule(node) {
    const w = getSplitRuleWidget(node);
    return String(w?.value ?? "");
}

function setSplitRule(node, rule) {
    const w = getSplitRuleWidget(node);
    if (!w) return;
    w.value = String(rule ?? "");
    w.callback?.(w.value);
}

function getIndex1Based(node) {
    const w = getIndexWidget(node);
    const v = Number(w?.value);
    return Number.isFinite(v) ? Math.floor(v) : 0;
}

function setIndex1Based(node, idx) {
    const w = getIndexWidget(node);
    if (!w) return;
    const v = Number(idx);
    w.value = Number.isFinite(v) ? Math.floor(v) : 0;
    w.callback?.(w.value);
}

function isFilesDragEvent(e) {
    const dt = e?.dataTransfer;
    if (!dt) return false;
    const types = Array.from(dt.types || []);
    return types.includes("Files");
}

let _ioTextListDndGuardInstalled = false;
let _ioTextListExternalTextDropInstalled = false;
let _ioTextListActiveDragText = null;

function _isEditableElement(el) {
    if (!el) return false;
    const tag = String(el.tagName || "").toUpperCase();
    if (tag === "TEXTAREA") return true;
    if (tag === "INPUT") {
        const type = String(el.type || "").toLowerCase();
        return ["text", "search", "url", "email", "password", "number"].includes(type);
    }
    return !!el.isContentEditable;
}

function _findEditableElement(el) {
    let cur = el;
    let guard = 0;
    while (cur && guard++ < 12) {
        if (_isEditableElement(cur)) return cur;
        cur = cur.parentElement;
    }
    return null;
}

function _insertTextIntoEditable(el, text) {
    const s = String(text ?? "");
    if (!s) return;

    const tag = String(el.tagName || "").toUpperCase();
    if (tag === "TEXTAREA" || tag === "INPUT") {
        const start = typeof el.selectionStart === "number" ? el.selectionStart : el.value?.length ?? 0;
        const end = typeof el.selectionEnd === "number" ? el.selectionEnd : start;
        const prev = String(el.value ?? "");
        const next = prev.slice(0, start) + s + prev.slice(end);
        el.value = next;
        try {
            el.selectionStart = el.selectionEnd = start + s.length;
        } catch {}
        try {
            el.dispatchEvent(new Event("input", { bubbles: true }));
        } catch {}
        return;
    }

    try {
        el.focus?.();
    } catch {}
    try {
        document.execCommand?.("insertText", false, s);
        return;
    } catch {}
    try {
        const sel = window.getSelection?.();
        if (!sel || sel.rangeCount === 0) return;
        const range = sel.getRangeAt(0);
        range.deleteContents();
        range.insertNode(document.createTextNode(s));
        range.collapse(false);
        sel.removeAllRanges();
        sel.addRange(range);
        el.dispatchEvent?.(new Event("input", { bubbles: true }));
    } catch {}
}

function ensureExternalTextDropOnce() {
    if (_ioTextListExternalTextDropInstalled) return;
    _ioTextListExternalTextDropInstalled = true;

    const isInsideOurUi = (target) => {
        try {
            return !!target?.closest?.('[data-io-loadtextlist="1"]');
        } catch {
            return false;
        }
    };

    window.addEventListener(
        "dragover",
        (e) => {
            if (!_ioTextListActiveDragText) return;
            if (isFilesDragEvent(e)) return;
            const target = e.target;
            if (isInsideOurUi(target)) return;
            const editable = _findEditableElement(target);
            if (!editable) return;
            e.preventDefault();
            try {
                e.dataTransfer.dropEffect = "copy";
            } catch {}
        },
        { capture: true }
    );

    window.addEventListener(
        "drop",
        (e) => {
            if (!_ioTextListActiveDragText) return;
            if (isFilesDragEvent(e)) return;
            const target = e.target;
            if (isInsideOurUi(target)) return;
            const editable = _findEditableElement(target);
            if (!editable) return;
            e.preventDefault();
            e.stopPropagation();
            _insertTextIntoEditable(editable, _ioTextListActiveDragText);
        },
        { capture: true }
    );
}

function ensureGlobalDragDropPrevention() {
    if (_ioTextListDndGuardInstalled) return;
    _ioTextListDndGuardInstalled = true;
    window.addEventListener(
        "dragover",
        (e) => {
            if (!isFilesDragEvent(e)) return;
            e.preventDefault();
        },
        { capture: true }
    );
    window.addEventListener(
        "drop",
        (e) => {
            if (!isFilesDragEvent(e)) return;
            e.preventDefault();
        },
        { capture: true }
    );
}

function openEditorModal({ initialText, title, onSave }) {
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;";

    const panel = document.createElement("div");
    panel.style.cssText =
        "width:min(860px,90vw);height:min(560px,80vh);background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:10px;display:flex;flex-direction:column;gap:8px;padding:10px;";

    const head = document.createElement("div");
    head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";

    const t = document.createElement("div");
    t.textContent = title || "编辑";
    t.style.cssText = "font-size:13px;opacity:0.9;";

    const btnRow = document.createElement("div");
    btnRow.style.cssText = "display:flex;gap:8px;align-items:center;";

    const mkBtn = (label) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
            "padding:6px 10px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:6px;cursor:pointer;font-size:12px;";
        return b;
    };

    const cancelBtn = mkBtn("取消");
    const saveBtn = mkBtn("保存");
    btnRow.appendChild(cancelBtn);
    btnRow.appendChild(saveBtn);

    head.appendChild(t);
    head.appendChild(btnRow);

    const ta = document.createElement("textarea");
    ta.value = String(initialText ?? "");
    ta.style.cssText =
        "flex:1;width:100%;resize:none;padding:10px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:8px;line-height:1.4;";

    const close = () => {
        try {
            document.body.removeChild(overlay);
        } catch {}
    };

    cancelBtn.onclick = (e) => {
        e.preventDefault();
        close();
    };
    overlay.addEventListener("click", (e) => {
        if (e.target === overlay) close();
    });
    saveBtn.onclick = (e) => {
        e.preventDefault();
        onSave?.(ta.value);
        close();
    };

    panel.appendChild(head);
    panel.appendChild(ta);
    overlay.appendChild(panel);
    document.body.appendChild(overlay);
    requestAnimationFrame(() => ta.focus());
}

function createTextListUI(node) {
    const wList = getTextListWidget(node);
    if (!wList) return null;

    const container = document.createElement("div");
    container.dataset.ioLoadtextlist = "1";
    container.style.cssText =
        "width:100%;padding:8px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:10px 0 6px 0;pointer-events:none;display:flex;flex-direction:row;gap:8px;";

    const sidebar = document.createElement("div");
    sidebar.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:56px;width:56px;pointer-events:auto;";

    const mkBtn = (label) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
            "padding:4px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;cursor:pointer;font-size:10px;width:100%;text-align:center;";
        return b;
    };

    const addBtn = mkBtn("新增");
    const importOneBtn = mkBtn("单txt");
    const importFolderBtn = mkBtn("文件夹");
    const sortBtn = mkBtn("顺序");
    const clearBtn = mkBtn("清空");
    const hideBtn = mkBtn("隐藏");
    const sizeBtn = mkBtn("尺寸");

    sidebar.appendChild(addBtn);
    sidebar.appendChild(importOneBtn);
    sidebar.appendChild(importFolderBtn);
    sidebar.appendChild(sortBtn);
    sidebar.appendChild(clearBtn);
    sidebar.appendChild(hideBtn);
    sidebar.appendChild(sizeBtn);

    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;gap:6px;pointer-events:auto;";

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--card-size,100px),1fr));gap:6px;flex:1;overflow-y:auto;background:var(--comfy-input-bg);padding:6px;border-radius:4px;";

    const hiddenOverlay = document.createElement("div");
    hiddenOverlay.style.cssText =
        "flex:1;display:none;align-items:center;justify-content:center;background:var(--comfy-input-bg);border-radius:4px;color:var(--input-text);font-size:12px;opacity:0.75;border:1px solid var(--border-color);";

    let previewsHidden = false;
    let nextSortIsAsc = true;
    let dragging = false;
    const DRAG_INDEX_MIME = "application/x-io-loadtextlist-index";

    const readTxtFile = (file) =>
        new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onerror = () => reject(new Error("read file error"));
            reader.onload = () => resolve(String(reader.result ?? ""));
            reader.readAsText(file);
        });

    const pickTxtFiles = ({ multiple = false, directory = false } = {}) =>
        new Promise((resolve) => {
            const input = document.createElement("input");
            input.type = "file";
            input.accept = ".txt,text/plain";
            input.multiple = !!multiple;
            if (directory) {
                input.webkitdirectory = true;
                input.directory = true;
                input.multiple = true;
            }
            input.onchange = () => resolve(Array.from(input.files || []));
            input.click();
        });

    const syncSizeUIFromWidget = () => {
        // 不显示尺寸数值，避免被填充到错误的地方
    };

    const increaseCardSize = () => {
        const currentSize = getCardSize(node);
        let newSize = currentSize + 40;
        if (newSize > 400) {
            newSize = 100;
        }
        setCardSize(node, newSize);
        redraw();
    };

    sizeBtn.onclick = increaseCardSize;
    syncSizeUIFromWidget();

    const redraw = () => {
        const items = getTextItems(node);
        const total = items.length;
        const cardSize = getCardSize(node);
        grid.style.setProperty("--card-size", `${cardSize}px`);

        if (previewsHidden) {
            grid.style.display = "none";
            hiddenOverlay.style.display = "flex";
            hiddenOverlay.textContent = `预览已隐藏（${total}）`;
            app.graph.setDirtyCanvas(true);
            return;
        }

        grid.style.display = "grid";
        hiddenOverlay.style.display = "none";
        grid.innerHTML = "";

        const lines = Math.max(2, Math.min(24, Math.floor((cardSize - 44) / 18)));

        const frag = document.createDocumentFragment();
        for (let i = 0; i < items.length; i++) {
            const text = String(items[i] ?? "");

            const card = document.createElement("div");
            card.draggable = true;
            const isSel = i + 1 === getIndex1Based(node);
            card.style.cssText =
                `position:relative;height:${cardSize}px;display:flex;flex-direction:column;gap:6px;padding:6px;background:var(--comfy-menu-bg);border:2px solid ${isSel ? "#fff" : "var(--border-color)"};border-radius:6px;user-select:none;cursor:pointer;`;

            card.addEventListener("click", (e) => {
                if (dragging) return;
                e.preventDefault();
                setIndex1Based(node, i + 1);
                redraw();
            });

            card.addEventListener("dragstart", (e) => {
                dragging = true;
                _ioTextListActiveDragText = text;
                e.dataTransfer?.setData?.(DRAG_INDEX_MIME, String(i));
                e.dataTransfer?.setData?.("text/plain", text);
                e.dataTransfer?.setData?.("text", text);
                e.dataTransfer?.setData?.("Text", text);
                e.dataTransfer && (e.dataTransfer.effectAllowed = "copyMove");
                card.style.opacity = "0.6";
            });

            card.addEventListener("dragend", () => {
                dragging = false;
                _ioTextListActiveDragText = null;
                card.style.opacity = "";
            });

            card.addEventListener("dragover", (e) => {
                e.preventDefault();
                e.dataTransfer && (e.dataTransfer.dropEffect = "move");
            });

            card.addEventListener("drop", (e) => {
                e.preventDefault();
                const fromRaw = e.dataTransfer?.getData?.(DRAG_INDEX_MIME);
                const from = Number(fromRaw);
                const to = i;
                if (!Number.isInteger(from) || from < 0 || from >= items.length) return;
                if (from === to) return;
                const next = items.slice();
                const [moved] = next.splice(from, 1);
                next.splice(to, 0, moved);
                setTextItems(node, next);
                redraw();
            });

            card.addEventListener("dblclick", (e) => {
                if (dragging) return;
                e.preventDefault();
                openEditorModal({
                    initialText: text,
                    title: `编辑 #${i + 1}`,
                    onSave: (nextText) => {
                        const next = getTextItems(node);
                        if (i < 0 || i >= next.length) return;
                        next[i] = String(nextText ?? "");
                        setTextItems(node, next);
                        redraw();
                    },
                });
            });

            const seq = document.createElement("div");
            seq.textContent = `#${i + 1}`;
            seq.style.cssText = "font-size:12px;opacity:0.85;white-space:nowrap;";

            const del = document.createElement("button");
            del.textContent = "×";
            del.title = "删除";
            del.style.cssText =
                "position:absolute;top:6px;right:6px;width:20px;height:20px;background:rgba(255,0,0,0.75);color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:16px;line-height:1;z-index:3;";
            del.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                const next = items.slice(0, i).concat(items.slice(i + 1));
                setTextItems(node, next);
                const idx = getIndex1Based(node);
                if (idx > next.length) setIndex1Based(node, 0);
                redraw();
            });

            const preview = document.createElement("div");
            preview.textContent = text;
            preview.style.cssText =
                `flex:1;padding:8px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;overflow:hidden;white-space:pre-wrap;word-break:break-word;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:${lines};line-height:1.35;`;

            card.appendChild(del);
            card.appendChild(seq);
            card.appendChild(preview);
            frag.appendChild(card);
        }

        grid.appendChild(frag);
        app.graph.setDirtyCanvas(true);
    };

    addBtn.onclick = (e) => {
        e.preventDefault();
        const next = getTextItems(node);
        next.push("");
        setTextItems(node, next);
        redraw();
        openEditorModal({
            initialText: "",
            title: `编辑 #${next.length}`,
            onSave: (txt) => {
                const cur = getTextItems(node);
                cur[cur.length - 1] = String(txt ?? "");
                setTextItems(node, cur);
                redraw();
            },
        });
    };

    importOneBtn.onclick = async (e) => {
        e.preventDefault();
        const fs = await pickTxtFiles({ multiple: false, directory: false });
        const f = fs?.[0];
        if (!f) return;
        const text = await readTxtFile(f);
        const parts = splitByRule(text, getSplitRule(node));
        if (parts.length === 0) return;
        const cur = getTextItems(node);
        cur.push(...parts);
        setTextItems(node, cur);
        redraw();
    };

    importFolderBtn.onclick = async (e) => {
        e.preventDefault();
        let fs = await pickTxtFiles({ multiple: true, directory: true });
        fs = Array.from(fs || []).filter((f) => (f?.name || "").toLowerCase().endsWith(".txt"));
        if (fs.length === 0) return;
        fs.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
        const rule = getSplitRule(node);
        const cur = getTextItems(node);
        for (const f of fs) {
            const text = await readTxtFile(f);
            const parts = splitByRule(text, rule);
            if (parts.length > 0) cur.push(...parts);
        }
        setTextItems(node, cur);
        redraw();
    };

    sortBtn.onclick = (e) => {
        e.preventDefault();
        const cur = getTextItems(node);
        const next = cur.slice().sort((a, b) => (nextSortIsAsc ? a.localeCompare(b) : b.localeCompare(a)));
        nextSortIsAsc = !nextSortIsAsc;
        sortBtn.textContent = nextSortIsAsc ? "顺序" : "逆序";
        setTextItems(node, next);
        redraw();
    };

    clearBtn.onclick = (e) => {
        e.preventDefault();
        setTextItems(node, []);
        setIndex1Based(node, 0);
        redraw();
    };

    hideBtn.onclick = (e) => {
        e.preventDefault();
        previewsHidden = !previewsHidden;
        hideBtn.textContent = previewsHidden ? "显示" : "隐藏";
        redraw();
    };

    mainContent.addEventListener("dragover", (e) => {
        if (!isFilesDragEvent(e)) return;
        e.preventDefault();
        e.stopPropagation();
        container.style.border = "2px dashed #4a6";
    });

    mainContent.addEventListener("dragleave", (e) => {
        if (!isFilesDragEvent(e)) return;
        container.style.border = "1px solid var(--border-color)";
    });

    mainContent.addEventListener("drop", async (e) => {
        if (!isFilesDragEvent(e)) return;
        e.preventDefault();
        e.stopPropagation();
        container.style.border = "1px solid var(--border-color)";
        let fs = Array.from(e.dataTransfer?.files || []);
        fs = fs.filter((f) => (f?.name || "").toLowerCase().endsWith(".txt"));
        if (fs.length === 0) return;
        fs.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
        const rule = getSplitRule(node);
        const cur = getTextItems(node);
        for (const f of fs) {
            const text = await readTxtFile(f);
            const parts = splitByRule(text, rule);
            if (parts.length > 0) cur.push(...parts);
        }
        setTextItems(node, cur);
        redraw();
    });

    mainContent.appendChild(grid);
    mainContent.appendChild(hiddenOverlay);
    container.appendChild(sidebar);
    container.appendChild(mainContent);

    redraw();
    return { container, redraw };
}

app.registerExtension({
    name: "IO_LoadTextList.Extension",
    async setup() {
        api.addEventListener("IO_LoadTextList_append", function (event) {
            const nodeId = parseInt(event.detail.node);
            const node = app.graph.nodes.find((n) => n.id === nodeId);
            if (!node) return;
            const itemsRaw = event.detail.items;
            const items = Array.isArray(itemsRaw) ? itemsRaw : itemsRaw == null ? [] : [itemsRaw];
            if (items.length === 0) return;
            const cur = getTextItems(node);
            cur.push(...items.map((x) => (x == null ? "" : String(x))));
            setTextItems(node, cur);
            node._ioLoadTextListUI?.redraw?.();
        });

        api.addEventListener("IO_LoadTextList_set_index", function (event) {
            const nodeId = parseInt(event.detail.node);
            const node = app.graph.nodes.find((n) => n.id === nodeId);
            if (!node) return;
            const idx = Number(event.detail.index);
            setIndex1Based(node, Number.isFinite(idx) ? idx : 0);
            node._ioLoadTextListUI?.redraw?.();
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_LoadTextList") return;
        ensureGlobalDragDropPrevention();
        ensureExternalTextDropOnce();

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const wList = getTextListWidget(this);
            if (wList) {
                wList.type = "hidden";
                wList.computeSize = () => [0, -4];
                if (!wList.value || String(wList.value).trim() === "") {
                    wList.value = "[]";
                    wList.callback?.(wList.value);
                }
            }

            const wSize = getCardSizeWidget(this);
            if (wSize) {
                wSize.type = "hidden";
                wSize.computeSize = () => [0, -4];
                // 只在值为 NaN、null 或 undefined 时才设置默认值
                const v = Number(wSize.value);
                if (wSize.value == null || !Number.isFinite(v) || v <= 0) {
                    wSize.value = 100;
                    wSize.callback?.(wSize.value);
                }
            }

            const wRule = getSplitRuleWidget(this);
            if (wRule) {
                // 不隐藏 widget，保持可见
                // 不强制设置默认值，保留从服务器传来的值
                if (wRule.value == null) {
                    wRule.value = "";
                    wRule.callback?.(wRule.value);
                }
            }

            const wIndex = getIndexWidget(this);
            if (wIndex) {
                // 只在值为 NaN、null 或 undefined 时才设置默认值
                const v = Number(wIndex.value);
                if (wIndex.value == null || !Number.isFinite(v) || v < 0) {
                    wIndex.value = 1;
                    wIndex.callback?.(wIndex.value);
                }
                const prevCb = wIndex.callback;
                const node = this;
                wIndex.callback = function () {
                    const r = prevCb?.apply(this, arguments);
                    node._ioLoadTextListUI?.redraw?.();
                    return r;
                };
            }

            const ui = createTextListUI(this);
            if (ui) {
                this._ioLoadTextListUI = ui;
                this.addDOMWidget("io_load_text_list", "customwidget", ui.container);
                const minW = 520;
                const minH = 420;
                if (!this.size || this.size[0] < minW || this.size[1] < minH) {
                    this.setSize([Math.max(this.size?.[0] || 0, minW), Math.max(this.size?.[1] || 0, minH)]);
                }
            }

            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = origOnConfigure?.apply(this, arguments);

            // 确保 widget 的值正确设置（只在无效时才设置默认值）
            const wSize = getCardSizeWidget(this);
            if (wSize) {
                const v = Number(wSize.value);
                // 只在值为 NaN、null 或 undefined 时才设置默认值
                if (wSize.value == null || !Number.isFinite(v) || v <= 0) {
                    wSize.value = 100;
                    wSize.callback?.(wSize.value);
                }
                // 否则保留用户的设置值
            }

            const wRule = getSplitRuleWidget(this);
            if (wRule) {
                const v = wRule.value;
                wRule.value = String(v ?? "");
                wRule.callback?.(wRule.value);
            }

            const wIndex = getIndexWidget(this);
            if (wIndex) {
                const v = Number(wIndex.value);
                // 只在值为 NaN、null 或 undefined 时才设置默认值
                if (wIndex.value == null || !Number.isFinite(v) || v < 0) {
                    wIndex.value = 1;
                    wIndex.callback?.(wIndex.value);
                }
            }

            this._ioLoadTextListUI?.redraw?.();
            return r;
        };
    },
});

