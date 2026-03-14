import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function getWidgetByName(node, name) {
    return node?.widgets?.find((w) => w.name === name);
}

function getCameraDataWidget(node) {
    return getWidgetByName(node, "camera_data");
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

function getTextListWidget(node) {
    return getWidgetByName(node, "text_list");
}

function getCardSizeWidget(node) {
    return getWidgetByName(node, "card_size");
}

function getIndexWidget(node) {
    return getWidgetByName(node, "index");
}

function parseTextList(text) {
    return (text || "")
        .split("\n")
        .map((s) => String(s || "").replace(/\r/g, "").trim())
        .filter((s) => s !== "");
}

function setTextList(node, lines) {
    const w = getTextListWidget(node);
    if (!w) return;
    const next = Array.isArray(lines) ? lines : [];
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

function openEditorModal({ initialText, title, onSave }) {
    const overlay = document.createElement("div");
    overlay.style.cssText =
        "position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;";

    const panel = document.createElement("div");
    panel.style.cssText =
        "width:min(860px,90vw);height:min(520px,80vh);background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:10px;display:flex;flex-direction:column;gap:8px;padding:10px;";

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

function _sanitizeSingleLineText(raw) {
    const s = String(raw ?? "")
        .replace(/\r/g, "")
        .replace(/\n/g, " ")
        .trim();
    return s;
}

function _linesFromText(raw) {
    return String(raw ?? "")
        .replace(/\r/g, "")
        .split("\n")
        .map((x) => String(x ?? "").trim())
        .filter((x) => x !== "");
}

async function _readTxtFiles(files) {
    const out = [];
    const arr = Array.isArray(files) ? files : [];
    for (const f of arr) {
        if (!f) continue;
        const name = String(f.name || "").toLowerCase();
        if (!name.endsWith(".txt")) continue;
        const txt = await f.text();
        out.push(..._linesFromText(txt));
    }
    return out;
}

function createTextListUI(node) {
    const container = document.createElement("div");
    container.style.cssText =
        "width:100%;padding:8px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:5px 0;pointer-events:none;display:flex;flex-direction:row;gap:8px;";
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

    const newBtn = mkBtn("新建");
    const singleTxtBtn = mkBtn("单txt");
    const folderTxtBtn = mkBtn("📂");
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

    sidebar.appendChild(newBtn);
    sidebar.appendChild(singleTxtBtn);
    sidebar.appendChild(folderTxtBtn);
    sidebar.appendChild(sortBtn);
    sidebar.appendChild(deleteBtn);
    sidebar.appendChild(clearBtn);
    sidebar.appendChild(hideBtn);
    sidebar.appendChild(sizeBtn);
    sidebar.appendChild(sizeSlider);
    sidebar.appendChild(sizeVal);

    let previewsHidden = false;
    let nextSortIsAsc = true;
    const SIZE_PRESETS = [120, 220, 320, 420];

    const syncSizeUIFromWidget = () => {
        const current = getCardSize(node);
        const idx = SIZE_PRESETS.indexOf(current);
        sizeSlider.value = String(idx >= 0 ? idx : 0);
        sizeVal.textContent = String(current);
    };

    const applyCardSizeFromSlider = () => {
        const idx = Math.max(0, Math.min(SIZE_PRESETS.length - 1, Math.floor(Number(sizeSlider.value))));
        const size = SIZE_PRESETS[idx] ?? 120;
        setCardSize(node, size);
        sizeVal.textContent = String(size);
        redraw();
    };

    sizeSlider.max = String(Math.max(0, SIZE_PRESETS.length - 1));
    sizeBtn.onclick = () => applyCardSizeFromSlider();
    sizeSlider.addEventListener("input", applyCardSizeFromSlider);
    syncSizeUIFromWidget();

    sortBtn.onclick = () => {
        const items = parseTextList(getTextListWidget(node)?.value);
        const sorted = [...items].sort((a, b) => (nextSortIsAsc ? a.localeCompare(b) : b.localeCompare(a)));
        setTextList(node, sorted);
        nextSortIsAsc = !nextSortIsAsc;
        sortBtn.textContent = nextSortIsAsc ? "顺序" : "逆序";
        redraw();
    };

    deleteBtn.onclick = () => {
        const items = parseTextList(getTextListWidget(node)?.value);
        if (items.length === 0) return;
        const idx = getIndex(node);
        const idx0 = Math.max(0, Math.min(items.length - 1, idx));
        const next = items.slice(0, idx0).concat(items.slice(idx0 + 1));
        setTextList(node, next);
        if (next.length === 0) {
            setIndex(node, 0);
        } else {
            setIndex(node, Math.max(0, Math.min(next.length - 1, idx)));
        }
        redraw();
    };

    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;pointer-events:auto;";
    mainContent.style.userSelect = "none";
    mainContent.style.webkitUserSelect = "none";

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--card-size,220px),1fr));gap:6px;flex:1;overflow-y:auto;background:var(--comfy-input-bg);padding:6px;border-radius:4px;";
    grid.style.userSelect = "none";
    grid.style.webkitUserSelect = "none";
    grid.style.touchAction = "none";

    const hiddenOverlay = document.createElement("div");
    hiddenOverlay.style.cssText =
        "flex:1;display:none;align-items:center;justify-content:center;background:var(--comfy-input-bg);border-radius:4px;color:var(--input-text);font-size:12px;opacity:0.75;";

    const redraw = () => {
        const items = parseTextList(getTextListWidget(node)?.value);
        grid.innerHTML = "";
        const cardSize = getCardSize(node);
        grid.style.setProperty("--card-size", `${cardSize}px`);

        if (previewsHidden) {
            grid.style.display = "none";
            hiddenOverlay.style.display = "flex";
            hiddenOverlay.textContent = `预览已隐藏（${items.length}）`;
            app.graph.setDirtyCanvas(true);
            return;
        }

        grid.style.display = "grid";
        hiddenOverlay.style.display = "none";

        const frag = document.createDocumentFragment();
        const idx = getIndex(node);
        items.forEach((text, idx0) => {
            const isSelected = idx0 === idx;

            const cell = document.createElement("div");
            cell.style.cssText = `display:flex;flex-direction:column;gap:6px;cursor:pointer;`;
            cell.dataset.ioTextCard = "1";
            cell.dataset.ioTextIndex0 = String(idx0);

            cell.onclick = (e) => {
                e.preventDefault();
                setIndex(node, idx0);
                redraw();
            };

            const card = document.createElement("div");
            card.style.cssText = `position:relative;border-radius:6px;border:2px solid ${
                isSelected ? "#4a6" : "var(--border-color)"
            };background:var(--comfy-menu-bg);padding:8px;display:flex;flex-direction:column;gap:6px;min-height:${Math.max(
                120,
                Math.floor(cardSize)
            )}px;max-height:${Math.max(120, Math.floor(cardSize))}px;overflow:hidden;`;
            card.style.userSelect = "none";
            card.style.webkitUserSelect = "none";

            const head = document.createElement("div");
            head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";

            const badge = document.createElement("div");
            badge.textContent = `#${idx0}`;
            badge.style.cssText =
                "font-size:11px;opacity:0.9;background:var(--comfy-input-bg);border:1px solid var(--border-color);border-radius:999px;padding:2px 8px;max-width:60px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";

            const headBtns = document.createElement("div");
            headBtns.style.cssText = "display:flex;gap:6px;align-items:center;";

            const del = document.createElement("button");
            del.textContent = "×";
            del.title = "删除";
            del.style.cssText =
                "width:20px;height:20px;background:rgba(255,0,0,0.75);color:#fff;border:none;border-radius:4px;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;";
            del.onclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                const all = parseTextList(getTextListWidget(node)?.value);
                const next = all.slice(0, idx0).concat(all.slice(idx0 + 1));
                setTextList(node, next);
                if (next.length === 0) {
                    setIndex(node, 0);
                } else {
                    setIndex(node, Math.max(0, Math.min(next.length - 1, idx)));
                }
                redraw();
            };

            const body = document.createElement("div");
            body.textContent = text;
            body.style.cssText =
                "flex:1;overflow:auto;white-space:pre-wrap;word-break:break-word;font-size:12px;line-height:1.35;opacity:0.95;padding-right:2px;";
            body.style.userSelect = "none";
            body.style.webkitUserSelect = "none";

            headBtns.appendChild(del);
            head.appendChild(badge);
            head.appendChild(headBtns);
            card.appendChild(head);
            card.appendChild(body);
            cell.appendChild(card);
            frag.appendChild(cell);

            card.ondblclick = (e) => {
                e.preventDefault();
                e.stopPropagation();
                openEditorModal({
                    initialText: text,
                    title: `编辑 #${idx0}`,
                    onSave: (v) => {
                        const nextLine = _sanitizeSingleLineText(v);
                        const all = parseTextList(getTextListWidget(node)?.value);
                        if (idx0 < 0 || idx0 >= all.length) return;
                        all[idx0] = nextLine;
                        setTextList(node, all);
                        setIndex(node, idx0);
                        redraw();
                    },
                });
            };
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
        lastApplyAt: 0,
        toIndex0: null,
        startText: "",
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
        const card = el?.closest?.("[data-io-text-card]");
        if (!card) return null;
        const v = Number(card.dataset.ioTextIndex0);
        return Number.isFinite(v) ? Math.floor(v) : null;
    };

    const _swap = (from0, to0) => {
        const all = parseTextList(getTextListWidget(node)?.value);
        if (from0 < 0 || from0 >= all.length) return;
        if (to0 < 0) to0 = 0;
        if (to0 >= all.length) to0 = all.length - 1;
        if (from0 === to0) return;
        const next = all.slice();
        const t = next[from0];
        next[from0] = next[to0];
        next[to0] = t;
        setTextList(node, next);
        setIndex(node, to0);
        redraw();
    };

    const _beginDrag = (e, idx0) => {
        dragState.active = true;
        dragState.pointerId = e.pointerId;
        dragState.fromIndex0 = idx0;
        dragState.toIndex0 = null;
        {
            const all = parseTextList(getTextListWidget(node)?.value);
            dragState.startText = idx0 >= 0 && idx0 < all.length ? String(all[idx0] ?? "") : "";
        }
        setIndex(node, idx0);
        redraw();
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

    const _isInsideContainerPoint = (x, y) => {
        const rect = container.getBoundingClientRect?.();
        if (!rect) return false;
        return x >= rect.left && x <= rect.right && y >= rect.top && y <= rect.bottom;
    };

    const _insertIntoInput = (el, text) => {
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
    };

    const _copyToClipboard = async (text) => {
        const t = String(text ?? "");
        if (t === "") return;
        try {
            await navigator.clipboard.writeText(t);
        } catch {}
    };

    grid.addEventListener(
        "pointerdown",
        (e) => {
            if (e.button != null && e.button !== 0) return;
            if (e.target?.closest?.("button")) return;
            const card = e.target?.closest?.("[data-io-text-card]");
            if (!card) return;
            const idx0 = Number(card.dataset.ioTextIndex0);
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
            redraw();
        },
        { capture: true }
    );

    const _onUpOrCancel = (e) => {
        if (dragState.pointerId == null) return;
        if (e && e.pointerId != null && e.pointerId !== dragState.pointerId) return;
        if (dragState.active) {
            try {
                grid.releasePointerCapture?.(dragState.pointerId);
            } catch {}

            const x = e?.clientX;
            const y = e?.clientY;
            if (typeof x === "number" && typeof y === "number") {
                if (_isInsideContainerPoint(x, y)) {
                    if (dragState.toIndex0 != null) {
                        _swap(dragState.fromIndex0, dragState.toIndex0);
                    } else {
                        setIndex(node, dragState.fromIndex0);
                        redraw();
                    }
                } else {
                    const el = document.elementFromPoint(x, y);
                    const target = el?.closest?.("textarea,input,[contenteditable='true']") || el;
                    const ok = _insertIntoInput(target, dragState.startText);
                    if (!ok) {
                        void _copyToClipboard(dragState.startText);
                    }
                }
            }
        }
        _endDrag();
    };

    grid.addEventListener("pointerup", _onUpOrCancel, { capture: true });
    grid.addEventListener("pointercancel", _onUpOrCancel, { capture: true });

    newBtn.onclick = () => {
        openEditorModal({
            initialText: "",
            title: "新建文本",
            onSave: (v) => {
                const line = _sanitizeSingleLineText(v);
                if (line === "") return;
                const all = parseTextList(getTextListWidget(node)?.value);
                all.push(line);
                setTextList(node, all);
                setIndex(node, all.length - 1);
                redraw();
            },
        });
    };

    singleTxtBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".txt,text/plain";
        input.multiple = false;
        input.style.display = "none";
        document.body.appendChild(input);

        input.onchange = async (e) => {
            try {
                const f = e?.target?.files?.[0];
                if (!f) return;
                const lines = await _readTxtFiles([f]);
                if (lines.length === 0) return;
                const all = parseTextList(getTextListWidget(node)?.value);
                all.push(...lines);
                setTextList(node, all);
                setIndex(node, all.length - 1);
                redraw();
            } finally {
                try {
                    document.body.removeChild(input);
                } catch {}
            }
        };

        input.click();
    };

    folderTxtBtn.onclick = () => {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = ".txt,text/plain";
        input.multiple = true;
        input.webkitdirectory = true;
        input.directory = true;
        input.style.display = "none";
        document.body.appendChild(input);

        input.onchange = async (e) => {
            try {
                let files = Array.from(e.target.files || []);
                files = files.filter((f) => String(f?.name || "").toLowerCase().endsWith(".txt"));
                files.sort((a, b) => (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name));
                const lines = await _readTxtFiles(files);
                if (lines.length === 0) return;
                const all = parseTextList(getTextListWidget(node)?.value);
                all.push(...lines);
                setTextList(node, all);
                setIndex(node, all.length - 1);
                redraw();
            } finally {
                try {
                    document.body.removeChild(input);
                } catch {}
            }
        };

        input.click();
    };

    clearBtn.onclick = () => {
        setTextList(node, []);
        setIndex(node, 0);
        redraw();
    };
    hideBtn.onclick = () => {
        previewsHidden = !previewsHidden;
        hideBtn.textContent = previewsHidden ? "显示" : "隐藏";
        redraw();
    };

    mainContent.appendChild(grid);
    mainContent.appendChild(hiddenOverlay);
    container.appendChild(sidebar);
    container.appendChild(mainContent);

    return { container, redraw };
}

app.registerExtension({
    name: "IO_LoadTextBatch.Extension",
    async setup() {
        api.addEventListener("IO_LoadTextBatch_set", (event) => {
            const nodeId = parseInt(event.detail.node);
            const nodes = app.graph?._nodes || app.graph?.nodes || [];
            const node = nodes.find((n) => n.id === nodeId);
            if (!node) return;

            const itemsRaw = event.detail.items;
            const items = Array.isArray(itemsRaw) ? itemsRaw : [];
            const wList = getTextListWidget(node);
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

            node._ioLoadTextListUI?.redraw?.();
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_LoadTextBatch") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const textListWidget = getTextListWidget(this);
            if (textListWidget) {
                textListWidget.type = "hidden";
                textListWidget.computeSize = () => [0, -4];
            }
            const cardSizeWidget = getCardSizeWidget(this);
            if (cardSizeWidget) {
                cardSizeWidget.type = "hidden";
                cardSizeWidget.computeSize = () => [0, -4];
            }

            const ui = createTextListUI(this);
            this._ioLoadTextListUI = ui;
            this.addDOMWidget("io_load_text_list", "customwidget", ui.container);
            const minW = 420;
            const minH = 360;
            if (!this.size || this.size[0] < minW || this.size[1] < minH) {
                this.setSize([Math.max(this.size?.[0] || 0, minW), Math.max(this.size?.[1] || 0, minH)]);
            }

            const wIndex = getIndexWidget(this);
            const wList = getTextListWidget(this);
            const wSize = getCardSizeWidget(this);

            for (const w of [wIndex, wList, wSize]) {
                if (!w) continue;
                const origCallback = w.callback;
                let lastValue = w.value;
                w.callback = function (value) {
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
            const textListWidget = getTextListWidget(this);
            if (textListWidget) {
                textListWidget.type = "hidden";
                textListWidget.computeSize = () => [0, -4];
            }
            const cardSizeWidget = getCardSizeWidget(this);
            if (cardSizeWidget) {
                cardSizeWidget.type = "hidden";
                cardSizeWidget.computeSize = () => [0, -4];
            }
            this._ioLoadTextListUI?.redraw?.();
            return r;
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
