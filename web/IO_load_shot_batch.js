import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function getWidgetByName(node, name) {
    return (node?.widgets || []).find((w) => w?.name === name) ?? null;
}

function getShotStateWidget(node) {
    return getWidgetByName(node, "shot_state");
}

function getShotPreviewWidget(node) {
    return getWidgetByName(node, "shot_preview");
}

function getIndexWidget(node) {
    return getWidgetByName(node, "index");
}

function getCardSizeWidget(node) {
    return getWidgetByName(node, "card_size");
}

function safeJsonParse(raw, fallback) {
    try {
        return JSON.parse(raw);
    } catch {
        return fallback;
    }
}

function normalizeOrigIndex(v) {
    const n = Number(v);
    return Number.isFinite(n) && n >= 0 ? Math.floor(n) : null;
}

function getShotPreviewItems(node) {
    const w = getShotPreviewWidget(node);
    const raw = String(w?.value ?? "[]").trim();
    const v = safeJsonParse(raw, []);
    if (!Array.isArray(v)) return [];
    return v
        .filter((x) => x && typeof x === "object")
        .map((x) => ({
            orig_index: normalizeOrigIndex(x.orig_index),
            title: x.title == null ? "" : String(x.title),
            content: x.content == null ? "" : String(x.content),
            thumb: x.thumb == null ? "" : String(x.thumb),
        }))
        .filter((x) => x.orig_index != null);
}

function setShotPreviewItems(node, items) {
    const w = getShotPreviewWidget(node);
    if (!w) return;
    const next = Array.isArray(items) ? items : [];
    w.value = JSON.stringify(next);
    w.callback?.(w.value);
}

function getShotStateItems(node) {
    const w = getShotStateWidget(node);
    const raw = String(w?.value ?? "[]").trim();
    const v = safeJsonParse(raw, []);
    if (!Array.isArray(v)) return [];
    return v
        .filter((x) => x && typeof x === "object")
        .map((x) => ({
            orig_index: normalizeOrigIndex(x.orig_index),
            title: x.title == null ? "" : String(x.title),
            content: x.content == null ? "" : String(x.content),
            removed: Boolean(x.removed),
        }))
        .filter((x) => x.orig_index != null);
}

function setShotStateItems(node, items) {
    const w = getShotStateWidget(node);
    if (!w) return;
    const next = Array.isArray(items) ? items : [];
    w.value = JSON.stringify(next);
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

function mergeIncomingIntoStateAndPreview({ incomingItems, stateItems }) {
    const incomingByOrig = new Map(incomingItems.map((x) => [x.orig_index, x]));
    const stateByOrig = new Map(stateItems.map((x) => [x.orig_index, x]));

    const mergedState = [];
    const seen = new Set();

    for (const st of stateItems) {
        const oi = st.orig_index;
        if (oi == null) continue;
        seen.add(oi);
        const inc = incomingByOrig.get(oi);
        const title = st.title != null && st.title !== "" ? st.title : inc?.title ?? "";
        const content = st.content != null && st.content !== "" ? st.content : inc?.content ?? "";
        mergedState.push({ orig_index: oi, title: String(title ?? ""), content: String(content ?? ""), removed: Boolean(st.removed) });
    }

    for (const inc of incomingItems) {
        const oi = inc.orig_index;
        if (oi == null || seen.has(oi)) continue;
        mergedState.push({ orig_index: oi, title: String(inc.title ?? ""), content: String(inc.content ?? ""), removed: false });
    }

    const mergedPreview = [];
    for (const st of mergedState) {
        const inc = incomingByOrig.get(st.orig_index);
        const thumb = inc?.thumb ?? "";
        mergedPreview.push({
            orig_index: st.orig_index,
            title: st.title,
            content: st.content,
            thumb,
        });
    }

    return { mergedState, mergedPreview };
}

function isAnyImportConnected(node) {
    const inputs = node?.inputs || [];
    const names = new Set(["shot", "img_list", "text_list"]);
    for (const inp of inputs) {
        if (!inp || !names.has(inp.name)) continue;
        if (inp.link != null) return true;
    }
    return false;
}

function clearShotUI(node) {
    setShotPreviewItems(node, []);
    setShotStateItems(node, []);
    setIndex(node, 0);
    try {
        node._ioLoadShotListHistory = [];
    } catch {}
}

function createShotListUI(node) {
    const container = document.createElement("div");
    container.dataset.ioLoadshotlist = "1";
    container.style.cssText =
        "width:100%;height:100%;min-width:0;box-sizing:border-box;overflow:hidden;padding:8px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;margin:10px 0 6px 0;pointer-events:none;display:flex;flex-direction:row;gap:8px;";

    const sidebar = document.createElement("div");
    sidebar.style.cssText = "display:flex;flex-direction:column;gap:2px;min-width:56px;width:56px;pointer-events:auto;";

    const mkBtn = (label) => {
        const b = document.createElement("button");
        b.textContent = label;
        b.style.cssText =
            "padding:4px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;cursor:pointer;font-size:10px;width:100%;text-align:center;";
        return b;
    };

    const sortBtn = mkBtn("顺序");
    const clearBtn = mkBtn("重置");
    const undoBtn = mkBtn("撤回");
    const hideBtn = mkBtn("隐藏");
    const sizeBtn = mkBtn("尺寸");

    const sizeSlider = document.createElement("input");
    sizeSlider.type = "range";
    sizeSlider.min = "120";
    sizeSlider.max = "520";
    sizeSlider.step = "10";
    sizeSlider.style.cssText = "width:100%;margin:2px 0 0 0;cursor:pointer;";

    const sizeVal = document.createElement("div");
    sizeVal.style.cssText = "font-size:10px;opacity:0.85;text-align:center;line-height:1.2;padding-bottom:2px;";

    sidebar.appendChild(sortBtn);
    sidebar.appendChild(clearBtn);
    sidebar.appendChild(undoBtn);
    sidebar.appendChild(hideBtn);
    sidebar.appendChild(sizeBtn);
    sidebar.appendChild(sizeSlider);
    sidebar.appendChild(sizeVal);

    const mainContent = document.createElement("div");
    mainContent.style.cssText = "flex:1;display:flex;flex-direction:column;gap:6px;pointer-events:auto;min-width:0;min-height:0;";

    const grid = document.createElement("div");
    grid.style.cssText =
        "display:grid;grid-template-columns:repeat(auto-fill,minmax(var(--card-size,120px),1fr));gap:6px;flex:1;min-width:0;min-height:0;overflow-y:auto;background:var(--comfy-input-bg);padding:6px;border-radius:4px;";

    const hiddenOverlay = document.createElement("div");
    hiddenOverlay.style.cssText =
        "flex:1;display:none;align-items:center;justify-content:center;background:var(--comfy-input-bg);border-radius:4px;color:var(--input-text);font-size:12px;opacity:0.75;border:1px solid var(--border-color);";

    let previewsHidden = false;
    let nextSortIsAsc = true;

    const getHistory = () => {
        if (!node._ioLoadShotListHistory) node._ioLoadShotListHistory = [];
        return node._ioLoadShotListHistory;
    };

    const pushHistory = () => {
        try {
            const snap = {
                preview: getShotPreviewItems(node),
                state: getShotStateItems(node),
                index: getIndex(node),
            };
            const h = getHistory();
            h.push(snap);
            if (h.length > 50) h.splice(0, h.length - 50);
        } catch {}
    };

    const popHistory = () => {
        const h = getHistory();
        return h.length ? h.pop() : null;
    };

    const restoreHistory = (snap) => {
        if (!snap) return;
        setShotPreviewItems(node, Array.isArray(snap.preview) ? snap.preview : []);
        setShotStateItems(node, Array.isArray(snap.state) ? snap.state : []);
        setIndex(node, Number.isFinite(Number(snap.index)) ? Number(snap.index) : 1);
        redraw();
    };

    const syncSizeUIFromWidget = () => {
        const cur = getCardSize(node);
        sizeSlider.value = String(cur);
        sizeVal.textContent = String(cur);
    };

    const applyCardSizeFromSlider = () => {
        const v = Number(sizeSlider.value);
        setCardSize(node, v);
        sizeVal.textContent = String(v);
        redraw();
    };

    sizeBtn.onclick = () => applyCardSizeFromSlider();
    sizeSlider.addEventListener("input", applyCardSizeFromSlider);
    syncSizeUIFromWidget();

    const redraw = () => {
        if (!isAnyImportConnected(node)) {
            clearShotUI(node);
        }
        const previewItems = getShotPreviewItems(node);
        const stateItems = getShotStateItems(node);
        const previewByOrig = new Map(previewItems.map((x) => [x.orig_index, x]));
        const used = new Set();
        const removedSet = new Set(stateItems.filter((x) => x.removed).map((x) => x.orig_index));
        const visibleItems = [];

        for (const st of stateItems) {
            if (st.removed) {
                used.add(st.orig_index);
                continue;
            }
            const it = previewByOrig.get(st.orig_index);
            if (!it) continue;
            used.add(st.orig_index);
            visibleItems.push({
                orig_index: it.orig_index,
                title: st.title ?? it.title,
                content: st.content ?? it.content,
                thumb: it.thumb,
            });
        }
        for (const it of previewItems) {
            if (used.has(it.orig_index)) continue;
            if (removedSet.has(it.orig_index)) continue;
            visibleItems.push(it);
        }
        const total = visibleItems.length;
        const size = Math.max(120, Math.min(520, getCardSize(node)));
        grid.style.setProperty("--card-size", `${size}px`);

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

        const frag = document.createDocumentFragment();

        if (visibleItems.length === 0) {
            const empty = document.createElement("div");
            empty.style.cssText =
                `height:${Math.max(200, Math.floor(size * 1.2))}px;display:flex;flex-direction:column;gap:6px;padding:10px;background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:6px;user-select:none;align-items:center;justify-content:center;opacity:0.85;`;
            const t1 = document.createElement("div");
            t1.textContent = "暂无分镜";
            t1.style.cssText = "font-size:12px;";
            const t2 = document.createElement("div");
            t2.textContent = "连接 shot 输入并运行一次队列以生成预览";
            t2.style.cssText = "font-size:11px;opacity:0.8;";
            empty.appendChild(t1);
            empty.appendChild(t2);
            frag.appendChild(empty);
            grid.appendChild(frag);
            app.graph.setDirtyCanvas(true);
            return;
        }

        const selected = getIndex(node);

        for (let i = 0; i < visibleItems.length; i++) {
            const item = visibleItems[i];
            const card = document.createElement("div");
            const isSel = i === selected;
            card.style.cssText =
                `position:relative;min-height:${Math.max(180, Math.floor(size * 1.3))}px;max-height:${Math.max(180, Math.floor(size * 1.3))}px;display:flex;flex-direction:column;gap:2px;padding:2px;background:var(--comfy-menu-bg);border:2px solid ${isSel ? "#fff" : "var(--border-color)"};border-radius:6px;user-select:none;cursor:pointer;overflow:hidden;`;

            card.addEventListener("click", (e) => {
                e.preventDefault();
                setIndex(node, i);
                redraw();
            });

            card.setAttribute("draggable", "true");
            card.addEventListener("dragstart", (e) => {
                e.stopPropagation();
                e.dataTransfer && (e.dataTransfer.effectAllowed = "move");
                e.dataTransfer?.setData?.("application/x-io-shot-index", String(i));
                card.style.opacity = "0.6";
            });
            card.addEventListener("dragend", () => {
                card.style.opacity = "";
            });
            card.addEventListener("dragover", (e) => {
                e.stopPropagation();
                e.preventDefault();
                e.dataTransfer && (e.dataTransfer.dropEffect = "move");
            });
            card.addEventListener("drop", (e) => {
                e.stopPropagation();
                e.preventDefault();
                const fromRaw = e.dataTransfer?.getData?.("application/x-io-shot-index");
                const from = Number(fromRaw);
                const to = i;
                if (!Number.isInteger(from) || from < 0 || from >= visibleItems.length) return;
                if (from === to) return;
                const state = getShotStateItems(node);
                const stateByOrig = new Map(state.map((x) => [x.orig_index, x]));
                const order = visibleItems.map((x) => x.orig_index);
                const [moved] = order.splice(from, 1);
                order.splice(to, 0, moved);
                const nextState = order.map((oi) => stateByOrig.get(oi) || { orig_index: oi, title: "", content: "", removed: false });
                const orderSet = new Set(order);
                for (const st of state) if (!orderSet.has(st.orig_index)) nextState.push(st);
                setShotStateItems(node, nextState);
                redraw();
            });

            const del = document.createElement("button");
            del.textContent = "×";
            del.style.cssText =
                "position:absolute;top:3px;right:3px;width:18px;height:18px;background:rgba(255,0,0,0.75);color:#fff;border:none;border-radius:3px;cursor:pointer;font-size:14px;line-height:1;z-index:3;";
            del.addEventListener("click", (e) => {
                e.preventDefault();
                e.stopPropagation();
                pushHistory();
                const state = getShotStateItems(node);
                const idx = state.findIndex((x) => x.orig_index === item.orig_index);
                if (idx >= 0) state[idx] = { ...state[idx], removed: true };
                else state.push({ orig_index: item.orig_index, title: item.title, content: item.content, removed: true });
                setShotStateItems(node, state);

                const prev = getShotPreviewItems(node);
                const k = prev.findIndex((x) => x.orig_index === item.orig_index);
                if (k >= 0) {
                    prev.splice(k, 1);
                    setShotPreviewItems(node, prev);
                }
                const curIdx = getIndex(node);
                const nextTotal = Math.max(0, total - 1);
                if (nextTotal <= 0) setIndex(node, 0);
                else if (curIdx > nextTotal) setIndex(node, nextTotal);
                redraw();
            });

            const seq = document.createElement("div");
            seq.textContent = `#${i}`;
            seq.style.cssText = "position:absolute;top:3px;left:3px;font-size:10px;font-weight:800;color:#000;white-space:nowrap;z-index:3;background:rgba(255,255,255,0.92);padding:0 3px;border-radius:2px;line-height:1.2;";

            const imgWrap = document.createElement("div");
            imgWrap.style.cssText =
                `width:100%;aspect-ratio:1/1;border:1px solid var(--border-color);border-radius:4px;overflow:hidden;background:var(--comfy-input-bg);display:flex;align-items:center;justify-content:center;`;
            const img = document.createElement("img");
            img.draggable = false;
            img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;";
            img.addEventListener("dragstart", (e) => {
                e.preventDefault();
                e.stopPropagation();
            });
            if (item.thumb) {
                img.src = item.thumb;
            } else {
                img.alt = "no image";
            }
            imgWrap.appendChild(img);
            imgWrap.addEventListener("dragstart", (e) => {
                e.stopPropagation();
            });
            imgWrap.addEventListener("dragover", (e) => {
                e.stopPropagation();
            });
            imgWrap.addEventListener("drop", (e) => {
                e.stopPropagation();
            });

            const textWrap = document.createElement("div");
            textWrap.style.cssText =
                "flex:1;display:flex;flex-direction:column;gap:2px;padding:3px;background:var(--comfy-input-bg);border:1px solid var(--border-color);border-radius:4px;overflow:hidden;min-height:0;";

            const title = document.createElement("div");
            title.textContent = item.title || "";
            title.style.cssText =
                "font-size:10px;font-weight:600;opacity:0.95;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#00ff4b;line-height:1.2;";

            const content = document.createElement("div");
            content.textContent = item.content || "";
            content.style.cssText =
                "flex:1;font-size:9px;opacity:0.85;white-space:pre-wrap;word-break:break-word;display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:6;overflow:hidden;line-height:1.2;min-height:0;";

            textWrap.appendChild(title);
            textWrap.appendChild(content);

            card.addEventListener("dblclick", (e) => {
                e.preventDefault();
                const st = getShotStateItems(node);
                const idx = st.findIndex((x) => x.orig_index === item.orig_index);
                const cur = idx >= 0 ? st[idx] : { orig_index: item.orig_index, title: item.title, content: item.content, removed: false };
                const overlay = document.createElement("div");
                overlay.style.cssText =
                    "position:fixed;left:0;top:0;right:0;bottom:0;background:rgba(0,0,0,0.55);z-index:99999;display:flex;align-items:center;justify-content:center;";
                const panel = document.createElement("div");
                panel.style.cssText =
                    "width:min(860px,90vw);height:min(620px,85vh);background:var(--comfy-menu-bg);border:1px solid var(--border-color);border-radius:10px;display:flex;flex-direction:column;gap:8px;padding:10px;";
                const head = document.createElement("div");
                head.style.cssText = "display:flex;align-items:center;justify-content:space-between;gap:8px;";
                const t = document.createElement("div");
                t.textContent = `编辑 #${i + 1}`;
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

                const titleInput = document.createElement("input");
                titleInput.type = "text";
                titleInput.value = String(cur.title ?? "");
                titleInput.placeholder = "标题";
                titleInput.style.cssText =
                    "width:100%;padding:8px 10px;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:8px;";

                const ta = document.createElement("textarea");
                ta.value = String(cur.content ?? "");
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
                    const nextTitle = String(titleInput.value ?? "");
                    const nextContent = String(ta.value ?? "");
                    const next = getShotStateItems(node);
                    const j = next.findIndex((x) => x.orig_index === item.orig_index);
                    if (j >= 0) next[j] = { ...next[j], title: nextTitle, content: nextContent };
                    else next.push({ orig_index: item.orig_index, title: nextTitle, content: nextContent, removed: false });
                    setShotStateItems(node, next);
                    const prev = getShotPreviewItems(node);
                    const k = prev.findIndex((x) => x.orig_index === item.orig_index);
                    if (k >= 0) prev[k] = { ...prev[k], title: nextTitle, content: nextContent };
                    setShotPreviewItems(node, prev);
                    redraw();
                    close();
                };

                panel.appendChild(head);
                panel.appendChild(titleInput);
                panel.appendChild(ta);
                overlay.appendChild(panel);
                document.body.appendChild(overlay);
                requestAnimationFrame(() => titleInput.focus());
            });

            card.appendChild(del);
            card.appendChild(seq);
            card.appendChild(imgWrap);
            card.appendChild(textWrap);
            frag.appendChild(card);
        }

        grid.appendChild(frag);
        app.graph.setDirtyCanvas(true);
    };

    sortBtn.onclick = (e) => {
        e.preventDefault();
        const preview = getShotPreviewItems(node).slice();
        if (preview.length === 0) return;
        preview.sort((a, b) => (nextSortIsAsc ? a.orig_index - b.orig_index : b.orig_index - a.orig_index));
        nextSortIsAsc = !nextSortIsAsc;
        sortBtn.textContent = nextSortIsAsc ? "顺序" : "逆序";
        setShotPreviewItems(node, preview);
        const state = getShotStateItems(node);
        const stateByOrig = new Map(state.map((x) => [x.orig_index, x]));
        const nextState = preview.map((x) => stateByOrig.get(x.orig_index) || { orig_index: x.orig_index, title: x.title, content: x.content, removed: false });
        const previewOrigs = new Set(preview.map((x) => x.orig_index));
        for (const st of state) if (!previewOrigs.has(st.orig_index)) nextState.push(st);
        setShotStateItems(node, nextState);
        redraw();
    };

    clearBtn.onclick = (e) => {
        e.preventDefault();
        pushHistory();
        setShotPreviewItems(node, []);
        setShotStateItems(node, []);
        setIndex(node, 0);
        redraw();
    };

    undoBtn.onclick = (e) => {
        e.preventDefault();
        const snap = popHistory();
        if (!snap) return;
        restoreHistory(snap);
    };

    hideBtn.onclick = (e) => {
        e.preventDefault();
        previewsHidden = !previewsHidden;
        hideBtn.textContent = previewsHidden ? "显示" : "隐藏";
        redraw();
    };

    mainContent.appendChild(grid);
    mainContent.appendChild(hiddenOverlay);
    container.appendChild(sidebar);
    container.appendChild(mainContent);

    redraw();
    return { container, redraw };
}

app.registerExtension({
    name: "IO_LoadShotBatch.Extension",
    async setup() {
        api.addEventListener("IO_LoadShotBatch_append", function (event) {
            const nodeId = parseInt(event.detail.node);
            const node = app.graph.nodes.find((n) => n.id === nodeId);
            if (!node) return;
            const itemsRaw = event.detail.items;
            const items = Array.isArray(itemsRaw) ? itemsRaw : itemsRaw == null ? [] : [itemsRaw];
            if (items.length === 0) return;
            const incoming = items
                .filter((x) => x && typeof x === "object")
                .map((x) => ({
                    orig_index: normalizeOrigIndex(x.orig_index),
                    title: x.title == null ? "" : String(x.title),
                    content: x.content == null ? "" : String(x.content),
                    thumb: x.thumb == null ? "" : String(x.thumb),
                }))
                .filter((x) => x.orig_index != null);
            if (incoming.length === 0) return;
            const curPreview = getShotPreviewItems(node);
            const curState = getShotStateItems(node);
            const { mergedState, mergedPreview } = mergeIncomingIntoStateAndPreview({ incomingItems: curPreview.concat(incoming), stateItems: curState });
            setShotPreviewItems(node, mergedPreview);
            setShotStateItems(node, mergedState);
            node._ioLoadShotListUI?.redraw?.();
        });

        api.addEventListener("IO_LoadShotBatch_set", function (event) {
            const nodeId = parseInt(event.detail.node);
            const node = app.graph.nodes.find((n) => n.id === nodeId);
            if (!node) return;
            const itemsRaw = event.detail.items;
            const items = Array.isArray(itemsRaw) ? itemsRaw : itemsRaw == null ? [] : [itemsRaw];
            const incoming = items
                .filter((x) => x && typeof x === "object")
                .map((x) => ({
                    orig_index: normalizeOrigIndex(x.orig_index),
                    title: x.title == null ? "" : String(x.title),
                    content: x.content == null ? "" : String(x.content),
                    thumb: x.thumb == null ? "" : String(x.thumb),
                }))
                .filter((x) => x.orig_index != null);
            setShotPreviewItems(node, incoming);

            const curState = getShotStateItems(node);
            const { mergedState, mergedPreview } = mergeIncomingIntoStateAndPreview({ incomingItems: incoming, stateItems: curState });
            setShotStateItems(node, mergedState);
            setShotPreviewItems(node, mergedPreview);
            node._ioLoadShotListUI?.redraw?.();
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_LoadShotBatch") return;

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = origOnNodeCreated?.apply(this, arguments);

            const wState = getShotStateWidget(this);
            if (wState) {
                wState.type = "textarea";
                wState.computeSize = () => [300, 48];
                if (!wState.value || String(wState.value).trim() === "") {
                    wState.value = "[]";
                    wState.callback?.(wState.value);
                }
            }

            const wPreview = getShotPreviewWidget(this);
            if (wPreview) {
                wPreview.hidden = true;
                wPreview.type = "converted-widget:shot_preview";
                wPreview.computeSize = () => [0, -4];
                if (!wPreview.value || String(wPreview.value).trim() === "") {
                    wPreview.value = "[]";
                    wPreview.callback?.(wPreview.value);
                }
            }

            const wSize = getCardSizeWidget(this);
            if (wSize) {
                wSize.type = "number";
                wSize.computeSize = () => [120, 24];
                const v = Number(wSize.value);
                if (!Number.isFinite(v) || v < 120) {
                    wSize.value = 120;
                    wSize.callback?.(wSize.value);
                }
            }

            const wIndex = getIndexWidget(this);
            if (wIndex) {
                const v = Number(wIndex.value);
                if (!Number.isFinite(v) || v < 1) {
                    wIndex.value = 1;
                    wIndex.callback?.(wIndex.value);
                }
            }

            const ui = createShotListUI(this);
            if (ui) {
                this._ioLoadShotListUI = ui;
                const minW = 520;
                const minH = 420;
                if (!this.size || this.size[0] < minW || this.size[1] < minH) {
                    this.setSize([Math.max(this.size?.[0] || 0, minW), Math.max(this.size?.[1] || 0, minH)]);
                }
                // Set minimum node size to prevent button overflow
                this.minWidth = Math.max(this.minWidth || 0, 520);
                this.minHeight = Math.max(this.minHeight || 0, 420);

                this.addDOMWidget("io_load_shot_list", "customwidget", ui.container);
            }

            return r;
        };

        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info, ioSlot) {
            const r = origOnConnectionsChange?.apply(this, arguments);
            try {
                if (ioSlot === "INPUT" && !connected) {
                    if (!isAnyImportConnected(this)) {
                        clearShotUI(this);
                        this._ioLoadShotListUI?.redraw?.();
                    }
                }
            } catch {}
            return r;
        };

        const origOnConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        const r = origOnConfigure?.apply(this, arguments);

        // 确保 widget 的值正确设置（只在无效时才设置默认值）
        const wSize = getCardSizeWidget(this);
        if (wSize) {
            wSize.type = "number";
            wSize.computeSize = () => [120, 24];
            const v = Number(wSize.value);
            // 只在值为 NaN、null 或 undefined 时才设置默认值
            if (wSize.value == null || !Number.isFinite(v) || v < 120) {
                wSize.value = 120;
                wSize.callback?.(wSize.value);
            }
        }

        const wState = getShotStateWidget(this);
        if (wState) {
            wState.type = "textarea";
            wState.computeSize = () => [300, 48];
        }

        const wPreview = getShotPreviewWidget(this);
        if (wPreview) {
            wPreview.hidden = true;
            wPreview.type = "converted-widget:shot_preview";
            wPreview.computeSize = () => [0, -4];
        }

        if (!isAnyImportConnected(this)) {
            clearShotUI(this);
        }
        this._ioLoadShotListUI?.redraw?.();
        return r;
    };
    },
});
