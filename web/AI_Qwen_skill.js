import { app } from "../../scripts/app.js";

const NODE_CLASS = "AI_Qwen_skill";
const MEDIA_RELAY_CLASSES = new Set(["basicIn_media", "AD_Media_editor"]);
const LINKS_PROP = "apt_qwen_skill_media_links";
const RELAY_LINKS_PROP = "apt_media_hub_links";
const MEDIA_LIMIT = 20;

function nodeClass(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "");
}

function widget(node, name) {
    return node.widgets?.find((item) => item?.name === name) || null;
}

function setGrayed(item, grayed) {
    if (!item) return;
    item.disabled = grayed;
    if (item._state) item._state.disabled = grayed;
    const element = item.inputEl || item.element;
    if (element && "disabled" in element) element.disabled = grayed;
}

function setWidgetValue(item, value) {
    if (!item) return;
    item.value = value;
    if (item._state) item._state.value = value;
}

function migrateLegacyNode(node) {
    const legacyNames = new Set(["session_id", "skills", "events"]);
    const legacy = (node.outputs || []).some((output) => legacyNames.has(String(output?.name || "")));
    for (let index = (node.outputs?.length || 0) - 1; index >= 0; index -= 1) {
        if (legacyNames.has(String(node.outputs[index]?.name || ""))) node.removeOutput?.(index);
    }
    if (!legacy) return;
    setWidgetValue(widget(node, "skill"), "不使用技能");
    setWidgetValue(widget(node, "llm_model"), "Qwen3.8-Max");
    setWidgetValue(widget(node, "custom_model"), "");
    setWidgetValue(widget(node, "api_key"), "");
}

function syncModel(node) {
    const custom = widget(node, "llm_model")?.value === "自定义";
    setGrayed(widget(node, "custom_model"), !custom);
    node.setDirtyCanvas?.(true, true);
}

function graphLink(graph, linkId) {
    if (linkId == null) return null;
    for (const values of [graph?.links, graph?._links]) {
        if (!values) continue;
        if (typeof values.get === "function") {
            const link = values.get(linkId) ?? values.get(String(linkId));
            if (link) return link;
        }
        const link = values[linkId] ?? values[String(linkId)];
        if (link) return link;
    }
    return null;
}

function sourceFromGraphLink(node, link) {
    if (!link) return null;
    const sourceId = Number(link.origin_id ?? link.originId ?? link.from_id ?? link.fromId);
    const sourceSlot = Number(link.origin_slot ?? link.originSlot ?? link.from_slot ?? link.fromSlot ?? 0) || 0;
    const source = (node.graph || app.graph)?.getNodeById?.(sourceId);
    if (!source || source === node) return null;
    const output = source.outputs?.[sourceSlot] || {};
    return { source_id: sourceId, source_slot: sourceSlot, source_type: String(output.type || link.type || "*").toUpperCase() };
}

function sourceFromLink(node, linkInfo = null) {
    const input = (node.inputs || []).find((item) => String(item?.name || "") === "media");
    const linkId = input?.link ?? linkInfo?.id ?? linkInfo?.link_id ?? linkInfo?.linkId;
    const link = graphLink(node.graph || app.graph, linkId) || linkInfo;
    return sourceFromGraphLink(node, link);
}

function links(node) {
    node.properties ||= {};
    if (!Array.isArray(node.properties[LINKS_PROP])) node.properties[LINKS_PROP] = [];
    return node.properties[LINKS_PROP];
}

function addSource(node, source) {
    const values = links(node);
    if (values.some((item) => item.source_id === source.source_id && item.source_slot === source.source_slot)) return;
    if (values.length < MEDIA_LIMIT) values.push(source);
}

function migrateMediaInputs(node) {
    const graph = node.graph || app.graph;
    for (const input of node.inputs || []) {
        if (!/^media(?:_\d+)?$/.test(String(input?.name || "")) || input.link == null) continue;
        const source = sourceFromGraphLink(node, graphLink(graph, input.link));
        if (source) addSource(node, source);
    }
    for (let index = (node.inputs?.length || 0) - 1; index >= 0; index -= 1) {
        if (/^media_\d+$/.test(String(node.inputs[index]?.name || ""))) node.removeInput?.(index);
    }
    let mediaIndex = (node.inputs || []).findIndex((item) => String(item?.name || "") === "media");
    if (mediaIndex < 0) {
        node.addInput?.("media", "IMAGE,VIDEO,STRING");
        mediaIndex = (node.inputs || []).findIndex((item) => String(item?.name || "") === "media");
    }
    if (mediaIndex >= 0 && node.inputs?.[mediaIndex]?.link != null) {
        node.__aptQwenSkillClearing = true;
        try {
            node.disconnectInput?.(mediaIndex);
            if (node.inputs?.[mediaIndex]) node.inputs[mediaIndex].link = null;
        } finally {
            node.__aptQwenSkillClearing = false;
        }
    }
}

function captureConnection(node, linkInfo = null) {
    if (node.__aptQwenSkillClearing) return;
    const source = sourceFromLink(node, linkInfo);
    if (!source) return;
    addSource(node, source);
    const inputIndex = (node.inputs || []).findIndex((item) => String(item?.name || "") === "media");
    node.__aptQwenSkillClearing = true;
    try {
        if (inputIndex >= 0 && node.inputs?.[inputIndex]?.link != null) node.disconnectInput?.(inputIndex);
        if (inputIndex >= 0 && node.inputs?.[inputIndex]) node.inputs[inputIndex].link = null;
    } finally {
        node.__aptQwenSkillClearing = false;
    }
    refreshEditor(node);
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
}

function expandedSources(node) {
    const graph = node.graph || app.graph;
    const result = [];
    for (const item of links(node)) {
        const source = graph?.getNodeById?.(Number(item.source_id));
        if (!source) continue;
        if (MEDIA_RELAY_CLASSES.has(nodeClass(source))) {
            for (const child of source.properties?.[RELAY_LINKS_PROP] || []) {
                const childNode = graph?.getNodeById?.(Number(child.source_id));
                if (!childNode) continue;
                const output = childNode.outputs?.[Number(child.source_slot) || 0] || {};
                result.push({
                    source_id: Number(child.source_id),
                    source_slot: Number(child.source_slot) || 0,
                    source_type: String(child.source_type || output.type || "*").toUpperCase(),
                    media_type: String(child.media_type || "").toLowerCase(),
                    cached: child.cached === true,
                    filename: String(child.filename || ""),
                    path: String(child.path || ""),
                    text: child.text == null ? "" : String(child.text),
                });
            }
        } else {
            result.push(item);
        }
    }
    return result.slice(0, MEDIA_LIMIT);
}

function mediaType(source, item) {
    const type = String(item?.source_type || source?.outputs?.[item?.source_slot]?.type || "").toUpperCase();
    if (type.includes("IMAGE")) return "Image";
    if (type.includes("VIDEO")) return "Video";
    if (type.includes("STRING") || type.includes("TEXT")) return "Text";
    return "";
}

function sourceDetail(source, type, item = null) {
    if (item?.cached) {
        if (type === "Text" && item.text) return item.text.length > 36 ? `${item.text.slice(0, 35)}…` : item.text;
        const cachedName = String(item.filename || item.path || "").split(/[\\/]/).pop();
        if (cachedName) return cachedName;
    }
    const preferred = type === "Image"
        ? ["image", "filename", "file"]
        : type === "Video" ? ["video", "filename", "file"] : ["text", "string", "prompt"];
    for (const name of preferred) {
        const item = widget(source, name);
        const raw = typeof item?.value === "object" ? item.value?.filename || item.value?.name : item?.value;
        const value = String(raw || "").trim();
        if (!value) continue;
        if (type === "Text") return value.length > 36 ? `${value.slice(0, 35)}…` : value;
        return value.split(/[\\/]/).pop() || value;
    }
    return String(source?.title || nodeClass(source));
}

function mediaViewUrlFromWidgets(node, preferredNames) {
    const preferred = new Set(preferredNames);
    const widgets = Array.isArray(node?.widgets) ? node.widgets : [];
    const candidates = [
        ...widgets.filter((item) => preferred.has(String(item?.name || "").toLowerCase())),
        ...widgets,
    ];
    for (const item of candidates) {
        const value = item?.value;
        if (!value) continue;
        const filename = typeof value === "object" ? value.filename : value;
        if (!filename) continue;
        const params = new URLSearchParams({
            filename: String(filename),
            type: typeof value === "object" ? String(value.type || "input") : "input",
        });
        if (typeof value === "object" && value.subfolder) params.set("subfolder", String(value.subfolder));
        return `/view?${params.toString()}`;
    }
    return "";
}

function videoFrameThumbnail(url) {
    const resolve = globalThis.__aptPresetVideoFrameThumbnail;
    return typeof resolve === "function" ? resolve(url) : "";
}

function sourcePreview(source, type, item = null) {
    if (!source || type === "Text") return { url: "", kind: "" };
    if (item?.cached) {
        const path = String(item.path || item.filename || "");
        if (path && type === "Image") return { url: `/view?filename=${encodeURIComponent(path)}&type=input`, kind: "image" };
        if (path && type === "Video") {
            const videoUrl = `/Apt_Preset_IO_LoadMedia_preview?path=${encodeURIComponent(path)}&media=video`;
            return { url: videoFrameThumbnail(videoUrl), kind: "image" };
        }
    }
    const image = (source.imgs || []).find((item) => item?.src);
    if (image?.src) return { url: image.src, kind: "image" };
    for (const item of source.widgets || []) {
        const element = item?.element || item?.inputEl;
        const preview = element?.matches?.("img") ? element : element?.querySelector?.("img");
        if (preview?.src) return { url: preview.src, kind: "image" };
        const video = element?.matches?.("video") ? element : element?.querySelector?.("video");
        if (type === "Video" && (video?.poster || video?.currentSrc || video?.src)) {
            const thumbnail = videoFrameThumbnail(video.currentSrc || video.src);
            return { url: thumbnail || video.poster || "", kind: "image" };
        }
    }
    const url = mediaViewUrlFromWidgets(
        source,
        type === "Video" ? ["video", "file", "filename", "video_file"] : ["image", "file", "filename"],
    );
    if (type === "Video") return { url: videoFrameThumbnail(url), kind: "image" };
    return { url, kind: url ? type.toLowerCase() : "" };
}

function mediaOptions(node) {
    const graph = node.graph || app.graph;
    const counts = { Image: 0, Video: 0, Text: 0 };
    const options = [];
    for (const item of expandedSources(node)) {
        const source = graph?.getNodeById?.(Number(item.source_id));
        const type = mediaType(source, item);
        if (!type) continue;
        counts[type] += 1;
        const label = `${type} ${counts[type]}`;
        const preview = sourcePreview(source, type, item);
        options.push({
            label,
            token: `@${label}`,
            type: type.toLowerCase(),
            detail: sourceDetail(source, type, item),
            previewUrl: preview.url,
            previewKind: preview.kind,
            sourceId: Number(item.source_id),
            sourceSlot: Number(item.source_slot) || 0,
        });
    }
    return options;
}

function optionKey(option) {
    return `${Number(option?.sourceId)}:${Number(option?.sourceSlot) || 0}:${String(option?.type || "")}`;
}

function replaceToken(text, from, to) {
    const escaped = String(from || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    return escaped ? String(text || "").replace(new RegExp(escaped, "gi"), to) : String(text || "");
}

function removeSource(node, index) {
    const values = links(node);
    if (index < 0 || index >= values.length) return false;
    const before = mediaOptions(node);
    values.splice(index, 1);
    const after = mediaOptions(node);
    const nextBySource = new Map(after.map((option) => [optionKey(option), option]));
    let text = String(widget(node, "text")?.value || "");
    for (const option of before) {
        text = replaceToken(text, option.token, nextBySource.get(optionKey(option))?.token || "");
    }
    text = text.replace(/[ \t]{2,}/g, " ");
    setStoredValue(node, "text", text);
    renderPlainText(node, text);
    node.__aptQwenSkillDeleteIndex = -1;
    closeMenu(node);
    node.graph?.change?.();
    node.setDirtyCanvas?.(true, true);
    return true;
}

function closeMenu(node) {
    node.__aptQwenSkillMenu?.remove?.();
    node.__aptQwenSkillMenu = null;
}

function makeThumb(option, menu = false) {
    if (option.previewUrl) {
        const preview = document.createElement(option.previewKind === "video" ? "video" : "img");
        preview.className = `apt-qwen-skill-chip-thumb${menu ? " is-menu" : ""}`;
        preview.draggable = false;
        preview.src = option.previewUrl;
        if (preview.tagName === "VIDEO") {
            preview.muted = true;
            preview.playsInline = true;
            preview.preload = "metadata";
            preview.addEventListener("loadedmetadata", () => {
                try {
                    preview.currentTime = Math.min(0.5, Math.max(0, Number(preview.duration || 0) * 0.1));
                } catch { /* keep the browser-selected first frame */ }
            }, { once: true });
        }
        preview.addEventListener("error", () => {
            preview.replaceWith(makeThumb({ ...option, previewUrl: "", previewKind: "" }, menu));
        }, { once: true });
        return preview;
    }
    const thumb = document.createElement("span");
    thumb.className = `apt-qwen-skill-chip-thumb is-${option.type}${menu ? " is-menu" : ""}`;
    return thumb;
}

function makeChip(option, menu = false) {
    const chip = document.createElement("span");
    chip.className = `apt-qwen-skill-chip${menu ? " is-menu" : ""}`;
    chip.contentEditable = "false";
    chip.dataset.token = option.token;
    const label = document.createElement("span");
    label.textContent = option.label;
    chip.append(makeThumb(option, menu), label);
    return chip;
}

function editorText(editor) {
    let value = "";
    function read(root) {
        for (const child of root.childNodes) {
            if (child.nodeType === Node.TEXT_NODE) value += child.nodeValue || "";
            else if (child.nodeType === Node.ELEMENT_NODE) {
                if (child.classList.contains("apt-qwen-skill-chip")) value += child.dataset.token || "";
                else if (child.tagName === "BR") value += "\n";
                else {
                    if (["DIV", "P"].includes(child.tagName) && value && !value.endsWith("\n")) value += "\n";
                    read(child);
                }
            }
        }
    }
    read(editor);
    return value.replace(/\n{3,}/g, "\n\n");
}

function syncTextWidget(node) {
    const textWidget = widget(node, "text");
    if (!textWidget || !node.__aptQwenSkillEditor) return;
    const value = editorText(node.__aptQwenSkillEditor);
    textWidget.value = value;
    if (textWidget._state) textWidget._state.value = value;
    textWidget.callback?.(value);
}

function insertChip(node, option) {
    const editor = node.__aptQwenSkillEditor;
    const selection = window.getSelection?.();
    if (!editor || !selection?.rangeCount) return;
    const range = selection.getRangeAt(0);
    let textNode = range.startContainer;
    let offset = range.startOffset;
    if (textNode.nodeType !== Node.TEXT_NODE) {
        textNode = textNode.childNodes?.[Math.max(0, offset - 1)];
        offset = textNode?.nodeValue?.length || 0;
    }
    if (textNode?.nodeType === Node.TEXT_NODE) {
        const before = String(textNode.nodeValue || "").slice(0, offset);
        const match = before.match(/@[^\s@\n]*$/);
        if (match) {
            const start = offset - match[0].length;
            textNode.deleteData(start, match[0].length);
            range.setStart(textNode, start);
            range.collapse(true);
        }
    }
    const chip = makeChip(option);
    const space = document.createTextNode(" ");
    range.insertNode(space);
    range.insertNode(chip);
    range.setStartAfter(space);
    range.collapse(true);
    selection.removeAllRanges();
    selection.addRange(range);
    closeMenu(node);
    syncTextWidget(node);
    editor.focus();
}

function openMenu(node, query = "") {
    closeMenu(node);
    const filter = String(query).toLowerCase();
    const options = mediaOptions(node).filter((item) => !filter
        || item.label.toLowerCase().includes(filter)
        || item.detail.toLowerCase().includes(filter));
    const menu = document.createElement("div");
    menu.className = "apt-qwen-skill-menu";
    const title = document.createElement("div");
    title.className = "apt-qwen-skill-menu-title";
    title.textContent = options.length ? "选择已连接素材" : "没有已连接素材";
    menu.append(title);
    for (const option of options) {
        const item = document.createElement("div");
        item.className = "apt-qwen-skill-menu-item";
        item.append(makeChip(option, true));
        const detail = document.createElement("span");
        detail.className = "apt-qwen-skill-menu-detail";
        detail.textContent = option.detail;
        item.append(detail);
        item.addEventListener("pointerdown", (event) => {
            event.preventDefault();
            insertChip(node, option);
        });
        menu.append(item);
    }
    document.body.append(menu);
    positionMenuAtCaret(menu, node.__aptQwenSkillEditor);
    node.__aptQwenSkillMenu = menu;
}

function caretRect(editor) {
    const selection = window.getSelection?.();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) return editor.getBoundingClientRect();
    const range = selection.getRangeAt(0).cloneRange();
    range.collapse(true);
    const rects = range.getClientRects?.();
    let rect = rects?.length ? rects[rects.length - 1] : range.getBoundingClientRect?.();
    if (rect && (rect.width || rect.height)) return rect;
    const marker = document.createElement("span");
    marker.textContent = "\u200b";
    range.insertNode(marker);
    rect = marker.getBoundingClientRect();
    marker.remove();
    return rect && (rect.width || rect.height) ? rect : editor.getBoundingClientRect();
}

function positionMenuAtCaret(menu, editor) {
    const rect = caretRect(editor);
    const width = Math.min(320, Math.max(250, menu.offsetWidth || 250));
    const height = Math.min(360, menu.offsetHeight || 120);
    let left = rect.left;
    let top = rect.bottom + 6;
    if (left + width > window.innerWidth - 8) left = window.innerWidth - width - 8;
    if (top + height > window.innerHeight - 8) top = Math.max(8, rect.top - height - 6);
    menu.style.left = `${Math.max(8, Math.round(left))}px`;
    menu.style.top = `${Math.max(8, Math.round(top))}px`;
    menu.style.width = `${width}px`;
}

function currentMention(editor) {
    const selection = window.getSelection?.();
    if (!selection?.rangeCount || !editor.contains(selection.anchorNode)) return null;
    const node = selection.anchorNode;
    if (node?.nodeType !== Node.TEXT_NODE) return null;
    const before = String(node.nodeValue || "").slice(0, selection.anchorOffset);
    return before.match(/@([^\s@\n]*)$/)?.[1] ?? null;
}

function renderPlainText(node, value) {
    const editor = node.__aptQwenSkillEditor;
    if (!editor) return;
    const source = String(value || "");
    editor.textContent = "";
    const options = new Map(mediaOptions(node).map((item) => [item.token.toLowerCase(), item]));
    const pattern = /@(Image|Video|Text)\s+\d+/gi;
    let cursor = 0;
    for (const match of source.matchAll(pattern)) {
        if (match.index > cursor) editor.append(document.createTextNode(source.slice(cursor, match.index)));
        const option = options.get(match[0].toLowerCase());
        editor.append(option ? makeChip(option) : document.createTextNode(match[0]));
        cursor = match.index + match[0].length;
    }
    if (cursor < source.length) editor.append(document.createTextNode(source.slice(cursor)));
}

function refreshEditor(node) {
    if (!node.__aptQwenSkillEditor || document.activeElement === node.__aptQwenSkillEditor) return;
    renderPlainText(node, widget(node, "text")?.value || "");
}

function hideWidget(node, name) {
    const item = widget(node, name);
    if (!item) return item;
    item.__aptQwenSkillHidden = true;
    item.hidden = true;
    item.type = "hidden";
    item.computeSize = () => [0, -4];
    item.options ||= {};
    item.options.hidden = true;
    item.options.canvasOnly = true;
    const element = item.inputEl || item.element;
    if (element) element.style.display = "none";
    if (item.inputEl) item.inputEl.style.display = "none";
    if (item.element) item.element.style.display = "none";
    if (item._state) {
        item._state.hidden = true;
        item._state.type = "hidden";
    }
    return item;
}

function conversationState(node) {
    try {
        const state = JSON.parse(String(widget(node, "conversation_history")?.value || ""));
        if (state && Array.isArray(state.turns)) return state;
    } catch { /* start with an empty conversation */ }
    return {
        version: 1,
        skill: String(widget(node, "skill")?.value || ""),
        model: String(widget(node, "llm_model")?.value || ""),
        turns: [],
    };
}

function setStoredValue(node, name, value) {
    const item = widget(node, name);
    if (!item) return;
    item.value = String(value ?? "");
    if (item._state) item._state.value = item.value;
}

function storeConversation(node, state) {
    setStoredValue(node, "conversation_history", JSON.stringify(state));
}

function resetConversation(node) {
    const state = {
        version: 1,
        skill: String(widget(node, "skill")?.value || ""),
        model: String(widget(node, "llm_model")?.value || ""),
        turns: [],
    };
    storeConversation(node, state);
    setStoredValue(node, "committed_result", "");
    setStoredValue(node, "conversation_action", "send");
    renderConversation(node);
    node.graph?.change?.();
}

function undoConversation(node) {
    const state = conversationState(node);
    state.turns.pop();
    storeConversation(node, state);
    const previous = state.turns.at(-1)?.assistant || "";
    if (widget(node, "output_mode")?.value === "实时输出") setStoredValue(node, "committed_result", previous);
    renderConversation(node);
    node.graph?.change?.();
}

function queueConversation(node, action = "send") {
    setStoredValue(node, "conversation_action", action);
    node.graph?.change?.();
    Promise.resolve(app.queuePrompt?.(0, 1)).catch((error) => {
        setStoredValue(node, "conversation_action", "send");
        console.error("[AI_Qwen_skill] queue failed", error);
    });
}

function renderConversation(node) {
    const state = conversationState(node);
    const turns = state.turns.slice(-12);
    if (node.__aptQwenSkillHistoryToggle) {
        node.__aptQwenSkillHistoryToggle.textContent = `对话记录（${turns.length}轮）`;
    }
    if (!node.__aptQwenSkillTranscript) return;
    const expanded = Boolean(node.properties?.apt_qwen_skill_history_open);
    node.__aptQwenSkillTranscript.hidden = !expanded;
    node.__aptQwenSkillTranscript.textContent = "";
    for (const turn of turns) {
        const user = document.createElement("div");
        user.className = "apt-qwen-skill-turn is-user";
        user.textContent = `你：${String(turn.user || "")}`;
        const assistant = document.createElement("div");
        assistant.className = "apt-qwen-skill-turn is-assistant";
        assistant.textContent = `模型：${String(turn.assistant || "")}`;
        node.__aptQwenSkillTranscript.append(user, assistant);
    }
    if (!turns.length) {
        const empty = document.createElement("div");
        empty.className = "apt-qwen-skill-history-empty";
        empty.textContent = "暂无对话记录";
        node.__aptQwenSkillTranscript.append(empty);
    }
    node.__aptQwenSkillDomWidget?.afterResize?.();
    node.setDirtyCanvas?.(true, true);
}

function conversationControls(node) {
    const controls = document.createElement("div");
    controls.className = "apt-qwen-skill-controls";
    const button = (label, callback, className = "") => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = `apt-qwen-skill-button ${className}`.trim();
        item.textContent = label;
        item.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            callback();
        });
        return item;
    };
    controls.append(button("发送", () => queueConversation(node, "send"), "is-primary"));
    controls.append(button("采用本轮", () => queueConversation(node, "adopt")));
    controls.append(button("新对话", () => resetConversation(node)));
    controls.append(button("撤回", () => undoConversation(node)));

    const mode = document.createElement("select");
    mode.className = "apt-qwen-skill-mode";
    for (const value of ["实时输出", "仅对话"]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        mode.append(option);
    }
    mode.value = String(widget(node, "output_mode")?.value || "实时输出");
    mode.addEventListener("change", () => {
        setStoredValue(node, "output_mode", mode.value);
        node.graph?.change?.();
    });
    controls.append(mode);

    const toggle = button("对话记录（0轮）", () => {
        node.properties ||= {};
        const expanded = !node.properties.apt_qwen_skill_history_open;
        node.properties.apt_qwen_skill_history_open = expanded;
        const width = Number(node.size?.[0] || 320);
        const height = Number(node.size?.[1] || 260);
        if (expanded) {
            node.properties.apt_qwen_skill_collapsed_height = height;
            node.setSize?.([width, height + 185]);
        } else {
            const collapsed = Number(node.properties.apt_qwen_skill_collapsed_height || 0);
            if (collapsed > 0) node.setSize?.([width, collapsed]);
        }
        renderConversation(node);
        node.graph?.change?.();
    }, "is-history");
    node.__aptQwenSkillHistoryToggle = toggle;
    controls.append(toggle);
    return controls;
}

function installEditor(node) {
    if (node.__aptQwenSkillEditor) return;
    const textWidget = widget(node, "text");
    if (!textWidget) return;
    hideWidget(node, "text");

    const wrap = document.createElement("div");
    wrap.className = "apt-qwen-skill-compose";
    const editor = document.createElement("div");
    editor.className = "apt-qwen-skill-editor";
    editor.contentEditable = "true";
    editor.dataset.placeholder = "输入指令；输入 @ 可引用已连接素材";
    node.__aptQwenSkillEditor = editor;
    renderPlainText(node, textWidget.value || "");
    editor.addEventListener("input", () => {
        syncTextWidget(node);
        const query = currentMention(editor);
        if (query == null) closeMenu(node);
        else openMenu(node, query);
    });
    editor.addEventListener("keyup", () => {
        const query = currentMention(editor);
        if (query != null) openMenu(node, query);
    });
    editor.addEventListener("blur", () => setTimeout(() => closeMenu(node), 120));
    const controls = conversationControls(node);
    const transcript = document.createElement("div");
    transcript.className = "apt-qwen-skill-transcript";
    node.__aptQwenSkillTranscript = transcript;
    wrap.append(editor, controls, transcript);
    const domWidget = node.addDOMWidget?.("apt_qwen_skill_editor", "customwidget", wrap, {
        getValue: () => editorText(editor),
        setValue: (value) => renderPlainText(node, value),
        serialize: false,
        getMinHeight: () => node.properties?.apt_qwen_skill_history_open ? 300 : 132,
        afterResize: () => {
            editor.style.maxHeight = node.properties?.apt_qwen_skill_history_open ? "220px" : "100%";
        },
    });
    if (domWidget) {
        domWidget.serialize = false;
        node.__aptQwenSkillDomWidget = domWidget;
    }
    node._widgetSlotsDirty = true;
    node.setSize?.([Number(node.size?.[0] || 320), Number(node.size?.[1] || 320)]);
    renderConversation(node);
}

function install(node) {
    migrateLegacyNode(node);
    links(node);
    migrateMediaInputs(node);
    for (const name of ["output_mode", "conversation_history", "committed_result", "conversation_action"]) {
        hideWidget(node, name);
    }
    const model = widget(node, "llm_model");
    if (model && !model.__aptQwenSkillBound) {
        model.__aptQwenSkillBound = true;
        const callback = model.callback;
        model.callback = function modelChanged() {
            const result = callback?.apply(this, arguments);
            syncModel(node);
            resetConversation(node);
            return result;
        };
    }
    const skill = widget(node, "skill");
    if (skill && !skill.__aptQwenSkillConversationBound) {
        skill.__aptQwenSkillConversationBound = true;
        const callback = skill.callback;
        skill.callback = function skillChanged() {
            const result = callback?.apply(this, arguments);
            resetConversation(node);
            return result;
        };
    }
    const apiKey = widget(node, "api_key");
    const keyInput = apiKey?.inputEl || apiKey?.element;
    if (keyInput?.tagName === "INPUT") keyInput.type = "password";
    syncModel(node);
    installEditor(node);
}

function applyConversationMessage(node, message) {
    const raw = Array.isArray(message?.conversation) ? message.conversation[0] : message?.conversation;
    if (!raw) return;
    let payload;
    try {
        payload = typeof raw === "string" ? JSON.parse(raw) : raw;
    } catch {
        return;
    }
    if (!payload || typeof payload !== "object") return;
    if (payload.history) storeConversation(node, payload.history);
    if (Object.hasOwn(payload, "committed_result")) setStoredValue(node, "committed_result", payload.committed_result);
    const action = String(widget(node, "conversation_action")?.value || "send");
    setStoredValue(node, "conversation_action", "send");
    if (!payload.error && payload.answer && action === "send") {
        setStoredValue(node, "text", "");
        renderPlainText(node, "");
    }
    node.__aptQwenSkillLastError = String(payload.error || "");
    renderConversation(node);
    node.graph?.change?.();
}

function pruneTransportInputs(nodeData) {
    const optional = nodeData?.input?.optional;
    if (!optional) return;
    for (const name of Object.keys(optional)) {
        if (/^media_\d+$/.test(name)) delete optional[name];
    }
}

function patchPrompt() {
    if (app.__aptQwenSkillPromptPatched || typeof app.graphToPrompt !== "function") return;
    app.__aptQwenSkillPromptPatched = true;
    const original = app.graphToPrompt;
    app.graphToPrompt = async function graphToPromptWithQwenSkillMedia() {
        const data = await original.apply(this, arguments);
        const output = data?.output || {};
        for (const node of app.graph?._nodes || []) {
            if (nodeClass(node) !== NODE_CLASS) continue;
            const promptNode = output[String(node.id)];
            if (!promptNode) continue;
            promptNode.inputs ||= {};
            delete promptNode.inputs.media;
            for (let index = 1; index <= MEDIA_LIMIT; index += 1) delete promptNode.inputs[`media_${index}`];
            expandedSources(node).forEach((item, index) => {
                if (!output[String(item.source_id)]) return;
                promptNode.inputs[`media_${index + 1}`] = [String(item.source_id), Number(item.source_slot) || 0];
            });
        }
        return data;
    };
}

function connectionPoint(node, input, slot) {
    if (typeof node?.getConnectionPos === "function") return node.getConnectionPos(input, slot);
    const y = Number(node?.pos?.[1] || 0) + 32 + slot * 20;
    return [Number(node?.pos?.[0] || 0) + (input ? 0 : Number(node?.size?.[0] || 160)), y];
}

function officialLinkMidpoint(canvas, start, target) {
    const point = canvas?.computeConnectionPoint?.(start, target, 0.5, globalThis.LiteGraph?.RIGHT, globalThis.LiteGraph?.LEFT);
    return point && Number.isFinite(point[0]) && Number.isFinite(point[1]) ? point : [(start[0] + target[0]) / 2, (start[1] + target[1]) / 2];
}

function linkMarker(node, item, index, canvas = app.canvas) {
    const graph = node.graph || app.graph;
    const source = graph?.getNodeById?.(Number(item.source_id));
    const inputIndex = (node.inputs || []).findIndex((input) => String(input?.name || "") === "media");
    if (!source || inputIndex < 0) return null;
    const start = connectionPoint(source, false, Number(item.source_slot) || 0);
    const target = connectionPoint(node, true, inputIndex);
    const midpoint = officialLinkMidpoint(canvas, start, target);
    return {
        node,
        index,
        source,
        start,
        target,
        x: midpoint[0],
        y: midpoint[1],
    };
}

function drawLinks(canvas, ctx) {
    const graph = canvas?.graph || app.graph;
    if (canvas.links_render_mode === globalThis.LiteGraph?.HIDDEN_LINK) return;
    for (const node of graph?._nodes || []) {
        if (nodeClass(node) !== NODE_CLASS) continue;
        const inputIndex = (node.inputs || []).findIndex((item) => String(item?.name || "") === "media");
        if (inputIndex < 0) continue;
        links(node).forEach((item, index) => {
            const geometry = linkMarker(node, item, index, canvas);
            if (!geometry) return;
            const type = String(item.source_type || "").toUpperCase();
            const colors = globalThis.LGraphCanvas?.link_type_colors || {};
            const color = colors[type] || colors[type.split(",")[0]] || "#7ec8ff";
            ctx.save();
            canvas.renderLink(ctx, geometry.start, geometry.target, null, false, false, color, globalThis.LiteGraph?.RIGHT, globalThis.LiteGraph?.LEFT);
            const deleting = Number(node.__aptQwenSkillDeleteIndex) === index;
            ctx.beginPath();
            ctx.arc(geometry.x, geometry.y, 10, 0, Math.PI * 2);
            ctx.fillStyle = deleting ? "#e53935" : "rgba(24,24,24,.96)";
            ctx.fill();
            ctx.lineWidth = 2;
            ctx.strokeStyle = deleting ? "#ffb3ad" : color;
            ctx.stroke();
            ctx.fillStyle = "#fff";
            ctx.font = deleting ? "bold 15px system-ui,sans-serif" : "bold 11px system-ui,sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText(deleting ? "×" : String(index + 1), geometry.x, geometry.y + (deleting ? 0 : 0.5));
            ctx.restore();
        });
    }
}

function graphPosition(canvas, event) {
    const rect = canvas?.canvas?.getBoundingClientRect?.();
    const scale = Number(canvas?.ds?.scale || 1);
    const offset = canvas?.ds?.offset || [0, 0];
    if (!rect) return [0, 0];
    return [
        (Number(event.clientX) - rect.left) / scale - Number(offset[0] || 0),
        (Number(event.clientY) - rect.top) / scale - Number(offset[1] || 0),
    ];
}

function markerAt(canvas, x, y) {
    const graph = canvas?.graph || app.graph;
    for (let nodeIndex = (graph?._nodes?.length || 0) - 1; nodeIndex >= 0; nodeIndex -= 1) {
        const node = graph._nodes[nodeIndex];
        if (nodeClass(node) !== NODE_CLASS) continue;
        const values = links(node);
        for (let index = values.length - 1; index >= 0; index -= 1) {
            const marker = linkMarker(node, values[index], index, canvas);
            if (marker && Math.hypot(marker.x - x, marker.y - y) <= 14) return marker;
        }
    }
    return null;
}

function patchCanvas() {
    const canvas = app.canvas;
    if (!canvas || canvas.__aptQwenSkillCanvasPatched || typeof canvas.drawConnections !== "function") return;
    canvas.__aptQwenSkillCanvasPatched = true;
    const original = canvas.drawConnections;
    canvas.drawConnections = function drawConnectionsWithQwenSkill(ctx) {
        const result = original.apply(this, arguments);
        const drawContext = ctx || this.bgctx || this.ctx;
        const onConnectionLayer = drawContext?.canvas === this?.bgcanvas || drawContext === this?.bgctx || !this?.bgcanvas;
        if (drawContext && onConnectionLayer) drawLinks(this, drawContext);
        return result;
    };
    canvas.canvas?.addEventListener?.("pointerdown", (event) => {
        const [x, y] = graphPosition(canvas, event);
        const hit = markerAt(canvas, x, y);
        if (hit) {
            if (Number(hit.node.__aptQwenSkillDeleteIndex) === hit.index) removeSource(hit.node, hit.index);
            else {
                hit.node.__aptQwenSkillDeleteIndex = hit.index;
                hit.node.setDirtyCanvas?.(true, true);
            }
            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation?.();
            return;
        }
        for (const node of canvas.graph?._nodes || []) {
            if (nodeClass(node) !== NODE_CLASS || Number(node.__aptQwenSkillDeleteIndex) < 0) continue;
            node.__aptQwenSkillDeleteIndex = -1;
            node.setDirtyCanvas?.(true, true);
        }
    }, true);
}

function addStyles() {
    if (document.getElementById("apt-qwen-skill-styles")) return;
    const style = document.createElement("style");
    style.id = "apt-qwen-skill-styles";
    style.textContent = `
      .apt-qwen-skill-compose { display: flex; flex-direction: column; width: 100%; height: 100%; min-height: 124px; overflow: hidden; box-sizing: border-box; }
      .apt-qwen-skill-editor { flex: 1 1 auto; width: 100%; min-height: 80px; max-height: 100%; padding: 8px; overflow: auto; box-sizing: border-box; border: 1px solid rgba(255,255,255,.14); border-radius: 2px; outline: none; background: #202020; color: #ddd; white-space: pre-wrap; overflow-wrap: anywhere; font: 14px/1.5 Consolas,"Courier New",monospace; }
      .apt-qwen-skill-editor:empty::before { content: attr(data-placeholder); color: #777; pointer-events: none; }
      .apt-qwen-skill-controls { display: flex; flex: 0 0 auto; flex-wrap: wrap; gap: 4px; padding: 5px 0 0; }
      .apt-qwen-skill-button, .apt-qwen-skill-mode { height: 26px; padding: 0 7px; border: 1px solid rgba(255,255,255,.16); border-radius: 5px; background: #252525; color: #ccc; font: 11px/24px system-ui,sans-serif; cursor: pointer; }
      .apt-qwen-skill-button:hover, .apt-qwen-skill-mode:hover { background: #353535; }
      .apt-qwen-skill-button.is-primary { border-color: rgba(0,226,187,.42); color: #8fffe6; }
      .apt-qwen-skill-button.is-history { margin-left: auto; color: #ff9a40; }
      .apt-qwen-skill-mode { min-width: 72px; padding-right: 2px; }
      .apt-qwen-skill-transcript { flex: 0 0 auto; max-height: 180px; margin-top: 5px; padding: 6px; overflow: auto; box-sizing: border-box; border: 1px solid rgba(255,255,255,.12); border-radius: 5px; background: #181818; font: 12px/1.45 system-ui,sans-serif; }
      .apt-qwen-skill-transcript[hidden] { display: none !important; }
      .apt-qwen-skill-turn { margin: 0 0 6px; padding: 5px 7px; border-radius: 5px; white-space: pre-wrap; overflow-wrap: anywhere; }
      .apt-qwen-skill-turn.is-user { background: rgba(90,169,240,.12); color: #b9ddff; }
      .apt-qwen-skill-turn.is-assistant { background: rgba(255,136,34,.10); color: #e7e7e7; }
      .apt-qwen-skill-history-empty { padding: 8px; color: #777; text-align: center; }
      .apt-qwen-skill-chip { display: inline; color: #ff8822; white-space: nowrap; }
      .apt-qwen-skill-chip-thumb { display: inline-block; width: 22px; height: 22px; margin-right: 3px; object-fit: cover; border-radius: 4px; vertical-align: -6px; background: #5aa9f0; }
      .apt-qwen-skill-chip-thumb.is-menu { flex: 0 0 48px; width: 48px; height: 48px; margin-right: 9px; border-radius: 6px; vertical-align: middle; }
      .apt-qwen-skill-chip-thumb.is-video { position: relative; background: linear-gradient(135deg,#1557b8,#49b6ff); }
      .apt-qwen-skill-chip-thumb.is-video::after { content:""; position:absolute; left:6px; top:4px; border-left:6px solid #fff; border-top:4px solid transparent; border-bottom:4px solid transparent; }
      .apt-qwen-skill-chip-thumb.is-text { background: #8f78d8; }
      .apt-qwen-skill-menu { position: fixed; z-index: 10080; max-height: 300px; overflow-y: auto; padding: 6px; border: 1px solid #666; border-radius: 8px; background: #181818; color: #eee; box-shadow: 0 8px 28px #000a; }
      .apt-qwen-skill-menu-title { padding: 5px 8px; color: #aaa; font-size: 12px; }
      .apt-qwen-skill-menu-item { display: flex; align-items: center; justify-content: space-between; gap: 12px; min-height: 54px; padding: 6px 8px; border-radius: 5px; cursor: pointer; }
      .apt-qwen-skill-chip.is-menu { display: inline-flex; align-items: center; font-size: 16px; font-weight: 600; }
      .apt-qwen-skill-menu-item:hover { background: #3a3a3a; }
      .apt-qwen-skill-menu-detail { max-width: 65%; overflow: hidden; color: #999; text-overflow: ellipsis; white-space: nowrap; }
    `;
    document.head.append(style);
}

app.registerExtension({
    name: "AptPreset.AIQwenSkill",
    setup() {
        addStyles();
        patchPrompt();
        patchCanvas();
        window.addEventListener("apt-media-relay-change", () => {
            for (const node of app.graph?._nodes || []) {
                if (nodeClass(node) === NODE_CLASS) refreshEditor(node);
            }
        });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;
        pruneTransportInputs(nodeData);

        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreated() {
            const result = created?.apply(this, arguments);
            setTimeout(() => install(this), 0);
            return result;
        };

        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigure() {
            const result = configured?.apply(this, arguments);
            setTimeout(() => install(this), 0);
            return result;
        };

        const changed = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function onConnectionsChange(type, index, connected, linkInfo) {
            const result = changed?.apply(this, arguments);
            const input = this.inputs?.[index];
            if (connected && Number(type) === Number(globalThis.LiteGraph?.INPUT ?? 1) && String(input?.name || "") === "media") {
                setTimeout(() => captureConnection(this, linkInfo), 0);
            }
            return result;
        };

        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function onExecuted(message) {
            const result = executed?.apply(this, arguments);
            applyConversationMessage(this, message);
            return result;
        };

        const menu = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function getExtraMenuOptions(_, options) {
            const result = menu?.apply(this, arguments);
            if (links(this).length) {
                options.push(null, {
                    content: "清空 Media 素材",
                    callback: () => {
                        this.properties[LINKS_PROP] = [];
                        refreshEditor(this);
                        this.graph?.change?.();
                        this.setDirtyCanvas?.(true, true);
                    },
                });
            }
            return result;
        };

        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function onRemoved() {
            closeMenu(this);
            return removed?.apply(this, arguments);
        };
    },
});
