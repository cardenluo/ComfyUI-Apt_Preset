import { app } from "../../scripts/app.js";

const MEDIA_RELAY_CLASS = "basicIn_media";
const MEDIA_UNPACK_CLASS = "basicIn_media_unpack";
const MEDIA_EDITOR_CLASS = "AD_Media_editor";
const MEDIA_UNION = "IMAGE,VIDEO,AUDIO,LATENT,STRING,ARRAY";
const MEDIA_UNPACK_MAX_INPUTS = 64;
const LINKS_PROP = "apt_media_hub_links";
const STAGE_PROMPT_DOCS_PROP = "ad_minimax_guide_stage_prompt_docs";
const CHANGE_EVENT = "apt-media-relay-change";
const LINK_BORDER = "rgba(0, 0, 0, 0.72)";
const BATCH_IMAGE_NODE_CLASS = "BatchImagesNode";

function nodeClass(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "");
}

function isMediaRelay(node) {
    return [MEDIA_RELAY_CLASS, MEDIA_EDITOR_CLASS].includes(nodeClass(node));
}

function isMediaUnpack(node) {
    return nodeClass(node) === MEDIA_UNPACK_CLASS;
}

function ensureLinks(node) {
    node.properties ||= {};
    if (!Array.isArray(node.properties[LINKS_PROP])) node.properties[LINKS_PROP] = [];
    return node.properties[LINKS_PROP];
}

function graphLink(graph, linkId) {
    if (!graph || linkId == null) return null;
    for (const links of [graph.links, graph._links]) {
        if (!links) continue;
        if (typeof links.get === "function") {
            const link = links.get(linkId) ?? links.get(String(linkId));
            if (link) return link;
        }
        const link = links[linkId] ?? links[String(linkId)];
        if (link) return link;
    }
    return null;
}

function sourceFromLink(node, linkInfo = null) {
    const graph = node.graph || app.graph;
    const input = node.inputs?.[0];
    const linkId = input?.link ?? linkInfo?.id ?? linkInfo?.link_id ?? linkInfo?.linkId;
    const link = graphLink(graph, linkId) || linkInfo;
    if (!link) return null;
    const sourceId = Number(link.origin_id ?? link.originId ?? link.from_id ?? link.fromId);
    const sourceSlot = Number(link.origin_slot ?? link.originSlot ?? link.from_slot ?? link.fromSlot ?? 0) || 0;
    const sourceNode = graph?.getNodeById?.(sourceId);
    if (!sourceNode || sourceNode === node || isMediaRelay(sourceNode)) return null;
    const output = sourceNode.outputs?.[sourceSlot] || {};
    return {
        source_id: sourceId,
        source_slot: sourceSlot,
        source_type: String(output.type || output.datatype || link.type || "*").toUpperCase(),
    };
}

function notifyChanged(node, removed = false) {
    window.dispatchEvent(new CustomEvent(CHANGE_EVENT, {
        detail: { nodeId: Number(node?.id), removed },
    }));
}

function refreshNode(node) {
    const count = ensureLinks(node).length;
    if (nodeClass(node) === MEDIA_RELAY_CLASS) node.title = count ? `Media (${count})` : "Media";
    const input = node.inputs?.[0];
    if (input) {
        input.name = "media";
        input.label = count ? `media · ${count}` : "media";
        input.type = MEDIA_UNION;
    }
    const output = node.outputs?.[0];
    if (output) {
        output.name = "media";
        output.label = "media";
        output.type = "*";
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function removeLink(node, index) {
    const links = ensureLinks(node);
    if (index < 0 || index >= links.length) return false;
    links.splice(index, 1);
    refreshNode(node);
    app.graph?.change?.();
    notifyChanged(node);
    return true;
}

function captureConnection(node, linkInfo = null) {
    if (!node || node.__aptMediaRelayClearing) return false;
    const source = sourceFromLink(node, linkInfo);
    if (!source) return false;
    const links = ensureLinks(node);
    const sourceNode = (node.graph || app.graph)?.getNodeById?.(source.source_id);
    const sourceMediaType = mediaType(source, sourceNode);
    const exists = links.some((item) => Number(item.source_id) === source.source_id && Number(item.source_slot) === source.source_slot);
    const duplicateBatch = sourceMediaType === "batch" && links.some((item) => {
        const itemNode = (node.graph || app.graph)?.getNodeById?.(Number(item.source_id));
        return mediaType(item, itemNode) === "batch";
    });
    if (!exists && !duplicateBatch) links.push(source);

    node.__aptMediaRelayClearing = true;
    try {
        if (node.inputs?.[0]?.link != null && typeof node.disconnectInput === "function") node.disconnectInput(0);
        if (node.inputs?.[0]) node.inputs[0].link = null;
    } finally {
        node.__aptMediaRelayClearing = false;
    }
    refreshNode(node);
    app.graph?.change?.();
    notifyChanged(node);
    return !exists && !duplicateBatch;
}

function captureSoon(node, linkInfo = null) {
    setTimeout(() => captureConnection(node, linkInfo), 0);
    if (!linkInfo) setTimeout(() => captureConnection(node), 50);
}

function connectionPosition(node, isInput, slot) {
    const direct = isInput ? node?.getInputPos?.(slot) : node?.getOutputPos?.(slot);
    if (Array.isArray(direct)) return direct;
    const out = [0, 0];
    try {
        if (typeof node?.getConnectionPos === "function") return node.getConnectionPos(isInput, slot, out) || out;
    } catch {
        // Fall back to stable LiteGraph geometry.
    }
    return [Number(node?.pos?.[0] || 0) + (isInput ? 0 : Number(node?.size?.[0] || 160)), Number(node?.pos?.[1] || 0) + 40 + slot * 20];
}

function curveMidpoint(source, target) {
    const cp1 = [source[0] + 80, source[1]];
    const cp2 = [target[0] - 80, target[1]];
    return [
        0.125 * source[0] + 0.375 * cp1[0] + 0.375 * cp2[0] + 0.125 * target[0],
        0.125 * source[1] + 0.375 * cp1[1] + 0.375 * cp2[1] + 0.125 * target[1],
    ];
}

function linkGeometry(hub, link) {
    const graph = hub.graph || app.graph;
    const sourceNode = graph?.getNodeById?.(Number(link.source_id));
    if (!sourceNode) return null;
    const source = connectionPosition(sourceNode, false, Number(link.source_slot) || 0);
    const target = connectionPosition(hub, true, 0);
    return { sourceNode, source, target, mid: curveMidpoint(source, target) };
}

function linkColor(canvas, link) {
    const colors = globalThis.LGraphCanvas?.link_type_colors || {};
    const type = String(link?.source_type || "");
    return colors[type] || colors[type.toUpperCase()] || canvas.default_link_color || globalThis.LiteGraph?.LINK_COLOR || "#9A9";
}

function mediaType(link, sourceNode) {
    const type = String(link?.source_type || "").toUpperCase();
    const sourceClass = nodeClass(sourceNode);
    if (type.includes("IMAGE") && sourceClass === BATCH_IMAGE_NODE_CLASS) return "image_batch";
    if (type === "ARRAY" && sourceClass === "text_MinimaxH3") return "batch";
    if (type.includes("LATENT")) return "latent";
    if (type.includes("AUDIO")) return "audio";
    if (type.includes("VIDEO")) return "video";
    if (type.includes("IMAGE")) return "image";
    if (type.includes("STRING")) return "text";
    return "image";
}

function relaySource(node) {
    const input = node?.inputs?.find?.((item) => String(item?.name || "").toLowerCase() === "media");
    const link = graphLink(node?.graph || app.graph, input?.link);
    if (!link) return null;
    const source = (node?.graph || app.graph)?.getNodeById?.(Number(link.origin_id ?? link.originId));
    return isMediaRelay(source) ? source : null;
}

function editorPromptTexts(node) {
    if (nodeClass(node) !== MEDIA_EDITOR_CLASS) return [];
    const docs = node?.properties?.[STAGE_PROMPT_DOCS_PROP];
    if (!Array.isArray(docs)) return [];
    const names = { image: "Picture", video: "Video", audio: "Audio" };
    return docs.map((doc) => {
        if (!Array.isArray(doc?.parts)) return String(doc?.text || "");
        return doc.parts.map((part) => {
            if (part?.type === "dialogue") return `<d>${String(part.text || "")}</d>`;
            if (part?.type !== "mention") return String(part?.text || "");
            if (part.textTag) return String(part.token || part.label || "");
            const kind = String(part.mediaType || "image").toLowerCase();
            const ordinal = Math.max(1, Number(part.ordinal) || 1);
            return names[kind] ? `<${names[kind]} ${ordinal}>` : String(part.token || "");
        }).join("");
    });
}

let graphToPromptPatched = false;

function patchGraphToPrompt() {
    if (graphToPromptPatched || typeof app.graphToPrompt !== "function") return;
    graphToPromptPatched = true;
    const original = app.graphToPrompt;
    app.graphToPrompt = async function graphToPromptWithMediaUnpack() {
        const promptData = await original.apply(this, arguments);
        const output = promptData?.output || {};
        for (const node of app.graph?._nodes || []) {
            if (!isMediaUnpack(node)) continue;
            const promptNode = output[String(node.id)];
            const relay = relaySource(node);
            if (!promptNode || !relay) continue;
            promptNode.inputs ||= {};
            delete promptNode.inputs.media;
            for (let index = 1; index <= MEDIA_UNPACK_MAX_INPUTS; index += 1) {
                delete promptNode.inputs[`media_${index}`];
                delete promptNode.inputs[`media_type_${index}`];
            }
            normalizeLinks(relay);
            const editor = nodeClass(relay) === MEDIA_EDITOR_CLASS;
            const links = ensureLinks(relay)
                .filter((link) => {
                    if (!editor) return true;
                    const sourceNode = app.graph?.getNodeById?.(Number(link.source_id));
                    return !["text", "batch"].includes(mediaType(link, sourceNode));
                })
                .slice(0, MEDIA_UNPACK_MAX_INPUTS);
            links.forEach((link, index) => {
                if (!output[String(link.source_id)]) return;
                const sourceNode = app.graph?.getNodeById?.(Number(link.source_id));
                const kind = mediaType(link, sourceNode);
                promptNode.inputs[`media_${index + 1}`] = [String(link.source_id), Number(link.source_slot) || 0];
                promptNode.inputs[`media_type_${index + 1}`] = kind;
            });
            const texts = editorPromptTexts(relay);
            if (texts.length && links.length < MEDIA_UNPACK_MAX_INPUTS) {
                const index = links.length + 1;
                promptNode.inputs[`media_${index}`] = texts
                    .map((text, textIndex) => `#segment${textIndex + 1}----------\n${text}`)
                    .join("\n");
                promptNode.inputs[`media_type_${index}`] = "batch";
            }
        }
        return promptData;
    };
}

function pruneUnpackTransportInputs(nodeData) {
    const optional = nodeData?.input?.optional;
    if (!optional) return;
    for (const name of Object.keys(optional)) {
        if (/^media_(?:type_)?\d+$/.test(name)) delete optional[name];
    }
}

function normalizeLinks(node) {
    const graph = node.graph || app.graph;
    const links = ensureLinks(node);
    let hasBatch = false;
    const filtered = links.filter((link) => {
        const sourceNode = graph?.getNodeById?.(Number(link.source_id));
        if (!sourceNode) return false;
        if (mediaType(link, sourceNode) !== "batch") return true;
        if (hasBatch) return false;
        hasBatch = true;
        return true;
    });
    if (filtered.length === links.length) return false;
    node.properties[LINKS_PROP] = filtered;
    refreshNode(node);
    notifyChanged(node);
    return true;
}

function drawRelayLinks(canvas, ctx) {
    const graph = canvas?.graph || app.graph;
    if (!graph?._nodes || canvas.links_render_mode === globalThis.LiteGraph?.HIDDEN_LINK) return;
    for (const hub of graph._nodes) {
        if (!isMediaRelay(hub)) continue;
        normalizeLinks(hub);
        const counts = { image: 0, video: 0, audio: 0, latent: 0, text: 0, batch: 0, image_batch: 0 };
        ensureLinks(hub).forEach((link) => {
            const geometry = linkGeometry(hub, link);
            if (!geometry) return;
            const type = mediaType(link, geometry.sourceNode);
            counts[type] = (counts[type] || 0) + 1;
            const width = canvas.connections_width || 3;
            ctx.save();
            ctx.beginPath();
            ctx.moveTo(geometry.source[0], geometry.source[1]);
            ctx.bezierCurveTo(geometry.source[0] + 80, geometry.source[1], geometry.target[0] - 80, geometry.target[1], geometry.target[0], geometry.target[1]);
            ctx.lineWidth = width + 4;
            ctx.strokeStyle = LINK_BORDER;
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(geometry.source[0], geometry.source[1]);
            ctx.bezierCurveTo(geometry.source[0] + 80, geometry.source[1], geometry.target[0] - 80, geometry.target[1], geometry.target[0], geometry.target[1]);
            ctx.lineWidth = width;
            ctx.strokeStyle = linkColor(canvas, link);
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(geometry.mid[0], geometry.mid[1], 9, 0, Math.PI * 2);
            ctx.fillStyle = "#e53935";
            ctx.fill();
            const textLink = type === "text";
            if (textLink) {
                ctx.beginPath();
                ctx.moveTo(geometry.mid[0] - 3.5, geometry.mid[1] - 3.5);
                ctx.lineTo(geometry.mid[0] + 3.5, geometry.mid[1] + 3.5);
                ctx.moveTo(geometry.mid[0] + 3.5, geometry.mid[1] - 3.5);
                ctx.lineTo(geometry.mid[0] - 3.5, geometry.mid[1] + 3.5);
                ctx.lineWidth = 2;
                ctx.lineCap = "round";
                ctx.strokeStyle = "#ffffff";
                ctx.stroke();
            } else {
                ctx.fillStyle = "#ffffff";
                ctx.font = "bold 11px system-ui, sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(String(counts[type]), geometry.mid[0], geometry.mid[1] + 0.5);
            }
            ctx.restore();
        });
    }
}

function graphPosition(canvas, event) {
    const result = canvas?.convertEventToCanvasOffset?.(event);
    if (Array.isArray(result)) return result;
    const rect = canvas?.canvas?.getBoundingClientRect?.();
    const scale = Number(canvas?.ds?.scale || 1);
    const offset = canvas?.ds?.offset || [0, 0];
    return [
        (Number(event?.clientX || 0) - Number(rect?.left || 0)) / scale - Number(offset[0] || 0),
        (Number(event?.clientY || 0) - Number(rect?.top || 0)) / scale - Number(offset[1] || 0),
    ];
}

function hitTest(graph, x, y) {
    let best = null;
    for (const hub of graph?._nodes || []) {
        if (!isMediaRelay(hub)) continue;
        ensureLinks(hub).forEach((link, index) => {
            const geometry = linkGeometry(hub, link);
            if (!geometry) return;
            const distance = Math.hypot(x - geometry.mid[0], y - geometry.mid[1]);
            if (distance <= 18 && (!best || distance < best.distance)) best = { hub, index, point: geometry.mid, distance };
        });
    }
    return best;
}

function openDeleteMenu(canvas, hit, event) {
    const rect = canvas?.canvas?.getBoundingClientRect?.();
    const scale = Number(canvas?.ds?.scale || 1);
    const offset = canvas?.ds?.offset || [0, 0];
    const clientX = Number(rect?.left || 0) + (hit.point[0] + Number(offset[0] || 0)) * scale;
    const clientY = Number(rect?.top || 0) + (hit.point[1] + Number(offset[1] || 0)) * scale;
    const menuEvent = typeof PointerEvent === "function"
        ? new PointerEvent("pointerdown", { clientX: clientX + 8, clientY: clientY + 8, bubbles: true, cancelable: true })
        : new MouseEvent("mousedown", { clientX: clientX + 8, clientY: clientY + 8, bubbles: true, cancelable: true });
    if (globalThis.LiteGraph?.ContextMenu) {
        new globalThis.LiteGraph.ContextMenu([
            { content: "删除", callback: () => removeLink(hit.hub, hit.index) },
        ], { event: menuEvent });
    }
    event?.preventDefault?.();
    event?.stopPropagation?.();
    event?.stopImmediatePropagation?.();
}

function patchCanvas() {
    const canvas = app.canvas;
    if (!canvas || canvas.__aptMediaRelayPatched || typeof canvas.drawConnections !== "function") return;
    canvas.__aptMediaRelayPatched = true;
    const originalDraw = canvas.drawConnections;
    canvas.drawConnections = function drawConnectionsWithMediaRelay(ctx) {
        const result = originalDraw.apply(this, arguments);
        const connectionContext = ctx || this.bgctx || this.ctx;
        const onConnectionLayer = connectionContext?.canvas === this?.bgcanvas || connectionContext === this?.bgctx || !this?.bgcanvas;
        if (connectionContext && onConnectionLayer) drawRelayLinks(this, connectionContext);
        return result;
    };
    canvas.canvas?.addEventListener?.("pointerdown", (event) => {
        const [x, y] = graphPosition(canvas, event);
        const hit = hitTest(canvas.graph || app.graph, x, y);
        if (hit) openDeleteMenu(canvas, hit, event);
    }, true);
}

app.registerExtension({
    name: "Apt_Preset.MediaRelay",
    setup() {
        patchCanvas();
        patchGraphToPrompt();
        for (const delay of [100, 500, 1200]) setTimeout(patchGraphToPrompt, delay);
        window.addEventListener(CHANGE_EVENT, (event) => {
            const node = app.graph?.getNodeById?.(Number(event?.detail?.nodeId));
            if (isMediaRelay(node)) refreshNode(node);
        });
        for (const delay of [100, 500, 1200]) setTimeout(patchCanvas, delay);
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === MEDIA_UNPACK_CLASS) {
            pruneUnpackTransportInputs(nodeData);
            return;
        }
        if (![MEDIA_RELAY_CLASS, MEDIA_EDITOR_CLASS].includes(nodeData?.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onMediaRelayCreated() {
            const result = onNodeCreated?.apply(this, arguments);
            ensureLinks(this);
            refreshNode(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onMediaRelayConfigure() {
            const result = onConfigure?.apply(this, arguments);
            ensureLinks(this);
            refreshNode(this);
            setTimeout(() => notifyChanged(this), 0);
            return result;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function onMediaRelayConnectionChange(type, index, connected, linkInfo) {
            const result = onConnectionsChange?.apply(this, arguments);
            if (connected && Number(index) === 0 && Number(type) === Number(globalThis.LiteGraph?.INPUT ?? 1)) captureSoon(this, linkInfo);
            return result;
        };

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function getMediaRelayMenuOptions(_, options) {
            const result = getExtraMenuOptions?.apply(this, arguments);
            const links = ensureLinks(this);
            if (links.length) {
                const graph = this.graph || app.graph;
                links.forEach((link, index) => {
                    const source = graph?.getNodeById?.(Number(link.source_id));
                    options.push({
                        content: `删除 ${index + 1}: ${String(source?.title || source?.comfyClass || link.source_type || "Media")}`,
                        callback: () => removeLink(this, index),
                    });
                });
                options.push(null, {
                    content: "清空 Media 素材",
                    callback: () => {
                        this.properties[LINKS_PROP] = [];
                        refreshNode(this);
                        app.graph?.change?.();
                        notifyChanged(this);
                    },
                });
            }
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function onMediaRelayRemoved() {
            notifyChanged(this, true);
            return onRemoved?.apply(this, arguments);
        };
    },
});
