import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "IO_Media_Size";
const ASPECT_RATIOS = {
    "1:1（正方形）": [1, 1], "1:2（超长竖屏）": [1, 2], "2:1（超宽横屏）": [2, 1],
    "2:3（竖版照片）": [2, 3], "3:2（横版照片）": [3, 2],
    "3:4（竖版标准）": [3, 4], "4:3（横版标准）": [4, 3],
    "4:5（人像）": [4, 5], "5:4（横版人像）": [5, 4],
    "5:7（竖版照片）": [5, 7], "7:5（横版照片）": [7, 5],
    "9:16（竖屏）": [9, 16], "16:9（横屏）": [16, 9],
    "10:16（竖版宽屏）": [10, 16], "16:10（横版宽屏）": [16, 10],
    "9:21（超宽屏竖版）": [9, 21], "21:9（超宽屏）": [21, 9],
};

function pythonRound(value) {
    const lower = Math.floor(value);
    const fraction = value - lower;
    if (Math.abs(fraction - 0.5) <= Number.EPSILON * Math.max(1, Math.abs(value))) {
        return lower % 2 === 0 ? lower : lower + 1;
    }
    return Math.round(value);
}

function widgetValue(node, name, fallback) {
    return getWidget(node, name)?.value ?? fallback;
}

function getWidget(node, name) {
    const widget = (node.widgets || []).find((w) => w.name === name);
    if (widget && !Object.hasOwn(widget, "__originalComputeSize")) widget.__originalComputeSize = widget.computeSize;
    return widget;
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    widget.hidden = !visible;
    if (widget._state) widget._state.hidden = !visible;
    widget.options ||= {};
    widget.options.hidden = !visible;
    widget.computeSize = visible ? widget.__originalComputeSize : () => [0, -4];
}

function syncModeVisibility(node) {
    const mode = widgetValue(node, "size_mode", "指定宽高");
    setWidgetVisible(getWidget(node, "target_size"), ["长边缩放", "短边缩放", "按宽缩放", "按高缩放"].includes(mode));
    setWidgetVisible(getWidget(node, "width"), mode === "指定宽高");
    setWidgetVisible(getWidget(node, "height"), mode === "指定宽高");
    setWidgetVisible(getWidget(node, "aspect_ratio"), mode === "像素量 + 比例");
    setWidgetVisible(getWidget(node, "megapixels"), mode === "像素量 + 比例");
    setWidgetVisible(getWidget(node, "resolution"), mode === "分辨率");
    const size = node.computeSize?.();
    if (size) node.setSize([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function calculateReferenceImageSize(node, sourceWidth, sourceHeight) {
    const mode = widgetValue(node, "size_mode", "指定宽高");
    let width = Number(widgetValue(node, "width", 1344));
    let height = Number(widgetValue(node, "height", 768));
    if (mode === "像素量 + 比例") {
        const ratio = ASPECT_RATIOS[widgetValue(node, "aspect_ratio", "16:9（横屏）")] || [16, 9];
        const scale = Math.sqrt(Number(widgetValue(node, "megapixels", 1.0)) * 1024 * 1024 / (ratio[0] * ratio[1]));
        width = ratio[0] * scale;
        height = ratio[1] * scale;
    } else if (mode === "分辨率") {
        const resolution = Number(widgetValue(node, "resolution", 1024));
        if (resolution) {
            const ratio = sourceWidth / sourceHeight;
            width = Math.sqrt(resolution * resolution * ratio);
            height = Math.sqrt(resolution * resolution / ratio);
        } else {
            width = sourceWidth;
            height = sourceHeight;
        }
    } else if (["长边缩放", "短边缩放", "按宽缩放", "按高缩放"].includes(mode)) {
        const target = Number(widgetValue(node, "target_size", 1024));
        const edge = mode === "长边缩放" ? Math.max(sourceWidth, sourceHeight)
            : mode === "短边缩放" ? Math.min(sourceWidth, sourceHeight)
            : mode === "按宽缩放" ? sourceWidth : sourceHeight;
        width = sourceWidth * target / edge;
        height = sourceHeight * target / edge;
    }
    const multiple = Math.max(1, Number(widgetValue(node, "size_multiple", 16)));
    width = Math.max(multiple, pythonRound(width / multiple) * multiple);
    height = Math.max(multiple, pythonRound(height / multiple) * multiple);
    const target = Number(widgetValue(node, "target_size", 1024));
    if (mode === "按宽缩放" || mode === "长边缩放" && sourceWidth >= sourceHeight || mode === "短边缩放" && sourceWidth < sourceHeight) width = target;
    if (mode === "按高缩放" || mode === "长边缩放" && sourceHeight > sourceWidth || mode === "短边缩放" && sourceHeight <= sourceWidth) height = target;
    return { width, height };
}

function defaultReferenceSourceSize(node) {
    const ratio = ASPECT_RATIOS[widgetValue(node, "aspect_ratio", "16:9（横屏）")] || [16, 9];
    return { source_width: ratio[0] * 1024, source_height: ratio[1] * 1024 };
}

function connectedSource(node, kind) {
    const input = node.inputs?.find((item) => item.name === kind);
    const link = input?.link != null ? app.graph?.links?.[input.link] : null;
    if (!link) return null;
    const source = app.graph?.getNodeById(link.origin_id);
    return source ? { source, slot: link.origin_slot } : null;
}

function loadMediaSize(node, kind, url, key) {
    const media = kind === "video" ? document.createElement("video") : new Image();
    if (kind === "video") media.preload = "metadata";
    const done = () => {
        if (node.__referenceMediaKeys?.[kind] !== key) return;
        const width = kind === "video" ? media.videoWidth : media.naturalWidth;
        const height = kind === "video" ? media.videoHeight : media.naturalHeight;
        if (width && height) {
            node.__referenceLiveSizes[kind] = { source_width: width, source_height: height };
            updateReferenceSizeDisplay(node);
        }
    };
    if (kind === "video") media.onloadedmetadata = done;
    else media.onload = done;
    media.src = url;
}

function syncConnectedSizes(node) {
    node.__referenceMediaKeys ||= {};
    node.__referenceLiveSizes ||= {};
    let changed = false;
    for (const kind of ["image", "video"]) {
        const connected = connectedSource(node, kind);
        const source = connected?.source;
        const preview = kind === "image" ? source?.imgs?.[0] : source?.video;
        const sourceType = source?.comfyClass || source?.type;
        const directLoader = kind === "image"
            ? sourceType === "LoadImage" || sourceType === "IO_load_mage"
            : sourceType === "LoadVideo";
        const fileWidget = kind === "video" ? "file" : "image";
        const file = directLoader ? source?.widgets?.find((widget) => widget.name === fileWidget)?.value : null;
        const key = source ? `${source.id}:${connected.slot}:${preview?.src || ""}:${file || ""}` : "";
        if (node.__referenceMediaKeys[kind] === key) continue;
        node.__referenceMediaKeys[kind] = key;
        delete node.__referenceLiveSizes[kind];
        changed = true;
        if (file && file !== "None") {
            const params = new URLSearchParams({ filename: String(file).replace(/ \[input\]$/, ""), type: "input" });
            loadMediaSize(node, kind, api.apiURL(`/view?${params}`), key);
            continue;
        }
        if (preview && !directLoader) {
            const width = kind === "video" ? preview.videoWidth : preview.naturalWidth;
            const height = kind === "video" ? preview.videoHeight : preview.naturalHeight;
            if (width && height) {
                node.__referenceLiveSizes[kind] = { source_width: width, source_height: height };
                continue;
            }
            if (preview.src) {
                loadMediaSize(node, kind, preview.src, key);
                continue;
            }
        }
    }
    if (changed) updateReferenceSizeDisplay(node);
}

function addReferenceSizeDisplay(node) {
    if (node.__referenceSizeDisplay) return node.__referenceSizeDisplay;
    const display = {
        name: "reference_size_result",
        type: "APT_REFERENCE_SIZE_RESULT",
        value: "端口连接后显示尺寸",
        options: { serialize: false },
        draw(ctx, _node, width, y, height) {
            const lines = String(this.value || "").split("\n");
            ctx.save();
            ctx.font = "600 13px Arial, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.lineJoin = "round";
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = "rgba(0, 0, 0, 0.68)";
            ctx.fillStyle = "#ff9a40";
            const lineHeight = 19;
            const startY = y + height * 0.5 - ((lines.length - 1) * lineHeight) / 2;
            lines.forEach((line, index) => {
                const lineY = startY + index * lineHeight;
                ctx.strokeText(line, width * 0.5, lineY);
                ctx.fillText(line, width * 0.5, lineY);
            });
            ctx.restore();
        },
        computeSize(width) {
            const lines = String(this.value || "").split("\n").length;
            return [width, Math.max(26, lines * 19 + 7)];
        },
        mouse() {
            return false;
        },
        async serializeValue() {
            return undefined;
        },
    };
    node.addCustomWidget(display);
    node.__referenceSizeDisplay = display;
    return display;
}

function updateReferenceSizeDisplay(node) {
    const display = addReferenceSizeDisplay(node);
    const sizes = node.__referenceLiveSizes || {};
    if (!sizes || (!sizes.image && !sizes.video)) {
        const source = defaultReferenceSourceSize(node);
        const output = calculateReferenceImageSize(node, source.source_width, source.source_height);
        display.value = `尺寸: ${output.width} × ${output.height}`;
        node.setDirtyCanvas?.(true, true);
        return;
    }

    const lines = [];
    if (sizes.image) {
        const output = calculateReferenceImageSize(node, sizes.image.source_width, sizes.image.source_height);
        lines.push(`image: ${output.width} × ${output.height}`);
    }
    if (sizes.video) {
        const output = calculateReferenceImageSize(node, sizes.video.source_width, sizes.video.source_height);
        lines.push(`video: ${output.width} × ${output.height}`);
    }
    display.value = lines.join("\n");
    const computed = node.computeSize?.();
    if (computed) node.setSize([Math.max(node.size?.[0] || 0, computed[0]), computed[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Apt_Preset.IO_Media_Size",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const self = this;
            addReferenceSizeDisplay(this);
            setTimeout(() => {
                syncModeVisibility(self);
                syncConnectedSizes(self);
            }, 0);
            this.__referenceSizeTimer = setInterval(() => {
                if (!self.graph) return;
                syncConnectedSizes(self);
            }, 1000);
            for (const name of ["size_mode", "target_size", "width", "height", "megapixels", "aspect_ratio", "resolution", "size_multiple"]) {
                const widget = getWidget(this, name);
                if (!widget) continue;
                const original = widget.callback;
                widget.callback = function () {
                    const r2 = original?.apply(this, arguments);
                    if (name === "size_mode") syncModeVisibility(self);
                    updateReferenceSizeDisplay(self);
                    return r2;
                };
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const self = this;
            setTimeout(() => {
                syncModeVisibility(self);
                syncConnectedSizes(self);
            }, 0);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function () {
            const result = onConnectionsChange?.apply(this, arguments);
            setTimeout(() => syncConnectedSizes(this), 0);
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            clearInterval(this.__referenceSizeTimer);
            return onRemoved?.apply(this, arguments);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = onExecuted?.apply(this, arguments);
            const sizes = message?.reference_sizes?.at(-1);
            if (sizes) {
                this.__referenceLiveSizes = sizes;
                updateReferenceSizeDisplay(this);
            }
            return result;
        };
    },
});
