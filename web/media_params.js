import { app } from "../../scripts/app.js";

function setWidgetVisible(node, widget, visible) {
    if (!widget) return;

    if (!widget.__mediaParamsOriginal) {
        widget.__mediaParamsOriginal = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
            optionsHidden: widget.options?.hidden,
            optionsCanvasOnly: widget.options?.canvasOnly,
        };
    }

    const original = widget.__mediaParamsOriginal;
    widget.options ||= {};
    if (visible) {
        widget.type = original.type;
        widget.hidden = original.hidden ?? false;
        if (original.computeSize === undefined) delete widget.computeSize;
        else widget.computeSize = original.computeSize;
        if (original.optionsHidden === undefined) delete widget.options.hidden;
        else widget.options.hidden = original.optionsHidden;
        if (original.optionsCanvasOnly === undefined) delete widget.options.canvasOnly;
        else widget.options.canvasOnly = original.optionsCanvasOnly;
    } else {
        widget.type = "hidden";
        widget.hidden = true;
        widget.options.hidden = true;
        widget.options.canvasOnly = true;
        widget.computeSize = () => [0, -4];
    }

    if (widget.inputEl) widget.inputEl.style.display = visible ? "" : "none";
    if (widget.element) widget.element.style.display = visible ? "" : "none";
    if (widget._state) {
        widget._state.type = widget.type;
        widget._state.hidden = widget.hidden;
    }

    const size = node.computeSize?.();
    if (size) node.setSize([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

const VIDEO_MODES = new Set([
    "Hunyuan-Video",
    "Wan2.x",
    "LTX-2",
    "CogVideoX-1.5",
    "MiniMax-H3",
]);

const ASPECT_RATIOS = {
    "1:1（正方形）": [1, 1],
    "2:3（竖版照片）": [2, 3],
    "3:2（横版照片）": [3, 2],
    "3:4（竖版标准）": [3, 4],
    "4:3（横版标准）": [4, 3],
    "9:16（竖屏）": [9, 16],
    "16:9（横屏）": [16, 9],
    "21:9（超宽屏）": [21, 9],
};

const TEMPORAL_RULES = {
    "Hunyuan-Video": [4, 1],
    "Wan2.x": [4, 1],
    "LTX-2": [8, 1],
    "CogVideoX-1.5": [8, 1],
    "MiniMax-H3": [17, 5],
};

const CALCULATION_INPUTS = [
    "mode",
    "size_multiple",
    "aspect_ratio",
    "megapixels",
    "time_s",
    "fps",
];

function pythonRound(value) {
    const lower = Math.floor(value);
    const fraction = value - lower;
    if (Math.abs(fraction - 0.5) <= Number.EPSILON * Math.max(1, Math.abs(value))) {
        return lower % 2 === 0 ? lower : lower + 1;
    }
    return Math.round(value);
}

function widgetValue(node, name, fallback) {
    const value = node.widgets?.find((widget) => widget.name === name)?.value;
    return value ?? fallback;
}

function calculateValues(node) {
    const mode = String(widgetValue(node, "mode", "MiniMax-H3"));
    const multiple = Math.max(1, Number(widgetValue(node, "size_multiple", 32)) || 32);
    const ratio = ASPECT_RATIOS[widgetValue(node, "aspect_ratio", "1:1（正方形）")] || [1, 1];
    const megapixels = Number(widgetValue(node, "megapixels", 0.6)) || 0.6;
    const seconds = Number(widgetValue(node, "time_s", 5.0)) || 5.0;
    const fps = Number(widgetValue(node, "fps", 24.0)) || 24.0;
    const scale = Math.sqrt(megapixels * 1024 * 1024 / (ratio[0] * ratio[1]));
    const width = Math.max(multiple, pythonRound(ratio[0] * scale / multiple) * multiple);
    const height = Math.max(multiple, pythonRound(ratio[1] * scale / multiple) * multiple);
    let frameCount = Math.max(1, pythonRound(seconds * fps));
    const temporalRule = TEMPORAL_RULES[mode];
    let length = frameCount;
    if (temporalRule) {
        const [frameMultiple, frameOffset] = temporalRule;
        frameCount = Math.max(frameOffset, frameCount);
        length = frameCount + ((frameOffset - (frameCount % frameMultiple)) % frameMultiple + frameMultiple) % frameMultiple;
    }
    return { width, height, length };
}

function addResultDisplay(node) {
    if (node.__mediaParamsDisplay) return node.__mediaParamsDisplay;
    const display = {
        name: "calculated_media_params",
        type: "APT_MEDIA_PARAMS_RESULT",
        value: "",
        options: { serialize: false },
        draw(ctx, _node, width, y, height) {
            ctx.save();
            ctx.font = "600 13px Arial, sans-serif";
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.lineJoin = "round";
            ctx.lineWidth = 2.5;
            ctx.strokeStyle = "rgba(0, 0, 0, 0.68)";
            ctx.fillStyle = "#ff9a40";
            const centerY = y + height * 0.5;
            ctx.strokeText(this.value, width * 0.5, centerY);
            ctx.fillText(this.value, width * 0.5, centerY);
            ctx.restore();
        },
        computeSize(width) {
            return [width, 26];
        },
        mouse() {
            return false;
        },
        async serializeValue() {
            return undefined;
        },
    };
    node.addCustomWidget(display);
    node.__mediaParamsDisplay = display;
    return display;
}

function updateResultDisplay(node) {
    const display = addResultDisplay(node);
    const values = calculateValues(node);
    display.value = `${values.width} × ${values.height}     length:${values.length}`;
    node.setDirtyCanvas?.(true, true);
}

function syncModeWidgets(node) {
    const mode = node.widgets?.find((widget) => widget.name === "mode");
    const sizeMultiple = node.widgets?.find((widget) => widget.name === "size_multiple");
    const time = node.widgets?.find((widget) => widget.name === "time_s");
    const fps = node.widgets?.find((widget) => widget.name === "fps");
    const showVideoParams = VIDEO_MODES.has(mode?.value);

    setWidgetVisible(node, sizeMultiple, true);
    setWidgetVisible(node, time, showVideoParams);
    setWidgetVisible(node, fps, showVideoParams);
}

app.registerExtension({
    name: "AptPreset.MediaParams",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "basicIn_Media_Params") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            addResultDisplay(this);
            for (const name of CALCULATION_INPUTS) {
                const widget = this.widgets?.find((candidate) => candidate.name === name);
                if (!widget) continue;
                const callback = widget.callback;
                widget.callback = (...args) => {
                    const callbackResult = callback?.apply(widget, args);
                    if (name === "mode") syncModeWidgets(this);
                    updateResultDisplay(this);
                    return callbackResult;
                };
            }
            setTimeout(() => {
                syncModeWidgets(this);
                updateResultDisplay(this);
            }, 0);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => {
                syncModeWidgets(this);
                updateResultDisplay(this);
            }, 0);
            return result;
        };
    },
});
