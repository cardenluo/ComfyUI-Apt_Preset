import { app } from "../../scripts/app.js";

const NODE_CLASSES = new Set([
    "UC_create_context",
    "UC_load_model",
    "UC_ksampler",
    "UC_Ksampler_refine",
]);
const SAMPLING_WIDGETS = ["steps", "cfg", "sampler", "scheduler"];
const LATENT_SIZE_WIDGETS = ["ratio_selected", "batch_size", "width", "height"];
const SAMPLE_PARAMETERS_WIDGET = "sample_parameters";
const COLLAPSE_WIDGET = "collapse_sampling_parameters";
const LATENT_SIZE_WIDGET = "latent_size";

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget?.name === name) || null;
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    if (!widget.__aptSamplingSettingsOriginal) {
        widget.__aptSamplingSettingsOriginal = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
            optionsHidden: widget.options?.hidden,
            optionsCanvasOnly: widget.options?.canvasOnly,
        };
    }

    const original = widget.__aptSamplingSettingsOriginal;
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
}

function resizeNode(node) {
    const size = node.computeSize?.();
    if (size) node.setSize?.([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

function applySamplingParametersState(node, controlName) {
    const controlWidget = getWidget(node, controlName);
    if (!controlWidget) return;
    const visible = controlWidget.value !== false;
    for (const name of SAMPLING_WIDGETS) {
        setWidgetVisible(getWidget(node, name), visible);
    }
    resizeNode(node);
}

function installSamplingSettingsControl(node, nodeName) {
    const usesSourceSwitch = nodeName === "UC_ksampler" || nodeName === "UC_Ksampler_refine";
    const controlName = usesSourceSwitch ? SAMPLE_PARAMETERS_WIDGET : COLLAPSE_WIDGET;
    const controlWidget = getWidget(node, controlName);
    if (!controlWidget) return;
    if (controlWidget.value == null) controlWidget.value = false;
    if (!controlWidget.__aptSamplingParametersBound) {
        controlWidget.__aptSamplingParametersBound = true;
        const originalCallback = controlWidget.callback;
        controlWidget.callback = function samplingParametersChanged() {
            const result = originalCallback?.apply(this, arguments);
            applySamplingParametersState(node, controlName);
            node.graph?.setDirtyCanvas?.(true, true);
            return result;
        };
    }
    applySamplingParametersState(node, controlName);
}

function applyLatentSizeState(node) {
    const sizeWidget = getWidget(node, LATENT_SIZE_WIDGET);
    if (!sizeWidget) return;
    const visible = sizeWidget.value !== false;
    for (const name of LATENT_SIZE_WIDGETS) {
        setWidgetVisible(getWidget(node, name), visible);
    }
    resizeNode(node);
}

function installLatentSizeControl(node) {
    const sizeWidget = getWidget(node, LATENT_SIZE_WIDGET);
    if (!sizeWidget) return;
    if (sizeWidget.value == null) sizeWidget.value = false;
    if (!sizeWidget.__aptLatentSizeBound) {
        sizeWidget.__aptLatentSizeBound = true;
        const originalCallback = sizeWidget.callback;
        sizeWidget.callback = function latentSizeChanged() {
            const result = originalCallback?.apply(this, arguments);
            applyLatentSizeState(node);
            node.graph?.setDirtyCanvas?.(true, true);
            return result;
        };
    }
    applyLatentSizeState(node);
}

app.registerExtension({
    name: "AptPreset.basicUnitContext",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_CLASSES.has(nodeData?.name)) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onBasicUnitContextCreated() {
            const result = originalCreated?.apply(this, arguments);
            installSamplingSettingsControl(this, nodeData.name);
            if (nodeData.name === "UC_ksampler") installLatentSizeControl(this);
            return result;
        };

        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onBasicUnitContextConfigured() {
            const result = originalConfigured?.apply(this, arguments);
            installSamplingSettingsControl(this, nodeData.name);
            if (nodeData.name === "UC_ksampler") installLatentSizeControl(this);
            return result;
        };
    },
});
