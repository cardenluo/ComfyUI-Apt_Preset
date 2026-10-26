import { app } from "../../../scripts/app.js";

const MAX_LORAS = 20;

function setVisible(widget, visible) {
    if (!widget) return;
    if (!widget._loraRowOriginal) {
        widget._loraRowOriginal = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
            optionsHidden: widget.options?.hidden,
            optionsCanvasOnly: widget.options?.canvasOnly,
        };
    }

    const original = widget._loraRowOriginal;
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

function setGrayed(widget, grayed) {
    if (!widget) return;
    widget.disabled = grayed;
    if (widget._state) widget._state.disabled = grayed;
    const element = widget.inputEl || widget.element;
    if (element && "disabled" in element) element.disabled = grayed;
}

function updateRows(node) {
    const oldInput = node.inputs?.findIndex(input => input.name === "loras") ?? -1;
    if (oldInput >= 0) node.removeInput(oldInput);

    const widgets = new Map((node.widgets || []).map(widget => [widget.name, widget]));
    let lastSelected = 0;
    for (let i = 1; i <= MAX_LORAS; i++) {
        const name = widgets.get(`lora_name_${i}`)?.value;
        if (name && name !== "None") lastSelected = i;
    }

    const visibleRows = Math.min(MAX_LORAS, Math.max(1, lastSelected + 1));
    for (let i = 1; i <= MAX_LORAS; i++) {
        const visible = i <= visibleRows;
        for (const prefix of ["lora_name", "strength", "enabled"]) {
            setVisible(widgets.get(`${prefix}_${i}`), visible);
        }
        const grayed = widgets.get(`enabled_${i}`)?.value === false;
        setGrayed(widgets.get(`lora_name_${i}`), grayed);
        setGrayed(widgets.get(`strength_${i}`), grayed);
    }

    const size = node.computeSize?.();
    if (size) node.setSize?.([Math.max(node.size?.[0] || 0, size[0]), size[1]]);
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "AptPreset.sumLoraRows",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "sum_lora") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            for (let i = 1; i <= MAX_LORAS; i++) {
                for (const prefix of ["lora_name", "enabled"]) {
                    const widget = this.widgets?.find(item => item.name === `${prefix}_${i}`);
                    if (!widget) continue;
                    const callback = widget.callback;
                    widget.callback = (...args) => {
                        callback?.apply(widget, args);
                        updateRows(this);
                    };
                }
            }
            updateRows(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            updateRows(this);
            return result;
        };
    },
});
