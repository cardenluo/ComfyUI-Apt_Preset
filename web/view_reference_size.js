import { app } from "../../scripts/app.js";

const NODE_CLASS = "view_Reference_Size";
const QWEN_MODE = "qwenImage21_image";

function isOurNode(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "") === NODE_CLASS;
}

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetVisible(widget, visible) {
    if (!widget) return;
    widget.hidden = !visible;
    if (widget._state) widget._state.hidden = !visible;
}

function syncModeVisibility(node) {
    const typeWidget = getWidget(node, "input_type");
    if (!typeWidget) return;
    const isQwen = String(typeWidget.value) === QWEN_MODE;
    setWidgetVisible(getWidget(node, "generation_width"), !isQwen);
    setWidgetVisible(getWidget(node, "generation_height"), !isQwen);
    setWidgetVisible(getWidget(node, "resolution"), isQwen);
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "Apt_Preset.view_Reference_Size",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const onCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onCreated?.apply(this, arguments);
            const self = this;
            setTimeout(() => syncModeVisibility(self), 0);
            const typeWidget = getWidget(this, "input_type");
            if (typeWidget) {
                const original = typeWidget.callback;
                typeWidget.callback = function () {
                    const r2 = original?.apply(this, arguments);
                    syncModeVisibility(self);
                    return r2;
                };
            }
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            const self = this;
            setTimeout(() => syncModeVisibility(self), 0);
            return r;
        };
    },
});