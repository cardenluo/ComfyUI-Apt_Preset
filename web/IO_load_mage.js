import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AptPreset.IO_load_mage",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_load_mage") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            const image = this.widgets?.find(widget => widget.name === "image");
            if (!image) return result;

            const updatePreview = () => {
                const values = image.options?.values;
                const empty = !image.value || image.value === "None" ||
                    (Array.isArray(values) && !values.includes(image.value));
                this.hideOutputImages = empty;
                if (empty) this.imgs = undefined;
                this.setDirtyCanvas?.(true, true);
                return empty;
            };
            const callback = image.callback;
            image.callback = function () {
                if (!updatePreview()) return callback?.apply(this, arguments);
            };
            updatePreview();
            return result;
        };
    },
});
