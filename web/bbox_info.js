

import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Apt_Preset.Coordinate_fromMask",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "Coordinate_fromMask") {
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // 如果有 bbox 数据，则更新显示
                if (message.bbox) {
                    // 查找 bbox widget
                    const bboxWidget = this.widgets?.find(w => w.name === "bbox");
                    if (bboxWidget) {
                        // 更新 bbox 显示格式
                        bboxWidget.value = `BoundingBox(x=${message.bbox.x}, y=${message.bbox.y}, width=${message.bbox.width}, height=${message.bbox.height})`;
                    }
                }
            };
        }
    }
});