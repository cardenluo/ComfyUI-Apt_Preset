import { app } from "../../../scripts/app.js";
import { api } from "../../../scripts/api.js";

function syncWidgetText(node, widget, text) {
	widget.value = text;
	if (widget.inputEl) {
		widget.inputEl.value = text;
	}
	if (typeof widget.callback === "function") {
		widget.callback(text);
	}
	if (typeof node.setDirtyCanvas === "function") {
		node.setDirtyCanvas(true, true);
	} else if (node.graph?.setDirtyCanvas) {
		node.graph.setDirtyCanvas(true, true);
	} else if (app.graph?.setDirtyCanvas) {
		app.graph.setDirtyCanvas(true, true);
	}
}

app.registerExtension({
	name: "Apt_Preset.view_Data_text",

	async setup() {
		// 兼容旧版自定义 WebSocket 事件
		api.addEventListener("view_Data_text_processed", function (event) {
			const nodeId = parseInt(event.detail.node);
			const widgetName = event.detail.widget;
			const text = event.detail.text;

			const node = app.graph?.nodes?.find(n => n.id === nodeId);
			if (!node) return;

			const widget = node.widgets?.find(w => w.name === widgetName);
			if (!widget) return;

			syncWidgetText(node, widget, text);
		});
	},

	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		const previewNodes = [
			"view_Data",
			"view_bridge_Text",
			"view_GetLength",
			"view_GetShape",
			"view_GetWidgetsValues",
			"view_node_Script",
		];

		if (!previewNodes.includes(nodeData.name)) return;

		const onExecuted = nodeType.prototype.onExecuted;
		nodeType.prototype.onExecuted = function (message) {
			onExecuted?.apply(this, arguments);

			// message 里的 key 对应后端 ui 字段（data / display / text 等）
			if (!message) return;

			const uiKeys = ["data", "display", "text", "output_keys"];
			for (const key of uiKeys) {
				if (message[key] != null) {
					const values = Array.isArray(message[key]) ? message[key] : [message[key]];
					const text = values.join("\n");

					const widget = this.widgets?.find(w => w.name === key);
					if (widget) {
						syncWidgetText(this, widget, text);
					}
				}
			}
		};
	},
});
