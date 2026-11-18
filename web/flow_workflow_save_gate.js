import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "flow_workflow_save_gate";
const SELECT_DIRECTORY_ENDPOINT = "/apt_preset/flow_workflow_save_gate/select_directory";

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name) || null;
}

function installDirectoryButton(node) {
    if (node.__aptWorkflowSaveDirectoryButton) return;
    const button = node.addWidget?.("button", "选择保存工作流文件夹", null, async () => {
        if (button.disabled) return;
        button.disabled = true;
        const originalLabel = button.name;
        button.name = "正在选择…";
        node.setDirtyCanvas?.(true, true);
        try {
            const directoryWidget = widget(node, "save_directory");
            const response = await api.fetchApi(SELECT_DIRECTORY_ENDPOINT, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    initial_directory: String(directoryWidget?.value || ""),
                }),
            });
            const result = await response.json();
            if (!response.ok || result?.ok !== true) {
                throw new Error(result?.error || "无法打开文件夹选择窗口");
            }
            if (result.directory && directoryWidget) {
                directoryWidget.value = result.directory;
                directoryWidget.callback?.(result.directory);
                node.graph?.setDirtyCanvas?.(true, true);
            }
        } catch (error) {
            window.alert(`选择工作流保存文件夹失败：${error?.message || error}`);
        } finally {
            button.disabled = false;
            button.name = originalLabel;
            node.setDirtyCanvas?.(true, true);
        }
    });
    if (!button) return;
    button.serialize = false;
    node.__aptWorkflowSaveDirectoryButton = button;
}

app.registerExtension({
    name: "AptPreset.flowWorkflowSaveGate",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onWorkflowSaveGateCreated() {
            const result = originalCreated?.apply(this, arguments);
            installDirectoryButton(this);
            return result;
        };
        const originalConfigured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onWorkflowSaveGateConfigured() {
            const result = originalConfigured?.apply(this, arguments);
            installDirectoryButton(this);
            return result;
        };
    },
});
