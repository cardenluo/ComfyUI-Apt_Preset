import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
function getOutputNodes(nodes) {
    return ((nodes === null || nodes === void 0 ? void 0 : nodes.filter((n) => {
        var _a;
        return (n.mode != LiteGraph.NEVER && ((_a = n.constructor.nodeData) === null || _a === void 0 ? void 0 : _a.output_node));
    })) || []);
}
async function queueOutputNodes(nodes) {
    const nodeIds = nodes.map(n => n.id);
    const originalQueuePrompt = api.queuePrompt;
    try {
        api.queuePrompt = async function(index, prompt, ...args) {
            if (prompt && prompt.output && nodeIds.length) {
                const oldOutput = prompt.output;
                let newOutput = {};
                function recursiveAddNodes(nodeId, oldOutput, newOutput) {
                    let currentId = nodeId;
                    let currentNode = oldOutput[currentId];
                    if (newOutput[currentId] == null) {
                        newOutput[currentId] = currentNode;
                        if (currentNode && currentNode.inputs) {
                            for (const inputValue of Object.values(currentNode.inputs)) {
                                if (Array.isArray(inputValue)) {
                                    recursiveAddNodes(String(inputValue[0]), oldOutput, newOutput);
                                }
                            }
                        }
                    }
                    return newOutput;
                }
                for (const queueNodeId of nodeIds) {
                    recursiveAddNodes(String(queueNodeId), oldOutput, newOutput);
                }
                prompt.output = newOutput;
            }
            return await originalQueuePrompt.call(this, index, prompt, ...args);
        };
        await app.queuePrompt(0);
    }
    catch (e) {
        console.error(`There was an error queuing nodes ${nodeIds}`, e);
    }
    finally {
        api.queuePrompt = originalQueuePrompt;
    }
}
function addQueueButtonToViewNodes(node) {
    // 定义所有需要添加按钮的节点类型
    const targetNodes = ["view_Mask_And_Img", "view_Data", "view_GetShape","view_GetLength","view_bridge_Text","IO_store_image","view_bridge_image","Apt_clear_cache","view_mulView"];
    const flowSchControlNode = "flow_sch_control";
    
    // 检查是否是普通视图节点
    if (targetNodes.includes(node.constructor.nodeData?.name)) {
        const queueButton = node.addWidget("button", "update", "queue", () => {
            if (node.constructor.nodeData?.output_node) {
                queueOutputNodes([node]);
            } else {
                app.queuePrompt(0);
            }
        });
        queueButton.options = { ...queueButton.options, class: "queue-button" };
    }
    // 检查是否是flow_sch_control节点
    else if (node.constructor.nodeData?.name === flowSchControlNode) {
        const queueButton = node.addWidget("button", "Run_total", "queue", async () => {
            // 获取节点的 total 参数值
            let total = 1;
            if (node.widgets) {
                const totalWidget = node.widgets.find(w => w.name === "total");
                if (totalWidget) {
                    total = totalWidget.value;
                }
            }
            
            // 批量运行 total 次
            for (let i = 0; i < total; i++) {
                if (node.constructor.nodeData?.output_node) {
                    await queueOutputNodes([node]);
                } else {
                    await app.queuePrompt(0);
                }
                
                // 每次运行后等待一小段时间，避免系统过载
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        });
        queueButton.options = { ...queueButton.options, class: "queue-button" };
    }
}
app.registerExtension({
    name: "AptPreset.ViewMaskAndImgQueueButton",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const targetNodes = ["view_Mask_And_Img", "view_Data", "view_GetShape", "view_GetLength","view_bridge_Text","IO_store_image","view_bridge_image","Apt_clear_cache","view_mulView", "flow_sch_control"];
        if (targetNodes.includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                addQueueButtonToViewNodes(this);
                return result;
            };
        }
    },
    async setup() {
        const targetNodes = ["view_Mask_And_Img", "view_Data", "view_GetShape","view_GetLength","view_bridge_Text","IO_store_image","view_bridge_image","Apt_clear_cache","view_mulView", "flow_sch_control"];
        app.graph._nodes.forEach(node => {
            if (targetNodes.includes(node.constructor.nodeData?.name)) {
                addQueueButtonToViewNodes(node);
            }
        });
    },
});
export { queueOutputNodes };
















