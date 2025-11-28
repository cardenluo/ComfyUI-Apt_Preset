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
    const targetNodes = ["view_Mask_And_Img", "view_Data", "view_GetShape","view_GetLength","view_bridge_Text"];
    if (!targetNodes.includes(node.constructor.nodeData?.name)) {
        return;
    }
    const queueButton = node.addWidget("button", "update", "queue", () => {
        if (node.constructor.nodeData?.output_node) {
            queueOutputNodes([node]);
        } else {
            app.queuePrompt(0);
        }
    });
    queueButton.options = { ...queueButton.options, class: "queue-button" };
}
app.registerExtension({
    name: "AptPreset.ViewMaskAndImgQueueButton",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const targetNodes = ["view_Mask_And_Img", "view_Data", "view_GetShape", "view_GetLength","view_bridge_Text"];
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
        const targetNodes = ["view_Mask_And_Img", "view_Data", "view_GetShape","view_GetLength","view_bridge_Text"];
        app.graph._nodes.forEach(node => {
            if (targetNodes.includes(node.constructor.nodeData?.name)) {
                addQueueButtonToViewNodes(node);
            }
        });
    },
});
export { queueOutputNodes };
















