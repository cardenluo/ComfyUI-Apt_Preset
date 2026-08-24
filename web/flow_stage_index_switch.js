import { app } from "../../../scripts/app.js";

const STAGE_BEGIN = "flow_stage_begin";
const STAGE_SWITCH = "flow_stage_index_switch";

function nodeClass(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "");
}

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name) || null;
}

function hideInlineStageWidget(node) {
    const stageWidget = widget(node, "stage_index");
    if (!stageWidget || stageWidget.__aptStageInlineHidden) return;
    stageWidget.__aptStageInlineHidden = true;
    stageWidget.type = "hidden";
    stageWidget.computeSize = () => [0, -4];
    stageWidget.serializeValue = async () => stageWidget.value;
    node.setSize?.(node.computeSize?.() || node.size);
}

function inlineStageIndexes(promptData) {
    const graphNodes = app.graph?._nodes || [];
    const switches = graphNodes.filter((node) => nodeClass(node) === STAGE_SWITCH);
    if (!switches.length) return;

    const output = promptData?.output || {};
    const inlineSwitches = switches.filter((node) => {
        const promptNode = output[String(node.id)];
        if (!promptNode) return false;
        const stageInput = node.inputs?.find((input) => input?.name === "stage_index");
        return stageInput?.link == null && !Array.isArray(promptNode.inputs?.stage_index);
    });
    if (!inlineSwitches.length) return;

    const stageBegins = graphNodes.filter((node) => nodeClass(node) === STAGE_BEGIN);
    if (stageBegins.length !== 1) {
        throw new Error(
            `flow_阶段编号开关：隐式模式要求工作流中恰好有一个 flow_阶段开始，当前找到 ${stageBegins.length} 个。`,
        );
    }

    const currentIndex = Number(widget(stageBegins[0], "current_index")?.value);
    if (!Number.isInteger(currentIndex) || currentIndex < 1) {
        throw new Error("flow_阶段编号开关：无法读取 flow_阶段开始 的 current_index。");
    }

    for (const node of inlineSwitches) {
        const promptNode = output[String(node.id)];
        promptNode.inputs ||= {};
        promptNode.inputs.stage_index = currentIndex;
    }
}

app.registerExtension({
    name: "AptPreset.FlowStageIndexSwitchInline",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== STAGE_SWITCH) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function onNodeCreatedStageIndexSwitch() {
            const result = originalCreated?.apply(this, arguments);
            hideInlineStageWidget(this);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function onConfigureStageIndexSwitch() {
            const result = originalConfigure?.apply(this, arguments);
            hideInlineStageWidget(this);
            return result;
        };
    },

    async setup() {
        const originalGraphToPrompt = app.graphToPrompt;
        app.graphToPrompt = async function graphToPromptWithInlineStageIndex() {
            const promptData = await originalGraphToPrompt.apply(this, arguments);
            inlineStageIndexes(promptData);
            return promptData;
        };
    },
});
