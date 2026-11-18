import { app } from "../../../scripts/app.js";

const STAGE_BEGIN = "flow_stage_begin";
const STAGE_END = "flow_stage_end";
const STAGE_UNPACK = "flow_stage_unpack";
const STAGE_ODD_EVEN_HOLD = "flow_stage_odd_even_hold";
const STAGE_SWITCH = "flow_stage_index_switch";
const FIXED_STAGE_PORTS = {
    [STAGE_BEGIN]: {
        inputs: [["initial_data_1", "*"], ["initial_data_2", "*"], ["initial_data_3", "*"]],
        outputs: [["stage_info", "FLOW_STAGE_INFO"], ["stage_index", "INT"]],
    },
    [STAGE_UNPACK]: {
        inputs: [["stage_info", "FLOW_STAGE_INFO"]],
        outputs: [
            ["stage_info", "FLOW_STAGE_INFO"],
            ["data1", "*"], ["data2", "*"], ["data3", "*"],
            ["total", "INT"], ["is_first", "BOOLEAN"], ["is_last", "BOOLEAN"],
            ["stage_info_mix", "FLOW_STAGE_INFO"],
            ["stage_info_data1", "FLOW_STAGE_INFO"],
            ["stage_info_data2", "FLOW_STAGE_INFO"],
            ["stage_info_data3", "FLOW_STAGE_INFO"],
        ],
    },
    [STAGE_END]: {
        inputs: [
            ["stage_info", "FLOW_STAGE_INFO"],
            ["data_1", "*"], ["data_2", "*"], ["data_3", "*"], ["blank", "*"],
        ],
        outputs: [["list_data1", "*"], ["list_data2", "*"], ["list_data3", "*"]],
    },
    [STAGE_ODD_EVEN_HOLD]: {
        inputs: [["stage_info", "FLOW_STAGE_INFO"], ["odd_data_1", "*"], ["even_data_2", "*"]],
        outputs: [["data_1", "*"], ["data_2", "*"]],
    },
};

function nodeClass(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "");
}

function widget(node, name) {
    return node?.widgets?.find((item) => item?.name === name) || null;
}

function graphLink(graph, linkId) {
    const links = graph?.links;
    if (Array.isArray(links)) {
        return links.find((link) => String(link?.id ?? link?.[0]) === String(linkId));
    }
    return links?.[linkId];
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

function normalizeStagePorts(node) {
    const spec = FIXED_STAGE_PORTS[nodeClass(node)];
    if (!spec) return false;
    let changed = false;
    for (const [slots, definitions] of [[node.inputs, spec.inputs], [node.outputs, spec.outputs]]) {
        definitions.forEach(([name, type], index) => {
            const slot = slots?.[index];
            if (!slot) return;
            for (const key of ["name", "label", "localized_name"]) {
                if (slot[key] !== name) {
                    slot[key] = name;
                    changed = true;
                }
            }
            if (slot.type !== type) {
                slot.type = type;
                changed = true;
            }
            const linkIds = Array.isArray(slot.links) ? slot.links : [slot.link];
            for (const linkId of linkIds) {
                if (linkId == null) continue;
                const link = graphLink(node.graph, linkId);
                if (Array.isArray(link) && link[5] !== type) {
                    link[5] = type;
                    changed = true;
                } else if (link && link.type !== type) {
                    link.type = type;
                    changed = true;
                }
            }
        });
    }
    if (changed) node.setDirtyCanvas?.(true, true);
    return changed;
}

function installFixedStagePorts(nodeType, nodeData) {
    if (!FIXED_STAGE_PORTS[nodeData?.name]) return;
    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function onNodeCreatedFixedStagePorts() {
        const result = originalCreated?.apply(this, arguments);
        normalizeStagePorts(this);
        setTimeout(() => normalizeStagePorts(this), 0);
        return result;
    };
    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function onConfigureFixedStagePorts() {
        const result = originalConfigure?.apply(this, arguments);
        normalizeStagePorts(this);
        return result;
    };
    const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function onConnectionsChangeFixedStagePorts() {
        const result = originalConnectionsChange?.apply(this, arguments);
        normalizeStagePorts(this);
        setTimeout(() => normalizeStagePorts(this), 0);
        return result;
    };
    const originalDrawForeground = nodeType.prototype.onDrawForeground;
    nodeType.prototype.onDrawForeground = function onDrawForegroundFixedStagePorts() {
        normalizeStagePorts(this);
        return originalDrawForeground?.apply(this, arguments);
    };
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

    const currentIndex = Number(widget(stageBegins[0], "stage_index")?.value);
    if (!Number.isInteger(currentIndex) || currentIndex < 0) {
        throw new Error("flow_阶段编号开关：无法读取 flow_阶段开始 的 stage_index。");
    }

    const zeroBasedIndex = currentIndex === 0 ? 0 : currentIndex - 1;
    for (const node of inlineSwitches) {
        const promptNode = output[String(node.id)];
        promptNode.inputs ||= {};
        promptNode.inputs.stage_index = zeroBasedIndex;
    }
}

app.registerExtension({
    name: "AptPreset.FlowStageIndexSwitchInline",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        installFixedStagePorts(nodeType, nodeData);
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
