import { app } from "../../scripts/app.js";

// sum_QwenImage2 专属：
//   1) latent_image 接入时灰化 width/height，断开时恢复手动尺寸。
//   2) 根据 ref_size_mode 控件的取值，灰化或启用 resolution widget。
//   3) 隐藏 minimax_guide_dual_ui40.js 公共资源里的「Image N→Picture N」预览行，
//      不动共享代码，只对本节点差异化处理。
// 其他 UI 行为（媒体台、stage_prompts 滚动、虚拟连线等）由 minimax_guide_dual_ui40.js 提供。

const NODE_CLASS = "sum_QwenImage2";
const LATENT_IMAGE_INPUT = "latent_image";
const MEDIA_EDITOR_CLASS = "AD_Media_editor";
const REF_SIZE_MODE_WIDGET = "ref_size_mode";
const REF_SIZE_MODE_ORIGINAL_VALUE = true; // True=各自原尺寸；False=统一分辨率
const RESOLUTION_WIDGET = "resolution";
const MAPPING_POLL_INTERVAL_MS = 250;
const MAPPING_POLL_MAX_LIFE_MS = 5 * 60 * 1000;
const STYLE_ELEMENT_ID = "sum-qwen-image2-style-overrides";

// 仅针对本节点注入的 CSS：约束 .ad-guide-prompt-editor-wrap 的最大宽度，
// 防止 mention chip / 文本编辑框把 wrap 撑出节点边界（minimax_guide_dual_ui40.js 的公共 CSS 不动）。
// 通过 .sum-qwen-image2-host 类挂在节点容器上做 scope。
function ensureNodeStyleOverride() {
    if (typeof document === "undefined") return;
    if (document.getElementById(STYLE_ELEMENT_ID)) return;
    const style = document.createElement("style");
    style.id = STYLE_ELEMENT_ID;
    style.textContent = `
      .sum-qwen-image2-host .ad-guide-prompt-editor-wrap {
        max-width: 100%;
        min-width: 0;
        box-sizing: border-box;
        overflow: hidden;
      }
      .sum-qwen-image2-host .ad-guide-prompt-editor,
      .sum-qwen-image2-host .ad-guide-material-tray {
        max-width: 100%;
        min-width: 0;
        box-sizing: border-box;
      }
    `;
    document.head?.appendChild(style) || document.documentElement.appendChild(style);
}

function tagNodeAsOurHost(node) {
    if (!node?.constructor?.nodeData) return false;
    const host = node?.graph?._nodes ? node : null;
    // LiteGraph 把 node 渲染成 .litegraph-node 容器；优先挂在这个 class 上做 scope
    const el = typeof node.getNodeElement === "function" ? node.getNodeElement() : null;
    const target = el || node?.element || null;
    if (target && !target.classList.contains("sum-qwen-image2-host")) {
        target.classList.add("sum-qwen-image2-host");
        return true;
    }
    return false;
}

function isOurNode(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "") === NODE_CLASS;
}

function nodeClass(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.nodeData?.name || "");
}

function getWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

function setWidgetGrayed(widget, grayed) {
    if (!widget) return;
    if (widget.disabled === grayed) return;
    widget.disabled = grayed;
    if (widget._state) widget._state.disabled = grayed;
    // 同步给 DOM input（如果已经渲染）
    const el = widget.inputEl || widget.element;
    if (el && "disabled" in el) el.disabled = grayed;
}

function inputHasLink(input) {
    if (!input) return false;
    return Array.isArray(input.link) ? input.link.some((id) => id != null) : input.link != null;
}

function graphLinkValues(graph) {
    const raw = graph?.links ?? graph?._links;
    if (raw instanceof Map) return [...raw.values()];
    if (Array.isArray(raw)) return raw.filter(Boolean);
    return raw && typeof raw === "object" ? Object.values(raw).filter(Boolean) : [];
}

function syncGraphTargetSlots(node) {
    const links = graphLinkValues(node.graph || app.graph);
    node.inputs.forEach((input, targetSlot) => {
        const ids = Array.isArray(input?.link) ? input.link : [input?.link];
        for (const id of ids) {
            if (id == null) continue;
            const link = links.find((item) => String(item?.id) === String(id));
            if (link) link.target_slot = targetSlot;
        }
    });
}

// 旧工作流可能没有新插槽；同时固定为 stage_index → latent_image → media。
function ensureLatentImageInput(node) {
    if (!Array.isArray(node?.inputs)) return false;
    let latentIndex = node.inputs.findIndex((input) => String(input?.name || "") === LATENT_IMAGE_INPUT);
    if (latentIndex < 0 && typeof node.addInput === "function") {
        node.addInput(LATENT_IMAGE_INPUT, "IMAGE");
        latentIndex = node.inputs.findIndex((input) => String(input?.name || "") === LATENT_IMAGE_INPUT);
    }
    if (latentIndex < 0) return false;

    const mediaIndex = node.inputs.findIndex((input) => String(input?.name || "") === "media");
    const stageIndex = node.inputs.findIndex((input) => String(input?.name || "") === "stage_index");
    let desiredIndex = mediaIndex >= 0 ? mediaIndex : Math.max(0, stageIndex + 1);
    if (latentIndex < desiredIndex) desiredIndex -= 1;
    if (latentIndex !== desiredIndex) {
        const [input] = node.inputs.splice(latentIndex, 1);
        node.inputs.splice(desiredIndex, 0, input);
    }

    syncGraphTargetSlots(node);
    node._widgetSlotsDirty = true;
    node.setDirtyCanvas?.(true, true);
    return true;
}

// 新插槽插在旧 media 前面。若旧工作流按槽位恢复，media 的线可能暂时
// 落到 latent_image；利用保存的输入名称把它迁回 media。
function repairLegacyMediaLink(node, info) {
    const savedInputs = Array.isArray(info?.inputs) ? info.inputs : [];
    if (!savedInputs.length || savedInputs.some((input) => String(input?.name || "") === LATENT_IMAGE_INPUT)) return false;
    const savedMedia = savedInputs.find((input) => String(input?.name || "") === "media");
    const savedLink = savedMedia?.link;
    if (savedLink == null) return false;
    const latentInput = (node.inputs || []).find((input) => String(input?.name || "") === LATENT_IMAGE_INPUT);
    const mediaInput = (node.inputs || []).find((input) => String(input?.name || "") === "media");
    if (!latentInput || !mediaInput || mediaInput.link != null || String(latentInput.link) !== String(savedLink)) return false;
    latentInput.link = null;
    mediaInput.link = savedLink;
    syncGraphTargetSlots(node);
    node.setDirtyCanvas?.(true, true);
    return true;
}

function repairWrongMediaEditorTarget(node) {
    if (node.__sumQwenRepairingMediaEditorLink) return false;
    const latentIndex = (node.inputs || []).findIndex((input) => String(input?.name || "") === LATENT_IMAGE_INPUT);
    const mediaIndex = (node.inputs || []).findIndex((input) => String(input?.name || "") === "media");
    if (latentIndex < 0 || mediaIndex < 0) return false;
    const latentLinkId = node.inputs[latentIndex]?.link;
    if (latentLinkId == null) return false;
    const link = graphLinkValues(node.graph || app.graph)
        .find((item) => String(item?.id) === String(latentLinkId));
    const sourceId = Number(link?.origin_id ?? link?.originId);
    const sourceSlot = Number(link?.origin_slot ?? link?.originSlot ?? 0) || 0;
    const sourceNode = (node.graph || app.graph)?.getNodeById?.(sourceId);
    if (nodeClass(sourceNode) !== MEDIA_EDITOR_CLASS) return false;

    node.__sumQwenRepairingMediaEditorLink = true;
    try {
        node.disconnectInput?.(latentIndex);
        if (node.inputs[mediaIndex]?.link == null) sourceNode.connect?.(sourceSlot, node, mediaIndex);
    } finally {
        node.__sumQwenRepairingMediaEditorLink = false;
    }
    node.setDirtyCanvas?.(true, true);
    return true;
}

function syncWidthHeightState(node) {
    const input = (node.inputs || []).find((item) => String(item?.name || "") === LATENT_IMAGE_INPUT);
    const useImageSize = inputHasLink(input);
    setWidgetGrayed(getWidget(node, "width"), useImageSize);
    setWidgetGrayed(getWidget(node, "height"), useImageSize);
}

function syncResolutionState(node) {
    const modeWidget = getWidget(node, REF_SIZE_MODE_WIDGET);
    // True=各自原尺寸 (灰化 resolution)；False=统一分辨率 (启用)
    const useOriginal = modeWidget?.value === REF_SIZE_MODE_ORIGINAL_VALUE;
    setWidgetGrayed(getWidget(node, RESOLUTION_WIDGET), useOriginal);
}

function attachRefSizeModeWatcher(node) {
    const modeWidget = getWidget(node, REF_SIZE_MODE_WIDGET);
    if (!modeWidget || modeWidget.__sumQwenRefSizeModeWatcher) return;
    const originalCallback = modeWidget.callback;
    modeWidget.callback = (...args) => {
        const result = originalCallback?.apply(modeWidget, args);
        syncResolutionState(node);
        return result;
    };
    modeWidget.__sumQwenRefSizeModeWatcher = true;
}

function hideStagePromptMapping(node) {
    const mapping = node.__adGuideStagePromptMapping;
    if (mapping && mapping.style.display !== "none") {
        mapping.style.display = "none";
    }
}

// 启动一个轮询：当 minimax_guide_dual_ui40.js 在节点上挂出
// __adGuideStagePromptMapping 时，立刻把它 display: none。
// 同时兼容 wrap 被销毁再重建的场景（mapping 会换成新元素）。
function startMappingHider(node) {
    if (node.__sumQwenMappingHiderStarted) return;
    node.__sumQwenMappingHiderStarted = true;

    let lastSeen = null;
    const startedAt = Date.now();
    const interval = setInterval(() => {
        const nodeGone = !node || node.removed || !node.graph;
        const tooLong = Date.now() - startedAt > MAPPING_POLL_MAX_LIFE_MS;
        if (nodeGone || tooLong) {
            clearInterval(interval);
            node.__sumQwenMappingHiderStarted = false;
            return;
        }
        const mapping = node.__adGuideStagePromptMapping;
        // 元素引用变化时（新 mapping 实例）也立即隐藏
        if (mapping && mapping !== lastSeen) {
            hideStagePromptMapping(node);
            lastSeen = mapping;
        }
    }, MAPPING_POLL_INTERVAL_MS);
}

app.registerExtension({
    name: "Apt_Preset.sum_QwenImage2",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            const self = this;
            ensureNodeStyleOverride();
            setTimeout(() => {
                ensureLatentImageInput(self);
                syncWidthHeightState(self);
                attachRefSizeModeWatcher(self);
                syncResolutionState(self);
                startMappingHider(self);
                tagNodeAsOurHost(self);
            }, 0);
            return r;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (info) {
            const r = onConfigure?.apply(this, arguments);
            const self = this;
            ensureNodeStyleOverride();
            // 旧工作流 schema 迁移由 minimax_guide_dual_ui40.js 中按明确布局
            // 处理的 repairQwen2ConfiguredWidgetValues 负责。这里不能再按数值
            // 类型重排，否则 width/height 会被误写到 resolution。
            setTimeout(() => {
                ensureLatentImageInput(self);
                repairWrongMediaEditorTarget(self);
                repairLegacyMediaLink(self, info);
                syncWidthHeightState(self);
                attachRefSizeModeWatcher(self);
                syncResolutionState(self);
                startMappingHider(self);
                tagNodeAsOurHost(self);
            }, 50);
            return r;
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index) {
            const r = onConnectionsChange?.apply(this, arguments);
            const input = this.inputs?.[Number(index)];
            if (String(input?.name || "") === LATENT_IMAGE_INPUT) {
                setTimeout(() => syncWidthHeightState(this), 0);
            }
            return r;
        };

        const onConnectInput = nodeType.prototype.onConnectInput;
        nodeType.prototype.onConnectInput = function (targetSlot, type, output, originNode) {
            const inputName = String(this.inputs?.[Number(targetSlot)]?.name || "");
            // 若某条前端路径绕过了 AD_Media_editor.connect 的主动重定向，
            // 仍不允许其宽类型 media 输出占用 latent_image。
            if (inputName === LATENT_IMAGE_INPUT && nodeClass(originNode) === MEDIA_EDITOR_CLASS) {
                console.warn("[sum_QwenImage2] AD_Media_editor 只能连接到 media 输入");
                return false;
            }
            return onConnectInput?.apply(this, arguments);
        };
    },
});
