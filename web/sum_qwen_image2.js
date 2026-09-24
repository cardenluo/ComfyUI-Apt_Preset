import { app } from "../../scripts/app.js";

// sum_QwenImage2 专属：
//   1) 根据 output_size 控件的取值，灰化或启用 width/height widget。
//   2) 根据 ref_size_mode 控件的取值，灰化或启用 resolution widget。
//   3) 隐藏 minimax_guide_dual_ui40.js 公共资源里的「Image N→Picture N」预览行，
//      不动共享代码，只对本节点差异化处理。
// 其他 UI 行为（媒体台、stage_prompts 滚动、虚拟连线等）由 minimax_guide_dual_ui40.js 提供。

const NODE_CLASS = "sum_QwenImage2";
const OUTPUT_SIZE_WIDGET = "output_size";
const OUTPUT_SIZE_CUSTOM_VALUE = true;   // True=自定义(width/height)；False=第一张图尺寸
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

function syncWidthHeightState(node) {
    const modeWidget = getWidget(node, OUTPUT_SIZE_WIDGET);
    // True=自定义 (启用)；False=第一张图尺寸 (灰化)
    const isCustom = modeWidget?.value === OUTPUT_SIZE_CUSTOM_VALUE;
    setWidgetGrayed(getWidget(node, "width"), !isCustom);
    setWidgetGrayed(getWidget(node, "height"), !isCustom);
}

function attachOutputSizeWatcher(node) {
    const modeWidget = getWidget(node, OUTPUT_SIZE_WIDGET);
    if (!modeWidget || modeWidget.__sumQwenOutputSizeWatcher) return;
    const originalCallback = modeWidget.callback;
    modeWidget.callback = (...args) => {
        const result = originalCallback?.apply(modeWidget, args);
        syncWidthHeightState(node);
        return result;
    };
    modeWidget.__sumQwenOutputSizeWatcher = true;
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

// widgets_values 位置错位迁移：在多次 schema 变更（删 latent_image / 加 size_mode→output_size / 加 ref_size_mode / 改 BOOLEAN）后，
// 旧 workflow 的 widgets_values 数组按位置塞进新 schema 会错位（例如 "blur" 落到 resolution 槽、1024 落到 negative_prompt 槽）。
// 这里按 widget 类型而非位置去匹配：
//   - STRING 槽去找类型为 STRING 的旧值；
//   - INT 槽去找有限数值；
//   - BOOLEAN 槽去找 true/false。
// 一旦修过就标记 __sumQwenSchemaMigrated，避免每次 onConfigure 都重跑。
function migrateMisalignedWidgetsValues(node, info) {
    if (node.__sumQwenSchemaMigrated) return false;
    const widgets = node.widgets || [];
    const widgetValues = info?.widgets_values || [];
    if (!widgets.length || !widgetValues.length) return false;

    // 期望类型映射（按 widget.name 区分），INT/BOOLEAN 都归到 number 类，正负值范围作为粗筛
    const typeByName = {
        stage_index: "int",
        resolution: "int",
        width: "int",
        height: "int",
        ref_size_mode: "bool",
        output_size: "bool",
        prompt: "string",
        stage_prompts: "string",
        negative_prompt: "string",
    };

    // 仅对"位置已错位"的 widget 做迁移；判断规则：当前 value 的类型与期望不符（且期望类型为 string/int/bool 之一）
    const needsMigration = widgets.some((w) => {
        const expected = typeByName[w.name];
        if (!expected) return false;
        const v = w.value;
        const actual = Array.isArray(v) ? "array" : v === null || v === undefined ? "null" : typeof v;
        if (expected === "string" && actual !== "string") return true;
        if (expected === "int" && (actual !== "number" || !Number.isFinite(v))) return true;
        if (expected === "bool" && actual !== "boolean") return true;
        return false;
    });

    if (!needsMigration) {
        node.__sumQwenSchemaMigrated = true;
        return false;
    }

    // 把旧 widgets_values 按类型分桶，保留原始索引以便反向写回
    const buckets = { string: [], int: [], bool: [], other: [] };
    for (let i = 0; i < widgetValues.length; i++) {
        const v = widgetValues[i];
        if (typeof v === "string") buckets.string.push({ index: i, value: v });
        else if (typeof v === "boolean") buckets.bool.push({ index: i, value: v });
        else if (typeof v === "number" && Number.isFinite(v)) buckets.int.push({ index: i, value: v });
        else buckets.other.push({ index: i, value: v });
    }

    // 按 widget.name 的"期望默认"值预填，避免空桶时全部归零
    const reassigned = new Set();
    const takeOne = (kind, predicate = () => true) => {
        const arr = buckets[kind];
        for (let k = 0; k < arr.length; k++) {
            if (reassigned.has(arr[k].index)) continue;
            if (!predicate(arr[k].value)) continue;
            const v = arr[k].value;
            reassigned.add(arr[k].index);
            return v;
        }
        return undefined;
    };

    let changed = false;
    for (const w of widgets) {
        const expected = typeByName[w.name];
        if (!expected) continue;
        const v = w.value;
        const actual = Array.isArray(v) ? "array" : v === null || v === undefined ? "null" : typeof v;
        const ok =
            (expected === "string" && actual === "string") ||
            (expected === "int" && actual === "number" && Number.isFinite(v)) ||
            (expected === "bool" && actual === "boolean");
        if (ok) continue;

        let next;
        if (expected === "string") {
            // negative_prompt 优先匹配"blur"或非空短字符串；prompt/stage_prompts 同
            next = takeOne("string", (s) => s.length > 0);
            if (next === undefined && w.options?.default !== undefined) next = w.options.default;
        } else if (expected === "int") {
            // resolution/width/height 优先取 32 倍数的合理分辨率（>= 32 且 <= 8192）
            next = takeOne("int", (n) => n >= 32 && n <= 8192 && n % 32 === 0);
            if (next === undefined) next = takeOne("int", (n) => n >= 32 && n <= 8192);
            if (next === undefined && w.options?.default !== undefined) next = w.options.default;
        } else if (expected === "bool") {
            next = takeOne("bool");
            if (next === undefined && w.options?.default !== undefined) next = w.options.default;
        }
        if (next !== undefined && next !== v) {
            w.value = next;
            changed = true;
        }
    }

    if (changed) {
        console.warn("[sum_QwenImage2] widgets_values misaligned after schema change; smart-migrated by type");
    }
    node.__sumQwenSchemaMigrated = true;
    return changed;
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
                attachOutputSizeWatcher(self);
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
            // 先做 schema 迁移（在 widget.value 被 positional 装载之后立刻修正）
            if (migrateMisalignedWidgetsValues(self, info)) {
                self.setDirtyCanvas?.(true, true);
            }
            setTimeout(() => {
                attachOutputSizeWatcher(self);
                syncWidthHeightState(self);
                attachRefSizeModeWatcher(self);
                syncResolutionState(self);
                startMappingHider(self);
                tagNodeAsOurHost(self);
            }, 50);
            return r;
        };
    },
});