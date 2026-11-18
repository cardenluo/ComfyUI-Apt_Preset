import { app, ComfyApp } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function sanitizeNodeId(nodeId) {
    const text = String(nodeId ?? "default");
    return text.replace(/[^a-zA-Z0-9_-]/g, "_");
}

function buildBridgeImageRef(nodeId, filename = "bridge_preview_0.png") {
    const safeNodeId = sanitizeNodeId(nodeId);
    return {
        filename,
        subfolder: `zml_image_memory/${safeNodeId}`,
        type: "input",
    };
}

function buildBridgeFileUrl(nodeId, filename = "bridge_preview_0.png") {
    const ref = buildBridgeImageRef(nodeId, filename);
    const params = new URLSearchParams({
        filename: ref.filename,
        subfolder: ref.subfolder,
        type: ref.type,
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function buildViewUrlFromRef(imageRef) {
    if (!imageRef?.filename) {
        return "";
    }
    const params = new URLSearchParams({
        filename: imageRef.filename,
        subfolder: imageRef.subfolder || "",
        type: imageRef.type || "input",
    });
    return api.apiURL(`/view?${params.toString()}`);
}

async function fetchImageAsBlob(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
        throw new Error(`读取图片失败: ${response.status}`);
    }
    return await response.blob();
}

function setBooleanWidget(node, name, value) {
    const widget = node.widgets?.find((w) => w.name === name);
    if (!widget) {
        return;
    }
    widget.value = value;
    if (typeof widget.callback === "function") {
        widget.callback(value);
    }
}

async function refreshNodePreview(node, filename = "bridge_preview_0.png") {
    const ref = buildBridgeImageRef(node.id, filename);
    const viewUrl = buildBridgeFileUrl(node.id, filename);
    const image = new Image();
    const imageUrl = `${viewUrl}${viewUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
    await new Promise((resolve, reject) => {
        image.onload = () => resolve();
        image.onerror = () => reject(new Error(`编辑预览加载失败: ${filename}`));
        image.src = imageUrl;
    });
    node.imgs = [image];
    node.images = [ref];
    node.imageIndex = 0;
    node.setDirtyCanvas(true, true);
}

function getNodeImageSignature(node) {
    const imageRef = node.images?.[0];
    if (imageRef?.filename) {
        return `${imageRef.subfolder || ""}/${imageRef.filename}[${imageRef.type || ""}]`;
    }
    const src = node.imgs?.[0]?.src || "";
    if (src) {
        return src.replace(/[?&]rand=[^&]*/g, "").replace(/[?&]t=\d+/g, "");
    }
    return "";
}

function getNodeImageUrl(node) {
    const imageRef = node.images?.[0];
    if (imageRef?.filename) {
        return buildViewUrlFromRef(imageRef);
    }

    const src = node.imgs?.[0]?.src || "";
    if (src) {
        return src;
    }
    return "";
}

function getMaskEditorServerRef(node) {
    const imageRef = node.images?.[0];
    if (!imageRef?.filename) {
        return null;
    }
    if (!/^clipspace-painted-masked-\d+\.png$/i.test(imageRef.filename)) {
        return null;
    }
    if ((imageRef.type || "input") !== "input") {
        return null;
    }
    return imageRef;
}

function cloneMaskEditorServerRef(imageRef) {
    if (!imageRef?.filename) {
        return null;
    }
    return {
        filename: imageRef.filename,
        subfolder: imageRef.subfolder || "",
        type: imageRef.type || "input",
    };
}

function getStoredMaskEditorServerRef(node) {
    return cloneMaskEditorServerRef(node.__flowBridgeLastMaskEditorServerRef);
}

function getMaskEditorServerUrl(node) {
    const imageRef = getMaskEditorServerRef(node);
    return imageRef ? buildViewUrlFromRef(imageRef) : "";
}

function hasBridgePreview(node) {
    const imageRef = node.images?.[0];
    return imageRef?.filename === "bridge_preview_0.png"
        && (imageRef.subfolder || "") === `zml_image_memory/${sanitizeNodeId(node.id)}`;
}

function queueNodeAndWait(node) {
    return new Promise((resolve, reject) => {
        const nodeId = String(node.id);
        const timeoutId = setTimeout(() => {
            cleanup();
            reject(new Error("等待节点生成编辑预览超时。"));
        }, 120000);

        const cleanup = () => {
            clearTimeout(timeoutId);
            api.removeEventListener("executed", onExecuted);
            api.removeEventListener("execution_cached", onExecutionCached);
            api.removeEventListener("execution_error", onExecutionError);
        };
        const finish = () => {
            cleanup();
            resolve();
        };
        const onExecuted = ({ detail }) => {
            if (String(detail?.node) === nodeId) {
                finish();
            }
        };
        const onExecutionCached = ({ detail }) => {
            if (detail?.nodes?.some((cachedNodeId) => String(cachedNodeId) === nodeId)) {
                finish();
            }
        };
        const onExecutionError = ({ detail }) => {
            cleanup();
            reject(new Error(detail?.exception_message || "工作流执行失败，无法生成编辑预览。"));
        };

        api.addEventListener("executed", onExecuted);
        api.addEventListener("execution_cached", onExecutionCached);
        api.addEventListener("execution_error", onExecutionError);
        Promise.resolve(app.queuePrompt(0)).catch((error) => {
            cleanup();
            reject(error);
        });
    });
}

async function prepareNodeForOfficialMaskEditor(node) {
    if (!hasBridgePreview(node)) {
        await queueNodeAndWait(node);
    }
    try {
        await refreshNodePreview(node, "bridge_editor_preview_0.png");
    } catch (error) {
        await queueNodeAndWait(node);
        await refreshNodePreview(node, "bridge_editor_preview_0.png");
    }
}

async function syncEditedMaskToBackend(node, options = {}) {
    const preferredRef = cloneMaskEditorServerRef(options.serverRef);
    const serverRef = preferredRef || getMaskEditorServerRef(node) || getStoredMaskEditorServerRef(node);
    const imageUrl = getMaskEditorServerUrl(node) || getNodeImageUrl(node);
    if (!serverRef && !imageUrl) {
        throw new Error("官方遮罩编辑器没有产出可同步的图片。");
    }

    const formData = new FormData();
    formData.append("node_id", String(node.id));
    if (serverRef) {
        formData.append("image_ref", JSON.stringify(serverRef));
    } else {
        const imageBlob = await fetchImageAsBlob(imageUrl);
        formData.append("image", imageBlob, `flow_bridge_image_${node.id}.png`);
    }

    const response = await api.fetchApi("/apt_preset/flow_bridge_image/save_edit", {
        method: "POST",
        body: formData,
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
        throw new Error(result.error || `保存失败: ${response.status}`);
    }

    if (serverRef) {
        node.__flowBridgeLastMaskEditorServerRef = cloneMaskEditorServerRef(serverRef);
    }

    setBooleanWidget(node, "disable_input", true);
    setBooleanWidget(node, "disable_output", false);
    await refreshNodePreview(node, "bridge_preview_0.png");
    restoreClipspaceReturnNode(node);
    setTimeout(() => {
        Promise.resolve(app.queuePrompt(0)).catch((error) => {
            console.error("[flow_bridge_image] 触发重新执行失败:", error);
        });
    }, 0);
}

async function repairFlowBridgeOutput(node) {
    if (isMaskEditorProbablyOpen()) {
        alert("请先关闭遮罩编辑器，再执行输出修复。");
        return;
    }

    const serverRef = getMaskEditorServerRef(node) || getStoredMaskEditorServerRef(node);
    if (!serverRef) {
        alert("当前没有可用于输出修复的遮罩编辑结果，请先完成一次二次编辑遮罩并保存。");
        return;
    }

    clearMaskWatcher(node);
    restoreClipspaceReturnNode(node);

    try {
        await syncEditedMaskToBackend(node, { serverRef });
    } catch (error) {
        console.error("[flow_bridge_image] 输出修复失败:", error);
        alert(error?.message || "输出修复失败");
    }
}

function clearMaskWatcher(node) {
    if (node.__flowBridgeMaskWatcher) {
        clearInterval(node.__flowBridgeMaskWatcher);
        node.__flowBridgeMaskWatcher = null;
    }
}

function isMaskEditorProbablyOpen() {
    return document.querySelectorAll(".p-dialog-mask").length > 0;
}

function restoreClipspaceReturnNode(node) {
    if (ComfyApp?.clipspace_return_node === node) {
        ComfyApp.clipspace_return_node = node.__flowBridgePreviousReturnNode ?? null;
    }
    node.__flowBridgePreviousReturnNode = null;
}

function waitForFinalNodeImage(node) {
    clearMaskWatcher(node);

    let attempts = 0;
    let lastSignature = "";
    let stableTicks = 0;
    node.__flowBridgeMaskWatcher = setInterval(async () => {
        attempts += 1;
        const currentSignature = getNodeImageSignature(node);
        const serverRef = getMaskEditorServerRef(node);
        if (!serverRef) {
            if (attempts > 7200) {
                clearMaskWatcher(node);
                restoreClipspaceReturnNode(node);
            }
            return;
        }

        if (!lastSignature || currentSignature !== lastSignature) {
            lastSignature = currentSignature;
            stableTicks = 0;
            return;
        }

        stableTicks += 1;
        if (stableTicks < 3 && attempts < 40) {
            return;
        }

        clearMaskWatcher(node);
        try {
            await syncEditedMaskToBackend(node);
        } catch (error) {
            console.error("[flow_bridge_image] 同步官方遮罩编辑结果失败:", error);
            alert(error?.message || "同步官方遮罩编辑结果失败");
        } finally {
            restoreClipspaceReturnNode(node);
        }
    }, 250);
}

async function openOfficialMaskEditor(node) {
    if (typeof ComfyApp?.copyToClipspace !== "function") {
        alert("当前前端没有可用的官方 Clipspace 接口。");
        return;
    }

    if (typeof ComfyApp?.open_maskeditor !== "function") {
        alert("当前前端版本没有暴露官方 MaskEditor 兼容入口。");
        return;
    }

    clearMaskWatcher(node);
    restoreClipspaceReturnNode(node);
    try {
        await prepareNodeForOfficialMaskEditor(node);
    } catch (error) {
        console.error("[flow_bridge_image] 打开遮罩编辑器失败:", error);
        alert(error?.message || "打开遮罩编辑器失败");
        return;
    }
    node.__flowBridgePreviousReturnNode = ComfyApp?.clipspace_return_node ?? null;
    ComfyApp.copyToClipspace(node);
    ComfyApp.clipspace_return_node = node;
    waitForFinalNodeImage(node);
    ComfyApp.open_maskeditor();
}

function addEditorButton(node) {
    if (node.__flowBridgeEditorButtonAdded) {
        return;
    }
    node.__flowBridgeEditorButtonAdded = true;

    const widget = node.addWidget("button", "二次编辑遮罩", "open", () => {
        openOfficialMaskEditor(node);
    });
    widget.options = { ...widget.options, class: "flow-bridge-open-editor" };

    const repairWidget = node.addWidget("button", "输出修复", "repair", () => {
        repairFlowBridgeOutput(node);
    });
    repairWidget.options = { ...repairWidget.options, class: "flow-bridge-output-repair" };
}

app.registerExtension({
    name: "AptPreset.FlowBridgeImageEditor",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "flow_bridge_image") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            addEditorButton(this);
            return result;
        };
    },
    async setup() {
        app.graph?._nodes?.forEach((node) => {
            if (node.constructor.nodeData?.name === "flow_bridge_image") {
                addEditorButton(node);
            }
        });
    },
});
