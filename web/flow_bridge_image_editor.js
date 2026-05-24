import { app, ComfyApp } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function debugReport(hypothesisId, location, msg, data = {}) {
    // #region debug-point common:report
    fetch("http://127.0.0.1:7777/event", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            sessionId: "mask-save-lag",
            runId: "post-fix",
            hypothesisId,
            location,
            msg: `[DEBUG] ${msg}`,
            data,
            ts: Date.now(),
        }),
    }).catch(() => {});
    // #endregion
}

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

function refreshNodePreview(node, filename = "bridge_preview_0.png") {
    const ref = buildBridgeImageRef(node.id, filename);
    const viewUrl = buildBridgeFileUrl(node.id, filename);
    const image = new Image();
    image.src = `${viewUrl}${viewUrl.includes("?") ? "&" : "?"}t=${Date.now()}`;
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
    if ((imageRef.subfolder || "") !== "clipspace") {
        return null;
    }
    if (!/^clipspace-painted-masked-\d+\.png$/i.test(imageRef.filename)) {
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

function prepareNodeForOfficialMaskEditor(node) {
    refreshNodePreview(node, "bridge_editor_preview_0.png");
}

async function syncEditedMaskToBackend(node, options = {}) {
    const preferredRef = cloneMaskEditorServerRef(options.serverRef);
    const serverRef = preferredRef || getMaskEditorServerRef(node) || getStoredMaskEditorServerRef(node);
    const imageUrl = getMaskEditorServerUrl(node) || getNodeImageUrl(node);
    if (!serverRef && !imageUrl) {
        throw new Error("官方遮罩编辑器没有产出可同步的图片。");
    }

    // #region debug-point A:sync-start
    debugReport("A", "flow_bridge_image_editor.js:syncEditedMaskToBackend:start", "准备同步编辑结果到后端", {
        nodeId: node.id,
        imageUrl,
        serverRef,
        signature: getNodeImageSignature(node),
        imageRef: node.images?.[0] || null,
        imgSrc: node.imgs?.[0]?.src || "",
    });
    // #endregion

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

    // #region debug-point E:sync-response
    debugReport("E", "flow_bridge_image_editor.js:syncEditedMaskToBackend:response", "后端已返回保存结果", {
        nodeId: node.id,
        ok: result.ok,
        view_url: result.view_url || "",
        imageRef: node.images?.[0] || null,
        imgSrc: node.imgs?.[0]?.src || "",
    });
    // #endregion

    setBooleanWidget(node, "disable_input", true);
    setBooleanWidget(node, "disable_output", false);
    refreshNodePreview(node, "bridge_preview_0.png");
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

function waitForFinalNodeImage(node, initialSignature) {
    clearMaskWatcher(node);

    let attempts = 0;
    let lastSignature = "";
    let stableTicks = 0;
    node.__flowBridgeMaskWatcher = setInterval(async () => {
        attempts += 1;
        const currentSignature = getNodeImageSignature(node);
        // #region debug-point C:watcher-tick
        debugReport("C", "flow_bridge_image_editor.js:waitForFinalNodeImage:tick", "等待节点最终图像稳定", {
            nodeId: node.id,
            attempts,
            initialSignature,
            currentSignature,
            lastSignature,
            stableTicks,
            serverRef: getMaskEditorServerRef(node),
            imageRef: node.images?.[0] || null,
            imgSrc: node.imgs?.[0]?.src || "",
        });
        // #endregion

        const serverRef = getMaskEditorServerRef(node);
        if (!serverRef) {
            if (attempts > 40) {
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

function watchMaskEditorDialogLifecycle(node, initialSignature) {
    clearMaskWatcher(node);

    let attempts = 0;
    let dialogSeen = false;
    node.__flowBridgeMaskWatcher = setInterval(() => {
        attempts += 1;
        const dialogOpen = isMaskEditorProbablyOpen();
        // #region debug-point C:dialog-lifecycle
        debugReport("C", "flow_bridge_image_editor.js:watchMaskEditorDialogLifecycle:tick", "轮询官方 MaskEditor 弹窗状态", {
            nodeId: node.id,
            attempts,
            dialogOpen,
            dialogSeen,
            initialSignature,
            imageRef: node.images?.[0] || null,
            imgSrc: node.imgs?.[0]?.src || "",
        });
        // #endregion

        if (dialogOpen) {
            dialogSeen = true;
            return;
        }

        if (dialogSeen) {
            clearMaskWatcher(node);
            setTimeout(() => {
                waitForFinalNodeImage(node, initialSignature);
            }, 150);
            return;
        }

        if (attempts > 80) {
            clearMaskWatcher(node);
            restoreClipspaceReturnNode(node);
        }
    }, 250);
}

function openOfficialMaskEditor(node) {
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
    prepareNodeForOfficialMaskEditor(node);
    const initialSignature = getNodeImageSignature(node);
    // #region debug-point B:open-editor
    debugReport("B", "flow_bridge_image_editor.js:openOfficialMaskEditor", "准备打开官方 MaskEditor", {
        nodeId: node.id,
        initialSignature,
        imageRef: node.images?.[0] || null,
        imgSrc: node.imgs?.[0]?.src || "",
    });
    // #endregion
    node.__flowBridgePreviousReturnNode = ComfyApp?.clipspace_return_node ?? null;
    ComfyApp.copyToClipspace(node);
    ComfyApp.clipspace_return_node = node;
    watchMaskEditorDialogLifecycle(node, initialSignature);
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
