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
    return src.replace(/[?&]rand=[^&]*/g, "").replace(/[?&]t=\d+/g, "");
}

function getNodeImageUrl(node) {
    const imageRef = node.images?.[0];
    if (imageRef?.filename) {
        const params = new URLSearchParams({
            filename: imageRef.filename,
            subfolder: imageRef.subfolder || "",
            type: imageRef.type || "input",
        });
        return api.apiURL(`/view?${params.toString()}`);
    }
    return node.imgs?.[0]?.src || "";
}

function prepareNodeForOfficialMaskEditor(node) {
    refreshNodePreview(node, "bridge_editor_preview_0.png");
}

async function syncEditedMaskToBackend(node) {
    const imageUrl = getNodeImageUrl(node);
    if (!imageUrl) {
        throw new Error("官方遮罩编辑器没有产出可同步的图片。");
    }

    const imageBlob = await fetchImageAsBlob(imageUrl);
    const formData = new FormData();
    formData.append("node_id", String(node.id));
    formData.append("image", imageBlob, `flow_bridge_image_${node.id}.png`);

    const response = await api.fetchApi("/apt_preset/flow_bridge_image/save_edit", {
        method: "POST",
        body: formData,
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
        throw new Error(result.error || `保存失败: ${response.status}`);
    }

    setBooleanWidget(node, "disable_input", true);
    setBooleanWidget(node, "disable_output", false);
    refreshNodePreview(node, "bridge_preview_0.png");
    await app.queuePrompt(0);
}

function clearMaskWatcher(node) {
    if (node.__flowBridgeMaskWatcher) {
        clearInterval(node.__flowBridgeMaskWatcher);
        node.__flowBridgeMaskWatcher = null;
    }
}

function restoreClipspaceReturnNode(node) {
    if (ComfyApp?.clipspace_return_node === node) {
        ComfyApp.clipspace_return_node = node.__flowBridgePreviousReturnNode ?? null;
    }
    node.__flowBridgePreviousReturnNode = null;
}

function watchOfficialMaskEditorSave(node, initialSignature) {
    clearMaskWatcher(node);

    let attempts = 0;
    node.__flowBridgeMaskWatcher = setInterval(async () => {
        attempts += 1;
        const currentSignature = getNodeImageSignature(node);
        if (!currentSignature || currentSignature === initialSignature) {
            if (attempts > 600) {
                clearMaskWatcher(node);
                restoreClipspaceReturnNode(node);
            }
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
    }, 500);
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

    prepareNodeForOfficialMaskEditor(node);
    const initialSignature = getNodeImageSignature(node);
    node.__flowBridgePreviousReturnNode = ComfyApp?.clipspace_return_node ?? null;
    ComfyApp.copyToClipspace(node);
    ComfyApp.clipspace_return_node = node;
    watchOfficialMaskEditorSave(node, initialSignature);
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
