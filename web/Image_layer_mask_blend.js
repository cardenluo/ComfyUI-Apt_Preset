import { app, ComfyApp } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_CLASS = "Image_layer_mask_blend";

function nodeKey(node) {
    return String(node?.id ?? "default").replace(/[^a-zA-Z0-9_-]/g, "_");
}

function editorRef(node) {
    return {
        filename: "layer_editor.png",
        subfolder: `Apt_Preset_layer_mask/${nodeKey(node)}`,
        type: "input",
    };
}

function viewUrl(ref) {
    const params = new URLSearchParams({
        filename: ref.filename,
        subfolder: ref.subfolder || "",
        type: ref.type || "input",
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function paintedRef(node) {
    const ref = node?.images?.[0];
    if (!ref?.filename || (ref.subfolder || "") !== "clipspace") return null;
    if (!/^clipspace-painted-masked-\d+\.png$/i.test(ref.filename)) return null;
    return { filename: ref.filename, subfolder: ref.subfolder, type: ref.type || "input" };
}

function clearWatcher(node) {
    if (node.__aptLayerMaskWatcher) {
        clearInterval(node.__aptLayerMaskWatcher);
        node.__aptLayerMaskWatcher = null;
    }
}

function restoreReturnNode(node) {
    if (ComfyApp?.clipspace_return_node === node) {
        ComfyApp.clipspace_return_node = node.__aptLayerMaskPreviousReturnNode ?? null;
    }
    node.__aptLayerMaskPreviousReturnNode = null;
}

async function savePaintedMask(node, ref) {
    const body = new FormData();
    body.append("node_id", String(node.id));
    body.append("image_ref", JSON.stringify(ref));
    const response = await api.fetchApi("/apt_preset/image_layer_mask_blend/save_mask", { method: "POST", body });
    const result = await response.json();
    if (!response.ok || !result.ok) throw new Error(result.error || `Mask save failed: ${response.status}`);
    await app.queuePrompt(0);
}

function waitForPaintedMask(node) {
    let attempts = 0;
    let lastSignature = "";
    let stableTicks = 0;
    node.__aptLayerMaskWatcher = setInterval(async () => {
        attempts += 1;
        const ref = paintedRef(node);
        if (!ref) {
            if (attempts > 40) {
                clearWatcher(node);
                restoreReturnNode(node);
                alert("没有收到遮罩编辑结果，请重新打开遮罩编辑器。");
            }
            return;
        }
        const signature = `${ref.subfolder}/${ref.filename}`;
        if (signature !== lastSignature) {
            lastSignature = signature;
            stableTicks = 0;
            return;
        }
        stableTicks += 1;
        if (stableTicks < 2) return;

        clearWatcher(node);
        try {
            await savePaintedMask(node, ref);
        } catch (error) {
            console.error("[Image_layer_mask_blend] Failed to save mask:", error);
            alert(error?.message || "遮罩保存失败。");
        } finally {
            restoreReturnNode(node);
        }
    }, 250);
}

function watchEditorClose(node) {
    let attempts = 0;
    let dialogSeen = false;
    node.__aptLayerMaskWatcher = setInterval(() => {
        attempts += 1;
        const dialogOpen = document.querySelectorAll(".p-dialog-mask").length > 0;
        if (dialogOpen) {
            dialogSeen = true;
            return;
        }
        if (dialogSeen) {
            clearWatcher(node);
            setTimeout(() => waitForPaintedMask(node), 150);
            return;
        }
        if (attempts > 80) {
            clearWatcher(node);
            restoreReturnNode(node);
        }
    }, 250);
}

function openMaskEditor(node) {
    if (typeof ComfyApp?.copyToClipspace !== "function" || typeof ComfyApp?.open_maskeditor !== "function") {
        alert("当前 ComfyUI 前端没有可用的原生遮罩编辑器接口。");
        return;
    }

    const ref = editorRef(node);
    const image = new Image();
    image.onload = () => {
        clearWatcher(node);
        restoreReturnNode(node);
        node.imgs = [image];
        node.images = [ref];
        node.imageIndex = 0;
        node.setDirtyCanvas?.(true, true);
        node.__aptLayerMaskPreviousReturnNode = ComfyApp.clipspace_return_node ?? null;
        ComfyApp.copyToClipspace(node);
        ComfyApp.clipspace_return_node = node;
        watchEditorClose(node);
        ComfyApp.open_maskeditor();
    };
    image.onerror = () => alert("请先运行一次节点，生成 A 图的遮罩编辑预览。");
    const url = viewUrl(ref);
    image.src = `${url}${url.includes("?") ? "&" : "?"}t=${Date.now()}`;
}

app.registerExtension({
    name: "AptPreset.ImageLayerMaskBlend",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;

        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function getLayerMaskMenuOptions(_, options) {
            const result = getExtraMenuOptions?.apply(this, arguments);
            options.unshift({
                content: "绘制上层镂空遮罩",
                callback: () => openMaskEditor(this),
            });
            return result;
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function onLayerMaskNodeRemoved() {
            clearWatcher(this);
            restoreReturnNode(this);
            return onRemoved?.apply(this, arguments);
        };
    },
});
