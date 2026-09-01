import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { queueOutputNodes } from "./viewMaskAndImgQueueButton.js";

const VIDEO_NODE_NAME = "View_video_compare";
const IMAGE_NODE_NAME = "View_image_compare";

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function videoUrl(info) {
    const params = new URLSearchParams({
        filename: info.filename,
        subfolder: info.subfolder || "",
        type: info.type,
        t: String(Date.now()),
    });
    return api.apiURL(`/view?${params.toString()}`);
}

function formatTime(value) {
    const seconds = Math.max(0, Number(value) || 0);
    const minutes = Math.floor(seconds / 60);
    const remain = Math.floor(seconds % 60);
    return `${String(minutes).padStart(2, "0")}:${String(remain).padStart(2, "0")}`;
}

function createCompareUI(node) {
    const root = document.createElement("div");
    root.style.cssText = "width:100%;height:360px;display:flex;flex-direction:column;gap:6px;color:#ddd;font:12px sans-serif;overflow:hidden;";

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;flex:1;min-height:260px;background:#080808;overflow:hidden;border:1px solid #444;border-radius:6px;touch-action:none;user-select:none;";

    const videoA = document.createElement("video");
    const videoB = document.createElement("video");
    for (const video of [videoA, videoB]) {
        video.playsInline = true;
        video.preload = "metadata";
        video.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#080808;";
        stage.appendChild(video);
    }
    videoB.muted = true;
    videoB.style.pointerEvents = "none";

    const divider = document.createElement("div");
    divider.style.cssText = "position:absolute;z-index:3;background:#fff;box-shadow:0 0 5px #000;pointer-events:none;";
    const handle = document.createElement("div");
    handle.textContent = "↔";
    handle.style.cssText = "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:30px;height:30px;border-radius:15px;background:#fff;color:#222;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 1px 5px #000;";
    divider.appendChild(handle);
    stage.appendChild(divider);

    const labelA = document.createElement("span");
    const labelB = document.createElement("span");
    labelA.textContent = "视频 A";
    labelB.textContent = "视频 B";
    for (const label of [labelA, labelB]) {
        label.style.cssText = "position:absolute;z-index:4;bottom:8px;padding:3px 7px;border-radius:4px;background:#000a;color:#fff;pointer-events:none;";
        stage.appendChild(label);
    }
    labelA.style.left = "8px";
    labelB.style.right = "8px";

    const empty = document.createElement("div");
    empty.textContent = "运行节点后显示视频对比";
    empty.style.cssText = "position:absolute;z-index:5;inset:0;display:flex;align-items:center;justify-content:center;background:#111;color:#999;";
    stage.appendChild(empty);

    const controls = document.createElement("div");
    controls.style.cssText = "display:grid;grid-template-columns:34px 1fr auto 34px 34px;align-items:center;gap:6px;height:30px;";
    const playButton = document.createElement("button");
    const timeline = document.createElement("input");
    const timeLabel = document.createElement("span");
    const muteButton = document.createElement("button");
    const fullscreenButton = document.createElement("button");
    playButton.textContent = "▶";
    muteButton.textContent = "🔇";
    fullscreenButton.textContent = "⛶";
    timeline.type = "range";
    timeline.min = "0";
    timeline.max = "1";
    timeline.step = "0.01";
    timeline.value = "0";
    timeLabel.textContent = "00:00 / 00:00";
    for (const button of [playButton, muteButton, fullscreenButton]) {
        button.style.cssText = "height:26px;padding:0;border:1px solid #555;border-radius:4px;background:#292929;color:#eee;cursor:pointer;";
    }
    controls.append(playButton, timeline, timeLabel, muteButton, fullscreenButton);
    root.append(stage, controls);

    let split = 50;
    let direction = "左右";
    let dragging = false;
    let syncing = false;

    const applySplit = () => {
        const value = Math.max(0, Math.min(100, Number(split) || 0));
        if (direction === "上下") {
            videoB.style.clipPath = `inset(0 0 ${100 - value}% 0)`;
            divider.style.left = "0";
            divider.style.right = "0";
            divider.style.top = `${value}%`;
            divider.style.bottom = "auto";
            divider.style.height = "2px";
            divider.style.width = "auto";
            divider.style.transform = "translateY(-1px)";
            handle.textContent = "↕";
            labelA.style.bottom = "8px";
            labelB.style.bottom = "auto";
            labelB.style.top = "8px";
        } else {
            videoB.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
            divider.style.top = "0";
            divider.style.bottom = "0";
            divider.style.left = `${value}%`;
            divider.style.right = "auto";
            divider.style.width = "2px";
            divider.style.height = "auto";
            divider.style.transform = "translateX(-1px)";
            handle.textContent = "↔";
            labelA.style.bottom = "8px";
            labelB.style.top = "auto";
            labelB.style.bottom = "8px";
        }
    };

    const setSplitFromPointer = (event) => {
        const rect = stage.getBoundingClientRect();
        split = direction === "上下"
            ? ((event.clientY - rect.top) / rect.height) * 100
            : ((event.clientX - rect.left) / rect.width) * 100;
        split = Math.max(0, Math.min(100, split));
        const widget = getWidget(node, "split_position");
        if (widget) widget.value = Math.round(split * 10) / 10;
        applySplit();
        node.setDirtyCanvas?.(true, true);
    };

    stage.addEventListener("pointerdown", (event) => {
        dragging = true;
        stage.setPointerCapture(event.pointerId);
        setSplitFromPointer(event);
    });
    stage.addEventListener("pointermove", (event) => {
        if (dragging) setSplitFromPointer(event);
    });
    stage.addEventListener("pointerup", (event) => {
        dragging = false;
        stage.releasePointerCapture(event.pointerId);
    });
    stage.addEventListener("pointercancel", () => { dragging = false; });

    const syncVideoB = (force = false) => {
        if (!Number.isFinite(videoA.currentTime)) return;
        if (force || Math.abs(videoB.currentTime - videoA.currentTime) > 0.08) {
            videoB.currentTime = Math.min(videoA.currentTime, videoB.duration || videoA.currentTime);
        }
        videoB.playbackRate = videoA.playbackRate;
    };

    const updateTime = () => {
        const duration = Number.isFinite(videoA.duration) ? videoA.duration : 0;
        timeline.max = String(duration || 1);
        timeline.value = String(videoA.currentTime || 0);
        timeLabel.textContent = `${formatTime(videoA.currentTime)} / ${formatTime(duration)}`;
        if (!videoA.paused) syncVideoB();
    };

    const playBoth = async () => {
        syncVideoB(true);
        try {
            await Promise.all([videoA.play(), videoB.play()]);
            playButton.textContent = "❚❚";
        } catch {
            pauseBoth();
        }
    };

    const pauseBoth = () => {
        videoA.pause();
        videoB.pause();
        playButton.textContent = "▶";
    };

    playButton.addEventListener("click", () => videoA.paused ? playBoth() : pauseBoth());
    timeline.addEventListener("input", () => {
        syncing = true;
        videoA.currentTime = Number(timeline.value);
        syncVideoB(true);
        updateTime();
        syncing = false;
    });
    muteButton.addEventListener("click", () => {
        videoA.muted = !videoA.muted;
        muteButton.textContent = videoA.muted ? "🔇" : "🔊";
    });
    fullscreenButton.addEventListener("click", () => stage.requestFullscreen?.());
    videoA.addEventListener("timeupdate", () => { if (!syncing) updateTime(); });
    videoA.addEventListener("seeked", () => syncVideoB(true));
    videoA.addEventListener("pause", () => { videoB.pause(); playButton.textContent = "▶"; });
    videoA.addEventListener("ended", () => {
        if (getWidget(node, "loop")?.value) {
            videoA.currentTime = 0;
            videoB.currentTime = 0;
            playBoth();
        } else {
            pauseBoth();
        }
    });

    const bindWidget = (name, callback) => {
        const widget = getWidget(node, name);
        if (!widget) return;
        const original = widget.callback;
        widget.callback = function (value) {
            original?.apply(this, arguments);
            callback(value);
        };
    };
    bindWidget("direction", (value) => { direction = value; applySplit(); });
    bindWidget("split_position", (value) => { split = value; applySplit(); });

    const load = (data) => {
        pauseBoth();
        direction = data.direction || getWidget(node, "direction")?.value || "左右";
        split = data.split_position ?? getWidget(node, "split_position")?.value ?? 50;
        videoA.loop = false;
        videoB.loop = false;
        videoA.src = videoUrl(data.video_a);
        videoB.src = videoUrl(data.video_b);
        empty.style.display = "none";
        applySplit();
        updateTime();
        if (data.autoplay) {
            videoA.muted = true;
            muteButton.textContent = "🔇";
            Promise.all([
                new Promise((resolve) => videoA.addEventListener("canplay", resolve, { once: true })),
                new Promise((resolve) => videoB.addEventListener("canplay", resolve, { once: true })),
            ]).then(playBoth);
        }
    };

    const destroy = () => {
        pauseBoth();
        for (const video of [videoA, videoB]) {
            video.removeAttribute("src");
            video.load();
        }
    };

    applySplit();
    return { root, load, destroy };
}

function createImageCompareUI(node) {
    const root = document.createElement("div");
    root.style.cssText = "width:100%;height:360px;display:flex;flex-direction:column;gap:6px;color:#ddd;font:12px sans-serif;overflow:hidden;";

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;flex:1;min-height:280px;background:#080808;overflow:hidden;border:1px solid #444;border-radius:6px;touch-action:none;user-select:none;";
    const imageA = document.createElement("img");
    const imageB = document.createElement("img");
    for (const image of [imageA, imageB]) {
        image.draggable = false;
        image.style.cssText = "position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#080808;pointer-events:none;";
        stage.appendChild(image);
    }

    const divider = document.createElement("div");
    divider.style.cssText = "position:absolute;z-index:3;background:#fff;box-shadow:0 0 5px #000;pointer-events:none;";
    const handle = document.createElement("div");
    handle.style.cssText = "position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:30px;height:30px;border-radius:15px;background:#fff;color:#222;display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 1px 5px #000;";
    divider.appendChild(handle);
    stage.appendChild(divider);

    const labelA = document.createElement("span");
    const labelB = document.createElement("span");
    labelA.textContent = "图片 A";
    labelB.textContent = "图片 B（上层）";
    for (const label of [labelA, labelB]) {
        label.style.cssText = "position:absolute;z-index:4;bottom:8px;padding:3px 7px;border-radius:4px;background:#000a;color:#fff;pointer-events:none;";
        stage.appendChild(label);
    }
    labelA.style.left = "8px";
    labelB.style.right = "8px";

    const empty = document.createElement("div");
    empty.textContent = "点击 update 显示图像对比";
    empty.style.cssText = "position:absolute;z-index:5;inset:0;display:flex;align-items:center;justify-content:center;background:#111;color:#999;";
    stage.appendChild(empty);

    const opacityRow = document.createElement("div");
    opacityRow.style.cssText = "display:grid;grid-template-columns:auto 1fr 42px 34px;align-items:center;gap:7px;height:30px;";
    const opacityLabel = document.createElement("span");
    const opacitySlider = document.createElement("input");
    const opacityValue = document.createElement("span");
    const fullscreenButton = document.createElement("button");
    opacityLabel.textContent = "上层透明度";
    opacitySlider.type = "range";
    opacitySlider.min = "0";
    opacitySlider.max = "1";
    opacitySlider.step = "0.01";
    opacitySlider.value = "1";
    opacityValue.textContent = "100%";
    fullscreenButton.textContent = "⛶";
    fullscreenButton.style.cssText = "height:26px;padding:0;border:1px solid #555;border-radius:4px;background:#292929;color:#eee;cursor:pointer;";
    opacityRow.append(opacityLabel, opacitySlider, opacityValue, fullscreenButton);
    root.append(stage, opacityRow);

    let split = 50;
    let direction = "左右";

    const applySplit = () => {
        const value = Math.max(0, Math.min(100, Number(split) || 0));
        if (direction === "上下") {
            imageB.style.clipPath = `inset(0 0 ${100 - value}% 0)`;
            divider.style.left = "0";
            divider.style.right = "0";
            divider.style.top = `${value}%`;
            divider.style.bottom = "auto";
            divider.style.width = "auto";
            divider.style.height = "2px";
            divider.style.transform = "translateY(-1px)";
            handle.textContent = "↕";
            labelB.style.top = "8px";
            labelB.style.bottom = "auto";
        } else {
            imageB.style.clipPath = `inset(0 ${100 - value}% 0 0)`;
            divider.style.top = "0";
            divider.style.bottom = "0";
            divider.style.left = `${value}%`;
            divider.style.right = "auto";
            divider.style.width = "2px";
            divider.style.height = "auto";
            divider.style.transform = "translateX(-1px)";
            handle.textContent = "↔";
            labelB.style.top = "auto";
            labelB.style.bottom = "8px";
        }
    };

    const applyOpacity = (value) => {
        const opacity = Math.max(0, Math.min(1, Number(value) || 0));
        imageB.style.opacity = String(opacity);
        opacitySlider.value = String(opacity);
        opacityValue.textContent = `${Math.round(opacity * 100)}%`;
    };

    let dragging = false;
    const setSplitFromPointer = (event) => {
        const rect = stage.getBoundingClientRect();
        split = direction === "上下"
            ? ((event.clientY - rect.top) / rect.height) * 100
            : ((event.clientX - rect.left) / rect.width) * 100;
        split = Math.max(0, Math.min(100, split));
        const widget = getWidget(node, "split_position");
        if (widget) widget.value = Math.round(split * 10) / 10;
        applySplit();
        node.setDirtyCanvas?.(true, true);
    };
    stage.addEventListener("pointerdown", (event) => {
        dragging = true;
        stage.setPointerCapture(event.pointerId);
        setSplitFromPointer(event);
    });
    stage.addEventListener("pointermove", (event) => {
        if (dragging) setSplitFromPointer(event);
    });
    stage.addEventListener("pointerup", (event) => {
        dragging = false;
        stage.releasePointerCapture(event.pointerId);
    });
    stage.addEventListener("pointercancel", () => { dragging = false; });

    opacitySlider.addEventListener("input", () => {
        const widget = getWidget(node, "opacity");
        if (widget) widget.value = Number(opacitySlider.value);
        applyOpacity(opacitySlider.value);
        node.setDirtyCanvas?.(true, true);
    });
    fullscreenButton.addEventListener("click", () => stage.requestFullscreen?.());

    const bindWidget = (name, callback) => {
        const widget = getWidget(node, name);
        if (!widget) return;
        const original = widget.callback;
        widget.callback = function (value) {
            original?.apply(this, arguments);
            callback(value);
        };
    };
    bindWidget("direction", (value) => { direction = value; applySplit(); });
    bindWidget("split_position", (value) => { split = value; applySplit(); });
    bindWidget("opacity", applyOpacity);

    const load = (data) => {
        direction = data.direction || getWidget(node, "direction")?.value || "左右";
        split = data.split_position ?? getWidget(node, "split_position")?.value ?? 50;
        imageA.src = videoUrl(data.image_a);
        imageB.src = videoUrl(data.image_b);
        empty.style.display = "none";
        applySplit();
        applyOpacity(data.opacity ?? getWidget(node, "opacity")?.value ?? 1);
    };

    direction = getWidget(node, "direction")?.value || direction;
    split = getWidget(node, "split_position")?.value ?? split;
    applySplit();
    applyOpacity(getWidget(node, "opacity")?.value ?? 1);
    return { root, load };
}

app.registerExtension({
    name: "Apt_Preset.View_video_compare",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === IMAGE_NODE_NAME) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const updateButton = this.addWidget("button", "update", "queue", () => queueOutputNodes([this]));
                updateButton.options = { ...updateButton.options, class: "queue-button" };
                const ui = createImageCompareUI(this);
                this._imageCompareUI = ui;
                const widget = this.addDOMWidget("image_compare", "image_compare", ui.root, {
                    serialize: false,
                    hideOnZoom: false,
                });
                widget.computeSize = (width) => [width, 370];
                this.setSize([Math.max(this.size[0], 540), Math.max(this.size[1], 500)]);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                const data = message?.image_compare?.[0];
                if (data) this._imageCompareUI?.load(data);
            };
            return;
        }

        if (nodeData.name !== VIDEO_NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const ui = createCompareUI(this);
            this._videoCompareUI = ui;
            const widget = this.addDOMWidget("video_compare", "video_compare", ui.root, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = (width) => [width, 370];
            this.setSize([Math.max(this.size[0], 540), Math.max(this.size[1], 500)]);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const data = message?.video_compare?.[0];
            if (data) this._videoCompareUI?.load(data);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._videoCompareUI?.destroy();
            onRemoved?.apply(this, arguments);
        };
    },
});
