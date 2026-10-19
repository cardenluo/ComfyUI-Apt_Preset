import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { queueOutputNodes } from "./viewMaskAndImgQueueButton.js";

const NODE_NAME = "View_video_combine";

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

function createVideoUI() {
    const root = document.createElement("div");
    root.style.cssText = "width:100%;height:360px;display:flex;flex-direction:column;gap:6px;color:#ddd;font:12px sans-serif;overflow:hidden;";

    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;flex:1;min-height:260px;background:#080808;overflow:hidden;border:1px solid #444;border-radius:6px;";
    const video = document.createElement("video");
    video.playsInline = true;
    video.preload = "metadata";
    video.style.cssText = "width:100%;height:100%;object-fit:contain;background:#080808;";
    stage.appendChild(video);

    const empty = document.createElement("div");
    empty.textContent = "点击 update 保存并预览视频";
    empty.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:#111;color:#999;pointer-events:none;";
    stage.appendChild(empty);

    const info = document.createElement("div");
    info.textContent = "视频尺寸 -- × --  ·  总帧数 --";
    info.style.cssText = "height:20px;display:flex;align-items:center;justify-content:center;color:#f6b94a;font-weight:600;white-space:nowrap;";

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
    root.append(stage, controls, info);

    const updateTime = () => {
        const duration = Number.isFinite(video.duration) ? video.duration : 0;
        timeline.max = String(duration || 1);
        timeline.value = String(video.currentTime || 0);
        timeLabel.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
    };
    const play = async () => {
        try {
            await video.play();
            playButton.textContent = "❚❚";
        } catch {
            video.pause();
        }
    };
    playButton.addEventListener("click", () => video.paused ? play() : video.pause());
    timeline.addEventListener("input", () => {
        video.currentTime = Number(timeline.value);
        updateTime();
    });
    muteButton.addEventListener("click", () => {
        video.muted = !video.muted;
        muteButton.textContent = video.muted ? "🔇" : "🔊";
    });
    fullscreenButton.addEventListener("click", () => stage.requestFullscreen?.());
    video.addEventListener("timeupdate", updateTime);
    video.addEventListener("play", () => { playButton.textContent = "❚❚"; });
    video.addEventListener("pause", () => { playButton.textContent = "▶"; });
    video.addEventListener("ended", () => { video.currentTime = 0; updateTime(); });

    const load = (data) => {
        video.pause();
        video.src = videoUrl(data.video);
        video.muted = true;
        muteButton.textContent = "🔇";
        empty.style.display = "none";
        info.textContent = `视频尺寸 ${data.width} × ${data.height}  ·  总帧数 ${data.frames}`;
        updateTime();
        video.addEventListener("canplay", play, { once: true });
    };
    const destroy = () => {
        video.pause();
        video.removeAttribute("src");
        video.load();
    };
    return { root, load, destroy };
}

app.registerExtension({
    name: "Apt_Preset.View_video_combine",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            const updateButton = this.addWidget("button", "update", "queue", () => queueOutputNodes([this]));
            updateButton.options = { ...updateButton.options, class: "queue-button" };
            const ui = createVideoUI();
            this._videoCombineUI = ui;
            const widget = this.addDOMWidget("video_combine", "video_combine", ui.root, {
                serialize: false,
                hideOnZoom: false,
            });
            widget.computeSize = (width) => [width, 370];
            this.setSize([Math.max(this.size[0], 540), Math.max(this.size[1], 520)]);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const data = message?.video_combine?.[0];
            if (data) this._videoCombineUI?.load(data);
        };

        const onRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this._videoCombineUI?.destroy();
            onRemoved?.apply(this, arguments);
        };
    },
});
