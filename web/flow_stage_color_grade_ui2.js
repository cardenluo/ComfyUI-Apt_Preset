import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const field = (node, name) => node.widgets?.find((item) => item.name === name);

function createEditor(node) {
    const root = document.createElement("div");
    root.className = "apt-video-grade";
    const style = document.createElement("style");
    style.textContent = `
        .apt-video-grade{display:flex;flex-direction:column;gap:10px;padding:12px;box-sizing:border-box;height:100%;color:var(--input-text,#ddd);background:var(--comfy-menu-bg,#222);font:12px sans-serif}
        .apt-video-grade .viewer{position:relative;flex:1;min-height:100px;background:#101215;border-radius:6px;overflow:hidden}
        .apt-video-grade video,.apt-video-grade .live-frame{position:absolute;width:100%;height:100%;object-fit:contain}
        .apt-video-grade .row{display:flex;align-items:center;gap:8px}
        .apt-video-grade button,.apt-video-grade select{color:inherit;background:#34383e;border:1px solid #555;border-radius:5px;padding:5px 9px;cursor:pointer}
        .apt-video-grade button:disabled{opacity:.45;cursor:default}
        .apt-video-grade .track{position:relative;height:36px;margin:0 8px;touch-action:none;cursor:crosshair;flex:none}
        .apt-video-grade .rail{position:absolute;left:0;right:0;top:12px;height:8px;border-radius:4px;background:#454b53}
        .apt-video-grade .selection{position:absolute;top:12px;height:8px;background:#7bb5e6;pointer-events:none}
        .apt-video-grade .handle{position:absolute;top:5px;width:12px;height:23px;padding:0;transform:translateX(-50%);background:#add7f7;border:1px solid #273a49;cursor:ew-resize;touch-action:none}
        .apt-video-grade .playhead{position:absolute;top:0;width:2px;height:34px;background:#fff;pointer-events:none;box-shadow:0 0 2px #000}
        .apt-video-grade .control{display:flex;align-items:center;gap:10px;min-height:22px}
        .apt-video-grade .control span{width:55px;flex:none}
        .apt-video-grade .control input{min-width:0;flex:1;accent-color:#8ac4f2}
        .apt-video-grade output{width:42px;text-align:right;font-variant-numeric:tabular-nums}
        .apt-video-grade .status{min-height:16px;color:#aeb7c3;font-size:11px}
    `;
    const video = document.createElement("video");
    video.playsInline = true;
    video.muted = true;
    video.preload = "auto";
    const viewer = document.createElement("div");
    viewer.className = "viewer";
    const liveFrame = document.createElement("img");
    liveFrame.className = "live-frame";
    liveFrame.alt = "当前帧实时调色预览";
    liveFrame.style.display = "none";
    viewer.append(video, liveFrame);
    const track = document.createElement("div");
    track.className = "track";
    track.title = "拖动两端选择调色范围；点击轨道定位画面";
    const rail = document.createElement("div");
    rail.className = "rail";
    const selection = document.createElement("div");
    selection.className = "selection";
    const head = document.createElement("div");
    head.className = "playhead";
    const start = document.createElement("button"), end = document.createElement("button");
    for (const [handle, label] of [[start, "开始帧"], [end, "结束帧"]]) {
        handle.className = "handle";
        handle.type = "button";
        handle.title = label;
        handle.setAttribute("role", "slider");
        handle.setAttribute("aria-label", label);
        handle.disabled = true;
    }
    track.append(rail, selection, start, end, head);
    const row = document.createElement("div");
    row.className = "row";
    const status = document.createElement("div");
    status.className = "status";
    status.textContent = "连接视频后加载，拖选范围即可调色。";
    const info = document.createElement("span");
    info.style.cssText = "flex:1;font-variant-numeric:tabular-nums";
    const state = { total: 0, fps: 24, current: 1, start: 1, end: 1, reference: 1, dragging: null };
    let source = null, revision = 0, timer = null, busy = false, disposed = false, frameUrl = null;
    function queueLive() {
        ++revision;
        clearTimeout(timer);
        if (!disposed && source && video.paused) timer = setTimeout(renderLive, 60);
    }
    async function renderLive() {
        if (disposed || busy || !source || !video.paused) return;
        busy = true;
        const requestRevision = revision;
        try {
            const controls = Object.fromEntries(["mode", "strength", "exposure", "saturation", "temperature"]
                .map((name) => [name, field(node, name).value]));
            const response = await api.fetchApi("/apt_preset/video_grade/frame", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ source, frame: state.current, start: state.start, end: state.end,
                    reference: state.reference, ...controls }),
            });
            if (!response.ok) {
                const result = await response.json();
                throw new Error(result.error || "单帧预览失败，请重新加载视频。");
            }
            const blob = await response.blob();
            if (disposed || requestRevision !== revision || !video.paused) return;
            const url = URL.createObjectURL(blob);
            try {
                const decoded = new Image();
                decoded.src = url;
                await decoded.decode();
                if (disposed || requestRevision !== revision || !video.paused) return;
                if (frameUrl) URL.revokeObjectURL(frameUrl);
                frameUrl = url;
                liveFrame.src = url;
                liveFrame.style.display = "block";
                status.textContent = "单帧实时预览；点击「应用调色」更新整段输出。";
            } finally {
                if (frameUrl !== url) URL.revokeObjectURL(url);
            }
        } catch (error) {
            if (!disposed && requestRevision === revision) status.textContent = `实时预览：${error.message}`;
        } finally {
            busy = false;
            if (!disposed && requestRevision !== revision && video.paused) {
                clearTimeout(timer);
                timer = setTimeout(renderLive, 60);
            }
        }
    }
    function setValue(name, value) {
        const item = field(node, name);
        item.value = value;
        item.callback?.(value);
        app.graph?.change?.();
        node.setDirtyCanvas?.(true, true);
    }
    function changed() {
        video.pause();
        status.textContent = source ? "正在更新当前帧…" : "请先加载视频，再实时调节。";
        queueLive();
    }
    function saveSelection() {
        setValue("selection", JSON.stringify({ start: state.start, end: state.end, reference: state.reference }));
        changed();
    }
    function renderTrack() {
        const percent = (frame) => 100 * (frame - 1) / Math.max(1, state.total - 1);
        start.style.left = selection.style.left = `${percent(state.start)}%`;
        end.style.left = `${percent(state.end)}%`;
        selection.style.width = `${percent(state.end) - percent(state.start)}%`;
        head.style.left = `${percent(state.current)}%`;
        for (const [handle, value] of [[start, state.start], [end, state.end]]) {
            handle.setAttribute("aria-valuemin", 1);
            handle.setAttribute("aria-valuemax", state.total || 1);
            handle.setAttribute("aria-valuenow", value);
            handle.disabled = !state.total;
        }
        info.textContent = state.total ? `${state.current} / ${state.total} 帧 · 选区 ${state.start}–${state.end}` : "尚未加载视频";
        reference.textContent = `取当前帧颜色（${state.reference}）`;
    }
    function seek(frame) {
        if (!state.total) return;
        video.pause();
        state.current = Math.max(1, Math.min(state.total, frame));
        liveFrame.style.display = "none";
        video.currentTime = (state.current - 1) / state.fps;
        renderTrack();
        queueLive();
    }
    function selectEdge(edge, frame) {
        frame = Math.max(1, Math.min(state.total, frame));
        if (edge === "start") state.start = Math.min(frame, state.end);
        else state.end = Math.max(frame, state.start);
        saveSelection();
        seek(state[edge]);
    }
    function pointerFrame(event) {
        const rect = track.getBoundingClientRect();
        const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / Math.max(1, rect.width)));
        return 1 + Math.round(fraction * (state.total - 1));
    }
    track.onpointerdown = (event) => {
        if (!state.total) return;
        event.preventDefault();
        event.stopPropagation();
        state.dragging = event.target === start ? "start" : event.target === end ? "end" : "head";
        track.setPointerCapture(event.pointerId);
        if (state.dragging === "head") seek(pointerFrame(event));
        else selectEdge(state.dragging, pointerFrame(event));
    };
    track.onpointermove = (event) => {
        if (!state.dragging) return;
        if (state.dragging === "head") seek(pointerFrame(event));
        else selectEdge(state.dragging, pointerFrame(event));
    };
    track.onpointerup = track.onpointercancel = () => { state.dragging = null; };
    for (const [handle, edge] of [[start, "start"], [end, "end"]]) {
        handle.onkeydown = (event) => {
            if (!state.total) return;
            const delta = { ArrowLeft: -1, ArrowDown: -1, ArrowRight: 1, ArrowUp: 1 }[event.key];
            if (delta === undefined && !["Home", "End"].includes(event.key)) return;
            event.preventDefault();
            event.stopPropagation();
            selectEdge(edge, event.key === "Home" ? 1 : event.key === "End" ? state.total : state[edge] + delta);
        };
    }
    function button(text, action) {
        const item = document.createElement("button");
        item.type = "button";
        item.textContent = text;
        item.onclick = action;
        row.append(item);
        return item;
    }
    button("‹", () => seek(state.current - 1)).title = "上一帧";
    const play = button("播放", async () => {
        if (!state.total) return;
        if (!video.paused) { video.pause(); return; }
        if (state.current < state.start || state.current >= state.end) seek(state.start);
        try { await video.play(); } catch (error) { status.textContent = error.message; }
    });
    button("›", () => seek(state.current + 1)).title = "下一帧";
    row.append(info);
    video.onplay = () => {
        play.textContent = "暂停";
        liveFrame.style.display = "none";
        ++revision;
        clearTimeout(timer);
        status.textContent = "播放上次已应用的整段视频；暂停后可实时调节当前帧。";
    };
    video.onpause = () => {
        play.textContent = "播放";
        state.current = Math.max(1, Math.min(state.total, Math.floor(video.currentTime * state.fps + 0.001) + 1));
        renderTrack();
        queueLive();
    };
    video.ontimeupdate = () => {
        state.current = Math.max(1, Math.min(state.total, Math.floor(video.currentTime * state.fps + 0.001) + 1));
        if (!video.paused && state.current > state.end) video.currentTime = (state.start - 1) / state.fps;
        renderTrack();
    };
    video.onended = () => { seek(state.start); };
    video.onerror = () => { status.textContent = "视频预览不可用，请重新加载。"; };

    const modes = document.createElement("div");
    modes.className = "row";
    const mode = document.createElement("select");
    mode.setAttribute("aria-label", "调色模式");
    for (const [value, label] of [["manual", "手动调色"], ["brightness_smooth", "亮度平滑"], ["reference_match", "颜色匹配"]]) {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = label;
        mode.append(option);
    }
    const reference = document.createElement("button");
    reference.type = "button";
    reference.onclick = () => { if (state.total) { state.reference = state.current; saveSelection(); renderTrack(); } };
    modes.append(mode, reference);
    const controls = [];
    for (const [name, label, min, max, step] of [["strength", "强度", 0, 1, .05], ["exposure", "亮度 EV", -4, 4, .05],
        ["saturation", "饱和度", 0, 2, .05], ["temperature", "色温", -1, 1, .05]]) {
        const group = document.createElement("label");
        group.className = "control";
        const text = document.createElement("span");
        text.textContent = label;
        const input = document.createElement("input");
        input.type = "range";
        input.min = min; input.max = max; input.step = step;
        input.setAttribute("aria-label", label);
        const value = document.createElement("output");
        input.oninput = () => { setValue(name, Number(input.value)); value.textContent = Number(input.value).toFixed(2); changed(); };
        group.append(text, input, value);
        controls.push({ name, group, input, value });
    }
    function syncControls() {
        mode.value = field(node, "mode").value;
        reference.style.display = mode.value === "reference_match" ? "" : "none";
        for (const control of controls) {
            if (control.name === "strength") {
                const max = mode.value === "brightness_smooth" ? 3 : 1;
                control.input.max = max;
                control.input.title = max === 3 ? "0–1 平滑亮度；1–3 加强极亮、极暗和颜色跳变修正" : "调色强度";
                if (Number(field(node, "strength").value) > max) setValue("strength", max);
            }
            control.input.value = field(node, control.name).value;
            control.value.textContent = Number(control.input.value).toFixed(2);
            control.group.style.display = mode.value === "brightness_smooth" && control.name !== "strength" ? "none" : "flex";
        }
    }
    mode.onchange = () => { setValue("mode", mode.value); syncControls(); changed(); };
    button("↺", () => {
        video.pause();
        for (const [name, value] of Object.entries({ mode: "manual", strength: 1, exposure: 0, saturation: 1, temperature: 0, selection: "{}" })) {
            setValue(name, value);
        }
        state.start = state.reference = 1;
        state.end = state.total || 1;
        syncControls();
        renderTrack();
        changed();
    }).title = "重置调色与选区；更换视频后可重新加载";
    const apply = button("加载视频", async () => {
        video.pause();
        if (mode.value === "brightness_smooth" && Number(field(node, "strength").value) > 0 && state.start === 1 && state.end === state.total) {
            status.textContent = "请拖选异常范围，并在选区外留出正常帧供亮度平滑参考。";
            return;
        }
        apply.disabled = true;
        try {
            const prompt = await app.graphToPrompt();
            const output = {};
            function include(id) {
                if (output[id] || !prompt.output[id]) return;
                output[id] = prompt.output[id];
                for (const value of Object.values(output[id].inputs || {})) {
                    if (Array.isArray(value) && value.length === 2) include(String(value[0]));
                }
            }
            include(String(node.id));
            if (!output[String(node.id)]) throw new Error("请先连接视频输入。");
            const response = await api.fetchApi("/prompt", {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ client_id: api.clientId, prompt: output, partial_execution_targets: [String(node.id)],
                    extra_data: { extra_pnginfo: { workflow: prompt.workflow } } }),
            });
            const result = await response.json();
            if (!response.ok) throw new Error(result.error?.message || JSON.stringify(result.error || result));
            status.textContent = "正在处理视频…";
        } catch (error) { status.textContent = `处理失败：${error.message}`; }
        finally { apply.disabled = false; }
    });
    root.append(style, viewer, track, row, modes, ...controls.map((item) => item.group), status);
    for (const name of ["mode", "strength", "exposure", "saturation", "temperature", "selection"]) {
        const item = field(node, name);
        item.type = "hidden";
        item.hidden = true;
        item.computeSize = () => [0, -4];
        if (item.element) item.element.style.display = "none";
        if (item.inputEl) item.inputEl.style.display = "none";
    }
    const dom = node.addDOMWidget("video_grade_editor", "div", root, { serialize: false });
    dom.computeSize = (width) => [width, 460];
    syncControls();
    renderTrack();
    return {
        configure() { syncControls(); },
        update(payload) {
            video.pause();
            source = payload.source?.filename || null;
            ++revision;
            clearTimeout(timer);
            liveFrame.style.display = "none";
            state.total = payload.total_frames;
            state.fps = payload.fps;
            state.start = payload.start;
            state.end = payload.end;
            state.reference = payload.reference;
            state.current = Math.max(1, Math.min(state.total, state.current));
            const item = payload.video;
            video.onloadedmetadata = () => { seek(state.current); };
            video.src = api.apiURL(`/view?${new URLSearchParams({ filename: item.filename, subfolder: item.subfolder || "", type: "temp" })}`);
            apply.textContent = "应用调色";
            status.textContent = "暂停后拖动滑块实时查看当前帧；应用调色后更新整段输出。";
            renderTrack();
        },
        dispose() {
            disposed = true;
            ++revision;
            video.pause();
            clearTimeout(timer);
            if (frameUrl) URL.revokeObjectURL(frameUrl);
            liveFrame.removeAttribute("src");
            video.removeAttribute("src");
            video.load();
        },
    };
}

app.registerExtension({
    name: "AptPreset.FlowStageColorGrade.UI2",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AD_Video_color_grad") return;
        const created = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = created?.apply(this, arguments);
            this.stageColorEditor = createEditor(this);
            this.setSize([Math.max(500, this.size[0]), this.computeSize()[1]]);
            return result;
        };
        const configured = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = configured?.apply(this, arguments);
            this.stageColorEditor?.configure();
            return result;
        };
        const executed = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = executed?.apply(this, arguments);
            const payload = message.grade_preview?.at(-1);
            if (payload) this.stageColorEditor?.update(payload);
            return result;
        };
        const removed = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            this.stageColorEditor?.dispose();
            return removed?.apply(this, arguments);
        };
    },
});
