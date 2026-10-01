import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let _aptPadPointerHooksInited = false;
let _aptPadPointerDown = false;
let _aptPadResizeNode = null;

const WORKSPACE_PADDING = 30;
const HANDLE_OFFSET = 24;
const HANDLE_SIZE = 10;
const HANDLE_HIT_RADIUS = 20;

const clampNonNegativeInt = (value) => Math.max(0, Math.round(Number.isFinite(+value) ? +value : 0));

const sanitizeExpandState = (state) => ({
    left: clampNonNegativeInt(state?.left),
    right: clampNonNegativeInt(state?.right),
    top: clampNonNegativeInt(state?.top),
    bottom: clampNonNegativeInt(state?.bottom),
});

const parseExpandStateValue = (value) => {
    if (!value) return sanitizeExpandState({});
    if (typeof value === "object") return sanitizeExpandState(value);
    try {
        return sanitizeExpandState(JSON.parse(value));
    } catch (error) {
        return sanitizeExpandState({});
    }
};

const computeEdgeColor = (img) => {
    const sampleW = Math.max(2, Math.min(96, img.naturalWidth || img.width || 2));
    const sampleH = Math.max(2, Math.min(96, img.naturalHeight || img.height || 2));
    const sampleCanvas = document.createElement("canvas");
    sampleCanvas.width = sampleW;
    sampleCanvas.height = sampleH;
    const sampleCtx = sampleCanvas.getContext("2d", { willReadFrequently: true });
    if (!sampleCtx) return "#808080";
    sampleCtx.drawImage(img, 0, 0, sampleW, sampleH);
    const data = sampleCtx.getImageData(0, 0, sampleW, sampleH).data;

    let r = 0;
    let g = 0;
    let b = 0;
    let count = 0;
    const addPixel = (x, y) => {
        const idx = (y * sampleW + x) * 4;
        r += data[idx];
        g += data[idx + 1];
        b += data[idx + 2];
        count += 1;
    };

    for (let x = 0; x < sampleW; x += 1) {
        addPixel(x, 0);
        addPixel(x, sampleH - 1);
    }
    for (let y = 1; y < sampleH - 1; y += 1) {
        addPixel(0, y);
        addPixel(sampleW - 1, y);
    }

    if (!count) return "#808080";
    return `rgb(${Math.round(r / count)}, ${Math.round(g / count)}, ${Math.round(b / count)})`;
};

const _aptPadEnsurePointerHooks = () => {
    if (_aptPadPointerHooksInited) return;
    _aptPadPointerHooksInited = true;

    const pickResizeNodeFromMouse = () => {
        try {
            const c = app?.canvas;
            const m = c?.graph_mouse || c?.mouse;
            if (!Array.isArray(m) || m.length < 2) return;
            const sn = c?.selected_nodes;
            const nodes = Array.isArray(sn)
                ? sn
                : (sn && typeof sn === "object" ? Object.values(sn) : []);
            for (const n of nodes) {
                if (!n || !Array.isArray(n.pos) || !Array.isArray(n.size)) continue;
                const x0 = n.pos[0];
                const y0 = n.pos[1];
                const x1 = x0 + n.size[0];
                const y1 = y0 + n.size[1];
                const mx = m[0];
                const my = m[1];
                if (mx < x0 || mx > x1 || my < y0 || my > y1) continue;
                if (mx >= x1 - 30 && my >= y1 - 30) {
                    _aptPadResizeNode = n;
                    return;
                }
            }
            const n = c?.node_over;
            if (n && Array.isArray(n.pos) && Array.isArray(n.size)) {
                const mx = m[0];
                const my = m[1];
                const x0 = n.pos[0];
                const y0 = n.pos[1];
                const x1 = x0 + n.size[0];
                const y1 = y0 + n.size[1];
                if (mx >= x1 - 30 && my >= y1 - 30) _aptPadResizeNode = n;
            }
        } catch (error) {}
    };

    const onDown = () => {
        _aptPadPointerDown = true;
        _aptPadResizeNode = null;
        pickResizeNodeFromMouse();
    };
    const onMove = () => {
        if (!_aptPadPointerDown) return;
        pickResizeNodeFromMouse();
    };
    const onUp = () => {
        _aptPadPointerDown = false;
        _aptPadResizeNode = null;
    };

    window.addEventListener("pointerdown", onDown, true);
    window.addEventListener("pointermove", onMove, true);
    window.addEventListener("pointerup", onUp, true);
    window.addEventListener("pointercancel", onUp, true);
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("mousemove", onMove, true);
    window.addEventListener("mouseup", onUp, true);
    window.addEventListener("blur", onUp, true);
};

app.registerExtension({
    name: "apt.Image_expand_canvase_visual",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        _aptPadEnsurePointerHooks();
        if (nodeData.name !== "Image_expand_canvase_visual") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const MIN_NODE_WIDTH = 240;
            const MIN_NODE_HEIGHT = 320;
            if (this.size[0] < MIN_NODE_WIDTH) this.size[0] = MIN_NODE_WIDTH;
            if (this.size[1] < MIN_NODE_HEIGHT) this.size[1] = MIN_NODE_HEIGHT;
            this.resizable = true;
            this.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
            this._apt_target_size = this.size;

            const originalOnResize = this.onResize;
            this.onResize = function (size) {
                this._apt_target_size = size;
                return originalOnResize?.apply(this, arguments);
            };

            const hideWidgetAndSlot = (widgetName) => {
                const w = this.widgets?.find((widget) => widget.name === widgetName);
                if (w) {
                    w.type = "hidden";
                    w.hidden = true;
                    w.computeSize = () => [0, 0];
                    w.draw = () => {};
                    if (w.inputEl) {
                        w.inputEl.style.display = "none";
                        if (w.inputEl.parentElement) w.inputEl.parentElement.style.display = "none";
                    }
                }
                if (this.inputs) {
                    const idx = this.inputs.findIndex((input) => input.name === widgetName);
                    if (idx !== -1) this.removeInput(idx);
                }
            };

            hideWidgetAndSlot("expand_state");

            const expandStateWidget = this.widgets?.find((w) => w.name === "expand_state");
            const widthWidget = this.widgets?.find((w) => w.name === "width");
            const heightWidget = this.widgets?.find((w) => w.name === "height");
            const constantColorWidget = this.widgets?.find((w) => w.name === "constant_color");
            const initialWidthValue = Number(widthWidget?.value ?? 0);
            const initialHeightValue = Number(heightWidget?.value ?? 0);

            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.marginTop = "0px";
            container.style.borderRadius = "6px";
            container.style.overflow = "hidden";
            container.style.backgroundColor = "transparent";

            const viewArea = document.createElement("div");
            viewArea.style.flex = "1";
            viewArea.style.position = "relative";
            viewArea.style.width = "100%";
            viewArea.style.height = "100%";
            viewArea.style.backgroundColor = "#1a1a1a";
            viewArea.style.overflow = "hidden";

            const canvas = document.createElement("canvas");
            canvas.style.position = "absolute";
            canvas.style.inset = "0";
            canvas.style.width = "100%";
            canvas.style.height = "100%";
            canvas.style.cursor = "default";
            viewArea.appendChild(canvas);
            container.appendChild(viewArea);

            const previewBar = document.createElement("div");
            previewBar.style.display = "flex";
            previewBar.style.alignItems = "center";
            previewBar.style.gap = "8px";
            previewBar.style.padding = "0 0px";
            previewBar.style.backgroundColor = "transparent";

            const isNodes2_0 = !!document.querySelector("comfy-app") ||
                !!document.querySelector(".comfy-vue") ||
                (window.comfyAPI && window.comfyAPI.vue);

            const resetBtn = document.createElement("button");
            resetBtn.innerText = "Reset";
            resetBtn.style.flex = "1";
            resetBtn.style.width = "auto";
            resetBtn.style.height = "24px";
            resetBtn.style.lineHeight = "22px";
            resetBtn.style.marginTop = "8px";
            resetBtn.style.border = "none";
            resetBtn.style.borderRadius = "8px";
            resetBtn.style.cursor = "pointer";
            resetBtn.style.fontSize = "10px";
            resetBtn.style.fontWeight = "bold";
            resetBtn.style.backgroundColor = "#4f5d6d";
            resetBtn.style.color = "#fff";
            resetBtn.style.transition = "all 0.2s ease";
            previewBar.appendChild(resetBtn);

            const runBtn = document.createElement("button");
            runBtn.innerText = "Preview";
            runBtn.style.flex = "1";
            runBtn.style.width = "auto";
            runBtn.style.height = "24px";
            runBtn.style.lineHeight = "22px";
            runBtn.style.marginTop = "8px";
            runBtn.style.border = "none";
            runBtn.style.borderRadius = "8px";
            runBtn.style.cursor = "pointer";
            runBtn.style.fontSize = "10px";
            runBtn.style.fontWeight = "bold";
            runBtn.style.backgroundColor = "#2d8a3e";
            runBtn.style.color = "#fff";
            runBtn.style.transition = "all 0.2s ease";
            previewBar.appendChild(runBtn);

            previewBar.style.marginBottom = isNodes2_0 ? "10px" : "0px";
            container.appendChild(previewBar);

            const widget = this.addDOMWidget("ImagePadExpandVisualUI", "div", container, { serialize: false, hideOnZoom: false });
            const nodeInstance = this;
            const ctx = canvas.getContext("2d");
            const UI_DEFAULT_HEIGHT = 300;
            widget.computeSize = function (width) {
                return [width, UI_DEFAULT_HEIGHT];
            };
            const UI_BASE_HEIGHT = typeof nodeInstance.computeSize === "function"
                ? nodeInstance.computeSize()[1]
                : nodeInstance.size[1];

            const getTargetNodeHeight = () => {
                const c = app?.canvas;
                const m = c?.graph_mouse || c?.mouse;
                if (_aptPadPointerDown && _aptPadResizeNode && (_aptPadResizeNode === nodeInstance || _aptPadResizeNode?.id === nodeInstance.id) && Array.isArray(m) && m.length > 1 && Array.isArray(nodeInstance.pos)) {
                    return Math.max(MIN_NODE_HEIGHT, m[1] - nodeInstance.pos[1]);
                }
                return nodeInstance.size[1];
            };

            widget.computeSize = function (width) {
                const targetH = getTargetNodeHeight();
                const extra = Math.max(0, (targetH - UI_BASE_HEIGHT) * 0.95);
                return [width, UI_DEFAULT_HEIGHT + extra];
            };

            let bgImg = null;
            let edgeColor = "#808080";
            let imgWidth = 0;
            let imgHeight = 0;
            let expandState = sanitizeExpandState({});
            let dragInfo = null;
            let userAdjusted = false;
            let internalSync = false;
            let originalSize = null;
            const setWidgetNumericValue = (widgetRef, value) => {
                if (!widgetRef) return;
                widgetRef.value = value;
                if (widgetRef.inputEl) widgetRef.inputEl.value = String(value);
            };

            const updateWidgetMinimums = () => {
                if (widthWidget) {
                    widthWidget.options = { ...(widthWidget.options || {}), min: Math.max(1, imgWidth || 1) };
                    if (widthWidget.inputEl) widthWidget.inputEl.min = String(Math.max(1, imgWidth || 1));
                }
                if (heightWidget) {
                    heightWidget.options = { ...(heightWidget.options || {}), min: Math.max(1, imgHeight || 1) };
                    if (heightWidget.inputEl) heightWidget.inputEl.min = String(Math.max(1, imgHeight || 1));
                }
            };

            const syncExpandStateWidget = () => {
                if (!expandStateWidget) return;
                expandStateWidget.value = JSON.stringify(expandState);
                app.graph?.setDirtyCanvas(true);
            };

            const syncSizeWidgetsFromState = () => {
                const nextWidth = Math.max(imgWidth || 1, (imgWidth || 0) + expandState.left + expandState.right);
                const nextHeight = Math.max(imgHeight || 1, (imgHeight || 0) + expandState.top + expandState.bottom);
                internalSync = true;
                setWidgetNumericValue(widthWidget, nextWidth);
                setWidgetNumericValue(heightWidget, nextHeight);
                internalSync = false;
                app.graph?.setDirtyCanvas(true);
            };

            const applyStateFromSizeWidgets = () => {
                if (!imgWidth || !imgHeight) return;
                const desiredWidth = Math.max(imgWidth, Math.round(Number(widthWidget?.value ?? imgWidth)));
                const desiredHeight = Math.max(imgHeight, Math.round(Number(heightWidget?.value ?? imgHeight)));
                const extraWidth = Math.max(0, desiredWidth - imgWidth);
                const extraHeight = Math.max(0, desiredHeight - imgHeight);
                expandState = {
                    left: Math.floor(extraWidth / 2),
                    right: extraWidth - Math.floor(extraWidth / 2),
                    top: Math.floor(extraHeight / 2),
                    bottom: extraHeight - Math.floor(extraHeight / 2),
                };
                syncSizeWidgetsFromState();
                syncExpandStateWidget();
            };

            const resetState = () => {
                expandState = sanitizeExpandState({});
                userAdjusted = false;
                updateWidgetMinimums();
                syncSizeWidgetsFromState();
                syncExpandStateWidget();
                draw();
            };

            const getCanvasSize = () => ({
                w: Math.max(imgWidth || 1, Math.round(Number(widthWidget?.value ?? (imgWidth || 1)))),
                h: Math.max(imgHeight || 1, Math.round(Number(heightWidget?.value ?? (imgHeight || 1)))),
            });

            const getPreviewRect = () => {
                const { w: totalW, h: totalH } = getCanvasSize();
                if (!totalW || !totalH || !canvas.width || !canvas.height) return null;

                const availableW = Math.max(1, canvas.width - WORKSPACE_PADDING * 2);
                const availableH = Math.max(1, canvas.height - WORKSPACE_PADDING * 2);
                const scale = Math.min(availableW / totalW, availableH / totalH);
                const drawW = totalW * scale;
                const drawH = totalH * scale;
                return {
                    x: (canvas.width - drawW) * 0.5,
                    y: (canvas.height - drawH) * 0.5,
                    w: drawW,
                    h: drawH,
                    scale,
                };
            };

            const getHandlePoints = (rect) => ([
                { side: "top", x: rect.x + rect.w * 0.5, y: rect.y - HANDLE_OFFSET, cursor: "ns-resize" },
                { side: "right", x: rect.x + rect.w + HANDLE_OFFSET, y: rect.y + rect.h * 0.5, cursor: "ew-resize" },
                { side: "bottom", x: rect.x + rect.w * 0.5, y: rect.y + rect.h + HANDLE_OFFSET, cursor: "ns-resize" },
                { side: "left", x: rect.x - HANDLE_OFFSET, y: rect.y + rect.h * 0.5, cursor: "ew-resize" },
            ]);

            const getMouse = (e) => {
                const rect = canvas.getBoundingClientRect();
                const sx = rect.width > 0 ? canvas.width / rect.width : 1;
                const sy = rect.height > 0 ? canvas.height / rect.height : 1;
                return {
                    x: (e.clientX - rect.left) * sx,
                    y: (e.clientY - rect.top) * sy,
                };
            };

            const hitTestHandle = (mouse) => {
                const rect = getPreviewRect();
                if (!rect || !bgImg) return null;
                for (const handle of getHandlePoints(rect)) {
                    const dx = mouse.x - handle.x;
                    const dy = mouse.y - handle.y;
                    if (Math.sqrt(dx * dx + dy * dy) <= HANDLE_HIT_RADIUS) return handle;
                }
                return null;
            };

            const getFillColor = () => {
                const colorMap = {
                    white: "#ffffff",
                    black: "#000000",
                    red: "#ff0000",
                    gray: "#808080",
                    edge: edgeColor,
                };
                return colorMap[constantColorWidget?.value ?? "black"] || "#000000";
            };

            const drawArrow = (x, y, side) => {
                ctx.save();
                ctx.translate(x, y);
                if (side === "right") ctx.rotate(Math.PI * 0.5);
                if (side === "bottom") ctx.rotate(Math.PI);
                if (side === "left") ctx.rotate(-Math.PI * 0.5);

                ctx.fillStyle = "#ffdc45";
                ctx.strokeStyle = "#1a1a1a";
                ctx.lineWidth = 1.5;
                ctx.beginPath();
                ctx.moveTo(0, -HANDLE_SIZE);
                ctx.lineTo(HANDLE_SIZE, HANDLE_SIZE);
                ctx.lineTo(-HANDLE_SIZE, HANDLE_SIZE);
                ctx.closePath();
                ctx.fill();
                ctx.stroke();
                ctx.restore();
            };

            const draw = () => {
                const clientW = canvas.clientWidth;
                const clientH = canvas.clientHeight;
                if (clientW > 0 && clientH > 0 && (canvas.width !== clientW || canvas.height !== clientH)) {
                    canvas.width = clientW;
                    canvas.height = clientH;
                }

                if (!canvas.width || !canvas.height) return;
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                ctx.fillStyle = "#1a1a1a";
                ctx.fillRect(0, 0, canvas.width, canvas.height);

                const rect = getPreviewRect();
                if (!rect) return;

                ctx.save();
                ctx.strokeStyle = "#353535";
                ctx.setLineDash([6, 6]);
                ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
                ctx.restore();

                ctx.fillStyle = getFillColor();
                ctx.fillRect(rect.x, rect.y, rect.w, rect.h);

                if (bgImg && imgWidth && imgHeight) {
                    const imageX = rect.x + expandState.left * rect.scale;
                    const imageY = rect.y + expandState.top * rect.scale;
                    const imageW = imgWidth * rect.scale;
                    const imageH = imgHeight * rect.scale;

                    ctx.drawImage(bgImg, imageX, imageY, imageW, imageH);

                    ctx.save();
                    ctx.strokeStyle = "#ff6a4d";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(imageX, imageY, imageW, imageH);
                    ctx.restore();

                    ctx.save();
                    ctx.strokeStyle = "#7a7a7a";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
                    ctx.restore();

                    for (const handle of getHandlePoints(rect)) {
                        ctx.save();
                        ctx.strokeStyle = "#ffdc45";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        if (handle.side === "top") {
                            ctx.moveTo(rect.x + rect.w * 0.5, rect.y);
                            ctx.lineTo(handle.x, handle.y + 8);
                        } else if (handle.side === "right") {
                            ctx.moveTo(rect.x + rect.w, rect.y + rect.h * 0.5);
                            ctx.lineTo(handle.x - 8, handle.y);
                        } else if (handle.side === "bottom") {
                            ctx.moveTo(rect.x + rect.w * 0.5, rect.y + rect.h);
                            ctx.lineTo(handle.x, handle.y - 8);
                        } else {
                            ctx.moveTo(rect.x, rect.y + rect.h * 0.5);
                            ctx.lineTo(handle.x + 8, handle.y);
                        }
                        ctx.stroke();
                        ctx.restore();
                        drawArrow(handle.x, handle.y, handle.side);
                    }
                }
            };

            const bindWidget = (widgetRef, onChange) => {
                if (!widgetRef) return;
                const originalCallback = widgetRef.callback;
                widgetRef.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    onChange?.();
                };
                if (widgetRef.inputEl) {
                    widgetRef.inputEl.addEventListener("change", onChange);
                    widgetRef.inputEl.addEventListener("input", onChange);
                }
            };

            bindWidget(constantColorWidget, () => draw());
            bindWidget(widthWidget, () => {
                if (internalSync) return;
                applyStateFromSizeWidgets();
                userAdjusted = true;
                draw();
            });
            bindWidget(heightWidget, () => {
                if (internalSync) return;
                applyStateFromSizeWidgets();
                userAdjusted = true;
                draw();
            });

            canvas.addEventListener("mousedown", (e) => {
                if (e.button !== 0) return;
                const mouse = getMouse(e);
                const handle = hitTestHandle(mouse);
                const rect = getPreviewRect();
                if (!handle || !rect) return;
                dragInfo = {
                    side: handle.side,
                    startMouse: mouse,
                    startState: { ...expandState },
                    scale: rect.scale,
                };
                e.preventDefault();
            });

            window.addEventListener("mousemove", (e) => {
                const mouse = getMouse(e);
                if (!dragInfo) {
                    const handle = hitTestHandle(mouse);
                    canvas.style.cursor = handle?.cursor || "default";
                    return;
                }

                const dx = (mouse.x - dragInfo.startMouse.x) / Math.max(0.0001, dragInfo.scale);
                const dy = (mouse.y - dragInfo.startMouse.y) / Math.max(0.0001, dragInfo.scale);
                const next = { ...dragInfo.startState };

                if (dragInfo.side === "left") next.left = clampNonNegativeInt(dragInfo.startState.left - dx);
                if (dragInfo.side === "right") next.right = clampNonNegativeInt(dragInfo.startState.right + dx);
                if (dragInfo.side === "top") next.top = clampNonNegativeInt(dragInfo.startState.top - dy);
                if (dragInfo.side === "bottom") next.bottom = clampNonNegativeInt(dragInfo.startState.bottom + dy);

                expandState = sanitizeExpandState(next);
                userAdjusted = true;
                syncSizeWidgetsFromState();
                syncExpandStateWidget();
                draw();
            });

            window.addEventListener("mouseup", () => {
                dragInfo = null;
            });

            resetBtn.onclick = () => resetState();

            runBtn.onclick = async () => {
                try {
                    const p = await app.graphToPrompt();
                    const prompt = p.output;
                    const selectedNodeId = String(this.id);
                    const isolatedPrompt = {};
                    const traceDependencies = (nodeId) => {
                        if (!prompt[nodeId] || isolatedPrompt[nodeId]) return;
                        isolatedPrompt[nodeId] = prompt[nodeId];
                        const inputs = prompt[nodeId].inputs;
                        for (const key in inputs) {
                            const val = inputs[key];
                            if (Array.isArray(val) && val.length === 2) traceDependencies(String(val[0]));
                        }
                    };
                    traceDependencies(selectedNodeId);
                    if (Object.keys(isolatedPrompt).length === 0) return;
                    const response = await api.fetchApi("/prompt", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            client_id: api.clientId,
                            prompt: isolatedPrompt,
                            extra_data: p.workflow ? { extra_pnginfo: { workflow: p.workflow } } : {},
                        }),
                    });
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.error || "Failed to queue prompt");
                    }
                } catch (error) {}
            };

            const resizeObserver = new ResizeObserver(() => draw());
            resizeObserver.observe(viewArea);

            this._aptImagePadKeepVisualUI = {
                setImage: (url, w, h) => {
                    if (!url) return;
                    const img = new Image();
                    img.onload = () => {
                        bgImg = img;
                        imgWidth = Math.max(1, Math.round(Number(w || img.naturalWidth || img.width || 1)));
                        imgHeight = Math.max(1, Math.round(Number(h || img.naturalHeight || img.height || 1)));
                        edgeColor = computeEdgeColor(img);
                        updateWidgetMinimums();

                        const parsedState = parseExpandStateValue(expandStateWidget?.value ?? "{}");
                        const sizeLooksDefault =
                            Number(widthWidget?.value ?? 0) === initialWidthValue &&
                            Number(heightWidget?.value ?? 0) === initialHeightValue;
                        const hasSavedExpansion = parsedState.left || parsedState.right || parsedState.top || parsedState.bottom;

                        if (hasSavedExpansion) {
                            expandState = parsedState;
                            syncSizeWidgetsFromState();
                            syncExpandStateWidget();
                        } else if (!sizeLooksDefault) {
                            applyStateFromSizeWidgets();
                        } else if (!userAdjusted) {
                            expandState = sanitizeExpandState({});
                            syncSizeWidgetsFromState();
                            syncExpandStateWidget();
                        }

                        draw();
                    };
                    img.src = url;
                },
                setOriginalSize: (w, h) => {
                    if (w > 0 && h > 0) {
                        originalSize = { w, h };
                        imgWidth = w;
                        imgHeight = h;
                        updateWidgetMinimums();
                    }
                },
                draw,
                setUserAdjusted: (value) => {
                    userAdjusted = !!value;
                },
                resetState,
            };

            setTimeout(() => {
                if (!originalSize && widthWidget && heightWidget && expandStateWidget) {
                    expandState = parseExpandStateValue(expandStateWidget.value);
                }
                draw();
            }, 120);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);

            const ui = this._aptImagePadKeepVisualUI;
            if (!ui) return;

            const origSize = Array.isArray(message?.original_size) ? message.original_size[0] : null;
            if (origSize && origSize.w > 0 && origSize.h > 0) {
                ui.setOriginalSize(origSize.w, origSize.h);
            }

            if (Array.isArray(message?.preview) && message.preview.length > 0) {
                const img = message.preview[0];
                const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                if (origSize && origSize.w > 0 && origSize.h > 0) {
                    ui.setImage(url, origSize.w, origSize.h);
                } else {
                    ui.setImage(url);
                }
            }
        };
    },
});
