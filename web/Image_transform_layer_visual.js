import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let _aptPointerHooksInited = false;
let _aptPointerDown = false;
let _aptResizeNode = null;
const _aptEnsurePointerHooks = () => {
    if (_aptPointerHooksInited) return;
    _aptPointerHooksInited = true;
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
                const x0 = n.pos[0], y0 = n.pos[1];
                const x1 = x0 + n.size[0], y1 = y0 + n.size[1];
                const mx = m[0], my = m[1];
                if (mx < x0 || mx > x1 || my < y0 || my > y1) continue;
                if (mx >= x1 - 30 && my >= y1 - 30) {
                    _aptResizeNode = n;
                    return;
                }
            }
            const n = c?.node_over;
            if (n && Array.isArray(n.pos) && Array.isArray(n.size)) {
                const mx = m[0], my = m[1];
                const x0 = n.pos[0], y0 = n.pos[1];
                const x1 = x0 + n.size[0], y1 = y0 + n.size[1];
                if (mx >= x1 - 30 && my >= y1 - 30) _aptResizeNode = n;
            }
        } catch (e) {}
    };
    const onDown = () => {
        _aptPointerDown = true;
        _aptResizeNode = null;
        pickResizeNodeFromMouse();
    };
    const onMove = () => {
        if (!_aptPointerDown) return;
        pickResizeNodeFromMouse();
    };
    const onUp = () => {
        _aptPointerDown = false;
        _aptResizeNode = null;
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

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const rad2deg = (r) => r * 180 / Math.PI;
const deg2rad = (d) => d * Math.PI / 180;

app.registerExtension({
    name: "apt.Image_transform_layer_visual",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        _aptEnsurePointerHooks();
        if (nodeData.name !== "Image_transform_layer_visual") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const MIN_NODE_WIDTH = 260;
            const MIN_NODE_HEIGHT = 420;
            if (this.size[0] < MIN_NODE_WIDTH) this.size[0] = MIN_NODE_WIDTH;
            if (this.size[1] < MIN_NODE_HEIGHT) this.size[1] = MIN_NODE_HEIGHT;
            this.resizable = true;
            this.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
            this._apt_target_size = this.size;
            const _apt_orig_onResize = this.onResize;
            this.onResize = function (size) {
                this._apt_target_size = size;
                return _apt_orig_onResize?.apply(this, arguments);
            };

            const hideWidgetAndSlot = (widgetName) => {
                const w = this.widgets?.find(w => w.name === widgetName);
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
                    const idx = this.inputs.findIndex(i => i.name === widgetName);
                    if (idx !== -1) this.removeInput(idx);
                }
            };

            hideWidgetAndSlot("transform_state");

            const transformStateWidget = this.widgets?.find(w => w.name === "transform_state");
            const bgFillWidget = this.widgets?.find(w => w.name === "bg_fill");
            const opacityWidget = this.widgets?.find(w => w.name === "opacity");
            const blendingModeWidget = this.widgets?.find(w => w.name === "blending_mode");
            const blendStrengthWidget = this.widgets?.find(w => w.name === "blend_strength");

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

            const bgImageLayer = document.createElement("div");
            bgImageLayer.className = "apt-preview-bg";
            bgImageLayer.style.position = "absolute";
            bgImageLayer.style.inset = "0";
            bgImageLayer.style.backgroundSize = "contain";
            bgImageLayer.style.backgroundPosition = "center";
            bgImageLayer.style.backgroundRepeat = "no-repeat";
            viewArea.appendChild(bgImageLayer);

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

            const loadBtn = document.createElement("button");
            loadBtn.innerText = "Preview";
            loadBtn.style.flex = "1";
            loadBtn.style.width = "auto";
            loadBtn.style.height = "24px";
            loadBtn.style.lineHeight = "22px";
            loadBtn.style.marginTop = "8px";
            loadBtn.style.border = "none";
            loadBtn.style.borderRadius = "8px";
            loadBtn.style.cursor = "pointer";
            loadBtn.style.fontSize = "10px";
            loadBtn.style.fontWeight = "bold";
            loadBtn.style.backgroundColor = "#4f5d6d";
            loadBtn.style.color = "#FFF";
            loadBtn.style.transition = "all 0.2s ease";

            const runPreview = async () => {
                try {
                    const p = await app.graphToPrompt();
                    const prompt = p.output;
                    const selectedNodeId = String(this.id);
                    const isolatedPrompt = {};
                    const traceDependencies = (nodeId) => {
                        if (!prompt[nodeId] || isolatedPrompt[nodeId]) return;
                        isolatedPrompt[nodeId] = prompt[nodeId];
                        const inputs = prompt[nodeId].inputs;
                        for (let key in inputs) {
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
                            extra_data: p.workflow ? { extra_pnginfo: { workflow: p.workflow } } : {}
                        })
                    });
                    if (!response.ok) {
                        const error = await response.json();
                        throw new Error(error.error || "Failed to queue prompt");
                    }
                } catch (err) {}
            };
            loadBtn.onclick = runPreview;

            previewBar.style.marginBottom = isNodes2_0 ? "10px" : "0px";
            previewBar.appendChild(loadBtn);
            container.appendChild(previewBar);

            const widget = this.addDOMWidget("ImageLayerVisualUI", "div", container, { serialize: false, hideOnZoom: false });
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
                if (_aptPointerDown && _aptResizeNode && (_aptResizeNode === nodeInstance || _aptResizeNode?.id === nodeInstance.id) && Array.isArray(m) && m.length > 1 && Array.isArray(nodeInstance.pos)) {
                    return Math.max(MIN_NODE_HEIGHT, m[1] - nodeInstance.pos[1]);
                }
                return nodeInstance.size[1];
            };
            widget.computeSize = function (width) {
                const targetH = getTargetNodeHeight();
                const extra = Math.max(0, (targetH - UI_BASE_HEIGHT) * 0.95);
                return [width, UI_DEFAULT_HEIGHT + extra];
            };

            const imageMeta = { bg_w: 0, bg_h: 0, layer_w: 0, layer_h: 0 };
            let bgImg = null;
            let fgImg = null;
            let userAdjusted = false;
            let dragInfo = null;
            let transformState = { cx: 0.5, cy: 0.5, w: 0.25, h: 0.25, rot: 0 };

            const getOpacity = () => {
                const v = parseFloat(opacityWidget?.value ?? 1);
                return Number.isFinite(v) ? clamp(v, 0, 1) : 1;
            };
            const getBlendStrength = () => {
                const v = parseFloat(blendStrengthWidget?.value ?? 1);
                return Number.isFinite(v) ? clamp(v, 0, 1) : 1;
            };
            const getBlendMode = () => {
                const v = String(blendingModeWidget?.value ?? "正常");
                return v || "正常";
            };
            const getCanvasComposite = (mode) => {
                if (mode === "正常" || mode === "溶解") return "source-over";
                if (mode === "变暗") return "darken";
                if (mode === "正片叠底") return "multiply";
                if (mode === "颜色加深") return "color-burn";
                if (mode === "线性加深") return "darken";
                if (mode === "深色") return "darken";
                if (mode === "变亮") return "lighten";
                if (mode === "滤色") return "screen";
                if (mode === "颜色减淡") return "color-dodge";
                if (mode === "线性减淡（添加）") return "lighter";
                if (mode === "浅色") return "lighten";
                if (mode === "叠加") return "overlay";
                if (mode === "柔光") return "soft-light";
                if (mode === "强光") return "hard-light";
                if (mode === "亮光") return "hard-light";
                if (mode === "线性光") return "hard-light";
                if (mode === "点光") return "hard-light";
                if (mode === "实色混合") return "hard-light";
                if (mode === "差值") return "difference";
                if (mode === "排除") return "exclusion";
                if (mode === "减去") return "difference";
                if (mode === "划分") return "color-dodge";
                if (mode === "色相") return "hue";
                if (mode === "饱和度") return "saturation";
                if (mode === "颜色") return "color";
                if (mode === "明度") return "luminosity";
                return "source-over";
            };

            const parseState = () => {
                if (!transformStateWidget) return;
                if (typeof transformStateWidget.value !== "string") return;
                try {
                    const parsed = JSON.parse(transformStateWidget.value || "{}");
                    if (parsed && typeof parsed === "object") transformState = { ...transformState, ...parsed };
                } catch (e) {}
            };
            const syncState = () => {
                if (!transformStateWidget) return;
                transformStateWidget.value = JSON.stringify({
                    cx: +transformState.cx,
                    cy: +transformState.cy,
                    w: +transformState.w,
                    h: +transformState.h,
                    rot: +transformState.rot
                });
                if (app.graph) app.graph.setDirtyCanvas(true);
            };

            const getImageRectOnCanvas = () => {
                if (!imageMeta.bg_w || !imageMeta.bg_h) return null;
                const w = canvas.width;
                const h = canvas.height;
                if (w <= 0 || h <= 0) return null;
                const pad = Math.max(12, Math.min(64, Math.floor(Math.min(w, h) * 0.06)));
                const availW = Math.max(1, w - pad * 2);
                const availH = Math.max(1, h - pad * 2);
                const scale = Math.min(availW / imageMeta.bg_w, availH / imageMeta.bg_h);
                const dw = imageMeta.bg_w * scale;
                const dh = imageMeta.bg_h * scale;
                return {
                    x: (w - dw) * 0.5,
                    y: (h - dh) * 0.5,
                    w: dw,
                    h: dh,
                    scale
                };
            };
            const canvasToImage = (mx, my) => {
                const ir = getImageRectOnCanvas();
                if (!ir) return null;
                const ix = (mx - ir.x) / ir.scale;
                const iy = (my - ir.y) / ir.scale;
                return { x: ix, y: iy };
            };
            const getMouse = (e) => {
                const rect = canvas.getBoundingClientRect();
                const sx = canvas.width / rect.width;
                const sy = canvas.height / rect.height;
                return {
                    x: (e.clientX - rect.left) * sx,
                    y: (e.clientY - rect.top) * sy
                };
            };

            const ensureInitState = () => {
                if (userAdjusted) return;
                if (!imageMeta.bg_w || !imageMeta.bg_h || !imageMeta.layer_w || !imageMeta.layer_h) return;
                transformState.cx = 0.5;
                transformState.cy = 0.5;
                transformState.w = imageMeta.layer_w / imageMeta.bg_w;
                transformState.h = imageMeta.layer_h / imageMeta.bg_h;
                transformState.rot = 0;
                syncState();
            };

            const getRectPx = () => {
                const bgW = imageMeta.bg_w || 1;
                const bgH = imageMeta.bg_h || 1;
                const cxPx = transformState.cx * bgW;
                const cyPx = transformState.cy * bgH;
                const wPx = Math.max(1, Math.abs(transformState.w) * bgW);
                const hPx = Math.max(1, Math.abs(transformState.h) * bgH);
                const rot = +transformState.rot || 0;
                return { cxPx, cyPx, wPx, hPx, rot };
            };

            const hitTest = (imgPt) => {
                if (!imgPt || !imageMeta.bg_w || !imageMeta.bg_h) return null;
                const { cxPx, cyPx, wPx, hPx, rot } = getRectPx();
                const dx = imgPt.x - cxPx;
                const dy = imgPt.y - cyPx;
                const a = deg2rad(rot);
                const ca = Math.cos(-a), sa = Math.sin(-a);
                const lx = dx * ca - dy * sa;
                const ly = dx * sa + dy * ca;
                const inside = Math.abs(lx) <= wPx * 0.5 && Math.abs(ly) <= hPx * 0.5;
                const ir = getImageRectOnCanvas();
                const handleR = ir ? (10 / ir.scale) : 8;
                const corners = [
                    { id: "tl", x: -wPx * 0.5, y: -hPx * 0.5 },
                    { id: "tr", x: wPx * 0.5, y: -hPx * 0.5 },
                    { id: "br", x: wPx * 0.5, y: hPx * 0.5 },
                    { id: "bl", x: -wPx * 0.5, y: hPx * 0.5 },
                ];
                const ca2 = Math.cos(a), sa2 = Math.sin(a);
                for (const c of corners) {
                    const gx = cxPx + c.x * ca2 - c.y * sa2;
                    const gy = cyPx + c.x * sa2 + c.y * ca2;
                    const ddx = imgPt.x - gx;
                    const ddy = imgPt.y - gy;
                    if (ddx * ddx + ddy * ddy <= handleR * handleR) return { mode: "scale", corner: c.id };
                }
                const rotOffset = ir ? (30 / ir.scale) : 30;
                const rotHandle = { x: 0, y: -hPx * 0.5 - rotOffset };
                const rhx = cxPx + rotHandle.x * ca2 - rotHandle.y * sa2;
                const rhy = cyPx + rotHandle.x * sa2 + rotHandle.y * ca2;
                const rdx = imgPt.x - rhx;
                const rdy = imgPt.y - rhy;
                if (rdx * rdx + rdy * rdy <= handleR * handleR) return { mode: "rotate" };
                if (inside) return { mode: "move" };
                return null;
            };

            const draw = () => {
                const clientW = canvas.clientWidth;
                const clientH = canvas.clientHeight;
                if (clientW > 0 && clientH > 0) {
                    if (canvas.width !== clientW || canvas.height !== clientH) {
                        canvas.width = clientW;
                        canvas.height = clientH;
                    }
                }
                const w = canvas.width;
                const h = canvas.height;
                if (!w || !h) return;
                ctx.clearRect(0, 0, w, h);
                const ir = getImageRectOnCanvas();
                if (!ir) return;
                ensureInitState();
                parseState();
                const { cxPx, cyPx, wPx, hPx, rot } = getRectPx();
                const opacity = getOpacity();
                const strength = getBlendStrength();
                const mode = getBlendMode();
                const gco = getCanvasComposite(mode);

                if (bgImg) {
                    ctx.save();
                    ctx.globalAlpha = 1.0;
                    ctx.globalCompositeOperation = "source-over";
                    ctx.drawImage(bgImg, ir.x, ir.y, ir.w, ir.h);
                    ctx.restore();
                }

                if (fgImg && imageMeta.layer_w && imageMeta.layer_h) {
                    const cx = ir.x + cxPx * ir.scale;
                    const cy = ir.y + cyPx * ir.scale;
                    ctx.save();
                    ctx.translate(cx, cy);
                    ctx.rotate(deg2rad(rot));
                    if (mode === "正常" || gco === "source-over") {
                        ctx.globalCompositeOperation = "source-over";
                        ctx.globalAlpha = opacity;
                        ctx.drawImage(
                            fgImg,
                            -wPx * 0.5 * ir.scale,
                            -hPx * 0.5 * ir.scale,
                            wPx * ir.scale,
                            hPx * ir.scale
                        );
                    } else {
                        const a1 = opacity * (1 - strength);
                        const a2 = opacity * strength;
                        if (a1 > 0) {
                            ctx.globalCompositeOperation = "source-over";
                            ctx.globalAlpha = a1;
                            ctx.drawImage(
                                fgImg,
                                -wPx * 0.5 * ir.scale,
                                -hPx * 0.5 * ir.scale,
                                wPx * ir.scale,
                                hPx * ir.scale
                            );
                        }
                        if (a2 > 0) {
                            ctx.globalCompositeOperation = gco;
                            ctx.globalAlpha = a2;
                            ctx.drawImage(
                                fgImg,
                                -wPx * 0.5 * ir.scale,
                                -hPx * 0.5 * ir.scale,
                                wPx * ir.scale,
                                hPx * ir.scale
                            );
                        }
                    }
                    ctx.restore();
                }

                const cx = ir.x + cxPx * ir.scale;
                const cy = ir.y + cyPx * ir.scale;
                ctx.save();
                ctx.translate(cx, cy);
                ctx.rotate(deg2rad(rot));
                ctx.strokeStyle = "#ff5a4f";
                ctx.lineWidth = 3;
                ctx.strokeRect(-wPx * 0.5 * ir.scale, -hPx * 0.5 * ir.scale, wPx * ir.scale, hPx * ir.scale);
                const hs = 6;
                ctx.fillStyle = "#ffea00";
                ctx.fillRect(-wPx * 0.5 * ir.scale - hs, -hPx * 0.5 * ir.scale - hs, hs * 2, hs * 2);
                ctx.fillRect(wPx * 0.5 * ir.scale - hs, -hPx * 0.5 * ir.scale - hs, hs * 2, hs * 2);
                ctx.fillRect(-wPx * 0.5 * ir.scale - hs, hPx * 0.5 * ir.scale - hs, hs * 2, hs * 2);
                ctx.fillRect(wPx * 0.5 * ir.scale - hs, hPx * 0.5 * ir.scale - hs, hs * 2, hs * 2);
                ctx.beginPath();
                ctx.strokeStyle = "#ffea00";
                ctx.lineWidth = 2;
                ctx.moveTo(0, -hPx * 0.5 * ir.scale);
                ctx.lineTo(0, -hPx * 0.5 * ir.scale - 30);
                ctx.stroke();
                ctx.fillStyle = "#ffea00";
                ctx.beginPath();
                ctx.arc(0, -hPx * 0.5 * ir.scale - 30, 6, 0, Math.PI * 2);
                ctx.fill();
                ctx.restore();
            };

            canvas.addEventListener("mousedown", (e) => {
                if (e.button !== 0) return;
                const ir = getImageRectOnCanvas();
                if (!ir) return;
                const m = getMouse(e);
                const imgPt = canvasToImage(m.x, m.y);
                const hit = hitTest(imgPt);
                canvas.style.cursor = hit ? (hit.mode === "rotate" ? "crosshair" : (hit.mode === "scale" ? "nwse-resize" : "move")) : "default";
                if (!hit) return;
                const { cxPx, cyPx, wPx, hPx, rot } = getRectPx();
                dragInfo = {
                    mode: hit.mode,
                    corner: hit.corner,
                    startCx: cxPx,
                    startCy: cyPx,
                    startW: wPx,
                    startH: hPx,
                    startRot: rot,
                    startX: imgPt.x,
                    startY: imgPt.y
                };
                if (hit.mode === "rotate") {
                    dragInfo.startAngle = Math.atan2(imgPt.y - cyPx, imgPt.x - cxPx);
                }
                e.preventDefault();
            });

            window.addEventListener("mousemove", (e) => {
                const m = getMouse(e);
                const imgPt = canvasToImage(m.x, m.y);
                if (!dragInfo) {
                    const hit = hitTest(imgPt);
                    canvas.style.cursor = hit ? (hit.mode === "rotate" ? "crosshair" : (hit.mode === "scale" ? "nwse-resize" : "move")) : "default";
                    return;
                }
                if (!imgPt) return;
                if (dragInfo.mode === "move") {
                    const dx = imgPt.x - dragInfo.startX;
                    const dy = imgPt.y - dragInfo.startY;
                    const nx = dragInfo.startCx + dx;
                    const ny = dragInfo.startCy + dy;
                    transformState.cx = nx / imageMeta.bg_w;
                    transformState.cy = ny / imageMeta.bg_h;
                } else if (dragInfo.mode === "scale") {
                    const cx = dragInfo.startCx;
                    const cy = dragInfo.startCy;
                    const rot = dragInfo.startRot;
                    const a = deg2rad(rot);
                    const ca = Math.cos(-a), sa = Math.sin(-a);
                    const dx = imgPt.x - cx;
                    const dy = imgPt.y - cy;
                    const lx = dx * ca - dy * sa;
                    const ly = dx * sa + dy * ca;
                    const scaleF = Math.max(
                        Math.abs(lx) / Math.max(1, dragInfo.startW * 0.5),
                        Math.abs(ly) / Math.max(1, dragInfo.startH * 0.5),
                        1e-3
                    );
                    const wPx = Math.max(2, dragInfo.startW * scaleF);
                    const hPx = Math.max(2, dragInfo.startH * scaleF);
                    transformState.w = wPx / imageMeta.bg_w;
                    transformState.h = hPx / imageMeta.bg_h;
                } else if (dragInfo.mode === "rotate") {
                    const cx = dragInfo.startCx;
                    const cy = dragInfo.startCy;
                    const ang = Math.atan2(imgPt.y - cy, imgPt.x - cx);
                    const delta = ang - dragInfo.startAngle;
                    transformState.rot = dragInfo.startRot + rad2deg(delta);
                }
                userAdjusted = true;
                syncState();
                draw();
            });

            window.addEventListener("mouseup", () => {
                dragInfo = null;
            });

            canvas.addEventListener("wheel", (e) => {
                if (!imageMeta.bg_w || !imageMeta.bg_h) return;
                e.preventDefault();
                const factor = e.deltaY < 0 ? 1.08 : (1 / 1.08);
                transformState.w = (+transformState.w || 0.001) * factor;
                transformState.h = (+transformState.h || 0.001) * factor;
                userAdjusted = true;
                syncState();
                draw();
            }, { passive: false });

            const bindWidgetRedraw = (w) => {
                if (!w) return;
                const originalCallback = w.callback;
                w.callback = function () {
                    if (originalCallback) originalCallback.apply(this, arguments);
                    draw();
                };
                if (w.inputEl) {
                    w.inputEl.addEventListener("change", draw);
                    w.inputEl.addEventListener("input", draw);
                }
            };
            bindWidgetRedraw(bgFillWidget);
            bindWidgetRedraw(opacityWidget);
            bindWidgetRedraw(blendingModeWidget);
            bindWidgetRedraw(blendStrengthWidget);

            const resizeObserver = new ResizeObserver(() => draw());
            resizeObserver.observe(viewArea);

            this._aptImageTransformLayerVisualUI = {
                setMeta: (meta) => {
                    if (!meta) return;
                    imageMeta.bg_w = parseInt(meta.bg_w) || 0;
                    imageMeta.bg_h = parseInt(meta.bg_h) || 0;
                    imageMeta.layer_w = parseInt(meta.layer_w) || 0;
                    imageMeta.layer_h = parseInt(meta.layer_h) || 0;
                    if (!userAdjusted) {
                        ensureInitState();
                    }
                    draw();
                },
                setFgUrl: (url) => {
                    if (!url) return;
                    const img = new Image();
                    img.onload = () => {
                        fgImg = img;
                        draw();
                    };
                    img.src = url;
                },
                    setBgUrl: (url) => {
                        if (!url) return;
                        const img = new Image();
                        img.onload = () => {
                            bgImg = img;
                            bgImageLayer.style.backgroundImage = "";
                            draw();
                        };
                        img.src = url;
                    },
                draw,
                setUserAdjusted: (v) => { userAdjusted = !!v; },
            };

            setTimeout(() => draw(), 120);
        };

        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            const uiWidget = this.widgets?.find(w => w.name === "ImageLayerVisualUI");
            const root = uiWidget?.element;
            if (!root) return;
            const bg = root.querySelector?.(".apt-preview-bg");
            if (bg && message?.bg_image?.length > 0) {
                const img = message.bg_image[0];
                const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                bg.style.backgroundImage = `url(${url})`;
                    this._aptImageTransformLayerVisualUI?.setBgUrl?.(url);
            }
            const ui = this._aptImageTransformLayerVisualUI;
            if (!ui) return;
            const meta = Array.isArray(message?.layer_meta) ? message.layer_meta[0] : null;
            if (meta) ui.setMeta(meta);
            if (Array.isArray(message?.fg_image) && message.fg_image.length > 0) {
                const img = message.fg_image[0];
                const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                ui.setFgUrl(url);
            }
        };
    },
});
