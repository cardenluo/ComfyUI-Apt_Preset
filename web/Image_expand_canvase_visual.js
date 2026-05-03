import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let _aptPadPointerHooksInited = false;
let _aptPadPointerDown = false;
let _aptPadResizeNode = null;

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
                const x0 = n.pos[0], y0 = n.pos[1];
                const x1 = x0 + n.size[0], y1 = y0 + n.size[1];
                const mx = m[0], my = m[1];
                if (mx < x0 || mx > x1 || my < y0 || my > y1) continue;
                if (mx >= x1 - 30 && my >= y1 - 30) {
                    _aptPadResizeNode = n;
                    return;
                }
            }
            const n = c?.node_over;
            if (n && Array.isArray(n.pos) && Array.isArray(n.size)) {
                const mx = m[0], my = m[1];
                const x0 = n.pos[0], y0 = n.pos[1];
                const x1 = x0 + n.size[0], y1 = y0 + n.size[1];
                if (mx >= x1 - 30 && my >= y1 - 30) _aptPadResizeNode = n;
            }
        } catch (e) {}
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

const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
const rad2deg = (r) => r * 180 / Math.PI;
const deg2rad = (d) => d * Math.PI / 180;
const SNAP_DISTANCE_PX = 16;

app.registerExtension({
    name: "apt.Image_expand_canvase_visual",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        _aptPadEnsurePointerHooks();
        if (nodeData.name !== "Image_expand_canvase_visual") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);

            const MIN_NODE_WIDTH = 280;
            const MIN_NODE_HEIGHT = 400;
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

            // 隐藏 transform_state widget
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
            const widthWidget = this.widgets?.find(w => w.name === "width");
            const heightWidget = this.widgets?.find(w => w.name === "height");
            const constantColorWidget = this.widgets?.find(w => w.name === "constant_color");
            const alignPositionWidget = this.widgets?.find(w => w.name === "align_position");

            // 创建UI容器
            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.flexDirection = "column";
            container.style.width = "100%";
            container.style.height = "100%";
            container.style.marginTop = "0px";
            container.style.borderRadius = "6px";
            container.style.overflow = "hidden";
            container.style.backgroundColor = "transparent";

            // 画布区域
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

            // 控制栏
            const controlBar = document.createElement("div");
            controlBar.style.display = "flex";
            controlBar.style.alignItems = "center";
            controlBar.style.gap = "8px";
            controlBar.style.padding = "8px 0";
            controlBar.style.backgroundColor = "transparent";

            // 重置按钮
            const resetBtn = document.createElement("button");
            resetBtn.innerText = "Reset";
            resetBtn.style.flex = "1";
            resetBtn.style.height = "24px";
            resetBtn.style.lineHeight = "22px";
            resetBtn.style.border = "none";
            resetBtn.style.borderRadius = "8px";
            resetBtn.style.cursor = "pointer";
            resetBtn.style.fontSize = "10px";
            resetBtn.style.fontWeight = "bold";
            resetBtn.style.backgroundColor = "#4f5d6d";
            resetBtn.style.color = "#FFF";
            resetBtn.style.transition = "all 0.2s ease";
            controlBar.appendChild(resetBtn);

            // 运行按钮
            const runBtn = document.createElement("button");
            runBtn.innerText = "Preview";
            runBtn.style.flex = "1";
            runBtn.style.height = "24px";
            runBtn.style.lineHeight = "22px";
            runBtn.style.border = "none";
            runBtn.style.borderRadius = "8px";
            runBtn.style.cursor = "pointer";
            runBtn.style.fontSize = "10px";
            runBtn.style.fontWeight = "bold";
            runBtn.style.backgroundColor = "#2d8a3e";
            runBtn.style.color = "#FFF";
            runBtn.style.transition = "all 0.2s ease";
            controlBar.appendChild(runBtn);

            container.appendChild(controlBar);

            // 隐藏信息显示
            // 移除信息栏代码

            const widget = this.addDOMWidget("ImagePadKeepVisualUI", "div", container, { serialize: false, hideOnZoom: false });
            const nodeInstance = this;
            const ctx = canvas.getContext("2d");
            
            const UI_DEFAULT_HEIGHT = 280;
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

            // 状态变量
            let bgImg = null;
            let canvasWidth = 512;
            let canvasHeight = 512;
            let imgWidth = 0;
            let imgHeight = 0;
            // 变换状态
            let transformState = { x: 0, y: 0, scale: 1.0, angle: 0 };
            let dragInfo = null;
            let userAdjusted = false;

            const parseState = () => {
                if (!transformStateWidget) return;
                if (typeof transformStateWidget.value !== "string") return;
                try {
                    const parsed = JSON.parse(transformStateWidget.value || "{}");
                    if (parsed && typeof parsed === "object") {
                        transformState = {
                            ...transformState,
                            x: Number.isFinite(+parsed.x) ? +parsed.x : transformState.x,
                            y: Number.isFinite(+parsed.y) ? +parsed.y : transformState.y,
                            scale: Number.isFinite(+parsed.scale) ? +parsed.scale : transformState.scale,
                            angle: Number.isFinite(+parsed.angle) ? +parsed.angle : transformState.angle
                        };
                    }
                } catch (e) {}
            };
            
            const syncState = () => {
                if (!transformStateWidget) return;
                transformStateWidget.value = JSON.stringify({
                    x: +transformState.x,
                    y: +transformState.y,
                    scale: +transformState.scale,
                    angle: +transformState.angle
                });
                if (app.graph) app.graph.setDirtyCanvas(true);
            };

            const getCanvasSize = () => {
                const w = parseInt(widthWidget?.value ?? 512);
                const h = parseInt(heightWidget?.value ?? 512);
                return { w: Math.max(1, w), h: Math.max(1, h) };
            };

            const getAlignAnchor = () => {
                const align = alignPositionWidget?.value ?? "mid-center";
                const map = {
                    "left-top": { x: 0, y: 0 },
                    "mid-top": { x: 0.5, y: 0 },
                    "right-top": { x: 1, y: 0 },
                    "left-center": { x: 0, y: 0.5 },
                    "mid-center": { x: 0.5, y: 0.5 },
                    "right-center": { x: 1, y: 0.5 },
                    "left-bottom": { x: 0, y: 1 },
                    "mid-bottom": { x: 0.5, y: 1 },
                    "right-bottom": { x: 1, y: 1 }
                };
                return map[align] || map["mid-center"];
            };

            const getImageRectOnCanvas = () => {
                if (!imgWidth || !imgHeight) return null;
                const cw = canvas.width;
                const ch = canvas.height;
                if (cw <= 0 || ch <= 0) return null;
                
                const pad = Math.max(12, Math.min(64, Math.floor(Math.min(cw, ch) * 0.06)));
                const availW = Math.max(1, cw - pad * 2);
                const availH = Math.max(1, ch - pad * 2);
                
                // 计算预览缩放比例（画布尺寸映射到显示区域）
                const previewScaleX = availW / canvasWidth;
                const previewScaleY = availH / canvasHeight;
                const previewScale = Math.min(previewScaleX, previewScaleY);
                
                const dw = canvasWidth * previewScale;
                const dh = canvasHeight * previewScale;
                
                return {
                    x: (cw - dw) * 0.5,
                    y: (ch - dh) * 0.5,
                    w: dw,
                    h: dh,
                    scale: previewScale
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
                if (!imgWidth || !imgHeight) return;
                
                // 初始化：图片居中，缩放为1.0（基础缩放已计算）
                transformState.x = 0;
                transformState.y = 0;
                transformState.scale = 1.0;
                transformState.angle = 0;
                syncState();
            };

            const getImageDisplayRect = (state = transformState) => {
                if (!imgWidth || !imgHeight) return null;

                const finalScale = Math.max(0.01, +state.scale || 1.0);
                const w = imgWidth * finalScale;
                const h = imgHeight * finalScale;

                const anchor = getAlignAnchor();
                const centerX = (canvasWidth - w) * anchor.x + w / 2 + (+state.x || 0);
                const centerY = (canvasHeight - h) * anchor.y + h / 2 + (+state.y || 0);

                return {
                    cx: centerX,
                    cy: centerY,
                    w,
                    h,
                    scale: finalScale,
                    imgAspect: imgWidth / imgHeight
                };
            };

            const getTransformedBounds = (state = transformState) => {
                const rect = getImageDisplayRect(state);
                if (!rect) return null;

                const angle = deg2rad(-(Number.isFinite(+state.angle) ? +state.angle : 0));
                const ca = Math.abs(Math.cos(angle));
                const sa = Math.abs(Math.sin(angle));
                const halfW = (rect.w * ca + rect.h * sa) * 0.5;
                const halfH = (rect.w * sa + rect.h * ca) * 0.5;

                return {
                    left: rect.cx - halfW,
                    right: rect.cx + halfW,
                    top: rect.cy - halfH,
                    bottom: rect.cy + halfH
                };
            };

            const getSnapDelta = (candidates) => {
                let best = null;
                for (const c of candidates) {
                    if (!Number.isFinite(c.delta) || Math.abs(c.distance) > SNAP_DISTANCE_PX) continue;
                    if (!best || Math.abs(c.distance) < Math.abs(best.distance)) best = c;
                }
                return best?.delta ?? 0;
            };

            const snapMoveState = (state) => {
                const bounds = getTransformedBounds(state);
                if (!bounds) return state;

                const dx = getSnapDelta([
                    { distance: bounds.left, delta: -bounds.left },
                    { distance: bounds.right - canvasWidth, delta: canvasWidth - bounds.right }
                ]);
                const dy = getSnapDelta([
                    { distance: bounds.top, delta: -bounds.top },
                    { distance: bounds.bottom - canvasHeight, delta: canvasHeight - bounds.bottom }
                ]);

                return {
                    ...state,
                    x: state.x + dx,
                    y: state.y + dy
                };
            };

            const snapScaleValue = (scale, baseState = transformState) => {
                const candidateState = {
                    x: Number.isFinite(+baseState.x) ? +baseState.x : 0,
                    y: Number.isFinite(+baseState.y) ? +baseState.y : 0,
                    scale: Math.max(0.1, Math.min(10.0, Number.isFinite(+scale) ? +scale : 1.0)),
                    angle: Number.isFinite(+baseState.angle) ? +baseState.angle : 0
                };
                const bounds = getTransformedBounds(candidateState);
                if (!bounds) return candidateState.scale;

                const anchor = getAlignAnchor();
                const angle = deg2rad(-candidateState.angle);
                const absCos = Math.abs(Math.cos(angle));
                const absSin = Math.abs(Math.sin(angle));
                const halfSpanX = 0.5 * (imgWidth * absCos + imgHeight * absSin);
                const halfSpanY = 0.5 * (imgWidth * absSin + imgHeight * absCos);
                const centerCoeffX = imgWidth * (0.5 - anchor.x);
                const centerCoeffY = imgHeight * (0.5 - anchor.y);
                const baseX = canvasWidth * anchor.x + candidateState.x;
                const baseY = canvasHeight * anchor.y + candidateState.y;

                const maybePush = (list, distance, target, denom) => {
                    if (Math.abs(distance) > SNAP_DISTANCE_PX || Math.abs(denom) < 1e-6) return;
                    const nextScale = target / denom;
                    if (Number.isFinite(nextScale) && nextScale > 0) {
                        list.push({ scale: nextScale, distance });
                    }
                };

                const candidates = [];
                maybePush(candidates, bounds.left, -baseX, centerCoeffX - halfSpanX);
                maybePush(candidates, bounds.right - canvasWidth, canvasWidth - baseX, centerCoeffX + halfSpanX);
                maybePush(candidates, bounds.top, -baseY, centerCoeffY - halfSpanY);
                maybePush(candidates, bounds.bottom - canvasHeight, canvasHeight - baseY, centerCoeffY + halfSpanY);

                if (candidates.length === 0) return candidateState.scale;

                candidates.sort((a, b) => Math.abs(a.distance) - Math.abs(b.distance));
                return clamp(candidates[0].scale, 0.1, 10.0);
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
                
                // 获取画布尺寸设置
                const canvasSize = getCanvasSize();
                canvasWidth = canvasSize.w;
                canvasHeight = canvasSize.h;
                
                const ir = getImageRectOnCanvas();
                if (!ir) return;
                
                ensureInitState();
                parseState();
                
                // 绘制画布边框
                ctx.save();
                ctx.strokeStyle = "#555";
                ctx.lineWidth = 2;
                ctx.strokeRect(ir.x, ir.y, ir.w, ir.h);
                
                // 绘制背景色（根据constant_color）
                const colorMap = {
                    "white": "#ffffff",
                    "black": "#000000",
                    "red": "#ff0000",
                    "gray": "#808080",
                    "edge": "#1a1a1a"
                };
                const bgColor = colorMap[constantColorWidget?.value ?? "black"] || "#000000";
                ctx.fillStyle = bgColor;
                ctx.fillRect(ir.x + 1, ir.y + 1, ir.w - 2, ir.h - 2);
                ctx.restore();
                
                if (bgImg && imgWidth && imgHeight) {
                    const rect = getImageDisplayRect();
                    if (rect) {
                        ctx.save();
                        
                        // 计算显示区域：将画布坐标映射到预览区域
                        const previewX = ir.x + rect.cx * ir.scale;
                        const previewY = ir.y + rect.cy * ir.scale;
                        const previewW = rect.w * ir.scale;
                        const previewH = rect.h * ir.scale;
                        
                        // 应用变换：先移动到中心，然后旋转
                        ctx.translate(previewX, previewY);
                        // 旋转方向取反，与后端一致
                        ctx.rotate(deg2rad(-transformState.angle));

                        // 绘制图片 - 保持原始宽高比
                        ctx.drawImage(bgImg, -previewW / 2, -previewH / 2, previewW, previewH);
                        
                        // 绘制边框 - 与图片边缘完全吻合
                        ctx.strokeStyle = "#ff5a4f";
                        ctx.lineWidth = 2;
                        const x = Math.floor(-previewW / 2) + 0.5;
                        const y = Math.floor(-previewH / 2) + 0.5;
                        const w = Math.floor(previewW);
                        const h = Math.floor(previewH);
                        ctx.strokeRect(x, y, w, h);
                        
                        // 绘制角点 - 位于边框外侧
                        const hs = 5;
                        ctx.fillStyle = "#ffea00";
                        ctx.fillRect(x - hs, y - hs, hs * 2, hs * 2);
                        ctx.fillRect(x + w - hs, y - hs, hs * 2, hs * 2);
                        ctx.fillRect(x - hs, y + h - hs, hs * 2, hs * 2);
                        ctx.fillRect(x + w - hs, y + h - hs, hs * 2, hs * 2);
                        
                        // 绘制旋转指示器 - 从顶部中心延伸
                        ctx.beginPath();
                        ctx.strokeStyle = "#ffea00";
                        ctx.lineWidth = 2;
                        ctx.moveTo(0, y);
                        ctx.lineTo(0, y - 25);
                        ctx.stroke();
                        ctx.beginPath();
                        ctx.arc(0, y - 25, 5, 0, Math.PI * 2);
                        ctx.fill();
                        
                        ctx.restore();
                    }
                }
                
                // 隐藏信息显示
                // 移除信息绘制代码
            };

            // 检测点击位置（图片内、旋转手柄、角点）
            const hitTest = (imgPt) => {
                if (!imgPt || !imgWidth || !imgHeight) return null;
                const rect = getImageDisplayRect();
                if (!rect) return null;
                
                const ir = getImageRectOnCanvas();
                if (!ir) return null;
                
                // 转换到图片本地坐标系
                const dx = imgPt.x - rect.cx;
                const dy = imgPt.y - rect.cy;
                const a = deg2rad(-transformState.angle);
                const ca = Math.cos(a), sa = Math.sin(a);
                const lx = dx * ca - dy * sa;
                const ly = dx * sa + dy * ca;
                
                // 检查是否在图片内
                const halfW = rect.w / 2;
                const halfH = rect.h / 2;
                const inside = Math.abs(lx) <= halfW && Math.abs(ly) <= halfH;
                
                // 检查旋转手柄（在图片上方）
                const rotHandleY = -halfH - 25 / ir.scale;
                const rotHandleDist = Math.sqrt(lx * lx + (ly - rotHandleY) * (ly - rotHandleY));
                const hitRotHandle = rotHandleDist <= 10 / ir.scale;
                
                // 检查四个角点（用于等比缩放）
                const corners = [
                    { id: "tl", x: -halfW, y: -halfH },
                    { id: "tr", x: halfW, y: -halfH },
                    { id: "bl", x: -halfW, y: halfH },
                    { id: "br", x: halfW, y: halfH }
                ];
                const handleR = 12 / ir.scale;
                for (const c of corners) {
                    const dist = Math.sqrt((lx - c.x) ** 2 + (ly - c.y) ** 2);
                    if (dist <= handleR) {
                        return { mode: "scale", corner: c.id };
                    }
                }
                
                if (hitRotHandle) return { mode: "rotate" };
                if (inside) return { mode: "move" };
                
                return null;
            };

            // 鼠标事件处理
            canvas.addEventListener("mousedown", (e) => {
                if (e.button !== 0) return;
                const ir = getImageRectOnCanvas();
                if (!ir || !bgImg) return;
                
                const m = getMouse(e);
                const imgPt = canvasToImage(m.x, m.y);
                if (!imgPt) return;
                
                const rect = getImageDisplayRect();
                if (!rect) return;
                
                // 使用hitTest检测点击位置
                const hit = hitTest(imgPt);
                
                if (hit) {
                    dragInfo = {
                        mode: hit.mode,
                        corner: hit.corner,
                        startX: imgPt.x,
                        startY: imgPt.y,
                        startState: { ...transformState }
                    };
                    if (hit.mode === "rotate") {
                        dragInfo.startAngle = Math.atan2(imgPt.y - rect.cy, imgPt.x - rect.cx);
                    }
                    e.preventDefault();
                }
            });

            window.addEventListener("mousemove", (e) => {
                const m = getMouse(e);
                const imgPt = canvasToImage(m.x, m.y);
                
                if (!dragInfo) {
                    // 更新鼠标样式
                    if (imgPt && bgImg) {
                        const hit = hitTest(imgPt);
                        if (hit?.mode === "rotate") {
                            canvas.style.cursor = "crosshair";
                        } else if (hit?.mode === "scale") {
                            canvas.style.cursor = "nwse-resize";
                        } else if (hit?.mode === "move") {
                            canvas.style.cursor = "move";
                        } else {
                            canvas.style.cursor = "default";
                        }
                    }
                    return;
                }
                
                if (!imgPt) return;
                
                if (dragInfo.mode === "move") {
                    const dx = imgPt.x - dragInfo.startX;
                    const dy = imgPt.y - dragInfo.startY;
                    const nextState = snapMoveState({
                        ...dragInfo.startState,
                        x: dragInfo.startState.x + dx,
                        y: dragInfo.startState.y + dy
                    });
                    transformState.x = nextState.x;
                    transformState.y = nextState.y;
                } else if (dragInfo.mode === "rotate") {
                    const rect = getImageDisplayRect();
                    if (rect) {
                        const ang = Math.atan2(imgPt.y - rect.cy, imgPt.x - rect.cx);
                        const delta = ang - dragInfo.startAngle;
                        // 旋转方向取反，与后端一致
                        transformState.angle = dragInfo.startState.angle - rad2deg(delta);
                    }
                } else if (dragInfo.mode === "scale") {
                    const startRect = getImageDisplayRect(dragInfo.startState);
                    if (startRect) {
                        const cx = startRect.cx;
                        const cy = startRect.cy;
                        const a = deg2rad(-dragInfo.startState.angle);
                        const ca = Math.cos(a), sa = Math.sin(a);
                        const dx = imgPt.x - cx;
                        const dy = imgPt.y - cy;
                        const lx = dx * ca - dy * sa;
                        const ly = dx * sa + dy * ca;
                        const scaleFactor = Math.max(
                            Math.abs(lx) / Math.max(1, startRect.w * 0.5),
                            Math.abs(ly) / Math.max(1, startRect.h * 0.5),
                            0.01
                        );
                        const nextScale = clamp(dragInfo.startState.scale * scaleFactor, 0.1, 10.0);
                        transformState.scale = snapScaleValue(nextScale, dragInfo.startState);
                    }
                }
                
                userAdjusted = true;
                syncState();
                draw();
            });

            window.addEventListener("mouseup", () => {
                dragInfo = null;
            });

            // 绑定widget变化重绘
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
            bindWidgetRedraw(widthWidget);
            bindWidgetRedraw(heightWidget);
            bindWidgetRedraw(constantColorWidget);
            bindWidgetRedraw(alignPositionWidget);

            // 重置按钮
            resetBtn.onclick = () => {
                transformState = { x: 0, y: 0, scale: 1.0, angle: 0 };
                userAdjusted = false;
                syncState();
                draw();
            };

            // 运行预览
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

            // 调整大小观察
            const resizeObserver = new ResizeObserver(() => draw());
            resizeObserver.observe(viewArea);

            // 存储原始图片尺寸（用于红框计算）
            let originalImgWidth = 0;
            let originalImgHeight = 0;
            
            // 存储UI引用
            this._aptImagePadKeepVisualUI = {
                setImage: (url, w, h) => {
                    if (!url) return;
                    const img = new Image();
                    img.onload = () => {
                        bgImg = img;
                        // 设置图片尺寸（优先使用传入的尺寸，否则使用图片实际尺寸）
                        imgWidth = w || img.naturalWidth;
                        imgHeight = h || img.naturalHeight;
                        if (!userAdjusted) {
                            ensureInitState();
                        }
                        draw();
                    };
                    img.src = url;
                },
                setOriginalSize: (w, h) => {
                    originalImgWidth = w;
                    originalImgHeight = h;
                },
                draw,
                setUserAdjusted: (v) => { userAdjusted = !!v; }
            };

            setTimeout(() => draw(), 120);
        };

        // 处理执行结果
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            
            const ui = this._aptImagePadKeepVisualUI;
            if (!ui) return;
            
            // 处理原始尺寸信息
            const origSize = Array.isArray(message?.original_size) ? message.original_size[0] : null;
            if (origSize && origSize.w > 0 && origSize.h > 0) {
                ui.setOriginalSize(origSize.w, origSize.h);
            }
            
            // 处理预览图（原始输入图片）
            if (Array.isArray(message?.preview) && message.preview.length > 0) {
                const img = message.preview[0];
                const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                // 使用原始尺寸
                if (origSize && origSize.w > 0 && origSize.h > 0) {
                    ui.setImage(url, origSize.w, origSize.h);
                } else {
                    ui.setImage(url);
                }
            }
        };
    },
});
