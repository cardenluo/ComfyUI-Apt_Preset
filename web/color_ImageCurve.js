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

app.registerExtension({
    name: "apt.colorImageCurve",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        _aptEnsurePointerHooks();
        if (nodeData.name === "color_ImageCurve") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                
                const MIN_NODE_WIDTH = 240;
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

                const hideWidgetAndSlot = (widgetName) => {
                    const w = this.widgets?.find(w => w.name === widgetName);
                    if (w) {
                        w.type = "hidden"; 
                        w.hidden = true;
                        w.computeSize = () => [0, 0];
                        w.draw = () => {};
                        // 移除DOM 元素及父容器
                        if (w.inputEl) {
                            w.inputEl.style.display = "none";
                            if (w.inputEl.parentElement) {
                                w.inputEl.parentElement.style.display = "none";
                            }
                        }
                    }
                    if (this.inputs) {
                        const idx = this.inputs.findIndex(i => i.name === widgetName);
                        if (idx !== -1) {
                            this.removeInput(idx);
                        }
                    }
                };

                // 立即隐藏 (curve_data)
                hideWidgetAndSlot("curve_data");
                hideWidgetAndSlot("preview_width");
                hideWidgetAndSlot("preview_height");
                
                const curveWidget = this.widgets?.find(w => w.name === "curve_data");
                const presetWidget = this.widgets?.find(w => w.name === "curve_preset");

                let curveData = {
                    RGB: [[0.0, 0.0], [1.0, 1.0]],
                    R: [[0.0, 0.0], [1.0, 1.0]],
                    G: [[0.0, 0.0], [1.0, 1.0]],
                    B: [[0.0, 0.0], [1.0, 1.0]]
                };
                let activeChannel = "RGB";
                
                if (curveWidget) {
                    const originalDescriptor = Object.getOwnPropertyDescriptor(curveWidget, 'value') || 
                                               Object.getOwnPropertyDescriptor(Object.getPrototypeOf(curveWidget), 'value');
                    
                    Object.defineProperty(curveWidget, 'value', {
                        get: function() {
                            if (originalDescriptor && originalDescriptor.get) {
                                return originalDescriptor.get.call(this);
                            }
                            return this._curveValue !== undefined ? this._curveValue : "";
                        },
                        set: function(v) {
                            if (originalDescriptor && originalDescriptor.set) {
                                originalDescriptor.set.call(this, v);
                            } else {
                                this._curveValue = v;
                            }
                            if (v && typeof v === 'string' && v.startsWith("{")) {
                                try {
                                    const newData = JSON.parse(v);
                                    if (newData && newData.RGB) {
                                        curveData = newData;
                                        if (typeof draw === 'function') {
                                            requestAnimationFrame(draw);
                                        }
                                    }
                                } catch (e) {}
                            }
                        },
                        configurable: true
                    });
                    
                    if (curveWidget.value && typeof curveWidget.value === 'string' && curveWidget.value.startsWith("{")) {
                        try {
                            curveData = JSON.parse(curveWidget.value);
                        } catch (e) {}
                    } else {
                        curveWidget.value = JSON.stringify(curveData);
                    }
                }

                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.width = "100%";
                container.style.height = "100%";
                container.style.marginTop = "0px";
                container.style.borderRadius = "6px";
                container.style.overflow = "hidden";
                container.style.backgroundColor = "transparent";
				
                const header = document.createElement("div");
                header.style.display = "flex";
                header.style.height = "20px";
                header.style.flexShrink = "0"; 
                header.style.backgroundColor = "#222";
                
                const channels = [
                    { id: "RGB", color: "#FFF" },
                    { id: "R", color: "#FF4444" },
                    { id: "G", color: "#44FF44" },
                    { id: "B", color: "#4488FF" }
                ];
                
                const buttons = {};
                channels.forEach(ch => {
                    const btn = document.createElement("button");
                    btn.innerText = ch.id;
                    btn.style.flex = "1";
                    btn.style.border = "none";
                    btn.style.cursor = "pointer";
                    btn.style.backgroundColor = ch.id === "RGB" ? "#555" : "transparent";
                    btn.style.color = ch.color;
                    btn.style.fontWeight = "";
                    btn.style.fontSize = "10px";
                    
                    btn.onclick = () => {
                        activeChannel = ch.id;
                        Object.values(buttons).forEach(b => b.style.backgroundColor = "transparent");
                        btn.style.backgroundColor = "#555";
                        if (typeof draw === 'function') draw();
                    };
                    buttons[ch.id] = btn;
                    header.appendChild(btn);
                });

                const rcBtn = document.createElement("button");
                rcBtn.innerText = "Rc";
                rcBtn.style.flex = "0.8";
                rcBtn.style.border = "none";
                rcBtn.style.cursor = "pointer";
                rcBtn.style.backgroundColor = "transparent";
                rcBtn.style.color = "#CCC";
                rcBtn.style.fontWeight = "";
                rcBtn.style.fontSize = "10px";
                
                rcBtn.onclick = () => {
                    curveData[activeChannel] = [[0.0, 0.0], [1.0, 1.0]];
                    if (typeof updateBackend === 'function') updateBackend(true);
                    if (typeof draw === 'function') draw();
                    if (typeof updateLivePreview === 'function') updateLivePreview(false);
                };
                header.appendChild(rcBtn);

                const rallBtn = document.createElement("button");
                rallBtn.innerText = "Rall";
                rallBtn.style.flex = "0.8";
                rallBtn.style.border = "none";
                rallBtn.style.cursor = "pointer";
                rallBtn.style.backgroundColor = "transparent";
                rallBtn.style.color = "#CCC";
                rallBtn.style.fontWeight = "bold";
                rallBtn.style.fontSize = "10px";
                
                rallBtn.onclick = () => {
                    curveData = {
                        RGB: [[0.0, 0.0], [1.0, 1.0]],
                        R: [[0.0, 0.0], [1.0, 1.0]],
                        G: [[0.0, 0.0], [1.0, 1.0]],
                        B: [[0.0, 0.0], [1.0, 1.0]]
                    };
                    if (typeof updateBackend === 'function') updateBackend(true);
                    if (typeof draw === 'function') draw();
                    if (typeof updateLivePreview === 'function') updateLivePreview(false);
                };
                header.appendChild(rallBtn);

                container.appendChild(header);

                const viewArea = document.createElement("div");
                viewArea.style.flex = "1";
                viewArea.style.position = "relative";
                viewArea.style.width = "100%";
                viewArea.style.height = "100%";
                viewArea.style.backgroundColor = "#1a1a1a";
                viewArea.style.overflow = "hidden";

                const bgImageLayer = document.createElement("div");
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
                canvas.style.cursor = "crosshair";
                viewArea.appendChild(canvas);

                container.appendChild(viewArea);
                const ctx = canvas.getContext("2d");
				
				// 同步
                const resizeObserver = new ResizeObserver((entries) => {
                    for (let entry of entries) {
                        const { width, height } = entry.contentRect;
                        if (width > 0 && height > 0) {
                            if (canvas.width !== width || canvas.height !== height) {
                                canvas.width = width;
                                canvas.height = height;
                                if (typeof draw === 'function') draw();
                            }
                        }
                    }
                });
                resizeObserver.observe(viewArea);
                // ===================================================================

                const widget = this.addDOMWidget("CurveUI", "div", container, { serialize: false, hideOnZoom: false });

                const nodeInstance = this;

                const CURVEUI_DEFAULT_HEIGHT = 310;
                widget.computeSize = function (width) {
                    return [width, CURVEUI_DEFAULT_HEIGHT];
                };
                const CURVEUI_BASE_HEIGHT = typeof nodeInstance.computeSize === "function"
                    ? nodeInstance.computeSize()[1]
                    : nodeInstance.size[1];
                const getTargetNodeHeight = () => {
                    const ts = nodeInstance._apt_target_size;
                    const th = Array.isArray(ts) ? ts[1] : undefined;
                    if (typeof th === "number" && Number.isFinite(th)) {
                        return Math.max(MIN_NODE_HEIGHT, th);
                    }
                    const c = app?.canvas;
                    const rn = c?.resizing_node || c?.node_resizing || c?.resized_node;
                    const m = c?.graph_mouse || c?.mouse;
                    if (rn === nodeInstance && Array.isArray(m) && m.length > 1 && Array.isArray(nodeInstance.pos)) {
                        return Math.max(MIN_NODE_HEIGHT, m[1] - nodeInstance.pos[1]);
                    }
                    return nodeInstance.size[1];
                };
                widget.computeSize = function (width) {
                    const targetH = getTargetNodeHeight();
                    const extra = Math.max(0, targetH - CURVEUI_BASE_HEIGHT);
                    return [width, CURVEUI_DEFAULT_HEIGHT + extra];
                };

                let lastPreviewTime = 0;
                let pendingUpdate = false;
                let isPreviewPending = false;

                const updateLivePreview = (isDragging = false) => {
                    const now = Date.now();
                    
                    const interval = isDragging ? 16 : 33;
                    if (now - lastPreviewTime < interval) {
                        if (!pendingUpdate) {
                            pendingUpdate = true;
                            setTimeout(() => {
                                pendingUpdate = false;
                                updateLivePreview(isDragging);
                            }, interval - (now - lastPreviewTime));
                        }
                        return;
                    }
                    
                    if (isPreviewPending) {
                        pendingUpdate = true;
                        return;
                    }
                    
                    lastPreviewTime = now;
                    pendingUpdate = false;
                    isPreviewPending = true;
                    
                    const satWidget = nodeInstance.widgets?.find(w => w.name === "saturation");
                    const presetWidget = nodeInstance.widgets?.find(w => w.name === "curve_preset");

                    const body = {
                        node_id: nodeInstance.id.toString(),
                        curve_data: JSON.stringify(curveData),
                        curve_preset: presetWidget ? String(presetWidget.value) : "Custom",
                        saturation: satWidget ? parseFloat(satWidget.value) : 1.0
                    };
                    
                    (async () => {
                        try {
                            const response = await api.fetchApi("/color_image_curve/live_preview", {
                                method: "POST",
                                body: JSON.stringify(body)
                            });
                            
                            isPreviewPending = false;
                            
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                            
                            if (response.ok) {
                                const img = await response.json();
                                if (img.filename) {
                                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                                    
                                    const imgLoader = new Image();
                                    imgLoader.onload = () => {
                                        if (bgImageLayer) bgImageLayer.style.backgroundImage = `url(${url})`;
                                    };
                                    imgLoader.src = url;
                                }
                            }
                        } catch(e) {
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                        }
                    })();
                };

                hideWidgetAndSlot("saturation");
                const satWidget = this.widgets?.find(w => w.name === "saturation");
                
                // ========== Sat 滑条 ==========
                const satControlArea = document.createElement("div");
                satControlArea.style.display = "flex";
                satControlArea.style.alignItems = "center";
                satControlArea.style.height = "24px";
                satControlArea.style.flexShrink = "0";
                satControlArea.style.backgroundColor = "#1a1a1a";
                satControlArea.style.padding = "0 10px";
                satControlArea.style.borderTop = "1px solid #333";
                satControlArea.style.gap = "8px";
                satControlArea.style.borderRadius = "0 0 6px 6px";
                
                const satLabel = document.createElement("span");
                satLabel.innerText = "Sat";
                satLabel.style.color = "#CCC";
                satLabel.style.fontSize = "10px";
                satLabel.style.fontWeight = "bold";
                satLabel.style.width = "28px";
                satLabel.style.flexShrink = "0";
                satControlArea.appendChild(satLabel);
                
                // 创建滑条
                const satSlider = document.createElement("input");
                satSlider.type = "range";
                satSlider.min = "0";
                satSlider.max = "2";
                satSlider.step = "0.01";
                satSlider.style.flex = "0.85";
                satSlider.style.height = "2px";
                satSlider.style.accentColor = "#FFFFFF";
                
                satSlider.id = "sat-slider-" + this.id;
                
                satControlArea.appendChild(satSlider);
                
                // 数值显示
                const satValueDisplay = document.createElement("span");
                satValueDisplay.style.color = "#ffffff";
                satValueDisplay.style.fontSize = "10px";
                satValueDisplay.style.fontFamily = "";
                satValueDisplay.style.width = "20px";
                satValueDisplay.style.textAlign = "right";
                satValueDisplay.style.flexShrink = "0";
                satControlArea.appendChild(satValueDisplay);
                
                const styleId = "sat-style-" + this.id;
                if (!document.getElementById(styleId)) {
                    const style = document.createElement("style");
                    style.id = styleId;
                    style.textContent = `
                        #${satSlider.id} {
                            -webkit-appearance: none !important;
                            appearance: none !important;
                        }
                        #${satSlider.id}::-webkit-slider-thumb {
                            -webkit-appearance: none !important;
                            appearance: none !important;
                            width: 8px !important;
                            height: 8px !important;
                            background: #FFFFFF !important;
                            border-radius: 50% !important;
                            margin-top: -3px !important;
                            border: none !important;
                            box-shadow: none !important;
                        }
                        #${satSlider.id}::-webkit-slider-runnable-track {
                            height: 2px !important;
                            background: #333 !important;
                            border-radius: 2px !important;
                        }
                    `;
                    document.head.appendChild(style);
                }
                
                
                const initialValue = satWidget ? parseFloat(satWidget.value) : 1.0;
                satSlider.value = initialValue;
                satValueDisplay.innerText = initialValue.toFixed(2);
                
                satSlider.addEventListener("input", (e) => {
                    const val = parseFloat(e.target.value);
                    satValueDisplay.innerText = val.toFixed(2);
                    if (satWidget) satWidget.value = val;
                    updateLivePreview(true);
                });
                
                satSlider.addEventListener("change", (e) => {
                    const val = parseFloat(e.target.value);
                    if (satWidget) satWidget.value = val;
                    updateLivePreview(false);
                });
                

                let satValueInternal = initialValue;
                let satUpdateRaf = null;
                
                Object.defineProperty(satWidget, "value", {
                    get: () => satValueInternal,
                    set: (v) => {
                        const newVal = parseFloat(v);
                        if (satValueInternal === newVal) return; // 避免重复设置相同值
                        satValueInternal = newVal;
                        
                        // 使用 RAF 批量更新 DOM，避免强制同步布局
                        if (satUpdateRaf) return;
                        satUpdateRaf = requestAnimationFrame(() => {
                            if (satSlider) satSlider.value = satValueInternal;
                            if (satValueDisplay) satValueDisplay.innerText = satValueInternal.toFixed(2);
                            satUpdateRaf = null;
                        });
                    },
                    configurable: true
                });
                
                // 双击数值重置
                satValueDisplay.style.cursor = "pointer";
                satValueDisplay.title = "双击重置为 1.0";
                satValueDisplay.addEventListener("dblclick", () => {
                    satSlider.value = "1.0";
                    satValueDisplay.innerText = "1.00";
                    if (satWidget) satWidget.value = 1.0;
                    updateLivePreview(false);
                });
                
                container.appendChild(satControlArea);
             
                // ======= 模式切换 (Load Output) ======
                const modeControlArea = document.createElement("div");
                modeControlArea.style.display = "flex";
                modeControlArea.style.alignItems = "center";
                modeControlArea.style.alignSelf = "stretch";
                modeControlArea.style.width = "100%";
                //modeControlArea.style.height = "32px";
                modeControlArea.style.flexShrink = "0";
                modeControlArea.style.backgroundColor = "transparent";
                modeControlArea.style.padding = "0 0px";
                modeControlArea.style.borderTop = "none";
                modeControlArea.style.gap = "8px";
                modeControlArea.style.boxSizing = "border-box";
				//modeControlArea.style.marginBottom = "0px";
				
				// Nodes 2.0？
                const isNodes2_0 = !!document.querySelector("comfy-app") || 
                                   !!document.querySelector(".comfy-vue") || 
                                   (window.comfyAPI && window.comfyAPI.vue);

                modeControlArea.style.marginBottom = isNodes2_0 ? "10px" : "0px";
                // ==========================================================
				
                
                // Load 按钮
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
                
                loadBtn.onclick = async () => {
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
                                if (Array.isArray(val) && val.length === 2) {
                                    traceDependencies(String(val[0]));
                                }
                            }
                        };
                        
                        traceDependencies(selectedNodeId);
                        
                        if (Object.keys(isolatedPrompt).length === 0) {
                            console.warn("No dependencies found for node", selectedNodeId);
                            return;
                        }
                        
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
                        
                        console.log("Successfully queued selected node execution");
                        
                    } catch (err) {
                        console.error("Failed to execute isolated node:", err);
                    }
                };
                
                modeControlArea.appendChild(loadBtn);
                container.appendChild(modeControlArea);
				

                let isDragging = false;
                let dragIndex = -1;
                let lastClickTime = 0;
                const PADDING = 14; 

                const setPresetValue = (name) => {
                    if (!presetWidget) return;
                    presetWidget.value = name;
                    if (presetWidget.inputEl) {
                        presetWidget.inputEl.value = name;
                    }
                };

                const updateBackend = (markCustom = false) => {
                    if (markCustom && presetWidget && String(presetWidget.value || "Custom") !== "Custom") {
                        setPresetValue("Custom");
                    }
                    if (curveWidget) {
                        curveWidget.value = JSON.stringify(curveData);
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true);
                };

                const CURVE_PRESETS = {
                    "Contrast (S)": { RGB: [[0.0, 0.0], [0.25, 0.18], [0.5, 0.5], [0.75, 0.82], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Soft Contrast": { RGB: [[0.0, 0.0], [0.3, 0.27], [0.7, 0.73], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Matte": { RGB: [[0.0, 0.08], [0.25, 0.28], [0.75, 0.88], [1.0, 0.95]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Fade Highlights": { RGB: [[0.0, 0.0], [0.6, 0.62], [1.0, 0.92]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Brighten": { RGB: [[0.0, 0.0], [0.25, 0.35], [0.5, 0.65], [0.75, 0.85], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Darken": { RGB: [[0.0, 0.0], [0.25, 0.15], [0.5, 0.38], [0.75, 0.68], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Warm": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.0], [0.5, 0.55], [1.0, 1.0]], G: [[0.0, 0.0], [0.5, 0.52], [1.0, 1.0]], B: [[0.0, 0.0], [0.5, 0.45], [1.0, 0.95]] },
                    "Cool": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.0], [0.5, 0.45], [1.0, 0.95]], G: [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], B: [[0.0, 0.0], [0.5, 0.55], [1.0, 1.0]] },
                    "Teal & Orange": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.0], [0.4, 0.38], [0.7, 0.78], [1.0, 1.0]], G: [[0.0, 0.02], [0.5, 0.5], [1.0, 1.0]], B: [[0.0, 0.08], [0.4, 0.45], [0.7, 0.62], [1.0, 0.95]] },
                    "Cross Process": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.06], [0.35, 0.33], [0.75, 0.86], [1.0, 1.0]], G: [[0.0, 0.0], [0.5, 0.48], [1.0, 0.95]], B: [[0.0, 0.0], [0.25, 0.22], [0.75, 0.88], [1.0, 1.0]] },
                    "Linear": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Soft S": { RGB: [[0.0, 0.0], [0.157, 0.141], [0.376, 0.361], [0.627, 0.784], [0.816, 0.894], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Strong S": { RGB: [[0.0, 0.0], [0.125, 0.078], [0.314, 0.275], [0.502, 0.502], [0.69, 0.784], [0.878, 0.949], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Bright Midtones": { RGB: [[0.0, 0.0], [0.251, 0.431], [0.502, 0.667], [0.753, 0.863], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Dark Mood": { RGB: [[0.0, 0.0], [0.188, 0.11], [0.376, 0.314], [0.502, 0.471], [0.753, 0.706], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Highlight Lift": { RGB: [[0.0, 0.0], [0.251, 0.353], [0.502, 0.588], [0.753, 0.824], [0.902, 0.961], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Shadow Lift": { RGB: [[0.0, 0.0], [0.047, 0.094], [0.251, 0.306], [0.502, 0.549], [0.753, 0.784], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Film Matte": { RGB: [[0.0, 0.0], [0.141, 0.11], [0.376, 0.376], [0.627, 0.745], [0.816, 0.878], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Fade Blacks": { RGB: [[0.0, 0.0], [0.031, 0.071], [0.188, 0.188], [0.502, 0.549], [0.753, 0.784], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Punchy": { RGB: [[0.0, 0.0], [0.188, 0.125], [0.376, 0.329], [0.502, 0.502], [0.627, 0.706], [0.816, 0.922], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "High Key": { RGB: [[0.0, 0.0], [0.251, 0.392], [0.502, 0.706], [0.753, 0.902], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Low Key": { RGB: [[0.0, 0.0], [0.157, 0.078], [0.376, 0.251], [0.502, 0.431], [0.753, 0.706], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Subtle S": { RGB: [[0.0, 0.0], [0.188, 0.173], [0.439, 0.416], [0.627, 0.722], [0.816, 0.863], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Retro Fade": { RGB: [[0.0, 0.0], [0.047, 0.071], [0.251, 0.282], [0.502, 0.533], [0.753, 0.784], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Vintage Warm": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.0], [0.188, 0.176], [0.376, 0.38], [0.502, 0.549], [0.753, 0.804], [1.0, 1.0]], G: [[0.0, 0.0], [0.188, 0.157], [0.376, 0.353], [0.502, 0.51], [0.753, 0.784], [1.0, 1.0]], B: [[0.0, 0.0], [0.188, 0.141], [0.376, 0.314], [0.502, 0.471], [0.753, 0.745], [1.0, 0.965]] },
                    "Vintage Cool": { RGB: [[0.0, 0.0], [1.0, 1.0]], R: [[0.0, 0.0], [0.188, 0.141], [0.376, 0.294], [0.502, 0.451], [0.753, 0.706], [1.0, 0.941]], G: [[0.0, 0.0], [0.188, 0.157], [0.376, 0.314], [0.502, 0.471], [0.753, 0.745], [1.0, 1.0]], B: [[0.0, 0.0], [0.188, 0.176], [0.376, 0.38], [0.502, 0.549], [0.753, 0.824], [1.0, 1.0]] },
                    "Cinematic S": { RGB: [[0.0, 0.0], [0.251, 0.188], [0.502, 0.502], [0.753, 0.816], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "HDR Boost": { RGB: [[0.0, 0.0], [0.125, 0.11], [0.251, 0.251], [0.502, 0.627], [0.753, 0.863], [1.0, 1.0]], R: [[0.0, 0.0], [1.0, 1.0]], G: [[0.0, 0.0], [1.0, 1.0]], B: [[0.0, 0.0], [1.0, 1.0]] },
                    "Film Negative": { RGB: [[0.0, 1.0], [0.125, 0.784], [0.376, 0.627], [0.627, 0.314], [0.878, 0.094], [1.0, 0.0]], R: [[0.0, 1.0], [1.0, 0.0]], G: [[0.0, 1.0], [1.0, 0.0]], B: [[0.0, 1.0], [1.0, 0.0]] }
                };

                const applyPreset = (presetName) => {
                    const preset = CURVE_PRESETS[presetName];
                    if (!preset) return;
                    curveData = JSON.parse(JSON.stringify(preset));
                    updateBackend();
                    draw();
                    updateLivePreview(false);
                };

                if (presetWidget) {
                    const bind = () => {
                        const name = String(presetWidget.value || "Custom");
                        if (name !== "Custom") applyPreset(name);
                    };

                    const originalCallback = presetWidget.callback;
                    presetWidget.callback = function(value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        bind();
                    };

                    if (presetWidget.inputEl) {
                        presetWidget.inputEl.addEventListener("change", bind);
                        presetWidget.inputEl.addEventListener("input", bind);
                    }

                    bind();
                }

                const getPos = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    
                    let mouseX = (e.clientX - rect.left) * scaleX;
                    let mouseY = (e.clientY - rect.top) * scaleY;
                    
                    const innerW = canvas.width - PADDING * 2;
                    const innerH = canvas.height - PADDING * 2;
                    
                    let x = (mouseX - PADDING) / innerW;
                    let y = 1.0 - (mouseY - PADDING) / innerH;
                    
                    return [Math.max(0, Math.min(1, x)), Math.max(0, Math.min(1, y))];
                };

                canvas.addEventListener("mousedown", (e) => {
                    const [x, y] = getPos(e);
                    const pts = curveData[activeChannel];
                    
                    let closestIdx = -1;
                    let minDist = 0.05; 
                    for (let i = 0; i < pts.length; i++) {
                        let dist = Math.hypot(pts[i][0] - x, pts[i][1] - y);
                        if (dist < minDist) {
                            minDist = dist;
                            closestIdx = i;
                        }
                    }

                    if (e.button === 0) { 
                        const now = Date.now();
                        if (now - lastClickTime < 300) {
                            if (closestIdx === -1) {
                                pts.push([x, y]);
                                pts.sort((a, b) => a[0] - b[0]);
                                updateBackend(true);
                                draw();
                                updateLivePreview(false);
                            }
                            lastClickTime = 0; 
                        } else {
                            if (closestIdx !== -1) {
                                isDragging = true;
                                dragIndex = closestIdx;
                            }
                            lastClickTime = now;
                        }
                    } else if (e.button === 2) { 
                        e.preventDefault();
                        if (closestIdx !== -1 && pts.length > 2) {
                            pts.splice(closestIdx, 1);
                            updateBackend(true);
                            draw();
                            updateLivePreview(false);
                        }
                    }
                });

                canvas.addEventListener("contextmenu", e => e.preventDefault());

                window.addEventListener("mousemove", (e) => {
                    if (!isDragging || dragIndex === -1) return;
                    const [x, y] = getPos(e);
                    const pts = curveData[activeChannel];
                    
                    let minX = dragIndex > 0 ? pts[dragIndex - 1][0] + 0.02 : 0;
                    let maxX = dragIndex < pts.length - 1 ? pts[dragIndex + 1][0] - 0.02 : 1;
                    
                    pts[dragIndex] = [Math.max(minX, Math.min(maxX, x)), y];
                    
                    draw();
                    updateBackend(true);
                    updateLivePreview(true);
                });

                window.addEventListener("mouseup", () => {
                    if (isDragging) {
                        isDragging = false;
                        updateBackend(true);
                    }
                });

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
                    if (w === 0 || h === 0) return;

                    ctx.clearRect(0, 0, w, h);
                    ctx.shadowColor = "rgba(0, 0, 0, 0.5)";
                    ctx.shadowBlur = 1;

                    const innerW = w - PADDING * 2;
                    const innerH = h - PADDING * 2;
                    const toPx = (val) => PADDING + val * innerW;
                    const toPy = (val) => PADDING + (1 - val) * innerH;

                    ctx.strokeStyle = "rgba(255, 255, 255, 0.25)";
                    ctx.lineWidth = 0.5; // 网格
                    ctx.beginPath();
                    for(let i=0; i<=4; i++) {
                        let px = Math.round(toPx(i / 4)) + 0.5;
                        let py = Math.round(toPy(i / 4)) + 0.5;
                        let startX = Math.round(toPx(0)) + 0.5;
                        let endX = Math.round(toPx(1)) + 0.5;
                        let startY = Math.round(toPy(0)) + 0.5;
                        let endY = Math.round(toPy(1)) + 0.5;
                        
                        ctx.moveTo(px, toPy(0)); ctx.lineTo(px, toPy(1));
                        ctx.moveTo(toPx(0), py); ctx.lineTo(toPx(1), py);
                    }
                    ctx.stroke();

                    channels.forEach(ch => {
                        if (ch.id !== activeChannel) drawCurve(curveData[ch.id], ch.color, 0.3);
                    });
                    
                    const activeColor = channels.find(c => c.id === activeChannel).color;
                    drawCurve(curveData[activeChannel], activeColor, 1.0);

                    const pts = curveData[activeChannel];
                    ctx.fillStyle = activeColor;
                    pts.forEach(p => {
                        ctx.beginPath();
                        ctx.arc(toPx(p[0]), toPy(p[1]), 4, 0, Math.PI * 2);
                        ctx.fill();
                    });
                };

                const drawCurve = (points, color, alpha) => {
                    if (points.length < 2) return;
                    ctx.strokeStyle = color;
                    ctx.globalAlpha = alpha;
                    ctx.lineWidth = 1.25;
                    ctx.beginPath();
                    
                    const w = canvas.width;
                    const h = canvas.height;
                    const innerW = w - PADDING * 2;
                    const innerH = h - PADDING * 2;
                    const toPx = (val) => PADDING + val * innerW;
                    const toPy = (val) => PADDING + (1 - val) * innerH;

                    if (points[0][0] > 0) {
                        ctx.moveTo(toPx(0), toPy(points[0][1]));
                        ctx.lineTo(toPx(points[0][0]), toPy(points[0][1]));
                    } else {
                        ctx.moveTo(toPx(points[0][0]), toPy(points[0][1]));
                    }

                    for (let i = 0; i < points.length - 1; i++) {
                        let p0 = points[i === 0 ? 0 : i - 1];
                        let p1 = points[i];
                        let p2 = points[i + 1];
                        let p3 = points[i + 2 >= points.length ? i + 1 : i + 2];

                        for (let t = 0; t <= 1; t += 0.05) {
                            let t2 = t * t;
                            let t3 = t2 * t;
                            let cx = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3);
                            let cy = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3);
                            
                            cx = Math.max(p1[0], Math.min(p2[0], cx)); 
                            cy = Math.max(0, Math.min(1, cy)); 
                            
                            ctx.lineTo(toPx(cx), toPy(cy));
                        }
                        ctx.lineTo(toPx(p2[0]), toPy(p2[1]));
                    }
                    
                    const lastP = points[points.length - 1];
                    if (lastP[0] < 1) {
                        ctx.lineTo(toPx(1), toPy(lastP[1]));
                    }
                    
                    ctx.stroke();
                    ctx.globalAlpha = 1.0;
                };

                setTimeout(draw, 100);
                
                setTimeout(() => {
                    draw();
                    if (this.onResize) this.onResize(this.size);
                }, 150);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                onExecuted?.apply(this, arguments);
                
                if (message?.bg_image?.length > 0) {
                    const img = message.bg_image[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                    
                    const curveWidget = this.widgets.find(w => w.name === "CurveUI");
                    if (curveWidget && curveWidget.element) {
                        const viewArea = curveWidget.element.childNodes[1]; 
                        if (viewArea && viewArea.firstChild) {
                            viewArea.firstChild.style.backgroundImage = `url(${url})`;
                        }
                    }
                }

                setTimeout(() => {
                    const satWidget = this.widgets.find(w => w.name === "saturation");
                    
                    if (satWidget) {
                        const currentVal = parseFloat(satWidget.value);
                        satWidget.value = currentVal;
                    }
                }, 50);
            };
        } else if (nodeData.name === "color_RadiaGradient_visual") {
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

                hideWidgetAndSlot("center_x");
                hideWidgetAndSlot("center_y");

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
                canvas.style.cursor = "crosshair";
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

                loadBtn.onclick = async () => {
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

                previewBar.style.marginBottom = isNodes2_0 ? "10px" : "0px";
                previewBar.appendChild(loadBtn);
                container.appendChild(previewBar);

                const widget = this.addDOMWidget("RadialGradientUI", "div", container, { serialize: false, hideOnZoom: false });

                const nodeInstance = this;
                const ctx = canvas.getContext("2d");

                const resizeObserver = new ResizeObserver((entries) => {
                    for (let entry of entries) {
                        const { width, height } = entry.contentRect;
                        if (width > 0 && height > 0) {
                            if (canvas.width !== width || canvas.height !== height) {
                                canvas.width = width;
                                canvas.height = height;
                                draw();
                            }
                        }
                    }
                });
                resizeObserver.observe(viewArea);

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

                const getWidget = (name) => nodeInstance.widgets?.find(w => w.name === name);
                const cxW = getWidget("center_x");
                const cyW = getWidget("center_y");
                const radiusW = getWidget("circle_radius");
                const centerBrightW = getWidget("center_bright");
                const edgeBrightW = getWidget("edge_bright");
                const overlayColorW = getWidget("overlay_color");
                const centerAlphaW = getWidget("center_alpha");
                const edgeAlphaW = getWidget("edge_alpha");
                const falloffModeW = getWidget("falloff_mode");
                const softEdgeW = getWidget("soft_edge");

                const clamp01 = (v) => Math.max(0, Math.min(1, v));

                let point = {
                    x: cxW ? clamp01(parseFloat(cxW.value)) : 0.5,
                    y: cyW ? clamp01(parseFloat(cyW.value)) : 0.5
                };

                const hexToRgb = (hex) => {
                    const s = String(hex || "").replace("#", "");
                    if (s.length !== 6) return [255, 255, 255];
                    const r = parseInt(s.slice(0, 2), 16);
                    const g = parseInt(s.slice(2, 4), 16);
                    const b = parseInt(s.slice(4, 6), 16);
                    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [255, 255, 255];
                    return [r, g, b];
                };

                let lastPreviewTime = 0;
                let pendingUpdate = false;
                let isPreviewPending = false;

                const updateLivePreview = (isDragging = false) => {
                    const now = Date.now();
                    const interval = isDragging ? 16 : 33;
                    if (now - lastPreviewTime < interval) {
                        if (!pendingUpdate) {
                            pendingUpdate = true;
                            setTimeout(() => {
                                pendingUpdate = false;
                                updateLivePreview(isDragging);
                            }, interval - (now - lastPreviewTime));
                        }
                        return;
                    }
                    if (isPreviewPending) {
                        pendingUpdate = true;
                        return;
                    }
                    lastPreviewTime = now;
                    pendingUpdate = false;
                    isPreviewPending = true;

                    const body = {
                        node_id: nodeInstance.id.toString(),
                        center_x: point.x,
                        center_y: point.y,
                        circle_radius: radiusW ? parseFloat(radiusW.value) : 0.2,
                        center_bright: centerBrightW ? parseFloat(centerBrightW.value) : 1.5,
                        edge_bright: edgeBrightW ? parseFloat(edgeBrightW.value) : 1.0,
                        overlay_color: overlayColorW ? String(overlayColorW.value) : "#FFFFFF",
                        center_alpha: centerAlphaW ? parseFloat(centerAlphaW.value) : 0.0,
                        edge_alpha: edgeAlphaW ? parseFloat(edgeAlphaW.value) : 0.0,
                        falloff_mode: falloffModeW ? String(falloffModeW.value) : "linear",
                        soft_edge: softEdgeW ? !!softEdgeW.value : true
                    };

                    (async () => {
                        try {
                            const response = await api.fetchApi("/color_radia_bright_gradient/live_preview", {
                                method: "POST",
                                body: JSON.stringify(body)
                            });
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                            if (response.ok) {
                                const img = await response.json();
                                if (img.filename) {
                                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                                    const imgLoader = new Image();
                                    imgLoader.onload = () => {
                                        bgImageLayer.style.backgroundImage = `url(${url})`;
                                    };
                                    imgLoader.src = url;
                                }
                            }
                        } catch (e) {
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                        }
                    })();
                };

                const updateBackendCoords = () => {
                    if (cxW) cxW.value = point.x;
                    if (cyW) cyW.value = point.y;
                    if (app.graph) app.graph.setDirtyCanvas(true);
                };

                const draw = () => {
                    if (!ctx) return;
                    const w = canvas.width;
                    const h = canvas.height;
                    ctx.clearRect(0, 0, w, h);

                    const centerBright = centerBrightW ? parseFloat(centerBrightW.value) : 1.5;
                    const edgeBright = edgeBrightW ? parseFloat(edgeBrightW.value) : 1.0;
                    const radius = radiusW ? parseFloat(radiusW.value) : 0.2;
                    const centerAlpha = centerAlphaW ? parseFloat(centerAlphaW.value) : 0.0;
                    const edgeAlpha = edgeAlphaW ? parseFloat(edgeAlphaW.value) : 0.0;
                    const [r, g, b] = hexToRgb(overlayColorW ? overlayColorW.value : "#FFFFFF");

                    const cx = point.x * w;
                    const cy = point.y * h;
                    const rr = radius * Math.max(w, h);

                    const hasBg = !!(bgImageLayer && bgImageLayer.style.backgroundImage && bgImageLayer.style.backgroundImage !== "none");
                    ctx.save();
                    if (hasBg) ctx.globalAlpha = 0.25;
                    const g0 = clamp01(centerBright / 2.0);
                    const g1 = clamp01(edgeBright / 2.0);
                    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(1, rr));
                    grad.addColorStop(0, `rgb(${Math.round(255 * g0)},${Math.round(255 * g0)},${Math.round(255 * g0)})`);
                    grad.addColorStop(1, `rgb(${Math.round(255 * g1)},${Math.round(255 * g1)},${Math.round(255 * g1)})`);
                    ctx.fillStyle = grad;
                    ctx.fillRect(0, 0, w, h);

                    const colGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(1, rr));
                    colGrad.addColorStop(0, `rgba(${r},${g},${b},${clamp01(centerAlpha)})`);
                    colGrad.addColorStop(1, `rgba(${r},${g},${b},${clamp01(edgeAlpha)})`);
                    ctx.fillStyle = colGrad;
                    ctx.fillRect(0, 0, w, h);
                    ctx.restore();

                    ctx.save();
                    ctx.strokeStyle = "rgba(255,255,255,0.65)";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.arc(cx, cy, rr, 0, Math.PI * 2);
                    ctx.stroke();
                    ctx.restore();

                    ctx.save();
                    ctx.fillStyle = "#ffffff";
                    ctx.strokeStyle = "#000000";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.arc(cx, cy, 6, 0, Math.PI * 2);
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                };

                let isDragging = false;
                const hitRadius = 10;

                const getMouse = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const sx = canvas.width / rect.width;
                    const sy = canvas.height / rect.height;
                    const mx = (e.clientX - rect.left) * sx;
                    const my = (e.clientY - rect.top) * sy;
                    return { x: mx, y: my };
                };

                canvas.addEventListener("mousedown", (e) => {
                    const m = getMouse(e);
                    const px = point.x * canvas.width;
                    const py = point.y * canvas.height;
                    const d = Math.hypot(m.x - px, m.y - py);
                    if (d <= hitRadius) isDragging = true;
                });

                window.addEventListener("mousemove", (e) => {
                    if (!isDragging) return;
                    const m = getMouse(e);
                    point.x = clamp01(m.x / canvas.width);
                    point.y = clamp01(m.y / canvas.height);
                    updateBackendCoords();
                    draw();
                    updateLivePreview(true);
                });

                window.addEventListener("mouseup", () => {
                    if (!isDragging) return;
                    isDragging = false;
                    updateLivePreview(false);
                });

                const bindWidget = (w) => {
                    if (!w) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        draw();
                        updateLivePreview(false);
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => { draw(); updateLivePreview(false); });
                        w.inputEl.addEventListener("input", () => { draw(); updateLivePreview(true); });
                    }
                };

                [radiusW, centerBrightW, edgeBrightW, overlayColorW, centerAlphaW, edgeAlphaW, falloffModeW, softEdgeW].forEach(bindWidget);

                draw();
                setTimeout(() => {
                    draw();
                    if (this.onResize) this.onResize(this.size);
                }, 150);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.bg_image?.length > 0) {
                    const img = message.bg_image[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                    const uiWidget = this.widgets.find(w => w.name === "RadialGradientUI");
                    const bg = uiWidget?.element?.querySelector?.(".apt-preview-bg");
                    if (bg) bg.style.backgroundImage = `url(${url})`;
                }
            };
        } else if (nodeData.name === "color_lineGradient_visual") {
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

                hideWidgetAndSlot("start_x");
                hideWidgetAndSlot("start_y");
                hideWidgetAndSlot("end_x");
                hideWidgetAndSlot("end_y");

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
                canvas.style.cursor = "crosshair";
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

                loadBtn.onclick = async () => {
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

                previewBar.style.marginBottom = isNodes2_0 ? "10px" : "0px";
                previewBar.appendChild(loadBtn);
                container.appendChild(previewBar);

                const widget = this.addDOMWidget("BrightGradientUI", "div", container, { serialize: false, hideOnZoom: false });

                const nodeInstance = this;
                const ctx = canvas.getContext("2d");

                const resizeObserver = new ResizeObserver((entries) => {
                    for (let entry of entries) {
                        const { width, height } = entry.contentRect;
                        if (width > 0 && height > 0) {
                            if (canvas.width !== width || canvas.height !== height) {
                                canvas.width = width;
                                canvas.height = height;
                                draw();
                            }
                        }
                    }
                });
                resizeObserver.observe(viewArea);

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

                const getWidget = (name) => nodeInstance.widgets?.find(w => w.name === name);
                const sxW = getWidget("start_x");
                const syW = getWidget("start_y");
                const exW = getWidget("end_x");
                const eyW = getWidget("end_y");
                const startBrightW = getWidget("start_bright");
                const endBrightW = getWidget("end_bright");
                const overlayColorW = getWidget("overlay_color");
                const startAlphaW = getWidget("start_alpha");
                const endAlphaW = getWidget("end_alpha");

                const clamp01 = (v) => Math.max(0, Math.min(1, v));

                let p0 = {
                    x: sxW ? clamp01(parseFloat(sxW.value)) : 0.0,
                    y: syW ? clamp01(parseFloat(syW.value)) : 0.0
                };
                let p1 = {
                    x: exW ? clamp01(parseFloat(exW.value)) : 1.0,
                    y: eyW ? clamp01(parseFloat(eyW.value)) : 1.0
                };

                const hexToRgb = (hex) => {
                    const s = String(hex || "").replace("#", "");
                    if (s.length !== 6) return [255, 255, 255];
                    const r = parseInt(s.slice(0, 2), 16);
                    const g = parseInt(s.slice(2, 4), 16);
                    const b = parseInt(s.slice(4, 6), 16);
                    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) return [255, 255, 255];
                    return [r, g, b];
                };

                let lastPreviewTime = 0;
                let pendingUpdate = false;
                let isPreviewPending = false;

                const updateLivePreview = (isDragging = false) => {
                    const now = Date.now();
                    const interval = isDragging ? 16 : 33;
                    if (now - lastPreviewTime < interval) {
                        if (!pendingUpdate) {
                            pendingUpdate = true;
                            setTimeout(() => {
                                pendingUpdate = false;
                                updateLivePreview(isDragging);
                            }, interval - (now - lastPreviewTime));
                        }
                        return;
                    }
                    if (isPreviewPending) {
                        pendingUpdate = true;
                        return;
                    }
                    lastPreviewTime = now;
                    pendingUpdate = false;
                    isPreviewPending = true;

                    const body = {
                        node_id: nodeInstance.id.toString(),
                        start_x: p0.x,
                        start_y: p0.y,
                        start_bright: startBrightW ? parseFloat(startBrightW.value) : 1.0,
                        end_x: p1.x,
                        end_y: p1.y,
                        end_bright: endBrightW ? parseFloat(endBrightW.value) : 1.0,
                        overlay_color: overlayColorW ? String(overlayColorW.value) : "#FFFFFF",
                        start_alpha: startAlphaW ? parseFloat(startAlphaW.value) : 0.0,
                        end_alpha: endAlphaW ? parseFloat(endAlphaW.value) : 0.0
                    };

                    (async () => {
                        try {
                            const response = await api.fetchApi("/color_bright_gradient/live_preview", {
                                method: "POST",
                                body: JSON.stringify(body)
                            });
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                            if (response.ok) {
                                const img = await response.json();
                                if (img.filename) {
                                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                                    const imgLoader = new Image();
                                    imgLoader.onload = () => {
                                        bgImageLayer.style.backgroundImage = `url(${url})`;
                                    };
                                    imgLoader.src = url;
                                }
                            }
                        } catch (e) {
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                        }
                    })();
                };

                const updateBackendCoords = () => {
                    if (sxW) sxW.value = p0.x;
                    if (syW) syW.value = p0.y;
                    if (exW) exW.value = p1.x;
                    if (eyW) eyW.value = p1.y;
                    if (app.graph) app.graph.setDirtyCanvas(true);
                };

                const draw = () => {
                    if (!ctx) return;
                    const w = canvas.width;
                    const h = canvas.height;
                    ctx.clearRect(0, 0, w, h);

                    const startBright = startBrightW ? parseFloat(startBrightW.value) : 1.0;
                    const endBright = endBrightW ? parseFloat(endBrightW.value) : 1.0;
                    const startAlpha = startAlphaW ? parseFloat(startAlphaW.value) : 0.0;
                    const endAlpha = endAlphaW ? parseFloat(endAlphaW.value) : 0.0;
                    const [r, g, b] = hexToRgb(overlayColorW ? overlayColorW.value : "#FFFFFF");

                    const x0 = p0.x * w;
                    const y0 = p0.y * h;
                    const x1 = p1.x * w;
                    const y1 = p1.y * h;

                    const hasBg = !!(bgImageLayer && bgImageLayer.style.backgroundImage && bgImageLayer.style.backgroundImage !== "none");
                    ctx.save();
                    if (hasBg) ctx.globalAlpha = 0.25;
                    const g0 = clamp01(startBright / 2.0);
                    const g1 = clamp01(endBright / 2.0);
                    const grad = ctx.createLinearGradient(x0, y0, x1, y1);
                    grad.addColorStop(0, `rgb(${Math.round(255 * g0)},${Math.round(255 * g0)},${Math.round(255 * g0)})`);
                    grad.addColorStop(1, `rgb(${Math.round(255 * g1)},${Math.round(255 * g1)},${Math.round(255 * g1)})`);
                    ctx.fillStyle = grad;
                    ctx.fillRect(0, 0, w, h);

                    const colGrad = ctx.createLinearGradient(x0, y0, x1, y1);
                    colGrad.addColorStop(0, `rgba(${r},${g},${b},${clamp01(startAlpha)})`);
                    colGrad.addColorStop(1, `rgba(${r},${g},${b},${clamp01(endAlpha)})`);
                    ctx.fillStyle = colGrad;
                    ctx.fillRect(0, 0, w, h);
                    ctx.restore();

                    ctx.save();
                    ctx.strokeStyle = "rgba(255,255,255,0.65)";
                    ctx.lineWidth = 1;
                    ctx.beginPath();
                    ctx.moveTo(x0, y0);
                    ctx.lineTo(x1, y1);
                    ctx.stroke();
                    ctx.restore();

                    const drawPt = (x, y) => {
                        ctx.save();
                        ctx.fillStyle = "#ffffff";
                        ctx.strokeStyle = "#000000";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.arc(x, y, 6, 0, Math.PI * 2);
                        ctx.fill();
                        ctx.stroke();
                        ctx.restore();
                    };
                    drawPt(x0, y0);
                    drawPt(x1, y1);
                };

                let drag = -1;
                const hitRadius = 10;

                const getMouse = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const sx = canvas.width / rect.width;
                    const sy = canvas.height / rect.height;
                    const mx = (e.clientX - rect.left) * sx;
                    const my = (e.clientY - rect.top) * sy;
                    return { x: mx, y: my };
                };

                canvas.addEventListener("mousedown", (e) => {
                    const m = getMouse(e);
                    const a = { x: p0.x * canvas.width, y: p0.y * canvas.height };
                    const b = { x: p1.x * canvas.width, y: p1.y * canvas.height };
                    if (Math.hypot(m.x - a.x, m.y - a.y) <= hitRadius) drag = 0;
                    else if (Math.hypot(m.x - b.x, m.y - b.y) <= hitRadius) drag = 1;
                });

                window.addEventListener("mousemove", (e) => {
                    if (drag === -1) return;
                    const m = getMouse(e);
                    if (drag === 0) {
                        p0.x = clamp01(m.x / canvas.width);
                        p0.y = clamp01(m.y / canvas.height);
                    } else {
                        p1.x = clamp01(m.x / canvas.width);
                        p1.y = clamp01(m.y / canvas.height);
                    }
                    updateBackendCoords();
                    draw();
                    updateLivePreview(true);
                });

                window.addEventListener("mouseup", () => {
                    if (drag === -1) return;
                    drag = -1;
                    updateLivePreview(false);
                });

                const bindWidget = (w) => {
                    if (!w) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        draw();
                        updateLivePreview(false);
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => { draw(); updateLivePreview(false); });
                        w.inputEl.addEventListener("input", () => { draw(); updateLivePreview(true); });
                    }
                };

                [startBrightW, endBrightW, overlayColorW, startAlphaW, endAlphaW].forEach(bindWidget);

                draw();
                setTimeout(() => {
                    draw();
                    if (this.onResize) this.onResize(this.size);
                }, 150);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.bg_image?.length > 0) {
                    const img = message.bg_image[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                    const uiWidget = this.widgets.find(w => w.name === "BrightGradientUI");
                    const bg = uiWidget?.element?.querySelector?.(".apt-preview-bg");
                    if (bg) bg.style.backgroundImage = `url(${url})`;
                }
            };
        } else if (nodeData.name === "Image_crop_visual") {
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

                hideWidgetAndSlot("crop_state");

                const cropStateWidget = this.widgets?.find(w => w.name === "crop_state");
                const cropWidthWidget = this.widgets?.find(w => w.name === "crop_width");
                const cropHeightWidget = this.widgets?.find(w => w.name === "crop_height");
                const fillWidget = this.widgets?.find(w => w.name === "fill");
                const marginWidget = this.widgets?.find(w => w.name === "margin");

                let cropState = { cx: 0.5, cy: 0.5, zoom: 1.0 };
                if (cropStateWidget && typeof cropStateWidget.value === "string") {
                    try {
                        const parsed = JSON.parse(cropStateWidget.value);
                        if (parsed && typeof parsed === "object") cropState = { ...cropState, ...parsed };
                    } catch (e) {}
                }

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

                const widget = this.addDOMWidget("ImageCropUI", "div", container, { serialize: false, hideOnZoom: false });
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

                const imageMeta = { img_w: 0, img_h: 0 };
                let userAdjusted = false;
                let dragInfo = null;

                const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
                const getCropDims = () => {
                    const cw = Math.max(1, parseInt(cropWidthWidget?.value ?? 512));
                    const ch = Math.max(1, parseInt(cropHeightWidget?.value ?? 512));
                    return { cw, ch };
                };
                const parseState = () => {
                    const cx = Number.isFinite(+cropState.cx) ? +cropState.cx : 0.5;
                    const cy = Number.isFinite(+cropState.cy) ? +cropState.cy : 0.5;
                    const zoom = Number.isFinite(+cropState.zoom) ? +cropState.zoom : 1.0;
                    cropState.cx = clamp(cx, 0, 1);
                    cropState.cy = clamp(cy, 0, 1);
                    cropState.zoom = Math.max(1e-4, zoom);
                };
                const syncState = () => {
                    parseState();
                    if (cropStateWidget) {
                        cropStateWidget.value = JSON.stringify({
                            cx: cropState.cx,
                            cy: cropState.cy,
                            zoom: cropState.zoom
                        });
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true);
                };
                const fitZoom = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return 1.0;
                    const { cw, ch } = getCropDims();
                    return Math.max(cw / imageMeta.img_w, ch / imageMeta.img_h, 1e-4);
                };
                const getMinZoomForImageBounds = () => {
                    return fitZoom();
                };
                const getMaxZoom = () => {
                    const { cw, ch } = getCropDims();
                    return Math.max(1, Math.min(cw, ch));
                };
                const getSourceRect = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return null;
                    const { cw, ch } = getCropDims();
                    const oldCx = cropState.cx;
                    const oldCy = cropState.cy;
                    const oldZoom = cropState.zoom;
                    const minZoom = fitZoom();
                    cropState.zoom = Math.max(cropState.zoom, minZoom);
                    const srcW = Math.min(imageMeta.img_w, Math.max(1, cw / cropState.zoom));
                    const srcH = Math.min(imageMeta.img_h, Math.max(1, ch / cropState.zoom));
                    let cxPx = clamp(cropState.cx * imageMeta.img_w, 0, imageMeta.img_w);
                    let cyPx = clamp(cropState.cy * imageMeta.img_h, 0, imageMeta.img_h);
                    cxPx = clamp(cxPx, srcW * 0.5, imageMeta.img_w - srcW * 0.5);
                    cyPx = clamp(cyPx, srcH * 0.5, imageMeta.img_h - srcH * 0.5);
                    cropState.cx = cxPx / imageMeta.img_w;
                    cropState.cy = cyPx / imageMeta.img_h;
                    if (cropState.cx !== oldCx || cropState.cy !== oldCy || cropState.zoom !== oldZoom) syncState();
                    return {
                        x: cxPx - srcW * 0.5,
                        y: cyPx - srcH * 0.5,
                        w: srcW,
                        h: srcH
                    };
                };
                const initState = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return;
                    const z = fitZoom();
                    cropState.cx = 0.5;
                    cropState.cy = 0.5;
                    cropState.zoom = z;
                    syncState();
                };
                const getImageRectOnCanvas = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return null;
                    const w = canvas.width;
                    const h = canvas.height;
                    if (w <= 0 || h <= 0) return null;
                    const scale = Math.min(w / imageMeta.img_w, h / imageMeta.img_h);
                    const dw = imageMeta.img_w * scale;
                    const dh = imageMeta.img_h * scale;
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
                    return {
                        x: clamp(ix, 0, imageMeta.img_w),
                        y: clamp(iy, 0, imageMeta.img_h)
                    };
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
                    const sr = getSourceRect();
                    if (!ir || !sr) return;
                    const rx = ir.x + sr.x * ir.scale;
                    const ry = ir.y + sr.y * ir.scale;
                    const rw = sr.w * ir.scale;
                    const rh = sr.h * ir.scale;

                    ctx.fillStyle = "rgba(0,0,0,0.35)";
                    ctx.fillRect(ir.x, ir.y, ir.w, Math.max(0, ry - ir.y));
                    ctx.fillRect(ir.x, ry + rh, ir.w, Math.max(0, ir.y + ir.h - (ry + rh)));
                    ctx.fillRect(ir.x, ry, Math.max(0, rx - ir.x), rh);
                    ctx.fillRect(rx + rw, ry, Math.max(0, ir.x + ir.w - (rx + rw)), rh);

                    ctx.strokeStyle = "#ff5a4f";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(rx, ry, rw, rh);

                    const hs = 6;
                    ctx.fillStyle = "#ffea00";
                    ctx.fillRect(rx - hs, ry - hs, hs * 2, hs * 2);
                    ctx.fillRect(rx + rw - hs, ry - hs, hs * 2, hs * 2);
                    ctx.fillRect(rx - hs, ry + rh - hs, hs * 2, hs * 2);
                    ctx.fillRect(rx + rw - hs, ry + rh - hs, hs * 2, hs * 2);
                };

                const resizeObserver = new ResizeObserver((entries) => {
                    for (let entry of entries) {
                        const { width, height } = entry.contentRect;
                        if (width > 0 && height > 0) {
                            if (canvas.width !== width || canvas.height !== height) {
                                canvas.width = width;
                                canvas.height = height;
                                draw();
                            }
                        }
                    }
                });
                resizeObserver.observe(viewArea);

                const getMouse = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const sx = canvas.width / rect.width;
                    const sy = canvas.height / rect.height;
                    return {
                        x: (e.clientX - rect.left) * sx,
                        y: (e.clientY - rect.top) * sy
                    };
                };

                canvas.addEventListener("mousedown", (e) => {
                    if (e.button !== 0) return;
                    const ir = getImageRectOnCanvas();
                    const sr = getSourceRect();
                    if (!ir || !sr) return;
                    const m = getMouse(e);
                    const rx = ir.x + sr.x * ir.scale;
                    const ry = ir.y + sr.y * ir.scale;
                    const rw = sr.w * ir.scale;
                    const rh = sr.h * ir.scale;
                    const inside = m.x >= rx && m.x <= rx + rw && m.y >= ry && m.y <= ry + rh;
                    canvas.style.cursor = inside ? "move" : "default";
                    if (!inside) return;
                    const imgPt = canvasToImage(m.x, m.y);
                    if (!imgPt) return;
                    dragInfo = {
                        mouseX: imgPt.x,
                        mouseY: imgPt.y,
                        cx: cropState.cx * imageMeta.img_w,
                        cy: cropState.cy * imageMeta.img_h
                    };
                    e.preventDefault();
                });

                window.addEventListener("mousemove", (e) => {
                    const m = getMouse(e);
                    const ir = getImageRectOnCanvas();
                    const sr = getSourceRect();
                    if (ir && sr) {
                        const rx = ir.x + sr.x * ir.scale;
                        const ry = ir.y + sr.y * ir.scale;
                        const rw = sr.w * ir.scale;
                        const rh = sr.h * ir.scale;
                        const over = m.x >= rx && m.x <= rx + rw && m.y >= ry && m.y <= ry + rh;
                        canvas.style.cursor = dragInfo ? "move" : (over ? "move" : "default");
                    }
                    if (!dragInfo) return;
                    const imgPt = canvasToImage(m.x, m.y);
                    if (!imgPt) return;
                    const dx = imgPt.x - dragInfo.mouseX;
                    const dy = imgPt.y - dragInfo.mouseY;
                    
                    // 计算新的中心点
                    const { cw, ch } = getCropDims();
                    const srcW = cw / cropState.zoom;
                    const srcH = ch / cropState.zoom;
                    
                    let newCx = dragInfo.cx + dx;
                    let newCy = dragInfo.cy + dy;
                    
                    // 最后限制不超出图像边界
                    if (srcW <= imageMeta.img_w) {
                        newCx = clamp(newCx, srcW * 0.5, imageMeta.img_w - srcW * 0.5);
                    } else {
                        newCx = imageMeta.img_w * 0.5;
                    }
                    if (srcH <= imageMeta.img_h) {
                        newCy = clamp(newCy, srcH * 0.5, imageMeta.img_h - srcH * 0.5);
                    } else {
                        newCy = imageMeta.img_h * 0.5;
                    }
                    
                    cropState.cx = newCx / imageMeta.img_w;
                    cropState.cy = newCy / imageMeta.img_h;
                    userAdjusted = true;
                    syncState();
                    draw();
                });

                window.addEventListener("mouseup", () => {
                    dragInfo = null;
                });

                canvas.addEventListener("wheel", (e) => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return;
                    e.preventDefault();
                    const m = getMouse(e);
                    const imgPt = canvasToImage(m.x, m.y);
                    if (!imgPt) return;
                    const old = getSourceRect();
                    if (!old) return;
                    const rx = clamp((imgPt.x - old.x) / old.w, 0, 1);
                    const ry = clamp((imgPt.y - old.y) / old.h, 0, 1);
                    // 注意：zoom 越大，红框越小；zoom 越小，红框越大
                    // 向上滚动（deltaY < 0）：放大红框，需要减小 zoom
                    // 向下滚动（deltaY > 0）：缩小红框，需要增加 zoom（但不能超过 maxZoom）
                    const factor = e.deltaY < 0 ? (1 / 1.08) : 1.08;
                    const maxZoom = getMaxZoom();
                    const minZoom = getMinZoomForImageBounds();
                    // zoom 范围：最小 minZoom（红框刚好不超出图像边界），最大 maxZoom（刚好包裹遮罩）
                    cropState.zoom = clamp(cropState.zoom * factor, minZoom, maxZoom);
                    const { cw, ch } = getCropDims();
                    const nw = Math.min(imageMeta.img_w, Math.max(1, cw / cropState.zoom));
                    const nh = Math.min(imageMeta.img_h, Math.max(1, ch / cropState.zoom));
                    let nx = imgPt.x - rx * nw;
                    let ny = imgPt.y - ry * nh;
                    nx = clamp(nx, 0, imageMeta.img_w - nw);
                    ny = clamp(ny, 0, imageMeta.img_h - nh);
                    cropState.cx = (nx + nw * 0.5) / imageMeta.img_w;
                    cropState.cy = (ny + nh * 0.5) / imageMeta.img_h;
                    userAdjusted = true;
                    syncState();
                    draw();
                }, { passive: false });

                const bindCropDim = (w) => {
                    if (!w) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        const maxZoom = getMaxZoom();
                        // 当裁剪尺寸变化时，如果当前 zoom 超过了新的 maxZoom，需要调整
                        cropState.zoom = Math.min(cropState.zoom, maxZoom);
                        if (!userAdjusted) initState();
                        syncState();
                        draw();
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => {
                            const maxZoom = getMaxZoom();
                            cropState.zoom = Math.min(cropState.zoom, maxZoom);
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                        });
                        w.inputEl.addEventListener("input", () => {
                            const maxZoom = getMaxZoom();
                            cropState.zoom = Math.min(cropState.zoom, maxZoom);
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                        });
                    }
                };
                bindCropDim(cropWidthWidget);
                bindCropDim(cropHeightWidget);

                let fillAutoTimer = null;
                const triggerFillAutoPreview = () => {
                    if (fillAutoTimer) clearTimeout(fillAutoTimer);
                    fillAutoTimer = setTimeout(() => {
                        fillAutoTimer = null;
                        runPreview();
                    }, 120);
                };
                const bindFillAuto = (w) => {
                    if (!w) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        if (!userAdjusted) initState();
                        syncState();
                        draw();
                        triggerFillAutoPreview();
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => {
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                            triggerFillAutoPreview();
                        });
                        w.inputEl.addEventListener("input", () => {
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                            triggerFillAutoPreview();
                        });
                    }
                };
                bindFillAuto(fillWidget);
                bindFillAuto(marginWidget);

                this._aptImageCropUI = {
                    imageMeta,
                    draw,
                    initState,
                    getUserAdjusted: () => userAdjusted
                };

                syncState();
                draw();
                setTimeout(() => {
                    draw();
                    if (this.onResize) this.onResize(this.size);
                }, 150);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.bg_image?.length > 0) {
                    const img = message.bg_image[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                    const uiWidget = this.widgets.find(w => w.name === "ImageCropUI");
                    const bg = uiWidget?.element?.querySelector?.(".apt-preview-bg");
                    if (bg) bg.style.backgroundImage = `url(${url})`;

                    const cropUi = this._aptImageCropUI;
                    if (!cropUi) return;
                    const cropMeta = Array.isArray(message?.crop_meta) ? message.crop_meta[0] : null;
                    if (cropMeta && cropMeta.img_w && cropMeta.img_h) {
                        cropUi.imageMeta.img_w = parseInt(cropMeta.img_w);
                        cropUi.imageMeta.img_h = parseInt(cropMeta.img_h);
                        if (!cropUi.getUserAdjusted()) cropUi.initState();
                        cropUi.draw();
                    } else {
                        const imgProbe = new Image();
                        imgProbe.onload = () => {
                            cropUi.imageMeta.img_w = imgProbe.naturalWidth;
                            cropUi.imageMeta.img_h = imgProbe.naturalHeight;
                            if (!cropUi.getUserAdjusted()) cropUi.initState();
                            cropUi.draw();
                        };
                        imgProbe.src = url;
                    }
                }
            };
        } else if (nodeData.name === "Image_mask_crop_visual") {
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

                hideWidgetAndSlot("crop_state");

                const cropStateWidget = this.widgets?.find(w => w.name === "crop_state");
                const cropWidthWidget = this.widgets?.find(w => w.name === "crop_width");
                const cropHeightWidget = this.widgets?.find(w => w.name === "crop_height");
                const cropImgBjWidget = this.widgets?.find(w => w.name === "crop_img_bj");

                let cropState = { cx: 0.5, cy: 0.5, zoom: 1.0 };
                if (cropStateWidget && typeof cropStateWidget.value === "string") {
                    try {
                        const parsed = JSON.parse(cropStateWidget.value);
                        if (parsed && typeof parsed === "object") cropState = { ...cropState, ...parsed };
                    } catch (e) {}
                }

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

                const widget = this.addDOMWidget("ImageMaskCropUI", "div", container, { serialize: false, hideOnZoom: false });
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

                const imageMeta = { img_w: 0, img_h: 0 };
                const maskMeta = { mask_x: 0, mask_y: 0, mask_w: 0, mask_h: 0 };
                let userAdjusted = false;
                let dragInfo = null;

                const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
                const getCropDims = () => {
                    const cw = Math.max(1, parseInt(cropWidthWidget?.value ?? 512));
                    const ch = Math.max(1, parseInt(cropHeightWidget?.value ?? 512));
                    return { cw, ch };
                };
                const parseState = () => {
                    const cx = Number.isFinite(+cropState.cx) ? +cropState.cx : 0.5;
                    const cy = Number.isFinite(+cropState.cy) ? +cropState.cy : 0.5;
                    const zoom = Number.isFinite(+cropState.zoom) ? +cropState.zoom : 1.0;
                    cropState.cx = clamp(cx, 0, 1);
                    cropState.cy = clamp(cy, 0, 1);
                    cropState.zoom = Math.max(1e-4, zoom);
                };
                const syncState = () => {
                    parseState();
                    if (cropStateWidget) {
                        cropStateWidget.value = JSON.stringify({
                            cx: cropState.cx,
                            cy: cropState.cy,
                            zoom: cropState.zoom
                        });
                    }
                    if (app.graph) app.graph.setDirtyCanvas(true);
                };
                
                // 计算能包裹遮罩的最小红框
                const getMinCropBoxForMask = () => {
                    if (!maskMeta.mask_w || !maskMeta.mask_h) return null;
                    const { cw, ch } = getCropDims();
                    const cropRatio = cw / ch;
                    const maskW = maskMeta.mask_w;
                    const maskH = maskMeta.mask_h;
                    
                    // 计算能包裹遮罩的最小等比框
                    let boxW, boxH;
                    if (maskW / maskH >= cropRatio) {
                        boxW = maskW;
                        boxH = maskW / cropRatio;
                    } else {
                        boxH = maskH;
                        boxW = maskH * cropRatio;
                    }
                    return { w: boxW, h: boxH, ratio: cropRatio };
                };
                
                // 计算最大zoom（刚好包裹遮罩时的zoom，此时红框最小）
                // zoom = 输出尺寸 / 源尺寸，zoom越大，源尺寸越小
                // 要包裹遮罩，源尺寸必须 >= 遮罩尺寸，所以 zoom <= 输出尺寸 / 遮罩尺寸
                const getMaxZoom = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return 1.0;
                    const minBox = getMinCropBoxForMask();
                    if (!minBox) return 1.0;
                    const { cw, ch } = getCropDims();
                    // 最大zoom = 输出尺寸 / 最小源尺寸（刚好包裹遮罩）
                    const maxZoomW = cw / minBox.w;
                    const maxZoomH = ch / minBox.h;
                    return Math.min(maxZoomW, maxZoomH);
                };
                
                // 计算最小zoom（红框刚好不超出图像边界时的zoom，此时红框最大）
                const getMinZoomForImageBounds = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return 0.01;
                    const { cw, ch } = getCropDims();
                    // 红框尺寸 = 输出尺寸 / zoom
                    // 红框不超出图像边界：红框尺寸 <= 图像尺寸
                    // 所以 zoom >= 输出尺寸 / 图像尺寸
                    const minZoomW = cw / imageMeta.img_w;
                    const minZoomH = ch / imageMeta.img_h;
                    return Math.max(minZoomW, minZoomH, 0.01);
                };
                
                // 获取源矩形（基于当前zoom和中心点）
                const getSourceRect = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return null;
                    const { cw, ch } = getCropDims();
                    const oldCx = cropState.cx;
                    const oldCy = cropState.cy;
                    const oldZoom = cropState.zoom;
                    const maxZoom = getMaxZoom();
                    const minZoom = getMinZoomForImageBounds();
                    // 确保 zoom 在有效范围内：
                    // - 不超过 maxZoom（刚好包裹遮罩，红框最小）
                    // - 不小于 minZoom（刚好不超出图像边界，红框最大）
                    // zoom 越大，红框越小；zoom 越小，红框越大
                    // 初始状态 zoom = maxZoom（红框刚好包裹遮罩）
                    // 只能减小 zoom（放大红框）到 minZoom，不能更小
                    cropState.zoom = clamp(cropState.zoom, minZoom, maxZoom);
                    
                    // 源裁剪区域的尺寸（在原始图像上的尺寸）
                    const srcW = cw / cropState.zoom;
                    const srcH = ch / cropState.zoom;
                    
                    // 遮罩中心
                    const maskCx = maskMeta.mask_x + maskMeta.mask_w * 0.5;
                    const maskCy = maskMeta.mask_y + maskMeta.mask_h * 0.5;
                    
                    // 中心点像素坐标 - 优先使用当前状态，但要确保能包裹遮罩
                    let cxPx = cropState.cx * imageMeta.img_w;
                    let cyPx = cropState.cy * imageMeta.img_h;
                    
                    // 计算红框边界
                    let left = cxPx - srcW * 0.5;
                    let right = cxPx + srcW * 0.5;
                    let top = cyPx - srcH * 0.5;
                    let bottom = cyPx + srcH * 0.5;
                    
                    // 遮罩边界
                    const maskLeft = maskMeta.mask_x;
                    const maskRight = maskMeta.mask_x + maskMeta.mask_w;
                    const maskTop = maskMeta.mask_y;
                    const maskBottom = maskMeta.mask_y + maskMeta.mask_h;
                    
                    // 调整位置确保遮罩被完全包裹
                    if (left > maskLeft) {
                        // 红框左边界在遮罩左边界的右边，需要左移
                        const shift = left - maskLeft;
                        cxPx -= shift;
                        left -= shift;
                        right -= shift;
                    }
                    if (right < maskRight) {
                        // 红框右边界在遮罩右边界的左边，需要右移
                        const shift = maskRight - right;
                        cxPx += shift;
                        left += shift;
                        right += shift;
                    }
                    if (top > maskTop) {
                        // 红框上边界在遮罩上边界的下边，需要上移
                        const shift = top - maskTop;
                        cyPx -= shift;
                        top -= shift;
                        bottom -= shift;
                    }
                    if (bottom < maskBottom) {
                        // 红框下边界在遮罩下边界的上边，需要下移
                        const shift = maskBottom - bottom;
                        cyPx += shift;
                        top += shift;
                        bottom += shift;
                    }
                    
                    // 最后限制不超出图像边界（如果可能的话，优先保证包裹遮罩）
                    if (srcW <= imageMeta.img_w) {
                        cxPx = clamp(cxPx, srcW * 0.5, imageMeta.img_w - srcW * 0.5);
                    } else {
                        // 如果红框比图像还大，居中
                        cxPx = imageMeta.img_w * 0.5;
                    }
                    if (srcH <= imageMeta.img_h) {
                        cyPx = clamp(cyPx, srcH * 0.5, imageMeta.img_h - srcH * 0.5);
                    } else {
                        cyPx = imageMeta.img_h * 0.5;
                    }
                    
                    cropState.cx = cxPx / imageMeta.img_w;
                    cropState.cy = cyPx / imageMeta.img_h;
                    if (cropState.cx !== oldCx || cropState.cy !== oldCy || cropState.zoom !== oldZoom) syncState();
                    
                    return {
                        x: cxPx - srcW * 0.5,
                        y: cyPx - srcH * 0.5,
                        w: srcW,
                        h: srcH
                    };
                };
                
                // 检查红框是否完全包裹遮罩
                const isMaskFullyContained = (srcRect) => {
                    if (!srcRect || !maskMeta.mask_w) return true;
                    const maskLeft = maskMeta.mask_x;
                    const maskRight = maskMeta.mask_x + maskMeta.mask_w;
                    const maskTop = maskMeta.mask_y;
                    const maskBottom = maskMeta.mask_y + maskMeta.mask_h;
                    
                    const rectLeft = srcRect.x;
                    const rectRight = srcRect.x + srcRect.w;
                    const rectTop = srcRect.y;
                    const rectBottom = srcRect.y + srcRect.h;
                    
                    return rectLeft <= maskLeft && rectRight >= maskRight &&
                           rectTop <= maskTop && rectBottom >= maskBottom;
                };
                
                const initState = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return;
                    const maxZoom = getMaxZoom();
                    // 初始状态：zoom = maxZoom（红框刚好包裹遮罩，最小状态）
                    cropState.zoom = maxZoom;
                    // 初始位置：红框居中于遮罩
                    if (maskMeta.mask_w && maskMeta.mask_h) {
                        cropState.cx = (maskMeta.mask_x + maskMeta.mask_w * 0.5) / imageMeta.img_w;
                        cropState.cy = (maskMeta.mask_y + maskMeta.mask_h * 0.5) / imageMeta.img_h;
                    } else {
                        cropState.cx = 0.5;
                        cropState.cy = 0.5;
                    }
                    syncState();
                };
                const getImageRectOnCanvas = () => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return null;
                    const w = canvas.width;
                    const h = canvas.height;
                    if (w <= 0 || h <= 0) return null;
                    const scale = Math.min(w / imageMeta.img_w, h / imageMeta.img_h);
                    const dw = imageMeta.img_w * scale;
                    const dh = imageMeta.img_h * scale;
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
                    return {
                        x: clamp(ix, 0, imageMeta.img_w),
                        y: clamp(iy, 0, imageMeta.img_h)
                    };
                };
                
                // 绘制遮罩边界框
                const drawMaskBoundingBox = (ir) => {
                    if (!maskMeta.mask_w || !maskMeta.mask_h) return;
                    const mx = ir.x + maskMeta.mask_x * ir.scale;
                    const my = ir.y + maskMeta.mask_y * ir.scale;
                    const mw = maskMeta.mask_w * ir.scale;
                    const mh = maskMeta.mask_h * ir.scale;
                    
                    ctx.strokeStyle = "#00ff00";
                    ctx.lineWidth = 2;
                    ctx.setLineDash([5, 5]);
                    ctx.strokeRect(mx, my, mw, mh);
                    ctx.setLineDash([]);
                    
                    // 绘制遮罩标签
                    ctx.fillStyle = "#00ff00";
                    ctx.font = "10px sans-serif";
                    ctx.fillText("Mask", mx, my - 3);
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
                    const sr = getSourceRect();
                    if (!ir || !sr) return;
                    const rx = ir.x + sr.x * ir.scale;
                    const ry = ir.y + sr.y * ir.scale;
                    const rw = sr.w * ir.scale;
                    const rh = sr.h * ir.scale;

                    // 绘制遮罩边界框（绿色虚线）
                    drawMaskBoundingBox(ir);

                    ctx.fillStyle = "rgba(0,0,0,0.35)";
                    ctx.fillRect(ir.x, ir.y, ir.w, Math.max(0, ry - ir.y));
                    ctx.fillRect(ir.x, ry + rh, ir.w, Math.max(0, ir.y + ir.h - (ry + rh)));
                    ctx.fillRect(ir.x, ry, Math.max(0, rx - ir.x), rh);
                    ctx.fillRect(rx + rw, ry, Math.max(0, ir.x + ir.w - (rx + rw)), rh);

                    ctx.strokeStyle = "#ff5a4f";
                    ctx.lineWidth = 3;
                    ctx.strokeRect(rx, ry, rw, rh);

                    const hs = 6;
                    ctx.fillStyle = "#ffea00";
                    ctx.fillRect(rx - hs, ry - hs, hs * 2, hs * 2);
                    ctx.fillRect(rx + rw - hs, ry - hs, hs * 2, hs * 2);
                    ctx.fillRect(rx - hs, ry + rh - hs, hs * 2, hs * 2);
                    ctx.fillRect(rx + rw - hs, ry + rh - hs, hs * 2, hs * 2);
                };

                const resizeObserver = new ResizeObserver((entries) => {
                    for (let entry of entries) {
                        const { width, height } = entry.contentRect;
                        if (width > 0 && height > 0) {
                            if (canvas.width !== width || canvas.height !== height) {
                                canvas.width = width;
                                canvas.height = height;
                                draw();
                            }
                        }
                    }
                });
                resizeObserver.observe(viewArea);

                const getMouse = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const sx = canvas.width / rect.width;
                    const sy = canvas.height / rect.height;
                    return {
                        x: (e.clientX - rect.left) * sx,
                        y: (e.clientY - rect.top) * sy
                    };
                };

                canvas.addEventListener("mousedown", (e) => {
                    if (e.button !== 0) return;
                    const ir = getImageRectOnCanvas();
                    const sr = getSourceRect();
                    if (!ir || !sr) return;
                    const m = getMouse(e);
                    const rx = ir.x + sr.x * ir.scale;
                    const ry = ir.y + sr.y * ir.scale;
                    const rw = sr.w * ir.scale;
                    const rh = sr.h * ir.scale;
                    const inside = m.x >= rx && m.x <= rx + rw && m.y >= ry && m.y <= ry + rh;
                    canvas.style.cursor = inside ? "move" : "default";
                    if (!inside) return;
                    const imgPt = canvasToImage(m.x, m.y);
                    if (!imgPt) return;
                    dragInfo = {
                        mouseX: imgPt.x,
                        mouseY: imgPt.y,
                        cx: cropState.cx * imageMeta.img_w,
                        cy: cropState.cy * imageMeta.img_h
                    };
                    e.preventDefault();
                });

                window.addEventListener("mousemove", (e) => {
                    const m = getMouse(e);
                    const ir = getImageRectOnCanvas();
                    const sr = getSourceRect();
                    if (ir && sr) {
                        const rx = ir.x + sr.x * ir.scale;
                        const ry = ir.y + sr.y * ir.scale;
                        const rw = sr.w * ir.scale;
                        const rh = sr.h * ir.scale;
                        const over = m.x >= rx && m.x <= rx + rw && m.y >= ry && m.y <= ry + rh;
                        canvas.style.cursor = dragInfo ? "move" : (over ? "move" : "default");
                    }
                    if (!dragInfo) return;
                    const imgPt = canvasToImage(m.x, m.y);
                    if (!imgPt) return;
                    const dx = imgPt.x - dragInfo.mouseX;
                    const dy = imgPt.y - dragInfo.mouseY;
                    const src = getSourceRect();
                    if (!src) return;
                    const halfW = src.w * 0.5;
                    const halfH = src.h * 0.5;
                    const nx = clamp(dragInfo.cx + dx, halfW, imageMeta.img_w - halfW);
                    const ny = clamp(dragInfo.cy + dy, halfH, imageMeta.img_h - halfH);
                    cropState.cx = nx / imageMeta.img_w;
                    cropState.cy = ny / imageMeta.img_h;
                    getSourceRect();
                    dragInfo.cx = cropState.cx * imageMeta.img_w;
                    dragInfo.cy = cropState.cy * imageMeta.img_h;
                    dragInfo.mouseX = imgPt.x;
                    dragInfo.mouseY = imgPt.y;
                    userAdjusted = true;
                    syncState();
                    draw();
                });

                window.addEventListener("mouseup", () => {
                    dragInfo = null;
                });

                canvas.addEventListener("wheel", (e) => {
                    if (!imageMeta.img_w || !imageMeta.img_h) return;
                    e.preventDefault();
                    const m = getMouse(e);
                    const imgPt = canvasToImage(m.x, m.y);
                    if (!imgPt) return;
                    const old = getSourceRect();
                    if (!old) return;
                    const rx = clamp((imgPt.x - old.x) / old.w, 0, 1);
                    const ry = clamp((imgPt.y - old.y) / old.h, 0, 1);
                    // 注意：zoom 越大，红框越小；zoom 越小，红框越大
                    // 向上滚动（deltaY < 0）：放大红框，需要减小 zoom
                    // 向下滚动（deltaY > 0）：缩小红框，需要增加 zoom（但不能超过 maxZoom）
                    const factor = e.deltaY < 0 ? (1 / 1.08) : 1.08;
                    const maxZoom = getMaxZoom();
                    // zoom 范围：最小 0.01（红框很大），最大 maxZoom（刚好包裹遮罩）
                    cropState.zoom = clamp(cropState.zoom * factor, 0.01, maxZoom);
                    const { cw, ch } = getCropDims();
                    const nw = cw / cropState.zoom;
                    const nh = ch / cropState.zoom;
                    let nx = imgPt.x - rx * nw;
                    let ny = imgPt.y - ry * nh;
                    
                    // 计算红框边界
                    let left = nx;
                    let right = nx + nw;
                    let top = ny;
                    let bottom = ny + nh;
                    
                    // 遮罩边界
                    const maskLeft = maskMeta.mask_x;
                    const maskRight = maskMeta.mask_x + maskMeta.mask_w;
                    const maskTop = maskMeta.mask_y;
                    const maskBottom = maskMeta.mask_y + maskMeta.mask_h;
                    
                    // 调整位置确保遮罩被完全包裹
                    if (left > maskLeft) {
                        nx -= (left - maskLeft);
                    }
                    if (right < maskRight) {
                        nx += (maskRight - right);
                    }
                    if (top > maskTop) {
                        ny -= (top - maskTop);
                    }
                    if (bottom < maskBottom) {
                        ny += (maskBottom - bottom);
                    }
                    
                    // 最后限制不超出图像边界
                    if (nw <= imageMeta.img_w) {
                        nx = clamp(nx, 0, imageMeta.img_w - nw);
                    } else {
                        nx = 0;
                    }
                    if (nh <= imageMeta.img_h) {
                        ny = clamp(ny, 0, imageMeta.img_h - nh);
                    } else {
                        ny = 0;
                    }
                    
                    cropState.cx = (nx + nw * 0.5) / imageMeta.img_w;
                    cropState.cy = (ny + nh * 0.5) / imageMeta.img_h;
                    userAdjusted = true;
                    syncState();
                    draw();
                }, { passive: false });

                const bindCropDim = (w) => {
                    if (!w) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        const maxZoom = getMaxZoom();
                        cropState.zoom = Math.min(cropState.zoom, maxZoom);
                        if (!userAdjusted) initState();
                        syncState();
                        draw();
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => {
                            const maxZoom = getMaxZoom();
                            cropState.zoom = Math.min(cropState.zoom, maxZoom);
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                        });
                        w.inputEl.addEventListener("input", () => {
                            const maxZoom = getMaxZoom();
                            cropState.zoom = Math.min(cropState.zoom, maxZoom);
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                        });
                    }
                };
                bindCropDim(cropWidthWidget);
                bindCropDim(cropHeightWidget);

                let fillAutoTimer = null;
                const triggerFillAutoPreview = () => {
                    if (fillAutoTimer) clearTimeout(fillAutoTimer);
                    fillAutoTimer = setTimeout(() => {
                        fillAutoTimer = null;
                        runPreview();
                    }, 120);
                };
                const bindFillAuto = (w) => {
                    if (!w) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        if (!userAdjusted) initState();
                        syncState();
                        draw();
                        triggerFillAutoPreview();
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => {
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                            triggerFillAutoPreview();
                        });
                        w.inputEl.addEventListener("input", () => {
                            if (!userAdjusted) initState();
                            syncState();
                            draw();
                            triggerFillAutoPreview();
                        });
                    }
                };
                bindFillAuto(cropImgBjWidget);

                this._aptImageMaskCropUI = {
                    imageMeta,
                    maskMeta,
                    draw,
                    initState,
                    getUserAdjusted: () => userAdjusted
                };

                syncState();
                draw();
                setTimeout(() => {
                    draw();
                    if (this.onResize) this.onResize(this.size);
                }, 150);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.bg_image?.length > 0) {
                    const img = message.bg_image[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                    const uiWidget = this.widgets.find(w => w.name === "ImageMaskCropUI");
                    const bg = uiWidget?.element?.querySelector?.(".apt-preview-bg");
                    if (bg) bg.style.backgroundImage = `url(${url})`;

                    const cropUi = this._aptImageMaskCropUI;
                    if (!cropUi) return;
                    const cropMeta = Array.isArray(message?.crop_meta) ? message.crop_meta[0] : null;
                    const maskMetaData = Array.isArray(message?.mask_meta) ? message.mask_meta[0] : null;
                    
                    if (cropMeta && cropMeta.img_w && cropMeta.img_h) {
                        cropUi.imageMeta.img_w = parseInt(cropMeta.img_w);
                        cropUi.imageMeta.img_h = parseInt(cropMeta.img_h);
                    }
                    
                    if (maskMetaData) {
                        cropUi.maskMeta.mask_x = parseInt(maskMetaData.mask_x) || 0;
                        cropUi.maskMeta.mask_y = parseInt(maskMetaData.mask_y) || 0;
                        cropUi.maskMeta.mask_w = parseInt(maskMetaData.mask_w) || 0;
                        cropUi.maskMeta.mask_h = parseInt(maskMetaData.mask_h) || 0;
                    }
                    
                    if ((cropMeta && cropMeta.img_w && cropMeta.img_h) || (maskMetaData && maskMetaData.mask_w)) {
                        if (!cropUi.getUserAdjusted()) cropUi.initState();
                        cropUi.draw();
                    } else {
                        const imgProbe = new Image();
                        imgProbe.onload = () => {
                            cropUi.imageMeta.img_w = imgProbe.naturalWidth;
                            cropUi.imageMeta.img_h = imgProbe.naturalHeight;
                            if (!cropUi.getUserAdjusted()) cropUi.initState();
                            cropUi.draw();
                        };
                        imgProbe.src = url;
                    }
                }
            };
        } else if ([
            "color_adjust_HSL_visual",
            "color_adjust_HDR_visual",
            "color_match_adv_visual",
            "Image_CnMapMix_visual",
            "Image_Detail_HL_frequencye_visual",
        ].includes(nodeData.name)) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                const MIN_NODE_WIDTH = 240;
                const MIN_NODE_HEIGHT = 280;
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

                const baseName = nodeData.name.endsWith("_visual") ? nodeData.name.slice(0, -7) : nodeData.name;
                const cfgByName = {
                    "color_adjust_HSL": {
                        endpoint: "/color_adjust_hsl/live_preview",
                        params: {
                            hue: "float",
                            brightness: "float",
                            contrast: "float",
                            saturation: "float",
                            sharpness: "float",
                            blur: "int",
                            gaussian_blur: "float",
                            edge_enhance: "float",
                            detail_enhance: "string",
                        }
                    },
                    "color_adjust_HDR": {
                        endpoint: "/color_adjust_hdr/live_preview",
                        params: {
                            HDR_intensity: "float",
                            underexposure_factor: "float",
                            overexposure_factor: "float",
                            gamma: "float",
                            highlight_detail: "float",
                            midtone_detail: "float",
                            shadow_detail: "float",
                            overall_intensity: "float",
                        }
                    },
                    "color_match_adv": {
                        endpoint: "/color_match_adv/live_preview",
                        params: {
                            strength: "float",
                            skin_protection: "float",
                            brightness_range: "float",
                            contrast_range: "float",
                            saturation_range: "float",
                            tone_strength: "float",
                        }
                    },
                    "Image_CnMapMix": {
                        endpoint: "/image_cnmapmix/live_preview",
                        params: {
                            bg_color: "string",
                            blur_1: "int",
                            blur_2: "int",
                            diff_sensitivity: "float",
                            diff_blur: "int",
                            blend_mode: "string",
                            blend_factor: "float",
                            contrast: "float",
                            brightness: "float",
                            mask2_smoothness: "int",
                            invert_mask: "bool",
                            image1_min_black: "float",
                            image1_max_white: "float",
                            image2_min_black: "float",
                            image2_max_white: "float",
                        }
                    },
                    "Image_Detail_HL_frequencye": {
                        endpoint: "/image_detail_hl_frequencye/live_preview",
                        params: {
                            keep_high_freq: "int",
                            erase_low_freq: "int",
                            mask_blur: "int",
                            blend_mode: "string",
                            blend_opacity: "int",
                            detail_strength: "int",
                            high_freq_threshold: "float",
                            invert_mask: "bool",
                        }
                    },
                };

                const cfg = cfgByName[baseName];
                if (!cfg) return;

                const container = document.createElement("div");
                container.style.display = "flex";
                container.style.flexDirection = "column";
                container.style.width = "100%";
                container.style.height = "100%";
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

                loadBtn.onclick = async () => {
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

                previewBar.style.marginBottom = isNodes2_0 ? "10px" : "0px";
                previewBar.appendChild(loadBtn);
                container.appendChild(previewBar);

                const nodeInstance = this;
                const UI_NAME = "AptLivePreviewUI";
                const widget = this.addDOMWidget(UI_NAME, "div", container, { serialize: false, hideOnZoom: false });
                const UI_DEFAULT_HEIGHT = 220;
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
                const getWidget = (name) => nodeInstance.widgets?.find(w => w.name === name);

                const parseWidgetValue = (w, kind) => {
                    if (!w) return undefined;
                    const v = w.value;
                    if (kind === "int") return parseInt(v);
                    if (kind === "float") return parseFloat(v);
                    if (kind === "bool") return !!v;
                    return v;
                };

                let lastPreviewTime = 0;
                let pendingUpdate = false;
                let isPreviewPending = false;

                const updateLivePreview = (isDragging = false) => {
                    const now = Date.now();
                    const interval = isDragging ? 16 : 33;
                    if (now - lastPreviewTime < interval) {
                        if (!pendingUpdate) {
                            pendingUpdate = true;
                            setTimeout(() => {
                                pendingUpdate = false;
                                updateLivePreview(isDragging);
                            }, interval - (now - lastPreviewTime));
                        }
                        return;
                    }
                    if (isPreviewPending) {
                        pendingUpdate = true;
                        return;
                    }
                    lastPreviewTime = now;
                    pendingUpdate = false;
                    isPreviewPending = true;

                    const body = { node_id: nodeInstance.id.toString() };
                    for (const k of Object.keys(cfg.params)) {
                        const w = getWidget(k);
                        const pv = parseWidgetValue(w, cfg.params[k]);
                        if (pv !== undefined && !Number.isNaN(pv)) body[k] = pv;
                    }

                    (async () => {
                        try {
                            const response = await api.fetchApi(cfg.endpoint, {
                                method: "POST",
                                body: JSON.stringify(body)
                            });
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                            if (response.ok) {
                                const img = await response.json();
                                if (img.filename) {
                                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                                    const imgLoader = new Image();
                                    imgLoader.onload = () => {
                                        bgImageLayer.style.backgroundImage = `url(${url})`;
                                    };
                                    imgLoader.src = url;
                                }
                            }
                        } catch (e) {
                            isPreviewPending = false;
                            if (pendingUpdate) {
                                pendingUpdate = false;
                                setTimeout(() => updateLivePreview(isDragging), 0);
                            }
                        }
                    })();
                };

                const bindWidget = (w) => {
                    if (!w) return;
                    if (w.hidden || w.type === "hidden") return;
                    if (w.name === UI_NAME) return;
                    const originalCallback = w.callback;
                    w.callback = function (value) {
                        if (originalCallback) originalCallback.apply(this, arguments);
                        updateLivePreview(false);
                    };
                    if (w.inputEl) {
                        w.inputEl.addEventListener("change", () => updateLivePreview(false));
                        w.inputEl.addEventListener("input", () => updateLivePreview(true));
                    }
                };

                (this.widgets || []).forEach(bindWidget);
            };

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message?.bg_image?.length > 0) {
                    const img = message.bg_image[0];
                    const url = api.apiURL(`/view?filename=${encodeURIComponent(img.filename)}&type=${img.type}&subfolder=${img.subfolder}&t=${Date.now()}`);
                    const uiWidget = this.widgets.find(w => w.name === "AptLivePreviewUI");
                    const bg = uiWidget?.element?.querySelector?.(".apt-preview-bg");
                    if (bg) bg.style.backgroundImage = `url(${url})`;
                }
            };
        }
    }
});
