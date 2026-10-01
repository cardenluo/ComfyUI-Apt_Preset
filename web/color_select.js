import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "color_select",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "color_select") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            const node = this;

            // --- 1. 获取 Widget ---
            const wColorCode = this.widgets.find((w) => w.name === "color_code");
            const wPreset = this.widgets.find((w) => w.name === "preset_color");

            // 默认尺寸 240 x 400
            this.setSize([240, 400]);

            const minWidth = 240;
            const minHeight = 400; 
            this.onResize = function(size) {
                size[0] = Math.max(size[0], minWidth);
                size[1] = Math.max(size[1], minHeight);
            };

            // --- 2. 构建响应式 DOM 结构 ---
            const container = document.createElement("div");
            Object.assign(container.style, {
                position: "relative",
                display: "flex",
                flexDirection: "column",
                width: "100%",
                marginBottom: "10px", 
                boxSizing: "border-box",
                padding: "10px 10px", 
                gap: "4px", 
                overflow: "hidden" 
            });

            // 注入样式
            const style = document.createElement("style");
            style.textContent = `
                .pix-color-slider {
                    -webkit-appearance: none;
                    width: 100%;
                    background: transparent;
                    outline: none;
                    margin: 0;
                    height: 14px;
                    display: block; 
                }
                .pix-color-slider::-webkit-slider-runnable-track {
                    width: 100%;
                    height: 14px;
                    cursor: pointer;
                    border-radius: 2px;
                }
                .pix-color-slider::-webkit-slider-thumb {
                    -webkit-appearance: none;
                    height: 18px;
                    width: 10px;
                    border: 2px solid #fff;
                    background: rgba(0,0,0,0.1);
                    cursor: pointer;
                    margin-top: -2px;
                    border-radius: 1px;
                    box-shadow: 0 0 3px rgba(0,0,0,0.5);
                }
                .pix-hex-text {
                    position: absolute;
                    top: 2px;
                    left: 4px;
                    font-family: sans-serif;
                    font-size: 12px;
                    color: rgba(255,255,255,0.4); 
                    background: transparent;
                    border: none;
                    outline: none;
                    width: 60px;
                    z-index: 10;
                    padding: 4px;
                    margin: 0;
                    transition: color 0.2s;
                }
                .pix-hex-text:hover, .pix-hex-text:focus {
                    color: rgba(255,255,255,0.9); 
                }
            `;
            container.appendChild(style);

            // --- 顶部控件 ---
            const hexInput = document.createElement("input");
            hexInput.type = "text";
            hexInput.className = "pix-hex-text";
            hexInput.value = "#FFFFFF";
            hexInput.maxLength = 7;
            hexInput.title = "Click to edit HEX";
            container.appendChild(hexInput);

            const eyeDropperBtn = document.createElement("button");
            eyeDropperBtn.innerHTML = "🖌️";
            eyeDropperBtn.title = "Pick Color";
            Object.assign(eyeDropperBtn.style, {
                position: "absolute",
                top: "2px", 
                right: "4px", 
                width: "24px",
                height: "24px",
                border: "none",
                background: "transparent",
                color: "rgba(255,255,255,0.6)",
                cursor: "pointer",
                padding: "0",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                zIndex: "10"
            });
            eyeDropperBtn.onmouseenter = () => eyeDropperBtn.style.color = "#fff";
            eyeDropperBtn.onmouseleave = () => eyeDropperBtn.style.color = "rgba(255,255,255,0.6)";
            container.appendChild(eyeDropperBtn);

            // --- 中间色轮 ---
            const wheelContainer = document.createElement("div");
            Object.assign(wheelContainer.style, {
                flex: "1 1 auto", 
                minHeight: "150px",   
                width: "100%",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                position: "relative",
                marginTop: "16px"
            });
            container.appendChild(wheelContainer);

            const canvas = document.createElement("canvas");
            canvas.style.cursor = "crosshair";
            canvas.style.borderRadius = "50%";
            canvas.style.maxWidth = "100%";
            canvas.style.maxHeight = "100%";
            canvas.style.display = "block"; 
            wheelContainer.appendChild(canvas);

            // --- 底部明度条 ---
            const sliderContainer = document.createElement("div");
            Object.assign(sliderContainer.style, {
                flex: "0 0 auto", 
                width: "100%",
                height: "24px",
                display: "flex",
                alignItems: "center",
                marginTop: "8px"
            });
            container.appendChild(sliderContainer);

            const valInput = document.createElement("input");
            valInput.type = "range";
            valInput.className = "pix-color-slider";
            valInput.min = "0"; valInput.max = "100"; valInput.step = "1";
            sliderContainer.appendChild(valInput);

            // --- 3. 逻辑与状态 ---
            let state = { h: 0, s: 0, v: 100 };
            let currentRadius = 80;
            let currentWheelSize = 160;
            const dpr = Math.max(window.devicePixelRatio || 1, 2);
            const ctx = canvas.getContext("2d");

            let isUpdating = false;

            // 预设颜色映射
            const colorPresets = {
                "None": "#FFFFFF",
                "红色": "#ff0000",
                "橙色": "#ff7f00",
                "黄色": "#ffff00",
                "绿色": "#00ff00",
                "蓝色": "#0000ff",
                "黑色": "#000000",
                "白色": "#ffffff",
                "中灰色": "#808080"
            };

            // --- 工具函数 ---
            function hsvToRgb(h, s, v) {
                s /= 100; v /= 100;
                let c = v * s;
                let x = c * (1 - Math.abs(((h / 60) % 2) - 1));
                let m = v - c;
                let r = 0, g = 0, b = 0;
                if (0 <= h && h < 60) { r = c; g = x; b = 0; }
                else if (60 <= h && h < 120) { r = x; g = c; b = 0; }
                else if (120 <= h && h < 180) { r = 0; g = c; b = x; }
                else if (180 <= h && h < 240) { r = 0; g = x; b = c; }
                else if (240 <= h && h < 300) { r = x; g = 0; b = c; }
                else if (300 <= h && h < 360) { r = c; g = 0; b = x; }
                return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
            }

            function rgbToHsv(r, g, b) {
                r /= 255; g /= 255; b /= 255;
                let max = Math.max(r, g, b), min = Math.min(r, g, b);
                let h, s, v = max;
                let d = max - min;
                s = max === 0 ? 0 : d / max;
                if (max === min) h = 0;
                else {
                    switch (max) {
                        case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                        case g: h = (b - r) / d + 2; break;
                        case b: h = (r - g) / d + 4; break;
                    }
                    h /= 6;
                }
                return { h: h * 360, s: s * 100, v: v * 100 };
            }

            function hexToRgb(hex) {
                let result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
                if(!result) {
                    result = /^#?([a-f\d]{1})([a-f\d]{1})([a-f\d]{1})$/i.exec(hex);
                    if(result) {
                         return {
                            r: parseInt(result[1]+result[1], 16),
                            g: parseInt(result[2]+result[2], 16),
                            b: parseInt(result[3]+result[3], 16)
                        };
                    }
                }
                return result ? {
                    r: parseInt(result[1], 16),
                    g: parseInt(result[2], 16),
                    b: parseInt(result[3], 16)
                } : null;
            }

            function rgbToHex(r, g, b) {
                return "#" + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1).toUpperCase();
            }

            function drawWheel() {
                const size = currentWheelSize;
                const r = currentRadius;
                const cx = size / 2;
                const cy = size / 2;

                ctx.clearRect(0, 0, size, size);

                ctx.save();
                ctx.beginPath();
                ctx.arc(cx, cy, r, 0, Math.PI * 2);
                ctx.closePath();
                ctx.clip();

                for (let i = 0; i < 360; i++) {
                    ctx.beginPath();
                    ctx.moveTo(cx, cy);
                    ctx.arc(cx, cy, r + 2, (i - 1) * (Math.PI / 180), (i + 1) * (Math.PI / 180));
                    ctx.closePath();
                    ctx.fillStyle = `hsl(${i}, 100%, 50%)`;
                    ctx.fill();
                }

                const gradWhite = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
                gradWhite.addColorStop(0, 'rgba(255, 255, 255, 1)');
                gradWhite.addColorStop(1, 'rgba(255, 255, 255, 0)');
                ctx.fillStyle = gradWhite;
                ctx.fillRect(0, 0, size, size);

                const brightness = 1 - (state.v / 100);
                if (brightness > 0) {
                    ctx.fillStyle = `rgba(0, 0, 0, ${brightness})`;
                    ctx.fillRect(0, 0, size, size);
                }

                ctx.restore();

                const angle = state.h * (Math.PI / 180);
                const dist = (state.s / 100) * r;
                const px = cx + Math.cos(angle) * dist;
                const py = cy + Math.sin(angle) * dist;

                ctx.strokeStyle = "#fff";
                ctx.lineWidth = 2;
                ctx.shadowColor = "rgba(0,0,0,0.5)";
                ctx.shadowBlur = 2;
                ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke();
                ctx.shadowColor = "transparent";
                ctx.shadowBlur = 0;
            }

            // UI -> Widget 更新逻辑
            function updateAll(skipHexUpdate = false) {
                if (isUpdating) return;
                isUpdating = true;

                try {
                    drawWheel();

                    const [r, g, b] = hsvToRgb(state.h, state.s, state.v);
                    const hexColor = rgbToHex(r, g, b);

                    valInput.value = state.v;
                    const [fullR, fullG, fullB] = hsvToRgb(state.h, state.s, 100);
                    valInput.style.background = `linear-gradient(to right, #000, rgb(${fullR},${fullG},${fullB}))`;

                    if (!skipHexUpdate && document.activeElement !== hexInput) {
                        hexInput.value = hexColor;
                    }

                    // 【核心修复】更新 color_code widget 的值
                    if (wColorCode) {
                        wColorCode.value = hexColor;
                    }

                    app.graph.setDirtyCanvas(true, true);
                } finally {
                    isUpdating = false;
                }
            }

            function updateFromRGB(r, g, b) {
                const hsv = rgbToHsv(r, g, b);
                state.h = hsv.h;
                state.s = hsv.s;
                state.v = hsv.v;
                updateAll();
            }

            function syncUIFromWidgets() {
                if (isUpdating) return;
                isUpdating = true;

                try {
                    // 从 color_code widget 读取当前值
                    if (wColorCode && wColorCode.value) {
                        const rgb = hexToRgb(wColorCode.value);
                        if (rgb) {
                            const hsv = rgbToHsv(rgb.r, rgb.g, rgb.b);
                            state.h = hsv.h;
                            state.s = hsv.s;
                            state.v = hsv.v;
                        }
                    }

                    drawWheel();
                    valInput.value = state.v;
                    const [fullR, fullG, fullB] = hsvToRgb(state.h, state.s, 100);
                    valInput.style.background = `linear-gradient(to right, #000, rgb(${fullR},${fullG},${fullB}))`;

                    const [r, g, b] = hsvToRgb(state.h, state.s, state.v);
                    if (document.activeElement !== hexInput) {
                        hexInput.value = rgbToHex(r, g, b);
                    }
                } finally {
                    isUpdating = false;
                }
            }

            // 设置 widget 回调
            function setupWidgetCallback(w) {
                if (!w) return;
                const originalCallback = w.callback;
                w.callback = function(v) {
                    if (originalCallback) originalCallback.call(w, v);
                    // 当 preset_color 变化时，从预设映射更新颜色
                    if (w.name === "preset_color" && colorPresets[v]) {
                        const rgb = hexToRgb(colorPresets[v]);
                        if (rgb) {
                            updateFromRGB(rgb.r, rgb.g, rgb.b);
                        }
                    }
                };
            }

            setupWidgetCallback(wPreset);
            setupWidgetCallback(wColorCode); 

            // --- 4. 响应式监听 ---
            const resizeObserver = new ResizeObserver(entries => {
                for (let entry of entries) {
                    const rect = entry.contentRect;
                    const w = rect.width;
                    const h = rect.height;

                    const minDim = Math.min(w, h);
                    if (minDim < 20) return; 

                    const newSize = Math.floor(minDim);
                    
                    if (Math.abs(newSize - currentWheelSize) > 0) {
                        currentWheelSize = newSize;
                        currentRadius = (newSize / 2) - 2; 

                        canvas.width = currentWheelSize * dpr;
                        canvas.height = currentWheelSize * dpr;
                        canvas.style.width = `${currentWheelSize}px`;
                        canvas.style.height = `${currentWheelSize}px`;
                        
                        ctx.scale(dpr, dpr);
                        drawWheel();
                    }
                }
            });
            resizeObserver.observe(wheelContainer);


            // --- 事件监听 ---
            canvas.addEventListener("dblclick", () => {
                // 双击可以切换到默认白色
                state.h = 0;
                state.s = 0;
                state.v = 100;
                updateAll();
            });

            let isDragging = false;
            function handleWheel(e) {
                if (!isDragging && e.type !== "mousedown") return;
                const rect = canvas.getBoundingClientRect();
                const x = e.clientX - rect.left - (rect.width / 2);
                const y = e.clientY - rect.top - (rect.height / 2);
                
                let angle = Math.atan2(y, x) * (180 / Math.PI);
                if (angle < 0) angle += 360;
                
                const r = rect.width / 2; 
                let dist = Math.sqrt(x*x + y*y);
                let sat = (dist / (r - 2)) * 100;
                if (sat > 100) sat = 100;
                
                state.h = angle; 
                state.s = sat;
                updateAll();
            }
            
            canvas.addEventListener("mousedown", (e) => { isDragging = true; handleWheel(e); });
            window.addEventListener("mouseup", () => { isDragging = false; });
            window.addEventListener("mousemove", (e) => { if(isDragging) handleWheel(e); });
            valInput.addEventListener("input", (e) => { state.v = parseInt(e.target.value); updateAll(); });
            
            hexInput.addEventListener("input", (e) => {
                const val = e.target.value;
                const rgb = hexToRgb(val);
                if (rgb) {
                    const hsv = rgbToHsv(rgb.r, rgb.g, rgb.b);
                    state.h = hsv.h;
                    state.s = hsv.s;
                    state.v = hsv.v;
                    updateAll(true); 
                }
            });

            eyeDropperBtn.onclick = async () => {
                if (!window.EyeDropper) { alert("Browser doesn't support EyeDropper API"); return; }
                try {
                    const dropper = new window.EyeDropper();
                    const result = await dropper.open();
                    const rgb = hexToRgb(result.sRGBHex);
                    if (rgb) updateFromRGB(rgb.r, rgb.g, rgb.b);
                } catch (err) {}
            };

            // 初始化
            // 先从 color_code widget 初始化，如果有预设颜色则使用预设颜色
            if (wPreset && wPreset.value !== "None" && colorPresets[wPreset.value]) {
                const rgb = hexToRgb(colorPresets[wPreset.value]);
                if (rgb) {
                    updateFromRGB(rgb.r, rgb.g, rgb.b);
                }
            } else {
                syncUIFromWidgets();
            }

            this.addDOMWidget("pix_color_picker_responsive", "ui", container);
            return r;
        };
    }
});