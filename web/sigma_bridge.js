
//声明：此节点来自   https://github.com/LAOGOU-666/ComfyUI-LG_SamplingUtils  


import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


app.registerExtension({
    name: "scheduler_interactive_sigmas",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "scheduler_interactive_sigmas") {
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                const result = onNodeCreated?.apply(this, arguments);
                
                this.sigmas_data = null;
                this.adjustments = [];
                this.dragging_point = -1;
                this.isAdjusting = false;
                
                this.resizable = true;
                
                this.size = [this.size[0] || 600, 400];
                
                this.setupWebSocket();
                
                return result;
            };
            
            nodeType.prototype.setupWebSocket = function() {
                const messageHandler = (event) => {
                    const data = event.detail;
                    
                    if (!data || !data.node_id || !data.sigmas_data) {
                        return;
                    }
                    
                    const targetNode = app.graph.getNodeById(parseInt(data.node_id));
                    
                    if (targetNode && targetNode === this) {
                        this.sigmas_data = data.sigmas_data.original;
                        this.adjustments = data.sigmas_data.adjusted || data.sigmas_data.original.slice();
                        
                        this.updateAdjustmentsWidget();
                        
                        if (this.canvas) {
                            this.updateCanvas(true);
                        } else {
                            setTimeout(() => {
                                if (this.canvas) {
                                    this.updateCanvas(true);
                                }
                            }, 100);
                        }
                        this.setDirtyCanvas(true, true);
                    }
                };
                
                api.addEventListener("sigmas_editor_update", messageHandler);
                
                this._sigmasEditorMessageHandler = messageHandler;
            };
            
            const onAdded = nodeType.prototype.onAdded;
            nodeType.prototype.onAdded = function() {
                const result = onAdded?.apply(this, arguments);
                
                if (!this.canvasContainer && this.id !== undefined && this.id !== -1) {
                    const container = document.createElement("div");
                    container.style.position = "relative";
                    container.style.width = "100%";
                    container.style.height = "100%";
                    container.style.minHeight = "300px";
                    container.style.backgroundColor = "#1e1e1e";
                    container.style.borderRadius = "8px";
                    container.style.overflow = "hidden";
                    
                    const canvas = document.createElement("canvas");
                    canvas.style.width = "100%";
                    canvas.style.height = "100%";
                    canvas.style.cursor = "crosshair";
                    
                    container.appendChild(canvas);
                    this.canvas = canvas;
                    this.canvasContainer = container;
                    
                    this.addCanvasEventListeners();
                    
                    this.widgets ||= [];
                    this.widgets_up = true;
                    
                    requestAnimationFrame(() => {
                        if (this.widgets) {
                            this.canvasWidget = this.addDOMWidget("sigmas_canvas", "canvas", container);
                            
                            this.updateCanvas(true);
                            this.setDirtyCanvas(true, true);
                        }
                    });
                }
                
                return result;
            };
            
            nodeType.prototype.addCanvasEventListeners = function() {
                const canvas = this.canvas;
                
                const getScaledMousePos = (e) => {
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    
                    const displayX = e.clientX - rect.left;
                    const displayY = e.clientY - rect.top;
                    
                    const canvasX = displayX * scaleX;
                    const canvasY = displayY * scaleY;
                    
                    return { x: canvasX, y: canvasY };
                };
                
                canvas.addEventListener("mousedown", (e) => {
                    const pos = getScaledMousePos(e);
                    const pointIdx = this.findNearestPoint(pos.x, pos.y);
                    
                    if (pointIdx !== -1) {
                        this.dragging_point = pointIdx;
                        this.isAdjusting = true;
                        this.updateCanvas();
                    }
                });
                
                canvas.addEventListener("mousemove", (e) => {
                    if (this.dragging_point !== -1 && this.isAdjusting) {
                        const pos = getScaledMousePos(e);
                        this.updatePointAdjustment(this.dragging_point, pos.y);
                        this.updateAdjustmentsWidget();
                        this.updateCanvas();
                    }
                });
                
                canvas.addEventListener("mouseup", (e) => {
                    if (this.dragging_point !== -1) {
                        this.dragging_point = -1;
                        this.isAdjusting = false;
                        
                        this.updateAdjustmentsWidget();
                        this.updateCanvas();
                    }
                });
                
                canvas.addEventListener("mouseleave", (e) => {
                    if (this.dragging_point !== -1) {
                        this.dragging_point = -1;
                        this.isAdjusting = false;
                        this.updateAdjustmentsWidget();
                        this.updateCanvas();
                    }
                });
            };
            
            nodeType.prototype.findNearestPoint = function(mouseX, mouseY) {
                if (!this.sigmas_data || this.sigmas_data.length === 0) return -1;
                
                const paddingLeft = 60;
                const paddingTop = 50;
                const paddingRight = 20;
                const paddingBottom = 50;
                
                const canvas = this.canvas;
                const chartWidth = canvas.width - paddingLeft - paddingRight;
                const chartHeight = canvas.height - paddingTop - paddingBottom;
                const chartX = paddingLeft;
                const chartY = paddingTop;
                
                const steps = this.sigmas_data.length;
                
                let closestDist = Infinity;
                let closestIdx = -1;
                
                for (let i = 0; i < steps; i++) {
                    const x = chartX + (chartWidth / (steps - 1)) * i;
                    const adjustedValue = this.adjustments[i] !== undefined ? this.adjustments[i] : this.sigmas_data[i];
                    const y = chartY + chartHeight - (adjustedValue * chartHeight);
                    
                    const dist = Math.sqrt(Math.pow(mouseX - x, 2) + Math.pow(mouseY - y, 2));
                    if (dist < 15 && dist < closestDist) {
                        closestDist = dist;
                        closestIdx = i;
                    }
                }
                
                return closestIdx;
            };
            
            nodeType.prototype.updatePointAdjustment = function(pointIdx, mouseY) {
                if (!this.sigmas_data || pointIdx < 0 || pointIdx >= this.sigmas_data.length) return;
                
                const paddingTop = 50;
                const paddingBottom = 50;
                
                const canvas = this.canvas;
                const chartHeight = canvas.height - paddingTop - paddingBottom;
                const chartY = paddingTop;
                
                const clampedY = Math.max(chartY, Math.min(chartY + chartHeight, mouseY));
                const newSigmaValue = (chartY + chartHeight - clampedY) / chartHeight;
                
                this.adjustments[pointIdx] = Math.max(0.0, Math.min(1.0, newSigmaValue));
            };
            
            nodeType.prototype.ceilToFixed = function(value, decimals) {
                const multiplier = Math.pow(10, decimals);
                return Math.ceil(value * multiplier) / multiplier;
            };
            
            nodeType.prototype.updateAdjustmentsWidget = function() {
                const widget = this.widgets?.find(w => w.name === "sigmas_adjustments");
                if (widget) {
                    const formattedValues = this.adjustments.map(v => {
                        const rounded = this.ceilToFixed(v, 4);
                        return rounded.toFixed(4);
                    });
                    widget.value = '[' + formattedValues.join(', ') + ']';
                }
            };
            
            nodeType.prototype.updateCanvas = function(forceResize = false) {
                if (!this.canvas) return;
                
                requestAnimationFrame(() => {
                    const canvas = this.canvas;
                    const ctx = canvas.getContext("2d");
                    
                    if (forceResize || !this._canvasInitialized) {
                        const rect = canvas.getBoundingClientRect();
                        
                        const width = rect.width > 0 ? rect.width : 600;
                        const height = rect.height > 0 ? rect.height : 300;
                        
                        if (canvas.width !== width || canvas.height !== height) {
                            canvas.width = width;
                            canvas.height = height;
                        }
                        
                        this._canvasInitialized = true;
                    }
                    
                    const paddingLeft = 60;
                    const paddingRight = 20;
                    const paddingTop = 50;
                    const paddingBottom = 50;
                    
                    const chartWidth = canvas.width - paddingLeft - paddingRight;
                    const chartHeight = canvas.height - paddingTop - paddingBottom;
                    const chartX = paddingLeft;
                    const chartY = paddingTop;
                    
                    ctx.fillStyle = "#1e1e1e";
                    ctx.fillRect(0, 0, canvas.width, canvas.height);
                    
                    if (!this.sigmas_data || this.sigmas_data.length === 0) {
                        ctx.fillStyle = "#999";
                        ctx.font = "14px Arial";
                        ctx.textAlign = "center";
                        ctx.fillText("Connect Sigmas Input & Execute Workflow", canvas.width / 2, canvas.height / 2);
                        return;
                    }
                    
                    ctx.fillStyle = "#2a2a2a";
                    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
                    
                    ctx.strokeStyle = "#444";
                    ctx.lineWidth = 1;
                    for (let i = 0; i <= 10; i++) {
                        const y = chartY + (chartHeight / 10) * i;
                        ctx.beginPath();
                        ctx.moveTo(chartX, y);
                        ctx.lineTo(chartX + chartWidth, y);
                        ctx.stroke();
                    }
                    
                    const steps = this.sigmas_data.length;
                    
                    if (this.adjustments.length !== steps) {
                        this.adjustments = this.sigmas_data.slice();
                    }
                    
                    ctx.fillStyle = "#999";
                    ctx.font = "10px Arial";
                    ctx.textAlign = "right";
                    for (let i = 0; i <= 10; i++) {
                        const value = 1.0 - (i / 10);
                        const y = chartY + (chartHeight / 10) * i;
                        ctx.fillText(value.toFixed(1), chartX - 5, y + 3);
                    }
                    
                    ctx.save();
                    ctx.translate(15, chartY + chartHeight / 2);
                    ctx.rotate(-Math.PI / 2);
                    ctx.textAlign = "center";
                    ctx.font = "10px Arial";
                    ctx.fillStyle = "#ffffffff";
                    ctx.fillText("Sigma_Value", 0, 0);
                    ctx.restore();
                    
                    ctx.fillStyle = "#999";
                    ctx.font = "10px Arial";
                    ctx.textAlign = "center";
                    
                    let stepInterval = 1;
                    if (steps > 30) stepInterval = 2;
                    if (steps > 50) stepInterval = 5;
                    if (steps > 100) stepInterval = 10;
                    
                    for (let i = 0; i < steps; i += stepInterval) {
                        const x = chartX + (chartWidth / (steps - 1)) * i;
                        ctx.fillText(i.toString(), x, chartY + chartHeight + 15);
                    }
                    if ((steps - 1) % stepInterval !== 0) {
                        const x = chartX + chartWidth;
                        ctx.fillText((steps - 1).toString(), x, chartY + chartHeight + 15);
                    }
                    
                    ctx.font = "12px Arial";
                    ctx.fillStyle = "#ccc";
                    ctx.fillText("Steps", chartX + chartWidth / 2, chartY + chartHeight + 30);
                    
                    ctx.strokeStyle = "#4a9eff";
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    
                    for (let i = 0; i < steps; i++) {
                        const x = chartX + (chartWidth / (steps - 1)) * i;
                        const adjustedValue = this.adjustments[i] !== undefined ? this.adjustments[i] : this.sigmas_data[i];
                        const clampedValue = Math.max(0, Math.min(1, adjustedValue));
                        const y = chartY + chartHeight - (clampedValue * chartHeight);
                        
                        if (i === 0) {
                            ctx.moveTo(x, y);
                        } else {
                            ctx.lineTo(x, y);
                        }
                    }
                    ctx.stroke();
                    
                    for (let i = 0; i < steps; i++) {
                        const x = chartX + (chartWidth / (steps - 1)) * i;
                        const adjustedValue = this.adjustments[i] !== undefined ? this.adjustments[i] : this.sigmas_data[i];
                        const clampedValue = Math.max(0, Math.min(1, adjustedValue));
                        const y = chartY + chartHeight - (clampedValue * chartHeight);
                        
                        ctx.fillStyle = this.dragging_point === i ? "#ff6b6b" : "#4a9eff";
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, Math.PI * 2);
                        ctx.fill();
                    }
                    
                    ctx.fillStyle = "#ffffffff";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Sigmas_Editor", canvas.width / 2, 20);
                    
                    if (this.dragging_point !== -1) {
                        const orig = this.sigmas_data[this.dragging_point];
                        const adjusted = this.adjustments[this.dragging_point];
                        const multiplier = orig > 0 ? (adjusted / orig) : 1.0;
                        
                        const origCeil = this.ceilToFixed(orig, 4);
                        const adjustedCeil = this.ceilToFixed(adjusted, 4);
                        
                        ctx.fillStyle = "#ff6b6b";
                        ctx.textAlign = "center";
                        ctx.font = "11px Arial";
                        ctx.fillText(
                            `Step ${this.dragging_point}: Original=${origCeil.toFixed(4)}, Adjusted=${adjustedCeil.toFixed(4)}, Multiplier=${multiplier.toFixed(2)}x`,
                            canvas.width / 2,
                            35
                        );
                    }
                });
            };
            
            const onResize = nodeType.prototype.onResize;
            nodeType.prototype.onResize = function(size) {
                const result = onResize?.apply(this, arguments);
                
                if (this.canvas) {
                    this._canvasInitialized = false;
                    this.updateCanvas(true);
                }
                
                return result;
            };
            
            const onRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function() {
                const result = onRemoved?.apply(this, arguments);
                
                if (this && this.canvas) {
                    const ctx = this.canvas.getContext("2d");
                    if (ctx) {
                        ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
                    }
                    this.canvas = null;
                }
                if (this) {
                    this.canvasContainer = null;
                }
                
                if (this._sigmasEditorMessageHandler) {
                    api.removeEventListener("sigmas_editor_update", this._sigmasEditorMessageHandler);
                    this._sigmasEditorMessageHandler = null;
                }
                
                return result;
            };
        }
    }
});