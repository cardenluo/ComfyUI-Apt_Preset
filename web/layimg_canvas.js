import { app } from "../../../scripts/app.js";
import { $el } from "../../../scripts/ui.js";
import { api } from "../../../scripts/api.js";

class TransformBox {
    constructor(image, canvas, initialOpacity = 1.0) {
        this.image = image;
        this.canvas = canvas;
        this.rotation = 0;
        this.scaleX = 1;
        this.scaleY = 1;
        this.position = { x: 0, y: 0 };
        
        this.calculateSizes();
        
        this.originalWidth = image.width;
        this.originalHeight = image.height;
        
        this.initializeScale();
        this.opacity = initialOpacity;
    }
    
    calculateSizes() {
        const canvasWidth = this.canvas.width;
        
        this.borderWidth = Math.max(8, Math.round(canvasWidth * 0.05));
        
        this.handleSize = this.borderWidth * 2;
        
        this.rotateHandleDistance = this.handleSize * 4;
        
        this.hitTestArea = this.handleSize * 2.5;
    }
    
    draw(ctx) {
        ctx.save();
        
        ctx.translate(this.canvas.width/2 + this.position.x, 
                     this.canvas.height/2 + this.position.y);
        ctx.rotate(this.rotation);
        ctx.scale(this.scaleX, this.scaleY);
        
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)';
        ctx.lineWidth = this.borderWidth * 0.5;
        ctx.setLineDash([this.borderWidth, this.borderWidth]);
        
        const padding = this.handleSize;
        ctx.strokeRect(
            -this.image.width/2 - padding,
            -this.image.height/2 - padding,
            this.image.width + padding * 2,
            this.image.height + padding * 2
        );
        
        ctx.setLineDash([]);
        this.drawHandles(ctx);
        
        this.drawRotationIndicator(ctx);
        
        ctx.restore();
    }
    
    drawHandles(ctx) {
        const handles = this.getHandlePositions();
        
        handles.forEach(handle => {
            ctx.beginPath();
            ctx.arc(handle.x, handle.y, this.handleSize * 0.8, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fill();
            
            ctx.beginPath();
            ctx.arc(handle.x, handle.y, this.handleSize * 0.5, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(255, 255, 255, 0.95)';
            ctx.strokeStyle = 'rgba(0, 0, 0, 0.9)';
            ctx.lineWidth = this.borderWidth * 0.5;
            ctx.fill();
            ctx.stroke();
        });
    }
    
    drawRotationIndicator(ctx) {
        const centerY = -this.image.height/2 - this.handleSize;
        
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(0, centerY - this.rotateHandleDistance);
        ctx.strokeStyle = '#000000ff';
        ctx.lineWidth = this.borderWidth * 0.5;
        ctx.stroke();
        
        const rotateHandleSize = this.handleSize * 1.2;
        ctx.beginPath();
        ctx.arc(0, centerY - this.rotateHandleDistance, rotateHandleSize, 0, Math.PI * 2);
        ctx.strokeStyle = '#000000ff';
        ctx.stroke();
        
        ctx.beginPath();
        ctx.arc(0, centerY - this.rotateHandleDistance, rotateHandleSize * 0.8, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fill();
    }
    
    initializeScale() {
        const canvasRatio = this.canvas.width / this.canvas.height;
        const imageRatio = this.originalWidth / this.originalHeight;
        
        if (imageRatio > canvasRatio) {
            this.scaleX = this.scaleY = (this.canvas.width * 0.8) / this.originalWidth;
        } else {
            this.scaleX = this.scaleY = (this.canvas.height * 0.8) / this.originalHeight;
        }
    }
    
    getHandlePositions() {
        const w = this.image.width;
        const h = this.image.height;
        const padding = this.handleSize;
        
        return [
            {x: -w/2 - padding, y: -h/2 - padding, type: 'corner'},
            {x: w/2 + padding, y: -h/2 - padding, type: 'corner'},
            {x: w/2 + padding, y: h/2 + padding, type: 'corner'},
            {x: -w/2 - padding, y: h/2 + padding, type: 'corner'},
            
            {x: 0, y: -h/2 - padding, type: 'edge'},
            {x: w/2 + padding, y: 0, type: 'edge'},
            {x: 0, y: h/2 + padding, type: 'edge'},
            {x: -w/2 - padding, y: 0, type: 'edge'},
            
            {x: 0, y: -h/2 - padding - this.rotateHandleDistance, type: 'rotate'}
        ];
    }
    
    hitTest(x, y) {
        const centerX = this.canvas.width / 2;
        const centerY = this.canvas.height / 2;

        const relativeX = x - centerX - this.position.x;
        const relativeY = y - centerY - this.position.y;

        const cos = Math.cos(-this.rotation);
        const sin = Math.sin(-this.rotation);
        
        const rotatedX = relativeX * cos - relativeY * sin;
        const rotatedY = relativeX * sin + relativeY * cos;
        
        const scaledX = rotatedX / this.scaleX;
        const scaledY = rotatedY / this.scaleY;

        const handles = this.getHandlePositions();
        for(let handle of handles) {
            const dist = Math.sqrt(
                Math.pow(scaledX - handle.x, 2) + 
                Math.pow(scaledY - handle.y, 2)
            );
            
            if(dist <= this.hitTestArea) {
                switch(handle.type) {
                    case 'corner':
                        this.canvas.style.cursor = 'nw-resize';
                        break;
                    case 'rotate':
                        this.canvas.style.cursor = 'crosshair';
                        break;
                    default:
                        this.canvas.style.cursor = 'move';
                }
                return handle;
            }
            
            const padding = this.handleSize;
            const isInside = Math.abs(scaledX) <= (this.image.width/2 + padding) &&
                            Math.abs(scaledY) <= (this.image.height/2 + padding);
            
            if(isInside) {
                this.canvas.style.cursor = 'move';
            } else {
                this.canvas.style.cursor = 'default';
            }
        }
        
        return null;
    }
    
    transform(handle, dx, dy, shiftKey = false) {
        if (!handle) return;
        
        switch(handle.type) {
            case 'rotate':
                const centerX = this.canvas.width/2 + this.position.x;
                const centerY = this.canvas.height/2 + this.position.y;
                
                const mouseX = dx;
                const mouseY = dy;
                
                if (!this._rotateStart) {
                    this._rotateStart = {
                        angle: this.rotation,
                        startAngle: Math.atan2(mouseY - centerY, mouseX - centerX),
                        lastAngle: Math.atan2(mouseY - centerY, mouseX - centerX)
                    };
                    return;
                }
                
                const currentAngle = Math.atan2(mouseY - centerY, mouseX - centerX);
                
                let deltaAngle = currentAngle - this._rotateStart.lastAngle;
                
                if (deltaAngle > Math.PI) deltaAngle -= Math.PI * 2;
                if (deltaAngle < -Math.PI) deltaAngle += Math.PI * 2;
                
                this.rotation += deltaAngle;
                
                this._rotateStart.lastAngle = currentAngle;
                
                if (shiftKey) {
                    const snapAngle = Math.PI / 12; // 15度
                    this.rotation = Math.round(this.rotation / snapAngle) * snapAngle;
                }
                break;
                
            case 'corner':
                const scaleFactor = 0.005;
                const cornerX = handle.x;
                const cornerY = handle.y;
                const scaleDirectionX = Math.sign(cornerX);
                const scaleDirectionY = Math.sign(cornerY);
                
                let scaleX = 1 + (dx * scaleFactor * scaleDirectionX);
                let scaleY = 1 + (dy * scaleFactor * scaleDirectionY);
                
                if (shiftKey) {
                    const scale = Math.max(Math.abs(scaleX), Math.abs(scaleY));
                    scaleX = scaleY = scale;
                }
                
                this.scaleX = Math.max(0.0001, this.scaleX * scaleX);
                this.scaleY = Math.max(0.0001, this.scaleY * scaleY);
                break;
                
            default:
                this.position.x += dx;
                this.position.y += dy;
        }
    }
    
    clearRotation() {
        this._rotateStart = null;
        this._lastAngle = null;
    }
}

const dialog = {
    element: null,
    canvas: null,
    ctx: null,
    backImage: null,
    foreImage: null,
    isDragging: false,
    lastPoint: null,
    transformBox: null,

    show(node, backImagePath, foreImagePath, canvasWidth, canvasHeight, maskValue = 1.0) {
        if (this.element) {
            this.element.remove();
        }

        this.overlay = $el("div.canvas-overlay", {
            parent: document.body,
            style: {
                position: "fixed",
                top: 0,
                left: 0,
                right: 0,
                bottom: 0,
                background: "rgba(0, 0, 0, 0.7)", // 半透明黑色背景
                backdropFilter: "blur(5px)", // 背景模糊效果
                zIndex: 9999
            }
        });

        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;
        const margin = 30; // 增加边距
        const toolbarWidth = 280;
        
        const maxCanvasWidth = Math.min(
            viewportWidth * 0.95 - toolbarWidth - margin * 4, // 增加左右边距
            viewportWidth - toolbarWidth - margin * 4
        );
        const maxCanvasHeight = Math.min(
            viewportHeight * 0.95 - margin * 2,
            viewportHeight - margin * 2
        );
        
        const aspectRatio = canvasWidth / canvasHeight;
        let canvasDisplayWidth, canvasDisplayHeight;
        
        if (aspectRatio > maxCanvasWidth / maxCanvasHeight) {
            canvasDisplayWidth = maxCanvasWidth;
            canvasDisplayHeight = maxCanvasWidth / aspectRatio;
        } else {
            canvasDisplayHeight = maxCanvasHeight;
            canvasDisplayWidth = maxCanvasHeight * aspectRatio;
        }

        this.element = $el("div.web-canvas-dialog", {
            parent: document.body,
            style: {
                position: "fixed",
                top: "50%",
                left: "50%",
                transform: "translate(-50%, -50%)",
                width: `${viewportWidth - margin * 4}px`, 
                height: `${viewportHeight - margin * 2}px`,
                display: "flex",
                flexDirection: "column",
                gap: "10px",
                background: "#1a1a1a",
                borderRadius: "12px",
                zIndex: 10000,
                overflow: "hidden",
                boxShadow: "0 8px 32px rgba(0, 0, 0, 0.5)",
                border: "1px solid rgba(255, 255, 255, 0.1)"
            }
        }, [
            $el("div.properties-panel", {
                style: {
                    display: "flex",
                    alignItems: "center",
                    gap: "20px",
                    padding: "15px 20px",
                    background: "#222",
                    borderBottom: "1px solid #333"
                }
            }, [
                $el("div.size-info", {
                    style: {
                        display: "flex",
                        gap: "20px",
                        color: "#aaa",
                        fontSize: "13px"
                    }
                }, [
                    $el("span", {}, ["尺寸: ", $el("span.value", { style: { color: "#fff" } })]),
                    $el("span", {}, ["缩放: ", $el("span.scale-value", { style: { color: "#fff" } })]),
                    $el("span", {}, ["旋转: ", $el("span.rotation-value", { style: { color: "#fff" } })]),
                    $el("span", {}, ["透明度: ", $el("span.opacity-value", { style: { color: "#fff" } })])
                ])
            ]),
            
            $el("div.toolbar-title", {
                style: {
                    position: "absolute",
                    top: "15px",
                    right: "20px",
                    color: "#fff",
                    fontSize: "20px",
                    fontWeight: "800",
                    letterSpacing: "1.5px",
                    fontFamily: "'Segoe UI', Arial, sans-serif",
                    textTransform: "uppercase",
                    textShadow: "0 1px 2px rgba(0,0,0,0.3)",
                    zIndex: "1001"
                }
            }, ["图片摆放"]),
            
            $el("div.main-content", {
                style: {
                    display: "flex",
                    flex: 1,
                    gap: "20px", // 增加间距
                    padding: "0 20px 20px 20px" // 增加内边距
                }
            }, [
                $el("div.canvas-container", {
                    style: {
                        flex: 1,
                        position: "relative",
                        background: "repeating-conic-gradient(#404040 0% 25%, #303030 0% 50%) 50% / 20px 20px",
                        borderRadius: "8px",
                        overflow: "hidden"
                    }
                }, [
                    $el("canvas", {
                        id: "editor-canvas",
                        width: canvasWidth,
                        height: canvasHeight,
                        style: {
                            position: "absolute",
                            left: "50%",
                            top: "50%",
                            transform: "translate(-50%, -50%)",
                            maxWidth: "100%",
                            maxHeight: "100%"
                        }
                    }),
                ]),
                
                $el("div.toolbar", {
                    style: {
                        width: `${toolbarWidth}px`,
                        minWidth: `${toolbarWidth}px`, // 添加最小宽度
                        display: "flex",
                        flexDirection: "column",
                        gap: "15px",
                        background: "#222",
                        padding: "20px",
                        borderRadius: "8px",
                        overflowY: "auto" // 添加滚动条
                    }
                }, [
                    $el("div.tools-section", {
                        style: {
                            display: "flex",
                            flexDirection: "column",
                            gap: "12px"
                        }
                    }, [
                        this.createOpacityControl(1.0),
                        
                        $el("div.transform-tools", {
                            style: {
                                display: "flex",
                                flexDirection: "column",
                                gap: "8px"
                            }
                        }, [
                            $el("div.rotation-buttons", {
                                style: {
                                    display: "grid",
                                    gridTemplateColumns: "1fr 1fr", // 使用网格布局
                                    gap: "8px",
                                    width: "100%" // 确保充满容器宽度
                                }
                            }, [
                                this.createToolButton("左旋转", () => this.rotate(-90)),
                                this.createToolButton("右旋转", () => this.rotate(90))
                            ]),
                            
                            this.createScaleInput(),
                            
                            this.createToolButton("重置", () => this.resetTransform()),
                            $el("div", {
                                style: {
                                    display: "grid",
                                    gridTemplateColumns: "auto 1fr",
                                    gap: "8px 12px",
                                    lineHeight: "1.6",
                                    marginTop: "12px"
                                }
                            }, [
                                $el("span", { style: { color: "#aaa", backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["Alt + 滚轮"]),
                                $el("span", { style: { backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["旋转"]),
                                $el("span", { style: { color: "#aaa", backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["滚轮"]),
                                $el("span", { style: { backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["缩放"]),
                                $el("span", { style: { color: "#aaa", backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["拖拽"]),
                                $el("span", { style: { backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["等比缩放"]),
                                $el("span", { style: { color: "#aaa", backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["Shift + 拖拽"]),
                                $el("span", { style: { backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["自由变形"]),
                                $el("span", { style: { color: "#aaa", backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["Esc"]),
                                $el("span", { style: { backgroundColor: "rgba(0, 0, 0, 0.1)" } }, ["取消"])
                            ])
                        ])
                    ]),
                    
                    $el("div.bottom-buttons", {
                        style: {
                            marginTop: "auto",
                            display: "flex",
                            flexDirection: "column",
                            gap: "10px"
                        }
                    }, [
                        this.createButton("确认", () => this.save(), "#4CAF50"),
                        this.createButton("取消", () => this.cancel())
                    ])
                ])
            ])
        ]);

        this.canvas = document.getElementById("editor-canvas");
        this.ctx = this.canvas.getContext("2d");
        
        this.loadImages(backImagePath, foreImagePath, 1.0);
        
        if(this.transformBox) {
            this.transformBox.originalMaskValue = maskValue;
        }
        
        this.setupEvents();
    },
    
    createToolButton(text, onClick) {
        return $el("button", {
            onclick: onClick,
            style: {
                padding: "8px 12px",
                width: "100%", // 让按钮填满容器
                background: "#333",
                border: "none",
                borderRadius: "4px",
                color: "#fff",
                cursor: "pointer",
                fontSize: "13px",
                transition: "background 0.2s",
                textAlign: "center",
                ":hover": {
                    background: "#444"
                }
            }
        }, [text]);
    },
    
    createScaleInput() {
        return $el("div.scale-control", {
            style: {
                display: "flex",
                alignItems: "center",
                gap: "8px",
                background: "#333",
                padding: "8px 12px",
                borderRadius: "4px",
                width: "100%"
            }
        }, [
            $el("span", {
                style: {
                    color: "#fff",
                    fontSize: "13px",
                    whiteSpace: "nowrap"
                }
            }, ["缩放:"]),
            $el("input", {
                type: "number",
                min: "0.01", // 最小值改为0.01
                // 移除max属性，允许任意大的数值
                value: "100",
                style: {
                    width: "60px", // 增加输入框宽度以适应更大的数字
                    padding: "4px",
                    background: "#444",
                    border: "1px solid #555",
                    borderRadius: "3px",
                    color: "#fff",
                    fontSize: "13px"
                },
                oninput: (e) => this.setScale(e.target.value / 100)
            })
        ]);
    },
    
    createButton(text, onClick, bgColor = "#333") {
        return $el("button", {
            onclick: onClick,
            style: {
                padding: "8px 25px",
                background: bgColor,
                border: "none",
                borderRadius: "4px",
                color: "#fff",
                cursor: "pointer",
                fontSize: "14px",
                transition: "background 0.2s"
            }
        }, [text]);
    },

    loadImages(backPath, forePath, maskValue = 1.0) {
        this.backImage = new Image();
        this.backImage.crossOrigin = "anonymous";
        this.backImage.onload = () => {
            this.foreImage = new Image();
            this.foreImage.crossOrigin = "anonymous";
            this.foreImage.onload = () => {
                this.transformBox = new TransformBox(this.foreImage, this.canvas, maskValue);
                this.render();
                
                const opacitySlider = this.element.querySelector('.opacity-control input[type="range"]');
                const opacityValue = this.element.querySelector('.opacity-value');
                if (opacitySlider && opacityValue) {
                    const percentage = Math.round(maskValue * 100);
                    opacitySlider.value = percentage;
                    opacityValue.textContent = percentage + '%';
                    opacitySlider.style.background = `linear-gradient(to right, #4CAF50 ${percentage}%, #333 ${percentage}%)`;
                }
            };
            this.foreImage.src = forePath;
        };
        this.backImage.src = backPath;
    },

    render() {
        if (!this.ctx || !this.backImage || !this.foreImage) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.drawImage(this.backImage, 0, 0, this.canvas.width, this.canvas.height);
        
        this.ctx.save();
        
        const transform = this.transformBox;
        this.ctx.translate(this.canvas.width/2 + transform.position.x, 
                          this.canvas.height/2 + transform.position.y);
        this.ctx.rotate(transform.rotation);
        this.ctx.scale(transform.scaleX, transform.scaleY);
        
        this.ctx.globalAlpha = transform.opacity;
        this.ctx.globalCompositeOperation = 'source-over';
        
        this.ctx.drawImage(
            this.foreImage,
            -this.foreImage.width/2,
            -this.foreImage.height/2
        );
        
        this.ctx.restore();
        
        this.transformBox.draw(this.ctx);
        
        this.updateInfo();
    },
    
    setupEvents() {
        let activeHandle = null;
        let startX, startY;
        let isShiftPressed = false;
        
        document.addEventListener('keydown', (e) => {
            if(e.key === 'Shift') {
                isShiftPressed = true;
            } else if(e.key === 'Escape') {
                this.cancel();
            }
        });
        
        document.addEventListener('keyup', (e) => {
            if(e.key === 'Shift') {
                isShiftPressed = false;
            }
        });

        this.canvas.addEventListener('mousedown', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
            const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
            
            activeHandle = this.transformBox.hitTest(x, y);
            
            if(activeHandle) {
                startX = x;
                startY = y;
                switch(activeHandle.type) {
                    case 'rotate':
                        this.canvas.style.cursor = 'crosshair';
                        break;
                    case 'corner':
                        this.canvas.style.cursor = 'nw-resize';
                        break;
                    default:
                        this.canvas.style.cursor = 'move';
                }
            } else {
                this.isDragging = true;
                this.lastPoint = { x, y };
                this.canvas.style.cursor = 'grabbing';
            }
        });

        document.addEventListener('mousemove', (e) => {
            if(!activeHandle && !this.isDragging) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
            const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
            
            if(activeHandle) {
                const dx = x - startX;
                const dy = y - startY;
                
                this.transformBox.transform(activeHandle, dx, dy, !isShiftPressed);
                this.render();
                
                if(activeHandle.type !== 'rotate') {
                    startX = x;
                    startY = y;
                }
            } else if(this.isDragging) {
                const dx = x - this.lastPoint.x;
                const dy = y - this.lastPoint.y;
                
                this.transformBox.position.x += dx;
                this.transformBox.position.y += dy;
                
                this.lastPoint = { x, y };
                this.render();
            }
        });

        document.addEventListener('mouseup', () => {
            if(this.transformBox) {
                this.transformBox.clearRotation();
            }
            activeHandle = null;
            this.isDragging = false;
            this.canvas.style.cursor = 'default';
        });

        this.canvas.addEventListener('wheel', (e) => {
            e.preventDefault();
            
            if (e.altKey) {
                const rotationDelta = e.deltaY > 0 ? -5 : 5;
                this.rotate(rotationDelta);
            } else {
                const scaleDelta = e.deltaY > 0 ? 0.95 : 1.05;
                const newScale = this.transformBox.scaleX * scaleDelta;
                this.setScale(newScale);
            }
        }, { passive: false });

        this.canvas.addEventListener('mousemove', (e) => {
            if (!this.transformBox) return;

            const rect = this.canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
            const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
            
            const handle = this.transformBox.hitTest(x, y);
            
            if (handle) {
                switch (handle.type) {
                    case 'corner':
                        this.canvas.style.cursor = 'nw-resize';
                        break;
                    case 'edge':
                        this.canvas.style.cursor = handle.x === 0 ? 'ns-resize' : 'ew-resize';
                        break;
                    case 'rotate':
                        this.canvas.style.cursor = 'crosshair';
                        break;
                }
            } else {
                this.canvas.style.cursor = this.isDragging ? 'grabbing' : 'grab';
            }
        });
    },
    
    updateInfo() {
        const transform = this.transformBox;
        const scalePercent = Math.round(transform.scaleX * 100);
        const rotation = Math.round((transform.rotation * 180 / Math.PI) % 360);
        const opacity = Math.round(transform.opacity * 100);
        
        const sizeInfo = this.element.querySelector('.size-info');
        if (sizeInfo) {
            sizeInfo.children[0].querySelector('.value').textContent = 
                `${this.canvas.width} × ${this.canvas.height}`;
            sizeInfo.children[1].querySelector('.scale-value').textContent = 
                `${scalePercent}%`;
            sizeInfo.children[2].querySelector('.rotation-value').textContent = 
                `${rotation}°`;
            sizeInfo.children[3].querySelector('.opacity-value').textContent = 
                `${opacity}%`;
        }
    },
    
    rotate(angle) {
        this.transformBox.rotation += angle * Math.PI / 180;
        this.render();
    },
    
    resetTransform() {
        this.transformBox.position = { x: 0, y: 0 };
        this.transformBox.rotation = 0;
        this.transformBox.scaleX = 1;
        this.transformBox.scaleY = 1;
        this.transformBox.initializeScale();
        this.render();
    },
    
    setScale(scale) {
        this.transformBox.scaleX = this.transformBox.scaleY = 
            Math.max(0.0001, scale); // 只保留最小值限制
        this.render();
    },
    
    generateMask() {
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = this.canvas.width;
        maskCanvas.height = this.canvas.height;
        const maskCtx = maskCanvas.getContext('2d');
        
        maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        
        maskCtx.save();
        const transform = this.transformBox;
        
        maskCtx.translate(maskCanvas.width/2 + transform.position.x,
                         maskCanvas.height/2 + transform.position.y);
        maskCtx.rotate(transform.rotation);
        maskCtx.scale(transform.scaleX, transform.scaleY);
        
        const width = this.foreImage.width;
        const height = this.foreImage.height;
        
        maskCtx.drawImage(this.foreImage,
                         -width/2, -height/2,
                         width, height);
        
        maskCtx.restore();
        
        const imageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
        const data = imageData.data;
        
        const finalMaskCanvas = document.createElement('canvas');
        finalMaskCanvas.width = maskCanvas.width;
        finalMaskCanvas.height = maskCanvas.height;
        const finalMaskCtx = finalMaskCanvas.getContext('2d');
        
        const newImageData = finalMaskCtx.createImageData(maskCanvas.width, maskCanvas.height);
        const newData = newImageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            const alpha = data[i + 3];
            newData[i] = alpha;     // R
            newData[i + 1] = alpha; // G
            newData[i + 2] = alpha; // B
            newData[i + 3] = 255;   // A
        }
        
        finalMaskCtx.putImageData(newImageData, 0, 0);
        return finalMaskCanvas;
    },

    async save() {
        try {
            const transform = this.transformBox;
            
            const finalCanvas = document.createElement('canvas');
            finalCanvas.width = this.canvas.width;
            finalCanvas.height = this.canvas.height;
            const ctx = finalCanvas.getContext('2d');
            
            ctx.drawImage(this.backImage, 0, 0, finalCanvas.width, finalCanvas.height);
            
            ctx.save();
            
            ctx.translate(finalCanvas.width/2 + transform.position.x,
                         finalCanvas.height/2 + transform.position.y);
            ctx.rotate(transform.rotation);
            ctx.scale(transform.scaleX, transform.scaleY);
            
            ctx.globalAlpha = transform.opacity;
            
            ctx.drawImage(this.foreImage, 
                         -this.foreImage.width/2,
                         -this.foreImage.height/2);
            
            ctx.restore();
            
            const maskCanvas = document.createElement('canvas');
            maskCanvas.width = finalCanvas.width;
            maskCanvas.height = finalCanvas.height;
            const maskCtx = maskCanvas.getContext('2d');
            
            maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
            
            maskCtx.save();
            maskCtx.translate(maskCanvas.width/2 + transform.position.x,
                             maskCanvas.height/2 + transform.position.y);
            maskCtx.rotate(transform.rotation);
            maskCtx.scale(transform.scaleX, transform.scaleY);
            
            maskCtx.globalAlpha = transform.opacity;
            
            maskCtx.drawImage(this.foreImage, 
                             -this.foreImage.width/2,
                             -this.foreImage.height/2);
            maskCtx.restore();
            
            const maskImageData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
            const maskPixelData = maskImageData.data;
            
            const grayMaskImageData = maskCtx.createImageData(maskCanvas.width, maskCanvas.height);
            const grayMaskData = grayMaskImageData.data;
            
            const originalMaskValue = this.transformBox.originalMaskValue || 1.0;
            
            for (let i = 0; i < maskPixelData.length; i += 4) {
                const alpha = (maskPixelData[i + 3] / 255.0) * originalMaskValue;
                
                const grayValue = Math.round(alpha * 255);
                
                grayMaskData[i] = grayValue;     // R
                grayMaskData[i + 1] = grayValue; // G
                grayMaskData[i + 2] = grayValue; // B
                grayMaskData[i + 3] = 255;       // A
            }
            
            maskCtx.putImageData(grayMaskImageData, 0, 0);
            
            const finalImageData = finalCanvas.toDataURL('image/png');
            const maskData = maskCanvas.toDataURL('image/png');
            
            const transformData = {
                image: finalImageData,
                mask: maskData,
                confirmed: true
            };
            
            const response = await api.fetchApi('/preset_canvas', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(transformData)
            });
            
            if (!response.ok) {
                throw new Error(`Save failed: ${response.status}`);
            }
        } catch (error) {
        } finally {
            this.hide();
        }
    },
    
    async cancel() {
        try {
            const response = await api.fetchApi('/preset_canvas', {
                method: 'POST',
                body: JSON.stringify({
            
                    confirmed: false
                })
            });
            
            if (!response.ok) {
                throw new Error(`Cancel failed: ${response.status}`);
            }
        } catch (error) {
        } finally {
            this.hide();
        }
    },
    
    hide() {
        if (this.overlay) {
            this.overlay.remove();
            this.overlay = null;
        }
        if (this.element) {
            this.element.remove();
            this.element = null;
        }
    },
    
    createOpacityControl(initialValue = 1.0) {
        const percentage = 100;
        return $el("div.opacity-control", {
            style: {
                display: "flex",
                flexDirection: "column",
                gap: "8px",
                padding: "10px",
                background: "#333",
                borderRadius: "4px"
            }
        }, [
            $el("div", {
                style: {
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center"
                }
            }, [
                $el("span", {
                    style: {
                        color: "#fff",
                        fontSize: "14px"
                    }
                }, ["透明度"]),
                $el("span.opacity-value", {
                    style: {
                        color: "#fff",
                        fontSize: "14px",
                        minWidth: "40px",
                        textAlign: "right"
                    }
                }, [`${percentage}%`])
            ]),
            $el("input", {
                type: "range",
                min: "0",
                max: "100",
                value: percentage, // 固定为100
                style: {
                    width: "100%",
                    height: "20px",
                    WebkitAppearance: "none",
                    background: `linear-gradient(to right, #4CAF50 ${percentage}%, #333 ${percentage}%)`,
                    borderRadius: "10px",
                    outline: "none",
                    opacity: "0.9",
                    transition: "opacity 0.2s",
                    cursor: "pointer",
                    ":hover": {
                        opacity: "1"
                    },
                    "::-webkit-slider-thumb": {
                        WebkitAppearance: "none",
                        appearance: "none",
                        width: "18px",
                        height: "18px",
                        background: "#fff",
                        borderRadius: "50%",
                        cursor: "pointer"
                    },
                    "::-moz-range-thumb": {
                        width: "18px",
                        height: "18px",
                        background: "#fff",
                        borderRadius: "50%",
                        cursor: "pointer"
                    }
                },
                oninput: (e) => {
                    const value = e.target.value / 100;
                    this.setOpacity(value);
                    e.target.style.background = `linear-gradient(to right, #4CAF50 ${e.target.value}%, #333 ${e.target.value}%)`;
                }
            })
        ]);
    },
    
    setOpacity(value) {
        if (this.transformBox) {
            this.transformBox.opacity = value;
            const opacityValue = this.element.querySelector('.opacity-value');
            if (opacityValue) {
                opacityValue.textContent = Math.round(value * 100) + '%';
            }
            const slider = this.element.querySelector('.opacity-control input[type="range"]');
            if (slider) {
                slider.style.background = `linear-gradient(to right, #4CAF50 ${value * 100}%, #333 ${value * 100}%)`;
            }
            this.render();
        }
    },
    

};

app.registerExtension({
    name: "comfy.canvas.dialog",
    async setup() {
        api.addEventListener("show_canvas", async ({detail}) => {
            console.log("Received show_canvas event:", detail);
            try {
                const maskValue = detail.mask_value || 1.0; // 如果后端没有传递，默认为1.0
                const prompt = await app.graphToPrompt();
                const res = prompt.workflow.nodes;
                console.log("res:", res);
                let haslay_imgCanvasNode = false;
                let seednumber = [];
                for (const node of res) {
                    if (node.type === "lay_imgCanvas") {
                        console.log("Found lay_imgCanvas node");
                        haslay_imgCanvasNode = true;
                        if (node.widgets_values[1]) {
                            seednumber.push(node.widgets_values[1])
                        }
                    }
                }
                console.log("detail_window_seed:", detail.window_id);
                if (haslay_imgCanvasNode && seednumber.length > 0) {
                    let is_catch_window_id = seednumber.includes(detail.window_id);
                    console.log("seednumber:", seednumber);
                    console.log("is_catch_window_id:", is_catch_window_id);
                    
                    if (!is_catch_window_id) {
                        console.log('Window ID not found in seednumber array, skipping dialog');
                        return;
                    }
                    
                    if (detail.layers) {
                        dialogPro.layers = detail.layers;
                        dialogPro.layerImages = detail.layers.map(layer => {
                            const img = new Image();
                            img.crossOrigin = "anonymous";
                            img.src = layer.url;
                            return img;
                        });
                    }
                    
                    dialog.show(
                        null,
                        detail.back_image,
                        detail.fore_image,
                        detail.canvas_width,
                        detail.canvas_height,
                        maskValue,
                    );
                } else {
                    console.log('No lay_imgCanvasNodePro node found, skipping dialog');
                }
            } catch (error) {
                console.error('Error getting graph output:', error);
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {  
        if (nodeData.name === "lay_imgCanvas" ) {
            nodeType.prototype.onNodeCreated = function() {
                const randomize = function() {
                    const randomNumber = Math.floor(Math.random() * 9000000000000000) + 1000000000000000;
                    return randomNumber;
                };
                
                let seed = this.addWidget("number", "windows_seed", randomize(), { 
                    min: 1000000000000000,
                    max: 9999999999999999,
                    precision: 0,
                    callback: (value) => {
                        this.properties = this.properties || {};
                        this.properties.windows_seed = value;
                    }
                });
                
                seed.value = randomize();
                
                this.properties = this.properties || {};
                this.properties.windows_seed = seed.value;
                // seed.hidden = true;
            };
        }
    }
}); 