// author.yichengup.Loadimage_brushmask 2025.01.XX
import { app } from "../../scripts/app.js";

const DEFAULT_LAYOUT = {
    shiftLeft: 80, // 左侧工具栏固定宽度80px
    shiftRight: 80, // 右侧端口区域固定宽度80px
    panelHeight: 0
};

const WIDGET_NAMES = {
    BRUSH_DATA: "brush_data",
    BRUSH_SIZE: "brush_size",
    IMAGE_WIDTH: "image_width",
    IMAGE_HEIGHT: "image_height",
    IMAGE_BASE64: "image_base64"
};

class IO_EasyMark {
    constructor(node) {
        this.node = node;
        this.state = createInitialState(node);
        initUIBindings(node, this.state);
        initInteractionBindings(node, this.state);
    }
}

function createInitialState(node) {
    if (!node.properties) {
        node.properties = {};
    }

    const defaults = {
        brushPaths: [],
        isDrawing: false,
        currentPath: [],
        brushSize: 4,
        brushOpacity: 1.0,
        brushMode: "brush",
        brushType: "free",
        brushColor: "255,0,0",
        eraserColor: "255,50,50",
        backgroundImage: null,
        imageWidth: 512,
        imageHeight: 512,
        buttons: [],
        inputs: [],
        colorPalette: null,
        markerPalette: null,
        activeMarker: null,
        backgroundImageObj: null,
        imageBase64Data: "",
        brushTypeButtons: null,
        actionButtons: null
    };

    node.properties = {
        ...defaults,
        ...node.properties
    };

    node.size = node.size || [500, 500];

    return {
        layout: { ...DEFAULT_LAYOUT },
        fontSize: LiteGraph?.NODE_SUBTEXT_SIZE ?? 10
    };
}

function initUIBindings(node, state) {
    const { shiftLeft, shiftRight, panelHeight } = state.layout;
    const fontsize = state.fontSize;

    setupHiddenWidgets(node);

    const originalOnResize = node.onResize;
    node.onResize = function () {
        if (originalOnResize) {
            originalOnResize.apply(this, arguments);
        }
        if (this.min_size) {
            this.size[0] = Math.max(this.size[0], this.min_size[0] || 0);
            this.size[1] = Math.max(this.size[1], this.min_size[1] || 0);
        }
    };

    node.initButtons = function () {
        if (this.widgets[1] && this.widgets[1].value) {
            this.properties.brushSize = this.widgets[1].value || 10;
        }

        // 重新排版：左侧工具栏固定宽度，深色主题
        const buttonX = 12;
        let buttonY = 10;
        const buttonWidth = 56;
        const buttonHeight = 16;
        const buttonSpacing = 4;

        // 1. 图像加载模块
        this.properties.buttons = [
            {
                text: "加载",
                label: "加载图片",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                height: buttonHeight,
                action: () => this.loadImageFromFile()
            }
        ];

        // 2. 画笔类型选择（画笔、方框、色块互斥）
        buttonY += buttonHeight + buttonSpacing + 10;
        this.properties.brushTypeButtons = {
            free: {
                text: "画笔",
                label: "画笔",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                height: buttonHeight,
                isToggle: true,
                active: true,
                action: () => {
                    this.properties.brushMode = "brush";
                    this.properties.brushType = "free";
                    this.properties.activeMarker = null;
                    this.properties.brushTypeButtons.free.active = true;
                    this.properties.brushTypeButtons.box.active = false;
                    this.properties.brushTypeButtons.square.active = false;
                    this.updateThisNodeGraph?.();
                }
            },
            box: {
                text: "方框",
                label: "方框",
                x: buttonX,
                y: buttonY + buttonHeight + buttonSpacing,
                width: buttonWidth,
                height: buttonHeight,
                isToggle: true,
                active: false,
                action: () => {
                    this.properties.brushMode = "brush";
                    this.properties.brushType = "box";
                    this.properties.activeMarker = null;
                    this.properties.brushTypeButtons.free.active = false;
                    this.properties.brushTypeButtons.box.active = true;
                    this.properties.brushTypeButtons.square.active = false;
                    this.updateThisNodeGraph?.();
                }
            },
            square: {
                text: "色块",
                label: "色块",
                x: buttonX,
                y: buttonY + (buttonHeight + buttonSpacing) * 2,
                width: buttonWidth,
                height: buttonHeight,
                isToggle: true,
                active: false,
                action: () => {
                    this.properties.brushMode = "brush";
                    this.properties.brushType = "square";
                    this.properties.activeMarker = null;
                    this.properties.brushTypeButtons.free.active = false;
                    this.properties.brushTypeButtons.box.active = false;
                    this.properties.brushTypeButtons.square.active = true;
                    this.updateThisNodeGraph?.();
                }
            }
        };

        // 3. 核心操作按钮
        buttonY += (buttonHeight + buttonSpacing) * 3 + 10;
        this.properties.actionButtons = [
            {
                text: "清除",
                label: "清除",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                height: buttonHeight,
                action: () => {
                    this.properties.brushPaths = [];
                    this.properties.currentPath = [];
                    this.updateThisNodeGraph?.();
                }
            },
            {
                text: "撤销",
                label: "撤销",
                x: buttonX,
                y: buttonY + buttonHeight + buttonSpacing,
                width: buttonWidth,
                height: buttonHeight,
                action: () => {
                    if (this.properties.brushPaths.length > 0) {
                        this.properties.brushPaths.pop();
                        this.updateThisNodeGraph?.();
                    }
                }
            }
        ];

        // 4. 参数调节滑块
        buttonY += (buttonHeight + buttonSpacing) * 2 + 10;
        const sliderHeight = 8;
        const labelHeight = 12;
        const sliderSpacing = 8;
        this.properties.sliders = [
            {
                label: "大小",
                x: buttonX,
                y: buttonY,
                width: buttonWidth,
                labelHeight: labelHeight,
                sliderHeight: sliderHeight,
                type: "size",
                min: 1,
                max: 100,
                value: this.properties.brushSize || 10,
                isDragging: false
            },
            {
                label: "透明度",
                x: buttonX,
                y: buttonY + labelHeight + sliderHeight + sliderSpacing,
                width: buttonWidth,
                labelHeight: labelHeight,
                sliderHeight: sliderHeight,
                type: "opacity",
                min: 10,
                max: 100,
                value: Math.round((this.properties.brushOpacity || 1.0) * 100),
                isDragging: false
            }
        ];

        // 5. 颜色选择器（下拉菜单）
        buttonY += (labelHeight + sliderHeight + sliderSpacing) * 2 + 10;
        const colorLabelHeight = 14;
        const swatchSize = 16;
        const swatchGap = 4;
        const swatchRows = 2;
        const swatchCols = 3;
        const swatchGridWidth = swatchCols * swatchSize + (swatchCols - 1) * swatchGap;
        this.properties.colorPalette = {
            label: "颜色",
            x: buttonX,
            y: buttonY,
            width: swatchGridWidth,
            labelHeight: colorLabelHeight,
            swatchSize,
            swatchGap,
            rows: swatchRows,
            cols: swatchCols,
            options: [
                { label: "Black", text: "黑", value: "0,0,0", textColor: "#ffffff" },
                { label: "White", text: "白", value: "255,255,255", textColor: "#000000" },
                { label: "Red", text: "红", value: "255,0,0", textColor: "#ffffff" },
                { label: "Green", text: "绿", value: "0,255,0", textColor: "#000000" },
                { label: "Blue", text: "蓝", value: "0,0,255", textColor: "#ffffff" },
                { label: "Gray", text: "灰", value: "128,128,128", textColor: "#000000" }
            ]
        };

        const markerGridY = buttonY + colorLabelHeight + swatchRows * swatchSize + (swatchRows - 1) * swatchGap + 6;
        const markerSize = swatchSize;
        const markerGap = swatchGap;
        const markerRows = 2;
        const markerCols = 3;
        const markerGridWidth = markerCols * markerSize + (markerCols - 1) * markerGap;
        this.properties.markerPalette = {
            label: "标记",
            x: buttonX,
            y: markerGridY,
            labelHeight: colorLabelHeight,
            markerSize,
            markerGap,
            rows: markerRows,
            cols: markerCols,
            width: markerGridWidth,
            options: [
                { text: "1", value: "1" },
                { text: "2", value: "2" },
                { text: "3", value: "3" },
                { text: "4", value: "4" },
                { text: "5", value: "5" },
                { text: "6", value: "6" }
            ]
        };

        const markerGridHeight = markerRows * markerSize + (markerRows - 1) * markerGap;
        const minHeight = Math.max(320, markerGridY + this.properties.markerPalette.labelHeight + markerGridHeight + 8);
        if (!this.min_size) {
            this.min_size = [300, minHeight];
        } else {
            this.min_size[1] = Math.max(this.min_size[1] || 0, minHeight);
        }
        if (this.size[1] < minHeight) {
            this.size[1] = minHeight;
        }
    };









    node.onAdded = function () {
        this.initButtons?.();
    };

    node.onConfigure = function () {
        const widthWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
        const heightWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);

        if (widthWidget && heightWidget && widthWidget.value && heightWidget.value) {
            this.updateImageSize(widthWidget.value, heightWidget.value);
        } else {
            this.updateImageSize(512, 512);
        }

        const brushSizeWidget = this.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_SIZE);
        this.properties.brushSize = brushSizeWidget && brushSizeWidget.value !== undefined
            ? brushSizeWidget.value
            : 10;

        if (this.properties.brushOpacity === undefined) {
            this.properties.brushOpacity = 1.0;
        }
        if (this.properties.brushColor === undefined) {
            this.properties.brushColor = "255,0,0";
        }
        if (this.properties.brushType === undefined) {
            this.properties.brushType = "free";
        }
        if (this.properties.eraserColor === undefined) {
            this.properties.eraserColor = "255,50,50";
        }

        if (this.properties.sliders) {
            for (const slider of this.properties.sliders) {
                if (slider.type === "size") {
                    slider.value = this.properties.brushSize;
                } else if (slider.type === "opacity") {
                    slider.value = Math.round((this.properties.brushOpacity || 1.0) * 100);
                }
            }
        }

        if (this.properties.colorPalette && this.properties.colorPalette.options) {
            const colorValue = this.properties.brushColor || "255,0,0";
            const match = this.properties.colorPalette.options.find(o => o.value === colorValue);
            if (!match) {
                this.properties.brushColor = "255,0,0";
            }
        }
        if (this.properties.markerPalette && this.properties.activeMarker) {
            const match = this.properties.markerPalette.options?.some(o => o.value === this.properties.activeMarker);
            if (!match) {
                this.properties.activeMarker = null;
            }
        }

        const imageBase64Widget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);
        if (imageBase64Widget && imageBase64Widget.value) {
            this.properties.imageBase64Data = imageBase64Widget.value;
            this.loadBackgroundImageFromBase64(imageBase64Widget.value);
        } else if (this.properties.imageBase64Data) {
            this.loadBackgroundImageFromBase64(this.properties.imageBase64Data);
        }

        this.properties.brushPaths = [];
        const brushDataWidget = this.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_DATA) || this.widgets[2];
        if (brushDataWidget && brushDataWidget.value) {
            try {
                const brushData = brushDataWidget.value;
                if (brushData && brushData.trim()) {
                    const strokes = brushData.split("|");
                    for (const stroke of strokes) {
                        if (!stroke.trim()) continue;
                        const parsed = parseStroke(stroke);
                        if (parsed.points.length > 0) {
                            this.properties.brushPaths.push(parsed);
                        }
                    }
                }
            } catch (e) {
                console.error("Error parsing brush data:", e);
                this.properties.brushPaths = [];
            }
        }

        this.initButtons?.();
    };

    node.updateImageSize = function (width, height) {
        if (!width || !height || width <= 0 || height <= 0) {
            return;
        }

        this.properties.imageWidth = width;
        this.properties.imageHeight = height;

        const widthWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
        const heightWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);
        if (widthWidget) widthWidget.value = width;
        if (heightWidget) heightWidget.value = height;

        const maxDisplaySize = 500;
        const scale = Math.min(
            maxDisplaySize / width,
            maxDisplaySize / height,
            1.0
        );

        const displayWidth = Math.max(300, Math.min(width * scale + shiftRight + shiftLeft, 800));
        const displayHeight = Math.max(420, this.min_size?.[1] || 0);

        this.size[0] = displayWidth;
        this.size[1] = displayHeight;

        this.updateThisNodeGraph?.();
    };

    node.onDrawForeground = function (ctx) {
        if (this.flags.collapsed) {
            return false;
        }
        ctx.save();
        try {

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;

        // 绘制左侧工具栏背景（深色主题）
        ctx.fillStyle = "#2a2a2a";
        ctx.fillRect(0, 0, shiftLeft, this.size[1]);

        // 绘制右侧画布区域
        let canvasAreaWidth = this.size[0] - shiftLeft - shiftRight;
        let canvasAreaHeight = this.size[1] - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        // 绘制画布容器背景（浅灰）
        ctx.fillStyle = "#3a3a3a";
        ctx.fillRect(shiftLeft, 0, canvasAreaWidth, this.size[1]);

        // 绘制网格背景（先绘制网格，再绘制图片，这样图片在网格上方）
        ctx.fillStyle = "rgba(100,100,100,0.3)";
        ctx.fillRect(offsetX, offsetY, scaledWidth, scaledHeight);
        ctx.strokeStyle = "rgba(150,150,150,0.3)";
        ctx.lineWidth = 1;
        const gridSize = 20;
        const gridScale = gridSize * scale;

        for (let x = offsetX; x <= offsetX + scaledWidth; x += gridScale) {
            ctx.beginPath();
            ctx.moveTo(x, offsetY);
            ctx.lineTo(x, offsetY + scaledHeight);
            ctx.stroke();
        }

        for (let y = offsetY; y <= offsetY + scaledHeight; y += gridScale) {
            ctx.beginPath();
            ctx.moveTo(offsetX, y);
            ctx.lineTo(offsetX + scaledWidth, y);
            ctx.stroke();
        }

        // 绘制背景图像（不透明，绘制在网格上方）
        if (this.properties.backgroundImageObj && this.properties.backgroundImageObj.complete) {
            try {
                ctx.globalAlpha = 1.0;
                ctx.drawImage(
                    this.properties.backgroundImageObj,
                    offsetX,
                    offsetY,
                    scaledWidth,
                    scaledHeight
                );
                ctx.globalAlpha = 1.0;
            } catch (e) {
                console.error("Error drawing background image:", e);
            }
        }

        // 绘制画笔路径
        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        // 保存原始的裁剪区域
        ctx.save();
        // 设置裁剪区域为有效画布范围
        ctx.rect(offsetX, offsetY, scaledWidth, scaledHeight);
        ctx.clip();

        for (const pathObj of this.properties.brushPaths) {
            const path = pathObj.points || pathObj;
            const mode = pathObj.mode || "brush";
            const type = pathObj.type || "free";
            const pathSize = pathObj.size !== undefined ? pathObj.size : this.properties.brushSize;
            const pathOpacity = pathObj.opacity !== undefined ? pathObj.opacity : this.properties.brushOpacity;
            const pathColor = pathObj.color || this.properties.brushColor || "255,0,0";

            if (path.length < 2) continue;

            if (type === "square") {
                // Square画笔：绘制实心方块
                const isMarker = !!pathObj.marker;
                const fillColor = isMarker ? "255,255,0" : pathColor;
                const rgb = fillColor.split(",").map(c => parseInt(c.trim()));
                ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pathOpacity})`;
                
                const minX = Math.min(...path.map(p => p.x));
                const maxX = Math.max(...path.map(p => p.x));
                const minY = Math.min(...path.map(p => p.y));
                const maxY = Math.max(...path.map(p => p.y));
                
                ctx.fillRect(offsetX + minX * scale, offsetY + minY * scale,
                          (maxX - minX) * scale, (maxY - minY) * scale);

                if (pathObj.marker) {
                    const centerX = offsetX + (minX + maxX) * 0.5 * scale;
                    const centerY = offsetY + (minY + maxY) * 0.5 * scale;
                    const sideScaled = Math.min((maxX - minX) * scale, (maxY - minY) * scale);
                    const fontSize = Math.max(10, Math.min(200, sideScaled * 0.6));
                    ctx.fillStyle = "#000000";
                    ctx.font = `bold ${Math.floor(fontSize)}px Arial`;
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(String(pathObj.marker), centerX, centerY);
                }
            } else if (type === "box") {
                // Box画笔：绘制空心矩形框
                const rgb = pathColor.split(",").map(c => parseInt(c.trim()));
                ctx.lineWidth = pathSize * scale;
                ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pathOpacity})`;
                
                const minX = Math.min(...path.map(p => p.x));
                const maxX = Math.max(...path.map(p => p.x));
                const minY = Math.min(...path.map(p => p.y));
                const maxY = Math.max(...path.map(p => p.y));
                
                ctx.strokeRect(offsetX + minX * scale, offsetY + minY * scale,
                             (maxX - minX) * scale, (maxY - minY) * scale);
            } else {
                // Free画笔：绘制连续路径
                const rgb = pathColor.split(",").map(c => parseInt(c.trim()));
                ctx.lineWidth = pathSize * scale;
                ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${pathOpacity})`;
                ctx.beginPath();
                for (let i = 0; i < path.length; i++) {
                    const x = offsetX + path[i].x * scale;
                    const y = offsetY + path[i].y * scale;
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }
        }

        // 恢复原始的裁剪区域
        ctx.restore();

        // 绘制当前正在绘制的路径
        if (this.properties.currentPath.length > 0) {
            ctx.globalCompositeOperation = "source-over";

            // 保存原始的裁剪区域
            ctx.save();
            // 设置裁剪区域为有效画布范围
            ctx.rect(offsetX, offsetY, scaledWidth, scaledHeight);
            ctx.clip();

            // 对于Box和Square画笔，绘制矩形预览
            if ((this.properties.brushType === "box" || this.properties.brushType === "square") && this.properties.boxStartPoint) {
                const startPoint = this.properties.boxStartPoint;
                const endPoint = this.properties.currentPath[this.properties.currentPath.length - 1];

                let minX = Math.min(startPoint.x, endPoint.x);
                let maxX = Math.max(startPoint.x, endPoint.x);
                let minY = Math.min(startPoint.y, endPoint.y);
                let maxY = Math.max(startPoint.y, endPoint.y);

                let currentColor = this.properties.brushColor;
                if (!currentColor) {
                    currentColor = "255,0,0";
                }
                const rgb = currentColor.split(",").map(c => parseInt(c.trim()));

                if (this.properties.brushType === "square") {
                    const marker = this.properties.activeMarker;
                    const fillColor = marker ? "255,255,0" : currentColor;
                    const fillRgb = fillColor.split(",").map(c => parseInt(c.trim()));
                    ctx.fillStyle = `rgba(${fillRgb[0]},${fillRgb[1]},${fillRgb[2]},${this.properties.brushOpacity})`;
                    if (marker) {
                        const dx = endPoint.x - startPoint.x;
                        const dy = endPoint.y - startPoint.y;
                        const side = Math.max(Math.abs(dx), Math.abs(dy));
                        const signX = dx >= 0 ? 1 : -1;
                        const signY = dy >= 0 ? 1 : -1;
                        const squareEndX = startPoint.x + signX * side;
                        const squareEndY = startPoint.y + signY * side;
                        minX = Math.min(startPoint.x, squareEndX);
                        maxX = Math.max(startPoint.x, squareEndX);
                        minY = Math.min(startPoint.y, squareEndY);
                        maxY = Math.max(startPoint.y, squareEndY);
                    }
                    ctx.fillRect(offsetX + minX * scale, offsetY + minY * scale,
                              (maxX - minX) * scale, (maxY - minY) * scale);
                    if (marker) {
                        const centerX = offsetX + (minX + maxX) * 0.5 * scale;
                        const centerY = offsetY + (minY + maxY) * 0.5 * scale;
                        const sideScaled = Math.min((maxX - minX) * scale, (maxY - minY) * scale);
                        const fontSize = Math.max(10, Math.min(200, sideScaled * 0.6));
                        ctx.fillStyle = "#000000";
                        ctx.font = `bold ${Math.floor(fontSize)}px Arial`;
                        ctx.textAlign = "center";
                        ctx.textBaseline = "middle";
                        ctx.fillText(String(marker), centerX, centerY);
                    }
                } else {
                    // Box画笔：显示空心矩形框预览
                    ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${this.properties.brushOpacity})`;
                    ctx.lineWidth = this.properties.brushSize * scale;
                    ctx.strokeRect(offsetX + minX * scale, offsetY + minY * scale,
                                 (maxX - minX) * scale, (maxY - minY) * scale);
                }
            } else if (this.properties.brushType === "free") {
                // Free画笔：绘制连续路径
                ctx.lineWidth = this.properties.brushSize * scale;

                let currentColor = this.properties.brushColor;
                if (!currentColor) {
                    currentColor = "255,0,0";
                }
                const rgb = currentColor.split(",").map(c => parseInt(c.trim()));

                ctx.strokeStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${this.properties.brushOpacity})`;
                ctx.beginPath();
                for (let i = 0; i < this.properties.currentPath.length; i++) {
                    const x = offsetX + this.properties.currentPath[i].x * scale;
                    const y = offsetY + this.properties.currentPath[i].y * scale;
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
                ctx.stroke();
            }

            // 恢复原始的裁剪区域
            ctx.restore();
        }

        // 绘制左侧工具栏按钮（无背景框）
        for (const button of this.properties.buttons) {
            drawButton(ctx, button, this);
        }

        // 绘制画笔类型按钮（无背景框）
        if (this.properties.brushTypeButtons) {
            drawButton(ctx, this.properties.brushTypeButtons.free, this);
            drawButton(ctx, this.properties.brushTypeButtons.box, this);
            drawButton(ctx, this.properties.brushTypeButtons.square, this);
        }

        // 绘制操作按钮（无背景框）
        if (this.properties.actionButtons) {
            for (const button of this.properties.actionButtons) {
                drawButton(ctx, button, this);
            }
        }

        // 绘制滑块
        for (const slider of this.properties.sliders) {
            const sliderX = slider.x;
            const sliderY = slider.y + slider.labelHeight;
            const sliderWidth = slider.width;
            const sliderHeight = slider.sliderHeight;

            // 绘制标签
            ctx.fillStyle = "#b0b0b0";
            ctx.font = "bold 9px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            const valueText = slider.type === "opacity" ? `${slider.value}%` : slider.value;
            ctx.fillText(`${slider.label}: ${valueText}`, sliderX, slider.y);

            // 绘制滑块轨道背景
            ctx.fillStyle = "#1a1a1a";
            ctx.fillRect(sliderX, sliderY, sliderWidth, sliderHeight);

            // 绘制滑块轨道边框
            ctx.strokeStyle = "#333333";
            ctx.lineWidth = 1;
            ctx.strokeRect(sliderX, sliderY, sliderWidth, sliderHeight);

            // 计算滑块位置
            const range = slider.max - slider.min;
            const progress = (slider.value - slider.min) / range;
            const thumbX = sliderX + progress * (sliderWidth - 4);

            // 绘制已填充区域
            ctx.fillStyle = "#4CAF50";
            ctx.fillRect(sliderX + 2, sliderY + 2, thumbX - sliderX - 2, sliderHeight - 4);

            // 绘制滑块拇指
            ctx.fillStyle = "#ffffff";
            ctx.fillRect(thumbX - 2, sliderY + 1, 4, sliderHeight - 2);
        }

        if (this.properties.colorPalette) {
            const palette = this.properties.colorPalette;
            const paletteX = palette.x;
            const paletteY = palette.y;
            const gridX = paletteX;
            const gridY = paletteY + palette.labelHeight;

            ctx.fillStyle = "#b0b0b0";
            ctx.font = "bold 9px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillText(palette.label, paletteX, paletteY);

            const selectedValue = this.properties.brushColor || "255,0,0";
            const { swatchSize, swatchGap, cols, options } = palette;

            for (let i = 0; i < options.length; i++) {
                const row = Math.floor(i / cols);
                const col = i % cols;
                const x = gridX + col * (swatchSize + swatchGap);
                const y = gridY + row * (swatchSize + swatchGap);

                const opt = options[i];
                const rgb = opt.value.split(",").map(c => parseInt(c.trim()));
                ctx.fillStyle = `rgba(${rgb[0]},${rgb[1]},${rgb[2]},1.0)`;
                ctx.fillRect(x, y, swatchSize, swatchSize);

                const isSelected = opt.value === selectedValue;
                ctx.strokeStyle = isSelected ? "#4CAF50" : "#555555";
                ctx.lineWidth = isSelected ? 2 : 1;
                ctx.strokeRect(x + 0.5, y + 0.5, swatchSize - 1, swatchSize - 1);

                ctx.fillStyle = opt.textColor || "#ffffff";
                ctx.font = "bold 10px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(opt.text, x + swatchSize / 2, y + swatchSize / 2);
            }
        }

        if (this.properties.markerPalette) {
            const palette = this.properties.markerPalette;
            const paletteX = palette.x;
            const paletteY = palette.y;
            const gridX = paletteX;
            const gridY = paletteY + palette.labelHeight;

            ctx.fillStyle = "#b0b0b0";
            ctx.font = "bold 9px Arial";
            ctx.textAlign = "left";
            ctx.textBaseline = "top";
            ctx.fillText(palette.label, paletteX, paletteY);

            const selectedValue = this.properties.activeMarker;
            const { markerSize, markerGap, cols, options } = palette;

            for (let i = 0; i < options.length; i++) {
                const row = Math.floor(i / cols);
                const col = i % cols;
                const x = gridX + col * (markerSize + markerGap);
                const y = gridY + row * (markerSize + markerGap);

                const opt = options[i];
                const isSelected = opt.value === selectedValue;

                ctx.fillStyle = "#ffff00";
                ctx.fillRect(x, y, markerSize, markerSize);
                ctx.strokeStyle = isSelected ? "#4CAF50" : "#444444";
                ctx.lineWidth = isSelected ? 2 : 1;
                ctx.strokeRect(x + 0.5, y + 0.5, markerSize - 1, markerSize - 1);

                ctx.fillStyle = "#000000";
                ctx.font = "bold 10px Arial";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText(opt.text, x + markerSize / 2, y + markerSize / 2);
            }
        }




        } finally {
            ctx.restore();
        }
        syncBrushDataWidget(this);
    };

    function drawButton(ctx, button, node) {
        const isHover = node.mouseX >= button.x && node.mouseX <= button.x + button.width &&
                        node.mouseY >= button.y && node.mouseY <= button.y + button.height;

        // 绘制按钮背景
        if (button.isToggle && button.active) {
            ctx.fillStyle = "#4CAF50";
            ctx.fillRect(button.x, button.y, button.width, button.height);
            ctx.strokeStyle = "#45a049";
            ctx.lineWidth = 1;
            ctx.strokeRect(button.x, button.y, button.width, button.height);
        } else if (isHover) {
            ctx.fillStyle = "#555555";
            ctx.fillRect(button.x, button.y, button.width, button.height);
            ctx.strokeStyle = "#666666";
            ctx.lineWidth = 1;
            ctx.strokeRect(button.x, button.y, button.width, button.height);
        } else {
            ctx.fillStyle = "#3a3a3a";
            ctx.fillRect(button.x, button.y, button.width, button.height);
            ctx.strokeStyle = "#4a4a4a";
            ctx.lineWidth = 1;
            ctx.strokeRect(button.x, button.y, button.width, button.height);
        }

        // 绘制按钮文字
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 9px Arial";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(button.text, button.x + button.width / 2, button.y + button.height / 2);
    }
}

function syncBrushDataWidget(node) {
    const brushDataStrings = (node.properties.brushPaths || []).map(pathObj => {
        const path = pathObj.points || pathObj;
        const mode = pathObj.mode || "brush";
        const type = pathObj.type || "free";
        const size = pathObj.size !== undefined ? pathObj.size : node.properties.brushSize;
        const opacity = pathObj.opacity !== undefined ? pathObj.opacity : node.properties.brushOpacity;
        const color = pathObj.color || node.properties.brushColor || "255,0,0";
        const pointsStr = path.map(p => `${p.x},${p.y}`).join(';');
        if (pathObj.marker) {
            return `${mode}:${type}:${size}:${opacity}:${color}:${pathObj.marker}:${pointsStr}`;
        }
        return `${mode}:${type}:${size}:${opacity}:${color}:${pointsStr}`;
    });

    const brushDataWidget = node.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_DATA) || node.widgets[2];
    if (brushDataWidget) {
        brushDataWidget.value = brushDataStrings.join("|");
    }
}

function setupHiddenWidgets(node) {
    const brushDataWidget = node.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_DATA);
    if (brushDataWidget) {
        brushDataWidget.hidden = true;
    }

    const brushSizeWidget = node.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_SIZE);
    if (brushSizeWidget) {
        brushSizeWidget.hidden = true;
    }

    let widthWidget = node.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
    let heightWidget = node.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);
    let imageBase64Widget = node.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);

    if (!widthWidget) {
        widthWidget = node.addWidget("number", WIDGET_NAMES.IMAGE_WIDTH, 512, () => { }, { min: 64, max: 4096 });
        widthWidget.hidden = true;
    }
    if (!heightWidget) {
        heightWidget = node.addWidget("number", WIDGET_NAMES.IMAGE_HEIGHT, 512, () => { }, { min: 64, max: 4096 });
        heightWidget.hidden = true;
    }
    if (!imageBase64Widget) {
        imageBase64Widget = node.addWidget("text", WIDGET_NAMES.IMAGE_BASE64, "", () => { });
    }
    if (imageBase64Widget) {
        imageBase64Widget.hidden = true;
    }

    node.properties.backgroundImageObj = null;
    node.properties.imageBase64Data = "";
}

function parseStroke(stroke) {
    let mode = "brush";
    let type = "free";
    let size = 10;
    let opacity = 1.0;
    let color = "255,0,0";
    let marker = null;
    let pointsStr = stroke;

    if (stroke.includes(":")) {
        const parts = stroke.split(":");
        if (parts[0] === "brush" || parts[0] === "erase") {
            mode = parts[0];
            if (parts.length >= 6) {
                const maybeType = parts[1];
                if (maybeType === "free" || maybeType === "box" || maybeType === "square") {
                    type = maybeType;
                }
                size = parseFloat(parts[2]) || 10;
                opacity = parseFloat(parts[3]);
                if (!Number.isFinite(opacity)) opacity = 1.0;
                const rgbParts = (parts[4] || "").split(",").map(v => parseInt(v.trim()));
                if (rgbParts.length === 3 && isValidRGB(rgbParts[0], rgbParts[1], rgbParts[2])) {
                    color = `${rgbParts[0]},${rgbParts[1]},${rgbParts[2]}`;
                }
                const maybeMarker = parts[5];
                const looksLikePoint = typeof maybeMarker === "string" && maybeMarker.includes(",");
                const isMarkerToken = typeof maybeMarker === "string" && ["1", "2", "3", "4", "5", "6"].includes(maybeMarker);
                if (isMarkerToken && !looksLikePoint && parts.length >= 7) {
                    marker = maybeMarker;
                    pointsStr = parts.slice(6).join(":");
                } else {
                    pointsStr = parts.slice(5).join(":");
                }
            } else {
                pointsStr = parts.slice(1).join(":");
            }
        } else if (parts[0] === "free" || parts[0] === "box" || parts[0] === "square") {
            type = parts[0];
            pointsStr = parts.slice(1).join(":");
        }
    }

    const points = pointsStr.split(";").filter(point => point.trim() !== "");
    const path = [];
    for (const point of points) {
        if (!point.trim()) continue;
        const coords = point.split(",");
        if (coords.length === 2) {
            path.push({
                x: parseFloat(coords[0]),
                y: parseFloat(coords[1])
            });
        }
    }

    return {
        points: path,
        mode,
        type,
        size,
        opacity,
        color,
        marker
    };
}

function isValidRGB(r, g, b) {
    return [r, g, b].every(v => Number.isFinite(v) && v >= 0 && v <= 255);
}

function initInteractionBindings(node, state) {
    const { shiftLeft, shiftRight, panelHeight } = state.layout;

    node.onKeyDown = function (e) {
        if (!this.capture) {
            return false;
        }

        return false;
    };

    node.onMouseDown = function (e) {
        if (e.canvasY - this.pos[1] < 0) {
            return false;
        }

        const mouseX = e.canvasX - this.pos[0];
        const mouseY = e.canvasY - this.pos[1];

        // 处理左侧工具栏的所有按钮和输入框的点击事件
        const buttonAreaWidth = 80;
        if (mouseX <= buttonAreaWidth) {
            // 处理滑块点击
            if (this.properties.sliders) {
                for (const slider of this.properties.sliders) {
                    const sliderX = slider.x;
                    const sliderY = slider.y + slider.labelHeight;
                    const sliderWidth = slider.width;
                    const sliderHeight = slider.sliderHeight;

                    if (mouseX >= sliderX && mouseX <= sliderX + sliderWidth &&
                        mouseY >= sliderY && mouseY <= sliderY + sliderHeight) {
                        // 开始拖动滑块
                        slider.isDragging = true;
                        this.capture = true;
                        this.captureInput(true);
                        return true;
                    }
                }
            }

            if (this.properties.colorPalette) {
                const palette = this.properties.colorPalette;
                const gridX = palette.x;
                const gridY = palette.y + palette.labelHeight;
                const gridWidth = palette.cols * palette.swatchSize + (palette.cols - 1) * palette.swatchGap;
                const gridHeight = palette.rows * palette.swatchSize + (palette.rows - 1) * palette.swatchGap;

                if (mouseX >= gridX && mouseX <= gridX + gridWidth &&
                    mouseY >= gridY && mouseY <= gridY + gridHeight) {
                    const col = Math.floor((mouseX - gridX) / (palette.swatchSize + palette.swatchGap));
                    const row = Math.floor((mouseY - gridY) / (palette.swatchSize + palette.swatchGap));

                    const cellX = gridX + col * (palette.swatchSize + palette.swatchGap);
                    const cellY = gridY + row * (palette.swatchSize + palette.swatchGap);
                    const inCell = mouseX >= cellX && mouseX <= cellX + palette.swatchSize &&
                        mouseY >= cellY && mouseY <= cellY + palette.swatchSize;
                    if (inCell) {
                        const idx = row * palette.cols + col;
                        const option = palette.options[idx];
                        if (option) {
                            this.properties.brushColor = option.value;
                            this.properties.brushMode = "brush";
                            this.updateThisNodeGraph?.();
                            return true;
                        }
                    }
                }
            }

            if (this.properties.markerPalette) {
                const palette = this.properties.markerPalette;
                const gridX = palette.x;
                const gridY = palette.y + palette.labelHeight;
                const gridWidth = palette.cols * palette.markerSize + (palette.cols - 1) * palette.markerGap;
                const gridHeight = palette.rows * palette.markerSize + (palette.rows - 1) * palette.markerGap;

                if (mouseX >= gridX && mouseX <= gridX + gridWidth &&
                    mouseY >= gridY && mouseY <= gridY + gridHeight) {
                    const col = Math.floor((mouseX - gridX) / (palette.markerSize + palette.markerGap));
                    const row = Math.floor((mouseY - gridY) / (palette.markerSize + palette.markerGap));

                    const cellX = gridX + col * (palette.markerSize + palette.markerGap);
                    const cellY = gridY + row * (palette.markerSize + palette.markerGap);
                    const inCell = mouseX >= cellX && mouseX <= cellX + palette.markerSize &&
                        mouseY >= cellY && mouseY <= cellY + palette.markerSize;
                    if (inCell) {
                        const idx = row * palette.cols + col;
                        const option = palette.options[idx];
                        if (option) {
                            this.properties.activeMarker = option.value;
                            this.properties.brushColor = "255,255,0";
                            this.properties.brushMode = "brush";
                            this.properties.brushType = "square";
                            if (this.properties.brushTypeButtons) {
                                this.properties.brushTypeButtons.free.active = false;
                                this.properties.brushTypeButtons.box.active = false;
                                this.properties.brushTypeButtons.square.active = false;
                            }
                            this.updateThisNodeGraph?.();
                            return true;
                        }
                    }
                }
            }
        }

        // 处理画笔类型按钮
        if (this.properties.brushTypeButtons) {
            const freeBtn = this.properties.brushTypeButtons.free;
            const boxBtn = this.properties.brushTypeButtons.box;
            const squareBtn = this.properties.brushTypeButtons.square;

            if (mouseX >= freeBtn.x && mouseX <= freeBtn.x + freeBtn.width &&
                mouseY >= freeBtn.y && mouseY <= freeBtn.y + freeBtn.height) {
                freeBtn.action();
                return true;
            }

            if (mouseX >= boxBtn.x && mouseX <= boxBtn.x + boxBtn.width &&
                mouseY >= boxBtn.y && mouseY <= boxBtn.y + boxBtn.height) {
                boxBtn.action();
                return true;
            }

            if (squareBtn && mouseX >= squareBtn.x && mouseX <= squareBtn.x + squareBtn.width &&
                mouseY >= squareBtn.y && mouseY <= squareBtn.y + squareBtn.height) {
                squareBtn.action();
                return true;
            }
        }

        // 处理操作按钮
        if (this.properties.actionButtons) {
            for (const button of this.properties.actionButtons) {
                if (button.action && mouseX >= button.x && mouseX <= button.x + button.width &&
                    mouseY >= button.y && mouseY <= button.y + button.height) {
                    button.action();
                    return true;
                }
            }
        }

        // 处理图像加载按钮
        if (this.properties.buttons) {
            for (const button of this.properties.buttons) {
                if (button.action && mouseX >= button.x && mouseX <= button.x + button.width &&
                    mouseY >= button.y && mouseY <= button.y + button.height) {
                    button.action();
                    return true;
                }
            }
        }

        // 处理画布区域的绘制
        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        let canvasAreaHeight = this.size[1] - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        // 只排除左侧工具栏和右侧端口区域，允许在画布外绘制
        if (mouseX <= shiftLeft || mouseX >= this.size[0] - shiftRight) return false;

        if (e.button === 0) {
            let localX = e.canvasX - this.pos[0] - offsetX;
            let localY = e.canvasY - this.pos[1] - offsetY;

            let realX = localX / scale;
            let realY = localY / scale;

            // 允许坐标超出画布范围，这样边缘才能绘画
            // realX = Math.max(0, Math.min(realX, canvasWidth - 1));
            // realY = Math.max(0, Math.min(realY, canvasHeight - 1));

            this.properties.isDrawing = true;
            this.properties.currentPath = [{ x: realX, y: realY }];

            // 对于Box和Square画笔，记录起始点
            if (this.properties.brushType === "box" || this.properties.brushType === "square") {
                this.properties.boxStartPoint = { x: realX, y: realY };
            }

            this.capture = true;
            this.captureInput(true);
            return true;
        }

        return false;
    };

    node.onMouseMove = function (e, _pos, canvas) {
        if (!this.capture) {
            return;
        }

        // 处理滑块拖动
        if (this.properties.sliders) {
            for (const slider of this.properties.sliders) {
                if (slider.isDragging) {
                    const sliderX = slider.x;
                    const sliderY = slider.y + slider.labelHeight;
                    const sliderWidth = slider.width;
                    const sliderHeight = slider.sliderHeight;

                    const mouseX = e.canvasX - this.pos[0];
                    const mouseY = e.canvasY - this.pos[1];

                    if (mouseX >= sliderX && mouseX <= sliderX + sliderWidth &&
                        mouseY >= sliderY && mouseY <= sliderY + sliderHeight) {
                        const range = slider.max - slider.min;
                        const progress = (mouseX - sliderX) / sliderWidth;
                        const newValue = Math.round(slider.min + progress * range);
                        slider.value = Math.max(slider.min, Math.min(slider.max, newValue));

                        if (slider.type === "size") {
                            this.properties.brushSize = slider.value;
                            const brushSizeWidget = this.widgets.find(w => w.name === WIDGET_NAMES.BRUSH_SIZE);
                            if (brushSizeWidget) {
                                brushSizeWidget.value = this.properties.brushSize;
                            }
                        } else if (slider.type === "opacity") {
                            this.properties.brushOpacity = slider.value / 100;
                        }

                        this.updateThisNodeGraph?.();
                        return;
                    }
                }
            }
        }

        if (!this.properties.isDrawing) {
            return;
        }

        if (canvas.pointer.isDown === false) {
            this.onMouseUp(e);
            return;
        }
        this.valueUpdate(e);
    };

    node.valueUpdate = function (e) {
        if (!this.properties.isDrawing) {
            return;
        }

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        let canvasAreaHeight = this.size[1] - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        let mouseX = e.canvasX - this.pos[0] - offsetX;
        let mouseY = e.canvasY - this.pos[1] - offsetY;

        let realX = mouseX / scale;
        let realY = mouseY / scale;

        // 允许坐标超出画布范围，这样边缘才能绘画
        // realX = Math.max(0, Math.min(realX, canvasWidth - 1));
        // realY = Math.max(0, Math.min(realY, canvasHeight - 1));

        const lastPoint = this.properties.currentPath[this.properties.currentPath.length - 1];
        const dist = Math.sqrt(
            Math.pow(realX - lastPoint.x, 2) +
            Math.pow(realY - lastPoint.y, 2)
        );

        if (dist > 1) {
            this.properties.currentPath.push({ x: realX, y: realY });
            this.updateThisNodeGraph?.();
        }
    };

    node.onMouseUp = function () {
        if (!this.capture) {
            return;
        }

        // 停止所有滑块拖动
        if (this.properties.sliders) {
            for (const slider of this.properties.sliders) {
                slider.isDragging = false;
            }
        }

        if (this.properties.isDrawing && this.properties.currentPath.length > 0) {
            // 处理Box和Square画笔：生成矩形路径
            if ((this.properties.brushType === "box" || this.properties.brushType === "square") && this.properties.boxStartPoint) {
                const startPoint = this.properties.boxStartPoint;
                const endPoint = this.properties.currentPath[this.properties.currentPath.length - 1];

                let minX = Math.min(startPoint.x, endPoint.x);
                let maxX = Math.max(startPoint.x, endPoint.x);
                let minY = Math.min(startPoint.y, endPoint.y);
                let maxY = Math.max(startPoint.y, endPoint.y);

                // 根据画笔类型选择颜色
                const color = this.properties.brushColor;

                if (this.properties.brushType === "box") {
                    // Box画笔：绘制空心矩形框
                    const boxPath = [
                        { x: minX, y: minY },
                        { x: maxX, y: minY },
                        { x: maxX, y: maxY },
                        { x: minX, y: maxY },
                        { x: minX, y: minY }
                    ];

                    this.properties.brushPaths.push({
                        points: boxPath,
                        mode: "brush",
                        type: "box",
                        size: this.properties.brushSize,
                        opacity: this.properties.brushOpacity,
                        color: color
                    });
                } else if (this.properties.brushType === "square") {
                    // Square画笔：绘制实心方块
                    const marker = this.properties.activeMarker;
                    if (marker) {
                        const dx = endPoint.x - startPoint.x;
                        const dy = endPoint.y - startPoint.y;
                        const side = Math.max(Math.abs(dx), Math.abs(dy));
                        const signX = dx >= 0 ? 1 : -1;
                        const signY = dy >= 0 ? 1 : -1;
                        const squareEndX = startPoint.x + signX * side;
                        const squareEndY = startPoint.y + signY * side;
                        minX = Math.min(startPoint.x, squareEndX);
                        maxX = Math.max(startPoint.x, squareEndX);
                        minY = Math.min(startPoint.y, squareEndY);
                        maxY = Math.max(startPoint.y, squareEndY);
                    }
                    if (marker && Math.abs(maxX - minX) < 5 && Math.abs(maxY - minY) < 5) {
                        const defaultSize = 60;
                        maxX = minX + defaultSize;
                        maxY = minY + defaultSize;
                    }
                    const squarePath = [
                        { x: minX, y: minY },
                        { x: maxX, y: minY },
                        { x: maxX, y: maxY },
                        { x: minX, y: maxY },
                        { x: minX, y: minY }
                    ];

                    const colorForSquare = marker ? "255,255,0" : color;
                    this.properties.brushPaths.push({
                        points: squarePath,
                        mode: "brush",
                        type: "square",
                        size: this.properties.brushSize,
                        opacity: this.properties.brushOpacity,
                        color: colorForSquare,
                        marker: marker
                    });
                }

                this.properties.boxStartPoint = null;
            } else {
                // 处理Free画笔：保存当前路径
                const color = this.properties.brushColor;

                this.properties.brushPaths.push({
                    points: [...this.properties.currentPath],
                    mode: "brush",
                    type: "free",
                    size: this.properties.brushSize,
                    opacity: this.properties.brushOpacity,
                    color: color
                });
            }

            this.properties.currentPath = [];
            syncBrushDataWidget(this);
        }

        this.properties.isDrawing = false;
        this.capture = false;
        this.captureInput(false);
        this.updateThisNodeGraph?.();
    };

    node.onSelected = function () {
        this.onMouseUp();
    };

    const originalOnConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function (type, slot, isInput, link, info) {
        if (originalOnConnectionsChange) {
            originalOnConnectionsChange.apply(this, arguments);
        }

        if (isInput && slot === 0 && type === 1) {
            setTimeout(() => { }, 100);
        }
    };

    const originalOnAfterExecuteNode = node.onAfterExecuteNode;
    node.onAfterExecuteNode = function (message) {
        if (originalOnAfterExecuteNode) {
            originalOnAfterExecuteNode.apply(this, arguments);
        }
        return message;
    };

    const originalOnWidgetChange = node.onWidgetChange;
    node.onWidgetChange = function (widget) {
        if (originalOnWidgetChange) {
            originalOnWidgetChange.apply(this, arguments);
        }

        if (!widget) {
            return;
        }

        if (widget.name === WIDGET_NAMES.BRUSH_SIZE) {
            this.properties.brushSize = widget.value || 10;
            this.updateThisNodeGraph?.();
        }

        if (widget.name === WIDGET_NAMES.IMAGE_WIDTH || widget.name === WIDGET_NAMES.IMAGE_HEIGHT) {
            const widthWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_WIDTH);
            const heightWidget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_HEIGHT);
            if (widthWidget && heightWidget && widthWidget.value && heightWidget.value) {
                this.updateImageSize(widthWidget.value, heightWidget.value);
            }
        }

        if (widget.name === WIDGET_NAMES.IMAGE_BASE64) {
            if (widget.value) {
                this.properties.imageBase64Data = widget.value;
                this.loadBackgroundImageFromBase64(widget.value);
            } else {
                this.properties.backgroundImageObj = null;
                this.properties.imageBase64Data = "";
                this.updateThisNodeGraph?.();
            }
        }
    };

    node.loadImageFromFile = function () {
        const input = document.createElement("input");
        input.type = "file";
        input.accept = "image/*";
        input.onchange = e => {
            const file = e.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            reader.onload = event => {
                try {
                    const dataURL = event.target.result;
                    let base64String = dataURL;
                    if (dataURL.includes(",")) {
                        base64String = dataURL.split(",")[1];
                    }

                    this.properties.imageBase64Data = base64String;

                    const imageBase64Widget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);
                    if (imageBase64Widget) {
                        imageBase64Widget.value = base64String;
                    }

                    this.loadBackgroundImageFromBase64(dataURL);

                    this.properties.brushPaths = [];
                    this.properties.currentPath = [];

                    console.log("Image loaded successfully, size:", base64String.length, "bytes");
                } catch (err) {
                    console.error("Error processing image file:", err);
                    alert("加载图片失败: " + err.message);
                }
            };
            reader.onerror = err => {
                console.error("Error reading file:", err);
                alert("读取文件失败");
            };
            reader.readAsDataURL(file);
        };
        input.click();
    };

    node.openColorPicker = function (type) {
        let currentColor = type === "eraser" ? this.properties.eraserColor : this.properties.brushColor;
        if (!currentColor) {
            currentColor = type === "eraser" ? "255,50,50" : "255,255,255";
        }

        const rgb = currentColor.split(",").map(c => parseInt(c.trim()));
        const hexColor = "#" + rgb.map(c => {
            const hex = c.toString(16);
            return hex.length === 1 ? "0" + hex : hex;
        }).join("");

        const colorInput = document.createElement("input");
        colorInput.type = "color";
        colorInput.value = hexColor;
        colorInput.style.position = "fixed";
        colorInput.style.left = "-9999px";
        document.body.appendChild(colorInput);

        colorInput.onchange = e => {
            const hex = e.target.value;
            const r = parseInt(hex.substr(1, 2), 16);
            const g = parseInt(hex.substr(3, 2), 16);
            const b = parseInt(hex.substr(5, 2), 16);
            const rgbColor = `${r},${g},${b}`;

            if (type === "eraser") {
                this.properties.eraserColor = rgbColor;
            } else {
                this.properties.brushColor = rgbColor;
            }

            if (this.properties.colorButtonGroup?.buttons) {
                for (const colorBtn of this.properties.colorButtonGroup.buttons) {
                    if (colorBtn.type === type) {
                        colorBtn.color = rgbColor;
                    }
                }
            }

            this.updateThisNodeGraph?.();
            document.body.removeChild(colorInput);
        };

        colorInput.onblur = () => {
            setTimeout(() => {
                if (document.body.contains(colorInput)) {
                    document.body.removeChild(colorInput);
                }
            }, 100);
        };

        colorInput.click();
    };

    node.loadBackgroundImageFromBase64 = function (base64String) {
        if (!base64String || base64String.trim() === "") {
            this.properties.backgroundImageObj = null;
            this.updateThisNodeGraph?.();
            return;
        }

        try {
            const img = new Image();
            img.onload = () => {
                this.properties.backgroundImageObj = img;
                this.updateImageSize(img.width, img.height);
                this.updateThisNodeGraph?.();
            };
            img.onerror = err => {
                console.error("Error loading background image from base64:", err);
                this.properties.backgroundImageObj = null;
            };
            if (base64String.startsWith("data:")) {
                img.src = base64String;
            } else {
                img.src = "data:image/png;base64," + base64String;
            }
        } catch (err) {
            console.error("Error creating image from base64:", err);
            this.properties.backgroundImageObj = null;
        }
    };

    // 拖拽功能：支持从桌面拖拽图像到画布
    node.onDragOver = function (e) {
        // 检查是否在画布区域内
        const mouseX = e.canvasX - this.pos[0];
        const mouseY = e.canvasY - this.pos[1];

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        let canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        // 判断鼠标是否在画布区域内
        if (mouseX >= offsetX && mouseX <= offsetX + scaledWidth &&
            mouseY >= offsetY && mouseY <= offsetY + scaledHeight) {

            // 检查是否有图像文件
            if (e.dataTransfer && e.dataTransfer.types) {
                const hasFiles = Array.from(e.dataTransfer.types).includes('Files');
                if (hasFiles) {
                    e.preventDefault();
                    e.stopPropagation();
                    return true;
                }
            }
        }
        return false;
    };

    // 注意：ComfyUI 使用 onDragDrop 而不是 onDrop
    node.onDragDrop = function (e) {
        console.log("onDragDrop triggered", e);

        // 检查是否在画布区域内
        const mouseX = e.canvasX - this.pos[0];
        const mouseY = e.canvasY - this.pos[1];

        const canvasWidth = this.properties.imageWidth || 512;
        const canvasHeight = this.properties.imageHeight || 512;
        let canvasAreaWidth = this.size[0] - shiftRight - shiftLeft;
        let canvasAreaHeight = this.size[1] - shiftLeft - shiftLeft - panelHeight;

        const scaleX = canvasAreaWidth / canvasWidth;
        const scaleY = canvasAreaHeight / canvasHeight;
        const scale = Math.min(scaleX, scaleY);

        const scaledWidth = canvasWidth * scale;
        const scaledHeight = canvasHeight * scale;
        const offsetX = shiftLeft + (canvasAreaWidth - scaledWidth) / 2;
        const offsetY = shiftLeft + panelHeight + (canvasAreaHeight - scaledHeight) / 2;

        // 判断鼠标是否在画布区域内
        if (mouseX < offsetX || mouseX > offsetX + scaledWidth ||
            mouseY < offsetY || mouseY > offsetY + scaledHeight) {
            console.log("Drop outside canvas area");
            return false;
        }

        // 处理拖拽的文件
        if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            console.log("File dropped:", file.name, file.type);

            // 检查是否是图像文件
            if (!file.type.startsWith('image/')) {
                console.warn('Only image files are supported');
                return false;
            }

            // 读取文件并转换为 base64
            const reader = new FileReader();
            reader.onload = event => {
                try {
                    const dataURL = event.target.result;
                    let base64String = dataURL;
                    if (dataURL.includes(",")) {
                        base64String = dataURL.split(",")[1];
                    }

                    this.properties.imageBase64Data = base64String;

                    const imageBase64Widget = this.widgets.find(w => w.name === WIDGET_NAMES.IMAGE_BASE64);
                    if (imageBase64Widget) {
                        imageBase64Widget.value = base64String;
                    }

                    this.loadBackgroundImageFromBase64(dataURL);

                    // 清空画笔路径
                    this.properties.brushPaths = [];
                    this.properties.currentPath = [];

                    console.log("Image loaded from drag & drop, size:", base64String.length, "bytes");
                } catch (err) {
                    console.error("Error processing dropped image:", err);
                    alert("加载图片失败: " + err.message);
                }
            };
            reader.onerror = err => {
                console.error("Error reading dropped file:", err);
                alert("读取文件失败");
            };
            reader.readAsDataURL(file);

            e.preventDefault();
            e.stopPropagation();
            return true;  // 返回 true 表示已处理
        }

        console.log("No files in drop event");
        return false;
    };
}

// author.yichengup.Loadimage_brushmask 2025.01.XX
app.registerExtension({
    name: "IO_EasyMark",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "IO_EasyMark") {
            return;
        }

            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, []);
            }
                this.IO_EasyMark = new IO_EasyMark(this);
                if (this.initButtons) {
                    this.initButtons();
                }
        };
    }
});

// author.yichengup.Loadimage_brushmask 2025.01.XX
