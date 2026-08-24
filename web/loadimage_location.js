import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";


app.registerExtension({
    name: "Comfy.Coordinate_loadImage",
    async beforeRegisterNodeDef(nodeType, nodeData, appInstance) {
        if (nodeData.name !== "Coordinate_loadImage") {
            return;
        }
        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            if (originalOnNodeCreated) {
                originalOnNodeCreated.apply(this, arguments);
            }
            this.pointsData = [];
            this.currentImage = null;
            this.imageUrl = null;
            this._isNewNode = true;
            this.markerBgColor = "red";
            this.markerTextColor = "white";
            this._createImagePreview();
            this._setupImageWidget();
            setTimeout(() => {
                this._hideBackendWidgets();
            }, 50);
            this.size = [420, 550]; // 调整高度，因为移除了颜色选择器
            this.serialize_widgets = true;
            this.resizable = false;
            this.imgs = null;
        };
        nodeType.prototype.onResize = function() {
            this.size = [420, 550]; // 调整高度
        };
        nodeType.prototype._hideBackendWidgets = function() {
            if (!this.widgets) return;
            for (let i = this.widgets.length - 1; i >= 0; i--) {
                const widget = this.widgets[i];
                // 隐藏相关参数
                if (widget.name === "points_data" || widget.name === "marker_bg_color" || widget.name === "marker_text_color") {
                    widget.type = "converted-widget";
                    widget.computeSize = () => [0, -4];
                    widget.serializeValue = async () => widget.value;
                    if (widget.inputEl) widget.inputEl.style.display = "none";
                    if (widget.element) widget.element.style.display = "none";
                }
            }
            if (this.inputs) {
                for (let i = this.inputs.length - 1; i >= 0; i--) {
                    const input = this.inputs[i];
                    if (input.name === "points_data" || input.name === "marker_bg_color" || input.name === "marker_text_color") {
                        this.removeInput(i);
                    }
                }
            }
        };
        const originalOnDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function(ctx) {
            this.imgs = null;
            if (originalOnDrawBackground) {
                originalOnDrawBackground.apply(this, arguments);
            }
        };
        nodeType.prototype._createImagePreview = function() {
            const node = this;
            this.CONTAINER_WIDTH = 400;
            this.CONTAINER_HEIGHT = 400;
            const previewContainer = document.createElement("div");
            previewContainer.style.cssText = `position: relative;width: 100%;height: ${this.CONTAINER_HEIGHT}px;background: #1a1a1a;border: 1px solid #333;border-radius: 4px;overflow: hidden;cursor: crosshair;`;
            const imgElement = document.createElement("img");
            imgElement.style.cssText = `width: 100%;height: 100%;object-fit: contain;display: none;`;
            previewContainer.appendChild(imgElement);
            const placeholder = document.createElement("div");
            placeholder.style.cssText = `position: absolute;top: 50%;left: 50%;transform: translate(-50%, -50%);color: #666;font-size: 14px;text-align: center;`;
            placeholder.textContent = "选择文件上传";
            previewContainer.appendChild(placeholder);
            const markerOverlay = document.createElement("div");
            markerOverlay.style.cssText = `position: absolute;top: 0;left: 0;width: 100%;height: 100%;pointer-events: none;`;
            previewContainer.appendChild(markerOverlay);
            this.previewContainer = previewContainer;
            this.imgElement = imgElement;
            this.placeholder = placeholder;
            this.markerOverlay = markerOverlay;
            previewContainer.addEventListener("click", (e) => {
                node._handleImageClick(e);
            });
            const widget = this.addDOMWidget("image_preview", "customWidget", previewContainer, {
                serialize: false,
                hideOnZoom: false
            });
            widget.computeSize = (width) => [width - 20, this.CONTAINER_HEIGHT + 10];
            
            const controlContainer = document.createElement("div");
            controlContainer.style.cssText = `display: flex;gap: 4px;margin-top: 5px;justify-content: space-between;box-sizing: border-box;padding: 0 2px;`;
            

            
            const deleteBtn = document.createElement("button");
            deleteBtn.textContent = "删除选中";
            // 补充flex居中属性，确保文字正中间
            deleteBtn.style.cssText = `flex: 1;padding: 4px 0;background: #444;border: none;border-radius: 3px;color: #fff;cursor: pointer;font-size: 11px;min-width: 80px;display: flex;align-items: center;justify-content: center;text-align: center;line-height: normal;`;
            deleteBtn.onclick = () => node._deleteSelectedMarker();
            controlContainer.appendChild(deleteBtn);

            const clearBtn = document.createElement("button");
            clearBtn.textContent = "清空";
            // 同步给清空按钮添加居中样式
            clearBtn.style.cssText = `flex: 1;padding: 4px 0;background: #444;border: none;border-radius: 3px;color: #fff;cursor: pointer;font-size: 11px;min-width: 80px;display: flex;align-items: center;justify-content: center;text-align: center;line-height: normal;`;
            clearBtn.onclick = () => node._clearAllMarkers();
            controlContainer.appendChild(clearBtn);


            // 移除颜色选择器相关代码
            
            const controlWidget = this.addDOMWidget("control_buttons", "customWidget", controlContainer, {
                serialize: false,
                hideOnZoom: false
            });
            controlWidget.computeSize = (width) => [width - 20, 30]; // 调整高度
        };
        nodeType.prototype._setupImageWidget = function() {
            const node = this;
            const imageWidget = this.widgets?.find(w => w.name === "image");
            if (!imageWidget) return;
            const originalCallback = imageWidget.callback;
            imageWidget.callback = function(value) {
                if (originalCallback) {
                    originalCallback.call(this, value);
                }
                if (value) {
                    node._loadImage(value);
                }
            };
            this._hideDefaultImagePreview();
        };
        nodeType.prototype._hideDefaultImagePreview = function() {
            setTimeout(() => {
                if (this.widgets) {
                    this.widgets.forEach(w => {
                        if (w.type === "IMAGEUPLOAD" || w.name === "upload") {
                            if (w.element) {
                                w.element.style.display = "none";
                            }
                        }
                    });
                }
                if (this.imgs) {
                    this.imgs = [];
                }
                this.setDirtyCanvas(true, true);
            }, 100);
        };
        nodeType.prototype._loadImage = function(filename) {
            if (!filename) return;
            const imageUrl = `/view?filename=${encodeURIComponent(filename)}&type=input&subfolder=`;
            this.imgElement.src = imageUrl;
            this.imgElement.style.display = "block";
            this.placeholder.style.display = "none";
            this.imageUrl = imageUrl;
            this.imgElement.onload = () => {
                this._clearAllMarkers();
                this.setDirtyCanvas(true);
            };
            this.imgElement.onerror = () => {
                console.error("图片加载失败:", filename);
                this.placeholder.textContent = "图片加载失败";
                this.placeholder.style.display = "block";
                this.imgElement.style.display = "none";
            };
        };
        nodeType.prototype._handleImageClick = function(e) {
            let target = e.target;
            while (target && target !== this.previewContainer) {
                if (target.classList && target.classList.contains("marker-point")) {
                    return;
                }
                target = target.parentElement;
            }
            if (!this.imgElement || this.imgElement.style.display === "none") {
                return;
            }
            const containerRect = this.previewContainer.getBoundingClientRect();
            const scaleX = containerRect.width / this.CONTAINER_WIDTH;
            const scaleY = containerRect.height / this.CONTAINER_HEIGHT;
            const clickX = (e.clientX - containerRect.left) / scaleX;
            const clickY = (e.clientY - containerRect.top) / scaleY;
            const displayInfo = this._getImageDisplayInfo();
            if (!displayInfo) return;
            const { x: imgDisplayX, y: imgDisplayY, width: imgDisplayWidth, height: imgDisplayHeight } = displayInfo;
            if (clickX < imgDisplayX || clickX > imgDisplayX + imgDisplayWidth || clickY < imgDisplayY || clickY > imgDisplayY + imgDisplayHeight) {
                return;
            }
            const relX = (clickX - imgDisplayX) / imgDisplayWidth;
            const relY = (clickY - imgDisplayY) / imgDisplayHeight;
            const newIndex = this.pointsData.length + 1;
            this.pointsData.push({
                x: relX,
                y: relY,
                index: newIndex
            });
            this._renderMarkers();
            this._updatePointsWidget();
        };
        nodeType.prototype._getImageDisplayInfo = function() {
            if (!this.imgElement) return null;
            const containerWidth = this.CONTAINER_WIDTH;
            const containerHeight = this.CONTAINER_HEIGHT;
            const imgNaturalWidth = this.imgElement.naturalWidth || 1;
            const imgNaturalHeight = this.imgElement.naturalHeight || 1;
            const imgAspect = imgNaturalWidth / imgNaturalHeight;
            const containerAspect = containerWidth / containerHeight;
            let imgDisplayWidth, imgDisplayHeight, imgDisplayX, imgDisplayY;
            if (imgAspect > containerAspect) {
                imgDisplayWidth = containerWidth;
                imgDisplayHeight = containerWidth / imgAspect;
                imgDisplayX = 0;
                imgDisplayY = (containerHeight - imgDisplayHeight) / 2;
            } else {
                imgDisplayHeight = containerHeight;
                imgDisplayWidth = containerHeight * imgAspect;
                imgDisplayX = (containerWidth - imgDisplayWidth) / 2;
                imgDisplayY = 0;
            }
            return {
                x: imgDisplayX,
                y: imgDisplayY,
                width: imgDisplayWidth,
                height: imgDisplayHeight
            };
        };
        nodeType.prototype._renderMarkers = function() {
            this.markerOverlay.innerHTML = "";
            const displayInfo = this._getImageDisplayInfo();
            if (!displayInfo) return;
            const { x: imgDisplayX, y: imgDisplayY, width: imgDisplayWidth, height: imgDisplayHeight } = displayInfo;
            const node = this;
            // 固定颜色映射
            const colorMap = {
                "red": "rgba(255, 69, 0, 0.9)",
                "white": "rgba(255, 255, 255, 0.9)"
            };
            this.pointsData.forEach((point, idx) => {
                const absX = imgDisplayX + point.x * imgDisplayWidth;
                const absY = imgDisplayY + point.y * imgDisplayHeight;
                const marker = document.createElement("div");
                marker.className = "marker-point";
                // 使用固定的红色背景和白色文字
                marker.style.cssText = `position: absolute;left: ${absX}px;top: ${absY}px;transform: translate(-50%, -50%);width: 22px;height: 22px;background: ${colorMap["red"]};border: 2px solid rgba(255, 255, 255, 0.9);border-radius: 50%;display: flex;align-items: center;justify-content: center;color: ${colorMap["white"]};font-size: 12px;font-weight: bold;pointer-events: auto;cursor: grab;box-shadow: 0 2px 4px rgba(0,0,0,0.3);user-select: none;z-index: 10;`;
                marker.textContent = point.index;
                marker.dataset.index = idx;
                let isDragging = false;
                let startX, startY;
                marker.onmousedown = (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                    isDragging = true;
                    startX = e.clientX;
                    startY = e.clientY;
                    marker.style.cursor = "grabbing";
                    node._selectMarker(idx);
                    const containerRect = node.previewContainer.getBoundingClientRect();
                    const scaleX = containerRect.width / node.CONTAINER_WIDTH;
                    const scaleY = containerRect.height / node.CONTAINER_HEIGHT;
                    const onMouseMove = (moveEvent) => {
                        if (!isDragging) return;
                        const deltaX = (moveEvent.clientX - startX) / scaleX;
                        const deltaY = (moveEvent.clientY - startY) / scaleY;
                        let newRelX = point.x + deltaX / imgDisplayWidth;
                        let newRelY = point.y + deltaY / imgDisplayHeight;
                        newRelX = Math.max(0, Math.min(1, newRelX));
                        newRelY = Math.max(0, Math.min(1, newRelY));
                        node.pointsData[idx].x = newRelX;
                        node.pointsData[idx].y = newRelY;
                        const newAbsX = imgDisplayX + newRelX * imgDisplayWidth;
                        const newAbsY = imgDisplayY + newRelY * imgDisplayHeight;
                        marker.style.left = `${newAbsX}px`;
                        marker.style.top = `${newAbsY}px`;
                        startX = moveEvent.clientX;
                        startY = moveEvent.clientY;
                    };
                    const onMouseUp = () => {
                        isDragging = false;
                        marker.style.cursor = "grab";
                        document.removeEventListener("mousemove", onMouseMove);
                        document.removeEventListener("mouseup", onMouseUp);
                        node._updatePointsWidget();
                    };
                    document.addEventListener("mousemove", onMouseMove);
                    document.addEventListener("mouseup", onMouseUp);
                };
                this.markerOverlay.appendChild(marker);
            });
        };
        nodeType.prototype._selectMarker = function(idx) {
            this.selectedMarkerIndex = idx;
            const markers = this.markerOverlay.children;
            for (let i = 0; i < markers.length; i++) {
                if (i === idx) {
                    markers[i].style.border = "3px solid #FFD700";
                } else {
                    markers[i].style.border = "2px solid white";
                }
            }
        };
        nodeType.prototype._deleteSelectedMarker = function() {
            if (this.selectedMarkerIndex === undefined || this.selectedMarkerIndex === null) {
                return;
            }
            const idx = this.selectedMarkerIndex;
            this.pointsData.splice(idx, 1);
            this.pointsData.forEach((point, i) => {
                point.index = i + 1;
            });
            this.selectedMarkerIndex = null;
            this._renderMarkers();
            this._updatePointsWidget();
        };
        nodeType.prototype._clearAllMarkers = function() {
            this.pointsData = [];
            this.selectedMarkerIndex = null;
            this.markerOverlay.innerHTML = "";
            this._updatePointsWidget();
        };
        nodeType.prototype._updatePointsWidget = function() {
            const pointsWidget = this.widgets?.find(w => w.name === "points_data");
            if (pointsWidget) {
                pointsWidget.value = JSON.stringify(this.pointsData);
            }
            if (this.graph) {
                this.graph.change();
            }
        };
        // 移除颜色更新方法
        const originalSerialize = nodeType.prototype.serialize;
        nodeType.prototype.serialize = function() {
            const data = originalSerialize ? originalSerialize.call(this) : {};
            data.pointsData = this.pointsData || [];
            data.currentImageName = this._getCurrentImageName();
            // 保留颜色信息但不提供修改界面
            data.marker_bg_color = "red";
            data.marker_text_color = "white";
            return data;
        };
        nodeType.prototype._getCurrentImageName = function() {
            const imageWidget = this.widgets?.find(w => w.name === "image");
            return imageWidget?.value || "";
        };
        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function(data) {
            if (originalConfigure) {
                originalConfigure.call(this, data);
            }
            const node = this;
            node._isNewNode = false;
            setTimeout(() => {
                const hasPointsData = data.pointsData && Array.isArray(data.pointsData) && data.pointsData.length > 0;
                const imageWidget = node.widgets?.find(w => w.name === "image");
                const hasImage = imageWidget && imageWidget.value;
                
                // 强制使用红色背景和白色文字，忽略可能的旧数据
                node.markerBgColor = "red";
                node.markerTextColor = "white";
                
                if (hasPointsData) {
                    node.pointsData = data.pointsData;
                    node._updatePointsWidget();
                } else {
                    node.pointsData = [];
                    node._updatePointsWidget();
                }
                if (hasImage) {
                    node._loadImage(imageWidget.value);
                    setTimeout(() => {
                        if (hasPointsData) {
                            node._renderMarkers();
                        }
                    }, 200);
                }
            }, 150);
        };
    }
});













