import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";

/**
 * ComfyUI 3D Image Editor Frontend Extension
 * This script provides the 3D editing functionality for the ComfyUI 3D Image Editor node.
 */

// Register the extension
class ThreeDEditorModal extends ComfyDialog {
    static instance = null;
    static getInstance() {
        if (!ThreeDEditorModal.instance) {
            ThreeDEditorModal.instance = new ThreeDEditorModal();
        }
        return ThreeDEditorModal.instance;
    }

    constructor() {
        super();
        this.currentNode = null;
        this.modalElement = null;
        this._initialized = false;
    }

    setNode(node) {
        this.currentNode = node;
    }

    show() {
        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }

        this.modalElement = $el("div.comfy-modal", { parent: document.body }, [
            $el("div.comfy-modal-content", {}, [])
        ]);
        this.modalElement.classList.add("comfy-modal-layout");
        this.modalElement.style.width = "85vw";
        this.modalElement.style.height = "85vh";
        this.modalElement.style.maxWidth = "100vw";
        this.modalElement.style.maxHeight = "100vh";
        const root = document.createElement("div");
        root.id = "comfyui-3d-editor-modal";
        root.style.display = "flex";
        root.style.flexDirection = "column";
        root.style.width = "100%";
        root.style.height = "100%";
        root.innerHTML = `
            <div class="bg-gray-900" style="display:flex;flex-direction:column;width:100%;height:100%;background:#1a202c;color:#fff;">
              <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #4a5568;padding:10px 12px;">
                <h2 style="margin:0;font-size:16px;font-weight:700;">3D Image Editor</h2>
                <button id="comfyui-3d-editor-close" style="background:none;border:none;color:#a0aec0;font-size:20px;cursor:pointer;">×</button>
              </div>
              <div style="flex:1;display:flex;gap:12px;padding:12px;overflow:hidden;min-height:0;">
                <div style="flex:0 0 22%;display:flex;flex-direction:column;gap:12px;overflow:auto;min-height:0;">

                  <div class="panel" style="background:#2d3748;border-radius:8px;padding:12px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">
                      <h3 style="margin:0;font-size:14px;">灯光角度</h3>
                      <div style="display:flex;gap:6px;align-items:center;">
                        <select id="light-preset-select" style="min-width:130px;background:#1a202c;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:4px 6px;font-size:12px;"></select>
                        <button id="reset-light-btn" style="background:#4a5568;color:#fff;border:none;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px;">重置</button>
                      </div>
                    </div>
                    <label style="display:block;font-size:12px;">Intensity</label>
                    <input type="range" id="light-intensity" min="0" max="10" step="0.1" value="1" />
                    <div style="font-size:12px;display:flex;justify-content:space-between;"><span>0</span><span id="intensity-value">1.0</span><span>10.0</span></div>
                    <label style="display:block;margin-top:8px;font-size:12px;">Color</label>
                    <input type="color" id="light-color-picker" value="#ffffff" style="width:100%;height:32px;background:#1a202c;border:1px solid #4a5568;border-radius:4px;cursor:pointer;padding:0 2px;" />
                    <label style="display:block;margin-top:8px;font-size:12px;">X</label><input type="range" id="light-x" min="-10" max="10" step="0.5" value="0" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>-10</span><span id="light-x-value">0.0</span><span>10</span></div>
                    <label style="display:block;margin-top:8px;font-size:12px;">Y</label><input type="range" id="light-y" min="-10" max="10" step="0.5" value="0" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>-10</span><span id="light-y-value">0.0</span><span>10</span></div>
                    <label style="display:block;margin-top:8px;font-size:12px;">Z</label><input type="range" id="light-z" min="-10" max="10" step="0.5" value="3" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>-10</span><span id="light-z-value">3.0</span><span>10</span></div>
                    <label style="display:block;margin-top:8px;font-size:12px;">Ambient</label><input type="range" id="ambient-light" min="0" max="1" step="0.1" value="0.2" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>0</span><span id="ambient-value">0.2</span><span>1.0</span></div>
                  </div>

                  <div class="panel" style="background:#2d3748;border-radius:8px;padding:12px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">
                      <h3 style="margin:0;font-size:14px;">摄像机视角</h3>
                      <div style="display:flex;gap:6px;align-items:center;">
                        <select id="camera-preset-select" style="min-width:130px;background:#1a202c;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:4px 6px;font-size:12px;"></select>
                        <button id="reset-camera-btn" style="background:#4a5568;color:#fff;border:none;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px;">重置</button>
                      </div>
                    </div>
                    <label style="display:block;font-size:12px;">Rotate</label><input type="range" id="rotation-y" min="0" max="360" step="1" value="45" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>0°</span><span id="rotation-y-value">45°</span><span>360°</span></div>
                    <label style="display:block;margin-top:8px;font-size:12px;">Vertical</label><input type="range" id="rotation-x" min="-90" max="90" step="1" value="0" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>-90°</span><span id="rotation-x-value">0°</span><span>90°</span></div>
                    <label style="display:block;margin-top:8px;font-size:12px;">Zoom</label><input type="range" id="camera-zoom" min="1" max="10" step="0.1" value="5" /><div style="font-size:12px;display:flex;justify-content:space-between;"><span>1.0</span><span id="camera-zoom-value">5.0</span><span>10.0</span></div>
                  </div>
                  <div class="panel" style="background:#2d3748;border-radius:8px;padding:12px;">
                    <h3 style="margin:0 0 8px 0;font-size:14px;">Actions</h3>
                    <div style="display:flex;gap:8px;flex-wrap:wrap;">
                      <button id="render-btn" disabled style="flex:1;background:#38a169;color:#fff;border:none;border-radius:4px;padding:8px;cursor:pointer;">生成词</button>
                      <button id="apply-node-btn" disabled style="flex:1;background:#d69e2e;color:#fff;border:none;border-radius:4px;padding:8px;cursor:pointer;">应用</button>
                    </div>
                  </div>
                </div>

                <div style="flex:1;display:flex;flex-direction:column;min-height:0;overflow:auto;gap:12px;">
                  <div style="background:#2d3748;border-radius:8px;padding:12px;display:flex;flex-direction:column;flex:1;min-height:0;">
                    <h3 style="margin:0 0 8px 0;font-size:14px;">3D Canvas</h3>
                    <div id="canvas-container" style="flex:1;position:relative;background:#1a202c;border-radius:6px;min-height:320px;">
                      <div id="canvas-placeholder" style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#a0aec0;">Load an image to begin</div>
                      <canvas id="editor-canvas" style="width:100%;height:100%;display:none;"></canvas>
                    </div>
                    <div style="margin-top:8px;">
                      <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
                        <span style="font-size:20px;font-weight:700;line-height:1;">灯光词</span>
                        <button id="save-light-record-btn" style="background:#2b6cb0;color:#fff;border:none;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px;">保存</button>
                        <div id="light-record-list" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
                      </div>
                      <textarea id="light-prompt-output" rows="3" readonly style="width:100%;background:#1a202c;border:1px solid #4a5568;border-radius:4px;color:#e2e8f0;padding:6px;"></textarea>
                      <div style="display:flex;align-items:center;gap:8px;margin:8px 0 4px 0;">
                        <span style="font-size:20px;font-weight:700;line-height:1;">摄像词</span>
                        <button id="save-camera-record-btn" style="background:#2b6cb0;color:#fff;border:none;border-radius:4px;padding:4px 8px;cursor:pointer;font-size:12px;">保存</button>
                        <div id="camera-record-list" style="display:flex;gap:6px;flex-wrap:wrap;"></div>
                      </div>
                      <textarea id="camera-prompt-output" rows="2" readonly style="width:100%;background:#1a202c;border:1px solid #4a5568;border-radius:4px;color:#e2e8f0;padding:6px;"></textarea>
                    </div>
                  </div>
                </div>
              </div>
            </div>
        `;
        this.modalElement.appendChild(root);

        const state = {
            scene: null, camera: null, renderer: null, controls: null, imagePlane: null,
            light: null, ambientLight: null, textureLoader: null, is3D: false, currentImage: null, sourceNode: null, cameraDistance: 8
        };

        const q = (sel) => root.querySelector(sel);
        const canvasContainer = q("#canvas-container");
        const canvasPlaceholder = q("#canvas-placeholder");
        const editorCanvas = q("#editor-canvas");

        const closeBtn = q("#comfyui-3d-editor-close");
        closeBtn.addEventListener("click", () => this.close());

        const lightIntensity = q("#light-intensity");
        const intensityValue = q("#intensity-value");
        const lightColorPicker = q("#light-color-picker");
        const lightX = q("#light-x");
        const lightXValue = q("#light-x-value");
        const lightY = q("#light-y");
        const lightYValue = q("#light-y-value");
        const lightZ = q("#light-z");
        const lightZValue = q("#light-z-value");
        const ambientLightSlider = q("#ambient-light");
        const ambientValue = q("#ambient-value");
        const lightPresetSelect = q("#light-preset-select");
        const resetLightBtn = q("#reset-light-btn");
        const rotationX = q("#rotation-x");
        const rotationY = q("#rotation-y");
        const cameraZoom = q("#camera-zoom");
        const rotationXValue = q("#rotation-x-value");
        const rotationYValue = q("#rotation-y-value");
        const cameraZoomValue = q("#camera-zoom-value");
        const cameraPresetSelect = q("#camera-preset-select");
        const resetCameraBtn = q("#reset-camera-btn");

        const renderBtn = q("#render-btn");
        const applyBtn = q("#apply-node-btn");
        const saveLightRecordBtn = q("#save-light-record-btn");
        const saveCameraRecordBtn = q("#save-camera-record-btn");
        const lightRecordList = q("#light-record-list");
        const cameraRecordList = q("#camera-record-list");
        const lightPromptOutput = q("#light-prompt-output");
        const cameraPromptOutput = q("#camera-prompt-output");
        const lightPromptHistory = [];
        const cameraPromptHistory = [];

        const createTextSprite = (message) => {
            const canvas = document.createElement("canvas");
            const ctx = canvas.getContext("2d");
            canvas.width = 256; canvas.height = 64;
            ctx.font = "Bold 40px Arial";
            ctx.fillStyle = "#fff";
            ctx.textAlign = "center";
            ctx.fillText(message, 128, 40);
            const texture = new THREE.CanvasTexture(canvas);
            const mat = new THREE.SpriteMaterial({ map: texture });
            const sprite = new THREE.Sprite(mat);
            sprite.scale.set(4, 1, 1);
            return sprite;
        };

        const getImageInputIndex = () => this.currentNode?.inputs?.findIndex((input) => input?.name === "image") ?? -1;

        const getConnectedImageSourceNode = () => {
            const imageInputIndex = getImageInputIndex();
            if (imageInputIndex < 0 || !this.currentNode?.inputs?.[imageInputIndex]) {
                return null;
            }
            const linkId = this.currentNode.inputs[imageInputIndex].link;
            if (linkId == null || !app.graph?.links) {
                return null;
            }
            const linkInfo = app.graph.links[linkId];
            if (!linkInfo?.origin_id) {
                return null;
            }
            return app.graph.getNodeById?.(linkInfo.origin_id) ?? app.graph._nodes?.find((node) => node.id === linkInfo.origin_id) ?? null;
        };

        const getImageWidgetValue = (node) => {
            return node?.widgets?.find((w) => w?.name === "image")?.value ?? null;
        };

        const buildViewUrlFromAnnotatedName = (annotatedName) => {
            const raw = String(annotatedName ?? "").trim();
            if (!raw) return null;
            const typeMatch = raw.match(/\[([^\]]+)\]\s*$/);
            const imageType = typeMatch?.[1] || "input";
            const unwrapped = raw.replace(/\s*\[[^\]]+\]\s*$/, "").replace(/\\/g, "/");
            const slashIndex = unwrapped.lastIndexOf("/");
            const filename = slashIndex >= 0 ? unwrapped.slice(slashIndex + 1) : unwrapped;
            const subfolder = slashIndex >= 0 ? unwrapped.slice(0, slashIndex) : "";
            const path = `/view?filename=${encodeURIComponent(filename)}&type=${encodeURIComponent(imageType)}&subfolder=${encodeURIComponent(subfolder)}`;
            return api.apiURL ? api.apiURL(path) : path;
        };

        const getNodePreviewUrl = (node) => {
            const previewImage = node?.imgs?.[0];
            if (!previewImage) {
                return null;
            }
            if (typeof previewImage === "string") {
                return previewImage;
            }
            if (previewImage?.src) {
                return previewImage.src;
            }
            return null;
        };

        const resolveInitialImageSource = () => {
            const sourceNode = getConnectedImageSourceNode();
            const sourcePreview = getNodePreviewUrl(sourceNode);
            if (sourceNode && sourcePreview) {
                return { node: sourceNode, url: sourcePreview };
            }

            const sourceWidgetValue = getImageWidgetValue(sourceNode);
            if (sourceNode && sourceWidgetValue) {
                const widgetUrl = buildViewUrlFromAnnotatedName(sourceWidgetValue);
                if (widgetUrl) {
                    return { node: sourceNode, url: widgetUrl };
                }
            }

            const currentPreview = getNodePreviewUrl(this.currentNode);
            if (currentPreview) {
                return { node: this.currentNode, url: currentPreview };
            }

            return null;
        };

        const loadImageFromUrl = (url) => {
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                state.currentImage = url;
                if (!state.textureLoader) {
                    initThree();
                } else if (!state.is3D) {
                    startThree();
                } else {
                    loadImageToPlane();
                }
            };
            img.onerror = () => alert("Failed to load node image.");
            img.src = url;
        };

        const loadImageFromNode = () => {
            const initial = resolveInitialImageSource();
            if (!initial?.url) {
                return;
            }
            state.sourceNode = initial.node;
            loadImageFromUrl(initial.url);
        };

        const updateLighting = () => {
            if (!state.light || !state.ambientLight) return;
            const intensity = parseFloat(lightIntensity.value);
            state.light.intensity = intensity;
            intensityValue.textContent = intensity.toFixed(1);
            const x = parseFloat(lightX.value), y = parseFloat(lightY.value), z = parseFloat(lightZ.value);
            state.light.position.set(x, y, z);
            lightXValue.textContent = x.toFixed(1);
            lightYValue.textContent = y.toFixed(1);
            lightZValue.textContent = z.toFixed(1);
            const ambient = parseFloat(ambientLightSlider.value);
            state.ambientLight.intensity = ambient;
            ambientValue.textContent = ambient.toFixed(1);
            if (this.lightHelper) {
                this.lightHelper.position.set(x, y, z);
                const colorHex = this.currentLightColor || lightColorPicker?.value || "#ffffff";
                this.lightHelper.material.color.set(colorHex);
            }
        };

        const updateRotation = () => {
            if (!this.camera) return;
            const vertical = parseFloat(rotationX.value);
            const rotate = parseFloat(rotationY.value);
            state.cameraDistance = parseFloat(cameraZoom.value);
            rotationXValue.textContent = `${vertical}°`;
            rotationYValue.textContent = `${rotate}°`;
            cameraZoomValue.textContent = state.cameraDistance.toFixed(1);
            const target = state.imagePlane?.position || new THREE.Vector3(0, 0, 0);
            const pitch = THREE.MathUtils.degToRad(vertical);
            const yaw = THREE.MathUtils.degToRad(rotate);
            const cosPitch = Math.cos(pitch);
            const relX = state.cameraDistance * Math.sin(yaw) * cosPitch;
            const relY = state.cameraDistance * Math.sin(pitch);
            const relZ = state.cameraDistance * Math.cos(yaw) * cosPitch;
            this.camera.position.set(target.x + relX, target.y + relY, target.z + relZ);
            this.camera.lookAt(target);
            if (state.controls) {
                state.controls.target.copy(target);
                state.controls.update();
            }
            if (this.cameraHelper) {
                this.cameraHelper.position.copy(this.camera.position);
            }
        };

        const generateLightingPrompt = () => {
            const lx = parseFloat(lightX.value), ly = parseFloat(lightY.value), lz = parseFloat(lightZ.value);
            const intensity = parseFloat(lightIntensity.value);
            const colorHex = this.currentLightColor || "#ffffff";
            let az = Math.atan2(lx, lz) * 180 / Math.PI; if (az < 0) az += 360;
            let e = Math.atan2(ly, Math.sqrt(lx*lx + lz*lz)) * 180 / Math.PI;
            let pos_desc = "";
            if (az >= 337.5 || az < 22.5) pos_desc = "light source in front";
            else if (az < 67.5) pos_desc = "light source from the front-right";
            else if (az < 112.5) pos_desc = "light source from the right";
            else if (az < 157.5) pos_desc = "light source from the back-right";
            else if (az < 202.5) pos_desc = "light source from behind";
            else if (az < 247.5) pos_desc = "light source from the back-left";
            else if (az < 292.5) pos_desc = "light source from the left";
            else pos_desc = "light source from the front-left";
            let elev_desc = "";
            if (e >= -90 && e < -30) elev_desc = "uplighting, light source positioned below the character, light shining upwards";
            else if (e >= -30 && e < -10) elev_desc = "low-angle light source from below, upward illumination";
            else if (e >= -10 && e < 20) elev_desc = "horizontal level light source";
            else if (e >= 20 && e < 60) elev_desc = "high-angle light source";
            else elev_desc = "overhead top-down light source";
            let scaled_intensity = intensity * 5;
            let int_desc = scaled_intensity < 3 ? "soft" : (scaled_intensity < 7 ? "bright" : "intense");
            const prefix = "SCENE LOCK, FIXED VIEWPOINT, maintaining character consistency and pose. RELIGHTING ONLY: ";
            return `${prefix}${pos_desc}, ${elev_desc}, ${int_desc} colored light (${colorHex}), cinematic relighting`;
        };

        const generateCameraPrompt = () => {
            if (!this.camera) return "";
            const target = state.imagePlane?.position || { x: 0, y: 0, z: 0 };
            const cx = this.camera.position.x - target.x;
            const cy = this.camera.position.y - target.y;
            const cz = this.camera.position.z - target.z;
            let h = Math.atan2(cx, cz) * 180/Math.PI; if (h < 0) h += 360;
            let v = Math.atan2(cy, Math.sqrt(cx*cx + cz*cz)) * 180/Math.PI;
            let zoom = Math.sqrt(cx*cx + cy*cy + cz*cz);
            let hdir = "";
            if (h < 22.5 || h >= 337.5) hdir = "front view";
            else if (h < 67.5) hdir = "front-right quarter view";
            else if (h < 112.5) hdir = "right side view";
            else if (h < 157.5) hdir = "back-right quarter view";
            else if (h < 202.5) hdir = "back view";
            else if (h < 247.5) hdir = "back-left quarter view";
            else if (h < 292.5) hdir = "left side view";
            else hdir = "front-left quarter view";
            let vdir = v < -15 ? "low-angle shot" : (v < 15 ? "eye-level shot" : (v < 45 ? "elevated shot" : "high-angle shot"));
            let dist = zoom < 4 ? "close-up" : (zoom < 8 ? "medium shot" : "wide shot");
            return `<sks> ${hdir} ${vdir} ${dist}`;
        };

        const generateFrontView = () => {
            if (!state.is3D || !this.renderer || !this.scene || !this.camera || !state.imagePlane) return;
            const lightPrompt = generateLightingPrompt();
            const cameraPrompt = generateCameraPrompt();
            lightPromptOutput.value = lightPrompt;
            cameraPromptOutput.value = cameraPrompt;
            applyBtn.disabled = false;
        };

        const renderHistoryChips = (container, history, onDelete) => {
            if (!container) return;
            container.innerHTML = "";
            history.forEach((_, index) => {
                const chip = document.createElement("div");
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.gap = "4px";
                chip.style.background = "#111827";
                chip.style.color = "#e2e8f0";
                chip.style.border = "1px solid #4a5568";
                chip.style.borderRadius = "4px";
                chip.style.padding = "2px 6px";
                chip.style.fontSize = "12px";
                chip.textContent = `${index + 1}`;
                const del = document.createElement("button");
                del.textContent = "×";
                del.style.background = "transparent";
                del.style.color = "#f56565";
                del.style.border = "none";
                del.style.cursor = "pointer";
                del.style.fontSize = "12px";
                del.addEventListener("click", () => onDelete(index));
                chip.appendChild(del);
                container.appendChild(chip);
            });
        };

        const pushPromptRecord = (history, text, container, onDelete) => {
            const value = String(text || "").trim();
            if (!value) return;
            history.push(value);
            renderHistoryChips(container, history, onDelete);
        };

        const mergePromptLines = (history, currentValue) => {
            const merged = [...history];
            const current = String(currentValue || "").trim();
            if (current && !merged.includes(current)) {
                merged.push(current);
            }
            return merged.join("\n");
        };
        const removeHistoryItem = (history, container, index) => {
            history.splice(index, 1);
            renderHistoryChips(container, history, (i) => removeHistoryItem(history, container, i));
        };

        const applyPromptsToNode = () => {
            if (!this.currentNode) return;
            const lightPrompt = mergePromptLines(lightPromptHistory, lightPromptOutput.value);
            const cameraPrompt = mergePromptLines(cameraPromptHistory, cameraPromptOutput.value);
            if (this.currentNode.widgets) {
                const lw = this.currentNode.widgets.find(w => w.name === "lighting_prompt");
                const cw = this.currentNode.widgets.find(w => w.name === "camera_prompt");
                if (lw) lw.value = lightPrompt;
                if (cw) cw.value = cameraPrompt;
            }
            if (app.graph) app.graph.setDirtyCanvas(true, true);
            this.close();
        };

        const defaultLightState = {
            intensity: 1.0, x: 0.0, y: 0.0, z: 3.0, ambient: 0.2, color: "#ffffff"
        };
        const defaultCameraState = {
            rotate: 45, vertical: 0, zoom: 5.0
        };
        const lightPresets = [
            { name: "通用光_正面主柔光", azimuth: 0, elevation: 20, intensity: 3.0, color: "#FFFFFF" },
            { name: "通用光_伦勃朗45°立体光", azimuth: 45, elevation: 45, intensity: 4.0, color: "#FFFFFF" },
            { name: "通用光_右侧轮廓光", azimuth: 90, elevation: 30, intensity: 6.0, color: "#FFFFFF" },
            { name: "通用光_左侧柔和补光", azimuth: 270, elevation: 15, intensity: 2.0, color: "#FFFFFF" },
            { name: "通用光_正后方逆光", azimuth: 180, elevation: 20, intensity: 7.0, color: "#FFFFFF" },
            { name: "通用光_右后侧氛围光", azimuth: 135, elevation: 10, intensity: 3.5, color: "#FFFFFF" },
            { name: "通用光_顶光", azimuth: 0, elevation: 90, intensity: 5.0, color: "#FFFFFF" },
            { name: "通用光_底部仰光", azimuth: 0, elevation: -60, intensity: 4.0, color: "#FFFFFF" },
            { name: "影视光_电影暗调逆光", azimuth: 180, elevation: 0, intensity: 9.0, color: "#333333" },
            { name: "产品光_产品右侧高光", azimuth: 80, elevation: 45, intensity: 6.5, color: "#FFFFFF" },
            { name: "人像光_柔和平光", azimuth: 0, elevation: 10, intensity: 2.5, color: "#FFFFFF" },
            { name: "轮廓光_左后侧轮廓光", azimuth: 225, elevation: 15, intensity: 5.5, color: "#FFFFFF" },
            { name: "风格化光_赛博朋克侧逆光", azimuth: 120, elevation: 10, intensity: 8.0, color: "#0066FF" },
            { name: "风格化光_霓虹轮廓背光", azimuth: 160, elevation: 20, intensity: 9.0, color: "#FF00FF" },
            { name: "风格化光_冷暖对比侧光", azimuth: 60, elevation: 30, intensity: 6.0, color: "#FF9900" },
            { name: "风格化光_哥特暗调底光", azimuth: 0, elevation: -45, intensity: 5.0, color: "#FF4444" },
            { name: "风格化光_科技顶射光", azimuth: 0, elevation: 80, intensity: 7.5, color: "#88CCFF" }
        ];
        const cameraPresets = [
            { name: "通用视角_正面平视中景", rotate: 0, vertical: 0, zoom: 5.0 },
            { name: "通用视角_右前45°平视中景", rotate: 45, vertical: 0, zoom: 5.0 },
            { name: "通用视角_左侧平视全身", rotate: 270, vertical: 0, zoom: 4.0 },
            { name: "通用视角_右后45°平视中景", rotate: 135, vertical: 0, zoom: 5.0 },
            { name: "通用视角_背面平视广角", rotate: 180, vertical: 0, zoom: 3.0 },
            { name: "通用视角_左前45°平视中景", rotate: 315, vertical: 0, zoom: 5.0 },
            { name: "人像视角_正面平视特写", rotate: 0, vertical: 0, zoom: 7.0 },
            { name: "人像视角_右前45°低角度特写", rotate: 45, vertical: -30, zoom: 7.0 },
            { name: "人像视角_正面高角度肖像", rotate: 0, vertical: 30, zoom: 6.0 },
            { name: "产品视角_右前45°高角度广角", rotate: 45, vertical: 45, zoom: 3.0 },
            { name: "产品视角_正面平视产品全景", rotate: 0, vertical: 15, zoom: 2.0 },
            { name: "影视视角_正面极端仰角特写", rotate: 0, vertical: -90, zoom: 7.0 },
            { name: "影视视角_背面鸟瞰全景", rotate: 180, vertical: 60, zoom: 2.0 },
            { name: "影视视角_右侧高角度俯视", rotate: 90, vertical: 75, zoom: 4.0 },
            { name: "风格化视角_正面顶视完全俯视", rotate: 0, vertical: 90, zoom: 5.0 },
            { name: "风格化视角_左后超低角度广角", rotate: 225, vertical: -60, zoom: 3.0 },
            { name: "风格化视角_右侧平视极端特写", rotate: 90, vertical: 0, zoom: 9.0 },
            { name: "场景视角_正面超广角全景", rotate: 0, vertical: 15, zoom: 1.0 },
            { name: "场景视角_背面鸟瞰超广角", rotate: 180, vertical: 80, zoom: 1.0 }
        ];

        const fillPresetSelect = (selectEl, presets, placeholder) => {
            if (!selectEl) return;
            selectEl.innerHTML = "";
            const defaultOption = document.createElement("option");
            defaultOption.value = "";
            defaultOption.textContent = placeholder;
            selectEl.appendChild(defaultOption);
            presets.forEach((preset, index) => {
                const option = document.createElement("option");
                option.value = String(index);
                option.textContent = preset.name;
                selectEl.appendChild(option);
            });
        };

        const applyLightPreset = (preset) => {
            if (!preset) return;
            const radius = 8.0;
            const az = THREE.MathUtils.degToRad(preset.azimuth);
            const el = THREE.MathUtils.degToRad(preset.elevation);
            const cosEl = Math.cos(el);
            const x = radius * Math.sin(az) * cosEl;
            const y = radius * Math.sin(el);
            const z = radius * Math.cos(az) * cosEl;
            lightIntensity.value = String(preset.intensity);
            lightX.value = String(x);
            lightY.value = String(y);
            lightZ.value = String(z);
            lightColorPicker.value = preset.color.toLowerCase();
            this.currentLightColor = lightColorPicker.value;
            updateLighting();
            if (state.light) state.light.color.set(this.currentLightColor);
            if (this.lightHelper) this.lightHelper.material.color.set(this.currentLightColor);
        };

        const applyCameraPreset = (preset) => {
            if (!preset) return;
            rotationY.value = String(preset.rotate);
            rotationX.value = String(preset.vertical);
            cameraZoom.value = String(preset.zoom);
            updateRotation();
        };

        const resetLighting = () => {
            lightIntensity.value = String(defaultLightState.intensity);
            lightX.value = String(defaultLightState.x);
            lightY.value = String(defaultLightState.y);
            lightZ.value = String(defaultLightState.z);
            ambientLightSlider.value = String(defaultLightState.ambient);
            lightColorPicker.value = defaultLightState.color;
            this.currentLightColor = defaultLightState.color;
            if (lightPresetSelect) lightPresetSelect.value = "";
            updateLighting();
            if (state.light) state.light.color.set(this.currentLightColor);
            if (this.lightHelper) this.lightHelper.material.color.set(this.currentLightColor);
        };

        const resetCamera = () => {
            rotationY.value = String(defaultCameraState.rotate);
            rotationX.value = String(defaultCameraState.vertical);
            cameraZoom.value = String(defaultCameraState.zoom);
            if (cameraPresetSelect) cameraPresetSelect.value = "";
            updateRotation();
        };

        const initThree = () => {
            if (!window.THREE) {
                const script = document.createElement("script");
                script.src = "https://cdn.jsdelivr.net/npm/three@0.140.1/build/three.min.js";
                script.onload = () => {
                    const cs = document.createElement("script");
                    cs.src = "https://cdn.jsdelivr.net/npm/three@0.140.1/examples/js/controls/OrbitControls.js";
                    cs.onload = () => {
                        state.textureLoader = new THREE.TextureLoader();
                        startThree();
                    };
                    document.head.appendChild(cs);
                };
                document.head.appendChild(script);
            } else {
                state.textureLoader = new THREE.TextureLoader();
                startThree();
            }
        };

        const startThree = () => {
            if (state.is3D) {
                loadImageToPlane();
                return;
            }
            state.is3D = true;
            canvasPlaceholder.style.display = "none";
            editorCanvas.style.display = "block";
            renderBtn.disabled = false;

            const width = canvasContainer.clientWidth, height = canvasContainer.clientHeight;
            this.scene = state.scene = new THREE.Scene();
            this.scene.background = new THREE.Color(0x1a202c);
            this.camera = state.camera = new THREE.PerspectiveCamera(75, width/height, 0.1, 1000);
            this.camera.position.z = 5;
            this.renderer = state.renderer = new THREE.WebGLRenderer({ canvas: editorCanvas, antialias: true });
            this.renderer.setSize(width, height);
            this.renderer.setPixelRatio(window.devicePixelRatio);
            state.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
            state.controls.enableDamping = true;
            this.scene.add(new THREE.GridHelper(20, 20, 0x4a5568, 0x2d3748));
            const initialLightColor = this.currentLightColor || lightColorPicker?.value || "#ffffff";
            state.light = new THREE.DirectionalLight(initialLightColor, parseFloat(lightIntensity.value));
            state.light.position.set(parseFloat(lightX.value), parseFloat(lightY.value), parseFloat(lightZ.value));
            this.scene.add(state.light);
            state.ambientLight = new THREE.AmbientLight(0xffffff, parseFloat(ambientLightSlider.value));
            this.scene.add(state.ambientLight);
            const lightGeo = new THREE.BoxGeometry(0.5,0.5,0.5);
            const lightMat = new THREE.MeshBasicMaterial({ color: initialLightColor });
            this.lightHelper = new THREE.Mesh(lightGeo, lightMat);
            this.lightHelper.position.copy(state.light.position);
            this.scene.add(this.lightHelper);
            const lightText = createTextSprite("Light");
            lightText.position.y = 0.8;
            this.lightHelper.add(lightText);
            const cameraGeo = new THREE.SphereGeometry(0.28, 24, 24);
            const cameraMat = new THREE.MeshBasicMaterial({ color: 0x63b3ed });
            this.cameraHelper = new THREE.Mesh(cameraGeo, cameraMat);
            this.cameraHelper.position.copy(this.camera.position);
            this.scene.add(this.cameraHelper);
            const cameraText = createTextSprite("Camera");
            cameraText.position.y = 0.8;
            this.cameraHelper.add(cameraText);
            if (state.controls) {
                state.controls.target.set(0, 0, 0);
            }
            updateRotation();
            loadImageToPlane();
            animate();
        };

        const loadImageToPlane = () => {
            if (!state.textureLoader || !state.currentImage) return;
            state.textureLoader.load(state.currentImage, (texture) => {
                const aspect = texture.image.width / texture.image.height;
                const geometry = new THREE.PlaneGeometry(5 * aspect, 5);
                const material = new THREE.MeshStandardMaterial({ map: texture, side: THREE.DoubleSide, roughness: 0.5, metalness: 0.1 });
                if (state.imagePlane) this.scene.remove(state.imagePlane);
                state.imagePlane = new THREE.Mesh(geometry, material);
                this.scene.add(state.imagePlane);
                updateRotation();
            });
        };

        const animate = () => {
            if (!state.is3D) return;
            requestAnimationFrame(animate);
            if (state.controls) state.controls.update();
            if (this.cameraHelper && this.camera) {
                this.cameraHelper.position.copy(this.camera.position);
            }
            if (this.renderer && this.scene && this.camera) this.renderer.render(this.scene, this.camera);
        };

        const onResize = () => {
            if (!state.is3D || !this.camera || !this.renderer) return;
            const width = canvasContainer.clientWidth, height = canvasContainer.clientHeight;
            this.camera.aspect = width / height;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(width, height);
        };
        window.addEventListener("resize", onResize);

        this.currentLightColor = lightColorPicker?.value || "#ffffff";
        lightColorPicker?.addEventListener("input", () => {
            const color = lightColorPicker.value;
            this.currentLightColor = color;
            if (state.light) state.light.color.set(color);
            if (this.lightHelper) this.lightHelper.material.color.set(color);
        });
        fillPresetSelect(lightPresetSelect, lightPresets, "灯光预设");
        fillPresetSelect(cameraPresetSelect, cameraPresets, "摄像机预设");
        lightIntensity.addEventListener("input", updateLighting);
        lightX.addEventListener("input", updateLighting);
        lightY.addEventListener("input", updateLighting);
        lightZ.addEventListener("input", updateLighting);
        ambientLightSlider.addEventListener("input", updateLighting);
        rotationX.addEventListener("input", updateRotation);
        rotationY.addEventListener("input", updateRotation);
        cameraZoom.addEventListener("input", updateRotation);
        lightPresetSelect?.addEventListener("change", () => {
            const index = parseInt(lightPresetSelect.value, 10);
            if (Number.isNaN(index)) return;
            applyLightPreset(lightPresets[index]);
        });
        cameraPresetSelect?.addEventListener("change", () => {
            const index = parseInt(cameraPresetSelect.value, 10);
            if (Number.isNaN(index)) return;
            applyCameraPreset(cameraPresets[index]);
        });
        resetLightBtn?.addEventListener("click", resetLighting);
        resetCameraBtn?.addEventListener("click", resetCamera);

        renderBtn.addEventListener("click", generateFrontView);
        applyBtn.addEventListener("click", applyPromptsToNode);
        saveLightRecordBtn?.addEventListener("click", () => {
            pushPromptRecord(lightPromptHistory, lightPromptOutput.value, lightRecordList, (index) => removeHistoryItem(lightPromptHistory, lightRecordList, index));
        });
        saveCameraRecordBtn?.addEventListener("click", () => {
            pushPromptRecord(cameraPromptHistory, cameraPromptOutput.value, cameraRecordList, (index) => removeHistoryItem(cameraPromptHistory, cameraRecordList, index));
        });

        loadImageFromNode();
    }

    close() {
        if (this.renderer) {
            this.renderer.dispose();
            this.renderer = null;
        }
        this.scene = null;
        this.camera = null;
        this.lightHelper = null;
        this.cameraHelper = null;
        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }
        super.close();
    }
}

app.registerExtension({
    name: "comfyui.APT_3dimageeditor",
    version: "1.0.0",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "text_mulAngle") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                if (!this.widgets || !this.widgets.find(w => w.name === "Open 3D Editor")) {
                    this.addWidget("button", "Open 3D Editor", "open", () => {
                        const modal = ThreeDEditorModal.getInstance();
                        modal.setNode(this);
                        modal.show();
                    });
                    if (this.computeSize) this.setSize(this.computeSize());
                }
                return r;
            };
        }
    },
    async setup() {
        if (app.graph && app.graph._nodes) {
            app.graph._nodes.forEach(node => {
                if (node.constructor.nodeData && node.constructor.nodeData.name === "text_mulAngle") {
                    if (!node.widgets || !node.widgets.find(w => w.name === "Open 3D Editor")) {
                        node.addWidget("button", "Open 3D Editor", "open", () => {
                            const modal = ThreeDEditorModal.getInstance();
                            modal.setNode(node);
                            modal.show();
                        });
                        if (node.computeSize) node.setSize(node.computeSize());
                    }
                }
            });
        }
    },
    init() {
        console.log("3D Image Editor Extension initialized");
    }
});

const style = document.createElement("style");
style.innerHTML = `
  .comfy-modal-layout {
    display:flex;
    flex-direction:column;
    width:85vw;
    height:85vh;
    max-width:100vw;
    max-height:100vh;
    padding:0;
    z-index:9999;
  }
`;
document.head.appendChild(style);





