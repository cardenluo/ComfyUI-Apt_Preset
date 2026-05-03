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
        // Keep history only for the current modal session.
        this.promptHistory = [];
        this.currentPalette = [];

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

class ColorPaletteModal extends ComfyDialog {
    static instance = null;
    static getInstance() {
        if (!ColorPaletteModal.instance) {
            ColorPaletteModal.instance = new ColorPaletteModal();
        }
        return ColorPaletteModal.instance;
    }

    constructor() {
        super();
        this.currentNode = null;
        this.modalElement = null;
        this.currentImage = null;
        this.currentPalette = [];
        this.promptHistory = [];
        this.currentPaletteNote = "";
    }

    setNode(node) {
        this.currentNode = node;
    }

    _getImageInputIndex() {
        return this.currentNode?.inputs?.findIndex((input) => input?.name === "image") ?? -1;
    }

    _getConnectedImageSourceNode() {
        const imageInputIndex = this._getImageInputIndex();
        if (imageInputIndex < 0 || !this.currentNode?.inputs?.[imageInputIndex]) return null;
        const linkId = this.currentNode.inputs[imageInputIndex].link;
        if (linkId == null || !app.graph?.links) return null;
        const linkInfo = app.graph.links[linkId];
        if (!linkInfo?.origin_id) return null;
        return app.graph.getNodeById?.(linkInfo.origin_id) ?? app.graph._nodes?.find((node) => node.id === linkInfo.origin_id) ?? null;
    }

    _getImageWidgetValue(node) {
        return node?.widgets?.find((w) => w?.name === "image")?.value ?? null;
    }

    _buildViewUrlFromAnnotatedName(annotatedName) {
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
    }

    _getNodePreviewUrl(node) {
        const previewImage = node?.imgs?.[0];
        if (!previewImage) return null;
        if (typeof previewImage === "string") return previewImage;
        if (previewImage?.src) return previewImage.src;
        return null;
    }

    _resolveInitialImageSource() {
        const sourceNode = this._getConnectedImageSourceNode();
        const sourcePreview = this._getNodePreviewUrl(sourceNode);
        if (sourceNode && sourcePreview) return { node: sourceNode, url: sourcePreview };
        const sourceWidgetValue = this._getImageWidgetValue(sourceNode);
        if (sourceNode && sourceWidgetValue) {
            const widgetUrl = this._buildViewUrlFromAnnotatedName(sourceWidgetValue);
            if (widgetUrl) return { node: sourceNode, url: widgetUrl };
        }
        const currentPreview = this._getNodePreviewUrl(this.currentNode);
        if (currentPreview) return { node: this.currentNode, url: currentPreview };
        return null;
    }

    _rgbToHex(r, g, b) {
        const toHex = (v) => {
            const n = Math.max(0, Math.min(255, Math.round(v)));
            return n.toString(16).padStart(2, "0");
        };
        return `#${toHex(r)}${toHex(g)}${toHex(b)}`;
    }

    _paletteToPrompt(hexList) {
        const colors = (hexList || []).join(", ");
        return `Color scheme strictly follows: ${colors}, overall unified and harmonious tones, low saturation, soft and natural colors, no messy or abrupt colors, clear hierarchy of primary and secondary colors in the image, advanced texture, accurate color reproduction.`;
    }

    _extractPaletteFromImage(imageEl, count) {
        const targetCount = Math.max(3, Math.min(10, Number(count) || 6));
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const maxDim = 220;
        let w = imageEl.naturalWidth || imageEl.width || 256;
        let h = imageEl.naturalHeight || imageEl.height || 256;
        if (w > h && w > maxDim) {
            h = Math.round(h * (maxDim / w));
            w = maxDim;
        } else if (h >= w && h > maxDim) {
            w = Math.round(w * (maxDim / h));
            h = maxDim;
        }
        canvas.width = Math.max(1, w);
        canvas.height = Math.max(1, h);
        ctx.drawImage(imageEl, 0, 0, canvas.width, canvas.height);
        const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
        const bins = new Map();
        for (let i = 0; i < data.length; i += 4) {
            const a = data[i + 3];
            if (a < 16) continue;
            const r = data[i] >> 4;
            const g = data[i + 1] >> 4;
            const b = data[i + 2] >> 4;
            const key = `${r},${g},${b}`;
            bins.set(key, (bins.get(key) || 0) + 1);
        }
        const top = [...bins.entries()]
            .sort((a, b) => b[1] - a[1])
            .slice(0, targetCount)
            .map(([k]) => {
                const [r, g, b] = k.split(",").map((v) => Number(v) * 16 + 8);
                return this._rgbToHex(r, g, b);
            });
        return top;
    }

    show() {
        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }

        this.modalElement = $el("div.comfy-modal", { parent: document.body }, [
            $el("div.comfy-modal-content", {}, []),
        ]);
        this.modalElement.classList.add("comfy-modal-layout");
        this.modalElement.style.width = "85vw";
        this.modalElement.style.height = "85vh";
        this.modalElement.style.maxWidth = "100vw";
        this.modalElement.style.maxHeight = "100vh";

        const root = document.createElement("div");
        root.style.display = "flex";
        root.style.flexDirection = "column";
        root.style.width = "100%";
        root.style.height = "100%";
        root.style.background = "#1a202c";
        root.style.color = "#fff";
        root.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #4a5568;padding:10px 12px;">
                <h2 style="margin:0;font-size:16px;font-weight:700;">Color Palette</h2>
                <button id="color-palette-close" style="background:none;border:none;color:#a0aec0;font-size:20px;cursor:pointer;">×</button>
            </div>
            <div style="flex:1;display:flex;gap:12px;padding:12px;overflow:hidden;min-height:0;">
                <div style="flex:0 0 34%;display:flex;flex-direction:column;gap:10px;overflow:auto;min-height:0;background:#2d3748;border-radius:8px;padding:10px;">
                    <div style="font-size:14px;font-weight:700;">方式一：提取图片颜色 - 基于图片</div>
                    <div id="cp-image-wrap" style="background:#111827;border:1px solid #4a5568;border-radius:6px;padding:6px;min-height:150px;display:flex;align-items:center;justify-content:center;">
                        <div id="cp-image-tip" style="color:#94a3b8;font-size:12px;">等待读取节点图片</div>
                        <img id="cp-image-preview" style="max-width:100%;max-height:200px;display:none;border-radius:4px;" />
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <label style="font-size:12px;">数量</label>
                        <input id="cp-color-count" type="range" min="3" max="10" step="1" value="6" style="flex:1;" />
                        <span id="cp-color-count-val" style="font-size:12px;">6</span>
                    </div>
                    <button id="cp-extract-btn" style="background:#2b6cb0;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">提取图片颜色</button>

                    <div style="font-size:14px;font-weight:700;margin-top:6px;">方式二：AI 生成配色方案 - 基于图片</div>
                    <select id="cp-scene-select" style="background:#111827;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:6px 8px;">
                        <option value="web">网页设计</option>
                        <option value="app">移动应用</option>
                        <option value="print">印刷设计</option>
                        <option value="seasonal">季节主题</option>
                        <option value="mood">情绪表达</option>
                    </select>
                    <button id="cp-ai-generate-btn" style="background:#805ad5;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">AI生成配色方案</button>

                    <div style="font-size:14px;font-weight:700;margin-top:6px;">方式三：预设配色</div>
                    <select id="cp-preset-select" style="background:#111827;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:6px 8px;"></select>
                    <button id="cp-apply-preset-btn" style="background:#2f855a;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">采用预设配色</button>

                    <div style="font-size:14px;font-weight:700;margin-top:6px;">方式四：动态衍生专业配色 - 基于图片</div>
                    <select id="cp-pro-derive-select" style="background:#111827;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:6px 8px;">
                        <option value="split_complementary">分裂互补色方案</option>
                        <option value="rectangle_tetradic">四色矩形（双互补）方案</option>
                        <option value="square_quadratic">正方形（四分色）方案</option>
                        <option value="mono_luma">渐变明度单色方案</option>
                        <option value="cool_warm_balance">冷暖平衡方案</option>
                        <option value="morandi_low_sat">低饱和莫兰迪方案</option>
                    </select>
                    <button id="cp-pro-derive-btn" style="background:#0ea5e9;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">动态衍生专业配色</button>
                    <div id="cp-pro-derive-note" style="font-size:12px;color:#93c5fd;line-height:1.4;"></div>
                </div>

                <div style="flex:1;display:flex;flex-direction:column;gap:10px;min-height:0;">
                    <div style="background:#2d3748;border-radius:8px;padding:10px;">
                        <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:8px;">
                            <div style="font-size:14px;font-weight:700;">当前配色</div>
                            <div id="cp-current-hex" style="font-size:12px;color:#94a3b8;"></div>
                        </div>
                        <div id="cp-swatch-row" style="display:flex;gap:8px;flex-wrap:wrap;"></div>
                    </div>
                    <div style="background:#2d3748;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:8px;min-height:0;">
                        <div style="display:flex;gap:8px;">
                            <button id="cp-generate-prompt-btn" style="background:#d69e2e;color:#111827;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">生成 color_prompt</button>
                            <button id="cp-save-prompt-btn" style="background:#2b6cb0;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">保存一条</button>
                            <button id="cp-apply-node-btn" style="background:#38a169;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">应用到节点</button>
                        </div>
                        <textarea id="cp-prompt-output" rows="5" style="width:100%;background:#111827;border:1px solid #4a5568;border-radius:4px;color:#e2e8f0;padding:8px;"></textarea>
                        <div style="font-size:14px;font-weight:700;">已保存列表</div>
                        <div id="cp-history-list" style="display:flex;gap:6px;flex-wrap:wrap;max-height:90px;overflow:auto;"></div>
                    </div>
                </div>
            </div>
        `;

        this.modalElement.appendChild(root);
        const q = (s) => root.querySelector(s);
        const closeBtn = q("#color-palette-close");
        const imageTip = q("#cp-image-tip");
        const imagePreview = q("#cp-image-preview");
        const countSlider = q("#cp-color-count");
        const countVal = q("#cp-color-count-val");
        const extractBtn = q("#cp-extract-btn");
        const sceneSelect = q("#cp-scene-select");
        const aiBtn = q("#cp-ai-generate-btn");
        const presetSelect = q("#cp-preset-select");
        const applyPresetBtn = q("#cp-apply-preset-btn");
        const proDeriveSelect = q("#cp-pro-derive-select");
        const proDeriveBtn = q("#cp-pro-derive-btn");
        const proDeriveNote = q("#cp-pro-derive-note");
        const swatchRow = q("#cp-swatch-row");
        const currentHex = q("#cp-current-hex");
        const generatePromptBtn = q("#cp-generate-prompt-btn");
        const savePromptBtn = q("#cp-save-prompt-btn");
        const applyNodeBtn = q("#cp-apply-node-btn");
        const promptOutput = q("#cp-prompt-output");
        const historyList = q("#cp-history-list");

        const presetMap = {
            "经典蓝": ["#1976D2", "#2196F3", "#42A5F5", "#67B7F5", "#93CDF5"],
            "薄荷绿": ["#1DE9B6", "#30DBB5", "#4DD0B6", "#64CCC5", "#80CBC4"],
            "珊瑚红": ["#FF7F50", "#FF8C69", "#FF997A", "#FFA58C", "#FFB2A0"],
            "紫罗兰": ["#8E44AD", "#9B59B6", "#AB66C4", "#BB78D3", "#CE91DB"],
            "向日葵": ["#FFC107", "#FFCD40", "#FFD770", "#FFE0A5", "#FFE9D8"],
            "高级灰": ["#424242", "#616161", "#808080", "#9E9E9E", "#BDBDBD"],
            "马卡龙粉": ["#FFB6C1", "#FFC3CD", "#FFCFD8", "#FFDBE2", "#FFE7EC"],
            "天空蓝": ["#ADD8E6", "#BBDFFB", "#C9E2FF", "#D7EBFA", "#E5F2FF"],
            "商务蓝": ["#003366", "#0055AA", "#0077CC", "#358CCC", "#6699CC"],
            "创意紫": ["#800080", "#8FBC8F", "#D2691E", "#4682B4", "#DC143C"],
            "活力橙": ["#FF4500", "#FF6347", "#FF7F50", "#FFA500", "#FFC107"],
            "单色黑": ["#000000", "#333333", "#666666", "#999999", "#CCCCCC"],
            "抹茶绿": ["#66BB6A", "#81C784", "#9CCC65", "#AED581", "#C8E699"],
            "玫瑰金": ["#E8BEB3", "#ECC9B0", "#F1D3B5", "#F4DDB2", "#F8E7D1"],
            "海洋蓝": ["#008080", "#009696", "#00B2B2", "#40BFBF", "#80CECE"],
            "薰衣草": ["#B5B5FF", "#C2C2FF", "#CFCFFF", "#DCDCFF", "#E9E9FF"],
            "秋叶橙": ["#D2691E", "#DE883C", "#E99C57", "#F1AE72", "#F7BF8F"],
            "科技蓝": ["#0070C0", "#118AD1", "#21A3E2", "#41B6FF", "#72BFF9"],
            "彩虹色": ["#FF0000", "#FF7F00", "#FFFF00", "#00FF00", "#0000FF"],
            "樱花粉": ["#FFC1CC", "#FFCECF", "#FFDBE5", "#FFE8F0", "#FFF3F7"],
            "珊瑚金": ["#DEB373", "#AC4E13", "#986E63", "#372F39", "#B48A6A"],
        };
        Object.keys(presetMap).forEach((name) => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            presetSelect.appendChild(opt);
        });

        const sceneMap = {
            web: ["#1f2937", "#3b82f6", "#93c5fd", "#f8fafc"],
            app: ["#0ea5e9", "#22c55e", "#e2e8f0", "#1e293b"],
            print: ["#1f2937", "#d97706", "#dc2626", "#f3f4f6"],
            seasonal: ["#8b5cf6", "#22c55e", "#f59e0b", "#ef4444"],
            mood: ["#372f39", "#deb373", "#ac4e13", "#986e63"],
        };

        const hexToRgb = (hex) => {
            const m = String(hex || "").trim().match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
            if (!m) return null;
            return {
                r: parseInt(m[1], 16),
                g: parseInt(m[2], 16),
                b: parseInt(m[3], 16),
            };
        };
        const rgbToHsl = (r, g, b) => {
            r /= 255; g /= 255; b /= 255;
            const max = Math.max(r, g, b), min = Math.min(r, g, b);
            let h, s, l = (max + min) / 2;
            if (max === min) {
                h = s = 0;
            } else {
                const d = max - min;
                s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
                switch (max) {
                    case r: h = (g - b) / d + (g < b ? 6 : 0); break;
                    case g: h = (b - r) / d + 2; break;
                    default: h = (r - g) / d + 4; break;
                }
                h /= 6;
            }
            return { h: Math.round(h * 360), s: Math.round(s * 100), l: Math.round(l * 100) };
        };
        const hslToRgb = (h, s, l) => {
            h /= 360; s /= 100; l /= 100;
            let r, g, b;
            if (s === 0) {
                r = g = b = l;
            } else {
                const hue2rgb = (p, q, t) => {
                    if (t < 0) t += 1;
                    if (t > 1) t -= 1;
                    if (t < 1 / 6) return p + (q - p) * 6 * t;
                    if (t < 1 / 2) return q;
                    if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
                    return p;
                };
                const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                const p = 2 * l - q;
                r = hue2rgb(p, q, h + 1 / 3);
                g = hue2rgb(p, q, h);
                b = hue2rgb(p, q, h - 1 / 3);
            }
            return { r: Math.round(r * 255), g: Math.round(g * 255), b: Math.round(b * 255) };
        };
        const generateComplementaryScheme = (hex) => {
            const rgb = hexToRgb(hex);
            if (!rgb) return [];
            const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
            const c = hslToRgb((hsl.h + 180) % 360, hsl.s, hsl.l);
            return [hex.toUpperCase(), this._rgbToHex(c.r, c.g, c.b).toUpperCase()];
        };
        const generateTriadicScheme = (hex) => {
            const rgb = hexToRgb(hex);
            if (!rgb) return [];
            const hsl = rgbToHsl(rgb.r, rgb.g, rgb.b);
            const c1 = hslToRgb((hsl.h + 120) % 360, hsl.s, hsl.l);
            const c2 = hslToRgb((hsl.h + 240) % 360, hsl.s, hsl.l);
            return [
                hex.toUpperCase(),
                this._rgbToHex(c1.r, c1.g, c1.b).toUpperCase(),
                this._rgbToHex(c2.r, c2.g, c2.b).toUpperCase(),
            ];
        };
        const mergeUniqueHex = (...groups) => {
            const out = [];
            const seen = new Set();
            groups.flat().forEach((h) => {
                const v = String(h || "").trim().toUpperCase();
                if (!/^#[0-9A-F]{6}$/.test(v)) return;
                if (seen.has(v)) return;
                seen.add(v);
                out.push(v);
            });
            return out;
        };

        const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
        const hslToHex = (h, s, l) => {
            const rgb = hslToRgb(((h % 360) + 360) % 360, clamp(s, 0, 100), clamp(l, 0, 100));
            return this._rgbToHex(rgb.r, rgb.g, rgb.b).toUpperCase();
        };
        const shiftHue = (h, delta) => ((h + delta) % 360 + 360) % 360;
        const extractMainColorHex = () => {
            if (this.currentImage) {
                const extracted = this._extractPaletteFromImage(this.currentImage, 1);
                if (extracted?.[0]) {
                    return String(extracted[0]).toUpperCase();
                }
            }
            if (this.currentPalette?.[0]) {
                return String(this.currentPalette[0]).toUpperCase();
            }
            return "#6B7280";
        };
        const proSchemeMap = {
            split_complementary: {
                name: "分裂互补色方案",
                copy: "对比克制、张力适中，比互补色更百搭不刺眼",
                scene: "适配场景：海报、APP 界面、品牌主视觉",
                build: (hsl) => {
                    const c1 = hslToHex(hsl.h, hsl.s, hsl.l);
                    const c2 = hslToHex(shiftHue(hsl.h, 150), Math.max(40, hsl.s), hsl.l);
                    const c3 = hslToHex(shiftHue(hsl.h, 210), Math.max(40, hsl.s), hsl.l);
                    const c4 = hslToHex(shiftHue(hsl.h, 0), Math.max(22, hsl.s - 28), clamp(hsl.l + 18, 0, 100));
                    return [c1, c2, c3, c4];
                },
            },
            rectangle_tetradic: {
                name: "四色矩形（双互补）方案",
                copy: "层次饱满、色彩丰富，活泼又不失平衡",
                scene: "适配场景：插画、电商详情页、活动专题",
                build: (hsl) => [
                    hslToHex(shiftHue(hsl.h, 0), hsl.s, hsl.l),
                    hslToHex(shiftHue(hsl.h, 30), hsl.s, clamp(hsl.l + 5, 0, 100)),
                    hslToHex(shiftHue(hsl.h, 180), clamp(hsl.s - 6, 0, 100), hsl.l),
                    hslToHex(shiftHue(hsl.h, 210), clamp(hsl.s - 8, 0, 100), clamp(hsl.l + 5, 0, 100)),
                ],
            },
            square_quadratic: {
                name: "正方形（四分色）方案",
                copy: "分布均衡、动感协调，视觉节奏感极佳",
                scene: "适配场景：Banner、文创、多区块排版设计",
                build: (hsl) => [
                    hslToHex(shiftHue(hsl.h, 0), hsl.s, hsl.l),
                    hslToHex(shiftHue(hsl.h, 90), clamp(hsl.s - 6, 0, 100), clamp(hsl.l + 4, 0, 100)),
                    hslToHex(shiftHue(hsl.h, 180), hsl.s, hsl.l),
                    hslToHex(shiftHue(hsl.h, 270), clamp(hsl.s - 6, 0, 100), clamp(hsl.l + 4, 0, 100)),
                ],
            },
            mono_luma: {
                name: "渐变明度单色方案",
                copy: "极致统一、极简高级，纵深层次干净",
                scene: "适配场景：商务官网、极简 UI、背景渐变",
                build: (hsl) => [
                    hslToHex(hsl.h, clamp(hsl.s + 4, 0, 100), clamp(hsl.l + 26, 0, 100)),
                    hslToHex(hsl.h, hsl.s, clamp(hsl.l + 12, 0, 100)),
                    hslToHex(hsl.h, hsl.s, hsl.l),
                    hslToHex(hsl.h, clamp(hsl.s + 2, 0, 100), clamp(hsl.l - 12, 0, 100)),
                    hslToHex(hsl.h, clamp(hsl.s + 4, 0, 100), clamp(hsl.l - 24, 0, 100)),
                ],
            },
            cool_warm_balance: {
                name: "冷暖平衡方案",
                copy: "冷暖中和、质感柔和，长时间观看无视觉压迫",
                scene: "适配场景：长页面、阅读类界面、产品详情",
                build: (hsl) => [
                    hslToHex(hsl.h, clamp(hsl.s - 6, 0, 100), hsl.l),
                    hslToHex(shiftHue(hsl.h, -24), clamp(hsl.s - 8, 0, 100), clamp(hsl.l + 8, 0, 100)),
                    hslToHex(shiftHue(hsl.h, 24), clamp(hsl.s - 8, 0, 100), clamp(hsl.l + 8, 0, 100)),
                    hslToHex(shiftHue(hsl.h, 180), clamp(hsl.s - 24, 0, 100), clamp(hsl.l + 4, 0, 100)),
                    hslToHex(shiftHue(hsl.h, 180), clamp(hsl.s - 36, 0, 100), clamp(hsl.l + 18, 0, 100)),
                ],
            },
            morandi_low_sat: {
                name: "低饱和莫兰迪方案",
                copy: "温柔高级、低对比度，氛围感拉满",
                scene: "适配场景：家居、美妆、生活方式品牌视觉",
                build: (hsl) => [
                    hslToHex(hsl.h, clamp(hsl.s * 0.42, 8, 36), clamp(hsl.l + 20, 0, 100)),
                    hslToHex(shiftHue(hsl.h, -18), clamp(hsl.s * 0.36, 6, 32), clamp(hsl.l + 10, 0, 100)),
                    hslToHex(hsl.h, clamp(hsl.s * 0.35, 6, 30), hsl.l),
                    hslToHex(shiftHue(hsl.h, 18), clamp(hsl.s * 0.34, 6, 30), clamp(hsl.l - 8, 0, 100)),
                    hslToHex(shiftHue(hsl.h, 36), clamp(hsl.s * 0.32, 6, 28), clamp(hsl.l - 16, 0, 100)),
                ],
            },
        };

        const renderSwatches = () => {
            swatchRow.innerHTML = "";
            const list = this.currentPalette || [];
            currentHex.textContent = list.join(", ");
            if (proDeriveNote) {
                proDeriveNote.textContent = this.currentPaletteNote || "";
            }
            if (!list.length) {
                const t = document.createElement("div");
                t.style.fontSize = "12px";
                t.style.color = "#94a3b8";
                t.textContent = "暂无配色";
                swatchRow.appendChild(t);
                return;
            }
            list.forEach((hex) => {
                const b = document.createElement("button");
                b.title = hex;
                b.style.width = "36px";
                b.style.height = "36px";
                b.style.borderRadius = "6px";
                b.style.border = "1px solid #4a5568";
                b.style.background = hex;
                b.style.cursor = "pointer";
                b.addEventListener("click", () => navigator.clipboard?.writeText(hex).catch(() => {}));
                swatchRow.appendChild(b);
            });
        };

        const renderHistory = () => {
            historyList.innerHTML = "";
            this.promptHistory.forEach((text, idx) => {
                const chip = document.createElement("div");
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.gap = "4px";
                chip.style.background = "#111827";
                chip.style.border = "1px solid #4a5568";
                chip.style.borderRadius = "4px";
                chip.style.padding = "2px 6px";
                chip.style.fontSize = "12px";
                chip.textContent = `${idx + 1}`;
                const del = document.createElement("button");
                del.textContent = "×";
                del.style.background = "transparent";
                del.style.color = "#f56565";
                del.style.border = "none";
                del.style.cursor = "pointer";
                del.onclick = () => {
                    this.promptHistory.splice(idx, 1);
                    renderHistory();
                };
                chip.appendChild(del);
                historyList.appendChild(chip);
            });
        };

        const loadImage = (url) => {
            if (!url) return;
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                this.currentImage = img;
                imagePreview.src = url;
                imagePreview.style.display = "block";
                imageTip.style.display = "none";
            };
            img.onerror = () => {};
            img.src = url;
        };

        closeBtn.onclick = () => this.close();
        countSlider.addEventListener("input", () => {
            countVal.textContent = String(countSlider.value);
        });
        extractBtn.onclick = () => {
            if (!this.currentImage) {
                alert("未读取到节点图片，请先确保 image 端口已连接并有预览。");
                return;
            }
            this.currentPalette = this._extractPaletteFromImage(this.currentImage, countSlider.value);
            renderSwatches();
        };
        aiBtn.onclick = () => {
            const scene = String(sceneSelect.value || "web");
            const base = [...(sceneMap[scene] || sceneMap.web)].map((x) => String(x).toUpperCase());
            const main = base[0] || "#372F39";
            const complementary = generateComplementaryScheme(main);
            const triadic = generateTriadicScheme(main);
            // AI smart mix: scenario base + complementary + triadic
            this.currentPalette = mergeUniqueHex(base, complementary, triadic).slice(0, 8);
            this.currentPaletteNote = "";
            renderSwatches();
        };
        applyPresetBtn.onclick = () => {
            const key = String(presetSelect.value || "");
            this.currentPalette = [...(presetMap[key] || [])];
            this.currentPaletteNote = "";
            renderSwatches();
        };
        proDeriveBtn.onclick = () => {
            const schemeKey = String(proDeriveSelect?.value || "split_complementary");
            const scheme = proSchemeMap[schemeKey];
            if (!scheme) return;
            const mainHex = extractMainColorHex();
            const mainRgb = hexToRgb(mainHex);
            if (!mainRgb) {
                alert("无法提取主色，请先提取图片颜色或确保图片已加载。");
                return;
            }
            const mainHsl = rgbToHsl(mainRgb.r, mainRgb.g, mainRgb.b);
            const palette = scheme.build(mainHsl);
            this.currentPalette = mergeUniqueHex(palette).slice(0, 8);
            this.currentPaletteNote = `${scheme.name}：${scheme.copy}；${scheme.scene}`;
            renderSwatches();
        };
        generatePromptBtn.onclick = () => {
            if (!this.currentPalette.length) {
                alert("请先通过任意方式生成配色。");
                return;
            }
            const basePrompt = this._paletteToPrompt(this.currentPalette);
            promptOutput.value = this.currentPaletteNote ? `${basePrompt}\n${this.currentPaletteNote}` : basePrompt;
        };
        savePromptBtn.onclick = () => {
            const value = String(promptOutput.value || "").trim();
            if (!value) return;
            this.promptHistory.push(value);
            renderHistory();
        };
        applyNodeBtn.onclick = () => {
            if (!this.currentNode?.widgets) return;
            const widget = this.currentNode.widgets.find((w) => w.name === "color_prompt");
            if (!widget) {
                alert("节点未找到 color_prompt 输入。请重启后端后再试。");
                return;
            }
            const merged = [...this.promptHistory];
            const current = String(promptOutput.value || "").trim();
            if (current && !merged.includes(current)) merged.push(current);
            widget.value = merged.join("\n");
            if (app.graph) app.graph.setDirtyCanvas(true, true);
            this.close();
        };

        const initial = this._resolveInitialImageSource();
        if (initial?.url) {
            loadImage(initial.url);
        }

        this.promptHistory = [];
        this.currentPaletteNote = "";
        promptOutput.value = "";
        renderHistory();
        renderSwatches();
    }

    close() {
        this.currentImage = null;
        this.currentPalette = [];
        this.promptHistory = [];
        this.currentPaletteNote = "";
        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }
        super.close();
    }
}

class Light2DPromptModal extends ComfyDialog {
    static instance = null;
    static getInstance() {
        if (!Light2DPromptModal.instance) {
            Light2DPromptModal.instance = new Light2DPromptModal();
        }
        return Light2DPromptModal.instance;
    }

    constructor() {
        super();
        this.currentNode = null;
        this.modalElement = null;
        this.promptHistory = [];
        this.imageEl = null;
        this.mode = "radial";
        this.isDragging = false;
        this.activeHandle = null;
        this.radialState = {
            center_x: 0.5,
            center_y: 0.5,
            circle_radius: 0.2,
            center_bright: 1.5,
            edge_bright: 1.0,
            overlay_color: "#FFFFFF",
            center_alpha: 0.0,
            edge_alpha: 0.0,
            falloff_mode: "linear",
            soft_edge: true,
        };
        this.linearState = {
            start_x: 0.0,
            start_y: 0.0,
            start_bright: 0.5,
            end_x: 1.0,
            end_y: 1.0,
            end_bright: 1.0,
            overlay_color: "#FFFFFF",
            start_alpha: 0.0,
            end_alpha: 0.0,
        };
    }

    setNode(node) {
        this.currentNode = node;
    }

    _getImageInputIndex() {
        return this.currentNode?.inputs?.findIndex((input) => input?.name === "image") ?? -1;
    }

    _getConnectedImageSourceNode() {
        const imageInputIndex = this._getImageInputIndex();
        if (imageInputIndex < 0 || !this.currentNode?.inputs?.[imageInputIndex]) return null;
        const linkId = this.currentNode.inputs[imageInputIndex].link;
        if (linkId == null || !app.graph?.links) return null;
        const linkInfo = app.graph.links[linkId];
        if (!linkInfo?.origin_id) return null;
        return app.graph.getNodeById?.(linkInfo.origin_id) ?? app.graph._nodes?.find((node) => node.id === linkInfo.origin_id) ?? null;
    }

    _getImageWidgetValue(node) {
        return node?.widgets?.find((w) => w?.name === "image")?.value ?? null;
    }

    _buildViewUrlFromAnnotatedName(annotatedName) {
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
    }

    _getNodePreviewUrl(node) {
        const previewImage = node?.imgs?.[0];
        if (!previewImage) return null;
        if (typeof previewImage === "string") return previewImage;
        if (previewImage?.src) return previewImage.src;
        return null;
    }

    _resolveInitialImageSource() {
        const sourceNode = this._getConnectedImageSourceNode();
        const sourcePreview = this._getNodePreviewUrl(sourceNode);
        if (sourceNode && sourcePreview) return { node: sourceNode, url: sourcePreview };
        const sourceWidgetValue = this._getImageWidgetValue(sourceNode);
        if (sourceNode && sourceWidgetValue) {
            const widgetUrl = this._buildViewUrlFromAnnotatedName(sourceWidgetValue);
            if (widgetUrl) return { node: sourceNode, url: widgetUrl };
        }
        const currentPreview = this._getNodePreviewUrl(this.currentNode);
        if (currentPreview) return { node: this.currentNode, url: currentPreview };
        return null;
    }

    _clamp01(v) {
        return Math.max(0, Math.min(1, Number(v) || 0));
    }

    _format2(v) {
        return Number(v || 0).toFixed(2);
    }

    _buildRadialPrompt() {
        const r = this.radialState;
        return `LOCK ALL ORIGINAL IMAGE CONTENT. ONLY RADIAL GRADIENT CORRECTION. CENTER: (${this._format2(r.center_x)},${this._format2(r.center_y)}), RADIUS: ${this._format2(r.circle_radius)} BRIGHTNESS: center=${this._format2(r.center_bright)}, edge=${this._format2(r.edge_bright)} COLOR TINT: ${r.overlay_color}, ALPHA: center=${this._format2(r.center_alpha)}, edge=${this._format2(r.edge_alpha)} FALLOFF: ${r.falloff_mode}, SOFT EDGE: ${r.soft_edge ? "True" : "False"} SMOOTH FADE, UNSELECTED AREAS UNTOUCHED, NO REGENERATION. Do not modify anything else at all costs.`;
    }

    _buildLinearPrompt() {
        const l = this.linearState;
        return `LOCK ALL ORIGINAL IMAGE CONTENT. ONLY LINEAR GRADIENT CORRECTION. GRADIENT LINE: (${this._format2(l.start_x)},${this._format2(l.start_y)}) -> (${this._format2(l.end_x)},${this._format2(l.end_y)}) BRIGHTNESS: ${this._format2(l.start_bright)} -> ${this._format2(l.end_bright)} COLOR TINT: ${l.overlay_color}, OPACITY: ${this._format2(l.start_alpha)}->${this._format2(l.end_alpha)} SMOOTH FADE, UNSELECTED AREAS UNTOUCHED, NO REGENERATION. Do not modify anything else at all costs.`;
    }

    _bandIndex(value, segments = 7) {
        const v = this._clamp01(value);
        return Math.min(segments - 1, Math.floor(v * segments));
    }

    _describeAxis(value, axis) {
        const bands = axis === "x"
            ? [
                "extreme left edge",
                "left outer area",
                "left side area",
                "near center line",
                "right side area",
                "right outer area",
                "extreme right edge",
            ]
            : [
                "extreme top edge",
                "upper outer area",
                "upper area",
                "near center line",
                "lower area",
                "lower outer area",
                "extreme bottom edge",
            ];
        return bands[this._bandIndex(value, 7)];
    }

    _describePosition(x, y) {
        const xIndex = this._bandIndex(x, 7);
        const yIndex = this._bandIndex(y, 7);
        const gridMap = [
            ["extreme upper-left corner", "far upper-left edge", "upper-left outer zone", "top-center zone", "upper-right outer zone", "far upper-right edge", "extreme upper-right corner"],
            ["left-top edge-adjacent area", "upper-left side zone", "upper-left inner zone", "upper-central area", "upper-right inner zone", "upper-right side zone", "right-top edge-adjacent area"],
            ["left upper offset", "left-upper region", "left-of-upper-center", "slightly upper center", "right-of-upper-center", "right-upper region", "right upper offset"],
            ["left-center region", "slightly left of center", "center-left inner area", "dead center", "center-right inner area", "slightly right of center", "right-center region"],
            ["left lower offset", "left-lower region", "left-of-lower-center", "slightly lower center", "right-of-lower-center", "right-lower region", "right lower offset"],
            ["left-bottom edge-adjacent area", "lower-left side zone", "lower-left inner zone", "lower-central area", "lower-right inner zone", "lower-right side zone", "right-bottom edge-adjacent area"],
            ["extreme lower-left corner", "far lower-left edge", "lower-left outer zone", "bottom-center zone", "lower-right outer zone", "far lower-right edge", "extreme lower-right corner"],
        ];
        return gridMap[yIndex][xIndex];
    }

    _describeRadius(radius) {
        const r = Math.max(0, Number(radius) || 0);
        if (r < 0.08) return "tiny pinpoint micro range";
        if (r < 0.15) return "tiny micro range";
        if (r < 0.22) return "small compact range";
        if (r < 0.30) return "small focused range";
        if (r < 0.40) return "medium focused range";
        if (r < 0.52) return "medium spread range";
        if (r < 0.68) return "broad moderate range";
        if (r < 0.85) return "large wide coverage";
        return "very large near full-frame coverage";
    }

    _describeBrightness(value) {
        const v = Math.max(0, Number(value) || 0);
        if (v < 0.25) return "very heavy darkening";
        if (v < 0.45) return "heavy darken";
        if (v < 0.65) return "clear dimming";
        if (v < 0.85) return "noticeable dimming";
        if (v < 0.98) return "soft slight dimming";
        if (v < 1.10) return "near neutral brightness";
        if (v < 1.22) return "very gentle brighten";
        if (v < 1.35) return "gentle subtle brighten";
        if (v < 1.55) return "soft clear brightening";
        if (v < 1.85) return "moderately strong brightening";
        if (v < 2.30) return "strong visible brightening";
        if (v < 3.20) return "intense highlight boost";
        return "extreme dramatic brightening";
    }

    _describeAlpha(value) {
        const v = Math.max(0, Number(value) || 0);
        if (v < 0.03) return "nearly invisible tint overlay";
        if (v < 0.08) return "almost transparent tint";
        if (v < 0.16) return "very faint tint overlay";
        if (v < 0.28) return "light tint overlay";
        if (v < 0.42) return "soft but noticeable tint overlay";
        if (v < 0.58) return "moderate tint overlay";
        if (v < 0.78) return "strong tint overlay";
        return "dense heavy tint overlay";
    }

    _describeFalloff(mode) {
        if (mode === "ease_out") return "fast emphasis near the key region with a softer outer fade";
        if (mode === "ease_in") return "slow initial transition that strengthens toward the far range";
        return "strict linear gradual transition";
    }

    _describeSoftEdge(enabled) {
        return enabled ? "smooth continuous natural fade" : "firmer harder edge transition";
    }

    _describeRadialIntent(centerBright, edgeBright) {
        if (centerBright > edgeBright + 0.2) return "localized spotlight-style enhancement";
        if (edgeBright > centerBright + 0.2) return "center-suppressed outward emphasis";
        return "balanced localized tonal modulation";
    }

    _describeLinearDirection(startX, startY, endX, endY) {
        const dx = endX - startX;
        const dy = endY - startY;
        const absX = Math.abs(dx);
        const absY = Math.abs(dy);
        if (absX < 0.12 && absY < 0.12) return "very short local directional transition";
        if (absX > absY * 1.8) return dx >= 0 ? "left-to-right horizontal sweep" : "right-to-left horizontal sweep";
        if (absY > absX * 1.8) return dy >= 0 ? "top-to-bottom vertical sweep" : "bottom-to-top vertical sweep";
        if (dx >= 0 && dy >= 0) return "upper-left to lower-right diagonal sweep";
        if (dx >= 0 && dy < 0) return "lower-left to upper-right diagonal sweep";
        if (dx < 0 && dy >= 0) return "upper-right to lower-left diagonal sweep";
        return "lower-right to upper-left diagonal sweep";
    }

    _describeGradientSpan(startX, startY, endX, endY) {
        const dx = endX - startX;
        const dy = endY - startY;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 0.18) return "very short transition span";
        if (dist < 0.35) return "short focused transition span";
        if (dist < 0.60) return "medium transition span";
        if (dist < 0.85) return "long sweeping transition span";
        return "near full-frame transition span";
    }

    _buildRadialPromptFuzzy() {
        const r = this.radialState;
        return `LOCK ALL ORIGINAL IMAGE CONTENT. ONLY RADIAL GRADIENT CORRECTION. Create a ${this._describeRadialIntent(r.center_bright, r.edge_bright)} around the ${this._describePosition(r.center_x, r.center_y)} region. The effective influence is a ${this._describeRadius(r.circle_radius)}. Brightness shifts from ${this._describeBrightness(r.center_bright)} near the core to ${this._describeBrightness(r.edge_bright)} toward the surrounding area. Keep color tint ${r.overlay_color}. Center tint strength is ${this._describeAlpha(r.center_alpha)}, outer tint strength is ${this._describeAlpha(r.edge_alpha)}. Use ${this._describeFalloff(r.falloff_mode)} with ${this._describeSoftEdge(r.soft_edge)}. Unselected areas untouched, no regeneration, do not modify anything else.`;
    }

    _buildLinearPromptFuzzy() {
        const l = this.linearState;
        return `LOCK ALL ORIGINAL IMAGE CONTENT. ONLY LINEAR GRADIENT CORRECTION. Build a ${this._describeLinearDirection(l.start_x, l.start_y, l.end_x, l.end_y)} from ${this._describePosition(l.start_x, l.start_y)} toward ${this._describePosition(l.end_x, l.end_y)} across a ${this._describeGradientSpan(l.start_x, l.start_y, l.end_x, l.end_y)}. The brightness transitions from ${this._describeBrightness(l.start_bright)} to ${this._describeBrightness(l.end_bright)}. Keep color tint ${l.overlay_color}. Opacity shifts from ${this._describeAlpha(l.start_alpha)} to ${this._describeAlpha(l.end_alpha)}. Use a strict linear gradual transition with smooth continuous natural fade. Unselected areas untouched, no regeneration, do not modify anything else.`;
    }

    show() {
        // Each open starts from a clean editing session.
        this.promptHistory = [];
        this.imageEl = null;
        this.mode = "radial";
        this.isDragging = false;
        this.activeHandle = null;
        this.radialState = {
            center_x: 0.5,
            center_y: 0.5,
            circle_radius: 0.2,
            center_bright: 1.5,
            edge_bright: 1.0,
            overlay_color: "#FFFFFF",
            center_alpha: 0.0,
            edge_alpha: 0.0,
            falloff_mode: "linear",
            soft_edge: true,
        };
        this.linearState = {
            start_x: 0.0,
            start_y: 0.0,
            start_bright: 0.5,
            end_x: 1.0,
            end_y: 1.0,
            end_bright: 1.0,
            overlay_color: "#FFFFFF",
            start_alpha: 0.0,
            end_alpha: 0.0,
        };

        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }

        this.modalElement = $el("div.comfy-modal", { parent: document.body }, [
            $el("div.comfy-modal-content", {}, []),
        ]);
        this.modalElement.classList.add("comfy-modal-layout");
        this.modalElement.style.width = "82vw";
        this.modalElement.style.height = "82vh";
        this.modalElement.style.maxWidth = "100vw";
        this.modalElement.style.maxHeight = "100vh";

        const root = document.createElement("div");
        root.style.display = "flex";
        root.style.flexDirection = "column";
        root.style.width = "100%";
        root.style.height = "100%";
        root.style.background = "#1a202c";
        root.style.color = "#fff";
        root.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #4a5568;padding:10px 12px;">
                <h2 style="margin:0;font-size:16px;font-weight:700;">Light2D Gradient Prompt</h2>
                <button id="l2d-close" style="background:none;border:none;color:#a0aec0;font-size:20px;cursor:pointer;">×</button>
            </div>
            <div style="flex:1;display:flex;gap:12px;padding:12px;overflow:hidden;min-height:0;">
                <div style="flex:0 0 34%;display:flex;flex-direction:column;gap:10px;overflow:auto;min-height:0;background:#2d3748;border-radius:8px;padding:10px;">
                    <label style="font-size:12px;">Operation Mode</label>
                    <select id="l2d-mode" style="background:#111827;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:6px 8px;">
                        <option value="radial">radial_gradient</option>
                        <option value="linear">line_gradient</option>
                    </select>

                    <div id="l2d-radial-controls" style="display:flex;flex-direction:column;gap:6px;">
                        <label style="font-size:12px;">circle_radius</label>
                        <input id="l2d-circle-radius" type="range" min="0" max="1" step="0.01" value="0.2" />
                        <label style="font-size:12px;">center_bright</label>
                        <input id="l2d-center-bright" type="range" min="0" max="5" step="0.01" value="1.5" />
                        <label style="font-size:12px;">edge_bright</label>
                        <input id="l2d-edge-bright" type="range" min="0" max="5" step="0.01" value="1.0" />
                        <label style="font-size:12px;">center_alpha</label>
                        <input id="l2d-center-alpha" type="range" min="0" max="1" step="0.01" value="0.0" />
                        <label style="font-size:12px;">edge_alpha</label>
                        <input id="l2d-edge-alpha" type="range" min="0" max="1" step="0.01" value="0.0" />
                        <label style="font-size:12px;">falloff_mode</label>
                        <select id="l2d-falloff" style="background:#111827;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:6px 8px;">
                            <option value="linear">linear</option>
                            <option value="ease_out">ease_out</option>
                            <option value="ease_in">ease_in</option>
                        </select>
                        <label style="font-size:12px;display:flex;align-items:center;gap:6px;">
                            <input id="l2d-soft-edge" type="checkbox" checked />
                            soft_edge
                        </label>
                    </div>

                    <div id="l2d-linear-controls" style="display:none;flex-direction:column;gap:6px;">
                        <label style="font-size:12px;">start_bright</label>
                        <input id="l2d-start-bright" type="range" min="0" max="5" step="0.01" value="0.5" />
                        <label style="font-size:12px;">end_bright</label>
                        <input id="l2d-end-bright" type="range" min="0" max="5" step="0.01" value="1.0" />
                        <label style="font-size:12px;">start_alpha</label>
                        <input id="l2d-start-alpha" type="range" min="0" max="1" step="0.01" value="0.0" />
                        <label style="font-size:12px;">end_alpha</label>
                        <input id="l2d-end-alpha" type="range" min="0" max="1" step="0.01" value="0.0" />
                    </div>

                    <label style="font-size:12px;">overlay_color</label>
                    <input id="l2d-color" type="color" value="#ffffff" style="width:100%;height:32px;background:#111827;border:1px solid #4a5568;border-radius:4px;cursor:pointer;padding:0 2px;" />
                    <div style="font-size:12px;color:#a0aec0;">Radial: 点击中心并拖动半径。Linear: 拖拽起点到终点。</div>
                </div>

                <div style="flex:1;display:flex;flex-direction:column;gap:10px;min-height:0;">
                    <div style="background:#2d3748;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:8px;min-height:0;flex:1;">
                        <div style="font-size:14px;font-weight:700;">Canvas</div>
                        <div style="background:#111827;border:1px solid #4a5568;border-radius:6px;flex:1;display:flex;align-items:center;justify-content:center;min-height:280px;">
                            <canvas id="l2d-canvas" style="max-width:100%;max-height:100%;"></canvas>
                        </div>
                    </div>
                    <div style="background:#2d3748;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:8px;">
                        <div style="display:flex;gap:8px;">
                            <button id="l2d-generate-precise" style="background:#d69e2e;color:#111827;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">生成精准light2d</button>
                            <button id="l2d-generate-fuzzy" style="background:#7c3aed;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">生成模糊light2d</button>
                            <button id="l2d-save" style="background:#2b6cb0;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">保存一条</button>
                            <button id="l2d-apply" style="background:#38a169;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">应用到节点</button>
                        </div>
                        <textarea id="l2d-output" rows="4" style="width:100%;background:#111827;border:1px solid #4a5568;border-radius:4px;color:#e2e8f0;padding:8px;"></textarea>
                        <div style="font-size:14px;font-weight:700;">已保存列表</div>
                        <div id="l2d-history" style="display:flex;gap:6px;flex-wrap:wrap;max-height:90px;overflow:auto;"></div>
                    </div>
                </div>
            </div>
        `;

        this.modalElement.appendChild(root);
        const q = (s) => root.querySelector(s);
        const closeBtn = q("#l2d-close");
        const modeSel = q("#l2d-mode");
        const radialPanel = q("#l2d-radial-controls");
        const linearPanel = q("#l2d-linear-controls");
        const colorInput = q("#l2d-color");
        const canvas = q("#l2d-canvas");
        const circleRadius = q("#l2d-circle-radius");
        const centerBright = q("#l2d-center-bright");
        const edgeBright = q("#l2d-edge-bright");
        const centerAlpha = q("#l2d-center-alpha");
        const edgeAlpha = q("#l2d-edge-alpha");
        const falloffMode = q("#l2d-falloff");
        const softEdge = q("#l2d-soft-edge");
        const startBright = q("#l2d-start-bright");
        const endBright = q("#l2d-end-bright");
        const startAlpha = q("#l2d-start-alpha");
        const endAlpha = q("#l2d-end-alpha");
        const genPreciseBtn = q("#l2d-generate-precise");
        const genFuzzyBtn = q("#l2d-generate-fuzzy");
        const saveBtn = q("#l2d-save");
        const applyBtn = q("#l2d-apply");
        const output = q("#l2d-output");
        const historyEl = q("#l2d-history");
        const ctx = canvas.getContext("2d");

        const syncStateFromWidgets = () => {
            this.radialState.circle_radius = Number(circleRadius.value);
            this.radialState.center_bright = Number(centerBright.value);
            this.radialState.edge_bright = Number(edgeBright.value);
            this.radialState.center_alpha = Number(centerAlpha.value);
            this.radialState.edge_alpha = Number(edgeAlpha.value);
            this.radialState.falloff_mode = String(falloffMode.value);
            this.radialState.soft_edge = !!softEdge.checked;
            this.radialState.overlay_color = String(colorInput.value || "#FFFFFF").toUpperCase();

            this.linearState.start_bright = Number(startBright.value);
            this.linearState.end_bright = Number(endBright.value);
            this.linearState.start_alpha = Number(startAlpha.value);
            this.linearState.end_alpha = Number(endAlpha.value);
            this.linearState.overlay_color = String(colorInput.value || "#FFFFFF").toUpperCase();
        };

        const renderHistory = () => {
            historyEl.innerHTML = "";
            this.promptHistory.forEach((_, idx) => {
                const chip = document.createElement("div");
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.gap = "4px";
                chip.style.background = "#111827";
                chip.style.border = "1px solid #4a5568";
                chip.style.borderRadius = "4px";
                chip.style.padding = "2px 6px";
                chip.style.fontSize = "12px";
                chip.textContent = `${idx + 1}`;
                const del = document.createElement("button");
                del.textContent = "×";
                del.style.background = "transparent";
                del.style.color = "#f56565";
                del.style.border = "none";
                del.style.cursor = "pointer";
                del.onclick = () => {
                    this.promptHistory.splice(idx, 1);
                    renderHistory();
                };
                chip.appendChild(del);
                historyEl.appendChild(chip);
            });
        };

        const clamp01 = (v) => Math.max(0, Math.min(1, v));
        const clamp255 = (v) => Math.max(0, Math.min(255, v));
        const hexToRgb = (hex) => {
            const m = String(hex || "").trim().match(/^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i);
            if (!m) return { r: 255, g: 255, b: 255 };
            return {
                r: parseInt(m[1], 16),
                g: parseInt(m[2], 16),
                b: parseInt(m[3], 16),
            };
        };
        const applyFalloffCurve = (t, mode) => {
            const x = clamp01(t);
            if (mode === "ease_out") {
                return x * x;
            }
            if (mode === "ease_in") {
                return Math.sqrt(x);
            }
            return x;
        };

        const applyRadialPreview = (imageData) => {
            const data = imageData.data;
            const w = imageData.width;
            const h = imageData.height;
            const c = this.radialState;
            const maxSide = Math.max(w, h);
            const cx = c.center_x * (w - 1);
            const cy = c.center_y * (h - 1);
            const radiusPx = c.circle_radius * maxSide;
            const softEdgePx = Math.max(1, Math.floor(radiusPx * 0.05));
            const maxFalloffDistance = Math.max(1e-6, maxSide - radiusPx);
            const overlay = hexToRgb(c.overlay_color);
            const oR = overlay.r / 255;
            const oG = overlay.g / 255;
            const oB = overlay.b / 255;

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const idx = (y * w + x) * 4;
                    const dx = x - cx;
                    const dy = y - cy;
                    const dist = Math.sqrt(dx * dx + dy * dy);

                    let inCircleSoft = 0;
                    if (c.soft_edge) {
                        inCircleSoft = clamp01((radiusPx + softEdgePx - dist) / (2 * softEdgePx));
                    } else {
                        inCircleSoft = dist <= radiusPx ? 1 : 0;
                    }

                    const falloffDistance = dist - radiusPx;
                    let normFalloff = clamp01(falloffDistance / maxFalloffDistance);
                    normFalloff = applyFalloffCurve(normFalloff, c.falloff_mode);

                    const brightOverlay = c.center_bright * (1 - normFalloff) + c.edge_bright * normFalloff;
                    const brightness = c.center_bright * inCircleSoft + brightOverlay * (1 - inCircleSoft);

                    const alphaOverlay = c.center_alpha * (1 - normFalloff) + c.edge_alpha * normFalloff;
                    const alpha = c.center_alpha * inCircleSoft + alphaOverlay * (1 - inCircleSoft);

                    let r = (data[idx] / 255) * brightness;
                    let g = (data[idx + 1] / 255) * brightness;
                    let b = (data[idx + 2] / 255) * brightness;

                    r = r * (1 - alpha) + oR * alpha;
                    g = g * (1 - alpha) + oG * alpha;
                    b = b * (1 - alpha) + oB * alpha;

                    data[idx] = clamp255(Math.round(r * 255));
                    data[idx + 1] = clamp255(Math.round(g * 255));
                    data[idx + 2] = clamp255(Math.round(b * 255));
                }
            }
        };

        const applyLinearPreview = (imageData) => {
            const data = imageData.data;
            const w = imageData.width;
            const h = imageData.height;
            const l = this.linearState;
            const sx = l.start_x * (w - 1);
            const sy = l.start_y * (h - 1);
            const ex = l.end_x * (w - 1);
            const ey = l.end_y * (h - 1);
            const dirX = ex - sx;
            const dirY = ey - sy;
            const len2 = dirX * dirX + dirY * dirY;
            const overlay = hexToRgb(l.overlay_color);
            const oR = overlay.r / 255;
            const oG = overlay.g / 255;
            const oB = overlay.b / 255;

            for (let y = 0; y < h; y++) {
                for (let x = 0; x < w; x++) {
                    const idx = (y * w + x) * 4;
                    let t = 0;
                    if (len2 > 1e-6) {
                        t = clamp01((((x - sx) * dirX) + ((y - sy) * dirY)) / len2);
                    }

                    const brightness = l.start_bright * (1 - t) + l.end_bright * t;
                    const alpha = l.start_alpha * (1 - t) + l.end_alpha * t;

                    let r = (data[idx] / 255) * brightness;
                    let g = (data[idx + 1] / 255) * brightness;
                    let b = (data[idx + 2] / 255) * brightness;

                    r = r * (1 - alpha) + oR * alpha;
                    g = g * (1 - alpha) + oG * alpha;
                    b = b * (1 - alpha) + oB * alpha;

                    data[idx] = clamp255(Math.round(r * 255));
                    data[idx + 1] = clamp255(Math.round(g * 255));
                    data[idx + 2] = clamp255(Math.round(b * 255));
                }
            }
        };

        const drawGuides = () => {
            ctx.lineWidth = 2;
            ctx.strokeStyle = "#38bdf8";
            ctx.fillStyle = "rgba(56,189,248,0.25)";
            if (this.mode === "radial") {
                const cx = this.radialState.center_x * canvas.width;
                const cy = this.radialState.center_y * canvas.height;
                const radius = this.radialState.circle_radius * Math.max(canvas.width, canvas.height);
                ctx.beginPath();
                ctx.arc(cx, cy, radius, 0, Math.PI * 2);
                ctx.stroke();
                ctx.beginPath();
                ctx.arc(cx, cy, 4, 0, Math.PI * 2);
                ctx.fillStyle = "#f97316";
                ctx.fill();
            } else {
                const sx = this.linearState.start_x * canvas.width;
                const sy = this.linearState.start_y * canvas.height;
                const ex = this.linearState.end_x * canvas.width;
                const ey = this.linearState.end_y * canvas.height;
                ctx.beginPath();
                ctx.moveTo(sx, sy);
                ctx.lineTo(ex, ey);
                ctx.stroke();
                ctx.fillStyle = "#f97316";
                ctx.beginPath();
                ctx.arc(sx, sy, 4, 0, Math.PI * 2);
                ctx.fill();
                ctx.fillStyle = "#22c55e";
                ctx.beginPath();
                ctx.arc(ex, ey, 4, 0, Math.PI * 2);
                ctx.fill();
            }
        };

        const draw = () => {
            if (!this.imageEl || !ctx) return;
            const maxW = 980;
            const maxH = 560;
            const imgW = this.imageEl.naturalWidth || this.imageEl.width;
            const imgH = this.imageEl.naturalHeight || this.imageEl.height;
            const ratio = Math.min(maxW / imgW, maxH / imgH, 1);
            canvas.width = Math.max(1, Math.round(imgW * ratio));
            canvas.height = Math.max(1, Math.round(imgH * ratio));
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(this.imageEl, 0, 0, canvas.width, canvas.height);
            const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
            if (this.mode === "radial") {
                applyRadialPreview(imageData);
            } else {
                applyLinearPreview(imageData);
            }
            ctx.putImageData(imageData, 0, 0);
            drawGuides();
        };

        const eventToNorm = (evt) => {
            const rect = canvas.getBoundingClientRect();
            const x = (evt.clientX - rect.left) / rect.width;
            const y = (evt.clientY - rect.top) / rect.height;
            return { x: this._clamp01(x), y: this._clamp01(y) };
        };

        const onPointerDown = (evt) => {
            if (!this.imageEl) return;
            const p = eventToNorm(evt);
            const hitRadius = 0.03;
            this.activeHandle = null;
            if (this.mode === "radial") {
                const dx = p.x - this.radialState.center_x;
                const dy = p.y - this.radialState.center_y;
                const d = Math.sqrt(dx * dx + dy * dy);
                if (d <= hitRadius) {
                    this.activeHandle = "radial_center";
                }
            } else {
                const dsx = p.x - this.linearState.start_x;
                const dsy = p.y - this.linearState.start_y;
                const dex = p.x - this.linearState.end_x;
                const dey = p.y - this.linearState.end_y;
                const dStart = Math.sqrt(dsx * dsx + dsy * dsy);
                const dEnd = Math.sqrt(dex * dex + dey * dey);
                if (dStart <= hitRadius || dEnd <= hitRadius) {
                    this.activeHandle = dStart <= dEnd ? "linear_start" : "linear_end";
                }
            }
            this.isDragging = !!this.activeHandle;
            if (this.isDragging) draw();
        };

        const onPointerMove = (evt) => {
            if (!this.isDragging || !this.imageEl || !this.activeHandle) return;
            const p = eventToNorm(evt);
            if (this.activeHandle === "radial_center") {
                this.radialState.center_x = p.x;
                this.radialState.center_y = p.y;
            } else if (this.activeHandle === "linear_start") {
                this.linearState.start_x = p.x;
                this.linearState.start_y = p.y;
            } else if (this.activeHandle === "linear_end") {
                this.linearState.end_x = p.x;
                this.linearState.end_y = p.y;
            }
            draw();
        };

        const onPointerUp = () => {
            this.isDragging = false;
            this.activeHandle = null;
        };

        const loadImage = (url) => {
            if (!url) return;
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                this.imageEl = img;
                draw();
            };
            img.src = url;
        };

        closeBtn.onclick = () => this.close();
        modeSel.onchange = () => {
            this.mode = String(modeSel.value || "radial");
            const isRadial = this.mode === "radial";
            radialPanel.style.display = isRadial ? "flex" : "none";
            linearPanel.style.display = isRadial ? "none" : "flex";
            draw();
        };

        [circleRadius, centerBright, edgeBright, centerAlpha, edgeAlpha, falloffMode, softEdge, startBright, endBright, startAlpha, endAlpha, colorInput].forEach((el) => {
            el?.addEventListener("input", () => {
                syncStateFromWidgets();
                draw();
            });
        });

        canvas.addEventListener("mousedown", onPointerDown);
        canvas.addEventListener("mousemove", onPointerMove);
        window.addEventListener("mouseup", onPointerUp);
        canvas.addEventListener("mouseleave", onPointerUp);

        genPreciseBtn.onclick = () => {
            syncStateFromWidgets();
            output.value = this.mode === "radial" ? this._buildRadialPrompt() : this._buildLinearPrompt();
        };
        genFuzzyBtn.onclick = () => {
            syncStateFromWidgets();
            output.value = this.mode === "radial" ? this._buildRadialPromptFuzzy() : this._buildLinearPromptFuzzy();
        };
        saveBtn.onclick = () => {
            const value = String(output.value || "").trim();
            if (!value) return;
            this.promptHistory.push(value);
            renderHistory();
        };
        applyBtn.onclick = () => {
            if (!this.currentNode?.widgets) return;
            const widget = this.currentNode.widgets.find((w) => w.name === "light2d_prompt");
            if (!widget) {
                alert("节点未找到 light2d_prompt 输入。请重启后端后再试。");
                return;
            }
            const merged = [...this.promptHistory];
            const current = String(output.value || "").trim();
            if (current && !merged.includes(current)) merged.push(current);
            widget.value = merged.join("\n");
            if (app.graph) app.graph.setDirtyCanvas(true, true);
            this.close();
        };

        const initial = this._resolveInitialImageSource();
        if (initial?.url) {
            loadImage(initial.url);
        }

        this.promptHistory = [];
        output.value = "";
        renderHistory();
        syncStateFromWidgets();
    }

    close() {
        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }
        this.imageEl = null;
        this.isDragging = false;
        this.activeHandle = null;
        this.promptHistory = [];
        super.close();
    }
}

class PositionPromptModal extends ComfyDialog {
    static instance = null;
    static getInstance() {
        if (!PositionPromptModal.instance) {
            PositionPromptModal.instance = new PositionPromptModal();
        }
        return PositionPromptModal.instance;
    }

    constructor() {
        super();
        this.currentNode = null;
        this.modalElement = null;
        this.promptHistory = [];
        this.imageEl = null;
        this.isDragging = false;
        this.dragMode = null;
        this.dragAnchor = null;
        this.startBox = null;
        this.outputMode = "center";
        this.boxState = {
            xMin: 0.25,
            yMin: 0.25,
            xMax: 0.75,
            yMax: 0.75,
        };
    }

    setNode(node) {
        this.currentNode = node;
    }

    _getImageInputIndex() {
        return this.currentNode?.inputs?.findIndex((input) => input?.name === "image") ?? -1;
    }

    _getConnectedImageSourceNode() {
        const imageInputIndex = this._getImageInputIndex();
        if (imageInputIndex < 0 || !this.currentNode?.inputs?.[imageInputIndex]) return null;
        const linkId = this.currentNode.inputs[imageInputIndex].link;
        if (linkId == null || !app.graph?.links) return null;
        const linkInfo = app.graph.links[linkId];
        if (!linkInfo?.origin_id) return null;
        return app.graph.getNodeById?.(linkInfo.origin_id) ?? app.graph._nodes?.find((node) => node.id === linkInfo.origin_id) ?? null;
    }

    _getImageWidgetValue(node) {
        return node?.widgets?.find((w) => w?.name === "image")?.value ?? null;
    }

    _buildViewUrlFromAnnotatedName(annotatedName) {
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
    }

    _getNodePreviewUrl(node) {
        const previewImage = node?.imgs?.[0];
        if (!previewImage) return null;
        if (typeof previewImage === "string") return previewImage;
        if (previewImage?.src) return previewImage.src;
        return null;
    }

    _resolveInitialImageSource() {
        const sourceNode = this._getConnectedImageSourceNode();
        const sourcePreview = this._getNodePreviewUrl(sourceNode);
        if (sourceNode && sourcePreview) return { node: sourceNode, url: sourcePreview };
        const sourceWidgetValue = this._getImageWidgetValue(sourceNode);
        if (sourceNode && sourceWidgetValue) {
            const widgetUrl = this._buildViewUrlFromAnnotatedName(sourceWidgetValue);
            if (widgetUrl) return { node: sourceNode, url: widgetUrl };
        }
        const currentPreview = this._getNodePreviewUrl(this.currentNode);
        if (currentPreview) return { node: this.currentNode, url: currentPreview };
        return null;
    }

    _clamp01(v) {
        return Math.max(0, Math.min(1, Number(v) || 0));
    }

    _format3(v) {
        return Number(v || 0).toFixed(3);
    }

    _normalizeBox(box) {
        const minSize = 0.02;
        let xMin = this._clamp01(Math.min(box.xMin, box.xMax));
        let xMax = this._clamp01(Math.max(box.xMin, box.xMax));
        let yMin = this._clamp01(Math.min(box.yMin, box.yMax));
        let yMax = this._clamp01(Math.max(box.yMin, box.yMax));
        if ((xMax - xMin) < minSize) {
            xMax = Math.min(1, xMin + minSize);
            xMin = Math.max(0, xMax - minSize);
        }
        if ((yMax - yMin) < minSize) {
            yMax = Math.min(1, yMin + minSize);
            yMin = Math.max(0, yMax - minSize);
        }
        return { xMin, yMin, xMax, yMax };
    }

    _getCenter(box = this.boxState) {
        return {
            x: (box.xMin + box.xMax) / 2,
            y: (box.yMin + box.yMax) / 2,
        };
    }

    _buildCenterPrompt() {
        const center = this._getCenter();
        return `Normalized coordinate system: top-left (0,0), bottom-right (1,1), center position coordinate [${this._format3(center.x)}, ${this._format3(center.y)}].`;
    }

    _buildBBoxPrompt() {
        const box = this.boxState;
        return `Normalized coordinate system: top-left (0,0), bottom-right (1,1), normalized bounding box bbox=[${this._format3(box.xMin)}, ${this._format3(box.yMin)}, ${this._format3(box.xMax)}, ${this._format3(box.yMax)}].`;
    }

    _bandIndex(value, segments = 7) {
        const v = this._clamp01(value);
        return Math.min(segments - 1, Math.floor(v * segments));
    }

    _describePosition(x, y) {
        const xIndex = this._bandIndex(x, 7);
        const yIndex = this._bandIndex(y, 7);
        const gridMap = [
            ["extreme upper-left corner", "far upper-left edge", "upper-left outer zone", "top-center zone", "upper-right outer zone", "far upper-right edge", "extreme upper-right corner"],
            ["left-top edge-adjacent area", "upper-left side zone", "upper-left inner zone", "upper-central area", "upper-right inner zone", "upper-right side zone", "right-top edge-adjacent area"],
            ["left upper offset", "left-upper region", "left-of-upper-center", "slightly upper center", "right-of-upper-center", "right-upper region", "right upper offset"],
            ["left-center region", "slightly left of center", "center-left inner area", "dead center", "center-right inner area", "slightly right of center", "right-center region"],
            ["left lower offset", "left-lower region", "left-of-lower-center", "slightly lower center", "right-of-lower-center", "right-lower region", "right lower offset"],
            ["left-bottom edge-adjacent area", "lower-left side zone", "lower-left inner zone", "lower-central area", "lower-right inner zone", "lower-right side zone", "right-bottom edge-adjacent area"],
            ["extreme lower-left corner", "far lower-left edge", "lower-left outer zone", "bottom-center zone", "lower-right outer zone", "far lower-right edge", "extreme lower-right corner"],
        ];
        return gridMap[yIndex][xIndex];
    }

    _describeBoxSize(box = this.boxState) {
        const width = Math.max(0, box.xMax - box.xMin);
        const height = Math.max(0, box.yMax - box.yMin);
        const area = width * height;
        if (area < 0.02) return "very tiny pinpoint target area";
        if (area < 0.05) return "tiny compact target area";
        if (area < 0.10) return "small focused target area";
        if (area < 0.18) return "small-to-medium target area";
        if (area < 0.30) return "medium moderate target area";
        if (area < 0.45) return "large broad target area";
        if (area < 0.65) return "very large dominant target area";
        return "near full-frame target coverage";
    }

    _describeBoxShape(box = this.boxState) {
        const width = Math.max(1e-6, box.xMax - box.xMin);
        const height = Math.max(1e-6, box.yMax - box.yMin);
        const ratio = width / height;
        if (ratio < 0.50) return "very tall narrow vertical box";
        if (ratio < 0.80) return "tall vertical box";
        if (ratio < 1.15) return "balanced square-like box";
        if (ratio < 1.60) return "balanced rectangular box";
        if (ratio < 2.20) return "wide horizontal box";
        return "very wide panoramic box";
    }

    _describeEdgeAffinity(box = this.boxState) {
        const nearTop = box.yMin < 0.10;
        const nearBottom = box.yMax > 0.90;
        const nearLeft = box.xMin < 0.10;
        const nearRight = box.xMax > 0.90;
        const tags = [];
        if (nearTop) tags.push("top edge");
        if (nearBottom) tags.push("bottom edge");
        if (nearLeft) tags.push("left edge");
        if (nearRight) tags.push("right edge");
        if (!tags.length) return "well inside the frame";
        if (tags.length === 1) return `close to the ${tags[0]}`;
        return `touching or hugging the ${tags.join(" and ")}`;
    }

    _describeBoxSpan(box = this.boxState) {
        const width = Math.max(0, box.xMax - box.xMin);
        const height = Math.max(0, box.yMax - box.yMin);
        const wDesc = width < 0.15 ? "narrow width span" : (width < 0.35 ? "moderate width span" : (width < 0.60 ? "broad width span" : "very wide width span"));
        const hDesc = height < 0.15 ? "short height span" : (height < 0.35 ? "moderate height span" : (height < 0.60 ? "tall height span" : "very tall height span"));
        return `${wDesc} with ${hDesc}`;
    }

    _describeBoxCoverage(box = this.boxState) {
        const center = this._getCenter(box);
        return `mainly centered around the ${this._describePosition(center.x, center.y)} area, extending from ${this._describePosition(box.xMin, box.yMin)} toward ${this._describePosition(box.xMax, box.yMax)}`;
    }

    _buildCenterPromptFuzzy() {
        const center = this._getCenter();
        return `Normalized coordinate system reference: top-left (0,0), bottom-right (1,1). The target center position is roughly around the ${this._describePosition(center.x, center.y)} area, not an exact numeric point.`;
    }

    _buildBBoxPromptFuzzy() {
        const box = this.boxState;
        return `Normalized coordinate system reference: top-left (0,0), bottom-right (1,1). The target bounding box is roughly a ${this._describeBoxShape(box)} with ${this._describeBoxSize(box)}, ${this._describeBoxSpan(box)}, ${this._describeEdgeAffinity(box)}, and ${this._describeBoxCoverage(box)}.`;
    }

    show() {
        this.promptHistory = [];
        this.imageEl = null;
        this.isDragging = false;
        this.dragMode = null;
        this.dragAnchor = null;
        this.startBox = null;
        this.outputMode = "center";
        this.boxState = {
            xMin: 0.25,
            yMin: 0.25,
            xMax: 0.75,
            yMax: 0.75,
        };

        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }

        this.modalElement = $el("div.comfy-modal", { parent: document.body }, [
            $el("div.comfy-modal-content", {}, []),
        ]);
        this.modalElement.classList.add("comfy-modal-layout");
        this.modalElement.style.width = "82vw";
        this.modalElement.style.height = "82vh";
        this.modalElement.style.maxWidth = "100vw";
        this.modalElement.style.maxHeight = "100vh";

        const root = document.createElement("div");
        root.style.display = "flex";
        root.style.flexDirection = "column";
        root.style.width = "100%";
        root.style.height = "100%";
        root.style.background = "#1a202c";
        root.style.color = "#fff";
        root.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #4a5568;padding:10px 12px;">
                <h2 style="margin:0;font-size:16px;font-weight:700;">Position Prompt Editor</h2>
                <button id="pos-close" style="background:none;border:none;color:#a0aec0;font-size:20px;cursor:pointer;">×</button>
            </div>
            <div style="flex:1;display:flex;gap:12px;padding:12px;overflow:hidden;min-height:0;">
                <div style="flex:0 0 34%;display:flex;flex-direction:column;gap:10px;overflow:auto;min-height:0;background:#2d3748;border-radius:8px;padding:10px;">
                    <label style="font-size:12px;">输出模式</label>
                    <select id="pos-output-mode" style="background:#111827;color:#e2e8f0;border:1px solid #4a5568;border-radius:4px;padding:6px 8px;">
                        <option value="center">center position</option>
                        <option value="bbox">bbox</option>
                    </select>
                    <div style="font-size:12px;color:#a0aec0;line-height:1.5;">拖动矩形内部可移动，拖动四个角点可缩放。所有值均为归一化坐标。</div>
                    <button id="pos-reset-box" style="background:#4a5568;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">重置矩形</button>
                    <div style="background:#111827;border:1px solid #4a5568;border-radius:6px;padding:8px;display:flex;flex-direction:column;gap:6px;">
                        <div style="font-size:13px;font-weight:700;">当前结果</div>
                        <div id="pos-center-text" style="font-size:12px;color:#e2e8f0;line-height:1.4;"></div>
                        <div id="pos-bbox-text" style="font-size:12px;color:#e2e8f0;line-height:1.4;"></div>
                    </div>
                </div>

                <div style="flex:1;display:flex;flex-direction:column;gap:10px;min-height:0;">
                    <div style="background:#2d3748;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:8px;min-height:0;flex:1;">
                        <div style="font-size:14px;font-weight:700;">Canvas</div>
                        <div style="background:#111827;border:1px solid #4a5568;border-radius:6px;flex:1;display:flex;align-items:center;justify-content:center;min-height:280px;">
                            <canvas id="pos-canvas" style="max-width:100%;max-height:100%;"></canvas>
                        </div>
                    </div>
                    <div style="background:#2d3748;border-radius:8px;padding:10px;display:flex;flex-direction:column;gap:8px;">
                        <div style="display:flex;gap:8px;">
                            <button id="pos-generate-precise" style="background:#d69e2e;color:#111827;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">生成精准位置</button>
                            <button id="pos-generate-fuzzy" style="background:#7c3aed;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">生成模糊位置</button>
                            <button id="pos-save" style="background:#2b6cb0;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">保存一条</button>
                            <button id="pos-apply" style="background:#38a169;color:#fff;border:none;border-radius:4px;padding:6px 10px;cursor:pointer;">应用到节点</button>
                        </div>
                        <textarea id="pos-output" rows="4" style="width:100%;background:#111827;border:1px solid #4a5568;border-radius:4px;color:#e2e8f0;padding:8px;"></textarea>
                        <div style="font-size:14px;font-weight:700;">已保存列表</div>
                        <div id="pos-history" style="display:flex;gap:6px;flex-wrap:wrap;max-height:90px;overflow:auto;"></div>
                    </div>
                </div>
            </div>
        `;

        this.modalElement.appendChild(root);
        const q = (s) => root.querySelector(s);
        const closeBtn = q("#pos-close");
        const outputModeSel = q("#pos-output-mode");
        const resetBoxBtn = q("#pos-reset-box");
        const centerText = q("#pos-center-text");
        const bboxText = q("#pos-bbox-text");
        const canvas = q("#pos-canvas");
        const genPreciseBtn = q("#pos-generate-precise");
        const genFuzzyBtn = q("#pos-generate-fuzzy");
        const saveBtn = q("#pos-save");
        const applyBtn = q("#pos-apply");
        const output = q("#pos-output");
        const historyEl = q("#pos-history");
        const ctx = canvas.getContext("2d");

        const renderHistory = () => {
            historyEl.innerHTML = "";
            this.promptHistory.forEach((_, idx) => {
                const chip = document.createElement("div");
                chip.style.display = "inline-flex";
                chip.style.alignItems = "center";
                chip.style.gap = "4px";
                chip.style.background = "#111827";
                chip.style.border = "1px solid #4a5568";
                chip.style.borderRadius = "4px";
                chip.style.padding = "2px 6px";
                chip.style.fontSize = "12px";
                chip.textContent = `${idx + 1}`;
                const del = document.createElement("button");
                del.textContent = "×";
                del.style.background = "transparent";
                del.style.color = "#f56565";
                del.style.border = "none";
                del.style.cursor = "pointer";
                del.onclick = () => {
                    this.promptHistory.splice(idx, 1);
                    renderHistory();
                };
                chip.appendChild(del);
                historyEl.appendChild(chip);
            });
        };

        const syncInfo = () => {
            const center = this._getCenter();
            centerText.textContent = `Center: [${this._format3(center.x)}, ${this._format3(center.y)}]`;
            bboxText.textContent = `BBox: [${this._format3(this.boxState.xMin)}, ${this._format3(this.boxState.yMin)}, ${this._format3(this.boxState.xMax)}, ${this._format3(this.boxState.yMax)}]`;
        };

        const eventToNorm = (evt) => {
            const rect = canvas.getBoundingClientRect();
            const x = (evt.clientX - rect.left) / rect.width;
            const y = (evt.clientY - rect.top) / rect.height;
            return { x: this._clamp01(x), y: this._clamp01(y) };
        };

        const getHandleRects = () => {
            const box = this.boxState;
            return {
                nw: { x: box.xMin, y: box.yMin },
                ne: { x: box.xMax, y: box.yMin },
                sw: { x: box.xMin, y: box.yMax },
                se: { x: box.xMax, y: box.yMax },
            };
        };

        const hitTest = (p) => {
            const handleRadius = 0.03;
            const handles = getHandleRects();
            for (const [name, hp] of Object.entries(handles)) {
                const dx = p.x - hp.x;
                const dy = p.y - hp.y;
                if (Math.sqrt(dx * dx + dy * dy) <= handleRadius) {
                    return name;
                }
            }
            const box = this.boxState;
            if (p.x >= box.xMin && p.x <= box.xMax && p.y >= box.yMin && p.y <= box.yMax) {
                return "move";
            }
            return null;
        };

        const updateOutput = (fuzzy = false) => {
            if (fuzzy) {
                output.value = this.outputMode === "bbox" ? this._buildBBoxPromptFuzzy() : this._buildCenterPromptFuzzy();
                return;
            }
            output.value = this.outputMode === "bbox" ? this._buildBBoxPrompt() : this._buildCenterPrompt();
        };

        const drawGuides = () => {
            const box = this.boxState;
            const x = box.xMin * canvas.width;
            const y = box.yMin * canvas.height;
            const w = (box.xMax - box.xMin) * canvas.width;
            const h = (box.yMax - box.yMin) * canvas.height;
            const center = this._getCenter();
            const cx = center.x * canvas.width;
            const cy = center.y * canvas.height;

            ctx.save();
            ctx.fillStyle = "rgba(15, 23, 42, 0.35)";
            ctx.beginPath();
            ctx.rect(0, 0, canvas.width, canvas.height);
            ctx.rect(x, y, w, h);
            ctx.fill("evenodd");

            ctx.strokeStyle = "#38bdf8";
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);

            ctx.strokeStyle = "rgba(56,189,248,0.9)";
            ctx.beginPath();
            ctx.moveTo(cx, y);
            ctx.lineTo(cx, y + h);
            ctx.moveTo(x, cy);
            ctx.lineTo(x + w, cy);
            ctx.stroke();

            const handles = [
                [x, y],
                [x + w, y],
                [x, y + h],
                [x + w, y + h],
            ];
            handles.forEach(([hx, hy]) => {
                ctx.fillStyle = "#f97316";
                ctx.beginPath();
                ctx.arc(hx, hy, 6, 0, Math.PI * 2);
                ctx.fill();
            });

            ctx.fillStyle = "#22c55e";
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        };

        const draw = () => {
            if (!this.imageEl || !ctx) return;
            const maxW = 980;
            const maxH = 560;
            const imgW = this.imageEl.naturalWidth || this.imageEl.width;
            const imgH = this.imageEl.naturalHeight || this.imageEl.height;
            const ratio = Math.min(maxW / imgW, maxH / imgH, 1);
            canvas.width = Math.max(1, Math.round(imgW * ratio));
            canvas.height = Math.max(1, Math.round(imgH * ratio));
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(this.imageEl, 0, 0, canvas.width, canvas.height);
            drawGuides();
            syncInfo();
        };

        const applyDrag = (p) => {
            if (!this.dragMode || !this.startBox || !this.dragAnchor) return;
            const minSize = 0.02;
            if (this.dragMode === "move") {
                const dx = p.x - this.dragAnchor.x;
                const dy = p.y - this.dragAnchor.y;
                const width = this.startBox.xMax - this.startBox.xMin;
                const height = this.startBox.yMax - this.startBox.yMin;
                let xMin = this.startBox.xMin + dx;
                let yMin = this.startBox.yMin + dy;
                xMin = Math.max(0, Math.min(1 - width, xMin));
                yMin = Math.max(0, Math.min(1 - height, yMin));
                this.boxState = {
                    xMin,
                    yMin,
                    xMax: xMin + width,
                    yMax: yMin + height,
                };
                return;
            }

            let next = { ...this.startBox };
            if (this.dragMode === "nw") {
                next.xMin = Math.min(p.x, this.startBox.xMax - minSize);
                next.yMin = Math.min(p.y, this.startBox.yMax - minSize);
            } else if (this.dragMode === "ne") {
                next.xMax = Math.max(p.x, this.startBox.xMin + minSize);
                next.yMin = Math.min(p.y, this.startBox.yMax - minSize);
            } else if (this.dragMode === "sw") {
                next.xMin = Math.min(p.x, this.startBox.xMax - minSize);
                next.yMax = Math.max(p.y, this.startBox.yMin + minSize);
            } else if (this.dragMode === "se") {
                next.xMax = Math.max(p.x, this.startBox.xMin + minSize);
                next.yMax = Math.max(p.y, this.startBox.yMin + minSize);
            }
            this.boxState = this._normalizeBox(next);
        };

        const onPointerDown = (evt) => {
            if (!this.imageEl) return;
            const p = eventToNorm(evt);
            const mode = hitTest(p);
            if (!mode) return;
            this.isDragging = true;
            this.dragMode = mode;
            this.dragAnchor = p;
            this.startBox = { ...this.boxState };
            evt.preventDefault();
        };

        const onPointerMove = (evt) => {
            if (!this.isDragging || !this.imageEl) return;
            const p = eventToNorm(evt);
            applyDrag(p);
            draw();
        };

        const onPointerUp = () => {
            this.isDragging = false;
            this.dragMode = null;
            this.dragAnchor = null;
            this.startBox = null;
        };

        const loadImage = (url) => {
            if (!url) return;
            const img = new Image();
            img.crossOrigin = "Anonymous";
            img.onload = () => {
                this.imageEl = img;
                draw();
                updateOutput();
            };
            img.src = url;
        };

        closeBtn.onclick = () => this.close();
        outputModeSel.onchange = () => {
            this.outputMode = String(outputModeSel.value || "center");
            updateOutput();
        };
        resetBoxBtn.onclick = () => {
            this.boxState = {
                xMin: 0.25,
                yMin: 0.25,
                xMax: 0.75,
                yMax: 0.75,
            };
            draw();
            updateOutput();
        };

        canvas.addEventListener("mousedown", onPointerDown);
        canvas.addEventListener("mousemove", onPointerMove);
        window.addEventListener("mouseup", onPointerUp);
        canvas.addEventListener("mouseleave", onPointerUp);

        genPreciseBtn.onclick = () => {
            updateOutput();
        };
        genFuzzyBtn.onclick = () => {
            updateOutput(true);
        };
        saveBtn.onclick = () => {
            const value = String(output.value || "").trim();
            if (!value) return;
            this.promptHistory.push(value);
            renderHistory();
        };
        applyBtn.onclick = () => {
            if (!this.currentNode?.widgets) return;
            const widget = this.currentNode.widgets.find((w) => w.name === "position_prompt");
            if (!widget) {
                alert("节点未找到 position_prompt 输入。请重启后端后再试。");
                return;
            }
            const merged = [...this.promptHistory];
            const current = String(output.value || "").trim();
            if (current && !merged.includes(current)) merged.push(current);
            widget.value = merged.join("\n");
            if (app.graph) app.graph.setDirtyCanvas(true, true);
            this.close();
        };

        const initial = this._resolveInitialImageSource();
        if (initial?.url) {
            loadImage(initial.url);
        }

        this.promptHistory = [];
        output.value = "";
        syncInfo();
        updateOutput();
        renderHistory();
    }

    close() {
        if (this.modalElement) {
            document.body.removeChild(this.modalElement);
            this.modalElement = null;
        }
        this.imageEl = null;
        this.isDragging = false;
        this.dragMode = null;
        this.dragAnchor = null;
        this.startBox = null;
        this.promptHistory = [];
        super.close();
    }
}

function ensureTextMulAngleButtons(node) {
    if (!node) return;
    if (node._textMulAngleButtonsAttached) return;
    const open3D = () => {
        const modal = ThreeDEditorModal.getInstance();
        modal.setNode(node);
        modal.show();
    };
    const openPalette = () => {
        const modal = ColorPaletteModal.getInstance();
        modal.setNode(node);
        modal.show();
    };
    const openLight2D = () => {
        const modal = Light2DPromptModal.getInstance();
        modal.setNode(node);
        modal.show();
    };
    const openPosition = () => {
        const modal = PositionPromptModal.getInstance();
        modal.setNode(node);
        modal.show();
    };

    if (typeof node.addDOMWidget === "function") {
        const wrap = document.createElement("div");
        wrap.style.display = "flex";
        wrap.style.gap = "4px";
        wrap.style.width = "100%";
        wrap.style.flexWrap = "nowrap";
        wrap.style.pointerEvents = "auto";
        wrap.style.alignItems = "center";
        wrap.style.minHeight = "34px";
        wrap.style.maxHeight = "34px";
        wrap.style.height = "34px";
        wrap.style.flex = "0 0 34px";
        wrap.style.overflow = "hidden";
        const b1 = document.createElement("button");
        b1.textContent = "editor_3d";
        b1.style.cssText = "flex:1 1 0;background:var(--comfy-input-bg);color:var(--input-text);border:1px solid var(--border-color);border-radius:4px;padding:0 6px;min-height:28px;max-height:28px;height:28px;line-height:1;font-size:11px;cursor:pointer;";
        const b2 = document.createElement("button");
        b2.textContent = "色板";
        b2.style.cssText = "flex:1 1 0;background:var(--comfy-input-bg);color:#ef4444;border:1px solid var(--border-color);border-radius:4px;padding:0 6px;min-height:28px;max-height:28px;height:28px;line-height:1;font-size:11px;cursor:pointer;font-weight:700;";
        const b3 = document.createElement("button");
        b3.textContent = "light2d";
        b3.style.cssText = "flex:1 1 0;background:var(--comfy-input-bg);color:#22c55e;border:1px solid var(--border-color);border-radius:4px;padding:0 6px;min-height:28px;max-height:28px;height:28px;line-height:1;font-size:11px;cursor:pointer;font-weight:700;";
        const b4 = document.createElement("button");
        b4.textContent = "位置";
        b4.style.cssText = "flex:1 1 0;background:var(--comfy-input-bg);color:#38bdf8;border:1px solid var(--border-color);border-radius:4px;padding:0 6px;min-height:28px;max-height:28px;height:28px;line-height:1;font-size:11px;cursor:pointer;font-weight:700;";
        b1.onclick = open3D;
        b2.onclick = openPalette;
        b3.onclick = openLight2D;
        b4.onclick = openPosition;
        wrap.appendChild(b1);
        wrap.appendChild(b2);
        wrap.appendChild(b3);
        wrap.appendChild(b4);
        const btnWidget = node.addDOMWidget("text_mulangle_dual_buttons", "customwidget", wrap, { serialize: false, hideOnZoom: false });
        if (btnWidget) {
            btnWidget.computeSize = (width) => [Math.max(120, (width || 0) - 20), 34];
        }
        const spacer = document.createElement("div");
        spacer.style.height = "12px";
        spacer.style.minHeight = "12px";
        spacer.style.maxHeight = "12px";
        const spacerWidget = node.addDOMWidget("text_mulangle_buttons_bottom_spacer", "customwidget", spacer, { serialize: false, hideOnZoom: false });
        if (spacerWidget) {
            spacerWidget.computeSize = (width) => [Math.max(0, (width || 0) - 20), 12];
        }
    } else {
        if (!node.widgets || !node.widgets.find((w) => w.name === "editor_3d")) {
            node.addWidget("button", "editor_3d", "open", open3D);
        }
        if (!node.widgets || !node.widgets.find((w) => w.name === "色板")) {
            node.addWidget("button", "色板", "open", openPalette);
        }
        if (!node.widgets || !node.widgets.find((w) => w.name === "light2d")) {
            node.addWidget("button", "light2d", "open", openLight2D);
        }
        if (!node.widgets || !node.widgets.find((w) => w.name === "位置")) {
            node.addWidget("button", "位置", "open", openPosition);
        }
    }
    node._textMulAngleButtonsAttached = true;
    const shrinkToMin = () => {
        if (!node.computeSize) return;
        const minSize = node.computeSize();
        node.setSize(minSize);
    };
    shrinkToMin();

    if (!node._textMulAngleCompactHeightLocked) {
        const originalOnResize = node.onResize;
        const lockToMinHeight = () => {
            if (!node.computeSize || !Array.isArray(node.size)) return;
            const minSize = node.computeSize();
            const minH = Number(minSize?.[1] || 0);
            if (minH > 0 && node.size[1] > minH + 12) {
                node.size[1] = minH + 12;
            }
        };

        node.onResize = function () {
            const result = originalOnResize ? originalOnResize.apply(this, arguments) : undefined;
            lockToMinHeight();
            return result;
        };

        // Ensure existing oversized nodes are snapped back immediately.
        lockToMinHeight();
        requestAnimationFrame(() => {
            lockToMinHeight();
            shrinkToMin();
            setTimeout(() => {
                lockToMinHeight();
                shrinkToMin();
                if (app.graph) app.graph.setDirtyCanvas(true, true);
            }, 120);
        });
        node._textMulAngleCompactHeightLocked = true;
    }
}

app.registerExtension({
    name: "comfyui.APT_3dimageeditor",
    version: "1.0.0",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "text_interPrompt") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                ensureTextMulAngleButtons(this);
                return r;
            };
        }
    },
    async setup() {
        if (app.graph && app.graph._nodes) {
            app.graph._nodes.forEach(node => {
                if (node.constructor.nodeData && node.constructor.nodeData.name === "text_interPrompt") {
                    ensureTextMulAngleButtons(node);
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





