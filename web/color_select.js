// apt_color_picker.js
import { app } from "../../../scripts/app.js";

// 加载外部库的通用函数
function loadScript(url) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${url}"]`)) {
            resolve();
            return;
        }
        const script = document.createElement('script');
        script.src = url;
        script.onload = resolve;
        script.onerror = reject;
        document.body.appendChild(script);
    });
}

// 创建和关闭模态框的辅助函数
function createModal(htmlContent) {
    const modal = document.createElement('div');
    modal.id = 'apt-colorpicker-modal';
    modal.innerHTML = htmlContent;
    document.body.appendChild(modal);
    return modal;
}

function closeModal(modal, stylesheet) {
    if (modal) modal.remove();
    if (stylesheet) stylesheet.remove();
}

// 获取扩展基础路径
function get_extension_base_path() {
    const scriptUrl = import.meta.url;
    const parts = scriptUrl.split('/');
    const extensionsIndex = parts.indexOf('extensions');
    if (extensionsIndex !== -1 && parts.length > extensionsIndex + 1) {
        return '/' + parts.slice(extensionsIndex, extensionsIndex + 2).join('/') + '/';
    }
    return '/extensions/ComfyUI-ZML-Image/';
}

// 显示颜色选择器模态框
function showColorPickerModal(node, widget) {
    const upstreamNode = node.getInputNode(0);
    const hasImage = upstreamNode && upstreamNode.imgs && upstreamNode.imgs.length > 0;
    
    // 如果没有图像连接，使用基本颜色选择器
    if (!hasImage) {
        showBasicColorPicker(node, widget);
        return;
    }

    const imageUrl = upstreamNode.imgs[0].src;
    const extensionBasePath = get_extension_base_path();
    
    const modalHtml = `
        <div class="apt-modal">
            <div class="apt-modal-content">
                <style>
                    .apt-modal { 
                        position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                        background: rgba(0,0,0,0.7); 
                        display: flex; justify-content: center; align-items: center; 
                        z-index: 1001; 
                    }
                    .apt-modal-content { 
                        background: #222; padding: 20px; border-radius: 8px; 
                        max-width: 90vw; max-height: 90vh; 
                        display: flex; flex-direction: column; gap: 10px; 
                    }
                    .apt-editor-main { 
                        flex-grow: 1; overflow: hidden; 
                        display: flex; justify-content: center; align-items: center; 
                    }
                    .apt-editor-controls { 
                        display: flex; justify-content: space-around; align-items: center; 
                        flex-wrap: wrap; gap: 10px;
                    }
                    .apt-editor-btn { 
                        padding: 8px 12px; color: white; border: none; 
                        border-radius: 4px; cursor: pointer; 
                    }
                    #apt-color-preview {
                        width: 100px; height: 30px; border: 1px solid #ccc;
                        margin: 0 10px;
                    }
                </style>
                <div class="apt-editor-main">
                    <canvas id="apt-colorpicker-canvas" style="max-width: 100%; max-height: 75vh;"></canvas>
                </div>
                <div style="display: flex; align-items: center; justify-content: center;">
                    <span>选中的颜色:</span>
                    <div id="apt-color-preview"></div>
                    <span id="apt-color-value"></span>
                </div>
                <div class="apt-editor-controls">
                    <button id="apt-confirm-btn" class="apt-editor-btn" style="background-color: #4CAF50;">确认</button>
                    <button id="apt-cancel-btn" class="apt-editor-btn" style="background-color: #f44336;">取消</button>
                </div>
            </div>
        </div>
    `;

    const modal = createModal(modalHtml);
    const canvas = modal.querySelector('#apt-colorpicker-canvas');
    const ctx = canvas.getContext('2d');
    const colorPreview = modal.querySelector('#apt-color-preview');
    const colorValue = modal.querySelector('#apt-color-value');
    
    let selectedColor = '#000000';

    // 加载图像并设置画布
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);

        // 点击画布获取颜色
        canvas.addEventListener('click', (e) => {
            const rect = canvas.getBoundingClientRect();
            const x = Math.floor((e.clientX - rect.left) * (canvas.width / rect.width));
            const y = Math.floor((e.clientY - rect.top) * (canvas.height / rect.height));
            
            const pixel = ctx.getImageData(x, y, 1, 1).data;
            selectedColor = `#${pixel[0].toString(16).padStart(2, '0')}${pixel[1].toString(16).padStart(2, '0')}${pixel[2].toString(16).padStart(2, '0')}`;
            
            colorPreview.style.backgroundColor = selectedColor;
            colorValue.textContent = selectedColor.toUpperCase();
        });

        // 触发一次点击事件以显示默认颜色（左上角）
        canvas.dispatchEvent(new MouseEvent('click', {
            clientX: canvas.getBoundingClientRect().left,
            clientY: canvas.getBoundingClientRect().top
        }));
    };

    // 确认按钮事件
    modal.querySelector('#apt-confirm-btn').onclick = () => {
        widget.data.value = selectedColor;
        node.onWidgetValue_changed?.(widget.data, widget.data.value);
        closeModal(modal);
    };

    // 取消按钮事件
    modal.querySelector('#apt-cancel-btn').onclick = () => closeModal(modal);
}

// 显示基本颜色选择器
function showBasicColorPicker(node, widget) {
    // 预定义颜色
    const presetColors = [
        { name: "白色", hex: "#FFFFFF" },
        { name: "黑色", hex: "#000000" },
        { name: "红色", hex: "#FF0000" },
        { name: "绿色", hex: "#00FF00" },
        { name: "蓝色", hex: "#0000FF" },
        { name: "黄色", hex: "#FFFF00" },
        { name: "青色", hex: "#00FFFF" },
        { name: "品红", hex: "#FF00FF" },
        { name: "橙色", hex: "#FFA500" },
        { name: "紫色", hex: "#800080" },
        { name: "粉色", hex: "#FFC0CB" },
        { name: "棕色", hex: "#A52A2A" },
        { name: "灰色", hex: "#808080" },
        { name: "浅灰", hex: "#D3D3D3" },
        { name: "深灰", hex: "#A9A9A9" },
        { name: "橄榄色", hex: "#808000" },
        { name: "酸橙绿", hex: "#008000" },
        { name: "蓝绿", hex: "#008080" },
        { name: "海军蓝", hex: "#000080" },
        { name: "栗色", hex: "#800000" },
        { name: "紫红色", hex: "#FF0080" },
        { name: "水色", hex: "#00FF80" },
        { name: "银色", hex: "#C0C0C0" },
        { name: "金色", hex: "#FFD700" },
        { name: "绿松石", hex: "#40E0D0" },
        { name: "淡紫色", hex: "#E6E6FA" },
        { name: "紫罗兰", hex: "#EE82EE" },
        { name: "珊瑚色", hex: "#FF7F50" },
        { name: "靛蓝色", hex: "#4B0082" }
    ];
    
    const modalHtml = `
        <div class="apt-modal">
            <div class="apt-modal-content">
                <style>
                    .apt-modal { 
                        position: fixed; top: 0; left: 0; width: 100%; height: 100%; 
                        background: rgba(0,0,0,0.7); 
                        display: flex; justify-content: center; align-items: center; 
                        z-index: 1001; 
                    }
                    .apt-modal-content { 
                        background: #222; padding: 20px; border-radius: 8px; 
                        max-width: 500px; 
                        display: flex; flex-direction: column; gap: 15px; 
                    }
                    .apt-color-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(60px, 1fr));
                        gap: 10px;
                    }
                    .apt-color-item {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        gap: 5px;
                        cursor: pointer;
                        padding: 5px;
                        border-radius: 4px;
                        transition: background-color 0.2s;
                    }
                    .apt-color-item:hover {
                        background-color: rgba(255,255,255,0.1);
                    }
                    .apt-color-swatch {
                        width: 40px;
                        height: 40px;
                        border-radius: 4px;
                        border: 1px solid #555;
                    }
                    .apt-color-name {
                        font-size: 10px;
                        color: #ccc;
                        text-align: center;
                    }
                    .apt-color-hex {
                        font-size: 9px;
                        color: #999;
                    }
                    .apt-editor-controls { 
                        display: flex; justify-content: space-around; align-items: center; 
                        flex-wrap: wrap; gap: 10px;
                    }
                    .apt-editor-btn { 
                        padding: 8px 12px; color: white; border: none; 
                        border-radius: 4px; cursor: pointer; 
                    }
                    .apt-custom-input {
                        display: flex;
                        align-items: center;
                        gap: 10px;
                    }
                    .apt-custom-input input {
                        background: #333;
                        color: white;
                        border: 1px solid #555;
                        padding: 5px;
                        border-radius: 4px;
                        width: 100px;
                    }
                </style>
                <div class="apt-color-grid">
                    ${presetColors.map(color => `
                        <div class="apt-color-item" data-hex="${color.hex}">
                            <div class="apt-color-swatch" style="background-color: ${color.hex}"></div>
                            <div class="apt-color-name">${color.name}</div>
                            <div class="apt-color-hex">${color.hex}</div>
                        </div>
                    `).join('')}
                </div>
                <div class="apt-custom-input">
                    <span>自定义颜色:</span>
                    <input type="color" id="apt-color-picker" value="#0E0B0B">
                    <span>或输入HEX:</span>
                    <input type="text" id="apt-color-text" value="#0E0B0B" maxlength="7">
                </div>
                <div class="apt-editor-controls">
                    <button id="apt-confirm-btn" class="apt-editor-btn" style="background-color: #4CAF50;">确认</button>
                    <button id="apt-cancel-btn" class="apt-editor-btn" style="background-color: #f44336;">取消</button>
                </div>
            </div>
        </div>
    `;

    const modal = createModal(modalHtml);
    const colorPicker = modal.querySelector('#apt-color-picker');
    const colorText = modal.querySelector('#apt-color-text');
    let selectedColor = '#0E0B0B';
    
    // 预设颜色点击事件
    modal.querySelectorAll('.apt-color-item').forEach(item => {
        item.addEventListener('click', () => {
            selectedColor = item.getAttribute('data-hex');
            colorPicker.value = selectedColor;
            colorText.value = selectedColor;
            
            // 高亮选中的颜色
            modal.querySelectorAll('.apt-color-item').forEach(el => {
                el.style.backgroundColor = '';
            });
            item.style.backgroundColor = 'rgba(255,255,255,0.2)';
        });
    });
    
    // 自定义颜色选择器变化事件
    colorPicker.addEventListener('change', (e) => {
        selectedColor = e.target.value;
        colorText.value = selectedColor;
    });
    
    // 文本输入变化事件
    colorText.addEventListener('input', (e) => {
        const value = e.target.value;
        if (value.match(/^#[0-9A-Fa-f]{6}$/)) {
            selectedColor = value;
            colorPicker.value = selectedColor;
        }
    });

    // 确认按钮事件
    modal.querySelector('#apt-confirm-btn').onclick = () => {
        widget.data.value = selectedColor;
        node.onWidgetValue_changed?.(widget.data, widget.data.value);
        closeModal(modal);
    };

    // 取消按钮事件
    modal.querySelector('#apt-cancel-btn').onclick = () => closeModal(modal);
}

// 注册扩展
app.registerExtension({
    name: "apt.colorselect",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "basicIn_color") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const node = this;
                const widget = {
                    data: this.widgets.find(w => w.name === "hex_str"),
                };
                this.addWidget("button", "hex_str", null, () => showColorPickerModal(node, widget));
            };
        }
    },
});