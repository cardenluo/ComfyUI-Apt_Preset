import { app } from "../../../scripts/app.js";


function createModal(htmlContent) {
    const modal = document.createElement('div');
    modal.id = 'preeditor-modal';
    modal.innerHTML = htmlContent;
    document.body.appendChild(modal);
    return modal;
}

function closeModal(modal, stylesheet) {
    if (modal) modal.remove();
    if (stylesheet) stylesheet.remove();
}


app.registerExtension({
    name: "color_select",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "color_select") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated?.apply(this, arguments);
                const node = this;
                const widget = {
                    data: this.widgets.find(w => w.name === "color_code"),
                };
                this.addWidget("button", "select_color", null, () => showColorPickerModal(node, widget));
            };
        }
    },
});




function showColorPickerModal(node, widget) {
    const modalHtml = `
        <div class="premodal" id="precolor-picker-modal">
            <div class="premodal-content" style="max-width: 600px; padding: 30px;">
                <style>
                    .premodal { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); display: flex; justify-content: center; align-items: center; z-index: 1001; }
                    .premodal-content { background: #222; padding: 30px; border-radius: 8px; max-width: 90vw; max-height: 90vh; display: flex; flex-direction: column; gap: 15px; }
                    .precolor-controls { display: flex; flex-direction: column; gap: 22.5px; align-items: center; }
                    .precolor-picker-input { width: 150px; height: 150px; border: none; padding: 0; background: none; cursor: pointer; }
                    .preeditor-btn { padding: 12px 18px; color: white; border: none; border-radius: 6px; cursor: pointer; }
                    .prebutton-group { display: flex; justify-content: center; width: 100%; gap: 15px; }
                </style>
                <div class="precolor-controls">
                    <label for="precolor-picker-input" style="color: white; font-size: 24px;">select_color:</label>
                    <input type="color" id="precolor-picker-input" class="precolor-picker-input" value="${widget.data.value || '#FFFFFF'}">
                    <div class="prebutton-group">
                        <button id="preeyedropper-btn" class="preeditor-btn" style="background-color: #5bc0de;">💉</button>
                        <button id="preconfirm-color-btn" class="preeditor-btn" style="background-color: #4CAF50;">Yes</button>
                        <button id="precancel-color-btn" class="preeditor-btn" style="background-color: #f44336;">No</button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const modal = createModal(modalHtml);
    const colorInput = modal.querySelector('#precolor-picker-input');
    const confirmBtn = modal.querySelector('#preconfirm-color-btn');
    const cancelBtn = modal.querySelector('#precancel-color-btn');
    const eyedropperBtn = modal.querySelector('#preeyedropper-btn');

    if (!window.EyeDropper) {
        eyedropperBtn.textContent = "The browser does not support the color picker.";
        eyedropperBtn.disabled = true;
    } else {
        eyedropperBtn.onclick = async () => {
            const eyeDropper = new EyeDropper();
            try {
                modal.style.display = 'none';
                const result = await eyeDropper.open();
                const selectedColor = result.sRGBHex.toUpperCase();
                colorInput.value = selectedColor;
            } catch (e) { console.log("Straws are canceled"); } 
            finally { modal.style.display = 'flex'; }
        };
    }

    confirmBtn.onclick = () => {
        widget.data.value = colorInput.value.toUpperCase();
        if (node.onWidgetValue_changed) { node.onWidgetValue_changed(widget.data.name, widget.data.value); }
        closeModal(modal);
    };
    cancelBtn.onclick = () => { closeModal(modal); };
}


