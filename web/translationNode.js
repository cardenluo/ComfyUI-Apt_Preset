import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "ComfyUI-Apt_Collect",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "GPT_ChineseToEnglish" || nodeData.name === "GPT_EnglishToChinese") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function() {
                onNodeCreated?.apply(this, arguments);
                this.size = [300, 150];
                this.setDirtyCanvas(true, true);
                this.updateInputOpacity();
            };

            nodeType.prototype.updateInputOpacity = function() {
                const inputWidget = this.widgets.find(w => w.name === "input_text");
                if (inputWidget && inputWidget.inputEl) {
                    const isOptionalConnected = this.inputs.find(input => input.name === "optional_input_text" && input.link !== null);
                    inputWidget.inputEl.style.opacity = isOptionalConnected ? "0.5" : "1";
                }
            };

            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function(type, index, connected, link_info) {
                onConnectionsChange?.apply(this, arguments);
                this.updateInputOpacity();
            };

            function updateTranslation(translatedText) {
                if (Array.isArray(translatedText)) {
                    translatedText = translatedText.join('');
                } else if (typeof translatedText !== 'string') {
                    translatedText = String(translatedText);
                }

                if (!this.resultWidget) {
                    this.resultWidget = ComfyWidgets["STRING"](this, "translated_text", ["STRING", { multiline: true }], app).widget;
                    this.resultWidget.inputEl.readOnly = true;
                    this.resultWidget.inputEl.style.opacity = 0.6;
                }

                this.resultWidget.value = translatedText;

                requestAnimationFrame(() => {
                    const sz = this.computeSize();
                    sz[0] = Math.max(sz[0], this.size[0]);
                    sz[1] = Math.max(sz[1], this.size[1]);
                    this.onResize?.(sz);
                    app.graph.setDirtyCanvas(true, false);
                });
            }

            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                if (message.text) {
                    updateTranslation.call(this, message.text);
                }
            };

            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (config) {
                onConfigure?.apply(this, arguments);
                if (config.widgets_values?.length > 1) {
                    updateTranslation.call(this, config.widgets_values[1]);
                }
                this.updateInputOpacity();
            };
        }
    },
});


