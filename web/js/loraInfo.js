import { app } from "../../../scripts/app.js";
import { LoraInfoDialog } from "../../ComfyUI-Custom-Scripts/js/modelInfo.js";

// 定义支持的节点类型和它们的LoRA参数名称模式
const supportedNodes = {
    "Stack_LoRA": {
        loraParamPattern: /^lora_name_\d+$/
    },
    "sum_lora": {
        loraParamPattern: /^lora_\d+$/
    },
    "sum_load_adv": {
        loraParamPattern: /^lora\d*$/
    },
    // 可以在这里添加更多节点类型...
};

app.registerExtension({
    name: "autotrigger.LoraInfo",
    beforeRegisterNodeDef(nodeType) {
        const nodeConfig = supportedNodes[nodeType.comfyClass];
        if (!nodeConfig) {
            return;
        }
        
        const getExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
            // 使用节点配置中的正则表达式识别LoRA小部件
            const loraWidgets = this.widgets.filter(w => nodeConfig.loraParamPattern.test(w.name));
            
            if (loraWidgets.length === 0) {
                return;
            }
            
            // 为每个LoRA添加单独的菜单项
            for (const widget of loraWidgets) {
                let value = widget.value;
                
                if (!value || (value === "None")) {
                    continue;
                }
                
                if (value.content) {
                    value = value.content;
                }
                
                // 提取编号部分（如果有的话）用于菜单项显示
                const match = widget.name.match(/\d+$/);
                const loraIndex = match ? match[0] : "";
                
                options.unshift({
                    content: `View info for LoRA ${loraIndex} (${value})`,
                    callback: async () => {
                        new LoraInfoDialog(value).show("loras", value);
                    },
                });
            }

            return getExtraMenuOptions?.apply(this, arguments);
        };
    }
});