import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "comfy.highway_node.Highway",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "Data_Highway") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = origOnNodeCreated?.apply(this, arguments);

                // 添加按钮：Update Ports
                this.addWidget("button", "update_ports", "Update Ports", () => {
                    this.updatePorts();
                    // 存储最后一次更新的端口配置
                    this.lastPortConfig = this.widgets.find(w => w.name === "port_config").value;
                });

                // 初始化时读取配置并更新端口
                const configWidget = this.widgets.find(w => w.name === "port_config");
                if (configWidget) {
                    const config = configWidget.value || "";
                    this.updatePortsWithConfig(config);
                    this.lastPortConfig = config;
                }

                return r;
            };

            // 新增方法：根据配置更新端口
            nodeType.prototype.updatePortsWithConfig = function (config) {
                const sections = config.split(";");

                // 清除已有端口（保留 bus）
                this.inputs = this.inputs.filter(i => i.name === "bus");
                this.outputs = this.outputs.filter(o => o.name === "bus");

                // 解析输入端口
                let inputNames = [];
                if (sections[0]?.trim()) {
                    inputNames = sections[0]
                        .trim()
                        .split(">")
                        .slice(1)
                        .map(n => n.trim())
                        .filter(n => n && n !== "bus");
                }

                // 解析输出端口，默认继承输入
                let outputNames = [];
                if (sections.length > 1 && sections[1]?.trim()) {
                    outputNames = sections[1]
                        .trim()
                        .split("<")
                        .slice(1)
                        .map(n => n.trim())
                        .filter(n => n && n !== "bus");
                } else {
                    outputNames = [...inputNames];
                }

                // 添加输入端口
                inputNames.forEach(name => {
                    this.addInput(name, "*");
                });

                // 添加输出端口
                outputNames.forEach(name => {
                    this.addOutput(name, "*");
                });

                console.log(`Ports updated: Inputs=[${inputNames.join(", ")}, bus], Outputs=[${outputNames.join(", ")}, bus]`);
            };

            nodeType.prototype.updatePorts = function () {
                const configWidget = this.widgets.find(w => w.name === "port_config");
                if (!configWidget) return;

                const config = configWidget.value || "";
                this.updatePortsWithConfig(config);
            };

            // 移除初始调用 updatePorts
            delete nodeType.prototype.onConnectionsChange;
        }
    },

    init() {
        console.log("Data_Highway loaded with dynamic ports support.");
    }
});