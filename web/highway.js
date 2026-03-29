import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "comfy.highway_node.Highway",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "Data_Highway") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = origOnNodeCreated?.apply(this, arguments);

                // 隐藏Python端预留的20个固定端口（output_0到output_19）
                // 只显示动态创建的端口和bus端口
                for (let i = this.outputs.length - 1; i >= 0; i--) {
                    if (this.outputs[i].name.startsWith("output_")) {
                        this.removeOutput(i);
                    }
                }

                // 添加按钮：Update Ports
                this.addWidget("button", "update_ports", "Update Ports", () => {
                    this.updatePorts();
                    // 存储最后一次更新的端口配置
                    this.lastPortConfig = this.widgets.find(w => w.name === "port_config").value;
                });

                // 初始化时不更新端口，避免破坏已有连接
                // 用户需要手动点击"Update Ports"按钮来更新端口
                console.log("Data_Highway node created. Click 'Update Ports' button to update ports.");

                return r;
            };



            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (config) {
                const r = origOnConfigure?.apply(this, arguments);

                // 隐藏Python端预留的20个固定端口（output_0到output_19）
                // 刷新加载时，这些端口会重新出现，需要再次隐藏
                for (let i = this.outputs.length - 1; i >= 0; i--) {
                    if (this.outputs[i].name.startsWith("output_")) {
                        this.removeOutput(i);
                    }
                }

                // 不自动更新端口，避免破坏已有连接
                // 配置已从工作流加载，但不会自动应用到端口
                // 用户需要手动点击"Update Ports"按钮来更新端口
                console.log("Data_Highway configured. Click 'Update Ports' button to update ports.");

                return r;
            };

            // 新增方法：根据配置更新端口（智能更新，只改变需要改变的端口）
            nodeType.prototype.updatePortsWithConfig = function (config) {
                const sections = config.split(";");

                if (this._processingPorts) return;
                this._processingPorts = true;
                try {
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

                    // 获取当前端口（排除bus和output_X）
                    const currentInputNames = this.inputs
                        .filter(p => p.name !== "bus")
                        .map(p => p.name);
                    const currentOutputNames = this.outputs
                        .filter(p => p.name !== "bus" && !p.name.startsWith("output_"))
                        .map(p => p.name);

                    // 检查端口是否真的需要更新
                    const inputsChanged = JSON.stringify(inputNames) !== JSON.stringify(currentInputNames);
                    const outputsChanged = JSON.stringify(outputNames) !== JSON.stringify(currentOutputNames);

                    if (!inputsChanged && !outputsChanged) {
                        console.log("Ports configuration unchanged, skipping update.");
                        return;
                    }

                    // 获取需要保留的端口名称（有连接的端口）
                    const connectedInputNames = this.inputs
                        .filter(p => p.name !== "bus" && p.link !== null)
                        .map(p => p.name);
                    const connectedOutputNames = this.outputs
                        .filter(p => p.name !== "bus" && !p.name.startsWith("output_") && p.links && p.links.length > 0)
                        .map(p => p.name);

                    // 智能更新：只删除没有连接且不需要的端口
                    // 删除输入端口
                    for (let i = this.inputs.length - 1; i >= 0; i--) {
                        const port = this.inputs[i];
                        if (port.name !== "bus" &&
                            !inputNames.includes(port.name) &&
                            !connectedInputNames.includes(port.name)) {
                            this.removeInput(i);
                        }
                    }

                    // 删除输出端口（同时删除Python端的output_X端口）
                    for (let i = this.outputs.length - 1; i >= 0; i--) {
                        const port = this.outputs[i];
                        if (port.name !== "bus" &&
                            port.name.startsWith("output_")) {
                            // 直接删除所有Python端的output_X端口
                            this.removeOutput(i);
                        } else if (port.name !== "bus" &&
                            !outputNames.includes(port.name) &&
                            !connectedOutputNames.includes(port.name)) {
                            // 删除不需要的动态端口
                            this.removeOutput(i);
                        }
                    }

                    // 添加新端口（不存在的）
                    inputNames.forEach(name => {
                        if (!this.inputs.find(p => p.name === name)) {
                            this.addInput(name, "*");
                        }
                    });

                    outputNames.forEach(name => {
                        if (!this.outputs.find(p => p.name === name)) {
                            this.addOutput(name, "*");
                        }
                    });

                    this.setSize(this.computeSize());
                    app.canvas.setDirty(true);
                    console.log(`Ports updated: Inputs=[${inputNames.join(", ")}, bus], Outputs=[${outputNames.join(", ")}, bus]`);
                } finally {
                    this._processingPorts = false;
                }
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
