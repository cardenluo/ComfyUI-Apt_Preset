
import { app } from "../../scripts/app.js";



export const DynamicPorts = {
    setupDynamicPorts: function(nodeType, options) {
        const config = {
            startIndex: 1,
            type: "input",
            ...options
        };

        const isInput = config.type === "input";
        const portArrayKey = isInput ? "inputs" : "outputs";
        const addPortMethod = isInput ? "addInput" : "addOutput";
        const removePortMethod = isInput ? "removeInput" : "removeOutput";
        const isConnectedMethod = isInput ? "isInputConnected" : "isOutputConnected";

        const origOnNodeCreated = nodeType.prototype.onNodeCreated;
        const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;

        nodeType.prototype.onNodeCreated = function() {
            if (origOnNodeCreated) {
                origOnNodeCreated.apply(this, arguments);
            }

            const baseName = `${config.baseName}_${config.startIndex}`;
            if (!this[portArrayKey] || !this[portArrayKey].some(p => p.name === baseName)) {
                this[addPortMethod](baseName, config.dataType);
                if (isInput && config.secondaryName) {
                    this[addPortMethod](`${config.secondaryName}_${config.startIndex}`, config.secondaryDataType);
                }
            }

            this.properties = this.properties || {};
            this.properties[`dynamic${isInput ? 'Inputs' : 'Outputs'}`] = true;

            if (isInput) {
                for (let i = this[portArrayKey].length - 1; i >= 0; i--) {
                    const portName = this[portArrayKey][i].name;
                    if ((portName.startsWith(`${config.baseName}_${config.startIndex + 1}`) ||
                        (config.secondaryName && portName.startsWith(`${config.secondaryName}_${config.startIndex + 1}`)))) {
                        this[removePortMethod](i);
                    }
                }
            }
        };

        nodeType.prototype.onConnectionsChange = function(type, index, connected, link) {
            if (origOnConnectionsChange) {
                origOnConnectionsChange.apply(this, arguments);
            }

            const shouldProcess = (isInput && type === 1) || (!isInput && type === 2);

            if (shouldProcess) {
                if (isInput) {
                    setTimeout(() => this.updateDynamicPorts(), 10);
                } else {
                    this.updateDynamicPorts();
                }
            }
        };

        nodeType.prototype.updateDynamicPorts = function() {
            if (!this.graph) return;

            const groups = new Set();
            const connectedGroups = new Set();

            if (this[portArrayKey]) {
                this[portArrayKey].forEach((port, index) => {
                    if (port.name.startsWith(`${config.baseName}_`)) {
                        const parts = port.name.split("_");
                        const idx = parseInt(parts[parts.length - 1]);
                        if (!isNaN(idx)) {
                            groups.add(idx);
                            if (this[isConnectedMethod](index)) {
                                connectedGroups.add(idx);
                            }
                        }
                    }
                });
            }

            const maxIdx = connectedGroups.size > 0 ? Math.max(...connectedGroups) : 0;

            if (connectedGroups.size > 0 && groups.size <= maxIdx) {
                const nextIdx = groups.size + 1;
                const newPortName = `${config.baseName}_${nextIdx}`;
                this[addPortMethod](newPortName, config.dataType);

                if (isInput && config.secondaryName) {
                    this[addPortMethod](`${config.secondaryName}_${nextIdx}`, config.secondaryDataType);
                }

                if (this.graph) {
                    this.graph._version++;
                    this.setDirtyCanvas(true, true);
                }
            }

            Array.from(groups)
                .filter(idx => !connectedGroups.has(idx) && idx > maxIdx + 1)
                .sort((a, b) => b - a)
                .forEach(idx => {
                    if (isInput) {
                        for (let i = this[portArrayKey].length - 1; i >= 0; i--) {
                            const portName = this[portArrayKey][i].name;
                            if ((portName === `${config.baseName}_${idx}`) ||
                                (config.secondaryName && portName === `${config.secondaryName}_${idx}`)) {
                                this[removePortMethod](i);
                            }
                        }
                    } else {
                        for (let i = this[portArrayKey].length - 1; i >= 0; i--) {
                            const portName = this[portArrayKey][i].name;
                            const parts = portName.split("_");
                            const portIdx = parseInt(parts[parts.length - 1]);
                            if (!isNaN(portIdx) && portIdx === idx && portName.startsWith(`${config.baseName}_`)) {
                                this[removePortMethod](i);
                            }
                        }
                    }
                });

            if (this.computeSize) {
                this.computeSize();
            }
            if (this.setDirtyCanvas) {
                this.setDirtyCanvas(true, true);
            }
        };
    },

    setupDynamicInputs: function(nodeType, options) {
        this.setupDynamicPorts(nodeType, {
            type: "input",
            baseName: options.baseInputName,
            dataType: options.inputType,
            startIndex: options.startIndex || 1,
            secondaryName: options.secondaryInputName,
            secondaryDataType: options.secondaryInputType
        });
    },

    setupDynamicOutputs: function(nodeType, options) {
        this.setupDynamicPorts(nodeType, {
            type: "output",
            baseName: options.baseOutputName,
            dataType: options.outputType,
            startIndex: options.startIndex || 1
        });
    }
};



app.registerExtension({
    name: "view_Primitive_WorkflowChange",
    async beforeRegisterNodeDef() {
        const origLoadGraph = app.loadGraph;
        app.loadGraph = function (graphData) {
            const result = origLoadGraph.apply(this, arguments);
            setTimeout(() => {
                const nodes = app.graph._nodes_by_id;
                for (const nodeId in nodes) {
                    const node = nodes[nodeId];
                    if (node.type === "view_Primitive" && typeof node.restoreWidgets === "function") {
                        node.restoreWidgets();
                    }
                }
            }, 100);
            return result;
        };
    },
});

app.registerExtension({
    name: "./viewprimitive.js",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "view_Primitive") {
            DynamicPorts.setupDynamicPorts(nodeType, {
                type: "output",
                baseName: "connect_to_widget_input",
                dataType: "*",
                startIndex: 1,
                maxPorts: 1,
            });

            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            const origSerialize = nodeType.prototype.serialize;
            const origConfigure = nodeType.prototype.configure;

            nodeType.prototype.init = function () {
                this.state = {
                    connections: {},
                    widgetValues: {},
                    widgets: {},
                };
            };

            nodeType.prototype.onNodeCreated = function () {
                if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
                this.init();

                if (!this.outputs || this.outputs.length === 0) {
                    this.addOutput("connect_to_widget_input_1", "*");
                }
                if (this.outputs && this.outputs.length > 1) {
                    const firstOutput = this.outputs[0];
                    this.outputs = [firstOutput];
                }

                this.size = [this.size[0], 30];

                this.onResize = function () {
                    if (this.widgets) {
                        for (const widget of this.widgets) {
                            if (widget.fullLabel) widget.label = this.truncateLabel(widget.fullLabel);
                        }
                    }
                    this.setDirtyCanvas(true, true);
                };

                this.onWidgetChange = function (outputName, value) {
                    const connection = this.state.connections[outputName];
                    if (connection) {
                        const targetNode = this.graph._nodes_by_id[connection.nodeId];
                        if (targetNode && targetNode.widgets) {
                            const targetWidget = targetNode.widgets.find((w) => w.name === connection.inputName);
                            if (targetWidget) {
                                targetWidget.value = value;
                                if (targetWidget.callback) targetWidget.callback(value, targetWidget, targetNode);
                                if (this.graph) this.graph.setDirtyCanvas(true, true);
                            }
                        }
                    }

                    this.state.widgetValues[outputName] = value;
                    const outputIndex = this.outputs.findIndex((o) => o.name === outputName);
                    if (outputIndex >= 0) {
                        const outVal = this.formatValue(value, true);
                        this.setOutputData(outputIndex, outVal);
                        this.notifyConnectedNodes(outputIndex);
                    }

                    const widget = this.widgets.find((w) => w.name === outputName);
                    if (widget && widget.fullLabel) widget.label = this.truncateLabel(widget.fullLabel);

                    return value;
                };

                this.onNodeMoved = function () {
                    this.restoreWidgets();
                };
                this.onNodeSelected = function () {
                    this.restoreWidgets();
                };
            };

            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (origOnConnectionsChange) origOnConnectionsChange.call(this, type, index, connected, link_info);
                if (type !== LiteGraph.OUTPUT || !this.outputs || index >= this.outputs.length) return;

                const outputName = this.outputs[index].name;

                if (connected && link_info) {
                    const targetNode = this.graph._nodes_by_id[link_info.target_id];
                    if (!targetNode) return;

                    const targetSlot = link_info.target_slot;
                    const targetInput = targetNode.inputs[targetSlot];
                    if (!targetInput) return;

                    this.state.connections[outputName] = {
                        nodeId: link_info.target_id,
                        slot: targetSlot,
                        inputName: targetInput.name,
                        type: targetInput.type,
                    };

                    let targetWidget = null;
                    if (targetNode.widgets) {
                        targetWidget = targetNode.widgets.find((w) => w.name === targetInput.name);
                    }

                    if (targetWidget) {
                        this.outputs[index].label = targetInput.name;
                        this.createMatchingWidget(outputName, targetInput.type, targetWidget, targetNode);
                    } else {
                        this.setOutputData(index, [""]);
                        this.state.widgetValues[outputName] = "";
                    }
                } else {
                    this.removeWidget(outputName);
                    delete this.state.connections[outputName];
                    this.outputs[index].label = null;

                    const currentValue = this.getOutputData(index);
                    if (currentValue === undefined || currentValue === null) {
                        this.setOutputData(index, [""]);
                    }
                }
            };

            nodeType.prototype.serialize = function () {
                const data = origSerialize ? origSerialize.apply(this, arguments) : {};
                if (this.state) {
                    data.state = {
                        connections: this.state.connections,
                        widgetValues: this.state.widgetValues,
                        widgets: {},
                    };
                    if (this.widgets) {
                        for (const widget of this.widgets) {
                            if (widget.name) {
                                data.state.widgets[widget.name] = {
                                    name: widget.name,
                                    type: widget.type,
                                    value: widget.value,
                                    options: widget.options,
                                    fullLabel: widget.fullLabel,
                                };
                            }
                        }
                    }
                }
                return data;
            };

            nodeType.prototype.configure = function (info) {
                if (origConfigure) origConfigure.apply(this, arguments);
                if (info.state) {
                    this.state = this.state || {};
                    this.state.connections = info.state.connections || {};
                    this.state.widgetValues = info.state.widgetValues || {};
                    this.state.widgets = info.state.widgets || {};
                    setTimeout(() => this.restoreWidgets(), 10);
                }
            };

            // ----------- restoreWidgets (Sequence Preserved Version) -------------
            nodeType.prototype.restoreWidgets = function () {
                if (!this.state || !this.state.connections || !this.state.widgets) return;

                const originalWidgets = this.widgets ? [...this.widgets] : [];

                for (let oIndex = 0; oIndex < (this.outputs ? this.outputs.length : 0); oIndex++) {
                    const output = this.outputs[oIndex];
                    const outputName = output.name;
                    const connection = this.state.connections[outputName];
                    if (!connection) continue;

                    const targetNode = this.graph._nodes_by_id[connection.nodeId];
                    if (!targetNode) continue;
                    if (!output.links || output.links.length === 0) continue;

                    const hasValidLink = output.links.some((linkId) => {
                        const link = this.graph.links[linkId];
                        return link && link.target_id === connection.nodeId && link.target_slot === connection.slot;
                    });
                    if (!hasValidLink) continue;

                    const widgetConfig = this.state.widgets[outputName];
                    if (!widgetConfig) continue;

                    let targetWidget = null;
                    if (targetNode.widgets) {
                        targetWidget = targetNode.widgets.find((w) => w.name === connection.inputName);
                    }

                    const fakeTargetWidget = targetWidget || {
                        name: widgetConfig.name || connection.inputName,
                        type: widgetConfig.type || "string",
                        value: widgetConfig.value,
                        options: widgetConfig.options || {},
                        label: widgetConfig.fullLabel || widgetConfig.name,
                    };

                    let existingWidget = this.widgets.find((w) => w.name === outputName);
                    if (existingWidget) {
                        existingWidget.value = fakeTargetWidget.value;
                        existingWidget.label = this.truncateLabel(this.createLabel(targetNode, fakeTargetWidget));
                        existingWidget.fullLabel = this.createLabel(targetNode, fakeTargetWidget);
                    } else {
                        const widget = this.createMatchingWidget(outputName, connection.type, fakeTargetWidget, targetNode);
                        if (widget) this.widgets.push(widget);
                    }

                    this.state.widgetValues[outputName] = fakeTargetWidget.value;
                }

                if (this.computeSize) this.computeSize();
                this.setDirtyCanvas(true, true);
            };

            // ----------- Other methods keep v2 logic -------------
            nodeType.prototype.isComboType = function (type, targetWidget) {
                if (Array.isArray(type) || (typeof type === "string" && type.toUpperCase() === "COMBO")) return true;
                if (targetWidget.type === "combo") return true;
                if (targetWidget.options) {
                    return (
                        Array.isArray(targetWidget.options.values) ||
                        Array.isArray(targetWidget.options.options) ||
                        (typeof targetWidget.options.values === "object" && targetWidget.options.values !== null)
                    );
                }
                return false;
            };

            nodeType.prototype.formatValue = function (value, forOutput = true) {
                if (value === null || value === undefined) return forOutput ? [""] : "";
                if (!forOutput) return value;
                if (Array.isArray(value)) return value.length === 0 ? [""] : value;
                return [value];
            };

            nodeType.prototype.createLabel = function (targetNode, targetWidget) {
                const prefix = targetNode.title || targetNode.type || "Widget";
                const widgetName = targetWidget.label || targetWidget.name || "";
                return `${prefix}•${widgetName}`;
            };

            nodeType.prototype.createMatchingWidget = function (outputName, type, targetWidget, targetNode) {
                this.removeWidget(outputName);

                const fullLabel = this.createLabel(targetNode, targetWidget);

                let widget = null;

                if (this.isComboType(type, targetWidget)) {
                    widget = this.addWidget(
                        "combo",
                        outputName,
                        targetWidget.value,
                        (v) => this.onWidgetChange(outputName, v),
                        targetWidget.options
                    );
                } else {
                    let widgetType = "string";
                    if (type === "BOOLEAN") widgetType = "toggle";
                    else if (type === "INT" || type === "FLOAT") widgetType = "number";

                    widget = this.addWidget(
                        widgetType,
                        outputName,
                        targetWidget.value,
                        (v) => this.onWidgetChange(outputName, v),
                        targetWidget.options
                    );
                }

                if (widget) {
                    widget.fullLabel = fullLabel;
                    widget.label = this.truncateLabel(fullLabel);

                    this.state.widgetValues[outputName] = targetWidget.value;

                    const outputIndex = this.outputs.findIndex((o) => o.name === outputName);
                    if (outputIndex >= 0) {
                        this.setOutputData(outputIndex, this.formatValue(targetWidget.value, true));
                    }

                    this.state.widgets[outputName] = {
                        name: widget.name,
                        type: widget.type,
                        value: widget.value,
                        options: widget.options,
                        fullLabel: widget.fullLabel,
                    };
                }

                if (this.computeSize) this.computeSize();
                return widget;
            };

            nodeType.prototype.notifyConnectedNodes = function (outputIndex) {
                const output = this.outputs[outputIndex];
                if (!output || !output.links || output.links.length === 0) return;

                const outputValue = this.getOutputData(outputIndex) || [""];

                for (const linkId of output.links) {
                    const link = this.graph.links[linkId];
                    if (!link) continue;

                    const targetNode = this.graph.getNodeById(link.target_id);
                    if (!targetNode) continue;

                    this.notifyTargetNode(targetNode, link.target_slot, outputValue);
                }

                if (this.graph) {
                    this.graph._version++;
                    this.graph.setDirtyCanvas(true, true);
                }
            };

            nodeType.prototype.notifyTargetNode = function (targetNode, targetSlot, outputValue) {
                if (typeof targetNode.onInputChanged === "function") {
                    targetNode.onInputChanged(targetSlot, outputValue);
                }
                if (targetNode.inputs_data) {
                    const inputName = targetNode.inputs[targetSlot]?.name;
                    if (inputName) targetNode.inputs_data[inputName] = outputValue;
                }
                if (typeof targetNode.onExecuted === "function") {
                    targetNode.onExecuted();
                }
            };

            nodeType.prototype.removeWidget = function (name) {
                if (!this.widgets) return;
                this.widgets = this.widgets.filter((w) => w.name !== name);
                if (this.computeSize) this.computeSize();
                this.setDirtyCanvas(true, true);
            };

            nodeType.prototype.truncateLabel = function (label) {
                if (!label) return "";
                const nodeWidth = this.size[0];
                const margin = 20;
                const portWidth = 20;
                const availableWidth = nodeWidth - margin * 2 - portWidth;
                const charWidth = 8;
                const maxChars = Math.floor(availableWidth / charWidth);
                if (label.length <= maxChars) return label;
                return "..." + label.slice(-maxChars + 3);
            };

            nodeType.prototype.onExecute = function () {
                if (this.widgets && this.outputs) {
                    for (const widget of this.widgets) {
                        const outputIndex = this.outputs.findIndex((o) => o.name === widget.name);
                        if (outputIndex >= 0) {
                            this.setOutputData(outputIndex, this.formatValue(widget.value, true));
                        }
                    }
                }
            };

            nodeType.prototype.onExecuted = function (message) {
                try {
                    if (message?.data?.[0]) {
                        const data = message.data[0];
                        if (typeof data === "object" && data !== null) {
                            for (const key in data) {
                                if (key.startsWith("connect_to_widget_input_")) {
                                    const outputIndex = this.outputs.findIndex((o) => o.name === key);
                                    if (outputIndex >= 0) {
                                        const value = this.formatValue(data[key], true);
                                        this.setOutputData(outputIndex, value);
                                        this.state.widgetValues[key] = Array.isArray(value) && value.length === 1 ? value[0] : value;
                                        this.notifyConnectedNodes(outputIndex);
                                    }
                                }
                            }
                        }
                    }

                    if (this.widgets) {
                        for (const widget of this.widgets) {
                            const outputIndex = this.outputs.findIndex((o) => o.name === widget.name);
                            if (outputIndex >= 0) {
                                this.setOutputData(outputIndex, this.formatValue(widget.value, true));
                            }
                        }
                    }

                    this.setDirtyCanvas(true, true);
                } catch (err) {
                    console.error("Primitive (Advanced) onExecuted error:", err);
                }
            };
        }
    },
});


