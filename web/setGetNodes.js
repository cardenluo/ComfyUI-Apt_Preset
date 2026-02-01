import { app } from "../../../scripts/app.js";

const LGraphNode = LiteGraph.LGraphNode;

function getLocalizedCategory(subCategory) {
    const settings = app?.ui?.settings;
    const lang =
        settings?.getSettingValue?.("Comfy.Locale") ||
        settings?.getSettingValue?.("Comfy.Language") ||
        "en-US";
    
    if (lang.startsWith("zh")) {
        return "Apt_Preset/流程";
    } else {
        return "Apt_Preset/" + subCategory;
    }
}

app.registerExtension({
    name: "Apt.SetNode",
    registerCustomNodes() {
        class SetNode extends LGraphNode {
            defaultVisibility = true;
            serialize_widgets = true;
            
            constructor(title) {
                super(title);
                if (!this.properties) {
                    this.properties = {
                        "previousName": ""
                    };
                }
                
                const node = this;
                
                this.addWidget(
                    "text",
                    "Variable",
                    '',
                    (s, t, u, v, x) => {
                        node.validateName(node.graph);
                        if (this.widgets[0].value !== '') {
                            this.title = "Set_" + this.widgets[0].value;
                        }
                        this.update();
                        this.properties.previousName = this.widgets[0].value;
                    },
                    {}
                );
                
                this.addInput("*", "*");
                this.addOutput("*", '*');
                
                this.onConnectionsChange = function(
                    slotType,
                    slot,
                    isChangeConnect,
                    link_info,
                    output
                ) {
                    if (slotType === 1 && !isChangeConnect) {
                        if (this.inputs && this.inputs[slot]) {
                            this.inputs[slot].type = '*';
                            this.inputs[slot].name = '*';
                        }
                        if (this.outputs && this.outputs[0]) {
                            this.outputs[0].type = '*';
                            this.outputs[0].name = '*';
                        }
                        const variableName = this.widgets?.[0]?.value;
                        this.title = variableName ? "Set_" + variableName : "Set_";
                    }
                    if (slotType === 2 && !isChangeConnect) {
                        if (this.outputs && this.outputs[slot]) {
                            this.outputs[slot].type = '*';
                            this.outputs[slot].name = '*';
                        }
                    }
                    if (link_info && node.graph && slotType === 1 && isChangeConnect) {
                        const fromNode = node.graph._nodes.find((otherNode) => String(otherNode.id) === String(link_info.origin_id));
                        
                        if (fromNode && fromNode.outputs && fromNode.outputs[link_info.origin_slot]) {
                            const type = fromNode.outputs[link_info.origin_slot].type;
                            
                            if (this.title === "Set_") {
                                this.title = "Set_" + type;
                            }
                            if (this.widgets[0].value === '') {
                                this.widgets[0].value = type
                            }
                            
                            this.validateName(node.graph);
                            this.inputs[0].type = type;
                            this.inputs[0].name = type;
                        }
                    }
                    if (link_info && node.graph && slotType === 2 && isChangeConnect) {
                        let type = this.inputs && this.inputs[0] ? this.inputs[0].type : '*';
                        if (type === '*' && link_info && node.graph) {
                            const targetNode = node.graph._nodes.find((otherNode) => String(otherNode.id) === String(link_info.target_id));
                            const targetInput = targetNode?.inputs?.[link_info.target_slot];
                            const inferred = targetInput?.type;
                            if (typeof inferred === "string" && inferred.length > 0 && inferred !== "*") {
                                type = inferred;
                                if (this.inputs && this.inputs[0]) {
                                    this.inputs[0].type = type;
                                    this.inputs[0].name = type;
                                }
                            }
                        }
                        if (this.outputs && this.outputs[0]) {
                            this.outputs[0].type = type;
                            this.outputs[0].name = type;
                        }
                    }
                    
                    this.update();
                }
                
                this.validateName = function(graph) {
                    let widgetValue = node.widgets[0].value;
                    if (!graph || !graph._nodes) {
                        return;
                    }
                    
                    if (widgetValue !== '') {
                        const baseValue = widgetValue;
                        let tries = 0;
                        const maxTries = 100;
                        const existingValues = new Set();
                        
                        graph._nodes.forEach(otherNode => {
                            if (otherNode !== this && otherNode.type === 'flow_Set_Value') {
                                const value = otherNode.widgets?.[0]?.value;
                                if (typeof value === "string" && value.length > 0) {
                                    existingValues.add(value);
                                }
                            }
                        });
                        
                        while (existingValues.has(widgetValue) && tries < maxTries) {
                            widgetValue = baseValue + "_" + tries;
                            tries++;
                        }
                        if (tries >= maxTries) {
                            widgetValue = baseValue + "_" + Math.random().toString(36).slice(2, 10);
                        }
                        
                        node.widgets[0].value = widgetValue;
                        this.update();
                    }
                }
                
                this.clone = function () {
                    const cloned = SetNode.prototype.clone.apply(this);
                    cloned.inputs[0].name = '*';
                    cloned.inputs[0].type = '*';
                    if (cloned.widgets && cloned.widgets[0]) {
                        cloned.widgets[0].value = '';
                    }
                    cloned.title = "Set_";
                    cloned.properties.previousName = '';
                    cloned.size = cloned.computeSize();
                    return cloned;
                };
                
                this.onAdded = function(graph) {
                    this.validateName(graph);
                }
                
                this.update = function() {
                    if (!node.graph) {
                        return;
                    }
                    
                    const getters = this.findGetters(node.graph);
                    getters.forEach(getter => {
                        getter.setType(this.inputs[0].type);
                    });
                    
                    if (this.widgets[0].value) {
                        const gettersWithPreviousName = this.findGetters(node.graph, true);
                        gettersWithPreviousName.forEach(getter => {
                            getter.setName(this.widgets[0].value);
                        });
                    }
                    
                    const allGetters = node.graph._nodes.filter(otherNode => otherNode.type === "flow_Get_Value");
                    allGetters.forEach(otherNode => {
                        if (otherNode.setComboValues) {
                            otherNode.setComboValues();
                        }
                    });
                }
                
                this.findGetters = function(graph, checkForPreviousName) {
                    if (!graph || !graph._nodes) {
                        return [];
                    }
                    const name = checkForPreviousName ? this.properties.previousName : this.widgets[0].value;
                    if (!name) {
                        return [];
                    }
                    return graph._nodes.filter(otherNode => otherNode.type === 'flow_Get_Value' && otherNode.widgets?.[0]?.value === name);
                }
                
                this.isVirtualNode = true;
            }
            
            onRemoved() {
                const nodes = this.graph && this.graph._nodes ? this.graph._nodes : [];
                const allGetters = nodes.filter((otherNode) => otherNode.type === "flow_Get_Value");
                allGetters.forEach((otherNode) => {
                    if (otherNode.setComboValues) {
                        otherNode.setComboValues([this]);
                    }
                })
            }
        }
        
        LiteGraph.registerNodeType(
            "flow_Set_Value",
            Object.assign(SetNode, {
                title: "flow_Set_Value",
                description: "",
            })
        );
        
        SetNode.category = getLocalizedCategory("flow");
    },
});

app.registerExtension({
    name: "Apt.GetNode",
    registerCustomNodes() {
        class GetNode extends LGraphNode {
            defaultVisibility = true;
            serialize_widgets = true;
            
            constructor(title) {
                super(title);
                if (!this.properties) {
                    this.properties = {};
                }
                
                const node = this;
                this.addWidget(
                    "combo",
                    "Variable",
                    "",
                    (e) => {
                        this.onRename();
                    },
                    {
                        values: () => {
                            const nodes = node.graph && node.graph._nodes ? node.graph._nodes : [];
                            const setterNodes = nodes.filter((otherNode) => otherNode.type === 'flow_Set_Value');
                            const values = setterNodes
                                .map((otherNode) => otherNode.widgets?.[0]?.value)
                                .filter((v) => typeof v === "string" && v.length > 0);
                            return Array.from(new Set(values)).sort();
                        }
                    }
                );
                
                this.addOutput("*", '*');
                
                this.onConnectionsChange = function(
                    slotType,
                    slot,
                    isChangeConnect,
                    link_info,
                    output
                ) {
                    this.validateLinks();
                }
                
                this.setName = function(name) {
                    node.widgets[0].value = name;
                    node.onRename();
                    node.serialize();
                }
                
                this.onRename = function() {
                    const setter = this.findSetter(node.graph);
                    if (setter) {
                        const linkType = setter.inputs?.[0]?.type ?? "*";
                        
                        this.setType(linkType);
                        const variableName = setter.widgets?.[0]?.value ?? "";
                        this.title = variableName ? "Get_" + variableName : "Get_";
                        
                    } else {
                        this.setType('*');
                        this.title = "Get_";
                    }
                }
                
                this.clone = function () {
                    const cloned = GetNode.prototype.clone.apply(this);
                    if (cloned.widgets && cloned.widgets[0]) {
                        cloned.widgets[0].value = "";
                    }
                    cloned.title = "Get_";
                    if (cloned.outputs && cloned.outputs[0]) {
                        cloned.outputs[0].type = "*";
                        cloned.outputs[0].name = "*";
                    }
                    cloned.size = cloned.computeSize();
                    return cloned;
                };
                
                this.validateLinks = function() {
                    const outputSlot = this.outputs && this.outputs[0] ? this.outputs[0] : null;
                    if (!outputSlot || outputSlot.type === '*' || !outputSlot.links || !node.graph || !node.graph.links) {
                        return;
                    }
                    outputSlot.links
                        .filter((linkId) => {
                            const link = node.graph.links[linkId];
                            if (!link) return false;
                            const linkType = typeof link.type === "string" ? link.type : "*";
                            if (linkType === "*") return false;
                            return !linkType.split(",").includes(outputSlot.type);
                        })
                        .forEach((linkId) => {
                            node.graph.removeLink(linkId);
                        });
                };
                
                this.setType = function(type) {
                    if (!this.outputs || !this.outputs[0]) {
                        return;
                    }
                    this.outputs[0].name = type;
                    this.outputs[0].type = type;
                    this.validateLinks();
                }
                
                this.findSetter = function(graph) {
                    if (!graph || !graph._nodes) {
                        return null;
                    }
                    const name = this.widgets?.[0]?.value;
                    if (!name) {
                        return null;
                    }
                    const foundNode = graph._nodes.find(otherNode => otherNode.type === 'flow_Set_Value' && otherNode.widgets?.[0]?.value === name);
                    return foundNode;
                };
                
                this.setComboValues = function(removedSetters) {
                    const widget = this.widgets && this.widgets[0] ? this.widgets[0] : null;
                    if (!widget || !widget.options) {
                        return;
                    }
                    
                    widget.options.values = () => {
                        const nodes = node.graph && node.graph._nodes ? node.graph._nodes : [];
                        const setterNodes = nodes.filter((otherNode) => 
                            otherNode.type === 'flow_Set_Value' && 
                            !removedSetters?.includes(otherNode)
                        );
                        const values = setterNodes
                            .map((otherNode) => otherNode.widgets?.[0]?.value)
                            .filter((v) => typeof v === "string" && v.length > 0);
                        return Array.from(new Set(values)).sort();
                    };

                    const currentValue = widget.value;
                    const values = widget.options.values();
                    const removedNames = Array.isArray(removedSetters)
                        ? new Set(
                            removedSetters
                                .map((n) => n?.widgets?.[0]?.value)
                                .filter((v) => typeof v === "string" && v.length > 0)
                        )
                        : null;
                    
                    const exists = typeof currentValue === "string" && currentValue.length > 0 && values.includes(currentValue);
                    const removed = removedNames ? removedNames.has(currentValue) : false;
                    
                    if (!exists || removed) {
                        widget.value = "";
                        this.setType("*");
                        this.title = "Get_";
                    } else {
                        this.onRename();
                    }
                    
                    if (app?.graph?.setDirtyCanvas) {
                        app.graph.setDirtyCanvas(true, true);
                    }
                }
                
                this.isVirtualNode = true;
            }
            
            getInputLink(slot) {
                const setter = this.findSetter(this.graph);
                
                if (setter) {
                    const slotInfo = setter.inputs ? setter.inputs[slot] : null;
                    let linkId = slotInfo ? slotInfo.link : null;
                    if (linkId == null && slotInfo && Array.isArray(slotInfo.links) && slotInfo.links.length > 0) {
                        linkId = slotInfo.links[0];
                    }
                    const link = linkId != null && this.graph && this.graph.links ? this.graph.links[linkId] : null;
                    return link || null;
                } else {
                    console.warn("No SetNode found for " + this.widgets?.[0]?.value + "(" + this.type + ")");
                }
            }
        }
        
        LiteGraph.registerNodeType(
            "flow_Get_Value",
            Object.assign(GetNode, {
                title: "flow_Get_Value",
                description: "",
            })
        );
        
        GetNode.category = getLocalizedCategory("flow");
    },
});