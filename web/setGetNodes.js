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

function getSlotDisplayName(slot, fallback = "*") {
    const candidates = [
        slot?.label,
        slot?.localized_name,
        slot?.localizedName,
        slot?.name
    ];
    for (const candidate of candidates) {
        if (typeof candidate === "string" && candidate.trim().length > 0) {
            return candidate;
        }
    }
    return fallback;
}

function getSlotMeta(slot, fallbackType = "*", fallbackName = null) {
    const type =
        typeof slot?.type === "string" && slot.type.length > 0
            ? slot.type
            : fallbackType;
    const name = getSlotDisplayName(slot, fallbackName ?? type);
    return { type, name };
}

function isSetNode(node) {
    return node?.type === "flow_Set_Value";
}

function isGetNode(node) {
    return node?.type === "flow_Get_Value";
}

function isFlowNode(node) {
    return isSetNode(node) || isGetNode(node);
}

function getFlowVariableName(node) {
    const value = node?.widgets?.[0]?.value;
    return typeof value === "string" && value.length > 0 ? value : "";
}

function getSelectedNodes(canvas) {
    const selected = canvas?.selected_nodes;
    if (Array.isArray(selected)) {
        return selected.filter(Boolean);
    }
    if (selected && typeof selected === "object") {
        return Object.values(selected).filter(Boolean);
    }
    return [];
}

function getLinkedFlowNodes(graph, sourceNode) {
    if (!graph || !Array.isArray(graph._nodes) || !isFlowNode(sourceNode)) {
        return [];
    }
    const variableName = getFlowVariableName(sourceNode);
    if (!variableName) {
        return [sourceNode];
    }
    return graph._nodes.filter((node) => isFlowNode(node) && getFlowVariableName(node) === variableName);
}

const FLOW_LINK_COLOR = "#66bb6a";

function getActiveFlowGroups(canvas) {
    const graph = canvas?.graph || app?.graph;
    if (!graph) {
        return [];
    }

    const groupMap = new Map();
    const selectedFlowNodes = getSelectedNodes(canvas).filter((node) => isFlowNode(node));
    selectedFlowNodes.forEach((selectedNode) => {
        const variableName = getFlowVariableName(selectedNode);
        const key = variableName || `node:${selectedNode.id}`;
        if (groupMap.has(key)) {
            return;
        }
        const nodes = getLinkedFlowNodes(graph, selectedNode);
        groupMap.set(key, {
            variableName,
            nodes,
            setter: nodes.find((node) => isSetNode(node)) || null,
            getters: nodes.filter((node) => isGetNode(node))
        });
    });

    return Array.from(groupMap.values());
}

function getCollapsedNodeSize(node) {
    if (!node?.flags?.collapsed) {
        return node?.size || [0, 0];
    }
    const titleWidth = node.title ? node.title.length * 8 : 80;
    return [titleWidth, 30];
}

function getNodeCenter(node) {
    const size = node?.flags?.collapsed ? getCollapsedNodeSize(node) : (node?.size || [0, 0]);
    const pos = node?.pos || [0, 0];
    const centerX = pos[0] + size[0] / 2;
    const centerY = pos[1] + (node?.flags?.collapsed ? -15 : size[1] / 2);
    return [centerX, centerY];
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
                
                for (let i = 0; i < 3; i++) {
                    this.addInput("*", "*");
                    this.addOutput("*", "*");
                }
                
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
                        if (this.outputs && this.outputs[slot]) {
                            this.outputs[slot].type = "*";
                            this.outputs[slot].name = "*";
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
                            const upstreamSlot = fromNode.outputs[link_info.origin_slot];
                            const { type, name } = getSlotMeta(upstreamSlot);
                            const defaultVariableName = name || type;
                            
                            if (slot === 0 && this.title === "Set_") {
                                this.title = "Set_" + defaultVariableName;
                            }
                            if (slot === 0 && this.widgets[0].value === '') {
                                this.widgets[0].value = defaultVariableName;
                            }
                            
                            this.validateName(node.graph);
                            if (this.inputs && this.inputs[slot]) {
                                this.inputs[slot].type = type;
                                this.inputs[slot].name = name;
                            }
                            if (this.outputs && this.outputs[slot]) {
                                this.outputs[slot].type = type;
                                this.outputs[slot].name = name;
                            }
                        }
                    }
                    if (link_info && node.graph && slotType === 2 && isChangeConnect) {
                        let type = this.inputs && this.inputs[slot] ? this.inputs[slot].type : "*";
                        let name = this.inputs && this.inputs[slot] ? this.inputs[slot].name : type;
                        if (type === '*' && link_info && node.graph) {
                            const targetNode = node.graph._nodes.find((otherNode) => String(otherNode.id) === String(link_info.target_id));
                            const targetInput = targetNode?.inputs?.[link_info.target_slot];
                            const inferred = targetInput?.type;
                            if (typeof inferred === "string" && inferred.length > 0 && inferred !== "*") {
                                type = inferred;
                                if (this.inputs && this.inputs[slot]) {
                                    this.inputs[slot].type = type;
                                    this.inputs[slot].name = getSlotDisplayName(this.inputs[slot], type);
                                }
                                name = this.inputs && this.inputs[slot] ? this.inputs[slot].name : type;
                            }
                        }
                        if (this.outputs && this.outputs[slot]) {
                            this.outputs[slot].type = type;
                            this.outputs[slot].name = name;
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
                    if (Array.isArray(cloned.inputs)) {
                        cloned.inputs.forEach((input) => {
                            input.name = "*";
                            input.type = "*";
                        });
                    }
                    if (Array.isArray(cloned.outputs)) {
                        cloned.outputs.forEach((output) => {
                            output.name = "*";
                            output.type = "*";
                        });
                    }
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
                    const slots = Array.from({ length: 3 }, (_, idx) => getSlotMeta(this.inputs?.[idx]));
                    getters.forEach(getter => {
                        if (typeof getter.setSlots === "function") {
                            getter.setSlots(slots);
                        } else if (typeof getter.setTypes === "function") {
                            getter.setTypes(slots.map((slotMeta) => slotMeta.type));
                        } else if (typeof getter.setType === "function") {
                            getter.setType(slots[0]?.type ?? "*");
                        }
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
                
                for (let i = 0; i < 3; i++) {
                    this.addOutput("*", "*");
                }
                
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
                        const slots = Array.from({ length: 3 }, (_, idx) => getSlotMeta(setter.inputs?.[idx]));
                        this.setSlots(slots);
                        const variableName = setter.widgets?.[0]?.value ?? "";
                        this.title = variableName ? "Get_" + variableName : "Get_";
                        
                    } else {
                        this.setType("*");
                        this.title = "Get_";
                    }
                }
                
                this.clone = function () {
                    const cloned = GetNode.prototype.clone.apply(this);
                    if (cloned.widgets && cloned.widgets[0]) {
                        cloned.widgets[0].value = "";
                    }
                    cloned.title = "Get_";
                    if (Array.isArray(cloned.inputs)) {
                        cloned.inputs.forEach((input) => {
                            input.type = "*";
                            input.name = "*";
                        });
                    }
                    if (Array.isArray(cloned.outputs)) {
                        cloned.outputs.forEach((output) => {
                            output.type = "*";
                            output.name = "*";
                        });
                    }
                    cloned.size = cloned.computeSize();
                    return cloned;
                };
                
                this.validateLinks = function() {
                    if (!Array.isArray(this.outputs) || !node.graph || !node.graph.links) {
                        return;
                    }
                    this.outputs.forEach((outputSlot) => {
                        if (!outputSlot || outputSlot.type === "*" || !outputSlot.links) {
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
                        });
                };
                
                this.setType = function(type) {
                    this.setTypes([type, type, type]);
                }

                this.setTypes = function(types) {
                    const slots = Array.from({ length: 3 }, (_, idx) => {
                        const type = types[idx] ?? "*";
                        return { type, name: type };
                    });
                    this.setSlots(slots);
                }

                this.setSlots = function(slots) {
                    if (!Array.isArray(slots)) {
                        return;
                    }
                    for (let i = 0; i < 3; i++) {
                        const slotMeta = slots[i] ?? {};
                        const type = slotMeta.type ?? "*";
                        const name = getSlotDisplayName(slotMeta, type);
                        if (this.outputs && this.outputs[i]) {
                            this.outputs[i].name = name;
                            this.outputs[i].type = type;
                        }
                    }
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

app.registerExtension({
    name: "Apt.SetGetNodeHighlight",
    async setup() {
        if (LGraphCanvas.prototype.__aptSetGetHighlightWrapped) {
            return;
        }
        LGraphCanvas.prototype.__aptSetGetHighlightWrapped = true;

        const originalDraw = LGraphCanvas.prototype.draw;
        LGraphCanvas.prototype.draw = function(...args) {
            const activeGroups = getActiveFlowGroups(this);
            const result = originalDraw.call(this, ...args);
            const ctx = this.ctx;
            if (ctx && activeGroups.length > 0) {
                ctx.save();
                ctx.imageSmoothingEnabled = true;
                ctx.lineJoin = "round";
                ctx.lineCap = "round";
                const scale = this.ds?.scale ?? 1;
                const offset = this.ds?.offset ?? [0, 0];
                const lineWidth = Math.max(2 * scale, 0.2);
                const dashSize = Math.max(5 * scale, 0.2);
                ctx.setLineDash(dashSize > 0 ? [dashSize, dashSize] : []);
                ctx.strokeStyle = FLOW_LINK_COLOR;
                ctx.lineWidth = lineWidth;

                activeGroups.forEach(({ setter, getters }) => {
                    if (!setter || !Array.isArray(getters) || getters.length === 0) {
                        return;
                    }
                    const [sourceX, sourceY] = getNodeCenter(setter);
                    getters.forEach((getter) => {
                        if (!getter) {
                            return;
                        }
                        const [targetX, targetY] = getNodeCenter(getter);
                        ctx.beginPath();
                        ctx.moveTo((sourceX + offset[0]) * scale, (sourceY + offset[1]) * scale);
                        ctx.lineTo((targetX + offset[0]) * scale, (targetY + offset[1]) * scale);
                        ctx.stroke();
                    });
                });
                ctx.restore();
            }
            return result;
        };
    }
});
