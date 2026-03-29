import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "OutputShareNode",
    async setup() {
        const SHARE_COLOR = "#4CAF50";
        const shareLinks = new Map();
        function isShareNode(node) {
            return !!node?.properties?.is_output_share;
        }

        function getSourceNode(shareNode) {
            const sourceId = shareNode?.properties?.source_node_id;
            if (!sourceId) {
                return null;
            }
            return app.graph.getNodeById(sourceId);
        }

        function registerShareNode(shareNode) {
            const sourceId = shareNode?.properties?.source_node_id;
            if (!sourceId) {
                return;
            }
            if (!shareLinks.has(sourceId)) {
                shareLinks.set(sourceId, []);
            }
            const list = shareLinks.get(sourceId);
            if (!list.some((n) => n?.id === shareNode.id)) {
                list.push(shareNode);
            }
        }

        function getForwardedLinks(shareNode) {
            if (!shareNode.properties) {
                shareNode.properties = {};
            }
            if (!shareNode.properties.forwarded_links || typeof shareNode.properties.forwarded_links !== "object") {
                shareNode.properties.forwarded_links = {};
            }
            return shareNode.properties.forwarded_links;
        }

        function unregisterShareNode(shareNode) {
            const sourceId = shareNode?.properties?.source_node_id;
            if (!sourceId || !shareLinks.has(sourceId)) {
                return;
            }
            const left = shareLinks.get(sourceId).filter((n) => n?.id !== shareNode.id);
            if (left.length === 0) {
                shareLinks.delete(sourceId);
                return;
            }
            shareLinks.set(sourceId, left);
        }

        function calculateNodeSize(node) {
            const titleWidth = node.title ? node.title.length * 8 : 80;
            const contentWidth = Math.max(titleWidth, 120);
            const width = Math.ceil(contentWidth * 1.15);
            const outputCount = node.outputs?.length || 0;
            const height = Math.ceil((outputCount * 20 + 20) * 1.2);
            return [Math.max(120, width), Math.max(40, height)];
        }

        function getShareableOutputs(sourceNode) {
            return sourceNode?.outputs || [];
        }

        function clearAllWidgets(node) {
            if (!node) {
                return;
            }
            if (Array.isArray(node.widgets) && typeof node.removeWidget === "function") {
                while (node.widgets.length > 0) {
                    const widget = node.widgets[node.widgets.length - 1];
                    try {
                        node.removeWidget(widget);
                    } catch (_) {
                        widget?.onRemove?.();
                        node.widgets.pop();
                    }
                }
            } else {
                node.widgets = [];
            }
            if (node.customWidgets && typeof node.customWidgets === "object") {
                for (const widget of Object.values(node.customWidgets)) {
                    try {
                        widget?.onRemove?.();
                    } catch (_) {}
                }
                node.customWidgets = {};
            }
            node.widgets_values = [];
        }

        function normalizeShareNodeShape(shareNode, sourceNode = null) {
            const alwaysMode = LiteGraph.ALWAYS ?? 0;
            shareNode.mode = alwaysMode;
            shareNode.isVirtualNode = false;
            shareNode.inputs = [];
            shareNode.serialize_widgets = false;
            shareNode.flags = {
                ...(shareNode.flags || {}),
                collapsed: false,
                pinned: false,
                locked: false
            };
            delete shareNode.flags.skip_repeated_outputs;
            shareNode.boxcolor = SHARE_COLOR;
            shareNode.color = sourceNode?.color ?? shareNode.color;
            shareNode.bgcolor = sourceNode?.bgcolor ?? shareNode.bgcolor;
            shareNode.shape = LiteGraph.BOX_SHAPE;
        }

        function updateShareNode(shareNode, sourceNode) {
            if (!isShareNode(shareNode) || !sourceNode) {
                return;
            }
            enforceShareNodeLayout(shareNode, sourceNode);
            shareNode.setDirtyCanvas(true, true);
        }

        function enforceShareNodeLayout(shareNode, sourceNode = null) {
            const linkedSourceNode = sourceNode || getSourceNode(shareNode);
            const oldOutputs = shareNode.outputs || [];
            const oldLinkMap = new Map(
                oldOutputs.map((output) => [`${output?.name ?? ""}|${output?.type ?? ""}`, Array.isArray(output?.links) ? [...output.links] : []])
            );
            const outputBase = linkedSourceNode
                ? getShareableOutputs(linkedSourceNode)
                : oldOutputs;
            shareNode.outputs = outputBase.map((output) => {
                const key = `${output?.name ?? ""}|${output?.type ?? ""}`;
                const existingLinks = oldLinkMap.get(key) || [];
                return {
                    ...output,
                    disabled: false,
                    links: existingLinks
                };
            });
            shareNode.inputs = [];
            clearAllWidgets(shareNode);
            normalizeShareNodeShape(shareNode, linkedSourceNode);
            shareNode.size = calculateNodeSize(shareNode);
        }

        function transferLinkOwnership(shareNode, sourceNode) {
            return;
        }

        function realignForwardedLinks(shareNode, sourceNode) {
            return;
        }

        function resolveSourceSlot(shareNode, sourceNode, shareSlot) {
            const shareOutput = shareNode?.outputs?.[shareSlot];
            if (!shareOutput || !Array.isArray(sourceNode?.outputs)) {
                return shareSlot;
            }
            const shareName = shareOutput.name;
            const shareType = shareOutput.type;
            const mapped = sourceNode.outputs.findIndex((output) => {
                if (!output) {
                    return false;
                }
                const sameName = shareName ? output.name === shareName : true;
                const sameType = shareType ? output.type === shareType : true;
                return sameName && sameType;
            });
            if (mapped >= 0) {
                return mapped;
            }
            return sourceNode.outputs.length > 0 ? 0 : -1;
        }

        function rewritePromptOutput(output) {
            if (!output || typeof output !== "object") {
                return;
            }
            const shareMap = new Map();
            for (const node of app.graph._nodes || []) {
                if (!isShareNode(node)) {
                    continue;
                }
                const sourceNode = getSourceNode(node);
                if (!sourceNode) {
                    continue;
                }
                shareMap.set(String(node.id), {
                    shareNode: node,
                    sourceId: String(sourceNode.id),
                    sourceNode
                });
            }
            if (shareMap.size === 0) {
                return;
            }
            for (const [, promptNode] of Object.entries(output)) {
                if (!promptNode?.inputs) {
                    continue;
                }
                for (const [inputName, inputValue] of Object.entries(promptNode.inputs)) {
                    if (!Array.isArray(inputValue) || inputValue.length !== 2) {
                        continue;
                    }
                    const linkedId = String(inputValue[0]);
                    const shareInfo = shareMap.get(linkedId);
                    if (!shareInfo) {
                        continue;
                    }
                    const shareSlot = Number(inputValue[1]) || 0;
                    const mappedSlot = resolveSourceSlot(shareInfo.shareNode, shareInfo.sourceNode, shareSlot);
                    if (mappedSlot < 0) {
                        delete promptNode.inputs[inputName];
                        continue;
                    }
                    if (!output[shareInfo.sourceId]) {
                        delete promptNode.inputs[inputName];
                        continue;
                    }
                    promptNode.inputs[inputName] = [shareInfo.sourceId, mappedSlot];
                }
            }
            for (const shareId of shareMap.keys()) {
                if (output[shareId]) {
                    delete output[shareId];
                }
            }
        }

        function rebuildShareLinks() {
            shareLinks.clear();
            for (const node of app.graph._nodes) {
                if (!isShareNode(node) || !node.properties?.source_node_id) {
                    continue;
                }
                const sourceNode = getSourceNode(node);
                registerShareNode(node);
                if (sourceNode) {
                    updateShareNode(node, sourceNode);
                    transferLinkOwnership(node, sourceNode);
                    realignForwardedLinks(node, sourceNode);
                }
            }
        }

        const originalGetOutputData = LGraphNode.prototype.getOutputData;
        LGraphNode.prototype.getOutputData = function(slot) {
            if (isShareNode(this)) {
                const sourceNode = getSourceNode(this);
                if (sourceNode) {
                    return sourceNode.getOutputData(slot);
                }
                return null;
            }
            return originalGetOutputData?.call(this, slot);
        };

        const originalNodeConfigure = LGraphNode.prototype.configure;
        LGraphNode.prototype.configure = function(info) {
            const configured = originalNodeConfigure?.call(this, info);
            if (isShareNode(this)) {
                enforceShareNodeLayout(this, getSourceNode(this));
            }
            return configured;
        };

        const originalOnConnectionsChange = LGraphNode.prototype.onConnectionsChange;
        LGraphNode.prototype.onConnectionsChange = function(type, slotIndex, isConnected, link, ioSlot) {
            if (originalOnConnectionsChange) {
                originalOnConnectionsChange.call(this, type, slotIndex, isConnected, link, ioSlot);
            }
            if (isShareNode(this)) {
                clearAllWidgets(this);
                this.inputs = [];
                this.setDirtyCanvas(true, true);
                return;
            }
            const sharedByThisNode = shareLinks.get(this.id) || [];
            for (const shareNode of sharedByThisNode) {
                if (!isShareNode(shareNode)) {
                    continue;
                }
                updateShareNode(shareNode, this);
                transferLinkOwnership(shareNode, this);
                realignForwardedLinks(shareNode, this);
            }
        };

        const originalAddNode = LGraph.prototype.add;
        LGraph.prototype.add = function(node) {
            const added = originalAddNode.call(this, node);
            if (isShareNode(node)) {
                const sourceNode = getSourceNode(node);
                normalizeShareNodeShape(node, sourceNode);
                registerShareNode(node);
                if (sourceNode) {
                    updateShareNode(node, sourceNode);
                    transferLinkOwnership(node, sourceNode);
                    realignForwardedLinks(node, sourceNode);
                }
            }
            return added;
        };

        const originalRemove = LGraph.prototype.remove;
        LGraph.prototype.remove = function(node) {
            if (isShareNode(node)) {
                unregisterShareNode(node);
            }
            return originalRemove.call(this, node);
        };

        const originalConfigure = LGraph.prototype.configure;
        LGraph.prototype.configure = function(data, keep_old) {
            const configured = originalConfigure.call(this, data, keep_old);
            queueMicrotask(() => rebuildShareLinks());
            return configured;
        };

        const originalGraphToPrompt = app.graphToPrompt?.bind(app);
        if (originalGraphToPrompt) {
            app.graphToPrompt = async function(...args) {
                const promptData = await originalGraphToPrompt(...args);
                rewritePromptOutput(promptData?.output);
                return promptData;
            };
        }

        const originalGetNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function(node) {
            const options = originalGetNodeMenuOptions.call(this, node);
            const shareIndex = Math.max(0, options.length - 2);
            options.splice(shareIndex, 0, null, {
                content: "Output Share Node",
                callback: () => {
                    const sourceNodeId = node.id;
                    const sourceNode = app.graph.getNodeById(sourceNodeId);
                    if (!sourceNode) {
                        return;
                    }
                    const shareNode = LiteGraph.createNode(sourceNode.type);
                    if (!shareNode) {
                        return;
                    }
                    app.graph.add(shareNode);
                    shareNode.title = `${sourceNode.title}_share`;
                    shareNode.pos = [...app.canvas.graph_mouse];
                    shareNode.color = sourceNode.color;
                    shareNode.bgcolor = sourceNode.bgcolor;
                    shareNode.properties = {
                        ...(shareNode.properties || {}),
                        is_output_share: true,
                        source_node_id: sourceNodeId,
                        original_type: sourceNode.type
                    };
                    registerShareNode(shareNode);
                    updateShareNode(shareNode, sourceNode);
                    transferLinkOwnership(shareNode, sourceNode);
                    realignForwardedLinks(shareNode, sourceNode);
                }
            });
            return options;
        };

        function getCollapsedNodeSize(node) {
            if (!node.flags?.collapsed) return node.size;
            let titleWidth = node.title ? node.title.length * 8 : 80;
            let height = 30;
            return [titleWidth, height];
        }
        function getNodeCenter(node) {
            let size = node.flags?.collapsed ? getCollapsedNodeSize(node) : node.size;
            let centerX = node.pos[0] + size[0] / 2;
            let centerY = node.pos[1] + (node.flags?.collapsed ? -15 : size[1] / 2);
            return [centerX, centerY];
        }
        const originalDraw = LGraphCanvas.prototype.draw;
        LGraphCanvas.prototype.draw = function() {
            originalDraw.call(this);
            const ctx = this.ctx;
            if (!ctx) return;

            ctx.save();
            ctx.imageSmoothingEnabled = true;
            ctx.lineJoin = "round";
            ctx.lineCap = "round";
            const scale = this.ds.scale;
            const offset = this.ds.offset;
            const lineWidth = Math.max(2 * scale, 0.2);
            const dashSize = Math.max(5 * scale, 0.2);
            ctx.setLineDash(dashSize > 0 ? [dashSize, dashSize] : []);
            ctx.strokeStyle = SHARE_COLOR;
            ctx.lineWidth = lineWidth;
            shareLinks.forEach((shareNodes, sourceId) => {
                const sourceNode = this.graph.getNodeById(sourceId);
                shareNodes.forEach(shareNode => {
                    if (sourceNode && shareNode) {
                        const shouldDrawFromSource = sourceNode.selected;
                        const shouldDrawFromShare = shareNode.selected;
                        if (shouldDrawFromSource || shouldDrawFromShare) {
                            ctx.beginPath();
                            const [sourceX, sourceY] = getNodeCenter(sourceNode);
                            const [shareX, shareY] = getNodeCenter(shareNode);
                            ctx.moveTo((sourceX + offset[0]) * scale, (sourceY + offset[1]) * scale);
                            ctx.lineTo((shareX + offset[0]) * scale, (shareY + offset[1]) * scale);
                            ctx.stroke();
                        }
                    }
                });
            });
            ctx.restore();
        };
        rebuildShareLinks();
    }
});
