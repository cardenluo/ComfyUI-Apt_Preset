import { app } from "../../../scripts/app.js";

function node_info_copy(src, dest, connect_both, copy_shape = true) {
    for (let i in src.inputs) {
        let input = src.inputs[i];
        if (input.widget !== undefined) {
            const destWidget = dest.widgets.find(x => x.name === input.widget.name);
            if (destWidget) {
                dest.convertWidgetToInput(destWidget);
            }
        }
        if (input.link) {
            let link = app.graph.links[input.link];
            if (link) {
                let src_node = app.graph.getNodeById(link.origin_id);
                if (src_node) {
                    src_node.connect(link.origin_slot, dest.id, input.name);
                }
            }
        }
    }

    if (connect_both) {
        let output_links = {};
        for (let i in src.outputs) {
            let output = src.outputs[i];
            if (output.links) {
                let links = [];
                for (let j in output.links) {
                    const link = app.graph.links[output.links[j]];
                    if (link) {
                        links.push(link);
                    }
                }
                output_links[output.name] = links;
            }
        }

        for (let i in dest.outputs) {
            let output = dest.outputs[i];
            let links = output_links[output.name];
            if (links) {
                for (let j in links) {
                    let link = links[j];
                    let target_node = app.graph.getNodeById(link.target_id);
                    if (target_node) {
                        dest.connect(parseInt(i), target_node, link.target_slot);
                    }
                }
            }
        }
    }

    if (copy_shape) {
        dest.color = src.color;
        dest.bgcolor = src.bgcolor;
        dest.size = [
            Math.max(src.size[0], dest.size[0]),
            Math.max(src.size[1], dest.size[1])
        ];
    }

    app.graph.afterChange();
}

app.registerExtension({
    name: "AptPreset.IOStoreImage.ClearStorage",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "IO_store_image") {
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = origOnNodeCreated?.apply(this, arguments);

                this.addWidget("button", "clear_storage", "🗑️ 清空存储", () => {
                    this.clearStorage();
                }, {
                    clickedColor: "#ff6b6b",
                    defaultColor: "#4a4a4a"
                });

                return r;
            };

            nodeType.prototype.clearStorage = async function () {
                const graph = app.graph;
                const nodeId = this.id;

                try {
                    const [x, y] = this.pos;
                    const new_node = LiteGraph.createNode(nodeType.comfyClass);
                    new_node.pos = [x, y];
                    graph.add(new_node, false);

                    node_info_copy(this, new_node, true);

                    graph.remove(this);

                    requestAnimationFrame(() => {
                        app.canvas.setDirty(true, true);
                    });

                    console.log(`IO_store_image node #${nodeId} storage cleared successfully`);
                } catch (e) {
                    console.error("Clear storage failed:", e);
                }
            };
        }
    },

    init() {
        console.log("IO_store_image Clear Storage Button loaded");
    }
});