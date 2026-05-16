/** ComfyUI/custom_nodes/CCNotes/js/dynamic_ports_register.js **/

import { app } from "../../../scripts/app.js";
import { DynamicPorts } from "./dynamic_ports.js";




export function setupOutputPortSync(nodeType, app) {
    const origOnNodeCreated = nodeType.prototype.onNodeCreated;
    const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    const origUpdateDynamicPorts = nodeType.prototype.updateDynamicPorts;

    nodeType.prototype.onNodeCreated = function () {
        if (origOnNodeCreated) {
            origOnNodeCreated.apply(this, arguments);
        }
        this.syncOutputPorts();
    };

    nodeType.prototype.onConnectionsChange = function (type, index, connected, link) {
        if (origOnConnectionsChange) {
            origOnConnectionsChange.apply(this, arguments);
        }
        if (type === 1) {
            this.syncOutputPorts();
        }
    };

    nodeType.prototype.updateDynamicPorts = function () {
        if (origUpdateDynamicPorts) {
            origUpdateDynamicPorts.apply(this, arguments);
        }
        this.syncOutputPorts();
    };

    nodeType.prototype.syncOutputPorts = function () {
        if (!this.inputs || !this.outputs) return;
        let connectedInputCount = 0;
        let maxInputIndex = 0;
        
        this.inputs.forEach((input) => {
            if (input.name.startsWith("input_")) {
                const parts = input.name.split("_");
                const index = parseInt(parts[parts.length - 1]);
                if (!isNaN(index)) {
                    maxInputIndex = Math.max(maxInputIndex, index);
                    connectedInputCount++;
                }
            }
        });

        const neededOutputs = Math.max(1, maxInputIndex);
        
        while (this.outputs.length > neededOutputs) {
            this.removeOutput(this.outputs.length - 1);
        }
        
        while (this.outputs.length < neededOutputs) {
            const index = this.outputs.length + 1;
            this.addOutput(`output_${index}`, "*");
        }
        
        if (this.computeSize) {
            this.computeSize();
        }
        if (this.setDirtyCanvas) {
            this.setDirtyCanvas(true, true);
        }
        if (this.graph) {
            this.graph._version++;
        }
    };
}



app.registerExtension({
    name: "view_mulView",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "view_mulView") return;

        DynamicPorts.setupDynamicInputs(nodeType, {
            baseInputName: "input",
            inputType: "*",
            startIndex: 1
        });

        // Setup port synchronization logic
        setupOutputPortSync(nodeType, app);
    },
});











