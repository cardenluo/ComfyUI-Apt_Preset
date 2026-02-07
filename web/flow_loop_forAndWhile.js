import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "Comfy.LoopDynamic",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        const validNodes = [
            "flow_forStart",
            "flow_forEnd",
            "flow_whileStart",
            "flow_whileEnd"
        ];
        
        if (!validNodes.includes(nodeData.name)) return;

        const inPrefix = "initial_value_";
        const outPrefix = "value_";

        const isWhileNode = nodeData.name.includes("While");
        const startIndex = isWhileNode ? 0 : 1;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) onNodeCreated.apply(this, arguments);
            this._processingInterfaces = true;
            try {
                this.cleanupLoopInterfaces(inPrefix, outPrefix, startIndex);
                this.setSize(this.computeSize());
            } finally {
                this._processingInterfaces = false;
            }
        };

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            if (onConnectionsChange) onConnectionsChange.apply(this, arguments);

            if (this._processingInterfaces) return;
            this._processingInterfaces = true;

            setTimeout(() => {
                try {
                    if (type === 1) { 
                        const input = this.inputs[index];
                        if (input && input.name.startsWith(inPrefix)) {
                            this.cleanupLoopInterfaces(inPrefix, outPrefix, startIndex);
                        }
                    }
                    this.setSize(this.computeSize());
                } finally {
                    this._processingInterfaces = false;
                    app.canvas.setDirty(true);
                }
            }, 0);
        };

        nodeType.prototype.cleanupLoopInterfaces = function (inPre, outPre, startIdx) {
            let maxLinkedNum = -1;
            const linkedNums = new Set();

            for (const input of this.inputs) {
                if (input.name.startsWith(inPre)) {
                    const num = parseInt(input.name.split("_").pop());
                    if (!isNaN(num) && input.link !== null) {
                        linkedNums.add(num);
                        if (num > maxLinkedNum) maxLinkedNum = num;
                    }
                }
            }

            const keepInUpTo = Math.max(startIdx, maxLinkedNum + 1);

            for (let i = this.inputs.length - 1; i >= 0; i--) {
                const input = this.inputs[i];
                if (input.name.startsWith(inPre)) {
                    const num = parseInt(input.name.split("_").pop());
                    if (num > keepInUpTo) {
                        this.removeInput(i);
                    }
                }
            }

            for (let n = startIdx; n <= keepInUpTo; n++) {
                const name = `${inPre}${n}`;
                if (!this.inputs.find(i => i.name === name)) {
                    this.addInput(name, "*");
                }
            }

            for (let i = this.outputs.length - 1; i >= 0; i--) {
                const output = this.outputs[i];
                if (output.name.startsWith(outPre)) {
                    const num = parseInt(output.name.split("_").pop());
                    if (!linkedNums.has(num)) {
                        if (!output.links || output.links.length === 0) {
                            this.removeOutput(i);
                        }
                    }
                }
            }

            for (const num of linkedNums) {
                const name = `${outPre}${num}`;
                if (!this.outputs.find(o => o.name === name)) {
                    this.addOutput(name, "*");
                }
            }
        };
    },
});