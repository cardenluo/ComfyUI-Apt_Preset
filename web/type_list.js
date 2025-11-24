
import { app } from "../../scripts/app.js"
// import { api } from "../../scripts/api.js"
import { ComfyWidgets } from "../../scripts/widgets.js"
// import { addConnectionLayoutSupport } from "./utils.js"

const _prefix = 'value'

const TypeSlot = {
    Input: 1,
    Output: 2,
}

const TypeSlotEvent = {
    Connect: true,
    Disconnect: false,
}

const dynamic_connection = (node, index, event, prefix = 'in_', type = '*', names = []
) => {
    if (!node.inputs[index].name.startsWith(prefix)) {
        return
    }
    // remove all non connected inputs
    if (event == TypeSlotEvent.Disconnect && node.inputs.length > 1) {
        if (node.widgets) {
            const widget = node.widgets.find((w) => w.name === node.inputs[index].name)
            if (widget) {
                widget.onRemoved?.()
                node.widgets.length = node.widgets.length - 1
            }
        }
        node.removeInput(index)

        // TODO type
        // make inputs sequential again
        for (let i = 0; i < node.inputs.length; i++) {
            const name = i < names.length ? names[i] : `${prefix}${i + 1}`
            node.inputs[i].label = name
            node.inputs[i].name = name
        }
    }

    // add an extra input
    if (node.inputs[node.inputs.length - 1].link != undefined) {
        const nextIndex = node.inputs.length
        const name = nextIndex < names.length
            ? names[nextIndex]
            : `${prefix}${nextIndex + 1}`
        node.addInput(name, type)
    }
}

/**
 * Get all unique types in the workflow.
 * @returns {Set} Unique set of all types used in the workflow
 */
function getWorkflowTypes(app) {
    const pythonSupportedTypes = [
        "*", "STRING", "INT", "FLOAT", "LIST", "SET", "TUPLE", "DICTIONARY", "BOOLEAN"];
    const types = new Set(pythonSupportedTypes);
    app.graph._nodes.forEach(node => {
        node.inputs.forEach(slot => {
            types.add(slot.type);
        });
        node.outputs.forEach(slot => {
            types.add(slot.type);
        });
    });
    return Array.from(types);
}






app.registerExtension({
    name: "godmt.ListUtils",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "pack_Pack" || nodeData.name === "create_any_batch" || nodeData.name === "create_image_batch"|| nodeData.name === "create_mask_batch") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                this.addInput(`${_prefix}_1`, '*')
                return r
            }

            // on copy and paste
            const onConfigure = nodeType.prototype.onConfigure
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined
                if (!app.configuringGraph && this.inputs) {
                    const length = this.inputs.length
                    for (let i = length - 1; i >= 0; i--) {
                        this.removeInput(i)
                    }
                    this.addInput(`${_prefix}_1`, '*')
                }
                return r
            }


            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                if (slotType === TypeSlot.Input) {
                    dynamic_connection(this, slot, event, `${_prefix}_`, '*')
                    if (event === TypeSlotEvent.Connect && link_info) {
                        const fromNode = this.graph._nodes.find(
                            (otherNode) => otherNode.id == link_info.origin_id
                        )
                        const type = fromNode.outputs[link_info.origin_slot].type
                        this.inputs[slot].type = type
                    } else if (event === TypeSlotEvent.Disconnect) {
                        this.inputs[slot].type = '*'
                        this.inputs[slot].label = `${_prefix}_${slot + 1}`
                    }
                }
                return r
            }


            
        } else if (nodeData.name === "math_Exec") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                this.addInput(`x[0]`, '*')
                return r
            }

            // on copy and paste
            const onConfigure = nodeType.prototype.onConfigure
            nodeType.prototype.onConfigure = function () {
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined
                if (!app.configuringGraph && this.inputs) {
                    const length = this.inputs.length
                    for (let i = length - 1; i >= 0; i--) {
                        this.removeInput(i)
                    }
                    this.addInput(`x[0]`, '*')
                }
                return r
            }

            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                if (slotType === TypeSlot.Input) {
                    // remove all non connected inputs
                    if (event == TypeSlotEvent.Disconnect && this.inputs.length > 1) {
                        if (this.widgets) {
                            const widget = this.widgets.find((w) => w.name === this.inputs[slot].name)
                            if (widget) {
                                widget.onRemoved?.()
                                this.widgets.length = this.widgets.length - 1
                            }
                        }
                        this.removeInput(slot)
                        // make inputs sequential again
                        for (let i = 0; i < this.inputs.length; i++) {
                            const name = `x[${i}]`
                            this.inputs[i].label = name
                            this.inputs[i].name = name
                        }
                    }

                    // TODO type
                    const type = "*"
                    // add an extra input
                    if (this.inputs[this.inputs.length - 1].link != undefined) {
                        const nextIndex = this.inputs.length
                        const name = `x[${nextIndex}]`
                        this.addInput(name, type)
                    }

                    if (event === TypeSlotEvent.Connect && link_info) {
                        const fromNode = this.graph._nodes.find(
                            (otherNode) => otherNode.id == link_info.origin_id
                        )
                        const type = fromNode.outputs[link_info.origin_slot].type
                        this.inputs[slot].type = type
                    } else if (event === TypeSlotEvent.Disconnect) {
                        this.inputs[slot].type = '*'
                        this.inputs[slot].label = `x[${slot}]`
                    }
                    if (this.widgets && this.widgets[0]) {
                        this.widgets[0].y = this.inputs.length * 20
                        // TODO update node height
                    }
                }
                return me
            }
        }
        // 如果节点名称为"create_any_List"或"list_MergeList"
        else if (nodeData.name === "create_any_List" || nodeData.name === "list_MergeList") {
            // 获取节点类型原型上的onNodeCreated方法
            const onNodeCreated = nodeType.prototype.onNodeCreated
            // 重写节点类型原型上的onNodeCreated方法
            nodeType.prototype.onNodeCreated = function () {
                // 调用原始的onNodeCreated方法
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                // 添加一个输入
                this.addInput(`${_prefix}_1`, '*')
                return r
            }

            // TODO 可能会有问题
            // 在复制、粘贴、加载时
            // 获取节点类型原型上的onConfigure方法
            const onConfigure = nodeType.prototype.onConfigure
            // 重写节点类型原型上的onConfigure方法
            nodeType.prototype.onConfigure = function () {
                // 调用原始的onConfigure方法
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined
                // 如果不是正在配置图形，并且有输入
                if (!app.configuringGraph && this.inputs) {
                    // 获取输入的长度
                    const length = this.inputs.length
                    // 从后往前遍历输入
                    for (let i = length - 1; i >= 0; i--) {
                        // 移除输入
                        this.removeInput(i)
                    }
                    // 添加一个输入
                    this.addInput(`${_prefix}_1`, '*')
                }
                return r
            }

            // 获取节点类型原型上的onConnectionsChange方法
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            // 重写节点类型原型上的onConnectionsChange方法
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                // 调用原始的onConnectionsChange方法
                const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                // 如果是输入类型的插槽
                if (slotType === TypeSlot.Input) {
                    // 如果输入名称不以_prefix开头，则返回
                    if (!this.inputs[slot].name.startsWith(_prefix)) {
                        return
                    }

                    // 如果是连接事件，并且有连接信息
                    if (event == TypeSlotEvent.Connect && link_info) {
                        // 如果是第一个插槽
                        if (slot == 0) {
                            // 获取连接的节点
                            const node = app.graph.getNodeById(link_info.origin_id)
                            // 获取连接的输出类型
                            const origin_type = node.outputs[link_info.origin_slot].type
                            // 设置输入和输出的类型
                            this.inputs[0].type = origin_type
                            this.outputs[0].type = origin_type
                            // 设置输出的标签和名称
                            this.outputs[0].label = origin_type
                            this.outputs[0].name = origin_type
                        }
                    }

                    // 移除所有未连接的输入
                    if (event == TypeSlotEvent.Disconnect && this.inputs.length > 0) {
                        // 如果有控件
                        if (this.widgets) {
                            // 获取控件
                            const widget = this.widgets.find((w) => w.name === this.inputs[slot].name)
                            // 如果有控件，则调用控件的onRemoved方法，并从控件数组中移除
                            if (widget) {
                                widget.onRemoved?.()
                                this.widgets.length = this.widgets.length - 1
                            }
                        }
                        // 移除输入
                        this.removeInput(slot)
                        // 使输入顺序重新排列
                        for (let i = 0; i < this.inputs.length; i++) {
                            // 设置输入的标签和名称
                            this.inputs[i].label = `${_prefix}_${i + 1}`
                            this.inputs[i].name = `${_prefix}_${i + 1}`
                        }
                    }

                    // 添加一个额外的输入
                    if (this.inputs[this.inputs.length - 1].link != undefined) {
                        // 获取下一个索引
                        const nextIndex = this.inputs.length
                        // 添加一个输入
                        this.addInput(`${_prefix}_${nextIndex + 1}`, this.inputs[0].type)
                    }
                }
                return me
            }
        } else if (nodeData.name == "batch_MergeBatch") {
            // 获取节点类型原型上的onNodeCreated方法
            const onNodeCreated = nodeType.prototype.onNodeCreated
            // 重写节点类型原型上的onNodeCreated方法
            nodeType.prototype.onNodeCreated = function () {
                // 调用原始的onNodeCreated方法
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                // 添加一个输入
                this.addInput(`${_prefix}_1`, 'LIST')
                return r
            }

            // 在复制和粘贴时
            // 获取节点类型原型上的onConfigure方法
            const onConfigure = nodeType.prototype.onConfigure
            // 重写节点类型原型上的onConfigure方法
            nodeType.prototype.onConfigure = function () {
                // 调用原始的onConfigure方法
                const r = onConfigure ? onConfigure.apply(this, arguments) : undefined
                // 如果不是正在配置图形，并且有输入
                if (!app.configuringGraph && this.inputs) {
                    // 获取输入的长度
                    const length = this.inputs.length
                    // 从后往前遍历输入
                    for (let i = length - 1; i >= 0; i--) {
                        // 移除输入
                        this.removeInput(i)
                    }
                    // 添加一个输入
                    this.addInput(`${_prefix}_1`, 'LIST')
                }
                return r
            }
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                if (slotType === TypeSlot.Input) {
                    dynamic_connection(this, slot, event, `${_prefix}_`, 'LIST')
                    if (event === TypeSlotEvent.Connect && link_info) {
                        const fromNode = this.graph._nodes.find(
                            (otherNode) => otherNode.id == link_info.origin_id
                        )
                        const type = fromNode.outputs[link_info.origin_slot].type
                        this.inputs[slot].type = type
                    } else if (event === TypeSlotEvent.Disconnect) {
                        this.inputs[slot].type = 'LIST'
                        this.inputs[slot].label = `${_prefix}_${slot + 1}`
                    }
                }
                return me
            }
        } 








        else if (nodeData.name === "pack_Unpack") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                // shrink outputs to 1
                this.outputs[0].type = "*"
                const output_len = this.outputs.length
                for (let i = output_len - 1; i > 0; i--) {
                    this.removeOutput(i)
                }
                return r

            }
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                const r = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                if (slotType === TypeSlot.Input) {
                    if (event === TypeSlotEvent.Connect && link_info) {
                        // find the origin Pack
                        let link_id = this.inputs[slot]?.link
                        let origin_id = app.graph.links[link_id]?.origin_id
                        let origin_node = null
                        for (let i = 0; i < 10; i++) {
                            origin_node = app.graph._nodes.find(n => n.id == origin_id)
                            if (!origin_node) {
                                break
                            }
                            if (origin_node.type === "pack_Pack") {
                                break
                            }
                            if (origin_node.inputs.length == 0) {
                                origin_node = null
                                break
                            }
                            let origin_slot = -1
                            for (let i in origin_node.inputs) {
                                if (origin_node.inputs[i].type === "PACK") {
                                    origin_slot = i
                                    break
                                } else if (origin_node.inputs[i].type === "*") {
                                    origin_slot = i
                                }
                            }
                            if (origin_slot == -1) {
                                origin_node = null
                                break
                            }
                            link_id = origin_node.inputs[origin_slot]?.link
                            origin_id = app.graph.links[link_id]?.origin_id
                            if (!origin_id) {
                                break
                            }
                            origin_node = null
                        }
                        
                        if (origin_node && origin_node.type === "pack_Pack") {
                            const origin_inputs = origin_node.inputs
                            const output_len = origin_inputs.length - 1  // end is empty socket
                            const cur_len = this.outputs.length
                            for (let i = cur_len - 1; i >= output_len; i--) {
                                this.removeOutput(i)
                            }
                            for (let i = cur_len; i < output_len; i++) {
                                this.addOutput(`${_prefix}_${i + 1}`, origin_inputs[i].type)
                            }
                            for (let i = 0; i < cur_len && i < output_len; i++) {
                                this.outputs[i].type = origin_inputs[i].type
                            }
                        }
                    }
                }
                return r
            }




            
        } else if (nodeData.name == "XXXtype_AnyCast") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined
                const onWidgetChanged = this.widgets[0].callback
                const thisNode = this
                this.widgets[0].callback = function () {
                    const me = onWidgetChanged ? onWidgetChanged.apply(this, arguments) : undefined
                    const output_type = thisNode.widgets[0].value
                    thisNode.outputs[0].type = output_type
                    thisNode.outputs[0].label = output_type
                    thisNode.outputs[0].name = output_type
                    return me
                }
                return me
            }
            // on copy, paste, load
            const onConfigure = nodeType.prototype.onConfigure
            nodeType.prototype.onConfigure = function () {
                const me = onConfigure ? onConfigure.apply(this, arguments) : undefined
                const output_type = this.widgets[0].value
                this.outputs[0].type = output_type
                this.outputs[0].label = output_type
                this.outputs[0].name = output_type
                return me
            }
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                if (slotType === TypeSlot.Input) {
                    if (event === TypeSlotEvent.Connect && link_info) {
                        const origin_node = app.graph.getNodeById(link_info.origin_id)
                        const origin_slot = origin_node.outputs[link_info.origin_slot]
                        const origin_type = origin_slot.type
                        const types = getWorkflowTypes(app)
                        this.widgets[0].options.values = types
                        const output_type = this.widgets[0].value
                        this.outputs[0].type = output_type
                        this.outputs[0].label = output_type
                        this.outputs[0].name = origin_type
                    } else if (event === TypeSlotEvent.Disconnect) {
                        this.outputs[0].type = "*"
                        this.outputs[0].label = "*"
                        this.outputs[0].name = "*"
                    }
                }
                return me
            }
        } else if (nodeData.name === "view_GetWidgetsValues") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined
                this.showValueWidget = ComfyWidgets["STRING"](this, "values", ["STRING", { multiline: true }], app).widget
            }
            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message])
                this.showValueWidget.value = message.text[0]
            }
        } else if (nodeData.name === "view_GetLength") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined
                this.showValueWidget = ComfyWidgets["STRING"](this, "length", ["STRING", { multiline: false }], app).widget
                // this.addWidget("STRING", "length", "", () => { }, { multiline: false })
            }
            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message])
                this.showValueWidget.value = message.text[0]
            }
        } else if (nodeData.name === "view_GetShape") {
            const onNodeCreated = nodeType.prototype.onNodeCreated
            nodeType.prototype.onNodeCreated = function () {
                onNodeCreated ? onNodeCreated.apply(this, []) : undefined
                this.showValueWidget = ComfyWidgets["STRING"](this, "WHBC", ["STRING", { multiline: false }], app).widget
            }
            const onExecuted = nodeType.prototype.onExecuted
            nodeType.prototype.onExecuted = function (message) {
                onExecuted === null || onExecuted === void 0 ? void 0 : onExecuted.apply(this, [message])
                this.showValueWidget.value = message.text[0]
            }
        } else if (nodeData.name === "XXlist_ListGetByIndex" || nodeData.name === "XXlist_ListSlice") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange
            nodeType.prototype.onConnectionsChange = function (slotType, slot, event, link_info, data) {
                const me = onConnectionsChange ? onConnectionsChange.apply(this, arguments) : undefined
                if (slotType === TypeSlot.Input) {
                    if (event === TypeSlotEvent.Connect && link_info) {
                        const node = app.graph.getNodeById(link_info.origin_id)
                        let origin_type = node.outputs[link_info.origin_slot].type
                        this.outputs[0].type = origin_type
                        this.outputs[0].label = origin_type
                        this.outputs[0].name = origin_type
                    } else if (event === TypeSlotEvent.Disconnect) {
                        this.outputs[0].type = "*"
                        this.outputs[0].label = "*"
                        this.outputs[0].name = "*"
                    }
                }
                return me
            }
        }
    }
})
