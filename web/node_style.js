import { app } from '../../scripts/app.js';

app.registerExtension({
    name: 'apt.node_width_and_style',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 检查是否是Apt_Preset节点
        const isAptPreset = nodeData.name && (
            nodeData.name.startsWith('sum_') || 
            nodeData.name.startsWith('Data_') || 
            nodeData.name.startsWith('AD_') || 
            nodeData.name.startsWith('basic_') || 
            nodeData.name.startsWith('chx_') || 
            nodeData.name.startsWith('Apply_') || 
            nodeData.name.startsWith('Stack_') || 
            nodeData.name.startsWith('param_') || 
            nodeData.name.startsWith('Model_') || 
            nodeData.name.startsWith('CN_') || 
            nodeData.name.startsWith('photoshop_') || 
            nodeData.name.startsWith('load_') ||
            nodeData.name.startsWith('IO_') ||
            nodeData.name.startsWith('view_') ||
            nodeData.name.startsWith('pack_') ||
            nodeData.name.startsWith('list_') ||
            nodeData.name.startsWith('batch_') ||
            nodeData.name.startsWith('type_') ||
            nodeData.name.startsWith('math_') ||
            nodeData.name.startsWith('model_') ||
            nodeData.name.startsWith('Image_') ||
            nodeData.name.startsWith('Mask_') ||
            nodeData.name.startsWith('latent_') ||
            nodeData.name.startsWith('text_') ||
            nodeData.name.startsWith('stack_') ||
            nodeData.name.startsWith('color_') ||
            nodeData.name.startsWith('img_') ||
            nodeData.name.startsWith('lay_') ||
            nodeData.name.startsWith('GPT_') ||
            nodeData.name.startsWith('pre_') ||
            nodeData.name.startsWith('basicIn_') ||
            nodeData.name.startsWith('sampler_') ||
            nodeData.name.startsWith('Amp_') ||
            nodeData.name.startsWith('excel_') ||
            nodeData.name.startsWith('create_') ||
            nodeData.name.startsWith('sch_') ||
            nodeData.name.startsWith('AI_') ||
            nodeData.name.startsWith('flow_') ||
            nodeData.name.startsWith('AD_')||
            nodeData.name.startsWith('texture')


        );

        if (isAptPreset) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);

                // 设置初始宽度为 260
                this.size[0] = 260;
                this.setSize([260, this.size[1]]);

                // 修改 computeSize 方法，确保宽度不小于 160
                const originalComputeSize = this.computeSize;
                this.computeSize = function () {
                    const size = originalComputeSize.call(this);
                    size[0] = Math.max(160, Math.min(500, size[0]));
                    return size;
                };


                return r;
            };
        }
        
    }
});
