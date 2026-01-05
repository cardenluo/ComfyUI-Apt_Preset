import { app } from '../../../../scripts/app.js';
import { api } from '../../../../scripts/api.js';

// 注册扩展
app.registerExtension({
    name: 'apt.sigma_editor',

    async beforeRegisterNodeDef(nodeType, nodeData) {
        // 检查是否是我们的交互式sigma节点
        if (nodeData.name === 'scheduler_interactive_sigmas') {
            // 保存原始的onNodeCreated函数
            const originalOnNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function() {
                // 调用原始函数
                if (originalOnNodeCreated) {
                    originalOnNodeCreated.apply(this, arguments);
                }

                const node = this;

                // 找到相关的widgets
                const adjustmentsWidget = this.widgets.find(w => w.name === 'adjustments');
                const stepsWidget = this.widgets.find(w => w.name === 'steps');
                const schedulerWidget = this.widgets.find(w => w.name === 'scheduler');

                // 保存原始的onExecuted函数
                const originalOnExecuted = nodeType.prototype.onExecuted;

                // 添加onExecuted方法来接收后端返回的sigmas_data
                const onExecutedHandler = function(message) {
                    console.log('[scheduler_interactive_sigmas] onExecuted called');
                    console.log('[scheduler_interactive_sigmas] message keys:', message ? Object.keys(message) : 'null');

                    // 调用原始的onExecuted
                    if (originalOnExecuted) {
                        originalOnExecuted.call(this, message);
                    }

                    // 保存后端返回的sigmas_data
                    if (message && message.ui && message.ui.sigmas_data) {
                        console.log('[scheduler_interactive_sigmas] ✅ Received sigmas_data from backend');
                        console.log('[scheduler_interactive_sigmas] sigmas_data:', message.ui.sigmas_data);
                        node.output_data = message;
                    } else {
                        console.log('[scheduler_interactive_sigmas] ⚠️ No sigmas_data in message.ui');
                        if (message) {
                            console.log('[scheduler_interactive_sigmas] message.ui:', message.ui);
                        }
                    }
                };

                // 替换节点的onExecuted方法
                node.onExecuted = onExecutedHandler;

                // 添加编辑按钮
                const editButton = this.addWidget("button", "编辑sigma曲线", "编辑", async () => {
                    // 从节点的widgets中获取参数
                    const steps = stepsWidget ? stepsWidget.value : 20;
                    const scheduler = schedulerWidget ? schedulerWidget.value : 'normal';

                    console.log('========== Sigma Editor Initialization ==========');
                    console.log('Node ID:', node.id);
                    console.log('Node type:', node.type);
                    console.log('Steps:', steps);
                    console.log('Scheduler:', scheduler);
                    console.log('================================================');

                    // 新逻辑：只获取步数，初始化所有点为0
                    // 调整值范围：-1到1（偏移量，会叠加到原始调度器的归一化值上）
                    const pointCount = steps + 1;  // sigmas数组长度 = steps + 1（因为最后一个是0）
                    const initialSigmas = new Array(pointCount).fill(0.0);  // 所有点初始化为0

                    console.log('\n📊 Initial editor state:');
                    console.log('📊 Point count:', pointCount);
                    console.log('📊 All points initialized to 0 (offset values)');
                    console.log('📊 Adjustment range: -1 to +1 (will be added to normalized scheduler values)');
                    console.log('========== End of initialization ==========\n');

                    // 显示交互式编辑器模态窗口
                    showSigmaEditorModal(node, adjustmentsWidget, initialSigmas, scheduler);
                });

                // 将按钮引用保存到节点对象，方便其他地方使用
                node.editButton = editButton;
            };
        }
    }
});

// 显示sigma编辑器模态窗口
function showSigmaEditorModal(node, adjustmentsWidget, currentSigmas, scheduler) {
    console.log('showSigmaEditorModal received sigmas:', currentSigmas ? currentSigmas.length : 0);
    console.log('showSigmaEditorModal scheduler:', scheduler);

    // 创建模态窗口HTML
    const modalHtml = `
        <div class="sigma-modal">
            <div class="sigma-modal-content">
                <style>
                    .sigma-modal {
                        position: fixed;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        background: rgba(0, 0, 0, 0.7);
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        z-index: 1001;
                    }

                    .sigma-modal-content {
                        background: #222;
                        padding: 20px;
                        border-radius: 8px;
                        max-width: 90vw;
                        max-height: 90vh;
                        display: flex;
                        flex-direction: column;
                        gap: 10px;
                        min-width: 600px;
                    }

                    .sigma-editor-main {
                        flex-grow: 1;
                        overflow: hidden;
                        background: white;
                        border-radius: 4px;
                        min-height: 300px;
                    }

                    .sigma-editor-controls {
                        display: flex;
                        justify-content: space-around;
                        align-items: center;
                        gap: 10px;
                    }

                    .sigma-editor-btn {
                        padding: 8px 16px;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 14px;
                        z-index: 1002;
                    }

                    .sigma-confirm-btn {
                        background-color: #4CAF50;
                    }

                    .sigma-cancel-btn {
                        background-color: #f44336;
                    }

                    .sigma-reset-btn {
                        background-color: #5bc0de;
                    }

                    canvas {
                        display: block;
                        width: 100%;
                        height: 100%;
                    }
                </style>
                <h2 style="color: white; margin: 0; text-align: center;">Sigma调整偏移量编辑器 (${scheduler})</h2>
                <div class="sigma-editor-main">
                    <canvas id="sigma-editor-canvas" style="width: 100%; height: 400px;"></canvas>
                </div>
                <div class="sigma-editor-controls">
                    <button class="sigma-editor-btn sigma-reset-btn" id="sigma-reset-btn">重置</button>
                    <button class="sigma-editor-btn sigma-confirm-btn" id="sigma-confirm-btn">确认</button>
                    <button class="sigma-editor-btn sigma-cancel-btn" id="sigma-cancel-btn">取消</button>
                </div>
            </div>
        </div>
    `;

    // 创建模态窗口
    const modal = document.createElement('div');
    modal.id = 'sigma-editor-modal';
    modal.innerHTML = modalHtml;
    document.body.appendChild(modal);

    // 直接获取canvas元素
    const canvas = modal.querySelector('#sigma-editor-canvas');
    canvas.width = 600;
    canvas.height = 400;

    // 获取2D上下文
    const ctx = canvas.getContext('2d');

    // 初始化编辑器
    const editor = new SigmaCurveEditor(canvas, ctx, node, adjustmentsWidget, currentSigmas);

    // 添加按钮事件监听器
    const resetBtn = modal.querySelector('#sigma-reset-btn');
    const confirmBtn = modal.querySelector('#sigma-confirm-btn');
    const cancelBtn = modal.querySelector('#sigma-cancel-btn');

    // 添加点击事件处理
    resetBtn.onclick = () => {
        editor.reset();
    };

    confirmBtn.onclick = () => {
        editor.save();
        modal.remove();
    };

    cancelBtn.onclick = () => {
        modal.remove();
    };

    // 添加键盘事件处理，按ESC键关闭
    const handleKeyDown = (e) => {
        if (e.key === 'Escape') {
            cleanup();
            modal.remove();
        }
    };

    document.addEventListener('keydown', handleKeyDown);

    // 清理函数
    const cleanup = () => {
        document.removeEventListener('keydown', handleKeyDown);
    };

    // 监听模态窗口移除事件 (使用MutationObserver替代已废弃的DOMNodeRemoved)
    const observer = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
            if (mutation.type === 'childList') {
                if (!document.body.contains(modal)) {
                    cleanup();
                    observer.disconnect();
                }
            }
        });
    });
    observer.observe(document.body, { childList: true });
}

// Sigma曲线编辑器类 - 用于模态窗口
class SigmaCurveEditor {
    constructor(canvas, ctx, node, adjustmentsWidget, currentSigmas = []) {
        this.canvas = canvas;
        this.ctx = ctx;
        this.node = node;
        this.adjustmentsWidget = adjustmentsWidget;
        this.sigmas = currentSigmas;
        this.adjustments = {};
        this.isDragging = false;
        this.dragIndex = -1;
        this.nodeSize = 8;

        // 加载现有调整和sigmas数据
        this.loadData();

        // 添加事件监听器
        this.canvas.addEventListener('mousedown', this.onMouseDown.bind(this));
        this.canvas.addEventListener('mousemove', this.onMouseMove.bind(this));
        this.canvas.addEventListener('mouseup', this.onMouseUp.bind(this));
        this.canvas.addEventListener('mouseleave', this.onMouseUp.bind(this));

        // 初始绘制
        this.draw();
    }

    // 从JSON字符串加载调整数据
    loadData() {
        try {
            let jsonStr = this.adjustmentsWidget.value || '{}';

            // 如果widget.value不是JSON格式（比如是"编辑"这种按钮文本），使用默认值
            if (typeof jsonStr !== 'string' || !jsonStr.trim().startsWith('[') && !jsonStr.trim().startsWith('{')) {
                console.warn('adjustmentsWidget.value is not a valid JSON string, using default');
                this.adjustments = {};
                return;
            }

            const data = JSON.parse(jsonStr);

            // 新格式：只包含调整值数组
            if (Array.isArray(data)) {
                this.adjustments = {};
                data.forEach(adj => {
                    this.adjustments[adj.index] = adj.value;
                });
                console.log('Loaded adjustments from array format:', this.adjustments);
            }
            // 兼容旧格式：包含sigmas和adjustments的对象
            else if (typeof data === 'object' && data !== null) {
                if (data.adjustments && Array.isArray(data.adjustments)) {
                    this.adjustments = {};
                    data.adjustments.forEach(adj => {
                        this.adjustments[adj.index] = adj.value;
                    });
                    console.log('Loaded adjustments from old format:', this.adjustments);
                } else if (data.reset === true) {
                    console.log('Detected reset state, clearing adjustments');
                    this.adjustments = {};
                } else {
                    // 旧格式，直接是调整数组
                    this.adjustments = {};
                    Object.keys(data).forEach(key => {
                        if (key !== 'sigmas' && key !== 'reset') {
                            this.adjustments[parseInt(key)] = data[key];
                        }
                    });
                    console.log('Loaded adjustments from object format:', this.adjustments);
                }
            }
        } catch (e) {
            console.warn('Error loading adjustments data:', e);
            console.warn('Adjustments widget value:', this.adjustmentsWidget.value);
            this.adjustments = {};
        }

        console.log('Final loaded adjustments data:', this.adjustments);
    }

    // 将调整数据保存为JSON字符串
    saveAdjustments() {
        const data = [];
        for (const [index, value] of Object.entries(this.adjustments)) {
            data.push({ index: parseInt(index), value });
        }
        // 只保存调整值数组（不保存sigmas，因为sigmas由后端生成）
        return JSON.stringify(data);
    }

    // 绘制sigma曲线 - 调整偏移量编辑器
    draw() {
        if (!this.ctx || this.sigmas.length === 0) {
            this.drawEmptyState();
            return;
        }

        const ctx = this.ctx;
        const canvas = this.canvas;

        // 定义绘图区域边距
        const margin = { top: 30, right: 30, bottom: 50, left: 60 };
        const plotWidth = canvas.width - margin.left - margin.right;
        const plotHeight = canvas.height - margin.top - margin.bottom;

        // 清空画布
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 调整值范围：-1到1
        const valueMin = -1.0;
        const valueMax = 1.0;

        // 调试信息：打印调整值
        console.log(`[SigmaEditor] Drawing ${this.sigmas.length} points`);
        console.log(`[SigmaEditor] Adjustment range: ${valueMin} to ${valueMax}`);
        console.log(`[SigmaEditor] Values (first 5): ${this.sigmas.slice(0, 5).map(s => s.toFixed(4))}`);
        console.log(`[SigmaEditor] Values (last 5): ${this.sigmas.slice(-5).map(s => s.toFixed(4))}`);

        // 计算横坐标刻度间隔 - 基于步数
        const totalSteps = this.sigmas.length - 1;
        let xTickInterval = 1;
        if (totalSteps > 50) {
            xTickInterval = 10;
        } else if (totalSteps > 20) {
            xTickInterval = 5;
        } else if (totalSteps > 10) {
            xTickInterval = 2;
        }

        // 绘制网格背景
        ctx.strokeStyle = '#e0e0e0';
        ctx.lineWidth = 0.5;

        // 纵向网格线 (根据步数)
        const xTicks = [];
        xTicks.push(0);
        for (let i = xTickInterval; i < totalSteps; i += xTickInterval) {
            xTicks.push(i);
        }
        if (totalSteps > 0) {
            xTicks.push(totalSteps);
        }

        xTicks.forEach(tick => {
            const x = margin.left + (tick / totalSteps) * plotWidth;
            ctx.beginPath();
            ctx.moveTo(x, margin.top);
            ctx.lineTo(x, margin.top + plotHeight);
            ctx.stroke();
        });

        // 横向网格线 - -1到1的范围，间隔0.5
        const yTicks = [-1.0, -0.5, 0.0, 0.5, 1.0];

        yTicks.forEach(tick => {
            // 将-1到1的值映射到画布高度（-1在底部，1在顶部）
            const normalizedValue = (tick - valueMin) / (valueMax - valueMin);
            const y = margin.top + (1 - normalizedValue) * plotHeight;
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(canvas.width - margin.right, y);
            ctx.stroke();
        });

        // 绘制零线（y=0），用不同的颜色突出显示
        const zeroY = margin.top + 0.5 * plotHeight;  // 0在正中间
        ctx.strokeStyle = '#999999';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);  // 虚线
        ctx.beginPath();
        ctx.moveTo(margin.left, zeroY);
        ctx.lineTo(canvas.width - margin.right, zeroY);
        ctx.stroke();
        ctx.setLineDash([]);  // 恢复实线

        // 应用调整 - 在原始调整值上进行调整
        const adjustedSigmas = this.sigmas.map((s, i) => {
            if (this.adjustments[i] !== undefined) {
                // 使用调整值(-1到1)
                return Math.max(-1.0, Math.min(1.0, this.adjustments[i]));
            }
            // 没有调整的点，使用原始值（应该是0）
            return s;
        });

        // 绘制曲线
        ctx.strokeStyle = '#007bff';
        ctx.lineWidth = 2;
        ctx.beginPath();

        adjustedSigmas.forEach((s, i) => {
            const x = margin.left + (i / (adjustedSigmas.length - 1)) * plotWidth;
            // 将-1到1的值映射到画布高度
            const normalizedValue = (s - valueMin) / (valueMax - valueMin);
            const y = margin.top + (1 - normalizedValue) * plotHeight;

            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });

        ctx.stroke();

        // 绘制调整点
        adjustedSigmas.forEach((s, i) => {
            const x = margin.left + (i / (adjustedSigmas.length - 1)) * plotWidth;
            const normalizedValue = (s - valueMin) / (valueMax - valueMin);
            const y = margin.top + (1 - normalizedValue) * plotHeight;

            // 绘制节点（调整过的点用红色，未调整的用蓝色）
            ctx.fillStyle = this.adjustments[i] !== undefined ? '#ff4757' : '#007bff';
            ctx.beginPath();
            ctx.arc(x, y, this.nodeSize, 0, 2 * Math.PI);
            ctx.fill();

            // 绘制节点边框
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });

        // 绘制坐标轴
        ctx.strokeStyle = '#333333';
        ctx.lineWidth = 1;

        // 横坐标轴
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top + plotHeight);
        ctx.lineTo(canvas.width - margin.right, margin.top + plotHeight);
        ctx.stroke();

        // 横坐标刻度
        ctx.fillStyle = '#333333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';

        xTicks.forEach(tick => {
            const x = margin.left + (tick / totalSteps) * plotWidth;

            // 绘制刻度线
            ctx.beginPath();
            ctx.moveTo(x, margin.top + plotHeight);
            ctx.lineTo(x, margin.top + plotHeight + 8);
            ctx.stroke();

            // 绘制刻度值
            ctx.fillText(tick.toString(), x, margin.top + plotHeight + 10);
        });

        // 横坐标标签
        ctx.font = '14px Arial';
        ctx.fillText('Steps (步数)', margin.left + plotWidth / 2, canvas.height - 10);

        // 纵坐标轴
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, margin.top + plotHeight);
        ctx.stroke();

        // 纵坐标刻度 (-1到1范围)
        ctx.fillStyle = '#333333';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';

        yTicks.forEach(tick => {
            const normalizedValue = (tick - valueMin) / (valueMax - valueMin);
            const y = margin.top + (1 - normalizedValue) * plotHeight;

            // 绘制刻度线
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(margin.left - 8, y);
            ctx.stroke();

            // 绘制刻度值
            ctx.fillText(tick.toFixed(1), margin.left - 10, y);
        });

        // 纵坐标标签（旋转）
        ctx.save();
        ctx.translate(15, margin.top + plotHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Adjustment Offset (偏移量)', 0, 0);
        ctx.restore();

        // 绘制标题
        ctx.fillStyle = '#333333';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText(`Sigma 调整偏移量 (总步数: ${totalSteps})`, margin.left + plotWidth / 2, 5);
    }

    // 绘制空状态
    drawEmptyState() {
        if (!this.ctx) return;

        const ctx = this.ctx;
        const canvas = this.canvas;

        // 清空画布
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 绘制提示文本
        ctx.fillStyle = '#888888';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('No sigmas data available', canvas.width / 2, canvas.height / 2);
    }

    // 获取鼠标位置对应的sigma索引和调整值
    getMousePosition(event) {
        const rect = this.canvas.getBoundingClientRect();

        // 计算缩放比例（canvas内部尺寸 vs 显示尺寸）
        const scaleX = this.canvas.width / rect.width;
        const scaleY = this.canvas.height / rect.height;

        const x = (event.clientX - rect.left) * scaleX;
        const y = (event.clientY - rect.top) * scaleY;

        // 定义绘图区域边距
        const margin = { top: 30, right: 30, bottom: 50, left: 60 };
        const plotWidth = this.canvas.width - margin.left - margin.right;
        const plotHeight = this.canvas.height - margin.top - margin.bottom;

        // 调整值范围：-1到1
        const valueMin = -1.0;
        const valueMax = 1.0;

        // 计算最近的点
        let minDist = Infinity;
        let closestIndex = -1;

        if (this.sigmas.length > 0) {
            for (let i = 0; i < this.sigmas.length; i++) {
                const adjustedSigma = this.adjustments[i] !== undefined
                    ? this.adjustments[i]
                    : this.sigmas[i];

                // 计算点的位置
                const normalizedValue = (adjustedSigma - valueMin) / (valueMax - valueMin);
                const px = margin.left + (i / (this.sigmas.length - 1)) * plotWidth;
                const py = margin.top + (1 - normalizedValue) * plotHeight;

                const dist = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
                if (dist < minDist && dist < this.nodeSize * 3) {  // 增加距离阈值
                    minDist = dist;
                    closestIndex = i;
                }
            }
        }

        // 计算鼠标位置对应的调整值（-1到1）
        const normalizedY = 1 - (y - margin.top) / plotHeight;  // Y轴向上（1在顶部）
        const adjustmentValue = valueMin + normalizedY * (valueMax - valueMin);

        // 添加调试日志
        if (Math.random() < 0.05) {  // 只有5%的概率输出，避免日志过多
            console.log(`[getMousePosition] Mouse: (${x.toFixed(1)}, ${y.toFixed(1)}), Closest index: ${closestIndex}, Distance: ${minDist.toFixed(1)}`);
        }

        return { index: closestIndex, value: Math.max(-1.0, Math.min(1.0, adjustmentValue)) };
    }

    // 鼠标按下事件
    onMouseDown(event) {
        const pos = this.getMousePosition(event);
        if (pos.index !== -1) {
            this.isDragging = true;
            this.dragIndex = pos.index;
            this.adjustments[pos.index] = pos.value;
            console.log(`[interactive_editor] Point ${pos.index} adjusted to ${pos.value}`);
            this.draw();
        }
    }

    // 鼠标移动事件
    onMouseMove(event) {
        if (this.isDragging && this.dragIndex !== -1) {
            const pos = this.getMousePosition(event);
            this.adjustments[this.dragIndex] = pos.value;
            // 添加调试日志，但限制输出频率
            if (Math.random() < 0.1) {  // 只有10%的概率输出，避免日志过多
                console.log(`[interactive_editor] Dragging point ${this.dragIndex} to ${pos.value}`);
            }
            this.draw();
        } else {
            const pos = this.getMousePosition(event);
            this.canvas.style.cursor = pos.index !== -1 ? 'pointer' : 'crosshair';
        }
    }

    // 鼠标释放事件
    onMouseUp(event) {
        this.isDragging = false;
        this.dragIndex = -1;
        this.canvas.style.cursor = 'crosshair';
    }

    // 重置调整
    reset() {
        this.adjustments = {};
        
        // 重置widget值为默认值
        this.adjustmentsWidget.value = '{}';
        if (this.adjustmentsWidget.inputEl) {
            this.adjustmentsWidget.inputEl.value = '{}';
        }
        
        // 触发节点更新
        if (this.node.onWidgetValue_changed) {
            this.node.onWidgetValue_changed(this.adjustmentsWidget, '{}');
        }
        
        // 重新绘制
        this.draw();
        
        // 添加重置成功的视觉反馈
        const canvas = this.canvas;
        const ctx = this.ctx;
        const originalAlpha = ctx.globalAlpha;
        
        // 显示重置成功的提示
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = '#4CAF50';
        ctx.fillRect(0, 0, canvas.width, 40);
        
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#FFFFFF';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('已重置为原始值', canvas.width / 2, 20);
        
        // 2秒后清除提示并重绘
        setTimeout(() => {
            this.draw();
        }, 1000);
    }

    // 保存调整
    save() {
        const jsonStr = this.saveAdjustments();
        this.adjustmentsWidget.value = jsonStr;
        if (this.adjustmentsWidget.inputEl) {
            this.adjustmentsWidget.inputEl.value = jsonStr;
        }
        
        // 添加调试日志
        console.log("[interactive_editor] Saving adjustments:", this.adjustments);
        console.log("[interactive_editor] JSON string:", jsonStr);

        // 触发节点更新
        if (this.node.onWidgetValue_changed) {
            this.node.onWidgetValue_changed(this.adjustmentsWidget, jsonStr);
        }
    }
}
