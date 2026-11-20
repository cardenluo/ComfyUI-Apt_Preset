import {api} from "../../../scripts/api.js";
import {app} from "../../../scripts/app.js";
import {$el} from "../../../scripts/ui.js";


let refreshRaf = null;
let lastRefreshAt = 0;
const REFRESH_INTERVAL_MS = 100; // throttle dirty-canvas flagging to ~10fps
const TASK_TIMER_SETTING = 'Apt_Preset.Display.ShowTaskTimer';
const NODE_TIMER_SETTING = 'Apt_Preset.Display.ShowNodeTimer';

function taskTimerEnabled() {
    return app.ui.settings.getSettingValue(TASK_TIMER_SETTING, false) === true;
}

function nodeTimerEnabled() {
    return app.ui.settings.getSettingValue(NODE_TIMER_SETTING, false) === true;
}

function stopRefreshTimer() {
    if (refreshRaf !== null) {
        cancelAnimationFrame(refreshRaf);
        refreshRaf = null;
    }
    lastRefreshAt = 0;
}

function startRefreshTimer() {
    stopRefreshTimer();
    const tick = (now) => {
        if (now - lastRefreshAt >= REFRESH_INTERVAL_MS) {
            lastRefreshAt = now;
            if (app?.graph) {
                app.graph.setDirtyCanvas(true, false);
            }
            updateTimerDisplay();
        }
        refreshRaf = requestAnimationFrame(tick);
    };
    refreshRaf = requestAnimationFrame(tick);
}


// endregion

function formatExecutionTime(time) {
    return `${(time / 1000.0).toFixed(2)}s`
}


function formatBytes(bytes, decimals) {
    if (bytes === 0) {
        return '0 B'
    }
    const k = 1024,
        dm = decimals || 2,
        sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'],
        i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}


function drawBadge(node, orig, restArgs) {
    let ctx = restArgs[0];
    const r = orig?.apply?.(node, restArgs);
    if (!nodeTimerEnabled()) return r;

    try {
        // Check if node is collapsed or has no title
        const NO_TITLE = LiteGraph?.NO_TITLE ?? 0;
        const NODE_TITLE_HEIGHT = LiteGraph?.NODE_TITLE_HEIGHT ?? 24;
        
        if (!node.flags?.collapsed && node.constructor?.title_mode != NO_TITLE) {
            let text = "";
            if (node.ty_et_execution_time !== undefined) {
                text = formatExecutionTime(node.ty_et_execution_time) + " - vram " + formatBytes(node.ty_et_vram_used, 2);
            } else if (node.ty_et_start_time !== undefined) {
                text = formatExecutionTime(performance.now() - node.ty_et_start_time);
            }
            if (!text) {
                return r;
            }
            let fgColor = "white";
            let bgColor = "#0F1F0F";

            ctx.save();
            ctx.font = "12px sans-serif";
            const textSize = ctx.measureText(text);
            ctx.fillStyle = bgColor;
            ctx.beginPath();
            const paddingHorizontal = 6;
            const badgeY = -NODE_TITLE_HEIGHT - 20;
            if (typeof ctx.roundRect === 'function') {
                ctx.roundRect(0, badgeY, textSize.width + paddingHorizontal * 2, 20, 5);
            } else {
                ctx.rect(0, badgeY, textSize.width + paddingHorizontal * 2, 20);
            }
            ctx.fill();

            ctx.fillStyle = fgColor;
            ctx.fillText(text, paddingHorizontal, badgeY + 14);
            ctx.restore();
        }
    } catch (e) {
        console.warn('[ExecutionTime] Error drawing badge:', e);
    }
    return r;
}

let lastRunningDate = null;
let runningData = null;

// region: Total Execution Timer Display
let executionStartTime = null;
let timerEl = null;
const TIMER_POS_KEY = 'apt-preset-et-timer-pos';
const HISTORY_KEY = 'apt-preset-et-history-v2';
const HISTORY_LIMIT = 30;
let executionHistory = [];
let pendingExecutionHistory = null;
let activeTaskIndex = null;
let selectedTaskIndex = null;
let timerSelectEl = null;
let timerValueEl = null;

function formatTotalTime(ms) {
    const totalSeconds = Math.floor(ms / 1000);
    const minutes = Math.floor(totalSeconds / 60);
    const seconds = totalSeconds % 60;
    return `${minutes}分${seconds}秒`;
}

function loadExecutionHistory() {
    try {
        const history = JSON.parse(localStorage.getItem(HISTORY_KEY));
        return Array.isArray(history) ? history.slice(0, HISTORY_LIMIT) : [];
    } catch (e) {
        return [];
    }
}

function saveExecutionHistory() {
    try {
        localStorage.setItem(HISTORY_KEY, JSON.stringify(executionHistory));
    } catch (e) { /* ignore */ }
}

function beginExecutionHistory() {
    pendingExecutionHistory = executionHistory;
    executionHistory = [];
    activeTaskIndex = 1;
    selectedTaskIndex = 1;
    renderTimerTasks();
}

function renderTimerTasks() {
    if (!timerSelectEl) return;
    const stages = new Set(executionHistory.map((entry) => entry.stage).filter(Number.isInteger));
    if (Number.isInteger(activeTaskIndex)) stages.add(activeTaskIndex);
    if (!stages.size) stages.add(1);
    timerSelectEl.replaceChildren();
    for (const stage of [...stages].sort((a, b) => a - b)) {
        const option = document.createElement('option');
        option.value = String(stage);
        option.textContent = `index${stage}`;
        timerSelectEl.appendChild(option);
    }
    selectedTaskIndex = selectedTaskIndex || activeTaskIndex || 1;
    if (!stages.has(selectedTaskIndex)) selectedTaskIndex = [...stages][0];
    timerSelectEl.value = String(selectedTaskIndex);
}

function saveTimerPosition() {
    if (!timerEl) return;
    try {
        localStorage.setItem(TIMER_POS_KEY, JSON.stringify({
            left: timerEl.style.left,
            top: timerEl.style.top,
        }));
    } catch (e) { /* ignore */ }
}

function ensureTimerDisplay() {
    if (timerEl) return;

    timerEl = document.createElement('div');
    timerEl.id = 'apt-preset-et-total-timer';

    // Restore saved position, or default to top-center
    let savedPos = null;
    try {
        const raw = localStorage.getItem(TIMER_POS_KEY);
        if (raw) savedPos = JSON.parse(raw);
    } catch (e) { /* ignore */ }

    Object.assign(timerEl.style, {
        position: 'fixed',
        top: savedPos?.top || '45px',
        left: savedPos?.left || '50%',
        transform: savedPos ? 'none' : 'translateX(-50%)',
        zIndex: '9999',
        background: 'rgba(0, 0, 0, 0.78)',
        color: '#00e676',
        padding: '5px 18px',
        borderRadius: '14px',
        fontFamily: '"Cascadia Code", "Fira Code", "Consolas", monospace',
        fontSize: '15px',
        fontWeight: '600',
        letterSpacing: '0.5px',
        cursor: 'grab',
        userSelect: 'none',
        boxShadow: '0 2px 8px rgba(0,0,0,0.35)',
        transition: 'box-shadow 0.15s',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
    });
    timerSelectEl = document.createElement('select');
    Object.assign(timerSelectEl.style, {
        color: '#00e676',
        background: 'transparent',
        border: 'none',
        font: 'inherit',
        fontWeight: '600',
        cursor: 'pointer',
        outline: 'none',
    });
    timerSelectEl.addEventListener('change', () => {
        selectedTaskIndex = Number(timerSelectEl.value);
        updateTimerDisplay();
    });
    timerValueEl = document.createElement('span');
    timerValueEl.textContent = '0分0秒';
    renderTimerTasks();
    timerEl.append(timerSelectEl, timerValueEl);

    // --- Drag logic ---
    let dragging = false;
    let dragStartX, dragStartY, startLeft, startTop;

    function onDragStart(e) {
        if (e.button !== undefined && e.button !== 0) return; // only left click
        if (e.target === timerSelectEl) return;
        dragging = true;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        dragStartX = clientX;
        dragStartY = clientY;
        startLeft = timerEl.offsetLeft;
        startTop = timerEl.offsetTop;
        timerEl.style.cursor = 'grabbing';
        timerEl.style.transition = 'none';
        timerEl.style.transform = 'none'; // drop centering once dragged
        if (e.preventDefault) e.preventDefault();
    }

    function onDragMove(e) {
        if (!dragging) return;
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        const dx = clientX - dragStartX;
        const dy = clientY - dragStartY;
        timerEl.style.left = Math.max(0, startLeft + dx) + 'px';
        timerEl.style.top = Math.max(0, startTop + dy) + 'px';
    }

    function onDragEnd() {
        if (!dragging) return;
        dragging = false;
        timerEl.style.cursor = 'grab';
        timerEl.style.transition = 'box-shadow 0.15s';
        saveTimerPosition();
    }

    timerEl.addEventListener('mousedown', onDragStart);
    timerEl.addEventListener('touchstart', onDragStart, {passive: false});
    window.addEventListener('mousemove', onDragMove);
    window.addEventListener('touchmove', onDragMove, {passive: false});
    window.addEventListener('mouseup', onDragEnd);
    window.addEventListener('touchend', onDragEnd);

    document.body.appendChild(timerEl);
    setTaskTimerVisible(taskTimerEnabled());
}

function setTaskTimerVisible(enabled) {
    if (timerEl) {
        timerEl.style.display = enabled ? 'flex' : 'none';
    }
}

function setNodeTimerVisible() {
    app.graph?.setDirtyCanvas(true, false);
}

const SETTINGS = [
    {
        id: TASK_TIMER_SETTING,
        name: '是否打开任务时间',
        type: 'boolean',
        defaultValue: false,
        onChange: setTaskTimerVisible,
    },
    {
        id: NODE_TIMER_SETTING,
        name: '是否打开节点上面的时间',
        type: 'boolean',
        defaultValue: false,
        onChange: setNodeTimerVisible,
    },
];

function showTimerDisplay() {
    ensureTimerDisplay();
    activeTaskIndex = 1;
    selectedTaskIndex = activeTaskIndex;
    renderTimerTasks();
    executionStartTime = performance.now();
    timerEl.style.color = '#00e676';
    updateTimerDisplay();
}

function updateTimerDisplay() {
    if (!timerValueEl) return;
    if (executionStartTime !== null && selectedTaskIndex === activeTaskIndex) {
        const elapsed = performance.now() - executionStartTime;
        timerValueEl.textContent = formatTotalTime(elapsed);
        return;
    }
    const history = executionHistory.find((entry) => entry.stage === selectedTaskIndex);
    timerValueEl.textContent = history ? formatTotalTime(history.execution_time) : '0分0秒';
}

function freezeTimerDisplay() {
    executionStartTime = null;
    if (timerEl) {
        timerEl.style.color = '#aaaaaa'; // dimmed to indicate idle
    }
}
// endregion

function exportTable(table, separator = ',') {
    // Select rows from table_id
    var rows = table.querySelectorAll('tr');
    // Construct csv
    var csv = [];
    for (var i = 0; i < rows.length; i++) {
        var row = [], cols = rows[i].querySelectorAll('td, th');
        for (var j = 0; j < cols.length; j++) {
            var data = cols[j].innerText.replace(/(\r\n|\n|\r)/gm, '').replace(/(\s\s)/gm, ' ')
            data = data.replace(/"/g, '""');
            row.push('"' + data + '"');
        }
        csv.push(row.join(separator));
    }
    var csv_string = csv.join('\n');
    // Download it
    var filename = 'execution_time' + new Date().toLocaleDateString() + '.csv';
    var link = document.createElement('a');
    link.style.display = 'none';
    link.setAttribute('target', '_blank');
    link.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv_string));
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function buildTableHtml() {
    const tableBody = $el("tbody")
    const tableFooter = $el("tfoot", {style: {"background": "var(--comfy-input-bg)"}})
    const headerThStyle = {"white-space": "nowrap"}
    const table = $el("table", {
        textAlign: "right",
        border: "1px solid var(--border-color)",
        style: {"border": "none", "border-spacing": "0", "font-size": "14px", "width": "100%"}
    }, [
        $el("thead", {style: {"background": "var(--comfy-input-bg)"}}, [
            $el("tr", [
                $el("th", {style: headerThStyle, "textContent": "Node Id"}),
                $el("th", {style: headerThStyle, "textContent": "Node Title"}),
                $el("th", {style: headerThStyle, "textContent": "Current Time"}),
                $el("th", {style: headerThStyle, "textContent": "Per Time"}),
                $el("th", {style: headerThStyle, "textContent": "Cur / Pre Time Diff"}),
                $el("th", {style: headerThStyle, "textContent": "VRAM Used"})
            ])
        ]),
        tableBody,
        tableFooter
    ]);
    if (runningData?.nodes_execution_time === undefined) {
        return table;
    }

    function diff(current, pre) {
        let diffText;
        let diffColor;
        if (pre) {
            const diffTime = current - pre;
            const diffPercentText = `${(diffTime * 100 / pre).toFixed(2)}%`;
            if (diffTime > 0) {
                diffColor = 'red';
                diffText = `+${formatExecutionTime(diffTime)} / +${diffPercentText}`;
            } else if (diffPercentText === '0.00%') {
                diffColor = 'white';
                diffText = formatExecutionTime(diffTime);
            } else {
                diffColor = 'green';
                diffText = `${formatExecutionTime(diffTime)} / ${diffPercentText}`;
            }
        }
        return [diffColor, diffText]
    }

    let max_execution_time = null
    let max_vram_used = null

    runningData.nodes_execution_time.forEach(function (item) {
        const nodeId = item.node;
        const node = app.graph.getNodeById(nodeId)
        const title = node?.title ?? nodeId
        const preExecutionTime = lastRunningDate?.nodes_execution_time?.find(x => x.node === nodeId)?.execution_time

        const [diffColor, diffText] = diff(item.execution_time, preExecutionTime);

        if (max_execution_time == null || item.execution_time > max_execution_time) {
            max_execution_time = item.execution_time
        }

        if (max_vram_used == null || item.vram_used > max_vram_used) {
            max_vram_used = item.vram_used
        }

        tableBody.append($el("tr", {
            style: {"cursor": "pointer"},
            onclick: () => {
                if (node) {
                    app.canvas.selectNode(node, false);
                }
            }
        }, [
            $el("td", {style: {"textAlign": "right"}, "textContent": nodeId}),
            $el("td", {style: {"textAlign": "right"}, "textContent": title}),
            $el("td", {style: {"textAlign": "right"}, "textContent": formatExecutionTime(item.execution_time)}),
            $el("td", {
                style: {"textAlign": "right"},
                "textContent": preExecutionTime !== undefined ? formatExecutionTime(preExecutionTime) : undefined
            }),
            $el("td", {
                style: {
                    "textAlign": "right",
                    "color": diffColor
                },
                "textContent": diffText
            }),
            $el("td", {style: {"textAlign": "right"}, "textContent": formatBytes(item.vram_used, 2)}),
        ]))
    });
    if (runningData.total_execution_time !== null) {
        const [diffColor, diffText] = diff(runningData.total_execution_time, lastRunningDate?.total_execution_time);

         tableFooter.append($el("tr", [
            $el("td", {style: {"textAlign": "right"}, "textContent": 'Max'}),
            $el("td", {style: {"textAlign": "right"}, "textContent": ''}),
            $el("td", {
                style: {"textAlign": "right"},
                "textContent": max_execution_time != null ? formatExecutionTime(max_execution_time) : ''
            }),
            $el("td", {
                style: {"textAlign": "right"},
                "textContent": ''
            }),
            $el("td", {
                style: {
                    "textAlign": "right",
                    "color": diffColor
                },
                "textContent": ''
            }),
            $el("td", {
                style: {"textAlign": "right"},
                "textContent": max_vram_used != null ? formatBytes(max_vram_used, 2) : ''
            }),
        ]))

        tableFooter.append($el("tr", [
            $el("td", {style: {"textAlign": "right"}, "textContent": 'Total'}),
            $el("td", {style: {"textAlign": "right"}, "textContent": ''}),
            $el("td", {
                style: {"textAlign": "right"},
                "textContent": formatExecutionTime(runningData.total_execution_time)
            }),
            $el("td", {
                style: {"textAlign": "right"},
                "textContent": lastRunningDate?.total_execution_time ? formatExecutionTime(lastRunningDate?.total_execution_time) : undefined
            }),
            $el("td", {
                style: {
                    "textAlign": "right",
                    "color": diffColor
                },
                "textContent": diffText
            }),
            $el("td", {style: {"textAlign": "right"}, "textContent": ""}),
        ]))
    }
    return table;
}

function refreshTable() {
    app.graph._nodes.forEach(function (node) {
        if (node.comfyClass === "TY_ExecutionTime" && node.widgets) {
            const tableWidget = node.widgets.find((w) => w.name === "Table");
            if (!tableWidget) {
                return;
            }
            tableWidget.inputEl.replaceChild(buildTableHtml(), tableWidget.inputEl.firstChild);
            const computeSize = node.computeSize();
            const newSize = [Math.max(node.size[0], computeSize[0]), Math.max(node.size[1], computeSize[1])];
            node.setSize(newSize);
            app.graph.setDirtyCanvas(true);
        }
    });
}

app.registerExtension({
    name: "ComfyUI-Apt_Preset.ExecutionTime",
    settings: SETTINGS,
    async setup() {
        // Initialize runningData
        if (!runningData) {
            runningData = {
                nodes_execution_time: [],
                total_execution_time: null
            };
        }
        executionHistory = loadExecutionHistory();


        ensureTimerDisplay();
        
        api.addEventListener("executing", ({detail}) => {
            const nodeId = detail;
            if (!nodeId) { // Finish
                return
            }
            const node = app.graph.getNodeById(nodeId)
            if (node) {
                node.ty_et_start_time = performance.now();
                app.graph.setDirtyCanvas(true, false);
            }
        });

        api.addEventListener("TyDev-Utils.ExecutionTime.executed", ({detail}) => {
            const node = app.graph.getNodeById(detail.node)
            if (node) {
                node.ty_et_execution_time = detail.execution_time;
                node.ty_et_vram_used = detail.vram_used;
            }
            const index = runningData.nodes_execution_time.findIndex(x => x.node === detail.node);
            const data = {
                node: detail.node,
                execution_time: detail.execution_time,
                vram_used: detail.vram_used
            };
            if (index >= 0) {
                runningData.nodes_execution_time[index] = data
            } else {
                runningData.nodes_execution_time.push(data)
            }
            if (detail.class_type === "flow_stage_begin") {
                const stageIndex = Number(detail.flow_stage_index);
                const stageTotal = Number(detail.flow_stage_total);
   
                const stage = Number.isInteger(stageIndex) && stageIndex >= 0
                    ? (stageIndex === 0 ? 1 : stageIndex)
                    : null;
                runningData.flow_stage = {
                    stage,
                    total: Number.isFinite(stageTotal) ? stageTotal : null,
                };
                if (stage === 1) {
                    executionHistory = [];
                    pendingExecutionHistory = null;
                    saveExecutionHistory();
                } else if (stage !== null && pendingExecutionHistory) {
                    executionHistory = pendingExecutionHistory;
                }
                if (stage !== null) {
                    activeTaskIndex = stage;
                    selectedTaskIndex = stage;
                    renderTimerTasks();
                    updateTimerDisplay();
                }
            }
            refreshTable();
        });

        api.addEventListener("execution_start", ({detail}) => {

            startRefreshTimer();
            beginExecutionHistory();
            showTimerDisplay();

            // Only save previous run data if a completed run exists
            if (runningData && runningData.total_execution_time != null) {
                lastRunningDate = runningData;
            }

            app.graph._nodes.forEach(function (node) {
                delete node.ty_et_start_time
                delete node.ty_et_execution_time
                delete node.ty_et_vram_used
            });
            runningData = {
                nodes_execution_time: [],
                total_execution_time: null
            };
        });

        api.addEventListener("TyDev-Utils.ExecutionTime.execution_end", ({detail}) => {
            stopRefreshTimer();
            freezeTimerDisplay();
            runningData.total_execution_time = detail.execution_time;
            const flowStage = runningData.flow_stage;
            const stage = flowStage?.stage ?? activeTaskIndex ?? 1;
            if (!flowStage?.stage) {
                executionHistory = [];
                activeTaskIndex = 1;
                selectedTaskIndex = 1;
            }
            const entry = {
                stage,
                total: flowStage?.total ?? null,
                execution_time: detail.execution_time,
                completed_at: Date.now(),
            };
            const existingIndex = executionHistory.findIndex((item) => item.stage === stage);
            if (existingIndex >= 0) {
                executionHistory[existingIndex] = entry;
            } else if (stage <= HISTORY_LIMIT) {
                executionHistory.push(entry);
            }
            executionHistory.sort((a, b) => a.stage - b.stage);
            saveExecutionHistory();
            pendingExecutionHistory = null;
            activeTaskIndex = null;
            selectedTaskIndex = stage;
            renderTimerTasks();
            updateTimerDisplay();
            refreshTable();
        })
    },
    async nodeCreated(node, app) {
        if (node.ty_et_swizzled) return;
        
        try {
            const orig = node.onDrawForeground;
            node.onDrawForeground = function (ctx) {
                return drawBadge(node, orig, arguments);
            };
            node.ty_et_swizzled = true;
        } catch (e) {
            console.warn('[ExecutionTime] Error setting up nodeCreated badge:', e);
        }
    },
    async loadedGraphNode(node, app) {
        if (node.ty_et_swizzled) return;
        
        try {
            const orig = node.onDrawForeground;
            node.onDrawForeground = function (ctx) {
                return drawBadge(node, orig, arguments);
            };
            node.ty_et_swizzled = true;
        } catch (e) {
            console.warn('[ExecutionTime] Error setting up loadedGraphNode badge:', e);
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply badge drawing to all node types at the prototype level
        try {
            const origOnDrawForeground = nodeType.prototype.onDrawForeground;
            if (origOnDrawForeground) {
                nodeType.prototype.onDrawForeground = function (ctx) {
                    return drawBadge(this, origOnDrawForeground, arguments);
                };
            }
        } catch (e) {
            console.warn('[ExecutionTime] Error applying beforeRegisterNodeDef badge:', e);
        }
        
        if (nodeType.comfyClass === "TY_ExecutionTime") {
            const originComputeSize = nodeType.prototype.computeSize || LGraphNode.prototype.computeSize;
            nodeType.prototype.computeSize = function () {
                const originSize = originComputeSize.apply(this, arguments);
                if (this.flags?.collapsed || !this.widgets) {
                    return originSize;
                }
                const tableWidget = this.widgets.find((w) => w.name === "Table");
                if (!tableWidget) {
                    return originSize;
                }
                const tableElem = tableWidget.inputEl.firstChild;
                const tableHeight = tableElem.getBoundingClientRect().height;
                const thHeight = tableElem.tHead.getBoundingClientRect().height;
                const thUnscaledHeight = 24;
                const tableUnscaledHeight = thUnscaledHeight * tableHeight / thHeight;
                const autoResizeMaxHeight = 300;
                return [Math.max(originSize[0], 600), originSize[1] + Math.min(tableUnscaledHeight, autoResizeMaxHeight) - LiteGraph.NODE_WIDGET_HEIGHT];
            }

            const nodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                nodeCreated?.apply(this, arguments);

                const tableWidget = {
                    type: "HTML",
                    name: "Table",
                    draw: function (ctx, node, widgetWidth, y, widgetHeight) {
                        const marginHorizontal = 14;
                        const marginTop = 0;
                        const marginBottom = 14;
                        const elRect = ctx.canvas.getBoundingClientRect();
                        const transform = new DOMMatrix()
                            .scaleSelf(elRect.width / ctx.canvas.width, elRect.height / ctx.canvas.height)
                            .multiplySelf(ctx.getTransform())
                            .translateSelf(marginHorizontal, marginTop + y);

                        const x = Math.max(0, Math.round(ctx.getTransform().a * (node.size[0] - this.inputEl.scrollWidth - 2 * marginHorizontal) / 2));
                        Object.assign(
                            this.inputEl.style,
                            {
                                transformOrigin: '0 0',
                                transform: transform,
                                left: `${x}px`,
                                top: `0px`,
                                position: "absolute",
                                width: `${widgetWidth - marginHorizontal * 2}px`,
                                height: `${node.size[1] - (marginTop + marginBottom) - y}px`,
                                overflow: `auto`,
                            }
                        );
                    },
                };
                tableWidget.inputEl = $el("div");

                document.body.appendChild(tableWidget.inputEl);

                this.addWidget("button", "Export CSV", "display: none", () => {
                    exportTable(tableWidget.inputEl.firstChild)
                });
                this.addCustomWidget(tableWidget);

                this.onRemoved = function () {
                    tableWidget.inputEl.remove();
                };
                this.serialize_widgets = false;
                this.isVirtualNode = true;

                const tableElem = buildTableHtml();

                tableWidget.inputEl.appendChild(tableElem)

                this.setSize(this.computeSize());
            }
        }
    }
});
