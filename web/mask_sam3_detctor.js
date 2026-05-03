import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let _sam3PointerDown = false;
let _sam3ResizeNode = null;
let _sam3PointerHooksInited = false;

const _sam3EnsurePointerHooks = () => {
  if (_sam3PointerHooksInited) return;
  _sam3PointerHooksInited = true;
  const pickResizeNodeFromMouse = () => {
    try {
      const c = app?.canvas;
      const m = c?.graph_mouse || c?.mouse;
      if (!Array.isArray(m) || m.length < 2) return;
      const sn = c?.selected_nodes;
      const nodes = Array.isArray(sn) ? sn : sn && typeof sn === "object" ? Object.values(sn) : [];
      for (const n of nodes) {
        if (!n || !Array.isArray(n.pos) || !Array.isArray(n.size)) continue;
        const x0 = n.pos[0],
          y0 = n.pos[1];
        const x1 = x0 + n.size[0],
          y1 = y0 + n.size[1];
        const mx = m[0],
          my = m[1];
        if (mx < x0 || mx > x1 || my < y0 || my > y1) continue;
        if (mx >= x1 - 30 && my >= y1 - 30) {
          _sam3ResizeNode = n;
          return;
        }
      }
      const n = c?.node_over;
      if (n && Array.isArray(n.pos) && Array.isArray(n.size)) {
        const mx = m[0],
          my = m[1];
        const x0 = n.pos[0],
          y0 = n.pos[1];
        const x1 = x0 + n.size[0],
          y1 = y0 + n.size[1];
        if (mx >= x1 - 30 && my >= y1 - 30) _sam3ResizeNode = n;
      }
    } catch (e) {}
  };
  const onDown = () => {
    _sam3PointerDown = true;
    _sam3ResizeNode = null;
    pickResizeNodeFromMouse();
  };
  const onMove = () => {
    if (!_sam3PointerDown) return;
    pickResizeNodeFromMouse();
  };
  const onUp = () => {
    _sam3PointerDown = false;
    _sam3ResizeNode = null;
  };
  window.addEventListener("pointerdown", onDown, true);
  window.addEventListener("pointermove", onMove, true);
  window.addEventListener("pointerup", onUp, true);
  window.addEventListener("pointercancel", onUp, true);
  window.addEventListener("mousedown", onDown, true);
  window.addEventListener("mousemove", onMove, true);
  window.addEventListener("mouseup", onUp, true);
  window.addEventListener("blur", onUp, true);
};

function findWidget(node, name) {
  return node?.widgets?.find((w) => w?.name === name) || null;
}

function hideWidget(widget) {
  if (!widget) return;
  widget.type = "converted-widget";
  widget.computeSize = () => [0, -4];
  if (widget.inputEl) widget.inputEl.style.display = "none";
  if (widget.element) widget.element.style.display = "none";
}

function getLinkById(graph, linkId) {
  if (!graph?.links || !linkId) return null;
  if (typeof graph.links.get === "function") return graph.links.get(linkId);
  return graph.links[linkId] || null;
}

app.registerExtension({
  name: "AptPreset.MaskSam3DetctorUI",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    _sam3EnsurePointerHooks();
    if (nodeData.name !== "mask_sam_detctor") return;
    const MAX_WIDGET_HEIGHT = 1200;
    const MIN_WIDGET_HEIGHT = 140;
    const MIN_NODE_WIDTH = 260;
    const MIN_NODE_HEIGHT = 360;

    const oldCreated = nodeType.prototype.onNodeCreated;
      nodeType.prototype.onNodeCreated = function () {
        oldCreated?.apply(this, arguments);
  
        if (!this.size || !Array.isArray(this.size)) {
          this.size = [420, 480];
        }
        if (this.size[0] < MIN_NODE_WIDTH) this.size[0] = MIN_NODE_WIDTH;
        if (this.size[1] < MIN_NODE_HEIGHT) this.size[1] = MIN_NODE_HEIGHT;
        this.resizable = true;
        this.min_size = [MIN_NODE_WIDTH, MIN_NODE_HEIGHT];
        this._apt_target_size = this.size;
        const _apt_orig_onResize = this.onResize;
        this.onResize = function (size) {
          this._apt_target_size = size;
          return _apt_orig_onResize?.apply(this, arguments);
        };

      this._samUi = {
        mode: "point",
        posPoints: [],
        negPoints: [],
        bboxes: [],
        pointHistory: [],
        boxHistory: [],
        activeBoxIndex: -1,
        boxDrag: null,
        imageObj: null,
        imageSrc: "",
      };

      const positiveWidget = findWidget(this, "ui_positive_coords");
      const negativeWidget = findWidget(this, "ui_negative_coords");
      const bboxesJsonWidget = findWidget(this, "ui_bboxes_json");
      hideWidget(positiveWidget);
      hideWidget(negativeWidget);
      hideWidget(bboxesJsonWidget);

      const container = document.createElement("div");
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.gap = "6px";
      container.style.width = "100%";
      container.style.height = "100%";
      container.style.overflow = "hidden";

      const toolbar = document.createElement("div");
      toolbar.style.display = "flex";
      toolbar.style.gap = "6px";
      toolbar.style.alignItems = "center";
      toolbar.style.flexWrap = "nowrap";
      toolbar.style.overflow = "hidden";

      const mkBtn = (label, bg = "#2a2a2a") => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.style.padding = "4px 10px";
        btn.style.border = "1px solid #4d4d4d";
        btn.style.borderRadius = "6px";
        btn.style.background = bg;
        btn.style.color = "#ddd";
        btn.style.cursor = "pointer";
        btn.style.height = "28px";
        return btn;
      };

      const previewBtn = mkBtn("Preview", "#334155");
      previewBtn.style.width = "100%";
      previewBtn.style.height = "36px";
      previewBtn.style.borderRadius = "10px";
      previewBtn.style.fontWeight = "700";
      previewBtn.style.marginTop = "2px";
      const modeBtn = mkBtn("点模式", "#14532d");
      const addBoxBtn = mkBtn("添加框", "#4b5563");
      const undoBtn = mkBtn("后退", "#3f3f46");
      const clearBtn = mkBtn("清空", "#7f1d1d");
      const hint = document.createElement("span");
      hint.style.color = "#9ca3af";
      hint.style.fontSize = "12px";
      hint.style.whiteSpace = "nowrap";
      hint.textContent = "点模式: 左蓝右红；框模式: 选中后拖动平移/缩放";

      toolbar.appendChild(modeBtn);
      toolbar.appendChild(addBoxBtn);
      toolbar.appendChild(undoBtn);
      toolbar.appendChild(clearBtn);
      toolbar.appendChild(hint);

      const pointInputRow = document.createElement("div");
      pointInputRow.style.display = "flex";
      pointInputRow.style.gap = "6px";
      pointInputRow.style.alignItems = "stretch";

      const mkTextArea = (placeholder) => {
        const input = document.createElement("textarea");
        input.placeholder = placeholder;
        input.style.flex = "1";
        input.style.height = "42px";
        input.style.minHeight = "42px";
        input.style.maxHeight = "42px";
        input.style.resize = "none";
        input.style.border = "1px solid #3f3f46";
        input.style.borderRadius = "6px";
        input.style.background = "#111827";
        input.style.color = "#d1d5db";
        input.style.padding = "4px 6px";
        input.style.fontSize = "12px";
        return input;
      };

      const negInput = mkTextArea("红点(右键): x,y; x,y");
      const posInput = mkTextArea("蓝点(左键): x,y; x,y");
      pointInputRow.appendChild(negInput);
      pointInputRow.appendChild(posInput);

      const canvasWrap = document.createElement("div");
      canvasWrap.style.flex = "1";
      canvasWrap.style.minHeight = "180px";
      canvasWrap.style.border = "1px solid #333";
      canvasWrap.style.borderRadius = "8px";
      canvasWrap.style.background = "#111827";
      canvasWrap.style.overflow = "hidden";
      canvasWrap.style.position = "relative";

      const canvas = document.createElement("canvas");
      canvas.style.width = "100%";
      canvas.style.height = "100%";
      canvas.style.display = "block";
      canvasWrap.appendChild(canvas);
      container.appendChild(toolbar);
      container.appendChild(pointInputRow);
      container.appendChild(canvasWrap);
      container.appendChild(previewBtn);

      const uiWidget = this.addDOMWidget("mask_sam3_detctor_ui", "customwidget", container, {
        serialize: false,
        hideOnZoom: false,
      });
      const nodeInstance = this;
      const UI_DEFAULT_HEIGHT = 300;
      uiWidget.computeSize = function (width) {
        return [width, UI_DEFAULT_HEIGHT];
      };
      const UI_BASE_HEIGHT = typeof nodeInstance.computeSize === "function" ? nodeInstance.computeSize()[1] : nodeInstance.size[1];
      const getTargetNodeHeight = () => {
        const c = app?.canvas;
        const m = c?.graph_mouse || c?.mouse;
        if (_sam3PointerDown && _sam3ResizeNode && (_sam3ResizeNode === nodeInstance || _sam3ResizeNode?.id === nodeInstance.id) && Array.isArray(m) && m.length > 1 && Array.isArray(nodeInstance.pos)) {
            return Math.max(MIN_NODE_HEIGHT, m[1] - nodeInstance.pos[1]);
        }
        return nodeInstance.size[1];
      };
      uiWidget.computeSize = function (w) {
        const targetH = getTargetNodeHeight();
        const delta = Math.max(0, (targetH - UI_BASE_HEIGHT) * 0.95);
        const wanted = UI_DEFAULT_HEIGHT + delta;
        const h = Math.max(MIN_WIDGET_HEIGHT, Math.min(MAX_WIDGET_HEIGHT, Math.round(wanted)));
        return [w || nodeInstance.size[0], h];
      };

      const parsePointsText = (text, iw, ih) => {
        const src = String(text || "").trim();
        if (!src) return [];
        const clampXY = (x, y) => ({
          x: Math.max(0, Math.min(iw, Math.round(Number(x) || 0))),
          y: Math.max(0, Math.min(ih, Math.round(Number(y) || 0))),
        });
        try {
          const parsed = JSON.parse(src);
          if (Array.isArray(parsed)) {
            const out = [];
            for (const item of parsed) {
              if (Array.isArray(item) && item.length >= 2) out.push(clampXY(item[0], item[1]));
              else if (item && typeof item === "object") out.push(clampXY(item.x, item.y));
            }
            return out;
          }
        } catch (_) {}

        return src
          .split(/[\n;]+/)
          .map((line) => line.trim())
          .filter(Boolean)
          .map((line) => {
            const parts = line.split(/[,\s]+/).filter(Boolean);
            if (parts.length < 2) return null;
            return clampXY(parts[0], parts[1]);
          })
          .filter(Boolean);
      };

      const formatPointsText = (points) => points.map((p) => `${Math.round(p.x)},${Math.round(p.y)}`).join("; ");

      const clampNum = (v, min, max) => Math.max(min, Math.min(max, v));

      const normalizeBoxes = (boxes, iw, ih) =>
        (boxes || []).map((b) => {
          const x = clampNum(Math.round(Number(b.x) || 0), 0, Math.max(0, iw - 1));
          const y = clampNum(Math.round(Number(b.y) || 0), 0, Math.max(0, ih - 1));
          const maxW = Math.max(1, iw - x);
          const maxH = Math.max(1, ih - y);
          const width = clampNum(Math.round(Number(b.width) || 1), 1, maxW);
          const height = clampNum(Math.round(Number(b.height) || 1), 1, maxH);
          return { x, y, width, height };
        });

      const pushPointHistory = () => {
        const s = this._samUi;
        const snap = {
          pos: s.posPoints.map((p) => ({ ...p })),
          neg: s.negPoints.map((p) => ({ ...p })),
        };
        s.pointHistory.push(snap);
        if (s.pointHistory.length > 5) s.pointHistory.shift();
      };

      const pushBoxHistory = () => {
        const s = this._samUi;
        const snap = {
          bboxes: s.bboxes.map((b) => ({ ...b })),
          activeBoxIndex: s.activeBoxIndex,
        };
        s.boxHistory.push(snap);
        if (s.boxHistory.length > 5) s.boxHistory.shift();
      };

      let syncingPointInputs = false;
      const syncWidgets = () => {
        const s = this._samUi;
        const posText = JSON.stringify(s.posPoints.map((p) => ({ x: p.x, y: p.y })));
        const negText = JSON.stringify(s.negPoints.map((p) => ({ x: p.x, y: p.y })));
        const iw = s.imageObj?.naturalWidth || s.imageObj?.width || 0;
        const ih = s.imageObj?.naturalHeight || s.imageObj?.height || 0;
        const normBoxes = iw > 0 && ih > 0 ? normalizeBoxes(s.bboxes, iw, ih) : s.bboxes;
        const boxText = normBoxes.length > 0 ? JSON.stringify(normBoxes) : "";
        
        // 如果是从后端回传进来的外接框，不要去覆盖 widget value（防止弄脏工作流状态），保持它受外部输入控制
        // 只同步手画的框
        if (!isBboxesLinked() && bboxesJsonWidget) {
            bboxesJsonWidget.value = boxText;
        }
        // 关键：左键打蓝点 -> negPoints -> ui_positive_coords
        //       右键打红点 -> posPoints -> ui_negative_coords
        // 按你的最新要求，这里交叉映射赋值给 widget
        if (positiveWidget) positiveWidget.value = negText;
        if (negativeWidget) negativeWidget.value = posText;
        
        syncingPointInputs = true;
        
        // 同步给输入框时也要交叉映射
        posInput.value = formatPointsText(s.negPoints);
        negInput.value = formatPointsText(s.posPoints);
        
        syncingPointInputs = false;
        this.graph?.change?.();
      };

      const applyPointInputs = () => {
        if (syncingPointInputs || this._samUi.mode !== "point") return;
        const img = this._samUi.imageObj;
        const iw = img?.naturalWidth || img?.width || 0;
        const ih = img?.naturalHeight || img?.height || 0;
        if (!iw || !ih) return;
        
        // 反向映射：从 ui_positive_coords 框读出来的其实是蓝点（negPoints）
        this._samUi.negPoints = parsePointsText(posInput.value, iw, ih);
        this._samUi.posPoints = parsePointsText(negInput.value, iw, ih);
        syncWidgets();
        draw();
      };

      const getImageRect = () => {
        const s = this._samUi;
        if (!s.imageObj) return null;
        
        const PADDING = 10;
        const cw = Math.max(1, canvas.width - PADDING * 2);
        const ch = Math.max(1, canvas.height - PADDING * 2);
        
        const iw = s.imageObj.naturalWidth || s.imageObj.width || 1;
        const ih = s.imageObj.naturalHeight || s.imageObj.height || 1;
        const ir = iw / ih;
        const cr = cw / ch;
        let w = cw;
        let h = ch;
        let x = 0;
        let y = 0;
        if (ir > cr) {
          h = Math.round(cw / ir);
          y = Math.round((ch - h) / 2);
        } else {
          w = Math.round(ch * ir);
          x = Math.round((cw - w) / 2);
        }
        
        x += PADDING;
        y += PADDING;
        
        return { x, y, w, h, iw, ih };
      };

      const toImageXY = (evt, allowOutside = false) => {
        const rect = canvas.getBoundingClientRect();
        const cx = ((evt.clientX - rect.left) / rect.width) * canvas.width;
        const cy = ((evt.clientY - rect.top) / rect.height) * canvas.height;
        const imgRect = getImageRect();
        if (!imgRect) return null;
        if (!allowOutside && (cx < imgRect.x || cy < imgRect.y || cx > imgRect.x + imgRect.w || cy > imgRect.y + imgRect.h)) return null;
        const nx = (cx - imgRect.x) / imgRect.w;
        const ny = (cy - imgRect.y) / imgRect.h;
        if (allowOutside) {
            return {
              x: nx * imgRect.iw,
              y: ny * imgRect.ih,
            };
        }
        return {
          x: Math.max(0, Math.min(imgRect.iw, nx * imgRect.iw)),
          y: Math.max(0, Math.min(imgRect.ih, ny * imgRect.ih)),
        };
      };

      const draw = () => {
        const s = this._samUi;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        canvas.width = Math.max(1, Math.floor(canvasWrap.clientWidth));
        canvas.height = Math.max(1, Math.floor(canvasWrap.clientHeight));
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const imgRect = getImageRect();
        if (s.imageObj && imgRect) {
          ctx.drawImage(s.imageObj, imgRect.x, imgRect.y, imgRect.w, imgRect.h);
        } else {
          ctx.fillStyle = "#6b7280";
          ctx.font = "13px sans-serif";
          ctx.textAlign = "center";
          ctx.fillText("等待 image 输入", canvas.width / 2, canvas.height / 2);
          return;
        }

        const drawBox = (b, color, active = false) => {
          const x = imgRect.x + (b.x / imgRect.iw) * imgRect.w;
          const y = imgRect.y + (b.y / imgRect.ih) * imgRect.h;
          const w = (b.width / imgRect.iw) * imgRect.w;
          const h = (b.height / imgRect.ih) * imgRect.h;
          ctx.strokeStyle = color;
          ctx.lineWidth = active ? 2.5 : 2;
          ctx.strokeRect(x, y, Math.max(1, w), Math.max(1, h));
          if (active && !isBboxesLinked()) {
            const hs = 4;
            const handles = [
              [x, y],
              [x + w, y],
              [x + w, y + h],
              [x, y + h],
            ];
            ctx.fillStyle = "#22d3ee";
            handles.forEach(([hx, hy]) => ctx.fillRect(hx - hs, hy - hs, hs * 2, hs * 2));
          }
        };

        s.bboxes.forEach((b, idx) => drawBox(b, "#f59e0b", idx === s.activeBoxIndex));

        const drawPoint = (p, color) => {
          const x = imgRect.x + (p.x / imgRect.iw) * imgRect.w;
          const y = imgRect.y + (p.y / imgRect.ih) * imgRect.h;
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, Math.PI * 2);
          ctx.fillStyle = color;
          ctx.fill();
          ctx.lineWidth = 1;
          ctx.strokeStyle = "#ffffff";
          ctx.stroke();
        };
        s.posPoints.forEach((p) => drawPoint(p, "#ef4444"));
        s.negPoints.forEach((p) => drawPoint(p, "#3b82f6"));
      };

      const hitTestBox = (p, imgRect) => {
        const hs = 8;
        const handles = (b) => {
          const x = imgRect.x + (b.x / imgRect.iw) * imgRect.w;
          const y = imgRect.y + (b.y / imgRect.ih) * imgRect.h;
          const w = (b.width / imgRect.iw) * imgRect.w;
          const h = (b.height / imgRect.ih) * imgRect.h;
          return {
            nw: { x, y },
            ne: { x: x + w, y },
            se: { x: x + w, y: y + h },
            sw: { x, y: y + h },
            body: { x, y, w, h },
          };
        };
        const rect = canvas.getBoundingClientRect();
        const cx = ((p.clientX - rect.left) / rect.width) * canvas.width;
        const cy = ((p.clientY - rect.top) / rect.height) * canvas.height;
        for (let i = this._samUi.bboxes.length - 1; i >= 0; i--) {
          const h = handles(this._samUi.bboxes[i]);
          for (const key of ["nw", "ne", "se", "sw"]) {
            if (Math.abs(cx - h[key].x) <= hs && Math.abs(cy - h[key].y) <= hs) return { index: i, kind: key };
          }
          if (cx >= h.body.x && cx <= h.body.x + h.body.w && cy >= h.body.y && cy <= h.body.y + h.body.h) {
            return { index: i, kind: "move" };
          }
        }
        return null;
      };

      const runPreview = async () => {
        try {
            const p = await app.graphToPrompt();
            const prompt = p.output;
            const selectedNodeId = String(this.id);
            const isolatedPrompt = {};
            const traceDependencies = (nodeId) => {
                if (!prompt[nodeId] || isolatedPrompt[nodeId]) return;
                isolatedPrompt[nodeId] = prompt[nodeId];
                const inputs = prompt[nodeId].inputs;
                for (let key in inputs) {
                    const val = inputs[key];
                    if (Array.isArray(val) && val.length === 2) traceDependencies(String(val[0]));
                }
            };
            traceDependencies(selectedNodeId);
            
            if (Object.keys(isolatedPrompt).length > 0) {
                await api.fetchApi("/prompt", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        client_id: api.clientId,
                        prompt: isolatedPrompt,
                        extra_data: p.workflow ? { extra_pnginfo: { workflow: p.workflow } } : {}
                    })
                });
            }
        } catch(e) {
            console.warn("Failed to queue prompt for preview", e);
        }
      };

      const ro = new ResizeObserver(() => draw());
      ro.observe(canvasWrap);
      this._samUiResizeObserver = ro;

      const isBboxesLinked = () => this.inputs?.find((i) => i.name === "permil_str")?.link != null;

      const updateModeUI = () => {
        const pointMode = this._samUi.mode === "point";
        modeBtn.textContent = pointMode ? "点模式" : "框模式";
        modeBtn.style.background = pointMode ? "#14532d" : "#1f2937";
        
        const linked = isBboxesLinked();
        if (!pointMode && linked) {
            addBoxBtn.style.display = "inline-block";
            addBoxBtn.style.opacity = "0.5";
            addBoxBtn.style.pointerEvents = "none";
            addBoxBtn.title = "外部已连接 bbox，当前为只读模式";
            undoBtn.style.opacity = "0.5";
            undoBtn.style.pointerEvents = "none";
            clearBtn.style.opacity = "0.5";
            clearBtn.style.pointerEvents = "none";
        } else {
            addBoxBtn.style.display = pointMode ? "none" : "inline-block";
            addBoxBtn.style.opacity = "1";
            addBoxBtn.style.pointerEvents = "auto";
            addBoxBtn.title = "";
            undoBtn.style.opacity = "1";
            undoBtn.style.pointerEvents = "auto";
            clearBtn.style.opacity = "1";
            clearBtn.style.pointerEvents = "auto";
        }
        pointInputRow.style.display = pointMode ? "flex" : "none";
      };

      previewBtn.onclick = () => runPreview();

      modeBtn.onclick = () => {
        this._samUi.mode = this._samUi.mode === "point" ? "box" : "point";
        updateModeUI();
        if (this._samUi.mode === "point") applyPointInputs();
        draw();
      };

      addBoxBtn.onclick = () => {
        const s = this._samUi;
        const iw = s.imageObj?.naturalWidth || s.imageObj?.width || 0;
        const ih = s.imageObj?.naturalHeight || s.imageObj?.height || 0;
        if (!iw || !ih) return;
        pushBoxHistory();
        const width = Math.max(1, Math.round(iw * 0.5));
        const height = Math.max(1, Math.round(ih * 0.5));
        const x = Math.max(0, Math.round((iw - width) * 0.5));
        const y = Math.max(0, Math.round((ih - height) * 0.5));
        s.bboxes.push({ x, y, width, height });
        s.activeBoxIndex = s.bboxes.length - 1;
        syncWidgets();
        draw();
      };

      undoBtn.onclick = () => {
        const s = this._samUi;
        if (s.mode === "point") {
          const last = s.pointHistory.pop();
          if (!last) return;
          s.posPoints = last.pos.map((p) => ({ ...p }));
          s.negPoints = last.neg.map((p) => ({ ...p }));
        } else {
          const last = s.boxHistory.pop();
          if (!last) return;
          s.bboxes = (last.bboxes || []).map((b) => ({ ...b }));
          s.activeBoxIndex = Number.isInteger(last.activeBoxIndex) ? last.activeBoxIndex : -1;
        }
        syncWidgets();
        draw();
      };

      clearBtn.onclick = () => {
        const s = this._samUi;
        if (s.mode === "point") {
          pushPointHistory();
          s.posPoints = [];
          s.negPoints = [];
        } else {
          pushBoxHistory();
          s.bboxes = [];
          s.activeBoxIndex = -1;
          s.boxDrag = null;
        }
        syncWidgets();
        draw();
      };

      negInput.addEventListener("input", applyPointInputs);
      posInput.addEventListener("input", applyPointInputs);

      canvas.addEventListener("contextmenu", (e) => e.preventDefault());
      canvas.addEventListener("mousedown", (e) => {
        const s = this._samUi;
        const p = toImageXY(e);
        if (!p) return;
        if (s.mode === "point") {
          pushPointHistory();
          // e.button === 0 是左键，要求蓝点代表 negative
          if (e.button === 0) s.negPoints.push(p);
          // e.button === 2 是右键，要求红点代表 positive
          else if (e.button === 2) s.posPoints.push(p);
          syncWidgets();
          draw();
          return;
        }
        if (e.button !== 0) return;
        if (isBboxesLinked()) return; // 外部连接了 bboxes 时禁止编辑
        const imgRect = getImageRect();
        if (!imgRect) return;
        const hit = hitTestBox(e, imgRect);
        if (!hit) {
          s.activeBoxIndex = -1;
          draw();
          return;
        }
        s.activeBoxIndex = hit.index;
        pushBoxHistory();
        s.boxDrag = {
          index: hit.index,
          kind: hit.kind,
          start: { x: p.x, y: p.y },
          box: { ...s.bboxes[hit.index] },
        };
        draw();
      });

      canvas.addEventListener("mousemove", (e) => {
        const s = this._samUi;
        if (s.mode !== "box" || !s.boxDrag) return;
        const p = toImageXY(e, true); // 拖动框时允许鼠标移出边界
        if (!p) return;
        const iw = s.imageObj?.naturalWidth || s.imageObj?.width || 0;
        const ih = s.imageObj?.naturalHeight || s.imageObj?.height || 0;
        if (!iw || !ih) return;
        const { index, kind, start, box } = s.boxDrag;
        const dx = p.x - start.x;
        const dy = p.y - start.y;
        let x = box.x;
        let y = box.y;
        let right = box.x + box.width;
        let bottom = box.y + box.height;
        if (kind === "move") {
          s.bboxes[index] = { x: Math.round(box.x + dx), y: Math.round(box.y + dy), width: box.width, height: box.height };
          syncWidgets();
          draw();
          return;
        }
        if (kind === "nw" || kind === "sw") x = box.x + dx;
        if (kind === "ne" || kind === "se") right = box.x + box.width + dx;
        if (kind === "nw" || kind === "ne") y = box.y + dy;
        if (kind === "sw" || kind === "se") bottom = box.y + box.height + dy;
        s.bboxes[index] = {
          x: Math.round(x),
          y: Math.round(y),
          width: Math.max(1, Math.round(right - x)),
          height: Math.max(1, Math.round(bottom - y)),
        };
        syncWidgets();
        draw();
      });

      const onMouseUp = () => {
        const s = this._samUi;
        if (!s.boxDrag) return;
        s.boxDrag = null;
        syncWidgets();
        draw();
      };
      window.addEventListener("mouseup", onMouseUp);
      this._samUiMouseUpHandler = onMouseUp;

      this._samUiForceDraw = () => {
          updateModeUI();
          applyPointInputs();
          syncWidgets();
          draw();
      };

      updateModeUI();
      draw();
    };

    const oldExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        oldExecuted?.apply(this, arguments);
        
        let needsDraw = false;

        // 如果后端传回了强制同步的外部 bbox
        if (message?.external_bboxes && this._samUi) {
            this._samUi.bboxes = message.external_bboxes;
            this._samUi.mode = "box"; // 自动切换到框模式
            
            // 找到模式按钮并更新
            const modeBtn = this.widgets.find(w => w.name === "ui_bboxes_json")?.inputEl?.parentElement?.querySelector('button');
            if (modeBtn) {
                 modeBtn.textContent = "框模式";
                 modeBtn.style.background = "#1f2937";
                 const addBoxBtn = modeBtn.nextElementSibling;
                 if (addBoxBtn) addBoxBtn.style.display = "inline-block";
                 const pointInputRow = modeBtn.parentElement?.nextElementSibling;
                 if (pointInputRow) pointInputRow.style.display = "none";
            }
            needsDraw = true;
        }

        if (message?.bg_image?.length > 0) {
            const imgInfo = message.bg_image[0];
            const url = api.apiURL(`/view?filename=${encodeURIComponent(imgInfo.filename)}&type=${imgInfo.type}&subfolder=${imgInfo.subfolder}&t=${Date.now()}`);
            
            if (this._samUi && url !== this._samUi.imageSrc) {
                const img = new Image();
                img.onload = () => {
                    this._samUi.imageObj = img;
                    this._samUi.imageSrc = url;
                    
                    if (this._samUiForceDraw) this._samUiForceDraw();
                };
                img.src = url;
                needsDraw = false; // will draw on load
            }
        }

        if (needsDraw && this._samUiForceDraw) {
            this._samUiForceDraw();
        }
    };

    const oldOnConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
        oldOnConnectionsChange?.apply(this, arguments);
        if (type === 1 && this.inputs[index]?.name === "permil_str") {
            if (connected && this._samUi) {
                this._samUi.mode = "box";
            }
            if (this._samUiForceDraw) this._samUiForceDraw();
        }
    };

    const oldRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this._samUiResizeObserver) {
        this._samUiResizeObserver.disconnect();
        this._samUiResizeObserver = null;
      }
      if (this._samUiMouseUpHandler) {
        window.removeEventListener("mouseup", this._samUiMouseUpHandler);
        this._samUiMouseUpHandler = null;
      }
      oldRemoved?.apply(this, arguments);
    };
  },
});

