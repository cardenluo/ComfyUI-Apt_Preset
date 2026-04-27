import { app } from "../../../scripts/app.js";

const DOU_BAO_DOWNLOAD_DIR_KEY = "aptpreset_doubao_download_dir";
const DOU_BAO_URL = "https://www.doubao.com/";
const SAVE_IMAGE_API = "/apt_preset/doubao_capture/save_remote_image";

async function requestJson(url, method = "GET", payload = null) {
  const options = { method };
  if (payload !== null) {
    options.headers = { "Content-Type": "application/json" };
    options.body = JSON.stringify(payload);
  }
  const resp = await fetch(url, options);
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok || data?.ok === false) {
    throw new Error(data?.message || data?.error || `HTTP ${resp.status}`);
  }
  return data;
}

function isDoubaoOrigin(origin) {
  return typeof origin === "string" && /https:\/\/([a-z0-9-]+\.)?doubao\.com/i.test(origin);
}

function extractNoWatermarkUrls(payload) {
  const list = Array.isArray(payload) ? payload : [];
  const urls = [];
  for (const item of list) {
    const url = (item?.no_watermark_url || "").toString().trim();
    if (url.startsWith("http")) {
      urls.push(url);
    }
  }
  return urls;
}

function bindDirPersistence(widget) {
  if (!widget || widget.__aptPersistBound) return;
  widget.__aptPersistBound = true;
  const originalCallback = widget.callback;
  widget.callback = function (value) {
    originalCallback?.apply(this, arguments);
    try {
      const dir = (value ?? "").toString().trim();
      if (dir) {
        window.localStorage.setItem(DOU_BAO_DOWNLOAD_DIR_KEY, dir);
      } else {
        window.localStorage.removeItem(DOU_BAO_DOWNLOAD_DIR_KEY);
      }
    } catch (error) {
      console.warn("保存下载路径失败:", error);
    }
  };
}

app.registerExtension({
  name: "AptPreset.DoubaoWebPreviewadv",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AI_DoubaoWebPreview") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const downloadDirWidget = this.widgets?.find((w) => w.name === "download_dir");

      const container = document.createElement("div");
      container.style.width = "100%";
      container.style.height = "900px";
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.background = "#151515";
      container.style.border = "1px solid #333";
      container.style.borderRadius = "8px";
      container.style.overflow = "hidden";

      const topBar = document.createElement("div");
      topBar.style.display = "flex";
      topBar.style.flexWrap = "nowrap";
      topBar.style.gap = "8px";
      topBar.style.padding = "8px";
      topBar.style.background = "#1f1f1f";
      topBar.style.borderBottom = "1px solid #333";
      topBar.style.alignItems = "center";

      const status = document.createElement("span");
      status.style.color = "#9aa0a6";
      status.style.fontSize = "12px";
      status.textContent = "";

      const captureInfo = document.createElement("span");
      captureInfo.style.color = "#66d9ef";
      captureInfo.style.fontSize = "12px";
      captureInfo.textContent = "状态：未启动";

      let captureEnabled = false;

      const refreshCaptureStatus = () => {
        captureInfo.textContent = captureEnabled ? `状态：运行中，已保存${saveCounter}张` : "状态：已停止";
      };

      const createButton = (label, onClick, bg = "#2a2a2a") => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.textContent = label;
        btn.style.padding = "4px 10px";
        btn.style.border = "1px solid #4d4d4d";
        btn.style.borderRadius = "6px";
        btn.style.background = bg;
        btn.style.color = "#ddd";
        btn.style.cursor = "pointer";
        btn.onclick = onClick;
        return btn;
      };

      const startBtn = createButton("启动", () => {
        saveCounter = 0;
        captureEnabled = true;
        refreshCaptureStatus();
      }, "#1f6f43");

      const stopBtn = createButton("停止", () => {
        captureEnabled = false;
        refreshCaptureStatus();
      }, "#7a2f2f");

      // 读取上次保存的下载路径，避免每次重新输入
      try {
        const savedDir = window.localStorage.getItem(DOU_BAO_DOWNLOAD_DIR_KEY) || "";
        if (savedDir && downloadDirWidget && !downloadDirWidget.value) {
          downloadDirWidget.value = savedDir;
        }
      } catch (error) {
        console.warn("读取下载路径失败:", error);
      }
      bindDirPersistence(downloadDirWidget);

      topBar.appendChild(startBtn);
      topBar.appendChild(stopBtn);
      topBar.appendChild(status);
      topBar.appendChild(captureInfo);

      const iframe = document.createElement("iframe");
      iframe.src = DOU_BAO_URL;
      iframe.style.flex = "1";
      iframe.style.width = "100%";
      iframe.style.border = "none";
      iframe.referrerPolicy = "no-referrer";

      container.appendChild(topBar);
      container.appendChild(iframe);

      const domWidget = this.addDOMWidget("doubao_web_ui", "customwidget", container, {
        serialize: false,
        hideOnZoom: false,
      });
      let saveCounter = 0;
      refreshCaptureStatus();
      const onWindowMessage = async (event) => {
        if (!iframe.contentWindow || event.source !== iframe.contentWindow) {
          return;
        }
        if (!isDoubaoOrigin(event.origin)) {
          return;
        }
        const msgType = event?.data?.type;
        const isBridgeType = msgType === "aptpreset_doubao_image_data";
        if (!isBridgeType) {
          return;
        }
        if (!captureEnabled) {
          return;
        }
        const urls = extractNoWatermarkUrls(event?.data?.data || []);
        if (!urls.length) {
          return;
        }

        for (const imageUrl of urls) {
          try {
            const targetDir = (downloadDirWidget?.value ?? "").toString().trim();
            const data = await requestJson(SAVE_IMAGE_API, "POST", {
              url: imageUrl,
              download_dir: targetDir,
            });
            saveCounter += 1;
            refreshCaptureStatus();
          } catch (error) {
            console.error("保存无水印图失败:", error);
          }
        }
      };
      window.addEventListener("message", onWindowMessage);

      const oldRemoved = this.onRemoved;
      this.onRemoved = function () {
        window.removeEventListener("message", onWindowMessage);
        oldRemoved?.apply(this, arguments);
      };
      this.size = [1200, 1000];
      domWidget.computeSize = (width) => [width || this.size[0], 900];
    };
  },
});
