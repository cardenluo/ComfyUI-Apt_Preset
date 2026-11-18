import { app } from "../../../scripts/app.js";

const DOU_BAO_DOWNLOAD_DIR_KEY = "aptpreset_doubao_download_dir";
const DOU_BAO_PAGE_URL_KEY = "aptpreset_doubao_page_url";
const DOU_BAO_FAVORITES_KEY = "aptpreset_doubao_favorites";
const DEFAULT_PREVIEW_URL = "https://www.tugaigai.com/online_ps/";
const SAVE_IMAGE_API = "/apt_preset/doubao_capture/save_remote_image";
const TASK_EVENT_API = "/apt_preset/doubao_capture/report_task_event";
const DOWNLOAD_PROBE_API = "/apt_preset/doubao_capture/download_probe";
const LOCAL_PREVIEW_REGISTER_API = "/apt_preset/local_preview/register";
const AUTO_REMOTE_DOWNLOAD_ENABLED = false;
const DEFAULT_FAVORITES = {
  "在线豆包": "https://www.doubao.com/",
  "在线PS工具": "https://www.tugaigai.com/online_ps/",
  "lovart画布": "https://www.lovart.ai/canvas",
  "pico在线图像编辑": "https://picoimage.com/zh",
  "pixel全能工具箱": "https://www.ppimage.com",
  "在线工具": "https://youronlinetools.com/zh",
  "leaderai词库": "https://www.leaderai.top/mid-api/lab/image_prompt/index.html",
};

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

function bindStringWidgetPersistence(widget, storageKey) {
  if (!widget || widget.__aptPersistBound) return;
  widget.__aptPersistBound = true;
  const originalCallback = widget.callback;
  widget.callback = function (value) {
    originalCallback?.apply(this, arguments);
    try {
      const input = (value ?? "").toString().trim();
      if (input) {
        window.localStorage.setItem(storageKey, input);
      } else {
        window.localStorage.removeItem(storageKey);
      }
    } catch (error) {
      console.warn("保存节点输入失败:", error);
    }
  };
}

function normalizePageUrl(inputUrl) {
  const value = (inputUrl ?? "").toString().trim();
  if (!value) return DEFAULT_PREVIEW_URL;
  // 兼容历史误存: https://F:\xx\index.html 或 https://F:/xx/index.html
  const malformedHttpDrive = value.match(/^https?:\/\/([a-zA-Z]:[\\/].*)$/i);
  if (malformedHttpDrive) {
    return `file:///${malformedHttpDrive[1].replace(/\\/g, "/")}`;
  }
  if (/^file:\/\//i.test(value)) return value;
  if (/^https?:\/\//i.test(value)) return value;
  // Windows 本地绝对路径: F:\xx\index.html -> file:///F:/xx/index.html
  if (/^[a-zA-Z]:[\\/]/.test(value)) {
    return `file:///${value.replace(/\\/g, "/")}`;
  }
  // 兼容 file://F:/xx 这类少斜杠写法
  if (/^file:[\\/]/i.test(value)) {
    const pathPart = value.replace(/^file:[\\/]+/i, "");
    return `file:///${pathPart.replace(/\\/g, "/")}`;
  }
  return `https://${value}`;
}

function isLocalPageLike(url) {
  const value = (url || "").toString().trim();
  if (!value) return false;
  if (/^file:\/\//i.test(value)) return true;
  if (/^[a-zA-Z]:[\\/]/.test(value)) return true;
  if (/^https?:\/\/[a-zA-Z]:[\\/]/i.test(value)) return true;
  return false;
}

function loadFavorites() {
  const initial = { ...DEFAULT_FAVORITES };
  try {
    const raw = window.localStorage.getItem(DOU_BAO_FAVORITES_KEY);
    if (!raw) return initial;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return initial;
    for (const [name, url] of Object.entries(parsed)) {
      const key = (name ?? "").toString().trim();
      const value = (url ?? "").toString().trim();
      if (!key || !value) continue;
      initial[key] = normalizePageUrl(value);
    }
  } catch (error) {
    console.warn("读取收藏失败:", error);
  }
  return initial;
}

function saveFavorites(favoritesMap) {
  try {
    window.localStorage.setItem(DOU_BAO_FAVORITES_KEY, JSON.stringify(favoritesMap || {}));
  } catch (error) {
    console.warn("保存收藏失败:", error);
  }
}

app.registerExtension({
  name: "AptPreset.WebToolPreview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "AI_web_tool") {
      return;
    }

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onNodeCreated?.apply(this, arguments);

      const downloadDirWidget = this.widgets?.find((w) => w.name === "download_dir");
      const pageUrlWidget = this.widgets?.find((w) => w.name === "page_url");
      const imageAddressWidget = this.widgets?.find((w) => w.name === "image_address");

      const container = document.createElement("div");
      container.style.width = "100%";
      container.style.height = "100%";
      container.style.minWidth = "0";
      container.style.minHeight = "0";
      container.style.display = "flex";
      container.style.flexDirection = "column";
      container.style.background = "#151515";
      container.style.border = "1px solid #333";
      container.style.borderRadius = "8px";
      container.style.overflow = "hidden";

      const topBar = document.createElement("div");
      topBar.style.display = "flex";
      topBar.style.flexWrap = "nowrap";
      topBar.style.gap = "10px";
      topBar.style.padding = "10px";
      topBar.style.background = "#1f1f1f";
      topBar.style.borderBottom = "1px solid #333";
      topBar.style.alignItems = "stretch";

      const status = document.createElement("span");
      status.style.color = "#9aa0a6";
      status.style.fontSize = "12px";
      status.textContent = "";
      const downloadState = document.createElement("span");
      downloadState.style.color = "#ff4d4f";
      downloadState.style.fontSize = "12px";
      downloadState.style.fontWeight = "600";
      downloadState.textContent = "";
      let favoritesMap = loadFavorites();

      let captureSessionId = 1;
      let taskCompleted = false;
      let manualDownloading = false;
      let downloadProbeInitialized = false;
      let lastDownloadMtime = 0;
      let downloadCompleteUntil = 0;
      let lastClipboardImageUrl = "";

      const isLikelyImageUrl = (text) => {
        const value = (text || "").toString().trim();
        if (!/^https?:\/\//i.test(value)) return false;
        const lower = value.toLowerCase();
        if (/\.(png|jpe?g|webp|bmp|gif|svg)(\?|#|$)/i.test(lower)) return true;
        if (lower.includes("x-oss-process=image")) return true;
        if (lower.includes("/image/")) return true;
        return false;
      };
      const syncImageAddressFromClipboard = async () => {
        if (!imageAddressWidget || !navigator.clipboard?.readText) return;
        try {
          const clipText = (await navigator.clipboard.readText())?.trim() || "";
          if (!clipText || clipText === lastClipboardImageUrl) return;
          if (!isLikelyImageUrl(clipText)) return;
          imageAddressWidget.value = clipText;
          imageAddressWidget.callback?.(clipText);
          lastClipboardImageUrl = clipText;
          status.textContent = "已同步图片地址到 image_address";
        } catch (_) {
          // 部分浏览器环境未授权读取剪贴板，静默忽略
        }
      };

      const reportTaskEvent = async (event, sessionId, batchTotal = 0) => {
        try {
          await requestJson(TASK_EVENT_API, "POST", {
            event,
            session_id: sessionId,
            batch_total: batchTotal,
          });
        } catch (error) {
          console.warn(`任务事件上报失败(${event}):`, error);
        }
      };
      const queryTaskState = async () => {
        try {
          const data = await requestJson(TASK_EVENT_API, "POST", {
            event: "status",
            session_id: captureSessionId,
            batch_total: 0,
          });
          return data?.task_state || null;
        } catch (error) {
          return null;
        }
      };
      const queryDownloadProbe = async () => {
        try {
          const targetDir = (downloadDirWidget?.value ?? "").toString().trim();
          const query = targetDir ? `?download_dir=${encodeURIComponent(targetDir)}` : "";
          return await requestJson(`${DOWNLOAD_PROBE_API}${query}`, "GET");
        } catch (error) {
          return null;
        }
      };
      const refreshDownloadState = async () => {
        if (manualDownloading) {
          downloadState.textContent = "正在下载";
          return;
        }
        const probe = await queryDownloadProbe();
        if (probe) {
          const latestMtime = Number(probe.latest_mtime || 0);
          if (!downloadProbeInitialized) {
            downloadProbeInitialized = true;
            lastDownloadMtime = latestMtime;
          } else if (latestMtime > lastDownloadMtime + 1e-6) {
            lastDownloadMtime = latestMtime;
            downloadCompleteUntil = Date.now() + 5000;
          }
          if (Boolean(probe.has_temp_downloading)) {
            downloadState.textContent = "正在下载";
            return;
          }
          if (Date.now() < downloadCompleteUntil) {
            downloadState.textContent = "已完成";
            return;
          }
        }
        const taskState = await queryTaskState();
        if (!taskState) return;
        const batchTotal = Number(taskState.batch_total || 0);
        const downloaded = Number(taskState.downloaded || 0);
        const failed = Number(taskState.failed || 0);
        const hasBatch = batchTotal > 0;
        if (hasBatch && Boolean(taskState.active) && (downloaded + failed) < batchTotal) {
          downloadState.textContent = "正在下载";
          return;
        }
        if (hasBatch && (downloaded + failed) >= batchTotal) {
          downloadState.textContent = "已完成";
          return;
        }
        downloadState.textContent = "";
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

      const iframe = document.createElement("iframe");
      const applyIframeUrl = async () => {
        const nextUrl = normalizePageUrl(pageUrlWidget?.value);
        let finalUrl = nextUrl;
        if (isLocalPageLike(nextUrl)) {
          try {
            const data = await requestJson(LOCAL_PREVIEW_REGISTER_API, "POST", { path: nextUrl });
            finalUrl = (data?.preview_url || "").toString().trim() || nextUrl;
          } catch (error) {
            status.textContent = "本地预览注册失败";
          }
        }
        iframe.src = finalUrl;
        if (pageUrlWidget) {
          pageUrlWidget.value = nextUrl;
        }
      };

      const refreshBtn = createButton("刷新", () => {
        void applyIframeUrl();
      }, "#2f4f7a");
      const fullscreenBtn = createButton("全屏", async () => {
        try {
          if (document.fullscreenElement === container) {
            await document.exitFullscreen();
          } else if (!document.fullscreenElement) {
            await container.requestFullscreen();
          }
        } catch (error) {
          status.textContent = "全屏切换失败";
        }
      }, "#2f4f7a");
      const favoriteBtn = createButton("收藏", () => {
        openFavoriteModal();
      }, "#7a5f2f");
      const favoriteSelect = document.createElement("select");
      favoriteSelect.style.height = "28px";
      favoriteSelect.style.width = "160px";
      favoriteSelect.style.minWidth = "160px";
      favoriteSelect.style.border = "1px solid #4d4d4d";
      favoriteSelect.style.borderRadius = "6px";
      favoriteSelect.style.background = "#2a2a2a";
      favoriteSelect.style.color = "#ddd";
      favoriteSelect.style.padding = "0 8px";
      favoriteSelect.style.fontSize = "12px";

      const renderFavoriteOptions = (selectedName = "") => {
        favoriteSelect.innerHTML = "";
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "选择收藏地址";
        favoriteSelect.appendChild(placeholder);
        for (const [name, url] of Object.entries(favoritesMap)) {
          const option = document.createElement("option");
          option.value = name;
          option.textContent = name;
          option.title = url;
          favoriteSelect.appendChild(option);
        }
        if (selectedName && favoritesMap[selectedName]) {
          favoriteSelect.value = selectedName;
        } else {
          const currentUrl = normalizePageUrl(pageUrlWidget?.value || DEFAULT_PREVIEW_URL);
          const matched = Object.entries(favoritesMap).find(([, url]) => normalizePageUrl(url) === currentUrl);
          favoriteSelect.value = matched ? matched[0] : "";
        }
      };
      favoriteSelect.addEventListener("change", () => {
        const selectedName = (favoriteSelect.value || "").trim();
        if (!selectedName || !favoritesMap[selectedName]) return;
        const nextUrl = normalizePageUrl(favoritesMap[selectedName]);
        if (pageUrlWidget) {
          pageUrlWidget.value = nextUrl;
        }
        void applyIframeUrl();
      });
      const filePickerInput = document.createElement("input");
      filePickerInput.type = "file";
      filePickerInput.accept = "image/*";
      filePickerInput.style.display = "none";
      const fileToPngBlob = async (file) => {
        const objectUrl = URL.createObjectURL(file);
        try {
          const image = await new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = objectUrl;
          });
          const canvas = document.createElement("canvas");
          canvas.width = image.naturalWidth || image.width;
          canvas.height = image.naturalHeight || image.height;
          const ctx = canvas.getContext("2d");
          if (!ctx) {
            throw new Error("无法创建画布上下文");
          }
          ctx.drawImage(image, 0, 0);
          const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
          if (!blob) {
            throw new Error("图片转换失败");
          }
          return blob;
        } finally {
          URL.revokeObjectURL(objectUrl);
        }
      };
      const copyImageFileToClipboard = async (file) => {
        if (typeof ClipboardItem === "undefined" || !navigator.clipboard?.write) {
          throw new Error("当前浏览器不支持图片写入剪贴板。");
        }
        const pngBlob = await fileToPngBlob(file);
        await navigator.clipboard.write([new ClipboardItem({ "image/png": pngBlob })]);
      };
      const copySelectionBtn = createButton("选图复制", () => {
        filePickerInput.click();
      }, "#5a3f7a");
      copySelectionBtn.style.width = "100px";
      copySelectionBtn.style.height = "32px";
      favoriteBtn.style.width = "60px";
      favoriteBtn.style.height = "30px";
      refreshBtn.style.width = "60x";
      refreshBtn.style.height = "30px";
      fullscreenBtn.style.width = "60px";
      fullscreenBtn.style.height = "30px";
      const syncFullscreenButtonLabel = () => {
        fullscreenBtn.textContent = document.fullscreenElement === container ? "退出" : "全屏";
      };
      document.addEventListener("fullscreenchange", syncFullscreenButtonLabel);
      filePickerInput.addEventListener("change", async () => {
        const imageFile = filePickerInput.files?.[0];
        if (!imageFile) return;
        try {
          await copyImageFileToClipboard(imageFile);
          status.textContent = "图片已复制到剪贴板，请到网页内 Ctrl+V";
        } catch (error) {
          console.warn("选图复制失败:", error);
          try {
            await navigator.clipboard.writeText("图片复制失败：请重试或使用拖拽兜底上传。");
          } catch (_) {}
          status.textContent = "选图复制失败";
        } finally {
          filePickerInput.value = "";
        }
      });

      // 读取上次保存的下载路径和网址，避免每次重新输入
      try {
        const savedDir = window.localStorage.getItem(DOU_BAO_DOWNLOAD_DIR_KEY) || "";
        if (savedDir && downloadDirWidget && !downloadDirWidget.value) {
          downloadDirWidget.value = savedDir;
        }

        const savedUrl = window.localStorage.getItem(DOU_BAO_PAGE_URL_KEY) || "";
        if (pageUrlWidget) {
          const initialUrl = pageUrlWidget.value || savedUrl || DEFAULT_PREVIEW_URL;
          pageUrlWidget.value = normalizePageUrl(initialUrl);
        }
      } catch (error) {
        console.warn("读取节点缓存失败:", error);
      }
      bindStringWidgetPersistence(downloadDirWidget, DOU_BAO_DOWNLOAD_DIR_KEY);
      bindStringWidgetPersistence(pageUrlWidget, DOU_BAO_PAGE_URL_KEY);
      renderFavoriteOptions();

      const favoriteModalOverlay = document.createElement("div");
      favoriteModalOverlay.style.position = "fixed";
      favoriteModalOverlay.style.inset = "0";
      favoriteModalOverlay.style.background = "rgba(0, 0, 0, 0.55)";
      favoriteModalOverlay.style.display = "none";
      favoriteModalOverlay.style.alignItems = "center";
      favoriteModalOverlay.style.justifyContent = "center";
      favoriteModalOverlay.style.zIndex = "9999";

      const favoriteModal = document.createElement("div");
      favoriteModal.style.width = "420px";
      favoriteModal.style.maxWidth = "90vw";
      favoriteModal.style.background = "#1f1f1f";
      favoriteModal.style.border = "1px solid #3b3b3b";
      favoriteModal.style.borderRadius = "10px";
      favoriteModal.style.padding = "14px";
      favoriteModal.style.boxSizing = "border-box";
      favoriteModal.style.display = "flex";
      favoriteModal.style.flexDirection = "column";
      favoriteModal.style.gap = "8px";

      const modalTitle = document.createElement("div");
      modalTitle.textContent = "收藏管理";
      modalTitle.style.color = "#eee";
      modalTitle.style.fontSize = "14px";
      modalTitle.style.fontWeight = "600";

      const nameInput = document.createElement("input");
      nameInput.type = "text";
      nameInput.placeholder = "名称";
      nameInput.style.height = "30px";
      nameInput.style.border = "1px solid #4d4d4d";
      nameInput.style.borderRadius = "6px";
      nameInput.style.background = "#2a2a2a";
      nameInput.style.color = "#ddd";
      nameInput.style.padding = "0 8px";

      const urlInput = document.createElement("input");
      urlInput.type = "text";
      urlInput.placeholder = "地址";
      urlInput.style.height = "30px";
      urlInput.style.border = "1px solid #4d4d4d";
      urlInput.style.borderRadius = "6px";
      urlInput.style.background = "#2a2a2a";
      urlInput.style.color = "#ddd";
      urlInput.style.padding = "0 8px";

      const modalActions = document.createElement("div");
      modalActions.style.display = "flex";
      modalActions.style.gap = "8px";
      modalActions.style.justifyContent = "flex-end";

      const closeFavoriteModal = () => {
        favoriteModalOverlay.style.display = "none";
      };
      const openFavoriteModal = () => {
        const selectedName = (favoriteSelect.value || "").trim();
        const selectedUrl = selectedName && favoritesMap[selectedName] ? favoritesMap[selectedName] : "";
        nameInput.value = selectedName;
        urlInput.value = selectedUrl || normalizePageUrl(pageUrlWidget?.value || DEFAULT_PREVIEW_URL);
        favoriteModalOverlay.style.display = "flex";
        nameInput.focus();
        nameInput.select();
      };

      const modalSaveBtn = createButton("添加/更新", () => {
        const name = (nameInput.value || "").trim();
        const inputUrl = (urlInput.value || "").trim();
        if (!name || !inputUrl) {
          status.textContent = "名称和地址不能为空";
          return;
        }
        const nextUrl = normalizePageUrl(inputUrl);
        favoritesMap[name] = nextUrl;
        saveFavorites(favoritesMap);
        renderFavoriteOptions(name);
        if (pageUrlWidget) {
          pageUrlWidget.value = nextUrl;
        }
        void applyIframeUrl();
        status.textContent = `已保存：${name}`;
        closeFavoriteModal();
      }, "#2f6f4a");

      const modalDeleteBtn = createButton("删除", () => {
        const name = (nameInput.value || "").trim();
        if (!name || !favoritesMap[name]) {
          status.textContent = "未找到可删除的收藏";
          return;
        }
        delete favoritesMap[name];
        saveFavorites(favoritesMap);
        renderFavoriteOptions();
        status.textContent = `已删除：${name}`;
        closeFavoriteModal();
      }, "#7a2f2f");

      const modalCancelBtn = createButton("取消", () => {
        closeFavoriteModal();
      }, "#3b3b3b");

      favoriteModalOverlay.addEventListener("click", (event) => {
        if (event.target === favoriteModalOverlay) {
          closeFavoriteModal();
        }
      });
      favoriteModal.appendChild(modalTitle);
      favoriteModal.appendChild(nameInput);
      favoriteModal.appendChild(urlInput);
      modalActions.appendChild(modalCancelBtn);
      modalActions.appendChild(modalDeleteBtn);
      modalActions.appendChild(modalSaveBtn);
      favoriteModal.appendChild(modalActions);
      favoriteModalOverlay.appendChild(favoriteModal);
      document.body.appendChild(favoriteModalOverlay);
      const rightPanel = document.createElement("div");
      rightPanel.style.flex = "1";
      rightPanel.style.display = "flex";
      rightPanel.style.flexDirection = "column";
      rightPanel.style.gap = "8px";
      rightPanel.style.minWidth = "0";
      const rightTopRow = document.createElement("div");
      rightTopRow.style.display = "flex";
      rightTopRow.style.gap = "8px";
      rightTopRow.style.alignItems = "center";
      rightTopRow.style.flexWrap = "nowrap";
      const rightInfoRow = document.createElement("div");
      rightInfoRow.style.display = "flex";
      rightInfoRow.style.gap = "18px";
      rightInfoRow.style.alignItems = "center";
      rightInfoRow.style.minHeight = "28px";
      status.style.color = "#aeb4bc";
      rightTopRow.appendChild(copySelectionBtn);
      rightTopRow.appendChild(favoriteBtn);
      rightTopRow.appendChild(favoriteSelect);
      rightTopRow.appendChild(refreshBtn);
      rightTopRow.appendChild(fullscreenBtn);
      rightTopRow.appendChild(filePickerInput);
      rightInfoRow.appendChild(status);
      rightInfoRow.appendChild(downloadState);
      rightPanel.appendChild(rightTopRow);
      rightPanel.appendChild(rightInfoRow);
      topBar.appendChild(rightPanel);

      const preventDefaultDrop = (event) => {
        event.preventDefault();
      };
      window.addEventListener("dragover", preventDefaultDrop);
      window.addEventListener("drop", preventDefaultDrop);

      void applyIframeUrl();
      iframe.style.flex = "1";
      iframe.style.minHeight = "0";
      iframe.style.width = "100%";
      iframe.style.border = "none";
      iframe.referrerPolicy = "no-referrer";
      iframe.allow = "clipboard-read; clipboard-write";

      container.appendChild(topBar);
      container.appendChild(iframe);

      const domWidget = this.addDOMWidget("doubao_web_ui", "customwidget", container, {
        serialize: false,
        hideOnZoom: false,
      });
      if (!this.size || this.size[1] < 600) {
        this.setSize([Math.max(this.size?.[0] || 0, 1200), Math.max(this.size?.[1] || 0, 600)]);
      }
      this.minHeight = Math.max(this.minHeight || 0, 600);
      this.maxHeight = 2000;
      const prevOnResize = this.onResize;
      this.onResize = function (size) {
        if (Array.isArray(size) && size.length >= 2 && size[1] > 2000) {
          this.setSize([size[0], 2000]);
        }
        return prevOnResize?.apply(this, arguments);
      };
      let saveCounter = 0;
      void reportTaskEvent("session_start", captureSessionId, 0);
      const downloadStateTimer = window.setInterval(() => {
        void refreshDownloadState();
      }, 1000);
      const clipboardSyncTimer = window.setInterval(() => {
        void syncImageAddressFromClipboard();
      }, 1200);
      void refreshDownloadState();
      void syncImageAddressFromClipboard();
      const onWindowMessage = async (event) => {
        if (!AUTO_REMOTE_DOWNLOAD_ENABLED) {
          return;
        }
        if (!iframe.contentWindow || event.source !== iframe.contentWindow) {
          return;
        }
        const msgType = event?.data?.type;
        const isBridgeType = msgType === "aptpreset_doubao_image_data";
        if (!isBridgeType) {
          return;
        }
        const currentSessionId = captureSessionId;
        const urls = extractNoWatermarkUrls(event?.data?.data || []);
        if (!urls.length) {
          return;
        }
        manualDownloading = true;
        downloadState.textContent = "正在下载";
        taskCompleted = false;
        void reportTaskEvent("batch_begin", currentSessionId, urls.length);

        for (const imageUrl of urls) {
          try {
            const targetDir = (downloadDirWidget?.value ?? "").toString().trim();
            const data = await requestJson(SAVE_IMAGE_API, "POST", {
              url: imageUrl,
              download_dir: targetDir,
              session_id: currentSessionId,
              batch_total: urls.length,
            });
            taskCompleted = Boolean(data?.task_state?.task_completed);
            saveCounter += 1;
            const completionText = taskCompleted ? "，本轮任务完成" : "";
            status.textContent = `已保存${saveCounter}张${completionText}`;
            if (taskCompleted) {
              downloadState.textContent = "已完成";
            }
          } catch (error) {
            console.error("保存无水印图失败:", error);
            void reportTaskEvent("download_failed", currentSessionId, urls.length);
          }
        }
        manualDownloading = false;
        void refreshDownloadState();
      };
      window.addEventListener("message", onWindowMessage);

      const oldRemoved = this.onRemoved;
      this.onRemoved = function () {
        window.removeEventListener("message", onWindowMessage);
        window.removeEventListener("dragover", preventDefaultDrop);
        window.removeEventListener("drop", preventDefaultDrop);
        window.clearInterval(downloadStateTimer);
        window.clearInterval(clipboardSyncTimer);
        document.removeEventListener("fullscreenchange", syncFullscreenButtonLabel);
        favoriteModalOverlay.remove();
        void reportTaskEvent("session_stop", captureSessionId, 0);
        oldRemoved?.apply(this, arguments);
      };
      this.resizable = true;
    };
  },
});
