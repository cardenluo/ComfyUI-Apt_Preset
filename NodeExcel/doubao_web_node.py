import os
import io
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional

import numpy as np
import torch
from PIL import Image

import folder_paths

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None


class _DoubaoCaptureService:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._seen_hashes = set()
        self._seen_limit = 800
        self._running = False
        self._download_dir = ""
        self._captured_count = 0
        self._last_file = ""
        self._last_error = ""
        self._hint = "未启动"

    @staticmethod
    def _default_download_dir() -> str:
        home = os.path.expanduser("~")
        return os.path.join(home, "Downloads")

    @staticmethod
    def _guess_ext(url: str, content_type: str) -> str:
        ctype = (content_type or "").lower()
        if "png" in ctype:
            return ".png"
        if "jpeg" in ctype or "jpg" in ctype:
            return ".jpg"
        if "webp" in ctype:
            return ".webp"
        if "bmp" in ctype:
            return ".bmp"
        lower_url = (url or "").lower()
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
            if ext in lower_url:
                return ext
        return ".png"

    @staticmethod
    def _is_target_image(url: str, content_type: str) -> bool:
        u = (url or "").lower()
        ct = (content_type or "").lower()
        is_image = ("image/" in ct) or any(ext in u for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"))
        if not is_image:
            return False
        # 降低噪声：只抓豆包常见图片源
        host_hit = any(k in u for k in ("doubao.com", "byteimg.com", "volces.com", "tos-cn"))
        return host_hit

    def _set_status(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, f"_{key}", value)

    def status(self):
        with self._lock:
            return {
                "running": self._running,
                "download_dir": self._download_dir,
                "captured_count": self._captured_count,
                "last_file": self._last_file,
                "last_error": self._last_error,
                "hint": self._hint,
            }

    def start(self, download_dir: str = "", headless: bool = False, timeout_sec: int = 1800):
        _ = (download_dir, headless, timeout_sec)
        self._set_status(
            running=False,
            last_error="浏览器自动抓取功能已移除。",
            hint="请改用 save_remote_image 接口传入图片 URL 保存。",
        )
        return False, "浏览器自动抓取功能已移除。"

    def stop(self):
        with self._lock:
            if not self._running:
                return False, "当前没有运行中的抓取任务。"
            self._stop_event.set()
            self._hint = "正在停止抓取..."
            return True, "停止指令已发送。"

    def _save_download_file(self, download, output_dir: str):
        suggested_name = ""
        try:
            suggested_name = str(download.suggested_filename or "").strip()
        except Exception:
            suggested_name = ""
        ext = os.path.splitext(suggested_name)[1].lower()
        if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            ext = ".png"
        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        file_name = f"doubao_auto_{ts}_{ms:03d}{ext}"
        file_path = os.path.join(output_dir, file_name)
        download.save_as(file_path)
        if not os.path.isfile(file_path) or os.path.getsize(file_path) <= 0:
            return
        with self._lock:
            self._captured_count += 1
            self._last_file = file_path
            self._hint = f"已抓取 {self._captured_count} 张"

    def _run_capture(self, output_dir: str, headless: bool, timeout_sec: int):
        _ = (output_dir, headless, timeout_sec)
        self._set_status(
            running=False,
            last_error="浏览器自动抓取功能已移除。",
            hint="请改用 save_remote_image 接口传入图片 URL 保存。",
        )


_DOUBAO_CAPTURE_SERVICE = _DoubaoCaptureService()


class _DoubaoBridgeTaskTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._session_id = 0
        self._batch_total = 0
        self._downloaded = 0
        self._failed = 0
        self._task_completed = False
        self._active = False

    @staticmethod
    def _to_int(value, default=0):
        try:
            return int(value)
        except Exception:
            return default

    def _reset_for_session(self, session_id: int):
        self._session_id = max(0, int(session_id))
        self._batch_total = 0
        self._downloaded = 0
        self._failed = 0
        self._task_completed = False
        self._active = True

    def start_session(self, session_id: int):
        with self._lock:
            self._reset_for_session(session_id)
            return self.snapshot_locked()

    def stop_session(self, session_id: int):
        with self._lock:
            if int(session_id) == self._session_id:
                self._active = False
                self._task_completed = self._batch_total > 0 and self._downloaded >= self._batch_total
            return self.snapshot_locked()

    def begin_batch(self, session_id: int, batch_total: int):
        with self._lock:
            if int(session_id) != self._session_id:
                self._reset_for_session(session_id)
            self._batch_total = max(0, self._to_int(batch_total, 0))
            self._downloaded = 0
            self._failed = 0
            self._task_completed = self._batch_total == 0
            self._active = self._batch_total > 0
            return self.snapshot_locked()

    def mark_success(self, session_id: int, batch_total: int = 0):
        with self._lock:
            if int(session_id) != self._session_id:
                return self.snapshot_locked()
            normalized_total = max(0, self._to_int(batch_total, 0))
            if normalized_total > 0 and normalized_total != self._batch_total:
                self._batch_total = normalized_total
            self._downloaded += 1
            self._task_completed = self._batch_total > 0 and self._downloaded >= self._batch_total
            if self._task_completed:
                self._active = False
            return self.snapshot_locked()

    def mark_failed(self, session_id: int, batch_total: int = 0):
        with self._lock:
            if int(session_id) != self._session_id:
                return self.snapshot_locked()
            normalized_total = max(0, self._to_int(batch_total, 0))
            if normalized_total > 0 and normalized_total != self._batch_total:
                self._batch_total = normalized_total
            self._failed += 1
            if self._batch_total > 0 and (self._downloaded + self._failed) >= self._batch_total:
                self._active = False
                self._task_completed = self._downloaded >= self._batch_total
            return self.snapshot_locked()

    def snapshot_locked(self):
        return {
            "session_id": self._session_id,
            "batch_total": self._batch_total,
            "downloaded": self._downloaded,
            "failed": self._failed,
            "task_completed": self._task_completed,
            "active": self._active,
        }

    def snapshot(self):
        with self._lock:
            return self.snapshot_locked()


_DOUBAO_BRIDGE_TASK_TRACKER = _DoubaoBridgeTaskTracker()
_LOCAL_PREVIEW_LOCK = threading.Lock()
_LOCAL_PREVIEW_MAP = {}
_LOCAL_PREVIEW_SEQ = 0


def _normalize_local_page_path(raw_value: str) -> str:
    value = str(raw_value or "").strip().strip('"').strip("'")
    if not value:
        return ""
    if value.lower().startswith("file://"):
        parsed = urllib.parse.urlparse(value)
        path_value = urllib.parse.unquote(parsed.path or "")
        if os.name == "nt" and len(path_value) >= 3 and path_value[0] == "/" and path_value[2] == ":":
            path_value = path_value[1:]
        value = path_value
    value = urllib.parse.unquote(value)
    value = os.path.expanduser(value)
    value = os.path.expandvars(value)
    return os.path.normpath(value)


def _register_local_preview(file_path: str) -> str:
    global _LOCAL_PREVIEW_SEQ
    root_dir = os.path.dirname(file_path)
    entry_name = os.path.basename(file_path)
    with _LOCAL_PREVIEW_LOCK:
        _LOCAL_PREVIEW_SEQ += 1
        key = f"lp{int(time.time())}_{_LOCAL_PREVIEW_SEQ}"
        _LOCAL_PREVIEW_MAP[key] = {
            "root_dir": root_dir,
            "entry_name": entry_name,
        }
    return key


def _probe_download_dir(download_dir: str):
    latest_file = ""
    latest_mtime = 0.0
    has_temp_downloading = False
    image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    temp_exts = (".crdownload", ".part", ".tmp")

    if not download_dir or not os.path.isdir(download_dir):
        return {
            "download_dir": download_dir,
            "latest_file": latest_file,
            "latest_mtime": latest_mtime,
            "has_temp_downloading": has_temp_downloading,
        }

    for root, _, files in os.walk(download_dir):
        for file_name in files:
            lower = file_name.lower()
            if lower.endswith(temp_exts):
                has_temp_downloading = True
            ext = os.path.splitext(lower)[1]
            if ext not in image_exts:
                continue
            full_path = os.path.join(root, file_name)
            if not os.path.isfile(full_path):
                continue
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue
            if mtime >= latest_mtime:
                latest_mtime = mtime
                latest_file = full_path

    return {
        "download_dir": download_dir,
        "latest_file": latest_file,
        "latest_mtime": latest_mtime,
        "has_temp_downloading": has_temp_downloading,
    }


if PromptServer is not None and web is not None:
    @PromptServer.instance.routes.post("/apt_preset/local_preview/register")
    async def apt_preset_local_preview_register(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        input_path = _normalize_local_page_path(payload.get("path", ""))
        if not input_path:
            return web.json_response({"ok": False, "error": "路径不能为空。"}, status=400)
        if not os.path.isabs(input_path):
            return web.json_response({"ok": False, "error": "仅支持本地绝对路径。"}, status=400)
        if not os.path.isfile(input_path):
            return web.json_response({"ok": False, "error": f"文件不存在: {input_path}"}, status=404)
        key = _register_local_preview(input_path)
        return web.json_response(
            {
                "ok": True,
                "preview_url": f"/apt_preset/local_preview/{key}/",
            }
        )

    @PromptServer.instance.routes.get("/apt_preset/local_preview/{key}/{tail:.*}")
    async def apt_preset_local_preview_file(request):
        key = str(request.match_info.get("key", "")).strip()
        tail = str(request.match_info.get("tail", "")).strip()
        with _LOCAL_PREVIEW_LOCK:
            meta = _LOCAL_PREVIEW_MAP.get(key)
        if not meta:
            return web.Response(status=404, text="Local preview key not found.")

        root_dir = os.path.normpath(str(meta.get("root_dir", "")))
        entry_name = str(meta.get("entry_name", ""))
        rel_path = entry_name if not tail else urllib.parse.unquote(tail)
        candidate = os.path.normpath(os.path.join(root_dir, rel_path))
        root_case = os.path.normcase(root_dir)
        cand_case = os.path.normcase(candidate)
        allowed = cand_case == root_case or cand_case.startswith(root_case + os.sep)
        if not allowed:
            return web.Response(status=403, text="Access denied.")
        if os.path.isdir(candidate):
            candidate = os.path.join(candidate, "index.html")
        if not os.path.isfile(candidate):
            return web.Response(status=404, text="File not found.")
        return web.FileResponse(candidate)

    @PromptServer.instance.routes.post("/apt_preset/doubao_capture/start")
    async def apt_preset_doubao_capture_start(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        ok, message = _DOUBAO_CAPTURE_SERVICE.start(
            download_dir=payload.get("download_dir", ""),
            headless=bool(payload.get("headless", False)),
            timeout_sec=int(payload.get("timeout_sec", 1800)),
        )
        data = _DOUBAO_CAPTURE_SERVICE.status()
        data.update({"ok": ok, "message": message})
        return web.json_response(data, status=200 if ok else 400)

    @PromptServer.instance.routes.post("/apt_preset/doubao_capture/stop")
    async def apt_preset_doubao_capture_stop(request):
        ok, message = _DOUBAO_CAPTURE_SERVICE.stop()
        data = _DOUBAO_CAPTURE_SERVICE.status()
        data.update({"ok": ok, "message": message})
        return web.json_response(data, status=200 if ok else 400)

    @PromptServer.instance.routes.get("/apt_preset/doubao_capture/status")
    async def apt_preset_doubao_capture_status(request):
        data = _DOUBAO_CAPTURE_SERVICE.status()
        data.update({"ok": True})
        return web.json_response(data)

    @PromptServer.instance.routes.get("/apt_preset/doubao_capture/download_probe")
    async def apt_preset_doubao_capture_download_probe(request):
        download_dir = str(request.rel_url.query.get("download_dir", "")).strip() or _DoubaoCaptureService._default_download_dir()
        probe = _probe_download_dir(download_dir)
        probe.update({"ok": True})
        return web.json_response(probe)

    @PromptServer.instance.routes.post("/apt_preset/doubao_capture/save_remote_image")
    async def apt_preset_doubao_capture_save_remote_image(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}

        image_url = str(payload.get("url", "")).strip()
        session_id = _DoubaoBridgeTaskTracker._to_int(payload.get("session_id", 0), 0)
        batch_total = _DoubaoBridgeTaskTracker._to_int(payload.get("batch_total", 0), 0)
        download_dir = str(payload.get("download_dir", "")).strip() or _DoubaoCaptureService._default_download_dir()
        if not image_url.startswith(("http://", "https://")):
            _DOUBAO_BRIDGE_TASK_TRACKER.mark_failed(session_id, batch_total)
            return web.json_response({"ok": False, "error": "无效图片地址。"}, status=400)

        os.makedirs(download_dir, exist_ok=True)
        req = urllib.request.Request(
            image_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            },
            method="GET",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                content_type = (resp.headers.get("Content-Type") or "").lower()
                body = resp.read()
        except Exception as e:
            _DOUBAO_BRIDGE_TASK_TRACKER.mark_failed(session_id, batch_total)
            return web.json_response({"ok": False, "error": f"下载失败: {e}"}, status=500)

        if not body:
            _DOUBAO_BRIDGE_TASK_TRACKER.mark_failed(session_id, batch_total)
            return web.json_response({"ok": False, "error": "下载内容为空。"}, status=500)
        if "image/" not in content_type:
            _DOUBAO_BRIDGE_TASK_TRACKER.mark_failed(session_id, batch_total)
            return web.json_response({"ok": False, "error": f"返回不是图片: {content_type}"}, status=400)

        ext = ".png"
        if "jpeg" in content_type or "jpg" in content_type:
            ext = ".jpg"
        elif "webp" in content_type:
            ext = ".webp"
        elif "bmp" in content_type:
            ext = ".bmp"
        else:
            parsed = urllib.parse.urlparse(image_url)
            guessed = os.path.splitext(parsed.path)[1].lower()
            if guessed in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                ext = guessed

        ts = time.strftime("%Y%m%d_%H%M%S")
        ms = int((time.time() % 1) * 1000)
        file_name = f"doubao_bridge_{ts}_{ms:03d}{ext}"
        file_path = os.path.join(download_dir, file_name)
        try:
            with open(file_path, "wb") as f:
                f.write(body)
        except Exception as e:
            _DOUBAO_BRIDGE_TASK_TRACKER.mark_failed(session_id, batch_total)
            return web.json_response({"ok": False, "error": f"保存失败: {e}"}, status=500)
        task_state = _DOUBAO_BRIDGE_TASK_TRACKER.mark_success(session_id, batch_total)

        return web.json_response(
            {
                "ok": True,
                "file_name": file_name,
                "file_path": file_path,
                "size": len(body),
                "task_state": task_state,
            }
        )

    @PromptServer.instance.routes.post("/apt_preset/doubao_capture/report_task_event")
    async def apt_preset_doubao_capture_report_task_event(request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        event_type = str(payload.get("event", "")).strip()
        session_id = _DoubaoBridgeTaskTracker._to_int(payload.get("session_id", 0), 0)
        batch_total = _DoubaoBridgeTaskTracker._to_int(payload.get("batch_total", 0), 0)
        if event_type == "session_start":
            state = _DOUBAO_BRIDGE_TASK_TRACKER.start_session(session_id)
        elif event_type == "session_stop":
            state = _DOUBAO_BRIDGE_TASK_TRACKER.stop_session(session_id)
        elif event_type == "batch_begin":
            state = _DOUBAO_BRIDGE_TASK_TRACKER.begin_batch(session_id, batch_total)
        elif event_type == "download_failed":
            state = _DOUBAO_BRIDGE_TASK_TRACKER.mark_failed(session_id, batch_total)
        elif event_type == "status":
            state = _DOUBAO_BRIDGE_TASK_TRACKER.snapshot()
        else:
            return web.json_response({"ok": False, "error": f"未知事件类型: {event_type}"}, status=400)
        return web.json_response({"ok": True, "task_state": state})


class AI_web_tool:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "download_dir": ("STRING", {"default": ""}),
                "page_url": ("STRING", {"default": "https://www.doubao.com/"}),
                "image_address": ("STRING", {"default": ""}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_preview"
    CATEGORY = "Apt_Preset/AI_tool"

    @staticmethod
    def _empty_result():
        return ([torch.zeros([1, 64, 64, 3])],)

    @staticmethod
    def _clean_name(name: str) -> str:
        # 兼容诸如 "xxx.png [output]" 这类前端字符串
        if "[" in name:
            name = name.split("[", 1)[0]
        return name.strip()

    @staticmethod
    def _is_image_file(file_name: str) -> bool:
        ext = os.path.splitext(file_name)[1].lower()
        return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def _find_latest_by_mtime(self, base_dir: str, recursive: bool, limit: int = 1):
        if not base_dir or not os.path.isdir(base_dir):
            return []

        candidates = []

        if recursive:
            iterator = os.walk(base_dir)
        else:
            iterator = [(base_dir, [], os.listdir(base_dir))]

        for root, _, files in iterator:
            for file_name in files:
                if not self._is_image_file(file_name):
                    continue
                full_path = os.path.join(root, file_name)
                if not os.path.isfile(full_path):
                    continue
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    continue
                candidates.append((mtime, full_path))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_n = max(1, int(limit))
        return [path for _, path in candidates[:top_n]]

    def _find_latest_images(self, output_dir: str, limit: int) -> list:
        return self._find_latest_by_mtime(output_dir, recursive=True, limit=limit)

    def _default_download_dir(self) -> str:
        home = os.path.expanduser("~")
        return os.path.join(home, "Downloads")

    def _find_latest_download_images(self, download_dir: str, limit: int) -> list:
        # 严格按“下载目录中的最新修改时间”选择，避免文件名权重导致选错旧图。
        return self._find_latest_by_mtime(download_dir, recursive=True, limit=limit)

    def _resolve_image_paths(self, download_dir: str, limit: int) -> list:
        output_dir = folder_paths.get_output_directory()
        target_download_dir = download_dir.strip() if download_dir else self._default_download_dir()
        latest_download = self._find_latest_download_images(target_download_dir, limit=limit)
        if latest_download:
            return latest_download
        return self._find_latest_images(output_dir, limit=limit)

    def _resolve_image_paths_with_address(self, image_address: str, download_dir: str, limit: int) -> list:
        raw = (image_address or "").strip().strip('"').strip("'")
        if raw.startswith(("http://", "https://")):
            # image_address 非空时强优先：网页地址直接走 URL 加载流程
            return [raw]
        # 兼容 URL 编码地址和 file:// URI（仅对本地路径场景解码）
        raw = urllib.parse.unquote(raw)
        if raw.lower().startswith("file://"):
            parsed = urllib.parse.urlparse(raw)
            uri_path = urllib.parse.unquote(parsed.path or "")
            if os.name == "nt" and len(uri_path) >= 3 and uri_path[0] == "/" and uri_path[2] == ":":
                uri_path = uri_path[1:]
            raw = uri_path
        if not raw:
            return self._resolve_image_paths(download_dir, limit=limit)

        cleaned = raw
        storage_hint = ""
        if "[" in raw and "]" in raw:
            cleaned = raw.split("[", 1)[0].strip()
            storage_hint = raw.split("[", 1)[1].split("]", 1)[0].strip().lower()
        else:
            cleaned = self._clean_name(raw)

        candidates = []
        if cleaned:
            candidates.append(cleaned)
            normalized = os.path.normpath(cleaned)
            if normalized not in candidates:
                candidates.append(normalized)

        if cleaned and not os.path.isabs(cleaned):
            hint_dirs = []
            if storage_hint == "output":
                hint_dirs.append(folder_paths.get_output_directory())
            elif storage_hint == "input":
                get_input_dir = getattr(folder_paths, "get_input_directory", None)
                if callable(get_input_dir):
                    hint_dirs.append(get_input_dir())
            elif storage_hint == "temp":
                get_temp_dir = getattr(folder_paths, "get_temp_directory", None)
                if callable(get_temp_dir):
                    hint_dirs.append(get_temp_dir())
            else:
                hint_dirs.append(folder_paths.get_output_directory())
                get_input_dir = getattr(folder_paths, "get_input_directory", None)
                if callable(get_input_dir):
                    hint_dirs.append(get_input_dir())
                get_temp_dir = getattr(folder_paths, "get_temp_directory", None)
                if callable(get_temp_dir):
                    hint_dirs.append(get_temp_dir())

            for base_dir in hint_dirs:
                if base_dir:
                    candidates.append(os.path.join(base_dir, cleaned))

        checked_dirs = set()
        for candidate in candidates:
            if os.path.isfile(candidate) and self._is_image_file(candidate):
                return [candidate]
            if os.path.isdir(candidate):
                normalized_dir = os.path.normpath(candidate)
                if normalized_dir in checked_dirs:
                    continue
                checked_dirs.add(normalized_dir)
                from_dir = self._find_latest_by_mtime(candidate, recursive=True, limit=limit)
                if from_dir:
                    return from_dir

        print(f"AI_web_tool: image_address 无效或未命中 -> {image_address}")
        # image_address 非空时强优先：不再回退下载目录/输出目录自动扫描
        return []

    @classmethod
    def IS_CHANGED(
        cls,
        download_dir="",
        page_url="",
        image_address="",
        image_count=1,
        **kwargs,
    ):
        _ = (download_dir, page_url, image_address, image_count, kwargs)
        # 每次运行都强制刷新，重新扫描最新图片
        return time.time_ns()

    def load_preview(
        self,
        download_dir="",
        page_url="",
        image_address="",
        image_count=1,
    ):
        image_paths = self._resolve_image_paths_with_address(
            image_address=image_address,
            download_dir=download_dir,
            limit=max(1, int(image_count)),
        )
        if not image_paths:
            print("AI_web_tool: 未找到可输出的图片文件。")
            return self._empty_result()
        images = []
        for image_path in image_paths:
            try:
                if str(image_path).startswith(("http://", "https://")):
                    req = urllib.request.Request(
                        image_path,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                        },
                        method="GET",
                    )
                    with urllib.request.urlopen(req, timeout=20) as resp:
                        body = resp.read()
                    if not body:
                        print(f"AI_web_tool: 远程图片内容为空 -> {image_path}")
                        continue
                    image = Image.open(io.BytesIO(body)).convert("RGB")
                else:
                    if not os.path.exists(image_path):
                        print(f"AI_web_tool: 文件不存在 -> {image_path}")
                        continue
                    image = Image.open(image_path).convert("RGB")
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                images.append(image_tensor)
            except Exception as e:
                print(f"AI_web_tool: 读取图片失败 -> {e}")
                continue

        if not images:
            return self._empty_result()
        return (images,)



class AI_DoubaoWebPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "download_dir": ("STRING", {"default": ""}),
                "image_count": ("INT", {"default": 1, "min": 1, "max": 32, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "task_completed")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "load_preview"
    CATEGORY = "Apt_Preset/AI_tool"

    @staticmethod
    def _empty_result():
        task_completed = bool(_DOUBAO_BRIDGE_TASK_TRACKER.snapshot().get("task_completed", False))
        return ([torch.zeros([1, 64, 64, 3])], task_completed)

    @staticmethod
    def _clean_name(name: str) -> str:
        if "[" in name:
            name = name.split("[", 1)[0]
        return name.strip()

    @staticmethod
    def _is_image_file(file_name: str) -> bool:
        ext = os.path.splitext(file_name)[1].lower()
        return ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    def _find_latest_by_mtime(self, base_dir: str, recursive: bool, limit: int = 1):
        if not base_dir or not os.path.isdir(base_dir):
            return []

        candidates = []

        if recursive:
            iterator = os.walk(base_dir)
        else:
            iterator = [(base_dir, [], os.listdir(base_dir))]

        for root, _, files in iterator:
            for file_name in files:
                if not self._is_image_file(file_name):
                    continue
                full_path = os.path.join(root, file_name)
                if not os.path.isfile(full_path):
                    continue
                try:
                    mtime = os.path.getmtime(full_path)
                except OSError:
                    continue
                candidates.append((mtime, full_path))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        top_n = max(1, int(limit))
        return [path for _, path in candidates[:top_n]]

    def _find_latest_images(self, output_dir: str, limit: int) -> list:
        return self._find_latest_by_mtime(output_dir, recursive=True, limit=limit)

    def _default_download_dir(self) -> str:
        home = os.path.expanduser("~")
        return os.path.join(home, "Downloads")

    def _find_latest_download_images(self, download_dir: str, limit: int) -> list:
        return self._find_latest_by_mtime(download_dir, recursive=True, limit=limit)

    def _resolve_image_paths(self, download_dir: str, limit: int) -> list:
        output_dir = folder_paths.get_output_directory()
        target_download_dir = download_dir.strip() if download_dir else self._default_download_dir()
        latest_download = self._find_latest_download_images(target_download_dir, limit=limit)
        if latest_download:
            return latest_download
        return self._find_latest_images(output_dir, limit=limit)

    @classmethod
    def IS_CHANGED(
        cls,
        download_dir="",
    ):
        # 每次运行都强制刷新，重新扫描最新图片
        return time.time_ns()

    def load_preview(
        self,
        download_dir="",
        image_count=1,
    ):
        image_paths = self._resolve_image_paths(download_dir, limit=max(1, int(image_count)))
        if not image_paths:
            print("AI_DoubaoWebPreview: 未找到可输出的图片文件。")
            return self._empty_result()
        images = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"AI_DoubaoWebPreview: 文件不存在 -> {image_path}")
                continue
            try:
                image = Image.open(image_path).convert("RGB")
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                images.append(image_tensor)
            except Exception as e:
                print(f"AI_DoubaoWebPreview: 读取图片失败 -> {e}")
                continue

        if not images:
            return self._empty_result()
        task_completed = bool(_DOUBAO_BRIDGE_TASK_TRACKER.snapshot().get("task_completed", False))
        return (images, task_completed)

