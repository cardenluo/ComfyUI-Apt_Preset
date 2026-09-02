import folder_paths
from comfy import model_management
from comfy.utils import common_upscale
import torch
import numpy as np
import av
from PIL import Image
import base64
import io
import json
import hashlib
import asyncio
import re
import math
from datetime import datetime
from typing import Tuple
from collections.abc import Mapping
from fractions import Fraction
from server import PromptServer
from aiohttp import web
import os
import inspect
import subprocess
import nodes
import comfy.utils
import comfy.nested_tensor
from comfy_api.latest import InputImpl, Types
from comfy_execution.graph_utils import ExecutionBlocker



from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader



#region----------------lowcpu--------------------------



GIB = 1024 ** 3


def get_auto_reserved_vram(total_vram):
    if total_vram <= 8.0:
        return 0.6
    if total_vram <= 16.0:
        return 0.8
    return 1.0

class flow_low_gpu:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any_type, {}),
                "reserved": ("FLOAT", {
                    "default": 0.6,
                    "min": 0.0,
                    "max": 24.0,
                    "step": 0.1
                }),
                "mode": (["manual", "auto"], {
                    "default": "auto",
                    "display": "Mode"
                })
            },
            "hidden": {"unique_id": "UNIQUE_ID", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    OUTPUT_NODE = True
    FUNCTION = "set_vram"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def set_vram(self, anything, reserved, mode="auto", unique_id=None, extra_pnginfo=None):
        requested_reserved = max(0.0, reserved)
        device = model_management.get_torch_device()
        device_type = getattr(device, "type", None)

        if device_type in (None, "cpu", "mps"):
            final_reserved = requested_reserved
            print(f'flow_low_gpu: 未检测到独立GPU，使用预留值 {final_reserved:.2f}GB')
        else:
            total_vram = model_management.get_total_memory(device) / GIB
            max_reserved = max(0.0, total_vram - 0.8)
            if mode == "auto":
                auto_reserved = get_auto_reserved_vram(total_vram)
                final_reserved = min(max(requested_reserved, auto_reserved), max_reserved)
                print(f'flow_low_gpu: 自动显存预留生效 | 设备={device} | 总显存={total_vram:.2f}GB | 预留={final_reserved:.2f}GB')
            else:
                final_reserved = min(requested_reserved, max_reserved)
                print(f'flow_low_gpu: 手动显存预留生效 | 设备={device} | 预留={final_reserved:.2f}GB')

        model_management.EXTRA_RESERVED_VRAM = int(final_reserved * GIB)

        return (anything,)



#endregion----------------lowcpu--------------------------




#region----------------flow_bridge_image--------------------------

try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    class ExecutionBlocker:
        def __init__(self, value):
            self.value = value


import torch
import numpy as np
from PIL import Image, PngImagePlugin
import os
import folder_paths
import uuid
import json

lazy_options = {
    "lazy": True
}

ExecutionBlocker = None
try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    class ExecutionBlocker:
        def __init__(self, value):
            self.value = value


class flow_bridge_image:
    OUTPUT_NODE = True

    def __init__(self):
        self.temp_subfolder = "zml_image_memory"
        self.input_dir = folder_paths.get_input_directory()
        self.prompt = None
        self.extra_pnginfo = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disable_input": ("BOOLEAN", {"default": False}),
                "disable_output": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE", lazy_options),
                "mask": ("MASK", lazy_options),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "store_and_retrieve"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def IS_CHANGED(s, disable_input, disable_output, image=None, mask=None, unique_id=None, **kwargs):
        import hashlib
        subfolder_path = s._get_node_cache_dir(unique_id)
        if os.path.exists(subfolder_path):
            m = hashlib.sha256()
            for filename in sorted(os.listdir(subfolder_path)):
                if (
                    (filename.startswith("bridge_image_") and filename.endswith(".png"))
                    or (filename.startswith("bridge_mask_edit_") and filename.endswith(".png"))
                ):
                    filepath = os.path.join(subfolder_path, filename)
                    if os.path.isfile(filepath):
                        with open(filepath, 'rb') as f:
                            m.update(f.read())
                elif filename.endswith(".sourcehash"):
                    filepath = os.path.join(subfolder_path, filename)
                    if os.path.isfile(filepath):
                        with open(filepath, 'rb') as f:
                            m.update(f.read())
            return m.digest().hex()
        return ""

    def check_lazy_status(self, disable_input, **kwargs):
        required_inputs = []
        if not disable_input:
            if "image" in kwargs:
                required_inputs.append("image")
            if "mask" in kwargs:
                required_inputs.append("mask")
        return required_inputs

    def store_and_retrieve(self, disable_input, disable_output, image=None, mask=None, prompt=None, extra_pnginfo=None, unique_id=None):
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo

        subfolder_path = self._get_node_cache_dir(unique_id)
        os.makedirs(subfolder_path, exist_ok=True)

        image_to_output = None
        mask_to_output = None

        if disable_input:
            image_to_output, mask_to_output = self._load_from_local(subfolder_path)
        elif image is not None:
            # 未禁用输入时，缓存必须完全按上游 image/mask 刷新，不能保留本地编辑结果。
            self._clear_cache_files(subfolder_path)
            self._save_to_local(subfolder_path, image, mask)
            self._save_source_hash(subfolder_path, self._compute_source_hash(image, mask))
            image_to_output, mask_to_output = self._load_from_local(subfolder_path)
            if image_to_output is None:
                image_to_output = image
                mask_to_output = mask
        else:
            image_to_output, mask_to_output = self._load_from_local(subfolder_path)

        if image_to_output is None:
            default_size = 1
            image_to_output = torch.zeros((1, default_size, default_size, 3), dtype=torch.float32, device="cpu")

        if mask_to_output is None:
            batch_size, height, width, _ = image_to_output.shape
            mask_to_output = torch.ones((batch_size, height, width), dtype=torch.float32, device="cpu")

        self._save_preview_images(subfolder_path, image_to_output, mask_to_output)
        ui_image_data = self._build_ui_image_data(subfolder_path, unique_id)

        if disable_output and ExecutionBlocker is not None:
            output_image = ExecutionBlocker(None)
            output_mask = ExecutionBlocker(None)
        else:
            output_image = image_to_output
            output_mask = mask_to_output

        return {"ui": {"images": ui_image_data}, "result": (output_image, output_mask)}

    @classmethod
    def _get_node_cache_dir(cls, unique_id=None):
        node_folder = str(unique_id) if unique_id is not None else "default"
        safe_folder = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_folder)
        return os.path.join(folder_paths.get_input_directory(), "zml_image_memory", safe_folder)

    def _build_png_metadata(self):
        metadata = PngImagePlugin.PngInfo()
        if self.prompt is not None:
            try:
                metadata.add_text("prompt", json.dumps(self.prompt))
            except Exception:
                pass
        if self.extra_pnginfo is not None:
            for key, value in self.extra_pnginfo.items():
                try:
                    metadata.add_text(key, json.dumps(value))
                except Exception:
                    pass
        return metadata

    def _build_ui_image_data(self, subfolder_path, unique_id=None):
        ui_image_data = []
        relative_subfolder = os.path.join(self.temp_subfolder, self._get_node_cache_name(unique_id)).replace("\\", "/")
        preview_files = self._list_preview_images(subfolder_path)
        if not preview_files:
            preview_files = self._list_source_images(subfolder_path)
        for filename in preview_files:
            ui_image_data.append({"filename": filename, "subfolder": relative_subfolder, "type": "input"})
        return ui_image_data

    @classmethod
    def _get_node_cache_name(cls, unique_id=None):
        node_folder = str(unique_id) if unique_id is not None else "default"
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_folder)

    def _save_to_local(self, subfolder_path, image_tensor, mask_tensor):
        try:
            batch_size = image_tensor.shape[0]
            metadata = self._build_png_metadata()
            for i in range(batch_size):
                current_image = image_tensor[i:i+1]
                current_mask = mask_tensor[i:i+1] if mask_tensor is not None else None

                image_np = (current_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np).convert("RGB")

                save_path = os.path.join(subfolder_path, f"bridge_image_{i}.png")
                pil_image.save(save_path, "PNG", pnginfo=metadata, compress_level=4)

                if current_mask is not None:
                    mask_np = (current_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    if mask_np.ndim == 3:
                        mask_np = mask_np.squeeze(0)
                else:
                    mask_np = np.full((pil_image.height, pil_image.width), 255, dtype=np.uint8)
                mask_image = Image.fromarray(mask_np, mode='L')
                mask_save_path = os.path.join(subfolder_path, f"bridge_mask_edit_{i}.png")
                mask_image.save(mask_save_path, "PNG", compress_level=4)
        except Exception as e:
            print(f"Failed to save image locally: {e}")

    def _save_preview_images(self, subfolder_path, image_tensor, mask_tensor):
        try:
            self._remove_files_by_prefix(subfolder_path, "bridge_preview_")
            self._remove_files_by_prefix(subfolder_path, "bridge_editor_preview_")
            batch_size = image_tensor.shape[0]
            metadata = self._build_png_metadata()
            for i in range(batch_size):
                current_image = image_tensor[i:i+1]
                current_mask = mask_tensor[i:i+1] if mask_tensor is not None else None

                image_np = (current_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                pil_image = Image.fromarray(image_np).convert("RGB")
                if current_mask is not None:
                    mask_np = (current_mask.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    if mask_np.ndim == 3:
                        mask_np = mask_np.squeeze(0)
                    pil_mask = Image.fromarray(mask_np, mode='L')
                    if pil_mask.size != pil_image.size:
                        pil_mask = pil_mask.resize(pil_image.size, Image.NEAREST)
                    pil_image.putalpha(pil_mask)

                save_path = os.path.join(subfolder_path, f"bridge_preview_{i}.png")
                pil_image.save(save_path, "PNG", pnginfo=metadata, compress_level=4)

                editor_pil_image = Image.fromarray(image_np).convert("RGB")
                if current_mask is not None:
                    inverted_mask_np = 255 - mask_np
                    editor_mask = Image.fromarray(inverted_mask_np, mode='L')
                    if editor_mask.size != editor_pil_image.size:
                        editor_mask = editor_mask.resize(editor_pil_image.size, Image.NEAREST)
                    editor_pil_image.putalpha(editor_mask)

                editor_save_path = os.path.join(subfolder_path, f"bridge_editor_preview_{i}.png")
                editor_pil_image.save(editor_save_path, "PNG", pnginfo=metadata, compress_level=4)
        except Exception as e:
            print(f"[flow_bridge_image] Failed to save preview image locally: {e}")

    def _load_from_local(self, subfolder_path):
        try:
            if not os.path.exists(subfolder_path):
                return None, None

            source_files = self._list_source_images(subfolder_path)
            if not source_files:
                return None, None

            images = []
            masks = []

            for filename in source_files:
                file_path = os.path.join(subfolder_path, filename)
                with Image.open(file_path) as pil_image:
                    rgb_np = np.array(pil_image.convert("RGB")).astype(np.float32) / 255.0
                    images.append(rgb_np)

                    mask_index = self._extract_file_index(filename)
                    mask_path = os.path.join(subfolder_path, f"bridge_mask_edit_{mask_index}.png")
                    if os.path.exists(mask_path):
                        with Image.open(mask_path) as mask_image:
                            masks.append(self._extract_mask_array(mask_image))
                    else:
                        rgba_image = pil_image.convert("RGBA")
                        rgba_np = np.array(rgba_image).astype(np.float32) / 255.0
                        masks.append(rgba_np[:, :, 3])

            if images:
                image_tensor = torch.from_numpy(np.stack(images))
                mask_tensor = torch.from_numpy(np.stack(masks))
                return image_tensor, mask_tensor
        except Exception as e:
            print(f"[flow_bridge_image] Failed to load image from local file: {e}")
            import traceback
            traceback.print_exc()
        return None, None

    def _compute_tensor_hash(self, tensor):
        if tensor is None:
            return "none"
        import hashlib
        m = hashlib.sha256()
        m.update(str(tuple(tensor.shape)).encode("utf-8"))
        m.update(str(tensor.dtype).encode("utf-8"))
        m.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return m.digest().hex()

    def _compute_source_hash(self, image_tensor, mask_tensor):
        import hashlib
        m = hashlib.sha256()
        m.update(self._compute_tensor_hash(image_tensor).encode("utf-8"))
        m.update(self._compute_tensor_hash(mask_tensor).encode("utf-8"))
        return m.digest().hex()

    def _save_source_hash(self, subfolder_path, hash_value):
        try:
            hash_path = os.path.join(subfolder_path, "bridge_image.sourcehash")
            with open(hash_path, 'w') as f:
                f.write(hash_value)
        except Exception as e:
            print(f"[flow_bridge_image] Failed to save source hash: {e}")

    def _load_source_hash(self, subfolder_path):
        try:
            hash_path = os.path.join(subfolder_path, "bridge_image.sourcehash")
            if os.path.exists(hash_path):
                with open(hash_path, 'r') as f:
                    return f.read().strip()
        except Exception as e:
            print(f"[flow_bridge_image] Failed to load source hash: {e}")
        return ""

    def _list_source_images(self, subfolder_path):
        bridge_files = [f for f in os.listdir(subfolder_path) if f.startswith("bridge_image_") and f.endswith(".png")]
        bridge_files.sort(key=self._extract_file_index)
        return bridge_files

    def _list_preview_images(self, subfolder_path):
        preview_files = [f for f in os.listdir(subfolder_path) if f.startswith("bridge_preview_") and f.endswith(".png")]
        preview_files.sort(key=self._extract_file_index)
        return preview_files

    def _extract_mask_array(self, pil_image):
        rgba_image = pil_image.convert("RGBA")
        rgba_np = np.array(rgba_image).astype(np.float32) / 255.0
        alpha = rgba_np[:, :, 3]
        if float(alpha.max() - alpha.min()) > 1e-6 and not np.allclose(alpha, 1.0, atol=1e-4):
            return alpha

        rgb = rgba_np[:, :, :3]
        return rgb.max(axis=2)

    def _clear_cache_files(self, subfolder_path):
        self._remove_files_by_prefix(subfolder_path, "bridge_image_")
        self._remove_files_by_prefix(subfolder_path, "bridge_mask_edit_")
        self._remove_files_by_prefix(subfolder_path, "bridge_preview_")
        self._remove_files_by_prefix(subfolder_path, "bridge_editor_preview_")

    def _remove_files_by_prefix(self, subfolder_path, prefix):
        for filename in os.listdir(subfolder_path):
            if filename.startswith(prefix) and filename.endswith(".png"):
                try:
                    os.remove(os.path.join(subfolder_path, filename))
                except OSError as e:
                    print(f"[flow_bridge_image] Failed to remove cache file {filename}: {e}")

    @staticmethod
    def _extract_file_index(filename):
        stem = os.path.splitext(filename)[0]
        try:
            return int(stem.rsplit("_", 1)[-1])
        except ValueError:
            return 0


@PromptServer.instance.routes.post("/apt_preset/flow_bridge_image/save_edit")
async def apt_preset_flow_bridge_image_save_edit(request):
    try:
        reader = await request.multipart()
    except Exception:
        return web.json_response({"ok": False, "error": "请求格式错误，必须使用 multipart/form-data。"}, status=400)

    fields = {}
    image_bytes = None
    image_ref = None

    while True:
        part = await reader.next()
        if part is None:
            break
        if part.name == "image":
            image_bytes = await part.read(decode=False)
        elif part.name == "image_ref":
            image_ref = await part.text()
        else:
            fields[part.name] = await part.text()

    node_id = str(fields.get("node_id", "")).strip()
    if not node_id:
        return web.json_response({"ok": False, "error": "缺少 node_id。"}, status=400)
    if not image_bytes and not image_ref:
        return web.json_response({"ok": False, "error": "缺少编辑后的图片数据。"}, status=400)

    if image_ref and not image_bytes:
        try:
            ref_info = json.loads(image_ref)
            filename = str(ref_info.get("filename", "")).strip()
            subfolder = str(ref_info.get("subfolder", "")).strip().replace("\\", "/")
            if not filename:
                return web.json_response({"ok": False, "error": "image_ref 缺少 filename。"}, status=400)
            input_dir = os.path.abspath(folder_paths.get_input_directory())
            source_path = os.path.abspath(os.path.join(input_dir, subfolder, filename))
            if not source_path.startswith(input_dir):
                return web.json_response({"ok": False, "error": "image_ref 路径非法。"}, status=400)
            if not os.path.exists(source_path):
                return web.json_response({"ok": False, "error": "image_ref 指向的文件不存在。"}, status=400)
            with open(source_path, "rb") as f:
                image_bytes = f.read()
        except Exception as e:
            return web.json_response({"ok": False, "error": f"读取 image_ref 失败: {e}"}, status=400)

    # #region debug-point D:save-edit-received
    import urllib.request
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://127.0.0.1:7777/event",
            data=json.dumps({
                "sessionId": "mask-save-lag",
                "runId": "post-fix",
                "hypothesisId": "D",
                "location": "C_flow.py:apt_preset_flow_bridge_image_save_edit:received",
                "msg": "[DEBUG] 后端收到编辑后的图片上传",
                "data": {
                    "node_id": node_id,
                    "image_bytes": len(image_bytes),
                }
            }).encode(),
            headers={"Content-Type": "application/json"}
        )).read()
    except Exception:
        pass
    # #endregion

    cache_dir = flow_bridge_image._get_node_cache_dir(node_id)
    os.makedirs(cache_dir, exist_ok=True)

    try:
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            rgba_image = pil_image.convert("RGBA")
            rgba_np = np.array(rgba_image).astype(np.uint8)
            alpha = rgba_np[:, :, 3]
            if int(alpha.max()) != int(alpha.min()):
                mask_array = alpha
            else:
                mask_array = rgba_np[:, :, :3].max(axis=2).astype(np.uint8)
            mask_array = (255 - mask_array).astype(np.uint8)
            # #region debug-point D:save-edit-parsed
            try:
                urllib.request.urlopen(urllib.request.Request(
                    "http://127.0.0.1:7777/event",
                    data=json.dumps({
                        "sessionId": "mask-save-lag",
                        "runId": "post-fix",
                        "hypothesisId": "D",
                        "location": "C_flow.py:apt_preset_flow_bridge_image_save_edit:parsed",
                        "msg": "[DEBUG] 后端解析上传图片完成",
                        "data": {
                            "node_id": node_id,
                            "mode": pil_image.mode,
                            "size": list(pil_image.size),
                            "alpha_min": int(alpha.min()),
                            "alpha_max": int(alpha.max()),
                            "mask_min": int(mask_array.min()),
                            "mask_max": int(mask_array.max()),
                        }
                    }).encode(),
                    headers={"Content-Type": "application/json"}
                )).read()
            except Exception:
                pass
            # #endregion
            gray_image = Image.fromarray(mask_array, mode="L")
            for filename in os.listdir(cache_dir):
                if filename.startswith("bridge_mask_edit_") and filename.endswith(".png"):
                    try:
                        os.remove(os.path.join(cache_dir, filename))
                    except OSError as e:
                        print(f"[flow_bridge_image] Failed to remove cache file {filename}: {e}")

            save_path = os.path.join(cache_dir, "bridge_mask_edit_0.png")
            gray_image.save(save_path, "PNG", compress_level=4)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"保存编辑结果失败: {e}"}, status=500)

    safe_node_id = flow_bridge_image._get_node_cache_name(node_id)
    view_url = f"/view?filename=bridge_mask_edit_0.png&subfolder=zml_image_memory/{safe_node_id}&type=input"
    return web.json_response({"ok": True, "view_url": view_url})

#endregion----------    





class flow_case_tentor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "case_judge": (
                    ["横向图：宽>高，为True", 
                     "竖向图：高>宽，为True",  
                     "正方图：宽=高，为True", 
                     "分辨率>面积阈值,为True", 
                     "分辨率=面积阈值,为True",                     
                     "宽高比>比例阈值,为True", 
                     "宽高比=比例阈值,为True",
                     "长边>边阈值,为True",
                     "长边=边阈值,为True",
                     "短边>边阈值,为True",
                     "短边=边阈值,为True",
                     "高度>边阈值,为True",  
                     "高度=边阈值,为True",
                     "宽度>边阈值,为True",
                     "宽度=边阈值,为True",
                     "张量存在,为True",
                     "张量数量>批次阈值,为True",
                     "张量数量=批次阈值,为True",
                     ], ),  
                "area_threshold": ("STRING", {"default": "1048576.0", "tooltip": "支持加减乘除四则运算表达式，例如:1024*1024、(2000+500)/2"}),
                "ratio_threshold": ("STRING", {"default": "1.0", "tooltip": "支持加减乘除四则运算表达式，例如:16/9、4/3+0.2"}),
                "edge_threshold": ("INT", {"default": 1024, "min": 1, "max": 99999, "step": 1}),
                "batch_threshold": ("INT", {"default": 1, "min": 1, "max": 9999, "step": 1, "tooltip": "遮罩或图片或latent，批次数量"}),

            },
            "optional": {
                "data": (any_type,),
            }
        }  
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "check_event"
    CATEGORY = "Apt_Preset/flow"

    # 新增：安全解析表达式并返回float的核心方法
    def safe_calc_float(self, expr_str):
        if not expr_str or expr_str.strip() == "":
            return 0.0
        # 只保留 数字/+-*/().  过滤所有非法字符，保证安全执行
        safe_expr = ''.join([c for c in expr_str.strip() if c in '0123456789+-*/().'])
        try:
            # 执行表达式计算并强转float
            result = float(eval(safe_expr))
            return result if result >= 0 else 0.0
        except:
            # 表达式解析失败/计算报错，返回默认值
            return 0.0
    
    def check_event(self, case_judge, area_threshold,  batch_threshold, ratio_threshold, edge_threshold, data=None) -> Tuple[bool]:
        # ========== 核心修复1：空data(空图片) 直接返回 False，取消抛异常 ==========
        if data is None:
            return (False,)
        
        # 核心修改：解析文本表达式为float数值
        area_threshold_val = self.safe_calc_float(area_threshold)
        ratio_threshold_val = self.safe_calc_float(ratio_threshold)
            
        if case_judge == "横向图：宽>高，为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                result = width > height
        
        elif case_judge == "竖向图：高>宽，为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                result = height > width
        
        elif case_judge == "正方图：宽=高，为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                result = width == height
        
        elif case_judge == "分辨率>面积阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                resolution = width * height
                result = resolution > area_threshold_val
        
        elif case_judge == "分辨率=面积阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                resolution = width * height
                result = resolution == area_threshold_val
        
        elif case_judge == "宽高比>比例阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                if height == 0:
                    result = False
                else:
                    aspect_ratio = width / height
                    result = aspect_ratio > ratio_threshold_val
        
        elif case_judge == "宽高比=比例阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                if height == 0:
                    result = False
                else:
                    aspect_ratio = width / height
                    result = aspect_ratio == ratio_threshold_val
        
        elif case_judge == "长边>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                long_side = max(width, height)
                result = long_side > edge_threshold
        
        elif case_judge == "长边=边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                long_side = max(width, height)
                result = long_side == edge_threshold
        
        elif case_judge == "短边>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                short_side = min(width, height)
                result = short_side > edge_threshold
        
        elif case_judge == "短边=边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height, width = data.shape[1], data.shape[2]
                short_side = min(width, height)
                result = short_side == edge_threshold
        
        elif case_judge == "高度>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height = data.shape[1]
                result = height > edge_threshold
        
        elif case_judge == "高度=边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                height = data.shape[1]
                result = height == edge_threshold
        
        elif case_judge == "宽度>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                width = data.shape[2]
                result = width > edge_threshold
        
        elif case_judge == "宽度=边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                result = False
            else:
                width = data.shape[2]
                result = width == edge_threshold
        
        elif case_judge == "张量存在,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                result = False
            else:
                mask_sum = torch.sum(data).item()  
                result = mask_sum > 0  
        
        elif case_judge == "张量数量>批次阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                result = False
            else:
                batch_size = data.shape[0]  
                result = batch_size > batch_threshold
        
        elif case_judge == "张量数量=批次阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                result = False
            else:
                batch_size = data.shape[0]  
                result = batch_size == batch_threshold
        
        else:
            # ========== 核心修复2：未知判断模式 也返回 False，取消抛异常 ==========
            result = False
        
        return (result,)




class XXXXflow_sch_XXXcontrol:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                #"seed": ("INT", {"default": 0, "min": -999999, "max": 0xffffffffffffffff}),
                "total": ("INT", {"default": 10, "min": 0, "max": 5000} ),
                "种子": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "种子控制": (["随机", "固定", "递增"], {"default": "递增"}),
            },
            "optional": {
            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("seedIndex", "total",)
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        seed_control = kwargs.get("种子控制", "随机")
        seed = kwargs.get("种子", -1)
        if seed_control in ["随机", "递增"]:
            return float("nan")
        return seed
    def __init__(self):
        self.last_seed = -1
    def _effective_seed(self, seed: int, seed_control: str) -> int:
        import random
        if seed_control == "固定":
            effective_seed = seed if seed != -1 else random.randint(0, 2147483647)
        elif seed_control == "随机":
            effective_seed = random.randint(0, 2147483647)
        elif seed_control == "递增":
            if self.last_seed == -1:
                effective_seed = seed if seed != -1 else random.randint(0, 2147483647)
            else:
                effective_seed = self.last_seed + 1
        else:
            effective_seed = random.randint(0, 2147483647)
        self.last_seed = effective_seed
        return effective_seed

    def set_range(
        self,
        seed,
        total,
    ):
        
        value = seed + 1    
        return (value, total)




class flow_sch_control:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": -999999, "max": 0xffffffffffffffff}),
                "total": ("INT", {"default": 10, "min": 0, "max": 5000} ),
            },
            "optional": {
            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("INT", "INT",)
    RETURN_NAMES = ("seedIndex", "total",)
    CATEGORY = "Apt_Preset/flow/other"

    def set_range(
        self,
        seed,
        total,
    ):
        
        value = seed + 1    
        return (value, total)




class flow_QueueTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "Index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "total": ("INT", {"default": 10, "min": 1, "max": 0xffffffffffffffff}),
                    "mode": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                    },
                "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID"}
                }

    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("Index", "total")
    OUTPUT_NODE = True     
    NAME = "flow_QueueTrigger"


    def doit(self, Index, total, mode, unique_id):  
        if mode:
            if Index < total - 1:
                PromptServer.instance.send_sync("node-feedback",
                                                {"node_id": unique_id, "widget_name": "Index", "type": "int", "value": Index + 1})
                PromptServer.instance.send_sync("add-queue", {})
            elif Index >= total - 1:
                PromptServer.instance.send_sync("node-feedback",
                                                {"node_id": unique_id, "widget_name": "Index", "type": "int", "value": 0})

        return (Index, total)






class flow_tensor_Unify:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keep_alpha": ("BOOLEAN", {"default": False, "label_on": "4 Channels", "label_off": "3 Channels"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",)
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("unified_image", "unified_mask")
    FUNCTION = "unify_media"
    CATEGORY = "Apt_Preset/flow/other"
    
    def unify_media(self, keep_alpha=False, image=None, mask=None):
        if image is None:
            c = 4 if keep_alpha else 3
            unified_image = torch.zeros((1, 64, 64, c), dtype=torch.float32)
        else:
            img_np = image.cpu().numpy()
            b, h, w, c = img_np.shape
            
            if c == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
                c = 3
            elif c in [3,4]:
                pass
            elif b in [3,4] and c == 1:
                img_np = np.transpose(img_np, (1, 2, 0))[np.newaxis, ...]
                b, h, w, c = img_np.shape

            if img_np.dtype != np.float32:
                img_np = img_np.astype(np.float32) / 255.0 if img_np.max() > 1 else img_np.astype(np.float32)

            img_np = np.clip(img_np, 0.0, 1.0)

            if keep_alpha:
                if c == 3:
                    alpha_channel = np.ones((b, h, w, 1), dtype=img_np.dtype)
                    img_np = np.concatenate([img_np, alpha_channel], axis=-1)
            else:
                if c >= 3:
                    img_np = img_np[:, :, :, :3]

            unified_image = torch.from_numpy(img_np).to(image.device)

        if mask is None:
            unified_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        else:
            mask_np = mask.cpu().numpy()

            if len(mask_np.shape) == 4:
                mask_np = mask_np[..., 0]
            elif len(mask_np.shape) == 3 and mask_np.shape[-1] in [1,3,4]:
                mask_np = mask_np[..., 0]
            elif len(mask_np.shape) == 2:
                mask_np = mask_np[np.newaxis, ...]

            if mask_np.dtype != np.float32:
                mask_np = mask_np.astype(np.float32) / 255.0 if mask_np.max() > 1 else mask_np.astype(np.float32)

            mask_np = np.clip(mask_np, 0.0, 1.0)

            unified_mask = torch.from_numpy(mask_np).to(mask.device)

        return (unified_image, unified_mask)



#region--------------IN/out-switch--------------------------

class flow_BooleanSwitch:
    def __init__(self):
        self.stored_data = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "switch": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
                "store": ("BOOLEAN", {"default": True,}),
            },
            "optional": {
                "any_input": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(self, switch, store, any_input=None):
        if store and any_input is not None:
            self.stored_data = any_input

        if switch:
            if any_input is not None:
                return (any_input,)
            elif store and self.stored_data is not None:
                return (self.stored_data,)
            else:
                if ExecutionBlocker is not None:
                    return (ExecutionBlocker(None),)
                else:
                    return ({},)
        else:
            if ExecutionBlocker is not None:
                return (ExecutionBlocker(None),)
            else:
                return ({},)


class flow_stage_index_switch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_index": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "open_stage_index": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
            },
            "optional": {
                "any_input": (any_type, {"lazy": True}),
            },
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check_lazy_status(self, stage_index, open_stage_index, any_input=None):
        if stage_index == open_stage_index and any_input is None:
            return ["any_input"]
        return []

    def process(self, stage_index, open_stage_index, any_input=None):
        if stage_index == open_stage_index and any_input is not None:
            return (any_input,)

        if ExecutionBlocker is not None:
            return (ExecutionBlocker(None),)
        return ({},)


def _frame_slice_indices(length, start_frame, end_frame, device=None):
    if length < 1:
        raise ValueError("flow_frame_slice: input contains no frames")

    def normalize(index):
        index = int(index)
        if index < 0:
            index += length
        return max(0, min(length - 1, index))

    start = normalize(start_frame)
    end = normalize(end_frame)
    step = 1 if start <= end else -1
    return torch.arange(start, end + step, step, device=device, dtype=torch.long)


def _frame_slice_tensor(tensor, indices, dim=0):
    return torch.index_select(tensor, dim, indices.to(tensor.device)).clone()


class flow_frame_slice:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_frame": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
                "end_frame": ("INT", {"default": -1, "min": -2147483648, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "VIDEO")
    RETURN_NAMES = ("image", "latent", "video")
    FUNCTION = "slice_frames"
    CATEGORY = "Apt_Preset/flow"

    def slice_frames(self, start_frame=0, end_frame=-1, image=None, latent=None, video=None):
        image_out = ExecutionBlocker(None)
        latent_out = ExecutionBlocker(None)
        video_out = ExecutionBlocker(None)

        if image is not None:
            indices = _frame_slice_indices(image.shape[0], start_frame, end_frame, image.device)
            image_out = _frame_slice_tensor(image, indices)

        if latent is not None:
            samples = latent.get("samples")
            if not isinstance(samples, torch.Tensor):
                raise TypeError("flow_frame_slice: latent samples must be a tensor")
            indices = _frame_slice_indices(samples.shape[0], start_frame, end_frame, samples.device)
            latent_out = latent.copy()
            latent_out["samples"] = _frame_slice_tensor(samples, indices)

            noise_mask = latent.get("noise_mask")
            if isinstance(noise_mask, torch.Tensor):
                if noise_mask.shape[0] == samples.shape[0]:
                    latent_out["noise_mask"] = _frame_slice_tensor(noise_mask, indices)
                else:
                    latent_out["noise_mask"] = noise_mask.clone()

            batch_index = latent.get("batch_index")
            if isinstance(batch_index, (list, tuple)) and len(batch_index) == samples.shape[0]:
                latent_out["batch_index"] = [batch_index[index] for index in indices.cpu().tolist()]

        if video is not None:
            components = video.get_components()
            images = components.images
            indices = _frame_slice_indices(images.shape[0], start_frame, end_frame, images.device)
            video_images = _frame_slice_tensor(images, indices)

            alpha = components.alpha
            if isinstance(alpha, torch.Tensor):
                if alpha.ndim > 0 and alpha.shape[0] == images.shape[0]:
                    alpha = _frame_slice_tensor(alpha, indices)
                else:
                    alpha = alpha.clone()

            audio = components.audio
            if audio is not None:
                frame_rate = float(components.frame_rate)
                if frame_rate <= 0:
                    raise ValueError("flow_frame_slice: video frame rate must be greater than zero")
                waveform = audio.get("waveform")
                sample_rate = int(audio.get("sample_rate", 0))
                if isinstance(waveform, torch.Tensor) and sample_rate > 0:
                    selected = indices.cpu()
                    first_frame = int(selected.min().item())
                    last_frame = int(selected.max().item()) + 1
                    sample_start = round(first_frame / frame_rate * sample_rate)
                    sample_end = round(last_frame / frame_rate * sample_rate)
                    waveform = waveform[..., sample_start:sample_end].clone()
                    if int(selected[0]) > int(selected[-1]):
                        waveform = torch.flip(waveform, dims=(-1,))
                    audio = dict(audio)
                    audio["waveform"] = waveform

            video_components = Types.VideoComponents(
                images=video_images,
                alpha=alpha,
                audio=audio,
                frame_rate=components.frame_rate,
                metadata=components.metadata,
            )
            bit_depth = video.get_bit_depth() if hasattr(video, "get_bit_depth") else 8
            video_out = InputImpl.VideoFromComponents(video_components, bit_depth=bit_depth)

        return image_out, latent_out, video_out



class flow_judge_output:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (any_type, {}),
                "judge": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (any_type, any_type)
    RETURN_NAMES = ("true", "false")
    FUNCTION = "judge_output"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = False

    def judge_output(self, data, judge=True):
        # 根据judge布尔值判断输出端口
        if judge:
            true_output = data
            false_output = ExecutionBlocker(None) if ExecutionBlocker is not None else {}
        else:
            true_output = ExecutionBlocker(None) if ExecutionBlocker is not None else {}
            false_output = data
            
        return {"ui": {"value": [judge]}, "result": (true_output, false_output)}


class flow_judge_input:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "judge": ("BOOLEAN", {"default": True, "label_on": "✅ True", "label_off": "❌ False"}), # 美化开关文字
            },
            "optional": {
                "true": (any_type, {"lazy": True}),
                "false": (any_type, {"lazy": True}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("data",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = False

    # 懒加载校验逻辑不变
    def check_lazy_status(self, judge, true=None, false=None):
        needed = []
        if judge:
            if true is None:
                needed.append('true')
        else:
            if false is None:
                needed.append('false')
        return needed

    def execute(self, judge, true=None, false=None):
        if judge:
            result_value = true if true is not None else false
        else:
            result_value = false if false is not None else true
            
        # 空值兜底不变
        if result_value is None:
            try:
                from nodes import ExecutionBlocker # 显式导入，兼容性更强
                result_value = ExecutionBlocker(None)
            except:
                result_value = {}
        
        return (result_value,)



class flow_switch_output:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_input": (any_type, {}),
                "index": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            }
        }

    RETURN_TYPES = (any_type, any_type, any_type, any_type, any_type)
    RETURN_NAMES = ("output_1", "output_2", "output_3", "output_4", "output_5")
    FUNCTION = "switch_output"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = False

    def switch_output(self, any_input, index=1):
        outputs = []
        for i in range(5):
            if i == index - 1:  
                outputs.append(any_input)
            else: 
                if ExecutionBlocker is not None:
                    outputs.append(ExecutionBlocker(None))
                else:
                    outputs.append({})
        
        return tuple(outputs)



class flow_switch_input:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_method": ("BOOLEAN", {"default": True, "label_on": "第一个有效值", "label_off": "按编号"}),
                "input_index": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
            },
            "optional": {
                "in1": (any_type,),
                "in2": (any_type,),
                "in3": (any_type,),
                "in4": (any_type,),
                "in5": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ('out',)
    CATEGORY = "Apt_Preset/flow"
    FUNCTION = "switch"

    def switch(self, input_method, input_index,
               in1=None, in2=None, in3=None, in4=None, in5=None):
        inputs = [in1, in2, in3, in4, in5]
        
        if input_method:
            selected_value = None
            for value in inputs:
                if not self.is_none(value):
                    selected_value = value
                    break
        else:
            index = input_index - 1
            if 0 <= index < len(inputs):
                selected_value = inputs[index]
            else:
                selected_value = None
        
        if selected_value is None:
            for value in inputs:
                if value is not None:
                    selected_value = value
                    break
    
        if selected_value is None:
            if ExecutionBlocker is not None:
                return (ExecutionBlocker(None),)
            else:
                return ({},)
        
        return (selected_value,)

    def is_none(self, value):
        if value is not None:
            if isinstance(value, dict) and 'model' in value and 'clip' in value:
                return all(v is None for v in value.values())
        return value is None


#endregion----------------IN/out-switch--------------------------





#region---------------loop team-------------


class AlwaysEqualProxy(str):
    def __eq__(self, _): return True
    def __ne__(self, _): return False

any_type = AlwaysEqualProxy("*")
def ByPassTypeTuple(t): return t


_STAGE_BRIDGE_VERSION = 1
_STAGE_BRIDGE_TYPES = ["auto", "latent", "image", "mask", "video", "audio", "tensor", "json"]
_STAGE_INFO_TYPE = "FLOW_STAGE_INFO"
_STAGE_ACTIVE_RUN_IDS = {}
_STAGE_BEGIN_NODE_IDS = {}


def _stage_safe_name(value):
    value = str(value or "default").strip() or "default"
    readable = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:48]
    return f"{readable}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:10]}"


def _stage_root_dir():
    root = os.path.abspath(os.path.join(folder_paths.get_output_directory(), ".apt_stage_bridge"))
    os.makedirs(root, exist_ok=True)
    return root


def _stage_run_path(run_id):
    root = _stage_root_dir()
    path = os.path.abspath(os.path.join(root, _stage_safe_name(run_id)))
    if os.path.commonpath((root, path)) != root:
        raise ValueError("flow_stage: invalid run_id")
    return path


def _stage_run_dir(run_id):
    path = _stage_run_path(run_id)
    os.makedirs(path, exist_ok=True)
    return path


def _stage_new_run_id():
    base = datetime.now().strftime("%y%m%d%H%M%S")
    root = _stage_root_dir()
    candidate = base
    suffix = 0
    while os.path.exists(os.path.join(root, _stage_safe_name(candidate))):
        suffix += 1
        candidate = f"{base}_{suffix:03d}"
    return candidate


def _stage_feedback(unique_id, widget_name, value):
    if unique_id is None:
        return
    try:
        server = PromptServer.instance
        PromptServer.instance.send_sync(
            "node-feedback",
            {"node_id": unique_id, "widget_name": widget_name, "type": "value", "value": value},
            server.client_id,
        )
    except Exception:
        pass


def _stage_state_path(run_dir):
    return os.path.join(run_dir, "state.json")


def _stage_load_state(run_dir):
    path = _stage_state_path(run_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("version") != _STAGE_BRIDGE_VERSION:
        raise ValueError("flow_stage: unsupported state version")
    return state


def _stage_write_json(path, value):
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def _stage_json_value(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_stage_json_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _stage_json_value(item) for key, item in value.items()}
    raise TypeError(f"flow_stage: {type(value).__name__} cannot be stored as JSON")


def _stage_cpu_tensor(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"flow_stage: expected a tensor, got {type(value).__name__}")
    return value.detach().to(device="cpu").contiguous()


def _stage_detect_type(data, requested):
    if requested != "auto":
        return requested
    if hasattr(data, "get_components"):
        return "video"
    if isinstance(data, Mapping) and "samples" in data:
        return "latent"
    if isinstance(data, Mapping) and isinstance(data.get("latent"), Mapping) and "samples" in data["latent"]:
        return "latent"
    if isinstance(data, Mapping) and "waveform" in data:
        return "audio"
    if isinstance(data, torch.Tensor):
        if data.ndim == 4 and data.shape[-1] in (1, 3, 4):
            return "image"
        if data.ndim == 3:
            return "mask"
        return "tensor"
    return "json"


def _stage_encode_payload(data, requested_type):
    payload_type = _stage_detect_type(data, requested_type)
    if payload_type == "latent" and isinstance(data, Mapping) and "samples" not in data:
        data = data.get("latent")
    elif payload_type == "image" and isinstance(data, Mapping) and isinstance(data.get("images"), torch.Tensor):
        data = data["images"]
    tensors = {"stage_format": torch.empty(0, dtype=torch.uint8)}
    descriptor = {"version": _STAGE_BRIDGE_VERSION, "type": payload_type}

    if payload_type == "latent":
        if not isinstance(data, Mapping) or "samples" not in data:
            raise TypeError("flow_stage: latent data must contain samples")
        samples = data["samples"]
        if isinstance(samples, comfy.nested_tensor.NestedTensor):
            names = []
            for index, tensor in enumerate(samples.unbind()):
                name = f"samples_{index}"
                tensors[name] = _stage_cpu_tensor(tensor)
                names.append(name)
            descriptor["samples"] = {"nested": True, "names": names}
        else:
            tensors["samples"] = _stage_cpu_tensor(samples)
            descriptor["samples"] = {"nested": False, "names": ["samples"]}

        fields = []
        for index, (key, value) in enumerate(data.items()):
            if key == "samples":
                continue
            if isinstance(value, comfy.nested_tensor.NestedTensor):
                names = []
                for nested_index, tensor in enumerate(value.unbind()):
                    name = f"field_{index}_{nested_index}"
                    tensors[name] = _stage_cpu_tensor(tensor)
                    names.append(name)
                fields.append({"key": str(key), "nested_tensors": names})
            elif isinstance(value, torch.Tensor):
                name = f"field_{index}"
                tensors[name] = _stage_cpu_tensor(value)
                fields.append({"key": str(key), "tensor": name})
            else:
                fields.append({"key": str(key), "value": _stage_json_value(value)})
        descriptor["fields"] = fields

    elif payload_type in ("image", "mask", "tensor"):
        tensors["data"] = _stage_cpu_tensor(data)

    elif payload_type == "audio":
        if not isinstance(data, Mapping) or "waveform" not in data:
            raise TypeError("flow_stage: audio data must contain waveform")
        tensors["waveform"] = _stage_cpu_tensor(data["waveform"])
        descriptor["sample_rate"] = int(data["sample_rate"])

    elif payload_type == "video":
        if not hasattr(data, "get_components"):
            raise TypeError("flow_stage: video data must provide get_components()")
        components = data.get_components()
        tensors["images"] = _stage_cpu_tensor(components.images)
        if components.alpha is not None:
            tensors["alpha"] = _stage_cpu_tensor(components.alpha)
            descriptor["alpha"] = True
        audio = components.audio
        if audio is not None:
            tensors["audio_waveform"] = _stage_cpu_tensor(audio["waveform"])
            descriptor["audio_sample_rate"] = int(audio["sample_rate"])
        frame_rate = Fraction(components.frame_rate)
        descriptor["frame_rate"] = [frame_rate.numerator, frame_rate.denominator]
        descriptor["bit_depth"] = int(data.get_bit_depth()) if hasattr(data, "get_bit_depth") else 8
        if components.metadata is not None:
            descriptor["metadata"] = _stage_json_value(components.metadata)

    elif payload_type == "json":
        descriptor["value"] = _stage_json_value(data)
    else:
        raise ValueError(f"flow_stage: unsupported data type {payload_type}")

    return tensors, descriptor


def _stage_decode_payload(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"flow_stage: checkpoint not found: {path}")
    tensors, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
    if metadata is None or "stage_payload" not in metadata:
        raise ValueError("flow_stage: checkpoint metadata is missing")
    descriptor = json.loads(metadata["stage_payload"])
    if descriptor.get("version") != _STAGE_BRIDGE_VERSION:
        raise ValueError("flow_stage: unsupported checkpoint version")
    payload_type = descriptor["type"]

    if payload_type == "latent":
        sample_info = descriptor["samples"]
        sample_tensors = [tensors[name] for name in sample_info["names"]]
        samples = comfy.nested_tensor.NestedTensor(sample_tensors) if sample_info["nested"] else sample_tensors[0]
        data = {"samples": samples}
        for field in descriptor.get("fields", []):
            if "nested_tensors" in field:
                data[field["key"]] = comfy.nested_tensor.NestedTensor(
                    [tensors[name] for name in field["nested_tensors"]]
                )
            else:
                data[field["key"]] = tensors[field["tensor"]] if "tensor" in field else field.get("value")
        return data
    if payload_type in ("image", "mask", "tensor"):
        return tensors["data"]
    if payload_type == "audio":
        return {"waveform": tensors["waveform"], "sample_rate": int(descriptor["sample_rate"])}
    if payload_type == "video":
        audio = None
        if "audio_waveform" in tensors:
            audio = {
                "waveform": tensors["audio_waveform"],
                "sample_rate": int(descriptor["audio_sample_rate"]),
            }
        numerator, denominator = descriptor["frame_rate"]
        components = Types.VideoComponents(
            images=tensors["images"],
            alpha=tensors.get("alpha"),
            audio=audio,
            frame_rate=Fraction(numerator, denominator),
            metadata=descriptor.get("metadata"),
        )
        return InputImpl.VideoFromComponents(components, bit_depth=int(descriptor.get("bit_depth", 8)))
    if payload_type == "json":
        return descriptor.get("value")
    raise ValueError(f"flow_stage: unsupported checkpoint type {payload_type}")


def _stage_checkpoint_filename(stage_index, channel):
    suffix = "1" if channel == "data1" else "2"
    return f"stage_{int(stage_index):05d}_checkpoint_{suffix}.safetensors"


def _stage_load_checkpoint(run_dir, stage_index, channel):
    path = os.path.join(run_dir, _stage_checkpoint_filename(stage_index, channel))
    return _stage_decode_payload(path) if os.path.isfile(path) else None


class flow_stage_begin:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "run_id": ("STRING", {"default": "default"}),
                "total": ("INT", {"default": 3, "min": 1, "max": 5000}),
                "stage_index": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 5000,
                    "tooltip": "当前阶段（1～总阶段数）；可手动选择断点阶段，完成后自动回到1",
                }),
            },
            "optional": {
                "initial_data_1": (any_type,),
                "initial_data_2": (any_type,),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (_STAGE_INFO_TYPE, "INT")
    RETURN_NAMES = ("stage_info", "stage_index")
    FUNCTION = "begin"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def IS_CHANGED(cls, run_id, total, stage_index=1, unique_id=None, **kwargs):
        node_key = str(unique_id or "")
        effective_run_id = str(run_id or "").strip()
        single_stage = int(total) == 1 and int(stage_index) == 1
        if single_stage:
            effective_run_id = "default"
        elif int(stage_index) == 1:
            effective_run_id = ""
        elif (not effective_run_id or effective_run_id == "default") and node_key:
            effective_run_id = str(_STAGE_ACTIVE_RUN_IDS.get(node_key) or "")

        files = []
        if effective_run_id and not single_stage:
            run_dir = _stage_run_path(effective_run_id)
            state_path = _stage_state_path(run_dir)
            if os.path.isfile(state_path):
                state_stat = os.stat(state_path)
                files.append((os.path.basename(state_path), state_stat.st_mtime_ns, state_stat.st_size))
                state = _stage_load_state(run_dir)
                if state is not None:
                    requested_stage = int(stage_index) - 1
                    if requested_stage > 0:
                        for suffix in ("", "_2"):
                            payload_path = os.path.join(run_dir, f"stage_{requested_stage - 1:05d}{suffix}.safetensors")
                            if os.path.isfile(payload_path):
                                payload_stat = os.stat(payload_path)
                                files.append((os.path.basename(payload_path), payload_stat.st_mtime_ns, payload_stat.st_size))
            requested_stage = int(stage_index) - 1
            for channel in ("data1", "data2"):
                checkpoint_path = os.path.join(run_dir, _stage_checkpoint_filename(requested_stage, channel))
                if os.path.isfile(checkpoint_path):
                    checkpoint_stat = os.stat(checkpoint_path)
                    files.append((os.path.basename(checkpoint_path), checkpoint_stat.st_mtime_ns, checkpoint_stat.st_size))

        return json.dumps(
            [effective_run_id, int(total), int(stage_index), files],
            separators=(",", ":"),
        )

    def begin(self, run_id, total, stage_index=1,
              initial_data_1=None, initial_data_2=None, unique_id=None):
        total = int(total)
        requested_index = int(stage_index)
        if requested_index < 1 or requested_index > total:
            raise ValueError(f"flow_stage_begin: stage_index must be between 1 and {total}")

        node_key = str(unique_id or "")
        effective_run_id = str(run_id or "").strip()
        single_stage = total == 1 and requested_index == 1
        if single_stage:
            effective_run_id = "default"
        elif requested_index == 1:
            effective_run_id = ""
        elif (not effective_run_id or effective_run_id == "default") and node_key:
            effective_run_id = str(_STAGE_ACTIVE_RUN_IDS.get(node_key) or "")

        state = None
        run_dir = None
        if effective_run_id:
            run_dir = _stage_run_dir(effective_run_id)
            state = _stage_load_state(run_dir)

        stage_index = requested_index - 1
        expanding_total = False
        if stage_index == 0:
            if not single_stage:
                effective_run_id = _stage_new_run_id()
                run_dir = _stage_run_dir(effective_run_id)
                state = None
            data_1 = initial_data_1
            data_2 = initial_data_2
        else:
            if not effective_run_id or state is None:
                raise ValueError(
                    "flow_stage_begin: cannot resume from this stage because the previous run checkpoint is missing"
                )
            saved_total = int(state["total"])
            if total < saved_total:
                raise ValueError("flow_stage_begin: total cannot be smaller than the saved run")
            expanding_total = total > saved_total
            if expanding_total and stage_index != int(state["next_stage"]):
                raise ValueError(
                    f"flow_stage_begin: expanding total must continue from stage {int(state['next_stage']) + 1}"
                )
            if not state.get("complete", False) and int(state["next_stage"]) == stage_index:
                filename = state.get("payload")
            else:
                filename = f"stage_{stage_index - 1:05d}.safetensors"
            if filename is None:
                if not state.get("control_only_1", state.get("control_only", False)):
                    raise ValueError("flow_stage_begin: previous stage checkpoint is missing")
                data_1 = None
            else:
                if not filename or os.path.basename(filename) != filename:
                    raise ValueError("flow_stage_begin: invalid checkpoint filename")
                payload_path = os.path.join(run_dir, filename)
                if os.path.isfile(payload_path):
                    data_1 = _stage_decode_payload(payload_path)
                elif state.get("control_only_1", state.get("control_only", False)):
                    filename = None
                    data_1 = None
                else:
                    raise FileNotFoundError(f"flow_stage_begin: checkpoint not found: {payload_path}")
            if not state.get("complete", False) and int(state["next_stage"]) == stage_index:
                filename_2 = state.get("payload_2")
            else:
                filename_2 = f"stage_{stage_index - 1:05d}_2.safetensors"
            if filename_2 is None:
                data_2 = None
            else:
                if not filename_2 or os.path.basename(filename_2) != filename_2:
                    raise ValueError("flow_stage_begin: invalid bridge 2 checkpoint filename")
                payload_path_2 = os.path.join(run_dir, filename_2)
                data_2 = _stage_decode_payload(payload_path_2) if os.path.isfile(payload_path_2) else None
            if expanding_total or state.get("complete", False) or int(state["next_stage"]) != stage_index:
                state = {
                    "version": _STAGE_BRIDGE_VERSION,
                    "run_id": effective_run_id,
                    "total": total,
                    "completed_stage": stage_index - 1,
                    "next_stage": stage_index,
                    "payload": filename,
                    "payload_2": filename_2,
                    "payload_type": "auto" if filename is not None else None,
                    "control_only": filename is None and filename_2 is None,
                    "control_only_1": filename is None,
                    "control_only_2": filename_2 is None,
                    "complete": False,
                    "restart_pending": True,
                }
                _stage_write_json(_stage_state_path(run_dir), state)

        checkpoint_data_1 = _stage_load_checkpoint(run_dir, stage_index, "data1")
        checkpoint_data_2 = _stage_load_checkpoint(run_dir, stage_index, "data2")

        if node_key:
            _STAGE_ACTIVE_RUN_IDS[node_key] = effective_run_id
            _STAGE_BEGIN_NODE_IDS[effective_run_id] = node_key
        _stage_feedback(unique_id, "run_id", effective_run_id)
        _stage_feedback(unique_id, "stage_index", stage_index + 1)

        stage_info = {
            "version": _STAGE_BRIDGE_VERSION,
            "run_id": effective_run_id,
            "stage_index": stage_index,
            "total": total,
            "is_first": stage_index == 0,
            "is_last": stage_index == total - 1,
            "stage_data": data_1,
            "stage_data_1": data_1,
            "stage_data_2": data_2,
            "checkpoint_data_1": checkpoint_data_1,
            "checkpoint_data_2": checkpoint_data_2,
        }
        return stage_info, stage_index + 1


def _stage_validate_info(stage_info):
    if not isinstance(stage_info, Mapping):
        raise TypeError("flow_stage: stage_info must come from flow_stage_begin")
    if int(stage_info.get("version", -1)) != _STAGE_BRIDGE_VERSION:
        raise ValueError("flow_stage: unsupported stage_info version")
    run_id = str(stage_info.get("run_id") or "").strip()
    stage_index = int(stage_info.get("stage_index", -1))
    total = int(stage_info.get("total", 0))
    if not run_id or total < 1 or stage_index < 0 or stage_index >= total:
        raise ValueError("flow_stage: invalid stage_info")
    return run_id, stage_index, total


class flow_stage_unpack:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_info": (_STAGE_INFO_TYPE,),
            },
        }

    RETURN_TYPES = (any_type, any_type, "INT", "INT", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("data_1", "data_2", "total", "stage_index", "is_first", "is_last")
    FUNCTION = "unpack"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def unpack(self, stage_info):
        _, stage_index, total = _stage_validate_info(stage_info)
        data_1 = stage_info.get("stage_data_1", stage_info.get("stage_data"))
        data_2 = stage_info.get("stage_data_2")
        return (
            data_1 if data_1 is not None else ExecutionBlocker(None),
            data_2 if data_2 is not None else ExecutionBlocker(None),
            total,
            stage_index + 1,
            stage_index == 0,
            stage_index == total - 1,
        )


def _stage_save_checkpoint_data(stage_info, data, bridge):
    run_id, stage_index, _total = _stage_validate_info(stage_info)
    if bridge not in ("data1", "data2"):
        raise ValueError(f"flow_stage_end: invalid bridge: {bridge}")

    tensors, descriptor = _stage_encode_payload(data, "auto")
    run_dir = _stage_run_dir(run_id)
    filename = _stage_checkpoint_filename(stage_index, bridge)
    path = os.path.join(run_dir, filename)
    temp_path = path + ".tmp"
    try:
        comfy.utils.save_torch_file(
            tensors,
            temp_path,
            metadata={"stage_payload": json.dumps(descriptor, ensure_ascii=False)},
        )
        os.replace(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
def _stage_list_collect(run_dir, total, suffix=""):
    items = []
    for i in range(total):
        path = os.path.join(run_dir, f"stage_{i:05d}{suffix}.safetensors")
        if os.path.isfile(path):
            items.append(_stage_decode_payload(path))
    return items


class flow_stage_end:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_info": (_STAGE_INFO_TYPE,),
                "unload_models": ("BOOLEAN", {"default": False}),
                "free_memory": ("BOOLEAN", {"default": True}),
                "free_memory_interval": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "data_1": (any_type, {"lazy": True}),
                "data_2": (any_type, {"lazy": True}),
            },
        }

    RETURN_TYPES = (any_type, any_type)
    RETURN_NAMES = ("list_data1", "list_data2")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "commit"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_lazy_status(self, stage_info, **kwargs):
        if "data_1" in kwargs and kwargs["data_1"] is None:
            return ["data_1"]
        if "data_2" in kwargs and kwargs["data_2"] is None:
            if "data_1" in kwargs:
                _stage_save_checkpoint_data(stage_info, kwargs["data_1"], "data1")
            return ["data_2"]
        return []

    def commit(self, data_1=None, data_2=None, stage_info=None, unload_models=False, free_memory=True, free_memory_interval=1):
        run_id, stage_index, total = _stage_validate_info(stage_info)

        channel_1 = data_1.get("apt_h3_bridge_channel") if isinstance(data_1, Mapping) else None
        channel_2 = data_2.get("apt_h3_bridge_channel") if isinstance(data_2, Mapping) else None
        if channel_2 == "data2" and channel_1 != "data1":
            raise ValueError(
                "flow_stage_end: H3 data_1 is missing or invalid; connect "
                "the first-pass generate context to Data_basic.context, then connect "
                "Data_basic.latent to flow_stage_end.data_1"
            )

        run_dir = _stage_run_dir(run_id)
        state = _stage_load_state(run_dir)
        if stage_index == 0:
            restart_pending = bool(
                state is not None
                and state.get("restart_pending", False)
                and int(state.get("next_stage", -1)) == 0
            )
            if state is not None and not state.get("complete", False) and not restart_pending:
                raise ValueError("flow_stage_end: an active run already exists for this run_id")
        else:
            if state is None or state.get("complete", False):
                raise ValueError("flow_stage_end: previous stage state is missing")
            if int(state["total"]) != total or int(state["next_stage"]) != stage_index:
                raise ValueError("flow_stage_end: stage order does not match the saved state")

        def write_bridge(data, suffix):
            if data is None:
                return None, None, None
            tensors, descriptor = _stage_encode_payload(data, "auto")
            filename = f"stage_{stage_index:05d}{suffix}.safetensors"
            temp_path = os.path.join(run_dir, filename + ".tmp")
            try:
                comfy.utils.save_torch_file(
                    tensors,
                    temp_path,
                    metadata={"stage_payload": json.dumps(descriptor, ensure_ascii=False)},
                )
            except Exception:
                if os.path.isfile(temp_path):
                    os.remove(temp_path)
                raise
            return filename, descriptor["type"], temp_path

        temp_path_1 = None
        temp_path_2 = None
        try:
            filename, payload_type, temp_path_1 = write_bridge(data_1, "")
            filename_2, payload_type_2, temp_path_2 = write_bridge(data_2, "_2")
            if temp_path_1 is not None:
                os.replace(temp_path_1, os.path.join(run_dir, filename))
            if temp_path_2 is not None:
                os.replace(temp_path_2, os.path.join(run_dir, filename_2))
        finally:
            for temp_path in (temp_path_1, temp_path_2):
                if temp_path is not None and os.path.isfile(temp_path):
                    os.remove(temp_path)

        complete = stage_index == total - 1
        next_state = {
            "version": _STAGE_BRIDGE_VERSION,
            "run_id": str(run_id),
            "total": total,
            "completed_stage": stage_index,
            "next_stage": stage_index + 1,
            "payload": filename,
            "payload_2": filename_2,
            "payload_type": payload_type,
            "payload_type_2": payload_type_2,
            "control_only": data_1 is None and data_2 is None,
            "control_only_1": data_1 is None,
            "control_only_2": data_2 is None,
            "complete": complete,
        }
        _stage_write_json(_stage_state_path(run_dir), next_state)
        for channel in ("data1", "data2"):
            checkpoint_path = os.path.join(run_dir, _stage_checkpoint_filename(stage_index, channel))
            if os.path.isfile(checkpoint_path):
                os.remove(checkpoint_path)

        begin_node_id = _STAGE_BEGIN_NODE_IDS.get(run_id)
        _stage_feedback(begin_node_id, "stage_index", 1 if complete else stage_index + 2)
        if complete:
            _STAGE_BEGIN_NODE_IDS.pop(run_id, None)

        PromptServer.instance.prompt_queue.set_flag("unload_models", bool(unload_models))
        should_free_memory = free_memory and (stage_index + 1) % free_memory_interval == 0
        PromptServer.instance.prompt_queue.set_flag("free_memory", should_free_memory)

        list_data1 = ExecutionBlocker(None)
        list_data2 = ExecutionBlocker(None)
        if complete:
            list_data1 = _stage_list_collect(run_dir, total)
            list_data2 = _stage_list_collect(run_dir, total, "_2")

        if not complete:
            server = PromptServer.instance
            server.send_sync("add-queue", {}, server.client_id)

        if filename is None and filename_2 is None:
            message = f"stage {stage_index + 1}/{total} completed (control only)"
        else:
            saved = ", ".join(name for name in (filename, filename_2) if name is not None)
            message = f"stage {stage_index + 1}/{total} saved: {saved}"
        return {"ui": {"text": [message]}, "result": (list_data1, list_data2)}


def _stage_color_number(value, name, minimum, maximum, integer=False):
    number = float(value)
    if not math.isfinite(number) or not minimum <= number <= maximum or (integer and not number.is_integer()):
        raise ValueError(f"AD_Video_color_grad: {name} must be between {minimum} and {maximum}")
    return int(number) if integer else number


def _stage_color_profile(frame, detailed=False):
    stride = max(1, min(frame.shape[:2]) // 128)
    pixels = frame[::stride, ::stride, :3].float().reshape(-1, 3)
    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] if detailed else [0.05, 0.25, 0.5, 0.75, 0.95]
    return torch.quantile(pixels, pixels.new_tensor(quantiles), dim=0)


def _stage_color_match(rgb, target, detailed=False):
    source = _stage_color_profile(rgb, detailed=detailed)
    output = rgb.clone()
    for channel in range(3):
        if float(source[-1, channel] - source[0, channel]) < 1e-4:
            continue
        x = torch.cat((source.new_zeros(1), source[:, channel], source.new_ones(1))).contiguous()
        y = torch.cat((target.new_zeros(1), target[:, channel], target.new_ones(1)))
        pixels = rgb[..., channel].contiguous()
        indices = (torch.searchsorted(x, pixels, right=True) - 1).clamp(0, len(x) - 2)
        t = (pixels - x[indices]) / (x[indices + 1] - x[indices]).clamp_min(1e-6)
        mapped = y[indices] + (y[indices + 1] - y[indices]) * t
        output[..., channel] = pixels + (mapped - pixels).clamp(-0.2, 0.2)
    return output


def _stage_color_log_brightness(frame):
    stride = max(1, min(frame.shape[:2]) // 96)
    rgb = frame[::stride, ::stride, :3].float().clamp(0, 1)
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))
    luminance = (linear * linear.new_tensor([0.2126, 0.7152, 0.0722])).sum(dim=-1)
    levels = torch.quantile(luminance.flatten(), luminance.new_tensor([0.25, 0.5, 0.75]))
    return float(levels.clamp_min(1e-5).log2().mean())


def _stage_color_smooth_targets(levels, start, end):
    """Interpolate tone statistics between nearby unselected frames."""
    if start == 1 and end == len(levels):
        raise ValueError("AD_Video_color_grad: leave normal frames outside the selection for brightness smoothing")
    left = levels[max(0, start - 4):start - 1]
    right = levels[end:end + 3]
    anchors = []
    if left:
        anchors.append((start - (len(left) + 1) / 2, np.median(left, axis=0)))
    if right:
        anchors.append((end + (len(right) + 1) / 2, np.median(right, axis=0)))
    targets = []
    for frame in range(start, end + 1):
        target = anchors[0][1]
        if len(anchors) == 2:
            (x0, y0), (x1, y1) = anchors
            target = target + (y1 - y0) * (frame - x0) / (x1 - x0)
        targets.append(target.tolist())
    return targets


def _stage_color_luma_profile(rgb):
    stride = max(1, min(rgb.shape[:2]) // 128)
    pixels = rgb[::stride, ::stride, :3].float()
    luminance = pixels[..., 0] * 0.2126 + pixels[..., 1] * 0.7152 + pixels[..., 2] * 0.0722
    return torch.quantile(luminance.flatten(), luminance.new_tensor([0.05, 0.25, 0.5, 0.75, 0.95]))


def _stage_color_smooth_tone(rgb, target):
    source = _stage_color_luma_profile(rgb)
    target = source.new_tensor(target)
    if float(source[-1] - source[0]) < 1e-4 or float((source - target).abs().max()) < 0.005:
        return rgb
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    white = luminance.amax().clamp_min(1.0).reshape(1)
    source = torch.cat((source.new_zeros(1), source, white))
    target = torch.cat((target.new_zeros(1), target, target.new_ones(1)))
    indices = (torch.searchsorted(source, luminance.contiguous(), right=True) - 1).clamp(0, len(source) - 2)
    t = (luminance - source[indices]) / (source[indices + 1] - source[indices]).clamp_min(1e-6)
    mapped = target[indices] + (target[indices + 1] - target[indices]) * t
    return rgb + (mapped - luminance).clamp(-0.15, 0.15).unsqueeze(-1)


def _stage_color_adjust(rgb, exposure, saturation, temperature):
    if exposure or temperature:
        linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb.clamp_min(0.0) + 0.055) / 1.055).pow(2.4))
        gains = rgb.new_tensor([2.0 ** (exposure + temperature * 0.2), 2.0 ** exposure,
                                2.0 ** (exposure - temperature * 0.2)])
        linear = linear * gains
        rgb = torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(0.0).pow(1.0 / 2.4) - 0.055)
    if saturation != 1.0:
        luminance = rgb[..., 0:1] * 0.2126 + rgb[..., 1:2] * 0.7152 + rgb[..., 2:3] * 0.0722
        rgb = luminance + (rgb - luminance) * saturation
    return rgb


def _stage_color_render_frame(rgb, mode, strength, exposure, saturation, temperature,
                              frame, start, end, count, fps, correction=None, tone=None, profile=None, color_tone=None):
    if not start <= frame <= end or not strength:
        return rgb
    fade = min(max(1, round(float(fps) * 0.2)), max(0, (end - start - 1) // 2))
    t = min(1.0, (frame - start + 1) / (fade + 1) if start > 1 else 1.0,
            (end - frame + 1) / (fade + 1) if end < count else 1.0)
    weight = min(strength, 1.0) * t * t * (3.0 - 2.0 * t)
    if mode == "brightness_smooth":
        adjusted = _stage_color_adjust(rgb, correction, 1.0, 0.0)
        adjusted = _stage_color_smooth_tone(adjusted, tone)
        if strength > 1.0:
            adjusted = adjusted.clamp(0.0, 1.0)
            matched = _stage_color_match(adjusted, rgb.new_tensor(color_tone), detailed=True)
            matched = _stage_color_smooth_tone(matched, tone)
            adjusted = adjusted + (matched - adjusted) * ((strength - 1.0) / 2.0)
    else:
        adjusted = _stage_color_match(rgb, profile) if profile is not None else rgb
        adjusted = _stage_color_adjust(adjusted, exposure, saturation, temperature)
    return (rgb + (adjusted - rgb) * weight).clamp(0.0, 1.0)


def _stage_color_preview(images, frame_rate, source=False):
    height, width = images.shape[1:3]
    scale = min(1.0, 640 / max(height, width))
    size = (max(2, round(height * scale / 2) * 2), max(2, round(width * scale / 2) * 2))
    directory, name, counter, subfolder, _ = folder_paths.get_save_image_path(
        "video_grade_source" if source else "video_grade", folder_paths.get_temp_directory(), width, height)
    filename = f"{name}_{counter:05d}_.mp4"
    path = os.path.join(directory, filename)
    statistics = {"fps": float(frame_rate), "count": len(images), "levels": [], "tones": [], "colors": [], "detail_colors": []}
    with av.open(path, mode="w", format="mp4", options={"movflags": "+faststart"}) as output:
        stream = output.add_stream("libx264", rate=frame_rate)
        stream.width, stream.height = size[1], size[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "22", "preset": "ultrafast", "g": str(max(1, round(float(frame_rate))))}
        for frame in images:
            if source:
                statistics["levels"].append(_stage_color_log_brightness(frame))
                statistics["tones"].append(_stage_color_luma_profile(frame).tolist())
                colors = _stage_color_profile(frame, detailed=True).tolist()
                statistics["colors"].append([colors[index] for index in (1, 3, 4, 5, 7)])
                statistics["detail_colors"].append(colors)
            small = torch.nn.functional.interpolate(frame[..., :3].movedim(-1, 0).unsqueeze(0).float(), size=size, mode="area")
            pixels = (small[0].movedim(0, -1).clamp(0, 1) * 255).round().to(device="cpu", dtype=torch.uint8).numpy()
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                output.mux(packet)
        for packet in stream.encode(None):
            output.mux(packet)
    if source:
        with open(path + ".json", "w", encoding="utf-8") as file:
            json.dump(statistics, file)
    return {"filename": filename, "subfolder": subfolder, "type": "temp"}


def _stage_color_preview_frame(data):
    filename = data["source"]
    if not isinstance(filename, str) or not re.fullmatch(r"video_grade_source_[0-9]+_\.mp4", filename):
        raise ValueError("Invalid video preview source")
    directory = os.path.realpath(folder_paths.get_temp_directory())
    path = os.path.join(directory, filename)
    for candidate in (path, path + ".json"):
        if os.path.commonpath((directory, os.path.realpath(candidate))) != directory:
            raise ValueError("Invalid video preview source")
    with open(path + ".json", encoding="utf-8") as file:
        stats = json.load(file)
    count, fps = stats["count"], stats["fps"]
    frame = _stage_color_number(data["frame"], "frame", 1, count, True)
    start = _stage_color_number(data["start"], "start", 1, count, True)
    end = _stage_color_number(data["end"], "end", start, count, True)
    reference = _stage_color_number(data["reference"], "reference", 1, count, True)
    mode = data["mode"]
    if mode not in ("manual", "brightness_smooth", "reference_match"):
        raise ValueError("Invalid grading mode")
    strength = _stage_color_number(data["strength"], "strength", 0, 3 if mode == "brightness_smooth" else 1)
    exposure = _stage_color_number(data["exposure"], "exposure", -4, 4)
    saturation = _stage_color_number(data["saturation"], "saturation", 0, 2)
    temperature = _stage_color_number(data["temperature"], "temperature", -1, 1)
    rgb = None
    with av.open(path) as video:
        stream = video.streams.video[0]
        rate = stream.average_rate
        timestamp = Fraction(frame - 1, 1) / rate
        video.seek(int(timestamp / stream.time_base), stream=stream, backward=True)
        for decoded in video.decode(stream):
            index = round(decoded.pts * decoded.time_base * rate) + 1
            if index == frame:
                rgb = torch.from_numpy(decoded.to_ndarray(format="rgb24")).float() / 255
                break
            if index > frame:
                break
    if rgb is None:
        raise ValueError("Preview frame is unavailable; reload the video")
    correction, tone, profile, color_tone = None, None, None, None
    if strength and start <= frame <= end:
        if mode == "brightness_smooth":
            targets = _stage_color_smooth_targets(stats["levels"], start, end)
            correction = max(-1.0, min(1.0, targets[frame - start] - stats["levels"][frame - 1]))
            tone = _stage_color_smooth_targets(stats["tones"], start, end)[frame - start]
            if strength > 1.0:
                if "detail_colors" not in stats:
                    raise ValueError("Reload the video to enable stronger smoothing")
                color_tone = _stage_color_smooth_targets(stats["detail_colors"], start, end)[frame - start]
        elif mode == "reference_match":
            profile = rgb.new_tensor(stats["colors"][reference - 1])
    graded = _stage_color_render_frame(rgb, mode, strength, exposure, saturation, temperature,
                                       frame, start, end, count, fps, correction, tone, profile, color_tone)
    pixels = (graded.clamp(0, 1) * 255).round().byte().numpy()
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG", compress_level=1)
    return buffer.getvalue()


@PromptServer.instance.routes.post("/apt_preset/video_grade/frame")
async def apt_preset_video_grade_frame(request):
    try:
        data = await request.json()
        if not isinstance(data, dict):
            raise ValueError("Invalid frame request")
        result = await asyncio.to_thread(_stage_color_preview_frame, data)
    except (ValueError, TypeError, KeyError) as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except FileNotFoundError:
        return web.json_response({"error": "Preview expired; reload the video"}, status=404)
    return web.Response(body=result, content_type="image/png", headers={"Cache-Control": "no-store"})


class AD_Video_color_grad:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "mode": (["manual", "brightness_smooth", "reference_match"], {"default": "manual"}),
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            "exposure": ("FLOAT", {"default": 0.0, "min": -4.0, "max": 4.0, "step": 0.05}),
            "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
            "selection": ("STRING", {"default": "{}"}),
        }}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "grade"
    CATEGORY = "Apt_Preset/AD"
    OUTPUT_NODE = True
    DESCRIPTION = "视频调色：在轨道上拖选范围，调整亮度、色彩或自动平滑。保留完整视频和音频，不需要 latent / VAE。"

    def grade(self, video, mode="manual", strength=1.0, exposure=0.0, saturation=1.0, temperature=0.0, selection="{}"):
        if mode not in ("manual", "brightness_smooth", "reference_match"):
            raise ValueError("AD_Video_color_grad: invalid mode")
        strength = _stage_color_number(strength, "strength", 0, 3 if mode == "brightness_smooth" else 1)
        exposure = _stage_color_number(exposure, "exposure", -4, 4)
        saturation = _stage_color_number(saturation, "saturation", 0, 2)
        temperature = _stage_color_number(temperature, "temperature", -1, 1)
        selected = json.loads(selection)
        if not isinstance(selected, dict) or set(selected) - {"start", "end", "reference"}:
            raise ValueError("AD_Video_color_grad: invalid timeline selection")
        components = video.get_components()
        images = components.images
        count = int(images.shape[0])
        if count == 0:
            raise ValueError("AD_Video_color_grad: video has no frames")
        start = _stage_color_number(selected.get("start", 1), "start", 1, count, True)
        end = _stage_color_number(selected.get("end", count), "end", start, count, True)
        reference = _stage_color_number(selected.get("reference", max(1, start - 1)), "reference", 1, count, True)
        curve, tones, profile, color_tones = None, None, None, None
        if strength and mode == "brightness_smooth":
            # Only the selection and its immediate neighbours contribute to the curve.
            first, last = max(0, start - 4), min(count, end + 3)
            levels = [_stage_color_log_brightness(frame) for frame in images[first:last]]
            targets = _stage_color_smooth_targets(levels, start - first, end - first)
            curve = [max(-1.0, min(1.0, target - levels[index - first]))
                     for index, target in zip(range(start - 1, end), targets)]
            profiles = [_stage_color_luma_profile(frame).tolist() for frame in images[first:last]]
            tones = _stage_color_smooth_targets(profiles, start - first, end - first)
            if strength > 1.0:
                colors = [_stage_color_profile(frame, detailed=True).tolist() for frame in images[first:last]]
                color_tones = _stage_color_smooth_targets(colors, start - first, end - first)
        elif strength and mode == "reference_match":
            profile = _stage_color_profile(images[reference - 1])
        changed = strength and (mode != "manual" or exposure != 0 or saturation != 1 or temperature != 0)
        output = images.clone() if changed else images
        if changed:
            for index in range(start - 1, end):
                rgb = images[index, ..., :3].float()
                graded = _stage_color_render_frame(rgb, mode, strength, exposure, saturation, temperature,
                    index + 1, start, end, count, components.frame_rate,
                    curve[index - start + 1] if curve is not None else None,
                    tones[index - start + 1] if tones is not None else None, profile,
                    color_tones[index - start + 1] if color_tones is not None else None)
                output[index, ..., :3] = graded.to(images.dtype)
        result = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=output, audio=components.audio, frame_rate=components.frame_rate),
            bit_depth=video.get_bit_depth())
        source = _stage_color_preview(images, components.frame_rate, source=True)
        preview = _stage_color_preview(output, components.frame_rate) if changed else source
        return {"ui": {"grade_preview": [{"video": preview, "source": source, "total_frames": count,
                    "fps": float(components.frame_rate), "start": start, "end": end, "reference": reference}]},
                "result": (result,)}


def _stage_batch_dir(run_id):
    run_dir = _stage_run_dir(run_id)
    path = os.path.join(run_dir, "batches")
    os.makedirs(path, exist_ok=True)
    return path


def _stage_batch_save(data, batch_dir, kind, stage_index, storage_kind=None):
    tensors, descriptor = _stage_encode_payload(data, kind)
    filename = f"{storage_kind or kind}_{stage_index:05d}.safetensors"
    path = os.path.join(batch_dir, filename)
    temp_path = path + ".tmp"
    try:
        comfy.utils.save_torch_file(
            tensors,
            temp_path,
            metadata={"stage_payload": json.dumps(descriptor, ensure_ascii=False)},
        )
        os.replace(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)


def _stage_batch_load(batch_dir, kind, stage_index):
    path = os.path.join(batch_dir, f"{kind}_{stage_index:05d}.safetensors")
    if not os.path.isfile(path):
        return None
    return _stage_decode_payload(path)


def _stage_batch_concat_image(items):
    if not items:
        return ExecutionBlocker(None)
    ref = items[0]
    ref_h, ref_w = int(ref.shape[1]), int(ref.shape[2])
    out = [ref]
    for tensor in items[1:]:
        if tensor.shape[1:] != ref.shape[1:]:
            tensor = common_upscale(
                tensor.movedim(-1, 1), ref_w, ref_h, "bilinear", "center"
            ).movedim(1, -1)
        out.append(tensor)
    return torch.cat(out, dim=0)


def _stage_batch_concat_mask(items):
    if not items:
        return ExecutionBlocker(None)
    normalized = [tensor.unsqueeze(0) if tensor.ndim == 2 else tensor for tensor in items]
    ref = normalized[0]
    ref_h, ref_w = int(ref.shape[-2]), int(ref.shape[-1])
    out = [ref]
    for tensor in normalized[1:]:
        if tensor.shape[-2:] != ref.shape[-2:]:
            tensor = common_upscale(
                tensor.unsqueeze(1), ref_w, ref_h, "bilinear", "center"
            ).squeeze(1)
        out.append(tensor)
    return torch.cat(out, dim=0)


def _stage_latent_to_list(samples):
    if isinstance(samples, comfy.nested_tensor.NestedTensor):
        return list(samples.unbind())
    return [samples[index] for index in range(samples.shape[0])]


def _stage_batch_concat_latent(items):
    if not items:
        return ExecutionBlocker(None)
    merged = items[0]
    samples_out = merged.copy()
    s1 = merged["samples"]
    use_nested = isinstance(s1, comfy.nested_tensor.NestedTensor)
    for nxt in items[1:]:
        s2 = nxt["samples"]
        if use_nested or isinstance(s2, comfy.nested_tensor.NestedTensor):
            use_nested = True
            s1 = comfy.nested_tensor.NestedTensor(_stage_latent_to_list(s1) + _stage_latent_to_list(s2))
        else:
            if s1.shape[1:] != s2.shape[1:]:
                s2 = common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
            s1 = torch.cat((s1, s2), dim=0)
    samples_out["samples"] = s1
    return samples_out


def _stage_batch_concat_audio(items):
    if not items:
        return ExecutionBlocker(None)
    sample_rate = int(items[0]["sample_rate"])
    waveforms = [item["waveform"] for item in items]
    return {"waveform": torch.cat(waveforms, dim=2), "sample_rate": sample_rate}


def _stage_video_segment_path(batch_dir, stage_index):
    return os.path.join(batch_dir, "video_segments", f"{stage_index:05d}.mp4")


def _stage_video_reference_path(batch_dir):
    return os.path.join(batch_dir, "video_reference.json")


def _stage_video_normalize_audio(audio, sample_rate, channels, length):
    if audio is None:
        waveform = torch.zeros((1, channels, length), dtype=torch.float32)
    else:
        waveform = audio["waveform"]
        source_rate = int(audio["sample_rate"])
        if source_rate != sample_rate:
            resampled_length = max(1, round(waveform.shape[-1] * sample_rate / source_rate))
            waveform = torch.nn.functional.interpolate(waveform, size=resampled_length, mode="linear", align_corners=False)
        source_channels = int(waveform.shape[1])
        if source_channels != channels:
            if channels == 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            elif source_channels == 1:
                waveform = waveform.repeat(1, channels, 1)
            elif source_channels > channels:
                waveform = waveform[:, :channels]
            else:
                waveform = torch.cat((waveform, waveform[:, -1:].repeat(1, channels - source_channels, 1)), dim=1)
        if waveform.shape[-1] < length:
            waveform = torch.nn.functional.pad(waveform, (0, length - waveform.shape[-1]))
        else:
            waveform = waveform[..., :length]
    return {"waveform": waveform, "sample_rate": sample_rate}


def _stage_save_video_segment(video, batch_dir, stage_index):
    components = video.get_components()
    images = components.images
    audio = components.audio
    reference_path = _stage_video_reference_path(batch_dir)
    if stage_index == 0:
        audio_sample_rate = int(audio["sample_rate"]) if audio is not None else 0
        source_audio_channels = int(audio["waveform"].shape[1]) if audio is not None else 0
        audio_channels = source_audio_channels if source_audio_channels in (1, 2, 6) else (2 if source_audio_channels else 0)
        frame_rate = Fraction(components.frame_rate)
        reference = {
            "width": int(images.shape[2]),
            "height": int(images.shape[1]),
            "frame_rate": [frame_rate.numerator, frame_rate.denominator],
            "bit_depth": int(video.get_bit_depth()) if hasattr(video, "get_bit_depth") else 8,
            "audio_sample_rate": audio_sample_rate,
            "audio_channels": audio_channels,
        }
        _stage_write_json(reference_path, reference)
    else:
        if not os.path.isfile(reference_path):
            raise ValueError("flow_stage: first video segment information is missing")
        with open(reference_path, "r", encoding="utf-8") as handle:
            reference = json.load(handle)

    width = int(reference["width"])
    height = int(reference["height"])
    if images.shape[1:3] != (height, width):
        images = common_upscale(images.movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
    numerator, denominator = reference["frame_rate"]
    frame_rate = Fraction(numerator, denominator)
    normalized = Types.VideoComponents(
        images=images,
        audio=(
            _stage_video_normalize_audio(
                audio,
                int(reference["audio_sample_rate"]),
                int(reference["audio_channels"]),
                max(1, math.ceil(images.shape[0] * int(reference["audio_sample_rate"]) / frame_rate)),
            )
            if int(reference["audio_sample_rate"]) > 0 else None
        ),
        frame_rate=frame_rate,
    )
    path = _stage_video_segment_path(batch_dir, stage_index)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp.mp4"
    try:
        normalized_video = InputImpl.VideoFromComponents(normalized, bit_depth=int(reference["bit_depth"]))
        normalized_video.save_to(
            temp_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            crf=18.0,
        )
        os.replace(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
    return path


def _stage_concat_video_segments(paths, output_path):
    temp_path = output_path + ".tmp.mp4"
    try:
        with av.open(paths[0], mode="r") as first:
            templates = [stream for stream in first.streams if stream.type in ("video", "audio")]
            if not templates:
                raise ValueError("flow_stage: saved segment has no usable stream")
            with av.open(temp_path, mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
                output_streams = [output.add_stream_from_template(stream, opaque=True) for stream in templates]
                timeline = Fraction(0)
                for path in paths:
                    with av.open(path, mode="r") as source:
                        streams = [stream for stream in source.streams if stream.type in ("video", "audio")]
                        if len(streams) != len(templates):
                            raise ValueError("flow_stage: segment stream layouts do not match")
                        bases = {}
                        segment_end = Fraction(0)
                        for packet in source.demux(streams):
                            if packet.dts is None and packet.pts is None:
                                continue
                            stream_index = streams.index(packet.stream)
                            template = templates[stream_index]
                            if packet.stream.type != template.type or packet.stream.codec_context.name != template.codec_context.name:
                                raise ValueError("flow_stage: segment codecs do not match")
                            time_base = Fraction(packet.time_base or packet.stream.time_base)
                            timestamps = [value for value in (packet.dts, packet.pts) if value is not None]
                            base = bases.setdefault(stream_index, min(timestamps))
                            offset = int(timeline / time_base)
                            if packet.dts is not None:
                                packet.dts = packet.dts - base + offset
                            if packet.pts is not None:
                                packet.pts = packet.pts - base + offset
                            packet.time_base = time_base
                            packet.stream = output_streams[stream_index]
                            end_value = max(value for value in (packet.dts, packet.pts) if value is not None)
                            segment_end = max(segment_end, (end_value + int(packet.duration or 0)) * time_base - timeline)
                            output.mux(packet)
                        if segment_end <= 0:
                            raise ValueError("flow_stage: saved segment is empty")
                        timeline += segment_end
        os.replace(temp_path, output_path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
    return InputImpl.VideoFromFile(output_path)


def _stage_collect_outputs(batch_dir, total):
    images = [_stage_batch_load(batch_dir, "image", index) for index in range(total)]
    masks = [_stage_batch_load(batch_dir, "mask", index) for index in range(total)]
    latents = [_stage_batch_load(batch_dir, "latent", index) for index in range(total)]
    audios = [_stage_batch_load(batch_dir, "audio", index) for index in range(total)]
    video_paths = [_stage_video_segment_path(batch_dir, index) for index in range(total)]
    video_paths = [path for path in video_paths if os.path.isfile(path)]
    merged_video = (
        _stage_concat_video_segments(video_paths, os.path.join(batch_dir, "merged_video.mp4"))
        if video_paths else ExecutionBlocker(None)
    )
    return (
        _stage_batch_concat_image([item for item in images if item is not None]),
        _stage_batch_concat_mask([item for item in masks if item is not None]),
        _stage_batch_concat_latent([item for item in latents if item is not None]),
        merged_video,
        _stage_batch_concat_audio([item for item in audios if item is not None]),
    )


def _stage_collect_blockers():
    return tuple(ExecutionBlocker(None) for _ in range(5))


class _StageBatchDynamicInputs(dict):
    """Provide the dynamic `value_*` slot for flow_stage_list / flow_stage_collect_multi.

    This dict MUST be a real mapping (with at least one key) so it round-trips
    through `json.dumps` / `json.loads` correctly. The previous implementation
    relied on the ``__contains__`` / ``__getitem__`` overrides of a dict subclass,
    but ``json`` only serializes the underlying items, so the frontend received
    an empty ``{}`` and the optional slot was never materialised on Linux.
    """

    def __init__(self, max_count=1):
        super().__init__()
        for i in range(1, max_count + 1):
            self[f"value_{i}"] = (any_type, {"lazy": True})

    def __contains__(self, key):
        return isinstance(key, str) and key.startswith("value_")

    def __getitem__(self, key):
        if key in self:
            return (any_type, {"lazy": True})
        raise KeyError(key)


class flow_stage_list:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_info": (_STAGE_INFO_TYPE,),
            },
            "optional": _StageBatchDynamicInputs(),
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (any_type, "LIST")
    RETURN_NAMES = ("list", "array")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "accumulate"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check_lazy_status(self, stage_info, **kwargs):
        _, stage_index, _ = _stage_validate_info(stage_info)
        input_name = f"value_{stage_index + 1}"
        if input_name in kwargs and kwargs[input_name] is None:
            return [input_name]
        return []

    def accumulate(self, stage_info, unique_id=None, **kwargs):
        run_id, stage_index, total = _stage_validate_info(stage_info)
        input_name = f"value_{stage_index + 1}"
        data = kwargs.get(input_name)
        if data is None:
            raise ValueError(f"flow_stage_list: {input_name} is not connected")

        kind = _stage_detect_type(data, "auto")

        batch_dir = os.path.join(_stage_batch_dir(run_id), _stage_safe_name(f"multi_{unique_id}"))
        os.makedirs(batch_dir, exist_ok=True)
        type_path = os.path.join(batch_dir, "multi_type.json")
        if stage_index == 0:
            _stage_write_json(type_path, {"type": kind})
        else:
            if not os.path.isfile(type_path):
                raise ValueError("flow_stage_list: first stage type information is missing")
            with open(type_path, "r", encoding="utf-8") as handle:
                first_kind = json.load(handle).get("type")
            if kind != first_kind:
                raise TypeError(f"flow_stage_list: {input_name} must be {first_kind}, got {kind}")

        _stage_batch_save(data, batch_dir, kind, stage_index, storage_kind="multi")
        if stage_index != total - 1:
            blocker = ExecutionBlocker(None)
            return (blocker, blocker)

        items = [_stage_batch_load(batch_dir, "multi", index) for index in range(total)]
        if any(item is None for item in items):
            missing = [str(index + 1) for index, item in enumerate(items) if item is None]
            raise ValueError(f"flow_stage_list: missing stage data: {', '.join(missing)}")
        return (items, items)


class flow_stage_collect_single:
    """Accumulate tensor batches and merge video/audio across flow_stage stages."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_info": (_STAGE_INFO_TYPE,),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "latent": ("LATENT",),
                "video": ("VIDEO",),
                "audio": ("AUDIO",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "VIDEO", "AUDIO")
    RETURN_NAMES = ("image_batch", "mask_batch", "latent_batch", "merged_video", "merged_audio")
    FUNCTION = "accumulate"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def accumulate(self, stage_info, image=None, mask=None, latent=None, video=None, audio=None, unique_id=None):
        run_id, stage_index, total = _stage_validate_info(stage_info)
        batch_dir = os.path.join(_stage_batch_dir(run_id), _stage_safe_name(f"single_{unique_id}"))
        os.makedirs(batch_dir, exist_ok=True)

        if image is not None:
            _stage_batch_save(image, batch_dir, "image", stage_index)
        if mask is not None:
            _stage_batch_save(mask, batch_dir, "mask", stage_index)
        if latent is not None:
            _stage_batch_save(latent, batch_dir, "latent", stage_index)
        if video is not None:
            _stage_save_video_segment(video, batch_dir, stage_index)
        if audio is not None:
            _stage_batch_save(audio, batch_dir, "audio", stage_index)

        if stage_index != total - 1:
            return _stage_collect_blockers()

        message = f"single-port stages {total}/{total} merged: {batch_dir}"
        return {
            "ui": {"text": [message]},
            "result": _stage_collect_outputs(batch_dir, total),
        }


class flow_stage_collect_multi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stage_info": (_STAGE_INFO_TYPE,),
            },
            "optional": _StageBatchDynamicInputs(),
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "LATENT", "VIDEO", "AUDIO")
    RETURN_NAMES = ("image_batch", "mask_batch", "latent_batch", "merged_video", "merged_audio")
    FUNCTION = "accumulate"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def check_lazy_status(self, stage_info, **kwargs):
        _, stage_index, _ = _stage_validate_info(stage_info)
        input_name = f"value_{stage_index + 1}"
        if input_name in kwargs and kwargs[input_name] is None:
            return [input_name]
        return []

    def accumulate(self, stage_info, unique_id=None, **kwargs):
        run_id, stage_index, total = _stage_validate_info(stage_info)
        input_name = f"value_{stage_index + 1}"
        data = kwargs.get(input_name)
        if data is None:
            raise ValueError(f"flow_stage_collect_multi: {input_name} is not connected")

        kind = _stage_detect_type(data, "auto")
        if kind not in ("image", "mask", "latent", "video", "audio"):
            raise TypeError(f"flow_stage_collect_multi: {input_name} has unsupported type {kind}")

        batch_dir = os.path.join(_stage_batch_dir(run_id), _stage_safe_name(f"multi_{unique_id}"))
        os.makedirs(batch_dir, exist_ok=True)
        type_path = os.path.join(batch_dir, "type.json")
        if stage_index == 0:
            _stage_write_json(type_path, {"type": kind})
        else:
            if not os.path.isfile(type_path):
                raise ValueError("flow_stage_collect_multi: first stage type information is missing")
            with open(type_path, "r", encoding="utf-8") as handle:
                first_kind = json.load(handle).get("type")
            if kind != first_kind:
                raise TypeError(f"flow_stage_collect_multi: {input_name} must be {first_kind}, got {kind}")

        if kind == "video":
            _stage_save_video_segment(data, batch_dir, stage_index)
        else:
            _stage_batch_save(data, batch_dir, kind, stage_index)
        if stage_index != total - 1:
            return _stage_collect_blockers()

        if kind == "video":
            items = [
                path if os.path.isfile(path) else None
                for path in (_stage_video_segment_path(batch_dir, index) for index in range(total))
            ]
        else:
            items = [_stage_batch_load(batch_dir, kind, index) for index in range(total)]
        if any(item is None for item in items):
            missing = [str(index + 1) for index, item in enumerate(items) if item is None]
            raise ValueError(f"flow_stage_collect_multi: missing stage data: {', '.join(missing)}")

        message = f"multi-port stages {total}/{total} merged: {batch_dir}"
        return {
            "ui": {"text": [message]},
            "result": _stage_collect_outputs(batch_dir, total),
        }


FLOW_VALUE_STORE = {}


def _normalize_flow_var_name(variable):
    if variable is None:
        return ""
    return str(variable).strip()



MAX_FLOW_NUM = 20


from comfy_execution.graph_utils import GraphBuilder, is_link


class flow_whileStart:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "condition": ("BOOLEAN", {"default": True}),
            },
            "optional": {},
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value_%d" % i] = (any_type,)
        return inputs
    
    NAME="loop_whileStart"
    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CL"] + [any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow"] + ["value_%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_open"
    CATEGORY = "Apt_Preset/flow/other"

    def while_loop_open(self, condition, **kwargs):
        
        values = []
        for i in range(MAX_FLOW_NUM):
            val = kwargs.get("initial_value_%d" % i, None)
            values.append(val if condition else ExecutionBlocker(None))
        return tuple(["stub"] + values)


class flow_whileEnd:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "flow": ("FLOW_CL", {"rawLink": True}),
                "condition": ("BOOLEAN", {}),
            },
            "optional": {},
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID",
            }
        }
        for i in range(MAX_FLOW_NUM):
            inputs["optional"]["initial_value_%d" % i] = (any_type,)
        return inputs
    NAME="loop_whileEnd"
    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * MAX_FLOW_NUM))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value_%d" % i for i in range(MAX_FLOW_NUM)]))
    FUNCTION = "while_loop_close"
    CATEGORY = "Apt_Preset/flow/other"

    def explore_dependencies(self, node_id, dynprompt, upstream, parent_ids):
        
        node_info = dynprompt.get_node(node_id)
        if "inputs" not in node_info:
            return

        for k, v in node_info["inputs"].items():
            if is_link(v):
                parent_id = v[0]
                display_id = dynprompt.get_display_node_id(parent_id)
                display_node = dynprompt.get_node(display_id)
                class_type = display_node["class_type"]
                loop_node_types = [
                    'flow_forEnd', 'flow_forEnd',
                    'flow_whileEnd', 'flow_whileEnd'
                ]
                if class_type not in loop_node_types:
                    parent_ids.append(display_id)
                if parent_id not in upstream:
                    upstream[parent_id] = []
                    self.explore_dependencies(parent_id, dynprompt, upstream, parent_ids)
                upstream[parent_id].append(node_id)

    def collect_contained(self, node_id, upstream, contained):
        if node_id not in upstream:
            return
        for child_id in upstream[node_id]:
            if child_id not in contained:
                contained[child_id] = True
                self.collect_contained(child_id, upstream, contained)

    def explore_output_nodes(self, dynprompt, upstream, output_nodes, parent_ids):
        for parent_id in upstream:
            display_id = dynprompt.get_display_node_id(parent_id)
            for output_id in output_nodes:
                input_link = output_nodes[output_id]
                if not is_link(input_link):
                    continue
                source_id = input_link[0]
                if source_id in parent_ids and display_id == source_id and output_id not in upstream[parent_id]:
                    if "." in parent_id:
                        arr = parent_id.split(".")
                        arr[len(arr) - 1] = output_id
                        upstream[parent_id].append(".".join(arr))
                    else:
                        upstream[parent_id].append(output_id)

    def while_loop_close(self, flow, condition, dynprompt=None, unique_id=None, **kwargs):
        if not condition:
            return tuple(kwargs.get("initial_value_%d" % i, None) for i in range(MAX_FLOW_NUM))

        
        upstream = {}
        parent_ids = []
        self.explore_dependencies(unique_id, dynprompt, upstream, parent_ids)
        parent_ids = list(set(parent_ids))

        output_nodes = {}
        prompts = dynprompt.get_original_prompt()
        for node_id in prompts:
            node = prompts[node_id]
            if "inputs" not in node:
                continue
            class_type = node.get("class_type")
            class_def = nodes.NODE_CLASS_MAPPINGS.get(class_type)
            if class_def is None:
                continue
            if hasattr(class_def, "OUTPUT_NODE") and class_def.OUTPUT_NODE is True:
                for _, v in node["inputs"].items():
                    if is_link(v):
                        output_nodes[node_id] = v
                        break
        
        graph = GraphBuilder()
        self.explore_output_nodes(dynprompt, upstream, output_nodes, parent_ids)
        contained = {}
        
        if flow is None or len(flow) == 0:
             return tuple([None] * MAX_FLOW_NUM)

        open_node = flow[0]
        self.collect_contained(open_node, upstream, contained)
        contained[unique_id] = True
        contained[open_node] = True

        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.node(original_node["class_type"], "Recurse" if node_id == unique_id else node_id)
            node.set_override_display_id(node_id)
            
        for node_id in contained:
            original_node = dynprompt.get_node(node_id)
            node = graph.lookup_node("Recurse" if node_id == unique_id else node_id)
            for k, v in original_node["inputs"].items():
                if is_link(v) and v[0] in contained:
                    parent = graph.lookup_node(v[0])
                    node.set_input(k, parent.out(v[1]))
                else:
                    node.set_input(k, v)

        new_open = graph.lookup_node(open_node)
        for i in range(MAX_FLOW_NUM):
            key = "initial_value_%d" % i
            new_open.set_input(key, kwargs.get(key, None))
            
        my_clone = graph.lookup_node("Recurse")
        result = [my_clone.out(i) for i in range(MAX_FLOW_NUM)]
        
        return {
            "result": tuple(result),
            "expand": graph.finalize(),
        }


class flow_forStart:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total": ("INT", {"default": 1, "min": 1, "max": 100000}),
            },
            "optional": {
                "initial_value_%d" % i: (any_type,) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "initial_value_0": (any_type,),
                "unique_id": "UNIQUE_ID"
            }
        }
    NAME="loop_forStart"
    RETURN_TYPES = ByPassTypeTuple(tuple(["FLOW_CL", "INT"] + [any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["flow", "index"] + ["value_%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "loop_start"
    CATEGORY = "Apt_Preset/flow"

    def loop_start(self, total, **kwargs):
        graph = GraphBuilder()
        i = kwargs.get("initial_value_0", 0)

        outputs = []
        initial_vals = {}
        for n in range(1, MAX_FLOW_NUM):
            val = kwargs.get(f"initial_value_{n}")
            if n == MAX_FLOW_NUM - 1 and val is None:
                val = total
            outputs.append(val)
            initial_vals[f"initial_value_{n}"] = val

        graph.node(
            "flow_whileStart",
            condition=total,
            initial_value_0=i,
            **initial_vals
        )

        return {
            "result": tuple(["stub", i] + outputs),
            "expand": graph.finalize(),
        }
    

class flow_forEnd:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "flow": ("FLOW_CL", {"rawLink": True}),
                "batch_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "initial_value_%d" % i: (any_type, {"rawLink": True}) for i in range(1, MAX_FLOW_NUM)
            },
            "hidden": {
                "dynprompt": "DYNPROMPT",
                "unique_id": "UNIQUE_ID"
            },
        }
    
    NAME = "loop_forEnd"
    RETURN_TYPES = ByPassTypeTuple(tuple([any_type] * (MAX_FLOW_NUM - 1)))
    RETURN_NAMES = ByPassTypeTuple(tuple(["value_%d" % i for i in range(1, MAX_FLOW_NUM)]))
    FUNCTION = "loop_end"
    CATEGORY = "Apt_Preset/flow"

    def loop_end(self, flow, batch_output=True, dynprompt=None, unique_id=None, **kwargs):
        
        graph = GraphBuilder()
        
        if flow is None or not isinstance(flow, (list, tuple)) or len(flow) == 0:
            return tuple(kwargs.get(f"initial_value_{i}") for i in range(1, MAX_FLOW_NUM))
            
        while_open_id = flow[0]
        start_node = dynprompt.get_node(while_open_id)
        
        if start_node is None:
             return tuple(kwargs.get(f"initial_value_{i}") for i in range(1, MAX_FLOW_NUM))

        total = None
        total_input = start_node.get("inputs", {}).get("total")
        if total_input is not None:
            if is_link(total_input):
                total = total_input
            else:
                try:
                    if isinstance(total_input, torch.Tensor):
                        total = int(total_input.item()) if total_input.numel() == 1 else 0
                    else:
                        total = int(total_input)
                except (ValueError, TypeError):
                    total = 0
        
        if total is None or (isinstance(total, list) and len(total) == 0):
            total = MAX_FLOW_NUM

        sub = graph.node(
            "math_calculate", 
            preset="a + b", 
            expression="", 
            a=[while_open_id, 1], 
            b=1,
            c=None
        )
        cond = graph.node(
            "math_calculate", 
            preset="a < b", 
            expression="", 
            a=sub.out(1),
            b=total,
            c=None
        )

        input_values = {}
        for i in range(1, MAX_FLOW_NUM):
            key = f"initial_value_{i}"
            v = kwargs.get(key)
            
            if batch_output and is_link(v):
                collector = graph.node("flow_createbatch", any_1=[while_open_id, i + 1], any_2=v)
                input_values[key] = collector.out(0)
            else:
                input_values[key] = v
        
        while_close = graph.node(
            "flow_whileEnd", 
            flow=flow, 
            condition=cond.out(2),
            initial_value_0=sub.out(1),
            **input_values
        )
        
        results = []
        for i in range(1, MAX_FLOW_NUM):
            out = while_close.out(i)
            results.append(out)

        return {
            "result": tuple(results),
            "expand": graph.finalize(),
        }


class flow_createbatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any_1": (any_type, {}),
                "any_2": (any_type, {})
            }
        }
    
    NAME="loop_createbatch"
    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("batch",)

    FUNCTION = "batch"
    CATEGORY = "Apt_Preset/stack/register"

    def latentBatch(self, any_1, any_2):
        samples_out = any_1.copy()
        s1 = any_1["samples"]
        s2 = any_2["samples"]

        if s1.shape[1:] != s2.shape[1:]:
            s2 = comfy.utils.common_upscale(s2, s1.shape[3], s1.shape[2], "bilinear", "center")
        s = torch.cat((s1, s2), dim=0)
        samples_out["samples"] = s
        samples_out["batch_index"] = any_1.get("batch_index",
                                               [x for x in range(0, s1.shape[0])]) + any_2.get(
            "batch_index", [x for x in range(0, s2.shape[0])])

        return samples_out

    def batch(self, any_1, any_2):
        if isinstance(any_1, torch.Tensor) or isinstance(any_2, torch.Tensor):
            if any_1 is None:
                return (any_2,)
            elif any_2 is None:
                return (any_1,)
            if any_1.shape[1:] != any_2.shape[1:]:
                any_2 = comfy.utils.common_upscale(any_2.movedim(-1, 1), any_1.shape[2], any_1.shape[1], "bilinear",
                                                   "center").movedim(1, -1)
            return (torch.cat((any_1, any_2), 0),)
        elif isinstance(any_1, (str, float, int)):
            if any_2 is None:
                return (any_1,)
            elif isinstance(any_2, tuple):
                return (any_2 + (any_1,),)
            elif isinstance(any_2, list):
                return (any_2 + [any_1],)
            return ([any_1, any_2],)
        elif isinstance(any_2, (str, float, int)):
            if any_1 is None:
                return (any_2,)
            elif isinstance(any_1, tuple):
                return (any_1 + (any_2,),)
            elif isinstance(any_1, list):
                return (any_1 + [any_2],)
            return ([any_2, any_1],)
        elif isinstance(any_1, dict) and 'samples' in any_1:
            if any_2 is None:
                return (any_1,)
            elif isinstance(any_2, dict) and 'samples' in any_2:
                return (self.latentBatch(any_1, any_2),)
        elif isinstance(any_2, dict) and 'samples' in any_2:
            if any_1 is None:
                return (any_2,)
            elif isinstance(any_1, dict) and 'samples' in any_1:
                return (self.latentBatch(any_2, any_1),)
        else:
            if any_1 is None:
                return (any_2,)
            elif any_2 is None:
                return (any_1,)
            return (any_1 + any_2,)






import time
import subprocess
import sys
import threading
import os


class flow_AutoShutdown:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "current_task_count": ("INT", {"default": 0, "min": 0,  "forceinput": True}),
                "target_task_count": ("INT", {"default": 10, "min": 1, }),
                "action_delay_minutes": ("FLOAT", {"default": 5.0, "min": 0.0, "step": 0.5, }),
                "action_type": (["None", "关机", "睡眠"], {"default": "None", }),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "check_and_execute"
    OUTPUT_NODE = True
    CATEGORY = "Apt_Preset/flow/other"
    DESCRIPTION = "当完成任务数达到目标值时执行指定操作（关机/睡眠/无操作）"

    def check_and_execute(self, current_task_count, target_task_count, action_delay_minutes, action_type):
        # 如果选择None（无操作），直接返回
        if action_type == "None":
            return ()
        
        # 检查任务数是否达标
        if current_task_count == target_task_count:
            print(f"[自动关机] 检测到完成任务数 {current_task_count} 达到目标值 {target_task_count}，准备执行：{action_type}")
            threading.Thread(
                target=self.delayed_action,
                args=(action_delay_minutes, action_type),
                daemon=True
            ).start()
        
        return ()

    def delayed_action(self, delay_minutes, action_type):
        delay_seconds = delay_minutes * 60
        
        if delay_seconds > 0:
            print(f"[自动关机] 将在 {delay_minutes} 分钟({delay_seconds}秒)后执行：{action_type}")
            time.sleep(delay_seconds)
        
        try:
            if action_type == "关机":
                self.shutdown_computer()
            elif action_type == "睡眠":
                self.sleep_computer()
            print(f"[自动关机] {action_type} 命令已执行")
        except Exception as e:
            print(f"[自动关机] {action_type} 执行失败：{str(e)}")

    def shutdown_computer(self):
        if sys.platform == "win32":
            subprocess.run(["shutdown", "/s", "/t", "0"], check=True)
        elif sys.platform in ["linux", "darwin"]:
            subprocess.run(["sudo", "shutdown", "-h", "now"], check=True)

    def sleep_computer(self):
        if sys.platform == "win32":
            subprocess.run(["powercfg", "-hibernate", "off"], check=True)
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"], check=True)
        elif sys.platform == "darwin":
            subprocess.run(["pmset", "sleepnow"], check=True)
        elif sys.platform == "linux":
            subprocess.run(["systemctl", "suspend"], check=True)






import time
import hashlib
import pickle
from collections import defaultdict

class flow_ChangeDetector:
    object_cache = defaultdict(dict)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "object1": ("*",),
                "object2": ("*",),
                "delay_threshold_seconds": ("FLOAT", {
                    "default": 10.0, 
                    "min": 0.1, 
                    "max": 300.0, 
                    "step": 0.1
                }),
            },
            "optional": {
                "cache_key1": ("STRING", {"default": "obj1"}),
                "cache_key2": ("STRING", {"default": "obj2"}),
            }
        }
    
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("BOTH_STABLE",)
    FUNCTION = "detect_double_stable"
    CATEGORY = "Apt_Preset/flow/other"

    def _get_object_hash(self, obj):
        try:
            serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            return hashlib.md5(serialized).hexdigest()
        except:
            return str(obj) + str(id(obj))

    def _is_single_stable(self, obj, delay, cache_key):
        current_time = time.time()
        current_hash = self._get_object_hash(obj)
        cache = self.object_cache[cache_key]

        if not cache:
            cache["last_hash"] = current_hash
            cache["last_change_time"] = current_time
            return False
        if cache["last_hash"] != current_hash:
            cache["last_hash"] = current_hash
            cache["last_change_time"] = current_time
            return False
        return (current_time - cache["last_change_time"]) >= delay

    def detect_double_stable(self, object1, object2, delay_threshold_seconds=10.0, cache_key1="obj1", cache_key2="obj2"):
        # 分别检测两个对象是否稳定
        obj1_stable = self._is_single_stable(object1, delay_threshold_seconds, cache_key1)
        obj2_stable = self._is_single_stable(object2, delay_threshold_seconds, cache_key2)
        # 仅当两个对象同时稳定时，返回True
        both_stable = obj1_stable and obj2_stable
        return (both_stable,)


    
#endregion---------------loop team-------------













