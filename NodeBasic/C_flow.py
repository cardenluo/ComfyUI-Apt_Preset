import folder_paths
from comfy import model_management
from comfy.utils import common_upscale
import torch
import numpy as np
from PIL import Image
import base64
import io
import json
from typing import Tuple
from server import PromptServer
from aiohttp import web
import os
import inspect
import nodes
import comfy.utils



from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader



#region----------------lowcpu--------------------------



has_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()

def get_gpu_memory_info(gpu_index=0):
    if not has_gpu or gpu_index >= gpu_count:
        return None, None
    try:
        gpu_prop = torch.cuda.get_device_properties(gpu_index)
        total = gpu_prop.total_memory / (1024 ** 3)
        used = torch.cuda.memory_allocated(gpu_index) / (1024 ** 3)
        return round(total, 2), round(used, 2)
    except Exception as e:
        print(f"获取GPU{gpu_index}信息出错: {e}")
        return None, None

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

    def set_vram(self, anything, reserved, mode="auto", unique_id=None, extra_pnginfo=None):
        reserved_bytes = int(max(0.0, reserved) * (1024 ** 3))
        
        if mode == "auto":
            if has_gpu:
                total_gpu, used_gpu = get_gpu_memory_info(gpu_index=0)
                if total_gpu and used_gpu and total_gpu > 0:
                    auto_reserved = used_gpu + reserved
                    auto_reserved = max(0.0, min(auto_reserved, total_gpu))
                    model_management.EXTRA_RESERVED_VRAM = int(auto_reserved * (1024 ** 3))
                    print(f'✅ 自动显存预留模式生效 | 总显存={total_gpu}GB | 已用={used_gpu}GB | 最终预留={auto_reserved:.2f}GB')
                else:
                    model_management.EXTRA_RESERVED_VRAM = reserved_bytes
                    print(f'⚠️ 自动模式读取显存失败，启用兜底预留值: {reserved}GB')
            else:
                model_management.EXTRA_RESERVED_VRAM = reserved_bytes
                print(f'⚠️ 无可用GPU，自动模式失效，使用手动预留值: {reserved}GB')
        else:
            model_management.EXTRA_RESERVED_VRAM = reserved_bytes
            print(f'✅ 手动显存预留模式生效 | 固定预留={reserved}GB')

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




class flow_auto_pixel:
    upscale_methods = ["bicubic","nearest-exact", "bilinear", "area",  "lanczos"]
    crop_methods = ["disabled", "center"]
    # 包含英文的选项列表
    threshold_types = ["(W+H) < threshold", "W*H < threshold", "width <= height", "width > height"]
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
                "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                "image": ("IMAGE",), 
                "threshold_type": (s.threshold_types,),  # 使用更新后的选项列表
                "pixels_threshold": ("INT", { "min": 0, "max": 90000,  "step": 1,}),
                "upscale_method_True": (s.upscale_methods,),
                "upscale_method_False": (s.upscale_methods,),
                "low_pixels_True": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),      # 名称修改
                "high_pixels_False": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01}),   # 名称修改
                "divisible_by": ("INT", { "default": 8, "min": 0, "max": 512, "step": 1, }),
                }
                }
    

    RETURN_TYPES = ("IMAGE", )  
    RETURN_NAMES = ("image", )  
    FUNCTION = "auto_pixel"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def auto_pixel(self, model_name, image, threshold_type, 
                pixels_threshold, upscale_method_True, upscale_method_False, low_pixels_True, high_pixels_False, divisible_by):


        # 处理不同维度的图像张量
        if len(image.shape) == 3:
            # 形状为 (H, W, C) 的单张图像
            height, width, channels = image.shape
            batch_size = 1
        elif len(image.shape) == 4:
            # 形状为 (B, H, W, C) 的批次图像
            batch_size, height, width, channels = image.shape
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # 根据选择的threshold_type确定使用哪种逻辑
        if threshold_type == "(W+H) < threshold":
            if (width + height) < pixels_threshold:
                megapixels = low_pixels_True
                upscale_method = upscale_method_True
            else:
                megapixels = high_pixels_False
                upscale_method = upscale_method_False
        elif threshold_type == "W*H < threshold":
            if (width * height) < pixels_threshold:
                megapixels = low_pixels_True
                upscale_method = upscale_method_True
            else:
                megapixels = high_pixels_False
                upscale_method = upscale_method_False
        elif threshold_type == "width <= height":
            megapixels = low_pixels_True
            upscale_method = upscale_method_True
        elif threshold_type == "width > height":
            megapixels = high_pixels_False
            upscale_method = upscale_method_False
            
        model = UpscaleModelLoader().load_model(model_name)[0]
        image = ImageUpscaleWithModel().upscale(model, image)[0]

        if len(image.shape) == 3:
            H, W, C = image.shape
        else:  # len(image.shape) == 4
            B, H, W, C = image.shape
        
        if divisible_by > 1:
            new_width = W - (W % divisible_by)
            new_height = H - (H % divisible_by)
            
            if new_width == 0:
                new_width = divisible_by
            if new_height == 0:
                new_height = divisible_by
            if new_width != W or new_height != H:
                # 根据图像维度调整处理方式
                if len(image.shape) == 3:
                    image = image.movedim(-1, 0)  # (H, W, C) -> (C, H, W)
                    image = common_upscale(image.unsqueeze(0), new_width, new_height, upscale_method, "center")
                    image = image.squeeze(0).movedim(0, -1)  # (C, H, W) -> (H, W, C)
                else:  # len(image.shape) == 4
                    image = image.movedim(-1, 1)  # (B, H, W, C) -> (B, C, H, W)
                    image = common_upscale(image, new_width, new_height, upscale_method, "center")
                    image = image.movedim(1, -1)  # (B, C, H, W) -> (B, H, W, C)

        return (image,)



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
    CATEGORY = "Apt_Preset/flow"

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
    CATEGORY = "Apt_Preset/flow"
    
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

FLOW_VALUE_STORE = {}


def _normalize_flow_var_name(variable):
    if variable is None:
        return ""
    return str(variable).strip()



MAX_FLOW_NUM = 20


from comfy_execution.graph_utils import GraphBuilder, is_link
from comfy_execution.graph import ExecutionBlocker


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
    CATEGORY = "Apt_Preset/flow"

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
    CATEGORY = "Apt_Preset/flow"

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
    CATEGORY = "Apt_Preset/flow"
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
    CATEGORY = "Apt_Preset/flow"

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













