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
import os
import inspect
import nodes
import comfy.utils



from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader



#region----------------lowcpu--------------------------

try:
    import pynvml
    try:
        pynvml.nvmlInit()
        pynvml_installed = True  
    except:
        pynvml_installed = False  
except ImportError:
    pynvml_installed = False


def get_gpu_memory_info():
    """获取GPU显存信息"""
    if not pynvml_installed:
        return None, None
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = memory_info.total / (1024 * 1024 * 1024)  
        used = memory_info.used / (1024 * 1024 * 1024)    
        return total, used
    except Exception as e:
        print(f"获取GPU信息出错: {e}")
        return None, None
#endregion----------------lowcpu--------------------------






class AlwaysEqual(str):
    def __eq__(self, _):
        return True

    def __ne__(self, _):
        return False


class AlwaysTuple(tuple):
    def __getitem__(self, i):
        if i < super().__len__():
            return AlwaysEqual(super().__getitem__(i))
        else:
            return AlwaysEqual(super().__getitem__(-1))


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
    CATEGORY = "Apt_Preset/flow"

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




class flow_low_gpu:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anything": (any_type, {}),
                "reserved": ("FLOAT", {
                    "default": 0.6,
                    "min": -2.0,
                    "step": 0.1,
                    "tooltip": "reserved (GB)"
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
        if mode == "auto":
            if pynvml_installed:
                total, used = get_gpu_memory_info()
                if total and used:
                    auto_reserved = used + reserved
                    auto_reserved = max(0, auto_reserved)  # 确保不小于0
                    model_management.EXTRA_RESERVED_VRAM = int(auto_reserved * 1024 * 1024 * 1024)
                    print(f'set EXTRA_RESERVED_VRAM={auto_reserved:.2f}GB (自动模式: 总显存={total:.2f}GB, 已用={used:.2f}GB)')
                else:
                    model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)
            else:
                model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)
        else:
            # 手动模式
            reserved = max(0, reserved)
            model_management.EXTRA_RESERVED_VRAM = int(reserved * 1024 * 1024 * 1024)

        return (anything,)




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
                     "宽高比>比例阈值,为True", 
                     "长边>边阈值,为True",
                     "短边>边阈值,为True",
                     "高度>边阈值,为True",  
                     "宽度>边阈值,为True",
                     "张量存在,为True",
                     "张量数量>批次阈值,为True",
                     "张量数量=批次阈值,为True",
                     "张量数量<批次阈值,为True",
                     ], ),  # 已移除三个事件

                "area_threshold": ("INT", {"default": 1048576, "min": 1, "max": 9999999999, "step": 1}),
                "ratio_threshold": ("FLOAT", {"default": 1.0, "min": 0.0001, "max": 10000.0, "step": 0.001}),
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
    
    def check_event(self, case_judge, area_threshold,  batch_threshold, ratio_threshold, edge_threshold, data=None) -> Tuple[bool]:
        if data is None:
            raise ValueError("必须输入data参数")
            
        if case_judge == "横向图：宽>高，为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            result = width > height
        
        elif case_judge == "竖向图：高>宽，为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            result = height > width
        
        elif case_judge == "正方图：宽=高，为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            result = width == height
        
        elif case_judge == "分辨率>面积阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            resolution = width * height
            result = resolution > area_threshold
        
        elif case_judge == "张量存在,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                raise ValueError(f"模式 '{case_judge}' 必须输入遮罩、图像、latent类型数据")
            mask_sum = torch.sum(data).item()  
            result = mask_sum > 0  
        
        elif case_judge == "张量数量>批次阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像或遮罩类型数据（3/4维张量）")
            batch_size = data.shape[0]  
            result = batch_size > batch_threshold
        
        elif case_judge == "张量数量=批次阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像或遮罩类型数据（3/4维张量）")
            batch_size = data.shape[0]  
            result = batch_size == batch_threshold
        
        elif case_judge == "张量数量<批次阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) in [3, 4]):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像或遮罩类型数据（3/4维张量）")
            batch_size = data.shape[0]  
            result = batch_size < batch_threshold
        
        elif case_judge == "宽高比>比例阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            if height == 0:
                raise ValueError(f"模式 '{case_judge}' 图像高度不能为0")
            aspect_ratio = width / height
            result = aspect_ratio > ratio_threshold
        
        elif case_judge == "长边>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            long_side = max(width, height)
            result = long_side > edge_threshold
        
        elif case_judge == "短边>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height, width = data.shape[1], data.shape[2]
            short_side = min(width, height)
            result = short_side > edge_threshold
        
        elif case_judge == "高度>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            height = data.shape[1]
            result = height > edge_threshold
        
        elif case_judge == "宽度>边阈值,为True":
            if not (isinstance(data, torch.Tensor) and len(data.shape) == 4):
                raise ValueError(f"模式 '{case_judge}' 必须输入图像类型数据")
            width = data.shape[2]
            result = width > edge_threshold
        
        else:
            raise ValueError(f"不支持的判断模式: {case_judge}")
        
        return (result,)





class flow_sch_control:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                "total": ("INT", {"default": 500, "min": 0, "max": 5000} ),
                "min_value": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "max_value": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "easing": (EASING_TYPES,{"default": "Linear"},
                ),
            },
            "optional": {
            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("INT","FLOAT","FLOAT", "INT",)
    RETURN_NAMES = ("Index","float","normalized","total",)
    CATEGORY = "Apt_Preset/flow"

    def set_range(
        self,
        min_value,
        max_value,
        easing,
        seed,
        total,
    ):
        
        value = seed + 1    
        if total < value:
            raise ValueError("pls stop running")

        try:
            float_value = float(value)
        except ValueError:
            raise ValueError("Invalid value for conversion to float")
        
        if 0 == total:
            normalized_value = 0
        else:
            normalized_value = (float_value - 0) / (total - 0)
        
        normalized_value = max(min(normalized_value, 1), 0)
        eased_value = apply_easing(normalized_value, easing)
        
        res_float = min_value + (max_value - min_value) * eased_value

        return (value, float_value, res_float, total)



class flow_QueueTrigger:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "count": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "total": ("INT", {"default": 10, "min": 1, "max": 0xffffffffffffffff}),
                    "mode": ("BOOLEAN", {"default": True, "label_on": "Trigger", "label_off": "Don't trigger"}),
                    "min_value": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),  # 新增：映射最小值
                    "max_value": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),  # 新增：映射最大值
                    },
                "optional": {},
                "hidden": {"unique_id": "UNIQUE_ID"}
                }

    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/flow"
    RETURN_TYPES = ("INT", "INT", "FLOAT")  # 新增：浮点型重映射结果
    RETURN_NAMES = ("count", "total", "remapped_value")  # 新增输出名称
    OUTPUT_NODE = True
    NAME = "flow_QueueTrigger"


    def doit(self, count, total, mode, min_value, max_value, unique_id):  # 新增参数：min_value, max_value
        # 处理计数逻辑（保持原有逻辑不变）
        if mode:
            if count < total - 1:
                PromptServer.instance.send_sync("node-feedback",
                                                {"node_id": unique_id, "widget_name": "count", "type": "int", "value": count + 1})
                PromptServer.instance.send_sync("add-queue", {})
            elif count >= total - 1:
                PromptServer.instance.send_sync("node-feedback",
                                                {"node_id": unique_id, "widget_name": "count", "type": "int", "value": 0})

        # 新增：重映射逻辑（将count从[0, total-1]映射到[min_value, max_value]）
        if total == 1:
            # 特殊情况：total=1时，count始终为0，直接映射为min_value
            remapped_value = min_value
        else:
            # 归一化count到[0, 1]范围，再映射到目标区间
            normalized = count / (total - 1)
            remapped_value = min_value + (max_value - min_value) * normalized

        # 返回原count、total，以及新增的重映射结果
        return (count, total, remapped_value)



class flow_ValueSender:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "value": (any_typ, ),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": 99999999 ,"step": 1}),
                    },
                "optional": {

                    }
                }

    OUTPUT_NODE = True
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/flow"
    RETURN_TYPES = (any_typ, )
    RETURN_NAMES = ("-", )
    NAME = "flow_ValueSender"


    def doit(self, value, link_id=0, ):
        PromptServer.instance.send_sync("value-send", {"link_id": link_id, "value": value})
        return ()



class flow_ValueReceiver:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "typ": (["STRING", "INT", "FLOAT", "BOOLEAN"], ),
                    "value": ("STRING", {"default": ""}),
                    "link_id": ("INT", {"default": 0, "min": 0, "max": 99999999, "step": 1}),
                    },
                }
    NAME = "flow_ValueReceiver"
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/flow"
    RETURN_TYPES = (any_typ, )
    def doit(self, typ, value, link_id=0):
        if typ == "INT":
            return (int(value), )
        elif typ == "FLOAT":
            return (float(value), )
        elif typ == "BOOLEAN":
            return (value.lower() == "true", )
        else:
            return (value, )





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
            if keep_alpha:
                unified_image = torch.zeros((1, 64, 64, 4), dtype=torch.float32)
            else:
                unified_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        else:
            img_np = image.cpu().numpy().squeeze()
            
            if len(img_np.shape) == 2:
                img_np = np.stack([img_np]*3, axis=-1)
            elif len(img_np.shape) == 3:
                if img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)
                elif img_np.shape[0] == 3:
                    img_np = np.transpose(img_np, (1, 2, 0))
                elif img_np.shape[0] == 4:
                    img_np = np.transpose(img_np, (1, 2, 0))
            
            if img_np.dtype != np.float32:
                img_np = img_np.astype(np.float32) / 255.0 if img_np.max() > 1 else img_np.astype(np.float32)
            
            img_np = np.clip(img_np, 0.0, 1.0)
            
            if keep_alpha and img_np.shape[-1] < 4:
                if img_np.shape[-1] == 3:
                    alpha_channel = np.ones((*img_np.shape[:2], 1), dtype=img_np.dtype)
                    img_np = np.concatenate([img_np, alpha_channel], axis=-1)
                elif img_np.shape[-1] == 1:
                    img_np = np.repeat(img_np, 3, axis=-1)
                    alpha_channel = np.ones((*img_np.shape[:2], 1), dtype=img_np.dtype)
                    img_np = np.concatenate([img_np, alpha_channel], axis=-1)
            elif not keep_alpha and img_np.shape[-1] >= 3:
                img_np = img_np[:, :, :3]
            
            if len(img_np.shape) == 3:
                img_np = img_np[np.newaxis, ...]
            
            unified_image = torch.from_numpy(img_np).to(image.device)
        
        if mask is None:
            unified_mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        else:
            mask_np = mask.cpu().numpy().squeeze()
            
            if len(mask_np.shape) == 3:
                if mask_np.shape[-1] in (1, 3, 4):
                    mask_np = mask_np[..., 0]
                else:
                    mask_np = mask_np[0]
            
            if mask_np.dtype != np.float32:
                mask_np = mask_np.astype(np.float32) / 255.0 if mask_np.max() > 1 else mask_np.astype(np.float32)
            
            mask_np = np.clip(mask_np, 0.0, 1.0)
            
            if len(mask_np.shape) == 2:
                mask_np = mask_np[np.newaxis, ...]
            
            unified_mask = torch.from_numpy(mask_np).to(mask.device)
        
        return (unified_image, unified_mask)






import torch
import numpy as np
from PIL import Image, PngImagePlugin
import os
import folder_paths
from pathlib import Path
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
        self.stored_image = None
        self.stored_mask = None
        self.temp_subfolder = "zml_image_memory_previews"
        self.temp_output_dir = folder_paths.get_temp_directory()
        self.persistence_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image_memory_cache.png")
        self.prompt = None
        self.extra_pnginfo = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "disable_input": ("BOOLEAN", {"default": False}),
                "disable_output": ("BOOLEAN", {"default": False}),
                "select_output_index": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
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
    
    def check_lazy_status(self, disable_input, **kwargs):
        if disable_input:
            return None
        required_inputs = []
        if "image" in kwargs:
            required_inputs.append("image")
        if "mask" in kwargs:
            required_inputs.append("mask")
        return required_inputs

    def store_and_retrieve(self, disable_input, disable_output, select_output_index, image=None, mask=None, prompt=None, extra_pnginfo=None, unique_id=None):
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        
        image_to_output = None
        mask_to_output = None

        if disable_input:
            image_to_output = self.stored_image
            mask_to_output = self.stored_mask
        elif image is not None:
            self.stored_image = image
            self.stored_mask = mask
            image_to_output = image
            mask_to_output = mask
        else:
            image_to_output = self.stored_image
            mask_to_output = self.stored_mask

        if image_to_output is None:
            default_size = 1
            image_to_output = torch.zeros((1, default_size, default_size, 3), dtype=torch.float32, device="cpu")
            
        if mask_to_output is None:
            # Create a default mask matching the image dimensions
            if image_to_output is not None:
                batch_size, height, width, _ = image_to_output.shape
                mask_to_output = torch.ones((batch_size, height, width), dtype=torch.float32, device="cpu")
            else:
                mask_to_output = torch.ones((1, default_size, default_size), dtype=torch.float32, device="cpu")

        subfolder_path = os.path.join(self.temp_output_dir, self.temp_subfolder)
        os.makedirs(subfolder_path, exist_ok=True)

        ui_image_data = []
        batch_size = image_to_output.shape[0]
        
        for i in range(batch_size):
            current_image = image_to_output[i:i+1]
            
            if current_image.shape[1] == 1 and current_image.shape[2] == 1:
                preview_image_tensor = torch.zeros((1, 32, 32, 3), dtype=torch.float32, device=current_image.device)
                pil_image = Image.fromarray((preview_image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
            else:
                pil_image = Image.fromarray((current_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8))

            filename = f"zml_image_memory_batch_{i}_{uuid.uuid4()}.png"
            file_path = os.path.join(subfolder_path, filename)

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

            pil_image.save(file_path, pnginfo=metadata, compress_level=4)
            ui_image_data.append({"filename": filename, "subfolder": self.temp_subfolder, "type": "temp"})
        
        if select_output_index == 0:
            selected_image = image_to_output
            selected_mask = mask_to_output
        else:
            zero_based_index = select_output_index - 1
            selected_index = min(zero_based_index, batch_size - 1) if batch_size > 0 else 0
            selected_image = image_to_output[selected_index:selected_index+1]
            if mask_to_output is not None and selected_index < mask_to_output.shape[0]:
                selected_mask = mask_to_output[selected_index:selected_index+1]
            else:
                selected_mask = mask_to_output[:1] if mask_to_output is not None else None

        if disable_output and ExecutionBlocker is not None:
            output_image = ExecutionBlocker(None)
            output_mask = ExecutionBlocker(None)
        else:
            output_image = selected_image
            output_mask = selected_mask
            
        return {"ui": {"images": ui_image_data}, "result": (output_image, output_mask)}

    def _save_to_local(self, image_tensor):
        try:
            pil_image = Image.fromarray((image_tensor.squeeze(0).cpu().numpy() * 255).astype(np.uint8))
            pil_image.save(self.persistence_file, "PNG")
        except Exception as e:
            print(f"Failed to save image locally: {e}")

    def _load_from_local(self):
        if os.path.exists(self.persistence_file):
            try:
                pil_image = Image.open(self.persistence_file).convert('RGB')
                image_np = np.array(pil_image).astype(np.float32) / 255.0
                return torch.from_numpy(image_np).unsqueeze(0)
            except Exception as e:
                print(f"Failed to load image from local file: {e}")
        return None






class flow_BooleanSwitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "any_input": (any_type,),
                "switch": ("BOOLEAN", {"default": True, "label_on": "On", "label_off": "Off"}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("any_output",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/flow"

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def process(self, switch, any_input=None):
        if switch:
            return (any_input,)
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
                "judge": ("BOOLEAN", {"default": True, "label_on": "True", "label_off": "False"}),
            },
            "optional": {
                "true": (any_type, {"lazy": True}),
                "false": (any_type, {"lazy": True}),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("data",)
    FUNCTION = "judge_bool"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = False

    def check_lazy_status(self, judge, **kwargs):
        needed = []
        if judge:
            needed.append('true')
        else:
            needed.append('false')
        return needed

    def judge_bool(self, judge, true=None, false=None):
        if judge:
            branch_name = "true"
            result_value = true
        else:
            branch_name = "false"
            result_value = false
        
        if result_value is None:
            fallback_value = false if judge else true
            if fallback_value is not None:
                result_value = fallback_value
                branch_name = f"{branch_name} (fallback to {'false' if judge else 'true'})"
            else:
                raise ValueError(
                    f"Boolean judgment node error:\n"
                    f"Currently selected {branch_name} branch, but no data input in this branch, and the other branch is also empty!\n"
                    f"Please ensure at least one branch is connected with valid data."
                )
        
        return {
            "ui": {
                "judge_status": [judge],
                "used_branch": [branch_name]
            },
            "result": (result_value,)
        }




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












