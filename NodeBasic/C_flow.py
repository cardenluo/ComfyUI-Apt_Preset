import folder_paths
from comfy import model_management
from comfy_extras.nodes_post_processing import ImageScaleToTotalPixels
from comfy.utils import common_upscale
import torch
import numpy as np
from PIL import Image
import base64
import io
import json
from typing import Tuple
from server import PromptServer


from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader



#region----------------lowcpu--------------------------

try:
    import pynvml
    pynvml_installed = True
    pynvml.nvmlInit()
except ImportError:
    pynvml_installed = False
    print("警告：未安装pynvml库，auto选项将不可用。")


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


class flow_judge:
 
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "judge": (AlwaysEqual("*"),),
                "true": (AlwaysEqual("*"), {"lazy": True}),
                "false": (AlwaysEqual("*"), {"lazy": True}),
            }
        }

    RETURN_TYPES = (AlwaysEqual("*"),)
    RETURN_NAMES = ("data",)
    FUNCTION = "judge_bool"
    CATEGORY = "Apt_Preset/flow"
    OUTPUT_NODE = False

    def check_lazy_status(self, judge, true, false):
        needed = []
        if judge:
            needed.append('true')
        else:
            needed.append('false')
        return needed


    def judge_bool(self, judge, true, false):
        return {"ui": {"value": [True if judge else False]}, "result": (true if judge else false,)}
    




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




class flow_switch:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_method": ("BOOLEAN", {"default": True, "label_on": "第一个有效值", "label_off": "按编号"}),
                "input_index": ("INT", {"default": 1, "min": 1, "max": 5, "step": 1}),
                "output_method": ("BOOLEAN", {"default": True, "label_on": "按有效输入", "label_off": "按匹配编号"}),
            },
            "optional": {
                "in1": (any_type,),
                "in2": (any_type,),
                "in3": (any_type,),
                "in4": (any_type,),
                "in5": (any_type,),
            }
        }

    RETURN_TYPES = (any_type, any_type, any_type, any_type, any_type,)
    RETURN_NAMES = ('out1', 'out2', 'out3', 'out4', 'out5',)
    CATEGORY = "Apt_Preset/flow"
    FUNCTION = "switch"

    DESCRIPTION = """
    - input_method: 自动检测并选择第一个非空输入数据（第一个有效值）或手动选择输入端口索引（按编号）
    - input_index: 手动选择输入端口索引（1-5），在"按编号"模式下生效
    - output_method: 为真时按选中值输出（所有输出口相同，按有效输入），为假时按输入输出1对1匹配（按匹配编号）
    """

    def switch(self, input_method, input_index, output_method,
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
        
        if output_method:
            output = [selected_value] * 5
        else:
            output = inputs.copy()
        
        return tuple(output)

    def is_none(self, value):
        if value is not None:
            if isinstance(value, dict) and 'model' in value and 'clip' in value:
                return all(v is None for v in value.values())
        return value is None



class XXflow_sch_control:
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
    RETURN_NAMES = ("result",)
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




























