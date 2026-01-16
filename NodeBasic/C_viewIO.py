
from nodes import MAX_RESOLUTION, SaveImage, common_ksampler
import torch
import os
import sys
import folder_paths
import random
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
import json

from comfy.cli_args import args
import numpy as np
import inspect
import re
import traceback
import itertools
import comfy
from server import PromptServer
from PIL import Image, ImageOps, ImageSequence
import node_helpers
import hashlib
import ast
import io
import base64
from typing import List, Dict, Any,Tuple,Optional
import glob
import torch.nn.functional as F




from ..main_unit import *
from ..office_unit import ImageCompositeMasked



#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False








#优先从当前文件所在目录下的 comfy 子目录中查找模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))  

def updateTextWidget(node, widget, text):
    PromptServer.instance.send_sync("view_Data_text_processed", {"node": node, "widget": widget, "text": text})




#region-----------------------收纳-------------------------------------------------------#





class view_mask(SaveImage):
    
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"mask": ("MASK",), },  
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True
    DESCRIPTION = "show mask"
    
    def execute(self, mask, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None):
        # 处理列表类型的遮罩
        if isinstance(mask, list):
            # 存储所有处理后的遮罩
            processed_masks = []
            for m in mask:
                # 确保每个元素都是张量
                if isinstance(m, torch.Tensor):
                    processed = self.process_single_mask(m)
                    processed_masks.append(processed)
            
            # 合并所有遮罩为一个批次
            if processed_masks:
                preview = torch.cat(processed_masks, dim=0)
            else:
                # 处理空列表情况
                return {"ui": {"images": []}}
        # 处理单个张量遮罩
        elif isinstance(mask, torch.Tensor):
            preview = self.process_single_mask(mask)
        else:
            # 处理其他不支持的类型
            return {"ui": {"images": []}}
        
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)
    
    def process_single_mask(self, mask_tensor):
        """处理单个遮罩张量，转换为正确的预览格式"""
        # 根据张量维度进行不同处理
        if mask_tensor.dim() == 2:  # 形状为 (H, W)
            # 添加批次和通道维度: (1, 1, H, W) -> 转换后 (1, H, W, 3)
            return mask_tensor.unsqueeze(0).unsqueeze(0).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask_tensor.dim() == 3:  # 形状为 (B, H, W) 或 (1, H, W)
            # 添加通道维度并转换: (B, 1, H, W) -> (B, H, W, 3)
            return mask_tensor.unsqueeze(1).movedim(1, -1).expand(-1, -1, -1, 3)
        else:  # 其他维度，使用reshape确保兼容性
            return mask_tensor.reshape((-1, 1, mask_tensor.shape[-2], mask_tensor.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
    


class view_combo:     # web_node/view_Data_text.js

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "prompt": ("STRING", {"multiline": True, "default": "text"}),
                    "start_index": ("INT", {"default": 0, "min": 0, "max": 9999}),
                    "max_rows": ("INT", {"default": 1000, "min": 1, "max": 9999}),
                    },
            "hidden":{
                "workflow_prompt": "PROMPT", "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = (any_type, any_type)
    RETURN_NAMES = ("STRING", "COMBO")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "generate_strings"
    CATEGORY = "Apt_Preset/PreView"
    NAME = "view_combo"


    def generate_strings(self, prompt, start_index, max_rows, workflow_prompt=None, my_unique_id=None):
        lines = prompt.split('\n')

        start_index = max(0, min(start_index, len(lines) - 1))
        end_index = min(start_index + max_rows, len(lines))
        rows = lines[start_index:end_index]

        return (rows, rows)




class IO_node_Script:
    def __init__(self):
        self.node_list = []
        self.custom_node_list = []
        self.update_node_list()

    def update_node_list(self):
        try:
            import nodes
            self.node_list = []
            self.custom_node_list = []
            
            for node_name, node_class in nodes.NODE_CLASS_MAPPINGS.items():
                try:
                    module = inspect.getmodule(node_class)
                    module_path = getattr(module, '__file__', '')
                    is_custom = 'custom_nodes' in module_path

                    node_info = {
                        'name': node_name,
                        'class_name': node_class.__name__,
                        'category': getattr(node_class, 'CATEGORY', 'Uncategorized'),
                        'description': getattr(node_class, 'DESCRIPTION', ''),
                        'is_custom': is_custom
                    }
                    
                    self.node_list.append(node_info)
                    if is_custom:
                        self.custom_node_list.append(node_info)
                except Exception as e:
                    logging.error(f"Error processing node {node_name}: {str(e)}")
                    continue
            
            self.node_list.sort(key=lambda x: x['name'])
            self.custom_node_list.sort(key=lambda x: x['name'])
            
        except Exception as e:
            logging.error(f"Error updating node list: {str(e)}")
            traceback.print_exc()

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import nodes
            node_names = sorted(list(nodes.NODE_CLASS_MAPPINGS.keys()))
            if not node_names:
                node_names = ["No nodes found"]
                
            return {
                "required": {
                    "selected_node": (node_names, {
                        "default": node_names[0]
                    }),
                    "search": ("STRING", {
                        "default": "",
                        "multiline": False
                    }),
                    "show_all": ("BOOLEAN", {
                        "default": True,
                        "label": "Show All Nodes"
                    }),
                    "refresh_list": ("BOOLEAN", {
                        "default": False,
                        "label": "Refresh Node List"
                    })
                }
            }
        except Exception as e:
            print(f"Error in INPUT_TYPES: {str(e)}")
            return {
                "required": {
                    "search": ("STRING", {"default": "", "multiline": False}),
                    "show_all": ("BOOLEAN", {"default": True, "label": "Show All Nodes"}),
                    "refresh_list": ("BOOLEAN", {"default": False, "label": "Refresh Node List"})
                }
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("node_source",)
    FUNCTION = "find_script"
    CATEGORY = "Apt_Preset/IO_Port"
    NAME = "IO_node_Script"

    def get_node_source_code(self, node_name):
        try:
            import nodes
            import inspect
            import os

            node_class = nodes.NODE_CLASS_MAPPINGS.get(node_name)
            if not node_class:
                return f"Node '{node_name}' not found"

            module = inspect.getmodule(node_class)
            if not module:
                return f"Could not find module for {node_name}"

            try:
                file_path = inspect.getfile(module)
            except TypeError:
                return f"Could not determine file path for {node_name}"

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
            except Exception as e:
                return f"Error reading file: {str(e)}"

            class_def = f"class {node_class.__name__}:"
            class_start = file_content.find(class_def)
            
            if class_start == -1:
                return f"Could not find class definition for {node_name}"

            lines = file_content[class_start:].split('\n')
            class_lines = []
            indent_level = None

            for line in lines:
                if indent_level is None:
                    if line.strip().startswith('class'):
                        indent_level = len(line) - len(line.lstrip())
                    continue

                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and line.strip():
                    break

                class_lines.append(line)

            source_output = f"=== Node: {node_name} ===\n"
            source_output += f"File: {file_path}\n\n"
            source_output += "=== Source Code ===\n"
            source_output += "\n".join(class_lines)

            return source_output

        except Exception as e:
            return f"Error retrieving source code: {str(e)}"

    def find_script(self, selected_node, search, show_all, refresh_list):
        try:
            if refresh_list:
                self.update_node_list()

            if selected_node:
                source_code = self.get_node_source_code(selected_node)
                return (source_code,)
            return ("Please select a node to view its source code",)

        except Exception as e:
            logging.error(f"Error in find_script: {str(e)}")
            traceback.print_exc()
            return (traceback.format_exc(),)









class IPA_clip_vision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { 
            "clip_name": (folder_paths.get_filename_list("clip_vision"), ),
            "image": ("IMAGE",),
        }}
    RETURN_TYPES = ("CLIP_VISION_OUTPUT",)
    FUNCTION = "combined_process"

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def combined_process(self, clip_name, image):
        # 加载 CLIP Vision 模型
        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_name)
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise RuntimeError("ERROR: clip vision file is invalid and does not contain a valid vision model.")
        
        output = clip_vision.encode_image(image, crop="center")
        return (output,)





class view_Data:   

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "any": (anyType, {"forceInput": True}),
                "data": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (anyType,)  # 只需要一个输出端口
    RETURN_NAMES = ("record",)  # 输出名称
    INPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    NAME = "view_Data"
    CATEGORY = "Apt_Preset/PreView"
    FUNCTION = "process"

    def process(self, data, unique_id, any=None):
        displayText = self.render(any)

        updateTextWidget(unique_id, "data", displayText)
        if isinstance(any, list) and len(any) == 1:
            return {"ui": {"data": displayText}, "result": (any[0],)}
        else:
            return {"ui": {"data": displayText}, "result": (any,)}

    def render(self, any):
        if not isinstance(any, list):
            return str(any)

        listLen = len(any)

        if listLen == 0:
            return ""

        if listLen == 1:
            return str(any[0])

        result = "List:\n"

        for i, element in enumerate(any):
            result += f"- {str(any[i])}\n"

        return result




class view_GetLength:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY" : (ANY_TYPE, {}), 
            },
        }
    
    TITLE = "Get Length"
    RETURN_TYPES = ("INT", )
    RETURN_NAMES = ("length", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True
    NAME = "view_GetLength"

    def run(self, ANY):
        length = len(ANY)
        return { "ui": {"text": (f"{length}",)}, "result": (length, ) }




class view_GetShape:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor" : ("IMAGE,LATENT,MASK", {}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "view_GetShape"
    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("width", "height", "batch_size", "channels")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, input_types):
        return True

    def run(self, tensor, unique_id, prompt, extra_pnginfo):  
        node_list = extra_pnginfo["workflow"]["nodes"]
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        link_id = cur_node["inputs"][0]["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_type = in_node["outputs"][in_socket_id]["type"]
        
        B, H, W, C = 1, 1, 1, 1
        
        if input_type == "IMAGE":
            B, H, W, C = tensor.shape
        elif input_type == "LATENT" or (type(tensor) is dict and "samples" in tensor):
            samples_shape = tensor["samples"].shape
            if len(samples_shape) == 4:
                B, C, H, W = samples_shape
            elif len(samples_shape) == 3:
                B, H, W = samples_shape
                C = 4
            else:
                print(f"Unexpected latent shape: {samples_shape}")
                B, C, H, W = 1, 4, 64, 64
            
            H *= 8
            W *= 8
        else:
            shape = tensor.shape
            if len(shape) == 2:
                H, W = shape
            elif len(shape) == 3:
                B, H, W = shape
            elif len(shape) == 4:
                if shape[3] <= 4:
                    B, H, W, C = tensor.shape
                else:
                    B, C, H, W = shape
        
        return { "ui": {"text": (f"{W}, {H}, {B}, {C}",)}, "result": (W, H, B, C) }



class view_GetWidgetsValues:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY" : (ANY_TYPE, {}), 
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "view_GetWidgetsValues"
    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("LIST", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True

    def run(self, ANY, unique_id, prompt, extra_pnginfo):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        link_id = cur_node["inputs"][0]["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        return { "ui": {"text": (f"{in_node['widgets_values']}",)}, "result": (in_node["widgets_values"], ) }







class XXXview_bridge_image:   
    def __init__(self):
        self.image_id = None
        self.cached_mask = None  

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
            "optional": {
                "mask": ("MASK",), 
                "output_mask": ("BOOLEAN", {"default": False, "label_off": "Refresh Mask", "label_on": "Store Mask"}),  
                "operation": (["+", "-", "*", "&", "None"], {"default": "+"}),
                "image_update": ("IMAGE_FILE",),  

            }
        }

    CATEGORY = "Apt_Preset/PreView"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "edit"
    OUTPUT_NODE = True
    NAME = "view_bridge_image"

    def edit(self, image, mask=None, operation="None", image_update=None, output_mask=False):
        if self.image_id is None:
            self.image_id = tensor_to_hash(image)
            image_update = None
        else:
            image_id = tensor_to_hash(image)
            if image_id != self.image_id:
                image_update = None
                self.image_id = image_id
                # 图像ID变化时重置缓存遮罩
                if not output_mask:
                    self.cached_mask = None

        # 优先使用 image_update 中的图像
        if image_update is not None and 'images' in image_update:
            images = image_update['images']
            filename = images[0]['filename']
            subfolder = images[0]['subfolder']
            type = images[0]['type']
            name, base_dir = folder_paths.annotated_filepath(filename)

            if type.endswith("output"):
                base_dir = folder_paths.get_output_directory()
            elif type.endswith("input"):
                base_dir = folder_paths.get_input_directory()
            elif type.endswith("temp"):
                base_dir = folder_paths.get_temp_directory()

            image_path = os.path.join(base_dir, subfolder, name)
            img = node_helpers.pillow(Image.open, image_path)
        else:
            # 否则使用 preview_image
            if mask is not None:
                try:
                    masked_result = generate_masked_black_image(image, mask)
                    preview_image = masked_result["result"][0]
                except Exception as e:
                    print(f"[Error] Failed to apply mask for preview: {e}")
                    preview_image = image
            else:
                preview_image = image

            image_path, images = create_temp_file(preview_image)
            img = node_helpers.pillow(Image.open, image_path)

        # 从图像中提取 mask
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_pil = i.convert("RGB")

            if len(output_masks) == 0:
                w = image_pil.size[0]
                h = image_pil.size[1]

            if image_pil.size[0] != w or image_pil.size[1] != h:
                continue

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = 1. - torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_masks.append(mask_tensor.unsqueeze(0))

        if len(output_masks) > 1 and img.format not in excluded_formats:
            output_mask_val = torch.cat(output_masks, dim=0)
        else:
            output_mask_val = output_masks[0] if output_masks else torch.zeros_like(image[0, :, :, 0])

        # 新增 Mask 运算逻辑
        mask1 = mask
        mask2 = output_mask_val

        # 计算当前运算结果
        if mask1 is None or operation == "None":
            current_result = mask2
        else:
            invert_mask1 = False
            invert_mask2 = False

            if invert_mask1:
                mask1 = 1 - mask1
            if invert_mask2:
                mask2 = 1 - mask2

            if mask1.dim() == 2:
                mask1 = mask1.unsqueeze(0)
            if mask2.dim() == 2:
                mask2 = mask2.unsqueeze(0)

            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            if mask1.shape != (b, h, w):
                mask1 = torch.zeros((b, h, w), dtype=mask1.dtype, device=mask1.device)
            if mask2.shape != (b, h, w):
                mask2 = torch.zeros((b, h, w), dtype=mask2.dtype, device=mask2.device)

            algorithm = "torch"  # 简化逻辑，直接使用torch

            if algorithm == "torch":
                if operation == "-":
                    current_result = torch.clamp(mask1 - mask2, min=0, max=1)
                elif operation == "+":
                    current_result = torch.clamp(mask1 + mask2, min=0, max=1)
                elif operation == "*":
                    current_result = torch.clamp(mask1 * mask2, min=0, max=1)
                elif operation == "&":
                    current_result = (torch.round(mask1).bool() & torch.round(mask2).bool()).float()
                else:
                    current_result = mask2  # 默认操作为 mask2

        # 根据output_mask控制是否保留遮罩
        if output_mask:
            # 如果是第一次启用启用保留，缓存当前结果
            if self.cached_mask is None:
                # 为避免显存问题，只在需要时保存缓存，并将其移至CPU
                self.cached_mask = current_result.detach().cpu()
            # 使用缓存的遮罩作为结果（需要时移回GPU）
            final_mask = self.cached_mask.to(current_result.device) if self.cached_mask.device != current_result.device else self.cached_mask
        else:
            # 不保留时更新缓存为当前结果
            self.cached_mask = current_result.detach().cpu()  # 移至CPU以节省GPU显存
            final_mask = current_result

        # 返回结果
        return {"ui": {"images": images}, "result": (image, final_mask)}

    # 以下静态方法保持不变
    @staticmethod
    def subtract_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def add_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def multiply_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def and_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            print("Warning: The two masks have different shapes")
            return mask1







class view_bridge_Text:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "text": ("STRING", {"forceInput": True}),
                "display": ("STRING", {"default": "", "multiline": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    OUTPUT_NODE = True
    NAME = "view_bridge_Text"
    CATEGORY = "Apt_Preset/PreView"
    FUNCTION = "process"
 
    def __init__(self):
        self.last_text = None
        # 新增：标记是否是首次执行（解决第一次无显示问题）
        self.is_first_run = True

    def process(self, text="", display="", unique_id=None):
        if isinstance(text, list):
            current_text = text[0] if text else ""
        else:
            current_text = text

        # 核心修改：区分「首次执行」和「内容无变化」
        if self.is_first_run:
            # 首次执行：强制使用 current_text，同时标记首次执行结束
            use_simple = False
            self.is_first_run = False
        else:
            # 非首次：保留原有逻辑（内容无变化时用 display）
            use_simple = (current_text == self.last_text)

        self.last_text = current_text

        if use_simple:
            input_text = display
        else:
            input_text = current_text

        displayText = self.render(input_text)
        updateTextWidget(unique_id, "display", displayText)
        
        return {"ui": {"display": displayText}, "result": (input_text,)}

    def render(self, input):
        if not isinstance(input, list):
            return input

        listLen = len(input)

        if listLen == 0:
            return ""

        if listLen == 1:
            return input[0]

        result = "List:\n"

        for i, element in enumerate(input):
            result += f"- {element}\n"
        return result




#endregion-----------------------旧-------------------------------------------------------#.


class view_Mask_And_Img(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),                
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }


    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True
    NAME = "view_Mask_And_Img"

    def execute(self, mask_opacity, filename_prefix="ComfyUI", image=None, mask=None, prompt=None, extra_pnginfo=None):
        if mask is not None and image is None:
            preview = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask is None and image is not None:
            if image.shape[-1] == 4:
                image = image[..., :3]
            preview = image
        elif mask is not None and image is not None:
            if image.shape[-1] == 4:
                image = image[..., :3]
            mask_adjusted = mask * mask_opacity
            mask_image = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3).clone()
            color_list = [0, 0, 0]
            mask_image[:, :, :, 0] = color_list[0] / 255
            mask_image[:, :, :, 1] = color_list[1] / 255
            mask_image[:, :, :, 2] = color_list[2] / 255
            preview, = ImageCompositeMasked.composite(self, image, mask_image, 0, 0, True, mask_adjusted)
        else:
            preview = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")

        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)



class IO_input_any:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", }),
            },
            "optional": {
                "split_preset": (
                    ["Custom", "None", "Line", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe"],
                    {"default": "None"}
                ),
                "delimiter": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "output_type": (["float", "int", "string", "anytype", "dictionary", "set", "tuple", "boolean"], {"default": "anytype"}),
            }
        }

    RETURN_TYPES = (ANY_TYPE, "LIST")
    RETURN_NAMES = ("data", "list")
    FUNCTION = "process_text"
    CATEGORY = "Apt_Preset/IO_Port"
    OUTPUT_IS_LIST = (True, False)
    DESCRIPTION = """
    文本拆分预设说明
    None：不做任何处理。
    Line：按行拆分。
    Space：按空格拆分。
    Comma：按逗号拆分。
    Period：按句号拆分。
    Semicolon：按分号拆分。
    Tab：用\\t（制表符）拆分。
    Pipe：以|（竖线）拆分。
    Custom：使用自定义分隔符
    """ 


    def process_text(self, text, split_preset="None", delimiter="", output_type="anytype"):
        preset_map = {
            "None": [],
            "Line": ["\n", "\r\n"],
            "Space": [" ", "　"],
            "Comma": [",", "，"],
            "Period": [".", "。"],
            "Semicolon": [";", "；"],
            "Tab": ["\t"],
            "Pipe": ["|"],
            "Custom": []
        }
        separators = preset_map.get(split_preset, [])
        
        # 修改部分：只有当split_preset为"Custom"时才使用自定义分隔符
        # 当split_preset为"None"时，即使提供了delimiter也不使用
        if split_preset == "Custom" and delimiter:
            delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\s", " ").replace("\\r", "\r")
            separators = [delimiter]
        
        text = text.strip()
        
        if output_type == "boolean":
            try:
                if text.lower() in ["true", "yes", "1", "on"]:
                    return ([True], [text])
                elif text.lower() in ["false", "no", "0", "off"]:
                    return ([False], [text])
                else:
                    return ([bool(ast.literal_eval(text))], [text])
            except Exception as e:
                print(f"解析布尔值失败: {e}")
                return ([False], [text])
        
        if output_type == "dictionary" or (output_type == "anytype" and text.startswith("{") and text.endswith("}")):
            try:
                parsed = ast.literal_eval(text)
                if not isinstance(parsed, dict):
                    raise ValueError("解析结果不是字典")
                return ([parsed], [text])
            except Exception as e:
                print(f"解析字典失败: {e}")
        
        elif output_type == "set" or (output_type == "anytype" and text.startswith("{") and text.endswith("}") and ":" not in text):
            try:
                if text == "{}":
                    parsed = set()
                else:
                    set_content = text[1:-1]
                    parsed = set(ast.literal_eval(f"({set_content},)"))
                return ([parsed], [text])
            except Exception as e:
                print(f"解析集合失败: {e}")
        
        elif output_type == "tuple" or (output_type == "anytype" and (text.startswith("(") and text.endswith(")") or "," in text)):
            try:
                if text == "()":
                    parsed = ()
                elif text.endswith(",") and not text.startswith("("):
                    parsed = ast.literal_eval(f"({text})")
                else:
                    parsed = ast.literal_eval(text)
                if not isinstance(parsed, tuple):
                    parsed = (parsed,)
                return ([parsed], [text])
            except Exception as e:
                print(f"解析元组失败: {e}")
        
        # 修改部分：当split_preset为"None"时不进行任何分割
        if split_preset == "None":
            items = [text] if text else []
        elif separators:
            escaped_seps = [re.escape(sep) for sep in separators if sep]
            sep_pattern = '|'.join(escaped_seps)
            items = re.split(f'(?:{sep_pattern})', text)
        else:
            # 只有当不是"None"且没有明确分隔符时才使用默认分割
            items = re.split(r'[\s,]+', text)
        
        items = [item.strip() for item in items if item.strip()]
        
        converted_result = []
        for item in items:
            if output_type == "int":
                try:
                    converted_result.append(int(item))
                except ValueError:
                    converted_result.append(0)
            elif output_type == "float":
                try:
                    converted_result.append(float(item))
                except ValueError:
                    converted_result.append(0.0)
            elif output_type == "string":
                converted_result.append(item)
            elif output_type == "boolean":
                if item.lower() in ["true", "yes", "1", "on"]:
                    converted_result.append(True)
                elif item.lower() in ["false", "no", "0", "off"]:
                    converted_result.append(False)
                else:
                    converted_result.append(bool(item))
            elif output_type == "anytype":
                try:
                    num = int(item)
                    converted_result.append(num)
                except ValueError:
                    try:
                        num = float(item)
                        converted_result.append(num)
                    except ValueError:
                        if item.lower() in ["true", "yes", "1", "on"]:
                            converted_result.append(True)
                        elif item.lower() in ["false", "no", "0", "off"]:
                            converted_result.append(False)
                        else:
                            converted_result.append(item)
        
        return (converted_result, items)






class IO_load_anyimage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "file_path": ("STRING", {"default": "./input/Apt_in",}),
            },
            "optional": {
                "max_images": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "keyword_filter": ("STRING", {"default": "", "multiline": False})
            }
        }
    RETURN_TYPES = ('IMAGE', 'MASK',)
    FUNCTION = "get_transparent_image"
    CATEGORY = "Apt_Preset/IO_Port"
    
    def get_transparent_image(self, file_path, max_images=0, keyword_filter=""):
        try:
            mask_tensor = None
            
            file_path = file_path.strip('"')
            
            if os.path.isdir(file_path):
                images = []
                
                image_files = [f for f in os.listdir(file_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                
                if keyword_filter:
                    image_files = [f for f in image_files if keyword_filter in f]
                
                image_files.sort()
                
                if max_images > 0:
                    image_files = image_files[:max_images]
                
                for filename in image_files:
                    img_path = os.path.join(file_path, filename)
                    image = Image.open(img_path).convert('RGBA')
                    images.append(image)
                
                if not images:
                    return None, None
                
                target_size = images[0].size
                resized_images = []
                for image in images:
                    if image.size != target_size:
                        image = image.resize(target_size, Image.BILINEAR)
                    resized_images.append(image)
                
                batch_images = np.stack([np.array(img) for img in resized_images], axis=0).astype(np.float32) / 255.0
                batch_tensor = torch.from_numpy(batch_images)
                
                return batch_tensor, mask_tensor        
            else:
                image = Image.open(file_path)
                if image is not None:
                    image_rgba = image.convert('RGBA')
                    
                    if keyword_filter and keyword_filter not in os.path.basename(file_path):
                        print(f"文件 {file_path} 不包含关键字 '{keyword_filter}'，跳过加载")
                        return None, None
                    
                    image_np = np.array(image_rgba).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None, :, :, :]
            
                    return (image_tensor, mask_tensor)
            
        except Exception as e:
            print(f"出错请重置节点：{e}")
        return None, None




class IO_image_select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "1,2"}),
                "canvas_operations": (["None", "Horizontal Flip", "Vertical Flip", "90 Degree Rotation", 
                                       "180 Degree Rotation", "Horizontal Flip + 90 Degree Rotation", "Horizontal Flip + 180 Degree Rotation"], {"default": "None"}),
            },
            "optional": {
                # 已移除 index 参数
            }
        }
    
    CATEGORY = "Apt_Preset/IO_Port"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("select_img", "exclude_img")
    FUNCTION = "SelectImages"

    DESCRIPTION = """indexes按图像索引选择输出
    索引方式：正 “1,3,5” 逆向 “-1,-3”（-1为最后一张）
    范围方式： “2-4”"""
    
    def parse_indexes(self, indexes_str, max_length):
        # 此方法保持不变
        indexes = []
        parts = [p.strip() for p in indexes_str.split(',') if p.strip()]
        for part in parts:
            if '-' in part and not part.startswith('-'):
                try:
                    start, end = part.split('-', 1)
                    start = int(start.strip())
                    end = int(end.strip())
                    start = start if start > 0 else max_length + start + 1
                    end = end if end > 0 else max_length + end + 1
                    if start > end:
                        start, end = end, start
                    start = max(1, min(start, max_length))
                    end = max(1, min(end, max_length))
                    indexes.extend(range(start, end + 1))
                except ValueError:
                    continue
            else:
                try:
                    num = int(part)
                    if num < 0:
                        num = max_length + num + 1
                    if 1 <= num <= max_length:
                        indexes.append(num)
                except ValueError:
                    continue
        seen = set()
        unique_indexes = []
        for num in indexes:
            if num not in seen:
                seen.add(num)
                unique_indexes.append(num)
        return unique_indexes
    
    def SelectImages(self, images, indexes, canvas_operations):
        # 移除了 index 参数
        max_length = len(images)
        if max_length == 0:
            return (images, [])
        
        # 直接使用 indexes 解析逻辑
        select_numbers = self.parse_indexes(indexes, max_length)
        
        if not select_numbers:
            print("Warning: No valid indexes found, return original input.")
            return (images, [])
        
        select_list1 = np.array(select_numbers) - 1  # 转换为0-based索引
        exclude_list = np.setdiff1d(np.arange(max_length), select_list1)
        print(f"Selected images (1-based): {select_numbers}")
        
        selected_images = images[torch.tensor(select_list1, dtype=torch.long)]
        excluded_images = images[torch.tensor(exclude_list, dtype=torch.long)] if len(exclude_list) > 0 else []
        
        def apply_operation(images_tensor):
            if not isinstance(images_tensor, torch.Tensor) or len(images_tensor) == 0:
                return images_tensor
            if canvas_operations == "Horizontal Flip":
                return torch.flip(images_tensor, [2])
            elif canvas_operations == "Vertical Flip":
                return torch.flip(images_tensor, [1])
            elif canvas_operations == "90 Degree Rotation":
                return torch.rot90(images_tensor, 1, [1, 2])
            elif canvas_operations == "180 Degree Rotation":
                return torch.rot90(images_tensor, 2, [1, 2])
            elif canvas_operations == "Horizontal Flip + 90 Degree Rotation":
                flipped = torch.flip(images_tensor, [2])
                return torch.rot90(flipped, 1, [1, 2])
            elif canvas_operations == "Horizontal Flip + 180 Degree Rotation":
                flipped = torch.flip(images_tensor, [2])
                return torch.rot90(flipped, 2, [1, 2])
            return images_tensor
        
        selected_images = apply_operation(selected_images)
        excluded_images = apply_operation(excluded_images)
        
        return (selected_images, excluded_images)



class IO_getFilePath:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        comfy_folder_options = list(folder_paths.folder_names_and_paths.keys()) if (folder_paths and hasattr(folder_paths, 'folder_names_and_paths')) else []
        return {
            "required": {
                "folder_path": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "绝对路径或相对路径"
                }),
                "recursive": ("BOOLEAN", {
                    "default": True,
                    "label_on": "递归",
                    "label_off": "非递归",
                    "tooltip": "递归，展开所有文件夹里的文件"
                }),
                "file_extensions": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "过滤扩展名（例：.png,.jpg 多个用逗号分隔，留空匹配所有文件）"
                }),
            },
            "optional": {
                "comfy_folder": (comfy_folder_options,),
            }
        }

    RETURN_TYPES = ("STRING", "LIST",)
    RETURN_NAMES = ("文件路径", "路径列表",)
    FUNCTION = "get_file_paths"
    CATEGORY = "Apt_Preset/IO_Port"
    OUTPUT_IS_LIST = (False, True)

    def get_file_paths(
        self,
        folder_path: str,
        recursive: bool = True,
        file_extensions: str = "",
        comfy_folder: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        folder_path = folder_path.strip().strip('"')
        
        if comfy_folder and folder_paths and hasattr(folder_paths, 'folder_names_and_paths'):
            comfy_folder_data = folder_paths.folder_names_and_paths.get(comfy_folder)
            if comfy_folder_data and isinstance(comfy_folder_data, (list, tuple)) and len(comfy_folder_data) > 0:
                base_paths = comfy_folder_data[0]
                if isinstance(base_paths, (list, tuple)) and len(base_paths) > 0:
                    base_path = base_paths[0]
                else:
                    base_path = base_paths

                if isinstance(base_path, str) and folder_path.strip():
                    folder_path = os.path.join(base_path, folder_path.strip())
                elif isinstance(base_path, str):
                    folder_path = base_path

        if folder_path.strip():
            if not os.path.isabs(folder_path):
                base_dir = folder_paths.base_path if (folder_paths and hasattr(folder_paths, 'base_path')) else ""
                folder_path = os.path.abspath(os.path.join(base_dir, folder_path))

            if not os.path.isdir(folder_path):
                print(f"警告：文件夹不存在或不是目录：{folder_path}")
                return ("", [])
        else:
            print("警告：文件夹路径为空")
            return ("", [])

        extensions = []
        if file_extensions.strip():
            extensions = [ext.strip().lower() for ext in file_extensions.split(",") if ext.strip()]

        file_paths = []

        try:
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if extensions:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext not in extensions:
                            continue

                    full_path = os.path.abspath(os.path.join(root, file))
                    file_paths.append(full_path)

                if not recursive:
                    break
        except Exception as e:
            print(f"获取文件路径失败：{str(e)}")
            return ("", [])

        file_paths.sort()
        paths_string = "\n".join(file_paths) if file_paths else ""
        paths_list = file_paths

        return (paths_string, paths_list)




class IO_save_image:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "file_format": (["png", "webp", "jpg", "tif"],),
                "output_path": ("STRING", {"default": "./output/Apt", "multiline": False}),
                "filename_mid": ("STRING", {"default": "Apt"}),
            },
            "optional": {
                "number_prefix": ("BOOLEAN", {"default": False, "label_on": "前置编号", "label_off": "后置编号"}),
                "number_digits": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1, "tooltip": "编号位数，如设置为3则为001格式"}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("STRING", )
    RETURN_NAMES = ("out_path",)
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "Apt_Preset/IO_Port"

    @staticmethod
    def find_highest_numeric_value(directory, filename_mid):
        highest_value = -1
        if not os.path.exists(directory):
            return highest_value
        for filename in os.listdir(directory):
            if filename.startswith(filename_mid):
                try:
                    numeric_part = filename[len(filename_mid):]
                    numeric_str = re.search(r'\d+', numeric_part).group()
                    numeric_value = int(numeric_str)
                    if numeric_value > highest_value:
                        highest_value = numeric_value
                except (ValueError, AttributeError):
                    continue
        return highest_value

    def save_image(self, image, file_format, filename_mid="Apt", output_path="", number_prefix=False, number_digits=5,
                   save_workflow_as_json=False, prompt=None, extra_pnginfo=None):
        batch_size = image.shape[0]
        images_list = [image[i:i + 1, ...] for i in range(batch_size)]
        output_dir = folder_paths.get_output_directory()
        output_paths = []

        if isinstance(output_path, str):
            os.makedirs(output_path, exist_ok=True)
            output_paths = [output_path] * batch_size
        elif isinstance(output_path, list) and len(output_path) == batch_size:
            for path in output_path:
                os.makedirs(path, exist_ok=True)
            output_paths = output_path
        else:
            print("Invalid output_path format. Using default output directory.")
            output_paths = [output_dir] * batch_size

        base_dir = output_paths[0]
        counter = self.find_highest_numeric_value(base_dir, filename_mid) + 1
        absolute_paths = []

        for idx, img_tensor in enumerate(images_list):
            output_image = img_tensor.cpu().numpy()
            img_np = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np[0])
            out_path = output_paths[idx]

            numbering = f"{counter + idx:0{number_digits}d}"
            if number_prefix:
                output_filename = f"{numbering}_{filename_mid}"
            else:
                output_filename = f"{filename_mid}_{numbering}"

            resolved_image_path = os.path.join(out_path, f"{output_filename}.{file_format}")
            img_params = {
                'png': {'compress_level': 4},
                'webp': {'method': 6, 'lossless': False, 'quality': 80},
                'jpg': {'quality': 95, 'format': 'JPEG'},
                'tif': {'format': 'TIFF'}
            }
            img.save(resolved_image_path, **img_params[file_format])
            absolute_paths.append(os.path.abspath(resolved_image_path))

            if save_workflow_as_json:
                try:
                    workflow = (extra_pnginfo or {}).get('workflow')
                    if workflow is not None:
                        json_file_path = os.path.join(out_path, f"{output_filename}.json")
                        with open(json_file_path, 'w') as f:
                            json.dump(workflow, f)
                except Exception as e:
                    print(f"Failed to save workflow JSON: {e}")

        return (absolute_paths, )





#region-------------IO_store_image-------------
import torch
import numpy as np
import io
import base64
import json
from PIL import Image
from typing import Optional, Dict, Any, List

GLOBAL_STORED_IMAGES: List[torch.Tensor] = []
GLOBAL_DISPLAY_DATA: List[Dict[str, Any]] = []

class IO_store_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image": ("IMAGE", {}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                "release_total": ("INT", {"default": 0, "min": 0, "step": 1}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }
    NAME = "IO_store_image"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "total")
    FUNCTION = "store_image"
    CATEGORY = "Apt_Preset/IO_Port"
    OUTPUT_NODE = True
    DESCRIPTION = """
    输出逻辑：
    - 尺寸一致时：
      - release_total ≤ 0 → 输出全部存储的图像
      - release_total > 0 → 
        1. 存储总数 = release_total → 输出全部存储的图像
        2. 存储总数 < release_total → 用最后一张图像尺寸的白底图补齐数量
        3. 存储总数 > release_total → 输出倒数 release_total 张图像
    - 尺寸不一致时：仅输出最后一张图像
"""

    def __init__(self):
        global GLOBAL_STORED_IMAGES, GLOBAL_DISPLAY_DATA
        GLOBAL_STORED_IMAGES = []
        GLOBAL_DISPLAY_DATA = []
        print("IO_store_image node initialized (storage reset)")

    def store_image(self, image: Optional[torch.Tensor] = None, 
                   prompt: Any = None, image_output: str = None, 
                   extra_pnginfo: Any = None, release_total: float = 0) -> Dict[str, Any]:
        global GLOBAL_STORED_IMAGES, GLOBAL_DISPLAY_DATA
        
        if image is not None:
            if not GLOBAL_STORED_IMAGES:
                GLOBAL_STORED_IMAGES.append(image)
                GLOBAL_DISPLAY_DATA.append(self._prepare_image_display(image))
            else:
                last_img = GLOBAL_STORED_IMAGES[-1]
                if image.shape != last_img.shape:
                    GLOBAL_STORED_IMAGES.append(image)
                    GLOBAL_DISPLAY_DATA.append(self._prepare_image_display(image))
                else:
                    if not torch.allclose(image, last_img):
                        GLOBAL_STORED_IMAGES.append(image)
                        GLOBAL_DISPLAY_DATA.append(self._prepare_image_display(image))
        
        total_stored = len(GLOBAL_STORED_IMAGES)
        output_image = None

        if total_stored > 0:
            channels = [img.shape[1] for img in GLOBAL_STORED_IMAGES]
            if len(set(channels)) > 1:
                print("Warning: Images have different channel counts, outputting last image")
                output_image = GLOBAL_STORED_IMAGES[-1]
                current_total = total_stored
                return self._prepare_return_data(output_image, current_total, image_output, prompt, extra_pnginfo)
        
        if total_stored > 0:
            sizes = [(img.shape[2], img.shape[3]) for img in GLOBAL_STORED_IMAGES]
            sizes_consistent = len(set(sizes)) == 1
        else:
            sizes_consistent = False
        
        if total_stored == 0:
            output_image = image
            current_total = 0
        else:
            if not sizes_consistent:
                output_image = GLOBAL_STORED_IMAGES[-1]
                current_total = total_stored
            else:
                release_total = int(release_total)
                if release_total <= 0:
                    output_image = torch.cat(GLOBAL_STORED_IMAGES, dim=0)
                    current_total = total_stored
                else:
                    if total_stored == release_total:
                        output_image = torch.cat(GLOBAL_STORED_IMAGES, dim=0)
                        current_total = total_stored
                    elif total_stored < release_total:
                        last_img = GLOBAL_STORED_IMAGES[-1]
                        white_img = torch.ones_like(last_img)
                        need_white = release_total - total_stored
                        extended_images = GLOBAL_STORED_IMAGES + [white_img] * need_white
                        output_image = torch.cat(extended_images, dim=0)
                        current_total = release_total
                    else:
                        start_idx = total_stored - release_total
                        selected_imgs = GLOBAL_STORED_IMAGES[start_idx:]
                        output_image = torch.cat(selected_imgs, dim=0)
                        current_total = release_total
        
        if output_image is None:
            output_image = image
            current_total = 0

        return self._prepare_return_data(output_image, current_total, image_output, prompt, extra_pnginfo)

    def _prepare_image_display(self, image_tensor: torch.Tensor) -> Dict[str, Any]:
        try:
            img_np = image_tensor[0].cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))
            img_np = (img_np * 255).astype(np.uint8)
            
            buffer = io.BytesIO()
            Image.fromarray(img_np).save(buffer, format="PNG")
            img_b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            
            return {
                "type": "image",
                "shape": image_tensor.shape,
                "size": f"{image_tensor.shape[2]}x{image_tensor.shape[3]}",
                "data": img_b64,
                "index": len(GLOBAL_STORED_IMAGES)
            }
        except Exception as e:
            return {
                "type": "image",
                "error": str(e),
                "shape": image_tensor.shape if isinstance(image_tensor, torch.Tensor) else "invalid"
            }
    
    def _prepare_return_data(self, output_image: torch.Tensor, current_total: int, 
                            image_output: str, prompt: Any, extra_pnginfo: Any) -> Dict[str, Any]:
        try:
            results = []
            for img in GLOBAL_STORED_IMAGES:
                results.extend(easySave(img, 'easyPreview', image_output, prompt, extra_pnginfo))
        except NameError:
            results = []
            print("Warning: easySave function not found")
        
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {}, "result": (output_image, current_total)}
        return {"ui": {"images": results}, "result": (output_image, current_total)}

    @classmethod
    def IS_CHANGED(cls, image: Optional[torch.Tensor] = None, 
                   release_total: float = 0, image_output: str = None) -> str:
        img_id = f"{image.shape}-{id(image)}" if isinstance(image, torch.Tensor) else "none"
        return json.dumps({
            "image_id": img_id, 
            "release_total": int(release_total),
            "image_output": image_output
        })

    def get_display_content(self) -> Dict[str, Any]:
        return {
            "total_images": len(GLOBAL_STORED_IMAGES),
            "images": GLOBAL_DISPLAY_DATA,
            "last_updated": str(len(GLOBAL_STORED_IMAGES))
        }

def __reload__(module):
    global GLOBAL_STORED_IMAGES, GLOBAL_DISPLAY_DATA
    GLOBAL_STORED_IMAGES = []
    GLOBAL_DISPLAY_DATA = []
    print("IO_store_image module reloaded (storage reset)")

#endregion-------------IO_store_image------------------








import os
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence
import folder_paths
import node_helpers

def tensor_to_hash(tensor):
    return hash(tuple(tensor.cpu().numpy().ravel()[:1000]))

def tensor2pil(image):
    img_np = np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)
    if len(img_np.shape) == 4:img_np = img_np[0]
    while len(img_np.shape) > 3:img_np = img_np.squeeze(0)
    return Image.fromarray(img_np)

def create_temp_file(image):
    import tempfile
    temp_dir = folder_paths.get_temp_directory()
    temp_path = os.path.join(temp_dir, f"temp_{hash(image)}.png")
    img = tensor2pil(image)
    img.save(temp_path, format='PNG')
    return temp_path, [{"filename": os.path.basename(temp_path), "subfolder": "", "type": "temp"}]





class view_bridge_image:   
    def __init__(self):
        self.image_id = None
        self.cached_mask = None  

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            },
            "optional": {
                "mask": ("MASK",), 
                "output_mask": ("BOOLEAN", {"default": False, "label_off": "Refresh Mask", "label_on": "Store Mask"}),  
                "operation": (["+", "-", "*", "&", "None"], {"default": "+"}),
                "image_update": ("IMAGE_FILE",),  

            }
        }

    CATEGORY = "Apt_Preset/PreView"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "edit"
    OUTPUT_NODE = True
    NAME = "view_bridge_image"

    def edit(self, image, mask=None, operation="None", image_update=None, output_mask=False):
        if self.image_id is None:
            self.image_id = tensor_to_hash(image)
            image_update = None
        else:
            image_id = tensor_to_hash(image)
            if image_id != self.image_id:
                image_update = None
                self.image_id = image_id
                # 图像ID变化时重置缓存遮罩
                if not output_mask:
                    self.cached_mask = None

        # 优先使用 image_update 中的图像
        if image_update is not None and 'images' in image_update:
            images = image_update['images']
            filename = images[0]['filename']
            subfolder = images[0]['subfolder']
            type = images[0]['type']
            name, base_dir = folder_paths.annotated_filepath(filename)

            if type.endswith("output"):
                base_dir = folder_paths.get_output_directory()
            elif type.endswith("input"):
                base_dir = folder_paths.get_input_directory()
            elif type.endswith("temp"):
                base_dir = folder_paths.get_temp_directory()

            image_path = os.path.join(base_dir, subfolder, name)
            img = node_helpers.pillow(Image.open, image_path)
        else:
            # 否则使用 preview_image
            if mask is not None:
                try:
                    masked_result = generate_masked_black_image(image, mask)
                    preview_image = masked_result["result"][0]
                except Exception as e:
                    print(f"[Error] Failed to apply mask for preview: {e}")
                    preview_image = image
            else:
                preview_image = image

            image_path, images = create_temp_file(preview_image)
            img = node_helpers.pillow(Image.open, image_path)

        # 从图像中提取 mask
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image_pil = i.convert("RGB")

            if len(output_masks) == 0:
                w = image_pil.size[0]
                h = image_pil.size[1]

            if image_pil.size[0] != w or image_pil.size[1] != h:
                continue

            if 'A' in i.getbands():
                mask_np = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask_tensor = 1. - torch.from_numpy(mask_np)
            else:
                mask_tensor = torch.zeros((h, w), dtype=torch.float32, device="cpu")

            output_masks.append(mask_tensor.unsqueeze(0))

        if len(output_masks) > 1 and img.format not in excluded_formats:
            output_mask_val = torch.cat(output_masks, dim=0)
        else:
            output_mask_val = output_masks[0] if output_masks else torch.zeros_like(image[0, :, :, 0])

        # 新增 Mask 运算逻辑
        mask1 = mask
        mask2 = output_mask_val

        # 计算当前运算结果
        if mask1 is None or operation == "None":
            current_result = mask2
        else:
            invert_mask1 = False
            invert_mask2 = False

            if invert_mask1:
                mask1 = 1 - mask1
            if invert_mask2:
                mask2 = 1 - mask2

            if mask1.dim() == 2:
                mask1 = mask1.unsqueeze(0)
            if mask2.dim() == 2:
                mask2 = mask2.unsqueeze(0)

            b, h, w = image.shape[0], image.shape[1], image.shape[2]
            if mask1.shape != (b, h, w):
                mask1 = torch.zeros((b, h, w), dtype=mask1.dtype, device=mask1.device)
            if mask2.shape != (b, h, w):
                mask2 = torch.zeros((b, h, w), dtype=mask2.dtype, device=mask2.device)

            algorithm = "torch"  # 简化逻辑，直接使用torch

            if algorithm == "torch":
                if operation == "-":
                    current_result = torch.clamp(mask1 - mask2, min=0, max=1)
                elif operation == "+":
                    current_result = torch.clamp(mask1 + mask2, min=0, max=1)
                elif operation == "*":
                    current_result = torch.clamp(mask1 * mask2, min=0, max=1)
                elif operation == "&":
                    current_result = (torch.round(mask1).bool() & torch.round(mask2).bool()).float()
                else:
                    current_result = mask2  # 默认操作为 mask2

        # 根据output_mask控制是否保留遮罩
        if output_mask:
            # 如果是第一次启用启用保留，缓存当前结果
            if self.cached_mask is None:
                # 为避免显存问题，只在需要时保存缓存，并将其移至CPU
                self.cached_mask = current_result.detach().cpu()
            # 使用缓存的遮罩作为结果（需要时移回GPU）
            final_mask = self.cached_mask.to(current_result.device) if self.cached_mask.device != current_result.device else self.cached_mask
        else:
            # 不保留时更新缓存为当前结果
            self.cached_mask = current_result.detach().cpu()  # 移至CPU以节省GPU显存
            final_mask = current_result

        # 返回结果
        return {"ui": {"images": images}, "result": (image, final_mask)}

    # 以下静态方法保持不变
    @staticmethod
    def subtract_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def add_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def multiply_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            print("Warning: The two masks have different shapes")
            return mask1

    @staticmethod
    def and_masks(mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            print("Warning: The two masks have different shapes")
            return mask1









def handle_error_safe(e: Exception, msg: str = "Operation failed", port_count: int = 1):
    print(f"[CCNotes] {msg}: {e}")
    return tuple([[""] for _ in range(port_count)])





class view_Primitive:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }
    NAME = "view_Primitive"
    RETURN_TYPES = tuple(any_type for _ in range(15))
    RETURN_NAMES = tuple(f"widget_input_{i+1}" for i in range(15))
    FUNCTION = "proxy_widget"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_IS_LIST = tuple(True for _ in range(15))

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def normalize_value(self, value):
        if isinstance(value, list) and len(value) >= 2 and isinstance(value[1], int):
            value = value[0]
        if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
            return [""]
        elif not isinstance(value, list):
            return [value]
        else:
            return [item if item is not None else "" for item in value]

    def proxy_widget(self, prompt=None, unique_id=None, extra_pnginfo=None, **kwargs):
        try:
            input_values = {}
            for key, value in kwargs.items():
                if key.startswith("widget_input"):
                    input_values[key] = self.normalize_value(value)
            if not input_values and prompt and unique_id:
                node_info = prompt.get(str(unique_id), {})
                if node_info and 'inputs' in node_info:
                    for key, value in node_info['inputs'].items():
                        if key.startswith("widget_input"):
                            input_values[key] = self.normalize_value(value)
            output_values = []
            for i in range(15):
                port_key = f"widget_input_{i+1}"
                port_value = input_values.get(port_key, [""])
                output_values.append(port_value)
            return tuple(output_values)
        except Exception as e:
            return handle_error_safe(e, "view_Primitive failed", 15)
















