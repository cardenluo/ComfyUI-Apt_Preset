from __future__ import annotations
import os
import io
import json
import time
import wave
import hashlib
import zipfile
import numpy as np
import torch
from PIL import Image, ImageOps
from PIL.PngImagePlugin import PngInfo
from typing import Dict, Any, Optional, Tuple, List
import folder_paths
import node_helpers
from comfy.cli_args import args
from comfy_api.input_impl import VideoFromFile

#--------------------------------------------------------------------

from nodes import MAX_RESOLUTION, SaveImage, common_ksampler
import os
import sys
import random
from pathlib import Path
from PIL.PngImagePlugin import PngInfo
from comfy.cli_args import args
import inspect
import re
import traceback
import itertools
import comfy
from server import PromptServer
from aiohttp import web
from PIL import Image, ImageOps, ImageSequence
import node_helpers
import ast
import base64
import glob
import torch.nn.functional as F



from ..main_unit import *
from ..office_unit import ImageCompositeMasked



#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  

try:
    import soundfile as _sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    _sf = None
    SOUNDFILE_AVAILABLE = False





#优先从当前文件所在目录下的 comfy 子目录中查找模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))  

def updateTextWidget(node, widget, text):
    PromptServer.instance.send_sync("view_Data_text_processed", {"node": node, "widget": widget, "text": text})

routes = PromptServer.instance.routes


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
        if isinstance(mask, list):
            processed_masks = []
            for m in mask:
                if isinstance(m, torch.Tensor):
                    processed = self.process_single_mask(m)
                    processed_masks.append(processed)
            
            if processed_masks:
                preview = torch.cat(processed_masks, dim=0)
            else:
                return {"ui": {"images": []}}
        elif isinstance(mask, torch.Tensor):
            preview = self.process_single_mask(mask)
        else:
            return {"ui": {"images": []}}
        
        return self.save_images(preview, filename_prefix, prompt, extra_pnginfo)
    
    def process_single_mask(self, mask_tensor):
        if mask_tensor.dim() == 2:
            return mask_tensor.unsqueeze(0).unsqueeze(0).movedim(1, -1).expand(-1, -1, -1, 3)
        elif mask_tensor.dim() == 3:
            return mask_tensor.unsqueeze(1).movedim(1, -1).expand(-1, -1, -1, 3)
        else:
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

    RETURN_TYPES = (anyType,)  
    RETURN_NAMES = ("record",)  
    OUTPUT_NODE = True
    NAME = "view_Data"
    CATEGORY = "Apt_Preset/PreView"
    FUNCTION = "process"
    INPUT_IS_LIST = True

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
        
        if split_preset == "None":
            items = [text] if text else []
        elif separators:
            escaped_seps = [re.escape(sep) for sep in separators if sep]
            sep_pattern = '|'.join(escaped_seps)
            items = re.split(f'(?:{sep_pattern})', text)
        else:
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
                "remove_extension": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "控制输出的文件名是否包含扩展名"
                }),
            },
            "optional": {
                "max_images": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "Include_keyword": ("STRING", {"default": "", "multiline": False})
            }
        }
    RETURN_TYPES = ('IMAGE', "STRING", "STRING",)
    RETURN_NAMES = ("images", "file_names", "file_paths")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "get_transparent_image"
    CATEGORY = "Apt_Preset/IO_Port"
    
    def get_transparent_image(self, file_path, remove_extension=False, max_images=0, Include_keyword=""):
        try:
            image_list = []
            file_names = []
            file_paths = []
            
            file_path = file_path.strip('"')
            
            if os.path.isdir(file_path):
                image_files = [f for f in os.listdir(file_path) 
                            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
                
                if Include_keyword:
                    image_files = [f for f in image_files if Include_keyword in f]
                
                image_files.sort()
                
                if max_images > 0:
                    image_files = image_files[:max_images]
                
                for filename in image_files:
                    img_path = os.path.join(file_path, filename)
                    image = Image.open(img_path).convert('RGBA')
                    
                    # 直接处理为张量，不调整尺寸
                    image_np = np.array(image).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None, :, :, :]
                    image_list.append(image_tensor)
                    
                    # 处理文件名，根据remove_extension决定是否移除扩展名
                    if remove_extension:
                        file_names.append(os.path.splitext(filename)[0])
                    else:
                        file_names.append(filename)
                    
                    # 添加完整文件路径
                    file_paths.append(img_path)
                
                if not image_list:
                    return [], [], []
                
                return image_list, file_names, file_paths        
            else:
                image = Image.open(file_path)
                if image is not None:
                    image_rgba = image.convert('RGBA')
                    
                    if Include_keyword and Include_keyword not in os.path.basename(file_path):
                        print(f"文件 {file_path} 不包含关键字 '{Include_keyword}'，跳过加载")
                        return [], [], []
                    
                    image_np = np.array(image_rgba).astype(np.float32) / 255.0
                    image_tensor = torch.from_numpy(image_np)[None, :, :, :]
                    
                    # 处理文件名
                    filename = os.path.basename(file_path)
                    if remove_extension:
                        filename = os.path.splitext(filename)[0]
                    
                    return [image_tensor], [filename], [file_path]
            
        except Exception as e:
            print(f"出错请重置节点：{e}")
        return [], [], []




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
                "naming_format": (
                    [
                        "序号",
                        "命名+序号",
                        "序号+命名",
                        "命名",
                    ],
                    {"default": "命名+序号"}
                ),
                "number_digits": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "save_workflow_as_json": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("imagelist", "pathlist")
    FUNCTION = "save_image"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True, True)
    CATEGORY = "Apt_Preset/IO_Port"

    @staticmethod
    def find_highest_numeric_value(directory, filename_mid):
        highest_value = -1
        if not os.path.exists(directory):
            return highest_value
        
        for filename in os.listdir(directory):
            try:
                # 尝试匹配常见的编号模式
                # 1. 纯数字格式：00001.png
                # 2. 前缀_编号格式：8a_00001.png
                # 3. 编号_后缀格式：00001_8a.png
                
                # 首先尝试提取末尾的数字（通常是编号）
                name_part = filename.split('.')[0]
                parts = name_part.split('_')
                for part in reversed(parts):
                    if part.isdigit() and len(part) >= 3:  # 编号通常至少3位
                        numeric_value = int(part)
                        if numeric_value > highest_value:
                            highest_value = numeric_value
                        break
                
                # 如果没有找到，尝试其他模式
                if highest_value == -1:
                    # 提取所有数字部分
                    numeric_parts = re.findall(r'\d+', filename)
                    if numeric_parts:
                        # 过滤掉可能是日期的长数字（8位以上）
                        valid_parts = [p for p in numeric_parts if len(p) < 8]
                        if valid_parts:
                            # 取最长的数字串作为编号
                            longest_num = max(valid_parts, key=len)
                            numeric_value = int(longest_num)
                            if numeric_value > highest_value:
                                highest_value = numeric_value
            except (ValueError, AttributeError):
                continue
        
        return highest_value

    def save_image(self, image, file_format, filename_mid="Apt", output_path="", naming_format="命名_序号", number_digits=5,
                   save_workflow_as_json=False, prompt=None, extra_pnginfo=None):
        import datetime
        import time
        
        if isinstance(image, list):
            image = np.concatenate(image, axis=0)
        
        batch_size = image.shape[0]
        images_list = [image[i:i + 1, ...] for i in range(batch_size)]
        output_dir = folder_paths.get_output_directory()
        output_paths = []

        if isinstance(output_path, list):
            if len(output_path) == batch_size:
                for path in output_path:
                    os.makedirs(path, exist_ok=True)
                output_paths = output_path
            else:
                print(f"output_path列表长度({len(output_path)})与图片数量({batch_size})不匹配，使用默认路径")
                output_paths = [output_dir] * batch_size
        else:
            os.makedirs(output_path, exist_ok=True)
            output_paths = [output_path] * batch_size

        base_dir = output_paths[0]
        # 查找最高编号，确保准确
        import re
        existing_counters = []
        
        # 生成可能的命名格式模式
        patterns = []
        patterns.append(r"^(\d+)$")  # 序号
        patterns.append(r"^" + re.escape(filename_mid) + r"(\d+)$")  # 命名+序号
        patterns.append(r"^(\d+)" + re.escape(filename_mid) + r"$")  # 序号+命名
        patterns.append(r"^" + re.escape(filename_mid) + r"(\d+)$")  # 命名
        
        # 扫描目录中的文件
        if os.path.exists(base_dir):
            for filename in os.listdir(base_dir):
                for pattern in patterns:
                    match = re.match(pattern, os.path.splitext(filename)[0])
                    if match:
                        try:
                            counter = int(match.group(1))
                            existing_counters.append(counter)
                        except:
                            pass
        
        # 确定起始编号
        if existing_counters:
            counter = max(existing_counters) + 1
        else:
            counter = 1
        
        absolute_paths = []

        for idx, img_tensor in enumerate(images_list):
            output_image = img_tensor.cpu().numpy()
            img_np = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_np[0])
            out_path = output_paths[idx]

            # 生成序号
            numbering = f"{counter:0{number_digits}d}"
            
            # 生成新文件名
            if naming_format == "序号":
                output_filename = f"{numbering}"
            elif naming_format == "命名+序号":
                output_filename = f"{filename_mid}{numbering}"
            elif naming_format == "序号+命名":
                output_filename = f"{numbering}{filename_mid}"
            elif naming_format == "命名":
                output_filename = f"{filename_mid}{numbering}"
            else:
                output_filename = f"{filename_mid}{numbering}"
            
            # 递增计数器
            counter += 1

            # 构建完整路径
            resolved_image_path = os.path.join(out_path, f"{output_filename}.{file_format}")
            
            # 确保文件名唯一，避免覆盖
            unique_counter = 1
            original_filename = output_filename
            while os.path.exists(resolved_image_path):
                # 如果文件已存在，添加后缀
                output_filename = f"{original_filename}_{unique_counter}"
                resolved_image_path = os.path.join(out_path, f"{output_filename}.{file_format}")
                unique_counter += 1
                # 避免无限循环
                if unique_counter > 100:
                    break

            # 保存图片
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

            # 稍微延迟，避免并发问题
            time.sleep(0.01)

        return (images_list, absolute_paths)



#region-------------view_bridge_image------------------

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

#endregion-------------view_bridge_image------------------







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





#region-------------IO_store_image-------------
try:
    from comfy_execution.graph import ExecutionBlocker
except ImportError:
    class ExecutionBlocker:
        def __init__(self, value):
            self.value = value

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
    RETURN_NAMES = ("image", "release_total")
    FUNCTION = "store_image"
    CATEGORY = "Apt_Preset/IO_Port"
    OUTPUT_NODE = True
    DESCRIPTION = """
    输出逻辑：
    - 全部尺寸一致时：
      - release_total ≤ 0 → 输出全部存储的图像
      - release_total > 0 → 
        1. 存储总数 = release_total → 输出全部
        2. 存储总数 < release_total → 关闭输出
        3. 存储总数 > release_total → 从后面数够数量的图像输出
    - 若不是全部一致时：仅输出最后一张图像
"""

    def __init__(self):
        global GLOBAL_STORED_IMAGES, GLOBAL_DISPLAY_DATA
        GLOBAL_STORED_IMAGES = []
        GLOBAL_DISPLAY_DATA = []
        print("IO_store_image node initialized (storage reset)")

    # ========== 核心改造：原IMAGE_BooleanSwitch合并为内部方法 ==========
    def _image_boolean_switch(self, switch: bool, image: Optional[torch.Tensor] = None):
        """原IMAGE_BooleanSwitch的核心逻辑，合并为内部私有方法"""
        if switch is True:
            return (image,)
        else:
            if ExecutionBlocker is not None:
                return (ExecutionBlocker(None),)
            else:
                return ({},)

    def store_image(self, image: Optional[torch.Tensor] = None, 
                   prompt: Any = None, image_output: str = None, 
                   extra_pnginfo: Any = None, release_total: float = 0) -> Dict[str, Any]:
        global GLOBAL_STORED_IMAGES, GLOBAL_DISPLAY_DATA
        
        # 图像存储逻辑：去重+追加，原逻辑保留不变
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
        switch_boolean = False  # 核心布尔开关，默认关闭输出

        # 通道数不一致的判断逻辑
        if total_stored > 0:
            channels = [img.shape[1] for img in GLOBAL_STORED_IMAGES]
            if len(set(channels)) > 1:
                print("Warning: Images have different channel counts, outputting last image")
                output_image = GLOBAL_STORED_IMAGES[-1]
                current_total = total_stored
                return self._prepare_return_data(output_image, current_total, image_output, prompt, extra_pnginfo)
        
        # 判断所有存储图像尺寸是否一致
        if total_stored > 0:
            sizes = [(img.shape[2], img.shape[3]) for img in GLOBAL_STORED_IMAGES]
            sizes_consistent = len(set(sizes)) == 1
        else:
            sizes_consistent = False
        
        # ========== 核心改造：全新业务逻辑 + 标准IF布尔判断 ==========
        if total_stored == 0:
            output_image = image
            current_total = 0
        else:
            # 尺寸不一致 → 固定输出最后一张图像
            if not sizes_consistent:
                output_image = GLOBAL_STORED_IMAGES[-1]
                current_total = total_stored
            # 尺寸一致 → 执行新的核心分支逻辑
            else:
                release_total = int(release_total)
                # 分支1: release_total ≤ 0 → 输出全部存储的图像，开关为真
                if release_total <= 0:
                    switch_boolean = True
                    output_image = torch.cat(GLOBAL_STORED_IMAGES, dim=0)
                    current_total = total_stored
                # 分支2: release_total > 0 → 三层IF布尔判断
                else:
                    # 子分支1: 存储总数 = release_total → 布尔真，打开输出，输出全部
                    if total_stored == release_total:
                        switch_boolean = True
                        output_image = torch.cat(GLOBAL_STORED_IMAGES, dim=0)
                        current_total = total_stored
                    # 子分支2: 存储总数 < release_total → 布尔假，关闭输出
                    elif total_stored < release_total:
                        switch_boolean = False
                        output_image = None
                        current_total = total_stored
                    # 子分支3: 存储总数 > release_total → 布尔真，打开输出，输出倒数N张
                    elif total_stored > release_total:
                        switch_boolean = True
                        start_idx = total_stored - release_total
                        selected_imgs = GLOBAL_STORED_IMAGES[start_idx:]
                        output_image = torch.cat(selected_imgs, dim=0)
                        current_total = release_total

        # 兜底空值处理
        if output_image is None:
            output_image = image
            current_total = 0

        # ========== 核心调用：使用合并后的内部开关逻辑，控制最终输出 ==========
        final_output_image = self._image_boolean_switch(switch_boolean, output_image)[0]

        return self._prepare_return_data(final_output_image, current_total, image_output, prompt, extra_pnginfo)

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





#region------------XXXXIO_store_image补齐-------------



import torch
import numpy as np
import io
import base64
import json
from PIL import Image
from typing import Optional, Dict, Any, List

GLOBAL_STORED_IMAGES: List[torch.Tensor] = []
GLOBAL_DISPLAY_DATA: List[Dict[str, Any]] = []

class XXXXIO_store_image:
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
    RETURN_NAMES = ("image", "release_total")
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





#region-------------IO_EasyMark------------------

import torch
import numpy as np
import nodes
from PIL import Image
from PIL import ImageDraw, ImageFont
import io
import base64

class IO_EasyMark:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "brush_data": ("STRING", {"default": "", "multiline": True}),
                "brush_size": ("INT", {"default": 4, "min": 1, "max": 100, "step": 1}),
                "image_base64": ("STRING", {"default": "", "multiline": True}),
            },
        }

    NAME="IO_EasyMark"
    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("原图", "合成图", "总mask", "黑mask", "白mask", "红mask", "绿mask", "蓝mask", "灰mask")
    FUNCTION = "main"
    CATEGORY = "Apt_Preset/IO_Port"

    def main(self, brush_data, brush_size, image_base64):
        background_img_tensor = None
        
        if image_base64 and image_base64.strip():
            try:
                base64_data = image_base64.strip()
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[-1]
                
                img_bytes = base64.b64decode(base64_data)
                img_pil = Image.open(io.BytesIO(img_bytes))
                if img_pil.mode != 'RGB':
                    img_pil = img_pil.convert('RGB')
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                background_img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            except Exception as e:
                print(f"Error loading image from base64: {e}")
                import traceback
                traceback.print_exc()
        
        if background_img_tensor is None:
            background_img_tensor = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
        
        batch_size = background_img_tensor.shape[0]
        height = background_img_tensor.shape[1]
        width = background_img_tensor.shape[2]
        
        black_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        white_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        red_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        green_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        blue_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        gray_mask = torch.zeros((batch_size, height, width), dtype=torch.float32)
        marker_annotations = []
        
        if brush_data and brush_data.strip():
            try:
                strokes = brush_data.split('|')
                
                black_mask_np = black_mask[0].numpy().copy()
                white_mask_np = white_mask[0].numpy().copy()
                red_mask_np = red_mask[0].numpy().copy()
                green_mask_np = green_mask[0].numpy().copy()
                blue_mask_np = blue_mask[0].numpy().copy()
                gray_mask_np = gray_mask[0].numpy().copy()
                
                color_mapping = {
                    "0,0,0": "black",
                    "255,255,255": "white",
                    "255,0,0": "red",
                    "0,255,0": "green",
                    "0,0,255": "blue",
                    "128,128,128": "gray"
                }
                
                for stroke_idx, stroke in enumerate(strokes):
                    if not stroke.strip():
                        continue
                    
                    mode = 'brush'
                    stroke_type = 'free'
                    stroke_size = brush_size
                    stroke_opacity = 1.0
                    stroke_color = "255,255,255"
                    points_str = stroke
                    marker_id = None
                    
                    if ':' in stroke:
                        parts = stroke.split(':')
                        if len(parts) >= 6:
                            mode = parts[0] if parts[0] in ('brush', 'erase') else 'brush'
                            stroke_type = parts[1] if parts[1] in ('free', 'box', 'square') else 'free'
                            stroke_size = int(float(parts[2]))
                            stroke_opacity = float(parts[3])
                            stroke_color = parts[4]
                            if len(parts) >= 7 and parts[5] in ('1', '2', '3', '4', '5', '6') and ',' not in parts[5]:
                                marker_id = parts[5]
                                points_str = ':'.join(parts[6:])
                            else:
                                points_str = ':'.join(parts[5:])
                        elif len(parts) >= 2:
                            if parts[0] in ('brush', 'erase'):
                                mode = parts[0]
                                points_str = ':'.join(parts[1:])
                            elif parts[0] in ('free', 'box', 'square'):
                                stroke_type = parts[0]
                                points_str = ':'.join(parts[1:])
                    
                    radius = max(1, stroke_size // 2)
                    
                    point_list = points_str.split(';')
                    path_points = []
                    
                    for point_str in point_list:
                        if not point_str.strip():
                            continue
                        try:
                            coords = point_str.split(',', 1)
                            if len(coords) == 2:
                                x = int(float(coords[0]))
                                y = int(float(coords[1]))
                                path_points.append((x, y))
                        except (ValueError, IndexError):
                            continue
                    
                    if len(path_points) == 0:
                        continue
                    
                    stroke_color_key = stroke_color
                    color_type = color_mapping.get(stroke_color_key, "default")
                    
                    x_coords = [p[0] for p in path_points]
                    y_coords = [p[1] for p in path_points]
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)

                    if marker_id is not None and stroke_type == 'square' and mode != 'erase':
                        valid_min_x = max(0, min_x)
                        valid_max_x = min(width - 1, max_x)
                        valid_min_y = max(0, min_y)
                        valid_max_y = min(height - 1, max_y)
                        if valid_max_x > valid_min_x and valid_max_y > valid_min_y:
                            marker_annotations.append({
                                "id": marker_id,
                                "min_x": int(valid_min_x),
                                "max_x": int(valid_max_x),
                                "min_y": int(valid_min_y),
                                "max_y": int(valid_max_y),
                            })
                        continue
                    
                    if stroke_type == 'square':
                        valid_min_x = max(0, min_x)
                        valid_max_x = min(width, max_x + 1)
                        valid_min_y = max(0, min_y)
                        valid_max_y = min(height, max_y + 1)
                        
                        if valid_max_x <= valid_min_x or valid_max_y <= valid_min_y:
                            continue
                        
                        if mode == 'erase':
                            black_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
                            white_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
                            red_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
                            green_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
                            blue_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
                            gray_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = 0.0
                        else:
                            if color_type == "black":
                                black_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity
                            elif color_type == "white":
                                white_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity
                            elif color_type == "red":
                                red_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity
                            elif color_type == "green":
                                green_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity
                            elif color_type == "blue":
                                blue_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity
                            elif color_type == "gray":
                                gray_mask_np[valid_min_y:valid_max_y, valid_min_x:valid_max_x] = stroke_opacity
                    elif stroke_type == 'box':
                        x0 = max(0, min_x)
                        x1 = min(width, max_x + 1)
                        y0 = max(0, min_y)
                        y1 = min(height, max_y + 1)

                        if x1 <= x0 or y1 <= y0:
                            continue

                        thickness = max(1, int(stroke_size))
                        top_y1 = min(y1, y0 + thickness)
                        bottom_y0 = max(y0, y1 - thickness)
                        left_x1 = min(x1, x0 + thickness)
                        right_x0 = max(x0, x1 - thickness)

                        edges = [
                            (slice(y0, top_y1), slice(x0, x1)),
                            (slice(bottom_y0, y1), slice(x0, x1)),
                            (slice(y0, y1), slice(x0, left_x1)),
                            (slice(y0, y1), slice(right_x0, x1)),
                        ]

                        if mode == 'erase':
                            for ys, xs in edges:
                                black_mask_np[ys, xs] = 0.0
                                white_mask_np[ys, xs] = 0.0
                                red_mask_np[ys, xs] = 0.0
                                green_mask_np[ys, xs] = 0.0
                                blue_mask_np[ys, xs] = 0.0
                                gray_mask_np[ys, xs] = 0.0
                        else:
                            target_mask = None
                            if color_type == "black":
                                target_mask = black_mask_np
                            elif color_type == "white":
                                target_mask = white_mask_np
                            elif color_type == "red":
                                target_mask = red_mask_np
                            elif color_type == "green":
                                target_mask = green_mask_np
                            elif color_type == "blue":
                                target_mask = blue_mask_np
                            elif color_type == "gray":
                                target_mask = gray_mask_np
                            if target_mask is not None:
                                for ys, xs in edges:
                                    target_mask[ys, xs] = stroke_opacity
                    else:
                        for i, (x, y) in enumerate(path_points):
                                if i > 0:
                                    prev_x, prev_y = path_points[i-1]
                                    if mode == 'erase':
                                        self._erase_line(black_mask_np, prev_x, prev_y, x, y, radius)
                                        self._erase_line(white_mask_np, prev_x, prev_y, x, y, radius)
                                        self._erase_line(red_mask_np, prev_x, prev_y, x, y, radius)
                                        self._erase_line(green_mask_np, prev_x, prev_y, x, y, radius)
                                        self._erase_line(blue_mask_np, prev_x, prev_y, x, y, radius)
                                        self._erase_line(gray_mask_np, prev_x, prev_y, x, y, radius)
                                    else:
                                        if color_type == "black":
                                            self._draw_line(black_mask_np, prev_x, prev_y, x, y, radius)
                                        elif color_type == "white":
                                            self._draw_line(white_mask_np, prev_x, prev_y, x, y, radius)
                                        elif color_type == "red":
                                            self._draw_line(red_mask_np, prev_x, prev_y, x, y, radius)
                                        elif color_type == "green":
                                            self._draw_line(green_mask_np, prev_x, prev_y, x, y, radius)
                                        elif color_type == "blue":
                                            self._draw_line(blue_mask_np, prev_x, prev_y, x, y, radius)
                                        elif color_type == "gray":
                                            self._draw_line(gray_mask_np, prev_x, prev_y, x, y, radius)
                                else:
                                    if mode == 'erase':
                                        self._erase_circle(black_mask_np, x, y, radius)
                                        self._erase_circle(white_mask_np, x, y, radius)
                                        self._erase_circle(red_mask_np, x, y, radius)
                                        self._erase_circle(green_mask_np, x, y, radius)
                                        self._erase_circle(blue_mask_np, x, y, radius)
                                        self._erase_circle(gray_mask_np, x, y, radius)
                                    else:
                                        if color_type == "black":
                                            self._draw_circle(black_mask_np, x, y, radius)
                                        elif color_type == "white":
                                            self._draw_circle(white_mask_np, x, y, radius)
                                        elif color_type == "red":
                                            self._draw_circle(red_mask_np, x, y, radius)
                                        elif color_type == "green":
                                            self._draw_circle(green_mask_np, x, y, radius)
                                        elif color_type == "blue":
                                            self._draw_circle(blue_mask_np, x, y, radius)
                                        elif color_type == "gray":
                                            self._draw_circle(gray_mask_np, x, y, radius)
                
                black_mask[0] = torch.from_numpy(black_mask_np)
                white_mask[0] = torch.from_numpy(white_mask_np)
                red_mask[0] = torch.from_numpy(red_mask_np)
                green_mask[0] = torch.from_numpy(green_mask_np)
                blue_mask[0] = torch.from_numpy(blue_mask_np)
                gray_mask[0] = torch.from_numpy(gray_mask_np)
                        
            except Exception as e:
                print(f"Error parsing brush data: {e}")
                import traceback
                traceback.print_exc()
        
        black_mask = torch.clamp(black_mask, 0.0, 1.0)
        white_mask = torch.clamp(white_mask, 0.0, 1.0)
        red_mask = torch.clamp(red_mask, 0.0, 1.0)
        green_mask = torch.clamp(green_mask, 0.0, 1.0)
        blue_mask = torch.clamp(blue_mask, 0.0, 1.0)
        gray_mask = torch.clamp(gray_mask, 0.0, 1.0)
        
        sum_mask = torch.maximum(black_mask, white_mask)
        sum_mask = torch.maximum(sum_mask, red_mask)
        sum_mask = torch.maximum(sum_mask, green_mask)
        sum_mask = torch.maximum(sum_mask, blue_mask)
        sum_mask = torch.maximum(sum_mask, gray_mask)
        
        sum_image = background_img_tensor.clone()
        
        black_mask_4d = black_mask.unsqueeze(-1)
        white_mask_4d = white_mask.unsqueeze(-1)
        red_mask_4d = red_mask.unsqueeze(-1)
        green_mask_4d = green_mask.unsqueeze(-1)
        blue_mask_4d = blue_mask.unsqueeze(-1)
        gray_mask_4d = gray_mask.unsqueeze(-1)
        
        sum_image = sum_image * (1 - black_mask_4d) + torch.tensor([0.0, 0.0, 0.0]).to(sum_image.device) * black_mask_4d
        sum_image = sum_image * (1 - white_mask_4d) + torch.tensor([1.0, 1.0, 1.0]).to(sum_image.device) * white_mask_4d
        sum_image = sum_image * (1 - red_mask_4d) + torch.tensor([1.0, 0.0, 0.0]).to(sum_image.device) * red_mask_4d
        sum_image = sum_image * (1 - green_mask_4d) + torch.tensor([0.0, 1.0, 0.0]).to(sum_image.device) * green_mask_4d
        sum_image = sum_image * (1 - blue_mask_4d) + torch.tensor([0.0, 0.0, 1.0]).to(sum_image.device) * blue_mask_4d
        sum_image = sum_image * (1 - gray_mask_4d) + torch.tensor([0.5, 0.5, 0.5]).to(sum_image.device) * gray_mask_4d

        if marker_annotations:
            font_cache = {}
            pil_font_path = os.path.join(os.path.dirname(ImageFont.__file__), "fonts", "DejaVuSans.ttf")
            def get_font(font_size: int):
                font_size = int(font_size)
                cached = font_cache.get(font_size)
                if cached is not None:
                    return cached
                font = None
                for candidate in ("arial.ttf", "DejaVuSans.ttf", pil_font_path):
                    try:
                        font = ImageFont.truetype(candidate, size=font_size)
                        break
                    except Exception:
                        font = None
                if font is None:
                    font = ImageFont.load_default()
                font_cache[font_size] = font
                return font
            for b in range(batch_size):
                img_np = (sum_image[b].detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np, mode='RGB')
                draw = ImageDraw.Draw(img_pil)
                for m in marker_annotations:
                    x0 = int(m["min_x"])
                    y0 = int(m["min_y"])
                    x1 = int(m["max_x"])
                    y1 = int(m["max_y"])
                    side = max(1, min(abs(x1 - x0), abs(y1 - y0)))
                    font = get_font(max(12, int(side * 0.6)))
                    draw.rectangle([x0, y0, x1, y1], fill=(255, 255, 0), outline=(0, 0, 0), width=2)
                    text = str(m["id"])
                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except Exception:
                        tw, th = font.getsize(text)
                    tx = x0 + max(0, (x1 - x0 - tw) // 2)
                    ty = y0 + max(0, (y1 - y0 - th) // 2)
                    draw.text((tx, ty), text, fill=(0, 0, 0), font=font)

                img_out = np.array(img_pil).astype(np.float32) / 255.0
                sum_image[b] = torch.from_numpy(img_out).to(sum_image.device)
        
        return (background_img_tensor, sum_image, sum_mask, black_mask, white_mask, red_mask, green_mask, blue_mask, gray_mask)
    
    def _draw_circle(self, mask, x, y, radius):
        h, w = mask.shape
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        
        if x_max <= x_min or y_max <= y_min:
            return
        
        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        
        dist_sq = (x_coords - x)**2 + (y_coords - y)**2
        radius_sq = radius * radius
        
        mask[y_min:y_max, x_min:x_max] = np.maximum(
            mask[y_min:y_max, x_min:x_max],
            (dist_sq <= radius_sq).astype(np.float32)
        )
    
    def _draw_line(self, mask, x1, y1, x2, y2, radius):
        if x1 == x2 and y1 == y2:
            self._draw_circle(mask, x1, y1, radius)
            return
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if radius > 10:
            step_size = max(1, radius // 3)
        else:
            step_size = 1
        
        steps = max(1, int(length / step_size) + 1)
        
        if steps <= 0:
            self._draw_circle(mask, x1, y1, radius)
            return
        
        t_values = np.linspace(0, 1, steps + 1)
        x_coords = (x1 + dx * t_values).astype(np.int32)
        y_coords = (y1 + dy * t_values).astype(np.int32)
        
        h, w = mask.shape
        valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        if len(x_coords) > 0:
            coords = np.column_stack((y_coords, x_coords))
            unique_coords = np.unique(coords, axis=0)
            
            for y, x in unique_coords:
                self._draw_circle(mask, int(x), int(y), radius)
    
    def _erase_circle(self, mask, x, y, radius):
        h, w = mask.shape
        y_min = max(0, y - radius)
        y_max = min(h, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(w, x + radius + 1)
        
        if x_max <= x_min or y_max <= y_min:
            return
        
        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        
        dist_sq = (x_coords - x)**2 + (y_coords - y)**2
        radius_sq = radius * radius
        
        erase_mask = dist_sq <= radius_sq
        mask[y_min:y_max, x_min:x_max] = np.where(
            erase_mask,
            0.0,
            mask[y_min:y_max, x_min:x_max]
        )
    
    def _erase_line(self, mask, x1, y1, x2, y2, radius):
        if x1 == x2 and y1 == y2:
            self._erase_circle(mask, x1, y1, radius)
            return
        
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx*dx + dy*dy)
        
        if radius > 10:
            step_size = max(1, radius // 3)
        else:
            step_size = 1
        
        steps = max(1, int(length / step_size) + 1)
        
        if steps <= 0:
            self._erase_circle(mask, x1, y1, radius)
            return
        
        t_values = np.linspace(0, 1, steps + 1)
        x_coords = (x1 + dx * t_values).astype(np.int32)
        y_coords = (y1 + dy * t_values).astype(np.int32)
        
        h, w = mask.shape
        valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        
        if len(x_coords) > 0:
            coords = np.column_stack((y_coords, x_coords))
            unique_coords = np.unique(coords, axis=0)
            
            for y, x in unique_coords:
                self._erase_circle(mask, int(x), int(y), radius)

#endregion-------------IO_EasyMark------------------






#region----------------IO_load_image_list



import os
import hashlib
import json
import shutil

import numpy as np
import torch
from PIL import Image, ImageOps, ImageSequence

import folder_paths
import node_helpers


@routes.get("/Apt_Preset_IO_LoadImgList_thumb")
async def apt_preset_io_loadimglist_thumb(request):
    filename = request.query.get("filename", "")
    size_raw = request.query.get("size", "64")
    try:
        size = int(size_raw)
    except Exception:
        size = 64
    if size < 32:
        size = 32
    if size > 2048:
        size = 2048
    render_size = 256 if size < 256 else size

    if not filename or not folder_paths.exists_annotated_filepath(filename):
        return web.Response(status=404)

    image_path = folder_paths.get_annotated_filepath(filename)
    try:
        img0 = node_helpers.pillow(Image.open, image_path)
        frame0 = next(ImageSequence.Iterator(img0))
        frame0 = node_helpers.pillow(ImageOps.exif_transpose, frame0)
        frame0 = frame0.convert("RGB")
    except Exception:
        return web.Response(status=500)

    resample = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.BICUBIC)
    contained = ImageOps.contain(frame0, (render_size, render_size), method=resample)
    canvas = Image.new("RGB", (render_size, render_size), (0, 0, 0))
    ox = (render_size - contained.size[0]) // 2
    oy = (render_size - contained.size[1]) // 2
    canvas.paste(contained, (ox, oy))

    buf = io.BytesIO()
    if render_size <= 256:
        canvas.save(buf, format="PNG", optimize=True)
        return web.Response(body=buf.getvalue(), content_type="image/png")
    canvas.save(buf, format="JPEG", quality=92, optimize=True)
    return web.Response(body=buf.getvalue(), content_type="image/jpeg")


class IO_LoadImgList:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_list": ("STRING", {"multiline": True, "default": "",},),

            },
            "optional": {
                "import_image": ("IMAGE", {"forceInput": True}),
                "selected_indices": ("STRING", {"default": "[]", "multiline": False}),
                "thumb_size": ("INT", {"default": 64, "min": 64, "max": 384, "step": 1}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }
    NAME="IO_LoadImgList"
    CATEGORY = "Apt_Preset/IO_Port"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image_list", "path_list")
    FUNCTION = "load_images"
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_NODE = True

    def load_images(self, image_list: str, selected_indices: str, thumb_size: int = 64, import_image=None, import_path: str = "", unique_id: str = ""):
        names = [x.strip() for x in (image_list or "").splitlines()]
        names = [x for x in names if x]

        imported_names = []
        if import_image is not None:
            imported_from_image = self._import_from_image(import_image, existing_names=names)
            if imported_from_image:
                names = names + imported_from_image
                imported_names.extend(imported_from_image)
        if import_path:
            imported_from_path = self._import_from_path(import_path, existing_names=names)
            if imported_from_path:
                names = names + imported_from_path
                imported_names.extend(imported_from_path)
        if unique_id and imported_names:
            try:
                ps = getattr(PromptServer, "instance", None)
                if ps is not None and hasattr(ps, "send_sync"):
                    ps.send_sync("IO_LoadImgList_append", {"node": unique_id, "names": imported_names})
            except Exception:
                pass

        try:
            selected = json.loads(selected_indices)
            if isinstance(selected, list):
                valid_indices = [i for i in selected if isinstance(i, int) and i >= 0 and i < len(names)]
                if valid_indices:
                    names = [names[i] for i in valid_indices]
                elif len(names) > 0:
                    names = [names[-1]]
                else:
                    names = []
            elif len(names) > 0:
                names = [names[-1]]
            else:
                names = []
        except Exception as e:
            if len(names) > 0:
                names = [names[-1]]
            else:
                names = []

        if len(names) == 0:
            return ([], [])

        output_images = []
        output_names = []
        output_paths = []

        excluded_formats = ["MPO"]

        for name in names:
            if not folder_paths.exists_annotated_filepath(name):
                continue

            image_path = folder_paths.get_annotated_filepath(name)
            img = node_helpers.pillow(Image.open, image_path)

            w, h = None, None
            frames = []

            for i in ImageSequence.Iterator(img):
                i = node_helpers.pillow(ImageOps.exif_transpose, i)

                if i.mode == "I":
                    i = i.point(lambda p: p * (1 / 255))
                pil_image = i.convert("RGB")

                if len(frames) == 0:
                    w = pil_image.size[0]
                    h = pil_image.size[1]

                if pil_image.size[0] != w or pil_image.size[1] != h:
                    continue

                arr = np.array(pil_image).astype(np.float32) / 255.0
                tensor = torch.from_numpy(arr)[None,]
                frames.append(tensor)

            if len(frames) == 0:
                continue

            if len(frames) > 1 and img.format not in excluded_formats:
                image_tensor = torch.cat(frames, dim=0)
            else:
                image_tensor = frames[0]

            output_images.append(image_tensor)
            output_names.append(name)
            output_paths.append(image_path)

        if len(output_images) == 0:
            return ([], [])
        return (output_images, output_paths)

    def _import_from_image(self, import_image, existing_names: list):
        if import_image is None:
            return []

        try:
            if isinstance(import_image, list):
                tensors = [t for t in import_image if isinstance(t, torch.Tensor)]
                if len(tensors) == 0:
                    return []
                import_image = torch.cat(tensors, dim=0)
        except Exception:
            pass

        if not isinstance(import_image, torch.Tensor):
            return []

        try:
            img = import_image.detach()
        except Exception:
            img = import_image

        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.dim() != 4:
            return []

        allowed_ext = ".png"
        input_dir = folder_paths.get_input_directory()
        existing_set = {str(x).lower() for x in (existing_names or [])}

        imported = []
        for b in range(img.shape[0]):
            try:
                frame = img[b]
                frame_np = np.clip(255.0 * frame.cpu().numpy(), 0, 255).astype(np.uint8)
                h = hashlib.sha256(frame_np.tobytes()).hexdigest()[:12]
                base = f"io_loadimg_{h}{allowed_ext}"
                if base.lower() in existing_set:
                    continue
                dst_path = os.path.join(input_dir, base)
                if not os.path.exists(dst_path):
                    Image.fromarray(frame_np).save(dst_path, format="PNG", optimize=True)
                imported.append(base)
                existing_set.add(base.lower())
            except Exception:
                continue
        return imported

    def _import_from_path(self, import_path: str, existing_names: list):
        p = (import_path or "").strip().strip('"').strip("'")
        if not p:
            return []
        p = os.path.expandvars(os.path.expanduser(p))
        p = os.path.abspath(p)

        allowed_ext = {".png", ".jpg", ".jpeg", ".webp"}
        input_dir = folder_paths.get_input_directory()
        existing_set = {str(x).lower() for x in (existing_names or [])}
        imported_set = set()

        def import_file(src_path: str):
            src_path = os.path.abspath(src_path)
            if not os.path.isfile(src_path):
                return None
            ext = os.path.splitext(src_path)[1].lower()
            if ext not in allowed_ext:
                return None

            base = os.path.basename(src_path)
            if base.lower() in existing_set or base.lower() in imported_set:
                return None

            dst_name = base
            dst_path = os.path.join(input_dir, dst_name)

            try:
                if os.path.exists(dst_path):
                    if os.path.getsize(dst_path) == os.path.getsize(src_path):
                        imported_set.add(dst_name.lower())
                        return dst_name
            except Exception:
                pass

            if os.path.exists(dst_path):
                stem = os.path.splitext(base)[0]
                i = 1
                while True:
                    cand = f"{stem}_{i}{ext}"
                    cand_path = os.path.join(input_dir, cand)
                    if cand.lower() not in existing_set and cand.lower() not in imported_set and not os.path.exists(cand_path):
                        dst_name = cand
                        dst_path = cand_path
                        break
                    i += 1

            try:
                shutil.copy2(src_path, dst_path)
            except Exception:
                return None

            imported_set.add(dst_name.lower())
            return dst_name

        imported = []
        if os.path.isdir(p):
            try:
                entries = [os.path.join(p, fn) for fn in os.listdir(p)]
            except Exception:
                entries = []
            entries = [x for x in entries if os.path.isfile(x) and os.path.splitext(x)[1].lower() in allowed_ext]
            entries.sort(key=lambda x: os.path.basename(x).lower())
            for src in entries:
                name = import_file(src)
                if name:
                    imported.append(name)
        elif os.path.isfile(p):
            name = import_file(p)
            if name:
                imported.append(name)
        return imported

    @classmethod
    def IS_CHANGED(s, image_list: str, selected_indices: str, thumb_size: int = 64, import_image=None, import_path: str = ""):
        m = hashlib.sha256()
        names = [x.strip() for x in (image_list or "").splitlines()]
        names = [x for x in names if x]

        try:
            selected = json.loads(selected_indices)
            if isinstance(selected, list):
                valid_indices = [i for i in selected if isinstance(i, int) and i >= 0 and i < len(names)]
                if valid_indices:
                    names = [names[i] for i in valid_indices]
        except Exception:
            pass

        m.update(selected_indices.encode("utf-8"))
        if isinstance(import_image, torch.Tensor):
            try:
                t = import_image.detach().cpu()
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                if t.dim() == 4 and t.numel() > 0:
                    sample = t.reshape(-1)[:4096].numpy().tobytes()
                    m.update(sample)
            except Exception:
                pass
        if import_path:
            p = (import_path or "").strip().strip('"').strip("'")
            p = os.path.expandvars(os.path.expanduser(p))
            p = os.path.abspath(p)
            allowed_ext = {".png", ".jpg", ".jpeg", ".webp"}
            m.update(p.encode("utf-8"))
            try:
                if os.path.isdir(p):
                    entries = [os.path.join(p, fn) for fn in os.listdir(p)]
                    entries = [x for x in entries if os.path.isfile(x) and os.path.splitext(x)[1].lower() in allowed_ext]
                    entries.sort(key=lambda x: os.path.basename(x).lower())
                    for fp in entries:
                        st = os.stat(fp)
                        m.update(os.path.basename(fp).encode("utf-8"))
                        m.update(str(st.st_mtime_ns).encode("utf-8"))
                        m.update(str(st.st_size).encode("utf-8"))
                elif os.path.isfile(p) and os.path.splitext(p)[1].lower() in allowed_ext:
                    st = os.stat(p)
                    m.update(os.path.basename(p).encode("utf-8"))
                    m.update(str(st.st_mtime_ns).encode("utf-8"))
                    m.update(str(st.st_size).encode("utf-8"))
            except Exception:
                pass
        for name in names:
            m.update(name.encode("utf-8"))
            if folder_paths.exists_annotated_filepath(name):
                image_path = folder_paths.get_annotated_filepath(name)
                if os.path.isfile(image_path):
                    with open(image_path, "rb") as f:
                        m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image_list: str, selected_indices: str, thumb_size: int = 64, import_image=None, import_path: str = ""):
        names = [x.strip() for x in (image_list or "").splitlines()]
        names = [x for x in names if x]

        if len(names) == 0:
            return True

        valid = False
        for name in names:
            if folder_paths.exists_annotated_filepath(name):
                valid = True
                break

        if not valid:
            return "No valid images in image_list"

        return True




#endregion----------------load_image_list---------------------------






class IO_PathProcessor:
    CATEGORY = "Apt_Preset/IO_Port"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("NewPathList", "FileNameList", "FolderList")
    FUNCTION = "process_paths"
    INPUT_IS_LIST = (True,)
    OUTPUT_IS_LIST = (True, True, True)

    DESCRIPTION = r"""
    【正则排序规则（三种常用写法）】
    1. 开头匹配：^(\d+) → （如12AI图片.png中的12）
    2. 结尾匹配：(\d{2})(?=\.\w+$) → （如AI图片63.png中的63）
    3. 括号匹配：\((\d+)\) → （如图片(45).png中的45）
    """


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path_list": ("STRING", {"forceInput": True}),
                "remove_file_suffix": ("BOOLEAN", {"default": True}),
                "filter_suffixes": ("STRING", {"default": "", "placeholder": "e.g.: png|jpg|jpeg, leave empty for all"}),
                "regex_filter": ("STRING", {"default": "", "placeholder": "Regex for filtering paths, e.g.: ^(?!.*00291).*$ (exclude 00291)"}),
                "regex_sort_pattern": ("STRING", {"default": "", "placeholder": "匹配括号数字留空，其他填正则"}),
                "sort_order": (["up", "down"], {"default": "up"}),
                "path_duplication_count": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
            }
        }

    def _extract_regex_sort_key(self, path: str, pattern: str) -> float:
        filename = os.path.basename(path)
        if not pattern or pattern.strip() == "":
            match = re.search(r'\((\d+)\)', filename)
        else:
            try:
                match = re.search(pattern, filename)
            except re.error:
                match = re.search(r'\((\d+)\)', filename)

        if match and len(match.groups()) > 0:
            try:
                return float(match.group(1))
            except:
                return 0.0

        return sum(ord(c) for c in filename) / 10000.0

    def process_paths(self, path_list, remove_file_suffix, filter_suffixes, regex_filter, regex_sort_pattern, sort_order, path_duplication_count):
        if isinstance(remove_file_suffix, list):
            remove_file_suffix = remove_file_suffix[0]
        if isinstance(filter_suffixes, list):
            filter_suffixes = filter_suffixes[0]
        if isinstance(regex_filter, list):
            regex_filter = regex_filter[0]
        if isinstance(regex_sort_pattern, list):
            regex_sort_pattern = regex_sort_pattern[0]
        if isinstance(sort_order, list):
            sort_order = sort_order[0]
        if isinstance(path_duplication_count, list):
            path_duplication_count = path_duplication_count[0]

        if isinstance(path_list, list):
            raw_paths = []
            for item in path_list:
                if isinstance(item, str):
                    paths = [p.strip().strip('"').strip("'") for p in item.split("\n") if p.strip()]
                    raw_paths.extend(paths)
                elif isinstance(item, list):
                    for sub_item in item:
                        paths = [p.strip().strip('"').strip("'") for p in str(sub_item).split("\n") if p.strip()]
                        raw_paths.extend(paths)
        else:
            raw_paths = [p.strip().strip('"').strip("'") for p in str(path_list).split("\n") if p.strip()]

        processed_paths = [p for p in raw_paths if os.path.normpath(p)]

        filtered_paths = []
        filter_suffixes_list = [s.strip().lower() for s in filter_suffixes.split("|") if s.strip()]
        if not filter_suffixes_list:
            filtered_paths = processed_paths.copy()
        else:
            for path in processed_paths:
                file_ext = os.path.splitext(path)[1].lower().lstrip(".")
                if file_ext in filter_suffixes_list:
                    filtered_paths.append(path)

        if regex_filter:
            try:
                filter_re = re.compile(regex_filter, re.IGNORECASE)
                filtered_paths = [path for path in filtered_paths if filter_re.match(path)]
            except re.error as e:
                pass

        if filtered_paths:
            sorted_paths = sorted(
                filtered_paths,
                key=lambda x: self._extract_regex_sort_key(x, regex_sort_pattern),
                reverse=(sort_order == "down")
            )
            filtered_paths = sorted_paths

        reused_paths = []
        for path in filtered_paths:
            reused_paths.extend([path] * path_duplication_count)
        filtered_paths = reused_paths

        folder_names = []
        file_names = []
        new_path_list = filtered_paths.copy()
        seen_folders = set()

        for path in filtered_paths:
            parent_dir = os.path.dirname(os.path.normpath(path))
            folder_name = os.path.basename(parent_dir)
            if folder_name not in seen_folders:
                seen_folders.add(folder_name)

            file_name = os.path.basename(os.path.normpath(path))
            if remove_file_suffix:
                file_name = os.path.splitext(file_name)[0]
            file_names.append(file_name)

        folder_names = list(seen_folders)

        return (new_path_list, file_names, folder_names)




class IO_RegexPreset:
    CATEGORY = "Apt_Preset/IO_Port"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("regex_pattern", "rule_description")
    FUNCTION = "generate_regex"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["排除空内容", "排除空+纯空白", "包含关键词", "不包含关键词", "以关键词开头", "以关键词结尾", "多关键词包含(或)", "多关键词包含(且)", "批量排除关键词", "匹配数字", "匹配字母", "匹配中文", "匹配数字+字母", "排除特定值", "仅匹配特定值", "匹配文件名开头数字", "匹配文件名结尾数字", "匹配括号中的数字", "自定义"], {"default": "自定义"}),
            },
            "optional": {
                "custom_regex": ("STRING", {"default": "", "placeholder": "自定义正则（无需/包裹，例：^[\u4e00-\u9fa5]+$）"}),
                "keyword": ("STRING", {"default": "", "placeholder": "输入关键词，多值用|分隔（例：油画|水彩）"}),
                "ignore_case": ("BOOLEAN", {"default": True, "label_on": "忽略大小写", "label_off": "区分大小写"}),
                "escape_special_chars": ("BOOLEAN", {"default": True, "label_on": "转义特殊字符", "label_off": "保留特殊字符"}),
                "preview_test_text": ("STRING", {"default": "", "multiline": True, "placeholder": "测试文本，多值用|分隔（例：12图片.png|图片(34).png）"})
            }
        }

    def get_rule_desc(self, preset, keyword, ignore_case):
        desc_map = {
            "排除空内容": "筛选掉空字符串选项，仅保留含至少1个字符的选项",
            "排除空+纯空白": "筛选掉空/纯空格/制表符选项，仅保留非空白字符选项",
            "包含关键词": f"仅显示包含「{keyword}」的选项（{'忽略' if ignore_case else '区分'}大小写）",
            "不包含关键词": f"排除所有包含「{keyword}」的选项（{'忽略' if ignore_case else '区分'}大小写）",
            "以关键词开头": f"仅显示以「{keyword}」开头的选项（{'忽略' if ignore_case else '区分'}大小写）",
            "以关键词结尾": f"仅显示以「{keyword}」结尾的选项（{'忽略' if ignore_case else '区分'}大小写）",
            "多关键词包含(或)": f"仅显示包含「{keyword.replace('|', '」或「')}」的选项（{'忽略' if ignore_case else '区分'}大小写）",
            "多关键词包含(且)": f"仅显示同时包含「{keyword.replace('|', '」和「')}」的选项（{'忽略' if ignore_case else '区分'}大小写）",
            "批量排除关键词": f"排除包含「{keyword.replace('|', '」或「')}」的所有选项（{'忽略' if ignore_case else '区分'}大小写）",
            "匹配数字": "仅显示包含数字（0-9）的选项",
            "匹配字母": "仅显示包含英文字母（a-z/A-Z）的选项",
            "匹配中文": "仅显示包含中文（\u4e00-\u9fa5）的选项",
            "匹配数字+字母": "仅显示包含数字或字母的选项",
            "排除特定值": f"精准排除「{keyword}」这个选项（完全匹配）",
            "仅匹配特定值": f"仅显示「{keyword}」这个选项（完全匹配）",
            "匹配文件名开头数字": "仅匹配文件名开头的连续数字（如12图片.png中的12）",
            "匹配文件名结尾数字": "仅匹配文件名扩展名前的最后两位数字（如图片63.png中的63）",
            "匹配括号中的数字": "仅匹配括号内的连续数字（如图片(45).png中的45）",
            "自定义": f"使用自定义正则：{self.custom_regex if hasattr(self, 'custom_regex') else '无'}"
        }
        return desc_map.get(preset, "未知筛选规则")


    def preview_regex_effect(self, regex_pattern, test_text):
        if not regex_pattern or not test_text:
            return "无测试数据"
        core_regex = regex_pattern.strip('/')
        if not core_regex:
            return "正则格式错误"
        test_options = [opt.strip() for opt in test_text.split('|')]
        matched_options = []
        try:
            flags = re.IGNORECASE if self.ignore_case else 0
            pattern = re.compile(core_regex, flags)
            for opt in test_options:
                if pattern.search(opt):
                    matched_options.append(opt)
        except Exception as e:
            return f"正则错误：{str(e)}"
        return f"匹配结果：{', '.join(matched_options) if matched_options else '无匹配项'}"


    def generate_regex(self, preset, keyword, ignore_case, escape_special_chars, custom_regex="", preview_test_text=""):
        self.ignore_case = ignore_case
        self.custom_regex = custom_regex
        keyword = keyword.strip() if keyword else ""
        processed_keyword = re.escape(keyword) if escape_special_chars and keyword else keyword
        keyword_list = [k.strip() for k in processed_keyword.split('|') if k.strip()]
        case_flag = "i" if ignore_case else ""
        regex_map = {
            "排除空内容": rf"/.+/{case_flag}",
            "排除空+纯空白": rf"/^\S+/{case_flag}",
            "包含关键词": rf"/{processed_keyword}/{case_flag}" if keyword else rf"/.+/{case_flag}",
            "不包含关键词": rf"/^(?!.*{processed_keyword}).*$/{case_flag}" if keyword else rf"/.+/{case_flag}",
            "以关键词开头": rf"/^{processed_keyword}/{case_flag}" if keyword else rf"/.+/{case_flag}",
            "以关键词结尾": rf"/{processed_keyword}$/{case_flag}" if keyword else rf"/.+/{case_flag}",
            "多关键词包含(或)": rf"/{'|'.join(keyword_list)}/{case_flag}" if keyword_list else rf"/.+/{case_flag}",
            "多关键词包含(且)": rf"/^(?=.*{')(?=.*'.join(keyword_list)}).*$/{case_flag}" if keyword_list else rf"/.+/{case_flag}",
            "批量排除关键词": rf"/^(?!.*({'|'.join(keyword_list)})).*$/{case_flag}" if keyword_list else rf"/.+/{case_flag}",
            "匹配数字": rf"/\d+/{case_flag}",
            "匹配字母": rf"/[a-zA-Z]+/{case_flag}",
            "匹配中文": rf"/[\u4e00-\u9fa5]+/{case_flag}",
            "匹配数字+字母": rf"/[a-zA-Z0-9]+/{case_flag}",
            "排除特定值": rf"/^(?!{processed_keyword}$).*$/{case_flag}" if keyword else rf"/.+/{case_flag}",
            "仅匹配特定值": rf"/^{processed_keyword}$/{case_flag}" if keyword else rf"/.+/{case_flag}",
            "匹配文件名开头数字": rf"/^\d+/{case_flag}",
            "匹配文件名结尾数字": rf"/\d{{2}}(?=\.\w+$)/{case_flag}",
            "匹配括号中的数字": rf"/\((\d+)\)/{case_flag}",
            "自定义": rf"/{custom_regex}/{case_flag}" if custom_regex else rf"/.+/{case_flag}"
        }
        final_regex = regex_map.get(preset, rf"/.+/{case_flag}")
        rule_desc = self.get_rule_desc(preset, keyword, ignore_case)
        if preview_test_text:
            preview_result = self.preview_regex_effect(final_regex, preview_test_text)
            rule_desc += f"\n{preview_result}"
        return (final_regex, rule_desc)








