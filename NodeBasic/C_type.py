
import comfy
import torch
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F
import comfy.utils

from ..main_unit import *


PACK_PREFIX = 'value'

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)
    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)
    return mask



#region---------------type---------------

class Pack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "pack"
    RETURN_TYPES = ("PACK", )
    RETURN_NAMES = ("PACK", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        data = {}
        pack = {
            "id": unique_id,
            "data": data,
        }
        for k, v in kwargs.items():
            if k.startswith('value'):
                i = int(k.split("_")[1])
                data[i - 1] = {
                    "name": cur_node["inputs"][i - 1]["name"],
                    "type": cur_node["inputs"][i - 1]["type"],
                    "value": v,
                }

        return (pack, )


class ByPassTypeTuple(tuple):
	def __getitem__(self, index):
		if index > 0:
			index = 0
		item = super().__getitem__(index)
		if isinstance(item, str):
			return AnyType(item)
		return item



class Unpack:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "PACK": ("PACK", ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "unpack"
    RETURN_TYPES = ByPassTypeTuple(("*", ))
    RETURN_NAMES = ByPassTypeTuple(("value_1", ))
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def run(self, PACK: dict, unique_id, prompt, extra_pnginfo):
        length = len(PACK["data"])
        types = []
        names = []
        outputs = []
        for i in range(length):
            d = PACK["data"][i]
            names.append(d["name"])
            types.append(d["type"])
            outputs.append(d["value"])
        return tuple(outputs)



class type_BasiPIPE:
    
    @classmethod
    def INPUT_TYPES(self):
        return {
            "optional": {
                "context": ("RUN_CONTEXT", ),
            },
        }
    RETURN_TYPES = ("BASIC_PIPE",)
    RETURN_NAMES = ("basic_pipe",)
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    
    FUNCTION = "fn"

    def fn(self, context):
        pipe = (context['model'], context['clip'], context['vae'], context['positive'], context['negative'])
        return pipe,



#region-------批次与列表---------------

class type_Image_List2Batch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "images": ("IMAGE", ),
                    }
                }

    INPUT_IS_LIST = True

    RETURN_TYPES = ("IMAGE", )
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/data/list|Batch"

    def doit(self, images):
        if len(images) <= 1:
            return (images[0],)
        else:
            image1 = images[0]
            for image2 in images[1:]:
                if image1.shape[1:] != image2.shape[1:]:
                    image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
                image1 = torch.cat((image1, image2), dim=0)
            return (image1,)




class type_Image_List2Batch_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_list": ("IMAGE", {"forceInput": True}),
                "mode": (["Crop", "Pad", "Stretch"], {"default": "Pad"}),
                "alignment": (["Top Left", "Top", "Top Right", "Left", "Center", "Right", "Bottom Left", "Bottom", "Bottom Right"], {"default": "Center"}),
                "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
                "background_color": (["white", "black", "green", "red", "blue", "gray"], {"default": "black"})
            }
        }

    INPUT_IS_LIST = True
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "convert"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def parse_color(self, color_input):
        color_map = {
            "white": [1.0, 1.0, 1.0],
            "black": [0.0, 0.0, 0.0],
            "green": [0.0, 1.0, 0.0],
            "red": [1.0, 0.0, 0.0],
            "blue": [0.0, 0.0, 1.0],
            "gray": [0.5, 0.5, 0.5]
        }
        
        if isinstance(color_input, str) and color_input in color_map:
            return color_map[color_input]
        
        return [0.0, 0.0, 0.0]

    def calculate_alignment(self, align_type, diff_w, diff_h):
        diff_w, diff_h = max(0, diff_w), max(0, diff_h)
        x_offset = diff_w // 2
        y_offset = diff_h // 2

        if "Left" in align_type: x_offset = 0
        elif "Right" in align_type: x_offset = diff_w
        
        if "Top" in align_type: y_offset = 0
        elif "Bottom" in align_type: y_offset = diff_h
            
        return x_offset, y_offset

    def convert(self, image_list, mode, alignment, width, height, background_color=None):
        if isinstance(mode, list): mode = mode[0]
        if isinstance(alignment, list): alignment = alignment[0]
        if isinstance(width, list): width = width[0]
        if isinstance(height, list): height = height[0]
        if isinstance(background_color, list): background_color = background_color[0]
        
        bg_rgb = self.parse_color(background_color)
        
        if not image_list:
            return (torch.zeros([1, 64, 64, 3]),)

        raw_images = []
        for item in image_list:
            if isinstance(item, torch.Tensor):
                if item.shape[-1] == 1:
                    item = item.repeat(1, 1, 1, 3) if item.ndim == 4 else item.unsqueeze(-1).repeat(1, 1, 1, 3)
                elif item.shape[-1] == 4:
                    item = item[:, :, :, :3]
                elif item.shape[-1] != 3:
                     item = item[:, :, :, :3]

                if item.ndim == 3:
                    raw_images.append(item.unsqueeze(0))
                elif item.ndim == 4:
                    raw_images.append(item)

        if not raw_images:
            return (torch.zeros([1, 64, 64, 3]),)

        target_h, target_w = height, width
        
        processed_images = []
        
        for img_batch in raw_images:
            for i in range(img_batch.shape[0]):
                img = img_batch[i:i+1]
                curr_h, curr_w = img.shape[1], img.shape[2]
                
                img_chw = img.permute(0, 3, 1, 2)
                
                if mode == "Stretch":
                    img_out = F.interpolate(img_chw, size=(target_h, target_w), mode='bilinear', align_corners=False)
                
                elif mode == "Crop":
                    scale = max(target_w / curr_w, target_h / curr_h)
                    new_w, new_h = max(target_w, round(curr_w * scale)), max(target_h, round(curr_h * scale))
                    resized = F.interpolate(img_chw, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    diff_w, diff_h = new_w - target_w, new_h - target_h
                    x_off, y_off = self.calculate_alignment(alignment, diff_w, diff_h)
                    
                    img_out = resized[:, :, y_off:y_off+target_h, x_off:x_off+target_w]
                
                elif mode == "Pad":
                    scale = min(target_w / curr_w, target_h / curr_h)
                    new_w, new_h = max(1, round(curr_w * scale)), max(1, round(curr_h * scale))
                    resized = F.interpolate(img_chw, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    
                    diff_w, diff_h = target_w - new_w, target_h - new_h
                    x_off, y_off = self.calculate_alignment(alignment, diff_w, diff_h)
                    
                    bg_tensor = torch.tensor(bg_rgb, device=img.device, dtype=img.dtype).view(1, 3, 1, 1)
                    final_canvas = bg_tensor.expand(1, 3, target_h, target_w).clone()
                    
                    final_canvas[:, :, y_off:y_off+new_h, x_off:x_off+new_w] = resized
                    img_out = final_canvas

                processed_images.append(img_out.permute(0, 2, 3, 1))
        
        if not processed_images:
             return (torch.zeros([1, target_h, target_w, 3]),)
             
        final_batch = torch.cat(processed_images, dim=0)
        return (final_batch,)




class type_Image_Batch2List:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/data/list|Batch"

    def doit(self, image):
        images = [image[i:i + 1, ...] for i in range(image.shape[0])]
        return (images, )


class type_Mask_Batch2List:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "masks": ("MASK", ),
                    }
                }

    RETURN_TYPES = ("MASK", )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def doit(self, masks):
        if masks is None:
            empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            return ([empty_mask], )

        res = []

        for mask in masks:
            res.append(mask)

        print(f"mask len: {len(res)}")

        res = [make_3d_mask(x) for x in res]

        return (res, )


class type_Mask_List2Batch:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                        "mask": ("MASK", ),
                    }
                }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("MASK", )
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def doit(self, mask):
        if len(mask) == 1:
            mask = make_3d_mask(mask[0])
            return (mask,)
        elif len(mask) > 1:
            mask1 = make_3d_mask(mask[0])

            for mask2 in mask[1:]:
                mask2 = make_3d_mask(mask2)
                if mask1.shape[1:] != mask2.shape[1:]:
                    mask2 = comfy.utils.common_upscale(mask2.movedim(-1, 1), mask1.shape[2], mask1.shape[1], "lanczos", "center").movedim(1, -1)
                mask1 = torch.cat((mask1, mask2), dim=0)

            return (mask1,)
        else:
            empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu").unsqueeze(0)
            return (empty_mask,)


class type_BatchToList:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LIST": ("LIST", {"forceInput": True}),
            }
        }
    
    TITLE = "Batch To List"
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, LIST: list):
        return (LIST, )


class type_ListToBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {"forceInput": True}),
            }
        }
    
    TITLE = "List To Batch"
    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("LIST", )
    INPUT_IS_LIST = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, ANY: list):
        return (ANY, )



#endregion---------------------------






class type_AnyCast:
    def __init__(self):
        # 类型构造函数映射
        self.type_constructor = {
            "LIST": list,
            "SET": set,
            "DICTIONARY": dict,
            "TUPLE": tuple,
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {}),
                "TYPE": (["anytype", "STRING", "INT", "FLOAT", "LIST", "SET", "TUPLE", "DICTIONARY", "BOOLEAN", "UTF8_STRING"], {}),
            },
        }
    
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, ANY, TYPE):
        # 处理 None 值
        if ANY is None:
            if TYPE in ["INT", "FLOAT"]:
                return (0 if TYPE == "INT" else 0.0,)
            elif TYPE == "STRING":
                return ("",)
            elif TYPE == "BOOLEAN":
                return (False,)
            elif TYPE in ["LIST", "SET", "DICTIONARY", "TUPLE"]:
                return (self.type_constructor[TYPE](),)
            elif TYPE == "UTF8_STRING":
                return ("",)
            else:
                return (None,)

        if TYPE == "LIST":
            if isinstance(ANY, list):
                return ([self.try_cast(item, "ANY") for item in ANY],)
            elif isinstance(ANY, set):
                return (list(ANY),)
            else:
                return ([ANY],)
                
        elif TYPE == "SET":
            if isinstance(ANY, set):
                return (ANY,)
            elif isinstance(ANY, list):
                return (set(ANY),)
            elif isinstance(ANY, dict):
                return (set(ANY.keys()),)
            else:
                return ({ANY},)
                
        elif TYPE == "TUPLE":
            if isinstance(ANY, tuple):
                return (ANY,)
            elif isinstance(ANY, list):
                return (tuple(ANY),)
            else:
                return ((ANY,),)
                
        elif TYPE == "DICTIONARY":
            if isinstance(ANY, dict):
                return ({k: self.try_cast(v, "ANY") for k, v in ANY.items()},)
            elif isinstance(ANY, str):
                try:
                    import json
                    parsed = json.loads(ANY)
                    return (self.try_cast(parsed, "DICTIONARY")[0],)
                except (json.JSONDecodeError, TypeError):
                    try:
                        import ast
                        parsed = ast.literal_eval(ANY)
                        return (self.try_cast(parsed, "DICTIONARY")[0],)
                    except (SyntaxError, ValueError):
                        return ({},)
            elif isinstance(ANY, list) and len(ANY) > 0:
                if isinstance(ANY[0], (list, tuple)) and len(ANY[0]) == 2:
                    return (dict(ANY),)
            return ({},)
            
        elif TYPE == "BOOLEAN":
            if isinstance(ANY, str):
                return (ANY.lower() in ["true", "1", "yes"],)
            return (bool(ANY),)
            
        elif TYPE == "INT":
            if isinstance(ANY, str):
                try:
                    return (int(float(ANY)),)
                except ValueError:
                    return (0,)
            return (int(ANY),)
            
        elif TYPE == "FLOAT":
            if isinstance(ANY, str):
                try:
                    return (float(ANY),)
                except ValueError:
                    return (0.0,)
            return (float(ANY),)
            
        elif TYPE == "STRING":
            return (str(ANY),)
            
        elif TYPE == "UTF8_STRING":
            try:
                # 执行 UTF-8 编码转换
                encoded_bytes = str(ANY).encode('utf-8', 'ignore')
                encoded_text = encoded_bytes.decode('utf-8', 'replace')
                return (encoded_text,)
            except Exception as e:
                return (f"Error during UTF-8 encoding: {e}",)
                
        else:  # 其他类型或 ANY_TYPE
            return (ANY,)

    def try_cast(self, value, target_type):
        # 递归调用 run 方法处理嵌套结构
        return self.run(value, target_type)[0]






#endregion---------------type---------------






#region---------------create--------------


class create_any_List:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "create_any_List"
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith(PACK_PREFIX):
                output_list.append(v)
        return (output_list, )



class create_any_batch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "create_any_batch"
    RETURN_TYPES = ("LIST",)  
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith(PACK_PREFIX):
                output_list.append(v)
        return (output_list, )





class create_mask_batch:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "doit"
    NAME = "create_mask_batch"

    CATEGORY = "Apt_Preset/data"

    def doit(self, unique_id, prompt, extra_pnginfo, **kwargs):
        # 处理所有输入遮罩，统一格式和类型
        masks = []
        for value in kwargs.values():
            if value is None:
                continue
                
            # 转换为3D遮罩并标准化数据类型
            mask = make_3d_mask(value)
            
            # 确保数据类型正确（转换为float32）
            if mask.dtype != torch.float32:
                mask = mask.to(torch.float32) / 255.0  # 处理可能的0-255范围数据
            
            # 确保维度正确 (1, H, W)
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            elif len(mask.shape) > 3:
                mask = mask.squeeze()  # 移除多余维度
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0)
            
            masks.append(mask)
        
        if len(masks) == 0:
            return (torch.zeros((1, 64, 64), dtype=torch.float32),)
        
        # 确定最小尺寸作为统一尺寸（使用裁切而非缩放）
        min_height = min(mask.shape[1] for mask in masks)
        min_width = min(mask.shape[2] for mask in masks)
        
        # 统一所有遮罩到最小尺寸（居中裁切）
        processed_masks = []
        for mask in masks:
            h, w = mask.shape[1], mask.shape[2]
            
            # 计算裁切区域（居中裁切）
            h_start = (h - min_height) // 2
            h_end = h_start + min_height
            w_start = (w - min_width) // 2
            w_end = w_start + min_width
            
            # 执行裁切
            cropped = mask[:, h_start:h_end, w_start:w_end]
            processed_masks.append(cropped)
        
        # 拼接所有遮罩
        combined_mask = torch.cat(processed_masks, dim=0)
        return (combined_mask,)



#endregion---------------create--------------



def make_3d_mask(mask):
    if mask.dim() == 2:
        return mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[0] == 1:
        return mask
    elif mask.dim() == 3 and mask.shape[2] == 1:
        return mask.permute(2, 0, 1)
    else:
        return mask


class create_image_batch:
    @classmethod 
    def INPUT_TYPES(s):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "doit"
    NAME = "create_image_batch"
    CATEGORY = "Apt_Preset/data"

    def doit(self, unique_id, prompt, extra_pnginfo, **kwargs):
        images = [value for value in kwargs.values() if value is not None]
        
        if len(images) == 0:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        
        image1 = images[0]
        for image2 in images[1:]:
            if image1.shape[1:] != image2.shape[1:]:
                image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "bicubic", "center").movedim(1, -1)
            image1 = torch.cat((image1, image2), dim=0)
        return (image1,)





class ImageBatchMultiple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
                "method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], { "default": "lanczos" }),
            }, "optional": {
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
            },
        }
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "essentials/image batch"

    def execute(self, image_1, method, image_2=None, image_3=None, image_4=None, image_5=None):
        out = image_1

        if image_2 is not None:
            if image_1.shape[1:] != image_2.shape[1:]:
                image_2 = comfy.utils.common_upscale(image_2.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((image_1, image_2), dim=0)
        if image_3 is not None:
            if image_1.shape[1:] != image_3.shape[1:]:
                image_3 = comfy.utils.common_upscale(image_3.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_3), dim=0)
        if image_4 is not None:
            if image_1.shape[1:] != image_4.shape[1:]:
                image_4 = comfy.utils.common_upscale(image_4.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_4), dim=0)
        if image_5 is not None:
            if image_1.shape[1:] != image_5.shape[1:]:
                image_5 = comfy.utils.common_upscale(image_5.movedim(-1,1), image_1.shape[2], image_1.shape[1], method, "center").movedim(1,-1)
            out = torch.cat((out, image_5), dim=0)

        return (out,)



