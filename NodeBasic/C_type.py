
import torch
import comfy
import torch
import numpy as np
from PIL import Image, ImageOps

import torch
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
    
    TITLE = "Pack"
    RETURN_TYPES = ("PACK", )
    RETURN_NAMES = ("PACK", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

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
    
    TITLE = "Unpack"
    RETURN_TYPES = ByPassTypeTuple(("*", ))
    RETURN_NAMES = ByPassTypeTuple(("value_1", ))
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/😺backup"

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



#region---------------废弃---------------

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

    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

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


class type_Image_Batch2List:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "doit"

    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

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
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

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
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

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
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

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
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def run(self, ANY: list):
        return (ANY, )


#endregion---------------废弃---------------





class xxxtype_AnyCast:
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
    
    TITLE = "Create List"
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
    
    TITLE = "Create Batch"
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

    CATEGORY = "Apt_Preset/data"

    def doit(self, unique_id, prompt, extra_pnginfo, **kwargs):
        images = [value for value in kwargs.values() if value is not None]
        
        if len(images) == 0:
            return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
        
        image1 = images[0]
        for image2 in images[1:]:
            if image1.shape[1:] != image2.shape[1:]:
                image2 = comfy.utils.common_upscale(image2.movedim(-1, 1), image1.shape[2], image1.shape[1], "lanczos", "center").movedim(1, -1)
            image1 = torch.cat((image1, image2), dim=0)
        return (image1,)




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


class type_AnyCast:
    def __init__(self):
        self.type_constructor = {
            "list": list,
            "set": set,
            "dictionary": dict,
            "tuple": tuple,
        }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {}),
                "TYPE": ([
                    "anytype", 
                    "string", 
                    "int", 
                    "float", 
                    "list", 
                    "set", 
                    "tuple", 
                    "dictionary", 
                    "boolean", 
                    "utf8_string",
                    "-------------------------",
                    "image_list_to_batch",
                    "image_batch_to_list",
                    "mask_list_to_batch",
                    "mask_batch_to_list",
                    "batch_to_list",
                    "list_to_batch"
                ], {}),
            },
        }
    
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data"
    OUTPUT_IS_LIST = (True, )
    INPUT_IS_LIST = True
    # 根据类型动态设置是否返回列表
    def get_output_flags(self, TYPE):
        if TYPE in ["mask_batch_to_list", "image_batch_to_list", "batch_to_list"]:
            return {"OUTPUT_IS_LIST": (True,)}
        return {}

    def run(self, ANY, TYPE):
        if ANY is None:
            if TYPE in ["int", "float"]:
                return (0 if TYPE == "int" else 0.0,)
            elif TYPE == "string":
                return ("",)
            elif TYPE == "boolean":
                return (False,)
            elif TYPE in ["list", "set", "dictionary", "tuple"]:
                return (self.type_constructor[TYPE](),)
            elif TYPE == "utf8_string":
                return ("",)
            elif TYPE == "image_list_to_batch":
                return (torch.zeros((1, 64, 64, 3), dtype=torch.float32),)
            elif TYPE in ["mask_list_to_batch", "mask_batch_to_list"]:
                empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                return ([empty_mask],) if TYPE == "mask_batch_to_list" else (torch.zeros((1, 64, 64), dtype=torch.float32),)
            elif TYPE == "image_batch_to_list":
                return ([torch.zeros((64, 64, 3), dtype=torch.float32)],)
            elif TYPE == "batch_to_list":
                return ([],)
            elif TYPE == "list_to_batch":
                return ([],)
            else:
                return (None,)

        if TYPE == "image_list_to_batch":
            if not isinstance(ANY, list):
                return (ANY,)
            
            if len(ANY) <= 1:
                return (ANY[0],)
            else:
                image1 = ANY[0]
                for image2 in ANY[1:]:
                    if image1.shape[1:] != image2.shape[1:]:
                        image2 = comfy.utils.common_upscale(
                            image2.movedim(-1, 1), 
                            image1.shape[2], 
                            image1.shape[1], 
                            "lanczos", 
                            "center"
                        ).movedim(1, -1)
                    image1 = torch.cat((image1, image2), dim=0)
                return (image1,)
        
        elif TYPE == "image_batch_to_list":
            if not isinstance(ANY, torch.Tensor):
                return ([ANY],)
            
            if len(ANY.shape) == 3:
                return ([ANY],)
            elif len(ANY.shape) == 4:
                images = [ANY[i] for i in range(ANY.shape[0])]
                return (images,)
            else:
                return ([ANY],)
        
        elif TYPE == "mask_list_to_batch":
            if not isinstance(ANY, list):
                mask = make_3d_mask(ANY)
                return (mask,)
                
            if len(ANY) == 1:
                mask = make_3d_mask(ANY[0])
                return (mask,)
            elif len(ANY) > 1:
                mask1 = make_3d_mask(ANY[0])

                for mask2 in ANY[1:]:
                    mask2 = make_3d_mask(mask2)
                    if mask1.shape[1:] != mask2.shape[1:]:
                        mask2 = comfy.utils.common_upscale(
                            mask2.movedim(-1, 1), 
                            mask1.shape[2], 
                            mask1.shape[1], 
                            "lanczos", 
                            "center"
                        ).movedim(1, -1)
                    mask1 = torch.cat((mask1, mask2), dim=0)

                return (mask1,)
            else:
                empty_mask = torch.zeros((1, 64, 64), dtype=torch.float32, device="cpu")
                return (empty_mask,)
        

        elif TYPE == "mask_batch_to_list":
            # 确保返回遮罩列表，每个元素都是2D张量
            res = []
            if ANY is None:
                empty_mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
                return ([empty_mask],)

            if isinstance(ANY, torch.Tensor):
                if len(ANY.shape) == 2:  # 单个遮罩，包装成列表
                    return ([ANY],)
                elif len(ANY.shape) == 3:  # 批次遮罩 (B, H, W)
                    for i in range(ANY.shape[0]):
                        res.append(ANY[i])  # 每个元素都是2D张量
                elif len(ANY.shape) == 4:  # 处理四维遮罩格式
                    if ANY.shape[0] == 1:  # (1, B, H, W)
                        for i in range(ANY.shape[1]):
                            res.append(ANY[0, i])
                    else:  # (B, 1, H, W)
                        for i in range(ANY.shape[0]):
                            res.append(ANY[i, 0])
                return (res,)
            else:  # 已经是列表
                for mask in ANY:
                    res.append(make_3d_mask(mask)[0] if make_3d_mask(mask).dim() == 3 else mask)
                return (res,)

        elif TYPE == "batch_to_list":
            if isinstance(ANY, list):
                return (ANY,)
            else:
                return ([ANY],)
        
        elif TYPE == "list_to_batch":
            if isinstance(ANY, list):
                return (ANY,)
            else:
                return ([ANY],)

        elif TYPE == "list":
            if isinstance(ANY, list):
                return ([self.try_cast(item, "anytype") for item in ANY],)
            elif isinstance(ANY, set):
                return (list(ANY),)
            else:
                return ([ANY],)
                
        elif TYPE == "set":
            if isinstance(ANY, set):
                return (ANY,)
            elif isinstance(ANY, list):
                return (set(ANY),)
            elif isinstance(ANY, dict):
                return (set(ANY.keys()),)
            else:
                return ({ANY},)
                
        elif TYPE == "tuple":
            if isinstance(ANY, tuple):
                return (ANY,)
            elif isinstance(ANY, list):
                return (tuple(ANY),)
            else:
                return ((ANY,),)
                
        elif TYPE == "dictionary":
            if isinstance(ANY, dict):
                return ({k: self.try_cast(v, "anytype") for k, v in ANY.items()},)
            elif isinstance(ANY, str):
                try:
                    import json
                    parsed = json.loads(ANY)
                    return (self.try_cast(parsed, "dictionary")[0],)
                except (json.JSONDecodeError, TypeError):
                    try:
                        import ast
                        parsed = ast.literal_eval(ANY)
                        return (self.try_cast(parsed, "dictionary")[0],)
                    except (SyntaxError, ValueError):
                        return ({},)
            elif isinstance(ANY, list) and len(ANY) > 0:
                if isinstance(ANY[0], (list, tuple)) and len(ANY[0]) == 2:
                    return (dict(ANY),)
            return ({},)
            
        elif TYPE == "boolean":
            if isinstance(ANY, str):
                return (ANY.lower() in ["true", "1", "yes"],)
            return (bool(ANY),)
            
        elif TYPE == "int":
            if isinstance(ANY, str):
                try:
                    return (int(float(ANY)),)
                except ValueError:
                    return (0,)
            return (int(ANY),)
            
        elif TYPE == "float":
            if isinstance(ANY, str):
                try:
                    return (float(ANY),)
                except ValueError:
                    return (0.0,)
            return (float(ANY),)
            
        elif TYPE == "string":
            return (str(ANY),)
            
        elif TYPE == "utf8_string":
            try:
                encoded_bytes = str(ANY).encode('utf-8', 'ignore')
                encoded_text = encoded_bytes.decode('utf-8', 'replace')
                return (encoded_text,)
            except Exception as e:
                return (f"Error during UTF-8 encoding: {e}",)
                
        else:
            return (ANY,)

    def try_cast(self, value, target_type):
        return self.run(value, target_type)[0]

















