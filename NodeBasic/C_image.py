
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
import torchvision.transforms.functional as TF
from typing import Literal, Any
import math
from comfy_extras.nodes_mask import composite
from comfy.utils import common_upscale
import re
from math import ceil, sqrt
from typing import cast



from PIL import Image, ImageDraw,  ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageOps

from tqdm import tqdm

from enum import Enum
import random
import folder_paths

import copy
from pymatting import estimate_alpha_cf, estimate_foreground_ml, fix_trimap
import ast
from nodes import CLIPTextEncode, common_ksampler,InpaintModelConditioning


import torch.nn.functional as F

import node_helpers
from typing import Tuple

import torch.nn.functional as F
import comfy.utils




from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader






if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#---------------------安全导入

try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False

try:
    from scipy.interpolate import CubicSpline
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    CubicSpline = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False

try:
    import onnxruntime as ort
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    ort = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False





#--------------------------------------------------------------------------------------#








#region --------batch-------------------------



class Blend: # 调用
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "blend_factor": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01
                }),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blend_images"

    CATEGORY = "image/postprocessing"

    def blend_images(self, image1: torch.Tensor, image2: torch.Tensor, blend_factor: float, blend_mode: str):
        image1, image2 = node_helpers.image_alpha_fix(image1, image2)
        image2 = image2.to(image1.device)
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(image2, image1.shape[2], image1.shape[1], upscale_method='bicubic', crop='center')
            image2 = image2.permute(0, 2, 3, 1)

        blended_image = self.blend_mode(image1, image2, blend_mode)
        blended_image = image1 * (1 - blend_factor) + blended_image * blend_factor
        blended_image = torch.clamp(blended_image, 0, 1)
        return (blended_image,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            return torch.where(img2 <= 0.5, img1 - (1 - 2 * img2) * img1 * (1 - img1), img1 + (2 * img2 - 1) * (self.g(img1) - img1))
        elif mode == "difference":
            return img1 - img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def g(self, x):
        return torch.where(x <= 0.25, ((16 * x - 12) * x + 4) * x, torch.sqrt(x))



class str_edit:
    def __init__(self):
        pass
    @classmethod
    def convert_list(cls, string_input,arrangement=True):
        if string_input == "":
            return ([],)
        if arrangement:
            string_input = cls.tolist_v1(string_input)
        if string_input[0] != "[":
            string_input = "[" + string_input + "]"
            return (ast.literal_eval(string_input),)
        else:
            return (ast.literal_eval(string_input),)
        
    def tolist_v1(cls,user_input):#转换为简单的带负数多维数组格式
        user_input = user_input.replace('{', '[').replace('}', ']')# 替换大括号
        user_input = user_input.replace('(', '[').replace(')', ']')# 替换小括号
        user_input = user_input.replace('，', ',')# 替换中文逗号
        user_input = re.sub(r'\s+', '', user_input)#去除空格和换行符
        user_input = re.sub(r'[^\d,.\-[\]]', '', user_input)#去除非数字字符，但不包括,.-[]
        return user_input
    @classmethod
    def tolist_v2(cls,str_input,to_list=True,to_oneDim=False,to_int=False,positive=False):#转换为数组格式
        if str_input == "":
            if to_list:return ([],)
            else:return ""
        else:
            str_input = str_input.replace('，', ',')# 替换中文逗号
            if to_oneDim:
                str_input = re.sub(r'[\(\)\[\]\{\}（）【】｛｝]', "" , str_input)
                str_input = "[" + str_input + "]"
            else:
                text=re.sub(r'[\(\[\{（【｛]', '[', text)#替换括号
                text=re.sub(r'[\)\]\}）】｝]', ']', text)#替换反括号
                if str_input[0] != "[":str_input = "[" + str_input + "]"
            str_input = re.sub(r'[^\d,.\-[\]]', '', str_input)#去除非数字字符，但不包括,.-[]
            str_input = re.sub(r'(?<![0-9])[,]', '', str_input)#如果,前面不是数字则去除
            #str_input = re.sub(r'(-{2,}|\.{2,})', '', str_input)#去除多余的.和-
            str_input = re.sub(r'\.{2,}', '.', str_input)#去除多余的.
            if positive:
                str_input = re.sub(r'-','', str_input)#移除-
            else:
                str_input = re.sub(r'-{2,}', '-', str_input)#去除多余的-
            list1=np.array(ast.literal_eval(str_input))
            if to_int:
                list1=list1.astype(int)
            if to_list:
                return list1.tolist()
            else:
                return str_input
            
    def repair_brackets(cls,str_input):#括号补全(待开发)
        pass




class Image_batch_select:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "indexes": ("STRING", {"default": "1,2"}),  
                "canvas_operations": (["None", "Horizontal Flip", "Vertical Flip", "90 Degree Rotation", 
                                       "180 Degree Rotation", "Horizontal Flip + 90 Degree Rotation", "Horizontal Flip + 180 Degree Rotation"], {"default": "None"}),
            },
        }
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("select_img", "exclude_img")
    FUNCTION = "SelectImages"



    def parse_indexes(self, indexes_str, max_length):
        """解析索引字符串，支持逗号分隔和范围格式"""
        indexes = []
        # 分割逗号处理多个表达式
        parts = [p.strip() for p in indexes_str.split(',') if p.strip()]
        
        for part in parts:
            # 处理范围格式（如 2-5）
            if '-' in part:
                try:
                    start, end = part.split('-', 1)
                    start = int(start.strip())
                    end = int(end.strip())
                    # 确保范围合法性
                    if start > end:
                        start, end = end, start  # 自动纠正倒序范围
                    # 限制在有效范围内
                    start = max(1, min(start, max_length))
                    end = max(1, min(end, max_length))
                    # 添加范围内所有数字
                    indexes.extend(range(start, end + 1))
                except ValueError:
                    continue  # 格式错误则跳过该部分
            # 处理单个数字格式
            else:
                try:
                    num = int(part)
                    if 1 <= num <= max_length:
                        indexes.append(num)
                except ValueError:
                    continue  # 格式错误则跳过该数字
        
        # 去重并保持顺序
        seen = set()
        unique_indexes = []
        for num in indexes:
            if num not in seen:
                seen.add(num)
                unique_indexes.append(num)
        return unique_indexes

    def SelectImages(self, images, indexes, canvas_operations):
        max_length = len(images)
        if max_length == 0:
            return (images, [])
        
        # 解析索引字符串
        select_numbers = self.parse_indexes(indexes, max_length)
        if not select_numbers:  # 若解析结果为空则返回原输入
            print("Warning: No valid indexes found, return original input.")
            return (images, [])
        
        # 转换为0基索引
        select_list1 = np.array(select_numbers) - 1
        
        # 计算排除列表
        exclude_list = np.setdiff1d(np.arange(max_length), select_list1)
        
        print(f"Selected images (1-based): {select_numbers}")

        # 选择图像
        selected_images = images[torch.tensor(select_list1, dtype=torch.long)]
        excluded_images = images[torch.tensor(exclude_list, dtype=torch.long)] if len(exclude_list) > 0 else []

        # 应用画布操作
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




class Image_batch_composite:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bg_image": ("IMAGE",),
                "batch_image": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
                "Invert": ("BOOLEAN", {"default": False}),

            },
            "optional": {
                "batch_mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite"
    CATEGORY = "Apt_Preset/image/😺backup"

    def composite(self, batch_image, bg_image, x, y, resize_source, Invert, batch_mask = None):
        if Invert and batch_mask is not None:
            batch_mask = 1 - batch_mask
        batch_image = batch_image.clone().movedim(-1, 1)


        output = composite(batch_image, bg_image.movedim(-1, 1), x, y, batch_mask, 1, resize_source).movedim(1, -1)
        return (output,)


#endregion--------batch-------------------------




class Image_Upscaletile:

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"image": ("IMAGE",),
                     "upscale_model": (folder_paths.get_filename_list("upscale_models"), ),
                     "mode": (["rescale", "resize"],),
                     "rescale_factor": ("FLOAT", {"default": 2, "min": 0.01, "max": 16.0, "step": 0.01}),
                     "resize_long_side": ("INT", {"default": 1024, "min": 1, "max": 48000, "step": 1}),
                     "resampling_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),                    
                     "rounding_modulus": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 2}),
                     }
                }

    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    CATEGORY = "Apt_Preset/image"
    
    
    @classmethod
    def apply_resize_image(cls, image: Image.Image, original_width, original_height, rounding_modulus, mode='scale', factor: int = 2, resize_long_side: int = 1024, resample='bicubic'): 
        if mode == 'rescale':
            new_width, new_height = int(original_width * factor), int(original_height * factor)               
        else:
            m = rounding_modulus
            # 确定原始图像的长边和短边
            is_width_longer = original_width > original_height
            
            if is_width_longer:
                # 宽度是长边，按宽度缩放
                scale_ratio = resize_long_side / original_width
                new_width = resize_long_side
                new_height = int(original_height * scale_ratio)
            else:
                # 高度是长边，按高度缩放
                scale_ratio = resize_long_side / original_height
                new_height = resize_long_side
                new_width = int(original_width * scale_ratio)
            
            # 应用 rounding modulus
            new_width = new_width if new_width % m == 0 else new_width + (m - new_width % m)
            new_height = new_height if new_height % m == 0 else new_height + (m - new_height % m)

        resample_filters = {'nearest': 0, 'bilinear': 2, 'bicubic': 3, 'lanczos': 1}
        image = image.resize((new_width * 8, new_height * 8), resample=Image.Resampling(resample_filters[resample]))
        resized_image = image.resize((new_width, new_height), resample=Image.Resampling(resample_filters[resample]))
        
        return resized_image  

    def upscale(self, image, upscale_model, rounding_modulus=8, mode="rescale",  resampling_method="bilinear", rescale_factor=2, resize_long_side=1024):
        up_model = load_upscale_model(upscale_model)  
        up_image = upscale_with_model(up_model, image)  

        # 获取原始图像尺寸
        original_width, original_height = 0, 0
        for img in image:
            pil_img = tensor2pil(img)
            original_width, original_height = pil_img.size
            break  # 只需要第一张图像的尺寸

        scaled_images = []
        for img in up_image:
            scaled_images.append(pil2tensor(self.apply_resize_image(
                tensor2pil(img), 
                original_width, 
                original_height, 
                rounding_modulus, 
                mode, 
                rescale_factor, 
                resize_long_side, 
                resampling_method
            )))
        
        images_out = torch.cat(scaled_images, dim=0)
        return (images_out,)
    



class Image_pad_keep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "y": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.001, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0, "step": 1, "min": -360, "max": 360}),
                "shear": ("FLOAT", {"default": 0, "step": 1, "min": -4096, "max": 4096}),
                "border_handling": (["edge", "constant", "reflect", "symmetric"], {"default": "edge"}),
            },
            "optional": {
                "scale_mode":(["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "constant_color": (["white", "black", "red", "green", "blue", "gray"], {"default": "black"}),
            },
        }

    FUNCTION = "transform"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "pad_mask")
    CATEGORY = "Apt_Preset/image"

    def transform(
        self,
        image: torch.Tensor,
        x: float,
        y: float,
        zoom: float,
        angle: float,
        shear: float,
        border_handling="edge",
        constant_color=None,
        scale_mode="bilinear",
    ):
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "gray": (128, 128, 128)
        }
        
        filter_map = {
            "nearest": Image.NEAREST,
            "box": Image.BOX,
            "bilinear": Image.BILINEAR,
            "hamming": Image.HAMMING,
            "bicubic": Image.BICUBIC,
            "lanczos": Image.LANCZOS,
        }
        resampling_filter = filter_map[scale_mode]

        x = int(x)
        y = int(y)
        angle = int(angle)

        if image.size(0) == 0:
            return (torch.zeros(0), torch.zeros(0))
        
        transformed_images = []
        padding_masks = []
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        diagonal = sqrt(frame_width**2 + frame_height**2)
        max_padding = ceil(diagonal * zoom - min(frame_width, frame_height))
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [
            max(0, pw + x),
            max(0, ph + y),
            max(0, pw - x),
            max(0, ph - y),
        ]

        constant_color = color_map.get(constant_color, (0, 0, 0))

        for img in list_tensor2pil(image):
            # 1. 处理图像：填充→仿射变换→裁剪
            padded_img = TF.pad(img, padding=padding, padding_mode=border_handling, fill=constant_color or 0)
            affine_img = cast(Image.Image, TF.affine(
                padded_img,
                angle=angle,
                scale=zoom,
                translate=[x, y],
                shear=shear,
                interpolation=resampling_filter,
            ))
            left, upper = abs(padding[0]), abs(padding[1])
            right, bottom = affine_img.width - abs(padding[2]), affine_img.height - abs(padding[3])
            final_img = affine_img.crop((left, upper, right, bottom))
            transformed_images.append(final_img)

            # 2. 生成填充遮罩：填充区域(255)，原图像区域(0)
            # 步骤1：创建原始填充区域的遮罩（未仿射前）
            mask_padded = Image.new("L", padded_img.size, 255)
            draw = ImageDraw.Draw(mask_padded)
            # 原图像在填充后的位置（黑色0区域）
            orig_x, orig_y = padding[0], padding[1]
            orig_w, orig_h = img.size
            draw.rectangle([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h], fill=0)
            
            # 步骤2：对遮罩应用与图像相同的仿射变换
            mask_affine = cast(Image.Image, TF.affine(
                mask_padded,
                angle=angle,
                scale=zoom,
                translate=[x, y],
                shear=shear,
                interpolation=Image.NEAREST,
            ))
            
            # 步骤3：对遮罩裁剪到最终尺寸
            final_mask = mask_affine.crop((left, upper, right, bottom))
            padding_masks.append(final_mask)

        # 转换为tensor并返回（图像+遮罩）
        return (
            list_pil2tensor(transformed_images),
            list_pil2tensor(padding_masks).squeeze(1)  # MASK格式为 [B, H, W]，移除通道维
        )




#region --------------------color--------------------


class color_balance_adv:
    def __init__(self):
        self.NODE_NAME = 'ColorBalance'

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "image": ("IMAGE", ),
                "cyan_red": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "magenta_green": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001}),
                "yellow_blue": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.001})
            },
            "optional": {
                "mask": ("MASK",),
                "lock_color": ("BOOLEAN", {"default": False}),
                "lock_color_hex": ("STRING", {"default": "#000000"}),
                "lock_color_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "lock_color_smoothness": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "mask_smoothness": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'color_balance'
    CATEGORY = "Apt_Preset/image/😺backup"

    def RGB2RGBA(self, image, mask):
        (R, G, B) = image.convert('RGB').split()
        return Image.merge('RGBA', (R, G, B, mask.convert('L')))

    def apply_color_balance(self, image, shadows, midtones, highlights,
                            shadow_center=0.15, midtone_center=0.5, highlight_center=0.8,
                            shadow_max=0.1, midtone_max=0.3, highlight_max=0.2,
                            preserve_luminosity=False):
        img = pil2tensor(image)
        img_copy = img.clone()
        if preserve_luminosity:
            original_luminance = 0.2126 * img_copy[..., 0] + 0.7152 * img_copy[..., 1] + 0.0722 * img_copy[..., 2]
        def adjust(x, center, value, max_adjustment):
            value = value * max_adjustment
            points = torch.tensor([[0, 0], [center, center + value], [1, 1]])
            cs = CubicSpline(points[:, 0], points[:, 1])
            return torch.clamp(torch.from_numpy(cs(x)), 0, 1)
        for i, (s, m, h) in enumerate(zip(shadows, midtones, highlights)):
            img_copy[..., i] = adjust(img_copy[..., i], shadow_center, s, shadow_max)
            img_copy[..., i] = adjust(img_copy[..., i], midtone_center, m, midtone_max)
            img_copy[..., i] = adjust(img_copy[..., i], highlight_center, h, highlight_max)
        if preserve_luminosity:
            current_luminance = 0.2126 * img_copy[..., 0] + 0.7152 * img_copy[..., 1] + 0.0722 * img_copy[..., 2]
            img_copy *= (original_luminance / current_luminance).unsqueeze(-1)
        return tensor2pil(img_copy)

    def color_balance(self, image, cyan_red, magenta_green, yellow_blue, mask=None, lock_color=False, lock_color_hex="#000000", lock_color_threshold=10, mask_smoothness=0, lock_color_smoothness=0):
        l_images = []
        l_masks = []
        ret_images = []
        
        for l in image:
            l_images.append(torch.unsqueeze(l, 0))
            m = tensor2pil(l)
            if m.mode == 'RGBA':
                l_masks.append(m.split()[-1])
            else:
                l_masks.append(Image.new('L', m.size, 'white'))
        
        for i in range(len(l_images)):
            _image = l_images[i]
            _mask = l_masks[i]
            orig_image = tensor2pil(_image)
            
            if mask is not None:
                mask_pil = tensor2pil(mask[i] if mask.dim() > 3 else mask)
                mask_pil = mask_pil.convert('L')
                
                if mask_pil.size != orig_image.size:
                    mask_pil = mask_pil.resize(orig_image.size, Image.BILINEAR)
                
                if mask_smoothness > 0:
                    mask_pil = tensor2pil(smoothness_mask(pil2tensor(mask_pil), mask_smoothness))
                
                if lock_color:
                    lock_color_hex = lock_color_hex.lstrip('#')
                    target_color = tuple(int(lock_color_hex[i:i+2], 16) for i in (0, 2, 4))
                    
                    orig_array = np.array(orig_image.convert('RGB'))
                    target_array = np.array(target_color)
                    
                    diff = np.sqrt(np.sum((orig_array - target_array) ** 2, axis=2))
                    
                    lock_mask = (diff <= lock_color_threshold).astype(np.uint8) * 255
                    lock_mask_pil = Image.fromarray(lock_mask, mode='L')
                    
                    if lock_color_smoothness > 0:
                        lock_mask_pil = tensor2pil(smoothness_mask(pil2tensor(lock_mask_pil), lock_color_smoothness))
                    
                    combined_mask = ImageChops.multiply(mask_pil, lock_mask_pil)
                else:
                    combined_mask = mask_pil
                
                masked_image = Image.composite(orig_image, Image.new('RGB', orig_image.size, (0, 0, 0)), combined_mask)
                
                ret_image = self.apply_color_balance(masked_image,
                                                     [cyan_red, magenta_green, yellow_blue],
                                                     [cyan_red, magenta_green, yellow_blue],
                                                     [cyan_red, magenta_green, yellow_blue],
                                                     shadow_center=0.15,
                                                     midtone_center=0.5,
                                                     midtone_max=1,
                                                     preserve_luminosity=True)
                
                ret_image = Image.composite(ret_image, orig_image, combined_mask)
            else:
                if lock_color:
                    lock_color_hex = lock_color_hex.lstrip('#')
                    target_color = tuple(int(lock_color_hex[i:i+2], 16) for i in (0, 2, 4))
                    
                    orig_array = np.array(orig_image.convert('RGB'))
                    target_array = np.array(target_color)
                    
                    diff = np.sqrt(np.sum((orig_array - target_array) ** 2, axis=2))
                    
                    lock_mask = (diff <= lock_color_threshold).astype(np.uint8) * 255
                    lock_mask_pil = Image.fromarray(lock_mask, mode='L')
                    
                    if lock_color_smoothness > 0:
                        lock_mask_pil = tensor2pil(smoothness_mask(pil2tensor(lock_mask_pil), lock_color_smoothness))
                    
                    masked_image = Image.composite(orig_image, Image.new('RGB', orig_image.size, (0, 0, 0)), lock_mask_pil)
                    
                    ret_image = self.apply_color_balance(masked_image,
                                                         [cyan_red, magenta_green, yellow_blue],
                                                         [cyan_red, magenta_green, yellow_blue],
                                                         [cyan_red, magenta_green, yellow_blue],
                                                         shadow_center=0.15,
                                                         midtone_center=0.5,
                                                         midtone_max=1,
                                                         preserve_luminosity=True)
                    
                    ret_image = Image.composite(ret_image, orig_image, lock_mask_pil)
                else:
                    ret_image = self.apply_color_balance(orig_image,
                                                         [cyan_red, magenta_green, yellow_blue],
                                                         [cyan_red, magenta_green, yellow_blue],
                                                         [cyan_red, magenta_green, yellow_blue],
                                                         shadow_center=0.15,
                                                         midtone_center=0.5,
                                                         midtone_max=1,
                                                         preserve_luminosity=True)
            
            if orig_image.mode == 'RGBA':
                ret_image = self.RGB2RGBA(ret_image, orig_image.split()[-1])
            
            ret_images.append(pil2tensor(ret_image))
        
        return (torch.cat(ret_images, dim=0),)



class texture_apply:  #法向光源图

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_img": ("IMAGE",),
                "normal_map": ("IMAGE",),
                "specular_map": ("IMAGE",),
                "light_yaw": ("FLOAT", {"default": 45, "min": -180, "max": 180, "step": 1}),
                "light_pitch": ("FLOAT", {"default": 30, "min": -90, "max": 90, "step": 1}),
                "specular_power": ("FLOAT", {"default": 32, "min": 1, "max": 200, "step": 1}),
                "ambient_light": ("FLOAT", {"default": 0.50, "min": 0, "max": 1, "step": 0.01}),
                "NormalDiffuseStrength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
                "SpecularHighlightsStrength": ("FLOAT", {"default": 1.00, "min": 0, "max": 5.0, "step": 0.01}),
                "TotalGain": ("FLOAT", {"default": 1.00, "min": 0, "max": 2.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/imgEffect"

    def execute(self, target_img, normal_map, specular_map, light_yaw, light_pitch, specular_power, ambient_light,NormalDiffuseStrength,SpecularHighlightsStrength,TotalGain):

        diffuse_tensor = target_img.permute(0, 3, 1, 2)  
        normal_tensor = normal_map.permute(0, 3, 1, 2) * 2.0 - 1.0  
        specular_tensor = specular_map.permute(0, 3, 1, 2)  

        normal_tensor = torch.nn.functional.normalize(normal_tensor, dim=1)
        light_direction = self.euler_to_vector(light_yaw, light_pitch, 0 )
        light_direction = light_direction.view(1, 3, 1, 1)  
        camera_direction = self.euler_to_vector(0,0,0)
        camera_direction = camera_direction.view(1, 3, 1, 1) 


        diffuse = torch.sum(normal_tensor * light_direction, dim=1, keepdim=True)
        diffuse = torch.clamp(diffuse, 0, 1)

        half_vector = torch.nn.functional.normalize(light_direction + camera_direction, dim=1)
        specular = torch.sum(normal_tensor * half_vector, dim=1, keepdim=True)
        specular = torch.pow(torch.clamp(specular, 0, 1), specular_power)

        output_tensor = ( diffuse_tensor * (ambient_light + diffuse * NormalDiffuseStrength ) + specular_tensor * specular * SpecularHighlightsStrength) * TotalGain

        output_tensor = output_tensor.permute(0, 2, 3, 1)  

        return (output_tensor,)


    def euler_to_vector(self, yaw, pitch, roll):
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)

        cos_pitch = np.cos(pitch_rad)
        sin_pitch = np.sin(pitch_rad)
        cos_yaw = np.cos(yaw_rad)
        sin_yaw = np.sin(yaw_rad)
        cos_roll = np.cos(roll_rad)
        sin_roll = np.sin(roll_rad)

        direction = np.array([
            sin_yaw * cos_pitch,
            sin_pitch,
            cos_pitch * cos_yaw
        ])


        return torch.from_numpy(direction).float()

    def convert_tensor_to_image(self, tensor):
        tensor = tensor.squeeze(0)  
        tensor = tensor.clamp(0, 1)  
        image = Image.fromarray((tensor.detach().cpu().numpy() * 255).astype(np.uint8))
        return image




class color_adjust_HDR:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {
                }),
                "HDR_intensity": ("FLOAT", {
                    "default": 1,
                    "min": 0.5,
                    "max": 3.0,
                    "step": 0.01,
                }),
                "underexposure_factor": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "overexposure_factor": ("FLOAT", {
                    "default": 1,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.01,
                }),
                "gamma": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 3.0,
                    "step": 0.01,
                }),
                "highlight_detail": ("FLOAT", {
                    "default": 1/30.0,
                    "min": 1/1000.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "midtone_detail": ("FLOAT", {
                    "default": 0.25,
                    "min": 1/1000.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "shadow_detail": ("FLOAT", {
                    "default": 2,
                    "min": 1/1000.0,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "overall_intensity": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/image"

    def execute(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, highlight_detail, midtone_detail, shadow_detail, overall_intensity):
        try:
            image = self.ensure_image_format(image)

            processed_image = self.apply_hdr(image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, [highlight_detail, midtone_detail, shadow_detail])

            blended_image = cv2.addWeighted(processed_image, overall_intensity, image, 1 - overall_intensity, 0)

            if isinstance(blended_image, np.ndarray):
                blended_image = np.expand_dims(blended_image, axis=0)

            blended_image = torch.from_numpy(blended_image).float()
            blended_image = blended_image / 255.0
            blended_image = blended_image.to(torch.device('cpu'))

            return [blended_image]
        except Exception as e:
            if image is not None and hasattr(image, 'shape'):
                black_image = torch.zeros((1, 3, image.shape[0], image.shape[1]), dtype=torch.float32)
            else:
                black_image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
            return [black_image.to(torch.device('cpu'))]

    def ensure_image_format(self, image):
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image = image.squeeze(0)
            image = image.numpy() * 255
            image = image.astype(np.uint8)
        return image

    def apply_hdr(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, exposure_times):
        hdr = cv2.createMergeDebevec()

        times = np.array(exposure_times, dtype=np.float32)

        exposure_images = [
            np.clip(image * underexposure_factor, 0, 255).astype(np.uint8),  # Underexposed
            image,  # Normal exposure
            np.clip(image * overexposure_factor, 0, 255).astype(np.uint8)   # Overexposed
        ]

        hdr_image = hdr.process(exposure_images, times=times.copy())

        tonemap = cv2.createTonemapReinhard(gamma=gamma)
        ldr_image = tonemap.process(hdr_image)

        ldr_image = ldr_image * HDR_intensity
        ldr_image = np.clip(ldr_image, 0, 1)
        ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

        return ldr_image


class color_adjust_HSL:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hue": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 2.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),

                "sharpness": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "gaussian_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.1}),
                "edge_enhance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "detail_enhance": (["false", "true"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_adjust_HSL"
    CATEGORY = "Apt_Preset/image"

    def color_adjust_HSL(self, image, brightness, contrast, saturation, hue, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance):
        tensors = []
        if len(image) > 1:
            for img in image:
                pil_image = None
                if brightness > 0.0 or brightness < 0.0:
                    img = np.clip(img + brightness, 0.0, 1.0)
                if contrast > 1.0 or contrast < 1.0:
                    img = np.clip(img * contrast, 0.0, 1.0)
                if saturation > 1.0 or saturation < 1.0 or hue != 0.0:
                    pil_image = tensor2pil(img)
                    if saturation != 1.0:
                        pil_image = ImageEnhance.Color(pil_image).enhance(saturation)
                    if hue != 0.0:
                        pil_image = self.adjust_hue(pil_image, hue)
                if sharpness > 1.0 or sharpness < 1.0:
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)
                if blur > 0:
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    for _ in range(blur):
                        pil_image = pil_image.filter(ImageFilter.BLUR)
                if gaussian_blur > 0.0:
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))
                if edge_enhance > 0.0:
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                    blend_mask = Image.new(mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                    pil_image = Image.composite(edge_enhanced_img, pil_image, blend_mask)
                    del blend_mask, edge_enhanced_img
                if detail_enhance == "true":
                    pil_image = pil_image if pil_image else tensor2pil(img)
                    pil_image = pil_image.filter(ImageFilter.DETAIL)
                out_image = (pil2tensor(pil_image) if pil_image else img)
                tensors.append(out_image)
            tensors = torch.cat(tensors, dim=0)
        else:
            pil_image = None
            img = image
            if brightness > 0.0 or brightness < 0.0:
                img = np.clip(img + brightness, 0.0, 1.0)
            if contrast > 1.0 or contrast < 1.0:
                img = np.clip(img * contrast, 0.0, 1.0)
            if saturation > 1.0 or saturation < 1.0 or hue != 0.0:
                pil_image = tensor2pil(img)
                if saturation != 1.0:
                    pil_image = ImageEnhance.Color(pil_image).enhance(saturation)
                if hue != 0.0:
                    pil_image = self.adjust_hue(pil_image, hue)
            if sharpness > 1.0 or sharpness < 1.0:
                pil_image = pil_image if pil_image else tensor2pil(img)
                pil_image = ImageEnhance.Sharpness(pil_image).enhance(sharpness)
            if blur > 0:
                pil_image = pil_image if pil_image else tensor2pil(img)
                for _ in range(blur):
                    pil_image = pil_image.filter(ImageFilter.BLUR)
            if gaussian_blur > 0.0:
                pil_image = pil_image if pil_image else tensor2pil(img)
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=gaussian_blur))
            if edge_enhance > 0.0:
                pil_image = pil_image if pil_image else tensor2pil(img)
                edge_enhanced_img = pil_image.filter(ImageFilter.EDGE_ENHANCE_MORE)
                blend_mask = Image.new(mode="L", size=pil_image.size, color=(round(edge_enhance * 255)))
                pil_image = Image.composite(edge_enhanced_img, pil_image, blend_mask)
                del blend_mask, edge_enhanced_img
            if detail_enhance == "true":
                pil_image = pil_image if pil_image else tensor2pil(img)
                pil_image = pil_image.filter(ImageFilter.DETAIL)
            out_image = (pil2tensor(pil_image) if pil_image else img)
            tensors = out_image
        return (tensors, )

    def adjust_hue(self, image, hue_shift):
        if hue_shift == 0:
            return image
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()
        h = h.point(lambda x: (x + int(hue_shift * 255)) % 256)
        hsv_image = Image.merge('HSV', (h, s, v))
        return hsv_image.convert('RGB')




class color_tool:
    """Returns to inverse of a color"""

    def __init__(self) -> None:
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "color": ("STRING", {"default": "#000000"}),  # 改为STRING类型，默认黑色
            },
            "optional": {
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hex_str": ("STRING",  {"forceInput": True}, ),
                "r": ("INT", {"forceInput": True, "min": 0, "max": 255}, ),
                "g": ("INT", {"forceInput": True, "min": 0, "max": 255}, ),
                "b": ("INT", {"forceInput": True, "min": 0, "max": 255}, ),
                "a": ("FLOAT", {"forceInput": True, "min": 0.0, "max": 1.0,} ,),
            }
        }

    CATEGORY = "Apt_Preset/image/😺backup"
    # 修改返回类型，添加 alpha 通道
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "FLOAT",)
    # 修改返回名称，添加 alpha 通道
    RETURN_NAMES = ("hex_str", "R", "G", "B", "A",)

    FUNCTION = "execute"

    def execute(self, color, alpha, hex_str=None, r=None, g=None, b=None, a=None):
        if hex_str:
            hex_color = hex_str
        else:
            hex_color = color

        hex_color = hex_color.lstrip("#")
        original_r, original_g, original_b = hex_to_rgb_tuple(hex_color)

        # 若有 r, g, b, a 输入则替换对应值
        final_r = r if r is not None else original_r
        final_g = g if g is not None else original_g
        final_b = b if b is not None else original_b
        final_a = a if a is not None else alpha

        final_hex_color = "#{:02x}{:02x}{:02x}".format(final_r, final_g, final_b)
        return (final_hex_color, final_r, final_g, final_b, final_a)
    

class color_OneColor_replace:
    """Replace Color in an Image"""
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                "target_color": ("STRING", {"default": "#09FF00"}),
                "replace_color": ("STRING", {"default": "#FF0000"}),
                "clip_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_remove_color"

    CATEGORY = "Apt_Preset/image/😺backup"

    def image_remove_color(self, image, clip_threshold=10, target_color='#ffffff',replace_color='#ffffff'):
        return (pil2tensor(self.apply_remove_color(tensor2pil(image), clip_threshold, hex_to_rgb_tuple(target_color), hex_to_rgb_tuple(replace_color))), )

    def apply_remove_color(self, image, threshold=10, color=(255, 255, 255), rep_color=(0, 0, 0)):
        # Create a color image with the same size as the input image
        color_image = Image.new('RGB', image.size, color)

        # Calculate the difference between the input image and the color image
        diff_image = ImageChops.difference(image, color_image)

        # Convert the difference image to grayscale
        gray_image = diff_image.convert('L')

        # Apply a threshold to the grayscale difference image
        mask_image = gray_image.point(lambda x: 255 if x > threshold else 0)

        # Invert the mask image
        mask_image = ImageOps.invert(mask_image)

        # Apply the mask to the original image
        result_image = Image.composite(
            Image.new('RGB', image.size, rep_color), image, mask_image)

        return result_image


class color_OneColor_keep:  #保留一色
    NAME = "Color Stylizer"
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "image": ("IMAGE",),
                # Modified to directly input color
                "color": ("STRING", {"default": "#000000"}),
                "falloff": ("FLOAT", {
                    "default": 30.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "display": "number"
                }),
                "gain": ("FLOAT", {
                    "default": 1.5,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.5,
                    "display": "number"
                })
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "stylize"
    OUTPUT_NODE = False
    CATEGORY = "Apt_Preset/image/😺backup"

    def stylize(self, image, color, falloff, gain):
        print(f"Type of color: {type(color)}, Value of color: {color}")  # Add debugging information
        try:
            # Extract BGR values from the color tuple
            target_b, target_g, target_r = [int(c * 255) if isinstance(c, (int, float)) else 0 for c in color]
            target_color = (target_b, target_g, target_r)
        except Exception as e:
            print(f"Error converting color: {e}")
            target_color = (0, 0, 0)  # Set default color if conversion fails

        image = image.squeeze(0)
        image = image.mul(255).byte().numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
        falloff_mask = self.create_falloff_mask(image, target_color, falloff)
        image_amplified = image.copy()
        image_amplified[:, :, 2] = np.clip(image_amplified[:, :, 2] * gain, 0, 255).astype(np.uint8)
        stylized_img = (image_amplified * falloff_mask + gray_img * (1 - falloff_mask)).astype(np.uint8)
        stylized_img = cv2.cvtColor(stylized_img, cv2.COLOR_BGR2RGB)
        stylized_img_tensor = to_tensor(stylized_img).float()
        stylized_img_tensor = stylized_img_tensor.permute(1, 2, 0).unsqueeze(0)
        return (stylized_img_tensor,)

    def create_falloff_mask(self, img, target_color, falloff):
        target_color = np.array(target_color, dtype=np.uint8)

        target_color = np.full_like(img, target_color)

        diff = cv2.absdiff(img, target_color)

        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(diff, falloff, 255, cv2.THRESH_BINARY_INV)
        mask = cv2.GaussianBlur(mask, (0, 0), falloff / 2)
        mask = mask / 255.0
        mask = mask.reshape(*mask.shape, 1)
        return mask



class color_selector:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {
                "color": ("COLOR", {"default": "#FFFFFF"},),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("HEX",)
    FUNCTION = 'selector'
    CATEGORY = "Apt_Preset/image"
    
    def selector(self, color):
        return (color,)


#endregion --------------------color--------------------




#region----color_match adv----------


def image_stats(image):
    return np.mean(image[:, :, 1:], axis=(0, 1)), np.std(image[:, :, 1:], axis=(0, 1))


def is_skin_or_lips(lab_image):
    l, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]
    skin = (l > 20) & (l < 250) & (a > 120) & (a < 180) & (b > 120) & (b < 190)
    lips = (l > 20) & (l < 200) & (a > 150) & (b > 140)
    return (skin | lips).astype(np.float32)


def adjust_brightness(image, factor, mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        v = np.where(mask > 0, np.clip(v * factor, 0, 255), v)
    else:
        v = np.clip(v * factor, 0, 255)
    hsv[:, :, 2] = v.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_saturation(image, factor, mask=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        s = np.where(mask > 0, np.clip(s * factor, 0, 255), s)
    else:
        s = np.clip(s * factor, 0, 255)
    hsv[:, :, 1] = s.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor, mask=None):
    mean = np.mean(image)
    adjusted = image.astype(np.float32)
    if mask is not None:
        mask = mask.squeeze()
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        adjusted = np.where(mask > 0, np.clip((adjusted - mean) * factor + mean, 0, 255), adjusted)
    else:
        adjusted = np.clip((adjusted - mean) * factor + mean, 0, 255)
    return adjusted.astype(np.uint8)


def adjust_tone(source, target, tone_strength=0.7, mask=None):
    h, w = target.shape[:2]
    source = cv2.resize(source, (w, h))
    lab_image = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_image = lab_image[:,:,0]
    l_source = lab_source[:,:,0]

    if mask is not None:
        mask = cv2.resize(mask, (w, h))
        mask = mask.astype(np.float32) / 255.0
        l_adjusted = np.copy(l_image)
        mean_source = np.mean(l_source[mask > 0])
        std_source = np.std(l_source[mask > 0])
        mean_target = np.mean(l_image[mask > 0])
        std_target = np.std(l_image[mask > 0])
        l_adjusted[mask > 0] = (l_image[mask > 0] - mean_target) * (std_source / (std_target + 1e-6)) * 0.7 + mean_source
        l_adjusted[mask > 0] = np.clip(l_adjusted[mask > 0], 0, 255)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_adjusted.astype(np.uint8))
        l_final = cv2.addWeighted(l_adjusted, 0.7, l_enhanced.astype(np.float32), 0.3, 0)
        l_final = np.clip(l_final, 0, 255)
        l_contrast = cv2.addWeighted(l_final, 1.3, l_final, 0, -20)
        l_contrast = np.clip(l_contrast, 0, 255)
        l_image[mask > 0] = l_image[mask > 0] * (1 - tone_strength) + l_contrast[mask > 0] * tone_strength
    else:
        mean_source = np.mean(l_source)
        std_source = np.std(l_source)
        l_mean = np.mean(l_image)
        l_std = np.std(l_image)
        l_adjusted = (l_image - l_mean) * (std_source / (l_std + 1e-6)) * 0.7 + mean_source
        l_adjusted = np.clip(l_adjusted, 0, 255)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
        l_enhanced = clahe.apply(l_adjusted.astype(np.uint8))
        l_final = cv2.addWeighted(l_adjusted, 0.7, l_enhanced.astype(np.float32), 0.3, 0)
        l_final = np.clip(l_final, 0, 255)
        l_contrast = cv2.addWeighted(l_final, 1.3, l_final, 0, -20)
        l_contrast = np.clip(l_contrast, 0, 255)
        l_image = l_image * (1 - tone_strength) + l_contrast * tone_strength

    lab_image[:,:,0] = l_image
    return cv2.cvtColor(lab_image.astype(np.uint8), cv2.COLOR_LAB2BGR)


def tensor2cv2(image: torch.Tensor) -> np.array:
    if image.dim() == 4:
        image = image.squeeze()
    npimage = image.numpy()
    cv2image = np.uint8(npimage * 255 / npimage.max())
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)


def color_transfer(source, target, mask=None, strength=1.0, skin_protection=0.2, auto_brightness=True,
                   brightness_range=0.5, auto_contrast=False, contrast_range=0.5,
                   auto_saturation=False, saturation_range=0.5, auto_tone=False, tone_strength=0.7):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_means, src_stds = image_stats(source_lab)
    tar_means, tar_stds = image_stats(target_lab)

    skin_lips_mask = is_skin_or_lips(target_lab.astype(np.uint8))
    skin_lips_mask = cv2.GaussianBlur(skin_lips_mask, (5, 5), 0)

    if mask is not None:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        mask = mask.astype(np.float32) / 255.0

    result_lab = target_lab.copy()
    for i in range(1, 3):
        adjusted_channel = (target_lab[:, :, i] - tar_means[i - 1]) * (src_stds[i - 1] / (tar_stds[i - 1] + 1e-6)) + \
                           src_means[i - 1]
        adjusted_channel = np.clip(adjusted_channel, 0, 255)

        if mask is not None:
            result_lab[:, :, i] = target_lab[:, :, i] * (1 - mask) + \
                                  (target_lab[:, :, i] * skin_lips_mask * skin_protection + \
                                   adjusted_channel * skin_lips_mask * (1 - skin_protection) + \
                                   adjusted_channel * (1 - skin_lips_mask)) * mask
        else:
            result_lab[:, :, i] = target_lab[:, :, i] * skin_lips_mask * skin_protection + \
                                  adjusted_channel * skin_lips_mask * (1 - skin_protection) + \
                                  adjusted_channel * (1 - skin_lips_mask)

    result_bgr = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    final_result = cv2.addWeighted(target, 1 - strength, result_bgr, strength, 0)

    if mask is not None:
        mask = cv2.resize(mask, (target.shape[1], target.shape[0]))
        mask = mask.astype(np.float32) / 255.0
        if auto_brightness:
            source_brightness = np.mean(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
            target_brightness = np.mean(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, brightness_range*-1, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor, mask)
        if auto_contrast:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, contrast_range*-1, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor, mask)
        if auto_saturation:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, saturation_range*-1, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor, mask)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength, mask)
    else:
        if auto_brightness:
            source_brightness = np.mean(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY))
            target_brightness = np.mean(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, brightness_range*-1, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor)
        if auto_contrast:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, contrast_range*-1, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor)
        if auto_saturation:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, saturation_range*-1, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength)

    return final_result


class color_match_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "target_image": ("IMAGE",),
                "ref_img": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "skin_protection": ("FLOAT", {"default": 0.2, "min": 0, "max": 1.0, "step": 0.1}),
                "brightness_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "contrast_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "saturation_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "tone_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "ref_mask": ("MASK", {"default": None}),
            },
        }

    CATEGORY = "Apt_Preset/image"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match_hue"

    def match_hue(self, ref_img, target_image, strength, skin_protection,  brightness_range,
                contrast_range, saturation_range, tone_strength, ref_mask=None):
        
        auto_brightness =True
        auto_contrast =True
        auto_tone =True
        auto_saturation =True


        for img in ref_img:
            img_cv1 = tensor2cv2(img)

        for img in target_image:
            img_cv2 = tensor2cv2(img)

        img_cv3 = None
        if ref_mask is not None:
            for img3 in ref_mask:
                img_cv3 = img3.cpu().numpy()
                img_cv3 = (img_cv3 * 255).astype(np.uint8)

        result_img = color_transfer(img_cv1, img_cv2, img_cv3, strength, skin_protection, auto_brightness,
                                    brightness_range,auto_contrast, contrast_range, auto_saturation,
                                    saturation_range, auto_tone, tone_strength)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        rst = torch.from_numpy(result_img.astype(np.float32) / 255.0).unsqueeze(0)

        return (rst,)





#endregion-----------------------color_transfer----------------





#region------图像-双图合并---总控制---------

class Image_Pair_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "box2": ("BOX2",),
                "crop_image": ("BOOLEAN", {"default": False,"label_on": "img2", "label_off": "img1"}),

            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("裁剪图像","裁剪遮罩")
    FUNCTION = "cropbox"

    def cropbox(self, mask=None, image=None, box2=None, crop_image=False):
        if box2 is None:
            return (None, None)
            
        box1_w, box1_h, box1_x, box1_y, box2_w, box2_h, box2_x, box2_y = box2
        
        if crop_image:
            region_width, region_height, center_x, center_y = box2_w, box2_h, box2_x, box2_y
        else:
            region_width, region_height, center_x, center_y = box1_w, box1_h, box1_x, box1_y
        
        x_start = max(0, int(center_x - region_width // 2))
        y_start = max(0, int(center_y - region_height // 2))
        
        x_end = x_start + region_width
        y_end = y_start + region_height
        
        cropped_image = None
        if image is not None:
            img_h, img_w = image.shape[1], image.shape[2]
            x_start = min(x_start, img_w)
            y_start = min(y_start, img_h)
            x_end = min(x_end, img_w)
            y_end = min(y_end, img_h)
            cropped_image = image[:, y_start:y_end, x_start:x_end, :]

        cropped_mask = None
        if mask is not None:
            mask_h, mask_w = mask.shape[1], mask.shape[2]
            x_start = min(x_start, mask_w)
            y_start = min(y_start, mask_h)
            x_end = min(x_end, mask_w)
            y_end = min(y_end, mask_h)
            cropped_mask = mask[:, y_start:y_end, x_start:x_end]

        return (cropped_image, cropped_mask,)



class Image_Pair_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layout_mode": ([
                                 "左右-中心对齐", "左右-高度对齐", "左右-宽度对齐",
                                 "上下-中心对齐", "上下-宽度对齐", "上下-高度对齐", 
                                 "居中-自动对齐","居中-中心对齐", "居中-高度对齐", "居中-宽度对齐"],), 
                "bg_mode": (["crop_image","image", "transparent", "white", "black", "red", "green", "blue"],),
                "size_mode": (["auto", "输出宽度", "输出高度"],),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "divider_thickness": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "image1": ("IMAGE",), 
                "mask1": ("MASK",),
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
                "mask1_stack": ("MASK_STACK2",),
                "mask2_stack": ("MASK_STACK2",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX2", "IMAGE", "MASK", )
    RETURN_NAMES = ("composite_image", "composite_mask", "box2", "new_img2", "new_mask2", )
    FUNCTION = "composite2"
    CATEGORY = "Apt_Preset/image"

    DESCRIPTION = """
    - 输入参数：
    - "crop_image": 使用裁切图像作为背景
    - "image": 使用原图填充背景
    - "transparent": 使用透明背景
    - "white": 使用白色背景
    - size_mode：决定最终输出图像的尺寸计算方式
    - target_size: 目标尺寸值，配合size_mode使用
    - divider_thickness: 分隔线厚度，用于左右或上下排列模式中的分隔线

    - -----------------------  
    - 重要逻辑：
    - image1和image2是同步处理的，根据layout_mode决定排列方式
    - 在居中叠加模式下，image2会叠加在image1上
    - 在左右/上下排列模式下，可以通过divider_thickness添加分隔线
    """



    def composite2(self, layout_mode, bg_mode, size_mode, target_size, divider_thickness, 
                image1=None, image2=None, mask1=None, mask2=None, mask1_stack=None, mask2_stack=None):

        # 处理 mask1
        if mask1_stack and mask1 is not None:
            if hasattr(mask1, 'convert'):
                mask1_tensor = pil2tensor(mask1.convert('L'))
            else:
                if isinstance(mask1, torch.Tensor):
                    mask1_tensor = mask1 if len(mask1.shape) <= 3 else mask1.squeeze(-1) if mask1.shape[-1] == 1 else mask1
                else:
                    mask1_tensor = mask1
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask1_stack            
            
            separated_result = Mask_transform_sum().separate(
                bg_mode=bg_mode, 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0, 
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image1.clone() if image1 is not None else image1,  # 使用clone避免修改原始图像
                mask=mask1_tensor, 
                crop_to_mask=False, 
                divisible_by=1
            )
            image1_processed = separated_result[0]  # 保存处理后的图像
            mask1 = separated_result[1]
        # 如果没有 mask1_stack 或 mask1 为 None，则直接使用 mask1（可能为 None）
        
        # 处理 mask2
        if mask2_stack and mask2 is not None: 
            if hasattr(mask2, 'convert'):
                mask2_tensor = pil2tensor(mask2.convert('L'))
            else:  
                if isinstance(mask2, torch.Tensor):
                    mask2_tensor = mask2 if len(mask2.shape) <= 3 else mask2.squeeze(-1) if mask2.shape[-1] == 1 else mask2
                else:
                    mask2_tensor = mask2
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask2_stack            
            
            separated_result = Mask_transform_sum().separate(  
                bg_mode=bg_mode, 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0, 
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image2.clone() if image2 is not None else image2,  # 使用clone避免修改原始图像
                mask=mask2_tensor, 
                crop_to_mask=False, 
                divisible_by=1
            )
            image2_processed = separated_result[0]  # 保存处理后的图像
            mask2 = separated_result[1]
    
        # 如果经过处理，使用处理后的图像；否则使用原始图像
        if 'image1_processed' in locals():
            image1 = image1_processed
        if 'image2_processed' in locals():
            image2 = image2_processed
        

        if image1 is not None and not isinstance(image1, torch.Tensor):
            if hasattr(image1, 'numpy'):
                image1 = torch.from_numpy(image1.numpy())
            else:
                image1 = torch.from_numpy(np.array(image1))
        
        if image2 is not None and not isinstance(image2, torch.Tensor):
            if hasattr(image2, 'numpy'):
                image2 = torch.from_numpy(image2.numpy())
            else:
                image2 = torch.from_numpy(np.array(image2))

        if image1 is not None:
            if len(image1.shape) == 3:
                image1 = image1.unsqueeze(0)
            if image1.shape[-1] != 3:
                raise ValueError(f"image1 应该是3通道图像，实际形状: {image1.shape}")
        
        if image2 is not None:
            if len(image2.shape) == 3:
                image2 = image2.unsqueeze(0)
            if image2.shape[-1] != 3:
                raise ValueError(f"image2 应该是3通道图像，实际形状: {image2.shape}")

        if image1 is None and image2 is not None:
            image1 = image2.clone()
            
        if image1 is None and image2 is None:
            image1 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image2 = torch.zeros((1, 512, 512, 3), dtype=torch.float32)

        if mask1 is None:
            mask1 = torch.zeros((image1.shape[0], image1.shape[1], image1.shape[2]), dtype=torch.float32)
        else:
            if not isinstance(mask1, torch.Tensor):
                mask1 = torch.from_numpy(np.array(mask1)) if hasattr(mask1, 'numpy') else torch.from_numpy(mask1)
            
            if len(mask1.shape) == 2:
                mask1 = mask1.unsqueeze(0)
            if mask1.shape[1:] != image1.shape[1:3]:
                mask1 = torch.nn.functional.interpolate(
                    mask1.unsqueeze(1) if len(mask1.shape) == 3 else mask1, 
                    size=(image1.shape[1], image1.shape[2]), 
                    mode='nearest'
                )
                if len(mask1.shape) == 4:
                    mask1 = mask1.squeeze(1)

        if mask2 is None:
            mask2 = torch.ones((image2.shape[0], image2.shape[1], image2.shape[2]), dtype=torch.float32)
        else:
            if not isinstance(mask2, torch.Tensor):
                mask2 = torch.from_numpy(np.array(mask2)) if hasattr(mask2, 'numpy') else torch.from_numpy(mask2)
            
            if len(mask2.shape) == 2:
                mask2 = mask2.unsqueeze(0)
            if mask2.shape[1:] != image2.shape[1:3]:
                mask2 = torch.nn.functional.interpolate(
                    mask2.unsqueeze(1) if len(mask2.shape) == 3 else mask2, 
                    size=(image2.shape[1], image2.shape[2]), 
                    mode='nearest'
                )
                if len(mask2.shape) == 4:
                    mask2 = mask2.squeeze(1)

        final_img_tensor, final_mask_tensor, box2, image2, mask2 = Pair_Merge().composite(
            layout_mode=layout_mode, bg_mode=bg_mode, size_mode=size_mode, 
            target_size=target_size, divider_thickness=divider_thickness, 
            image1=image1, image2=image2, mask1=mask1, mask2=mask2
        )

        return (final_img_tensor, final_mask_tensor, box2, image2, mask2, )



class Pair_Merge:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "layout_mode": ([
                                "左右-中心对齐", "左右-高度对齐", "左右-宽度对齐",
                                "上下-中心对齐", "上下-宽度对齐", "上下-高度对齐",
                                "居中-自动对齐","居中-中心对齐", "居中-高度对齐", "居中-宽度对齐"],),  
                "bg_mode": (BJ_MODE,),  
                "size_mode": (["auto", "输出宽度", "输出高度"],),
                "target_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 8}),
                "divider_thickness": ("INT", {"default": 0, "min": 0, "max": 20, "step": 1}),
            },
            "optional": {
                "image1": ("IMAGE",), 
                "mask1": ("MASK",),
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "BOX2", "IMAGE", "MASK")
    RETURN_NAMES = ("composite_image", "composite_mask", "box2", "new_img2", "new_mask2")
    FUNCTION = "composite"
    CATEGORY = "Apt_Preset/image"

    def composite(self, layout_mode, bg_mode, size_mode, target_size, divider_thickness, image1=None, image2=None, mask1=None, mask2=None):
        # 映射resize_mode到OpenCV的插值方法
        
        resize_mode="bicubic"
        interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "bilinear": cv2.INTER_LINEAR,
            "bicubic": cv2.INTER_CUBIC,
            "lanczos": cv2.INTER_LANCZOS4
        }
        interpolation = interpolation_map[resize_mode]
        
        composite_img = None
        composite_mask = None
        box1_w, box1_h, box1_x, box1_y = 0, 0, 0, 0
        box2_w, box2_h, box2_x, box2_y = 0, 0, 0, 0

        adjusted_img2_np = np.zeros((512, 512, 3), dtype=np.float32)
        adjusted_mask2_np = np.ones((512, 512), dtype=np.float32)

        if image1 is None and image2 is not None:
            image1 = image2
        elif image1 is None and image2 is None:
            default_img = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image1 = default_img
            image2 = default_img

        def get_img_size(img):
            if isinstance(img, torch.Tensor):
                return img.shape[1], img.shape[2]
            else:
                return img.shape[0], img.shape[1] if len(img.shape) >= 3 else (img.shape[0], img.shape[1])

        h1, w1 = get_img_size(image1)
        h2, w2 = get_img_size(image2)

        def tensor2numpy(img_tensor):
            if isinstance(img_tensor, torch.Tensor):
                return img_tensor.cpu().numpy()[0]
            else:
                return img_tensor[0] if len(img_tensor.shape) == 4 else img_tensor

        img1_np = tensor2numpy(image1)
        img2_np = tensor2numpy(image2)

        def process_mask(mask, img_h, img_w):
            if mask is None:
                return np.ones((img_h, img_w), dtype=np.float32)
            mask_np = tensor2numpy(mask)
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            if mask_np.shape != (img_h, img_w):
                mask_np = cv2.resize(mask_np, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            return mask_np

        mask1_np = process_mask(mask1, h1, w1)
        mask2_np = process_mask(mask2, h2, w2)

        if layout_mode == "居中-自动对齐":
            if h2 > h1:
                layout_mode = "居中-高度对齐"
            elif w2 > w1:
                layout_mode = "居中-宽度对齐"
            else:
                layout_mode = "居中-中心对齐"

        if layout_mode == "左右-中心对齐":
            img2_resized = get_image_resize(torch.from_numpy(img2_np).unsqueeze(0), torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (w1, h1), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1 + w1
            new_h = h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_w, box2_h = w1, h1
            box2_x = w1 + (w1 // 2)
            box2_y = h1 // 2
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[:h1, w1:w1+w1] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[:h1, w1:w1+w1] = mask2_resized_np

        elif layout_mode == "左右-高度对齐":
            ratio = h1 / h2
            box2_w = int(w2 * ratio)
            box2_h = h1
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1 + box2_w
            new_h = h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_x = w1 + (box2_w // 2)
            box2_y = h1 // 2
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[:box2_h, w1:w1+box2_w] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[:box2_h, w1:w1+box2_w] = mask2_resized_np

        elif layout_mode == "左右-宽度对齐":
            ratio = w1 / w2
            box2_w = w1
            box2_h = int(h2 * ratio)
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1 + box2_w
            new_h = max(h1, box2_h)
            y1_offset = (new_h - h1) // 2
            y2_offset = (new_h - box2_h) // 2
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = y1_offset + (h1 // 2)
            box2_x = w1 + (box2_w // 2)
            box2_y = y2_offset + (box2_h // 2)
            
            # 确保背景图像与输入通道数一致
            effective_bg_mode = "black" if bg_mode in ["image", "transparent"] else bg_mode
            composite_img = create_background(effective_bg_mode, new_w, new_h, img1_np)
            
            # 确保composite_img和img1_np/img2_resized_np通道数一致
            if composite_img.shape[-1] == 4 and img1_np.shape[-1] == 3:
                # 如果背景是4通道但原图是3通道，只使用前3个通道
                composite_img[y1_offset:y1_offset+h1, :w1, :3] = img1_np
                composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w, :3] = img2_resized_np
                # 对于alpha通道，设置为全不透明
                composite_img[y1_offset:y1_offset+h1, :w1, 3] = 1.0
                composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w, 3] = 1.0
            elif composite_img.shape[-1] == 3 and img1_np.shape[-1] == 3:
                # 如果都是3通道，直接赋值
                composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
                composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w] = img2_resized_np
            else:
                # 其他情况，确保通道数匹配
                if len(img1_np.shape) == 2:  # 灰度图
                    img1_np = np.stack([img1_np, img1_np, img1_np], axis=-1)
                if len(img2_resized_np.shape) == 2:  # 灰度图
                    img2_resized_np = np.stack([img2_resized_np, img2_resized_np, img2_resized_np], axis=-1)
                    
                # 确保目标区域与源通道数匹配
                target_slice_1 = composite_img[y1_offset:y1_offset+h1, :w1]
                target_slice_2 = composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w]
                
                if target_slice_1.shape[-1] != img1_np.shape[-1]:
                    if target_slice_1.shape[-1] == 4 and img1_np.shape[-1] == 3:
                        composite_img[y1_offset:y1_offset+h1, :w1, :3] = img1_np
                        composite_img[y1_offset:y1_offset+h1, :w1, 3] = 1.0
                    elif target_slice_1.shape[-1] == 3 and img1_np.shape[-1] == 4:
                        composite_img[y1_offset:y1_offset+h1, :w1] = img1_np[:, :, :3]
                else:
                    composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
                    
                if target_slice_2.shape[-1] != img2_resized_np.shape[-1]:
                    if target_slice_2.shape[-1] == 4 and img2_resized_np.shape[-1] == 3:
                        composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w, :3] = img2_resized_np
                        composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w, 3] = 1.0
                    elif target_slice_2.shape[-1] == 3 and img2_resized_np.shape[-1] == 4:
                        composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w] = img2_resized_np[:, :, :3]
                else:
                    composite_img[y2_offset:y2_offset+box2_h, w1:w1+box2_w] = img2_resized_np
            
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[y1_offset:y1_offset+h1, :w1] = mask1_np
            composite_mask[y2_offset:y2_offset+box2_h, w1:w1+box2_w] = mask2_resized_np



        elif layout_mode == "上下-中心对齐":
            img2_resized = get_image_resize(torch.from_numpy(img2_np).unsqueeze(0), torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (w1, h1), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = h1 + h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_w, box2_h = w1, h1
            box2_x = w1 // 2
            box2_y = h1 + (h1 // 2)
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[h1:h1+h1, :w1] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[h1:h1+h1, :w1] = mask2_resized_np

        elif layout_mode == "上下-宽度对齐":
            ratio = w1 / w2
            box2_w = w1
            box2_h = int(h2 * ratio)
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = h1 + box2_h
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_x = w1 // 2
            box2_y = h1 + (box2_h // 2)
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, :w1] = img1_np
            composite_img[h1:h1+box2_h, :w1] = img2_resized_np
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, :w1] = mask1_np
            composite_mask[h1:h1+box2_h, :w1] = mask2_resized_np

        elif layout_mode == "上下-高度对齐":
            ratio = h1 / h2
            box2_w = int(w2 * ratio)
            box2_h = h1
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = max(w1, box2_w)
            new_h = h1 + box2_h
            x1_offset = (new_w - w1) // 2
            x2_offset = (new_w - box2_w) // 2
            box1_w, box1_h = w1, h1
            box1_x = x1_offset + (w1 // 2)
            box1_y = h1 // 2
            box2_x = x2_offset + (box2_w // 2)
            box2_y = h1 + (box2_h // 2)
            
            # 确保背景图像与输入通道数一致
            effective_bg_mode = "black" if bg_mode in ["image", "transparent"] else bg_mode
            composite_img = create_background(effective_bg_mode, new_w, new_h, img1_np)
            
            # 确保composite_img和img1_np/img2_resized_np通道数一致
            if composite_img.shape[-1] == 4 and img1_np.shape[-1] == 3:
                # 如果背景是4通道但原图是3通道，只使用前3个通道
                composite_img[:h1, x1_offset:x1_offset+w1, :3] = img1_np
                composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w, :3] = img2_resized_np
                # 对于alpha通道，设置为全不透明
                composite_img[:h1, x1_offset:x1_offset+w1, 3] = 1.0
                composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w, 3] = 1.0
            elif composite_img.shape[-1] == 3 and img1_np.shape[-1] == 3:
                # 如果都是3通道，直接赋值
                composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
                composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w] = img2_resized_np
            else:
                # 其他情况，确保通道数匹配
                if len(img1_np.shape) == 2:  # 灰度图
                    img1_np = np.stack([img1_np, img1_np, img1_np], axis=-1)
                if len(img2_resized_np.shape) == 2:  # 灰度图
                    img2_resized_np = np.stack([img2_resized_np, img2_resized_np, img2_resized_np], axis=-1)
                    
                # 确保目标区域与源通道数匹配
                target_slice_1 = composite_img[:h1, x1_offset:x1_offset+w1]
                target_slice_2 = composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w]
                
                if target_slice_1.shape[-1] != img1_np.shape[-1]:
                    if target_slice_1.shape[-1] == 4 and img1_np.shape[-1] == 3:
                        composite_img[:h1, x1_offset:x1_offset+w1, :3] = img1_np
                        composite_img[:h1, x1_offset:x1_offset+w1, 3] = 1.0
                    elif target_slice_1.shape[-1] == 3 and img1_np.shape[-1] == 4:
                        composite_img[:h1, x1_offset:x1_offset+w1] = img1_np[:, :, :3]
                else:
                    composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
                    
                if target_slice_2.shape[-1] != img2_resized_np.shape[-1]:
                    if target_slice_2.shape[-1] == 4 and img2_resized_np.shape[-1] == 3:
                        composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w, :3] = img2_resized_np
                        composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w, 3] = 1.0
                    elif target_slice_2.shape[-1] == 3 and img2_resized_np.shape[-1] == 4:
                        composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w] = img2_resized_np[:, :, :3]
                else:
                    composite_img[h1:h1+box2_h, x2_offset:x2_offset+box2_w] = img2_resized_np
            
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, x1_offset:x1_offset+w1] = mask1_np
            composite_mask[h1:h1+box2_h, x2_offset:x2_offset+box2_w] = mask2_resized_np



        elif layout_mode == "居中-中心对齐":
            img2_resized = get_image_resize(torch.from_numpy(img2_np).unsqueeze(0), torch.from_numpy(img1_np).unsqueeze(0))
            img2_resized_np = img2_resized.numpy()[0]
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (w1, h1), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = h1
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = h1 // 2
            box2_w, box2_h = w1, h1
            box2_x = w1 // 2
            box2_y = h1 // 2
            composite_img = img1_np.copy()
            alpha = mask2_resized_np[..., np.newaxis]
            composite_img = composite_img * (1 - alpha) + img2_resized_np * alpha
            composite_mask = mask2_resized_np.copy()

        elif layout_mode == "居中-高度对齐":
            ratio = h1 / h2
            box2_w = int(w2 * ratio)
            box2_h = h1
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = max(w1, box2_w)
            new_h = h1
            x1_offset = (new_w - w1) // 2
            x2_offset = (new_w - box2_w) // 2
            box1_w, box1_h = w1, h1
            box1_x = x1_offset + (w1 // 2)
            box1_y = h1 // 2
            box2_x = x2_offset + (box2_w // 2)
            box2_y = h1 // 2
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[:h1, x1_offset:x1_offset+w1] = img1_np
            overlap_start_x = max(x1_offset, x2_offset)
            overlap_end_x = min(x1_offset + w1, x2_offset + box2_w)
            if overlap_start_x < overlap_end_x:
                overlap_width = overlap_end_x - overlap_start_x
                img1_overlap_idx = overlap_start_x - x1_offset
                img2_overlap_idx = overlap_start_x - x2_offset
                alpha = mask2_resized_np[:, img2_overlap_idx:img2_overlap_idx+overlap_width][..., np.newaxis]
                composite_img[:h1, overlap_start_x:overlap_end_x] = composite_img[:h1, overlap_start_x:overlap_end_x] * (1 - alpha) + img2_resized_np[:h1, img2_overlap_idx:img2_overlap_idx+overlap_width] * alpha
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[:h1, x1_offset:x1_offset+w1] = mask1_np
            if overlap_start_x < overlap_end_x:
                composite_mask[:h1, overlap_start_x:overlap_end_x] = mask2_resized_np[:h1, img2_overlap_idx:img2_overlap_idx+overlap_width]

        elif layout_mode == "居中-宽度对齐":
            ratio = w1 / w2
            box2_w = w1
            box2_h = int(h2 * ratio)
            img2_resized_np = cv2.resize(img2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_img2_np = img2_resized_np.copy()
            mask2_resized_np = cv2.resize(mask2_np, (box2_w, box2_h), interpolation=interpolation)
            adjusted_mask2_np = mask2_resized_np.copy()
            new_w = w1
            new_h = max(h1, box2_h)
            y1_offset = (new_h - h1) // 2
            y2_offset = (new_h - box2_h) // 2
            box1_w, box1_h = w1, h1
            box1_x = w1 // 2
            box1_y = y1_offset + (h1 // 2)
            box2_x = w1 // 2
            box2_y = y2_offset + (box2_h // 2)
            composite_img = np.zeros((new_h, new_w, 3), dtype=np.float32)
            composite_img[y1_offset:y1_offset+h1, :w1] = img1_np
            overlap_start_y = max(y1_offset, y2_offset)
            overlap_end_y = min(y1_offset + h1, y2_offset + box2_h)
            if overlap_start_y < overlap_end_y:
                overlap_height = overlap_end_y - overlap_start_y
                img1_overlap_idx = overlap_start_y - y1_offset
                img2_overlap_idx = overlap_start_y - y2_offset
                alpha = mask2_resized_np[img2_overlap_idx:img2_overlap_idx+overlap_height, :][..., np.newaxis]
                composite_img[overlap_start_y:overlap_end_y, :w1] = composite_img[overlap_start_y:overlap_end_y, :w1] * (1 - alpha) + img2_resized_np[img2_overlap_idx:img2_overlap_idx+overlap_height, :w1] * alpha
            composite_mask = np.zeros((new_h, new_w), dtype=np.float32)
            composite_mask[y1_offset:y1_offset+h1, :w1] = mask1_np
            if overlap_start_y < overlap_end_y:
                composite_mask[overlap_start_y:overlap_end_y, :w1] = mask2_resized_np[img2_overlap_idx:img2_overlap_idx+overlap_height, :w1]

        scale_ratio = 1.0
        if size_mode != "auto":
            current_h, current_w = composite_img.shape[:2]
            if size_mode == "输出宽度":
                scale_ratio = target_size / current_w
            else:
                scale_ratio = target_size / current_h
            new_w = int(current_w * scale_ratio)
            new_h = int(current_h * scale_ratio)
            composite_img = cv2.resize(composite_img, (new_w, new_h), interpolation=interpolation)
            composite_mask = cv2.resize(composite_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            adjusted_img2_np = cv2.resize(adjusted_img2_np, (int(adjusted_img2_np.shape[1]*scale_ratio), int(adjusted_img2_np.shape[0]*scale_ratio)), interpolation=interpolation)
            adjusted_mask2_np = cv2.resize(adjusted_mask2_np, (int(adjusted_mask2_np.shape[1]*scale_ratio), int(adjusted_mask2_np.shape[0]*scale_ratio)), interpolation=cv2.INTER_NEAREST)
            box1_w = int(box1_w * scale_ratio)
            box1_h = int(box1_h * scale_ratio)
            box1_x = int(box1_x * scale_ratio)
            box1_y = int(box1_y * scale_ratio)
            box2_w = int(box2_w * scale_ratio)
            box2_h = int(box2_h * scale_ratio)
            box2_x = int(box2_x * scale_ratio)
            box2_y = int(box2_y * scale_ratio)

        if divider_thickness > 0 and layout_mode not in ["居中-高度对齐", "居中-宽度对齐", "居中-中心对齐"]:
            if layout_mode.startswith("左右"):
                divider_x = box1_x + (box1_w // 2)
                start_x = max(0, divider_x - divider_thickness)
                end_x = min(composite_img.shape[1], divider_x)
                if start_x < end_x:
                    composite_img[:, start_x:end_x, :] = 0
            else:
                divider_y = box1_y + (box1_h // 2)
                start_y = max(0, divider_y - divider_thickness)
                end_y = min(composite_img.shape[0], divider_y)
                if start_y < end_y:
                    composite_img[start_y:end_y, :, :] = 0

        def np2tensor(np_arr, is_mask=False):
            if is_mask:
                return torch.from_numpy(np_arr).float().unsqueeze(0)
            else:
                if np_arr.shape[-1] == 3 and bg_mode == "transparent":
                    alpha = composite_mask[..., np.newaxis]
                    np_arr = np.concatenate([np_arr, alpha], axis=-1)
                return torch.from_numpy(np_arr).float().unsqueeze(0)

        final_img_tensor = np2tensor(composite_img)
        final_mask_tensor = np2tensor(composite_mask, is_mask=True)
        adjusted_img2_tensor = torch.from_numpy(adjusted_img2_np).float().unsqueeze(0)
        adjusted_mask2_tensor = torch.from_numpy(adjusted_mask2_np).float().unsqueeze(0)

        if not isinstance(final_img_tensor, torch.Tensor):
            final_img_tensor = torch.from_numpy(final_img_tensor).float() if isinstance(final_img_tensor, np.ndarray) else torch.tensor(final_img_tensor, dtype=torch.float32)
        if not isinstance(final_mask_tensor, torch.Tensor):
            final_mask_tensor = torch.from_numpy(final_mask_tensor).float() if isinstance(final_mask_tensor, np.ndarray) else torch.tensor(final_mask_tensor, dtype=torch.float32)
        if not isinstance(adjusted_img2_tensor, torch.Tensor):
            adjusted_img2_tensor = torch.from_numpy(adjusted_img2_tensor).float() if isinstance(adjusted_img2_tensor, np.ndarray) else torch.tensor(adjusted_img2_tensor, dtype=torch.float32)
        if not isinstance(adjusted_mask2_tensor, torch.Tensor):
            adjusted_mask2_tensor = torch.from_numpy(adjusted_mask2_tensor).float() if isinstance(adjusted_mask2_tensor, np.ndarray) else torch.tensor(adjusted_mask2_tensor, dtype=torch.float32)

        box2 = (box1_w, box1_h, box1_x, box1_y, box2_w, box2_h, box2_x, box2_y)
        return (final_img_tensor, final_mask_tensor, box2, adjusted_img2_tensor, adjusted_mask2_tensor)

#endregion----图像-双图合并---总控制---------




 
class Stack_sample_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == no change"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == no change"}),
                "sampler": ([None] + list(comfy.samplers.KSampler.SAMPLERS), {"default": "euler"}),  
                "scheduler": ([None] + list(comfy.samplers.KSampler.SCHEDULERS), {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("SAMPLE_STACK", )
    RETURN_NAMES = ("sample_stack", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def sample(self, steps, cfg, sampler, scheduler):
        sample_stack = (steps, cfg, sampler, scheduler)     
        return (sample_stack, )
    



class XXXchx_Ksampler_Kontext_inpaint:   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "work_pattern": (["ksampler", "only_adjust_mask"], {"default": "ksampler"}),
                "mask_sampling": ("BOOLEAN", {"default": True, }),
                "crop_mode": (["no_crop", "no_scale_crop", "scale_crop_image"], {"default": "no_scale_crop"}),            
                "long_side": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
                "expand_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "out_smoothness": ("INT", {"default": 2, "min": 0, "max": 150, "step": 1}),
            },
            "optional": {
                "mask": ("MASK", ),
                "pos": ("STRING", {"multiline": True, "default": ""}),
                "mask_stack": ("MASK_STACK2",),  
                "sample_stack": ("SAMPLE_STACK",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("context",  "image",  "cropped_mask", "cropped_image")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def run(self, context, seed, image=None, mask=None, denoise=1, prompt_weight=0.5, pos="",
            work_pattern="ksampler", sample_stack=None, mask_sampling=False, out_smoothness=2,
            mask_stack=None, crop_mode="no_crop", long_side=512,
            expand_width=0, expand_height=0, ):
        
        divisible_by=1
        if mask is None:
            batch_size, height, width, _ = image.shape
            mask = torch.ones((batch_size, height, width), dtype=torch.float32)
            
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")

        if sample_stack is not None:
            steps, cfg, sampler, scheduler = sample_stack   
            if steps == 0: 
                steps = context.get("steps")
            if cfg == 0: 
                cfg = context.get("cfg")
            if scheduler == None: 
                scheduler = context.get("scheduler")
            if sampler == None: 
                sampler = context.get("sampler")    
        else:
            steps = context.get("steps")       
            cfg = context.get("cfg")
            scheduler = context.get("scheduler")
            sampler = context.get("sampler")  

        guidance = context.get("guidance", 3.5)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)

        background_tensor = None
        background_mask_tensor = None
        cropped_image_tensor = None
        cropped_mask_tensor = None
        stitch = None

        if image is not None and mask is not None :
            background_tensor, background_mask_tensor, cropped_image_tensor, cropped_mask_tensor, stitch = Image_solo_crop().inpaint_crop(
                    image=image,
                    crop_mode = crop_mode,
                    long_side = long_side,  
                    upscale_method ="bicubic", 
                    expand_width = expand_width, 
                    expand_height = expand_height, 
                    auto_expand_square=False,
                    divisible_by = divisible_by,
                    mask=mask, 
                    mask_stack=mask_stack, 
                    crop_img_bj="image")

            processed_image = cropped_image_tensor
            processed_mask = cropped_mask_tensor

            if work_pattern == "only_adjust_mask": 
                return  (context, image, cropped_mask_tensor, cropped_image_tensor)

            encoded_result = encode(vae, processed_image)[0]
            if isinstance(encoded_result, dict):
                if "samples" in encoded_result:
                    encoded_latent = encoded_result["samples"]
                else:
                    raise ValueError(f"Encoded result dict doesn't contain 'samples' key. Keys: {list(encoded_result.keys())}")
            elif torch.is_tensor(encoded_result):
                encoded_latent = encoded_result
            else:
                try:
                    encoded_latent = torch.tensor(encoded_result)
                except Exception as e:
                    raise TypeError(f"Cannot convert encoded result to tensor. Type: {type(encoded_result)}, Error: {e}")

            if encoded_latent.dim() == 5:
                if encoded_latent.shape[2] == 1:
                    encoded_latent = encoded_latent.squeeze(2)
                else:
                     encoded_latent = encoded_latent.view(encoded_latent.shape[0], 
                                                    encoded_latent.shape[1], 
                                                    encoded_latent.shape[3], 
                                                    encoded_latent.shape[4])
            elif encoded_latent.dim() == 3:
                encoded_latent = encoded_latent.unsqueeze(0)
            elif encoded_latent.dim() != 4:
                raise ValueError(f"Unexpected latent dimensions: {encoded_latent.dim()}. Expected 4D tensor (B,C,H,W). Shape: {encoded_latent.shape}")

            if encoded_latent.size(0) > 1:
                encoded_latent = encoded_latent[:1]

            latent2 = encoded_latent              
            if not isinstance(latent2, dict):
                if torch.is_tensor(latent2):
                    latent2 = {"samples": latent2}
                else:
                    raise ValueError(f"Unexpected latent format: {type(latent2)}")
            if "samples" not in latent2:
                raise ValueError("Latent dictionary must contain 'samples' key")

            if mask_sampling == False:
                latent3 = latent2
            else:
                if processed_mask is not None:
                    if not torch.is_tensor(processed_mask):
                        processed_mask = torch.tensor(processed_mask)
                    if processed_mask.dim() == 3:
                        processed_mask = processed_mask.unsqueeze(0)
                    import copy
                    latent3 = copy.deepcopy(latent2)
                    if processed_mask.shape[1] == 1:
                        processed_mask = processed_mask.repeat(1, 4, 1, 1)                    
                    latent3["noise_mask"] = processed_mask
                else:
                    latent3 = latent2

            if work_pattern == "ksampler":
                if positive is not None and prompt_weight > 0:
                    latent_samples = None
                    if isinstance(latent3, dict) and "samples" in latent3:
                        latent_samples = latent3["samples"]
                    elif torch.is_tensor(latent3):
                        latent_samples = latent3                     
                    if latent_samples is not None and latent_samples.numel() > 0:
                        try:
                            influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                            scaled_latent = latent_samples * influence
                            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                        except Exception as e:
                            print(f"Warning: Failed to process kontext sampling: {e}")

            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            result = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent3, denoise=denoise)
            latent_result = result[0]
            output_image = decode(vae, latent_result)[0]

            fimage, output_image, original_image = Image_solo_stitch().inpaint_stitch(
                inpainted_image=output_image,
                smoothness=out_smoothness, 
                mask=cropped_mask_tensor, 
                stitch=stitch, 
                blend_factor=1.0, 
                blend_mode="normal", 
                opacity=1.0, 
                stitch_mode="crop_mask", 
                recover_method="bicubic")


            latent = encode(vae, output_image)[0]
            context = new_context(context, latent=latent, images=output_image)

            return (context, output_image, cropped_mask_tensor, cropped_image_tensor)



class chx_Ksampler_Kontext_inpaint:   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "work_pattern": (["ksampler", "only_adjust_mask"], {"default": "ksampler"}),
                "mask_sampling": ("BOOLEAN", {"default": True, }),
                "crop_mode": (["no_crop", "no_scale_crop", "scale_crop_image"], {"default": "no_scale_crop"}),            
                "long_side": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
                "expand_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "out_smoothness": ("INT", {"default": 2, "min": 0, "max": 150, "step": 1}),
            },
            "optional": {
                "mask": ("MASK", ),
                "pos": ("STRING", {"multiline": True, "default": ""}),
                "mask_stack": ("MASK_STACK2",),  
                "sample_stack": ("SAMPLE_STACK",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("context",  "image",  "cropped_mask", "cropped_image")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def run(self, context, seed, image=None, mask=None, denoise=1, prompt_weight=0.5, pos="",
            work_pattern="ksampler", sample_stack=None, mask_sampling=True, out_smoothness=2,
            mask_stack=None, crop_mode="no_crop", long_side=512,
            expand_width=0, expand_height=0, ):
        
        divisible_by=1
        if mask is None:
            batch_size, height, width, _ = image.shape
            mask = torch.ones((batch_size, height, width), dtype=torch.float32)
            
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")

        if sample_stack is not None:
            steps, cfg, sampler, scheduler = sample_stack   
            if steps == 0: 
                steps = context.get("steps")
            if cfg == 0: 
                cfg = context.get("cfg")
            if scheduler == None: 
                scheduler = context.get("scheduler")
            if sampler == None: 
                sampler = context.get("sampler")    
        else:
            steps = context.get("steps")       
            cfg = context.get("cfg")
            scheduler = context.get("scheduler")
            sampler = context.get("sampler")  

        guidance = context.get("guidance", 3.5)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)

        background_tensor = None
        background_mask_tensor = None
        cropped_image_tensor = None
        cropped_mask_tensor = None
        stitch = None

        if image is not None and mask is not None :
            background_tensor, background_mask_tensor, cropped_image_tensor, cropped_mask_tensor, stitch = Image_solo_crop().inpaint_crop(
                    image=image,
                    crop_mode = crop_mode,
                    long_side = long_side,  
                    upscale_method ="bicubic", 
                    expand_width = expand_width, 
                    expand_height = expand_height, 
                    auto_expand_square=False,
                    divisible_by = divisible_by,
                    mask=mask, 
                    mask_stack=mask_stack, 
                    crop_img_bj="image")

            processed_image = cropped_image_tensor
            processed_mask = cropped_mask_tensor

            if work_pattern == "only_adjust_mask": 
                return  (context, image, cropped_mask_tensor, cropped_image_tensor)

            encoded_result = encode(vae, processed_image)[0]
            if isinstance(encoded_result, dict):
                if "samples" in encoded_result:
                    encoded_latent = encoded_result["samples"]
                else:
                    raise ValueError(f"Encoded result dict doesn't contain 'samples' key. Keys: {list(encoded_result.keys())}")
            elif torch.is_tensor(encoded_result):
                encoded_latent = encoded_result
            else:
                try:
                    encoded_latent = torch.tensor(encoded_result)
                except Exception as e:
                    raise TypeError(f"Cannot convert encoded result to tensor. Type: {type(encoded_result)}, Error: {e}")

            if encoded_latent.dim() == 5:
                if encoded_latent.shape[2] == 1:
                    encoded_latent = encoded_latent.squeeze(2)
                else:
                     encoded_latent = encoded_latent.view(encoded_latent.shape[0], 
                                                    encoded_latent.shape[1], 
                                                    encoded_latent.shape[3], 
                                                    encoded_latent.shape[4])
            elif encoded_latent.dim() == 3:
                encoded_latent = encoded_latent.unsqueeze(0)
            elif encoded_latent.dim() != 4:
                raise ValueError(f"Unexpected latent dimensions: {encoded_latent.dim()}. Expected 4D tensor (B,C,H,W). Shape: {encoded_latent.shape}")

            if encoded_latent.size(0) > 1:
                encoded_latent = encoded_latent[:1]

            latent2 = encoded_latent              
            if not isinstance(latent2, dict):
                if torch.is_tensor(latent2):
                    latent2 = {"samples": latent2}
                else:
                    raise ValueError(f"Unexpected latent format: {type(latent2)}")
            if "samples" not in latent2:
                raise ValueError("Latent dictionary must contain 'samples' key")

            if mask_sampling == False:
                latent3 = latent2
            else:
                if processed_mask is not None:
                    if not torch.is_tensor(processed_mask):
                        processed_mask = torch.tensor(processed_mask, device=encoded_latent.device)
                    if processed_mask.dim() == 3:
                        processed_mask = processed_mask.unsqueeze(0)
                    latent_h, latent_w = encoded_latent.shape[2], encoded_latent.shape[3]
                    processed_mask = torch.nn.functional.interpolate(
                        processed_mask.unsqueeze(1),
                        size=(latent_h, latent_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(1)
                    if processed_mask.shape[1] == 1:
                        processed_mask = processed_mask.repeat(1, 4, 1, 1)
                    processed_mask = torch.clamp(processed_mask, 0.0, 1.0)
                    latent3 = copy.deepcopy(latent2)
                    latent3["noise_mask"] = processed_mask
                else:
                    latent3 = latent2

            if work_pattern == "ksampler":
                if positive is not None and prompt_weight > 0:
                    latent_samples = None
                    if isinstance(latent3, dict) and "samples" in latent3:
                        latent_samples = latent3["samples"]
                    elif torch.is_tensor(latent3):
                        latent_samples = latent3                     
                    if latent_samples is not None and latent_samples.numel() > 0:
                        try:
                            influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                            scaled_latent = latent_samples * influence
                            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                        except Exception as e:
                            print(f"Warning: Failed to process kontext sampling: {e}")

            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

            result = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent3, denoise=denoise)
            latent_result = result[0]
            output_image = decode(vae, latent_result)[0]

            fimage, output_image, original_image = Image_solo_stitch().inpaint_stitch(
                inpainted_image=output_image,
                smoothness=out_smoothness, 
                mask=cropped_mask_tensor, 
                stitch=stitch, 
                blend_factor=1.0, 
                blend_mode="normal", 
                opacity=1.0, 
                stitch_mode="crop_mask", 
                recover_method="bicubic")

            latent = encode(vae, output_image)[0]
            context = new_context(context, latent=latent, images=output_image)

            return (context, output_image, cropped_mask_tensor, cropped_image_tensor)







class Mask_transform_sum:
    def __init__(self):
        self.colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_mode": (["crop_image","image", "transparent", "white", "black", "red", "green", "blue"],),
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block", "circle", "outline_circle"], {"default": "original"}),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "outline_thickness": ("INT", {"default": 3, "min": 1, "max": 400, "step": 1}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 0.1}),
                "tapered_corners": ("BOOLEAN", {"default": True}),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "crop_to_mask": ("BOOLEAN", {"default": False}),
                "expand_width": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "rescale_crop": ("FLOAT", {"default": 1.00, "min": 0.1, "max": 10.0, "step": 0.01}),
                "divisible_by": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
            },
            "optional": {"base_image": ("IMAGE",), "mask": ("MASK",)}
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"
    
    def separate(self, bg_mode, mask_mode="fill", 
                 ignore_threshold=100, opacity=1.0, outline_thickness=1, 
                 smoothness=1, mask_expand=0,
                 expand_width=0, expand_height=0, rescale_crop=1.0,
                 tapered_corners=True, mask_min=0.0, mask_max=1.0,
                 base_image=None, mask=None, crop_to_mask=False, divisible_by=8):
        
        if mask is None:
            if base_image is not None:
                combined_image_tensor = base_image
                empty_mask = torch.zeros_like(base_image[:, :, :, 0])
            else:
                empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
                combined_image_tensor = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (combined_image_tensor, empty_mask)
        
        def tensorMask2cv2img(tensor_mask):
            mask_np = tensor_mask.cpu().numpy().squeeze()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return (mask_np * 255).astype(np.uint8)
        
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= ignore_threshold:
                filtered_contours.append(contour)
        
        contours_with_positions = []
        for contour in filtered_contours:
            x, y, w, h = cv2.boundingRect(contour)
            contours_with_positions.append((x, y, contour))
        contours_with_positions.sort(key=lambda item: (item[1], item[0]))
        sorted_contours = [item[2] for item in contours_with_positions]
        
        final_mask = np.zeros_like(binary_mask)
        c = 0 if tapered_corners else 1
        kernel = np.array([[c, 1, c], [1, 1, 1], [c, 1, c]], dtype=np.uint8)
        
        for contour in sorted_contours[:8]:
            temp_mask = np.zeros_like(binary_mask)
            
            if mask_mode == "original":
                cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                temp_mask = cv2.bitwise_and(opencv_gray_image, temp_mask)
            elif mask_mode == "fill":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "fill_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline":
                cv2.drawContours(temp_mask, [contour], 0, (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "outline_block":
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(temp_mask, (x, y), (x+w, y+h), (255, 255, 255), thickness=outline_thickness)
            elif mask_mode == "circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=cv2.FILLED)
            elif mask_mode == "outline_circle":
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(temp_mask, center, radius, (255, 255, 255), thickness=outline_thickness)
            
            if mask_expand != 0:
                expand_amount = abs(mask_expand)
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, kernel, iterations=expand_amount)
                else:
                    temp_mask = cv2.erode(temp_mask, kernel, iterations=expand_amount)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        if smoothness > 0:
            final_mask_pil = Image.fromarray(final_mask)
            final_mask_pil = final_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(final_mask_pil)
        
        original_h, original_w = final_mask.shape[:2]
        coords = cv2.findNonZero(final_mask)
        crop_params = None

        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            center_x = x + w / 2.0
            center_y = y + h / 2.0
            max_expand_left = center_x - 0
            max_expand_right = original_w - center_x
            max_expand_top = center_y - 0
            max_expand_bottom = original_h - center_y
            actual_expand_x = min(expand_width, max_expand_left, max_expand_right)
            actual_expand_y = min(expand_height, max_expand_top, max_expand_bottom)
            x_start = int(round(center_x - (w / 2.0) - actual_expand_x))
            x_end = int(round(center_x + (w / 2.0) + actual_expand_x))
            y_start = int(round(center_y - (h / 2.0) - actual_expand_y))
            y_end = int(round(center_y + (h / 2.0) + actual_expand_y))
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(original_w, x_end)
            y_end = min(original_h, y_end)
            width = x_end - x_start
            height = y_end - y_start
            if width % 2 != 0:
                if x_end < original_w:
                    x_end += 1
                elif x_start > 0:
                    x_start -= 1
            if height % 2 != 0:
                if y_end < original_h:
                    y_end += 1
                elif y_start > 0:
                    y_start -= 1
            x_start = max(0, x_start)
            y_start = max(0, y_start)
            x_end = min(original_w, x_end)
            y_end = min(original_h, y_end)
            crop_params = (x_start, y_start, x_end, y_end)
        else:
            crop_params = (0, 0, original_w, original_h)

        if base_image is None:
            base_image_np = np.zeros((original_h, original_w, 3), dtype=np.float32)
        else:
            base_image_np = base_image[0].cpu().numpy() * 255.0
            base_image_np = base_image_np.astype(np.float32)
        
        if crop_to_mask and crop_params is not None:
            x_start, y_start, x_end, y_end = crop_params[:4]
            cropped_final_mask = final_mask[y_start:y_end, x_start:x_end]
            cropped_base_image = base_image_np[y_start:y_end, x_start:x_end].copy()
            
            if rescale_crop != 1.0:
                scaled_w = int(cropped_final_mask.shape[1] * rescale_crop)
                scaled_h = int(cropped_final_mask.shape[0] * rescale_crop)
                cropped_final_mask = cv2.resize(cropped_final_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
                cropped_base_image = cv2.resize(cropped_base_image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            final_mask = cropped_final_mask
            base_image_np = cropped_base_image
        else:
            if base_image_np.shape[:2] != (original_h, original_w):
                base_image_np = cv2.resize(base_image_np, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        
        h, w = base_image_np.shape[:2]
        background = np.zeros((h, w, 3), dtype=np.float32)
        if bg_mode in self.colors:
            background[:] = self.colors[bg_mode]
        elif bg_mode == "image" and base_image is not None:
            background = base_image_np.copy()
        elif bg_mode == "transparent":
            background = np.zeros((h, w, 3), dtype=np.float32)
        
        if background.shape[:2] != (h, w):
            background = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
        
        if bg_mode == "crop_image":
            combined_image = base_image_np.copy()
        elif bg_mode in ["white", "black", "red", "green", "blue", "transparent"]:
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            mask_float = mask_float[:, :, np.newaxis]
            combined_image = mask_float * base_image_np + (1 - mask_float) * background
        elif bg_mode == "image":
            combined_image = background.copy()
            mask_float = final_mask.astype(np.float32) / 255.0
            if mask_float.ndim == 3:
                mask_float = mask_float.squeeze()
            mask_max_val = np.max(mask_float) if np.max(mask_float) > 0 else 1
            mask_float = (mask_float / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_float = np.clip(mask_float, 0.0, 1.0)
            color = np.array(self.colors["white"], dtype=np.float32)
            for c in range(3):
                combined_image[:, :, c] = (mask_float * (opacity * color[c] + (1 - opacity) * combined_image[:, :, c]) + 
                                         (1 - mask_float) * combined_image[:, :, c])
        
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
        final_mask = final_mask.astype(np.uint8)
        
        if divisible_by > 1:
            h, w = combined_image.shape[:2]
            new_h = ((h + divisible_by - 1) // divisible_by) * divisible_by
            new_w = ((w + divisible_by - 1) // divisible_by) * divisible_by
            if new_h != h or new_w != w:
                padded_image = np.zeros((new_h, new_w, 3), dtype=combined_image.dtype)
                padded_image[:h, :w, :] = combined_image
                padded_mask = np.zeros((new_h, new_w), dtype=final_mask.dtype)
                padded_mask[:h, :w] = final_mask
                combined_image = padded_image
                final_mask = padded_mask
        
        combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
        combined_image_tensor = combined_image_tensor.unsqueeze(0)
        final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
        final_mask_tensor = final_mask_tensor.unsqueeze(0)
        
        return (combined_image_tensor, final_mask_tensor)



class Image_transform_layer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bj_img": ("IMAGE",),  
                "fj_img": ("IMAGE",),  
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_mode": (["original", "fill", "fill_block", "outline", "outline_block", "circle", "outline_circle"], {"default": "original"}),
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], {"default": "black"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": (BLEND_METHODS , {"default": "normal"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
            },
            "optional": {
                "mask": ("MASK",),      
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE",  "MASK", )
    RETURN_NAMES = ( "bj_composite", "mask", "composite","line_mask",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image"
  
    def process( self, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, mask_expand, smoothness,
                opacity, blending_mode, blend_strength, bj_img=None, fj_img=None,mask_mode="fill", mask=None ):
        
        color_mapping = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }
        if fj_img is None: raise ValueError("前景图像(fj_img)是必需的输入")
        if bj_img is None: raise ValueError("背景图像(bj_img)是必需的输入")
        
        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
        
        # 画布尺寸应该严格按照背景图和前景图的最大尺寸来确定
        canvas_width = max(bj_pil.size[0], fj_pil.size[0])
        canvas_height = max(bj_pil.size[1], fj_pil.size[1])
        canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2
        
        # 当没有输入遮罩时，默认赋值一个fj_img全遮罩
        if mask is None:
            mask = torch.ones((1, fj_pil.size[1], fj_pil.size[0]), dtype=torch.float32)

        mask_tensor = None
        if mask is not None: 
            if hasattr(mask, 'convert'):
                mask_tensor = pil2tensor(mask.convert('L'))
            else:  
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask if len(mask.shape) <= 3 else mask.squeeze(-1) if mask.shape[-1] == 1 else mask
                else:
                    mask_tensor = mask

        separated_result = Mask_transform_sum().separate(  
            bg_mode="crop_image", 
            mask_mode=mask_mode,
            ignore_threshold=0, 
            opacity=1, 
            outline_thickness=1, 
            smoothness=smoothness,
            mask_expand=mask_expand,
            expand_width=0, 
            expand_height=0,
            rescale_crop=1.0,
            tapered_corners=True,
            mask_min=0, 
            mask_max=1,
            base_image=fj_img, 
            mask=mask_tensor, 
            crop_to_mask=False,
            divisible_by=1
        )

        fj_img = separated_result[0]
        mask = separated_result[1]
        
        # 处理遮罩和前景图，保持它们的相对位置关系
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
            
            # 确保遮罩尺寸与前景图一致
            if mask_pil.size != fj_pil.size:
                mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
            
            fj_with_mask = fj_pil.copy()
            fj_with_mask.putalpha(mask_pil)
            
            # 不再进行裁剪，保持原始位置关系
            fj_processed = fj_with_mask
            mask_processed = mask_pil
        else:
            # 没有遮罩时，使用整个前景图
            mask_processed = Image.new("L", fj_pil.size, 255)
            fj_processed = fj_pil.copy()
            fj_processed.putalpha(mask_processed)
        
        # 获取处理后图像的尺寸
        processed_width, processed_height = fj_processed.size
        # 使用图像中心作为旋转中心
        center_x, center_y = processed_width // 2, processed_height // 2
        
        adjusted_fj = fj_processed
        adjusted_mask = mask_processed

        rotation = float(rotation)
        
        # 应用旋转和缩放
        if rotation != 0 or scale != 1.0:
            adjusted_fj = adjusted_fj.rotate(rotation, center=(center_x, center_y), resample=Image.BICUBIC, expand=True)
            adjusted_mask = adjusted_mask.rotate(rotation, center=(center_x, center_y), resample=Image.BICUBIC, expand=True)
            
            if scale != 1.0:
                new_size = (int(adjusted_fj.size[0] * scale), int(adjusted_fj.size[1] * scale))
                adjusted_fj = adjusted_fj.resize(new_size, Image.LANCZOS)
                adjusted_mask = adjusted_mask.resize(new_size, Image.LANCZOS)
            
            # 更新中心点
            center_x, center_y = adjusted_fj.size[0] // 2, adjusted_fj.size[1] // 2
        
        # 计算背景图的中心点
        bj_center_x, bj_center_y = bj_pil.size[0] // 2, bj_pil.size[1] // 2
        
        # 计算前景图应该放置的位置（相对于画布中心）
        x_position = canvas_center_x - center_x + x_offset
        y_position = canvas_center_y - center_y + y_offset
        
        paste_x = int(x_position)
        paste_y = int(y_position)
        
        # 应用透明度
        if opacity < 1.0:
            r, g, b, a = adjusted_fj.split()
            a = a.point(lambda p: p * opacity)
            adjusted_fj = Image.merge("RGBA", (r, g, b, a))
        
        # 创建画布并放置背景图
        composite_pil = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))
        # 背景图居中放置
        bj_x = (canvas_width - bj_pil.size[0]) // 2
        bj_y = (canvas_height - bj_pil.size[1]) // 2
        composite_pil.paste(bj_pil, (bj_x, bj_y))
        
        # 应用混合模式并粘贴前景图
        if blending_mode != "normal":
            temp_img = Image.new('RGBA', composite_pil.size, (0, 0, 0, 0))
            temp_img.paste(adjusted_fj, (paste_x, paste_y), adjusted_fj)
            blended_pil = Image.new('RGBA', composite_pil.size, (0, 0, 0, 0))
            
            for x in range(max(0, paste_x), min(canvas_width, paste_x + adjusted_fj.size[0])):
                for y in range(max(0, paste_y), min(canvas_height, paste_y + adjusted_fj.size[1])):
                    if temp_img.getpixel((x, y))[3] > 0:
                        bg_pixel = composite_pil.getpixel((x, y))
                        fg_pixel = temp_img.getpixel((x, y))
                        bg_pixel_img = Image.new('RGBA', (1, 1), bg_pixel)
                        fg_pixel_img = Image.new('RGBA', (1, 1), fg_pixel)
                        blended_pixel_img = apply_blending_mode(
                            bg_pixel_img, fg_pixel_img, blending_mode, blend_strength
                        )
                        blended_pil.putpixel((x, y), blended_pixel_img.getpixel((0, 0)))
            composite_pil = Image.alpha_composite(composite_pil, blended_pil)
        else:
            composite_pil.paste(adjusted_fj, (paste_x, paste_y), adjusted_fj)
        
        # 处理边缘检测
        if edge_detection:
            if edge_color in color_mapping:
                r, g, b = color_mapping[edge_color]
            else:
                r, g, b = 0, 0, 0
            
            threshold = 128
            full_size_mask = Image.new("L", composite_pil.size, 0)
            
            # 正确放置遮罩到完整画布上
            mask_left = max(0, paste_x)
            mask_top = max(0, paste_y)
            mask_right = min(paste_x + adjusted_mask.size[0], canvas_width)
            mask_bottom = min(paste_y + adjusted_mask.size[1], canvas_height)
            
            if mask_right > mask_left and mask_bottom > mask_top:
                crop_left = max(0, -paste_x)
                crop_top = max(0, -paste_y)
                crop_right = crop_left + (mask_right - mask_left)
                crop_bottom = crop_top + (mask_bottom - mask_top)
                
                if crop_right <= adjusted_mask.size[0] and crop_bottom <= adjusted_mask.size[1]:
                    mask_crop = adjusted_mask.crop((crop_left, crop_top, crop_right, crop_bottom))
                    full_size_mask.paste(mask_crop, (mask_left, mask_top, mask_right, mask_bottom))
            
            mask_array = np.array(full_size_mask)
            binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            binary_mask_pil = Image.fromarray(binary_mask)
            
            edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
            edge_draw = ImageDraw.Draw(edge_image)
            mask_cv = np.array(binary_mask_pil)
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for i in range(edge_thickness):
                    points = [tuple(point[0]) for point in contour]
                    edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness-i+1)
            
            composite_pil = Image.alpha_composite(composite_pil, edge_image)
            edge_mask = np.zeros_like(mask_cv)
            cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
            line_mask_pil = Image.fromarray(edge_mask)
        else:
            # 创建完整尺寸的遮罩
            full_size_mask = Image.new("L", composite_pil.size, 0)
            
            # 正确放置遮罩到完整画布上
            mask_left = max(0, paste_x)
            mask_top = max(0, paste_y)
            mask_right = min(paste_x + adjusted_mask.size[0], canvas_width)
            mask_bottom = min(paste_y + adjusted_mask.size[1], canvas_height)
            
            if mask_right > mask_left and mask_bottom > mask_top:
                crop_left = max(0, -paste_x)
                crop_top = max(0, -paste_y)
                crop_right = crop_left + (mask_right - mask_left)
                crop_bottom = crop_top + (mask_bottom - mask_top)
                
                if crop_right <= adjusted_mask.size[0] and crop_bottom <= adjusted_mask.size[1]:
                    mask_crop = adjusted_mask.crop((crop_left, crop_top, crop_right, crop_bottom))
                    full_size_mask.paste(mask_crop, (mask_left, mask_top))
            
            line_mask_pil = Image.new("L", composite_pil.size, 0)
        
        composite_pil = composite_pil.convert("RGB")
        
        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        mask_np = np.array(full_size_mask).astype(np.float32) / 255.0
        line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0
        
        if len(composite_np.shape) == 2:
            composite_np = np.stack([composite_np] * 3, axis=-1)
        
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)
        
        # 创建裁剪后的输出（仅包含背景图尺寸范围）
        bj_x_start = (canvas_width - bj_pil.size[0]) // 2
        bj_y_start = (canvas_height - bj_pil.size[1]) // 2
        bj_x_end = bj_x_start + bj_pil.size[0]
        bj_y_end = bj_y_start + bj_pil.size[1]
        
        cropped_composite = composite_tensor[:, bj_y_start:bj_y_end, bj_x_start:bj_x_end, :]
        
        return ( cropped_composite, mask_tensor,composite_tensor, line_mask_tensor, )



class Image_Resize2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),

                "model_scale": (["None"] +folder_paths.get_filename_list("upscale_models"), {"default": "None"}  ),                
                "pixel_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),

                "width_max": ("INT", { "default": 512, "min": 0, "max": 99999, "step": 1, }),
                "height_max": ("INT", { "default": 512, "min": 0, "max": 99999, "step": 1, }),

                "keep_ratio": ("BOOLEAN", { "default": False }),
                "divisible_by": ("INT", { "default": 8, "min": 0, "max": 512, "step": 1, }),
            },
            "optional" : {
                "get_image_size": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def resize(self, image, width_max, height_max, keep_ratio, divisible_by,pixel_method="bilinear", model_scale=None, get_image_size=None,):

        if len(image.shape) == 3:
            H, W, C = image.shape
        else:  
            B, H, W, C = image.shape

        crop = "center"

        if get_image_size is not None:
            _, height_max, width_max, _ = get_image_size.shape
        

        if keep_ratio and get_image_size is None:

                if width_max == 0 and height_max != 0:
                    ratio = height_max / H
                    width_max = round(W * ratio)
                elif height_max == 0 and width_max != 0:
                    ratio = width_max / W
                    height_max = round(H * ratio)
                elif width_max != 0 and height_max != 0:

                    ratio = min(width_max / W, height_max / H)
                    width_max = round(W * ratio)
                    height_max = round(H * ratio)
        else:
            if width_max == 0:
                width_max = W
            if height_max == 0:
                height_max = H
    
        if model_scale != "None":
            model = UpscaleModelLoader().load_model(model_scale)[0]
            image = ImageUpscaleWithModel().upscale(model, image)[0]     
 
        if divisible_by > 1 and get_image_size is None:
            width_max = width_max - (width_max % divisible_by)
            height_max = height_max - (height_max % divisible_by)
        
         
        image = image.movedim(-1,1)
        image = common_upscale(image, width_max, height_max, pixel_method, crop)
        image = image.movedim(1,-1)

        return(image,image.shape[2], image.shape[1])





#region----------------------尺寸调整组合---------


class Image_tensor_Converter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "convert_to_pil": ("BOOLEAN", {"default": True, "label_on": "启用", "label_off": "禁用"}),
                "multiple": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1}),
                "convert_to_single": ("BOOLEAN", {"default": True,}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert_precision"
    CATEGORY = "Apt_Preset/image/😺backup"

    def convert_precision(self, image, convert_to_pil=True, multiple=8, convert_to_single=False):
        batch_size = image.shape[0]
        converted_images = []
        
        for i in range(batch_size):
            single_image = image[i]
            
            # 确保图像是3通道RGB格式
            if single_image.shape[2] == 1:  # 处理单通道图像
                single_image = single_image.repeat(1, 1, 3)  # 转换为3通道
            elif single_image.shape[2] == 4:  # 处理RGBA图像
                single_image = single_image[:, :, :3]  # 保留RGB通道
            
            h, w = single_image.shape[:2]
            new_h = ((h + multiple - 1) // multiple) * multiple
            new_w = ((w + multiple - 1) // multiple) * multiple
            
            if new_h != h or new_w != w:
                pil_image = tensor2pil(single_image)
                pil_image = pil_image.resize((new_w, new_h), Image.LANCZOS)
                single_image = pil2tensor(pil_image)
            
            if convert_to_pil:
                # 确保数据类型正确
                pil_image = tensor2pil(single_image)
                # 强制转换为RGB模式，避免RGBA或其他模式问题
                if pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")
                single_image = pil2tensor(pil_image)
            
            converted_images.append(single_image.unsqueeze(0))
        
        # 拼接处理后的图像
        result = torch.cat(converted_images, dim=0)
        
        # 如果需要转为单张图，取第一张并移除多余的批次维度
        if convert_to_single and result.shape[0] > 0:
            # 将形状从 (1, H, W, 3) 转换为 (H, W, 3)
            result = result.squeeze(0)
        
        return (result,)



class Image_Resize_longsize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", {"default": 512, "min": 0, "step": 1, "max": 99999}),
                "interpolation_mode":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/image"

    def execute(self, image: torch.Tensor, size: int, interpolation_mode: str):
        assert isinstance(image, torch.Tensor)
        assert isinstance(size, int)
        assert isinstance(interpolation_mode, str)

        interpolation_modes = {
            "bicubic": Image.BICUBIC,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST,
            "nearest exact": Image.NEAREST,
        }
        interpolation_mode = interpolation_modes[interpolation_mode]

        _, h, w, _ = image.shape

        if h >= w:
            new_h = size
            new_w = round(w * new_h / h)
        else:  # h < w
            new_w = size
            new_h = round(h * new_w / w)

        # Convert to PIL images, resize, and convert back to tensors
        resized_images = []
        for i in range(image.shape[0]):
            pil_image = tensor2pil(image[i])
            resized_pil_image = pil_image.resize((new_w, new_h), interpolation_mode)
            resized_tensor = pil2tensor(resized_pil_image)
            resized_images.append(resized_tensor)
        
        resized_batch = torch.cat(resized_images, dim=0)
        return (resized_batch,)



class Image_Resize_sum_data:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "stitch": ("STITCH3",),
            },
            "optional": {
                "image": ("IMAGE",),  # 用于计算缩放因子的图像输入
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "width",          # 排除填充的有效宽度（final_width * scale_factor）
        "height",         # 排除填充的有效高度（final_height * scale_factor）
        "x_offset",       # pad模式时有效图像左上角X坐标 * scale_factor
        "y_offset",       # pad模式时有效图像左上角Y坐标 * scale_factor
        "pad_left",       # 左侧填充像素数 * scale_factor
        "pad_right",      # 右侧填充像素数 * scale_factor
        "pad_top",        # 顶部填充像素数 * scale_factor
        "pad_bottom",     # 底部填充像素数 * scale_factor
        "full_width",     # 包含填充的输出图像实际宽度 * scale_factor
        "full_height",    # 包含填充的输出图像实际高度 * scale_factor
        "scale_factor"    # 计算得到的缩放因子（image宽度 / full_width）
    )
    FUNCTION = "extract_info"
    CATEGORY = "Apt_Preset/image/😺backup"
    DESCRIPTION = """
    从Image_Resize_sum输出的stitch信息中提取关键参数，并根据输入图像自动计算缩放因子：
    1. 缩放因子 = 输入图像宽度 / 原始全宽（包含填充）
    2. 所有输出参数会乘以缩放因子后取整
    3. 若未提供输入图像，缩放因子默认为1.0
    """

    def extract_info(self, stitch: dict, image: torch.Tensor = None) -> Tuple[int, int, int, int, int, int, int, int, int, int, float]:
        # 提取基础信息
        valid_width = stitch.get("final_size", (0, 0))[0]
        valid_height = stitch.get("final_size", (0, 0))[1]
        
        pad_left = stitch.get("pad_info", (0, 0, 0, 0))[0]
        pad_right = stitch.get("pad_info", (0, 0, 0, 0))[1]
        pad_top = stitch.get("pad_info", (0, 0, 0, 0))[2]
        pad_bottom = stitch.get("pad_info", (0, 0, 0, 0))[3]
        
        full_width = valid_width + pad_left + pad_right
        full_height = valid_height + pad_top + pad_bottom
        
        x_offset, y_offset = stitch.get("image_position", (0, 0))

        # 计算缩放因子：image宽度 / full_width（若image存在且full_width不为0）
        if image is not None and full_width > 0:
            # 获取输入图像的宽度（处理批次和单张图像情况）
            if len(image.shape) == 4:  # 批次图像：(B, H, W, C)
                img_width = image.shape[2]
            else:  # 单张图像：(H, W, C)
                img_width = image.shape[1]
            scale_factor = img_width / full_width
        else:
            scale_factor = 1.0  # 默认缩放因子

        # 应用缩放并取整（四舍五入）
        scaled = lambda x: int(round(x * scale_factor))
        
        return (
            scaled(valid_width),
            scaled(valid_height),
            scaled(x_offset),
            scaled(y_offset),
            scaled(pad_left),
            scaled(pad_right),
            scaled(pad_top),
            scaled(pad_bottom),
            scaled(full_width),
            scaled(full_height),
            round(scale_factor, 6)  # 保留6位小数，避免精度问题
        )
    


class Image_pad_restore:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "image": ("IMAGE",),
                "stitch": ("STITCH3",),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image", )
    FUNCTION = "image_crop"
    CATEGORY = "Apt_Preset/image"


    def calculate_scale_factor(self, image: torch.Tensor, stitch: dict) -> float:
        pad_left = stitch.get("pad_info", (0, 0, 0, 0))[0]
        pad_right = stitch.get("pad_info", (0, 0, 0, 0))[1]
        valid_width = stitch.get("final_size", (0, 0))[0]
        full_width = valid_width + pad_left + pad_right
        
        if full_width <= 0:
            return 1.0
            
        if len(image.shape) == 4:
            img_width = image.shape[2]
        else:
            img_width = image.shape[1]
            
        return img_width / full_width


    def extract_info(self, stitch: dict, scale_factor: float) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        valid_width = stitch.get("final_size", (0, 0))[0]
        valid_height = stitch.get("final_size", (0, 0))[1]
        
        pad_left = stitch.get("pad_info", (0, 0, 0, 0))[0]
        pad_right = stitch.get("pad_info", (0, 0, 0, 0))[1]
        pad_top = stitch.get("pad_info", (0, 0, 0, 0))[2]
        pad_bottom = stitch.get("pad_info", (0, 0, 0, 0))[3]
        
        full_width = valid_width + pad_left + pad_right
        full_height = valid_height + pad_top + pad_bottom
        
        x_offset, y_offset = stitch.get("image_position", (0, 0))

        scaled = lambda x: int(round(x * scale_factor))
        
        return (
            scaled(valid_width),
            scaled(valid_height),
            scaled(x_offset),
            scaled(y_offset),
            scaled(pad_left),
            scaled(pad_right),
            scaled(pad_top),
            scaled(pad_bottom),
            scaled(full_width),
            scaled(full_height)
        )
    

    def image_crop(self, image, stitch):
        scale_factor = self.calculate_scale_factor(image, stitch)
        valid_width, valid_height, x_offset, y_offset, _, _, _, _, _, _ = self.extract_info(stitch, scale_factor)

        x = min(x_offset, image.shape[2] - 1)
        y = min(y_offset, image.shape[1] - 1)
        to_x = valid_width + x
        to_y = valid_height + y
        
        to_x = min(to_x, image.shape[2])
        to_y = min(to_y, image.shape[1])
        
        img = image[:, y:to_y, x:to_x, :]
     

        return (img,)
    


class Image_Resize_sum:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": ("INT", { "default": 512, "min": 0, "max": 9999, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 0, "max": 9999, "step": 1, }),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "keep_proportion": (["resize", "stretch", "pad", "pad_edge", "crop"], ),
                "pad_color": (["black", "white", "red", "green", "blue", "gray"], { "default": "black" }),
                "crop_position": (["center", "top", "bottom", "left", "right"], { "default": "center" }),
                "divisible_by": ("INT", { "default": 2, "min": 0, "max": 512, "step": 1, }),
                "pad_mask_remove": ("BOOLEAN", {"default": True,}),
            },
            "optional" : {
                "mask": ("MASK",),
                "get_image_size": ("IMAGE",),
                "mask_stack": ("MASK_STACK2",),
            },

        }

    # 增加了remove_pad_mask输出
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH3", "FLOAT", )
    RETURN_NAMES = ("IMAGE", "mask", "stitch",  "scale_factor", )
    FUNCTION = "resize"
    CATEGORY = "Apt_Preset/image"

    DESCRIPTION = """
    - 输入参数：
    - resize：按比例缩放图像至宽和高的限制范围，保持宽高比，不填充、不裁剪
    - stretch：拉伸图像以完全匹配指定的宽度和高度，保持宽高比、像素扭曲
    - pad：按比例缩放图像后，在目标尺寸内居中放置，用指定颜色填充多余区域
    - pad_edge：与pad类似，但使用图像边缘像素颜色进行填充
    - crop：按目标尺寸比例裁剪原图像，然后缩放到指定尺寸
    - -----------------------  
    - 输出参数：
    - scale_factor：缩放倍率，用于精准还原，可以减少一次缩放导致的模糊
    - remove_pad_mask：移除填充部分的遮罩，保持画布尺寸不变
    """



    def resize(self, image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, get_image_size=None, mask=None, mask_stack=None,pad_mask_remove=True):
        if len(image.shape) == 3:
            B, H, W, C = 1, image.shape[0], image.shape[1], image.shape[2]
            original_image = image.unsqueeze(0)
        else:  
            B, H, W, C = image.shape
            original_image = image.clone()
            
        original_H, original_W = H, W

        if width == 0:
            width = W
        if height == 0:
            height = H

        if get_image_size is not None:
            _, height, width, _ = get_image_size.shape
        
        new_width, new_height = width, height
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        crop_x, crop_y, crop_w, crop_h = 0, 0, W, H
        scale_factor = 1.0
        
        processed_mask = mask
        if mask is not None and mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask_stack
            
            separated_result = Mask_transform_sum().separate(  
                bg_mode="crop_image", 
                mask_mode=mask_mode,
                ignore_threshold=0, 
                opacity=1, 
                outline_thickness=1, 
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0, 
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min, 
                mask_max=mask_max,
                base_image=image.clone(), 
                mask=mask, 
                crop_to_mask=False,
                divisible_by=1
            )
            processed_mask = separated_result[1]
        
        if keep_proportion == "resize" or keep_proportion.startswith("pad"):
            if width == 0 and height != 0:
                scale_factor = height / H
                new_width = round(W * scale_factor)
                new_height = height
            elif height == 0 and width != 0:
                scale_factor = width / W
                new_width = width
                new_height = round(H * scale_factor)
            elif width != 0 and height != 0:
                scale_factor = min(width / W, height / H)
                new_width = round(W * scale_factor)
                new_height = round(H * scale_factor)

            if keep_proportion.startswith("pad"):
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

        elif keep_proportion == "crop":
            old_aspect = W / H
            new_aspect = width / height
            
            if old_aspect > new_aspect:
                crop_h = H
                crop_w = round(H * new_aspect)
                scale_factor = height / H
            else:
                crop_w = W
                crop_h = round(W / new_aspect)
                scale_factor = width / W
            
            if crop_position == "center":
                crop_x = (W - crop_w) // 2
                crop_y = (H - crop_h) // 2
            elif crop_position == "top":
                crop_x = (W - crop_w) // 2
                crop_y = 0
            elif crop_position == "bottom":
                crop_x = (W - crop_w) // 2
                crop_y = H - crop_h
            elif crop_position == "left":
                crop_x = 0
                crop_y = (H - crop_h) // 2
            elif crop_position == "right":
                crop_x = W - crop_w
                crop_y = (H - crop_h) // 2

        final_width = new_width
        final_height = new_height
        if divisible_by > 1:
            final_width = final_width - (final_width % divisible_by)
            final_height = final_height - (final_height % divisible_by)
            if new_width != 0:
                scale_factor *= (final_width / new_width)
            if new_height != 0:
                scale_factor *= (final_height / new_height)

        out_image = image.clone()
        out_mask = processed_mask.clone() if processed_mask is not None else None
        padding_mask = None

        if keep_proportion == "crop":
            out_image = out_image.narrow(-2, crop_x, crop_w).narrow(-3, crop_y, crop_h)
            if out_mask is not None:
                out_mask = out_mask.narrow(-1, crop_x, crop_w).narrow(-2, crop_y, crop_h)

        out_image = common_upscale(
            out_image.movedim(-1, 1),
            final_width,
            final_height,
            upscale_method,
            crop="disabled"
        ).movedim(1, -1)

        if out_mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(
                    out_mask.unsqueeze(1).repeat(1, 3, 1, 1),
                    final_width,
                    final_height,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)[:, :, :, 0]
            else:
                out_mask = common_upscale(
                    out_mask.unsqueeze(1),
                    final_width,
                    final_height,
                    upscale_method,
                    crop="disabled"
                ).squeeze(1)

        # 保存原始out_mask用于创建remove_pad_mask
        original_out_mask = out_mask.clone() if out_mask is not None else None

        if keep_proportion.startswith("pad") and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            padded_width = final_width + pad_left + pad_right
            padded_height = final_height + pad_top + pad_bottom
            if divisible_by > 1:
                width_remainder = padded_width % divisible_by
                height_remainder = padded_height % divisible_by
                if width_remainder > 0:
                    extra_width = divisible_by - width_remainder
                    pad_right += extra_width
                    padded_width += extra_width
                if height_remainder > 0:
                    extra_height = divisible_by - height_remainder
                    pad_bottom += extra_height
                    padded_height += extra_height
            
            color_map = {
                "black": "0, 0, 0",
                "white": "255, 255, 255",
                "red": "255, 0, 0",
                "green": "0, 255, 0",
                "blue": "0, 0, 255",
                "gray": "128, 128, 128"
            }
            pad_color_value = color_map[pad_color]
            
            out_image, padding_mask = self.resize_pad(
                out_image,
                pad_left,
                pad_right,
                pad_top,
                pad_bottom,
                0,
                pad_color_value,
                "edge" if keep_proportion == "pad_edge" else "color"
            )
            
            if out_mask is not None:
                out_mask = out_mask.unsqueeze(1).repeat(1, 3, 1, 1).movedim(1, -1)
                out_mask, _ = self.resize_pad(
                    out_mask,
                    pad_left,
                    pad_right,
                    pad_top,
                    pad_bottom,
                    0,
                    pad_color_value,
                    "edge" if keep_proportion == "pad_edge" else "color"
                )
                out_mask = out_mask[:, :, :, 0]
            else:
                out_mask = torch.ones((B, padded_height, padded_width), dtype=out_image.dtype, device=out_image.device)
                out_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = 0.0

        if out_mask is None:
            if keep_proportion != "crop":
                out_mask = torch.zeros((out_image.shape[0], out_image.shape[1], out_image.shape[2]), dtype=torch.float32)
            else:
                out_mask = torch.zeros((out_image.shape[0], out_image.shape[1], out_image.shape[2]), dtype=torch.float32)

        if padding_mask is not None:
            composite_mask = torch.clamp(padding_mask + out_mask, 0, 1)
        else:
            composite_mask = out_mask.clone()

        if keep_proportion.startswith("pad") and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
            # 获取最终尺寸
            final_padded_height, final_padded_width = composite_mask.shape[1], composite_mask.shape[2]

            remove_pad_mask = torch.zeros_like(composite_mask)
            
            if original_out_mask is not None:
                if original_out_mask.shape[1] != final_height or original_out_mask.shape[2] != final_width:
                    resized_original_mask = common_upscale(
                        original_out_mask.unsqueeze(1),
                        final_width,
                        final_height,
                        upscale_method,
                        crop="disabled"
                    ).squeeze(1)
                else:
                    resized_original_mask = original_out_mask
        
                remove_pad_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = resized_original_mask
            else:
                remove_pad_mask[:, pad_top:pad_top+final_height, pad_left:pad_left+final_width] = 0.0
        else:
            remove_pad_mask = composite_mask.clone()

        stitch_info = {
            "original_image": original_image,
            "original_shape": (original_H, original_W),
            "resized_shape": (out_image.shape[1], out_image.shape[2]),
            "crop_position": (crop_x, crop_y),
            "crop_size": (crop_w, crop_h),
            "pad_info": (pad_left, pad_right, pad_top, pad_bottom),
            "keep_proportion": keep_proportion,
            "upscale_method": upscale_method,
            "scale_factor": scale_factor,
            "final_size": (final_width, final_height),
            "image_position": (pad_left, pad_top) if keep_proportion.startswith("pad") else (0, 0),
            "has_input_mask": mask is not None,
            "original_mask": mask.clone() if mask is not None else None
        }
        
        scale_factor = 1/scale_factor

        if pad_mask_remove:
           Fina_mask =  remove_pad_mask.cpu()
        else:
           Fina_mask =  composite_mask.cpu()

        return (out_image.cpu(), Fina_mask, stitch_info, scale_factor, )


    def resize_pad(self, image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape

        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        bg_color = [int(x.strip()) / 255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # 新增逻辑：判断是否需要跳过缩放
        should_skip_resize = False
        if target_width is not None and target_height is not None:
            # 判断长边是否已经等于目标尺寸
            current_long_side = max(W, H)
            target_long_side = max(target_width, target_height)
            if current_long_side == target_long_side:
                should_skip_resize = True

        if not should_skip_resize and target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "bilinear", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            pad_left = (target_width - W) // 2
            pad_right = target_width - W - pad_left
            pad_top = (target_height - H) // 2
            pad_bottom = target_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

        padded_width = W + pad_left + pad_right
        padded_height = H + pad_top + pad_bottom

        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
            if pad_mode == "edge":
                top_edge = image[b, 0, :, :]
                bottom_edge = image[b, H-1, :, :]
                left_edge = image[b, :, 0, :]
                right_edge = image[b, :, W-1, :]

                out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
            else:
                out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        padding_mask = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
        for m in range(B):
            padding_mask[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, padding_mask)



class Image_Resize_sum_restore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resized_image": ("IMAGE",),
                #"mask": ("MASK",),
                "stitch": ("STITCH3",),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
            },
        }

    CATEGORY = "Apt_Preset/image"
    # 添加原始图像到返回类型
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("restored_image", "restored_mask", "original_image")
    FUNCTION = "restore"

    def restore(self, resized_image, stitch, upscale_method="bicubic"):
        # 从stitch中提取关键信息
        original_h, original_w = stitch["original_shape"]
        pad_left, pad_right, pad_top, pad_bottom = stitch["pad_info"]
        keep_proportion = stitch["keep_proportion"]
        final_width, final_height = stitch["final_size"]  # 缩放后有效区域原始尺寸
        original_mask = stitch.get("original_mask")
        has_input_mask = stitch.get("has_input_mask", False)
        crop_x, crop_y = stitch["crop_position"]
        crop_w, crop_h = stitch["crop_size"]
        
        # 获取原始图像
        original_image = stitch.get("original_image", None)
        
        # 获取当前resized_image的实际尺寸
        current_b, current_h, current_w, current_c = resized_image.shape

        if keep_proportion.startswith("pad"):
            # 计算原始有效区域在填充后图像中的占比（用于处理比例变化）
            original_padded_w = final_width + pad_left + pad_right
            original_padded_h = final_height + pad_top + pad_bottom
            
            # 计算当前图像中有效区域的实际位置和尺寸（考虑比例变化）
            # 1. 计算缩放比例：当前图像 / 原始填充后尺寸
            scale_w = current_w / original_padded_w if original_padded_w != 0 else 1.0
            scale_h = current_h / original_padded_h if original_padded_h != 0 else 1.0
            
            # 2. 计算有效区域在当前图像中的位置（按比例映射）
            current_pad_left = int(round(pad_left * scale_w))
            current_pad_top = int(round(pad_top * scale_h))
            current_valid_w = int(round(final_width * scale_w))
            current_valid_h = int(round(final_height * scale_h))
            
            # 3. 安全裁剪：确保不超出当前图像范围
            valid_left = max(0, current_pad_left)
            valid_right = min(current_w, current_pad_left + current_valid_w)
            valid_top = max(0, current_pad_top)
            valid_bottom = min(current_h, current_pad_top + current_valid_h)
            
            # 4. 裁剪有效区域
            valid_image = resized_image[:, valid_top:valid_bottom, valid_left:valid_right, :]
            
            # 5. 单次缩放至原始尺寸（关键优化：只缩放一次）
            restored_image = common_upscale(
                valid_image.movedim(-1, 1),
                original_w, original_h,
                upscale_method,
                crop="disabled"
            ).movedim(1, -1)

        elif keep_proportion == "crop":
            # 处理crop模式的比例适配
            original_cropped_ratio = crop_w / crop_h if crop_h != 0 else 1.0
            current_ratio = current_w / current_h if current_h != 0 else 1.0
            
            # 计算需要裁剪的区域（保持原始裁剪比例）
            if abs(current_ratio - original_cropped_ratio) > 1e-6:
                if current_ratio > original_cropped_ratio:
                    # 当前图像更宽，按高度裁剪宽度
                    target_w = int(round(current_h * original_cropped_ratio))
                    crop_left = (current_w - target_w) // 2
                    crop_right = current_w - target_w - crop_left
                    valid_image = resized_image[:, :, crop_left:current_w - crop_right, :]
                else:
                    # 当前图像更高，按宽度裁剪高度
                    target_h = int(round(current_w / original_cropped_ratio))
                    crop_top = (current_h - target_h) // 2
                    crop_bottom = current_h - target_h - crop_top
                    valid_image = resized_image[:, crop_top:current_h - crop_bottom, :, :]
            else:
                valid_image = resized_image
            
            # 缩放至原始裁剪区域尺寸
            crop_restored = common_upscale(
                valid_image.movedim(-1, 1),
                crop_w, crop_h,
                upscale_method,
                crop="disabled"
            ).movedim(1, -1)
            
            # 放回原始图像位置
            if stitch.get("original_image") is not None:
                restored_image = stitch["original_image"].clone()
            else:
                restored_image = torch.zeros(
                    (current_b, original_h, original_w, current_c),
                    dtype=resized_image.dtype,
                    device=resized_image.device
                )
            restored_image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = crop_restored

        else:  # resize/stretch模式
            # 直接按原始尺寸比例缩放
            restored_image = common_upscale(
                resized_image.movedim(-1, 1),
                original_w, original_h,
                upscale_method,
                crop="disabled"
            ).movedim(1, -1)

        # 处理mask
        restored_mask = original_mask if (original_mask is not None and has_input_mask) else (
            torch.zeros((current_b, original_h, original_w), dtype=torch.float32, device=resized_image.device)
        )
       
        restored_image = convert_pil_image(restored_image)
        
        # 如果原始图像存在则返回，否则返回一个默认图像
        if original_image is not None:
            output_original_image = original_image
        else:
            output_original_image = torch.zeros((1, original_h, original_w, 3), dtype=torch.float32)

        return (restored_image.cpu(), restored_mask.cpu(), output_original_image.cpu())




#endregion----------------------------合并----------




#region--------------------裁切组合------------



class Image_transform_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "bj_img": ("IMAGE",),  
                "fj_img": ("IMAGE",),  
                "mask": ("MASK",),      
                "stitch": ("STITCH2",),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 1}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "edge_detection": ("BOOLEAN", {"default": False}),
                "edge_thickness": ("INT", {"default": 1, "min": 1, "max": 100, "step": 1}),
                "edge_color": (["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta"], {"default": "black"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": (BLEND_METHODS , {"default": "normal"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
            },
            "optional": {

            }
        }
    RETURN_TYPES = ("IMAGE","IMAGE", "MASK", "MASK", )
    RETURN_NAMES = ("composite","recover_img", "mask", "line_mask",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image"
    DESCRIPTION = """
-贴合中心是遮罩本地中心坐标，所以对接裁切节点时，会有偏心问题
-如果要消除偏心，则取消使用扩展宽等调节
"""


    def process( self, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, mask_expand, smoothness,
                opacity, blending_mode, blend_strength, bj_img=None, fj_img=None, stitch=None, mask=None,upscale_method="area"): 
        
        color_mapping = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }
        if fj_img is None: raise ValueError("前景图像(fj_img)是必需的输入")
        
        if bj_img is None:
            bj_img = fj_img.clone()
        
        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
        
        canvas_width, canvas_height = bj_pil.size
        canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2
        
        original_image_info = {
            "width": canvas_width,
            "height": canvas_height,
            "center_x": canvas_center_x,
            "center_y": canvas_center_y
        }
        
        stitch_original_position = None
        if stitch is not None:
            original_shape = stitch.get("original_shape", (canvas_height, canvas_width))
            crop_position = stitch.get("crop_position", (0, 0))
            crop_size = stitch.get("crop_size", (canvas_width, canvas_height))
            original_image_h, original_image_w = stitch["original_image_shape"]

            original_center_x = crop_position[0] + (crop_size[0] // 2)
            original_center_y = crop_position[1] + (crop_size[1] // 2)
            
            stitch_original_position = {
                "x": original_center_x,
                "y": original_center_y,
                "width": crop_size[0],
                "height": crop_size[1],
            }
        mask_tensor = None
        if mask is not None: 
            if hasattr(mask, 'convert'):
                mask_tensor = pil2tensor(mask.convert('L'))
            else:  
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask if len(mask.shape) <= 3 else mask.squeeze(-1) if mask.shape[-1] == 1 else mask
                else:
                    mask_tensor = mask

        separated_result = Mask_transform_sum().separate(  
            bg_mode="crop_image", 
            mask_mode="original",
            ignore_threshold=0, 
            opacity=1, 
            outline_thickness=1, 
            smoothness=smoothness,
            mask_expand=mask_expand,
            expand_width=0, 
            expand_height=0,
            rescale_crop=1.0,
            tapered_corners=True,
            mask_min=0, 
            mask_max=1,
            base_image=fj_img, 
            mask=mask_tensor, 
            crop_to_mask=False,
            divisible_by=1
        )

        fj_img = separated_result[0]
        mask = separated_result[1]
        
        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
            if mask_pil.size != fj_pil.size:
                mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
            
            fj_with_mask = fj_pil.copy()
            fj_with_mask.putalpha(mask_pil)
            
            bbox = mask_pil.getbbox()
            if bbox:
                fj_cropped = fj_with_mask.crop(bbox)
                mask_cropped = mask_pil.crop(bbox)
            else:
                fj_cropped = fj_with_mask
                mask_cropped = mask_pil
        else:
            mask_cropped = Image.new("L", fj_pil.size, 255)
            fj_cropped = fj_pil.copy()
            fj_cropped.putalpha(mask_cropped)
        
        cropped_width, cropped_height = fj_cropped.size
        mask_center_x, mask_center_y = cropped_width // 2, cropped_height // 2
        scale_x, scale_y = 1.0, 1.0
        adjusted_fj = fj_cropped
        adjusted_mask = mask_cropped

 #----------------------------------------------------------     
 
   
        adjusted_width, adjusted_height = adjusted_fj.size
        rotation = float(rotation)
        
        if rotation != 0 or scale != 1.0:
            adjusted_fj = adjusted_fj.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            adjusted_mask = adjusted_mask.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
            
            if scale != 1.0:
                new_size = (int(adjusted_fj.size[0] * scale), int(adjusted_fj.size[1] * scale))
                scale_x *= scale
                scale_y *= scale
                adjusted_fj = adjusted_fj.resize(new_size, Image.LANCZOS)
                adjusted_mask = adjusted_mask.resize(new_size, Image.LANCZOS)
            
            mask_center_x, mask_center_y = adjusted_fj.size[0] // 2, adjusted_fj.size[1] // 2
        
        if stitch is not None and stitch_original_position is not None:
            x_position = stitch_original_position["x"] - mask_center_x + x_offset
            y_position = stitch_original_position["y"] - mask_center_y + y_offset
        else:
            x_position = canvas_center_x - mask_center_x + x_offset
            y_position = canvas_center_y - mask_center_y + y_offset
        
        paste_x = max(0, x_position)
        paste_y = max(0, y_position)
        
        if opacity < 1.0:
            r, g, b, a = adjusted_fj.split()
            a = a.point(lambda p: p * opacity)
            adjusted_fj = Image.merge("RGBA", (r, g, b, a))
        
        # 修复前景图像裁剪坐标
        # 计算裁剪坐标
        left = max(0, -x_position)
        top = max(0, -y_position)
        right = min(adjusted_fj.size[0], canvas_width - x_position)
        bottom = min(adjusted_fj.size[1], canvas_height - y_position)
        
        # 确保坐标有效（左 <= 右，上 <= 下）
        left = min(left, right)
        top = min(top, bottom)
        right = max(left, right)
        bottom = max(top, bottom)
        
        # 执行裁剪
        cropped_fj = adjusted_fj.crop((left, top, right, bottom))
        
        if blending_mode != "normal":
            temp_img = Image.new('RGBA', bj_pil.size, (0, 0, 0, 0))
            temp_img.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
            composite_pil = Image.new('RGBA', bj_pil.size, (0, 0, 0, 0))
            
            for x in range(canvas_width):
                for y in range(canvas_height):
                    if temp_img.getpixel((x, y))[3] > 0:
                        bg_pixel = bj_pil.getpixel((x, y))
                        fg_pixel = temp_img.getpixel((x, y))
                        bg_pixel_img = Image.new('RGBA', (1, 1), bg_pixel)
                        fg_pixel_img = Image.new('RGBA', (1, 1), fg_pixel)
                        blended_pixel_img = apply_blending_mode(
                            bg_pixel_img, fg_pixel_img, blending_mode, blend_strength
                        )
                        composite_pil.putpixel((x, y), blended_pixel_img.getpixel((0, 0)))
                    else:
                        composite_pil.putpixel((x, y), bj_pil.getpixel((x, y)))
        else:
            composite_pil = bj_pil.copy()
            composite_pil.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
        
        if edge_detection:
            if edge_color in color_mapping:
                r, g, b = color_mapping[edge_color]
            else:
                r, g, b = 0, 0, 0
            
            threshold = 128
            full_size_mask = Image.new("L", composite_pil.size, 0)
            
            # 修复掩码裁剪坐标
            # 计算掩码裁剪坐标
            mask_left = max(0, -x_position)
            mask_top = max(0, -y_position)
            mask_right = min(adjusted_mask.size[0], canvas_width - x_position)
            mask_bottom = min(adjusted_mask.size[1], canvas_height - y_position)
            
            # 确保坐标有效
            mask_left = min(mask_left, mask_right)
            mask_top = min(mask_top, mask_bottom)
            mask_right = max(mask_left, mask_right)
            mask_bottom = max(mask_top, mask_bottom)
            
            # 执行掩码裁剪和粘贴
            full_size_mask.paste(adjusted_mask.crop((
                mask_left,
                mask_top,
                mask_right,
                mask_bottom
            )), (paste_x, paste_y))
            
            mask_array = np.array(full_size_mask)
            binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            binary_mask_pil = Image.fromarray(binary_mask)
            
            edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
            edge_draw = ImageDraw.Draw(edge_image)
            mask_cv = np.array(binary_mask_pil)
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                for i in range(edge_thickness):
                    points = [tuple(point[0]) for point in contour]
                    edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness-i+1)
            
            composite_pil = Image.alpha_composite(composite_pil, edge_image)
            edge_mask = np.zeros_like(mask_cv)
            cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
            line_mask_pil = Image.fromarray(edge_mask)
        else:
            full_size_mask = Image.new("L", composite_pil.size, 0)
            
            # 修复掩码裁剪坐标（else分支）
            # 计算掩码裁剪坐标
            mask_left = max(0, -x_position)
            mask_top = max(0, -y_position)
            mask_right = min(adjusted_mask.size[0], canvas_width - x_position)
            mask_bottom = min(adjusted_mask.size[1], canvas_height - y_position)
            
            # 确保坐标有效
            mask_left = min(mask_left, mask_right)
            mask_top = min(mask_top, mask_bottom)
            mask_right = max(mask_left, mask_right)
            mask_bottom = max(mask_top, mask_bottom)
            
            # 执行掩码裁剪和粘贴
            full_size_mask.paste(adjusted_mask.crop((
                mask_left,
                mask_top,
                mask_right,
                mask_bottom
            )), (paste_x, paste_y))
            line_mask_pil = Image.new("L", composite_pil.size, 0)
        
        composite_pil = Image.alpha_composite(bj_pil.convert("RGBA"), composite_pil)
        composite_pil = composite_pil.convert("RGB")
        
        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        mask_np = np.array(full_size_mask).astype(np.float32) / 255.0
        line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0
        
        if len(composite_np.shape) == 2:
            composite_np = np.stack([composite_np] * 3, axis=-1)
        
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)
        

        recover_img, Fina_mask, stitch_info, scale_factor = Image_Resize_sum().resize(
            image=composite_tensor,
            width=original_image_w, 
            height=original_image_h, 
            keep_proportion="stretch",
            upscale_method= upscale_method, 
            divisible_by=1,  
            pad_color="black", 
            crop_position="center", 
            get_image_size=None, 
            mask=None, 
            mask_stack=None,
            pad_mask_remove=True
        )
        return (composite_tensor,recover_img, mask_tensor, line_mask_tensor, )



class Image_Solo_data:
    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "stitch": ("STITCH2",),  # 仅依赖现有STITCH2，无需额外输入
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "valid_width",    # 裁切图有效宽（crop_size[0]）
        "valid_height",   # 裁切图有效高（crop_size[1]）
        "x_offset",       # 裁切图在原图左上角X坐标（crop_position[0]）
        "y_offset",       # 裁切图在原图左上角Y坐标（crop_position[1]）
        "full_width",     # 原图宽（original_shape[1]）
        "full_height",    # 原图高（original_shape[0]）
        "scale_factor"    # 输入长边 / 原始裁切图长边（核心调整）
    )
    FUNCTION = "extract_info"
    CATEGORY = "Apt_Preset/image/😺backup"


    def extract_info(self, stitch: dict) -> Tuple[int, int, int, int, int, int, float]:
        # 1. 提取原图尺寸（STITCH2现有字段：original_shape存储为(高, 宽)）
        original_height, original_width = stitch.get("original_shape", (0, 0))
        full_width = int(original_width)
        full_height = int(original_height)

        # 2. 提取裁切图尺寸与坐标（STITCH2现有字段）
        crop_width, crop_height = stitch.get("crop_size", (0, 0))
        valid_width = int(crop_width)
        valid_height = int(crop_height)
        x_offset, y_offset = stitch.get("crop_position", (0, 0))
        x_offset = int(x_offset)
        y_offset = int(y_offset)

        # 3. 提取计算缩放因子所需的两个长边（均为STITCH2现有字段）
        input_long_side = stitch.get("input_long_side", 512)  # 之前补充的「输入长边」
        crop_long_side = stitch.get("crop_long_side", max(valid_width, valid_height))  # 「原始裁切图长边」

        # 4. 计算缩放因子：输入长边 ÷ 原始裁切图长边（避免除以0）
        scale_factor = 1.0
        if crop_long_side > 0:
            scale_factor = round(  crop_long_side /input_long_side, 6)

        return (
            valid_width,
            valid_height,
            x_offset,
            y_offset,
            full_width,
            full_height,
            scale_factor
        )




class Image_solo_stitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "stitch": ("STITCH2",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1, "display": "slider"}),
                "blend_factor": ("FLOAT", {"default": 1.0,"min": 0.0,"max": 1.0,"step": 0.01}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stitch_mode": (["crop_mask", "crop_image"], {"default": "crop_mask"}),
                "recover_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
            },
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE","IMAGE","IMAGE") 
    RETURN_NAMES = ("image","recover_image","original_image") 
    FUNCTION = "inpaint_stitch"


    def apply_smooth_blur(self, image, mask, smoothness, bg_color="Alpha"):
        batch_size = image.shape[0]
        result_images = []
        smoothed_masks = []       
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "gray": (128, 128, 128) }       
        for i in range(batch_size):
            current_image = image[i].clone()
            current_mask = mask[i] if i < mask.shape[0] else mask[0]
            if smoothness > 0:
                mask_tensor = smoothness_mask(current_mask, smoothness)
            else:
                mask_tensor = current_mask.clone()
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze()
                while mask_tensor.dim() > 2:
                    mask_tensor = mask_tensor.squeeze(0)          
            smoothed_mask = mask_tensor.clone()
            unblurred_tensor = current_image.clone()
            if current_image.shape[-1] != 3:
                if current_image.shape[-1] == 4:
                    current_image = current_image[:, :, :3]
                    unblurred_tensor = unblurred_tensor[:, :, :3]
                elif current_image.shape[-1] == 1:
                    current_image = current_image.repeat(1, 1, 3)
                    unblurred_tensor = unblurred_tensor.repeat(1, 1, 3)         
            mask_expanded = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)
            result_tensor = current_image * mask_expanded + unblurred_tensor * (1 - mask_expanded)         
            if bg_color != "Alpha":
                bg_tensor = torch.zeros_like(current_image)
                if bg_color in color_map:
                    r, g, b = color_map[bg_color]
                    bg_tensor[:, :, 0] = r / 255.0
                    bg_tensor[:, :, 1] = g / 255.0
                    bg_tensor[:, :, 2] = b / 255.0
                
                result_tensor = result_tensor * mask_expanded + bg_tensor * (1 - mask_expanded)            
            result_images.append(result_tensor.unsqueeze(0))
            smoothed_masks.append(smoothed_mask.unsqueeze(0))       
        final_image = torch.cat(result_images, dim=0)
        final_mask = torch.cat(smoothed_masks, dim=0)
        return (final_image, final_mask)

    def create_feather_mask(self, width, height, feather_size):
        if feather_size <= 0:
            return np.ones((height, width), dtype=np.float32)
        
        feather = min(feather_size, min(width, height) // 2)
        mask = np.ones((height, width), dtype=np.float32)
        
        for y in range(feather):
            mask[y, :] = y / feather
        for y in range(height - feather, height):
            mask[y, :] = (height - y) / feather
        for x in range(feather):
            mask[:, x] = np.minimum(mask[:, x], x / feather)
        for x in range(width - feather, width):
            mask[:, x] = np.minimum(mask[:, x], (width - x) / feather)
            
        return mask

    def inpaint_stitch(self, inpainted_image, smoothness, mask, stitch, blend_factor, blend_mode, opacity, stitch_mode, recover_method):
        original_h, original_w = stitch["original_shape"]
        crop_x, crop_y = stitch["crop_position"]
        crop_w, crop_h = stitch["crop_size"]
        mask_crop_x, mask_crop_y, mask_crop_x2, mask_crop_y2 = stitch["mask_cropped_position"]
        original_image_h, original_image_w = stitch["original_image_shape"]

        # 从stitch字典中获取背景图像数据
        if "bj_image" in stitch:
            # 如果stitch中包含背景图像数据，则使用它
            bj_image = stitch["bj_image"]
        else:
            # 如果stitch中不包含背景图像，则创建一个黑色背景
            bj_image = torch.zeros((1, original_h, original_w, 3), dtype=torch.float32)

        # 获取原始输入图像
        if "original_image" in stitch:
            original_image = stitch["original_image"]
        else:
            # 如果stitch中不包含原始图像，则创建一个黑色背景作为默认值
            original_image = torch.zeros((1, original_image_h, original_image_w, 3), dtype=torch.float32)

        if opacity < 1.0:
            inpainted_image = inpainted_image * opacity

        if inpainted_image.shape[1:3] != mask.shape[1:3]:               
            mask = F.interpolate(mask.unsqueeze(1), size=(inpainted_image.shape[1], inpainted_image.shape[2]), mode='nearest').squeeze(1)

        inpainted_np = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        background_np = (bj_image[0].cpu().numpy() * 255).astype(np.uint8)

        inpainted_resized = cv2.resize(inpainted_np, (crop_w, crop_h))
        mask_resized = cv2.resize(mask_np, (crop_w, crop_h))
        background_resized = cv2.resize(background_np, (original_w, original_h))

        result = np.zeros((original_h, original_w, 4), dtype=np.uint8)
        result[:, :, :3] = background_resized.copy()
        result[:, :, 3] = 255

        if stitch_mode == "crop_mask":
            inpainted_image, mask = self.apply_smooth_blur(inpainted_image, mask, smoothness, bg_color="Alpha")
            inpainted_blurred = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
            mask_blurred = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            
            inpainted_blurred = cv2.resize(inpainted_blurred, (crop_w, crop_h))
            mask_blurred = cv2.resize(mask_blurred, (crop_w, crop_h))
            
            mask_content = mask_blurred[mask_crop_y:mask_crop_y2, mask_crop_x:mask_crop_x2]
            inpaint_content = inpainted_blurred[mask_crop_y:mask_crop_y2, mask_crop_x:mask_crop_x2]
            
            if mask_content.size == 0 or inpaint_content.size == 0:
                print("Warning: Mask content is empty, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值
            
            paste_x_start = max(0, crop_x + mask_crop_x)
            paste_x_end = min(original_w, crop_x + mask_crop_x2)
            paste_y_start = max(0, crop_y + mask_crop_y)
            paste_y_end = min(original_h, crop_y + mask_crop_y2)
            
            if paste_x_start >= paste_x_end or paste_y_start >= paste_y_end:
                print("Warning: Invalid paste region, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值
            
            alpha = mask_content / 255.0
            expected_h = paste_y_end - paste_y_start
            expected_w = paste_x_end - paste_x_start
            
            if alpha.shape[0] != expected_h or alpha.shape[1] != expected_w:
                alpha = cv2.resize(alpha, (expected_w, expected_h))
            
            alpha = np.expand_dims(alpha, axis=-1)
            
            background_content = result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3]
            
            if (background_content.shape[0] != alpha.shape[0] or 
                background_content.shape[1] != alpha.shape[1]):
                print("Warning: Dimension mismatch after processing, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值
            
            if (inpaint_content.shape[0] < alpha.shape[0] or 
                inpaint_content.shape[1] < alpha.shape[1]):
                inpaint_content = cv2.resize(inpaint_content, (alpha.shape[1], alpha.shape[0]))
            
            inpaint_content = inpaint_content[:alpha.shape[0], :alpha.shape[1]]
            
            if len(inpaint_content.shape) == 3 and inpaint_content.shape[2] > 3:
                inpaint_content = inpaint_content[:, :, :3]
            elif len(inpaint_content.shape) == 2:
                inpaint_content = np.stack([inpaint_content, inpaint_content, inpaint_content], axis=-1)
            
            if len(background_content.shape) == 2:
                background_content = np.stack([background_content, background_content, background_content], axis=-1)
            elif len(background_content.shape) == 3 and background_content.shape[2] > 3:
                background_content = background_content[:, :, :3]
            
            try:
                blended = (inpaint_content * alpha + background_content * (1 - alpha)).astype(np.uint8)
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3] = blended
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, 3] = (alpha * 255).astype(np.uint8).squeeze()
            except Exception as e:
                print(f"Warning: Error during blending operation: {e}, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值

        else:
            feather_mask = self.create_feather_mask(crop_w, crop_h, smoothness)
            
            paste_x_start = max(0, crop_x)
            paste_x_end = min(original_w, crop_x + crop_w)
            paste_y_start = max(0, crop_y)
            paste_y_end = min(original_h, crop_y + crop_h)
            
            inpaint_content = inpainted_resized[
                max(0, paste_y_start - crop_y) : max(0, paste_y_end - crop_y),
                max(0, paste_x_start - crop_x) : max(0, paste_x_end - crop_x)
            ]
            
            if inpaint_content.size == 0:
                print("Warning: Inpaint content is empty in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值
            
            if paste_x_start >= paste_x_end or paste_y_start >= paste_y_end:
                print("Warning: Invalid paste region in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值
            
            alpha_mask = feather_mask[
                max(0, paste_y_start - crop_y) : max(0, paste_y_end - crop_y),
                max(0, paste_x_start - crop_x) : max(0, paste_x_end - crop_x)
            ]
            alpha = np.expand_dims(alpha_mask, axis=-1)
            
            background_content = result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3]
            
            if (background_content.shape[0] != alpha.shape[0] or 
                background_content.shape[1] != alpha.shape[1] or
                inpaint_content.shape[0] != alpha.shape[0] or
                inpaint_content.shape[1] != alpha.shape[1]):
                print("Warning: Dimension mismatch in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值
            
            if len(inpaint_content.shape) == 3 and inpaint_content.shape[2] > 3:
                inpaint_content = inpaint_content[:, :, :3]
            elif len(inpaint_content.shape) == 2:
                inpaint_content = np.stack([inpaint_content, inpaint_content, inpaint_content], axis=-1)
            
            if len(background_content.shape) == 2:
                background_content = np.stack([background_content, background_content, background_content], axis=-1)
            elif len(background_content.shape) == 3 and background_content.shape[2] > 3:
                background_content = background_content[:, :, :3]
            
            try:
                blended = (inpaint_content * alpha + background_content * (1 - alpha)).astype(np.uint8)
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, :3] = blended
                result[paste_y_start:paste_y_end, paste_x_start:paste_x_end, 3] = (alpha * 255).astype(np.uint8).squeeze()
            except Exception as e:
                print(f"Warning: Error during blending operation in crop_image mode: {e}, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                return (fimage, fimage, original_image)  # 返回三个值

        final_rgb = result[:, :, :3]
        final_image_tensor = torch.from_numpy(final_rgb / 255.0).float().unsqueeze(0)
        fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]

        recover_img, Fina_mask, stitch_info, scale_factor = Image_Resize_sum().resize(
            image=fimage,
            width=original_image_w, 
            height=original_image_h, 
            keep_proportion="stretch",
            upscale_method=recover_method, 
            divisible_by=1, 
            pad_color="black", 
            crop_position="center", 
            get_image_size=None, 
            mask=None, 
            mask_stack=None,
            pad_mask_remove=True)

        fimage = convert_pil_image(fimage)
        recover_img = convert_pil_image(recover_img)

        return (fimage, recover_img, original_image)  # 返回三个值



class Image_solo_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mode": (["no_crop", "no_scale_crop", "scale_crop_image", "scale_bj_image"], {"default": "no_scale_crop"}),
                "long_side": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear"}),
                "expand_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "divisible_by": ("INT", {"default": 2, "min": 0, "max": 128, "step": 2}),

            },
            "optional": {
                "mask": ("MASK",),
                "mask_stack": ("MASK_STACK2",),
                "crop_img_bj": (["image", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"], {"default": "image"}),
                "auto_expand_square": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("bj_image", "bj_mask", "crop_image", "crop_mask", "stitch")
    FUNCTION = "inpaint_crop"
    DESCRIPTION = """
    - no_scale_crop: 原始裁切图。不支持缩放
    - scale_crop_image: 原始裁切图的长边缩放。
    - scale_bj_image: 背景图的长边缩放。不支持扩展
    - no_crop: 不进行裁剪，仅处理遮罩。
    - auto_expand_square自动扩展正方形，仅no_scale_crop和scale_crop_image模式
    - 遮罩控制: 微调尺寸【目标尺寸相差2~8个像素时】
    """

    def get_mask_bounding_box(self, mask):
        mask_np = (mask[0].cpu().numpy() > 0.5).astype(np.uint8)
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            raise ValueError("Mask is empty")
        x, y, w, h = cv2.boundingRect(coords)
        return w, h, x, y

    def process_resize(self, image, mask, crop_mode, long_side, divisible_by, upscale_method="bilinear"):
        batch_size, img_height, img_width, channels = image.shape
        image_ratio = img_width / img_height
        mask_w, mask_h, mask_x, mask_y = self.get_mask_bounding_box(mask)
        mask_ratio = mask_w / mask_h
        new_width, new_height = img_width, img_height

        if crop_mode == "scale_bj_image":
            if img_width >= img_height:
                new_width = long_side
                new_height = int(new_width / image_ratio)
            else:
                new_height = long_side
                new_width = int(new_height * image_ratio)
        elif crop_mode == "scale_crop_image":
            if mask_w >= mask_h:
                new_mask_width = long_side
                new_mask_height = int(new_mask_width / mask_ratio)
                mask_scale = new_mask_width / mask_w
            else:
                new_mask_height = long_side
                new_mask_width = int(new_mask_height * mask_ratio)
                mask_scale = new_mask_height / mask_h
            new_width = int(img_width * mask_scale)
            new_height = int(img_height * mask_scale)
        elif crop_mode == "no_crop":
            new_width, new_height = img_width, img_height

        if divisible_by > 1:
            if new_width % divisible_by != 0:
                new_width += (divisible_by - new_width % divisible_by)
            if new_height % divisible_by != 0:
                new_height += (divisible_by - new_height % divisible_by)
        else:
            if new_width % 2 != 0:
                new_width += 1
            if new_height % 2 != 0:
                new_height += 1

        torch_upscale_method = upscale_method
        if upscale_method == "lanczos":
            torch_upscale_method = "bicubic"

        image_t = image.permute(0, 3, 1, 2)
        crop_image = F.interpolate(image_t, size=(new_height, new_width), mode=upscale_method, align_corners=False if upscale_method in ["bilinear", "bicubic"] else None)
        crop_image = crop_image.permute(0, 2, 3, 1)

        mask_t = mask.unsqueeze(1) if mask.ndim == 3 else mask
        crop_mask = F.interpolate(mask_t, size=(new_height, new_width), mode="nearest")
        crop_mask = crop_mask.squeeze(1)

        return (crop_image, crop_mask)

    def inpaint_crop(self, image, crop_mode, long_side, upscale_method="bilinear",
                    expand_width=0, expand_height=0, auto_expand_square=False, divisible_by=2,
                    mask=None, mask_stack=None, crop_img_bj="image"):
        colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5)
        }
        batch_size, height, width, _ = image.shape
        if mask is None:
            mask = torch.ones((batch_size, height, width), dtype=torch.float32, device=image.device)

        if mask_stack is not None:
            mask_mode, smoothness, mask_expand, mask_min, mask_max = mask_stack
            if hasattr(mask, 'convert'):
                mask_tensor = pil2tensor(mask.convert('L'))
            else:
                if isinstance(mask, torch.Tensor):
                    mask_tensor = mask if len(mask.shape) <= 3 else mask.squeeze(-1) if mask.shape[-1] == 1 else mask
                else:
                    mask_tensor = mask
            separated_result = Mask_transform_sum().separate(
                bg_mode="crop_image",
                mask_mode=mask_mode,
                ignore_threshold=0,
                opacity=1,
                outline_thickness=1,
                smoothness=smoothness,
                mask_expand=mask_expand,
                expand_width=0,
                expand_height=0,
                rescale_crop=1.0,
                tapered_corners=True,
                mask_min=mask_min,
                mask_max=mask_max,
                base_image=image,
                mask=mask_tensor,
                crop_to_mask=False,
                divisible_by=1
            )
            processed_mask = separated_result[1]
        else:
            processed_mask = mask

        crop_image, original_crop_mask = self.process_resize(
            image, processed_mask, crop_mode, long_side, divisible_by, upscale_method)

        # 第一步：先计算auto_expand_square=False时的原始扩展结果（获取基准长边）
        # 1.1 基于原始扩展量计算边界
        orig_expand_w, orig_expand_h = expand_width, expand_height
        ideal_x_new = x - (orig_expand_w // 2) if 'x' in locals() else 0
        ideal_y_new = y - (orig_expand_h // 2) if 'y' in locals() else 0
        ideal_x_end = (x + w + (orig_expand_w // 2)) if 'x' in locals() else 0
        ideal_y_end = (y + h + (orig_expand_h // 2)) if 'y' in locals() else 0

        # 1.2 处理遮罩边界（提前计算，为后续基准长边获取做准备）
        image_np = crop_image[0].cpu().numpy()
        mask_np = original_crop_mask[0].cpu().numpy()
        original_h, original_w = image_np.shape[0], image_np.shape[1]
        coords = cv2.findNonZero((mask_np > 0.5).astype(np.uint8))
        if coords is None:
            raise ValueError("Mask is empty after processing")
        x, y, w, h = cv2.boundingRect(coords)

        # 1.3 计算False时的原始扩展边界
        false_x_new = max(0, x - (orig_expand_w // 2))
        false_y_new = max(0, y - (orig_expand_h // 2))
        false_x_end = min(original_w, x + w + (orig_expand_w // 2))
        false_y_end = min(original_h, y + h + (orig_expand_h // 2))

        # 1.4 处理False时的边界补偿
        if (x - (orig_expand_w // 2)) < 0:
            add = abs(x - (orig_expand_w // 2))
            false_x_end = min(original_w, false_x_end + add)
        elif (x + w + (orig_expand_w // 2)) > original_w:
            add = (x + w + (orig_expand_w // 2)) - original_w
            false_x_new = max(0, false_x_new - add)

        if (y - (orig_expand_h // 2)) < 0:
            add = abs(y - (orig_expand_h // 2))
            false_y_end = min(original_h, false_y_end + add)
        elif (y + h + (orig_expand_h // 2)) > original_h:
            add = (y + h + (orig_expand_h // 2)) - original_h
            false_y_new = max(0, false_y_new - add)

        # 1.5 计算False时的最终尺寸（获取基准长边）
        false_w = false_x_end - false_x_new
        false_h = false_y_end - false_y_new
        false_long_side = max(false_w, false_h)  # 这是auto_expand_square=False时的长边，作为正方形基准

        # 第二步：根据auto_expand_square状态分支处理
        if auto_expand_square and crop_mode in ["no_scale_crop", "scale_crop_image"]:
            # 正方形模式：以False时的长边为目标边长，修正扩展量
            target_square_side = false_long_side
            # 计算需要的总扩展量（目标边长 - 原始遮罩尺寸）
            total_needed_expand_w = target_square_side - w
            total_needed_expand_h = target_square_side - h
            # 分配扩展量（左右/上下均分）
            expand_width = total_needed_expand_w
            expand_height = total_needed_expand_h

            # 重新计算正方形扩展边界
            ideal_x_new = x - (expand_width // 2)
            ideal_y_new = y - (expand_height // 2)
            ideal_x_end = x + w + (expand_width // 2)
            ideal_y_end = y + h + (expand_height // 2)

            # 处理正方形边界限制
            x_new = max(0, ideal_x_new)
            y_new = max(0, ideal_y_new)
            x_end = min(original_w, ideal_x_end)
            y_end = min(original_h, ideal_y_end)

            # 补偿扩展确保边长达标
            if x_new > ideal_x_new:
                x_end = min(original_w, x_end + (ideal_x_new - x_new))
            if x_end < ideal_x_end:
                x_new = max(0, x_new - (ideal_x_end - x_end))
            if y_new > ideal_y_new:
                y_end = min(original_h, y_end + (ideal_y_new - y_new))
            if y_end < ideal_y_end:
                y_new = max(0, y_new - (ideal_y_end - y_end))

            # 最终修正为正方形（确保宽高=目标边长）
            current_w = x_end - x_new
            current_h = y_end - y_new
            if current_w != target_square_side:
                diff = target_square_side - current_w
                x_new = max(0, x_new - (diff // 2))
                x_end = min(original_w, x_end + (diff - (diff // 2)))
            if current_h != target_square_side:
                diff = target_square_side - current_h
                y_new = max(0, y_new - (diff // 2))
                y_end = min(original_h, y_end + (diff - (diff // 2)))

            # 兼容divisible_by要求
            if divisible_by > 1:
                final_side = x_end - x_new
                remainder = final_side % divisible_by
                if remainder != 0:
                    final_side += (divisible_by - remainder)
                    diff = final_side - (x_end - x_new)
                    x_new = max(0, x_new - (diff // 2))
                    x_end = min(original_w, x_end + (diff - (diff // 2)))
                    y_new = max(0, y_new - (diff // 2))
                    y_end = min(original_h, y_end + (diff - (diff // 2)))
            x_end = x_new + (x_end - x_new)
            y_end = y_new + (x_end - x_new)  # 强制高=宽，确保正方形
        else:
            # 非正方形模式：完全沿用False时的原始逻辑结果
            x_new, y_new = false_x_new, false_y_new
            x_end, y_end = false_x_end, false_y_end

            # 原始尺寸修正逻辑
            if divisible_by > 1:
                current_w = x_end - x_new
                remainder_w = current_w % divisible_by
                if remainder_w != 0:
                    if x_end + (divisible_by - remainder_w) <= original_w:
                        x_end += (divisible_by - remainder_w)
                    elif x_new - (divisible_by - remainder_w) >= 0:
                        x_new -= (divisible_by - remainder_w)
                    else:
                        current_w -= remainder_w
                        x_end = x_new + current_w

                current_h = y_end - y_new
                remainder_h = current_h % divisible_by
                if remainder_h != 0:
                    if y_end + (divisible_by - remainder_h) <= original_h:
                        y_end += (divisible_by - remainder_h)
                    elif y_new - (divisible_by - remainder_h) >= 0:
                        y_new -= (divisible_by - remainder_h)
                    else:
                        current_h -= remainder_h
                        y_end = y_new + current_h
            else:
                current_w = x_end - x_new
                if current_w % 2 != 0:
                    if x_end < original_w:
                        x_end += 1
                    elif x_new > 0:
                        x_new -= 1

                current_h = y_end - y_new
                if current_h % 2 != 0:
                    if y_end < original_h:
                        y_end += 1
                    elif y_new > 0:
                        y_new -= 1

        # 最终裁剪尺寸
        current_w = x_end - x_new
        current_h = y_end - y_new

        bj_mask_tensor = original_crop_mask
        bj_image = crop_image.clone()

        if crop_img_bj != "image" and crop_img_bj in colors:
            r, g, b = colors[crop_img_bj]
            h_bg, w_bg, _ = crop_image.shape[1:]
            background = torch.zeros((crop_image.shape[0], h_bg, w_bg, 3), device=crop_image.device)
            background[:, :, :, 0] = r
            background[:, :, :, 1] = g
            background[:, :, :, 2] = b
            if crop_image.shape[3] >= 4:
                alpha = crop_image[:, :, :, 3].unsqueeze(3)
                image_rgb = crop_image[:, :, :, :3]
                crop_image = image_rgb * alpha + background * (1 - alpha)
            else:
                alpha = original_crop_mask.unsqueeze(3)
                image_rgb = crop_image[:, :, :, :3]
                crop_image = image_rgb * alpha + background * (1 - alpha)

        mask_x_start = 0
        mask_y_start = 0
        mask_x_end = 0
        mask_y_end = 0

        if crop_mode == "no_crop":
            cropped_image_tensor = crop_image.clone()
            cropped_mask_tensor = original_crop_mask.clone()
            current_crop_position = (0, 0)
            current_crop_size = (original_w, original_h)
            mask_x_start = x
            mask_y_start = y
            mask_x_end = x + w
            mask_y_end = y + h
        else:
            cropped_image_tensor = crop_image[:, y_new:y_end, x_new:x_end, :].clone()
            cropped_mask_tensor = original_crop_mask[:, y_new:y_end, x_new:x_end].clone()
            mask_x_start = max(0, x - x_new)
            mask_y_start = max(0, y - y_new)
            mask_x_end = min(current_w, (x + w) - x_new)
            mask_y_end = min(current_h, (y + h) - y_new)
            current_crop_position = (x_new, y_new)
            current_crop_size = (current_w, current_h)

        orig_long_side = max(original_w, original_h)
        crop_long_side = max(current_crop_size[0], current_crop_size[1])
        original_image_h, original_image_w = image.shape[1], image.shape[2]
        stitch = {
            "original_shape": (original_h, original_w),
            "original_image_shape": (original_image_h, original_image_w),
            "crop_position": current_crop_position,
            "crop_size": current_crop_size,
            "expand_width": expand_width,
            "expand_height": expand_height,
            "auto_expand_square": auto_expand_square,
            "expanded_region": (x_new, y_new, x_end, y_end),
            "mask_original_position": (x, y, w, h),
            "mask_cropped_position": (mask_x_start, mask_y_start, mask_x_end, mask_y_end),
            "original_long_side": orig_long_side,
            "crop_long_side": crop_long_side,
            "input_long_side": long_side,
            "false_long_side": false_long_side,  # 记录False时的基准长边
            "bj_image": bj_image,
            "original_image": image
        }

        return (bj_image, bj_mask_tensor, cropped_image_tensor, cropped_mask_tensor, stitch)






#endregion----------------裁切组合--------------


class Mask_simple_adjust:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1}),
                "mask_expand": ("INT", {"default": 0, "min": -500, "max": 1000, "step": 0.1}),
                "is_fill": ("BOOLEAN", {"default": False}),
                "is_invert": ("BOOLEAN", {"default": False}),
                "input_mask": ("MASK",),
                "mask_min": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 1.0, "step": 0.01}),
                "mask_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("processed_mask",)
    FUNCTION = "process_mask"
    CATEGORY = "Apt_Preset/mask"
    
    def process_mask(self, smoothness=0, mask_expand=0, is_fill=False, is_invert=False, input_mask=None, mask_min=0.0, mask_max=1.0):
        if input_mask is None:
            empty_mask = torch.zeros(1, 64, 64, dtype=torch.float32)
            return (empty_mask,)
        
        def tensorMask2cv2img(tensor_mask):
            mask_np = tensor_mask.cpu().numpy().squeeze()
            if len(mask_np.shape) == 3:
                mask_np = mask_np[:, :, 0]
            return (mask_np * 255).astype(np.uint8)
        
        def cv2img2tensorMask(cv2_mask):
            mask_np = cv2_mask.astype(np.float32) / 255.0
            # 应用mask_min和mask_max调整蒙版动态范围
            mask_max_val = np.max(mask_np) if np.max(mask_np) > 0 else 1.0
            mask_np = (mask_np / mask_max_val) * (mask_max - mask_min) + mask_min
            mask_np = np.clip(mask_np, 0.0, 1.0)
            return torch.from_numpy(mask_np).unsqueeze(0)
        
        opencv_gray_mask = tensorMask2cv2img(input_mask)
        _, binary_mask = cv2.threshold(opencv_gray_mask, 1, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 1]
        
        final_mask = np.zeros_like(binary_mask)
        expand_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        
        for contour in valid_contours:
            temp_mask = np.zeros_like(binary_mask)
            if is_fill:
                cv2.drawContours(temp_mask, [contour], 0, 255, thickness=cv2.FILLED)
            else:
                cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                temp_mask = cv2.bitwise_and(opencv_gray_mask, temp_mask)
            
            if mask_expand != 0:
                expand_iter = abs(int(mask_expand))
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, expand_kernel, iterations=expand_iter)
                else:
                    temp_mask = cv2.erode(temp_mask, expand_kernel, iterations=expand_iter)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        if smoothness > 0:
            mask_pil = Image.fromarray(final_mask)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(mask_pil)
        
        if is_invert:
            final_mask = cv2.bitwise_not(final_mask)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        processed_mask_tensor = cv2img2tensorMask(final_mask)
        return (processed_mask_tensor,)



class Image_Channel_Apply:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "channel": (["A", "R", "G", "B"],),
                "invert_channel": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "channel_data": ("MASK",),
                "background_color": (
                    ["none", "image", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"],
                    {"default": "none"}
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    CATEGORY = "Apt_Preset/image"
    FUNCTION = "Image_Channel_Apply"

    def Image_Channel_Apply(self, images: torch.Tensor, channel, invert_channel=False, 
                           channel_data=None, background_color="none"):
        channel_colors = {
            "R": (1.0, 0.0, 0.0),
            "G": (0.0, 1.0, 0.0),
            "B": (0.0, 0.0, 1.0),
            "A": (1.0, 1.0, 1.0)
        }
        
        colors = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "red": (1.0, 0.0, 0.0),
            "green": (0.0, 1.0, 0.0),
            "blue": (0.0, 0.0, 1.0),
            "yellow": (1.0, 1.0, 0.0),
            "cyan": (0.0, 1.0, 1.0),
            "magenta": (1.0, 0.0, 1.0),
            "gray": (0.5, 0.5, 0.5)
        }
        
        merged_images = []
        output_masks = []
        
        if len(images.shape) < 4:
            images = images.unsqueeze(3).repeat(1, 1, 1, 3)
        
        channel_index = ["R", "G", "B", "A"].index(channel)
        input_provided = channel_data is not None

        for i, image in enumerate(images):
            # 保存原始图像的副本用于可能作为背景
            original_background = image.cpu().clone()
            
            if channel == "A" and image.shape[2] < 4:
                base_mask = torch.zeros((image.shape[0], image.shape[1]))
            else:
                base_mask = image[:, :, channel_index].clone()
            
            if input_provided:
                input_mask = channel_data
                
                # 处理不同维度的输入mask
                if len(input_mask.shape) == 4:
                    # 4D张量 (batch, height, width, 1)
                    input_mask = input_mask.squeeze(-1)  # 移除最后一个维度
                    if input_mask.shape[0] > i:
                        input_mask = input_mask[i]  # 选择对应批次
                    else:
                        input_mask = input_mask[0]  # 如果批次不足，使用第一个
                elif len(input_mask.shape) == 3:
                    # 3D张量 (batch, height, width) 或 (height, width, 1)
                    if input_mask.shape[-1] == 1:
                        input_mask = input_mask.squeeze(-1)
                    if len(input_mask.shape) == 3 and input_mask.shape[0] > 1:
                        # 多批次mask
                        if input_mask.shape[0] > i:
                            input_mask = input_mask[i]
                        else:
                            input_mask = input_mask[0]
                elif len(input_mask.shape) == 2:
                    # 2D张量 (height, width)
                    pass  # 已经是正确的格式
                
                # 确保input_mask是2D的
                if len(input_mask.shape) > 2:
                    input_mask = input_mask.squeeze()
                
                # 确保维度匹配
                if input_mask.shape != base_mask.shape:
                    input_mask = input_mask.unsqueeze(0).unsqueeze(0) if len(input_mask.shape) == 2 else input_mask.unsqueeze(0)
                    input_mask = torch.nn.functional.interpolate(
                        input_mask,
                        size=base_mask.shape[-2:],  # 只使用最后两个维度
                        mode='bilinear',
                        align_corners=False
                    )
                    input_mask = input_mask.squeeze()
                
                processed_input_mask = 1.0 - input_mask if invert_channel else input_mask
                merged_mask = processed_input_mask + base_mask
                merged_mask = torch.clamp(merged_mask, 0.0, 1.0)
            else:
                merged_mask = base_mask
                processed_input_mask = merged_mask
            
            original_image = image.cpu().clone()
            image = original_image.clone()

            if channel != "A":
                if input_provided:
                    if channel == "R":
                        image[:, :, 0] = merged_mask
                    elif channel == "G":
                        image[:, :, 1] = merged_mask
                    else:
                        image[:, :, 2] = merged_mask
                else:
                    r, g, b = channel_colors[channel]
                    channel_color_image = torch.zeros_like(image[:, :, :3])
                    channel_color_image[:, :, 0] = r
                    channel_color_image[:, :, 1] = g
                    channel_color_image[:, :, 2] = b
                    
                    mask = base_mask.unsqueeze(2)
                    image[:, :, :3] = original_image[:, :, :3] * (1 - mask) + channel_color_image * mask
            else:
                if input_provided:
                    if image.shape[2] < 4:
                        image = torch.cat([image, torch.ones((image.shape[0], image.shape[1], 1))], dim=2)
                    
                    image[:, :, 3] = processed_input_mask
                    mask = processed_input_mask.unsqueeze(2)
                    image[:, :, :3] = original_image[:, :, :3] * mask

            # 处理背景
            if background_color != "none":
                if background_color == "image":
                    # 使用输入的原始图像作为背景
                    background = original_background[:, :, :3]  # 只取RGB通道
                else:
                    # 使用颜色作为背景
                    r, g, b = colors[background_color]
                    h, w, _ = image.shape
                    background = torch.zeros((h, w, 3))
                    background[:, :, 0] = r
                    background[:, :, 1] = g
                    background[:, :, 2] = b
                
                # 应用背景
                if image.shape[2] >= 4:
                    alpha = image[:, :, 3].unsqueeze(2)
                    image_rgb = image[:, :, :3]
                    image = image_rgb * alpha + background * (1 - alpha)
                else:
                    image = background

            if channel == "A" and input_provided:
                output_mask = processed_input_mask
            else:
                output_mask = merged_mask

            merged_images.append(image)
            output_masks.append(output_mask)

        return (torch.stack(merged_images), torch.stack(output_masks))


class Image_target_adjust:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # 移除了 adjust_mode 参数
                "target_width": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "target_height": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 16}),
                "multiple": ("INT", {"default": 64, "min": 1, "max": 256, "step": 1}),
                "upscale_method": (["bicubic", "nearest-exact", "bilinear", "area", "lanczos"],),
                "adjustment_method": (["stretch", "crop", "pad"], {
                    "default": "stretch"
                }),
            },
            "optional": {
                "region_position": (["top", "bottom", "left", "right", "center"], {
                    "default": "center"
                }),
                "pad_background": (
                    ["none", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"],
                    {"default": "black"}
                ),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "resize_image"
    CATEGORY = "Apt_Preset/image"
    
    def calculate_optimal_dimensions(self, original_width, original_height, target_width, target_height, multiple):
        # 固定使用 long*short 模式 (保持原始宽高比，以长边和短边为参考)
        original_aspect = original_width / original_height
        target_area = target_width * target_height
        ideal_width = np.sqrt(target_area * original_aspect)
        ideal_height = np.sqrt(target_area / original_aspect)
        width = round(ideal_width / multiple) * multiple
        height = round(ideal_height / multiple) * multiple
        width = max(multiple, width)
        height = max(multiple, height)
        return width, height
    
    def get_color_value(self, color_name, image=None):
        """获取颜色值，支持预设颜色和使用原图作为背景"""
        if color_name == "image" and image is not None:
            # 对于图像背景，我们会在填充时处理，这里返回特殊标记
            return "image"
            
        colors = {
            "none": (0, 0, 0, 0),  # 透明
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "gray": (128, 128, 128)
        }
        
        return colors.get(color_name, (0, 0, 0))
    
    def create_background_image(self, size, original_image, position, padded_area):
        """创建与原图风格一致的背景填充图像"""
        width, height = size
        original_np = (original_image * 255).astype(np.uint8)
        original_pil = Image.fromarray(original_np)
        
        # 创建与目标大小相同的背景图
        background = Image.new('RGB', (width, height))
        
        # 根据填充区域和位置，从原图中提取合适的区域作为背景
        if padded_area == "top" or padded_area == "bottom":
            # 上下填充，使用原图左右边缘
            src_width, src_height = original_pil.size
            cropped = original_pil.crop((0, 0, src_width, min(src_height, height)))
            resized = cropped.resize((width, height), Image.BILINEAR)
            background.paste(resized)
        elif padded_area == "left" or padded_area == "right":
            # 左右填充，使用原图上下边缘
            src_width, src_height = original_pil.size
            cropped = original_pil.crop((0, 0, min(src_width, width), src_height))
            resized = cropped.resize((width, height), Image.BILINEAR)
            background.paste(resized)
            
        return background
    
    def resize_image(self, image, target_width, target_height, multiple, upscale_method, 
                    adjustment_method, region_position="center", pad_background="black"):
        batch_size, original_height, original_width, channels = image.shape
        output_width, output_height = self.calculate_optimal_dimensions(
            original_width, original_height, target_width, target_height, multiple
        )
        
        original_area = original_width * original_height
        target_area = output_width * output_height
        
        if original_area > target_area:
            method = "area"
        else:
            method = upscale_method
        
        def resize_fn(img):
            img_np = img.cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            
            original_aspect = original_width / original_height
            target_aspect = output_width / output_height
            
            if adjustment_method == "stretch":
                resized_pil = pil_img.resize((output_width, output_height), resample=self.get_resample_method(method))
                
            elif adjustment_method == "crop":
                if original_aspect > target_aspect:
                    # 宽高比更大，需要裁剪宽度
                    scale = output_height / original_height
                    scaled_width = int(original_width * scale)
                    scaled_height = output_height
                    resized = pil_img.resize((scaled_width, scaled_height), resample=self.get_resample_method(method))
                    
                    # 根据位置裁剪宽度
                    excess = scaled_width - output_width
                    if region_position == "left":
                        left, right = 0, output_width
                    elif region_position == "right":
                        left, right = excess, scaled_width
                    else:  # center
                        left = excess // 2
                        right = left + output_width
                    resized_pil = resized.crop((left, 0, right, scaled_height))
                else:
                    # 宽高比更小，需要裁剪高度
                    scale = output_width / original_width
                    scaled_width = output_width
                    scaled_height = int(original_height * scale)
                    resized = pil_img.resize((scaled_width, scaled_height), resample=self.get_resample_method(method))
                    
                    # 根据位置裁剪高度
                    excess = scaled_height - output_height
                    if region_position == "top":
                        top, bottom = 0, output_height
                    elif region_position == "bottom":
                        top, bottom = excess, scaled_height
                    else:  # center
                        top = excess // 2
                        bottom = top + output_height
                    resized_pil = resized.crop((0, top, scaled_width, bottom))
                    
            elif adjustment_method == "pad":
                # 获取填充颜色
                pad_color = self.get_color_value(pad_background, img_np)
                
                if original_aspect > target_aspect:
                    # 宽高比更大，需要在高度方向填充
                    scale = output_width / original_width
                    scaled_width = output_width
                    scaled_height = int(original_height * scale)
                    resized = pil_img.resize((scaled_width, scaled_height), resample=self.get_resample_method(method))
                    
                    # 计算填充量
                    pad_total = output_height - scaled_height
                    if region_position == "top":
                        pad_top, pad_bottom = pad_total, 0
                        padded_area = "top"
                    elif region_position == "bottom":
                        pad_top, pad_bottom = 0, pad_total
                        padded_area = "bottom"
                    else:  # center
                        pad_top = pad_total // 2
                        pad_bottom = pad_total - pad_top
                        padded_area = "center"
                    
                    # 处理图像背景填充
                    if pad_background == "image":
                        # 创建与原图风格一致的背景
                        bg_size = (output_width, output_height)
                        background = self.create_background_image(bg_size, img_np, region_position, padded_area)
                        # 将缩放后的图像粘贴到背景上
                        y_offset = pad_top
                        background.paste(resized, (0, y_offset))
                        resized_pil = background
                    else:
                        # 使用指定颜色填充
                        resized_pil = ImageOps.expand(resized, (0, pad_top, 0, pad_bottom), fill=pad_color)
                else:
                    # 宽高比更小，需要在宽度方向填充
                    scale = output_height / original_height
                    scaled_width = int(original_width * scale)
                    scaled_height = output_height
                    resized = pil_img.resize((scaled_width, scaled_height), resample=self.get_resample_method(method))
                    
                    # 计算填充量
                    pad_total = output_width - scaled_width
                    if region_position == "left":
                        pad_left, pad_right = pad_total, 0
                        padded_area = "left"
                    elif region_position == "right":
                        pad_left, pad_right = 0, pad_total
                        padded_area = "right"
                    else:  # center
                        pad_left = pad_total // 2
                        pad_right = pad_total - pad_left
                        padded_area = "center"
                    
                    # 处理图像背景填充
                    if pad_background == "image":
                        # 创建与原图风格一致的背景
                        bg_size = (output_width, output_height)
                        background = self.create_background_image(bg_size, img_np, region_position, padded_area)
                        # 将缩放后的图像粘贴到背景上
                        x_offset = pad_left
                        background.paste(resized, (x_offset, 0))
                        resized_pil = background
                    else:
                        # 使用指定颜色填充
                        resized_pil = ImageOps.expand(resized, (pad_left, 0, pad_right, 0), fill=pad_color)
            
            resized_np = np.array(resized_pil).astype(np.float32) / 255.0
            return torch.from_numpy(resized_np)
        
        resized_images = torch.stack([resize_fn(img) for img in image])
        return (resized_images,)
    
    def get_resample_method(self, method):
        methods = {
            "bicubic": Image.BICUBIC,
            "nearest-exact": Image.NEAREST,
            "bilinear": Image.BILINEAR,
            "area": Image.LANCZOS if method == "area" and Image.__version__ >= "9.1.0" else Image.BILINEAR,
            "lanczos": Image.LANCZOS
        }
        return methods.get(method, Image.BILINEAR)


class Image_pad_adjust_restore:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pad_image": ("IMAGE",),
                "stitch": ("STITCH4",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1, "display": "slider"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("restored_image", "restored_mask", "original_image")
    FUNCTION = "restore"
    CATEGORY = "Apt_Preset/image"

    def create_feather_mask(self, width, height, feather_size):
        if feather_size <= 0:
            return np.ones((height, width), dtype=np.float32)
        feather = min(feather_size, min(width, height) // 2)
        mask = np.ones((height, width), dtype=np.float32)
        for y in range(feather):
            mask[y, :] = y / feather
        for y in range(height - feather, height):
            mask[y, :] = (height - y) / feather
        for x in range(feather):
            mask[:, x] = np.minimum(mask[:, x], x / feather)
        for x in range(width - feather, width):
            mask[:, x] = np.minimum(mask[:, x], (width - x) / feather)
        return mask

    def restore(self, pad_image, stitch, smoothness):
        original_image = stitch["original_image"]
        original_h, original_w = stitch["original_shape"]
        orig_left, orig_top, orig_right, orig_bottom, act_left, act_top, act_right, act_bottom = stitch["pad_info"]
        crop_offset_left, crop_offset_top = stitch.get("crop_offsets", (0, 0))
        has_mask = stitch["has_mask"]
        original_mask = stitch["original_mask"]
        
        current_b, current_h, current_w, current_c = pad_image.shape
        batch_size = original_image.shape[0] if len(original_image.shape) == 4 else 1
        
        restored_images = []
        for i in range(batch_size):
            orig_img = original_image[i] if len(original_image.shape) == 4 else original_image
            restored_img = orig_img.clone()
            processed_img = pad_image[i] if pad_image.shape[0] > 1 else pad_image[0]
            
            crop_left = crop_offset_left
            crop_top = crop_offset_top
            crop_right = max(-orig_right, 0)
            crop_bottom = max(-orig_bottom, 0)
            
            pad_left = act_left
            pad_top = act_top
            pad_right = act_right
            pad_bottom = act_bottom
            
            valid_left = pad_left
            valid_top = pad_top
            valid_right = processed_img.shape[1] - pad_right
            valid_bottom = processed_img.shape[0] - pad_bottom
            
            valid_left = max(0, min(valid_left, processed_img.shape[1]))
            valid_top = max(0, min(valid_top, processed_img.shape[0]))
            valid_right = max(valid_left, min(valid_right, processed_img.shape[1]))
            valid_bottom = max(valid_top, min(valid_bottom, processed_img.shape[0]))
            
            content_img = processed_img[valid_top:valid_bottom, valid_left:valid_right, :]
            
            dst_left = crop_left
            dst_top = crop_top
            dst_right = min(original_w - crop_right, dst_left + content_img.shape[1])
            dst_bottom = min(original_h - crop_bottom, dst_top + content_img.shape[0])
            
            src_width = dst_right - dst_left
            src_height = dst_bottom - dst_top
            
            if src_width > 0 and src_height > 0:
                content = content_img[:src_height, :src_width, :].cpu().numpy()
                background = orig_img[dst_top:dst_bottom, dst_left:dst_right, :].cpu().numpy()
                
                if smoothness > 0:
                    feather_mask = self.create_feather_mask(src_width, src_height, smoothness)
                    feather_mask = np.expand_dims(feather_mask, axis=-1)
                    blended = background * (1 - feather_mask) + content * feather_mask
                    restored_img[dst_top:dst_bottom, dst_left:dst_right, :] = torch.from_numpy(blended).to(restored_img.device)
                else:
                    restored_img[dst_top:dst_bottom, dst_left:dst_right, :] = content_img[:src_height, :src_width, :]
            
            restored_images.append(restored_img.unsqueeze(0))
        
        restored_image = torch.cat(restored_images, dim=0)
        
        if has_mask and original_mask is not None:
            restored_mask = original_mask
        else:
            restored_mask = torch.zeros((restored_image.shape[0], restored_image.shape[1], restored_image.shape[2]), dtype=torch.float32, device=restored_image.device)
        
        return (restored_image, restored_mask, original_image)


class Image_pad_adjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "top": ("INT", {"default": 0, "step": 1, "min": -14096, "max": 14096}),
                "bottom": ("INT", {"default": 0, "step": 1, "min": -14096, "max": 14096}),
                "left": ("INT", {"default": 0, "step": 1, "min": -14096, "max": 14096}),
                "right": ("INT", {"default": 0, "step": 1, "min": -14096, "max": 14096}),
                "bg_color": (["white", "black", "red", "green", "blue", "gray"], {"default": "black"}),
                "smoothness": ("INT", {"default": 0, "step": 1, "min": 0, "max": 500}),
                "divisible_by": ("INT", {"default": 2, "min": 1, "max": 512, "step": 1}),
                "auto_pad": (["None", "auto_square", "target_WxH"], {"default": "None"}),
                "pad_position": (["left-top", "mid-top", "right-top", "left-center", "mid-center", "right-center", "left-bottom", "mid-bottom", "right-bottom"], {"default": "mid-center"}),               
                "target_W": ("INT", {"default": 512, "min": 1, "max": 14096, "step": 1}),
                "target_H": ("INT", {"default": 512, "min": 1, "max": 14096, "step": 1}),
                "pad_mask_remove": ("BOOLEAN", {"default": True,}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH4")
    RETURN_NAMES = ("image", "mask", "stitch")
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image"
    DESCRIPTION = """
    - bg_color: 填充的颜色
    - smoothness: 遮罩边缘平滑
    - divisible_by: 输出图像尺寸需整除的值
    - auto_pad自动填充:None表示关闭自动填充
    - auto_square按长边填充成正方形，target_WxH按输入的宽高填充
    """

    def process(self, left, top, right, bottom, bg_color, smoothness, divisible_by, auto_pad, target_W, target_H, pad_position, pad_mask_remove, image=None, mask=None):
        original_shape = (image.shape[1], image.shape[2])
        original_left, original_top, original_right, original_bottom = left, top, right, bottom
        original_image = image.clone()

        if auto_pad == "auto_square":
            image, mask, left, top, right, bottom = self.auto_padding(image, mask, left, top, right, bottom, pad_position)
        elif auto_pad == "target_WxH":
            image, mask, left, top, right, bottom = self.target_padding(image, mask, left, top, right, bottom, target_W, target_H, pad_position)

        cropped_image, cropped_mask = self.crop_image(image, mask, left, top, right, bottom)
        padded_image, actual_left, actual_top, actual_right, actual_bottom = self.add_padding(
            cropped_image, max(left, 0), max(top, 0), max(right, 0), max(bottom, 0), bg_color, divisible_by)
        
        # 1. 生成原始遮罩和填充区域掩码（无平滑）
        raw_mask, padding_mask = self.create_mask(cropped_image, actual_left, actual_top, actual_right, actual_bottom, 
                                    0, cropped_mask, divisible_by)
        
        # 2. 标记原始图像区域（非填充区）
        original_region = (1 - padding_mask) > 0.5  # 硬边界，确保填充区为False
        
        # 3. 仅对原始区域内的遮罩进行平滑处理
        if smoothness > 0 and pad_mask_remove:
            # 先将填充区域的遮罩清零，避免平滑扩散到填充区
            masked_raw = raw_mask * original_region.float()
            # 对清理后的遮罩进行平滑
            smoothed_mask, _ = self.create_mask_from_tensor(cropped_image, actual_left, actual_top, actual_right, actual_bottom, 
                                    smoothness, masked_raw, divisible_by)
            # 再次确保填充区域无残留
            final_mask = smoothed_mask * original_region.float()
        elif smoothness > 0:
            # 不移除填充区时，正常平滑全部遮罩
            final_mask, _ = self.create_mask(cropped_image, actual_left, actual_top, actual_right, actual_bottom, 
                                    smoothness, cropped_mask, divisible_by)
        else:
            # 无平滑时，根据pad_mask_remove决定是否保留填充区遮罩
            if pad_mask_remove:
                final_mask = raw_mask * original_region.float()
            else:
                final_mask = raw_mask  # 关键修复：当pad_mask_remove=False时，直接使用原始遮罩（保留填充区）
        
        crop_offset_left = max(-left, 0)
        crop_offset_top = max(-top, 0)
        
        pad_info = (original_left, original_top, original_right, original_bottom,
                    actual_left, actual_top, actual_right, actual_bottom)
        
        stitch_info = {
            "original_image": original_image,
            "original_shape": original_shape,
            "pad_info": pad_info,
            "crop_offsets": (crop_offset_left, crop_offset_top),
            "bg_color": bg_color,
            "has_mask": mask is not None,
            "original_mask": mask.clone() if mask is not None else None
        }
        return (padded_image, final_mask, stitch_info)


    def create_mask_from_tensor(self, image, left, top, right, bottom, smoothness, mask_tensor, divisible_by=1):
        masks = []
        padding_masks = []
        image = [tensor2pil(img) for img in image]
        mask = [tensor2pil(m) for m in mask_tensor] if isinstance(mask_tensor, torch.Tensor) and mask_tensor.dim() > 3 else [tensor2pil(mask_tensor)]
        
        for i, img in enumerate(image):
            target_width = img.width + left + right
            target_height = img.height + top + bottom
            if divisible_by > 1:
                target_width = math.ceil(target_width / divisible_by) * divisible_by
                target_height = math.ceil(target_height / divisible_by) * divisible_by
                adjusted_right = target_width - img.width - left
                adjusted_bottom = target_height - img.height - top
            else:
                adjusted_right = right
                adjusted_bottom = bottom
                
            mask_image = Image.new("L", (target_width, target_height), 0)
            mask_to_paste = mask[i] if len(mask) > 1 else mask[0]
            mask_image.paste(mask_to_paste, (left, top))
            
            padding_mask = Image.new("L", (target_width, target_height), 255)
            padding_draw = ImageDraw.Draw(padding_mask)
            padding_draw.rectangle((left, top, img.width + left, img.height + top), fill=0)
            
            if smoothness > 0:
                smoothed_mask_tensor = smoothness_mask(mask_image, smoothness)
                masks.append(smoothed_mask_tensor)
                smoothed_padding_pil = pil2tensor(padding_mask)
                smoothed_padding_mask = smoothness_mask(tensor2pil(smoothed_padding_pil), smoothness)
                padding_masks.append(smoothed_padding_mask)
            else:
                masks.append(pil2tensor(mask_image))
                padding_masks.append(pil2tensor(padding_mask))
                    
        final_masks = torch.cat(masks, dim=0) if len(masks) > 1 else masks[0].unsqueeze(0)
        final_padding_masks = torch.cat(padding_masks, dim=0) if len(padding_masks) > 1 else padding_masks[0].unsqueeze(0)
        final_padding_masks = torch.clamp(final_padding_masks, 0, 1)
        
        return final_masks, final_padding_masks


    def target_padding(self, image, mask, left, top, right, bottom, target_W, target_H, pad_position):
        batch_size, height, width, _ = image.shape
        
        # 计算需要的填充或裁切量
        delta_width = target_W - width
        delta_height = target_H - height
        
        # 根据pad_position计算左右和上下的填充/裁切量
        left_pad, right_pad, top_pad, bottom_pad = self.calculate_padding_or_cropping(delta_width, delta_height, pad_position)
        
        new_left = left + left_pad
        new_right = right + right_pad
        new_top = top + top_pad
        new_bottom = bottom + bottom_pad
        
        return image, mask, new_left, new_top, new_right, new_bottom
    
    def calculate_padding_or_cropping(self, delta_width, delta_height, pad_position):
        """
        根据delta值（正数表示填充，负数表示裁切）和pad_position计算左右和上下的填充或裁切量
        支持9个位置：left-top, mid-top, right-top, 
                   left-center, mid-center, right-center,
                   left-bottom, mid-bottom, right-bottom
        """
        # 解析位置参数
        parts = pad_position.split('-')
        if len(parts) != 2:
            # 如果格式不正确，默认使用mid-center
            left_pad, right_pad = self.calculate_horizontal_adjustment(delta_width, "mid")
            top_pad, bottom_pad = self.calculate_vertical_adjustment(delta_height, "center")
            return left_pad, right_pad, top_pad, bottom_pad
        
        horizontal_pos, vertical_pos = parts
        
        # 计算水平方向调整（左右）
        left_pad, right_pad = self.calculate_horizontal_adjustment(delta_width, horizontal_pos)
        
        # 计算垂直方向调整（上下）
        top_pad, bottom_pad = self.calculate_vertical_adjustment(delta_height, vertical_pos)
        
        return left_pad, right_pad, top_pad, bottom_pad
    
    def calculate_horizontal_adjustment(self, delta_width, position):
        """
        计算水平平方向（左右）的调整量（正数充或裁切）
        delta_width: 正数表示填充，负数表示裁切
        position: left, mid, right
        """
        if delta_width >= 0:
            # 填充模式
            if position == "left":
                return delta_width, 0
            elif position == "right":
                return 0, delta_width
            elif position == "mid":
                left = delta_width // 2
                right = delta_width - left
                return left, right
            else:  # 默认使用mid
                left = delta_width // 2
                right = delta_width - left
                return left, right
        else:
            # 裁切模式
            crop_width = -delta_width
            if position == "left":
                return -crop_width, 0  # 从左边裁切
            elif position == "right":
                return 0, -crop_width  # 从右边裁切
            elif position == "mid":
                left = crop_width // 2
                right = crop_width - left
                return -left, -right  # 从左右两边平均裁切
            else:  # 默认使用mid
                left = crop_width // 2
                right = crop_width - left
                return -left, -right
    
    def calculate_vertical_adjustment(self, delta_height, position):
        """
        计算垂直方向（上下）的调整量（填充或裁切）
        delta_height: 正数表示填充，负数表示裁切
        position: top, center, bottom
        """
        if delta_height >= 0:
            # 填充模式
            if position == "top":
                return delta_height, 0
            elif position == "bottom":
                return 0, delta_height
            elif position == "center":
                top = delta_height // 2
                bottom = delta_height - top
                return top, bottom
            else:  # 默认使用center
                top = delta_height // 2
                bottom = delta_height - top
                return top, bottom
        else:
            # 裁切模式
            crop_height = -delta_height
            if position == "top":
                return -crop_height, 0  # 从顶部裁切
            elif position == "bottom":
                return 0, -crop_height  # 从底部裁切
            elif position == "center":
                top = crop_height // 2
                bottom = crop_height - top
                return -top, -bottom  # 从上下两边平均裁切
            else:  # 默认使用center
                top = crop_height // 2
                bottom = crop_height - top
                return -top, -bottom

    def auto_padding(self, image, mask, left, top, right, bottom, pad_position):
        batch_size, height, width, _ = image.shape
        target_size = max(width, height)
        delta_width = target_size - width
        delta_height = target_size - height
        
        # 根据pad_position计算左右和上下的填充量
        left_pad, right_pad, top_pad, bottom_pad = self.calculate_padding_or_cropping(delta_width, delta_height, pad_position)
        
        new_left = left + left_pad
        new_right = right + right_pad
        new_top = top + top_pad
        new_bottom = bottom + bottom_pad
        
        return image, mask, new_left, new_top, new_right, new_bottom

    def crop_image(self, image, mask, left, top, right, bottom):
        crop_left = max(-left, 0)
        crop_top = max(-top, 0)
        crop_right = max(-right, 0)
        crop_bottom = max(-bottom, 0)
        images = [tensor2pil(img) for img in image]
        cropped_images = []
        for img in images:
            width, height = img.size
            new_left = crop_left
            new_top = crop_top
            new_right = width - crop_right
            new_bottom = height - crop_bottom
            new_left = min(max(new_left, 0), width)
            new_top = min(max(new_top, 0), height)
            new_right = max(min(new_right, width), new_left)
            new_bottom = max(min(new_bottom, height), new_top)
            cropped_img = img.crop((new_left, new_top, new_right, new_bottom))
            cropped_images.append(pil2tensor(cropped_img))
        cropped_masks = None
        if mask is not None:
            masks = [tensor2pil(m) for m in mask] if isinstance(mask, torch.Tensor) and mask.dim() > 3 else [tensor2pil(mask)]
            cropped_masks = []
            for m in masks:
                width, height = m.size
                new_left = crop_left
                new_top = crop_top
                new_right = width - crop_right
                new_bottom = height - crop_bottom
                new_left = min(max(new_left, 0), width)
                new_top = min(max(new_top, 0), height)
                new_right = max(min(new_right, width), new_left)
                new_bottom = max(min(new_bottom, height), new_top)
                cropped_mask = m.crop((new_left, new_top, new_right, new_bottom))
                cropped_masks.append(pil2tensor(cropped_mask))
            cropped_masks = torch.cat(cropped_masks, dim=0)
        return torch.cat(cropped_images, dim=0), cropped_masks

    def add_padding(self, image, left, top, right, bottom, bg_color, divisible_by=1):
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "gray": (128, 128, 128)
        }
        color = color_map.get(bg_color, (0, 0, 0))
        padded_images = []
        image = [tensor2pil(img) for img in image]
        for img in image:
            target_width = img.width + left + right
            target_height = img.height + top + bottom
            if divisible_by > 1:
                target_width = math.ceil(target_width / divisible_by) * divisible_by
                target_height = math.ceil(target_height / divisible_by) * divisible_by
                adjusted_right = target_width - img.width - left
                adjusted_bottom = target_height - img.height - top
            else:
                adjusted_right = right
                adjusted_bottom = bottom
            padded_image = Image.new("RGB", (target_width, target_height), color)
            padded_image.paste(img, (left, top))
            padded_images.append(pil2tensor(padded_image))
        return torch.cat(padded_images, dim=0), left, top, adjusted_right, adjusted_bottom

    def create_mask(self, image, left, top, right, bottom, smoothness, mask=None, divisible_by=1):
        masks = []
        padding_masks = []
        image = [tensor2pil(img) for img in image]
        if mask is not None:
            mask = [tensor2pil(m) for m in mask] if isinstance(mask, torch.Tensor) and mask.dim() > 3 else [tensor2pil(mask)]
        for i, img in enumerate(image):
            target_width = img.width + left + right
            target_height = img.height + top + bottom
            if divisible_by > 1:
                target_width = math.ceil(target_width / divisible_by) * divisible_by
                target_height = math.ceil(target_height / divisible_by) * divisible_by
                adjusted_right = target_width - img.width - left
                adjusted_bottom = target_height - img.height - top
            else:
                adjusted_right = right
                adjusted_bottom = bottom
            shape = (left, top, img.width + left, img.height + top)
            mask_image = Image.new("L", (target_width, target_height), 255)
            draw = ImageDraw.Draw(mask_image)
            draw.rectangle(shape, fill=0)
            if mask is not None:
                mask_to_paste = mask[i] if len(mask) > 1 else mask[0]
                mask_image.paste(mask_to_paste, (left, top))
            
            # 创建padding_mask（原图像区域为0，填充区域为1）
            padding_mask = Image.new("L", (target_width, target_height), 255)  # 默认全为255（填充区域）
            padding_draw = ImageDraw.Draw(padding_mask)
            padding_draw.rectangle(shape, fill=0)  # 原图像区域为0
            
            if smoothness > 0:
                smoothed_mask_tensor = smoothness_mask(mask_image, smoothness)
                masks.append(smoothed_mask_tensor)
                # 对padding_mask也应用平滑处理，但需要转换为0-1范围
                smoothed_padding_pil = pil2tensor(padding_mask)  # 转换为tensor (0-1范围)
                smoothed_padding_mask = smoothness_mask(tensor2pil(smoothed_padding_pil), smoothness)
                padding_masks.append(smoothed_padding_mask)
            else:
                masks.append(pil2tensor(mask_image))
                padding_masks.append(pil2tensor(padding_mask))
                
        final_masks = torch.cat(masks, dim=0) if len(masks) > 1 else masks[0].unsqueeze(0)
        final_padding_masks = torch.cat(padding_masks, dim=0) if len(padding_masks) > 1 else padding_masks[0].unsqueeze(0)
        
        # 确保padding_mask在0-1范围内
        final_padding_masks = torch.clamp(final_padding_masks, 0, 1)
        
        return final_masks, final_padding_masks



class Image_smooth_blur:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 150, "step": 1, "display": "slider"}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_expansion": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "mask_color": (["image", "Alpha", "white", "black", "red", "green", "blue", "gray"], {"default": "white"}),
                "bg_color": (["image", "Alpha", "white", "black", "red", "green", "blue", "gray"], {"default": "Alpha"}),
            },
            "optional": {
                "brightness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("image", "smooth_mask","invert_mask")

    CATEGORY = "Apt_Preset/image"

    FUNCTION = "apply_smooth_blur"
    def apply_smooth_blur(self, image, mask, smoothness, invert_mask=False, mask_expansion=0, mask_color="image", brightness=1.0, bg_color="Alpha"):
        batch_size = image.shape[0]
        result_images = []
        smoothed_masks = []
        
        color_map = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "gray": (128, 128, 128)
        }
        
        for i in range(batch_size):
            current_image = image[i].clone()
            current_mask = mask[i] if i < mask.shape[0] else mask[0]
            
            if current_image.shape[-1] == 4:
                current_image = current_image[:, :, :3]
            
            if smoothness > 0:
                mask_tensor = smoothness_mask(current_mask, smoothness)  # 沿用原始平滑函数
            else:
                mask_tensor = current_mask.clone()
            
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            elif mask_tensor.dim() > 2:
                mask_tensor = mask_tensor.squeeze()
                while mask_tensor.dim() > 2:
                    mask_tensor = mask_tensor.squeeze(0)
            
            if mask_expansion != 0:
                kernel_size = abs(mask_expansion) * 2 + 1
                if mask_expansion > 0:
                    from torch.nn import functional as F
                    mask_tensor = F.max_pool2d(mask_tensor.unsqueeze(0).unsqueeze(0), kernel_size, 1, padding=mask_expansion).squeeze()
                else:
                    from torch.nn import functional as F
                    mask_tensor = F.avg_pool2d(mask_tensor.unsqueeze(0).unsqueeze(0), kernel_size, 1, padding=-mask_expansion).squeeze()
                    mask_tensor = (mask_tensor > 0.5).float()
            
            if invert_mask:
                mask_tensor = 1.0 - mask_tensor
            
            smoothed_mask = mask_tensor.clone()
            
            unblurred_tensor = current_image.clone()
            
            if current_image.shape[-1] != 3:
                if current_image.shape[-1] == 1:
                    current_image = current_image.repeat(1, 1, 3)
                    unblurred_tensor = unblurred_tensor.repeat(1, 1, 3)
            
            mask_expanded = mask_tensor.unsqueeze(-1).repeat(1, 1, 3)
            
            # -------------------------- 新增：亮度调节逻辑 --------------------------
            # 仅当mask_color为"image"时，对遮罩区域进行灰度化+亮度调节
            adjusted_gray_image = current_image.clone()
            if mask_color == "image":
                # 1. Tensor转PIL图像（适配亮度调节接口）
                current_image_pil = Image.fromarray((255. * current_image).cpu().numpy().astype(np.uint8))
                # 2. 转为灰度图（消除色彩，保留亮度通道）
                gray_image_pil = current_image_pil.convert('L').convert('RGB')
                # 3. 根据brightness参数调节灰度亮度（0.0纯黑，10.0纯白）
                brightness_enhancer = ImageEnhance.Brightness(gray_image_pil)
                adjusted_gray_pil = brightness_enhancer.enhance(brightness)
                # 4. PIL转回Tensor（匹配原数据格式）
                adjusted_gray_np = np.array(adjusted_gray_pil).astype(np.float32) / 255.0
                adjusted_gray_image = torch.from_numpy(adjusted_gray_np)
            # ----------------------------------------------------------------------

            # 处理mask_color填充逻辑（将原current_image替换为调节后的adjusted_gray_image）
            if mask_color == "image":
                mask_fill = adjusted_gray_image  # 使用亮度调节后的灰度图填充遮罩
            elif mask_color == "Alpha":
                mask_fill = torch.zeros_like(current_image)  # 透明区域用黑色填充
            else:
                mask_fill = torch.zeros_like(current_image)
                r, g, b = color_map[mask_color]
                mask_fill[:, :, 0] = r / 255.0
                mask_fill[:, :, 1] = g / 255.0
                mask_fill[:, :, 2] = b / 255.0
            
            result_tensor = mask_fill * mask_expanded + unblurred_tensor * (1 - mask_expanded)
            
            # 处理bg_color背景逻辑（保持原始逻辑不变）
            if bg_color == "image":
                bg_tensor = unblurred_tensor  # 使用原图作为背景
            elif bg_color != "Alpha":
                bg_tensor = torch.zeros_like(current_image)
                if bg_color in color_map:
                    r, g, b = color_map[bg_color]
                    bg_tensor[:, :, 0] = r / 255.0
                    bg_tensor[:, :, 1] = g / 255.0
                    bg_tensor[:, :, 2] = b / 255.0
                
                result_tensor = result_tensor * mask_expanded + bg_tensor * (1 - mask_expanded)
            
            result_images.append(result_tensor.unsqueeze(0))
            smoothed_masks.append(smoothed_mask.unsqueeze(0))
        
        final_image = torch.cat(result_images, dim=0)
        final_mask = torch.cat(smoothed_masks, dim=0)
        final_invert_mask = 1.0 - final_mask
        return (final_image, final_mask, final_invert_mask)




class Image_tensor_Converter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "output_rgb": ("BOOLEAN", {"default": True, "label_on": "RGB 3通道", "label_off": "含Alpha 4通道"}),
                "keep_batch_dim": ("BOOLEAN", {"default": False, "label_on": "保留批次维度", "label_off": "仅最后一个元素"}),
                "divisible": ("INT", {"default": 8, "min": 1, "max": 128, "step": 1, "label": "尺寸调整为该数的整数倍"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "Apt_Preset/image/😺backup"


    def convert(self, image, output_rgb=True, keep_batch_dim=True, divisible=8):
        # 确保输入是4维张量（B, H, W, C）
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]
        converted_images = []
        
        for i in range(batch_size):
            single = image[i]
            h, w = single.shape[:2]
            c = single.shape[2]
            
            # 调整尺寸为divisible的整数倍
            new_h = ((h + divisible - 1) // divisible) * divisible
            new_w = ((w + divisible - 1) // divisible) * divisible
            if new_h != h or new_w != w:
                pil = tensor2pil(single)
                pil = pil.resize((new_w, new_h), Image.LANCZOS)
                single = pil2tensor(pil)
                h, w = new_h, new_w
            
            # 处理通道数（RGB 3通道或含Alpha 4通道）
            if output_rgb:
                if c == 1:
                    single = single.repeat(1, 1, 3)  # 单通道转RGB
                elif c == 4:
                    single = single[:, :, :3]  # 去除Alpha通道
            else:
                if c == 1:
                    rgb = single.repeat(1, 1, 3)
                    alpha = torch.ones((h, w, 1), dtype=single.dtype, device=single.device)
                    single = torch.cat([rgb, alpha], dim=2)  # 单通道转RGBA
                elif c == 3:
                    alpha = torch.ones((h, w, 1), dtype=single.dtype, device=single.device)
                    single = torch.cat([single, alpha], dim=2)  # RGB转RGBA
            
            converted_images.append(single.unsqueeze(0))
        
        result = torch.cat(converted_images, dim=0)
        
        # 不保留批次维度时，仅输出最后一个元素（保持4维）
        if not keep_batch_dim:
            result = result[-1].unsqueeze(0)
        
        return (result,)





class Image_CnMapMix:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "blur_1": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "blur_2": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "diff_sensitivity": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.01}),
                "diff_blur": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "blend_mode": (
                    ["normal", "multiply", "screen", "overlay", "soft_light", 
                     "difference", "add", "subtract", "lighten", "darken"],
                ),
                "blend_factor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "mask2_smoothness": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                # 新增：image1 黑-白重定向参数（min=黑，max=白）
                "image1_min_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image1_max_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                # 新增：image2 黑-白重定向参数（min=黑，max=白）
                "image2_min_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image2_max_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image1": ("IMAGE",),  # 原image2
                "image2": ("IMAGE",),  # 原image1
                "mask2": ("MASK",),    # 原mask1
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAME = ("image",)
    FUNCTION = "fuse_depth"
    CATEGORY = "Apt_Preset/image"
    DESCRIPTION = "为image1和image2分别添加黑-白重定向：低于min的转纯黑，高于max的转纯白，中间像素比例拉伸"

    def map_black_white_range(self, image, min_black, max_white):
        """黑-白重定向：低于min_black→纯黑，高于max_white→纯白，中间范围线性拉伸"""
        # 处理min >= max的极端情况（避免除以0）
        if min_black >= max_white:
            # 此时所有像素要么转黑要么转白，取中间值分割
            mid = (min_black + max_white) / 2
            return torch.where(image < mid, torch.tensor(0.0, device=image.device), torch.tensor(1.0, device=image.device))
        
        # 线性拉伸公式：(x - min_black) / (max_white - min_black)
        mapped = (image - min_black) / (max_white - min_black)
        # 确保输出在0-1范围（超出部分已被min/max截断为纯黑/纯白）
        return torch.clamp(mapped, 0.0, 1.0)

    def smoothness_mask(self, mask, smoothness):
        if smoothness <= 0:
            return mask
        kernel_size = smoothness * 2 + 1
        sigma = smoothness / 3
        kernel = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=mask.device)
        kernel = torch.exp(-0.5 * (kernel / sigma)**2)
        kernel = kernel / kernel.sum()
        kernel_2d = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        batch_size, height, width, channels = mask.shape
        blurred = torch.nn.functional.conv2d(
            mask.permute(0, 3, 1, 2), 
            kernel_2d.repeat(channels, 1, 1, 1), 
            padding=padding, 
            groups=channels
        ).permute(0, 2, 3, 1)
        return blurred

    def process_mask(self, mask, target_shape, device):
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze()
            if len(mask.shape) == 3:
                mask = mask[0]
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np).convert("L")
        else:
            mask_pil = mask.convert("L") if hasattr(mask, 'convert') else Image.fromarray(mask).convert("L")
        
        target_h, target_w = target_shape[1], target_shape[2]
        if mask_pil.size != (target_w, target_h):
            mask_pil = mask_pil.resize((target_w, target_h), Image.LANCZOS)
        
        mask_np = np.array(mask_pil).astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(-1).to(device)
        
        return torch.clamp(mask_tensor, 0.0, 1.0)

    def fuse_depth(self, blur_1, blur_2, diff_blur, blend_mode, 
                  blend_factor, contrast, brightness, mask2_smoothness,
                  diff_sensitivity, invert_mask,
                  image1_min_black, image1_max_white, image2_min_black, image2_max_white,
                  image1=None, image2=None, mask2=None):
        # 处理image1和image2的默认逻辑（参数名对调）
        if image1 is None and image2 is not None:
            image1 = image2.clone()
        elif image2 is None and image1 is not None:
            image2 = image1.clone()
        elif image1 is None and image2 is None:
            raise ValueError("至少需要提供image1或image2中的一个")
        
        # 尺寸对齐逻辑（参数名对调）
        if image1.shape != image2.shape:
            image2 = image2.permute(0, 3, 1, 2)
            image2 = comfy.utils.common_upscale(
                image2,
                image1.shape[2],
                image1.shape[1],
                upscale_method='bicubic',
                crop='center'
            )
            image2 = image2.permute(0, 2, 3, 1)

        # 处理mask2（原mask1）
        if mask2 is None:
            mask2 = torch.ones_like(image1[..., :1], device=image1.device)
        else:
            mask2 = self.process_mask(mask2, image1.shape, image2.device)
        
        # 遮罩反转
        if invert_mask:
            mask2 = 1.0 - mask2
        
        # 平滑处理
        mask2 = self.smoothness_mask(mask2, mask2_smoothness)

        # 图像转灰度（参数名对调）
        if image1.shape[-1] == 3:
            image1 = (image1 * torch.tensor([0.299, 0.587, 0.114], device=image1.device)).sum(dim=-1, keepdim=True)
        else:
            image1 = image1[:, :, :, 0:1]

        if image2.shape[-1] == 3:
            image2 = (image2 * torch.tensor([0.299, 0.587, 0.114], device=image2.device)).sum(dim=-1, keepdim=True)
        else:
            image2 = image2[:, :, :, 0:1]

        # 关键新增：执行黑-白重定向
        image1 = self.map_black_white_range(image1, image1_min_black, image1_max_white)
        image2 = self.map_black_white_range(image2, image2_min_black, image2_max_white)

        # 设备对齐
        image1 = image1.to(image2.device)
        mask2 = mask2.to(image2.device)
        
        # 关键修改：模糊参数与图像的对应关系对调
        blurred_a = self.gaussian_blur(image1, blur_1)  # 新image1（原image2）使用blur_1
        blurred_b = self.gaussian_blur(image2, blur_2)  # 新image2（原image1）使用blur_2

        # 差异计算与融合逻辑
        diff = torch.abs(blurred_a - blurred_b) - diff_sensitivity
        mask_raw = (diff > 0).float()
        mask_blurred = self.gaussian_blur(mask_raw, diff_blur)

        mode_result = self.apply_blend_mode(blurred_a, blurred_b, blend_mode)
        
        blended_mode = blurred_a * (1 - blend_factor) + mode_result * blend_factor
        fused = blurred_a * (1 - mask2 * mask_blurred) + blended_mode * (mask2 * mask_blurred)

        # 对比度和亮度调整
        fused = (fused - 0.5) * contrast + 0.5 + brightness
        fused = torch.clamp(fused, 0.0, 1.0)

        # 转RGB输出
        fused_rgb = torch.cat([fused, fused, fused], dim=-1)
        return (fused_rgb,)

    def apply_blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        elif mode == "multiply":
            return img1 * img2
        elif mode == "screen":
            return 1 - (1 - img1) * (1 - img2)
        elif mode == "overlay":
            return torch.where(img1 <= 0.5, 2 * img1 * img2, 1 - 2 * (1 - img1) * (1 - img2))
        elif mode == "soft_light":
            factor = 2 * img2 - 1
            low_values = img1 + factor * (img1 - img1 * img1)
            high_values = img1 + factor * (torch.sqrt(img1) - img1)
            return torch.where(img2 <= 0.5, low_values, high_values)
        elif mode == "difference":
            return torch.abs(img1 - img2)
        elif mode == "add":
            return torch.clamp(img1 + img2, 0.0, 1.0)
        elif mode == "subtract":
            return torch.clamp(img1 - img2, 0.0, 1.0)
        elif mode == "lighten":
            return torch.max(img1, img2)
        elif mode == "darken":
            return torch.min(img1, img2)
        return img2

    def gaussian_blur(self, image, radius):
        if radius == 0:
            return image
        
        kernel_size = int(radius * 6 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        
        sigma = radius if radius > 0 else 0.5
        
        kernel = torch.linspace(-(kernel_size//2), kernel_size//2, kernel_size, device=image.device)
        kernel = torch.exp(-0.5 * (kernel / sigma)**2)
        kernel = kernel / kernel.sum()
        
        kernel_2d = torch.outer(kernel, kernel).unsqueeze(0).unsqueeze(0)
        padding = kernel_size // 2
        
        batch_size, height, width, channels = image.shape
        blurred = torch.nn.functional.conv2d(
            image.permute(0, 3, 1, 2),
            kernel_2d.repeat(channels, 1, 1, 1),
            padding=padding,
            groups=channels
        ).permute(0, 2, 3, 1)
        
        return blurred



class chx_Ksampler_inpaint:   
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "latent_image": ("IMAGE", ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0, "max": 1, "step": 0.01}),
                "work_pattern": (["ksampler", "only_adjust_mask"], {"default": "ksampler"}),
                "crop_mode": (["no_crop", "no_scale_crop", "scale_crop_image", ], {"default": "no_scale_crop"}),
                "long_side": ("INT", {"default": 512, "min": 16, "max": 2048, "step": 2}),
                "expand_width": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "expand_height": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),     
                "out_smoothness": ("INT", {"default": 2, "min": 0, "max": 150, "step": 1}),
            },
            "optional": {
                "latent_mask": ("MASK", ),
                "pos": ("STRING", {"multiline": True, "default": "", "placeholder": ""}),
                "mask_stack": ("MASK_STACK2",),  
                "sample_stack": ("SAMPLE_STACK",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("context", "image",  "cropped_mask","cropped_image")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample"

    def run(self, context, seed, latent_image=None, latent_mask=None, denoise=1, pos="",
            work_pattern="ksampler", sample_stack=None, mask_sampling=True, out_smoothness=0.0,
            mask_stack=None,  crop_mode="no_crop", long_side=512,
            expand_width=0, expand_height=0, ):
        
        divisible_by=1
        if latent_mask is None:
            batch_size, height, width, _ = latent_image.shape
            latent_mask = torch.ones((batch_size, height, width), dtype=torch.float32)
            
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")

        if sample_stack is not None:
            steps, cfg, sampler, scheduler = sample_stack   
            if steps == 0: 
                steps = context.get("steps")
            if cfg == 0: 
                cfg = context.get("cfg")
            if scheduler == None: 
                scheduler = context.get("scheduler")
            if sampler == None: 
                sampler = context.get("sampler")    
        else:
            steps = context.get("steps")       
            cfg = context.get("cfg")
            scheduler = context.get("scheduler")
            sampler = context.get("sampler")  

        guidance = context.get("guidance", 3.5)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)

        background_tensor = None
        background_mask_tensor = None
        cropped_image_tensor = None
        cropped_mask_tensor = None
        stitch = None

        if latent_image is not None and latent_mask is not None :
            background_tensor, background_mask_tensor, cropped_image_tensor, cropped_mask_tensor, stitch = Image_solo_crop().inpaint_crop(
                    image=latent_image,
                    crop_mode = crop_mode,
                    long_side = long_side,  
                    upscale_method ="bicubic", 
                    expand_width = expand_width, 
                    expand_height = expand_height, 
                    auto_expand_square=False,
                    divisible_by = divisible_by,
                    mask=latent_mask, 
                    mask_stack=mask_stack, 
                    crop_img_bj="image")

            processed_image = cropped_image_tensor     
            processed_mask = cropped_mask_tensor

            if work_pattern == "only_adjust_mask": 
                return (context, latent_image, cropped_mask_tensor, cropped_image_tensor)

            encoded_result = encode(vae, processed_image)[0]
            if isinstance(encoded_result, dict):
                if "samples" in encoded_result:
                    encoded_latent = encoded_result["samples"]
                else:
                    raise ValueError(f"Encoded result dict doesn't contain 'samples' key. Keys: {list(encoded_result.keys())}")
            elif torch.is_tensor(encoded_result):
                encoded_latent = encoded_result
            else:
                try:
                    encoded_latent = torch.tensor(encoded_result)
                except Exception as e:
                    raise TypeError(f"Cannot convert encoded result to tensor. Type: {type(encoded_result)}, Error: {e}")

            if encoded_latent.dim() == 5:
                if encoded_latent.shape[2] == 1:
                    encoded_latent = encoded_latent.squeeze(2)
                else:
                     encoded_latent = encoded_latent.view(encoded_latent.shape[0], 
                                                    encoded_latent.shape[1], 
                                                    encoded_latent.shape[3], 
                                                    encoded_latent.shape[4])
            elif encoded_latent.dim() == 3:
                encoded_latent = encoded_latent.unsqueeze(0)
            elif encoded_latent.dim() != 4:
                raise ValueError(f"Unexpected latent dimensions: {encoded_latent.dim()}. Expected 4D tensor (B,C,H,W). Shape: {encoded_latent.shape}")

            if encoded_latent.size(0) > 1:
                encoded_latent = encoded_latent[:1]

            latent2 = encoded_latent              
            if not isinstance(latent2, dict):
                if torch.is_tensor(latent2):
                    latent2 = {"samples": latent2}
                else:
                    raise ValueError(f"Unexpected latent format: {type(latent2)}")
            if "samples" not in latent2:
                raise ValueError("Latent dictionary must contain 'samples' key")

            # 核心修复：遮罩适配（匹配latent尺寸+通道数）
            if mask_sampling == False:
                latent3 = latent2
            else:
                if processed_mask is not None:
                    if not torch.is_tensor(processed_mask):
                        processed_mask = torch.tensor(processed_mask, device=encoded_latent.device)
                    # 调整遮罩维度为 (B, 1, H, W)
                    if processed_mask.dim() == 2:
                        processed_mask = processed_mask.unsqueeze(0).unsqueeze(1)
                    elif processed_mask.dim() == 3:
                        processed_mask = processed_mask.unsqueeze(1)
                    # 缩放到latent尺寸
                    latent_h, latent_w = encoded_latent.shape[2], encoded_latent.shape[3]
                    processed_mask = torch.nn.functional.interpolate(
                        processed_mask,
                        size=(latent_h, latent_w),
                        mode='bilinear',
                        align_corners=False
                    )
                    # 扩展为4通道（匹配latent）
                    if processed_mask.shape[1] == 1:
                        processed_mask = processed_mask.repeat(1, 4, 1, 1)
                    # 归一化遮罩值
                    processed_mask = torch.clamp(processed_mask, 0.0, 1.0)
                    
                    latent3 = copy.deepcopy(latent2)
                    latent3["noise_mask"] = processed_mask
                else:
                    latent3 = latent2

            result = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent3, denoise=denoise)
            latent_result = result[0]
            output_image = decode(vae, latent_result)[0]

            fimage, output_image, original_image = Image_solo_stitch().inpaint_stitch(
                inpainted_image=output_image,
                smoothness=out_smoothness, 
                mask=cropped_mask_tensor, 
                stitch=stitch, 
                blend_factor=1.0, 
                blend_mode="normal", 
                opacity=1.0, 
                stitch_mode="crop_mask", 
                recover_method="bicubic")

            latent = encode(vae, output_image)[0]
            context = new_context(context, latent=latent, images=output_image)

            return (context, output_image, cropped_mask_tensor, cropped_image_tensor)











