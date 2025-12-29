
import torch
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor

import torchvision.transforms.functional as TF
import math
from comfy.utils import common_upscale
import re
from math import ceil, sqrt
from typing import cast
from PIL import Image, ImageDraw,  ImageFilter, ImageEnhance, ImageDraw, ImageFont, ImageOps
import random
import folder_paths
import copy
import ast
from nodes import CLIPTextEncode, common_ksampler,InpaintModelConditioning
import torch.nn.functional as F
import node_helpers
from typing import Tuple

import comfy.utils


from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader,composite



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#---------------------安全导入

try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  

try:
    from scipy.interpolate import CubicSpline
    REMOVER_AVAILABLE = True  
except ImportError:
    CubicSpline = None
    REMOVER_AVAILABLE = False 

try:
    import onnxruntime as ort
    REMOVER_AVAILABLE = True  
except ImportError:
    ort = None
    REMOVER_AVAILABLE = False  

try:   
    from scipy.ndimage import distance_transform_edt
    REMOVER_AVAILABLE = True  
except ImportError:
    distance_transform_edt = None
    REMOVER_AVAILABLE = False 



try:
    from tqdm import tqdm
    REMOVER_AVAILABLE = True  
except ImportError:
    tqdm = None
    REMOVER_AVAILABLE = False  


try:
    from enum import Enum
    REMOVER_AVAILABLE = True  
except ImportError:
    Enum = None
    REMOVER_AVAILABLE = False  





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
    CATEGORY = "Apt_Preset/image/ImgBatch"

    def composite(self, batch_image, bg_image, x, y, resize_source, Invert, batch_mask = None):
        if Invert and batch_mask is not None:
            batch_mask = 1 - batch_mask
        batch_image = batch_image.clone().movedim(-1, 1)


        output = composite(batch_image, bg_image.movedim(-1, 1), x, y, batch_mask, 1, resize_source).movedim(1, -1)
        return (output,)





import asyncio
from comfy_api.latest import io, ui
import torch



class Image_batch_select(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image_batch_select",
            display_name="Image_batch_select",
            category="Apt_Preset/image/ImgBatch",

            inputs=[
                io.Image.Input("image"),
                io.Combo.Input("mode", options=["index", "step", "uniform"], default="step"),
                io.Int.Input("index", default=0, min=-8192, max=8192, step=1),
                io.Int.Input("step", default=4, min=1, max=8192, step=1),
                io.Int.Input("uniform", default=4, min=0, max=8192, step=1),
                io.Int.Input("max_output", default=10, min=0, max=8192, step=1),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    async def execute(
        cls,
        image: torch.Tensor,
        mode: str,
        index: int,
        step: int,
        uniform: int,
        max_output: int,
    ) -> io.NodeOutput:
        try:
            batch_size = int(image.shape[0])

            indices = await asyncio.to_thread(
                cls._get_extract_indices, batch_size, mode, index, step, uniform
            )

            if not indices:
                empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                return io.NodeOutput(empty)

            if max_output > 0 and len(indices) > max_output:
                indices = indices[:max_output]

            valid = [i for i in indices if 0 <= i < batch_size]
            invalid = [i for i in indices if not (0 <= i < batch_size)]

            if not valid:
                empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
                return io.NodeOutput(empty)

            chunk_size = 512
            chunks = [valid[i : i + chunk_size] for i in range(0, len(valid), chunk_size)]
            async def _gather_chunk(chunk):
                def _select():
                    return image[chunk]
                return await asyncio.to_thread(_select)

            tasks = [_gather_chunk(ch) for ch in chunks]
            parts = await asyncio.gather(*tasks)
            extracted = torch.cat(parts, dim=0)

            return io.NodeOutput(extracted)
        except Exception as e:
            empty = torch.empty((0,) + image.shape[1:], dtype=image.dtype, device=image.device)
            return io.NodeOutput(empty)

    @classmethod
    def _get_extract_indices(cls, batch_size: int, mode: str, index: int, step: int, uniform: int):
        try:
            if mode == "index":
                return cls._parse_custom_index(index, batch_size)
            if mode == "step":
                if step < 1:
                    return []
                return cls._calculate_step_indices(batch_size, step)
            if mode == "uniform":
                if uniform <= 0:
                    return []
                return cls._calculate_count_indices(batch_size, uniform)
            return []
        except Exception as e:
            return []

    @staticmethod
    def _parse_custom_index(index: int, batch_size: int | None = None):
        try:
            idx = index
            if batch_size is not None and idx < 0:
                idx = batch_size + idx
            return [idx]
        except Exception as e:
            return []

    @staticmethod
    def _calculate_step_indices(batch_size: int, step: int):
        idxs = list(range(0, batch_size, step))
        return idxs

    @staticmethod
    def _calculate_count_indices(batch_size: int, count: int):
        if count <= 0:
            return []
        if count == 1:
            return [0]
        if count == 2:
            return [0, batch_size - 1] if batch_size > 1 else [0]
        if count >= batch_size:
            return list(range(batch_size))
        step = (batch_size - 1) / float(count - 1)
        idxs = [int(round(i * step)) for i in range(count)]
        idxs[-1] = batch_size - 1
        idxs = sorted(list(set(idxs)))
        return idxs








class Image_UpscaleModel:

    @classmethod
    def INPUT_TYPES(s):
        upscale_model_list = ["None"] + folder_paths.get_filename_list("upscale_models")
        upscale_pixel_list = ["None", "nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
        
        return {"required":
                    {"image": ("IMAGE",),
                     "upscale_model": (upscale_model_list, ),
                     "pixel_upscale": (upscale_pixel_list, {"default": "bilinear" }),
                     "pixel_mode": (["rescale", "resize"],{"default": "resize" }),
                     "pixel_rescale": ("FLOAT", {"default": 1, "min": 0.1, "max": 64.0, "step": 0.1}),
                     "pixel_resize_long_side": ("INT", {"default": 1024, "min": 1, "max": 48000, "step": 1}),
                     "rounding_modulus": ("INT", {"default": 8, "min": 0, "max": 1024, "step": 2}),
                     }
                }

    FUNCTION = "upscale"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image", )
    CATEGORY = "Apt_Preset/image/ImgResize"
    DESCRIPTION = """
    - 当upscale_model=None时，仅像素缩放
    - 当upscale_pixel=None时，进模型缩放
    - 当同时存在时，则先进行模型缩放，再进行像素缩放
    """ 

    @classmethod
    def apply_resize_image(cls, image: Image.Image, original_width, original_height, rounding_modulus, mode='scale', factor: float = 2.0, resize_long_side: int = 1024, resample='bicubic'):
        if resample is None:
            return image
            
        if mode == 'rescale':
            new_width = int(original_width * factor)
            new_height = int(original_height * factor)
        else:
            is_width_longer = original_width > original_height
            
            if is_width_longer:
                scale_ratio = resize_long_side / original_width
                new_width = resize_long_side
                new_height = int(original_height * scale_ratio)
            else:
                scale_ratio = resize_long_side / original_height
                new_height = resize_long_side
                new_width = int(original_width * scale_ratio)
        
        m = rounding_modulus
        if m != 0:
            new_width = new_width if new_width % m == 0 else new_width + (m - new_width % m)
            new_height = new_height if new_height % m == 0 else new_height + (m - new_height % m)
        
        resample_filters = {
            'nearest': 0,
            'bilinear': 2,
            'bicubic': 3,
            'lanczos': 1,
            'area': Image.Resampling.BOX
        }
        if resample == "nearest-exact":
            resample_type = Image.Resampling.NEAREST
        else:
            resample_type = Image.Resampling(resample_filters[resample])
        
        image = image.resize((new_width * 8, new_height * 8), resample=resample_type)
        resized_image = image.resize((new_width, new_height), resample=resample_type)
        
        return resized_image

    def upscale(self, image, upscale_model, rounding_modulus=8, pixel_mode="rescale", pixel_upscale="bilinear", pixel_rescale=2.0, pixel_resize_long_side=1024):
        if upscale_model == "None" and pixel_upscale == "None":
            raise ValueError(f"Select at least one scaling model or pixel scaling")
        
        if upscale_model != "None":
            up_model = load_upscale_model(upscale_model)
            up_image = upscale_with_model(up_model, image)
        else:
            up_image = image
        
        original_width, original_height = 0, 0
        for img in image:
            pil_img = tensor2pil(img)
            original_width, original_height = pil_img.size
            break
        
        scaled_images = []
        for img in up_image:
            pil_img = tensor2pil(img)
            
            if pixel_upscale != "None":
                resized_pil = self.apply_resize_image(
                    pil_img,
                    original_width,
                    original_height,
                    rounding_modulus,
                    pixel_mode,
                    pixel_rescale,
                    pixel_resize_long_side,
                    pixel_upscale
                )
            else:
                if upscale_model != "None":
                    m = rounding_modulus
                    if m != 0:
                        current_width, current_height = pil_img.size
                        new_width = current_width if current_width % m == 0 else current_width + (m - current_width % m)
                        new_height = current_height if current_height % m == 0 else current_height + (m - current_height % m)
                        resized_pil = pil_img.resize((new_width, new_height), Image.Resampling.BILINEAR)
                    else:
                        resized_pil = pil_img
                else:
                    resized_pil = pil_img
            
            scaled_images.append(pil2tensor(resized_pil))
        
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
                "canvas_image": ("IMAGE",),
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
        canvas_image=None,
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
            "nearest-exact": Image.NEAREST,
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

        use_canvas = canvas_image is not None
        if use_canvas:
            canvas_frames, canvas_h, canvas_w, canvas_ch = canvas_image.size()
            if canvas_ch != 3:
                raise ValueError("Canvas image must be 3-channel RGB image")
            canvas_pil_list = list_tensor2pil(canvas_image)
            canvas_width, canvas_height = canvas_w, canvas_h
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            img_half_w = frame_width // 2
            img_half_h = frame_height // 2
        else:
            canvas_width, canvas_height = int(frame_width * zoom), int(frame_height * zoom)
            canvas_pil_list = None
            canvas_center_x = canvas_width // 2
            canvas_center_y = canvas_height // 2
            img_half_w = frame_width // 2
            img_half_h = frame_height // 2

        max_transform_range = math.ceil(math.sqrt(canvas_width**2 + canvas_height**2))
        constant_color = color_map.get(constant_color, (0, 0, 0))

        for idx, img in enumerate(list_tensor2pil(image)):
            if use_canvas:
                canvas_pil = canvas_pil_list[idx % len(canvas_pil_list)]
                canvas_pil = canvas_pil.resize((canvas_width, canvas_height), resampling_filter)
                
                buffer_size = max(max_transform_range, canvas_width, canvas_height) * 2
                buffer_center_x = buffer_size // 2
                buffer_center_y = buffer_size // 2
                
                buffer_img = Image.new("RGB", (buffer_size, buffer_size), (0, 0, 0))
                buffer_paste_x = buffer_center_x - img_half_w
                buffer_paste_y = buffer_center_y - img_half_h
                buffer_img.paste(img, (buffer_paste_x, buffer_paste_y))
                
                translate_x = x
                translate_y = y
                
                transformed_buffer = cast(Image.Image, TF.affine(
                    buffer_img,
                    angle=angle,
                    scale=zoom,
                    translate=[translate_x, translate_y],
                    shear=shear,
                    interpolation=resampling_filter,
                    fill=0
                ))
                
                crop_left = buffer_center_x - canvas_center_x
                crop_top = buffer_center_y - canvas_center_y
                crop_right = crop_left + canvas_width
                crop_bottom = crop_top + canvas_height
                transformed_img = transformed_buffer.crop((crop_left, crop_top, crop_right, crop_bottom))
                
                img_alpha = Image.new("L", img.size, 255)
                buffer_alpha = Image.new("L", (buffer_size, buffer_size), 0)
                buffer_alpha.paste(img_alpha, (buffer_paste_x, buffer_paste_y))
                transformed_alpha = cast(Image.Image, TF.affine(
                    buffer_alpha,
                    angle=angle,
                    scale=zoom,
                    translate=[translate_x, translate_y],
                    shear=shear,
                    interpolation=Image.NEAREST,
                    fill=0
                ))
                transformed_alpha = transformed_alpha.crop((crop_left, crop_top, crop_right, crop_bottom))
                
                final_img = Image.composite(transformed_img, canvas_pil, transformed_alpha)
            else:
                pw = int(frame_width - canvas_width)
                ph = int(frame_height - canvas_height)
                padding = [
                    max(0, pw + x),
                    max(0, ph + y),
                    max(0, pw - x),
                    max(0, ph - y),
                ]
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

            if use_canvas:
                final_mask = transformed_alpha
            else:
                pw = int(frame_width - canvas_width)
                ph = int(frame_height - canvas_height)
                padding = [
                    max(0, pw + x),
                    max(0, ph + y),
                    max(0, pw - x),
                    max(0, ph - y),
                ]
                mask_padded = Image.new("L", (padded_img.width, padded_img.height), 255)
                draw = ImageDraw.Draw(mask_padded)
                orig_x, orig_y = padding[0], padding[1]
                orig_w, orig_h = img.size
                draw.rectangle([orig_x, orig_y, orig_x + orig_w, orig_y + orig_h], fill=0)
                mask_affine = cast(Image.Image, TF.affine(
                    mask_padded,
                    angle=angle,
                    scale=zoom,
                    translate=[x, y],
                    shear=shear,
                    interpolation=Image.NEAREST,
                ))
                left, upper = abs(padding[0]), abs(padding[1])
                right, bottom = mask_affine.width - abs(padding[2]), mask_affine.height - abs(padding[3])
                final_mask = mask_affine.crop((left, upper, right, bottom))

            padding_masks.append(final_mask)

        return (
            list_pil2tensor(transformed_images),
            list_pil2tensor(padding_masks).squeeze(1)
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
    CATEGORY = "Apt_Preset/image/color_adjust"

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
    CATEGORY = "Apt_Preset/imgEffect/texture"

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
    CATEGORY = "Apt_Preset/image/color_adjust"

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
    CATEGORY = "Apt_Preset/image/color_adjust"

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




class color_TransforTool:
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

    CATEGORY = "Apt_Preset/image/color_adjust"
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

    CATEGORY = "Apt_Preset/image/color_adjust"

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
    CATEGORY = "Apt_Preset/image/color_adjust"

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

    CATEGORY = "Apt_Preset/image/color_adjust"

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
                "long_side": ("INT", {"default": 512, "min": 16, "max": 9999, "step": 2}),
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
                "blending_mode": (BLEND_METHODS, {"default": "normal"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK",)
    RETURN_NAMES = ("bj_composite", "mask", "composite", "line_mask",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image/ImgLayer"

    def process(self, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, mask_expand, smoothness,
                opacity, blending_mode, blend_strength, bj_img=None, fj_img=None, mask_mode="fill", mask=None):
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
        if fj_img is None:
            raise ValueError("前景图像(fj_img)是必需的输入")
        if bj_img is None:
            raise ValueError("背景图像(bj_img)是必需的输入")

        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")

        canvas_width = max(bj_pil.size[0], fj_pil.size[0])
        canvas_height = max(bj_pil.size[1], fj_pil.size[1])
        canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2

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

        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")

            if mask_pil.size != fj_pil.size:
                mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)

            fj_with_mask = fj_pil.copy()
            fj_with_mask.putalpha(mask_pil)

            fj_processed = fj_with_mask
            mask_processed = mask_pil
        else:
            mask_processed = Image.new("L", fj_pil.size, 255)
            fj_processed = fj_pil.copy()
            fj_processed.putalpha(mask_processed)

        processed_width, processed_height = fj_processed.size
        center_x, center_y = processed_width // 2, processed_height // 2

        adjusted_fj = fj_processed
        adjusted_mask = mask_processed

        rotation = float(rotation)

        if rotation != 0 or scale != 1.0:
            adjusted_fj = adjusted_fj.rotate(rotation, center=(center_x, center_y), resample=Image.BICUBIC, expand=True)
            adjusted_mask = adjusted_mask.rotate(rotation, center=(center_x, center_y), resample=Image.BICUBIC, expand=True)

            if scale != 1.0:
                new_size = (int(adjusted_fj.size[0] * scale), int(adjusted_fj.size[1] * scale))
                adjusted_fj = adjusted_fj.resize(new_size, Image.LANCZOS)
                adjusted_mask = adjusted_mask.resize(new_size, Image.LANCZOS)

            center_x, center_y = adjusted_fj.size[0] // 2, adjusted_fj.size[1] // 2

        bj_center_x, bj_center_y = bj_pil.size[0] // 2, bj_pil.size[1] // 2

        x_position = canvas_center_x - center_x + x_offset
        y_position = canvas_center_y - center_y + y_offset

        paste_x = int(x_position)
        paste_y = int(y_position)

        if opacity < 1.0:
            r, g, b, a = adjusted_fj.split()
            a = a.point(lambda p: p * opacity)
            adjusted_fj = Image.merge("RGBA", (r, g, b, a))

        composite_mask_pil = Image.new("L", (canvas_width, canvas_height), 0)
        mask_paste_x = paste_x
        mask_paste_y = paste_y
        mask_left = max(0, mask_paste_x)
        mask_top = max(0, mask_paste_y)
        mask_right = min(canvas_width, mask_paste_x + adjusted_mask.size[0])
        mask_bottom = min(canvas_height, mask_paste_y + adjusted_mask.size[1])

        if mask_right > mask_left and mask_bottom > mask_top:
            crop_left = max(0, -mask_paste_x)
            crop_top = max(0, -mask_paste_y)
            crop_right = crop_left + (mask_right - mask_left)
            crop_bottom = crop_top + (mask_bottom - mask_top)
            mask_crop = adjusted_mask.crop((crop_left, crop_top, crop_right, crop_bottom))
            composite_mask_pil.paste(mask_crop, (mask_left, mask_top))

        composite_pil = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))
        bj_x = (canvas_width - bj_pil.size[0]) // 2
        bj_y = (canvas_height - bj_pil.size[1]) // 2
        composite_pil.paste(bj_pil, (bj_x, bj_y))

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

        if edge_detection:
            if edge_color in color_mapping:
                r, g, b = color_mapping[edge_color]
            else:
                r, g, b = 0, 0, 0

            threshold = 128
            mask_array = np.array(composite_mask_pil)
            binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
            binary_mask_pil = Image.fromarray(binary_mask)

            edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
            edge_draw = ImageDraw.Draw(edge_image)
            mask_cv = np.array(binary_mask_pil)
            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                for i in range(edge_thickness):
                    points = [tuple(point[0]) for point in contour]
                    edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness - i + 1)

            composite_pil = Image.alpha_composite(composite_pil, edge_image)
            edge_mask = np.zeros_like(mask_cv)
            cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
            line_mask_pil = Image.fromarray(edge_mask)
        else:
            line_mask_pil = Image.new("L", composite_pil.size, 0)

        composite_pil = composite_pil.convert("RGB")

        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        mask_np = np.array(composite_mask_pil).astype(np.float32) / 255.0
        line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0

        if len(composite_np.shape) == 2:
            composite_np = np.stack([composite_np] * 3, axis=-1)

        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)

        bj_x_start = (canvas_width - bj_pil.size[0]) // 2
        bj_y_start = (canvas_height - bj_pil.size[1]) // 2
        bj_x_end = bj_x_start + bj_pil.size[0]
        bj_y_end = bj_y_start + bj_pil.size[1]

        cropped_composite = composite_tensor[:, bj_y_start:bj_y_end, bj_x_start:bj_x_end, :]

        cropped_mask_np = mask_np[bj_y_start:bj_y_end, bj_x_start:bj_x_end]
        cropped_mask_tensor = torch.from_numpy(cropped_mask_np).unsqueeze(0).unsqueeze(0)

        cropped_line_mask_np = line_mask_np[bj_y_start:bj_y_end, bj_x_start:bj_x_end]
        cropped_line_mask_tensor = torch.from_numpy(cropped_line_mask_np).unsqueeze(0).unsqueeze(0)

        return (
            cropped_composite,
            cropped_mask_tensor,
            composite_tensor,
            cropped_line_mask_tensor
        )







#region----------------------尺寸调整组合---------



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
    CATEGORY = "Apt_Preset/image/ImgResize"

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
    CATEGORY = "Apt_Preset/image/ImgResize"
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




#endregion----------------------------合并----------




#region--------------------裁切组合------------


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
    CATEGORY = "Apt_Preset/image/ImgResize"


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



class Image_solo_crop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mode": (["no_crop", "no_scale_crop", "scale_crop_image", "scale_bj_image"], {"default": "no_scale_crop"}),
                "long_side": ("INT", {"default": 512, "min": 16, "max": 9999, "step": 2}),
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

    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
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
        mask_np = mask[0].cpu().numpy()
        mask_np = np.squeeze(mask_np)  
        mask_np = (mask_np > 0.5).astype(np.uint8)
        if mask_np.ndim != 2:
            raise ValueError(f"Mask must be 2D array, got {mask_np.ndim}D instead")
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




class Image_solo_crop2:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_mode": (["不裁切", "无缩放_裁切", "缩放后_裁切", ], {"default": "无缩放_裁切"}),
                "long_side": ("INT", {"default": 512, "min": 16, "max": 9999, "step": 2}),
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
    RETURN_TYPES = ( "IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ( "crop_image", "crop_mask", "stitch")
    FUNCTION = "inpaint_crop"
    DESCRIPTION = """
    - 不缩放_裁切: 原始裁切图。不支持缩放
    - 缩放后_裁切: 原始裁切图的长边缩放。
    - 无裁切: 不进行裁剪，原图输出(可增加贴合羽化功能)
    - auto_expand_square自动扩展正方形
    - 遮罩控制: 微调尺寸【目标尺寸相差2~8个像素时】
    """


    def get_mask_bounding_box(self, mask):
        mask_np = mask[0].cpu().numpy()
        mask_np = np.squeeze(mask_np)  
        mask_np = (mask_np > 0.5).astype(np.uint8)
        if mask_np.ndim != 2:
            raise ValueError(f"Mask must be 2D array, got {mask_np.ndim}D instead")
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


        if crop_mode == "缩放后_裁切":
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
        elif crop_mode == "不裁切":
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
        if auto_expand_square and crop_mode in ["无缩放_裁切", "缩放后_裁切"]:
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

        if crop_mode == "不裁切":
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

        return (cropped_image_tensor, cropped_mask_tensor, stitch)












def create_mask_feather(mask, smoothness):
    if smoothness <= 0:
        return mask.clone() if isinstance(mask, torch.Tensor) else torch.tensor(mask).float()
    if isinstance(mask, torch.Tensor):
        mask_np = mask.squeeze().cpu().detach().numpy()
        device = mask.device
    else:
        mask_np = mask.squeeze()
        device = torch.device("cpu")
    mask_np = (mask_np > 0.5).astype(np.uint8)
    dist = cv2.distanceTransform(mask_np, distanceType=cv2.DIST_L2, maskSize=5)
    dist = np.clip(dist, 0, smoothness)
    feather_mask = dist / smoothness
    feather_mask = torch.tensor(feather_mask).float().unsqueeze(0).to(device)
    if isinstance(mask, torch.Tensor) and mask.ndim == 3:
        feather_mask = feather_mask.repeat(mask.shape[0], 1, 1)
    return feather_mask

def create_feather_mask(width, height, feather_size):
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




class Image_solo_stitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inpainted_image": ("IMAGE",),
                "mask": ("MASK",),
                "stitch": ("STITCH2",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, }),
                "blend_factor": ("FLOAT", {"default": 1.0,"min": 0.0,"max": 1.0,"step": 0.01}),
                "blend_mode": (["normal", "multiply", "screen", "overlay", "soft_light", "difference"],),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stitch_mode": (["crop_mask", "crop_image"], {"default": "crop_mask"}),
                "recover_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                # 新增偏移和缩放参数
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
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
            "gray": (128, 128, 128)
        }
        for i in range(batch_size):
            current_image = image[i].clone()
            current_mask = mask[i] if i < mask.shape[0] else mask[0]
            if smoothness > 0:
                mask_tensor = create_mask_feather(current_mask, smoothness)
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

    def inpaint_stitch(self, inpainted_image, smoothness, mask, stitch, blend_factor, blend_mode, opacity, stitch_mode, recover_method, x_offset, y_offset, scale):
        original_h, original_w = stitch["original_shape"]
        crop_x, crop_y = stitch["crop_position"]
        crop_w, crop_h = stitch["crop_size"]
        mask_crop_x, mask_crop_y, mask_crop_x2, mask_crop_y2 = stitch["mask_cropped_position"]
        original_image_h, original_image_w = stitch["original_image_shape"]
        
        # 应用缩放：调整裁剪尺寸和mask裁剪范围
        scaled_crop_w = int(crop_w * scale)
        scaled_crop_h = int(crop_h * scale)
        scaled_mask_crop_x = int(mask_crop_x * scale)
        scaled_mask_crop_y = int(mask_crop_y * scale)
        scaled_mask_crop_x2 = int(mask_crop_x2 * scale)
        scaled_mask_crop_y2 = int(mask_crop_y2 * scale)
        
        # 应用偏移：调整裁剪位置
        crop_x += x_offset
        crop_y += y_offset

        if "bj_image" in stitch:
            bj_image = stitch["bj_image"]
        else:
            bj_image = torch.zeros((1, original_h, original_w, 3), dtype=torch.float32)
        if "original_image" in stitch:
            original_image = stitch["original_image"]
        else:
            original_image = torch.zeros((1, original_image_h, original_image_w, 3), dtype=torch.float32)
        if opacity < 1.0:
            inpainted_image = inpainted_image * opacity
        if inpainted_image.shape[1:3] != mask.shape[1:3]:
            mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(inpainted_image.shape[1], inpainted_image.shape[2]), mode='nearest').squeeze(1)
        
        inpainted_np = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask_np = (mask[0].cpu().numpy() * 255).astype(np.uint8)
        background_np = (bj_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 使用缩放后的尺寸调整图像和mask
        inpainted_resized = cv2.resize(inpainted_np, (scaled_crop_w, scaled_crop_h))
        mask_resized = cv2.resize(mask_np, (scaled_crop_w, scaled_crop_h))
        background_resized = cv2.resize(background_np, (original_w, original_h))
        
        result = np.zeros((original_h, original_w, 4), dtype=np.uint8)
        result[:, :, :3] = background_resized.copy()
        result[:, :, 3] = 255

        if stitch_mode == "crop_mask":
            inpainted_image, mask = self.apply_smooth_blur(inpainted_image, mask, smoothness, bg_color="Alpha")
            inpainted_blurred = (inpainted_image[0].cpu().numpy() * 255).astype(np.uint8)
            mask_blurred = (mask[0].cpu().numpy() * 255).astype(np.uint8)
            
            # 使用缩放后的尺寸调整模糊后的图像和mask
            inpainted_blurred = cv2.resize(inpainted_blurred, (scaled_crop_w, scaled_crop_h))
            mask_blurred = cv2.resize(mask_blurred, (scaled_crop_w, scaled_crop_h))
            
            # 使用缩放后的mask裁剪范围
            mask_content = mask_blurred[scaled_mask_crop_y:scaled_mask_crop_y2, scaled_mask_crop_x:scaled_mask_crop_x2]
            inpaint_content = inpainted_blurred[scaled_mask_crop_y:scaled_mask_crop_y2, scaled_mask_crop_x:scaled_mask_crop_x2]

            if mask_content.size == 0 or inpaint_content.size == 0:
                print("Warning: Mask content is empty, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            
            # 计算粘贴位置（含偏移）
            paste_x_start = max(0, crop_x + scaled_mask_crop_x)
            paste_x_end = min(original_w, crop_x + scaled_mask_crop_x2)
            paste_y_start = max(0, crop_y + scaled_mask_crop_y)
            paste_y_end = min(original_h, crop_y + scaled_mask_crop_y2)

            if paste_x_start >= paste_x_end or paste_y_start >= paste_y_end:
                print("Warning: Invalid paste region, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            
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
                recover_img = fimage
                return (fimage, recover_img, original_image)
            
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
                recover_img = fimage
                return (fimage, recover_img, original_image)
        else:
            feather_mask = create_feather_mask(scaled_crop_w, scaled_crop_h, smoothness)  # 使用缩放后的尺寸
            # 计算粘贴位置（含偏移）
            paste_x_start = max(0, crop_x)
            paste_x_end = min(original_w, crop_x + scaled_crop_w)
            paste_y_start = max(0, crop_y)
            paste_y_end = min(original_h, crop_y + scaled_crop_h)

            inpaint_content = inpainted_resized[
                max(0, paste_y_start - crop_y) : max(0, paste_y_end - crop_y),
                max(0, paste_x_start - crop_x) : max(0, paste_x_end - crop_x)
            ]

            if inpaint_content.size == 0:
                print("Warning: Inpaint content is empty in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            
            if paste_x_start >= paste_x_end or paste_y_start >= paste_y_end:
                print("Warning: Invalid paste region in crop_image mode, returning background image")
                final_image_tensor = torch.from_numpy(background_resized / 255.0).float().unsqueeze(0)
                fimage = Blend().blend_images(bj_image, final_image_tensor, blend_factor, blend_mode)[0]
                recover_img = fimage
                return (fimage, recover_img, original_image)
            
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
                recover_img = fimage
                return (fimage, recover_img, original_image)
            
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
                recover_img = fimage
                return (fimage, recover_img, original_image)

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
        
        return (fimage, recover_img, original_image)




#endregion----------------裁切组合--------------



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
    CATEGORY = "Apt_Preset/image/color_adjust"
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
    CATEGORY = "Apt_Preset/image/ImgResize"
    
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





class Image_safe_size:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "constraint_mode": ("BOOLEAN", {"default": True, "label_off": "Size_limit", "label_on": "Resolution_limit"}),
                "min_width": ("INT", {"default": 2560, "min": 64, "max": 4096, "step": 1}),
                "min_height": ("INT", {"default": 1440, "min": 64, "max": 4096, "step": 1}),
                "max_width": ("INT", {"default": 4096, "min": 128, "max": 8192, "step": 1}),
                "max_height": ("INT", {"default": 4096, "min": 128, "max": 8192, "step": 1}),
                "divisor": ("INT", {"default": 8, "min": 1, "max": 64, "step": 1}),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_safe_size"
    CATEGORY = "Apt_Preset/image/ImgResize"
    
    DESCRIPTION = """
       seedream4.5: 2560*1440 - 4096*4096
       seedream4.0: 1280x720 - 4096x4096
    """


    def adjust_safe_size(self, image, min_width, min_height, max_width, max_height, divisor, constraint_mode):
        divisor = max(1, divisor)
        batch_size, orig_h, orig_w, channels = image.shape
        aspect_ratio = orig_w / orig_h

        # 确保min ≤ max，避免参数设置错误
        min_width = min(min_width, max_width)
        min_height = min(min_height, max_height)

        if constraint_mode:
            target_w, target_h = self._resolve_resolution_constraint(orig_w, orig_h, min_width, min_height, max_width, max_height, divisor, aspect_ratio)
            processed_image = self._scale_image(image, orig_w, orig_h, target_w, target_h)
        else:
            target_w, target_h, cropped_image = self._resolve_size_constraint(image, orig_w, orig_h, min_width, min_height, max_width, max_height, divisor, aspect_ratio)
            processed_image = cropped_image

        target_w = self._align_to_divisor(target_w, divisor)
        target_h = self._align_to_divisor(target_h, divisor)

        return (processed_image,)

    def _resolve_resolution_constraint(self, orig_w, orig_h, min_w, min_h, max_w, max_h, divisor, aspect_ratio):
        min_area = min_w * min_h
        max_area = max_w * max_h
        orig_area = orig_w * orig_h

        target_w = self._align_to_divisor(orig_w, divisor)
        target_h = self._align_to_divisor(orig_h, divisor)
        current_area = target_w * target_h

        if current_area < min_area:
            scale_factor = np.sqrt(min_area / current_area)
            target_w = self._align_to_divisor(target_w * scale_factor, divisor)
            target_h = self._align_to_divisor(target_h * scale_factor, divisor)
            current_area = target_w * target_h

        if current_area > max_area:
            scale_factor = np.sqrt(max_area / current_area)
            target_w = self._align_to_divisor(target_w * scale_factor, divisor)
            target_h = self._align_to_divisor(target_h * scale_factor, divisor)
            current_area = target_w * target_h

        if target_w * target_h < min_area:
            scale_factor = np.sqrt(min_area / (target_w * target_h))
            target_w = self._align_to_divisor(target_w * scale_factor, divisor)
            target_h = self._align_to_divisor(target_h * scale_factor, divisor)
        elif target_w * target_h > max_area:
            scale_factor = np.sqrt(max_area / (target_w * target_h))
            target_w = self._align_to_divisor(target_w * scale_factor, divisor)
            target_h = self._align_to_divisor(target_h * scale_factor, divisor)

        return target_w, target_h

    def _resolve_size_constraint(self, image, orig_w, orig_h, min_w, min_h, max_w, max_h, divisor, aspect_ratio):
        # 优化缩放逻辑：优先判断是否需要缩小，避免无效放大
        if orig_w > max_w or orig_h > max_h:
            scale_factor_max_w = max_w / orig_w
            scale_factor_max_h = max_h / orig_h
            scale_factor_max = min(scale_factor_max_w, scale_factor_max_h)
            scaled_w = orig_w * scale_factor_max
            scaled_h = orig_h * scale_factor_max
        else:
            # 仅当原始尺寸小于最小值时放大
            scale_factor_min_w = min_w / orig_w if orig_w < min_w else 1.0
            scale_factor_min_h = min_h / orig_h if orig_h < min_h else 1.0
            scale_factor_min = max(scale_factor_min_w, scale_factor_min_h)
            scaled_w = orig_w * scale_factor_min
            scaled_h = orig_h * scale_factor_min

            # 放大后检查是否超过最大值，若超过则缩小
            scale_factor_max_w = max_w / scaled_w if scaled_w > max_w else 1.0
            scale_factor_max_h = max_h / scaled_h if scaled_h > max_h else 1.0
            scale_factor_max = min(scale_factor_max_w, scale_factor_max_h)
            scaled_w *= scale_factor_max
            scaled_h *= scale_factor_max

        # 对齐到divisor
        scaled_w = self._align_to_divisor(scaled_w, divisor)
        scaled_h = self._align_to_divisor(scaled_h, divisor)

        # 缩放图像
        scaled_image = self._scale_image(image, orig_w, orig_h, scaled_w, scaled_h)

        # 确定目标尺寸（限制在min/max范围内）
        target_w = self._align_to_divisor(max(min_w, min(max_w, scaled_w)), divisor)
        target_h = self._align_to_divisor(max(min_h, min(max_h, scaled_h)), divisor)

        # 替换填充为中心裁切
        cropped_image = self._center_crop_image(scaled_image, scaled_w, scaled_h, target_w, target_h)

        return target_w, target_h, cropped_image

    def _align_to_divisor(self, value, divisor):
        return int(np.ceil(value / divisor) * divisor)

    def _scale_image(self, image, orig_w, orig_h, target_w, target_h):
        if orig_w == target_w and orig_h == target_h:
            return image
        scaled = comfy.utils.common_upscale(
            image.movedim(-1, 1),
            target_w,
            target_h,
            "bilinear",
            "center"
        ).movedim(1, -1)
        return scaled

    def _center_crop_image(self, image, current_w, current_h, target_w, target_h):
        if current_w == target_w and current_h == target_h:
            return image
        
        # 计算裁切区域（仅当当前尺寸大于目标尺寸时裁切）
        crop_left = (current_w - target_w) // 2 if current_w > target_w else 0
        crop_top = (current_h - target_h) // 2 if current_h > target_h else 0
        crop_right = crop_left + target_w
        crop_bottom = crop_top + target_h

        # 确保裁切区域在图像范围内
        crop_left = max(0, crop_left)
        crop_top = max(0, crop_top)
        crop_right = min(current_w, crop_right)
        crop_bottom = min(current_h, crop_bottom)

        # 执行裁切（ComfyUI图像格式：[batch, height, width, channels]）
        cropped = image[:, crop_top:crop_bottom, crop_left:crop_right, :]

        return cropped



class Image_pad_adjust_restore:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pad_image": ("IMAGE",),
                "stitch": ("STITCH4",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, }),
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
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 2000, "step": 1, }),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "mask_expansion": ("INT", {"default": 0, "min": -50, "max": 50, "step": 1}),
                "mask_color": (["image", "Alpha", "white", "black", "red", "green", "blue", "gray"], {"default": "white"}),
                "bg_color": (["image", "Alpha", "white", "black", "red", "green", "blue", "gray"], {"default": "Alpha"}),
            },
            "optional": {
                "brightness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, }),
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
    




class Image_precision_Converter:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_precision": (
                    ["8-bit (uint8)", "4-bit (uint4)"],
                    {"default": "8-bit (uint8)"}
                )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("low_precision",)
    FUNCTION = "convert_precision"
    CATEGORY = "Apt_Preset/image/ImgResize"

    def convert_precision(self, image, target_precision):
        input_dtype = image.dtype
        normalized_img = self._normalize_input(image, input_dtype)

        if target_precision == "8-bit (uint8)":
            low_prec_img = self._to_uint8(normalized_img)
        elif target_precision == "4-bit (uint4)":
            low_prec_img = self._to_uint4(normalized_img)
        else:
            raise ValueError(f"不支持的目标精度：{target_precision}")

        return (low_prec_img,)

    def _normalize_input(self, image, input_dtype):
        img = image.float()

        if input_dtype == torch.uint16:
            img = img / 65535.0
        elif input_dtype in (torch.float16, torch.half):
            pass
        elif input_dtype == torch.float32:
            pass
        else:
            raise TypeError(f"不支持的输入图像类型：{input_dtype}，仅支持 uint16/float16/float32")

        return torch.clamp(img, 0.0, 1.0)

    def _to_uint8(self, normalized_img):
        uint8_img = (normalized_img * 255).round().to(torch.uint8)
        return uint8_img.float() / 255.0

    def _to_uint4(self, normalized_img):
        quantized = (normalized_img * 15).round().to(torch.uint8)
        return quantized.float() / 15.0




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
                "long_side": ("INT", {"default": 512, "min": 16, "max": 9999, "step": 2}),
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


class Image_Resize_sum_restore:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resized_image": ("IMAGE",),
                "stitch": ("STITCH3",),
                "upscale_method":  (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "pad_crop_no_scale": ("BOOLEAN", {"default": False, }),
            },
        }

    CATEGORY = "Apt_Preset/image"
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE")
    RETURN_NAMES = ("restored_image", "restored_mask", "original_image")
    FUNCTION = "restore"

    DESCRIPTION = """
    - 新增参数：
    - pad_crop_no_scale：布尔值，True时无缩放直接裁切填充部分
      （计算输入图像与Image_Resize_sum输出图像的尺寸倍数，对填充部分做同倍数裁切）
    """

    def restore(self, resized_image, stitch, upscale_method="bicubic", pad_crop_no_scale=False):
        # 从stitch中提取关键信息
        original_h, original_w = stitch["original_shape"]
        pad_left, pad_right, pad_top, pad_bottom = stitch["pad_info"]
        keep_proportion = stitch["keep_proportion"]
        final_width, final_height = stitch["final_size"]  # Image_Resize_sum输出的有效区域尺寸
        original_mask = stitch.get("original_mask")
        has_input_mask = stitch.get("has_input_mask", False)
        crop_x, crop_y = stitch["crop_position"]
        crop_w, crop_h = stitch["crop_size"]
        original_resized_shape = stitch["resized_shape"]  # Image_Resize_sum输出的完整尺寸（含填充）
        
        # 获取原始图像
        original_image = stitch.get("original_image", None)
        
        # 获取当前输入图像的尺寸
        current_b, current_h, current_w, current_c = resized_image.shape

        # 计算尺寸倍数：当前输入图像尺寸 / Image_Resize_sum输出的完整尺寸
        scale_w = current_w / original_resized_shape[1] if original_resized_shape[1] != 0 else 1.0
        scale_h = current_h / original_resized_shape[0] if original_resized_shape[0] != 0 else 1.0

        if pad_crop_no_scale:
            # 无缩放直接裁切模式：按倍数计算实际填充区域并裁切
            if keep_proportion.startswith("pad"):
                # 计算当前图像中实际的填充区域（按尺寸倍数缩放）
                current_pad_left = int(round(pad_left * scale_w))
                current_pad_right = int(round(pad_right * scale_w))
                current_pad_top = int(round(pad_top * scale_h))
                current_pad_bottom = int(round(pad_bottom * scale_h))
                
                # 安全裁切：确保裁切范围在当前图像内
                crop_left = max(0, current_pad_left)
                crop_right = max(0, current_w - current_pad_right)
                crop_top = max(0, current_pad_top)
                crop_bottom = max(0, current_h - current_pad_bottom)
                
                # 裁切填充部分，得到有效区域（不缩放）
                valid_image = resized_image[:, crop_top:crop_bottom, crop_left:crop_right, :]
                
                # 直接使用裁切后的有效区域作为还原图像（保持原有尺寸，仅移除填充）
                restored_image = valid_image
            elif keep_proportion == "crop":
                # crop模式下：按原始裁剪比例裁切当前图像（不缩放）
                original_cropped_ratio = crop_w / crop_h if crop_h != 0 else 1.0
                current_ratio = current_w / current_h if current_h != 0 else 1.0
                
                if abs(current_ratio - original_cropped_ratio) > 1e-6:
                    if current_ratio > original_cropped_ratio:
                        # 当前图像更宽，按高度裁切宽度
                        target_w = int(round(current_h * original_cropped_ratio))
                        crop_left = (current_w - target_w) // 2
                        valid_image = resized_image[:, :, crop_left:crop_left+target_w, :]
                    else:
                        # 当前图像更高，按宽度裁切高度
                        target_h = int(round(current_w / original_cropped_ratio))
                        crop_top = (current_h - target_h) // 2
                        valid_image = resized_image[:, crop_top:crop_top+target_h, :, :]
                else:
                    valid_image = resized_image
                
                # 放回原始图像位置（不缩放）
                if stitch.get("original_image") is not None:
                    restored_image = stitch["original_image"].clone()
                else:
                    restored_image = torch.zeros(
                        (current_b, original_h, original_w, current_c),
                        dtype=resized_image.dtype,
                        device=resized_image.device
                    )
                
                # 调整有效图像尺寸以匹配原始裁剪区域（仅调整尺寸，不缩放内容）
                valid_image_resized = common_upscale(
                    valid_image.movedim(-1, 1),
                    crop_w, crop_h,
                    "nearest-exact",  # 直接调整尺寸，不插值
                    crop="disabled"
                ).movedim(1, -1)
                
                restored_image[:, crop_y:crop_y + crop_h, crop_x:crop_x + crop_w, :] = valid_image_resized
            else:
                # resize/stretch模式：直接返回当前图像（不缩放）
                restored_image = resized_image
        else:
            # 原有逻辑：带缩放的还原
            if keep_proportion.startswith("pad"):
                # 计算原始有效区域在填充后图像中的占比（用于处理比例变化）
                original_padded_w = final_width + pad_left + pad_right
                original_padded_h = final_height + pad_top + pad_bottom
                
                # 计算当前图像中有效区域的实际位置和尺寸（考虑比例变化）
                scale_w = current_w / original_padded_w if original_padded_w != 0 else 1.0
                scale_h = current_h / original_padded_h if original_padded_h != 0 else 1.0
                
                current_pad_left = int(round(pad_left * scale_w))
                current_pad_top = int(round(pad_top * scale_h))
                current_valid_w = int(round(final_width * scale_w))
                current_valid_h = int(round(final_height * scale_h))
                
                # 安全裁剪
                valid_left = max(0, current_pad_left)
                valid_right = min(current_w, current_pad_left + current_valid_w)
                valid_top = max(0, current_pad_top)
                valid_bottom = min(current_h, current_pad_top + current_valid_h)
                
                valid_image = resized_image[:, valid_top:valid_bottom, valid_left:valid_right, :]
                
                # 缩放至原始尺寸
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
                
                if abs(current_ratio - original_cropped_ratio) > 1e-6:
                    if current_ratio > original_cropped_ratio:
                        target_w = int(round(current_h * original_cropped_ratio))
                        crop_left = (current_w - target_w) // 2
                        crop_right = current_w - target_w - crop_left
                        valid_image = resized_image[:, :, crop_left:current_w - crop_right, :]
                    else:
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

        # 处理mask（无缩放模式下同步裁切mask）
        if pad_crop_no_scale and original_mask is not None and has_input_mask:
            # 对原始mask按相同倍数裁切填充区域
            mask_h, mask_w = original_mask.shape[1], original_mask.shape[2]
            mask_scale_w = mask_w / original_resized_shape[1] if original_resized_shape[1] != 0 else 1.0
            mask_scale_h = mask_h / original_resized_shape[0] if original_resized_shape[0] != 0 else 1.0
            
            mask_pad_left = int(round(pad_left * mask_scale_w))
            mask_pad_right = int(round(pad_right * mask_scale_w))
            mask_pad_top = int(round(pad_top * mask_scale_h))
            mask_pad_bottom = int(round(pad_bottom * mask_scale_h))
            
            mask_crop_left = max(0, mask_pad_left)
            mask_crop_right = max(0, mask_w - mask_pad_right)
            mask_crop_top = max(0, mask_pad_top)
            mask_crop_bottom = max(0, mask_h - mask_pad_bottom)
            
            restored_mask = original_mask[:, mask_crop_top:mask_crop_bottom, mask_crop_left:mask_crop_right]
        else:
            restored_mask = original_mask if (original_mask is not None and has_input_mask) else (
                torch.zeros((current_b, original_h, original_w), dtype=torch.float32, device=resized_image.device)
            )
       
        # 确保输出为PIL图像（保持原有逻辑）
        if isinstance(restored_image, torch.Tensor):
            restored_image = convert_pil_image(restored_image)
        
        # 处理原始图像输出
        if original_image is not None:
            output_original_image = original_image
        else:
            output_original_image = torch.zeros((1, original_h, original_w, 3), dtype=torch.float32)

        return (restored_image.cpu(), restored_mask.cpu(), output_original_image.cpu())





class Image_CnMapMix:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bg_color": (["image", "white", "black", "gray"], {"default": "image"}),
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
                "mask2_smoothness": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
                "invert_mask": ("BOOLEAN", {"default": False}),
                "image1_min_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image1_max_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image2_min_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image2_max_white": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAME = ("image",)
    FUNCTION = "fuse_depth"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def map_black_white_range(self, image, min_black, max_white):
        if min_black >= max_white:
            mid = (min_black + max_white) / 2
            return torch.where(image < mid, torch.tensor(0.0, device=image.device), torch.tensor(1.0, device=image.device))
        
        mapped = (image - min_black) / (max_white - min_black)
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

    def create_solid_color_image(self, color, shape, device):
        color_map = {
            "white": 1.0,
            "black": 0.0,
            "gray": 0.5
        }
        value = color_map[color]
        return torch.full(shape, value, device=device, dtype=torch.float32)

    def fuse_depth(self, bg_color, blur_1, blur_2, diff_blur, blend_mode, 
                  blend_factor, contrast, brightness, mask2_smoothness,
                  diff_sensitivity, invert_mask,
                  image1_min_black, image1_max_white, image2_min_black, image2_max_white,
                  image1=None, image2=None, mask2=None):
        
        # 处理纯色替换逻辑
        if bg_color != "image":
            if image1 is not None:
                target_shape = image1.shape
            elif image2 is not None:
                target_shape = image2.shape
            else:
                raise ValueError("至少需要提供image1或image2中的一个以确定纯色块尺寸")
            
            solid_color = self.create_solid_color_image(bg_color, target_shape, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            image1 = solid_color

        # 处理image1和image2的默认逻辑
        if image1 is None and image2 is not None:
            image1 = image2.clone()
        elif image2 is None and image1 is not None:
            image2 = image1.clone()
        elif image1 is None and image2 is None:
            raise ValueError("至少需要提供image1或image2中的一个")
        
        # 尺寸对齐逻辑
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

        # 处理mask2
        if mask2 is None:
            mask2 = torch.ones_like(image1[..., :1], device=image1.device)
        else:
            mask2 = self.process_mask(mask2, image1.shape, image2.device)
        
        # 遮罩反转
        if invert_mask:
            mask2 = 1.0 - mask2
        
        # 平滑处理
        mask2 = self.smoothness_mask(mask2, mask2_smoothness)

        # 图像转灰度
        if image1.shape[-1] == 3:
            image1 = (image1 * torch.tensor([0.299, 0.587, 0.114], device=image1.device)).sum(dim=-1, keepdim=True)
        else:
            image1 = image1[:, :, :, 0:1]

        if image2.shape[-1] == 3:
            image2 = (image2 * torch.tensor([0.299, 0.587, 0.114], device=image2.device)).sum(dim=-1, keepdim=True)
        else:
            image2 = image2[:, :, :, 0:1]

        # 执行黑-白重定向
        image1 = self.map_black_white_range(image1, image1_min_black, image1_max_white)
        image2 = self.map_black_white_range(image2, image2_min_black, image2_max_white)

        # 设备对齐
        image1 = image1.to(image2.device)
        mask2 = mask2.to(image2.device)
        
        # 模糊处理
        blurred_a = self.gaussian_blur(image1, blur_1)
        blurred_b = self.gaussian_blur(image2, blur_2)

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


class Image_CnMap_Resize:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
            "get_resize_image": ("IMAGE",),  
            "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask",)
    FUNCTION = "resize_and_composite"
    CATEGORY = "Apt_Preset/image/ImgResize"


    DESCRIPTION = """
    get_resize_image有输入时，则按照get_resize_image尺寸输出
    否则使用image尺寸输出
    """


    def resize_and_composite(self, image, mask=None, get_resize_image=None):
        # 处理输入图像维度
        if len(image.shape) == 3:
            B, H, W, C = 1, image.shape[0], image.shape[1], image.shape[2]
            input_image = image.unsqueeze(0)
        else:
            B, H, W, C = image.shape
            input_image = image.clone()
        
        # 确定目标尺寸：优先使用 get_resize_image 的尺寸，否则使用原图像尺寸
        if get_resize_image is not None:
            # 处理参考图像的维度（支持单张图像和批量图像）
            if len(get_resize_image.shape) == 3:
                target_height, target_width = get_resize_image.shape[0], get_resize_image.shape[1]
            else:
                # 批量图像取第一张的尺寸
                target_height, target_width = get_resize_image.shape[1], get_resize_image.shape[2]
        else:
            target_height, target_width = H, W
        
        upscale_method = "area"
        bg_color = "image"

        out_mask = None
        if mask is not None:
            if upscale_method == "lanczos":
                out_mask = common_upscale(
                    mask.unsqueeze(1).repeat(1, 3, 1, 1),
                    target_width,
                    target_height,
                    upscale_method,
                    crop="disabled"
                ).movedim(1, -1)[:, :, :, 0]
            else:
                out_mask = common_upscale(
                    mask.unsqueeze(1),
                    target_width,
                    target_height,
                    upscale_method,
                    crop="disabled"
                ).squeeze(1)
        else:
            # 当没有输入mask时，创建与目标尺寸匹配的全黑mask
            out_mask = torch.zeros((B, target_height, target_width), dtype=torch.float32, device=input_image.device)

        color_map = {
            "white": (1.0, 1.0, 1.0),
            "black": (0.0, 0.0, 0.0),
            "gray": (0.5, 0.5, 0.5),
        }
        
        # 生成背景图像（使用目标尺寸）
        if bg_color == "image":
            background = common_upscale(
                input_image.movedim(-1, 1),
                target_width,
                target_height,
                upscale_method,
                crop="disabled"
            ).movedim(1, -1)
        else:
            bg_rgb = color_map[bg_color]
            background = torch.tensor(
                bg_rgb, dtype=input_image.dtype, device=input_image.device
            ).unsqueeze(0).unsqueeze(0).unsqueeze(0)
            background = background.repeat(B, target_height, target_width, 1)

        # 图像合成
        mask_unsqueezed = out_mask.unsqueeze(-1)
        composite_image = background * (1.0 - mask_unsqueezed)

        # 后处理：确保数值范围在 [0,1] 并转移到CPU
        composite_image = composite_image.cpu().clamp(0.0, 1.0)
        out_mask = out_mask.cpu()

        return (composite_image, out_mask,)



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
                "extract_to_block": ("BOOLEAN", {"default": False}),  # 矩形化开关（与参考代码一致）
                "block_size": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
            },
            "optional": {}
        }
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("processed_mask",)
    FUNCTION = "process_mask"
    CATEGORY = "Apt_Preset/mask"
    
    def process_mask(self, smoothness=0, mask_expand=0, is_fill=False, is_invert=False, 
                    input_mask=None, mask_min=0.0, mask_max=1.0, extract_to_block=True,block_size=0):
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
        
        # 完全复用参考代码的轮廓检测逻辑：提取外部轮廓，简化轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 1]
        
        final_mask = np.zeros_like(binary_mask)
        expand_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        
        for contour in valid_contours:
            temp_mask = np.zeros_like(binary_mask)
            
            # 核心修改：完全复用参考代码的矩形化逻辑
            if extract_to_block:
                # 1. 计算轮廓的最小外接矩形（与参考代码一致）
                x, y, w, h = cv2.boundingRect(contour)
                # 2. 绘制填充矩形（与参考代码的 cv2.rectangle + FILLED 逻辑一致）
                cv2.rectangle(temp_mask, (x, y), (x + w, y + h), 255, thickness=cv2.FILLED)
                
                # 3. 非填充模式兼容：如果 is_fill=False，矩形内保留原mask灰度（保持原逻辑）
                if not is_fill:
                    # 提取原mask的矩形区域灰度信息
                    roi = opencv_gray_mask[y:y+h, x:x+w]
                    # 用原灰度信息替换矩形内的纯白填充
                    temp_mask[y:y+h, x:x+w] = roi
            else:
                # 保持原逻辑：使用原始轮廓
                if is_fill:
                    cv2.drawContours(temp_mask, [contour], 0, 255, thickness=cv2.FILLED)
                else:
                    cv2.drawContours(temp_mask, [contour], 0, 255, -1)
                    temp_mask = cv2.bitwise_and(opencv_gray_mask, temp_mask)
            
            # 扩展/收缩处理（保持原逻辑不变）
            if mask_expand != 0:
                expand_iter = abs(int(mask_expand))
                if mask_expand > 0:
                    temp_mask = cv2.dilate(temp_mask, expand_kernel, iterations=expand_iter)
                else:
                    temp_mask = cv2.erode(temp_mask, expand_kernel, iterations=expand_iter)
            
            final_mask = cv2.bitwise_or(final_mask, temp_mask)
        
        # 平滑处理（保持原逻辑不变）
        if smoothness > 0:
            mask_pil = Image.fromarray(final_mask)
            mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            final_mask = np.array(mask_pil)
        
        # 反转处理（保持原逻辑不变）
        if is_invert:
            final_mask = cv2.bitwise_not(final_mask)
            _, final_mask = cv2.threshold(final_mask, 127, 255, cv2.THRESH_BINARY)
        
        processed_mask_tensor = cv2img2tensorMask(final_mask)

        if block_size > 0:  
            processed_mask_tensor = BlockifyMask(processed_mask_tensor, block_size)

        return (processed_mask_tensor,)







class Image_merge2image:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image1_position": (["up", "down", "left", "right", "center"],),
                "blendsmooth": ("INT", {"default": 100, "min": 1, "max": 4000, "step": 1}),
            },
            "optional": {
                "mask2": ("MASK",),
                "x_compress": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "y_compress": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("blended_image",)
    FUNCTION = "blend_images"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def blend_images(self, image1, image2, image1_position, blendsmooth, mask2=None, x_compress=0.0, y_compress=0.0):
        img1 = image1[0].cpu().numpy()
        img2 = image2[0].cpu().numpy()
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        img2_resized = None
        new_h2, new_w2 = 0, 0
        mask2_resized = None

        if image1_position == "center":
            target_h, target_w = h1, w1
            img2_resized = np.zeros((target_h, target_w, 3), dtype=np.float32)
            mask2_resized = np.zeros((target_h, target_w), dtype=np.float32)
            img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
            img2_scaled = img2_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
            img2_scaled = np.array(img2_scaled) / 255.0
            img2_resized[:, :, :] = img2_scaled

            if mask2 is not None:
                mask2_np = mask2[0].cpu().numpy()
                mask2_pil = Image.fromarray((mask2_np * 255).astype(np.uint8))
                mask2_scaled = mask2_pil.resize((target_w, target_h), Image.Resampling.LANCZOS)
                mask2_scaled = np.array(mask2_scaled) / 255.0
                if len(mask2_scaled.shape) == 3:
                    mask2_scaled = mask2_scaled[..., 0]
                mask2_resized[:, :] = mask2_scaled
            else:
                mask2_resized = np.ones((target_h, target_w), dtype=np.float32)

        elif image1_position in ["left", "right"]:
            target_h = h1
            scale = target_h / h2
            new_w2 = int(w2 * scale)
            new_h2 = target_h

            img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
            img2_resized = img2_pil.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
            img2_resized = np.array(img2_resized) / 255.0

            if mask2 is not None:
                mask2_np = mask2[0].cpu().numpy()
                mask2_pil = Image.fromarray((mask2_np * 255).astype(np.uint8))
                mask2_resized = mask2_pil.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
                mask2_resized = np.array(mask2_resized) / 255.0
                if len(mask2_resized.shape) == 3:
                    mask2_resized = mask2_resized[..., 0]
            else:
                mask2_resized = np.ones((new_h2, new_w2), dtype=np.float32)

        else:
            target_w = w1
            scale = target_w / w2
            new_h2 = int(h2 * scale)
            new_w2 = target_w

            img2_pil = Image.fromarray((img2 * 255).astype(np.uint8))
            img2_resized = img2_pil.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
            img2_resized = np.array(img2_resized) / 255.0

            if mask2 is not None:
                mask2_np = mask2[0].cpu().numpy()
                mask2_pil = Image.fromarray((mask2_np * 255).astype(np.uint8))
                mask2_resized = mask2_pil.resize((new_w2, new_h2), Image.Resampling.LANCZOS)
                mask2_resized = np.array(mask2_resized) / 255.0
                if len(mask2_resized.shape) == 3:
                    mask2_resized = mask2_resized[..., 0]
            else:
                mask2_resized = np.ones((new_h2, new_w2), dtype=np.float32)

        max_feather = min(blendsmooth, new_h2 // 2, new_w2 // 2) if image1_position != "center" else min(blendsmooth, h1 // 2, w1 // 2)
        feather_range = max_feather if max_feather > 0 else 1

        base_mask = np.ones_like(mask2_resized)
        if image1_position == "center":
            binary_mask = (mask2_resized > 0.5).astype(np.uint8)
            if np.any(binary_mask):
                try:
                    dist_in = distance_transform_edt(binary_mask)
                    base_mask = np.clip(dist_in / feather_range, 0.0, 1.0)
                except:
                    dilated = np.array(Image.fromarray(binary_mask * 255).filter(ImageFilter.MaxFilter(feather_range))) / 255.0
                    eroded = np.array(Image.fromarray(binary_mask * 255).filter(ImageFilter.MinFilter(feather_range))) / 255.0
                    feather_zone = dilated - eroded
                    base_mask = eroded + feather_zone * np.linspace(0, 1, feather_range)[-1]
            else:
                base_mask = np.zeros_like(mask2_resized)
        else:
            feather_region = np.linspace(0, 1, feather_range, dtype=np.float32)
            if image1_position == "left":
                base_mask[:, :feather_range] = feather_region[np.newaxis, :]
            elif image1_position == "right":
                base_mask[:, -feather_range:] = feather_region[::-1][np.newaxis, :]
            elif image1_position == "up":
                base_mask[:feather_range, :] = feather_region[:, np.newaxis]
            elif image1_position == "down":
                base_mask[-feather_range:, :] = feather_region[::-1][:, np.newaxis]

        final_mask = base_mask * mask2_resized
        final_mask = final_mask[:, :, np.newaxis]
        final_mask = final_mask[:img2_resized.shape[0], :img2_resized.shape[1], :]

        if image1_position == "center":
            total_height, total_width = h1, w1
            blended = np.copy(img1)
        elif image1_position in ["left", "right"]:
            original_overlap = min(feather_range, w1, new_w2)
            max_possible_overlap = min(w1, new_w2)
            compressed_overlap = original_overlap + (max_possible_overlap - original_overlap) * x_compress
            compressed_overlap = int(compressed_overlap)
            compressed_overlap = max(0, min(compressed_overlap, max_possible_overlap))
            
            total_width = w1 + new_w2 - compressed_overlap
            total_height = h1
            blended = np.zeros((total_height, total_width, 3), dtype=np.float32)
        else:
            original_overlap = min(feather_range, h1, new_h2)
            max_possible_overlap = min(h1, new_h2)
            compressed_overlap = original_overlap + (max_possible_overlap - original_overlap) * y_compress
            compressed_overlap = int(compressed_overlap)
            compressed_overlap = max(0, min(compressed_overlap, max_possible_overlap))
            
            total_height = h1 + new_h2 - compressed_overlap
            total_width = w1
            blended = np.zeros((total_height, total_width, 3), dtype=np.float32)

        if image1_position == "left":
            blended[:, :w1, :] = img1
        elif image1_position == "right":
            blended[:, -w1:, :] = img1
        elif image1_position == "up":
            blended[:h1, :, :] = img1
        elif image1_position == "down":
            blended[-h1:, :, :] = img1

        if image1_position == "center":
            blended = img1 * (1 - final_mask) + img2_resized * final_mask
        elif image1_position == "left":
            start_x = w1 - compressed_overlap
            end_x = start_x + new_w2
            start_x = max(0, start_x)
            end_x = min(total_width, end_x)
            
            background = blended[:, start_x:end_x, :]
            img2_crop = img2_resized[:, :background.shape[1], :] if background.shape[1] < new_w2 else img2_resized
            mask_crop = final_mask[:, :background.shape[1], :] if background.shape[1] < new_w2 else final_mask
            
            blended[:, start_x:end_x, :] = background * (1 - mask_crop) + img2_crop * mask_crop
        elif image1_position == "right":
            start_x = total_width - w1 - new_w2 + compressed_overlap
            start_x = max(0, start_x)
            end_x = start_x + new_w2
            end_x = min(total_width, end_x)
            
            background = blended[:, start_x:end_x, :]
            img2_crop = img2_resized[:, :background.shape[1], :] if background.shape[1] < new_w2 else img2_resized
            mask_crop = final_mask[:, :background.shape[1], :] if background.shape[1] < new_w2 else final_mask
            
            blended[:, start_x:end_x, :] = background * (1 - mask_crop) + img2_crop * mask_crop
        elif image1_position == "up":
            start_y = h1 - compressed_overlap
            end_y = start_y + new_h2
            start_y = max(0, start_y)
            end_y = min(total_height, end_y)
            
            background = blended[start_y:end_y, :, :]
            img2_crop = img2_resized[:background.shape[0], :, :] if background.shape[0] < new_h2 else img2_resized
            mask_crop = final_mask[:background.shape[0], :, :] if background.shape[0] < new_h2 else final_mask
            
            blended[start_y:end_y, :, :] = background * (1 - mask_crop) + img2_crop * mask_crop
        elif image1_position == "down":
            start_y = total_height - h1 - new_h2 + compressed_overlap
            start_y = max(0, start_y)
            end_y = start_y + new_h2
            end_y = min(total_height, end_y)
            
            background = blended[start_y:end_y, :, :]
            img2_crop = img2_resized[:background.shape[0], :, :] if background.shape[0] < new_h2 else img2_resized
            mask_crop = final_mask[:background.shape[0], :, :] if background.shape[0] < new_h2 else final_mask
            
            blended[start_y:end_y, :, :] = background * (1 - mask_crop) + img2_crop * mask_crop

        blended_tensor = torch.from_numpy(blended).unsqueeze(0)
        blended_tensor = torch.clamp(blended_tensor, 0.0, 1.0)

        return (blended_tensor,)


class color_RadiaBrightGradient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "circle_radius": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_bright": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 5.0, "step": 0.01}),
                "edge_bright": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "overlay_color": ("STRING", {"default": "#FFFFFF"}),
                "center_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "falloff_mode": (["linear", "ease_out", "ease_in"], {"default": "linear"}),
                "soft_edge": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("radial_gradient_image",)
    FUNCTION = "apply_radial_gradient"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (255, 255, 255)
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        except:
            return (255, 255, 255)

    def apply_falloff_curve(self, norm_falloff, mode):
        if mode == "ease_out":
            return 1 - (1 - norm_falloff) ** 2
        elif mode == "ease_in":
            return norm_falloff ** 2
        else:
            return norm_falloff

    def apply_radial_gradient(self, image, center_x, center_y, circle_radius,
                               center_bright, edge_bright, overlay_color,
                               center_alpha, edge_alpha, falloff_mode, soft_edge):
        batch_size, h, w, c = image.shape
        device = image.device
        max_side = max(w, h)

        cx = center_x * (w - 1)
        cy = center_y * (h - 1)
        radius_px = circle_radius * max_side

        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )
        coords = torch.stack([x_coords, y_coords], dim=-1).float()

        distance = torch.sqrt((coords[..., 0] - cx) ** 2 + (coords[..., 1] - cy) ** 2)

        if soft_edge:
            soft_edge_px = max(1, int(radius_px * 0.05))
            in_circle_soft = torch.clamp((radius_px + soft_edge_px - distance) / (2 * soft_edge_px), 0.0, 1.0)
        else:
            in_circle_soft = (distance <= radius_px).float()

        falloff_distance = distance - radius_px
        max_falloff_distance = max_side - radius_px
        max_falloff_distance = max(1e-6, max_falloff_distance)

        norm_falloff = torch.clamp(falloff_distance / max_falloff_distance, 0.0, 1.0)
        norm_falloff = self.apply_falloff_curve(norm_falloff, falloff_mode)

        bright_overlay = center_bright * (1 - norm_falloff) + edge_bright * norm_falloff
        brightness_map = center_bright * in_circle_soft + bright_overlay * (1 - in_circle_soft)
        brightness_map = brightness_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, h, w, c)

        result = image * brightness_map

        rgb_255 = self.hex_to_rgb(overlay_color)
        rgb_norm = torch.tensor(rgb_255, device=device, dtype=torch.float32) / 255.0
        color_layer = rgb_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, h, w, 3)

        alpha_overlay = center_alpha * (1 - norm_falloff) + edge_alpha * norm_falloff
        alpha_map = center_alpha * in_circle_soft + alpha_overlay * (1 - in_circle_soft)
        alpha_map = alpha_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, h, w, 1)

        result_rgb = result[..., :3]
        result_rgb = result_rgb * (1 - alpha_map) + color_layer * alpha_map

        if c >= 4:
            result_alpha = result[..., 3:4]
            result = torch.cat([result_rgb, result_alpha], dim=-1)
        else:
            result = result_rgb

        result = torch.clamp(result, 0.0, 1.0)

        return (result,)




class color_brightGradient:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "start_x": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_y": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "start_bright": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "end_x": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_y": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_bright": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01}),
                "overlay_color": ("STRING", {"default": "#FFFFFF", "description": "Hex color code (e.g. #FFFFFF)"}),
                "start_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_alpha": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("gradient_image",)
    FUNCTION = "apply_gradient"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def hex_to_rgb(self, hex_color):
        hex_color = hex_color.lstrip('#')
        if len(hex_color) != 6:
            return (255, 255, 255)
        try:
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            return (r, g, b)
        except:
            return (255, 255, 255)

    def apply_gradient(self, image, start_x, start_y, start_bright, end_x, end_y, end_bright,
                       overlay_color, start_alpha, end_alpha):
        batch_size, h, w, c = image.shape
        device = image.device
        
        start_x_px = int(start_x * (w - 1))
        start_y_px = int(start_y * (h - 1))
        end_x_px = int(end_x * (w - 1))
        end_y_px = int(end_y * (h - 1))
        
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing="ij"
        )
        coords = torch.stack([x_coords, y_coords], dim=-1).float()
        
        brightness_map = self._compute_gradient_map(
            coords=coords,
            p0=(start_x_px, start_y_px),
            p1=(end_x_px, end_y_px),
            val0=start_bright,
            val1=end_bright,
            device=device
        )
        
        alpha_overlay_map = self._compute_gradient_map(
            coords=coords,
            p0=(start_x_px, start_y_px),
            p1=(end_x_px, end_y_px),
            val0=start_alpha,
            val1=end_alpha,
            device=device
        )
        
        brightness_map = brightness_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, h, w, c)
        result = image * brightness_map
        
        rgb_255 = self.hex_to_rgb(overlay_color)
        rgb_norm = torch.tensor(rgb_255, device=device, dtype=torch.float32) / 255.0
        color_layer = rgb_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, h, w, 3)
        
        alpha_overlay_map = alpha_overlay_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, h, w, 1)
        
        result_rgb = result[..., :3]
        result_rgb = result_rgb * (1 - alpha_overlay_map) + color_layer * alpha_overlay_map
        
        if c >= 4:
            result_alpha = result[..., 3:4]
            result = torch.cat([result_rgb, result_alpha], dim=-1)
        else:
            result = result_rgb
        
        result = torch.clamp(result, 0.0, 1.0)
        
        return (result,)

    def _compute_gradient_map(self, coords, p0, p1, val0, val1, device):
        p0 = torch.tensor(p0, device=device, dtype=torch.float32)
        p1 = torch.tensor(p1, device=device, dtype=torch.float32)
        
        dir_vec = p1 - p0
        dir_len = torch.norm(dir_vec)
        
        if dir_len < 1e-6:
            return torch.full((coords.shape[0], coords.shape[1]), val0, device=device, dtype=torch.float32)
        
        coords_centered = coords - p0
        coords_flat = coords_centered.reshape(-1, 2)
        
        proj = torch.matmul(coords_flat, dir_vec.unsqueeze(-1)).squeeze(-1) / dir_len
        proj = proj.reshape(coords.shape[0], coords.shape[1])
        
        proj_norm = torch.clamp(proj / dir_len, 0.0, 1.0)
        gradient_map = val0 * (1 - proj_norm) + val1 * proj_norm
        
        return gradient_map









import json


BLEND_METHODS = [
    "正常",
    "溶解",
    "变暗",
    "正片叠底",
    "颜色加深",
    "线性加深",
    "深色",
    "变亮",
    "滤色",
    "颜色减淡",
    "线性减淡（添加）",
    "浅色",
    "叠加",
    "柔光",
    "强光",
    "亮光",
    "线性光",
    "点光",
    "实色混合",
    "差值",
    "排除",
    "减去",
    "划分",
    "色相",
    "饱和度",
    "颜色",
    "明度"
]

def rgb_to_hsv(r, g, b):
    r = r / 255.0
    g = g / 255.0
    b = b / 255.0
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    delta = max_val - min_val
    
    h = 0
    if delta > 0:
        if max_val == r:
            h = ((g - b) / delta) % 6
        elif max_val == g:
            h = ((b - r) / delta) + 2
        else:
            h = ((r - g) / delta) + 4
        h *= 60
        if h < 0:
            h += 360
    
    s = 0 if max_val == 0 else delta / max_val
    v = max_val
    
    return h, s, v

def hsv_to_rgb(h, s, v):
    h = h % 360
    h /= 60
    s = max(0, min(1, s))
    v = max(0, min(1, v))
    
    i = int(h)
    f = h - i
    p = v * (1 - s)
    q = v * (1 - s * f)
    t = v * (1 - s * (1 - f))
    
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    
    return int(r * 255), int(g * 255), int(b * 255)

def apply_blending_mode(bg_img, fg_img, mode, strength=1.0):
    bg_pixel = bg_img.getpixel((0, 0))
    fg_pixel = fg_img.getpixel((0, 0))
    
    bg_r, bg_g, bg_b, bg_a = bg_pixel
    fg_r, fg_g, fg_b, fg_a = fg_pixel
    
    fg_a = fg_a / 255.0 * strength
    bg_a = bg_a / 255.0
    
    r, g, b = bg_r, bg_g, bg_b
    
    if mode == "正常":
        r = int((fg_r * fg_a) + (bg_r * bg_a * (1 - fg_a)))
        g = int((fg_g * fg_a) + (bg_g * bg_a * (1 - fg_a)))
        b = int((fg_b * fg_a) + (bg_b * bg_a * (1 - fg_a)))
    
    elif mode == "溶解":
        r = int((fg_r * fg_a) + (bg_r * bg_a * (1 - fg_a)))
        g = int((fg_g * fg_a) + (bg_g * bg_a * (1 - fg_a)))
        b = int((fg_b * fg_a) + (bg_b * bg_a * (1 - fg_a)))
    
    elif mode == "变暗":
        r = int(min(bg_r, fg_r) * fg_a + bg_r * (1 - fg_a))
        g = int(min(bg_g, fg_g) * fg_a + bg_g * (1 - fg_a))
        b = int(min(bg_b, fg_b) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "正片叠底":
        r = int((bg_r * fg_r / 255) * fg_a + bg_r * (1 - fg_a))
        g = int((bg_g * fg_g / 255) * fg_a + bg_g * (1 - fg_a))
        b = int((bg_b * fg_b / 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "颜色加深":
        r = int((255 - (255 - bg_r) * 255 / max(fg_r, 1)) * fg_a + bg_r * (1 - fg_a))
        g = int((255 - (255 - bg_g) * 255 / max(fg_g, 1)) * fg_a + bg_g * (1 - fg_a))
        b = int((255 - (255 - bg_b) * 255 / max(fg_b, 1)) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "线性加深":
        r = int(max(0, bg_r + fg_r - 255) * fg_a + bg_r * (1 - fg_a))
        g = int(max(0, bg_g + fg_g - 255) * fg_a + bg_g * (1 - fg_a))
        b = int(max(0, bg_b + fg_b - 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "深色":
        bg_sum = bg_r + bg_g + bg_b
        fg_sum = fg_r + fg_g + fg_b
        if fg_sum < bg_sum:
            r = int(fg_r * fg_a + bg_r * (1 - fg_a))
            g = int(fg_g * fg_a + bg_g * (1 - fg_a))
            b = int(fg_b * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "变亮":
        r = int(max(bg_r, fg_r) * fg_a + bg_r * (1 - fg_a))
        g = int(max(bg_g, fg_g) * fg_a + bg_g * (1 - fg_a))
        b = int(max(bg_b, fg_b) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "滤色":
        r = int((255 - (255 - bg_r) * (255 - fg_r) / 255) * fg_a + bg_r * (1 - fg_a))
        g = int((255 - (255 - bg_g) * (255 - fg_g) / 255) * fg_a + bg_g * (1 - fg_a))
        b = int((255 - (255 - bg_b) * (255 - fg_b) / 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "颜色减淡":
        r = int((bg_r / max(255 - fg_r, 1) * 255) * fg_a + bg_r * (1 - fg_a))
        g = int((bg_g / max(255 - fg_g, 1) * 255) * fg_a + bg_g * (1 - fg_a))
        b = int((bg_b / max(255 - fg_b, 1) * 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "线性减淡（添加）":
        r = int(min(255, bg_r + fg_r) * fg_a + bg_r * (1 - fg_a))
        g = int(min(255, bg_g + fg_g) * fg_a + bg_g * (1 - fg_a))
        b = int(min(255, bg_b + fg_b) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "浅色":
        bg_sum = bg_r + bg_g + bg_b
        fg_sum = fg_r + fg_g + fg_b
        if fg_sum > bg_sum:
            r = int(fg_r * fg_a + bg_r * (1 - fg_a))
            g = int(fg_g * fg_a + bg_g * (1 - fg_a))
            b = int(fg_b * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "叠加":
        if bg_r < 128:
            r = int((2 * bg_r * fg_r / 255) * fg_a + bg_r * (1 - fg_a))
        else:
            r = int((255 - 2 * (255 - bg_r) * (255 - fg_r) / 255) * fg_a + bg_r * (1 - fg_a))
        
        if bg_g < 128:
            g = int((2 * bg_g * fg_g / 255) * fg_a + bg_g * (1 - fg_a))
        else:
            g = int((255 - 2 * (255 - bg_g) * (255 - fg_g) / 255) * fg_a + bg_g * (1 - fg_a))
        
        if bg_b < 128:
            b = int((2 * bg_b * fg_b / 255) * fg_a + bg_b * (1 - fg_a))
        else:
            b = int((255 - 2 * (255 - bg_b) * (255 - fg_b) / 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "柔光":
        if fg_r < 128:
            r = int((2 * (bg_r / 255) * (fg_r / 255) * 255) * fg_a + bg_r * (1 - fg_a))
        else:
            r = int((255 - 2 * (1 - bg_r / 255) * (1 - fg_r / 255) * 255) * fg_a + bg_r * (1 - fg_a))
        
        if fg_g < 128:
            g = int((2 * (bg_g / 255) * (fg_g / 255) * 255) * fg_a + bg_g * (1 - fg_a))
        else:
            g = int((255 - 2 * (1 - bg_g / 255) * (1 - fg_g / 255) * 255) * fg_a + bg_g * (1 - fg_a))
        
        if fg_b < 128:
            b = int((2 * (bg_b / 255) * (fg_b / 255) * 255) * fg_a + bg_b * (1 - fg_a))
        else:
            b = int((255 - 2 * (1 - bg_b / 255) * (1 - fg_b / 255) * 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "强光":
        if fg_r < 128:
            r = int((2 * bg_r * fg_r / 255) * fg_a + bg_r * (1 - fg_a))
        else:
            r = int((255 - 2 * (255 - bg_r) * (255 - fg_r) / 255) * fg_a + bg_r * (1 - fg_a))
        
        if fg_g < 128:
            g = int((2 * bg_g * fg_g / 255) * fg_a + bg_g * (1 - fg_a))
        else:
            g = int((255 - 2 * (255 - bg_g) * (255 - fg_g) / 255) * fg_a + bg_g * (1 - fg_a))
        
        if fg_b < 128:
            b = int((2 * bg_b * fg_b / 255) * fg_a + bg_b * (1 - fg_a))
        else:
            b = int((255 - 2 * (255 - bg_b) * (255 - fg_b) / 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "亮光":
        if fg_r < 128:
            r = int((255 - (255 - bg_r) * 255 / (2 * fg_r)) * fg_a + bg_r * (1 - fg_a))
        else:
            r = int((bg_r * 255 / (2 * (255 - fg_r))) * fg_a + bg_r * (1 - fg_a))
        
        if fg_g < 128:
            g = int((255 - (255 - bg_g) * 255 / (2 * fg_g)) * fg_a + bg_g * (1 - fg_a))
        else:
            g = int((bg_g * 255 / (2 * (255 - fg_g))) * fg_a + bg_g * (1 - fg_a))
        
        if fg_b < 128:
            b = int((255 - (255 - bg_b) * 255 / (2 * fg_b)) * fg_a + bg_b * (1 - fg_a))
        else:
            b = int((bg_b * 255 / (2 * (255 - fg_b))) * fg_a + bg_b * (1 - fg_a))
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
    
    elif mode == "线性光":
        r = int(max(0, min(255, bg_r + 2 * fg_r - 255)) * fg_a + bg_r * (1 - fg_a))
        g = int(max(0, min(255, bg_g + 2 * fg_g - 255)) * fg_a + bg_g * (1 - fg_a))
        b = int(max(0, min(255, bg_b + 2 * fg_b - 255)) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "点光":
        if fg_r < 128:
            r = int(max(bg_r, 2 * fg_r - 255) * fg_a + bg_r * (1 - fg_a))
        else:
            r = int(min(bg_r, 2 * fg_r) * fg_a + bg_r * (1 - fg_a))
        
        if fg_g < 128:
            g = int(max(bg_g, 2 * fg_g - 255) * fg_a + bg_g * (1 - fg_a))
        else:
            g = int(min(bg_g, 2 * fg_g) * fg_a + bg_g * (1 - fg_a))
        
        if fg_b < 128:
            b = int(max(bg_b, 2 * fg_b - 255) * fg_a + bg_b * (1 - fg_a))
        else:
            b = int(min(bg_b, 2 * fg_b) * fg_a + bg_b * (1 - fg_a))
        
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
    
    elif mode == "实色混合":
        r = 255 if (bg_r + fg_r) > 255 else 0
        g = 255 if (bg_g + fg_g) > 255 else 0
        b = 255 if (bg_b + fg_b) > 255 else 0
        r = int(r * fg_a + bg_r * (1 - fg_a))
        g = int(g * fg_a + bg_g * (1 - fg_a))
        b = int(b * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "差值":
        r = int(abs(bg_r - fg_r) * fg_a + bg_r * (1 - fg_a))
        g = int(abs(bg_g - fg_g) * fg_a + bg_g * (1 - fg_a))
        b = int(abs(bg_b - fg_b) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "排除":
        r = int((bg_r + fg_r - 2 * bg_r * fg_r / 255) * fg_a + bg_r * (1 - fg_a))
        g = int((bg_g + fg_g - 2 * bg_g * fg_g / 255) * fg_a + bg_g * (1 - fg_a))
        b = int((bg_b + fg_b - 2 * bg_b * fg_b / 255) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "减去":
        r = int(max(0, bg_r - fg_r) * fg_a + bg_r * (1 - fg_a))
        g = int(max(0, bg_g - fg_g) * fg_a + bg_g * (1 - fg_a))
        b = int(max(0, bg_b - fg_b) * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "划分":
        r = int((bg_r / max(fg_r, 1) * 255) * fg_a + bg_r * (1 - fg_a))
        g = int((bg_g / max(fg_g, 1) * 255) * fg_a + bg_g * (1 - fg_a))
        b = int((bg_b / max(fg_b, 1) * 255) * fg_a + bg_b * (1 - fg_a))
        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
    
    elif mode == "色相":
        bg_h, bg_s, bg_v = rgb_to_hsv(bg_r, bg_g, bg_b)
        fg_h, fg_s, fg_v = rgb_to_hsv(fg_r, fg_g, fg_b)
        new_r, new_g, new_b = hsv_to_rgb(fg_h, bg_s, bg_v)
        r = int(new_r * fg_a + bg_r * (1 - fg_a))
        g = int(new_g * fg_a + bg_g * (1 - fg_a))
        b = int(new_b * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "饱和度":
        bg_h, bg_s, bg_v = rgb_to_hsv(bg_r, bg_g, bg_b)
        fg_h, fg_s, fg_v = rgb_to_hsv(fg_r, fg_g, fg_b)
        new_r, new_g, new_b = hsv_to_rgb(bg_h, fg_s, bg_v)
        r = int(new_r * fg_a + bg_r * (1 - fg_a))
        g = int(new_g * fg_a + bg_g * (1 - fg_a))
        b = int(new_b * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "颜色":
        bg_h, bg_s, bg_v = rgb_to_hsv(bg_r, bg_g, bg_b)
        fg_h, fg_s, fg_v = rgb_to_hsv(fg_r, fg_g, fg_b)
        new_r, new_g, new_b = hsv_to_rgb(fg_h, fg_s, bg_v)
        r = int(new_r * fg_a + bg_r * (1 - fg_a))
        g = int(new_g * fg_a + bg_g * (1 - fg_a))
        b = int(new_b * fg_a + bg_b * (1 - fg_a))
    
    elif mode == "明度":
        bg_h, bg_s, bg_v = rgb_to_hsv(bg_r, bg_g, bg_b)
        fg_h, fg_s, fg_v = rgb_to_hsv(fg_r, fg_g, fg_b)
        new_r, new_g, new_b = hsv_to_rgb(bg_h, bg_s, fg_v)
        r = int(new_r * fg_a + bg_r * (1 - fg_a))
        g = int(new_g * fg_a + bg_g * (1 - fg_a))
        b = int(new_b * fg_a + bg_b * (1 - fg_a))
    
    r = max(0, min(255, int(r)))
    g = max(0, min(255, int(g)))
    b = max(0, min(255, int(b)))
    a = int(max(bg_a, fg_a) * 255)
    
    result_img = Image.new('RGBA', (1, 1), (r, g, b, a))
    return result_img

class Image_transform_layer_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bj_img": ("IMAGE",),
                "fj_img": ("IMAGE",),
                "coordinates": ("STRING", {
                    "default": '[{"index":1,"x":0.5,"y":0.5}]',
                    "multiline": True
                }),
                "coordinate_align": (
                    ["top_left", "top_center", "top_right", 
                     "left_center", "center", "right_center", 
                     "bottom_left", "bottom_center", "bottom_right"],
                    {"default": "center"}
                ),
                "bg_fill": (
                    ["black", "white", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"],
                    {"default": "black"}
                ),
                "x_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "rotation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blending_mode": (BLEND_METHODS, {"default": "正常"}),
                "blend_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE",)
    RETURN_NAMES = ("composite", "mask", "bg_fill",)
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image/ImgLayer"

    def process(self, x_offset, y_offset, rotation, scale, opacity, blending_mode, blend_strength,
                bj_img=None, fj_img=None, mask=None, coordinates="[]", coordinate_align="center", bg_fill="black"):
        color_mapping = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "gray": (128, 128, 128),
        }

        if fj_img is None:
            raise ValueError("前景图像(fj_img)是必需的输入")
        if bj_img is None:
            raise ValueError("背景图像(bj_img)是必需的输入")

        bj_np = bj_img[0].cpu().numpy()
        fj_np = fj_img[0].cpu().numpy()
        bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
        fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
        canvas_width, canvas_height = bj_pil.size

        if mask is None:
            mask = torch.ones((1, fj_pil.size[1], fj_pil.size[0]), dtype=torch.float32)
        mask_tensor = mask if isinstance(mask, torch.Tensor) else pil2tensor(mask.convert('L')) if hasattr(mask, 'convert') else mask

        if mask is not None:
            mask_np = mask[0].cpu().numpy()
            mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
            if mask_pil.size != fj_pil.size:
                mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
            fj_with_mask = fj_pil.copy()
            fj_with_mask.putalpha(mask_pil)
            fj_processed = fj_with_mask
            mask_processed = mask_pil
        else:
            mask_processed = Image.new("L", fj_pil.size, 255)
            fj_processed = fj_pil.copy()
            fj_processed.putalpha(mask_processed)

        processed_width, processed_height = fj_processed.size
        center_x, center_y = processed_width // 2, processed_height // 2
        adjusted_fj = fj_processed
        adjusted_mask = mask_processed

        rotation = float(rotation)
        if rotation != 0 or scale != 1.0:
            adjusted_fj = adjusted_fj.rotate(rotation, center=(center_x, center_y), resample=Image.BICUBIC, expand=True)
            adjusted_mask = adjusted_mask.rotate(rotation, center=(center_x, center_y), resample=Image.BICUBIC, expand=True)
            if scale != 1.0:
                new_size = (int(adjusted_fj.size[0] * scale), int(adjusted_fj.size[1] * scale))
                adjusted_fj = adjusted_fj.resize(new_size, Image.LANCZOS)
                adjusted_mask = adjusted_mask.resize(new_size, Image.LANCZOS)
            processed_width, processed_height = adjusted_fj.size
            center_x, center_y = processed_width // 2, processed_height // 2

        target_x, target_y = 0.5, 0.5
        try:
            points = json.loads(coordinates.strip())
            if isinstance(points, list) and len(points) > 0:
                for point in points:
                    if isinstance(point, dict):
                        temp_x = point.get("x", 0.5)
                        temp_y = point.get("y", 0.5)
                        if isinstance(temp_x, (int, float)) and 0 <= temp_x <= 1:
                            target_x = float(temp_x)
                        if isinstance(temp_y, (int, float)) and 0 <= temp_y <= 1:
                            target_y = float(temp_y)
                        break
            print(f"解析坐标成功：x={target_x}, y={target_y}")
        except json.JSONDecodeError as e:
            print(f"坐标格式解析失败（使用默认值）：{e}，输入内容：{coordinates}")
        except Exception as e:
            print(f"坐标处理异常（使用默认值）：{e}")

        bg_target_x = canvas_width * target_x
        bg_target_y = canvas_height * target_y

        anchor_offset_x, anchor_offset_y = 0, 0
        if coordinate_align == "top_left":
            anchor_offset_x = 0
            anchor_offset_y = 0
        elif coordinate_align == "top_center":
            anchor_offset_x = processed_width / 2
            anchor_offset_y = 0
        elif coordinate_align == "top_right":
            anchor_offset_x = processed_width
            anchor_offset_y = 0
        elif coordinate_align == "left_center":
            anchor_offset_x = 0
            anchor_offset_y = processed_height / 2
        elif coordinate_align == "center":
            anchor_offset_x = processed_width / 2
            anchor_offset_y = processed_height / 2
        elif coordinate_align == "right_center":
            anchor_offset_x = processed_width
            anchor_offset_y = processed_height / 2
        elif coordinate_align == "bottom_left":
            anchor_offset_x = 0
            anchor_offset_y = processed_height
        elif coordinate_align == "bottom_center":
            anchor_offset_x = processed_width / 2
            anchor_offset_y = processed_height
        elif coordinate_align == "bottom_right":
            anchor_offset_x = processed_width
            anchor_offset_y = processed_height

        x_position = bg_target_x - anchor_offset_x + x_offset
        y_position = bg_target_y - anchor_offset_y + y_offset
        paste_x = int(x_position)
        paste_y = int(y_position)

        if opacity < 1.0:
            r, g, b, a = adjusted_fj.split()
            a = a.point(lambda p: p * opacity)
            adjusted_fj = Image.merge("RGBA", (r, g, b, a))

        composite_mask_pil = Image.new("L", (canvas_width, canvas_height), 0)
        mask_paste_x = paste_x
        mask_paste_y = paste_y
        mask_left = max(0, mask_paste_x)
        mask_top = max(0, mask_paste_y)
        mask_right = min(canvas_width, mask_paste_x + adjusted_mask.size[0])
        mask_bottom = min(canvas_height, mask_paste_y + adjusted_mask.size[1])

        if mask_right > mask_left and mask_bottom > mask_top:
            crop_left = max(0, -mask_paste_x)
            crop_top = max(0, -mask_paste_y)
            crop_right = crop_left + (mask_right - mask_left)
            crop_bottom = crop_top + (mask_bottom - mask_top)
            mask_crop = adjusted_mask.crop((crop_left, crop_top, crop_right, crop_bottom))
            composite_mask_pil.paste(mask_crop, (mask_left, mask_top))

        bj_composite_pil = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 255))
        bj_composite_pil.paste(bj_pil, (0, 0))
        if blending_mode != "正常":
            temp_img = Image.new('RGBA', bj_composite_pil.size, (0, 0, 0, 0))
            temp_img.paste(adjusted_fj, (paste_x, paste_y), adjusted_fj)
            blended_pil = Image.new('RGBA', bj_composite_pil.size, (0, 0, 0, 0))
            for x in range(max(0, paste_x), min(canvas_width, paste_x + adjusted_fj.size[0])):
                for y in range(max(0, paste_y), min(canvas_height, paste_y + adjusted_fj.size[1])):
                    if temp_img.getpixel((x, y))[3] > 0:
                        bg_pixel = bj_composite_pil.getpixel((x, y))
                        fg_pixel = temp_img.getpixel((x, y))
                        bg_pixel_img = Image.new('RGBA', (1, 1), bg_pixel)
                        fg_pixel_img = Image.new('RGBA', (1, 1), fg_pixel)
                        blended_pixel_img = apply_blending_mode(
                            bg_pixel_img, fg_pixel_img, blending_mode, blend_strength
                        )
                        blended_pil.putpixel((x, y), blended_pixel_img.getpixel((0, 0)))
            bj_composite_pil = Image.alpha_composite(bj_composite_pil, blended_pil)
        else:
            bj_composite_pil.paste(adjusted_fj, (paste_x, paste_y), adjusted_fj)

        bg_color = color_mapping.get(bg_fill, (0, 0, 0))
        composite_pil = Image.new("RGBA", (canvas_width, canvas_height), bg_color + (255,))
        if blending_mode != "正常":
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

        bj_composite_pil = bj_composite_pil.convert("RGB")
        bj_composite_np = np.array(bj_composite_pil).astype(np.float32) / 255.0
        cropped_composite = torch.from_numpy(bj_composite_np).unsqueeze(0)

        mask_np = np.array(composite_mask_pil).astype(np.float32) / 255.0
        cropped_mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)

        composite_pil = composite_pil.convert("RGB")
        composite_np = np.array(composite_pil).astype(np.float32) / 255.0
        if len(composite_np.shape) == 2:
            composite_np = np.stack([composite_np] * 3, axis=-1)
        composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)

        return (
            cropped_composite,
            cropped_mask_tensor,
            composite_tensor,
        )









class color_select:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        preset_colors = [
            "None",
            "红色", "橙色", "黄色",
            "绿色", "蓝色", "黑色",
            "白色", "中灰色"
        ]
        
        return {
            "required": {
                "preset_color": (preset_colors, {"default": "None"}),

            },

            "optional": {
                "hex_str": ("COLORCODE", {"default": "#222222"}),
            }


        }
    NAME = "color_select"
    RETURN_TYPES = ("COLORCODE", "STRING")
    RETURN_NAMES = ("color", "hex_str")
    FUNCTION = "output_color"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def output_color(self, preset_color, hex_str):
        color_presets = {
            "红色": "#ff0000",
            "橙色": "#ff7f00",
            "黄色": "#ffff00",
            "绿色": "#00ff00",
            "蓝色": "#0000ff",
            "黑色": "#000000",
            "白色": "#ffffff",
            "中灰色": "#808080"
        }
        
        if preset_color != "None" and preset_color in color_presets:
            final_color = color_presets[preset_color]
        else:
            final_color = hex_str.strip().lower()
            if not final_color.startswith("#"):
                final_color = f"#{final_color}"
            if len(final_color) == 4:
                final_color = f"#{final_color[1]}{final_color[1]}{final_color[2]}{final_color[2]}{final_color[3]}{final_color[3]}"
            elif len(final_color) != 7:
                final_color = "#ffffff"
        
        return (final_color, final_color)










