import os
import torch
import numpy as np
import folder_paths
from PIL import ImageOps, ImageEnhance, Image, ImageOps, ImageChops, ImageFilter, ImageDraw, ImageFont
from torchvision.transforms.functional import to_pil_image, to_tensor
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import io
from typing import Literal, Any
import math
from comfy.utils import common_upscale
import typing as t
from pathlib import Path







from math import ceil, sqrt
from ..main_unit import *


#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False

try:
    import onnxruntime as ort
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    ort = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False




#region--------------def--------layout----------------------

font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
file_list = [f for f in os.listdir(font_dir) if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith(".ttf")]

color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (160, 85, 15),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (102, 102, 102),
    "olive": (128, 128, 0),
    "lime": (0, 128, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 128),
    "aqua": (0, 255, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),    
}


COLORS = ["white", "black", "red", "green", "blue", "yellow",
          "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
          "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
          "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
          "violet", "coral", "indigo"]




#endregion----------------------layout----------------------




#region----------lay_mul_image----------------------------------
import torch
import os
import sys
from PIL import Image, ImageFont, ImageDraw
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def tensor2pil(t_image: torch.Tensor) -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def get_font_list():
    font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
    font_extensions = ('.ttf', '.otf', '.ttc')
    font_list = []
    for file in os.listdir(font_dir):
        if file.lower().endswith(font_extensions):
            font_list.append(file)
    return font_list

file_list = get_font_list()



class lay_mul_image:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        self.file_list = [f for f in os.listdir(self.font_dir) if f.endswith(('.ttf', '.otf'))]

    @classmethod
    def INPUT_TYPES(cls):
        inst = cls()
        file_list = inst.file_list
        
        return {
            "required": {
                "sub_team_direction": (["row", "column"], {"default": "column"}),
                "main_position": (["top", "bottom", "left", "right"], {"default": "left"}),
                "font": (file_list,),
            },
            "optional": {
                "main_image": ("IMAGE",),
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                
                "main_text": ("STRING", {"multiline": False, "default": ""}),
                "main_text_size": ("INT", {"default": 60, "min": 8, "max": 128, "step": 2}),
                "main_long_size": ("INT", {"default": 1024, "min": 128, "max": 4096, "step": 32}),

                "image1_text": ("STRING", {"multiline": False, "default": ""}),
                "image2_text": ("STRING", {"multiline": False, "default": ""}),
                "image3_text": ("STRING", {"multiline": False, "default": ""}),
                "image4_text": ("STRING", {"multiline": False, "default": ""}),
                "sub_text_size": ("INT", {"default": 30, "min": 8, "max": 128, "step": 2}),
                "sub_team_size": ("INT", {"default": 512, "min": 128, "max": 4096, "step": 32}),
                "border": ("INT", {"default": 8, "min": 0, "max": 128, "step": 1}),

            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = 'process_images'
    CATEGORY = "Apt_Preset/imgEffect"
    DESCRIPTION = """
    - 子图排列方向：4个子图横向或纵向排列，
    - 主图位置：围绕子图组合的位置
    """


    def process_images(self, sub_team_direction, sub_team_size, sub_text_size, main_text_size, font, border,
                      image1=None, image1_text="", image2=None, image2_text="",
                      image3=None, image3_text="", image4=None, image4_text="",
                      main_image=None, main_text="", main_position="top", main_long_size=1024):
        
        sub_images = []
        sub_texts = []
        
        img_text_pairs = [
            (image1, image1_text),
            (image2, image2_text),
            (image3, image3_text),
            (image4, image4_text)
        ]
        
        for img, text in img_text_pairs:
            if img is not None:
                pil_img = tensor2pil(img.unsqueeze(0))
                sub_images.append(pil_img)
                sub_texts.append(text)
                
        # 当没有子图但有主图时，应该处理主图
        if not sub_images and main_image is not None:
            main_pil = tensor2pil(main_image)
            if main_text:
                main_pil = self.add_text_to_image(main_pil, main_text, main_text_size, font, border)
            main_resized = self.resize_main_image(main_pil, main_long_size)
            return (pil2tensor(main_resized),)


        if not sub_images:
            return (pil2tensor(Image.new('RGB', (512, 512), color='white')),)
        
        sub_grid = self.create_sub_grid(sub_images, sub_texts, sub_team_direction, sub_team_size, sub_text_size, font, border)
        
        if main_image is not None:
            main_pil = tensor2pil(main_image)
            if main_text:
                main_pil = self.add_text_to_image(main_pil, main_text, main_text_size, font, border)
            main_resized = self.resize_main_image(main_pil, main_long_size)
            result_image = self.combine_main_sub(main_resized, sub_grid, main_position, border)
        else:
            result_image = sub_grid
            
        return (pil2tensor(result_image),)
    
    def resize_sub_image(self, image, direction, target_size) -> Image:
        if direction == "row":
            scale = target_size / image.height
            new_width = int(image.width * scale)
            return image.resize((new_width, target_size), Image.LANCZOS)
        else:
            scale = target_size / image.width
            new_height = int(image.height * scale)
            return image.resize((target_size, new_height), Image.LANCZOS)
    
    def add_text_to_image(self, image, text, text_size, font_name, border) -> Image:
        if not text.strip():
            return image
            
        text_height = int(text_size * 1.8)
        new_image = Image.new('RGB', (image.width, image.height + text_height), color='white')
        new_image.paste(image, (0, 0))
        
        draw = ImageDraw.Draw(new_image)
        
        font_path = os.path.join(self.font_dir, font_name)
        font_obj = ImageFont.truetype(font_path, text_size)
        
        bbox = font_obj.getbbox(text)
        text_width = bbox[2] - bbox[0]
        x = (image.width - text_width) // 2
        y = image.height + (text_height - text_size) // 2
        
        try:
            draw.text((x, y), text, fill=(0, 0, 0), font=font_obj, features=['-liga'])
        except:
            draw.text((x, y), text, fill=(0, 0, 0), font=font_obj)
        
        return new_image
    
    def create_sub_grid(self, images, texts, direction, target_size, text_size, font_name, border) -> Image:
        processed_images = []
        for img, text in zip(images, texts):
            resized = self.resize_sub_image(img, direction, target_size)
            if text:
                with_text = self.add_text_to_image(resized, text, text_size, font_name, border)
                processed_images.append(with_text)
            else:
                processed_images.append(resized)
        
        spacing = border
        
        if direction == "row":
            total_width = sum(img.width for img in processed_images) + spacing * (len(processed_images) - 1)
            max_height = max(img.height for img in processed_images)
            grid_width = total_width + border * 2
            grid_height = max_height + border * 2
            
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            
            x = border
            for img in processed_images:
                y = border + (max_height - img.height) // 2
                grid_img.paste(img, (x, y))
                x += img.width + spacing
        else:
            max_width = max(img.width for img in processed_images)
            total_height = sum(img.height for img in processed_images) + spacing * (len(processed_images) - 1)
            grid_width = max_width + border * 2
            grid_height = total_height + border * 2
            
            grid_img = Image.new('RGB', (grid_width, grid_height), color='white')
            
            y = border
            for img in processed_images:
                x = border + (max_width - img.width) // 2
                grid_img.paste(img, (x, y))
                y += img.height + spacing
        
        return grid_img
    
    def resize_main_image(self, main_image, long_side) -> Image:
        width, height = main_image.size
        if width >= height:
            scale = long_side / width
            new_width = long_side
            new_height = int(height * scale)
        else:
            scale = long_side / height
            new_height = long_side
            new_width = int(width * scale)
        return main_image.resize((new_width, new_height), Image.LANCZOS)
    
    def combine_main_sub(self, main_img, sub_grid, position, border) -> Image:
        spacing = border * 2
        
        if position in ['top', 'bottom']:
            max_width = max(main_img.width, sub_grid.width)
            total_width = max_width + 2 * border
            total_height = main_img.height + sub_grid.height + spacing + 2 * border

            result = Image.new('RGB', (total_width, total_height), color='white')

            if position == 'top':
                main_y = border
                sub_y = main_y + main_img.height + spacing
            else:
                sub_y = border
                main_y = sub_y + sub_grid.height + spacing

            main_x = border + (max_width - main_img.width) // 2
            sub_x = border + (max_width - sub_grid.width) // 2

        else:
            max_height = max(main_img.height, sub_grid.height)
            total_height = max_height + 2 * border
            total_width = main_img.width + sub_grid.width + spacing + 2 * border

            result = Image.new('RGB', (total_width, total_height), color='white')

            if position == 'left':
                main_x = border
                sub_x = main_x + main_img.width + spacing
            else:
                sub_x = border
                main_x = sub_x + sub_grid.width + spacing

            main_y = border + (max_height - main_img.height) // 2
            sub_y = border + (max_height - sub_grid.height) // 2

        result.paste(main_img, (main_x, main_y))
        result.paste(sub_grid, (sub_x, sub_y))
        return result








#endregion-----------------纹理组---------------------------------




















