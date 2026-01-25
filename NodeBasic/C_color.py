


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

import torch.nn.functional as F
import node_helpers
from typing import Tuple



from ..main_unit import *
from ..office_unit import ImageUpscaleWithModel,UpscaleModelLoader,composite



if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#---------------------安全导入

try:
    from scipy.interpolate import CubicSpline
    REMOVER_AVAILABLE = True  
except ImportError:
    CubicSpline = None
    REMOVER_AVAILABLE = False 


try:   
    from scipy.ndimage import distance_transform_edt
    REMOVER_AVAILABLE = True  
except ImportError:
    distance_transform_edt = None
    REMOVER_AVAILABLE = False 




try:
    import onnxruntime as ort
    REMOVER_AVAILABLE = True  
except ImportError:
    ort = None
    REMOVER_AVAILABLE = False  



try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  








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
                "ref_img": ("IMAGE",),
                "target_image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 1.0, "step": 0.1}),
                "skin_protection": ("FLOAT", {"default": 0.2, "min": 0, "max": 1.0, "step": 0.1}),
                "brightness_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "contrast_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "saturation_range": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "tone_strength": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
            },
            "optional": {
                "target_mask": ("MASK", {"default": None}),
            },
        }

    CATEGORY = "Apt_Preset/image/color_adjust"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match_hue"

    def match_hue(self, ref_img, target_image, strength, skin_protection,  brightness_range,
                contrast_range, saturation_range, tone_strength, target_mask=None):
        
        auto_brightness =True
        auto_contrast =True
        auto_tone =True
        auto_saturation =True


        for img in ref_img:
            img_cv1 = tensor2cv2(img)

        for img in target_image:
            img_cv2 = tensor2cv2(img)

        img_cv3 = None
        if target_mask is not None:
            for img3 in target_mask:
                img_cv3 = img3.cpu().numpy()
                img_cv3 = (img_cv3 * 255).astype(np.uint8)

        result_img = color_transfer(img_cv1, img_cv2, img_cv3, strength, skin_protection, auto_brightness,
                                    brightness_range,auto_contrast, contrast_range, auto_saturation,
                                    saturation_range, auto_tone, tone_strength)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        rst = torch.from_numpy(result_img.astype(np.float32) / 255.0).unsqueeze(0)

        return (rst,)





#endregion-----------------------color_transfer----------------





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




class color_select:
    @classmethod
    def INPUT_TYPES(cls):
        preset_colors = [
            "None",
            "红色", "橙色", "黄色",
            "绿色", "蓝色", "黑色",
            "白色", "中灰色"
        ]
        
        return {
            "required": {
                "color_code": ("STRING", {"multiline": False, "default": "#FFFFFF", "widget": "hidden"}),
                "preset_color": (preset_colors, {"default": "None"}),
            }
        }
    
    NAME = "color_select"
    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("color_code",)
    FUNCTION = "get_color"
    CATEGORY = "Apt_Preset/image/color_adjust"
    OUTPUT_IS_LIST = (True,)



    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def get_color(self, color_code, preset_color):
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
            final_color = color_code.strip().lower()
            if not final_color.startswith("#"):
                final_color = f"#{final_color}"
            if len(final_color) == 4:
                final_color = f"#{final_color[1]}{final_color[1]}{final_color[2]}{final_color[2]}{final_color[3]}{final_color[3]}"
            elif len(final_color) != 7:
                final_color = "#FFFFFF"
        
        return ([final_color],)






class color_Fragment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "fragment_width": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
                "fragment_height": ("INT", {"default": 64, "min": 8, "max": 512, "step": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "recombine_color_Fragments"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def split_into_fragments(self, image, fragment_size):
        h, w = image.shape[:2]
        fh, fw = fragment_size
        fragments = []
        
        for y in range(0, h - fh + 1, fh):
            for x in range(0, w - fw + 1, fw):
                fragment = image[y:y+fh, x:x+fw]
                fragments.append(fragment)
        
        return fragments, (fh, fw), (h // fh, w // fw)

    def recombine_fragments(self, fragments, fragment_size, grid_size, seed):
        if not fragments:
            raise ValueError("图像碎片列表不能为空")
        
        random.seed(seed)
        random.shuffle(fragments)
        
        fh, fw = fragment_size
        rows, cols = grid_size
        recombined = np.zeros((rows * fh, cols * fw, 3), dtype=np.uint8)
        
        idx = 0
        for y in range(rows):
            for x in range(cols):
                if idx < len(fragments):
                    recombined[y*fh:(y+1)*fh, x*fw:(x+1)*fw] = fragments[idx]
                idx += 1
        
        return recombined

    def recombine_color_Fragments(self, image, fragment_width, fragment_height, seed):
        recombine_mode = "random"
        
        # 优化：增加类型判断，兼容不同输入类型
        if isinstance(image, torch.Tensor):
            image_np = 255.0 * image[0].cpu().numpy()
        else:
            image_np = 255.0 * image[0]
        image_np = image_np.astype(np.uint8)
        
        fragments, frag_size, grid_size = self.split_into_fragments(image_np, (fragment_height, fragment_width))
        recombined_image = self.recombine_fragments(fragments, frag_size, grid_size, seed)
        
        # 转换为ComfyUI标准格式：float32 + 增加batch维度 + 转Tensor
        recombined_image = recombined_image.astype(np.float32) / 255.0
        # 关键修改1：调整维度顺序（ComfyUI的IMAGE格式是 [batch, height, width, channels]）
        # 确保维度正确（有些情况下可能需要调整为 [H, W, C] -> [1, H, W, C]）
        if len(recombined_image.shape) == 3:
            recombined_image = np.expand_dims(recombined_image, axis=0)
        # 关键修改2：将numpy数组转换为PyTorch张量（这是修复报错的核心）
        recombined_image_tensor = torch.from_numpy(recombined_image)
        
        # 返回Tensor而不是numpy数组
        return (recombined_image_tensor,)





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
    CATEGORY = "Apt_Preset/image/ImgLayer"

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
    CATEGORY = "Apt_Preset/image/ImgLayer"

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












