


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
import copy
import ast
import torch.nn.functional as F
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
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "color_adjust_HSL"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def color_adjust_HSL(self, image, brightness, contrast, saturation, hue, sharpness, blur, gaussian_blur, edge_enhance, detail_enhance, unique_id, save_preview=True, return_ui=True):
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
        GLOBAL_IMAGE_CACHE[str(unique_id)] = {"image": image[0:1].cpu()}

        if not save_preview and not return_ui:
            return tensors

        preview_tensor = tensors
        if isinstance(preview_tensor, torch.Tensor) and preview_tensor.dim() == 3:
            preview_tensor = preview_tensor.unsqueeze(0)

        results = []
        if save_preview and isinstance(preview_tensor, torch.Tensor) and preview_tensor.dim() == 4 and preview_tensor.shape[0] > 0:
            temp_dir = folder_paths.get_temp_directory()
            t = preview_tensor[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"hsl_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        if return_ui:
            return {"ui": {"bg_image": results}, "result": (tensors,)}
        return tensors

    def adjust_hue(self, image, hue_shift):
        if hue_shift == 0:
            return image
        hsv_image = image.convert('HSV')
        h, s, v = hsv_image.split()
        h = h.point(lambda x: (x + int(hue_shift * 255)) % 256)
        hsv_image = Image.merge('HSV', (h, s, v))
        return hsv_image.convert('RGB')



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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/image/color_adjust"

    def execute(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, highlight_detail, midtone_detail, shadow_detail, overall_intensity, unique_id, save_preview=True, return_ui=True):
        GLOBAL_IMAGE_CACHE[str(unique_id)] = {"image": image[0:1].cpu()}
        try:
            # 处理批次图像
            processed_images = []
            
            for i in range(image.shape[0]):  # 遍历批次中的每张图像
                single_image = image[i:i+1]  # 保持批次维度
                processed_single = self.process_single_image(single_image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, highlight_detail, midtone_detail, shadow_detail, overall_intensity)
                processed_images.append(processed_single)
            
            # 将处理后的图像堆叠成批次
            result = torch.cat(processed_images, dim=0)
            if not save_preview and not return_ui:
                return result

            results = []
            if save_preview and isinstance(result, torch.Tensor) and result.dim() == 4 and result.shape[0] > 0:
                temp_dir = folder_paths.get_temp_directory()
                t = result[0]
                img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                filename = f"hdr_preview_{random.randint(1, 1000000)}.png"
                img_pil.save(os.path.join(temp_dir, filename))
                results.append({"filename": filename, "subfolder": "", "type": "temp"})

            if return_ui:
                return {"ui": {"bg_image": results}, "result": (result,)}
            return result
        except Exception as e:
            print(f"Error in color_adjust_HDR: {e}")
            # 返回与输入相同形状的黑色图像
            black_image = torch.zeros_like(image)
            if return_ui:
                return {"ui": {"bg_image": []}, "result": (black_image,)}
            return black_image

    def process_single_image(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, highlight_detail, midtone_detail, shadow_detail, overall_intensity):
        # 确保图像格式正确
        image = self.ensure_image_format(image)

        # 应用HDR效果
        processed_image = self.apply_hdr(image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, [highlight_detail, midtone_detail, shadow_detail])

        # 混合原图和处理后的图像
        blended_image = cv2.addWeighted(processed_image, overall_intensity, image, 1 - overall_intensity, 0)

        # 转换回ComfyUI格式
        if isinstance(blended_image, np.ndarray):
            if len(blended_image.shape) == 3:  # HWC
                blended_image = np.expand_dims(blended_image, axis=0)  # 添加批次维度 BCHW
            elif len(blended_image.shape) == 4:  # 已有批次维度
                pass  # 不做改变

        blended_image = torch.from_numpy(blended_image).float()
        blended_image = blended_image / 255.0
        blended_image = blended_image.to(torch.device('cpu'))

        return blended_image

    def ensure_image_format(self, image):
        if isinstance(image, torch.Tensor):
            # 转换为numpy格式并还原到0-255范围
            if image.dim() == 4:  # BCHW
                if image.shape[0] == 1:  # 单张图像，去掉批次维度
                    image = image.squeeze(0)
            elif image.dim() == 3:  # CHW
                pass  # 已经是合适的格式
            elif image.dim() == 2:  # HW
                image = image.unsqueeze(-1).repeat(1, 1, 3)  # 转为RGB
            
            # 转换为numpy并还原到0-255范围
            image = image.numpy()
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image

    def apply_hdr(self, image, HDR_intensity, underexposure_factor, overexposure_factor, gamma, exposure_times):
        # 如果是批次图像，处理第一张
        if len(image.shape) == 4 and image.shape[0] > 1:
            image = image[0]  # 取第一张图像进行处理
        
        # 确保图像是3通道
        if len(image.shape) == 3 and image.shape[-1] == 3:
            pass  # 已经是RGB格式
        elif len(image.shape) == 3 and image.shape[0] == 3:
            # CHW -> HWC
            image = np.transpose(image, (1, 2, 0))
        
        # 创建HDR处理对象
        if cv2 is not None:
            try:
                # 创建多曝光图像
                times = np.array(exposure_times, dtype=np.float32)

                # 创建不同曝光的图像
                underexposed = np.clip(image.astype(np.float32) * underexposure_factor, 0, 255).astype(np.uint8)
                normal_exposed = image.astype(np.uint8)
                overexposed = np.clip(image.astype(np.float32) * overexposure_factor, 0, 255).astype(np.uint8)

                exposure_images = [underexposed, normal_exposed, overexposed]

                # 如果OpenCV支持HDR处理
                if hasattr(cv2, 'createMergeDebevec'):
                    hdr = cv2.createMergeDebevec()
                    hdr_image = hdr.process([img.astype(np.float32)/255.0 for img in exposure_images], times=times)
                    
                    tonemap = cv2.createTonemapReinhard(gamma=gamma)
                    ldr_image = tonemap.process(hdr_image)
                else:
                    # 简化的HDR模拟
                    ldr_image = normal_exposed.astype(np.float32) / 255.0

                # 应用HDR强度
                ldr_image = ldr_image * HDR_intensity
                ldr_image = np.clip(ldr_image, 0, 1)
                ldr_image = np.clip(ldr_image * 255, 0, 255).astype(np.uint8)

                return ldr_image
            except Exception as e:
                print(f"HDR processing error: {e}")
                return image.astype(np.uint8)
        else:
            # 如果没有cv2，返回原始图像
            return image.astype(np.uint8)



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


def image_stats(image, mask=None):
    ab = image[:, :, 1:]
    if mask is None:
        return np.mean(ab, axis=(0, 1)), np.std(ab, axis=(0, 1))
    m = mask
    if m.ndim == 3:
        m = m.squeeze()
    m = m.astype(np.float32)
    if m.size == 0:
        return np.mean(ab, axis=(0, 1)), np.std(ab, axis=(0, 1))
    if np.max(m) <= 1.0:
        m = m > 0
    else:
        m = m > 0
    if not np.any(m):
        return np.mean(ab, axis=(0, 1)), np.std(ab, axis=(0, 1))
    v = ab[m]
    return np.mean(v, axis=0), np.std(v, axis=0)


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


def adjust_tone(source, target, tone_strength=0.7, mask=None, source_mask=None):
    h, w = target.shape[:2]
    source = cv2.resize(source, (w, h))
    lab_image = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_image = lab_image[:,:,0]
    l_source = lab_source[:,:,0]

    def _norm_mask(m):
        m = cv2.resize(m, (w, h))
        m = m.astype(np.float32)
        if m.size == 0:
            return m
        if np.max(m) <= 1.0:
            return m
        return m / 255.0

    src_mask = None
    if source_mask is not None:
        src_mask = _norm_mask(source_mask)

    if mask is not None:
        mask = _norm_mask(mask)
        l_adjusted = np.copy(l_image)
        if np.any(mask > 0):
            mean_target = np.mean(l_image[mask > 0])
            std_target = np.std(l_image[mask > 0])
        else:
            mean_target = np.mean(l_image)
            std_target = np.std(l_image)
        src_sel = src_mask if src_mask is not None else mask
        if src_sel is not None and np.any(src_sel > 0):
            mean_source = np.mean(l_source[src_sel > 0])
            std_source = np.std(l_source[src_sel > 0])
        else:
            mean_source = np.mean(l_source)
            std_source = np.std(l_source)
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
        if src_mask is not None and np.any(src_mask > 0):
            mean_source = np.mean(l_source[src_mask > 0])
            std_source = np.std(l_source[src_mask > 0])
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
        image = image[0]
    npimage = image.detach().cpu().numpy()
    cv2image = np.clip(npimage * 255.0, 0, 255).astype(np.uint8)
    if cv2image.ndim == 3 and cv2image.shape[2] == 4:
        return cv2.cvtColor(cv2image, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(cv2image, cv2.COLOR_RGB2BGR)


def color_transfer(source, target, mask=None, strength=1.0, skin_protection=0.2, auto_brightness=True,
                   brightness_range=0.5, auto_contrast=False, contrast_range=0.5,
                   auto_saturation=False, saturation_range=0.5, auto_tone=False, tone_strength=0.7, ref_mask=None):
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

    src_means, src_stds = image_stats(source_lab, ref_mask)
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
            src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            if ref_mask is not None:
                rm = ref_mask
                if rm.ndim == 3:
                    rm = rm.squeeze()
                rm = cv2.resize(rm.astype(np.float32), (src_gray.shape[1], src_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                if np.max(rm) > 1.0:
                    rm = rm / 255.0
                if np.any(rm > 0):
                    source_brightness = np.mean(src_gray[rm > 0])
                else:
                    source_brightness = np.mean(src_gray)
            else:
                source_brightness = np.mean(src_gray)
            target_brightness = np.mean(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, brightness_range*-1, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor, mask)
        if auto_contrast:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            if ref_mask is not None:
                rm = ref_mask
                if rm.ndim == 3:
                    rm = rm.squeeze()
                rm = cv2.resize(rm.astype(np.float32), (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                if np.max(rm) > 1.0:
                    rm = rm / 255.0
                if np.any(rm > 0):
                    source_contrast = np.std(source_gray[rm > 0])
                else:
                    source_contrast = np.std(source_gray)
            else:
                source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, contrast_range*-1, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor, mask)
        if auto_saturation:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            if ref_mask is not None:
                rm = ref_mask
                if rm.ndim == 3:
                    rm = rm.squeeze()
                rm = cv2.resize(rm.astype(np.float32), (source_hsv.shape[1], source_hsv.shape[0]), interpolation=cv2.INTER_NEAREST)
                if np.max(rm) > 1.0:
                    rm = rm / 255.0
                if np.any(rm > 0):
                    source_saturation = np.mean(source_hsv[:, :, 1][rm > 0])
                else:
                    source_saturation = np.mean(source_hsv[:, :, 1])
            else:
                source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, saturation_range*-1, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor, mask)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength, mask, source_mask=ref_mask)
    else:
        if auto_brightness:
            src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            if ref_mask is not None:
                rm = ref_mask
                if rm.ndim == 3:
                    rm = rm.squeeze()
                rm = cv2.resize(rm.astype(np.float32), (src_gray.shape[1], src_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                if np.max(rm) > 1.0:
                    rm = rm / 255.0
                if np.any(rm > 0):
                    source_brightness = np.mean(src_gray[rm > 0])
                else:
                    source_brightness = np.mean(src_gray)
            else:
                source_brightness = np.mean(src_gray)
            target_brightness = np.mean(cv2.cvtColor(target, cv2.COLOR_BGR2GRAY))
            brightness_difference = source_brightness - target_brightness
            brightness_factor = 1.0 + np.clip(brightness_difference / 255 * brightness_range, brightness_range*-1, brightness_range)
            final_result = adjust_brightness(final_result, brightness_factor)
        if auto_contrast:
            source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
            target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            if ref_mask is not None:
                rm = ref_mask
                if rm.ndim == 3:
                    rm = rm.squeeze()
                rm = cv2.resize(rm.astype(np.float32), (source_gray.shape[1], source_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
                if np.max(rm) > 1.0:
                    rm = rm / 255.0
                if np.any(rm > 0):
                    source_contrast = np.std(source_gray[rm > 0])
                else:
                    source_contrast = np.std(source_gray)
            else:
                source_contrast = np.std(source_gray)
            target_contrast = np.std(target_gray)
            contrast_difference = source_contrast - target_contrast
            contrast_factor = 1.0 + np.clip(contrast_difference / 255, contrast_range*-1, contrast_range)
            final_result = adjust_contrast(final_result, contrast_factor)
        if auto_saturation:
            source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
            if ref_mask is not None:
                rm = ref_mask
                if rm.ndim == 3:
                    rm = rm.squeeze()
                rm = cv2.resize(rm.astype(np.float32), (source_hsv.shape[1], source_hsv.shape[0]), interpolation=cv2.INTER_NEAREST)
                if np.max(rm) > 1.0:
                    rm = rm / 255.0
                if np.any(rm > 0):
                    source_saturation = np.mean(source_hsv[:, :, 1][rm > 0])
                else:
                    source_saturation = np.mean(source_hsv[:, :, 1])
            else:
                source_saturation = np.mean(source_hsv[:, :, 1])
            target_saturation = np.mean(target_hsv[:, :, 1])
            saturation_difference = source_saturation - target_saturation
            saturation_factor = 1.0 + np.clip(saturation_difference / 255, saturation_range*-1, saturation_range)
            final_result = adjust_saturation(final_result, saturation_factor)
        if auto_tone:
            final_result = adjust_tone(source, final_result, tone_strength, source_mask=ref_mask)

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
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    CATEGORY = "Apt_Preset/image/color_adjust"

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "match_hue"

    def match_hue(self, ref_img, target_image, strength, skin_protection,  brightness_range,
                contrast_range, saturation_range, tone_strength, target_mask=None, unique_id=None, save_preview=True, return_ui=True):
        
        auto_brightness =True
        auto_contrast =True
        auto_tone =True
        auto_saturation =True


        ref_alpha_mask = None
        for img in ref_img:
            if img.dim() == 3 and img.shape[-1] == 4:
                a = img[:, :, 3].detach().cpu().numpy()
                ref_alpha_mask = (a > 1e-6).astype(np.uint8) * 255
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
                                    saturation_range, auto_tone, tone_strength, ref_mask=ref_alpha_mask)
        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        rst = torch.from_numpy(result_img.astype(np.float32) / 255.0).unsqueeze(0)

        if unique_id is not None:
            cache = {
                "ref_img": ref_img[0:1].cpu(),
                "target_image": target_image[0:1].cpu(),
                "target_mask": None,
            }
            if target_mask is not None:
                cache["target_mask"] = target_mask[0:1].cpu()
            GLOBAL_IMAGE_CACHE[str(unique_id)] = cache

        if not save_preview and not return_ui:
            return rst

        results = []
        if save_preview and isinstance(rst, torch.Tensor) and rst.dim() == 4 and rst.shape[0] > 0:
            temp_dir = folder_paths.get_temp_directory()
            t = rst[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"match_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        if return_ui:
            return {"ui": {"bg_image": results}, "result": (rst,)}
        return rst




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
    CATEGORY = "Apt_Preset/image/visualize_edit"



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
        
        return (final_color,)






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
            ,
            "hidden": {
                "unique_id": "UNIQUE_ID"
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
                  image1=None, image2=None, mask2=None, unique_id=None, save_preview=True, return_ui=True):

        if unique_id is not None:
            cache = {"image1": None, "image2": None, "mask2": None}
            if image1 is not None:
                cache["image1"] = image1[0:1].cpu()
            if image2 is not None:
                cache["image2"] = image2[0:1].cpu()
            if mask2 is not None:
                cache["mask2"] = mask2[0:1].cpu()
            GLOBAL_IMAGE_CACHE[str(unique_id)] = cache
        
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
        if not save_preview and not return_ui:
            return fused_rgb

        results = []
        if save_preview and isinstance(fused_rgb, torch.Tensor) and fused_rgb.dim() == 4 and fused_rgb.shape[0] > 0:
            temp_dir = folder_paths.get_temp_directory()
            t = fused_rgb[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"cnmapmix_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        if return_ui:
            return {"ui": {"bg_image": results}, "result": (fused_rgb,)}
        return fused_rgb

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



#region---------------------------------------visualize---------------------------------------




#region-----------------color-curve--------------------------------------

from server import PromptServer
from aiohttp import web
import torch
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
import folder_paths
import os
import random
from PIL import Image

GLOBAL_IMAGE_CACHE = {}

CURVE_PRESETS = {
    "Contrast (S)": {
        "RGB": [[0.0, 0.0], [0.25, 0.18], [0.5, 0.5], [0.75, 0.82], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Soft Contrast": {
        "RGB": [[0.0, 0.0], [0.3, 0.27], [0.7, 0.73], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Matte": {
        "RGB": [[0.0, 0.08], [0.25, 0.28], [0.75, 0.88], [1.0, 0.95]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Fade Highlights": {
        "RGB": [[0.0, 0.0], [0.6, 0.62], [1.0, 0.92]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Brighten": {
        "RGB": [[0.0, 0.0], [0.25, 0.35], [0.5, 0.65], [0.75, 0.85], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Darken": {
        "RGB": [[0.0, 0.0], [0.25, 0.15], [0.5, 0.38], [0.75, 0.68], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Warm": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.0], [0.5, 0.55], [1.0, 1.0]],
        "G": [[0.0, 0.0], [0.5, 0.52], [1.0, 1.0]],
        "B": [[0.0, 0.0], [0.5, 0.45], [1.0, 0.95]],
    },
    "Cool": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.0], [0.5, 0.45], [1.0, 0.95]],
        "G": [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]],
        "B": [[0.0, 0.0], [0.5, 0.55], [1.0, 1.0]],
    },
    "Teal & Orange": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.0], [0.4, 0.38], [0.7, 0.78], [1.0, 1.0]],
        "G": [[0.0, 0.02], [0.5, 0.5], [1.0, 1.0]],
        "B": [[0.0, 0.08], [0.4, 0.45], [0.7, 0.62], [1.0, 0.95]],
    },
    "Cross Process": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.06], [0.35, 0.33], [0.75, 0.86], [1.0, 1.0]],
        "G": [[0.0, 0.0], [0.5, 0.48], [1.0, 0.95]],
        "B": [[0.0, 0.0], [0.25, 0.22], [0.75, 0.88], [1.0, 1.0]],
    },
    "Linear": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Soft S": {
        "RGB": [[0.0, 0.0], [0.157, 0.141], [0.376, 0.361], [0.627, 0.784], [0.816, 0.894], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Strong S": {
        "RGB": [[0.0, 0.0], [0.125, 0.078], [0.314, 0.275], [0.502, 0.502], [0.69, 0.784], [0.878, 0.949], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Bright Midtones": {
        "RGB": [[0.0, 0.0], [0.251, 0.431], [0.502, 0.667], [0.753, 0.863], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Dark Mood": {
        "RGB": [[0.0, 0.0], [0.188, 0.11], [0.376, 0.314], [0.502, 0.471], [0.753, 0.706], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Highlight Lift": {
        "RGB": [[0.0, 0.0], [0.251, 0.353], [0.502, 0.588], [0.753, 0.824], [0.902, 0.961], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Shadow Lift": {
        "RGB": [[0.0, 0.0], [0.047, 0.094], [0.251, 0.306], [0.502, 0.549], [0.753, 0.784], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Film Matte": {
        "RGB": [[0.0, 0.0], [0.141, 0.11], [0.376, 0.376], [0.627, 0.745], [0.816, 0.878], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Fade Blacks": {
        "RGB": [[0.0, 0.0], [0.031, 0.071], [0.188, 0.188], [0.502, 0.549], [0.753, 0.784], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Punchy": {
        "RGB": [[0.0, 0.0], [0.188, 0.125], [0.376, 0.329], [0.502, 0.502], [0.627, 0.706], [0.816, 0.922], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "High Key": {
        "RGB": [[0.0, 0.0], [0.251, 0.392], [0.502, 0.706], [0.753, 0.902], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Low Key": {
        "RGB": [[0.0, 0.0], [0.157, 0.078], [0.376, 0.251], [0.502, 0.431], [0.753, 0.706], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Subtle S": {
        "RGB": [[0.0, 0.0], [0.188, 0.173], [0.439, 0.416], [0.627, 0.722], [0.816, 0.863], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Retro Fade": {
        "RGB": [[0.0, 0.0], [0.047, 0.071], [0.251, 0.282], [0.502, 0.533], [0.753, 0.784], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Vintage Warm": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.0], [0.188, 0.176], [0.376, 0.38], [0.502, 0.549], [0.753, 0.804], [1.0, 1.0]],
        "G": [[0.0, 0.0], [0.188, 0.157], [0.376, 0.353], [0.502, 0.51], [0.753, 0.784], [1.0, 1.0]],
        "B": [[0.0, 0.0], [0.188, 0.141], [0.376, 0.314], [0.502, 0.471], [0.753, 0.745], [1.0, 0.965]],
    },
    "Vintage Cool": {
        "RGB": [[0.0, 0.0], [1.0, 1.0]],
        "R": [[0.0, 0.0], [0.188, 0.141], [0.376, 0.294], [0.502, 0.451], [0.753, 0.706], [1.0, 0.941]],
        "G": [[0.0, 0.0], [0.188, 0.157], [0.376, 0.314], [0.502, 0.471], [0.753, 0.745], [1.0, 1.0]],
        "B": [[0.0, 0.0], [0.188, 0.176], [0.376, 0.38], [0.502, 0.549], [0.753, 0.824], [1.0, 1.0]],
    },
    "Cinematic S": {
        "RGB": [[0.0, 0.0], [0.251, 0.188], [0.502, 0.502], [0.753, 0.816], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "HDR Boost": {
        "RGB": [[0.0, 0.0], [0.125, 0.11], [0.251, 0.251], [0.502, 0.627], [0.753, 0.863], [1.0, 1.0]],
        "R": [[0.0, 0.0], [1.0, 1.0]],
        "G": [[0.0, 0.0], [1.0, 1.0]],
        "B": [[0.0, 0.0], [1.0, 1.0]],
    },
    "Film Negative": {
        "RGB": [[0.0, 1.0], [0.125, 0.784], [0.376, 0.627], [0.627, 0.314], [0.878, 0.094], [1.0, 0.0]],
        "R": [[0.0, 1.0], [1.0, 0.0]],
        "G": [[0.0, 1.0], [1.0, 0.0]],
        "B": [[0.0, 1.0], [1.0, 0.0]],
    },
}

def resolve_curve_data(curve_preset, curve_data_value):
    has_curve_data = False
    parsed_curve_data = None
    if isinstance(curve_data_value, str):
        try:
            parsed_curve_data = json.loads(curve_data_value)
        except:
            parsed_curve_data = None
    elif isinstance(curve_data_value, dict):
        parsed_curve_data = curve_data_value
    if isinstance(parsed_curve_data, dict):
        has_curve_data = any(k in parsed_curve_data for k in ("RGB", "R", "G", "B"))
    if curve_preset != "Custom" and not has_curve_data:
        preset = CURVE_PRESETS.get(curve_preset)
        if preset is not None:
            return json.dumps(preset)
    if isinstance(parsed_curve_data, dict):
        return json.dumps(parsed_curve_data)
    if isinstance(curve_data_value, str):
        return curve_data_value
    return "{}"

def compute_curve_logic(image, curve_data_str, saturation, preview_mode=False):

    if preview_mode:
        h, w = int(image.shape[1]), int(image.shape[2])
        min_side = min(h, w)
        if min_side < 600:
            scale = 600.0 / float(min_side)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            image = torch.nn.functional.interpolate(
                image.permute(0, 3, 1, 2),
                size=(new_h, new_w),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
    
    try:
        data = json.loads(curve_data_str)
    except:
        data = {"RGB":[[0.0,0.0],[1.0,1.0]],"R":[[0.0,0.0],[1.0,1.0]],"G":[[0.0,0.0],[1.0,1.0]],"B":[[0.0,0.0],[1.0,1.0]]}
    
    x_eval = np.linspace(0, 1, 256)
    
    def get_lut(points):
        pts = np.array(points)
        if len(pts) < 2:
            return x_eval
        pts = pts[pts[:, 0].argsort()]
        x = pts[:, 0]
        y = pts[:, 1]
        if x[0] > 0:
            x = np.insert(x, 0, 0.0)
            y = np.insert(y, 0, y[0])
        if x[-1] < 1:
            x = np.append(x, 1.0)
            y = np.append(y, y[-1])
        interpolator = PchipInterpolator(x, y)
        y_eval = interpolator(x_eval)
        return np.clip(y_eval, 0.0, 1.0)
    
    lut_rgb = get_lut(data.get("RGB", [[0, 0], [1, 1]]))
    lut_r = get_lut(data.get("R", [[0, 0], [1, 1]]))
    lut_g = get_lut(data.get("G", [[0, 0], [1, 1]]))
    lut_b = get_lut(data.get("B", [[0, 0], [1, 1]]))
    
    final_r = np.clip(np.interp(lut_rgb, x_eval, lut_r), 0, 1)
    final_g = np.clip(np.interp(lut_rgb, x_eval, lut_g), 0, 1)
    final_b = np.clip(np.interp(lut_rgb, x_eval, lut_b), 0, 1)
    
    device = image.device
    lut_tensor = torch.tensor(np.stack([final_r, final_g, final_b]), dtype=image.dtype, device=device)
    
    out_images = []
    for img in image:
        out_img = img.clone()
        idx = (out_img[..., :3] * 255).clamp(0, 255).to(torch.int64)
        
        out_img[..., 0] = lut_tensor[0][idx[..., 0]]
        out_img[..., 1] = lut_tensor[1][idx[..., 1]]
        out_img[..., 2] = lut_tensor[2][idx[..., 2]]
        
        if saturation != 1.0:
            L = 0.299 * out_img[..., 0] + 0.587 * out_img[..., 1] + 0.114 * out_img[..., 2]
            L = L.unsqueeze(-1)
            out_img[..., :3] = L + (out_img[..., :3] - L) * saturation
            out_img[..., :3] = out_img[..., :3].clamp(0.0, 1.0)
            
        out_images.append(out_img)

    return torch.stack(out_images)




class color_ImageCurve:
    @classmethod
    def INPUT_TYPES(cls):
        preset_names = ["Custom"] + list(CURVE_PRESETS.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "curve_preset": (preset_names, {"default": "Custom"}),
                "curve_data": ("STRING", {"default": '{"RGB":[[0.0,0.0],[1.0,1.0]],"R":[[0.0,0.0],[1.0,1.0]],"G":[[0.0,0.0],[1.0,1.0]],"B":[[0.0,0.0],[1.0,1.0]]}'}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_curve"
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True

    def apply_curve(self, image, curve_preset, curve_data, saturation, unique_id):
        image_to_process = image[0:1]
        GLOBAL_IMAGE_CACHE[unique_id] = {
            "image": image_to_process.cpu()
        }

        curve_data_to_use = resolve_curve_data(curve_preset, curve_data)

        out_tensor = compute_curve_logic(image_to_process, curve_data_to_use, saturation, preview_mode=False)
        
        results = []
        temp_dir = folder_paths.get_temp_directory()
        
        if out_tensor.shape[0] > 0:
            t = out_tensor[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"curve_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        return { "ui": { "bg_image": results }, "result": (out_tensor,) }


@PromptServer.instance.routes.post("/color_image_curve/live_preview")
async def live_preview(request):
    data = await request.json()
    unique_id = data.get("node_id")
    
    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)
        
    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    image = cache_data["image"]
    
    curve_data_str = data.get("curve_data", "{}")
    curve_preset = data.get("curve_preset", "Custom")
    curve_data_str = resolve_curve_data(curve_preset, curve_data_str)

    saturation = data.get("saturation", 1.0)
    
    out_tensor = compute_curve_logic(image, curve_data_str, float(saturation), preview_mode=True)
    
    temp_dir = folder_paths.get_temp_directory()
    img_np = (255.0 * out_tensor[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    
    filename = f"curve_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)
    
    return web.json_response({
        "filename": filename,
        "subfolder": "",
        "type": "temp"
    })

#endregion--------------------------------color-curve--------------------------------------






#region-------------------------------gradient--------------------------------------

def _maybe_upscale_preview_image(image):
    h, w = int(image.shape[1]), int(image.shape[2])
    min_side = min(h, w)
    if min_side < 600:
        scale = 600.0 / float(min_side)
        new_h = int(round(h * scale))
        new_w = int(round(w * scale))
        return torch.nn.functional.interpolate(
            image.permute(0, 3, 1, 2),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).permute(0, 2, 3, 1)
    return image


def _hex_to_rgb_255(hex_color):
    hex_color = str(hex_color or "").lstrip("#")
    if len(hex_color) != 6:
        return (255, 255, 255)
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (r, g, b)
    except:
        return (255, 255, 255)


def _apply_falloff_curve(norm_falloff, mode):
    if mode == "ease_out":
        return 1 - (1 - norm_falloff) ** 2
    if mode == "ease_in":
        return norm_falloff**2
    return norm_falloff


def compute_radia_bright_gradient_logic(
    image,
    center_x,
    center_y,
    circle_radius,
    center_bright,
    edge_bright,
    overlay_color,
    center_alpha,
    edge_alpha,
    falloff_mode,
    soft_edge,
    preview_mode=False,
):
    if preview_mode:
        image = _maybe_upscale_preview_image(image)

    batch_size, h, w, c = image.shape
    device = image.device
    max_side = max(w, h)

    cx = float(center_x) * (w - 1)
    cy = float(center_y) * (h - 1)
    radius_px = float(circle_radius) * max_side

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    coords = torch.stack([x_coords, y_coords], dim=-1).float()

    distance = torch.sqrt((coords[..., 0] - cx) ** 2 + (coords[..., 1] - cy) ** 2)

    if bool(soft_edge):
        soft_edge_px = max(1, int(radius_px * 0.05))
        in_circle_soft = torch.clamp((radius_px + soft_edge_px - distance) / (2 * soft_edge_px), 0.0, 1.0)
    else:
        in_circle_soft = (distance <= radius_px).float()

    falloff_distance = distance - radius_px
    max_falloff_distance = max_side - radius_px
    max_falloff_distance = max(1e-6, max_falloff_distance)

    norm_falloff = torch.clamp(falloff_distance / max_falloff_distance, 0.0, 1.0)
    norm_falloff = _apply_falloff_curve(norm_falloff, str(falloff_mode))

    center_bright = float(center_bright)
    edge_bright = float(edge_bright)
    bright_overlay = center_bright * (1 - norm_falloff) + edge_bright * norm_falloff
    brightness_map = center_bright * in_circle_soft + bright_overlay * (1 - in_circle_soft)
    brightness_map = brightness_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, h, w, c)

    result = image * brightness_map

    rgb_255 = _hex_to_rgb_255(overlay_color)
    rgb_norm = torch.tensor(rgb_255, device=device, dtype=torch.float32) / 255.0
    color_layer = rgb_norm.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(batch_size, h, w, 3)

    center_alpha = float(center_alpha)
    edge_alpha = float(edge_alpha)
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

    return torch.clamp(result, 0.0, 1.0)


def _compute_linear_gradient_map(coords, p0, p1, val0, val1, device):
    p0 = torch.tensor(p0, device=device, dtype=torch.float32)
    p1 = torch.tensor(p1, device=device, dtype=torch.float32)

    dir_vec = p1 - p0
    dir_len = torch.norm(dir_vec)

    if dir_len < 1e-6:
        return torch.full((coords.shape[0], coords.shape[1]), float(val0), device=device, dtype=torch.float32)

    coords_centered = coords - p0
    coords_flat = coords_centered.reshape(-1, 2)

    proj = torch.matmul(coords_flat, dir_vec.unsqueeze(-1)).squeeze(-1) / dir_len
    proj = proj.reshape(coords.shape[0], coords.shape[1])

    proj_norm = torch.clamp(proj / dir_len, 0.0, 1.0)
    val0 = float(val0)
    val1 = float(val1)
    return val0 * (1 - proj_norm) + val1 * proj_norm


def compute_bright_gradient_logic(
    image,
    start_x,
    start_y,
    start_bright,
    end_x,
    end_y,
    end_bright,
    overlay_color,
    start_alpha,
    end_alpha,
    preview_mode=False,
):
    if preview_mode:
        image = _maybe_upscale_preview_image(image)

    batch_size, h, w, c = image.shape
    device = image.device

    start_x_px = int(float(start_x) * (w - 1))
    start_y_px = int(float(start_y) * (h - 1))
    end_x_px = int(float(end_x) * (w - 1))
    end_y_px = int(float(end_y) * (h - 1))

    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    coords = torch.stack([x_coords, y_coords], dim=-1).float()

    brightness_map = _compute_linear_gradient_map(
        coords=coords,
        p0=(start_x_px, start_y_px),
        p1=(end_x_px, end_y_px),
        val0=start_bright,
        val1=end_bright,
        device=device,
    )
    alpha_overlay_map = _compute_linear_gradient_map(
        coords=coords,
        p0=(start_x_px, start_y_px),
        p1=(end_x_px, end_y_px),
        val0=start_alpha,
        val1=end_alpha,
        device=device,
    )

    brightness_map = brightness_map.unsqueeze(0).unsqueeze(-1).expand(batch_size, h, w, c)
    result = image * brightness_map

    rgb_255 = _hex_to_rgb_255(overlay_color)
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

    return torch.clamp(result, 0.0, 1.0)





class color_RadiaGradient_visual:
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("radial_gradient_image",)
    FUNCTION = "apply_radial_gradient"
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True

    def apply_radial_gradient(
        self,
        image,
        center_x,
        center_y,
        circle_radius,
        center_bright,
        edge_bright,
        overlay_color,
        center_alpha,
        edge_alpha,
        falloff_mode,
        soft_edge,
        unique_id,
    ):
        image_to_cache = image[0:1]
        GLOBAL_IMAGE_CACHE[str(unique_id)] = {"image": image_to_cache.cpu()}

        result = compute_radia_bright_gradient_logic(
            image=image,
            center_x=center_x,
            center_y=center_y,
            circle_radius=circle_radius,
            center_bright=center_bright,
            edge_bright=edge_bright,
            overlay_color=overlay_color,
            center_alpha=center_alpha,
            edge_alpha=edge_alpha,
            falloff_mode=falloff_mode,
            soft_edge=soft_edge,
            preview_mode=False,
        )

        results = []
        temp_dir = folder_paths.get_temp_directory()
        if result.shape[0] > 0:
            t = result[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"radial_gradient_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        return {"ui": {"bg_image": results}, "result": (result,)}


@PromptServer.instance.routes.post("/color_radia_bright_gradient/live_preview")
async def live_preview_radia_bright_gradient(request):
    data = await request.json()
    unique_id = str(data.get("node_id"))

    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)

    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    image = cache_data["image"]

    out_tensor = compute_radia_bright_gradient_logic(
        image=image,
        center_x=data.get("center_x", 0.5),
        center_y=data.get("center_y", 0.5),
        circle_radius=data.get("circle_radius", 0.2),
        center_bright=data.get("center_bright", 1.5),
        edge_bright=data.get("edge_bright", 1.0),
        overlay_color=data.get("overlay_color", "#FFFFFF"),
        center_alpha=data.get("center_alpha", 0.0),
        edge_alpha=data.get("edge_alpha", 0.0),
        falloff_mode=data.get("falloff_mode", "linear"),
        soft_edge=data.get("soft_edge", True),
        preview_mode=True,
    )

    temp_dir = folder_paths.get_temp_directory()
    img_np = (255.0 * out_tensor[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    filename = f"radial_gradient_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)

    return web.json_response({"filename": filename, "subfolder": "", "type": "temp"})


@PromptServer.instance.routes.post("/color_adjust_hsl/live_preview")
async def live_preview_color_adjust_hsl(request):
    data = await request.json()
    unique_id = str(data.get("node_id"))
    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)
    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    image = cache_data.get("image", None)
    if image is None:
        return web.json_response({"error": "No image cached"}, status=400)

    out_tensor = color_adjust_HSL().color_adjust_HSL(
        image=image,
        hue=data.get("hue", 0.0),
        brightness=data.get("brightness", 0.0),
        contrast=data.get("contrast", 1.0),
        saturation=data.get("saturation", 1.0),
        sharpness=data.get("sharpness", 1.0),
        blur=data.get("blur", 0),
        gaussian_blur=data.get("gaussian_blur", 0.0),
        edge_enhance=data.get("edge_enhance", 0.0),
        detail_enhance=data.get("detail_enhance", "false"),
        unique_id=unique_id,
        save_preview=False,
        return_ui=False,
    )

    temp_dir = folder_paths.get_temp_directory()
    t = out_tensor[0] if isinstance(out_tensor, torch.Tensor) and out_tensor.dim() == 4 else out_tensor
    img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    filename = f"hsl_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)
    return web.json_response({"filename": filename, "subfolder": "", "type": "temp"})


@PromptServer.instance.routes.post("/color_adjust_hdr/live_preview")
async def live_preview_color_adjust_hdr(request):
    data = await request.json()
    unique_id = str(data.get("node_id"))
    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)
    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    image = cache_data.get("image", None)
    if image is None:
        return web.json_response({"error": "No image cached"}, status=400)

    out_tensor = color_adjust_HDR().execute(
        image=image,
        HDR_intensity=data.get("HDR_intensity", 1.0),
        underexposure_factor=data.get("underexposure_factor", 0.8),
        overexposure_factor=data.get("overexposure_factor", 1.0),
        gamma=data.get("gamma", 0.9),
        highlight_detail=data.get("highlight_detail", 1 / 30.0),
        midtone_detail=data.get("midtone_detail", 0.25),
        shadow_detail=data.get("shadow_detail", 2.0),
        overall_intensity=data.get("overall_intensity", 0.5),
        unique_id=unique_id,
        save_preview=False,
        return_ui=False,
    )

    temp_dir = folder_paths.get_temp_directory()
    t = out_tensor[0] if isinstance(out_tensor, torch.Tensor) and out_tensor.dim() == 4 else out_tensor
    img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    filename = f"hdr_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)
    return web.json_response({"filename": filename, "subfolder": "", "type": "temp"})


@PromptServer.instance.routes.post("/color_match_adv/live_preview")
async def live_preview_color_match_adv(request):
    data = await request.json()
    unique_id = str(data.get("node_id"))
    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)
    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    ref_img = cache_data.get("ref_img", None)
    target_image = cache_data.get("target_image", None)
    target_mask = cache_data.get("target_mask", None)
    if ref_img is None or target_image is None:
        return web.json_response({"error": "No image cached"}, status=400)

    out_tensor = color_match_adv().match_hue(
        ref_img=ref_img,
        target_image=target_image,
        target_mask=target_mask,
        strength=data.get("strength", 1.0),
        skin_protection=data.get("skin_protection", 0.2),
        brightness_range=data.get("brightness_range", 0.5),
        contrast_range=data.get("contrast_range", 0.5),
        saturation_range=data.get("saturation_range", 0.5),
        tone_strength=data.get("tone_strength", 0.5),
        unique_id=unique_id,
        save_preview=False,
        return_ui=False,
    )

    temp_dir = folder_paths.get_temp_directory()
    t = out_tensor[0] if isinstance(out_tensor, torch.Tensor) and out_tensor.dim() == 4 else out_tensor
    img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    filename = f"match_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)
    return web.json_response({"filename": filename, "subfolder": "", "type": "temp"})


@PromptServer.instance.routes.post("/image_cnmapmix/live_preview")
async def live_preview_image_cnmapmix(request):
    data = await request.json()
    unique_id = str(data.get("node_id"))
    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)
    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    image1 = cache_data.get("image1", None)
    image2 = cache_data.get("image2", None)
    mask2 = cache_data.get("mask2", None)
    if image1 is None and image2 is None:
        return web.json_response({"error": "No image cached"}, status=400)

    out_tensor = Image_CnMapMix().fuse_depth(
        bg_color=data.get("bg_color", "image"),
        blur_1=data.get("blur_1", 0),
        blur_2=data.get("blur_2", 0),
        diff_sensitivity=data.get("diff_sensitivity", 0.0),
        diff_blur=data.get("diff_blur", 0),
        blend_mode=data.get("blend_mode", "normal"),
        blend_factor=data.get("blend_factor", 0.5),
        contrast=data.get("contrast", 1.0),
        brightness=data.get("brightness", 0.0),
        mask2_smoothness=data.get("mask2_smoothness", 0),
        invert_mask=data.get("invert_mask", False),
        image1_min_black=data.get("image1_min_black", 0.0),
        image1_max_white=data.get("image1_max_white", 1.0),
        image2_min_black=data.get("image2_min_black", 0.0),
        image2_max_white=data.get("image2_max_white", 1.0),
        image1=image1,
        image2=image2,
        mask2=mask2,
        unique_id=unique_id,
        save_preview=False,
        return_ui=False,
    )

    temp_dir = folder_paths.get_temp_directory()
    t = out_tensor[0] if isinstance(out_tensor, torch.Tensor) and out_tensor.dim() == 4 else out_tensor
    img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)
    filename = f"cnmapmix_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)
    return web.json_response({"filename": filename, "subfolder": "", "type": "temp"})




class color_lineGradient_visual:
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
            },
            "hidden": {
                "unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("gradient_image",)
    FUNCTION = "apply_gradient"
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True

    def apply_gradient(
        self,
        image,
        start_x,
        start_y,
        start_bright,
        end_x,
        end_y,
        end_bright,
        overlay_color,
        start_alpha,
        end_alpha,
        unique_id,
    ):
        image_to_cache = image[0:1]
        GLOBAL_IMAGE_CACHE[str(unique_id)] = {"image": image_to_cache.cpu()}

        result = compute_bright_gradient_logic(
            image=image,
            start_x=start_x,
            start_y=start_y,
            start_bright=start_bright,
            end_x=end_x,
            end_y=end_y,
            end_bright=end_bright,
            overlay_color=overlay_color,
            start_alpha=start_alpha,
            end_alpha=end_alpha,
            preview_mode=False,
        )

        results = []
        temp_dir = folder_paths.get_temp_directory()
        if result.shape[0] > 0:
            t = result[0]
            img_np = (255.0 * t.cpu().numpy()).clip(0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filename = f"bright_gradient_preview_{random.randint(1, 1000000)}.png"
            img_pil.save(os.path.join(temp_dir, filename))
            results.append({"filename": filename, "subfolder": "", "type": "temp"})

        return {"ui": {"bg_image": results}, "result": (result,)}



@PromptServer.instance.routes.post("/color_bright_gradient/live_preview")
async def live_preview_bright_gradient(request):
    data = await request.json()
    unique_id = str(data.get("node_id"))

    if unique_id not in GLOBAL_IMAGE_CACHE:
        return web.json_response({"error": "No image cached"}, status=400)

    cache_data = GLOBAL_IMAGE_CACHE[unique_id]
    image = cache_data["image"]

    out_tensor = compute_bright_gradient_logic(
        image=image,
        start_x=data.get("start_x", 0.0),
        start_y=data.get("start_y", 0.0),
        start_bright=data.get("start_bright", 1.0),
        end_x=data.get("end_x", 1.0),
        end_y=data.get("end_y", 1.0),
        end_bright=data.get("end_bright", 1.0),
        overlay_color=data.get("overlay_color", "#FFFFFF"),
        start_alpha=data.get("start_alpha", 0.0),
        end_alpha=data.get("end_alpha", 0.0),
        preview_mode=True,
    )

    temp_dir = folder_paths.get_temp_directory()
    img_np = (255.0 * out_tensor[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
    img_pil = Image.fromarray(img_np)

    filename = f"bright_gradient_live_{unique_id}.png"
    filepath = os.path.join(temp_dir, filename)
    img_pil.save(filepath)

    return web.json_response({"filename": filename, "subfolder": "", "type": "temp"})


#endregion---------------------------------------gradient---------------------------------------    


def _crop_visual_parse_state(crop_state):
    if isinstance(crop_state, dict):
        data = crop_state
    else:
        try:
            data = json.loads(crop_state) if crop_state else {}
        except:
            data = {}
    cx = float(data.get("cx", 0.5))
    cy = float(data.get("cy", 0.5))
    zoom = float(data.get("zoom", 1.0))
    cx = min(1.0, max(0.0, cx))
    cy = min(1.0, max(0.0, cy))
    zoom = max(1e-4, zoom)
    return cx, cy, zoom


def _crop_visual_compute_box(img_w, img_h, crop_w, crop_h, center_x, center_y, zoom):
    src_w = min(float(img_w), max(1.0, float(crop_w) / float(zoom)))
    src_h = min(float(img_h), max(1.0, float(crop_h) / float(zoom)))
    cx_px = min(float(img_w), max(0.0, float(center_x) * float(img_w)))
    cy_px = min(float(img_h), max(0.0, float(center_y) * float(img_h)))
    half_w = src_w * 0.5
    half_h = src_h * 0.5
    cx_px = min(float(img_w) - half_w, max(half_w, cx_px))
    cy_px = min(float(img_h) - half_h, max(half_h, cy_px))
    left = cx_px - half_w
    top = cy_px - half_h
    right = left + src_w
    bottom = top + src_h
    return left, top, right, bottom


def _crop_visual_apply_single(img, crop_w, crop_h, center_x, center_y, zoom):
    h = int(img.shape[0])
    w = int(img.shape[1])
    left, top, right, bottom = _crop_visual_compute_box(w, h, crop_w, crop_h, center_x, center_y, zoom)
    x = torch.linspace(left, right, crop_w, device=img.device, dtype=img.dtype)
    y = torch.linspace(top, bottom, crop_h, device=img.device, dtype=img.dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    denom_w = float(max(1, w - 1))
    denom_h = float(max(1, h - 1))
    grid_x = (xx / denom_w) * 2.0 - 1.0
    grid_y = (yy / denom_h) * 2.0 - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    img_chw = img.permute(2, 0, 1).unsqueeze(0)
    out = torch.nn.functional.grid_sample(
        img_chw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out[0].permute(1, 2, 0).clamp(0.0, 1.0)


def _crop_visual_prepare_image(img, fill, margin):
    fill_mode = str(fill or "none").lower()
    if fill_mode == "none":
        return img
    h = int(img.shape[0])
    w = int(img.shape[1])
    side = max(h, w) + 2 * margin
    if side <= 0:
        return img
    if fill_mode == "white":
        val = 1.0
    elif fill_mode == "black":
        val = 0.0
    elif fill_mode == "grey":
        val = 0.5
    elif fill_mode == "edge":
        c = int(img.shape[2])
        canvas = torch.zeros((side, side, c), dtype=img.dtype, device=img.device)
        top = (side - h) // 2
        left = (side - w) // 2
        bottom_pad = side - (top + h)
        right_pad = side - (left + w)
        canvas[top:top + h, left:left + w, :] = img
        if top > 0:
            top_row = img[0:1, :, :].expand(top, w, c)
            canvas[0:top, left:left + w, :] = top_row
        if bottom_pad > 0:
            bottom_row = img[h - 1:h, :, :].expand(bottom_pad, w, c)
            canvas[top + h:side, left:left + w, :] = bottom_row
        if left > 0:
            left_col = img[:, 0:1, :].expand(h, left, c)
            canvas[top:top + h, 0:left, :] = left_col
        if right_pad > 0:
            right_col = img[:, w - 1:w, :].expand(h, right_pad, c)
            canvas[top + h:side, left + w:side, :] = right_col
        if top > 0 and left > 0:
            canvas[0:top, 0:left, :] = img[0, 0, :].view(1, 1, c)
        if top > 0 and right_pad > 0:
            canvas[0:top, left + w:side, :] = img[0, w - 1, :].view(1, 1, c)
        if bottom_pad > 0 and left > 0:
            canvas[top + h:side, 0:left, :] = img[h - 1, 0, :].view(1, 1, c)
        if bottom_pad > 0 and right_pad > 0:
            canvas[top + h:side, left + w:side, :] = img[h - 1, w - 1, :].view(1, 1, c)
        return canvas
    else:
        return img
    c = int(img.shape[2])
    canvas = torch.full((side, side, c), float(val), dtype=img.dtype, device=img.device)
    top = (side - h) // 2
    left = (side - w) // 2
    canvas[top:top + h, left:left + w, :] = img
    return canvas


class Image_crop_visual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "crop_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "fill": (["none", "white", "black", "grey", "edge"], {"default": "none"}),
                "margin": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1}),
                "crop_state": ("STRING", {"default": '{"cx":0.5,"cy":0.5,"zoom":1.0}'}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("cropped_image",)
    FUNCTION = "crop_image"
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True
    DESCRIPTION = """滚动鼠标，缩放裁剪框"""
    def crop_image(self, image, crop_width, crop_height, fill, margin, crop_state):
        crop_w = int(max(1, crop_width))
        crop_h = int(max(1, crop_height))
        cx, cy, zoom = _crop_visual_parse_state(crop_state)

        out_list = []
        image_for_preview = image[0]
        image_for_meta = image[0]
        for img in image:
            prepared = _crop_visual_prepare_image(img, fill, margin)
            out_list.append(_crop_visual_apply_single(prepared, crop_w, crop_h, cx, cy, zoom))
        if image.shape[0] > 0:
            image_for_preview = _crop_visual_prepare_image(image[0], fill, margin)
            image_for_meta = image_for_preview
        out_tensor = torch.stack(out_list, dim=0)

        bg_results = []
        temp_dir = folder_paths.get_temp_directory()
        if image.shape[0] > 0:
            src_np = (255.0 * image_for_preview.cpu().numpy()).clip(0, 255).astype(np.uint8)
            src_pil = Image.fromarray(src_np)
            filename = f"image_crop_visual_bg_{random.randint(1, 1000000)}.png"
            src_pil.save(os.path.join(temp_dir, filename))
            bg_results.append({"filename": filename, "subfolder": "", "type": "temp"})

        ui = {
            "bg_image": bg_results,
            "crop_meta": [{"img_w": int(image_for_meta.shape[1]), "img_h": int(image_for_meta.shape[0])}],
        }
        return {"ui": ui, "result": (out_tensor,)}


class Image_mask_crop_visual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "crop_width": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "crop_height": ("INT", {"default": 512, "min": 1, "max": 8192, "step": 1}),
                "crop_img_bj": (["image", "white", "black", "red", "green", "blue", "yellow", "cyan", "magenta", "gray"], {"default": "image"}),
                "crop_state": ("STRING", {"default": '{"cx":0.5,"cy":0.5,"zoom":1.0}'}),
            },
            "optional": {
                "mask_stack": ("MASK_STACK2",),
            }
        }

    CATEGORY = "Apt_Preset/image/visualize_edit"
    RETURN_TYPES = ("IMAGE", "MASK", "STITCH2")
    RETURN_NAMES = ("crop_image", "crop_mask", "stitch")
    FUNCTION = "crop_image"
    OUTPUT_NODE = True
    DESCRIPTION = """基于遮罩的可视化裁剪工具
    - 红框：仅代表宽高比例 + 位置，不直接决定输出像素尺寸
    - 输出尺寸：严格等于用户输入的「裁剪宽度 × 裁剪高度」
    - 遮罩最小外接矩形：必须被红框完全包裹
    - 红框比例：始终与「裁剪宽度：裁剪高度」保持一致
    """

    def get_mask_bounding_box(self, mask):
        mask_np = mask[0].cpu().numpy()
        mask_np = np.squeeze(mask_np)
        mask_np = (mask_np > 0.5).astype(np.uint8)
        if mask_np.ndim != 2:
            raise ValueError(f"Mask must be 2D array, got {mask_np.ndim}D instead")
        coords = cv2.findNonZero(mask_np)
        if coords is None:
            # 如果遮罩为空，返回全图
            h, w = mask_np.shape
            return w, h, 0, 0
        x, y, w, h = cv2.boundingRect(coords)
        return w, h, x, y

    def crop_image(self, image, mask, crop_width, crop_height, crop_img_bj, crop_state, mask_stack=None):
        crop_w = int(max(1, crop_width))
        crop_h = int(max(1, crop_height))
        cx, cy, zoom = _crop_visual_parse_state(crop_state)

        # 处理遮罩
        batch_size, height, width, _ = image.shape
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

        # 获取遮罩边界框
        mask_w, mask_h, mask_x, mask_y = self.get_mask_bounding_box(processed_mask)

        # 准备图像和遮罩
        out_list = []
        mask_out_list = []
        image_for_preview = image[0]
        image_for_meta = image[0]
        mask_for_meta = processed_mask[0]

        for img, msk in zip(image, processed_mask):
            # 应用裁剪
            prepared = img
            cropped_img = _crop_visual_apply_single(prepared, crop_w, crop_h, cx, cy, zoom)
            cropped_mask = _crop_visual_apply_single(msk.unsqueeze(2), crop_w, crop_h, cx, cy, zoom).squeeze(2)
            out_list.append(cropped_img)
            mask_out_list.append(cropped_mask)

        if image.shape[0] > 0:
            image_for_preview = image[0]
            image_for_meta = image[0]
            mask_for_meta = processed_mask[0]

        out_tensor = torch.stack(out_list, dim=0)
        mask_out_tensor = torch.stack(mask_out_list, dim=0)

        # 处理背景
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

        if crop_img_bj != "image" and crop_img_bj in colors:
            r, g, b = colors[crop_img_bj]
            h_bg, w_bg, _ = out_tensor.shape[1:]
            background = torch.zeros((out_tensor.shape[0], h_bg, w_bg, 3), device=out_tensor.device)
            background[:, :, :, 0] = r
            background[:, :, :, 1] = g
            background[:, :, :, 2] = b
            if out_tensor.shape[3] >= 4:
                alpha = out_tensor[:, :, :, 3].unsqueeze(3)
                image_rgb = out_tensor[:, :, :, :3]
                out_tensor = image_rgb * alpha + background * (1 - alpha)
            else:
                alpha = mask_out_tensor.unsqueeze(3)
                image_rgb = out_tensor[:, :, :, :3]
                out_tensor = image_rgb * alpha + background * (1 - alpha)

        # 生成预览图像
        bg_results = []
        temp_dir = folder_paths.get_temp_directory()
        if image.shape[0] > 0:
            src_np = (255.0 * image_for_preview.cpu().numpy()).clip(0, 255).astype(np.uint8)
            src_pil = Image.fromarray(src_np)
            filename = f"image_mask_crop_visual_bg_{random.randint(1, 1000000)}.png"
            src_pil.save(os.path.join(temp_dir, filename))
            bg_results.append({"filename": filename, "subfolder": "", "type": "temp"})

        # 构建 stitch 信息
        original_h, original_w = image.shape[1], image.shape[2]
        
        # 计算裁剪后的遮罩位置（在裁剪后的图像中的位置）
        # 红框在原始图像上的位置和尺寸
        # zoom 是从 crop_state 解析的，表示缩放比例
        # zoom 越大，红框在源图像上的尺寸越小；zoom 越小，红框越大
        src_rect = {
            'x': int(cx * original_w - crop_w / zoom * 0.5),
            'y': int(cy * original_h - crop_h / zoom * 0.5),
            'w': int(crop_w / zoom),
            'h': int(crop_h / zoom)
        }
        
        # 遮罩在裁剪后图像中的位置
        mask_crop_x_start = max(0, mask_x - src_rect['x'])
        mask_crop_y_start = max(0, mask_y - src_rect['y'])
        mask_crop_x_end = min(src_rect['w'], mask_x + mask_w - src_rect['x'])
        mask_crop_y_end = min(src_rect['h'], mask_y + mask_h - src_rect['y'])
        
        # 创建背景图像（用于 stitch 恢复）
        bj_image = image.clone()
        
        stitch = {
            "original_shape": (original_h, original_w),
            "original_image_shape": (original_h, original_w),
            "crop_position": (src_rect['x'], src_rect['y']),
            "crop_size": (src_rect['w'], src_rect['h']),
            "expand_width": 0,
            "expand_height": 0,
            "auto_expand_square": False,
            "expanded_region": (src_rect['x'], src_rect['y'], src_rect['x'] + src_rect['w'], src_rect['y'] + src_rect['h']),
            "mask_original_position": (mask_x, mask_y, mask_w, mask_h),
            "mask_cropped_position": (mask_crop_x_start, mask_crop_y_start, mask_crop_x_end, mask_crop_y_end),
            "original_long_side": max(original_w, original_h),
            "crop_long_side": max(src_rect['w'], src_rect['h']),
            "input_long_side": max(crop_w, crop_h),
            "false_long_side": max(src_rect['w'], src_rect['h']),
            "bj_image": bj_image,
            "original_image": image,
            "crop_state": crop_state,
            "mask_bounding_box": (mask_x, mask_y, mask_w, mask_h),
            "crop_img_bj": crop_img_bj,
        }

        ui = {
            "bg_image": bg_results,
            "crop_meta": [{"img_w": int(image_for_meta.shape[1]), "img_h": int(image_for_meta.shape[0])}],
            "mask_meta": [{"mask_x": mask_x, "mask_y": mask_y, "mask_w": mask_w, "mask_h": mask_h}],
        }
        return {"ui": ui, "result": (out_tensor, mask_out_tensor, stitch)}















class Image_CnMapMix_visual(Image_CnMapMix):
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True



class color_adjust_HSL_visual(color_adjust_HSL):
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True




class color_adjust_HDR_visual(color_adjust_HDR):
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True




class color_match_adv_visual(color_match_adv):
    CATEGORY = "Apt_Preset/image/visualize_edit"
    OUTPUT_NODE = True



#endregion---------------------------------------visualize---------------------------------------















