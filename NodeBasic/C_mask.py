
#region---------
import torch
import torch.nn.functional as F
import numpy as np
import io
import json

import folder_paths as comfy_paths
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageChops
import os,sys
import comfy.utils

from PIL import Image, ImageOps, ImageFilter
import math
from torchvision.transforms import functional as TF
import folder_paths


import random

from tqdm import tqdm

from ..main_unit import *
from ..office_unit import GrowMask

#endregion-------------------------------------------------------------------------------#

#---------------------安全导入------
try:
    import scipy.ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    scipy_ndimage = None
    SCIPY_AVAILABLE = False
#---------------------安全导入------


try:
    import cv2
    REMOVER_AVAILABLE = True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False



class Mask_math:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1":("MASK",),
                "mask2":("MASK",),
                "operation":(["-","+","*","&"],{"default": "+"}),
                "algorithm":(["cv2","torch"],{"default":"cv2"}),
                "invert_mask1":("BOOLEAN",{"default":False}),
                "invert_mask2":("BOOLEAN",{"default":False}),
            }
        }
    CATEGORY = "Apt_Preset/mask/😺backup"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "mask_math"
    def mask_math(self, mask1, mask2, operation, algorithm, invert_mask1, invert_mask2):
        #invert mask
        if invert_mask1:
            mask1 = 1-mask1
        if invert_mask2:
            mask2 = 1-mask2

        #repeat mask
        if mask1.dim() == 2:
            mask1 = mask1.unsqueeze(0)
        if mask2.dim() == 2:
            mask2 = mask2.unsqueeze(0)
        if mask1.shape[0] == 1 and mask2.shape[0] != 1:
            mask1 = mask1.repeat(mask2.shape[0],1,1)
        elif mask1.shape[0] != 1 and mask2.shape[0] == 1:
            mask2 = mask2.repeat(mask1.shape[0],1,1)

        #check cv2
        if algorithm == "cv2":
            try:
                import cv2
            except:
                print("prompt-mask_and_mask_math: cv2 is not installed, Using Torch")
                print("prompt-mask_and_mask_math: cv2 未安装, 使用torch")
                algorithm = "torch"

        #algorithm
        if algorithm == "cv2":
            if operation == "-":
                return (self.subtract_masks(mask1, mask2),)
            elif operation == "+":
                return (self.add_masks(mask1, mask2),)
            elif operation == "*":
                return (self.multiply_masks(mask1, mask2),)
            elif operation == "&":
                return (self.and_masks(mask1, mask2),)
        elif algorithm == "torch":
            if operation == "-":
                return (torch.clamp(mask1 - mask2, min=0, max=1),)
            elif operation == "+":
                return (torch.clamp(mask1 + mask2, min=0, max=1),)
            elif operation == "*":
                return (torch.clamp(mask1 * mask2, min=0, max=1),)
            elif operation == "&":
                mask1 = torch.round(mask1).bool()
                mask2 = torch.round(mask2).bool()
                return (mask1 & mask2, )

    def subtract_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.subtract(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1

    def add_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.add(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def multiply_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1) * 255
        cv2_mask2 = np.array(mask2) * 255
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.multiply(cv2_mask1, cv2_mask2)
            return torch.clamp(torch.from_numpy(cv2_mask) / 255.0, min=0, max=1)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1
    
    def and_masks(self, mask1, mask2):
        mask1 = mask1.cpu()
        mask2 = mask2.cpu()
        cv2_mask1 = np.array(mask1)
        cv2_mask2 = np.array(mask2)
        import cv2
        if cv2_mask1.shape == cv2_mask2.shape:
            cv2_mask = cv2.bitwise_and(cv2_mask1, cv2_mask2)
            return torch.from_numpy(cv2_mask)
        else:
            # do nothing - incompatible mask shape: mostly empty mask
            print("Warning-mask_math: The two masks have different shapes")
            return mask1





def tensorMask2cv2img(tensor) -> np.ndarray:   
    tensor = tensor.cpu().squeeze(0)
    array = tensor.numpy()
    array = (array * 255).astype(np.uint8)
    return array

class Mask_splitMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks", )
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask/😺backup"

    def separate(self, mask, ignore_threshold=100, index=0):
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmented_masks = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < ignore_threshold:
                continue
            segmented_mask = np.zeros_like(binary_mask)
            cv2.drawContours(segmented_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
            segmented_masks.append(segmented_mask)
        output_masks = []
        for segmented_mask in segmented_masks:
            numpy_mask = np.array(segmented_mask).astype(np.float32) / 255.0
            i_mask = torch.from_numpy(numpy_mask)
            output_masks.append(i_mask.unsqueeze(0))
        mask = output_masks
        if isinstance(mask, list):
            result = mask[index]
        else:
            result = mask
        if result is None:
            result = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
        return (result,)


class Mask_split_mulMask:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("mask1", "mask2", "mask3", "mask4", "rest_mask")
    FUNCTION = "separate"
    CATEGORY = "Apt_Preset/mask"

    def separate(self, mask, ignore_threshold=100):
        opencv_gray_image = tensorMask2cv2img(mask)
        _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 计算每个轮廓的边界框左上角坐标，并排序
        contours_with_positions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            contours_with_positions.append((x, y, contour))
        
        # 排序：先按y坐标，再按x坐标
        contours_with_positions.sort(key=lambda item: (item[1], item[0]))
        sorted_contours = [item[2] for item in contours_with_positions]
        
        # 处理排序后的轮廓
        segmented_masks = []
        remaining_contours = []
        
        for i, contour in enumerate(sorted_contours):
            area = cv2.contourArea(contour)
            if area < ignore_threshold:
                continue
            if i < 4:  # 前4个轮廓分别处理
                segmented_mask = np.zeros_like(binary_mask)
                cv2.drawContours(segmented_mask, [contour], 0, (255, 255, 255), thickness=cv2.FILLED)
                segmented_masks.append(segmented_mask)
            else:  # 第5个及以后的轮廓合并处理
                remaining_contours.append(contour)
        
        # 处理剩余的轮廓（如果有）
        if remaining_contours:
            mask5 = np.zeros_like(binary_mask)
            cv2.drawContours(mask5, remaining_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            segmented_masks.append(mask5)
        
        # 确保总是返回5个掩码
        output_masks = []
        for i in range(5):
            if i < len(segmented_masks):
                numpy_mask = np.array(segmented_masks[i]).astype(np.float32) / 255.0
                i_mask = torch.from_numpy(numpy_mask)
                output_masks.append(i_mask.unsqueeze(0))
            else:
                # 如果不足5个，添加全零掩码
                output_masks.append(torch.zeros((1, *binary_mask.shape), dtype=torch.float32, device="cpu"))
        
        return tuple(output_masks)



class create_AD_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", { "default": 512, "min": 1, "max": 5120, "step": 1, }),
                "height": ("INT", { "default": 512, "min": 1, "max": 5120, "step": 1, }),
                "frames": ("INT", { "default": 16, "min": 1, "max": 9999, "step": 1, }),
                "start_frame": ("INT", { "default": 0, "min": 0, "step": 1, }),
                "end_frame": ("INT", { "default": 9999, "min": 0, "step": 1, }),
                "transition_type": (["horizontal slide", "vertical slide", "horizontal bar", "vertical bar", "center box", "horizontal door", "vertical door", "circle", "fade"],),
                "method": (["linear", "in", "out", "in-out"],),
                
                "invertMask": ("BOOLEAN", {"default": False})
            }
        }

    FUNCTION = "run"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    CATEGORY = "Apt_Preset/mask/😺backup"

    def linear(self, i, t):
        return i/t
    def ease_in(self, i, t):
        return pow(i/t, 2)
    def ease_out(self, i, t):
        return 1 - pow(1 - i/t, 2)
    def ease_in_out(self, i, t):
        if i < t/2:
            return pow(i/(t/2), 2) / 2
        else:
            return 1 - pow(1 - (i - t/2)/(t/2), 2) / 2

    def run(self, width, height, frames, start_frame, end_frame, transition_type, method, invertMask):
        if method == 'in':
            method = self.ease_in
        elif method == 'out':
            method = self.ease_out
        elif method == 'in-out':
            method = self.ease_in_out
        else:
            method = self.linear

        out = []

        end_frame = min(frames, end_frame)
        transition = end_frame - start_frame

        if start_frame > 0:
            out = out + [torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")] * start_frame

        for i in range(transition):
            frame = torch.full((height, width), 0.0, dtype=torch.float32, device="cpu")
            progress = method(i, transition-1)

            if "horizontal slide" in transition_type:
                pos = round(width*progress)
                frame[:, :pos] = 1.0
            elif "vertical slide" in transition_type:
                pos = round(height*progress)
                frame[:pos, :] = 1.0
            elif "box" in transition_type:
                box_w = round(width*progress)
                box_h = round(height*progress)
                x1 = (width - box_w) // 2
                y1 = (height - box_h) // 2
                x2 = x1 + box_w
                y2 = y1 + box_h
                frame[y1:y2, x1:x2] = 1.0
            elif "circle" in transition_type:
                radius = math.ceil(math.sqrt(pow(width,2)+pow(height,2))*progress/2)
                c_x = width // 2
                c_y = height // 2
                # is this real life? Am I hallucinating?
                x = torch.arange(0, width, dtype=torch.float32, device="cpu")
                y = torch.arange(0, height, dtype=torch.float32, device="cpu")
                y, x = torch.meshgrid((y, x), indexing="ij")
                circle = ((x - c_x) ** 2 + (y - c_y) ** 2) <= (radius ** 2)
                frame[circle] = 1.0
            elif "horizontal bar" in transition_type:
                bar = round(height*progress)
                y1 = (height - bar) // 2
                y2 = y1 + bar
                frame[y1:y2, :] = 1.0
            elif "vertical bar" in transition_type:
                bar = round(width*progress)
                x1 = (width - bar) // 2
                x2 = x1 + bar
                frame[:, x1:x2] = 1.0
            elif "horizontal door" in transition_type:
                bar = math.ceil(height*progress/2)
                if bar > 0:
                    frame[:bar, :] = 1.0
                    frame[-bar:, :] = 1.0
            elif "vertical door" in transition_type:
                bar = math.ceil(width*progress/2)
                if bar > 0:
                    frame[:, :bar] = 1.0
                    frame[:, -bar:] = 1.0
            elif "fade" in transition_type:
                frame[:,:] = progress

            out.append(frame)

        if end_frame < frames:
            out = out + [torch.full((height, width), 1.0, dtype=torch.float32, device="cpu")] * (frames - end_frame)

        out = torch.stack(out, dim=0)
        
        if invertMask:
            out = 1.0 - out


        return (out, )




class Mask_image2mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask_type": (["White_Balance", "color", "channel", "depth"],),
                "WB_low_threshold": ("INT", {"default": 1, "min": 1, "max": 255, "step": 1}),
                "WB_high_threshold": ("INT", {"default": 255, "min": 1, "max": 255, "step": 1}),

                "color": ("STRING", {"default": "#000000"}),
                "color_threshold": ("INT", {"default": 10, "min": 0, "max": 255, "step": 1}),
                "channel": (["red", "green", "blue", "alpha"],),
                "depth": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.001, "display": "number"}),


                "blur_radius": ("INT", {"default": 1, "min": 1, "max": 32768, "step": 1}),
                "expand": ("INT", {"default": 0, "min": -150, "max": 150, "step": 1}),

            },
        }

    RETURN_TYPES = ( "MASK", "MASK", )
    RETURN_NAMES = ("mask", "invert_mask", )
    FUNCTION = "image_to_mask"
    CATEGORY = "Apt_Preset/mask"

    def image_to_mask(self, image, WB_low_threshold, WB_high_threshold, blur_radius, mask_type, depth, color, color_threshold, channel, expand):
        
        tapered_corners = True
        
        if isinstance(image, torch.Tensor):
            image_np = image.cpu().numpy()
        else:
            image_np = image

        out_image = image

        if mask_type == "White_Balance":
            image = 255. * image_np[0]
            image = Image.fromarray(np.clip(image, 0, 255).astype(np.uint8))
            image = ImageOps.grayscale(image)
            threshold_filter = lambda x: 255 if x > WB_high_threshold else 0 if x < WB_low_threshold else x
            image = image.convert("L").point(threshold_filter, mode="L")
            image = np.array(image).astype(np.float32) / 255.0
            mask = 1- torch.from_numpy(image)
            #mask2_img = torch.from_numpy(image)[None,]


        elif mask_type == "color":
            images = 255. * image_np
            images = np.clip(images, 0, 255).astype(np.uint8)
            images = [Image.fromarray(img) for img in images]
            images = [np.array(img) for img in images]

            black = [0, 0, 0]
            white = [255, 255, 255]
            new_images = []

            # 将十六进制颜色字符串转换为 RGB 值
            if isinstance(color, str) and color.startswith('#'):
                color = color.lstrip('#')
                rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
                color = np.array(rgb, dtype=np.float32)

            for img in images:
                new_image = np.full_like(img, black)

                color_distances = np.linalg.norm(img - color, axis=-1)
                complement_indexes = color_distances <= color_threshold
                new_image[complement_indexes] = white
                new_images.append(new_image)

            new_images = np.array(new_images).astype(np.float32) / 255.0
            new_images = torch.from_numpy(new_images).permute(3, 0, 1, 2)
            mask = new_images[0]

        elif mask_type == "channel":
            channels = ["red", "green", "blue", "alpha"]
            mask = image[:, :, :, channels.index(channel)]


        elif mask_type == "depth":
            bs = image.size()[0]
            width = image.size()[2]
            height = image.size()[1]
            mask1 = torch.zeros((bs, height, width))
            image = upscale(image, 'lanczos', width, height)[0]
            for k in range(bs):
                for i in range(width):
                    for j in range(height):
                        now_depth = image[k][j][i][0].item()
                        if now_depth < depth:
                            mask1[k][j][i] = 1
            mask = mask1


        if blur_radius > 0:
            mask=tensor2pil(mask)
            feathered_image = mask.filter(ImageFilter.GaussianBlur(blur_radius))
            mask=pil2tensor(feathered_image)


        mask = GrowMask().expand_mask(mask, expand, tapered_corners)[0]
        #mask2_img = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)
        invert_mask = 1.0 - mask
    
        return (mask, invert_mask, )



class Mask_splitMask_by_color:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "threshold_r": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_g": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
                "threshold_b": ("FLOAT", { "default": 0.15, "min": 0.0, "max": 1, "step": 0.01, }),
            }
        }

    RETURN_TYPES = ("MASK","MASK","MASK","MASK","MASK","MASK","MASK","MASK",)
    RETURN_NAMES = ("red","green","blue","cyan","magenta","yellow","black","white",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/mask"

    def execute(self, image, threshold_r, threshold_g, threshold_b):
        red = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] < threshold_b)).float()
        green = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()
        blue = ((image[..., 0] < threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] >= 1-threshold_b)).float()

        cyan = ((image[..., 0] < threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        magenta = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] < threshold_g) & (image[..., 2] > 1-threshold_b)).float()
        yellow = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] < threshold_b)).float()

        black = ((image[..., 0] <= threshold_r) & (image[..., 1] <= threshold_g) & (image[..., 2] <= threshold_b)).float()
        white = ((image[..., 0] >= 1-threshold_r) & (image[..., 1] >= 1-threshold_g) & (image[..., 2] >= 1-threshold_b)).float()
        
        return (red, green, blue, cyan, magenta, yellow, black, white,)





class create_mask_solo:

    @classmethod
    def INPUT_TYPES(s):
        color_options = ["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"]
        
        return {
            "required": {
                "wide": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": 5000, "step": 1}),
                "shape": (
                    [
                        'circle',
                        'square',
                        'rectangle',
                        'semicircle',
                        'quarter_circle',
                        'ellipse',
                        'triangle',
                        'cross',
                        'star',
                        'radial',
                    ],
                    {"default": "circle"},
                ),
                "coord_center": (
                    ['default', 'image_center'],
                    {"default": "default"}
                ),
                "X_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "Y_offset": ("INT", {"default": 0, "min": -10000, "max": 10000, "step": 1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "rotation": ("INT", {"default": 0, "min": 0, "max": 360, "step": 1}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.1}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "background_color": (color_options, {"default": "white"}),
                "shape_color": (color_options, {"default": "black"}),
            },
            "optional": {
                "bg_image": ("IMAGE", {"default": None}),  # 原base_image改名为bg_image
                "get_img_size": ("IMAGE", {"default": None}),  # 新增仅获取尺寸的图像输入
            },
        }

    CATEGORY = "Apt_Preset/mask"

    RETURN_TYPES = ("IMAGE", "MASK", "BOX2")
    RETURN_NAMES = ("image", "mask", "box2")
    FUNCTION = "drew_light_shape"

    def drew_light_shape(self, wide, height, shape, coord_center, X_offset, Y_offset, scale, rotation, opacity, blur_radius, background_color,
                         shape_color, bg_image=None, get_img_size=None):
        
        color_mapping = {
            "white": (255, 255, 255),
            "black": (0, 0, 0),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255)
        }
        
        bg_color_rgb = color_mapping[background_color]
        shape_color_rgb = color_mapping[shape_color]

        # 优先使用bg_image作为背景图，其次使用get_img_size的尺寸，最后使用wide/height
        if bg_image is not None:
            # 使用bg_image作为背景图
            bg_width = bg_image.shape[2]
            bg_height = bg_image.shape[1]
            # 将bg_image转换为PIL图像（保持原图像内容）
            background = Image.fromarray((bg_image.squeeze().cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")
        else:
            # 不使用背景图，确定输出尺寸
            if get_img_size is not None:
                # 从get_img_size获取尺寸，忽略图像内容
                bg_width = get_img_size.shape[2]
                bg_height = get_img_size.shape[1]
            else:
                # 使用默认的wide和height
                bg_width = wide
                bg_height = height
            
            # 创建纯色背景（RGBA格式用于后续合成）
            background = Image.new("RGBA", (bg_width, bg_height), (*bg_color_rgb, 255))

        # 创建形状图层（带透明度）
        shape_img = Image.new("RGBA", (bg_width, bg_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shape_img)
        
        # 创建mask（单通道）
        mask = Image.new("L", (bg_width, bg_height), 0)
        mask_draw = ImageDraw.Draw(mask)
        
        # 根据坐标中心选项计算基准点
        if coord_center == 'image_center':
            base_x = bg_width // 2
            base_y = bg_height // 2
        else:  # default - 左上角作为基准点
            base_x = 0
            base_y = 0
        
        # 计算最终的形状位置（基准点 + 偏移量）
        center_x = base_x + X_offset
        center_y = base_y + Y_offset
        
        # 使用wide和height控制形状大小
        shape_width = wide * scale
        shape_height = height * scale
        
        # 对于正方形、圆形等对称形状，使用最小边长的一半作为半径
        radius = int(min(shape_width, shape_height) / 2)
        
        # 绘制各种形状
        if shape == 'circle':
            draw.ellipse((center_x - radius, center_y - radius, 
                         center_x + radius, center_y + radius), 
                         fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.ellipse((center_x - radius, center_y - radius, 
                             center_x + radius, center_y + radius), 
                             fill=int(opacity * 255))
                             
        elif shape == 'square':
            draw.rectangle((center_x - radius, center_y - radius, 
                           center_x + radius, center_y + radius), 
                           fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.rectangle((center_x - radius, center_y - radius, 
                               center_x + radius, center_y + radius), 
                               fill=int(opacity * 255))
                               
        elif shape == 'rectangle':
            half_w = int(shape_width / 2)
            half_h = int(shape_height / 2)
            draw.rectangle((center_x - half_w, center_y - half_h, 
                           center_x + half_w, center_y + half_h), 
                           fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.rectangle((center_x - half_w, center_y - half_h, 
                               center_x + half_w, center_y + half_h), 
                               fill=int(opacity * 255))
                               
        elif shape == 'semicircle':
            draw.pieslice((center_x - radius, center_y - radius, 
                          center_x + radius, center_y + radius), 
                          0, 180, fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.pieslice((center_x - radius, center_y - radius, 
                             center_x + radius, center_y + radius), 
                             0, 180, fill=int(opacity * 255))
                             
        elif shape == 'quarter_circle':
            draw.pieslice((center_x - radius, center_y - radius, 
                          center_x + radius, center_y + radius), 
                          0, 90, fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.pieslice((center_x - radius, center_y - radius, 
                             center_x + radius, center_y + radius), 
                             0, 90, fill=int(opacity * 255))
                             
        elif shape == 'ellipse':
            draw.ellipse((center_x - radius, center_y - int(radius*0.7), 
                         center_x + radius, center_y + int(radius*0.7)), 
                         fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.ellipse((center_x - radius, center_y - int(radius*0.7), 
                            center_x + radius, center_y + int(radius*0.7)), 
                            fill=int(opacity * 255))
                            
        elif shape == 'triangle':
            points = [
                (center_x, center_y - radius),
                (center_x - radius, center_y + radius),
                (center_x + radius, center_y + radius)
            ]
            draw.polygon(points, fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.polygon(points, fill=int(opacity * 255))
            
        elif shape == 'cross':
            arm_width = int(radius / 3)
            draw.rectangle((center_x - radius, center_y - arm_width, 
                           center_x + radius, center_y + arm_width), 
                           fill=(*shape_color_rgb, int(opacity * 255)))
            draw.rectangle((center_x - arm_width, center_y - radius, 
                           center_x + arm_width, center_y + radius), 
                           fill=(*shape_color_rgb, int(opacity * 255)))
                           
            mask_draw.rectangle((center_x - radius, center_y - arm_width, 
                               center_x + radius, center_y + arm_width), 
                               fill=int(opacity * 255))
            mask_draw.rectangle((center_x - arm_width, center_y - radius, 
                               center_x + arm_width, center_y + radius), 
                               fill=int(opacity * 255))
                               
        elif shape == 'star':
            points = []
            for i in range(10):
                angle = i * 2 * np.pi / 10 - np.pi/2
                r = radius if i % 2 == 0 else radius / 2.5
                points.append((
                    center_x + r * np.cos(angle),
                    center_y + r * np.sin(angle)
                ))
            draw.polygon(points, fill=(*shape_color_rgb, int(opacity * 255)))
            mask_draw.polygon(points, fill=int(opacity * 255))
            
        elif shape == 'radial':
            segments = 8
            for i in range(segments):
                start_angle = i * 360 / segments
                end_angle = (i + 1) * 360 / segments
                draw.pieslice((center_x - radius, center_y - radius, 
                             center_x + radius, center_y + radius), 
                             start_angle, end_angle, fill=(*shape_color_rgb, int(opacity * 255)))
                mask_draw.pieslice((center_x - radius, center_y - radius, 
                                 center_x + radius, center_y + radius), 
                                 start_angle, end_angle, fill=int(opacity * 255))

        # 旋转处理
        if rotation != 0:
            shape_img = shape_img.rotate(rotation, center=(center_x, center_y), expand=False)
            mask = mask.rotate(rotation, center=(center_x, center_y), expand=False)

        # 模糊处理
        if blur_radius > 0:
            shape_img = shape_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        # 合成背景和形状
        background.paste(shape_img, (0, 0), shape_img)

        # 关键修改：将RGBA转换为RGB（3通道），去除alpha通道
        background_rgb = background.convert("RGB")
        
        # 转换为tensor输出（3通道）
        image_output = torch.from_numpy(np.array(background_rgb).astype(np.float32) / 255.0).unsqueeze(0)
        mask_output = torch.from_numpy(np.array(mask).astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        output_box2 = (bg_width, bg_height, X_offset, Y_offset)
        
        return (image_output, mask_output, output_box2)





class create_mask_array:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 99999, "step": 16}),
                "height": ("INT", {"default": 512, "min": 16, "max": 99999, "step": 16}),
                "split_mode": (["按宽分割", "按高分割"],),
                "split_pattern": ("STRING", {"default": "1-1", "description": "分割比例，格式如1-2-3"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "LIST",)
    RETURN_NAMES = ("合成图片", "遮罩阵列",)
    FUNCTION = "generate"
    CATEGORY = "Apt_Preset/mask"   
    OUTPUT_IS_LIST = (False,True,)

    def generate_unique_colors(self, count):
        colors = []
        while len(colors) < count:
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            if color not in colors:
                colors.append(color)
        return colors
    
    def parse_split_pattern(self, pattern):
        if not pattern or not pattern[0].isdigit() or not pattern[-1].isdigit():
            raise ValueError("分割模式必须以数字开始和结束")
        
        parts = pattern.split('-')
        try:
            return [int(part) for part in parts if part.strip()]
        except ValueError:
            raise ValueError("分割模式只能包含数字和连字符，如1-2-3")
    
    def generate(self, width, height, split_mode, split_pattern, ):
        split_ratios = self.parse_split_pattern(split_pattern)
        if len(split_ratios) < 1:
            raise ValueError("分割模式至少需要一个数字")
        
        total = sum(split_ratios)
        if total <= 0:
            raise ValueError("分割比例总和必须大于0")
        
        colors = self.generate_unique_colors(len(split_ratios))
        
        # 创建合成预览图像（RGB模式）
        composite_img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(composite_img)
        
        # 存储遮罩的列表
        masks = []
        current_pos = 0
        
        if split_mode == "按宽分割":
            for i, ratio in enumerate(split_ratios):
                segment_width = int(width * ratio / total)
                if i == len(split_ratios) - 1:
                    segment_width = width - current_pos
                
                # 绘制彩色合成图
                draw.rectangle(
                    [current_pos, 0, current_pos + segment_width, height],
                    fill=colors[i]
                )
                
                # 创建单个遮罩（L模式）
                mask = Image.new('L', (width, height), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(
                    [current_pos, 0, current_pos + segment_width, height],
                    fill=255
                )
                masks.append(mask)
                
                current_pos += segment_width
        
        else:  # 按高分
            for i, ratio in enumerate(split_ratios):
                segment_height = int(height * ratio / total)
                if i == len(split_ratios) - 1:
                    segment_height = height - current_pos
                
                # 绘制彩色合成图
                draw.rectangle(
                    [0, current_pos, width, current_pos + segment_height],
                    fill=colors[i]
                )
                
                # 创建单个遮罩（L模式）
                mask = Image.new('L', (width, height), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(
                    [0, current_pos, width, current_pos + segment_height],
                    fill=255
                )
                masks.append(mask)
                
                current_pos += segment_height
        
        # 转换为张量，使用你提供的pil2tensor函数
        composite_tensor = pil2tensor(composite_img)
        
        mask_tensors = [pil2tensor(mask) for mask in masks]



        return (composite_tensor, mask_tensors)
    



class create_mask_array:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 16, "max": 99999, "step": 16}),
                "height": ("INT", {"default": 512, "min": 16, "max": 99999, "step": 16}),
                "split_mode": (["按宽分割", "按高分割"],),
                "split_pattern": ("STRING", {"default": "1-1", "description": "分割比例，格式如1-2-3"}),
            }
        }
    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("合成图片", "遮罩阵列",)
    FUNCTION = "generate"
    CATEGORY = "Apt_Preset/mask/😺backup"   
    OUTPUT_IS_LIST = (False, True,)

    def generate_unique_colors(self, count):
        colors = []
        while len(colors) < count:
            color = (
                random.randint(50, 255),
                random.randint(50, 255),
                random.randint(50, 255)
            )
            if color not in colors:
                colors.append(color)
        return colors
    
    def parse_split_pattern(self, pattern):
        if not pattern or not pattern[0].isdigit() or not pattern[-1].isdigit():
            raise ValueError("分割模式必须以数字开始和结束")
        
        parts = pattern.split('-')
        try:
            return [int(part) for part in parts if part.strip()]
        except ValueError:
            raise ValueError("分割模式只能包含数字和连字符，如1-2-3")
    
    def generate(self, width, height, split_mode, split_pattern):
        split_ratios = self.parse_split_pattern(split_pattern)
        if len(split_ratios) < 1:
            raise ValueError("分割模式至少需要一个数字")
        
        total = sum(split_ratios)
        if total <= 0:
            raise ValueError("分割比例总和必须大于0")
        
        colors = self.generate_unique_colors(len(split_ratios))
        
        composite_img = Image.new('RGB', (width, height))
        draw = ImageDraw.Draw(composite_img)
        
        masks = []
        current_pos = 0
        
        if split_mode == "按宽分割":
            for i, ratio in enumerate(split_ratios):
                segment_width = int(width * ratio / total)
                if i == len(split_ratios) - 1:
                    segment_width = width - current_pos
                
                draw.rectangle(
                    [current_pos, 0, current_pos + segment_width, height],
                    fill=colors[i]
                )
                
                mask = Image.new('L', (width, height), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(
                    [current_pos, 0, current_pos + segment_width, height],
                    fill=255
                )
                masks.append(mask)
                
                current_pos += segment_width
        
        else:
            for i, ratio in enumerate(split_ratios):
                segment_height = int(height * ratio / total)
                if i == len(split_ratios) - 1:
                    segment_height = height - current_pos
                
                draw.rectangle(
                    [0, current_pos, width, current_pos + segment_height],
                    fill=colors[i]
                )
                
                mask = Image.new('L', (width, height), 0)
                mask_draw = ImageDraw.Draw(mask)
                mask_draw.rectangle(
                    [0, current_pos, width, current_pos + segment_height],
                    fill=255
                )
                masks.append(mask)
                
                current_pos += segment_height
        
        composite_tensor = pil2tensor(composite_img)
        mask_tensors = [pil2tensor(mask).squeeze(0) for mask in masks]

        return (composite_tensor, mask_tensors)


class mask_sam_detctor:
    _model_cache = {}

    @classmethod
    def _compute_cache_key(cls, ckpt_name, pos):
        """计算模型加载参数的缓存键"""
        import hashlib
        key_data = f"{ckpt_name}|{pos}"
        return hashlib.md5(key_data.encode()).hexdigest()

    @classmethod
    def _load_model_cached(cls, ckpt_name, pos):
        """带缓存的模型加载和提示词编码"""
        cache_key = cls._compute_cache_key(ckpt_name, pos)

        if cache_key in cls._model_cache:
            print(f"[mask_sam_detctor] Cache hit, reusing loaded model and conditioning")
            return cls._model_cache[cache_key]

        print(f"[mask_sam_detctor] Cache miss, loading model and encoding prompts...")
        model = None
        conditioning = None
        
        if ckpt_name:
            try:
                from nodes import CheckpointLoaderSimple, CLIPTextEncode
                model, clip, vae2 = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
                # 只有当 pos 确实有内容时，才生成 conditioning
                if clip is not None and pos and pos.strip():
                    (conditioning,) = CLIPTextEncode().encode(clip, pos.strip())
            except Exception as e:
                print(f"mask_sam_detctor: 加载模型失败 -> {e}")
                
        cls._model_cache[cache_key] = (model, conditioning)
        return model, conditioning

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine_iterations": ("INT", {"default": 2, "min": 0, "max": 5, "step": 1}),
                "individual_masks": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"),{"default":"sam3.1_multiplex_fp16.safetensors"}),
                "pos":  ("STRING", {"default": "", "multiline": False}),
                "permil_str": ("STRING", {"default": "", "forceInput": True}),
                "ui_positive_coords": ("STRING", {"default": "", "multiline": True}),
                "ui_negative_coords": ("STRING", {"default": "", "multiline": True}),
                "ui_bboxes_json": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("MASK", "BOUNDING_BOX")
    RETURN_NAMES = ("masks", "bboxes")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/mask"
    OUTPUT_NODE = True
    DESCRIPTION = """
文本输入多个相同名称,例如3个男孩,输入：boy:3
四者不能同时作用，多种同时存在，则按优先级：外部输入框>点>文本>手动框，
permil_str千分比对角框，批量格式："[[x1, y1, x2, y2], [x1, y1, x2, y2], ...]"
"""

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @staticmethod
    def _parse_bboxes_json(raw_text):
        if raw_text is None:
            return None
        text = str(raw_text).strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except Exception:
            return None

        def _norm_item(item):
            if not isinstance(item, dict):
                return None
            try:
                x = float(item.get("x", 0))
                y = float(item.get("y", 0))
                w = float(item.get("width", 0))
                h = float(item.get("height", 0))
            except Exception:
                return None
            if w <= 0 or h <= 0:
                return None
            return {"x": x, "y": y, "width": w, "height": h}

        if isinstance(data, dict):
            one = _norm_item(data)
            return one
        if isinstance(data, list):
            out = []
            for it in data:
                norm = _norm_item(it)
                if norm is not None:
                    out.append(norm)
            return out if out else None
        return None

    @staticmethod
    def _normalize_points_text(raw_text):
        if raw_text is None:
            return None
        text = str(raw_text).strip()
        if not text or text in ("[]", "{}", "null", "None"):
            return None
        try:
            data = json.loads(text)
            if isinstance(data, list) and len(data) == 0:
                return None
            if isinstance(data, dict) and len(data) == 0:
                return None
        except Exception:
            pass
        return text

    def run(
        self,
        image,
        threshold=0.5,
        refine_iterations=2,
        individual_masks=False,
        permil_str="",
        ui_positive_coords="",
        ui_negative_coords="",
        ui_bboxes_json="",
        ckpt_name="",
        pos=None,
    ):
        from .set_location import robust_extract_bbox

        # 使用缓存机制处理模型加载和文本编码
        model, conditioning = self._load_model_cached(ckpt_name, pos)

        batch, img_h, img_w, _ = image.shape
        # 解析外部输入的千分比框
        external_bboxes_ui = []
        parsed_bboxes = []
        if permil_str:
            permil_boxes = robust_extract_bbox(permil_str)
            if permil_boxes:
                for x1, y1, x2, y2 in permil_boxes:
                    px1 = x1 * img_w / 1000
                    py1 = y1 * img_h / 1000
                    px2 = x2 * img_w / 1000
                    py2 = y2 * img_h / 1000
                    
                    px1 = max(0, min(int(round(px1)), img_w))
                    py1 = max(0, min(int(round(py1)), img_h))
                    px2 = max(0, min(int(round(px2)), img_w))
                    py2 = max(0, min(int(round(py2)), img_h))
                    
                    if px2 < px1: px1, px2 = px2, px1
                    if py2 < py1: py1, py2 = py2, py1
                    
                    bw = max(1, px2 - px1)
                    bh = max(1, py2 - py1)
                    parsed_bboxes.append([px1, py1, px2, py2])
                    external_bboxes_ui.append({"x": px1, "y": py1, "width": bw, "height": bh})

        # 控制优先级逻辑：外部输入框 > 手动画框 > 点选 > 文本
        
        # 预设所有变量为 None
        merged_bboxes = None
        final_pos = None
        final_neg = None
        final_conditioning = None

        if parsed_bboxes:
            # 1. 外部输入框优先级最高，如果有外部框，屏蔽手动框、点、文本
            # SAM3_Detect.execute 内部会调用 _boxes_to_tensor() 将字典列表解析为正确的 Tensor。
            # 所以这里必须原样传递字典列表格式（List[Dict]），千万不能提前转成 Tensor！
            merged_bboxes = external_bboxes_ui
            
        else:
            # 没有外部框，检查手动框
            manual_bboxes = self._parse_bboxes_json(ui_bboxes_json)
            # 如果 manual_bboxes 解析出来是空的列表 []，应当视为“没有框”
            if manual_bboxes is not None and (not isinstance(manual_bboxes, list) or len(manual_bboxes) > 0):
                # 2. 有手动框，屏蔽点、文本
                # _parse_bboxes_json 返回的已经是单个 dict 或者 List[Dict] 了，直接用即可
                merged_bboxes = manual_bboxes
                
            else:
                # 3. 检查点选输入
                pos_str = self._normalize_points_text(ui_positive_coords)
                neg_str = self._normalize_points_text(ui_negative_coords)
                if pos_str or neg_str:
                    # 3. 有点选输入，屏蔽文本
                    final_pos = pos_str or None
                    final_neg = neg_str or None
                    
                else:
                    # 4. 只有当没有任何框和点时，才允许文本生效
                    # conditioning 保持从缓存加载的有效值
                    final_conditioning = conditioning

        # 生成用于前端预览的背景图，直接使用输入的 image
        bg_results = []
        try:
            import random
            import os
            import folder_paths
            import numpy as np
            from PIL import Image
            
            temp_dir = folder_paths.get_temp_directory()
            if image.shape[0] > 0:
                src_np = (255.0 * image[0].cpu().numpy()).clip(0, 255).astype(np.uint8)
                src_pil = Image.fromarray(src_np)
                filename = f"mask_sam3_detctor_bg_{random.randint(1, 1000000)}.png"
                src_pil.save(os.path.join(temp_dir, filename))
                bg_results.append({"filename": filename, "subfolder": "", "type": "temp"})
        except Exception as e:
            print(f"mask_sam_detctor: Preview 生成失败 -> {e}")

        ui = {"bg_image": bg_results}
        
        # 通知前端
        if external_bboxes_ui:
            ui["external_bboxes"] = external_bboxes_ui

        if model is None:
            b, h, w, _ = image.shape
            empty = torch.zeros((b, h, w), dtype=image.dtype, device=image.device)
            return {"ui": ui, "result": (empty, [])}

        try:
            from comfy_extras.nodes_sam3 import SAM3_Detect as _SAM3Detect
            result = _SAM3Detect.execute(
                model=model,
                image=image,
                conditioning=final_conditioning,
                bboxes=merged_bboxes,
                positive_coords=final_pos,
                negative_coords=final_neg,
                threshold=float(threshold),
                refine_iterations=int(refine_iterations),
                individual_masks=bool(individual_masks),
            )
            return {"ui": ui, "result": (result[0], result[1])}
        except Exception as e:
            print(f"mask_sam_detctor: SAM3 执行失败 -> {e}")
            b, h, w, _ = image.shape
            empty = torch.zeros((b, h, w), dtype=image.dtype, device=image.device)
            return {"ui": ui, "result": (empty, [])}

































