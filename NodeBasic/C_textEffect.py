
import re
import json
import os
import math
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps




class Stack_GradientAndStroke:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_gradient": ("BOOLEAN", {"default": False}),
                "start_color": ("STRING", {"default": "#ff0000"}),
                "end_color": ("STRING", {"default": "#0000ff"}),
                "angle": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "opacity_gradient_type": (["none", "linear", "radial"], {"default": "none"}),
                "opacity_start": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "opacity_end": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stroke_width": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "stroke_color": ("STRING", {"default": "#000000"}),
                "stroke_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stroke_type": (["normal", "smooth", "double"], {"default": "normal"}),
                "second_stroke_color": ("STRING", {"default": "#ffff00"}),
                "second_stroke_width": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1})
            },
            "optional": {}
        }

    RETURN_TYPES = ("GRADIENT_AND_STROKE_DATA",)
    RETURN_NAMES = ("GradientAndStroke",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def encode(self, use_gradient, start_color, end_color, angle, opacity_gradient_type, opacity_start, opacity_end,
               stroke_width, stroke_color, stroke_opacity, stroke_type, second_stroke_color, second_stroke_width):
        data = (use_gradient, start_color, end_color, angle, opacity_gradient_type, opacity_start, opacity_end,
                stroke_width, stroke_color, stroke_opacity, stroke_type, second_stroke_color, second_stroke_width)
        return (data,)


class Stack_ShadowAndGlow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shadow_x": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shadow_y": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "shadow_color": ("STRING", {"default": "#000000"}),
                "shadow_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_blur": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "shadow_type": (["single", "multi", "gradient"], {"default": "single"}),
                "multi_shadow_count": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "gradient_shadow_start": ("STRING", {"default": "#000000"}),
                "gradient_shadow_end": ("STRING", {"default": "#444444"}),
                "outer_glow_color": ("STRING", {"default": "#ffffff"}),
                "outer_glow_opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "outer_glow_radius": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1}),
                "inner_glow_color": ("STRING", {"default": "#ffffff"}),
                "inner_glow_opacity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "inner_glow_radius": ("INT", {"default": 0, "min": 0, "max": 50, "step": 1})
            },
            "optional": {}
        }

    RETURN_TYPES = ("SHADOW_AND_GLOW_DATA",)
    RETURN_NAMES = ("ShadowAndGlow",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def encode(self, shadow_x, shadow_y, shadow_color, shadow_opacity, shadow_blur, shadow_type, multi_shadow_count,
               gradient_shadow_start, gradient_shadow_end, outer_glow_color, outer_glow_opacity, outer_glow_radius,
               inner_glow_color, inner_glow_opacity, inner_glow_radius):
        data = (shadow_x, shadow_y, shadow_color, shadow_opacity, shadow_blur, shadow_type, multi_shadow_count,
                gradient_shadow_start, gradient_shadow_end, outer_glow_color, outer_glow_opacity, outer_glow_radius,
                inner_glow_color, inner_glow_opacity, inner_glow_radius)
        return (data,)


class Stack_EmbossAndFill:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "texture_image": ("IMAGE", {"default": None}),
                "use_texture_fill": ("BOOLEAN", {"default": False}),
                "texture_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "emboss_effect": ("BOOLEAN", {"default": False}),
                "emboss_depth": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "emboss_light_color": ("STRING", {"default": "#ffffff"}),
                "emboss_shadow_color": ("STRING", {"default": "#000000"})
            }
        }

    RETURN_TYPES = ("EMBOSS_AND_FILL_DATA",)
    RETURN_NAMES = ("EmbossAndFill",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def encode(self, texture_image=None, use_texture_fill=False, texture_scale=1.0, emboss_effect=False,
               emboss_depth=2, emboss_light_color="#ffffff", emboss_shadow_color="#000000"):
        data = (texture_image, use_texture_fill, texture_scale, emboss_effect, emboss_depth, emboss_light_color,
                emboss_shadow_color)
        return (data,)







#region-----lay_text_sum---------------------------------------------------

def get_comfyui_root():
    current_script = os.path.realpath(__file__)
    parent_dir = os.path.dirname(current_script)
    while parent_dir != os.path.dirname(parent_dir):
        if os.path.exists(os.path.join(parent_dir, "custom_nodes")) and os.path.exists(os.path.join(parent_dir, "models")):
            return parent_dir
        parent_dir = os.path.dirname(parent_dir)
    return os.path.dirname(os.path.dirname(current_script))

COMFYUI_ROOT = get_comfyui_root()
FONT_DIR_1 = os.path.join(COMFYUI_ROOT, "custom_nodes", "ComfyUI-Apt_Preset", "fonts")
FONT_DIR_2 = os.path.join(COMFYUI_ROOT, "models", "Apt_File", "fonts")

def get_all_fonts():
    font_map = {}
    for font_dir in [FONT_DIR_1, FONT_DIR_2]:
        if os.path.isdir(font_dir):
            for f in os.listdir(font_dir):
                if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith((".ttf", ".otf")):
                    if f not in font_map:
                        font_map[f] = os.path.join(font_dir, f)
    return font_map

FONT_MAP = get_all_fonts()
file_list = list(FONT_MAP.keys())

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_gradient(width, height, start_color_rgb, end_color_rgb, angle):
    gradient_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient_img)
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    proj_width = abs(width * cos_a) + abs(height * sin_a)
    proj_height = abs(width * sin_a) + abs(height * cos_a)
    length = max(proj_width, proj_height)
    if length == 0:
        length = 1
    for x in range(width):
        for y in range(height):
            t = (x * cos_a + y * sin_a + length / 2) / length
            t = max(0, min(1, t))
            r = int(start_color_rgb[0] + (end_color_rgb[0] - start_color_rgb[0]) * t)
            g = int(start_color_rgb[1] + (end_color_rgb[1] - start_color_rgb[1]) * t)
            b = int(start_color_rgb[2] + (end_color_rgb[2] - start_color_rgb[2]) * t)
            draw.point((x, y), fill=(r, g, b, 255))
    return gradient_img

def create_outer_glow(mask, color_rgb, opacity, radius):
    blur_radius = max(1, radius) if radius > 0 else 0
    glow_color = color_rgb + (int(255 * opacity),)
    glow_layer = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.bitmap((0, 0), mask, fill=glow_color)
    if blur_radius > 0:
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return glow_layer

def create_inner_glow(mask, color_rgb, opacity, radius):
    blur_radius = max(1, radius) if radius > 0 else 0
    glow_color = color_rgb + (int(255 * opacity),)
    glow_layer = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.bitmap((0, 0), mask, fill=glow_color)
    if blur_radius > 0:
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    inverted_mask = ImageOps.invert(mask)
    glow_layer = Image.composite(Image.new('RGBA', mask.size, (0,0,0,0)), glow_layer, inverted_mask)
    return glow_layer

def apply_texture_fill(text_mask, texture_image, texture_scale, text_width, text_height):
    if texture_image is None:
        return Image.new('RGBA', (text_width, text_height), (0,0,0,0))
    texture = texture_image[0].cpu().numpy()
    texture = (texture * 255).astype(np.uint8)
    texture_img = Image.fromarray(texture).convert('RGBA')
    scaled_width = int(text_width * texture_scale)
    scaled_height = int(text_height * texture_scale)
    texture_img = texture_img.resize((scaled_width, scaled_height), Image.BICUBIC)
    texture_text = Image.new('RGBA', (text_width, text_height), (0,0,0,0))
    for x in range(0, text_width, scaled_width):
        for y in range(0, text_height, scaled_height):
            texture_text.paste(texture_img, (x, y))
    result = Image.new('RGBA', (text_width, text_height), (0,0,0,0))
    result.paste(texture_text, (0,0), text_mask)
    return result

def apply_opacity_gradient(text_img, gradient_type, start_opacity, end_opacity):
    img_data = np.array(text_img)
    h, w = img_data.shape[:2]
    if gradient_type == "linear":
        for x in range(w):
            t = x / w
            alpha = int(255 * (start_opacity + (end_opacity - start_opacity) * t))
            img_data[:, x, 3] = np.clip(img_data[:, x, 3] * (alpha/255), 0, 255)
    elif gradient_type == "radial":
        center_x, center_y = w//2, h//2
        max_dist = math.hypot(center_x, center_y)
        for y in range(h):
            for x in range(w):
                dist = math.hypot(x - center_x, y - center_y)
                t = dist / max_dist if max_dist > 0 else 0
                t = max(0, min(1, t))
                alpha = int(255 * (start_opacity + (end_opacity - start_opacity) * t))
                img_data[y, x, 3] = np.clip(img_data[y, x, 3] * (alpha/255), 0, 255)
    return Image.fromarray(img_data)



class lay_text_sum:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"default": "", "multiline": True}),
                "font_file": (file_list, {"default": file_list[0] if file_list else ""}),
                "x": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "font_size": ("INT", {"default": 12, "min": 0, "max": 320, "step": 1}),
                "arrangement": (["horizontal", "vertical"], {"default": "horizontal"}),
                "text_rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "text_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "GradientAndStroke": ("GRADIENT_AND_STROKE_DATA",),
                "ShadowAndGlow": ("SHADOW_AND_GLOW_DATA",),
                "EmbossAndFill": ("EMBOSS_AND_FILL_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_text"
    CATEGORY = "Apt_Preset/imgEffect"

    def apply_text(self, image, text, font_file, x, y, font_size, arrangement, text_rotation,
                  text_color, text_opacity, GradientAndStroke=None, ShadowAndGlow=None, EmbossAndFill=None):
        # ========== 初始化可选参数默认值 ==========
        # 渐变和描边默认值
        use_gradient = False
        start_color = "#ff0000"
        end_color = "#0000ff"
        angle = 0
        opacity_gradient_type = "none"
        opacity_start = 1.0
        opacity_end = 0.5
        stroke_width = 0
        stroke_color = "#000000"
        stroke_opacity = 1.0
        stroke_type = "normal"
        second_stroke_color = "#ffff00"
        second_stroke_width = 0

        # 阴影和发光默认值
        shadow_x = 0
        shadow_y = 0
        shadow_color = "#000000"
        shadow_opacity = 1.0
        shadow_blur = 0
        shadow_type = "single"
        multi_shadow_count = 3
        gradient_shadow_start = "#000000"
        gradient_shadow_end = "#444444"
        outer_glow_color = "#ffffff"
        outer_glow_opacity = 0.8
        outer_glow_radius = 0
        inner_glow_color = "#ffffff"
        inner_glow_opacity = 0.8
        inner_glow_radius = 0

        # 浮雕和填充默认值
        texture_image = None
        use_texture_fill = False
        texture_scale = 1.0
        emboss_effect = False
        emboss_depth = 2
        emboss_light_color = "#ffffff"
        emboss_shadow_color = "#000000"

        # ========== 解析可选插槽数据 ==========
        # 解析渐变和描边数据
        if GradientAndStroke is not None and len(GradientAndStroke) >= 13:
            use_gradient, start_color, end_color, angle, opacity_gradient_type, opacity_start, opacity_end, \
            stroke_width, stroke_color, stroke_opacity, stroke_type, second_stroke_color, second_stroke_width = GradientAndStroke

        # 解析阴影和发光数据
        if ShadowAndGlow is not None and len(ShadowAndGlow) >= 15:
            shadow_x, shadow_y, shadow_color, shadow_opacity, shadow_blur, shadow_type, multi_shadow_count, \
            gradient_shadow_start, gradient_shadow_end, outer_glow_color, outer_glow_opacity, outer_glow_radius, \
            inner_glow_color, inner_glow_opacity, inner_glow_radius = ShadowAndGlow

        # 解析浮雕和填充数据
        if EmbossAndFill is not None and len(EmbossAndFill) >= 7:
            texture_image, use_texture_fill, texture_scale, emboss_effect, emboss_depth, \
            emboss_light_color, emboss_shadow_color = EmbossAndFill

        # ========== 原有业务逻辑 ==========
        if text == "":
            return (image,)
        
        # 加载字体
        if not font_file or font_file not in FONT_MAP:
            font = ImageFont.load_default()
        else:
            font_path = FONT_MAP[font_file]
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()

        # 基础图片转换
        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        base_img = Image.fromarray(img).convert('RGBA')
        img_width, img_height = base_img.size

        # 颜色转换
        text_color_rgb = hex_to_rgb(text_color)
        stroke_color_rgb = hex_to_rgb(stroke_color)
        second_stroke_color_rgb = hex_to_rgb(second_stroke_color)
        shadow_color_rgb = None if shadow_color == "none" else hex_to_rgb(shadow_color)
        start_color_rgb = hex_to_rgb(start_color)
        end_color_rgb = hex_to_rgb(end_color)
        outer_glow_color_rgb = hex_to_rgb(outer_glow_color)
        inner_glow_color_rgb = hex_to_rgb(inner_glow_color)
        emboss_light_rgb = hex_to_rgb(emboss_light_color)
        emboss_shadow_rgb = hex_to_rgb(emboss_shadow_color)

        # 横竖排处理
        if arrangement == "vertical":
            text = '\n'.join(list(text.replace('\n', '')))

        # 计算文本原始尺寸（旋转前）
        temp_draw = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        text_bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        safety_margin = font_size // 2
        text_h += safety_margin

        # 创建文本专用画布（旋转前）
        text_canvas = Image.new('RGBA', (text_w + 2*safety_margin, text_h + 2*safety_margin), (0,0,0,0))
        text_draw = ImageDraw.Draw(text_canvas)
        tx = safety_margin
        ty = safety_margin

        # ========== 1. 绘制阴影（最底层） ==========
        if shadow_color_rgb and (shadow_x != 0 or shadow_y != 0 or shadow_blur > 0):
            shadow_layer = Image.new('RGBA', text_canvas.size, (0,0,0,0))
            shadow_draw = ImageDraw.Draw(shadow_layer)
            if shadow_type == "single":
                shadow_draw.text((tx + shadow_x, ty + shadow_y), text, font=font,
                               fill=shadow_color_rgb + (int(255 * shadow_opacity),))
            elif shadow_type == "multi":
                for i in range(multi_shadow_count):
                    alpha = shadow_opacity * (1 - i/multi_shadow_count)
                    ox = shadow_x * (i+1) / multi_shadow_count
                    oy = shadow_y * (i+1) / multi_shadow_count
                    shadow_draw.text((int(tx+ox), int(ty+oy)), text, font=font,
                                   fill=shadow_color_rgb + (int(255*alpha),))
            elif shadow_type == "gradient":
                grad_start = hex_to_rgb(gradient_shadow_start)
                grad_end = hex_to_rgb(gradient_shadow_end)
                grad_img = create_gradient(text_canvas.width, text_canvas.height, grad_start, grad_end, angle)
                shadow_mask = Image.new('L', text_canvas.size, 0)
                ImageDraw.Draw(shadow_mask).text((tx+shadow_x, ty+shadow_y), text, font=font, fill=255)
                shadow_layer = Image.composite(grad_img, shadow_layer, shadow_mask)
            if shadow_blur > 0:
                shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            text_canvas = Image.alpha_composite(text_canvas, shadow_layer)

        # ========== 2. 绘制外发光 ==========
        if outer_glow_radius > 0:
            glow_mask = Image.new('L', text_canvas.size, 0)
            ImageDraw.Draw(glow_mask).text((tx, ty), text, font=font, fill=255)
            outer_glow = create_outer_glow(glow_mask, outer_glow_color_rgb, outer_glow_opacity, outer_glow_radius)
            text_canvas = Image.alpha_composite(text_canvas, outer_glow)

        # ========== 3. 绘制描边 ==========
        if stroke_width > 0:
            stroke_layer = Image.new('RGBA', text_canvas.size, (0,0,0,0))
            stroke_draw = ImageDraw.Draw(stroke_layer)
            
            if stroke_type == "normal":
                for dx in range(-stroke_width, stroke_width+1):
                    for dy in range(-stroke_width, stroke_width+1):
                        if dx != 0 or dy != 0:
                            stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                         fill=stroke_color_rgb + (int(255 * stroke_opacity),))
            elif stroke_type == "smooth":
                stroke_draw.text((tx, ty), text, font=font,
                               fill=stroke_color_rgb + (int(255 * stroke_opacity),))
                stroke_layer = stroke_layer.filter(ImageFilter.GaussianBlur(radius=1))
            elif stroke_type == "double":
                if second_stroke_width > 0:
                    for dx in range(-second_stroke_width, second_stroke_width+1):
                        for dy in range(-second_stroke_width, second_stroke_width+1):
                            if dx != 0 or dy != 0:
                                stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                             fill=second_stroke_color_rgb + (int(255 * stroke_opacity),))
                    for dx in range(-stroke_width, stroke_width+1):
                        for dy in range(-stroke_width, stroke_width+1):
                            if dx != 0 or dy != 0:
                                stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                             fill=stroke_color_rgb + (int(255 * stroke_opacity),))
            
            text_canvas = Image.alpha_composite(text_canvas, stroke_layer)

        # ========== 4. 绘制文本主体（严格遵循PS优先级逻辑） ==========
        text_main_layer = Image.new('RGBA', text_canvas.size, (0,0,0,0))
        text_main_draw = ImageDraw.Draw(text_main_layer)
        
        # 优先级1：浮雕效果（最高，覆盖所有文本颜色/渐变/纹理）
        if emboss_effect:
            light_color = emboss_light_rgb + (int(255*text_opacity),)
            shadow_color = emboss_shadow_rgb + (int(255*text_opacity),)
            text_main_draw.text((tx - emboss_depth, ty - emboss_depth), text, font=font, fill=light_color)
            text_main_draw.text((tx + emboss_depth, ty + emboss_depth), text, font=font, fill=shadow_color)
            text_main_draw.text((tx, ty), text, font=font, fill=(0,0,0,int(255*text_opacity))) # 浮雕底层黑色
        # 优先级2：纹理填充（次之，覆盖渐变/纯色）
        elif use_texture_fill and texture_image is not None:
            text_mask = Image.new('L', (text_w, text_h), 0)
            ImageDraw.Draw(text_mask).text((0, 0), text, font=font, fill=255)
            texture_fill = apply_texture_fill(text_mask, texture_image, texture_scale, text_w, text_h)
            texture_data = np.array(texture_fill)
            texture_data[..., 3] = (texture_data[..., 3] * text_opacity).astype(np.uint8)
            texture_fill = Image.fromarray(texture_data)
            text_main_layer.paste(texture_fill, (tx, ty), texture_fill)
        # 优先级3：渐变文本（覆盖纯色）
        elif use_gradient:
            grad_img = create_gradient(text_w, text_h, start_color_rgb, end_color_rgb, angle)
            text_mask = Image.new('L', (text_w, text_h), 0)
            ImageDraw.Draw(text_mask).text((0, 0), text, font=font, fill=255)
            grad_text = Image.new('RGBA', (text_w, text_h), (0,0,0,0))
            grad_text.paste(grad_img, (0,0), text_mask)
            grad_data = np.array(grad_text)
            grad_data[..., 3] = (grad_data[..., 3] * text_opacity).astype(np.uint8)
            grad_text = Image.fromarray(grad_data)
            if opacity_gradient_type != "none":
                grad_text = apply_opacity_gradient(grad_text, opacity_gradient_type, opacity_start, opacity_end)
            text_main_layer.paste(grad_text, (tx, ty), grad_text)
        # 优先级4：纯色文本（最低）
        else:
            text_main_draw.text((tx, ty), text, font=font, fill=text_color_rgb + (int(255 * text_opacity),))
            if opacity_gradient_type != "none":
                temp_text = Image.new('RGBA', text_main_layer.size, (0,0,0,0))
                temp_draw = ImageDraw.Draw(temp_text)
                temp_draw.text((tx, ty), text, font=font, fill=text_color_rgb + (int(255 * text_opacity),))
                temp_text = apply_opacity_gradient(temp_text, opacity_gradient_type, opacity_start, opacity_end)
                text_mask = Image.new('L', text_main_layer.size, 0)
                ImageDraw.Draw(text_mask).text((tx, ty), text, font=font, fill=255)
                text_main_layer = Image.composite(temp_text, text_main_layer, text_mask)

        # 合并文本主体图层
        text_canvas = Image.alpha_composite(text_canvas, text_main_layer)

        # ========== 5. 绘制内发光（最顶层） ==========
        if inner_glow_radius > 0:
            glow_mask = Image.new('L', text_canvas.size, 0)
            ImageDraw.Draw(glow_mask).text((tx, ty), text, font=font, fill=255)
            inner_glow = create_inner_glow(glow_mask, inner_glow_color_rgb, inner_glow_opacity, inner_glow_radius)
            text_canvas = Image.alpha_composite(text_canvas, inner_glow)

        # ========== 统一旋转文本画布 ==========
        if text_rotation != 0:
            text_canvas = text_canvas.rotate(text_rotation, expand=True, resample=Image.BICUBIC)

        # ========== 粘贴到主图 ==========
        rotated_w, rotated_h = text_canvas.size
        paste_x = x - (rotated_w // 2) + (text_w // 2)
        paste_y = y - (rotated_h // 2) + (text_h // 2)
        
        paste_x = max(0, min(paste_x, img_width - rotated_w))
        paste_y = max(0, min(paste_y, img_height - rotated_h))
        
        base_img.paste(text_canvas, (paste_x, paste_y), text_canvas)

        # 转换回tensor
        result = np.array(base_img).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]
        return (result,)



#endregion--------------------------------------------------------







#region-----lay_text_sum_mul-------------------------------------------------

def get_comfyui_root():
    current_script = os.path.realpath(__file__)
    parent_dir = os.path.dirname(current_script)
    while parent_dir != os.path.dirname(parent_dir):
        if os.path.exists(os.path.join(parent_dir, "custom_nodes")) and os.path.exists(os.path.join(parent_dir, "models")):
            return parent_dir
        parent_dir = os.path.dirname(parent_dir)
    return os.path.dirname(os.path.dirname(current_script))

COMFYUI_ROOT = get_comfyui_root()
FONT_DIR_1 = os.path.join(COMFYUI_ROOT, "custom_nodes", "ComfyUI-Apt_Preset", "fonts")
FONT_DIR_2 = os.path.join(COMFYUI_ROOT, "models", "Apt_File", "fonts")

def get_all_fonts():
    font_map = {}
    for font_dir in [FONT_DIR_1, FONT_DIR_2]:
        if os.path.isdir(font_dir):
            for f in os.listdir(font_dir):
                if os.path.isfile(os.path.join(font_dir, f)) and f.lower().endswith((".ttf", ".otf")):
                    if f not in font_map:
                        font_map[f] = os.path.join(font_dir, f)
    return font_map

FONT_MAP = get_all_fonts()
file_list = list(FONT_MAP.keys())

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def create_gradient(width, height, start_color_rgb, end_color_rgb, angle):
    gradient_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient_img)
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    proj_width = abs(width * cos_a) + abs(height * sin_a)
    proj_height = abs(width * sin_a) + abs(height * cos_a)
    length = max(proj_width, proj_height)
    if length == 0:
        length = 1
    for x in range(width):
        for y in range(height):
            t = (x * cos_a + y * sin_a + length / 2) / length
            t = max(0, min(1, t))
            r = int(start_color_rgb[0] + (end_color_rgb[0] - start_color_rgb[0]) * t)
            g = int(start_color_rgb[1] + (end_color_rgb[1] - start_color_rgb[1]) * t)
            b = int(start_color_rgb[2] + (end_color_rgb[2] - start_color_rgb[2]) * t)
            draw.point((x, y), fill=(r, g, b, 255))
    return gradient_img

def create_outer_glow(mask, color_rgb, opacity, radius):
    blur_radius = max(1, radius) if radius > 0 else 0
    glow_color = color_rgb + (int(255 * opacity),)
    glow_layer = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.bitmap((0, 0), mask, fill=glow_color)
    if blur_radius > 0:
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return glow_layer

def create_inner_glow(mask, color_rgb, opacity, radius):
    blur_radius = max(1, radius) if radius > 0 else 0
    glow_color = color_rgb + (int(255 * opacity),)
    glow_layer = Image.new('RGBA', mask.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)
    glow_draw.bitmap((0, 0), mask, fill=glow_color)
    if blur_radius > 0:
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    inverted_mask = ImageOps.invert(mask)
    glow_layer = Image.composite(Image.new('RGBA', mask.size, (0,0,0,0)), glow_layer, inverted_mask)
    return glow_layer

def apply_texture_fill(text_mask, texture_image, texture_scale, text_width, text_height):
    if texture_image is None:
        return Image.new('RGBA', (text_width, text_height), (0,0,0,0))
    texture = texture_image[0].cpu().numpy()
    texture = (texture * 255).astype(np.uint8)
    texture_img = Image.fromarray(texture).convert('RGBA')
    scaled_width = int(text_width * texture_scale)
    scaled_height = int(text_height * texture_scale)
    texture_img = texture_img.resize((scaled_width, scaled_height), Image.BICUBIC)
    texture_text = Image.new('RGBA', (text_width, text_height), (0,0,0,0))
    for x in range(0, text_width, scaled_width):
        for y in range(0, text_height, scaled_height):
            texture_text.paste(texture_img, (x, y))
    result = Image.new('RGBA', (text_width, text_height), (0,0,0,0))
    result.paste(texture_text, (0,0), text_mask)
    return result

def apply_opacity_gradient(text_img, gradient_type, start_opacity, end_opacity):
    img_data = np.array(text_img)
    h, w = img_data.shape[:2]
    if gradient_type == "linear":
        for x in range(w):
            t = x / w
            alpha = int(255 * (start_opacity + (end_opacity - start_opacity) * t))
            img_data[:, x, 3] = np.clip(img_data[:, x, 3] * (alpha/255), 0, 255)
    elif gradient_type == "radial":
        center_x, center_y = w//2, h//2
        max_dist = math.hypot(center_x, center_y)
        for y in range(h):
            for x in range(w):
                dist = math.hypot(x - center_x, y - center_y)
                t = dist / max_dist if max_dist > 0 else 0
                t = max(0, min(1, t))
                alpha = int(255 * (start_opacity + (end_opacity - start_opacity) * t))
                img_data[y, x, 3] = np.clip(img_data[y, x, 3] * (alpha/255), 0, 255)
    return Image.fromarray(img_data)


class lay_text_sum_mul:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "original_coordinate": ("STRING", {
                    "multiline": True,
                    "default": '[{"x":0.5,"y":0.5,"index":1}, {"x":0.6,"y":0.6,"index":2}, {"x":0.7,"y":0.7,"index":3}]',
                    "placeholder": "Input single coordinate JSON or array, e.g. [{\"x\":0.5,\"y\":0.5,\"index\":1}, {...}]"
                }),
                "index_replace": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Input rules line by line, e.g.\n1: hello\n2：world\n3:  11"
                }),
                "font_file": (file_list, {"default": file_list[0] if file_list else ""}),
                "font_size": ("INT", {"default": 12, "min": 0, "max": 320, "step": 1}),
                "arrangement": (["horizontal", "vertical"], {"default": "horizontal"}),
                "text_rotation": ("INT", {"default": 0, "min": -180, "max": 180, "step": 1}),
                "text_color": ("STRING", {"default": "#ffffff"}),
                "text_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "GradientAndStroke": ("GRADIENT_AND_STROKE_DATA",),
                "ShadowAndGlow": ("SHADOW_AND_GLOW_DATA",),
                "EmbossAndFill": ("EMBOSS_AND_FILL_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_text"
    CATEGORY = "Apt_Preset/imgEffect"

    def parse_replacement_rules(self, rules_text):
        replacement_map = {}
        lines = [line.strip() for line in rules_text.split("\n") if line.strip()]
        pattern = re.compile(r"^(\d+)\s*[:：]\s*(.*)$")
        
        for line_num, line in enumerate(lines, 1):
            match = pattern.match(line)
            if not match:
                raise ValueError(f"Invalid format in line {line_num}: '{line}' (expected: index:replacement)")
            
            index_str, replacement_text = match.groups()
            try:
                index = int(index_str)
                replacement_map[index] = replacement_text.strip()
            except ValueError:
                raise ValueError(f"Invalid index in line {line_num}: '{index_str}' (must be integer)")
        
        return replacement_map

    def replace_index_in_coordinates(self, original_coordinate, index_replace):
        try:
            replacement_map = self.parse_replacement_rules(index_replace)
            if not replacement_map:
                raise ValueError("No valid replacement rules found")

            coord_data = json.loads(original_coordinate)

            def replace_item(item):
                if isinstance(item, dict) and "index" in item:
                    current_index = item["index"]
                    if current_index in replacement_map:
                        item["text"] = replacement_map[current_index]
                        del item["index"]

            if isinstance(coord_data, list):
                for item in coord_data:
                    replace_item(item)
            elif isinstance(coord_data, dict):
                replace_item(coord_data)
            else:
                raise ValueError("JSON data must be a single coordinate object or array")

            return coord_data
        
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON parsing failed: {str(e)}")
        except Exception as e:
            raise ValueError(f"Processing failed: {str(e)}")

    def parse_text_coordinates(self, coord_data):
        result = []
        if not isinstance(coord_data, list):
            coord_data = [coord_data]
        
        for item in coord_data:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid item type: {type(item)}, expected dict")
            norm_x = float(item.get("x", 0.5))
            norm_y = float(item.get("y", 0.5))
            text = str(item.get("text", ""))
            if text.strip() == "":
                continue
            result.append((norm_x, norm_y, text))
        
        if not result:
            raise ValueError("No valid text coordinates found")
        return result


    def apply_text(self, image, original_coordinate, index_replace, font_file, font_size, arrangement, text_rotation,
                text_color, text_opacity, GradientAndStroke=None, ShadowAndGlow=None, EmbossAndFill=None):
        use_gradient = False
        start_color = "#ff0000"
        end_color = "#0000ff"
        angle = 0
        opacity_gradient_type = "none"
        opacity_start = 1.0
        opacity_end = 0.5
        stroke_width = 0
        stroke_color = "#000000"
        stroke_opacity = 1.0
        stroke_type = "normal"
        second_stroke_color = "#ffff00"
        second_stroke_width = 0

        shadow_x = 0
        shadow_y = 0
        shadow_color = "#000000"
        shadow_opacity = 1.0
        shadow_blur = 0
        shadow_type = "single"
        multi_shadow_count = 3
        gradient_shadow_start = "#000000"
        gradient_shadow_end = "#444444"
        outer_glow_color = "#ffffff"
        outer_glow_opacity = 0.8
        outer_glow_radius = 0
        inner_glow_color = "#ffffff"
        inner_glow_opacity = 0.8
        inner_glow_radius = 0

        texture_image = None
        use_texture_fill = False
        texture_scale = 1.0
        emboss_effect = False
        emboss_depth = 2
        emboss_light_color = "#ffffff"
        emboss_shadow_color = "#000000"

        if GradientAndStroke is not None and len(GradientAndStroke) >= 13:
            use_gradient, start_color, end_color, angle, opacity_gradient_type, opacity_start, opacity_end, \
            stroke_width, stroke_color, stroke_opacity, stroke_type, second_stroke_color, second_stroke_width = GradientAndStroke

        if ShadowAndGlow is not None and len(ShadowAndGlow) >= 15:
            shadow_x, shadow_y, shadow_color, shadow_opacity, shadow_blur, shadow_type, multi_shadow_count, \
            gradient_shadow_start, gradient_shadow_end, outer_glow_color, outer_glow_opacity, outer_glow_radius, \
            inner_glow_color, inner_glow_opacity, inner_glow_radius = ShadowAndGlow

        if EmbossAndFill is not None and len(EmbossAndFill) >= 7:
            texture_image, use_texture_fill, texture_scale, emboss_effect, emboss_depth, \
            emboss_light_color, emboss_shadow_color = EmbossAndFill

        img = image[0].cpu().numpy()
        img = (img * 255).astype(np.uint8)
        base_img = Image.fromarray(img).convert('RGBA')
        img_width, img_height = base_img.size

        try:
            replaced_coord_data = self.replace_index_in_coordinates(original_coordinate, index_replace)
            text_coord_list = self.parse_text_coordinates(replaced_coord_data)
        except ValueError as e:
            print(f"Text coordinates parse error: {e}")
            return (image,)

        if not font_file or font_file not in FONT_MAP:
            font = ImageFont.load_default()
        else:
            font_path = FONT_MAP[font_file]
            try:
                font = ImageFont.truetype(font_path, font_size)
            except IOError:
                font = ImageFont.load_default()

        text_color_rgb = hex_to_rgb(text_color)
        stroke_color_rgb = hex_to_rgb(stroke_color)
        second_stroke_color_rgb = hex_to_rgb(second_stroke_color)
        shadow_color_rgb = None if shadow_color == "none" else hex_to_rgb(shadow_color)
        start_color_rgb = hex_to_rgb(start_color)
        end_color_rgb = hex_to_rgb(end_color)
        outer_glow_color_rgb = hex_to_rgb(outer_glow_color)
        inner_glow_color_rgb = hex_to_rgb(inner_glow_color)
        emboss_light_rgb = hex_to_rgb(emboss_light_color)
        emboss_shadow_rgb = hex_to_rgb(emboss_shadow_color)

        for norm_x, norm_y, text in text_coord_list:
            if text.strip() == "":
                continue

            if arrangement == "vertical":
                text = '\n'.join(list(text.replace('\n', '')))

            temp_img = Image.new('RGBA', (1, 1))
            temp_draw = ImageDraw.Draw(temp_img)
            bbox = temp_draw.textbbox((0, 0), text, font=font)
            text_left = bbox[0]
            text_top = bbox[1]
            text_right = bbox[2]
            text_bottom = bbox[3]
            text_w = text_right - text_left
            text_h = text_bottom - text_top

            text_canvas = Image.new('RGBA', (text_w, text_h), (0,0,0,0))
            text_draw = ImageDraw.Draw(text_canvas)
            tx = -text_left
            ty = -text_top

            if shadow_color_rgb and (shadow_x != 0 or shadow_y != 0 or shadow_blur > 0):
                shadow_layer = Image.new('RGBA', text_canvas.size, (0,0,0,0))
                shadow_draw = ImageDraw.Draw(shadow_layer)
                if shadow_type == "single":
                    shadow_draw.text((tx + shadow_x, ty + shadow_y), text, font=font,
                                fill=shadow_color_rgb + (int(255 * shadow_opacity),))
                elif shadow_type == "multi":
                    for i in range(multi_shadow_count):
                        alpha = shadow_opacity * (1 - i/multi_shadow_count)
                        ox = shadow_x * (i+1) / multi_shadow_count
                        oy = shadow_y * (i+1) / multi_shadow_count
                        shadow_draw.text((int(tx+ox), int(ty+oy)), text, font=font,
                                    fill=shadow_color_rgb + (int(255*alpha),))
                elif shadow_type == "gradient":
                    grad_start = hex_to_rgb(gradient_shadow_start)
                    grad_end = hex_to_rgb(gradient_shadow_end)
                    grad_img = create_gradient(text_canvas.width, text_canvas.height, grad_start, grad_end, angle)
                    shadow_mask = Image.new('L', text_canvas.size, 0)
                    ImageDraw.Draw(shadow_mask).text((tx+shadow_x, ty+shadow_y), text, font=font, fill=255)
                    shadow_layer = Image.composite(grad_img, shadow_layer, shadow_mask)
                if shadow_blur > 0:
                    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
                text_canvas = Image.alpha_composite(text_canvas, shadow_layer)

            if outer_glow_radius > 0:
                glow_mask = Image.new('L', text_canvas.size, 0)
                ImageDraw.Draw(glow_mask).text((tx, ty), text, font=font, fill=255)
                outer_glow = create_outer_glow(glow_mask, outer_glow_color_rgb, outer_glow_opacity, outer_glow_radius)
                text_canvas = Image.alpha_composite(text_canvas, outer_glow)

            # 核心修复：初始化 stroke_layer，避免未定义错误
            stroke_layer = None
            if stroke_width > 0:
                stroke_layer = Image.new('RGBA', text_canvas.size, (0,0,0,0))
                stroke_draw = ImageDraw.Draw(stroke_layer)
                
                if stroke_type == "normal":
                    for dx in range(-stroke_width, stroke_width+1):
                        for dy in range(-stroke_width, stroke_width+1):
                            if dx != 0 or dy != 0:
                                stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                            fill=stroke_color_rgb + (int(255 * stroke_opacity),))
                elif stroke_type == "smooth":
                    stroke_draw.text((tx, ty), text, font=font,
                                fill=stroke_color_rgb + (int(255 * stroke_opacity),))
                    stroke_layer = stroke_layer.filter(ImageFilter.GaussianBlur(radius=1))
                elif stroke_type == "double":
                    if second_stroke_width > 0:
                        # 绘制第二层描边
                        for dx in range(-second_stroke_width, second_stroke_width+1):
                            for dy in range(-second_stroke_width, second_stroke_width+1):
                                if dx != 0 or dy != 0:
                                    stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                                fill=second_stroke_color_rgb + (int(255 * stroke_opacity),))
                        # 绘制第一层描边
                        for dx in range(-stroke_width, stroke_width+1):
                            for dy in range(-stroke_width, stroke_width+1):
                                if dx != 0 or dy != 0:
                                    stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                                fill=stroke_color_rgb + (int(255 * stroke_opacity),))
                    else:
                        # 无第二层描边时，降级为 normal 描边
                        for dx in range(-stroke_width, stroke_width+1):
                            for dy in range(-stroke_width, stroke_width+1):
                                if dx != 0 or dy != 0:
                                    stroke_draw.text((tx+dx, ty+dy), text, font=font,
                                                fill=stroke_color_rgb + (int(255 * stroke_opacity),))
            
            # 修复：仅当 stroke_layer 有效时才进行合成
            if stroke_layer is not None:
                text_canvas = Image.alpha_composite(text_canvas, stroke_layer)

            text_main_layer = Image.new('RGBA', text_canvas.size, (0,0,0,0))
            text_main_draw = ImageDraw.Draw(text_main_layer)
            
            if emboss_effect:
                light_color = emboss_light_rgb + (int(255*text_opacity),)
                shadow_color = emboss_shadow_rgb + (int(255*text_opacity),)
                text_main_draw.text((tx - emboss_depth, ty - emboss_depth), text, font=font, fill=light_color)
                text_main_draw.text((tx + emboss_depth, ty + emboss_depth), text, font=font, fill=shadow_color)
                text_main_draw.text((tx, ty), text, font=font, fill=(0,0,0,int(255*text_opacity)))
            elif use_texture_fill and texture_image is not None:
                text_mask = Image.new('L', text_canvas.size, 0)
                ImageDraw.Draw(text_mask).text((tx, ty), text, font=font, fill=255)
                texture_fill = apply_texture_fill(text_mask, texture_image, texture_scale, text_w, text_h)
                texture_data = np.array(texture_fill)
                texture_data[..., 3] = (texture_data[..., 3] * text_opacity).astype(np.uint8)
                texture_fill = Image.fromarray(texture_data)
                text_main_layer.paste(texture_fill, (0, 0), texture_fill)
            elif use_gradient:
                grad_img = create_gradient(text_w, text_h, start_color_rgb, end_color_rgb, angle)
                text_mask = Image.new('L', text_canvas.size, 0)
                ImageDraw.Draw(text_mask).text((tx, ty), text, font=font, fill=255)
                grad_text = Image.new('RGBA', text_canvas.size, (0,0,0,0))
                grad_text.paste(grad_img, (0,0), text_mask)
                grad_data = np.array(grad_text)
                grad_data[..., 3] = (grad_data[..., 3] * text_opacity).astype(np.uint8)
                grad_text = Image.fromarray(grad_data)
                if opacity_gradient_type != "none":
                    grad_text = apply_opacity_gradient(grad_text, opacity_gradient_type, opacity_start, opacity_end)
                text_main_layer.paste(grad_text, (0, 0), grad_text)
            else:
                text_main_draw.text((tx, ty), text, font=font, fill=text_color_rgb + (int(255 * text_opacity),))
                if opacity_gradient_type != "none":
                    temp_text = Image.new('RGBA', text_main_layer.size, (0,0,0,0))
                    temp_draw = ImageDraw.Draw(temp_text)
                    temp_draw.text((tx, ty), text, font=font, fill=text_color_rgb + (int(255 * text_opacity),))
                    temp_text = apply_opacity_gradient(temp_text, opacity_gradient_type, opacity_start, opacity_end)
                    text_mask = Image.new('L', text_main_layer.size, 0)
                    ImageDraw.Draw(text_mask).text((tx, ty), text, font=font, fill=255)
                    text_main_layer = Image.composite(temp_text, text_main_layer, text_mask)

            text_canvas = Image.alpha_composite(text_canvas, text_main_layer)

            if inner_glow_radius > 0:
                glow_mask = Image.new('L', text_canvas.size, 0)
                ImageDraw.Draw(glow_mask).text((tx, ty), text, font=font, fill=255)
                inner_glow = create_inner_glow(glow_mask, inner_glow_color_rgb, inner_glow_opacity, inner_glow_radius)
                text_canvas = Image.alpha_composite(text_canvas, inner_glow)

            if text_rotation != 0:
                text_canvas = text_canvas.rotate(text_rotation, expand=True, resample=Image.BICUBIC)

            center_x = img_width * norm_x
            center_y = img_height * norm_y
            rotated_w, rotated_h = text_canvas.size
            paste_x = center_x - (rotated_w / 2)
            paste_y = center_y - (rotated_h / 2)
            
            paste_x = max(0.0, min(paste_x, img_width - rotated_w))
            paste_y = max(0.0, min(paste_y, img_height - rotated_h))
            
            base_img.paste(text_canvas, (round(paste_x), round(paste_y)), text_canvas)

        result = np.array(base_img).astype(np.float32) / 255.0
        result = torch.from_numpy(result)[None,]
        return (result,)



#endregion--------------------------------------------------------









