# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import torch
import folder_paths
from PIL import Image, ImageDraw, ImageFont
import hashlib
import re
from ..main_unit import *


try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  




class Coordinate_MarkRender:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "fonts")
        self.file_list = [f for f in os.listdir(self.font_dir) if f.endswith(('.ttf', '.otf'))]
        self.default_font = self.file_list[0] if self.file_list else None
        self.color_mapping = {
            "black": (0, 0, 0),
            "white": (255, 255, 255),
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
        }

    @classmethod
    def INPUT_TYPES(cls):
        inst = cls()
        file_list = inst.file_list
        color_options = list(inst.color_mapping.keys())
        return {
            "required": {
                "image": ("IMAGE",),
                "coordinates": ("STRING", {"default": "[]", "multiline": True}),
                "base_size": ("INT", {"default": 32, "min": 8, "max": 256, "step": 2}),
                "bg_color": (color_options, {"default": "red"}),
                "text_size": ("INT", {"default": 24, "min": 8, "max": 128, "step": 2}),
                "text_color": (color_options, {"default": "white"}),
                "font": (file_list, {"default": inst.default_font if inst.default_font else ""}),
                "border_color": (color_options, {"default": "white"}),
            },
            "optional": {
                "border_width": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                "padding": ("INT", {"default": 4, "min": 0, "max": 32, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("marked_image",)
    FUNCTION = "mark_image"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    OUTPUT_NODE = False

    def load_font(self, font_name, font_size):
        if not font_name or not os.path.exists(os.path.join(self.font_dir, font_name)):
            return ImageFont.load_default(size=font_size)
        try:
            return ImageFont.truetype(os.path.join(self.font_dir, font_name), font_size)
        except:
            return ImageFont.load_default(size=font_size)

    def get_adaptive_box_size(self, text, font_obj, base_size, padding):
        bbox = ImageDraw.Draw(Image.new('RGB', (1,1))).textbbox((0, 0), text, font=font_obj)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        text_offset_x = bbox[0]
        text_offset_y = bbox[1]
        required_width = text_width + 2 * padding
        required_height = text_height + 2 * padding
        box_width = max(base_size, required_width)
        box_height = max(base_size, required_height)
        half_w = box_width // 2
        half_h = box_height // 2
        return half_w, half_h, text_width, text_height, text_offset_x, text_offset_y

    def mark_image(self, image, coordinates, base_size=32, text_size=24, font="",
                   bg_color="red", text_color="white", border_color="white",
                   border_width=2, padding=4):
        try:
            points = json.loads(coordinates)
            valid_points = []
            for p in points:
                if not isinstance(p, dict) or "x" not in p or "y" not in p:
                    continue
                text = p.get("text", p.get("index", ""))
                if text == "":
                    continue
                valid_points.append({
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "text": str(text)
                })
            points = valid_points
        except:
            points = []
        img_np = np.clip(image.cpu().numpy()[0] * 255, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np).convert("RGB")
        img_width, img_height = pil_img.size
        draw = ImageDraw.Draw(pil_img)
        font_obj = self.load_font(font, text_size)
        bg_rgb = self.color_mapping[bg_color]
        text_rgb = self.color_mapping[text_color]
        border_rgb = self.color_mapping[border_color]
        for point in points:
            center_x = int(point["x"] * img_width)
            center_y = int(point["y"] * img_height)
            text = point["text"]
            half_w, half_h, text_w, text_h, text_off_x, text_off_y = self.get_adaptive_box_size(
                text, font_obj, base_size, padding
            )
            draw.rectangle(
                [center_x - half_w, center_y - half_h,
                 center_x + half_w, center_y + half_h],
                fill=bg_rgb,
                outline=border_rgb,
                width=border_width
            )
            text_center_x = center_x - (text_w / 2) - text_off_x
            text_center_y = center_y - (text_h / 2) - text_off_y
            draw.text((text_center_x + 1, text_center_y + 1), text, fill=(0, 0, 0), font=font_obj)
            draw.text((text_center_x, text_center_y), text, fill=text_rgb, font=font_obj)
        img_array = np.array(pil_img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        return (img_tensor,)




class Coordinate_fromImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "points_data": ("STRING", {"default": "[{\"x\":0.5,\"y\":0.5,\"index\":1}]", "multiline": False}),
            },
        }
    
    NAME = "Coordinate_fromImage"
    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("ORIGINAL_IMAGE", "coordinates")
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    OUTPUT_NODE = True
    
    def process(self, image=None, points_data="[{\"x\":0.5,\"y\":0.5,\"index\":1}]"):
        if image is None:
            default_image = torch.zeros((1, 512, 512, 3), dtype=torch.float32)
            image = default_image
        
        img_np = np.clip(image.cpu().numpy()[0] * 255, 0, 255).astype(np.uint8)
        pil_image = Image.fromarray(img_np).convert("RGB")
        original_image = pil_image.copy()
        img_width, img_height = pil_image.size
        
        try:
            points = json.loads(points_data)
            if not isinstance(points, list):
                points = []
        except json.JSONDecodeError:
            points = []
        
        if not points:
            center_point = {
                "x": 0.5,
                "y": 0.5,
                "index": 1
            }
            points = [center_point]
        
        if points and len(points) > 0:
            pil_image = self._draw_markers(pil_image, points, img_width, img_height)
        
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        original_array = np.array(original_image).astype(np.float32) / 255.0
        original_tensor = torch.from_numpy(original_array).unsqueeze(0)
        
        coordinates_json = json.dumps(points, ensure_ascii=False)
        print(f"Coordinate_fromImage 输出坐标: {coordinates_json}")
        return (original_tensor, coordinates_json)
    
    def _draw_markers(self, image, points, img_width, img_height):
        marker_color = (255, 69, 0, 230)
        text_rgb = (255, 255, 255)
        if image.mode != 'RGBA':
            result_image = image.convert('RGBA')
        else:
            result_image = image.copy()
        overlay = Image.new('RGBA', result_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        marker_radius = max(14, min(img_width, img_height) // 40)
        outline_width = max(2, marker_radius // 10)
        outline_color = (255, 255, 255, 230)
        for point in points:
            rel_x = point.get("x", 0)
            rel_y = point.get("y", 0)
            index = point.get("index", 1)
            abs_x = int(rel_x * img_width)
            abs_y = int(rel_y * img_height)
            draw.ellipse(
                [
                    abs_x - marker_radius - outline_width,
                    abs_y - marker_radius - outline_width,
                    abs_x + marker_radius + outline_width,
                    abs_y + marker_radius + outline_width
                ],
                fill=outline_color
            )
            draw.ellipse(
                [
                    abs_x - marker_radius,
                    abs_y - marker_radius,
                    abs_x + marker_radius,
                    abs_y + marker_radius
                ],
                fill=marker_color
            )
            self._draw_number_geometric(draw, abs_x, abs_y, index, marker_radius, text_rgb)
        result_image = Image.alpha_composite(result_image, overlay)
        return result_image.convert('RGB')
    
    def _draw_number_geometric(self, draw, cx, cy, number, radius, color):
        num_str = str(number)
        digit_height = int(radius * 1.0)
        digit_width = int(radius * 0.55)
        spacing = int(radius * 0.15)
        line_width = max(2, radius // 5)
        total_width = len(num_str) * digit_width + (len(num_str) - 1) * spacing
        start_x = cx - total_width // 2
        for i, digit_char in enumerate(num_str):
            digit = int(digit_char)
            digit_cx = start_x + i * (digit_width + spacing) + digit_width // 2
            self._draw_single_digit(draw, digit_cx, cy, digit, digit_width, digit_height, line_width, color)
    
    def _draw_single_digit(self, draw, cx, cy, digit, width, height, line_width, color):
        hw = width // 2
        hh = height // 2
        gap = line_width // 2
        segments = {
            'a': [(cx - hw + gap, cy - hh), (cx + hw - gap, cy - hh)],
            'b': [(cx + hw, cy - hh + gap), (cx + hw, cy - gap)],
            'c': [(cx + hw, cy + gap), (cx + hw, cy + hh - gap)],
            'd': [(cx - hw + gap, cy + hh), (cx + hw - gap, cy + hh)],
            'e': [(cx - hw, cy + gap), (cx - hw, cy + hh - gap)],
            'f': [(cx - hw, cy - hh + gap), (cx - hw, cy - gap)],
            'g': [(cx - hw + gap, cy), (cx + hw - gap, cy)],
        }
        digit_segments = {
            0: ['a', 'b', 'c', 'd', 'e', 'f'],
            1: ['b', 'c'],
            2: ['a', 'b', 'g', 'e', 'd'],
            3: ['a', 'b', 'g', 'c', 'd'],
            4: ['f', 'g', 'b', 'c'],
            5: ['a', 'f', 'g', 'c', 'd'],
            6: ['a', 'f', 'e', 'd', 'c', 'g'],
            7: ['a', 'b', 'c'],
            8: ['a', 'b', 'c', 'd', 'e', 'f', 'g'],
            9: ['a', 'b', 'c', 'd', 'f', 'g'],
        }
        for seg_name in digit_segments.get(digit, []):
            if seg_name in segments:
                start, end = segments[seg_name]
                draw.line([start, end], fill=color, width=line_width)
                r = line_width // 2
                draw.ellipse([start[0]-r, start[1]-r, start[0]+r, start[1]+r], fill=color)
                draw.ellipse([end[0]-r, end[1]-r, end[0]+r, end[1]+r], fill=color)
    
    @classmethod
    def IS_CHANGED(cls, image, points_data="[]", **kwargs):
        return f"{points_data}"
    
    @classmethod
    def VALIDATE_INPUTS(cls, image=None, points_data="[{\"x\":0.5,\"y\":0.5,\"index\":1}]", **kwargs):
        try:
            points = json.loads(points_data)
            if not isinstance(points, list):
                return "坐标数据必须是数组格式"
        except json.JSONDecodeError:
            return "坐标数据格式错误，请输入有效的JSON格式"
        return True



class Coordinate_pointCombine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "coordinates1": ("STRING", {"default": "[]", "multiline": False}),
                "coordinates2": ("STRING", {"default": "[]", "multiline": False}),
                "coordinates3": ("STRING", {"default": "[]", "multiline": False}),
                "coordinates4": ("STRING", {"default": "[]", "multiline": False}),
                "coordinates5": ("STRING", {"default": "[]", "multiline": False}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "combine_coordinates"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    
    def combine_coordinates(self, coordinates1="[]", coordinates2="[]", coordinates3="[]", coordinates4="[]", coordinates5="[]"):
        all_points = []
        max_index = 0  # 缓存最大index，避免重复遍历
        
        coord_inputs = [coordinates1, coordinates2, coordinates3, coordinates4, coordinates5]
        
        for coord_str in coord_inputs:
            try:
                points = json.loads(coord_str)
                if isinstance(points, list):
                    for point in points:
                        if isinstance(point, dict) and "x" in point and "y" in point:
                            # 自动补充index
                            if "index" not in point:
                                max_index += 1
                                point["index"] = max_index
                            else:
                                # 更新最大index（如果手动指定了更大的值）
                                current_idx = point["index"]
                                if isinstance(current_idx, int) and current_idx > max_index:
                                    max_index = current_idx
                                
                            all_points.append(point)
            except json.JSONDecodeError:
                continue
        
        # 按index排序并重新分配连续index
        all_points.sort(key=lambda x: x.get("index", 0))
        for i, point in enumerate(all_points):
            point["index"] = i + 1
        
        combined_coordinates = json.dumps(all_points, ensure_ascii=False)
        return (combined_coordinates,)
    
    @classmethod
    def IS_CHANGED(cls, coordinates1="[]", coordinates2="[]", coordinates3="[]", coordinates4="[]", coordinates5="[]", **kwargs):
        # 改用哈希避免特殊字符导致的误判
        combined = f"{coordinates1}{coordinates2}{coordinates3}{coordinates4}{coordinates5}".encode('utf-8')
        return hashlib.md5(combined).hexdigest()
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True



class Coordinate_Generator:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "output_count": ("INT", {
                    "default": 5, 
                    "min": 1, 
                    "max": 5, 
                    "step": 1,
                    "tooltip": "输出的坐标点数量 (1-5)"
                })
            }
        }
        for i in range(1, 6):
            inputs["required"][f"x{i}"] = ("FLOAT", {
                "default": 0.0, 
                "min": 0.0, 
                "max": 1.0, 
                "step": 0.01,
                "tooltip": f"点{i}的X坐标 (0.0-1.0)"
            })
            inputs["required"][f"y{i}"] = ("FLOAT", {
                "default": 0.0, 
                "min": 0.0, 
                "max": 1.0, 
                "step": 0.01,
                "tooltip": f"点{i}的Y坐标 (0.0-1.0)"
            })
        return inputs
    
    NAME = "Coordinate Input 5"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("coordinates",)
    FUNCTION = "generate_coordinates"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    
    def generate_coordinates(self, **kwargs):
        output_count = kwargs.get("output_count", 5)
        output_count = max(1, min(5, output_count))
        
        points = []
        for i in range(1, output_count + 1):
            x_key = f"x{i}"
            y_key = f"y{i}"
            x = max(0.0, min(1.0, kwargs.get(x_key, 0.0)))
            y = max(0.0, min(1.0, kwargs.get(y_key, 0.0)))
            points.append({
                "x": x,
                "y": y,
                "index": i
            })
        coordinates = json.dumps(points)
        return (coordinates,)
    
    @classmethod
    def IS_CHANGED(cls,** kwargs):
        values = []
        values.append(str(kwargs.get("output_count", 5)))
        for i in range(1, 6):
            values.append(str(kwargs.get(f"x{i}", 0.0)))
            values.append(str(kwargs.get(f"y{i}", 0.0)))
        return "_".join(values)
    
    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        output_count = kwargs.get("output_count", 5)
        if not (1 <= output_count <= 5):
            return f"输出数量必须在1到5之间"
        
        for i in range(1, 6):
            x = kwargs.get(f"x{i}", 0.0)
            y = kwargs.get(f"y{i}", 0.0)
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return f"坐标值必须在0.0到1.0之间 (点{i})"
        return True
    





class Coordinate_Index2Text:
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    FUNCTION = "replace_index"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_Coordinate",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
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
            },
            "optional": {}
        }

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

    def replace_index(self, original_coordinate, index_replace):
        try:
            replacement_map = self.parse_replacement_rules(index_replace)
            if not replacement_map:
                raise ValueError("No valid replacement rules found")

            coord_data = json.loads(original_coordinate)

            def replace_item(item):
                if isinstance(item, dict) and "index" in item:
                    current_index = item["index"]
                    if current_index in replacement_map:
                        item["index"] = replacement_map[current_index]

            if isinstance(coord_data, list):
                for item in coord_data:
                    replace_item(item)
            elif isinstance(coord_data, dict):
                replace_item(coord_data)
            else:
                raise ValueError("JSON data must be a single coordinate object or array")

            result_json = json.dumps(coord_data, ensure_ascii=False, indent=2)
            return (result_json,)

        except json.JSONDecodeError as e:
            return (f"JSON parsing failed: {str(e)}",)
        except Exception as e:
            return (f"Processing failed: {str(e)}",)




from collections import namedtuple
BoundingBox = namedtuple('BoundingBox', ['x', 'y', 'width', 'height'])



class Coordinate_fromMask:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # ✅ 核心修改1：彻底删除 image 输入项
                "mask": ("MASK",),
                "coordinate_align": ([ "top_left", "top_right", "bottom_left", "bottom_right","center", "top_center", "bottom_center", "left_center", "right_center"], {
                    "default": "center",
                }),
                "index": ("INT", {"default": 0, "min": 0, "max": 99, "step": 1}),
                "ignore_threshold": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 1}),
            },
            "optional": {
            }
        }
    
    RETURN_TYPES = ("STRING", "BOUNDING_BOX", "FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("coordinates", "bbox", "norm_x", "norm_y", "px_x", "px_y")
    FUNCTION = "extract_coordinates"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    
    # ✅ 核心修改2：函数入参 删除 image=None
    def extract_coordinates(self, mask, coordinate_align, index=0, ignore_threshold=0):
        try:
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(np.array(mask))
            
            opencv_gray_image = self.tensorMask2cv2img(mask)
            _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= ignore_threshold:
                    valid_contours.append(contour)
            
            if not valid_contours:
                default_point = {
                    "x": 0.0,
                    "y": 0.0,
                    "index": 1
                }
                coordinates_json = json.dumps([default_point], ensure_ascii=False)
                bbox = BoundingBox(0, 0, 0, 0)
                return (coordinates_json, bbox, 0.0, 0.0, 0, 0)
            
            if index >= len(valid_contours):
                index = len(valid_contours) - 1
            
            contour = valid_contours[index]
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # ✅ 核心逻辑保留：mask的尺寸就是原图尺寸，直接用mask的宽高即可
            height, width = mask.shape[-2:]
            
            # ✅ 所有坐标对齐逻辑完全保留，无修改
            if coordinate_align == "center":
                center_y = y + h / 2.0
                center_x = x + w / 2.0
                x_norm = center_x / width
                y_norm = center_y / height
                px_x = int(center_x)
                px_y = int(center_y)
                
            elif coordinate_align == "top_left":
                x_norm = x / width
                y_norm = y / height
                px_x = x
                px_y = y
                
            elif coordinate_align == "top_right":
                x_norm = (x + w) / width
                y_norm = y / height
                px_x = x + w
                px_y = y
                
            elif coordinate_align == "bottom_left":
                x_norm = x / width
                y_norm = (y + h) / height
                px_x = x
                px_y = y + h
                
            elif coordinate_align == "bottom_right":
                x_norm = (x + w) / width
                y_norm = (y + h) / height
                px_x = x + w
                px_y = y + h
                
            elif coordinate_align == "top_center":
                center_x = x + w / 2.0
                x_norm = center_x / width
                y_norm = y / height
                px_x = int(center_x)
                px_y = y
                
            elif coordinate_align == "bottom_center":
                center_x = x + w / 2.0
                x_norm = center_x / width
                y_norm = (y + h) / height
                px_x = int(center_x)
                px_y = y + h
                
            elif coordinate_align == "left_center":
                center_y = y + h / 2.0
                x_norm = x / width
                y_norm = center_y / height
                px_x = x
                px_y = int(center_y)
                
            elif coordinate_align == "right_center":
                center_y = y + h / 2.0
                x_norm = (x + w) / width
                y_norm = center_y / height
                px_x = x + w
                px_y = int(center_y)
                
            else:
                center_y = y + h / 2.0
                center_x = x + w / 2.0
                x_norm = center_x / width
                y_norm = center_y / height
                px_x = int(center_x)
                px_y = int(center_y)
            
            # 归一化坐标边界限制 0~1
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            # ✅ 核心修改3：彻底删除所有 image 相关的判断分支，全部使用 mask 的宽高做像素坐标边界限制
            px_x = max(0, min(px_x, width - 1))
            px_y = max(0, min(px_y, height - 1))
            
            index_mapping = {
                "top_left": 1,
                "top_center": 2,
                "top_right": 3,
                "left_center": 4,
                "center": 5,
                "right_center": 6,
                "bottom_left": 7,
                "bottom_center": 8,
                "bottom_right": 9
            }
            
            point_data = {
                "x": float(x_norm),
                "y": float(y_norm),
                "index": index_mapping.get(coordinate_align, 5)
            }
            coordinates_json = json.dumps([point_data], ensure_ascii=False)
            
            bbox = BoundingBox(int(x), int(y), int(w), int(h))
                 
            return (coordinates_json, bbox, float(x_norm), float(y_norm), px_x, px_y)
            
        except Exception as e:
            print(f"遮罩坐标提取错误: {e}")
            default_point = {
                "x": 0.0,
                "y": 0.0,
                "index": 1
            }
            coordinates_json = json.dumps([default_point], ensure_ascii=False)
            bbox = BoundingBox(0, 0, 0, 0)
            return (coordinates_json, bbox, 0.0, 0.0, 0, 0)
    
    def tensorMask2cv2img(self, tensor) -> np.ndarray:   
        tensor = tensor.cpu().squeeze(0)
        array = tensor.numpy()
        array = (array * 255).astype(np.uint8)
        return array





class Coordinate_SplitIndex:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "Mark_image": ("IMAGE",), 
                "coordinates": ("STRING",{"forceInput": True}),
                "index": ("INT", {"default": 1, "min": 1, "step": 1}),     
            },
            "optional": {

            }
        }
    
    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("norm_x", "norm_y", "px_x", "px_y")
    FUNCTION = "get_coordinates"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    
    def get_coordinates(self, coordinates, index, Mark_image=None):  
        try:
            points = json.loads(coordinates)
            if not points or not isinstance(points, list):
                return (0.0, 0.0, 0, 0)
            
            x_norm, y_norm = 0.0, 0.0
            for point in points:
                if point.get("index") == index:
                    x_norm = float(point.get("x", 0.0))
                    y_norm = float(point.get("y", 0.0))
                    break
            
            x_px, y_px = 0, 0
            if Mark_image is not None:  # 使用新的参数名
                if len(Mark_image.shape) != 4:
                    raise ValueError(f"Invalid image shape: {Mark_image.shape}, expected [batch, H, W, C]")
                
                height = Mark_image.shape[1]
                width = Mark_image.shape[2]
                
                x_px = int(max(0, min(round(x_norm * width), width - 1)))
                y_px = int(max(0, min(round(y_norm * height), height - 1)))
            
            return (x_norm, y_norm, x_px, y_px)
        
        except json.JSONDecodeError:
            return (0.0, 0.0, 0, 0)
        except Exception as e:
            print(f"Coordinate conversion error: {str(e)}")
            return (0.0, 0.0, 0, 0)


class Coordinate_create_mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "x_min": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                "y_min": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001}),
                "x_max": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001}),
                "y_max": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("mask", "image")
    FUNCTION = "create_mask_and_crop"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"

    def _ensure_bhwc(self, image: torch.Tensor) -> torch.Tensor:
        # 兼容单图与非常见输入，统一转为 [B, H, W, C]
        if len(image.shape) == 4:
            return image

        if len(image.shape) == 3:
            # HWC
            if image.shape[-1] in (1, 3, 4):
                return image.unsqueeze(0)
            # CHW -> HWC
            if image.shape[0] in (1, 3, 4):
                return image.permute(1, 2, 0).unsqueeze(0)
            # 兜底按 HWC 处理
            return image.unsqueeze(0)

        if len(image.shape) == 2:
            # HW -> HWC(1) -> B,H,W,C
            return image.unsqueeze(-1).unsqueeze(0)

        raise ValueError(f"Invalid image shape: {image.shape}, expected HW/HWC/CHW/BHWC")

    def create_mask_and_crop(self, image, x_min, y_min, x_max, y_max):
        image = self._ensure_bhwc(image)

        batch, height, width, channels = image.shape
        if channels == 1:
            image = image.repeat(1, 1, 1, 3)

        x0 = max(0.0, min(float(x_min), float(x_max)))
        x1 = min(1.0, max(float(x_min), float(x_max)))
        y0 = max(0.0, min(float(y_min), float(y_max)))
        y1 = min(1.0, max(float(y_min), float(y_max)))

        px0 = int(np.floor(x0 * width))
        py0 = int(np.floor(y0 * height))
        px1 = int(np.ceil(x1 * width))
        py1 = int(np.ceil(y1 * height))

        px0 = max(0, min(px0, width - 1))
        py0 = max(0, min(py0, height - 1))
        px1 = max(px0 + 1, min(px1, width))
        py1 = max(py0 + 1, min(py1, height))

        cropped_image = image[:, py0:py1, px0:px1, :]

        # 遮罩保持原图尺寸，仅在矩形区域内为 1，其余为 0
        full_mask = torch.zeros((batch, height, width), dtype=image.dtype, device=image.device)
        full_mask[:, py0:py1, px0:px1] = 1.0
        return (full_mask, cropped_image)







import re
import json
from typing import List


def robust_extract_bbox(text: str) -> List[List[float]]:
    if not text or not text.strip():
        return []
    pat = re.findall(r'\[([^\[\]]+?)\]', text, re.DOTALL)
    res = []
    for g in pat:
        clean = re.sub(r'[^-0-9,.]', '', g)
        parts = [p.strip() for p in clean.split(",") if p.strip()]
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except:
                continue
        if len(nums) == 4:
            res.append(nums)
    return res


class Bbox_strToBbox:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "index": ("INT", {"default":0, "min":0}),
            },
            "optional": {
                "normalized_str": ("STRING", {"default": "", "forceInput": True}),
                "permil_str": ("STRING", {"default": "", "forceInput": True}),
            }
        }
    
    # 🔥 新增输出：normalized_output
    RETURN_TYPES = ("BOUNDING_BOX", "STRING", "MASK")
    RETURN_NAMES = ("bbox", "normalized_str", "mask")
    
    FUNCTION = "exec"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"

    DESCRIPTION = """
     将边界框字符串转为实际边界框与掩码。
    - normalized_str对角框：含归一化对坐标生成的边界框字符串【手绘生成】
    - permil_str千分比对角框：含千分比对坐标生成的边界框字符串【AI生成】
    - index：默认0，是所有边界框，1是第一个边界框
""" 
    @staticmethod
    def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 4:
            return image
        if len(image.shape) == 3:
            if image.shape[-1] in (1, 3, 4):
                return image.unsqueeze(0)
            if image.shape[0] in (1, 3, 4):
                return image.permute(1, 2, 0).unsqueeze(0)
            return image.unsqueeze(0)
        if len(image.shape) == 2:
            return image.unsqueeze(-1).unsqueeze(0)
        raise ValueError(f"Invalid image shape: {image.shape}")

    def exec(self, image, index, normalized_str="", permil_str=""):
        image = self._ensure_bhwc(image)
        batch, img_h, img_w, _ = image.shape

        boxes = []
        use_normalized = False

        # 优先使用归一化
        norm_boxes = robust_extract_bbox(normalized_str)
        if norm_boxes:
            boxes = norm_boxes
            use_normalized = True
        else:
            # 使用千分比
            permil_boxes = robust_extract_bbox(permil_str)
            if permil_boxes:
                boxes = permil_boxes
                use_normalized = False
            else:
                fallback_bbox = {"x": 0, "y": 0, "width": 100, "height": 100}
                fallback_mask = torch.zeros((batch, img_h, img_w), dtype=image.dtype, device=image.device)
                y2 = min(img_h, 100)
                x2 = min(img_w, 100)
                fallback_mask[:, 0:y2, 0:x2] = 1.0
                return (fallback_bbox, "[[0.0, 0.0, 0.1, 0.1]]", fallback_mask)

        # index 选择
        if index == 0:
            sel = boxes
        else:
            idx = index - 1
            if 0 <= idx < len(boxes):
                sel = [boxes[idx]]
            else:
                sel = [boxes[0]]

        # --------------------------
        # 1. 计算像素框
        # --------------------------
        out = []
        full_mask = torch.zeros((batch, img_h, img_w), dtype=image.dtype, device=image.device)
        for x1,y1,x2,y2 in sel:
            if use_normalized:
                px1 = x1 * img_w
                py1 = y1 * img_h
                px2 = x2 * img_w
                py2 = y2 * img_h
            else:
                px1 = x1 * img_w / 1000
                py1 = y1 * img_h / 1000
                px2 = x2 * img_w / 1000
                py2 = y2 * img_h / 1000

            px1 = max(0, min(int(round(px1)), img_w))
            py1 = max(0, min(int(round(py1)), img_h))
            px2 = max(0, min(int(round(px2)), img_w))
            py2 = max(0, min(int(round(py2)), img_h))

            if px2 < px1:
                px1, px2 = px2, px1
            if py2 < py1:
                py1, py2 = py2, py1

            bw = max(1, px2 - px1)
            bh = max(1, py2 - py1)
            out.append([px1, py1, bw, bh])
            full_mask[:, py1:py2, px1:px2] = 1.0

        # --------------------------
        # 2. 🔥 生成归一化坐标输出（你要的核心）
        # --------------------------
        normalized_result = []
        for x1, y1, x2, y2 in sel:
            if use_normalized:
                # 输入本身就是归一化 → 直接保留
                nx1 = round(x1, 4)
                ny1 = round(y1, 4)
                nx2 = round(x2, 4)
                ny2 = round(y2, 4)
            else:
                # 输入是千分比 → 转换成归一化
                nx1 = round(x1 / 1000, 4)
                ny1 = round(y1 / 1000, 4)
                nx2 = round(x2 / 1000, 4)
                ny2 = round(y2 / 1000, 4)

            # 限制 0~1 范围
            nx1 = max(0.0, min(1.0, nx1))
            ny1 = max(0.0, min(1.0, ny1))
            nx2 = max(0.0, min(1.0, nx2))
            ny2 = max(0.0, min(1.0, ny2))

            normalized_result.append([nx1, ny1, nx2, ny2])

        # 转成字符串输出
        normalized_str_out = json.dumps(normalized_result)

        # --------------------------
        # 与系统 BoundingBox 结构对齐：{x, y, width, height}
        bbox_items = [
            {"x": int(px1), "y": int(py1), "width": int(bw), "height": int(bh)}
            for px1, py1, bw, bh in out
        ]
        bbox_out = bbox_items[0] if len(bbox_items) == 1 else bbox_items
        return (bbox_out, normalized_str_out, full_mask)


class Bbox_BboxToStr:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "bbox": ("BOUNDING_BOX",{"forceInput": True}),
            }
        }
    RETURN_TYPES = ("STRING", "MASK")
    RETURN_NAMES = ("normalized_str", "mask")
    FUNCTION = "exec"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"

    @staticmethod
    def _ensure_bhwc(image: torch.Tensor) -> torch.Tensor:
        if len(image.shape) == 4:
            return image
        if len(image.shape) == 3:
            if image.shape[-1] in (1, 3, 4):
                return image.unsqueeze(0)
            if image.shape[0] in (1, 3, 4):
                return image.permute(1, 2, 0).unsqueeze(0)
            return image.unsqueeze(0)
        if len(image.shape) == 2:
            return image.unsqueeze(-1).unsqueeze(0)
        raise ValueError(f"Invalid image shape: {image.shape}")

    def exec(self, image, bbox):
        image = self._ensure_bhwc(image)
        batch, img_h, img_w, _ = image.shape
        norm_list = []
        permil_list = []
        full_mask = torch.zeros((batch, img_h, img_w), dtype=image.dtype, device=image.device)

        # 兼容系统 BoundingBox 数据结构：dict / 单个 bbox / bbox 列表
        normalized_boxes = []
        if isinstance(bbox, dict):
            normalized_boxes = [[bbox.get("x", 0), bbox.get("y", 0), bbox.get("width", 0), bbox.get("height", 0)]]
        elif hasattr(bbox, "x") and hasattr(bbox, "y") and hasattr(bbox, "width") and hasattr(bbox, "height"):
            normalized_boxes = [[bbox.x, bbox.y, bbox.width, bbox.height]]
        elif isinstance(bbox, (list, tuple)):
            if len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox):
                normalized_boxes = [[bbox[0], bbox[1], bbox[2], bbox[3]]]
            else:
                for item in bbox:
                    if isinstance(item, dict):
                        normalized_boxes.append([item.get("x", 0), item.get("y", 0), item.get("width", 0), item.get("height", 0)])
                    elif hasattr(item, "x") and hasattr(item, "y") and hasattr(item, "width") and hasattr(item, "height"):
                        normalized_boxes.append([item.x, item.y, item.width, item.height])
                    elif isinstance(item, (list, tuple)) and len(item) == 4:
                        normalized_boxes.append([item[0], item[1], item[2], item[3]])

        for box in normalized_boxes:
            x1, y1, w, h = box
            x1 = max(0, min(int(round(x1)), img_w))
            y1 = max(0, min(int(round(y1)), img_h))
            w = max(1, int(round(w)))
            h = max(1, int(round(h)))
            x2 = x1 + w
            y2 = y1 + h
            x2 = max(x1 + 1, min(x2, img_w))
            y2 = max(y1 + 1, min(y2, img_h))

            nx1 = round(x1 / img_w, 4)
            ny1 = round(y1 / img_h, 4)
            nx2 = round(x2 / img_w, 4)
            ny2 = round(y2 / img_h, 4)
            norm_list.append([nx1, ny1, nx2, ny2])

            px1 = int(round(nx1 * 1000))
            py1 = int(round(ny1 * 1000))
            px2 = int(round(nx2 * 1000))
            py2 = int(round(ny2 * 1000))
            permil_list.append([px1, py1, px2, py2])
            full_mask[:, y1:y2, x1:x2] = 1.0

        return (json.dumps(norm_list), full_mask)









