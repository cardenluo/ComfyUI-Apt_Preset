import os
import re
import numpy as np
from typing import Optional, List, Tuple
import os
import glob
import json
import toml 

import unicodedata
from typing import Dict, List
import datetime
import random

import base64
import urllib.parse
import html


try:
    import docx  # 导入整个docx库（备用）
    from docx import Document  
    from docx import Document as docx_Document  
    REMOVER_AVAILABLE = True  
except ImportError:
    docx = None
    Document = None
    docx_Document = None
    REMOVER_AVAILABLE = False


from ..main_unit import *




#region----------------------------------------------------------------------#




#region 镜头视角
LENS_MAP = {
    "None": "",
    "广角镜头": "广角镜头视角（等效焦距16-35mm），视野开阔宏大，景深深邃，边缘畸变自然，适合展现全景场景或空间纵深感",
    "超广角镜头": "超广角镜头视角（等效焦距8-15mm），视野极度宽广，空间拉伸感强，近大远小效果明显，适合狭小空间或震撼全景",
    "俯视镜头": "高空俯视角度拍摄，自上而下垂直/斜向视角，完整展现主体整体布局与环境关系，全局视角清晰",
    "仰视镜头": "低角度仰视拍摄，自下而上仰望视角，突出主体高耸感与压迫感，强化垂直维度视觉冲击",
    "特写镜头": "特写镜头聚焦主体局部（如面部、细节），细节放大突出，主体占据画面80%以上，背景轻微虚化，凸显纹理与质感",
    "大特写镜头": "大特写镜头极致聚焦微小细节（如眼睛、纹理），主体占据画面90%以上，细节纤毫毕现，背景完全虚化",
    "微距镜头": "超近距离微距摄影（放大倍率1:1以上），极致放大微观细节，纹理清晰锐利，色彩还原真实，突出材质肌理与微小结构",
    "近景镜头": "近景拍摄聚焦主体上半身/核心区域，主体突出鲜明，背景适度虚化（浅景深），兼顾主体细节与环境氛围",
    "中景镜头": "中景拍摄展现主体完整形态与周边环境，主体与背景比例协调，既能看清主体动作，又能体现环境关系",
    "远景镜头": "远景全景拍摄，主体与背景协调统一，展现完整场景格局，空间关系明确，氛围感强烈",
    "全景镜头": "全景镜头360度/宽幅覆盖，场景完整无遗漏，空间纵深感与广度兼具，适合宏大场景展现"
}

VIEW_MAP = {
    "None": "",
    "完整四视图": "工程制图标准四视图正交投影，包含前视图、侧视图、后视图、顶视图，比例精确，线条清晰无畸变，尺寸标注规范",
    "完整六视图": "工程制图标准六视图正交投影，包含前/后/左/右/顶/底视图，全方位无死角展示，机械设计标准规范",
    "正面视图": "正射投影正面视图，主体正面完整对称展现，中心构图均衡，结构细节无遮挡，轮廓线条规整",
    "侧面视图": "正射投影侧面视图，主体侧面轮廓清晰分明，深度维度与厚度关系明确，侧视角度无透视变形",
    "背面视图": "正射投影背面视图，主体背部结构完整呈现，后部细节无遗漏，轮廓与接口关系清晰",
    "顶部视图": "正射投影顶部视图，主体俯视结构完整展现，顶部布局与尺寸关系明确，无遮挡视角",
    "底部视图": "正射投影底部视图，主体仰视结构完整展现，底部细节与接口关系清晰，补充顶部视角盲区",
    "半侧面视图": "45度半侧正交视图，立体感与空间感兼具，前后层次关系明确，透视自然无畸变，兼顾正面与侧面细节",
    "30度侧视图": "30度侧视正交视图，侧面细节更突出，空间关系比45度更聚焦，适合展示单侧结构"
}

MOVE_CMD = {
    "向前平移": "镜头缓慢向前平移，主体逐渐放大，画面纵深感增强，前景细节清晰化",
    "向后平移": "镜头缓慢向后平移，主体逐渐缩小，场景范围扩大，背景元素更多纳入画面",
    "向左平移": "镜头平稳向左平移，主体位置右移，展现左侧环境延伸，构图平衡调整",
    "向右平移": "镜头平稳向右平移，主体位置左移，展现右侧环境延伸，构图重心偏移",
    "向上平移": "镜头缓慢向上平移，视角升高，突出主体下部细节与上方环境衔接",
    "向下平移": "镜头缓慢向下平移，视角轻微降低，突出主体上部细节与下方环境衔接",
    "向左上方平移": "镜头向左上方斜向平移，视角同时左移升高，展现左上方场景延伸",
    "向右上方平移": "镜头向右上方斜向平移，视角同时右移升高，展现右上方场景延伸",
    "向左下方平移": "镜头向左下方斜向平移，视角同时左移降低，展现左下方场景细节",
    "向右下方平移": "镜头向右下方斜向平移，视角同时右移降低，展现右下方场景细节"
}

ANGLE_CMD = {
    "水平向左转动": "镜头向左水平转动{}度，视角横向扩展，左侧场景纳入画面，构图左侧填充",
    "水平向右转动": "镜头向右水平转动{}度，视角横向扩展，右侧场景纳入画面，构图右侧填充",
    "向左倾斜旋转": "镜头向左旋转{}度，主体呈现左侧倾斜视角，增强动态张力，视觉重心左移",
    "向右倾斜旋转": "镜头向右旋转{}度，主体呈现右侧倾斜视角，增强动态张力，视觉重心右移",
    "向下俯视": "镜头向下俯视{}度，视角降低，突出主体顶部结构与地面/桌面环境的位置关系",
    "向上仰视": "镜头向上仰视{}度，视角升高，突出主体底部结构与天空/上方环境的位置关系",
    "向前倾斜旋转": "镜头向前旋转{}度，视角前倾，增强画面压迫感，主体近大远小效果强化",
    "向后倾斜旋转": "镜头向后旋转{}度，视角后仰，展现主体上部与天空/上方环境，画面开阔度提升",
    "顺时针旋转": "镜头顺时针旋转{}度，画面呈现旋转动态效果，增强动感与视觉冲击",
    "逆时针旋转": "镜头逆时针旋转{}度，画面呈现反向旋转动态效果，营造独特视觉体验"
}





class excel_Qwen_camera:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "镜头平移方向": (
                    [
                        "None", 
                        "向前平移", "向后平移", 
                        "向左平移", "向右平移", 
                        "向上平移", "向下平移",
                        "向左上方平移", "向右上方平移",
                        "向左下方平移", "向右下方平移"
                    ],
                    {"default": "None", "label": "镜头平移（None=不启用）"}
                ),
                
                "调整角度": (
                    [
                        "None",
                        "水平向左转动", "水平向右转动",
                        "向左倾斜旋转", "向右倾斜旋转",
                        "向下俯视", "向上仰视",
                        "向前倾斜旋转", "向后倾斜旋转",
                        "顺时针旋转", "逆时针旋转"
                    ],
                    {"default": "None", "label": "角度调整类型（None=不启用）"}
                ),
                "角度数值": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "max": 180, 
                    "step": 5, 
                    "display": "slider",
                }),
                
                "镜头类型": ([
                    "None", 
                    "广角镜头", "超广角镜头",
                    "俯视镜头", "仰视镜头",
                    "特写镜头", "大特写镜头",
                    "微距镜头",
                    "近景镜头", "中景镜头", "远景镜头", "全景镜头"
                ], {"default": "None", "label": "专业镜头选择（None=不启用）"}),
                
                "视图类型": ([
                    "None", 
                    "完整四视图", "完整六视图",
                    "正面视图", "侧面视图", "背面视图",
                    "顶部视图", "底部视图",
                    "半侧面视图（45度）", "30度侧视图"
                ], {"default": "None", "label": "正交视图选择（None=不启用）"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("提示词",)
    FUNCTION = "generate_prompt"
    CATEGORY = "Apt_Preset/prompt"

    def generate_prompt(self, 镜头平移方向, 调整角度, 角度数值, 
                       镜头类型, 视图类型):
        prompt_parts = []
        
        # 处理镜头平移：直接用下拉选项作为MOVE_CMD的键（无需额外映射）
        if 镜头平移方向 != "None":
            prompt_parts.append(MOVE_CMD.get(镜头平移方向, ""))
        
        # 处理角度调整：直接用下拉选项作为ANGLE_CMD的键
        if 调整角度 != "None" and 角度数值 > 0:
            prompt_parts.append(ANGLE_CMD.get(调整角度, "").format(角度数值))
        
        # 处理专业镜头
        if 镜头类型 != "None":
            prompt_parts.append(LENS_MAP.get(镜头类型, ""))
        
        # 处理正交视图
        view_key = 视图类型.replace("（45度）", "").replace("30度", "30度")
        if 视图类型 != "None":
            prompt_parts.append(VIEW_MAP.get(view_key, ""))
        
        # 过滤空值并优化提示词流畅度
        valid_parts = list(filter(None, prompt_parts))
        if valid_parts:
            if len(valid_parts) == 1:
                final_prompt = valid_parts[0] + "，画面构图协调，视觉效果自然"
            elif len(valid_parts) == 2:
                final_prompt = f"{valid_parts[0]}，同时{valid_parts[1]}，整体画面统一和谐"
            else:
                final_prompt = "，".join(valid_parts[:-1]) + f"，并{valid_parts[-1]}，画面层次丰富且协调"
            final_prompt += "，光影过渡自然，细节清晰可辨"
        else:
            final_prompt = "标准镜头视角（等效焦距50mm），视角自然无畸变，主体居中构图，景深适中，细节与环境兼顾，光影协调"
        
        return (final_prompt + "。",)







#endregion--------------------------------------



#region-----------------错开-------------


import os
import json
import numpy as np
import torch
import folder_paths
from PIL import Image, ImageDraw



class Coordinate_loadImage:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
                "points_data": ("STRING", {"default": "[]", "multiline": False}),
            },
        }
    
    NAME = "Coordinate_loadImage"
    RETURN_TYPES = ("IMAGE","IMAGE", "STRING",)
    RETURN_NAMES = ("ORIGINAL_IMAGE","img_tensor", "coordinates")
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/image/ImgCoordinate"
    OUTPUT_NODE = False
    
    def process(self, image, points_data="[]"):
        image_path = folder_paths.get_annotated_filepath(image)
        pil_image = Image.open(image_path).convert("RGB")
        original_image = pil_image.copy()
        img_width, img_height = pil_image.size
        
        try:
            points = json.loads(points_data)
        except json.JSONDecodeError:
            points = []
        
        if points and len(points) > 0:
            pil_image = self._draw_markers(pil_image, points, img_width, img_height)
        
        # 处理带标记的图片
        img_array = np.array(pil_image).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).unsqueeze(0)
        
        # 处理原图
        original_array = np.array(original_image).astype(np.float32) / 255.0
        original_tensor = torch.from_numpy(original_array).unsqueeze(0)
        
        return (original_tensor,img_tensor,points_data)
    
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
    def IS_CHANGED(cls, image, **kwargs):
        points_data = kwargs.get("points_data", "[]")
        return f"{image}_{points_data}"
    
    @classmethod
    def VALIDATE_INPUTS(cls, image,** kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return f"图片文件不存在: {image}"
        return True



#endregion-----------------------------




class text_mul_Split:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "delimiter": ("STRING", {
                    "default": "\\n",
                    "multiline": False,
                    "tooltip": "Use \\n for newline, \\t for tab, \\s for space"
                }),
            },
        }

    RETURN_TYPES = ("LIST", "STRING", "STRING", "STRING", "STRING", 
                    "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("list_output", "item1", "item2", "item3", "item4", 
                    "item5", "item6", "item7", "item8")
    #OUTPUT_IS_LIST = (True, False, False, False, False, False, False, False, False)
    FUNCTION = "split_text"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def split_text(self, text, delimiter):
        # 处理特殊转义字符
        if delimiter == "\\n":
            actual_delimiter = "\n"
        elif delimiter == "\\t":
            actual_delimiter = "\t"
        elif delimiter == "\\s":
            actual_delimiter = " "
        else:
            actual_delimiter = delimiter.strip()

        # 使用实际分隔符进行分割
        parts = [part.strip() for part in text.split(actual_delimiter)]

        # 生成8个固定输出，不足补空字符串
        output_items = parts[:8]
        while len(output_items) < 8:
            output_items.append("")


        list_out = []
        for text_item in parts:
            list_out.append(text_item)
        

        return (list_out, *output_items)


class text_list_combine :
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_list": (any_type,),  
                "delimiter": (["newline","comma","backslash","space"], {"default": "comma"}),
            },
        }
    
    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,)  

    def run(self, text_list, delimiter):
        try:
            delimiter = delimiter[0] if isinstance(delimiter, list) else delimiter
            delimiter_map = {
                "newline": "\n",
                "comma": ",",
                "backslash": "\\",
                "space": " "
            }
            final_delimiter = delimiter_map.get(delimiter, ",")

            flattened_list = []
            for item in text_list:
                if isinstance(item, list):
                    flattened_list.extend(item)
                else:
                    flattened_list.append(item)

            str_items = []
            for item in flattened_list:
                if item is None:
                    continue
                item_str = str(item).strip()
                if item_str:
                    str_items.append(item_str)

            combined_text = final_delimiter.join(str_items)
            return (combined_text,)
        
        except Exception as e:
            return (f"合并失败: {str(e)}",)



class text_mul_Join:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "join_rule": (
                    ["None", "Custom", "Line", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe"],
                    {"default": "Comma"}
                ),
                "custom_pattern": ("STRING", {
                    "multiline": True,
                    "default": "{text1},{text2},{text3},{text4},{text5},{text6},{text7},{text8}",
                }),
                **{f"text{i+1}": ("STRING", {"default": "", "multiline": False}) for i in range(8)},
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined_text",)
    FUNCTION = "smart_join"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    文本合并预设说明
    None：直接拼接所有文本（无分隔符）。
    Line：按行分隔（\\n）。
    Space：按空格分隔。
    Comma：按逗号分隔（支持中英文逗号）。
    Period：按句号分隔（支持中英文句号）。
    Semicolon：按分号分隔（支持中英文分号）。
    Tab：按制表符分隔（\\t）。
    Pipe：按竖线分隔（|）。
    Custom：自定义排版替换（支持{text1}-{text8}占位符自由组合，保留排版格式）
    """ 

    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = ''.join([c for c in text if c.isprintable() or c in ['\n', ' ']])
        return text.strip()

    def _get_separator_template(self, join_rule: str) -> str:
        rule_template_map = {
            "None": "",
            "Line": "\n",
            "Space": " ",
            "Comma": "{sep}",
            "Period": "{sep}",
            "Semicolon": "{sep}",
            "Tab": "\t",
            "Pipe": "|",
            "Custom": None
        }
        
        symbol_map = {
            "Comma": (",", "，"),
            "Period": (".", "。"),
            "Semicolon": (";", "；")
        }
        
        template = rule_template_map.get(join_rule, "\n")
        
        if join_rule in symbol_map and template == "{sep}":
            template = symbol_map[join_rule][1]
        
        return template

    def _replace_custom_placeholders(self, custom_pattern: str, text_dict: dict) -> str:
        normalized_pattern = self._normalize_text(custom_pattern)
        result = normalized_pattern
        
        for i in range(8):
            placeholder = f"{{text{i+1}}}"
            text_content = text_dict.get(f"text{i+1}", "")
            result = result.replace(placeholder, text_content)
        
        return result

    def smart_join(self, join_rule: str, custom_pattern: str, **kwargs):
        text_dict: Dict[str, str] = {}
        non_empty_texts: List[str] = []
        for i in range(8):
            text_key = f"text{i+1}"
            text_content = self._normalize_text(kwargs.get(text_key, ""))
            text_dict[text_key] = text_content
            if text_content:
                non_empty_texts.append(text_content)

        if join_rule == "Custom":
            joined_text = self._replace_custom_placeholders(custom_pattern, text_dict)
        else:
            separator = self._get_separator_template(join_rule)
            joined_text = separator.join(non_empty_texts) if non_empty_texts else ""

        joined_text = self._normalize_text(joined_text)
        if not joined_text:
            joined_text = "❌ No valid content to join"

        return (joined_text,)




import re
import random
import unicodedata
from typing import List



class text_Splitter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": "", "placeholder": "请输入要拆分的文本"}),
                # 👇 新增：按空行分割，位置放在最顺手的地方
                "split_rule": (["不分割", "自定义正则", "按行分割", "按空行分割", "按空格分割", "按逗号分割", "按句号分割", "按分号分割", "按制表符分割", "按竖线分割", "按序号", "按标题"], {"default": "按行分割"}),
                "custom_separator": ("STRING", {"multiline": False, "default": "", "placeholder": "自定义分隔符/正则表达式"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("STRING", "LIST", ANY_TYPE)
    RETURN_NAMES = ("调度输出", "拆分列表", "列表")
    FUNCTION = "smart_process"
    CATEGORY = "Apt_Preset/prompt"
    OUTPUT_IS_LIST = (False, False, True)

    DESCRIPTION = """
    自定义正则：自定义分隔符，支持正则表达式
    1) 匹配标题：【镜头1】、【镜头2】---->用：【[^】]+】
    2）匹配文字编号：镜头1、镜头2、镜头3...---->用：镜头\d+
    3）匹配编号：1）、2）、3）---->用：\d+）
    4）匹配内容：内容 A、内容 B、内容 C---->用：内容.
    文本拆分工具说明
    不分割：不进行任何拆分
    按行分割：按换行符拆分
    按空行分割：按连续空行拆分（适合分镜块、段落拆分）
    按空格分割：按空格拆分
    按逗号分割：按 , ， 拆分
    按句号分割：按 . 。 拆分
    按分号分割：按 ; ； 拆分
    按制表符分割：按 Tab 键拆分
    按竖线分割：按 | 拆分
    按序号：按常见列表序号拆分（如 1. / A. / (1)）
    按标题：按 Markdown 标题或“第X章/节/部分/篇”拆分
    """ 
    
    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = ''.join([c for c in text if c.isprintable() or c in ['\n', ' ']])
        return text.strip()

    def _get_smart_separator(self, rule: str, custom_sep: str) -> List[str]:
        # 👇 新增映射：按空行分割
        smart_sep_map = {
            "不分割": [],
            "按行分割": ["\n", "\r\n"],
            "按空行分割": [],  # 特殊处理
            "按空格分割": [" ", "　"],
            "按逗号分割": [",", "，"],
            "按句号分割": [".", "。"],
            "按分号分割": [";", "；"],
            "按制表符分割": ["\t"],
            "按竖线分割": ["|"],
            "自定义正则": [custom_sep.replace("\\n", "\n").replace("\\t", "\t")] if custom_sep else []
        }
        return smart_sep_map.get(rule, ["\n"])

    def _smart_split(self, text: str, separators: List[str], is_regex: bool = False, is_blank_line: bool = False, split_rule: str = "") -> List[str]:
        if not text:
            return [self._normalize_text(text)]
        
        if split_rule == "按序号":
            pattern = r"(?:^|\n)(?:\d+\.|[A-Za-z]\.|[\(（]?\d+[\)）]|[A-Za-z]\))\s*(.*)"
            split_result = re.findall(pattern, text, re.MULTILINE)
        elif split_rule == "按标题":
            pattern = r"(?:^|\n)(?:#{1,6}\s+|第\s*[一二三四五六七八九十\d]+\s*(?:章|节|部分|篇))\s*(.*)"
            split_result = re.findall(pattern, text, re.MULTILINE)
        elif is_blank_line:
            split_result = re.split(r'\n\s*\n', text)
        elif is_regex:
            try:
                split_result = re.split(separators[0], text)
            except re.error:
                return [self._normalize_text(text)]
        else:
            escaped_seps = [re.escape(sep) for sep in separators if sep]
            if not escaped_seps:
                return [self._normalize_text(text)]
            sep_pattern = '|'.join(escaped_seps)
            split_result = re.split(sep_pattern, text)

        result = []
        for item in split_result:
            cleaned = self._normalize_text(item)
            if cleaned:
                result.append(cleaned)
        return result if result else [""]

    def smart_process(self, text_input: str, split_rule: str, custom_separator: str, current_frame: int):
        text_content = self._normalize_text(text_input)
        if not text_content:
            return ("❌ 文本为空", [], [])
        
        separators = self._get_smart_separator(split_rule, custom_separator)
        is_regex_mode = (split_rule == "自定义正则")
        is_blank_line_mode = (split_rule == "按空行分割")
        
        split_list = self._smart_split(text_content, separators, is_regex_mode, is_blank_line_mode, split_rule)
        
        if not split_list or split_list == [""]:
            return ("❌ 拆分后无有效内容", [], [])
        
        frame_index = current_frame if current_frame >= 0 else 0
        selected_index = frame_index % len(split_list)
        single_text = split_list[selected_index]
        return (single_text, split_list, split_list)



class text_converter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": (any_type, {}),
            },
            "optional": {
                "operation": ([
                    "列表转合并文本",
                    "JSON转普通文本", 
                    "JSON压缩",
                    "JSON美化",
                    "Base64编码",
                    "Base64解码",
                    "URL编码",
                    "URL解码",
                    "HTML转义",
                    "HTML反转义",
                    "Unicode标准化",
                ], {"default": "列表转合并文本"}),
                "combine_rule": (["None", "Custom", "Line", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe"], {"default": "Comma"}),
                "custom_combine_rule": ("STRING", {"multiline": False, "default": "", "placeholder": "自定义分隔符（选择Custom时生效）"}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("输出文本",)
    FUNCTION = "convert"
    CATEGORY = "Apt_Preset/prompt"
    OUTPUT_IS_LIST = (False,)
    DESCRIPTION = """
    文本转换工具：自动识别上游列表格式，转为单一文本
    操作说明：
    1. 列表转合并文本：支持预设规则或自定义分隔符合并列表
       - None：不合并（返回第一个非空项）
       - Line：按换行\\n合并
       - Space：按空格（含全角）合并
       - Comma：按逗号（中英文）合并
       - Period：按句号（中英文）合并
       - Semicolon：按分号（中英文）合并
       - Tab：按制表符\\t合并
       - Pipe：按竖线|合并
       - Custom：使用自定义分隔符（支持\\n/\\t转义）
    2. JSON转普通文本：解析JSON并提取纯文本内容（移除所有JSON格式符号）
    3. JSON压缩/美化：仅对合法JSON字符串生效（多列表项自动转为JSON数组）
    4. 其他：编码/解码/格式化（自动适配列表/单值输入）
    """

    def _adapt_input_data(self, data):
        """扁平化嵌套列表，过滤None/空值"""
        flattened = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    flattened.extend(self._adapt_input_data(item))
                else:
                    if item is not None and str(item).strip() != "":
                        flattened.append(item)
        else:
            if data is not None and str(data).strip() != "":
                flattened.append(data)
        return flattened

    def _get_combine_separator(self, rule: str, custom_sep: str) -> str:
        """获取合并用的分隔符"""
        smart_sep_map = {
            "None": "",
            "Line": "\n",
            "Space": " ",
            "Comma": ",",
            "Period": ".",
            "Semicolon": ";",
            "Tab": "\t",
            "Pipe": "|",
            "Custom": custom_sep.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r").replace("\\\\", "\\") if custom_sep else ""
        }
        sep = smart_sep_map.get(rule, ",")
        
        if rule == "Space" and sep == " ":
            sep = " "
        if rule == "Comma":
            sep = ","
        elif rule == "Period":
            sep = "."
        elif rule == "Semicolon":
            sep = ";"
        
        return sep

    def _extract_json_text(self, json_data, separator=", "):
        """递归提取JSON中的所有纯文本内容，移除格式符号"""
        text_parts = []
        
        # 递归处理JSON对象/数组
        def _recursive_extract(obj):
            if isinstance(obj, dict):
                # 处理JSON对象：提取所有值（忽略键，或按需保留键）
                for value in obj.values():
                    _recursive_extract(value)
            elif isinstance(obj, list):
                # 处理JSON数组：遍历所有元素
                for item in obj:
                    _recursive_extract(item)
            elif isinstance(obj, str):
                # 字符串：直接添加（已去除引号）
                if obj.strip():
                    text_parts.append(obj.strip())
            elif isinstance(obj, (int, float, bool)):
                # 数字/布尔值：转为字符串
                text_parts.append(str(obj))
            # 忽略null/None值
        
        # 执行递归提取
        _recursive_extract(json_data)
        
        # 用分隔符合并所有文本片段
        return separator.join(text_parts) if text_parts else ""

    def convert(self, data, operation="列表转合并文本", combine_rule="Comma", custom_combine_rule=""):
        try:
            # 解包参数
            if isinstance(operation, list):
                operation = operation[0] if operation else "列表转合并文本"
            if isinstance(combine_rule, list):
                combine_rule = combine_rule[0] if combine_rule else "Comma"
            if isinstance(custom_combine_rule, list):
                custom_combine_rule = custom_combine_rule[0] if custom_combine_rule else ""
            
            # 适配输入数据
            normalized_data = self._adapt_input_data(data)
            # 获取合并分隔符
            final_sep = self._get_combine_separator(combine_rule, custom_combine_rule)

            # 1. 列表转合并文本
            if operation == "列表转合并文本":
                if combine_rule == "None" or final_sep == "":
                    result = str(normalized_data[0]).strip() if normalized_data else ""
                else:
                    str_items = []
                    for item in normalized_data:
                        item_str = str(item).strip()
                        if item_str:
                            cleaned = unicodedata.normalize('NFKC', item_str)
                            cleaned = ''.join([c for c in cleaned if c.isprintable() or c in ['\n', ' ', '\t']])
                            str_items.append(cleaned)
                    result = final_sep.join(str_items) if str_items else ""
                return (result,)

            # 2. 新增：JSON转普通文本（核心功能）
            elif operation == "JSON转普通文本":
                # 拼接输入数据为完整字符串
                if normalized_data:
                    str_data = final_sep.join([str(item).strip() for item in normalized_data])
                else:
                    str_data = str(data).strip() if data is not None else ""
                
                try:
                    # 解析JSON数据
                    parsed_json = json.loads(str_data)
                    # 提取纯文本内容（移除所有JSON格式）
                    result = self._extract_json_text(parsed_json, final_sep)
                except (json.JSONDecodeError, TypeError):
                    # 解析失败：返回原文本（或提示）
                    result = f"JSON解析失败，返回原文本：{str_data[:200]}"
                return (result,)

            # 3. JSON压缩
            elif operation == "JSON压缩":
                try:
                    if len(normalized_data) == 1:
                        str_data = str(normalized_data[0]).strip()
                        parsed = json.loads(str_data)
                    else:
                        parsed = [json.loads(str(item).strip()) for item in normalized_data if str(item).strip()]
                    result = json.dumps(parsed, ensure_ascii=False, separators=(',', ':'))
                except (json.JSONDecodeError, TypeError):
                    str_data = final_sep.join([str(item).strip() for item in normalized_data]) if normalized_data else str(data)
                    result = str_data
                return (result,)

            # 4. JSON美化
            elif operation == "JSON美化":
                try:
                    if len(normalized_data) == 1:
                        str_data = str(normalized_data[0]).strip()
                        parsed = json.loads(str_data)
                    else:
                        parsed = [json.loads(str(item).strip()) for item in normalized_data if str(item).strip()]
                    result = json.dumps(parsed, ensure_ascii=False, indent=2)
                except (json.JSONDecodeError, TypeError):
                    str_data = final_sep.join([str(item).strip() for item in normalized_data]) if normalized_data else str(data)
                    result = str_data
                return (result,)

            # 5. Base64编码
            elif operation == "Base64编码":
                str_data = final_sep.join([str(item).strip() for item in normalized_data]) if normalized_data else str(data)
                byte_data = str_data.encode('utf-8')
                result = base64.b64encode(byte_data).decode('ascii')
                return (result,)

            # 6. Base64解码
            elif operation == "Base64解码":
                str_data = str(normalized_data[0]).strip() if (normalized_data and len(normalized_data)>=1) else str(data).strip()
                try:
                    decoded_bytes = base64.b64decode(str_data.encode('ascii'))
                    result = decoded_bytes.decode('utf-8')
                except Exception as e:
                    result = f"Base64解码失败: {str(e)} | 原文: {str_data}"
                return (result,)

            # 7. URL编码
            elif operation == "URL编码":
                str_data = final_sep.join([str(item).strip() for item in normalized_data]) if normalized_data else str(data)
                result = urllib.parse.quote(str_data, encoding='utf-8')
                return (result,)

            # 8. URL解码
            elif operation == "URL解码":
                str_data = str(normalized_data[0]).strip() if (normalized_data and len(normalized_data)>=1) else str(data).strip()
                try:
                    result = urllib.parse.unquote(str_data, encoding='utf-8')
                except Exception as e:
                    result = f"URL解码失败: {str(e)} | 原文: {str_data}"
                return (result,)

            # 9. HTML转义
            elif operation == "HTML转义":
                str_data = final_sep.join([str(item).strip() for item in normalized_data]) if normalized_data else str(data)
                result = html.escape(str_data)
                return (result,)

            # 10. HTML反转义
            elif operation == "HTML反转义":
                str_data = str(normalized_data[0]).strip() if (normalized_data and len(normalized_data)>=1) else str(data).strip()
                result = html.unescape(str_data)
                return (result,)

            # 11. Unicode标准化
            elif operation == "Unicode标准化":
                str_data = final_sep.join([str(item).strip() for item in normalized_data]) if normalized_data else str(data)
                result = unicodedata.normalize('NFKC', str_data)
                return (result,)

            # 默认：列表转合并文本
            else:
                if combine_rule == "None" or final_sep == "":
                    result = str(normalized_data[0]).strip() if normalized_data else ""
                else:
                    str_items = [str(item).strip() for item in normalized_data if str(item).strip()]
                    result = final_sep.join(str_items) if str_items else ""
                return (result,)

        except Exception as e:
            return (f"转换错误: {str(e)} | 输入数据: {str(data)[:100]}",)




class text_filter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"multiline": True, "default": "", "placeholder": "Enter text to filter"}),
                "filter_rule": (["None", "custom", "@text@", "@text", "text @", '"text"', "'text'", "{text}", "(text)"], {"default": "None"}),
                "custom_rule": ("STRING", {"multiline": False, "default": "", "placeholder": "Custom filter rule, e.g., [text], [text, text]"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "match_all": ("BOOLEAN", {"default": False, "label_on": "All", "label_off": "Single"}),
                "reverse_filter": ("BOOLEAN", {"default": False, }),
            }
        }
    RETURN_TYPES = ("STRING", "LIST")
    RETURN_NAMES = ("调度文本", "Matched_list")
    FUNCTION = "smart_process"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    文本过滤预设说明
    None：不过滤，返回原始文本。
    @text@：提取@符号包裹的文本。
    @text：提取@符号后的所有文本。
    text @：提取@符号前的所有文本。
    "text"：提取双引号包裹的文本。
    'text'：提取单引号包裹的文本。
    (text)：提取小括号包裹的文本。
    Custom：使用自定义过滤规则，例如:
         [text] ：括号内的文本都会被提取并返回。
         [text ：括号后面的文本都会被提取并返回。
         text]：括号前面的文本都会被提取并返回。
    """
    def _normalize_text(self, text: str) -> str:
        text = unicodedata.normalize('NFKC', text)
        text = ''.join([c for c in text if c.isprintable() or c in ['\n', ' ']])
        return text.strip()
    def _get_filter_pattern(self, filter_rule: str, custom_rule: str) -> Optional[str]:
        # 处理自定义规则
        if filter_rule == "custom" and custom_rule.strip():
            target_rule = custom_rule.strip()
            if "text" in target_rule:
                if len(target_rule) == len("text") + 2 and target_rule.count("text") == 1:
                    prefix = target_rule.replace("text", "", 1)[0]
                    suffix = target_rule.replace("text", "", 1)[-1]
                    return re.escape(prefix) + r"(.*?)" + re.escape(suffix)
                elif target_rule.endswith("text") and len(target_rule) == len("text") + 1:
                    prefix = target_rule.replace("text", "")
                    return re.escape(prefix) + r"(.*)"
                elif target_rule.startswith("text") and len(target_rule) == len("text") + 1:
                    suffix = target_rule.replace("text", "")
                    return r"(.*?)" + re.escape(suffix)
        
        rule_pattern_map = {
            "None": None,
            "@text@": re.escape("@") + r"(.*?)" + re.escape("@"),
            "@text": re.escape("@") + r"(.*)",
            "text @": r"(.*?)" + re.escape("@"),
            '"text"': re.escape('"') + r"(.*?)" + re.escape('"'),
            "'text'": re.escape("'") + r"(.*?)" + re.escape("'"),
            "{text}": re.escape("{") + r"(.*?)" + re.escape("}"),
            "(text)": re.escape("(") + r"(.*?)" + re.escape(")"),
            "custom": None,  # 如果选择custom但没有提供规则，返回None
        }
        return rule_pattern_map.get(filter_rule, None)
    def _extract_non_matches(self, text: str, pattern: str) -> List[str]:
        non_match_parts = []
        last_end = 0
        for m in re.finditer(pattern, text, re.DOTALL):
            part = self._normalize_text(text[last_end:m.start()])
            if part:
                non_match_parts.append(part)
            last_end = m.end()
        tail = self._normalize_text(text[last_end:])
        if tail:
            non_match_parts.append(tail)
        return non_match_parts

    def _smart_filter(self, text: str, pattern: Optional[str], match_all: bool, reverse_filter: bool) -> Tuple[str, List[str]]:
        if pattern is None:
            normalized_text = self._normalize_text(text)
            return (normalized_text, [normalized_text] if normalized_text else [])
        if reverse_filter:
            filtered_results = self._extract_non_matches(text, pattern)
            if not filtered_results:
                return ("❌ No valid content after reverse filtering", [])
        else:
            filtered_results = re.findall(pattern, text, re.DOTALL)
            filtered_results = [res.strip() for res in filtered_results if res.strip()]
            if not filtered_results:
                return ("❌ No valid content after filtering", [])
        if match_all:
            main_result = "\n".join(filtered_results)
        else:
            main_result = filtered_results[0]
        return (main_result, filtered_results)
    def smart_process(self, text_input: str, filter_rule: str, custom_rule: str, current_frame: int, match_all: bool = False, reverse_filter: bool = False):
        text_content = self._normalize_text(text_input)
        if not text_content:
            return ("❌ Text input is empty", [])
        pattern = self._get_filter_pattern(filter_rule, custom_rule)
        main_result, match_list = self._smart_filter(text_content, pattern, match_all, reverse_filter)
        if not match_all and len(match_list) > 0:
            frame_index = current_frame if current_frame >= 0 else 0
            selected_index = frame_index % len(match_list)
            main_result = match_list[selected_index]
        return (main_result, match_list)




class text_modifier:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "repair_type": (
                    [
                        "None", "取数字", 
                        "取字母", "转大写", "转小写", "取中文", 
                        "去标点", "去换行", "去空行", "去空格", 
                        "去格式", "统计字数", "去特殊字符", 
                        "去重复行", "每行首字母大写"
                    ], 
                    {"default": "None"}
                ),
                "replace_targets": ("STRING", {"multiline": False, "default": "{text1}|{text2}", }),
                "replace_content": ("STRING", {"multiline": False, "default": "{A}|{B}", }),
                "remove_targets": ("STRING", {"multiline": False, "default": "{text1}|{text2}", })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process_text"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    多文本替换或移除，使用竖线分隔，支持正则表达式。
    例如：
    targets = "{man}|{dog}"
    replacements = "{girl}|{cat}" 同时替换
    words_to_remove = "{man}|{dog}" 同时移除
    """ 



    def process_text(self, text, repair_type, replace_targets, replace_content, remove_targets):
        text = text or ""
        
        # 处理替换文本
        if replace_targets.strip() and replace_content.strip():
            text = self.replace_text(text, replace_targets, replace_content)[0]
        
        # 处理移除文本
        if remove_targets.strip():
            text = self.remove_text(text, remove_targets)[0]
        
        # 处理其他修复类型
        if repair_type == "None":
            return (text,)
        else:
            return self.repair_text(text, repair_type)
    
    def replace_text(self, text, replace_targets, replace_content):
        """替换文本功能，使用 {text1}|{text2} 格式"""
        if not replace_targets.strip() or not replace_content.strip():
            return (text,)
        
        # 按 | 分割目标和替换内容
        targets = [t.strip() for t in replace_targets.split('|') if t.strip()]
        replacements = [r.strip() for r in replace_content.split('|') if r.strip()]
        
        if not targets or not replacements:
            return (text,)
        
        # 创建替换映射
        word_map = {}
        min_len = min(len(targets), len(replacements))
        for i in range(min_len):
            if targets[i]:
                word_map[targets[i]] = replacements[i]
        
        # 按长度排序，优先替换长的内容
        sorted_targets = sorted(word_map.keys(), key=len, reverse=True)
        
        result = text
        for target in sorted_targets:
            # 替换整个 {target} 为对应的替换内容
            pattern = re.escape(target)
            result = re.sub(pattern, word_map[target], result)
        
        return (result,)
    
    def remove_text(self, text, remove_targets):
        """移除文本功能，使用 {text1}|{text2} 格式"""
        if not remove_targets.strip():
            return (text,)
        
        # 按 | 分割要移除的内容
        remove_words = [w.strip() for w in remove_targets.split('|') if w.strip()]
        
        if not remove_words:
            return (text,)
        
        # 按长度排序，优先移除长的内容
        remove_words_sorted = sorted(remove_words, key=len, reverse=True)
        pattern = '|'.join(re.escape(word) for word in remove_words_sorted)
        cleaned_text = re.sub(pattern, '', text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return (cleaned_text,)
    

    
    def repair_text(self, input_string, option):
        input_string = input_string or ""
        
        if option == "取数字":
            result = ''.join(re.findall(r'\d', input_string))
        elif option == "取字母":
            processed = ''.join([self.full2half(c) for c in input_string])
            result = ''.join(filter(lambda char: char.isalpha() and not self.is_chinese(char), processed))
        elif option == "转大写":
            result = input_string.upper()
        elif option == "转小写":
            result = input_string.lower()
        elif option == "取中文":
            result = ''.join(filter(self.is_chinese, input_string))
        elif option == "去标点":
            result = re.sub(r'[^\d\w\s\u4e00-\u9fff]', '', input_string)
        elif option == "去换行":
            result = input_string.replace('\n', '').replace('\r', '')
        elif option == "去空行":
            result = '\n'.join(filter(lambda line: line.strip(), input_string.splitlines()))
        elif option == "去空格":
            result = input_string.replace(' ', '').replace('\t', '')
        elif option == "去格式":
            result = re.sub(r'\s+', '', input_string)
        elif option == "统计字数":
            clean_str = re.sub(r'\s+', '', input_string)
            result = str(len(clean_str))
        elif option == "去特殊字符":
            result = re.sub(r'[^\u4e00-\u9fffa-zA-Z0-9\s]', '', input_string)
        elif option == "去重复行":
            lines = input_string.splitlines()
            unique_lines = []
            seen = set()
            for line in lines:
                stripped_line = line.strip()
                if stripped_line not in seen:
                    seen.add(stripped_line)
                    unique_lines.append(stripped_line)
            result = '\n'.join(unique_lines)
        elif option == "每行首字母大写":
            def capitalize_line(line):
                if not line.strip():
                    return line
                stripped = line.lstrip()
                return line[:len(line)-len(stripped)] + stripped[0].upper() + stripped[1:]
            lines = input_string.splitlines()
            processed_lines = [capitalize_line(line) for line in lines]
            result = '\n'.join(processed_lines)
        else:
            result = input_string

        return (result,)
    
    @staticmethod
    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'
    
    @staticmethod
    def full2half(char):
        if ord(char) == 0x3000:
            return ' '
        if '\uff01' <= char <= '\uff5e':
            return chr(ord(char) - 0xfee0)
        return char
    
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return True





class text_saveText:
    @classmethod
    def INPUT_TYPES(cls):
        # 移除docx选项，仅保留其他格式
        file_types = ["text", "md", "json", "js", "py", "toml"]
        
        return {
            "required": {
                "content": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "输入要保存的内容"
                }),

                "file_path": ("STRING", {
                    "default": "",
                    "placeholder": "输入文件保存路径（包含文件名）"
                }),
                 "file_type": (file_types,),
            },
            "optional": {
                "custom_file_name": ("STRING", {
                    "default": "",
                    "placeholder": "自定义文件名（不含扩展名，留空则自动生成）"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "write_content"
    CATEGORY = "Apt_Preset/prompt"

    def write_content(self, content: str, file_path: str, file_type: str, custom_file_name: str = "") -> Tuple[str]:
        file_path = file_path.strip('\'"')
        
        if not file_path:
            raise ValueError("文件路径不能为空")
        
        # 移除docx映射，仅保留其他格式的扩展名映射
        ext_mapping = {
            "text": ".txt",
            "md": ".md",
            "json": ".json",
            "js": ".js",
            "py": ".py",
            "toml": ".toml"
        }
        target_ext = ext_mapping.get(file_type, "")
        if not target_ext:
            raise ValueError("不支持的文件类型")
        
        try:
            parent_dir = os.path.dirname(file_path)
            original_file_name = os.path.basename(file_path)
            
            if custom_file_name.strip():
                full_path = os.path.join(parent_dir, f"{custom_file_name.strip()}{target_ext}")
            elif not os.path.exists(file_path):
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                default_file_name = f"output_{timestamp}"
                full_path = os.path.join(parent_dir, f"{default_file_name}{target_ext}")
            else:
                full_path = file_path if file_path.lower().endswith(target_ext) else f"{file_path}{target_ext}"
            
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
            
            if file_type == "json":
                try:
                    json_data = json.loads(content)
                    with open(full_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=2)
                except (json.JSONDecodeError, TypeError):
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            elif file_type == "toml":
                try:
                    toml_data = toml.loads(content)
                    with open(full_path, 'w', encoding='utf-8') as f:
                        toml.dump(toml_data, f)
                except toml.TomlDecodeError:
                    with open(full_path, 'w', encoding='utf-8') as f:
                        f.write(content)
            
            # 移除所有docx相关的分支逻辑
            else:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            return (f"成功：文件已保存至 {full_path}",)
        
        except Exception as e:
            raise ValueError(f"写入失败：{str(e)}")



class text_loadText:
    @classmethod
    def INPUT_TYPES(cls):
        file_types = ["text", "md", "json", "js", "py", "toml"]
        
        return {
            "required": {
                "path": ("STRING", {
                    "default": "",
                    "placeholder": "输入文件路径"
                }),
                "file_type": (file_types,),
                "char_limit": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 100000,
                    "step": 10,
                }),
                "batch_mode": ("BOOLEAN", {
                    "default": False,
                    "label_on": "文件夹批量读取",
                    "label_off": "单文件读取"
                }),
                "remove_extension": ("BOOLEAN", {
                    "default": False,
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "file_paths", "file_names")
    FUNCTION = "read_content"
    CATEGORY = "Apt_Preset/prompt"

    def read_content(self, path: str, file_type: str, char_limit: int, batch_mode: bool, remove_extension: bool) -> Tuple[str, str, str]:
        path = path.strip('\'"')
        
        if not path:
            raise ValueError("路径不能为空")
        
        ext_mapping = {
            "text": ".txt",
            "md": ".md",
            "json": ".json",
            "js": ".js",
            "py": ".py",
            "toml": ".toml"
        }
        target_ext = ext_mapping.get(file_type, "")
        
        def process_file_name(file_name):
            if remove_extension:
                return os.path.splitext(file_name)[0]
            return file_name
        
        try:
            if batch_mode:
                if not os.path.isdir(path):
                    raise ValueError(f"批量模式下路径必须是文件夹 - {path}")
                
                if not target_ext:
                    raise ValueError(f"不支持的文件类型：{file_type}")
                
                search_pattern = os.path.join(path, f"*{target_ext}")
                file_paths = glob.glob(search_pattern)
                
                if not file_paths:
                    raise ValueError(f"警告：在 {path} 中未找到{target_ext}类型文件")
                
                file_paths.sort(key=lambda x: os.path.basename(x))
                read_paths = []
                read_names = []
                merged_content = []
                total_char_count = 0
                
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    processed_name = process_file_name(file_name)
                    merged_content.append(f"\n\n===== 开始：{file_name} =====")
                    
                    content = self._read_single_file(file_path, file_type)
                    merged_content.append(content)
                    merged_content.append(f"===== 结束：{file_name} =====")
                    
                    read_paths.append(file_path)
                    read_names.append(processed_name)
                    total_char_count += len(content)
                    
                    if char_limit > 0 and total_char_count > char_limit:
                        merged_content.append(f"\n\n...（已达字符限制 {char_limit}，后续文件未读取）")
                        break
                
                final_content = ''.join(merged_content)
                paths_str = "\n".join(read_paths)
                names_str = "\n".join(read_names)
                return (final_content, paths_str, names_str)
            
            else:
                if not os.path.isfile(path):
                    raise ValueError(f"文件不存在 - {path}")
                
                if not path.lower().endswith(target_ext):
                    new_path = path + target_ext
                    if os.path.isfile(new_path):
                        path = new_path
                    else:
                        raise ValueError(f"文件扩展名与所选类型不匹配（预期{target_ext}，当前路径：{path}）")
                
                content = self._read_single_file(path, file_type)
                file_name = os.path.basename(path)
                processed_name = process_file_name(file_name)
                
                if char_limit > 0 and len(content) > char_limit:
                    content = content[:char_limit] + f"\n\n...（内容已截断，原长度{len(content)}字符）"
                
                return (content, path, processed_name)
                
        except Exception as e:
            raise ValueError(f"读取失败：{str(e)}")
    
    def _read_single_file(self, file_path: str, file_type: str) -> str:
        try:
            if file_type == "json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    return json.dumps(json_data, ensure_ascii=False, indent=2)
            
            elif file_type == "toml":
                with open(file_path, 'r', encoding='utf-8') as f:
                    toml_data = toml.load(f)
                    return toml.dumps(toml_data)
            
            else:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='gbk') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"读取文件 {file_path} 失败：{str(e)}")




#endregion--------------------------------------





# region--------text_wildcards----------
import os
import random
import re
from pathlib import Path
import folder_paths
import unicodedata
import sys

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

MODEL_ROOT_DIR = os.path.join(folder_paths.models_dir, "Apt_File")
wildcards_dir1 = Path(MODEL_ROOT_DIR) / "wildcards"
os.makedirs(wildcards_dir1, exist_ok=True)

wildcards_dir2 = Path(__file__).parent.parent / "wildcards"
os.makedirs(wildcards_dir2, exist_ok=True)

full_dirs = [wildcards_dir1, wildcards_dir2]

wildcard_map = {}
for wildcard in wildcards_dir1.rglob("*.txt"):
    if wildcard.is_file():
        rel_path = str(wildcard.relative_to(wildcards_dir1))[:-4]
        wildcard_map[rel_path] = f"dir1 | {rel_path}"

for wildcard in wildcards_dir2.rglob("*.txt"):
    if wildcard.is_file():
        rel_path = str(wildcard.relative_to(wildcards_dir2))[:-4]
        if rel_path not in wildcard_map:
            wildcard_map[rel_path] = f"base_path | {rel_path}"

WILDCARDS_LIST = ["None"] + list(wildcard_map.values())

total_files = len(list(wildcards_dir1.rglob("*.txt"))) + len(list(wildcards_dir2.rglob("*.txt")))
unique_files = len(wildcard_map)
if total_files > unique_files:
    duplicate_count = total_files - unique_files
    print(f"ℹ️ 检测到 {duplicate_count} 个同名通配符文件，已优先使用 dir1 目录下的文件，base_path 中的同名文件已被过滤")

class text_wildcards:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "output_join_rule": (
                    ["Line", "Custom", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe"],
                    {"default": "Comma"}
                ),
                "custom_join": ("STRING", {
                    "multiline": True,
                    "default": "{text0}，{text1}，{text2}，{text3}，{text4}，{text5}，{text6}，{text7}，{text8}",
                    "placeholder": "在Custom模式下，才有效 It only works in Custom mode"
                }),
                "input_split_rule": (
                    ["Line", "Custom", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe"],
                    {"default": "Line"}
                ),
                "custom_split": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "placeholder": "在Custom模式下，才有效 It only works in Custom mode."
                }),
                "text0": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                **{f"text{i}": (WILDCARDS_LIST, {"default": WILDCARDS_LIST[0]}) for i in range(1, 9)},
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }
        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "stack_Wildcards"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
    1. 支持text0（自定义文本），text1-text8通配符文件选None则跳过
    2. text0为自定义文本前缀，可直接输入任意内容（完美支持中文）
    Line：按行分隔（\\n）。
    Space：按空格分隔（支持全角/半角空格）。
    Comma：按逗号分隔（支持中英文逗号）。
    Period：按句号分隔（支持中英文句号）。
    Semicolon：按分号分隔（支持中英文分号）。
    Tab：按制表符分隔（\\t）。
    Pipe：按竖线分隔（|）。
    Custom：自定义排版替换（支持{text1}-{text8}占位符自由排版）
    🌟 增强特性：完美支持中文文件名、中文文本处理、全角/半角符号统一；同名文件优先使用 dir1 目录下的版本
    """

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize('NFKC', text)
        allowed_chars = set(['\n', '\t', ' ', '　', '，', '。', '；', '：', '！', '？', '（', '）', '【', '】', '、', '…', '—'])
        text = ''.join([c for c in text if c.isprintable() or c in allowed_chars])
        text = text.strip(' \t\n\r　')
        return text

    def _get_split_separator(self, rule: str, custom_sep: str) -> list:
        split_sep_map = {
            "Line": ["\n", "\r\n"],
            "Space": [" ", "　"],
            "Comma": [",", "，"],
            "Period": [".", "。"],
            "Semicolon": [";", "；"],
            "Tab": ["\t"],
            "Pipe": ["|"],
            "Custom": [custom_sep.replace("\\n", "\n").replace("\\t", "\t").replace("\\s", " ").replace("\\u3000", "　")] if custom_sep else []
        }
        return split_sep_map.get(rule, ["\n"])

    def _get_join_separator(self, rule: str) -> str:
        join_sep_map = {
            "Line": "\n",
            "Space": " ",
            "Comma": "，",
            "Period": "。",
            "Semicolon": "；",
            "Tab": "\t",
            "Pipe": "|"
        }
        return join_sep_map.get(rule, "，")

    def _smart_split(self, text: str, separators: list) -> list:
        if not text:
            return []
        if not separators or all(not sep for sep in separators):
            cleaned = self._normalize_text(text)
            return [cleaned] if cleaned else []
        escaped_seps = []
        for sep in separators:
            if sep:
                escaped_seps.append(re.escape(sep))
        if not escaped_seps:
            cleaned = self._normalize_text(text)
            return [cleaned] if cleaned else []
        sep_pattern = '|'.join(escaped_seps)
        split_result = re.split(f'(?:{sep_pattern})', text)
        result = []
        for item in split_result:
            cleaned = self._normalize_text(item)
            if cleaned:
                result.append(cleaned)
        return result

    def _get_wildcard_content(self, wildcard_key: str, seed: int, input_split_rule: str, custom_split: str) -> str:
        if wildcard_key == "None":
            return ""
        target_dir = None
        wildcard_filename = ""
        if wildcard_key.startswith("dir1 | "):
            wildcard_filename = wildcard_key[len("dir1 | "):]
            target_dir = wildcards_dir1
        elif wildcard_key.startswith("base_path | "):
            wildcard_filename = wildcard_key[len("base_path | "):]
            target_dir = wildcards_dir2
        if not target_dir or not wildcard_filename:
            return ""
        wildcard_file = target_dir / f"{wildcard_filename}.txt"
        if not wildcard_file.exists():
            print(f"⚠️ 通配符文件未找到: {wildcard_file} (中文路径/文件名请确保编码正确)")
            return ""
        try:
            with open(wildcard_file, "r", encoding="utf-8", errors="replace") as f:
                file_content = f.read()
            separators = self._get_split_separator(input_split_rule, custom_split)
            split_lines = self._smart_split(file_content, separators)
            if not split_lines:
                print(f"⚠️ 通配符文件无有效内容: {wildcard_file}")
                return ""
            random.seed(seed)
            selected_content = random.choice(split_lines)
            return self._normalize_text(selected_content)
        except UnicodeDecodeError as e:
            print(f"⚠️ 通配符文件编码错误: {wildcard_file}，请确保文件为UTF-8编码。错误详情: {e}")
            return ""
        except Exception as e:
            print(f"⚠️ 读取通配符文件失败: {wildcard_file}，错误: {str(e)}")
            return ""

    def _replace_custom_placeholders(self, custom_join: str, text_dict: dict) -> str:
        if not custom_join:
            return ""
        result = self._normalize_text(custom_join)
        for i in range(0, 9):
            placeholder = f"{{text{i}}}"
            text_content = text_dict.get(f"text{i}", "")
            result = result.replace(placeholder, text_content)
        return self._normalize_text(result)

    def stack_Wildcards(self, output_join_rule: str, input_split_rule: str, custom_split: str, custom_join: str, seed: int, text0="", **kwargs):
        text_dict = {}
        non_empty_texts = []
        text0_content = self._normalize_text(text0)
        text_dict["text0"] = text0_content
        if text0_content:
            non_empty_texts.append(text0_content)
        for i in range(1, 9):
            text_key = f"text{i}"
            wildcard_selector = kwargs.get(text_key, "None")
            content = self._get_wildcard_content(wildcard_selector, seed, input_split_rule, custom_split)
            text_dict[text_key] = content
            if content:
                non_empty_texts.append(content)
        if output_join_rule == "Custom":
            joined_content = self._replace_custom_placeholders(custom_join, text_dict)
        else:
            separator = self._get_join_separator(output_join_rule)
            joined_content = separator.join(non_empty_texts) if non_empty_texts else ""
        final_result = self._normalize_text(joined_content)
        if not final_result:
            final_result = "❌ 无有效通配符内容"
        return (final_result,)

#endregion--------------------






import itertools



class text_StrMatrix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fstring": ("STRING", {
                    "multiline": True,
                    "default": "{a}_{b}_{c}_{d}",
                }),
                "max_count": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 999,
                    "step": 1,
                }),
            },
            "optional": {
                "a": (ANY_TYPE, {"default": None}),
                "b": (ANY_TYPE, {"default": None}),
                "c": (ANY_TYPE, {"default": None}),
                "d": (ANY_TYPE, {"default": None}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_NAMES = ("string_Matrix","string_lsit",)
    RETURN_TYPES = (ANY_TYPE,"LIST",)
    OUTPUT_IS_LIST = (True,False,)

    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"
    
    def execute(self, fstring, max_count, a=[], b=[], c=[], d=[]):
        fstring_template = fstring[0] if isinstance(fstring, list) and len(fstring) > 0 else "{a}_{b}_{c}_{d}"
        
        normalized = []
        for lst in [a, b, c, d]:
            if isinstance(lst, list) and len(lst) > 0:
                normalized.append(lst)
            else:
                normalized.append([None])
        
        formatted_strings = []
        current_count = 0
        max_count = max_count[0] if isinstance(max_count, list) else max_count
        
        try:
            for combination in itertools.product(*normalized):
                if current_count >= max_count:
                    break
                
                fmt_dict = {
                    'a': combination[0],
                    'b': combination[1],
                    'c': combination[2],
                    'd': combination[3]
                }
                
                fmt_dict = {k: v if v is not None else "" for k, v in fmt_dict.items()}
                formatted = fstring_template.format(**fmt_dict)
                formatted_strings.append(formatted)
                
                current_count += 1
        except Exception as e:
            print(f"[text_StrMatrix] 执行错误: {e}")
            formatted_strings = []

        return (formatted_strings,formatted_strings,)













