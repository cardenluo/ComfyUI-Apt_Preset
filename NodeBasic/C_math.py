import math
import torch
from typing import Any, Callable, Mapping
import comfy
import numpy as np
from nodes import NODE_CLASS_MAPPINGS 

from ..main_unit import *


#region-----------------def---------------


def get_input_nodes(extra_pnginfo, unique_id):
    node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
    node = next(n for n in node_list if n["id"] == unique_id)
    input_nodes = []
    for i, input in enumerate(node["inputs"]):
        link_id = input["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_nodes.append(in_node)
    return input_nodes


def get_input_types(extra_pnginfo, unique_id):
    node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
    node = next(n for n in node_list if n["id"] == unique_id)
    input_types = []
    for i, input in enumerate(node["inputs"]):
        link_id = input["link"]
        link = next(l for l in extra_pnginfo["workflow"]["links"] if l[0] == link_id)
        in_node_id, in_socket_id = link[1], link[2]
        in_node = next(n for n in node_list if n["id"] == in_node_id)
        input_type = in_node["outputs"][in_socket_id]["type"]
        input_types.append(input_type)
    return input_types


def keyframe_scheduler(schedule, schedule_alias, current_frame):
    schedule_lines = list()
    previous_params = ""
    for item in schedule:   
        alias = item[0]
        if alias == schedule_alias:
            schedule_lines.extend([(item)])
    for i, item in enumerate(schedule_lines):
        alias, line = item
        if not line.strip():
            print(f"[Warning] Skipped blank line at line {i}")
            continue
        frame_str, params = line.split('@', 1)
        frame = int(frame_str)
        params = params.lstrip()
        if frame < current_frame:
            previous_params = params
            continue
        if frame == current_frame:
            previous_params = params
        else:
            params = previous_params
        return params
    return previous_params

def prompt_scheduler(schedule, schedule_alias, current_frame):
    schedule_lines = list()
    previous_prompt = ""
    previous_keyframe = 0
    for item in schedule:   
        alias = item[0]
        if alias == schedule_alias:
            schedule_lines.extend([(item)])
    for i, item in enumerate(schedule_lines):
        alias, line = item
        frame_str, prompt = line.split('@', 1)
        frame_str = frame_str.strip('\"')
        frame = int(frame_str)
        prompt = prompt.lstrip()
        prompt = prompt.replace('"', '')        
        if frame < current_frame:
            previous_prompt = prompt
            previous_keyframe = frame
            continue
        elif frame == current_frame:
            next_prompt = prompt
            next_keyframe = frame             
            previous_prompt = prompt
            previous_keyframe = frame
        else:
            next_prompt = prompt
            next_keyframe = frame            
            prompt = previous_prompt
        return prompt, next_prompt, previous_keyframe, next_keyframe
    return previous_prompt, previous_prompt, previous_keyframe, previous_keyframe


#endregion---------------def---------------



#region---------------math---------------------


class math_Remap_data:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clamp": ("BOOLEAN", {"default": False}),
                "source_min": ("FLOAT", {"default": 0.0, "min": -9999, "max": 9999, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "min": -9999, "max": 9999, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -9999, "max": 9999, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -9999, "max": 9999, "step": 0.01}),
                "easing": (EASING_TYPES, {"default": "Linear"}),
            },
            "optional": {
                "value": (ANY_TYPE,),  # 移到optional中，变为可选项
            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT", "INT",)
    RETURN_NAMES = ("float", "int",)
    CATEGORY = "Apt_Preset/data"

    def set_range(
        self,
        clamp,
        source_min,
        source_max,
        target_min,
        target_max,
        easing,
        value=None,  # 设为可选参数，默认值为None
    ):
        # 处理value为None的情况（未输入时），这里默认设为0.0，可根据需求调整
        if value is None:
            float_value = 0.0
        else:
            try:
                float_value = float(value)
            except ValueError:
                raise ValueError("Invalid value for conversion to float")
        
        if source_min == source_max:
            normalized_value = 0
        else:
            normalized_value = (float_value - source_min) / (source_max - source_min)
        if clamp:
            normalized_value = max(min(normalized_value, 1), 0)
        eased_value = apply_easing(normalized_value, easing)
        if clamp:
           eased_value = max(min(eased_value, 1), 0)
        res_float = target_min + (target_max - target_min) * eased_value
        res_int = int(res_float)

        return (res_float, res_int)






class math_calculate:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        # 定义预设运算列表（使用扁平化命名，避免斜杠等特殊符号）
        presets = [
            # 单值运算
            ("custom", "自定义表达式"),
            ("sin(a)", "正弦函数 sin(a)"),
            ("cos(a)", "余弦函数 cos(a)"),
            ("tan(a)", "正切函数 tan(a)"),
            ("asin(a)", "反正弦函数 asin(a)"),
            ("acos(a)", "反余弦函数 acos(a)"),
            ("atan(a)", "反正切函数 atan(a)"),
            ("pow(a, 2)", "平方 a²"),
            ("sqrt(a)", "平方根 √a"),
            ("log(a)", "自然对数 log(a)"),
            ("log10(a)", "常用对数 log10(a)"),
            ("sinh(a)", "双曲正弦 sinh(a)"),
            ("cosh(a)", "双曲余弦 cosh(a)"),
            ("tanh(a)", "双曲正切 tanh(a)"),
            ("asinh(a)", "反双曲正弦 asinh(a)"),
            ("acosh(a)", "反双曲余弦 acosh(a)"),
            ("atanh(a)", "反双曲正切 atanh(a)"),
            ("radians(a)", "角度转弧度 radians(a)"),
            ("degrees(a)", "弧度转角度 degrees(a)"),
            ("fabs(a)", "绝对值 fabs(a)"),
            ("exp(a)", "指数函数 e的a次方"),
            ("round(a)", "四舍五入 round(a)"),
            ("ceil(a)", "向上取整 ceil(a)"),
            ("floor(a)", "向下取整 floor(a)"),
            ("abs(a)", "绝对值 abs(a)"),
            
            # 双值运算
            ("a + b", "加法 a + b"),
            ("a - b", "减法 a - b"),
            ("a * b", "乘法 a * b"),
            ("a ÷ b", "除法 a 除以 b"),
            ("a % b", "取模 a 模 b"),
            ("pow(a,b)", "幂运算 a的b次方"),
            ("ceil(a÷b)", "向上取整 ceil(a÷b)"),
            ("floor(a÷b)", "向下取整 floor(a÷b)"),
            ("max(a,b)", "最大值 max(a,b)"),
            ("min(a,b)", "最小值 min(a,b)"),
            ("a > b", "大于 a > b"),
            ("a < b", "小于 a < b"),
            ("a >= b", "大于等于 a >= b"),
            ("a <= b", "小于等于 a <= b"),
            ("a == b", "等于 a == b"),
            ("a != b", "不等于 a != b"),
            ("a & b", "按位与 a & b"),
            ("a | b", "按位或 a | b"),
            ("a ^ b", "按位异或 a ^ b"),
            ("a << b", "左移位 a << b"),
            ("a >> b", "右移位 a >> b"),
            ("atan2(a,b)", "四象限反正切 atan2(a,b)"),
            ("hypot(a,b)", "直角三角形斜边 hypot(a,b)"),
            ("copysign(a,b)", "复制符号 copysign(a,b)"),
            ("fmod(a,b)", "浮点数取模 fmod(a,b)"),
            
            # 三值运算（仅保留三个值的最大值和最小值）
            ("max(a,b,c)", "最大值 max(a,b,c)"),
            ("min(a,b,c)", "最小值 min(a,b,c)"),
            ("clamp(a,b,c)", "限制在b和c之间 clamp(a,b,c)"),
            ("lerp(a,b,c)", "线性插值 lerp(a,b,c)"),
        ]
        
        return {
            "required": {},
            "optional": {
                "a": (ANY_TYPE,),
                "b": (ANY_TYPE,),
                "c": (ANY_TYPE,),
                "preset": (
                    [p[0] for p in presets], 
                    {"default": "custom", "label": [p[1] for p in presets]}
                ),
                "expression": ("STRING", {"default": "", "multiline": False,}),
            }
        }

    RETURN_TYPES = ("FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("float_result", "int_result", "bool_result")
    FUNCTION = "calculate"
    CATEGORY = "Apt_Preset/data"
    DESCRIPTION = """
    - 基本运算：加 (+)、减 (-)、乘 (*)、除 (/)、模 (%)
    - 三角函数: sin(a)、cos、tan、asin、acos、atan、atan2(a,b)
    - 幂运算与开方: pow(a,2)=a*a、sqrt、hypot(a,b)
    - 对数运算: log、log10(a)、exp(a)
    - 双曲函数: sinh、cosh、tanh、asinh、acosh、atanh
    - 角度与弧度转换: radians、degrees
    - 绝对值与取整: fabs、abs、ceil、floor、round、sign
    - 位运算: &(与)、|(或)、^(异或)、<<(左移)、>>(右移)
    - 比大小: max(a,b,c) ,min(a,b,c)
    - 布尔运算: a>b,a<b,a>=b,a<=b,a==b,a!=b ,返回True或False
    - 其他运算: clamp(a,b,c)、lerp(a,b,c)、if(a,b,c)、copysign(a,b)、fmod(a,b)
    """

    def calculate(self, preset="custom", expression="", a=None, b=None, c=None):
        try:
            # 处理空值情况
            a = a if a is not None else 0
            b = b if b is not None else 0
            c = c if c is not None else 0
            
            # 类型转换：确保输入是数值类型
            def to_num(x):
                if isinstance(x, torch.Tensor):
                    return x.item() if x.numel() == 1 else 0
                try:
                    return float(x)
                except (ValueError, TypeError):
                    return 0
            
            a = to_num(a)
            b = to_num(b)
            c = to_num(c)
            
            if preset != "custom":
                current_expression = preset.replace("÷", "/")
            else:
                current_expression = expression.strip()
                if not current_expression:
                    raise ValueError("自定义表达式不能为空")
            
            namespace = {
                'a': a,
                'b': b,
                'c': c,
                'math': math,
                'sin': math.sin,
                'cos': math.cos,
                'tan': math.tan,
                'asin': math.asin,
                'acos': math.acos,
                'atan': math.atan,
                'atan2': math.atan2,
                'pow': math.pow,
                'sqrt': math.sqrt,
                'hypot': math.hypot,
                'log': math.log,
                'log10': math.log10,
                'exp': math.exp,
                'sinh': math.sinh,
                'cosh': math.cosh,
                'tanh': math.tanh,
                'asinh': math.asinh,
                'acosh': math.acosh,
                'atanh': math.atanh,
                'radians': math.radians,
                'degrees': math.degrees,
                'fabs': math.fabs,
                'abs': abs,
                'ceil': math.ceil,
                'floor': math.floor,
                'round': round,
                'max': max,
                'min': min,
                'copysign': math.copysign,
                'fmod': math.fmod,
                'clamp': lambda x, min_val, max_val: max(min(x, max_val), min_val),
                'lerp': lambda a, b, c: a + (b - a) * c,
            }
            
            # 执行表达式计算
            result = eval(current_expression, {"__builtins__": None}, namespace)
            
            # 处理不同类型的返回值
            float_result = float(result) if result is not None else 0.0
            int_result = int(round(float_result))  # 四舍五入转整数
            bool_result = bool(result) if isinstance(result, (int, float, bool)) else False
            
            return (float_result, int_result, bool_result)
            
        except Exception as e:
            print(f"Math calculation error: {e}")
            return (0.0, 0, False)


class math_text_compare:
    @classmethod
    def INPUT_TYPES(cls):
        compare_modes = [
            "完全相等",
            "不相等",
            "A包含B",
            "A不包含B",
            "文本是否为空",
            "长度A=长度B",
            "长度A<长度B",
            "长度A>长度B",
            "正则匹配",
        ]

        return {
            "required": {
                "ignore_case": ("BOOLEAN", {"default": True, "label_on": "忽略大小写", "label_off": "区分大小写"}),
                "mode": (compare_modes, {"default": "完全相等"}),
                "text_a": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "text_b": ("STRING", {"default": "", "multiline": True}),
                "regex_pattern": ("STRING", {"default": "", "multiline": False}),

            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("bool",)
    FUNCTION = "compare_text"
    CATEGORY = "Apt_Preset/data"
    DESCRIPTION = """
    - 大小写：先按 ignore_case 决定是否忽略大小写，再执行文本比较
    - 文本比较：完全相等 / 不相等 / 包含 / 不包含
    - 空值判断：把空字符串和纯空白都视为空
    - 长度对比：长度A=长度B / 长度A<长度B / 长度A>长度B
    - 正则匹配：开头/结尾/复杂规则建议用正则；完整匹配请直接写 ^...$
    - 开头匹配： ^black
    - 结尾匹配： black$
    - 完整匹配： ^black$
    """

    def compare_text(self, mode, text_a, text_b="", regex_pattern="", ignore_case=True):
        try:
            text_a = "" if text_a is None else str(text_a)
            text_b = "" if text_b is None else str(text_b)
            regex_pattern = "" if regex_pattern is None else str(regex_pattern)

            compare_text_a = text_a.casefold() if ignore_case else text_a
            compare_text_b = text_b.casefold() if ignore_case else text_b
            text_a_length = len(text_a)
            text_b_length = len(text_b)
            result = False

            if mode == "完全相等":
                result = compare_text_a == compare_text_b
            elif mode == "不相等":
                result = compare_text_a != compare_text_b
            elif mode == "A包含B":
                result = bool(compare_text_b) and compare_text_b in compare_text_a
            elif mode == "A不包含B":
                result = bool(compare_text_b) and compare_text_b not in compare_text_a
            elif mode == "文本是否为空":
                result = text_a.strip() == ""
            elif mode == "长度A=长度B":
                result = text_a_length == text_b_length
            elif mode == "长度A<长度B":
                result = text_a_length < text_b_length
            elif mode == "长度A>长度B":
                result = text_a_length > text_b_length
            elif mode == "正则匹配":
                if regex_pattern:
                    flags = re.IGNORECASE if ignore_case else 0
                    pattern = re.compile(regex_pattern, flags)
                    match = pattern.search(text_a)
                    result = match is not None
                else:
                    result = False

            return (result,)
        except re.error as e:
            print(f"Text compare regex error: {e}")
            return (False,)
        except Exception as e:
            print(f"Text compare error: {e}")
            return (False,)






#endregion---------------------math---------------------



#region---------------list---------------


class list_Slice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "list": (ANY_TYPE, {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": -1, "min": -9007199254740991}),
            }
        }
    
    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("data",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, )  # 确保输出是列表形式
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, list: list, start: list[int], end: list[int]):
        # 从输入列表中获取起始和结束值
        start_val = start[0] if start else 0
        end_val = end[0] if end else -1
        
        # 处理负数索引
        if start_val < 0:
            start_val = len(list) + start_val
        if end_val < 0:
            end_val = len(list) + end_val
        
        # 确保索引在有效范围内
        start_val = max(0, min(start_val, len(list)))
        end_val = max(0, min(end_val, len(list)))
        
        # 确保start不大于end
        if start_val > end_val:
            return ([], )

        if start_val == end_val and start_val < len(list):
            return ([list[start_val]], )
            
        # 执行切片操作
        sliced = list[start_val:end_val]
        return (sliced, )


class list_fliter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "filter_zero_content": ("BOOLEAN", {"default": True, "label_on": "filter", "label_off": "keep"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"
    DESCRIPTION = """
    过滤图片列表或遮罩列表中的空项。
    - 固定过滤：None、非法张量、空张量
    - filter_zero_content 开启时：全0图像、全0遮罩也会被过滤
    """

    @staticmethod
    def _to_items(data, data_type):
        if data is None:
            return []
        if isinstance(data, (list, tuple)):
            items = list(data)
        else:
            if not isinstance(data, torch.Tensor):
                return []
            if data.numel() == 0:
                return []

            if data_type == "image":
                if data.ndim == 4:
                    items = [data[i:i + 1] for i in range(data.shape[0])]
                elif data.ndim == 3:
                    items = [data.unsqueeze(0)]
                else:
                    items = [data]
            else:
                if data.ndim == 3:
                    items = [data[i:i + 1] for i in range(data.shape[0])]
                elif data.ndim == 2:
                    items = [data.unsqueeze(0)]
                else:
                    items = [data]

        normalized_items = []
        for item in items:
            if not isinstance(item, torch.Tensor):
                normalized_items.append(item)
                continue
            if data_type == "image" and item.ndim == 3:
                normalized_items.append(item.unsqueeze(0))
            elif data_type == "mask" and item.ndim == 2:
                normalized_items.append(item.unsqueeze(0))
            else:
                normalized_items.append(item)
        return normalized_items

    @staticmethod
    def _is_valid_item(item, data_type, filter_zero_content):
        if item is None or not isinstance(item, torch.Tensor):
            return False
        if item.numel() == 0:
            return False
        if any(dim <= 0 for dim in item.shape):
            return False

        if data_type == "image" and item.ndim not in (3, 4):
            return False
        if data_type == "mask" and item.ndim not in (2, 3):
            return False

        if filter_zero_content and not torch.any(item != 0):
            return False
        return True

    def run(self, filter_zero_content=True, images=None, masks=None):
        image_items = self._to_items(images, "image")
        mask_items = self._to_items(masks, "mask")

        valid_images = [
            item for item in image_items
            if self._is_valid_item(item, "image", filter_zero_content)
        ]
        valid_masks = [
            item for item in mask_items
            if self._is_valid_item(item, "mask", filter_zero_content)
        ]

        return (valid_images, valid_masks)






class list_value_range:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0}),
                "stop": ("FLOAT", {"default": 1}),
                "total": ("INT", {"default": 10, "min": 2}),
            },
        }
    
    TITLE = "Create Linspace"
    RETURN_TYPES = ("FLOAT", "LIST", "INT")
    RETURN_NAMES = ("data", "list", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, start: float, stop: float, total: int):
        range_list = list(np.linspace(start, stop, total))
        return (range_list, range_list, len(range_list))


#endregion---------------list---------------



#region---------------sch----------------




import re

class sch_split_text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "a b c"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 99999}),
            },
            "optional": {
                "preset": (
                    ["None", "Line", "Space", "Comma", "Period", "Semicolon", "Tab", "Pipe", "Custom"], 
                    {"default": "None"}
                ),
                "delimiter": (
                    "STRING", 
                    {"default": " ", "label": "Custom delimiter", }
                ),
            }
        }

    RETURN_TYPES = (ANY_TYPE, "STRING",)
    RETURN_NAMES = ('index_text', "list_text",)
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "text_to_list"
    CATEGORY = "Apt_Preset/IO_Port/schedule"
    DESCRIPTION = """
    None：不使用预设分隔符
    Line：按换行符拆分
    Space：按空格拆分
    Comma：按逗号拆分
    Period：按句号拆分
    Semicolon：按分号拆分
    Tab：按制表符拆分 \t
    Pipe：按竖线拆分 |
    Custom：使用自定义分隔符
    """ 

    def text_to_list(self, text, index, preset="None", delimiter=" "):
        preset_map = {
            "None": [],
            "Line": ["\n", "\r\n"],
            "Space": [" ", "　"],
            "Comma": [",", "，"],
            "Period": [".", "。"],
            "Semicolon": [";", "；"],
            "Tab": ["\t"],
            "Pipe": ["|"],
            "Custom": []
        }
        separators = preset_map.get(preset, [])
        
        if (preset == "Custom" or (preset == "None" and delimiter)) and delimiter:
            delimiter = delimiter.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")
            separators = [delimiter]
        
        if text.strip() == "":
            strList = []
        elif not separators:
            strList = [text.strip()] if text.strip() else []
        else:
            escaped_seps = [re.escape(sep) for sep in separators if sep]
            sep_pattern = '|'.join(escaped_seps)
            strList = re.split(f'(?:{sep_pattern})', text.strip())
            strList = [item.strip() for item in strList if item.strip()]
        
        list_length = len(strList)
        if index < 0 or index >= list_length:
            return ("", strList)
        
        return (strList[index], strList)



class sch_text:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"keyframe_list": ("STRING", {"multiline": True, "default": "index@text"}),
                            "current_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                },
                "optional": {
                }

        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("index_prompt", "list_prompt")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "simple_schedule"
    CATEGORY = "Apt_Preset/IO_Port/schedule"

    def simple_schedule(self, keyframe_list, current_index):
        keyframes = []
        for line in keyframe_list.splitlines():
            if line.strip():
                index, prompt = line.split('@', 1)
                keyframes.append((int(index.strip().strip('"')), prompt.strip().replace('"', '')))
        keyframes.sort(key=lambda item: item[0])
        index_prompt = next((prompt for index, prompt in reversed(keyframes) if index <= current_index), "")
        return (index_prompt, [prompt for _, prompt in keyframes])




class sch_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "schedule": ("STRING", {"multiline": True, "default": "index@value"}),
                "current_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "easing_type": (list(easing_functions.keys()), ),
            },
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "INT", "FLOAT")
    RETURN_NAMES = ("index_int", "index_float", "list_int", "list_float")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/IO_Port/schedule"


    def adv_schedule(self, schedule, current_index, easing_type):
        keyframes = []
        for line in schedule.splitlines():
            if line.strip():
                index, value = line.split('@', 1)
                keyframes.append((int(index.strip().strip('"')), float(value.strip())))
        keyframes.sort(key=lambda item: item[0])
        list_float = [value for _, value in keyframes]
        list_int = [int(value) for value in list_float]
        value_out = 0.0
        if keyframes and current_index >= keyframes[0][0]:
            value_out = keyframes[-1][1]
            for (from_index, current_value), (to_index, next_value) in zip(keyframes, keyframes[1:]):
                if current_index < to_index:
                    t = (current_index - from_index) / (to_index - from_index)
                    value_out = current_value + (next_value - current_value) * apply_easing(t, easing_type)
                    break
        return (int(value_out), value_out, list_int, list_float)




#endregion---------------batch_cycler---------------




class array_Slice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "array": ("LIST", {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": -1, "min": -9007199254740991}),  # 默认-1表示到末尾
            }
        }
    
    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("data", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Array"

    def run(self, array: list, start: int, end: int):
        list_length = len(array)
        
        # 处理负数索引
        if start < 0:
            start = list_length + start
        if end < 0:
            end = list_length + end
            
        # 确保索引在有效范围内
        start = max(0, min(start, list_length))
        end = max(0, min(end, list_length))
        
        # 确保start不大于end
        if start > end:
            # 返回空列表或适当的默认值
            # 检查输入数据类型以返回相应类型的空值
            if list_length > 0 and isinstance(array[0], torch.Tensor):
                # 如果是张量列表，返回空的张量
                return (torch.tensor([]), )
            return ([], )

        if start == end and start < list_length:
            return (array[start], )
            
        # 执行切片操作
        sliced_data = array[start:end]
        
        # 如果列表中的元素是张量，考虑将它们堆叠成一个张量
        if len(sliced_data) > 0 and isinstance(sliced_data[0], torch.Tensor):
            try:
                if sliced_data[0].ndim in (3, 4):
                    return (torch.cat(sliced_data, dim=0), )
                # 如果是相同形状的张量，尝试堆叠它们
                return (torch.stack(sliced_data), )
            except RuntimeError:
                # 如果形状不匹配，返回原列表
                return (sliced_data, )
        
        return (sliced_data, )






class type_AnyIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, ),
                "index": ("INT", {"default": 0, "min": -999990, "max": 1000000, "step": 1, "tooltip":"从0开始，0对应第一个元素；负数表示倒数，-1为最后一个元素"}),
            },
            "hidden":{
                "prompt": "PROMPT",
                "my_unique_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("out",)
    INPUT_IS_LIST = True
    FUNCTION = "getIndex"
    CATEGORY = "Apt_Preset/data"
    DESCRIPTION = """
    注意：第一元素是 0 ，最后一个是-1
    """

    def getIndex(self, any, index, prompt=None, my_unique_id=None):
        index = index[0]
        prompt = prompt[0]
        my_unique_id = my_unique_id[0]
        my_unique_id = my_unique_id.split('.')[len(my_unique_id.split('.')) - 1] if "." in my_unique_id else my_unique_id
        id, slot = prompt[my_unique_id]['inputs']['any']
        class_type = prompt[id]['class_type']
        node_class = NODE_CLASS_MAPPINGS [class_type]
        output_is_list = node_class.OUTPUT_IS_LIST[slot] if hasattr(node_class, 'OUTPUT_IS_LIST') else False

        if output_is_list or len(any) > 1:
            total = len(any)
            if index < 0:
                index = max(0, total + index)
            safe_index = min(index, total - 1) if total > 0 else 0
            return (any[safe_index],)
        elif isinstance(any[0], torch.Tensor):
            total = any[0].shape[0]
            if index < 0:
                index = max(0, total + index)
            batch_index = min(index, total - 1) if total > 0 else 0
            s = any[0][batch_index:batch_index + 1].clone()
            return (s,)
        else:
            total = len(any[0])
            if index < 0:
                index = max(0, total + index)
            safe_index = min(index, total - 1) if total > 0 else 0
            return (any[0][safe_index],)







