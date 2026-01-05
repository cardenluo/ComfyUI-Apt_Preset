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
                "source_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "source_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
                "target_min": ("FLOAT", {"default": 0.0, "min": -999, "max": 999, "step": 0.01}),
                "target_max": ("FLOAT", {"default": 1.0, "min": -999, "max": 999, "step": 0.01}),
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
            "required": {
                "preset": (
                    [p[0] for p in presets], 
                    {"default": "custom", "label": [p[1] for p in presets]}
                ),
                "expression": ("STRING", {"default": "", "multiline": False,}),
                "a": (ANY_TYPE, {"forceInput": True}),
            },
            "optional": {
                "b": (ANY_TYPE,),
                "c": (ANY_TYPE,),
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


    def calculate(self, preset, expression, a, b=None, c=None):
        try:
            # 确定使用预设表达式还是自定义表达式
            if preset != "custom":
                # 将显示用的÷替换回计算用的/
                current_expression = preset.replace("÷", "/")
            else:
                current_expression = expression
                
            # 定义命名空间，将输入变量和常用数学函数添加到其中
            namespace = {
                'a': a,
                'b': b if b is not None else 0,
                'c': c if c is not None else 0,
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
                'clamp': lambda x, min_val, max_val: max(min(x, max_val), min_val),  # 限制范围
                'lerp': lambda a, b, c: a + (b - a) * c,  # 线性插值
            }
            # 执行表达式计算
            result = eval(current_expression, namespace)
            
            # 转换为指定返回类型
            float_result = float(result)
            int_result = int(result)
            bool_result = bool(result)
            
            return (float_result, int_result, bool_result)
        except Exception as e:
            print(f"Error performing calculation: {e}")
            return (0.0, 0, False)

#endregion---------------------math---------------------



#region---------------list---------------


class list_Slice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ANY": (ANY_TYPE, {"forceInput": True}),
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

    def run(self, ANY: list, start: list[int], end: list[int]):
        # 从输入列表中获取起始和结束值
        start_val = start[0] if start else 0
        end_val = end[0] if end else -1
        
        # 处理负数索引
        if start_val < 0:
            start_val = len(ANY) + start_val
        if end_val < 0:
            end_val = len(ANY) + end_val
        
        # 确保索引在有效范围内
        start_val = max(0, min(start_val, len(ANY)))
        end_val = max(0, min(end_val, len(ANY)))
        
        # 确保start不大于end
        if start_val > end_val:
            return ([], )
            
        # 执行切片操作
        sliced = ANY[start_val:end_val]
        return (sliced, )


class list_Merge:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "list_Merge"
    INPUT_IS_LIST = True
    RETURN_TYPES = (ANY_TYPE, )
    OUTPUT_IS_LIST = (True, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        unique_id = unique_id[0]
        prompt = prompt[0]
        extra_pnginfo = extra_pnginfo[0]
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith('value'):
                output_list += v
        return (output_list, )



class list_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"schedule": ("STRING", {"multiline": True, "default": "frame_number@value"}),
                             "max_frames": ("INT", {"default": 100, "min": 1, "max": 99999}),  # 添加 max_frames 参数
                             "easing_type": (list(easing_functions.keys()), ),
                },
        }
    RETURN_TYPES = ("FLOAT","INT",  "FLOAT")
    RETURN_NAMES = ("float","int",  "weight")
    OUTPUT_IS_LIST = (True, True, True)  # 强制输出结果为列表
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def adv_schedule(self, schedule, max_frames, easing_type):
        schedule_lines = list()
        if schedule == "":
            print(f"[Warning] CR Advanced Value Scheduler. No lines in schedule")
            return ([], [], [])  # 返回空列表

        lines = schedule.split('\n')
        for line in lines:
            schedule_lines.extend([("ADV", line)])

        int_out_list = []
        value_out_list = []
        weight_list = []

        for current_frame in range(max_frames):
            params = keyframe_scheduler(schedule_lines, "ADV", current_frame)
            if params == "":
                print(f"[Warning] CR Advanced Value Scheduler. No schedule found for frame {current_frame}. Advanced schedules must start at frame 0.")
                int_out_list.append(0)
                value_out_list.append(0.0)
                weight_list.append(1.0)
                continue

            try:
                current_params, next_params, from_index, to_index = prompt_scheduler(schedule_lines, "ADV", current_frame)
                if to_index == from_index:
                    t = 1.0
                else:
                    t = (current_frame - from_index) / (to_index - from_index)
                if t < 0 or t > 1:
                    t = 1.0
                weight = apply_easing(t, easing_type)
                current_value = float(current_params)
                next_value = float(next_params)
                value_out = current_value + (next_value - current_value) * weight
                int_out = int(value_out)

                int_out_list.append(int_out)
                value_out_list.append(value_out)
                weight_list.append(weight)
            except ValueError:
                print(f"[Warning] CR Advanced Value Scheduler. Invalid params at frame {current_frame}")
                int_out_list.append(0)
                value_out_list.append(0.0)
                weight_list.append(1.0)

        return ( value_out_list, int_out_list, weight_list)



class list_num_range:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start": ("FLOAT", {"default": 0}),
                "stop": ("FLOAT", {"default": 1}),
                "num": ("INT", {"default": 10, "min": 2}),
            },
        }
    
    TITLE = "Create Linspace"
    RETURN_TYPES = ("FLOAT", "LIST", "INT")
    RETURN_NAMES = ("data", "list", "length")
    OUTPUT_IS_LIST = (True, False, False, )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, start: float, stop: float, num: int):
        range_list = list(np.linspace(start, stop, num))
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
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
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

    RETURN_TYPES = (ANY_TYPE, "INT",)
    RETURN_NAMES = ('i_text', "length",)
    FUNCTION = "text_to_list"
    CATEGORY = "Apt_Preset/data/schedule"
    DESCRIPTION = """
    文本拆分预设说明
    - **None**：不采用预设分隔符。
    - **Line**：以`\n`或`\r\n`（换行符）拆分。
    - **Space**：用` `（空格）或`　`（全角空格）拆分。
    - **Comma**：以`,`（逗号）或`，`（中文逗号）拆分。
    - **Period**：用`.`（句号）或`。`（中文句号）拆分。
    - **Semicolon**：以`;`（分号）或`；`（中文分号）拆分。
    - **Tab**：用`\t`（制表符）拆分。
    - **Pipe**：以`|`（竖线）拆分。
    - **Custom**：使用自定义分隔符（支持转义字符如`\\n`、`\\t`）。
    """ 

    def text_to_list(self, text, current_frame, preset="None", delimiter=" "):
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
        if current_frame < 0 or current_frame >= list_length:
            return ("", list_length)
        
        return (strList[current_frame], list_length)



class sch_text:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"keyframe_list": ("STRING", {"multiline": True, "default": "frame_number@text"}),  
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "easing_type": (list(easing_functions.keys()), ),
                },
                "optional": {
                }

        }
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("current_prompt", "next_prompt", "weight")
    FUNCTION = "simple_schedule"
    CATEGORY = "Apt_Preset/data/schedule"

    def simple_schedule(self, keyframe_list,  current_frame, easing_type,):
        keyframes = list()
        if keyframe_list == "":
            print(f"[Error] CR Simple Prompt Scheduler. No lines in keyframe list") 
            return ()   
        lines = keyframe_list.split('\n')
        for line in lines:
            if not line.strip():
                print(f"[Warning] CR Simple Prompt Scheduler. Skipped blank line at line {i}")
                continue                  
            keyframes.extend([("SIMPLE", line)])        
        current_prompt, next_prompt, current_keyframe, next_keyframe = prompt_scheduler(keyframes, "SIMPLE", current_frame)
        if current_prompt == "":
            print(f"[Warning] CR Simple Prompt Scheduler. No prompt found for frame. Simple schedules must start at frame 0.")
        else:        
            try:
                current_prompt_out = str(current_prompt)
                next_prompt_out = str(next_prompt)
                from_index = int(current_keyframe)
                to_index = int(next_keyframe)
            except ValueError:
                print(f"[Warning] CR Simple Text Scheduler. Invalid keyframe at frame {current_frame}")
            
            if from_index == to_index:
                weight_out = 1.0
            else:
                # 缓入缓出效果
                t = (to_index - current_frame) / (to_index - from_index)

                weight_out =  apply_easing(t, easing_type) 


            return(current_prompt_out, next_prompt_out, weight_out)




class sch_Value:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "schedule": ("STRING", {"multiline": True, "default": "frame_number@value"}),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "easing_type": (list(easing_functions.keys()), ),
            },
        }
    
    RETURN_TYPES = ("INT", "FLOAT", "FLOAT", "INT")
    RETURN_NAMES = ("INT", "FLOAT", "weight", "length")
    FUNCTION = "adv_schedule"
    CATEGORY = "Apt_Preset/data/schedule"

    def adv_schedule(self, schedule, current_frame, easing_type):
        int_out = 0
        value_out = 0.0
        weight = 0.0
        data_length = 0
        
        if schedule.strip() == "":
            print(f"[Warning] CR Advanced Value Scheduler. No lines in schedule")
        else:
            lines = [line.strip() for line in schedule.split('\n') if line.strip()]
            frame_numbers = []
            for line in lines:
                if '@' in line:
                    try:
                        frame_part = line.split('@')[0].strip()
                        frame = int(frame_part)
                        frame_numbers.append(frame)
                    except ValueError:
                        print(f"[Warning] CR Advanced Value Scheduler. Invalid frame number: {frame_part} in line: {line}")
            
            if frame_numbers:
                min_frame = min(frame_numbers)
                max_frame = max(frame_numbers)
                data_length = max_frame - min_frame + 1
                print(f"[Info] CR Advanced Value Scheduler. Actual data length (frame range): {data_length} frames (from {min_frame} to {max_frame})")
            else:
                print(f"[Warning] CR Advanced Value Scheduler. No valid frame numbers found in schedule")
        
        schedule_lines = list()
        if schedule == "":
            return (int_out, value_out, weight, data_length)
        
        lines = schedule.split('\n')
        for line in lines:        
            if line.strip():
                schedule_lines.extend([("ADV", line)])        
        
        params = keyframe_scheduler(schedule_lines, "ADV", current_frame)
        if params == "":
            print(f"[Warning] CR Advanced Value Scheduler. No schedule found for frame. Advanced schedules must start at frame 0.")
        else:
            try:
                current_params, next_params, from_index, to_index = prompt_scheduler(schedule_lines, "ADV", current_frame)
                if to_index == from_index:
                    t = 1.0
                else:
                    t = (current_frame - from_index) / (to_index - from_index)
                if t < 0 or t > 1:
                    t = 1.0
                weight = apply_easing(t, easing_type)
                current_value = float(current_params)
                next_value = float(next_params)
                value_out = current_value + (next_value - current_value) * weight
                int_out = int(value_out)
                
                print(f"[Info] CR Advanced Value Scheduler. Processing frame {current_frame}, current data index range: {from_index} -> {to_index}")
                
            except ValueError as e:
                print(f"[Warning] CR Advanced Value Scheduler. Invalid params at frame {current_frame}: {str(e)}")
        
        return (int_out, value_out, weight, data_length)



class sch_Prompt:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"clip": ("CLIP",),
                            "keyframe_list": ("STRING", {"multiline": True, "default": "frame_number@text"}),  
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),
                            "easing_type": (list(easing_functions.keys()), ),
                            }
        }
    
    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("CONDITIONING", )
    FUNCTION = "condition"
    CATEGORY = "Apt_Preset/data/schedule"

    def condition(self, clip, keyframe_list, current_frame, easing_type):      
        
        (current_prompt, next_prompt, weight) = sch_text().simple_schedule( keyframe_list, current_frame, easing_type)
        
        # CLIP text encoding
        tokens = clip.tokenize(str(next_prompt))
        cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        tokens = clip.tokenize(str(current_prompt))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
        print(weight)
        
        # Average conditioning
        conditioning_to_strength = weight
        conditioning_from = [[cond_from, {"pooled_output": pooled_from}]]
        conditioning_to = [[cond_to, {"pooled_output": pooled_to}]]
        out = []

        if len(conditioning_from) > 1:
            print("Warning: Conditioning from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from

            n = [tw, t_to]
            out.append(n)

        return (out,)




class sch_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "max_frames": ("INT", {"default": 99999, "min": 1, "max": 99999})  # 添加 max_frames 输入
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("selected_image",)
    FUNCTION = "select_image"
    CATEGORY = "Apt_Preset/data/schedule"

    def select_image(self, images, current_frame, max_frames):
        adjusted_frame = min(current_frame, max_frames - 1, len(images) - 1)  # 调整当前帧
        selected_image = images[adjusted_frame].unsqueeze(0)
        if current_frame > adjusted_frame:
            print(f"[Warning] Current frame {current_frame} exceeds max_frames or image count. Using frame {adjusted_frame}.")
        return (selected_image,)



class sch_mask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "current_frame": ("INT", {"default": 0, "min": 0, "max": 99999}),
                "max_frames": ("INT", {"default": 99999, "min": 1, "max": 99999})  # 添加 max_frames 输入
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("selected_mask",)
    FUNCTION = "select_mask"
    CATEGORY = "Apt_Preset/data/schedule"

    def select_mask(self, masks, current_frame, max_frames):
        adjusted_frame = min(current_frame, max_frames - 1, len(masks) - 1)  # 调整当前帧
        selected_mask = masks[adjusted_frame].unsqueeze(0)
        if current_frame > adjusted_frame:
            print(f"[Warning] Current frame {current_frame} exceeds max_frames or mask count. Using frame {adjusted_frame}.")
        return (selected_mask,)


#endregion---------------batch_cycler---------------




class BatchSlice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "LIST": ("LIST", {"forceInput": True}),
                "start": ("INT", {"default": 0, "min": -9007199254740991}),
                "end": ("INT", {"default": -1, "min": -9007199254740991}),  # 默认-1表示到末尾
            }
        }
    
    RETURN_TYPES = (ANY_TYPE, )
    RETURN_NAMES = ("Data", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, LIST: list, start: int, end: int):
        list_length = len(LIST)
        
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
            if list_length > 0 and isinstance(LIST[0], torch.Tensor):
                # 如果是张量列表，返回空的张量
                return (torch.tensor([]), )
            return ([], )
            
        # 执行切片操作
        sliced_data = LIST[start:end]
        
        # 如果列表中的元素是张量，考虑将它们堆叠成一个张量
        if len(sliced_data) > 0 and isinstance(sliced_data[0], torch.Tensor):
            try:
                # 如果是相同形状的张量，尝试堆叠它们
                return (torch.stack(sliced_data), )
            except RuntimeError:
                # 如果形状不匹配，返回原列表
                return (sliced_data, )
        
        return (sliced_data, )




class MergeBatch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT", 
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }
    
    NAME = "list_MergeBatch"
    RETURN_TYPES = ("LIST", )
    RETURN_NAMES = ("list", )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/data/list|Batch"

    def run(self, unique_id, prompt, extra_pnginfo, **kwargs):
        node_list = extra_pnginfo["workflow"]["nodes"]  # list of dict including id, type
        cur_node = next(n for n in node_list if str(n["id"]) == unique_id)
        output_list = []
        for k, v in kwargs.items():
            if k.startswith('value'):
                output_list += v
        return (output_list, )





class type_AnyIndex:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "any": (any_type, {}),
                "index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
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
            return (any[index],)
        elif isinstance(any[0], torch.Tensor):
            batch_index = min(any[0].shape[0] - 1, index)
            s = any[0][index:index + 1].clone()
            return (s,)
        else:
            return (any[0][index],)













