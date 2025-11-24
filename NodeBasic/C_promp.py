import os
import torch
import re
from comfy.sd1_clip import gen_empty_tokens
import random
from pathlib import Path
import folder_paths
import comfy
import os, re, io, base64, csv, shutil, requests, chardet, pathlib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


from PIL import Image as PILImage
from io import BytesIO



from ..main_unit import *


#------------------------------------------------------------
# 安全导入检查 -- 将导入语句修改为以下形式
try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ImportError:
    T5Tokenizer = None
    T5ForConditionalGeneration = None
    print("Warning: transformers not installed, SuperPrompter node will not be available")

try:
    import openpyxl
except ImportError:
    openpyxl = None
    print("Warning: openpyxl not installed, Excel-related nodes will not be available")

try:
    from openpyxl.drawing.image import Image as OpenpyxlImage
except ImportError:
    OpenpyxlImage = None
    print("Warning: openpyxl.drawing.image not available")

try:
    from openpyxl.utils import get_column_letter
except ImportError:
    get_column_letter = None
    print("Warning: openpyxl.utils.get_column_letter not available")

try:
    from openpyxl import Workbook
except ImportError:
    Workbook = None
    print("Warning: openpyxl.Workbook not available")

#------------------------------------------------------------


#region----------------------------------------------------------------------#

class text_SuperPrompter:
    def __init__(self):
        self.modelDir = os.path.expanduser("~") + "/.models"
        self.tokenizer = None
        self.model = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": "Enter prompt here"}),
                "max_new_tokens": ("INT", {"default": 77, "min": 1, "max": 2048}),
                "repetition_penalty": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 2.0, "step": 0.1}),
                "remove_incomplete_sentences": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate_text"
    CATEGORY = "Apt_Preset/prompt/😺backup"

    def remove_incomplete_sentence(self, paragraph):
        return re.sub(r'((?:\[^.!?\](?!\[.!?\]))\*+\[^.!?\\s\]\[^.!?\]\*$)', '', paragraph.rstrip())

    def download_models(self):
        model_name = "roborovski/superprompt-v1"
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16)
        os.makedirs(self.modelDir, exist_ok=True)
        self.tokenizer.save_pretrained(self.modelDir)
        self.model.save_pretrained(self.modelDir)
        print("Downloaded SuperPrompt-v1 model files to", self.modelDir)

    def load_models(self):
        if not all(os.path.exists(self.modelDir) for file in self.modelDir):
            self.download_models()
        else:
            print("Model files found. Skipping download.")

        self.tokenizer = T5Tokenizer.from_pretrained(self.modelDir)
        self.model = T5ForConditionalGeneration.from_pretrained(self.modelDir, torch_dtype=torch.float16)
        print("SuperPrompt-v1 model loaded successfully.")

    def generate_text(self, prompt, max_new_tokens, repetition_penalty, remove_incomplete_sentences):
        if self.tokenizer is None or self.model is None:
            self.load_models()

        seed = 1
        torch.manual_seed(seed)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        systemprompt = "Expand the following prompt to add more detail:"
        input_ids = self.tokenizer(systemprompt + prompt, return_tensors="pt").input_ids.to(device)
        if torch.cuda.is_available():
            self.model.to('cuda')

        outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                                      do_sample=True)

        dirty_text = self.tokenizer.decode(outputs[0])
        text = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        if remove_incomplete_sentences:
            text = self.remove_incomplete_sentence(text)
        
        return (text,)



class text_mul_replace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "target": ("STRING", {
                    "multiline": False,
                    "default": "man, dog "
                }),
                "replace": ("STRING", {
                    "multiline": False,
                    "default": "dog, man, "
                })
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "replace"
    CATEGORY = "Apt_Preset/prompt"

    def replace(self, text, target, replace):
        import re  # 注意补充re的导入（原代码可能遗漏）
        def split_with_quotes(s):
            pattern = r'"([^"]*)"|\s*([^,]+)'
            matches = re.finditer(pattern, s)
            return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]
        
        targets = split_with_quotes(target)
        exchanges = split_with_quotes(replace)
    
        word_map = {}
        for target, exchange in zip(targets, exchanges):
            target_clean = target.strip('"').strip()  # 去掉lower()，避免大小写转换影响（如原目标含大写时）
            exchange_clean = exchange.strip('"').strip()
            word_map[target_clean] = exchange_clean
    
        sorted_targets = sorted(word_map.keys(), key=len, reverse=True)
        
        result = text
        for target in sorted_targets:
            pattern = re.escape(target)
            result = re.sub(pattern, word_map[target], result)
        
        
        return (result,)

    @classmethod
    def IS_CHANGED(cls, text, target, replace):
        return (text, target, replace)



class text_mul_remove:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "words_to_remove": ("STRING", {
                    "multiline": False,
                    "default": "man, woman, world"
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "clean_prompt"
    CATEGORY = "Apt_Preset/prompt/😺backup"

    def clean_prompt(self, text, words_to_remove):
        # 拆分待移除的词（处理可能的空格和空字符串）
        remove_words = [word.strip() for word in words_to_remove.split(',') if word.strip()]
        if not remove_words:
            return (text,)
    
        remove_words_sorted = sorted(remove_words, key=lambda x: len(x), reverse=True)
        
        pattern = '|'.join(re.escape(word) for word in remove_words_sorted)
        cleaned_text = re.sub(pattern, '', text)

        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()    
        return (cleaned_text,)

    @classmethod
    def IS_CHANGED(cls, text, words_to_remove):
        return (text, words_to_remove)


class text_free_wildcards:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF
                }),
                "wildcard_symbol": ("STRING", {"default": "@@"}),
                "recursive_depth": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 10,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process_wildcards"
    CATEGORY = "Apt_Preset/prompt/😺backup"

    def process_wildcards(self, prompt, seed, wildcard_symbol, recursive_depth):
        random.seed(seed)
        wildcards_folder = Path(__file__).parent.parent  / "wildcards"
        
        logger.debug(f"Wildcards folder: {wildcards_folder}")
        logger.debug(f"Current working directory: {os.getcwd()}")
        logger.debug(f"Directory contents of wildcards folder: {os.listdir(wildcards_folder)}")
        
        def replace_wildcard(match, depth=0):
            if depth >= recursive_depth:
                logger.debug(f"Max depth reached: {depth}")
                return match.group(0)
            
            wildcard = match.group(1)
            file_path = os.path.join(wildcards_folder, f"{wildcard}.txt")
            logger.debug(f"Looking for file: {file_path} (depth: {depth})")
            logger.debug(f"File exists: {os.path.exists(file_path)}")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        choice = random.choice(lines)
                        logger.debug(f"Replaced {wildcard} with: {choice} (depth: {depth})")
                        
                        if wildcard_symbol in choice:
                            logger.debug(f"Found nested wildcard in: {choice}")
                            processed_choice = re.sub(pattern, lambda m: replace_wildcard(m, depth + 1), choice)
                            logger.debug(f"After recursive processing: {processed_choice} (depth: {depth})")
                            return processed_choice
                        else:
                            return choice
                    else:
                        logger.warning(f"File {file_path} is empty")
                        return match.group(0)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {str(e)}")
                    return match.group(0)
            else:
                logger.warning(f"File not found: {file_path}")
                return match.group(0)

        escaped_symbol = re.escape(wildcard_symbol)
        pattern = f"{escaped_symbol}([a-zA-Z0-9_]+)"
        

        
        processed_prompt = prompt
        for i in range(recursive_depth):
            new_prompt = re.sub(pattern, lambda m: replace_wildcard(m, 0), processed_prompt)
            if new_prompt == processed_prompt:
                break
            processed_prompt = new_prompt
            logger.debug(f"Iteration {i+1} result: {processed_prompt}")
        
        logger.debug(f"Final processed prompt: {processed_prompt}")
        
        return (processed_prompt,)

    @classmethod
    def IS_CHANGED(cls, prompt, seed, wildcard_symbol, recursive_depth):
        return float(seed)




#region---------------------------# Wildcards-------------


wildcards_dir1 = Path(__file__).parent.parent  / "wildcards"
os.makedirs(wildcards_dir1, exist_ok=True)
wildcards_dir2 = Path(folder_paths.base_path) / "wildcards"


full_dirs = [wildcards_dir1, wildcards_dir2]

WILDCARDS_LIST = (
    ["None"]
    + [
        "dir1 | " + str(wildcard.relative_to(wildcards_dir1))[:-4]
        for wildcard in wildcards_dir1.rglob("*.txt")
    ]
    + [
        "base_path | " + str(wildcard.relative_to(wildcards_dir2))[:-4]
        for wildcard in wildcards_dir2.rglob("*.txt")
    ]
)


class text_stack_wildcards:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "wildcards_count": (
                    "INT",
                    {"default": 1, "min": 1, "max": 50, "step": 1},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                    },
                ),
            },
        }

        for i in range(1, 10):
            inputs["required"][f"wildcard_name_{i}"] = (
                WILDCARDS_LIST,
                {"default": WILDCARDS_LIST[0]},
            )

        return inputs

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "stack_Wildcards"
    CATEGORY = "Apt_Preset/prompt/😺backup"

    def stack_Wildcards(self, wildcards_count, seed, text=None, **kwargs):

        selected_wildcards = [
            kwargs[f"wildcard_name_{i}"] for i in range(1, wildcards_count + 1)
        ]
        results = []

        for full_dir in full_dirs:
            for root, dirs, files in os.walk(full_dir):
                for wildcard in selected_wildcards:
                    if wildcard == "None":
                        continue
                    else:
                        if wildcard.startswith("dir1 | "):
                            wildcard_filename = wildcard[len("dir1 | ") :]
                            target_dir = wildcards_dir1
                        if wildcard.startswith("base_path | "):
                            wildcard_filename = wildcard[len("base_path | ") :]
                            target_dir = wildcards_dir2
                        if target_dir:
                            wildcard_file = (
                                Path(target_dir) / f"{wildcard_filename}.txt"
                            )
                            if wildcard_file.is_file():
                                with open(wildcard_file, "r", encoding="utf-8") as f:
                                    lines = f.readlines()
                                    if lines:
                                        selected_line_index = seed - 1
                                        selected_line_index %= len(lines)
                                        selected_line = lines[
                                            selected_line_index
                                        ].strip()
                                        results.append(selected_line)
                            else:
                                print(f"Wildcard File not found: {wildcard_file}")

                joined_result = ", ".join(results)
                if text == "":
                    joined_result = f"{joined_result}"
                else:
                    joined_result = f"{text},{joined_result}"
                return (joined_result,)


#endregion---------------------------# Wildcards-------------


class text_mul_Join:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                f"text{i+1}": ("STRING", {"default": "", "multiline": False}) for i in range(8)
            },
            "optional": {
                "delimiter": ("STRING", {
                    "default": "\\n",
                    "multiline": False,
                    "tooltip": "Use \\n for newline, \\t for tab, \\s for space"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("joined_text",)
    FUNCTION = "join_text"
    CATEGORY = "Apt_Preset/prompt"

    def join_text(self, delimiter, **kwargs):
        # 处理特殊转义字符
        if delimiter == "\\n":
            actual_delimiter = "\n"
        elif delimiter == "\\t":
            actual_delimiter = "\t"
        elif delimiter == "\\s":
            actual_delimiter = " "
        else:
            actual_delimiter = delimiter.strip()

        # 获取所有输入
        inputs = [kwargs[f"text{i+1}"] for i in range(8)]

        # 去除每个输入的首尾空白
        stripped_inputs = [text.strip() for text in inputs]

        # 找到最后一个非空索引
        last_non_empty_index = -1
        for i, text in enumerate(stripped_inputs):
            if text:
                last_non_empty_index = i

        # 构建结果
        result = []
        for i, text in enumerate(stripped_inputs):
            if i <= last_non_empty_index:
                result.append(text if text else "")  # 空字段也保留占位

        # 拼接并返回
        joined = actual_delimiter.join(result)
        return (joined,)


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
    CATEGORY = "Apt_Preset/prompt/😺backup"

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
        return {"required": {
                    "text_list": (any_type,),
                    "delimiter":(["newline","comma","backslash","space"],),
                            },
                }
    RETURN_TYPES = ("STRING",) 
    RETURN_NAMES = ("text",) 
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/prompt/😺backup"

    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,)

    def run(self,text_list,delimiter):
        delimiter=delimiter[0]
        if delimiter =='newline':
            delimiter='\n'
        elif delimiter=='comma':
            delimiter=','
        elif delimiter=='backslash':
            delimiter='\\'
        elif delimiter=='space':
            delimiter=' '
        t=''
        if isinstance(text_list, list):
            t=delimiter.join(text_list)
        return (t,)










#endregion--------------------------------------


# 常量定义：所有字典键值均使用中文，同步优化提示词
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

class text_Qwen_camera:
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










