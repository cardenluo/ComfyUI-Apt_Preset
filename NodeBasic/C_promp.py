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
    CATEGORY = "Apt_Preset/prompt/😺backup"

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
    CATEGORY = "Apt_Preset/prompt/😺backup"

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












