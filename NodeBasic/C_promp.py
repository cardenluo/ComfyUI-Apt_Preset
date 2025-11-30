import os
import re
import numpy as np
from typing import Optional, List, Tuple


from ..main_unit import *




#region----------------------------------------------------------------------#


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



class text_repair:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_string": ("STRING", {"multiline": True, "default": ""}),
                "option": (
                    [
                        "不改变", "取数字", "取字母", "转大写", "转小写", "取中文", 
                        "去标点", "去换行", "去空行", "去空格", "去格式", "统计字数",
                        "去特殊字符", "去重复行", "每行首字母大写"
                    ], 
                    {"default": "不改变"}
                ),
            },
            "optional": {},
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "process_string"
    CATEGORY = "Apt_Preset/prompt"
  
    def process_string(self, input_string, option):
        input_string = input_string or ""
        
        if option == "不改变":
            result = input_string
        elif option == "取数字":
            result = ''.join(re.findall(r'\d', input_string))
        elif option == "取字母":
            def full2half(char):
                if '\uff21' <= char <= '\uff3a':
                    return chr(ord(char) - 0xfee0)
                elif '\uff41' <= char <= '\uff5a':
                    return chr(ord(char) - 0xfee0)
                return char
            processed = ''.join([full2half(c) for c in input_string])
            result = ''.join(filter(lambda char: char.isalpha() and not self.is_chinese(char), processed))
        elif option == "转大写":
            result = input_string.upper()
        elif option == "转小写":
            result = input_string.lower()
        elif option == "取中文":
            result = ''.join(filter(self.is_chinese, input_string))
        elif option == "去标点":
            result = re.sub(r'[^\d\s\u4e00-\u9fff]', '', input_string)
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
            result = re.sub(r'[^\u4e00-\u9fff\w\s]', '', input_string)
        elif option == "去重复行":
            lines = input_string.splitlines()
            unique_lines = []
            for line in lines:
                stripped_line = line.strip()
                if stripped_line not in unique_lines:
                    unique_lines.append(stripped_line)
            result = '\n'.join(unique_lines)
        elif option == "每行首字母大写":
            lines = input_string.splitlines()
            processed_lines = [line.lstrip().capitalize() if line.strip() else line for line in lines]
            result = '\n'.join(processed_lines)

        return (result,)

    @staticmethod
    def is_chinese(char):
        return '\u4e00' <= char <= '\u9fff'



class text_filter:

    CATEGORY = "Apt_Preset/prompt"
    FUNCTION = "filter_text"
    RETURN_TYPES = ("STRING", "LIST",)
    RETURN_NAMES = ("Extracted Text", "All Matched Results",)
    INPUT_IS_LIST = False
    OUTPUT_IS_LIST = (False, True,)
    DESCRIPTION = """
  - custom_rule：自定义规则，例如，定义括号规则:
       [text] ：括号内的文本都会被提取并返回。
       [text ：括号后面的文本都会被提取并返回。
       text]：括号前面的文本都会被提取并返回。
 """ 



    RULE_OPTIONS = [
        "None",  # 新增 None 选项
        "@text@",
        "@text",
        "text @",
        '"text"',
        "'text'",
        "{text}",
        "(text)",
    ]

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "source_text": ("STRING", {"multiline": True, "default": "", "placeholder": ""}),
                "filter_rule": (cls.RULE_OPTIONS, {
                    "default": cls.RULE_OPTIONS[0],  # 默认选中 None
                    "label": ""
                }),
                "custom_rule": ("STRING", {"default": "", "placeholder": ""}),
            },
            "optional": {
                "match_all": ("BOOLEAN", {"default": False, "label_on": "", "label_off": ""}),
            }
        }

    def _get_pattern_by_rule_text(self, rule_text: str) -> Optional[str]:
        # 处理 None 选项
        if rule_text == "None":
            return None
        
        rule_core = rule_text.strip()
        if rule_core == "@text@":
            return re.escape("@") + r"(.*?)" + re.escape("@")
        elif rule_core == "@text":
            return re.escape("@") + r"(.*)"
        elif rule_core == "text @":
            return r"(.*?)" + re.escape("@")
        elif rule_core == '"text"':
            return re.escape('"') + r"(.*?)" + re.escape('"')
        elif rule_core == "'text'":
            return re.escape("'") + r"(.*?)" + re.escape("'")
        elif rule_core == "{text}":
            return re.escape("{") + r"(.*?)" + re.escape("}")
        elif rule_core == "(text)":
            return re.escape("(") + r"(.*?)" + re.escape(")")
        else:
            return None

    def filter_text(
        self,
        source_text: str,
        filter_rule: str,
        match_all: bool = False,
        custom_rule: str = ""
    ) -> Tuple[str, List[str]]:
        source_text = source_text.strip()
        pattern = None

        # 优先使用自定义规则
        if custom_rule.strip():
            target_rule = custom_rule.strip()
            if "text" in target_rule and len(target_rule) == len("text") + 2:
                prefix = target_rule.replace("text", "")[0]
                suffix = target_rule.replace("text", "")[-1]
                pattern = re.escape(prefix) + r"(.*?)" + re.escape(suffix)
            elif target_rule.endswith("text") and len(target_rule) == len("text") + 1:
                prefix = target_rule.replace("text", "")
                pattern = re.escape(prefix) + r"(.*)"
            elif target_rule.startswith("text") and len(target_rule) == len("text") + 1:
                suffix = target_rule.replace("text", "")
                pattern = r"(.*?)" + re.escape(suffix)

        # 自定义规则未配置时，使用预设规则（None 则返回空 pattern）
        if not pattern:
            pattern = self._get_pattern_by_rule_text(filter_rule)

        # 无有效规则/源文本时返回空
        if not source_text or not pattern:
            return ("", [])

        # 执行匹配并清洗结果
        match_results = re.findall(pattern, source_text, re.DOTALL)
        match_results = [res.strip() for res in match_results if res.strip()]

        # 确定最终返回值
        main_result = "\n".join(match_results) if (match_all and match_results) else (match_results[0] if match_results else "")

        return (main_result, match_results)





#endregion--------------------------------------












    