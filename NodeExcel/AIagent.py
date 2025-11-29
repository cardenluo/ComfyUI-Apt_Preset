



#region-----------GLM4.5V-------------------------------------

import os
import json
import base64
import random
from PIL import Image
import numpy as np
import io
import requests

try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except ImportError:
    ZhipuAI = None
    ZHIPUAI_AVAILABLE = False

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")


def load_text_prompts():
    if not os.path.exists(json_path):
        # JSON 文件不存在时返回默认字典，避免报错
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 返回 TEXT_PROMPTS，若不存在则返回默认字典
        return data.get("TEXT_PROMPTS", {"None": ""})
    except Exception:
        # JSON 格式错误时返回默认字典
        return {"None": ""}

TEXT_PROMPTS = load_text_prompts()

def _log_info(message):
    print(f"[GLM_Nodes] 信息：{message}")

def _log_warning(message):
    print(f"[GLM_Nodes] 警告：{message}")

def _log_error(message):
    print(f"[GLM_Nodes] 错误：{message}")

def get_zhipuai_api_key():
    env_api_key = os.getenv("ZHIPUAI_API_KEY")
    if env_api_key:
        _log_info("使用环境变量 API Key。")
        return env_api_key
    _log_warning("未设置环境变量 ZHIPUAI_API_KEY。")
    return ""

# 原有模型列表保留
ZHIPU_MODELS = [
    "GLM-4.5-Flash",
    "glm-4v-flash",
    "XX----下面的要开通支付-----XX",
    "glm-4.5-air",
    "glm-4.5",
    "glm-4.5-x",
    "glm-4.5-airx",
    "glm-4.5-flash",
    "glm-4-plus",
    "glm-z1-air",
    "glm-4v-plus-0111",
    "glm-4.1v-thinking-flash"
]

class AI_GLM4:
    @classmethod
    def INPUT_TYPES(cls):
        # 从 JSON 加载的 TEXT_PROMPTS 中获取键，作为预设选项
        prompt_keys = list(TEXT_PROMPTS.keys())
        default_selection = prompt_keys[0] if prompt_keys else ""

        return {
            "required": {
                "text_input": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "此处是语言模型的文本输入，图片分析用系统提示词输入"
                }),
                "prompt_preset": (prompt_keys, {"default": default_selection}),
                "prompt_override": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": " 输入系统提示词，留空则用预设"
                }),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "可选：智谱AI API Key (留空则尝试从环境变量或config.json读取)"
                }),
                "model_name": (ZHIPU_MODELS, {
                    "default": "glm-4v-flash",
                    "placeholder": "请输入模型名称，如 glm-4v-flash "
                }),
            },
            "optional": {
                "image_input": ("IMAGE", {"optional": True}),  # 移除 tooltip 备注
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/prompt"

    def execute(self, prompt_preset, prompt_override, api_key, model_name,
                max_tokens, seed, text_input, image_input=None):
        
        image_input_provided = image_input is not None
        
        if image_input_provided:
            return self._process_image(api_key, prompt_preset, prompt_override, model_name, seed,
                                     image_input, max_tokens)
        else:
            return self._process_text(text_input, api_key, prompt_preset, prompt_override, model_name,
                                    max_tokens, seed)

    def _process_text(self, text_input, api_key, prompt_preset, prompt_override, model_name,
                      max_tokens, seed):
        final_api_key = api_key.strip() or get_zhipuai_api_key()
        if not final_api_key:
            _log_error("API Key 未提供。")
            return ("API Key 未提供。",)

        _log_info("初始化智谱AI客户端。")

        try:
            client = ZhipuAI(api_key=final_api_key)
        except Exception as e:
            _log_error(f"客户端初始化失败: {e}")
            return (f"客户端初始化失败: {e}",)

        # 从 JSON 加载的 TEXT_PROMPTS 中获取提示词（核心修改）
        final_system_prompt = ""
        if prompt_override and prompt_override.strip():
            final_system_prompt = prompt_override.strip()
            _log_info("使用 'prompt_override'。")
        elif prompt_preset in TEXT_PROMPTS:
            final_system_prompt = TEXT_PROMPTS[prompt_preset]
            _log_info(f"使用预设提示词: '{prompt_preset}'。")
        else:
            # 若预设不存在，使用字典第一个值（兼容原有逻辑）
            final_system_prompt = next(iter(TEXT_PROMPTS.values()), "") if TEXT_PROMPTS else ""
            _log_warning("预设提示词未找到，使用默认提示词。")

        if not final_system_prompt:
            _log_error("系统提示词不能为空。")
            return ("系统提示词不能为空。",)

        if not isinstance(final_system_prompt, str):
            final_system_prompt = str(final_system_prompt)

        messages = [
            {"role": "system", "content": final_system_prompt},
            {"role": "user", "content": text_input}
        ]

        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        _log_info(f"内部种子: {effective_seed}。")
        random.seed(effective_seed)

        _log_info(f"调用 GLM-4 ({model_name})...")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.9,
                top_p=0.7,
                max_tokens=max_tokens,
            )
            response_text = response.choices[0].message.content
            _log_info("GLM-4 响应成功。")
            return (response_text,)
        except Exception as e:
            error_message = f"GLM-4 API 调用失败: {e}"
            return (error_message,)

    def _process_image(self, api_key, prompt_preset, prompt_override, model_name, seed,
                    image_input=None, max_tokens=1024):
        final_api_key = api_key.strip() or get_zhipuai_api_key()
        if not final_api_key:
            _log_error("API Key 未提供。")
            return ("API Key 未提供。",)
        _log_info("初始化智谱AI客户端。")

        try:
            client = ZhipuAI(api_key=final_api_key)
        except Exception as e:
            _log_error(f"客户端初始化失败: {e}")
            return (f"客户端初始化失败: {e}",)

        if image_input is None:
            _log_error("必须提供有效的IMAGE对象。")
            return ("必须提供有效的IMAGE对象。",)

        try:
            i = 255. * image_input.cpu().numpy()
            img_array = np.clip(i, 0, 255).astype(np.uint8)[0]
            img = Image.fromarray(img_array)
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            final_image_data = f"data:image/png;base64,{image_base64}"
            _log_info("IMAGE对象成功转换为Base64格式")
        except Exception as e:
            _log_error(f"图片格式转换失败: {str(e)}")
            return (f"图片格式转换失败: {str(e)}",)

        # 从 JSON 加载的 TEXT_PROMPTS 中获取提示词（核心修改）
        final_prompt_text = ""
        if prompt_override and prompt_override.strip():
            final_prompt_text = prompt_override.strip()
            _log_info("使用自定义提示词")
        elif prompt_preset in TEXT_PROMPTS:
            final_prompt_text = TEXT_PROMPTS[prompt_preset]
            _log_info(f"使用预设提示词: {prompt_preset}")
        else:
            final_prompt_text = next(iter(TEXT_PROMPTS.values()), "") if TEXT_PROMPTS else ""
            _log_warning("使用默认提示词")

        if not final_prompt_text:
            _log_error("提示词不能为空")
            return ("提示词不能为空",)

        content_parts = [
            {"type": "text", "text": final_prompt_text},
            {"type": "image_url", "image_url": {"url": final_image_data}}
        ]

        effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
        _log_info(f"使用内部种子: {effective_seed}")
        random.seed(effective_seed)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content_parts}],
                temperature=0.9,
                top_p=0.7,
                max_tokens=max_tokens,
            )
            response_content = str(response.choices[0].message.content)
            _log_info("GLM-4V图片识别成功")
            return (response_content,)
        except Exception as e:
            error_message = f"GLM-4V API调用失败: {str(e)}"
            _log_error(error_message)
            return (error_message,)



#endregion--------------------------------------------------------------------------



#region-----------qwen-image edit---

import requests
from io import BytesIO
import os
import io
import base64
import json
import folder_paths
import numpy as np
from PIL import Image


try:
    import dashscope
    REMOVER_AVAILABLE = True  
except ImportError:
    dashscope = None
    REMOVER_AVAILABLE = False  


current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")

def load_Image_Analysis():
    if not os.path.exists(json_path):
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("IMAGE_PROMPTS", {"None": ""})
    except Exception:
        return {"None": ""}

IMAGE_PROMPTS = load_Image_Analysis()

custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")
comfy_root = os.path.dirname(custom_nodes_paths[0]) if custom_nodes_paths else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))

key_path = os.path.join(comfy_root, "custom_nodes", "ComfyUI-Apt_Preset", "NodeExcel", "ApiKey_AI_Qwen.txt")

def get_aliyun_api_key():
    env_api_key = os.getenv("aliyun_API_KEY")
    if env_api_key and env_api_key.strip():
        return env_api_key.strip()
    if os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
        except:
            pass
    return ""

def encode_image(pil_image, save_tokens=True):
    buffered = io.BytesIO()
    if save_tokens:
        image = resize_to_limit(pil_image)
        image.save(buffered, format="JPEG", optimize=True, quality=75)
    else:
        pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def resize_to_limit(img, max_pixels=262144):
    width, height = img.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return img
    scale = (max_pixels / total_pixels) ** 0.5
    return img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        return [tensor2pil(image[i])[0] for i in range(batch_count)]
    return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]

def analyze_images(images, model, api_key, final_prompt, max_tokens, seed=None, max_retries=3):
    if not api_key:
        raise ValueError("API密钥未配置，请通过以下方式之一设置：\n1. 设置系统环境变量 aliyun_API_KEY\n2. 在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥")
    
    img_count = len(images)
    user_text = f"请按要求分析以下{img_count}张图片（按输入顺序）：{final_prompt}"
    if seed is not None and seed != 0:
        user_text += f"\n【固定种子：{seed}】"
    
    user_content = []
    for img in images:
        user_content.append({"image": f"data:image/png;base64,{encode_image(img)}"})
    user_content.append({"text": user_text})
    
    messages = [{"role": "user", "content": user_content}]
    
    for retry in range(max_retries):
        try:
            call_params = {
                "api_key": api_key, 
                "model": model, 
                "messages": messages, 
                "result_format": "message",
                "max_tokens": max_tokens
            }
            if seed is not None and seed != 0:
                call_params["seed"] = seed
            
            response = dashscope.MultiModalConversation.call(**call_params)
            if response.status_code == 200:
                return response.output.choices[0].message.content[0]["text"].strip()
            else:
                raise Exception(f"API请求失败: {response.message}")
        except Exception as e:
            if "seed" in str(e).lower() and retry < max_retries - 1:
                call_params.pop("seed", None)
                response = dashscope.MultiModalConversation.call(**call_params)
                if response.status_code == 200:
                    return response.output.choices[0].message.content[0]["text"].strip()
            if retry == max_retries - 1:
                raise Exception(f"多次尝试后失败: {str(e)}")
            

class AI_Qwen:
    def __init__(self):
        self.api_key = get_aliyun_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["qwen-vl-max", "qwen-vl-max-latest"], {"default": "qwen-vl-max-latest"}),
                "preset": (list(IMAGE_PROMPTS.keys()), {"default": "None"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {
                    "default": 1024,
                    "min": 10,
                    "max": 8192,
                    "step": 10,
                }),
            },
            "optional": {

                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "api_key_input": ("STRING", {"default": "", "multiline": False, }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "system_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Apt_Preset/prompt"

    def analyze(self, model, preset, text, max_tokens, api_key_input="", image_1=None, image_2=None, image_3=None, seed=0):
        # 获取preset对应的文本内容（键名映射到值）
        preset_text = IMAGE_PROMPTS.get(preset, "") if preset != "None" else ""
        text_content = text.strip()
        
        # 确定最终提示词
        if text_content:
            final_prompt = text_content
        else:
            final_prompt = preset_text if preset_text else "生成输入图片的详细中文描述"

        input_images = []
        if image_1 is not None:
            pil_img = tensor2pil(image_1)[0]
            if pil_img:
                input_images.append(pil_img)
        if image_2 is not None:
            pil_img = tensor2pil(image_2)[0]
            if pil_img:
                input_images.append(pil_img)
        if image_3 is not None:
            pil_img = tensor2pil(image_3)[0]
            if pil_img:
                input_images.append(pil_img)
        
        if not input_images:
            return ("错误：请至少输入1张有效图片", preset_text)
        
        # 使用传入的API key，如果没有则使用默认方式获取
        api_key = api_key_input.strip() if api_key_input.strip() else self.api_key
        if not api_key:
            return ("API密钥未配置，请通过以下方式之一设置：\n1. 在节点输入中直接填写API密钥\n2. 设置系统环境变量 aliyun_API_KEY\n3. 或在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥", preset_text)
        
        try:
            result = analyze_images(
                input_images, 
                model, 
                api_key, 
                final_prompt, 
                max_tokens,
                seed if seed != 0 else None
            )
            return (result, preset_text)
        except Exception as e:
            return (f"分析失败: {str(e)}", preset_text)

#endregion--------------------------------------------------------------------------



#region-----------qwen-prompt-----------------

try:
    import dashscope
    from dashscope import Conversation
    REMOVER_AVAILABLE = True
except ImportError:
    dashscope = None
    Conversation = None
    REMOVER_AVAILABLE = False

def analyze_text(model, api_key, final_prompt, max_tokens, seed=None, max_retries=3):
    if not api_key:
        raise ValueError("API密钥未配置，请通过以下方式之一设置：\n1. 设置系统环境变量 aliyun_API_KEY\n2. 在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥")
    
    if not Conversation:
        raise ImportError("dashscope.Conversation 导入失败，请确保dashscope库版本正确")
    
    user_text = final_prompt
    if seed is not None and seed != 0:
        user_text += f"\n【固定种子：{seed}】"
    
    messages = [{"role": "user", "content": user_text}]
    
    for retry in range(max_retries):
        try:
            conv = Conversation()
            call_params = {
                "api_key": api_key, 
                "model": model, 
                "messages": messages, 
                "result_format": "message",
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.8
            }
            if seed is not None and seed != 0:
                call_params["seed"] = seed
            
            response = conv.call(**call_params)
            if response.status_code == 200:
                return response.output.choices[0].message.content.strip()
            else:
                raise Exception(f"API请求失败: {response.message} (状态码：{response.status_code})")
        except Exception as e:
            if "seed" in str(e).lower() and retry < max_retries - 1:
                call_params.pop("seed", None)
                conv_retry = Conversation()
                response = conv_retry.call(**call_params)
                if response.status_code == 200:
                    return response.output.choices[0].message.content.strip()
            if retry == max_retries - 1:
                raise Exception(f"多次尝试后失败: {str(e)}")

# 修复系统提示词加载逻辑
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")

def load_text_prompts():
    if not os.path.exists(json_path):
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("TEXT_PROMPTS", {"None": ""})
    except Exception:
        return {"None": ""}

TEXT_PROMPTS = load_text_prompts()



class AI_Qwen_text:
    def __init__(self):
        self.api_key = get_aliyun_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (["qwen3-coder-plus", "qwen3-coder-plus-2025-09-23", "qwen3-coder-flash", 
                              "qwen3-coder-480b-a35b-instruct", "qwen3-coder-30b-a3b-instruct"], {
                    "default": "qwen3-coder-plus", }),

                "preset": (list(TEXT_PROMPTS.keys()), {"default": "None"}),

            },
            "optional": {
                "custom_system_prompt": ("STRING", {"default": "", "multiline": False, "description": "自定义系统提示词，填写后将替代预设的preset"}),
                "text": ("STRING", {"default": "", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 10, "max": 8192,"step": 10, }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "api_key_input": ("STRING", {"default": "", "multiline": False, }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "system_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Apt_Preset/prompt"

    def analyze(self, llm_model, preset, text, max_tokens, custom_system_prompt="", api_key_input="", seed=0):
        # 优先使用自定义系统提示词，无内容时使用预设
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            system_prompt = TEXT_PROMPTS.get(preset, "") if preset != "None" else ""
        
        text_content = text.strip()
        
        # 构建最终提示词
        if text_content and system_prompt:
            final_prompt = f"{system_prompt}\n\n{text_content}"
        elif text_content:
            final_prompt = text_content
        else:
            final_prompt = system_prompt if system_prompt else "请根据需求完成相关代码或文本任务（如代码生成、代码解释、编程问题解答等）"

        # 校验最终提示词
        if not final_prompt.strip():
            return ("错误：请输入有效文本或选择合适的预设/填写自定义系统提示词", system_prompt)
        
        # 检查依赖
        if not REMOVER_AVAILABLE or not Conversation:
            return ("错误：dashscope库导入失败，请安装最新版本dashscope（pip install -U dashscope）", system_prompt)
        
        # 处理API密钥
        api_key = api_key_input.strip() if api_key_input.strip() else self.api_key
        if not api_key:
            return ("API密钥未配置，请通过以下方式之一设置：\n1. 在节点输入中直接填写API密钥\n2. 设置系统环境变量 aliyun_API_KEY\n3. 或在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥", system_prompt)
        
        # 调用分析函数
        try:
            result = analyze_text(
                llm_model, 
                api_key, 
                final_prompt, 
                max_tokens,
                seed if seed != 0 else None
            )
            return (result, system_prompt)
        except Exception as e:
            return (f"处理失败: {str(e)}", system_prompt)








#endregion--------------------------------------------------------------------------









#region-----------------保存提示词------------

import os
import json
import folder_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")

# 确保JSON文件存在，不存在则创建默认文件
def init_json_file():
    if not os.path.exists(json_path):
        default_data = {
            "TEXT_PROMPTS": {"None": ""},
            "IMAGE_PROMPTS": {"None": ""}
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(default_data, f, ensure_ascii=False, indent=4)

init_json_file()

def load_prompt_dicts():
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_prompt_dicts(data):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




class AI_PresetSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target_dict": (["TEXT_PROMPTS", "IMAGE_PROMPTS"], {"default": "TEXT_PROMPTS"}),
                "prompt_title": ("STRING", {"default": "", "placeholder": "输入提示词标题（作为字典的键）"}),
                "prompt_content": ("STRING", {"default": "", "multiline": True, "placeholder": "输入提示词内容（作为字典的值）"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("result",)
    FUNCTION = "append_prompt"
    CATEGORY = "Apt_Preset/prompt"
 
    DESCRIPTION = """
    TEXT_PROMPTS: 文本提示词，对应AiPromptPreset.json中的TEXT_PROMPTS字典
    IMAGE_PROMPTS: 图像提示词，对应AiPromptPreset.json中的IMAGE_PROMPTS字典
   """ 


    def append_prompt(self, target_dict, prompt_title, prompt_content):
        prompt_title = prompt_title.strip()
        prompt_content = prompt_content.strip()
        if not prompt_title:
            return ("错误：提示词标题不能为空",)
        if not prompt_content:
            return ("错误：提示词内容不能为空",)
        try:
            data = load_prompt_dicts()
        except Exception as e:
            return (f"错误：加载JSON失败 - {str(e)}",)
        if target_dict not in data:
            return (f"错误：JSON中不存在{target_dict}字典",)
        if prompt_title in data[target_dict]:
            return (f"错误：{target_dict}中已存在标题「{prompt_title}」",)
        data[target_dict][prompt_title] = prompt_content
        try:
            save_prompt_dicts(data)
        except Exception as e:
            return (f"错误：保存JSON失败 - {str(e)}",)
        return (f"成功：向{target_dict}永久追加提示词\n标题：{prompt_title}\n内容：{prompt_content[:50]}...",)
    

#endregion--------------------------------------------------------------------------



#region-----------------ollama模型管理------------

import time
import base64
import json
import os
import subprocess
import threading
from io import BytesIO
from typing import Optional
import requests
import torch
import numpy as np
from PIL import Image
import comfy.sd
import comfy.utils

# 精准定位ComfyUI根目录，拼接模型路径
try:
    comfy_module_path = os.path.dirname(os.path.abspath(comfy.__file__))
    COMFYUI_ROOT = os.path.abspath(os.path.join(comfy_module_path, ".."))
except:
    current_file_path = os.path.abspath(__file__)
    COMFYUI_ROOT = current_file_path
    for _ in range(5):
        COMFYUI_ROOT = os.path.dirname(COMFYUI_ROOT)
        if os.path.exists(os.path.join(COMFYUI_ROOT, "comfy")):
            break
OLLAMA_MODEL_PATH = os.path.join(COMFYUI_ROOT, "models", "ollama")
os.makedirs(OLLAMA_MODEL_PATH, exist_ok=True)
os.environ["OLLAMA_MODELS"] = OLLAMA_MODEL_PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")

def load_Image_Analysis():
    if not os.path.exists(json_path):
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        preset_data = data.get("IMAGE_PROMPTS", {})
        if "None" not in preset_data:
            preset_data["None"] = ""
        return preset_data
    except Exception:
        return {"None": ""}

def load_TEXT_PROMPTS():
    if not os.path.exists(json_path):
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        preset_data = data.get("TEXT_PROMPTS", {})
        if "None" not in preset_data:
            preset_data["None"] = ""
        return preset_data
    except Exception:
        return {"None": ""}

def resize_to_limit(img, max_pixels=262144):
    width, height = img.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return img
    scale = (max_pixels / total_pixels) ** 0.5
    new_width = int(width * scale)
    new_height = int(height * scale)
    return img.resize((new_width, new_height), Image.LANCZOS)


IMAGE_PROMPTS = load_Image_Analysis()
TEXT_PROMPTS = load_TEXT_PROMPTS()
OLLMAMA_MODEL_NAME_IMAGE = ["qwen3-vl:latest",]
OLLMAMA_MODEL_NAME_TEXT = []



class AI_Ollama_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (OLLMAMA_MODEL_NAME_IMAGE, {"default": "qwen3-vl:latest"}),
                "preset": (list(IMAGE_PROMPTS.keys()), {"default": "None"}),
                "analysis_prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "enable_ocr": ("BOOLEAN", {"default": False}),
            },
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("analysis_result", "system_prompt")
    FUNCTION = "run_image_analysis"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = "CMD运行命令:  ollama run qwen3-vl:latest"
    def run_image_analysis(
        self,
        model_name,
        preset,
        analysis_prompt,
        temperature,
        max_tokens,
        seed=0,
        image_1=None,
        image_2=None,
        image_3=None,
        enable_ocr=False
    ):
        cleaned_analysis_prompt = analysis_prompt.strip()
        if cleaned_analysis_prompt:
            final_prompt = cleaned_analysis_prompt
        else:
            preset_prompt = IMAGE_PROMPTS.get(preset, "").strip()
            final_prompt = preset_prompt if preset_prompt else "请基于输入图片，详细描述内容、分析物体/场景/细节，若有多张图需对比异同"
        img_base64_list = []
        img_inputs = [image_1, image_2, image_3]
        for idx, img_tensor in enumerate(img_inputs, 1):
            if img_tensor is not None:
                try:
                    if len(img_tensor.shape) == 4:
                        img_tensor = img_tensor.squeeze(0)
                    if img_tensor.dtype == torch.float32:
                        img_tensor = (img_tensor * 255).byte()
                    img_np = img_tensor.cpu().numpy().astype(np.uint8)
                    if img_np.shape[-1] == 4:
                        img_np = img_np[..., :3]
                    img_pil = Image.fromarray(img_np)
                    img_pil = resize_to_limit(img_pil)
                    img_buffer = BytesIO()
                    img_pil.save(img_buffer, format="JPEG", quality=95, optimize=True)
                    img_buffer.seek(0)
                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8").strip()
                    img_base64_list.append(img_base64)
                except Exception as e:
                    return (f"图片{idx}处理/编码错误：{str(e)}", final_prompt)
        if not img_base64_list:
            return ("错误：未输入任何图片，请至少选择1张图片上传", final_prompt)
        if enable_ocr:
            img_count = len(img_base64_list)
            ocr_msg = f"\n\n特别指令：请提取{img_count}张图片中所有可见文本内容，按图片序号（1-{img_count}）分类，以列表形式整理文本内容及文字位置信息"
            final_prompt += ocr_msg
        ollama_host = "http://localhost:11434"
        try:
            data = {
                "model": model_name,
                "prompt": final_prompt,
                "images": img_base64_list,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
                "options": {
                    "num_ctx": max_tokens,
                    "num_thread": 4
                }
            }
            if seed != 0:
                data["seed"] = seed
            url = f"{ollama_host}/api/generate"
            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(data),
                stream=True,
                timeout=300
            )
            response.raise_for_status()
            analysis_result = ""
            for line in response.iter_lines():
                if line:
                    try:
                        line_data = json.loads(line.decode("utf-8"))
                        if "response" in line_data:
                            analysis_result += line_data["response"]
                        if line_data.get("error"):
                            return (f"模型错误：{line_data['error']}", final_prompt)
                        if line_data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
            analysis_result = analysis_result.strip()
            if not analysis_result:
                return ("错误：模型未返回有效结果，请检查模型是否支持视觉分析", final_prompt)
            return (analysis_result, final_prompt)
        except requests.exceptions.ConnectionError:
            error_msg = "连接错误：无法连接到Ollama服务，请检查Ollama是否已启动、服务地址正确、端口11434未被占用"
            return (error_msg, final_prompt)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                error_msg = f"模型未找到：请先执行 `ollama pull {model_name}` 下载模型到{OLLAMA_MODEL_PATH}（确保模型支持视觉分析，如qwen3-vl、llava等）"
            elif e.response.status_code == 500:
                error_msg = f"Ollama服务内部错误：1. 请更新Ollama到最新版本（≥0.12.7）；2. 重新拉取模型 `ollama pull {model_name}` 到{OLLAMA_MODEL_PATH}；3. 检查图片是否损坏"
            else:
                error_msg = f"HTTP错误：{str(e)}"
            return (error_msg, final_prompt)
        except requests.exceptions.Timeout:
            return ("错误：请求超时，请检查网络连接或增大超时时间", final_prompt)
        except Exception as e:
            error_msg = f"未知错误：{str(e)}"
            return (error_msg, final_prompt)



class AI_Ollama_text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # 移除 model_name 列表中的 llama3:8b
                "model_name": (["qwen2.5-coder:7b","qwen3-coder:30b","qwen3-vl:latest",], {"default": "qwen3-vl:latest"}),
                "preset": (list(TEXT_PROMPTS.keys()), {"default": "None"}),

            },
            "optional": {
                "custom_system_prompt": ("STRING", {"default": "", "multiline": False, "description": "自定义系统提示词，填写后将替代预设的preset"}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("analysis_result", "system_prompt")

    FUNCTION = "run"
    CATEGORY = "Apt_Preset/prompt"
    
    def run(self, model_name, preset, prompt, temperature, max_tokens, seed=0, custom_system_prompt=""):
        # 优先使用自定义系统提示词，无内容时使用预设
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            system_prompt = TEXT_PROMPTS.get(preset, "") if preset != "None" else ""
        
        ollama_host = "http://localhost:11434"
        
        try:
            url = f"{ollama_host}/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": model_name,
                "prompt": prompt,
                "system": system_prompt,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True
            }
            
            if seed != 0:
                data["seed"] = seed
            
            response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
            response.raise_for_status()
            
            result = ""
            for line in response.iter_lines():
                if line:
                    line_data = json.loads(line.decode('utf-8'))
                    if 'response' in line_data:
                        result += line_data['response']
                    if line_data.get('done', False):
                        break
            
            return (result,system_prompt)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return (f"模型未找到：请先执行 `ollama pull {model_name}` 下载模型到{OLLAMA_MODEL_PATH}",)
            else:
                return (f"HTTP错误：{str(e)}",)
        except Exception as e:
            return (f"错误: {str(e)}",)







class Ai_Ollama_RunModel:
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.is_running = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},  # 无任何必填参数，直接启动
            "optional": {
                "timeout": ("INT", {
                    "default": 20,
                    "min": 5,
                    "max": 30,
                    "step": 1,
                })
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run_ollama_service"
    CATEGORY = "Apt_Preset/prompt"


    def run_ollama_service(self, timeout: int = 10):
        # 步骤1：检查Ollama是否安装
        try:
            subprocess.run(["ollama", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except FileNotFoundError:
            return ("错误：未找到ollama命令！\n请确保：1. 已安装Ollama（https://ollama.com/download）\n       2. Ollama已添加到系统环境变量",)

        # 步骤2：停止已存在的Ollama服务（避免端口占用）
        self._stop_existing_ollama()

        # 步骤3：轻量启动Ollama API服务
        return self._lightweight_start_service(timeout)

    def _stop_existing_ollama(self):
        """停止已运行的Ollama进程"""
        try:
            if os.name == "nt":  # Windows
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            else:  # Linux/macOS
                subprocess.run(["pkill", "ollama"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            time.sleep(1)
        except Exception:
            pass  # 忽略停止失败（可能原本就没有运行的服务）

    def _lightweight_start_service(self, timeout: int) -> tuple:
        """纯轻量启动Ollama API服务（仅启动服务，不加载任何模型）"""
        cmd = ["ollama", "serve"]
        try:
            # 隐藏启动窗口（Windows），后台运行（Linux/macOS）
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags
            )

            # 等待服务启动并验证连通性
            start_time = time.time()
            service_alive = False
            while time.time() - start_time < timeout:
                if self.process.poll() is not None:
                    # 服务异常退出
                    stderr_data = self.process.stdout.read() if self.process.stdout else ""
                    return (f"启动失败：Ollama服务异常退出\n错误信息：{stderr_data}",)
                # 检查API是否可访问
                if self._check_service_connected():
                    service_alive = True
                    break
                time.sleep(0.5)

            if not service_alive:
                self.process.kill()
                return (f"启动失败：Ollama服务启动超时（{timeout}秒），API未连通",)

            self.is_running = True
            # 启动日志监听线程（仅打印关键日志）
            threading.Thread(target=self._print_service_log, args=(self.process,), daemon=True).start()

            # 启动成功信息
            success_msg = (
                f"✅ Ollama服务轻量启动成功！\n"
                f"📁 模型存储路径：{OLLAMA_MODEL_PATH}\n"
                f"🆔 服务PID：{self.process.pid}\n"
                f"🌐 API地址：http://localhost:11434\n"
                "💡 提示：\n"
                "  1. 服务已在后台运行，可通过API调用任意已下载的模型\n"
                "  2. 停止服务需手动结束PID（Windows任务管理器/Linux/macOS pkill ollama）\n"
                "  3. 模型需提前下载到上述路径（命令：ollama pull 模型名）"
            )
            return (success_msg,)

        except Exception as e:
            return (f"启动失败：{str(e)}",)

    def _check_service_connected(self) -> bool:
        """检查Ollama API服务是否连通"""
        try:
            # 调用Ollama的tags接口（无模型依赖，仅验证服务存活）
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def _print_service_log(self, process: subprocess.Popen):
        """轻量日志输出（仅打印错误和关键信息）"""
        while self.is_running and process.poll() is None:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                # 仅打印包含错误或关键状态的日志
                if "error" in line.lower() or "listening" in line.lower() or "started" in line.lower():
                    print(f"[Ollama服务日志] {line}")
        # 服务退出时打印日志
        if process.poll() is not None:
            print(f"[Ollama服务日志] 服务已退出（退出码：{process.returncode}）")
        self.is_running = False





#endregion--------------------------------------------------------------------------
















