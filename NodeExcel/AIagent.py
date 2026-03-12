



#region-------------------主图doubao_seedream+3+4
import torch
import numpy as np
from PIL import Image
import requests
import json
import base64
import time
import os
import io
import folder_paths

import comfy.utils

PROXIES = {
}

custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")
comfy_root = os.path.dirname(custom_nodes_paths[0]) if custom_nodes_paths else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
key_path = os.path.join(comfy_root, "custom_nodes", "ComfyUI-Apt_Preset", "NodeExcel", "ApiKey_doubao.txt")

def get_oubao_api_key():
    env_api_key = os.getenv("DOUBAO_API_KEY")
    if env_api_key and env_api_key.strip():
        return env_api_key.strip()
    if os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
        except Exception as e:
            print(f"读取API Key文件失败: {e}")
    return ""

def encode_image_to_base64(image_tensor):
    try:
        i = 255. * image_tensor.cpu().numpy().squeeze()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        byte_arr = io.BytesIO()
        img.save(byte_arr, format='PNG')
        byte_arr = byte_arr.getvalue()
        base64_bytes = base64.b64encode(byte_arr)
        base64_string = base64_bytes.decode('utf-8')
        return f"data:image/png;base64,{base64_string}"
    except Exception as e:
        print(f"ERROR: Image encoding to Base64 failed: {e}")
        return None

def process_image_to_tensor(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        np_image = np.array(img).astype(np.float32) / 255.0
        return torch.from_numpy(np_image)[None,]
    except Exception as e:
        print(f"ERROR: Failed to process image bytes into tensor: {e}")
        return None

def decode_base64_to_tensor(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]
        img_data = base64.b64decode(base64_string)
        return process_image_to_tensor(img_data)
    except Exception as e:
        print(f"ERROR: Image decoding from Base64 failed: {e}")
        return None

def download_image_to_tensor(url):
    try:
        print(f"Downloading image from URL: {url[:80]}...")
        response = requests.get(url, timeout=60, proxies=PROXIES)
        response.raise_for_status()
        return process_image_to_tensor(response.content)
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to download image from URL {url}: {e}")
        return None



class Ai_doubao_seedream:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "main_image1": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True,"default": ""}),
                "model": (["doubao-seedream-4-5-251128", "doubao-seedream-4-0-250828"],),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 15, "step": 1}),
                "auto_resize": (["crop", "pad", "stretch"], {"default": "crop"}),
            },
            "optional": {
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "image4": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False,"default": "" }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAME = ("image",)
    FUNCTION = "generate_image"
    CATEGORY = "Apt_Preset/prompt"

    def _process_image_channels(self, image):
        if image is None:
            return None
        if len(image.shape) == 4:
            b, h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        image = image.clamp(0.0, 1.0)
        return image

    def _auto_resize(self, image: torch.Tensor, target_h: int, target_w: int, auto_resize: str) -> torch.Tensor:
        if len(image.shape) == 4:
            b, h, w, c = image.shape
            if c in [3, 4]:
                image = image.permute(0, 3, 1, 2)
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c in [3, 4]:
                image = image.unsqueeze(0).permute(0, 3, 1, 2)
        
        if len(image.shape) != 4 or image.shape[1] not in [3, 4]:
            raise ValueError(f"Invalid image tensor format: {image.shape}, expected [B, C, H, W] with C=3/4")
        
        batch, ch, orig_h, orig_w = image.shape
        
        max_single_dim = 4096
        target_h = min(target_h, max_single_dim)
        target_w = min(target_w, max_single_dim)
        orig_h = max(orig_h, 32)
        orig_w = max(orig_w, 32)
        target_h = max(target_h, 32)
        target_w = max(target_w, 32)
        
        required_memory = batch * ch * target_h * target_w * 4
        max_allowed_memory = 16 * 1024 * 1024 * 1024
        if required_memory > max_allowed_memory:
            scale = np.sqrt(max_allowed_memory / (batch * ch * 4)) / np.sqrt(target_h * target_w)
            target_w = int(target_w * scale)
            target_h = int(target_h * scale)
            target_w = max(128, (target_w // 8) * 8)
            target_h = max(128, (target_h // 8) * 8)
            print(f"Warning: Target size exceeds memory limit, automatically adjusted to {target_w}x{target_h}")
        
        if auto_resize == "crop":
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            crop_h = min(target_h, new_h - y_offset)
            crop_w = min(target_w, new_w - x_offset)
            crop_h = max(crop_h, 32)
            crop_w = max(crop_w, 32)
            result = scaled[:, :, y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
            
        elif auto_resize == "pad":
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            black_bg = torch.zeros((batch, ch, target_h, target_w), dtype=image.dtype, device=image.device)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            black_bg[:, :, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled
            result = black_bg
            
        elif auto_resize == "stretch":
            result = comfy.utils.common_upscale(image, target_w, target_h, "bicubic", "disabled")
            
        else:
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            result = scaled[:, :, y_offset:y_offset + target_h, x_offset:x_offset + target_w]
        
        final_w = max(32, (result.shape[3] // 8) * 8)
        final_h = max(32, (result.shape[2] // 8) * 8)
        
        if final_w != result.shape[3] or final_h != result.shape[2]:
            x_offset = (result.shape[3] - final_w) // 2
            y_offset = (result.shape[2] - final_h) // 2
            result = result[:, :, y_offset:y_offset + final_h, x_offset:x_offset + final_w]
        
        result = result.permute(0, 2, 3, 1)
        return result

    def generate_image(self, main_image1, api_key, prompt, seed, model, max_images, auto_resize, image2=None, image3=None, image4=None):
        api_url = "https://ark.cn-beijing.volces.com/api/v3/images/generations"
        
        api_key = api_key.strip() if api_key.strip() else get_oubao_api_key()
        if not api_key or "在此输入" in api_key:
            print("ERROR: Volcano Engine API Key is missing.")
            return (torch.zeros(1, 512, 512, 3),)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        payload = {
            "model": model,
            "prompt": prompt,
            "seed": seed if seed != -1 else np.random.randint(0, 2147483647),
            "response_format": "b64_json",
            "watermark": False,
            "sequential_image_generation": "auto",
            "sequential_image_generation_options": {"max_images": max_images},
            "optimize_prompt_options": {"mode": "standard"}
        }

        min_total_pixels = 3686400
        target_w, target_h = 0, 0
        all_reference_images = []
        max_single_dim = 4096
        
        main_image1 = self._process_image_channels(main_image1)
        if main_image1 is not None and main_image1.nelement() > 0:
            orig_h = main_image1.shape[1]
            orig_w = main_image1.shape[2]
            orig_total = orig_w * orig_h
            
            if orig_total < min_total_pixels:
                scale = np.sqrt(min_total_pixels / orig_total)
                target_w = int(orig_w * scale)
                target_h = int(orig_h * scale)
                target_w = min(max(1920, (target_w // 8) * 8), max_single_dim)
                target_h = min(max(1920, (target_h // 8) * 8), max_single_dim)
                print(f"Original image size {orig_w}x{orig_h} has insufficient pixels, automatically scaled to {target_w}x{target_h}")
            else:
                target_w = min(orig_w, max_single_dim)
                target_h = min(orig_h, max_single_dim)
            
            main_image1 = self._auto_resize(main_image1, target_h, target_w, auto_resize)
            payload['size'] = f"{target_w}x{target_h}"
            
            main_base64 = encode_image_to_base64(main_image1)
            if main_base64:
                all_reference_images.append(main_base64)

            optional_images = [image2, image3, image4]
            for img_tensor in optional_images:
                if img_tensor is not None and img_tensor.nelement() > 0:
                    img_tensor = self._process_image_channels(img_tensor)
                    img_target_w = min(target_w, max_single_dim)
                    img_target_h = min(target_h, max_single_dim)
                    resized_img = self._auto_resize(img_tensor, img_target_h, img_target_w, auto_resize)
                    img_base64 = encode_image_to_base64(resized_img)
                    if img_base64:
                        all_reference_images.append(img_base64)
        else:
            payload['size'] = "2560x1440"
            print("No input image, using default compliant size 2560x1440")

        if all_reference_images:
            payload["image"] = all_reference_images

        payload_for_log = {k: v for k, v in payload.items() if k != 'image'}
        payload_for_log['image_count'] = len(payload.get('image', []))
        print(f"Sending single request to {api_url} with payload: {json.dumps(payload_for_log, indent=2)}")
        
        result_images = []
        try:
            start_time = time.time()
            response = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=180,
                proxies=PROXIES
            )
            end_time = time.time()
            
            print(f"API Response Status Code: {response.status_code}. Time taken: {end_time - start_time:.2f} seconds.")
            response.raise_for_status()
            
            result_json = response.json()

            if "data" in result_json and result_json["data"]:
                print(f"API returned {len(result_json['data'])} images.")
                for item in result_json["data"]:
                    processed_image = None
                    if item.get("b64_json"):
                        processed_image = decode_base64_to_tensor(item["b64_json"])
                    elif item.get("url"):
                        processed_image = download_image_to_tensor(item["url"])
                    
                    if processed_image is not None:
                        result_images.append(processed_image)
                    else:
                        print(f"Warning: Could not process API response item: {item}")
            else:
                print("Error: No 'data' found in API response.")
                print("Full API Response:", json.dumps(result_json, indent=2, ensure_ascii=False))

        except requests.exceptions.RequestException as e:
            print(f"ERROR: API request failed.")
            if e.response is not None:
                print(f"Error status code: {e.response.status_code}")
                try: 
                    print(f"--- API SERVER ERROR DETAILS ---")
                    print(json.dumps(e.response.json(), indent=2, ensure_ascii=False))
                    print(f"--------------------------------")
                except json.JSONDecodeError: 
                    print(f"Error response content: {e.response.text}")
            return (torch.zeros(1, 512, 512, 3),)

        if not result_images:
            print("ERROR: No images were generated.")
            return (torch.zeros(1, 512, 512, 3),) 
            
        final_batch = torch.cat(result_images, dim=0)
        return (final_batch,)




#endregion-------------------主图+3+4





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

MANUAL_COMFYUI_ROOT = "/comfy/mnt/ComfyUI"
COMFYUI_ROOT = None

if os.path.exists(os.path.join(MANUAL_COMFYUI_ROOT, "comfy")):
    COMFYUI_ROOT = MANUAL_COMFYUI_ROOT
else:
    current_file = os.path.abspath(__file__)
    for _ in range(10):
        parent_dir = os.path.dirname(current_file)
        if os.path.exists(os.path.join(parent_dir, "comfy")):
            COMFYUI_ROOT = parent_dir
            break
        current_file = parent_dir
    if COMFYUI_ROOT is None:
        COMFYUI_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

OLLAMA_MODEL_PATH = None
try:
    OLLAMA_MODEL_PATH = os.path.join(COMFYUI_ROOT, "models", "ollama")
    os.makedirs(OLLAMA_MODEL_PATH, exist_ok=True, mode=0o755)
    if not os.access(OLLAMA_MODEL_PATH, os.W_OK):
        raise PermissionError("无写入权限")
    os.environ["OLLAMA_MODELS"] = OLLAMA_MODEL_PATH
except:
    PLUGIN_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OLLAMA_MODEL_PATH = os.path.join(PLUGIN_ROOT, "ollama_models")
    os.makedirs(OLLAMA_MODEL_PATH, exist_ok=True, mode=0o755)
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
    except:
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
    except:
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
OLLMAMA_MODEL_NAME_IMAGE = ["None","qwen3-vl:latest","gemma3:12b"]
OLLMAMA_MODEL_NAME_TEXT = ["None","qwen2.5-coder:7b","qwen3-coder:30b","gemma3:1b-it-fp16","gemma3:12b"]





class AI_Ollama_image:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (OLLMAMA_MODEL_NAME_IMAGE, {"default": "qwen3-vl:latest"}),
                "preset": (list(IMAGE_PROMPTS.keys()), {"default": "None"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "custom_system_prompt：在预设preset=None时生效"}),
                "analysis_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "user prompt"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 2048, "min": 1, "max": 8192}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "enable_ocr": ("BOOLEAN", {"default": False}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),
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
        enable_ocr=False,
        custom_model="",
        custom_system_prompt=""
    ):
        if custom_model.strip():
            actual_model = custom_model.strip()
        else:
            actual_model = model_name
        
        if preset != "None":
            system_prompt = IMAGE_PROMPTS.get(preset, "")
        else:
            system_prompt = custom_system_prompt.strip()
        
        cleaned_analysis_prompt = analysis_prompt.strip()
        user_prompt = cleaned_analysis_prompt if cleaned_analysis_prompt else "请基于输入图片，详细描述内容、分析物体/场景/细节，若有多张图需对比异同"
        
        if enable_ocr:
            img_count = len([img for img in [image_1, image_2, image_3] if img is not None])
            ocr_msg = f"\n\n特别指令：请提取{img_count}张图片中所有可见文本内容，按图片序号（1-{img_count}）分类，以列表形式整理文本内容及文字位置信息"
            user_prompt += ocr_msg
        
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
                    return (f"图片{idx}处理/编码错误：{str(e)}", system_prompt)
        
        if not img_base64_list:
            return ("错误：未输入任何图片，请至少选择1张图片上传", system_prompt)
        
        ollama_host = "http://localhost:11434"
        try:
            data = {
                "model": actual_model,
                "prompt": user_prompt,
                "system": system_prompt,
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
                            return (f"模型错误：{line_data['error']}", system_prompt)
                        if line_data.get("done", False):
                            break
                    except:
                        continue
            
            analysis_result = analysis_result.strip()
            if not analysis_result:
                return ("错误：模型未返回有效结果，请检查模型是否支持视觉分析", system_prompt)
            
            return (analysis_result, system_prompt)
        
        except requests.exceptions.ConnectionError:
            error_msg = "连接错误：无法连接到Ollama服务，请检查Ollama是否已启动、服务地址正确、端口11434未被占用"
            return (error_msg, system_prompt)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                error_msg = f"模型未找到：请先执行 `ollama pull {actual_model}` 下载模型到{OLLAMA_MODEL_PATH}"
            elif e.response.status_code == 500:
                error_msg = f"Ollama服务内部错误：1. 请更新Ollama到最新版本（≥0.12.7）；2. 重新拉取模型 `ollama pull {actual_model}` 到{OLLAMA_MODEL_PATH}；3. 检查图片是否损坏"
            else:
                error_msg = f"HTTP错误：{str(e)}"
            return (error_msg, system_prompt)
        except requests.exceptions.Timeout:
            return ("错误：请求超时，请检查网络连接或增大超时时间", system_prompt)
        except Exception as e:
            error_msg = f"未知错误：{str(e)}"
            return (error_msg, system_prompt)




class AI_Ollama_text:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (OLLMAMA_MODEL_NAME_TEXT, {"default": "qwen2.5-coder:7b"}),
                "preset": (list(TEXT_PROMPTS.keys()), {"default": "None"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "custom_system_prompt：在预设preset=None时生效"}),
                "prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "user prompt"}),

            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
                "max_tokens": ("INT", {"default": 512, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("analysis_result", "system_prompt")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/prompt"
    
    def run(self, model_name, preset, prompt, temperature, max_tokens, seed=0, custom_system_prompt="", custom_model=""):
        if custom_model.strip():
            actual_model = custom_model.strip()
        else:
            actual_model = model_name
        
        if custom_system_prompt.strip():
            system_prompt = custom_system_prompt.strip()
        else:
            system_prompt = TEXT_PROMPTS.get(preset, "") if preset != "None" else ""
        
        ollama_host = "http://localhost:11434"
        
        try:
            url = f"{ollama_host}/api/generate"
            headers = {"Content-Type": "application/json"}
            data = {
                "model": actual_model,
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
            
            return (result, system_prompt)
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return (f"模型未找到：请先执行 `ollama pull {actual_model}` 下载模型到{OLLAMA_MODEL_PATH}", system_prompt)
            else:
                return (f"HTTP错误：{str(e)}", system_prompt)
        except Exception as e:
            return (f"错误: {str(e)}", system_prompt)




class Ai_Ollama_RunModel:
    _instance = None  # 用于跟踪实例
    
    def __init__(self):
        # 移除单例限制，允许ComfyUI正常创建实例
        # 如果已有实例，则复用其状态
        if Ai_Ollama_RunModel._instance is not None:
            # 复用现有实例的状态
            self.process = Ai_Ollama_RunModel._instance.process
            self.is_running = Ai_Ollama_RunModel._instance.is_running
        else:
            # 新实例初始化
            self.process: Optional[subprocess.Popen] = None
            self.is_running = False
            Ai_Ollama_RunModel._instance = self

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enable_service": ("BOOLEAN", {"default": True, "label_on": "启动", "label_off": "停止"})
            },
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
    FUNCTION = "control_ollama_service"
    CATEGORY = "Apt_Preset/prompt"

    def control_ollama_service(self, enable_service: bool, timeout: int = 20):
        if enable_service:
            return self._start_service(timeout)
        else:
            return self._stop_service()

    def _start_service(self, timeout: int = 20):
        # 检查是否已经运行
        if self.is_running and self.process and self.process.poll() is None:
            return (f"ℹ️ Ollama服务已经在运行中\n"
                    f"📁 模型存储路径：{OLLAMA_MODEL_PATH}\n"
                    f"🆔 服务PID：{self.process.pid}\n"
                    f"🌐 API地址：http://localhost:11434",)

        # 检查ollama命令是否存在
        try:
            subprocess.run(["ollama", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except FileNotFoundError:
            return ("错误：未找到ollama命令！\n请确保：1. 已安装Ollama（https://ollama.com/download）\n       2. Ollama已添加到系统环境变量",)

        # 停止现有的ollama进程
        self._stop_existing_ollama()

        # 启动服务
        return self._lightweight_start_service(timeout)

    def _stop_service(self):
        if not self.is_running and (self.process is None or self.process.poll() is not None):
            return ("ℹ️ Ollama服务未在运行",)

        try:
            # 终止进程
            if self.process and self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            # 杀死系统中的其他ollama进程
            self._stop_existing_ollama()
            
            self.is_running = False
            self.process = None
            
            return ("✅ Ollama服务已停止",)
        except Exception as e:
            return (f"停止服务时出错：{str(e)}",)

    def _stop_existing_ollama(self):
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/f", "/im", "ollama.exe"], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            else:
                subprocess.run(["pkill", "-f", "ollama"], 
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
            time.sleep(1)
        except:
            pass

    def _lightweight_start_service(self, timeout: int) -> tuple:
        cmd = ["ollama", "serve"]
        try:
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=creationflags
            )

            start_time = time.time()
            service_alive = False
            while time.time() - start_time < timeout:
                if self.process.poll() is not None:
                    stderr_data = self.process.stdout.read() if self.process.stdout else ""
                    return (f"启动失败：Ollama服务异常退出\n错误信息：{stderr_data}",)
                if self._check_service_connected():
                    service_alive = True
                    break
                time.sleep(0.5)

            if not service_alive:
                self.process.kill()
                return (f"启动失败：Ollama服务启动超时（{timeout}秒），API未连通",)

            self.is_running = True
            threading.Thread(target=self._print_service_log, args=(self.process,), daemon=True).start()

            success_msg = (
                f"✅ Ollama服务轻量启动成功！\n"
                f"📁 模型存储路径：{OLLAMA_MODEL_PATH}\n"
                f"🆔 服务PID：{self.process.pid}\n"
                f"🌐 API地址：http://localhost:11434\n"
                "💡 提示：\n"
                "  1. 服务已在后台运行，可通过API调用任意已下载的模型\n"
                "  2. 停止服务：启动器运行>关闭服务器或（Windows任务管理器/Linux/macOS pkill ollama）\n"
                "  3. 模型需提前下载到上述路径（命令：ollama pull 模型名）"
            )
            return (success_msg,)

        except Exception as e:
            return (f"启动失败：{str(e)}",)

    def _check_service_connected(self) -> bool:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False

    def _print_service_log(self, process: subprocess.Popen):
        while self.is_running and process.poll() is None:
            line = process.stdout.readline()
            if line:
                line = line.strip()
                if "error" in line.lower() or "listening" in line.lower() or "started" in line.lower():
                    pass
        if process.poll() is not None:
            self.is_running = False
            self.process = None

    @classmethod
    def cleanup(cls):
        """清理资源"""
        if cls._instance and cls._instance.is_running:
            cls._instance._stop_service()
        cls._instance = None






#endregion--------------------------------------------------------------------------




#region-----------GLM-combined-------------------------------------
import os
import json
import base64
import random
from PIL import Image
import numpy as np
import io
import requests
import folder_paths


ZHIPU_MODELS = [
    "None",
    "GLM-4.5-Flash",
    "glm-4v-flash",
    "XX----下面的要开通支付-----XX",
    "glm-5",
    "GLM-4.6V",
    "glm-4.7",
    "glm-4.5-air",
    "glm-4.5",

]




try:
    from zhipuai import ZhipuAI
    ZHIPUAI_AVAILABLE = True
except (ImportError, TypeError):
    ZhipuAI = None
    ZHIPUAI_AVAILABLE = False
    print("[GLM_Nodes] 警告：zhipuai 库导入失败，可能是 httpx 版本不兼容。GLM相关节点将不可用。")

current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")

def get_zhipuai_api_key():
    env_api_key = os.getenv("ZHIPUAI_API_KEY")
    if env_api_key and env_api_key.strip():
        return env_api_key.strip()
    
    custom_nodes_paths = []
    custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")

    comfy_root = os.path.dirname(custom_nodes_paths[0]) if custom_nodes_paths else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    key_path = os.path.join(comfy_root, "custom_nodes", "ComfyUI-Apt_Preset", "NodeExcel", "ApiKey_GLM.txt")
    
    if os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
        except Exception as e:
            print(f"[GLM_Nodes] 错误：读取GLM API Key文件失败: {e}")
    return ""

def load_prompts(prompt_type):
    if not os.path.exists(json_path):
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(prompt_type, {"None": ""})
    except Exception:
        return {"None": ""}

TEXT_PROMPTS = load_prompts("TEXT_PROMPTS")
IMAGE_PROMPTS = load_prompts("IMAGE_PROMPTS")



def analyze_glm_text_no_sdk(model, api_key, system_prompt, text_content, max_tokens, seed=None):
    if not api_key:
        raise ValueError("API密钥未配置，请通过节点输入/环境变量/ApiKey_GLM.txt配置")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if text_content:
        messages.append({"role": "user", "content": text_content})
    elif not system_prompt:
        messages.append({"role": "user", "content": "请作为专业AI助手提供帮助"})

    effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
    random.seed(effective_seed)

    try:
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.9,
            top_p=0.7,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"GLM API调用失败: {str(e)}")



class AI_GLM_text:
    def __init__(self):
        self.api_key = get_zhipuai_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (ZHIPU_MODELS, {"default": "GLM-4.5-Flash"}),

                "preset": (list(TEXT_PROMPTS.keys()), {"default": "None"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "custom_system_prompt：在预设preset=None时生效"}),
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "user prompt"}),
            },
            "optional": {
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "api_key_input": ("STRING", {"default": "", "multiline": False}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "system_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Apt_Preset/prompt"

    def analyze(self, model_name, preset, text, max_tokens, custom_system_prompt="", api_key_input="", seed=0, custom_model=""):
        actual_model = custom_model.strip() if custom_model.strip() else model_name
        
        if preset != "None":
            system_prompt = TEXT_PROMPTS.get(preset, "")
        else:
            system_prompt = custom_system_prompt.strip()
        
        text_content = text.strip()
        
        api_key = api_key_input.strip() if api_key_input.strip() else self.api_key
        if not api_key or "在此输入" in api_key:
            return ("API密钥未配置，请通过节点输入/环境变量/ApiKey_GLM.txt配置", system_prompt)

        try:
            result = analyze_glm_text_no_sdk(
                actual_model,
                api_key,
                system_prompt,
                text_content,
                max_tokens,
                seed if seed != 0 else None
            )
            return (result, system_prompt)
        except Exception as e:
            return (f"处理失败: {str(e)}", system_prompt)

def resize_to_limit(img, max_pixels=262144):
    width, height = img.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return img
    scale = (max_pixels / total_pixels) ** 0.5
    return img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)

def encode_image(pil_image, save_tokens=True):
    buffered = io.BytesIO()
    if save_tokens:
        image = resize_to_limit(pil_image)
        image.save(buffered, format="JPEG", optimize=True, quality=75)
    else:
        pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def tensor2pil(image):
    batch_count = image.size(0) if len(image.shape) > 3 else 1
    if batch_count > 1:
        return [tensor2pil(image[i])[0] for i in range(batch_count)]
    return [Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))]

def analyze_glm_image_no_sdk(images, model, api_key, system_prompt, user_prompt, max_tokens, seed=None):
    if not api_key:
        raise ValueError("API密钥未配置，请通过节点输入/环境变量/ApiKey_GLM.txt配置")
    
    content_parts = []
    if system_prompt:
        content_parts.append({"type": "text", "text": f"{system_prompt}\n{user_prompt}"})
    else:
        content_parts.append({"type": "text", "text": user_prompt})
    
    for i, img in enumerate(images):
        try:
            image_base64 = encode_image(img)
            content_parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}})
        except Exception as e:
            raise Exception(f"图片{i+1}格式转换失败: {str(e)}")

    effective_seed = seed if seed != 0 else random.randint(0, 0xffffffffffffffff)
    random.seed(effective_seed)

    try:
        client = ZhipuAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content_parts}],
            temperature=0.9,
            top_p=0.7,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise Exception(f"GLM-4V API调用失败: {str(e)}")

class AI_GLM_image:
    def __init__(self):
        self.api_key = get_zhipuai_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (ZHIPU_MODELS, {"default": "glm-4v-flash"}),
                "preset": (list(IMAGE_PROMPTS.keys()), {"default": "None"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "custom_system_prompt：在预设preset=None时生效"}),
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "user prompt"}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1024, "min": 10, "max": 4096, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
                "api_key_input": ("STRING", {"default": "", "multiline": False}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "system_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Apt_Preset/prompt"

    def analyze(self, model_name, preset, text, max_tokens, custom_system_prompt="", image_1=None, image_2=None, image_3=None, seed=0, api_key_input="", custom_model=""):
        actual_model = custom_model.strip() if custom_model.strip() else model_name
        
        if preset != "None":
            system_prompt = IMAGE_PROMPTS.get(preset, "")
        else:
            system_prompt = custom_system_prompt.strip()
        
        user_prompt = text.strip() if text.strip() else "生成输入图片的详细中文描述"
        
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
            return ("错误：请至少输入1张有效图片", system_prompt)
        
        api_key = api_key_input.strip() if api_key_input.strip() else self.api_key
        if not api_key or "在此输入" in api_key:
            return ("API密钥未配置，请通过节点输入/环境变量/ApiKey_GLM.txt配置", system_prompt)
        
        try:
            result = analyze_glm_image_no_sdk(
                input_images,
                actual_model,
                api_key,
                system_prompt,
                user_prompt,
                max_tokens,
                seed if seed != 0 else None
            )
            return (result, system_prompt)
        except Exception as e:
            return (f"分析失败: {str(e)}", system_prompt)


#endregion--------------------------------------------------------------------------







#region-----------qwen-combined（无SDK版）-----------------
import os
import json
import requests
import numpy as np
from io import BytesIO
import io
import base64
import folder_paths
from PIL import Image


current_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(current_dir, "AiPromptPreset.json")
QWEN_TEXT_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
QWEN_MULTIMODAL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

def get_aliyun_api_key():
    env_api_key = os.getenv("aliyun_API_KEY")
    if env_api_key and env_api_key.strip():
        return env_api_key.strip()
    
    custom_nodes_paths = []
    try:
        custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes")
    except ImportError:
        pass
    
    comfy_root = os.path.dirname(custom_nodes_paths[0]) if custom_nodes_paths else os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
    key_path = os.path.join(comfy_root, "custom_nodes", "ComfyUI-Apt_Preset", "NodeExcel", "ApiKey_AI_Qwen.txt")
    
    if os.path.exists(key_path):
        try:
            with open(key_path, "r", encoding="utf-8") as f:
                api_key = f.read().strip()
                if api_key:
                    return api_key
        except Exception as e:
            print(f"读取Qwen API Key文件失败: {e}")
    return ""

def load_prompts(prompt_type):
    if not os.path.exists(json_path):
        return {"None": ""}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(prompt_type, {"None": ""})
    except Exception:
        return {"None": ""}

TEXT_PROMPTS = load_prompts("TEXT_PROMPTS")
IMAGE_PROMPTS = load_prompts("IMAGE_PROMPTS")

def request_with_retry(url, headers, payload, seed, max_retries=3):
    for retry in range(max_retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60,
                proxies={}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if "seed" in str(e).lower() and retry < max_retries - 1:
                if "seed" in payload["parameters"]:
                    payload["parameters"].pop("seed", None)
                continue
            if retry == max_retries - 1:
                error_detail = f"请求失败: {str(e)}"
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail += f"\n服务器返回: {json.dumps(e.response.json(), indent=2)}"
                    except:
                        error_detail += f"\n服务器返回: {e.response.text}"
                raise Exception(error_detail)

def analyze_text_no_sdk(model, api_key, system_prompt, text_content, max_tokens, seed=None):
    if not api_key:
        raise ValueError("API密钥未配置，请通过以下方式之一设置：\n1. 设置系统环境变量 aliyun_API_KEY\n2. 在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥")

    messages = []
    # 修复：确保system_prompt是字符串且去空
    system_prompt_str = str(system_prompt).strip() if system_prompt else ""
    if system_prompt_str:
        messages.append({"role": "system", "content": system_prompt_str})
    if text_content:
        messages.append({"role": "user", "content": text_content})
    elif not system_prompt_str:
        messages.append({"role": "user", "content": "请根据需求完成相关代码或文本任务（如代码生成、代码解释、编程问题解答等）"})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": {"messages": messages},
        "parameters": {
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.8,
            "result_format": "message"
        }
    }

    if seed is not None and seed != 0:
        payload["parameters"]["seed"] = seed

    result_json = request_with_retry(QWEN_TEXT_API_URL, headers, payload, seed)
    
    if "output" in result_json and "choices" in result_json["output"]:
        return result_json["output"]["choices"][0]["message"]["content"].strip()
    else:
        raise Exception(f"API返回格式异常: {json.dumps(result_json, indent=2)}")

class AI_Qwen_text:
    def __init__(self):
        self.api_key = get_aliyun_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": (["None", "qwen3-coder-plus", "qwen3-coder-plus-2025-09-23", "qwen3-coder-flash", 
                              "qwen3-coder-480b-a35b-instruct", "qwen3-coder-30b-a3b-instruct"], {"default": "qwen3-coder-plus"}),

                "preset": (list(TEXT_PROMPTS.keys()), {"default": "None"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "custom_system_prompt：在预设preset=None时生效"}),
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "user prompt"}),

            },
            "optional": {
                "max_tokens": ("INT", {"default": 1024, "min": 10, "max": 8192,"step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "api_key_input": ("STRING", {"default": "", "multiline": False}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),                
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "system_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Apt_Preset/prompt"

    def analyze(self, llm_model, preset, text, max_tokens, custom_system_prompt="", api_key_input="", seed=0, custom_model=""):
        actual_model = custom_model.strip() if custom_model.strip() else llm_model
        
        if preset != "None":
            system_prompt = TEXT_PROMPTS.get(preset, "")
        else:
            system_prompt = custom_system_prompt.strip()
        
        text_content = text.strip()
        
        api_key = api_key_input.strip() if api_key_input.strip() else self.api_key
        if not api_key or "在此输入" in api_key:
            return ("API密钥未配置，请通过以下方式之一设置：\n1. 在节点输入中直接填写API密钥\n2. 设置系统环境变量 aliyun_API_KEY\n3. 或在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥", system_prompt)

        try:
            result = analyze_text_no_sdk(
                actual_model,
                api_key,
                system_prompt,
                text_content,
                max_tokens,
                seed if seed != 0 else None
            )
            return (result, system_prompt)
        except Exception as e:
            return (f"处理失败: {str(e)}", system_prompt)

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
    # 修复：增加空值判断，避免返回无效图片
    img_np = np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    if img_np.size == 0:
        return [None]
    return [Image.fromarray(img_np)]

def analyze_images_no_sdk(images, model, api_key, system_prompt, user_prompt, max_tokens, seed=None):
    if not api_key:
        raise ValueError("API密钥未配置，请通过以下方式之一设置：\n1. 设置系统环境变量 aliyun_API_KEY\n2. 在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥")
    
    # 修复：强制转换为字符串并去空，彻底避免list类型
    system_prompt_str = str(system_prompt).strip() if system_prompt else ""
    user_content = []
    
    for img in images:
        if img is None:
            continue
        user_content.append({"image": f"data:image/png;base64,{encode_image(img)}"})
    user_content.append({"text": user_prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "input": {"messages": []},
        "parameters": {
            "max_tokens": max_tokens,
            "result_format": "message"
        }
    }

    if system_prompt_str:
        payload["input"]["messages"].append({"role": "system", "content": system_prompt_str})
    payload["input"]["messages"].append({"role": "user", "content": user_content})

    if seed is not None and seed != 0:
        payload["parameters"]["seed"] = seed

    result_json = request_with_retry(QWEN_MULTIMODAL_API_URL, headers, payload, seed)
    
    # ========== 核心修复：处理多图场景下content可能为列表的情况 ==========
    if "output" in result_json and "choices" in result_json["output"]:
        content = result_json["output"]["choices"][0]["message"]["content"]
        # 如果content是列表，拼接成字符串；否则直接转字符串后去空
        if isinstance(content, list):
            # 遍历列表，提取所有文本内容拼接
            content_str = ""
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    content_str += item["text"] + "\n"
                elif isinstance(item, str):
                    content_str += item + "\n"
            return content_str.strip()
        else:
            # 确保是字符串后再调用strip
            return str(content).strip()
    else:
        raise Exception(f"API返回格式异常: {json.dumps(result_json, indent=2)}")

class AI_Qwen:
    def __init__(self):
        self.api_key = get_aliyun_api_key()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (["None", "qwen-vl-max", "qwen-vl-max-latest"], {"default": "qwen-vl-max-latest"}),
                "preset": (list(IMAGE_PROMPTS.keys()), {"default": "None"}),
                "custom_system_prompt": ("STRING", {"default": "", "multiline": True, "placeholder": "custom_system_prompt：在预设preset=None时生效"}),
                "text": ("STRING", {"default": "", "multiline": True, "placeholder": "user prompt"}),
            },
            "optional": {                
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "max_tokens": ("INT", {"default": 1024, "min": 10, "max": 8192, "step": 10}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999999, "step": 1}),
                "api_key_input": ("STRING", {"default": "", "multiline": False}),
                "custom_model": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("result", "system_prompt")
    FUNCTION = "analyze"
    CATEGORY = "Apt_Preset/prompt"

    def analyze(self, model, preset, text, max_tokens, custom_system_prompt="", api_key_input="", image_1=None, image_2=None, image_3=None, seed=0, custom_model=""):
        actual_model = custom_model.strip() if custom_model.strip() else model
        
        if preset != "None":
            system_prompt = IMAGE_PROMPTS.get(preset, "")
        else:
            system_prompt = custom_system_prompt.strip()
        
        user_prompt = text.strip() if text.strip() else "生成输入图片的详细中文描述"
        
        input_images = []
        # 修复：增加pil_img非空判断
        if image_1 is not None:
            pil_img = tensor2pil(image_1)[0]
            if pil_img and isinstance(pil_img, Image.Image):
                input_images.append(pil_img)
        if image_2 is not None:
            pil_img = tensor2pil(image_2)[0]
            if pil_img and isinstance(pil_img, Image.Image):
                input_images.append(pil_img)
        if image_3 is not None:
            pil_img = tensor2pil(image_3)[0]
            if pil_img and isinstance(pil_img, Image.Image):
                input_images.append(pil_img)
        
        if not input_images:
            return ("错误：请至少输入1张有效图片", system_prompt)
        
        api_key = api_key_input.strip() if api_key_input.strip() else self.api_key
        if not api_key or "在此输入" in api_key:
            return ("API密钥未配置，请通过以下方式之一设置：\n1. 在节点输入中直接填写API密钥\n2. 设置系统环境变量 aliyun_API_KEY\n3. 或在custom_nodes/ComfyUI-Apt_Preset/NodeExcel目录下创建ApiKey_AI_Qwen.txt并写入密钥", system_prompt)
        
        try:
            result = analyze_images_no_sdk(
                input_images,
                actual_model,
                api_key,
                system_prompt,
                user_prompt,
                max_tokens,
                seed if seed != 0 else None
            )
            return (result, system_prompt)
        except Exception as e:
            return (f"分析失败: {str(e)}", system_prompt)

#endregion--------------------------------------------------------------------------










