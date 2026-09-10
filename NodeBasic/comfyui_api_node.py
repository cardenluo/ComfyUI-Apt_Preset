import json
import requests
import random
from typing import Optional, Dict, Any, List

class ComfyUI_UniversalAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_endpoint": ("STRING", {"default": "https://api.openai.com/v1", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "node_type": (["text_generation", "vision_analysis", "image_generation"], {"default": "text_generation"}),
                "model": ("STRING", {"default": "gpt-4", "multiline": False}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "Hello, how are you?", "multiline": True}),
                "max_tokens": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 9999999999}),
            },
            "optional": {
                "image_input": ("IMAGE", {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING", "DICT")
    RETURN_NAMES = ("output", "raw_response")
    FUNCTION = "run"
    CATEGORY = "API Nodes"

    def run(self, api_endpoint, api_key, node_type, model, system_prompt, user_prompt, max_tokens, temperature, seed, image_input=None):
        if seed == 0:
            seed = random.randint(1, 9999999999)

        if node_type == "text_generation":
            return self._text_generation(api_endpoint, api_key, model, system_prompt, user_prompt, max_tokens, temperature, seed)
        elif node_type == "vision_analysis":
            return self._vision_analysis(api_endpoint, api_key, model, system_prompt, user_prompt, max_tokens, temperature, seed, image_input)
        elif node_type == "image_generation":
            return self._image_generation(api_endpoint, api_key, model, system_prompt, user_prompt, max_tokens, temperature, seed)
        else:
            return ("Invalid node type", {"error": "Invalid node type"})

    def _text_generation(self, api_endpoint, api_key, model, system_prompt, user_prompt, max_tokens, temperature, seed):
        url = f"{api_endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            output = data['choices'][0]['message']['content']
            return (output, data)
        except Exception as e:
            return (f"Error: {str(e)}", {"error": str(e)})

    def _vision_analysis(self, api_endpoint, api_key, model, system_prompt, user_prompt, max_tokens, temperature, seed, image_input):
        if not image_input:
            return ("Error: No image input provided for vision analysis", {"error": "No image input"})

        url = f"{api_endpoint.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Convert image to base64 (simplified implementation)
        # Note: In actual implementation, you'd need to properly handle the image input
        image_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            output = data['choices'][0]['message']['content']
            return (output, data)
        except Exception as e:
            return (f"Error: {str(e)}", {"error": str(e)})

    def _image_generation(self, api_endpoint, api_key, model, system_prompt, user_prompt, max_tokens, temperature, seed):
        url = f"{api_endpoint.rstrip('/')}/images/generations"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "prompt": f"{system_prompt}\n\n{user_prompt}",
            "n": 1,
            "size": "1024x1024",
            "seed": seed
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            image_url = data['data'][0]['url']
            return (image_url, data)
        except Exception as e:
            return (f"Error: {str(e)}", {"error": str(e)})

NODE_CLASS_MAPPINGS = {
    "UniversalAPI": ComfyUI_UniversalAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UniversalAPI": "Universal API Node"
}
