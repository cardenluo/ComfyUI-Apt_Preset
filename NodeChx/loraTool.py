
import sys
import os
import folder_paths
import comfy
from ..main_unit import *




#region---------------------------从外部导入模块，加载检查-------------------------------

#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Apt_Preset\NodeChx\loraTool.py"
#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Custom-Scripts"
current_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
target_dir = os.path.join(custom_nodes_dir, 'ComfyUI-Custom-Scripts')
#target_dir = r"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Custom-Scripts"


LoraLoaderWithImages = None

if not os.path.exists(target_dir):
    pass
else:
    sys.path.append(target_dir)
    try:
        from py.better_combos import (
            LoraLoaderWithImages
        )
    except ImportError as e:
        pass

# 定义一个函数用于检查插件是否安装
def check_Custom_Scripts_installed():
    if LoraLoaderWithImages is None:
        raise RuntimeError(" Please install Custom-Scripts  before using this function.")


#endregion---------------------------加载检查-------------------------------

class LoraLoader:
    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the diffusion model. This value can be negative."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to modify the CLIP model. This value can be negative."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    OUTPUT_TOOLTIPS = ("The modified diffusion model.", "The modified CLIP model.")
    FUNCTION = "load_lora"

    CATEGORY = "loaders"

    def load_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)


class XXLoraLoaderWithImages(LoraLoader):
    RETURN_TYPES = (*LoraLoader.RETURN_TYPES, "STRING",)
    RETURN_NAMES = (*getattr(LoraLoader, "RETURN_NAMES", LoraLoader.RETURN_TYPES), "example")

    @classmethod
    def INPUT_TYPES(s):
        types = super().INPUT_TYPES()
        types["optional"] = {"prompt": ("STRING", {"hidden": True})}
        return types

    def load_lora(self, **kwargs):
        prompt = kwargs.pop("prompt", "")
        return (*super().load_lora(**kwargs), prompt)






class pre_mul_lora(LoraLoaderWithImages):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name1": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the first LoRA."}),
                "strength1": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to apply the first LoRA to both model and clip."}),
                "lora_name2": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the second LoRA."}),
                "strength2": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to apply the second LoRA to both model and clip."}),
                "lora_name3": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the third LoRA."}),
                "strength3": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to apply the third LoRA to both model and clip."}),
                "lora_name4": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the fourth LoRA."}),
                "strength4": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to apply the fourth LoRA to both model and clip."}),
            },
            "optional": {
                "prompt1": ("STRING", {"hidden": True}),
                "prompt2": ("STRING", {"hidden": True}),
                "prompt3": ("STRING", {"hidden": True}),
                "prompt4": ("STRING", {"hidden": True})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "clip", "example1", "example2", "example3", "example4")
    FUNCTION = "apply_multi_loras"
    CATEGORY = "Apt_Preset/Lora"

    def apply_multi_loras(self,
                          model, clip,
                          lora_name1, strength1, lora_name2, strength2,
                          lora_name3, strength3, lora_name4, strength4,
                          prompt1="", prompt2="", prompt3="", prompt4=""):
        # 创建一个临时实例来调用 load_lora 方法
        loader = LoraLoaderWithImages()

        # 第一次 LoRA 应用
        model1, clip1, example1 = loader.load_lora(
            model=model, clip=clip, lora_name=lora_name1,
            strength_model=strength1, strength_clip=strength1,
            prompt=prompt1
        )

        # 第二次 LoRA 应用
        model2, clip2, example2 = loader.load_lora(
            model=model1, clip=clip1, lora_name=lora_name2,
            strength_model=strength2, strength_clip=strength2,
            prompt=prompt2
        )

        # 第三次 LoRA 应用
        model3, clip3, example3 = loader.load_lora(
            model=model2, clip=clip2, lora_name=lora_name3,
            strength_model=strength3, strength_clip=strength3,
            prompt=prompt3
        )

        # 第四次 LoRA 应用
        model4, clip4, example4 = loader.load_lora(
            model=model3, clip=clip3, lora_name=lora_name4,
            strength_model=strength4, strength_clip=strength4,
            prompt=prompt4
        )

        return (model4, clip4, example1, example2, example3, example4)


class SingleLoraNode(LoraLoaderWithImages):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model the LoRA will be applied to."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model the LoRA will be applied to."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the LoRA."}),
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01, "tooltip": "How strongly to apply the LoRA to both model and clip."}),
            },
            "optional": {
                "prompt": ("STRING", {"hidden": True})
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "example")
    FUNCTION = "apply_single_lora"
    CATEGORY = "Apt_Preset/Lora"

    def apply_single_lora(self, model, clip, lora_name, strength, prompt=""):
        # 调用父类的 load_lora 方法，并统一设置 strength_model 和 strength_clip
        model_out, clip_out, example = super().load_lora(
            model=model,
            clip=clip,
            lora_name=lora_name,
            strength_model=strength,
            strength_clip=strength,
            prompt=prompt
        )

        return (model_out, clip_out, example)