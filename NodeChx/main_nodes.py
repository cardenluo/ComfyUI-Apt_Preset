


#region-----------导入与全局函数-------------------------------------------------
# 内置库
import os
import sys
import glob
import csv
import json
import math
import re
#from turtle import width



# 第三方库
import numpy as np
import torch
import toml
import aiohttp
from aiohttp import web
from PIL import Image, ImageFilter
from tqdm import tqdm

import nodes
# 本地库
import comfy
import folder_paths
import node_helpers
import latent_preview
from server import PromptServer
from nodes import common_ksampler, CLIPTextEncode, ControlNetLoader, KSamplerAdvanced,SetLatentNoiseMask, ControlNetApplyAdvanced, VAEDecode, VAEEncode, DualCLIPLoader, CLIPLoader,ConditioningConcat, ConditioningAverage, InpaintModelConditioning, LoraLoader, CheckpointLoaderSimple, ImageScale,VAEDecodeTiled,VAELoader
from comfy.cli_args import args
from typing import Optional, Tuple, Dict, Any, Union, cast
from comfy.comfy_types.node_typing import IO

from math import ceil
from nodes import CLIPSetLastLayer, CheckpointLoaderSimple, UNETLoader
from comfy_extras.nodes_hidream import QuadrupleCLIPLoader
from comfy.utils import load_torch_file as comfy_load_torch_file
from nodes import CLIPVisionLoader, CLIPVisionEncode,KSampler



from .load_GGUF.nodes import  UnetLoaderGGUF2
from ..main_unit import *
from ..office_unit import *


#---------------------安全导入------


try:
    import cv2
    REMOVER_AVAILABLE = True  
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  

try:
    from comfy_extras.nodes_model_patch import ModelPatchLoader, QwenImageDiffsynthControlnet, USOStyleReference
    REMOVER_AVAILABLE = True  
except ImportError:
    ModelPatchLoader = None
    QwenImageDiffsynthControlnet = None
    USOStyleReference = None
    REMOVER_AVAILABLE = False  



#region---------------注册和检测gguf----------------------------------------

def update_folder_names_and_paths(key, targets=[]):
    base = folder_paths.folder_names_and_paths.get(key, ([], {}))
    base = base[0] if isinstance(base[0], (list, set, tuple)) else []
    target = next((x for x in targets if x in folder_paths.folder_names_and_paths), targets[0])
    orig, _ = folder_paths.folder_names_and_paths.get(target, ([], {}))
    folder_paths.folder_names_and_paths[key] = (orig or base, {".gguf"})
    if base and base != orig:
        logging.warning(f"Unknown file list already present on key {key}: {base}")
update_folder_names_and_paths("unet_gguf", ["diffusion_models", "unet"])
update_folder_names_and_paths("clip_gguf", ["text_encoders", "clip"])




def check_UnetLoaderGGUF2_installed():
    if UnetLoaderGGUF2 is None:
        raise RuntimeError(" Please install the plugin ComfyUI-GGUF first")





#endregion---------------注册gguf----------------------------------------




#region------------------------preset---------------------------------#


# 获取当前文件所在目录的上一级目录
parent_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

# 插入上一级目录下的 comfy 到 sys.path
sys.path.insert(0, os.path.join(parent_directory, "comfy"))

routes = PromptServer.instance.routes
@routes.post('/Apt_Preset_path')
async def my_function(request):
    the_data = await request.post()
    sum_load_adv.handle_my_message(the_data)
    return web.json_response({})

# 使用上一级目录作为基础路径
my_directory_path = parent_directory
presets_directory_path = os.path.join(my_directory_path, "presets")


preset_list = []
tmp_list = []
tmp_list += glob.glob(f"{presets_directory_path}/**/*.toml", recursive=True)
for l in tmp_list:
    preset_list.append(os.path.relpath(l, presets_directory_path))

if len(preset_list) > 1: preset_list.sort()


available_ckpt = folder_paths.get_filename_list("checkpoints")
available_unets = list(set(folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf")))
available_clips = list(set(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")))

CLIP_TYPE = ["sdxl", "sd3", "flux", "hunyuan_video", "stable_diffusion", "stable_audio", "mochi", 
             "ltxv", "pixart", "cosmos","lumina2", "wan", "hidream", "chroma", "ace", "omnigen2", "qwen_image"]



def getNewTomlnameExt(tomlname, folderpath, savetype):

    tomlnameExt = tomlname + ".toml"
    
    if savetype == "new save":

        filename_list = []
        tmp_list = []
        tmp_list += glob.glob(f"{folderpath}/**/*.toml", recursive=True)
        for l in tmp_list:
            filename_list.append(os.path.relpath(l, folderpath))
        
        duplication_flag = False
        for f in filename_list:
            if tomlnameExt == f:
                duplication_flag = True
                
        if duplication_flag:
            count = 1
            while duplication_flag:
                new_tomlnameExt = f"{tomlname}_{count}.toml"
                if not new_tomlnameExt in filename_list:
                    tomlnameExt = new_tomlnameExt
                    duplication_flag = False
                count += 1
                
    return tomlnameExt


#endregion------------------------preset---------------------------------#

#endregion


#region-----------基础节点context------------------------------------------------------------------------------#



class Data_chx_Merge:
    @classmethod
    def INPUT_TYPES(cls): 
        return {
            "required": {  },
            "optional": {
            "context": ("RUN_CONTEXT",),  
                "chx1": ("RUN_CONTEXT",),
                "chx2": ("RUN_CONTEXT",),
            },
        }

    RETURN_TYPES = "RUN_CONTEXT",
    RETURN_NAMES = "context",
    CATEGORY = "Apt_Preset/chx_load"
    FUNCTION = "merge"
    

    def get_return_tuple(self, ctx):
        return get_orig_context_return_tuple(ctx)

    def merge(self, context=None, chx1=None, chx2=None):
        ctxs = [context, chx1, chx2]  
        ctx = merge_new_context(*ctxs)
        return self.get_return_tuple(ctx)



class Data_basic:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  "context": ("RUN_CONTEXT",),   },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "latent_image": ("IMAGE",),
                "latent_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","CONDITIONING","LATENT","VAE","CLIP","IMAGE","MASK",)
    RETURN_NAMES = ("context", "model","positive","negative","latent","vae","clip","latent_image","latent_mask",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, context=None,model=None,positive=None,negative=None,latent=None,vae =None,clip =None,latent_image =None,latent_mask=None ):
        if model is None:
            model = context.get("model")
        if positive is None:
            positive = context.get("positive")
        if negative is None:
            negative = context.get("negative")
        if vae is None:
            vae = context.get("vae")
        if clip is None:
            clip = context.get("clip")
        
        if latent_image is not None: 
            latent = VAEEncode().encode(vae, latent_image)[0]     
            if latent_mask is not None:
                if isinstance(latent, dict) and "samples" in latent:
                    latent_copy = {"samples": latent["samples"].clone()}
                    latent = self.set_latent_mask2(latent_copy, latent_mask)
                else:
                    latent = self.set_latent_mask2(latent, latent_mask)

        elif latent is not None: 
            pass
        else:
            latent = context.get("latent")

        context = new_context(context,model=model,positive=positive,negative=negative,latent=latent,vae=vae,clip=clip,images=latent_image,mask=latent_mask)
        return (context, model, positive, negative, latent, vae, clip, latent_image, latent_mask)
    def set_latent_mask2(self,latent, mask):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent 必须是包含 'samples' 键的字典")
        newlatent = {
            "samples": latent["samples"].clone()  # 只对张量部分调用 clone()
        }
        
        if mask is not None:
            newlatent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        
        return newlatent




class Data_sampleData:  
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "optional": {  "context": ("RUN_CONTEXT",),   },
        }

    RETURN_TYPES = ("RUN_CONTEXT","INT","FLOAT", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("context","steps","cfg","sampler","scheduler" )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def sample(self, context, ):
        
        steps=context.get("steps",None)
        cfg=context.get("cfg",None) 
        sampler=context.get("sampler",None) 
        scheduler=context.get("scheduler",None) 
        
        return (context,steps,cfg,sampler,scheduler )



class Data_select:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { "context": ("RUN_CONTEXT",),           
                        },
        
            "optional": {
                "type": (["model", "clip", "positive", "negative", "vae", "latent", "images", "mask",
                        "clip1", 
                        "clip2", 
                        "clip3", 
                        "clip4",
                        "unet_name", 
                        "ckpt_name",
                        "pos", 
                        "neg",
                        "width",
                        "height",
                        "batch",
                        "data",
                        "data1",
                        "data2",
                        "data3",
                        "data4",
                        "data5",
                        "data6",
                        "data7",
                        "data8",
                        "data9",
                        ], {}),
            },
        }

    RETURN_TYPES = (ANY_TYPE,)
    RETURN_NAMES = ("data",)
    FUNCTION = "pipeout"

    CATEGORY = "Apt_Preset/chx_load"

    def pipeout(self, type, context=None):

        out = context[type]
        return (out,)


class Data_presetData:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "context": ("RUN_CONTEXT",),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT",                                     
                    available_ckpt,
                    available_unets, 
                    available_clips,
                    available_clips,
                    available_clips,  
                    available_clips,  
                    "STRING", 
                    "STRING", 
                    "INT",
                    "INT",
                    "INT",
                    )

    RETURN_NAMES = (
            "context",
            "ckpt_name",
            "unet_name", 
            "clip1_name", 
            "clip2_name", 
            "clip3_name", 
            "clip4_name", 
            "pos", 
            "neg",
            "width",
            "height",
            "batch",
                    )

    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    FUNCTION = "convert"

    def convert(self, context=None):
        ckpt_name = context.get("ckpt_name", None)
        unet_name = context.get("unet_name", None)
        clip1 = context.get("clip1", None)
        clip2 = context.get("clip2", None)
        clip3 = context.get("clip3", None)
        clip4 = context.get("clip3", None)
        pos = context.get("pos", None)
        neg = context.get("neg", None)
        width = context.get("width", None)
        height = context.get("height", None)
        batch = context.get("batch", None)

        return (
            context,
            ckpt_name,
            unet_name, 
            clip1, 
            clip2, 
            clip3, 
            clip4, 
            pos, 
            neg,
            width,
            height,
            batch
        )


class Data_bus_chx:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            
            "optional": {
                "context": ("RUN_CONTEXT",),   
                "data1": ( ANY_TYPE, ),
                "data2": ( ANY_TYPE, ),
                "data3": ( ANY_TYPE, ),
                "data4": ( ANY_TYPE, ),
                "data5": ( ANY_TYPE, ),
                "data6": ( ANY_TYPE, ),
                "data7": ( ANY_TYPE, ),
                "data8": ( ANY_TYPE, ),
            },
        }


    RETURN_TYPES = ("RUN_CONTEXT",ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,ANY_TYPE,)
    RETURN_NAMES = ("context", "data1","data2","data3","data4","data5","data6","data7","data8",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, context=None, data1=None, data2=None, data3=None, data4=None, data5=None, data6=None, data7=None, data8=None):
        # 先检查 context 是否为 None
        if context is None:
            # 如果 context 为 None，可以创建一个新的上下文或者根据需求处理
            # 这里假设 new_context 可以在没有输入的情况下创建一个默认的上下文
            context = {}  # 或者使用其他方式初始化一个新的 context

        if data1 is None:
            data1 = context.get("data1")
        if data2 is None:
            data2 = context.get("data2")
        if data3 is None:
            data3 = context.get("data3")
        if data4 is None:
            data4 = context.get("data4")
        if data5 is None:
            data5 = context.get("data5")
        if data6 is None:
            data6 = context.get("data6")
        if data7 is None:
            data7 = context.get("data7")
        if data8 is None:
            data8 = context.get("data8")

        context = new_context(context, data1=data1, data2=data2, data3=data3, data4=data4, data5=data5, data6=data6, data7=data7, data8=data8)

        return (context, data1,data2,data3,data4,data5,data6,data7,data8,)

#endregion




#region-----------加载器load-----------------------------------------------------------------------------------#



def safe_load_torch_file(path, device="cpu"):
    """兼容 PyTorch 2.6 的加载方式，先检测再降低安全等级"""
    try:
        return comfy_load_torch_file(path, device=device)
    except Exception as e:
        logger.warning(f"Failed with weights_only=True: {e}")
        logger.info("Retrying with weights_only=False (unsafe)")
        return torch.load(path, map_location=device, weights_only=False)



class sum_load_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional":{
                "preset": (["None"] + preset_list, {"default": "ckpt-sd.toml"}),
                "ckpt_name": (["None"] + available_ckpt,),
                "unet_name": (["None"] + available_unets,),
                "unet_Weight_Dtype": (["None", "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_type": (["None"] + CLIP_TYPE,),
                "clip1": (["None"] + available_clips,),
                "clip2": (["None"] + available_clips,),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "clip3": (["None"] + available_clips,),
                "clip4": (["None"] + available_clips,),
                "vae": (["None"] + available_vaes,),
                "lora": (["None"] + available_loras,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "pos": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "a girl"}),
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "worst quality, low quality"}),
                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),
                "lora_stack": ("LORASTACK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID"
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "PDATA")
    RETURN_NAMES = ("context", "model", "preset_save")
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"

    def process_settings(self,
                        node_id,
                        lora_strength,
                        width, height, steps, cfg, sampler, scheduler,
                        unet_Weight_Dtype, guidance, clip_type=None, device="default",
                        vae=None, lora=None, unet_name=None, ckpt_name=None,
                        clip1=None, clip2=None, clip3=None, clip4=None,
                        pos="default", neg="default", over_model=None, over_clip=None, lora_stack=None, preset=[]):

        if preset != "None":
            pass

        parameters_data = [{
            "run_Mode": None,
            "ckpt_name": ckpt_name,
            "clipnum": -2,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": clip3,
            "clip4": clip4,
            "vae": vae,
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": 1,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            "cache_threshold": None,
            "attention": None,
            "cpu_offload": None,
        }]

        # 分辨率修正
        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([1, 4, height // 8, width // 8])
        if latent.shape[1] != 16:
            latent = latent.repeat(1, 16 // 4, 1, 1)

        # 加载模型
        model = None
        clip = over_clip
        vae2 = None

        if over_model is not None:
            model = over_model
        elif ckpt_name != "None" and unet_name == "None":
            model, clip, vae2 = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
        elif unet_name != "None" and ckpt_name == "None":
            if unet_name.endswith(".gguf"):
                from .load_GGUF.nodes import UnetLoaderGGUF2
                if UnetLoaderGGUF2 is None:
                    raise RuntimeError("Please install ComfyUI-GGUF plugin.")
                result = UnetLoaderGGUF2().load_unet(unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None)
                model = result[0]
            else:
                model = UNETLoader().load_unet(unet_name, unet_Weight_Dtype)[0]
        elif ckpt_name != "None" and unet_name != "None":
            raise ValueError("ckpt_name and unet_name cannot be entered at the same time. Please enter only one of them.")

        # 加载 CLIP
        if over_clip is not None:
            clip = over_clip
        elif clip1 == "None" and clip2 == "None" and clip3 == "None" and clip4 == "None":
            pass
        elif clip1 != "None" and clip2 == "None" and clip3 == "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import CLIPLoaderGGUF2
                clip = CLIPLoaderGGUF2().load_clip(clip1, clip_type)[0]
            else:
                clip = CLIPLoader().load_clip(clip1, clip_type, device)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 == "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import DualCLIPLoaderGGUF2
                clip = DualCLIPLoaderGGUF2().load_clip(clip1, clip2, clip_type)[0]
            else:
                clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 != "None" and clip4 == "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import TripleCLIPLoaderGGUF2
                clip = TripleCLIPLoaderGGUF2().load_clip(clip1, clip2, clip3, clip_type="sd3")[0]
            else:
                clip = TripleCLIPLoader().load_clip(clip1, clip2, clip3)[0]
        elif clip1 != "None" and clip2 != "None" and clip3 != "None" and clip4 != "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import QuadrupleCLIPLoaderGGUF2
                clip = QuadrupleCLIPLoaderGGUF2().load_clip(clip1, clip2, clip3, clip4, clip_type="stable_diffusion")[0]
            else:
                clip = QuadrupleCLIPLoader().load_clip(clip1, clip2, clip3, clip4)[0]

        # 应用 LoRA
        if lora_stack is not None:
            model, clip = apply_lora_stack(model, clip, lora_stack)
        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)

        # 编码提示词
        if clip is not None:
            (positive,) = CLIPTextEncode().encode(clip, pos)
            (negative,) = CLIPTextEncode().encode(clip, neg)
        else:
            positive = None
            negative = None

        # 处理 guidance
        if clip1 != "None" and clip2 != "None":
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        # 处理 VAE
        if isinstance(vae, str) and vae != "None":
            vae = VAELoader().load_vae(vae)[0]
        elif vae2 is not None:
            vae = vae2

        # 构造 context
        context = new_context(None, **{
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,

            "clip1": clip1,
            "clip2": clip2,
            "clip3": clip3,
            "clip4": clip4,
            "unet_name": unet_name,
            "ckpt_name": ckpt_name,
            "pos": pos,
            "neg": neg,
            "width": width,
            "height": height,
            "batch": 1,
        })

        return (context, model, parameters_data)

    @classmethod
    def handle_my_message(cls, d):
        """从 Web 接收 preset 文件并发送到前端"""
        preset_path = os.path.join(presets_directory_path, d['message'])
        if not os.path.exists(preset_path):
            logger.error(f"Preset file not found: {preset_path}")
            return
        with open(preset_path, 'r', encoding='utf-8') as f:
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message": preset_data, "node": d['node_id']})



class load_basic:
    @classmethod
    def INPUT_TYPES(cls):


        return {
            "required": {
                "preset": (preset_list, ),
                "ckpt_name": (["None"] + available_ckpt,),  
                "clipnum": ("INT", {"default": 0, "min": -24, "max": 1}),
                "vae": (["None"] + available_vaes, ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                        },

            "hidden": { 
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    TITLE = "load_basic(Deprecated)"

    def process_settings(self, node_id,  lora_strength, clipnum, 
                        width, height, batch, steps, cfg, sampler, scheduler,  device="default", 
                        vae=None, lora=None, ckpt_name=None,  pos="default", neg="default", preset=[]):
        
        parameters_data = []
        parameters_data.append({
            "run_Mode": "basic",
            "ckpt_name": ckpt_name,
            "clipnum": clipnum,
            "unet_name": None,
            "unet_Weight_Dtype": None,
            "clip_type": None,
            "clip1": None,
            "clip2": None,
            "guidance": None,
            "clip3": None,
            "clip4": None,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            "cache_threshold": None,
            "attention": None,
            "cpu_offload": None,            
        })
        

        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  


        model_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(model_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        vae = out[2]
        clip = out[1].clone()
        if clip is not None:
            clip.clip_layer(clipnum)

        if isinstance(vae, str) and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": None, 
            "clip1": None,  
            "clip2": None,  
            "clip3": None, 
            "clip4": None, 
            "unet_name": None, 
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})


class load_FLUX:
    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "preset": (preset_list, ),
                "unet_name": (available_unets, ), 
                "unet_Weight_Dtype": ( ["fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], ),

                "clip_type": (["flux", "sdxl", "sd3", "hunyuan_video"], ),  # Flux
                "clip1": (available_clips,{"default": "clip_l.safetensors"} ),  
                "clip2": (available_clips, {"default": "t5xxl_fp8_e4m3fn.safetensors"} ), 
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),

                "vae": (available_vaes,{"default": "ae.safetensors"} ),
                "lora": (["None"] + available_loras, ),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),

                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                #"neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                        },
            
            
            "hidden": { 
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "PDATA", )
    RETURN_NAMES = ("context","model", "preset_save", )
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"


    def process_settings(self, node_id, lora_strength, 
                        width, height, batch, steps, cfg, sampler, scheduler, unet_Weight_Dtype, guidance, clip_type=None,device="default", 
                        vae=None, lora=None, unet_name=None, clip1=None, 
                        clip2=None, pos="default", neg="", preset=[]):
        
        neg="worst quality, low quality"
        # 非编码后的数据
        parameters_data = []
        parameters_data.append({
            "run_Mode": "FLUX",
            "ckpt_name": None,
            "clipnum": None,
            "unet_name": unet_name,
            "unet_Weight_Dtype": unet_Weight_Dtype,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": None,
            "clip4": None,
            "vae": vae,  
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            "cache_threshold": None,
            "attention": None,
            "cpu_offload": None,
        })
        
        
        model_options = {}
        if unet_Weight_Dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif unet_Weight_Dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif unet_Weight_Dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2


        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:  # Check if the latent has 16 channels
            latent = latent.repeat(1, 16 // 4, 1, 1)  

        if isinstance(vae, str) and vae!= "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        if unet_name!= "None":
            if unet_name.endswith(".gguf"):
                check_UnetLoaderGGUF2_installed()
                loader = UnetLoaderGGUF2()  # 创建实例
                result = loader.load_unet(unet_name, dequant_dtype=None, patch_dtype=None, patch_on_device=None)
                model = result[0]
            else:
                model=UNETLoader().load_unet(unet_name, unet_Weight_Dtype)[0]


        if clip1 != "None" and clip2 != "None" :
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import DualCLIPLoaderGGUF2
                clip = DualCLIPLoaderGGUF2().load_clip(clip1, clip2, clip_type)[0]
            else:
                clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]


        if lora != "None" and lora_strength != 0:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)  

        (positive,) = CLIPTextEncode().encode(clip, pos)
        (negative,) = CLIPTextEncode().encode(clip, neg)
        
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})


        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},      
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,

            "clip1": clip1, 
            "clip2": clip2, 
            "clip3": None, 
            "clip4": None,
            "unet_name": unet_name, 
            "ckpt_name": None, 
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, parameters_data, )

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})




class pre_controlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",),
            },
            "optional": {
                # 第一个ControlNet相关参数
                "image1": ("IMAGE",),
                "controlnet1": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # 第二个ControlNet相关参数
                "image2": ("IMAGE",),
                "controlnet2": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent2": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                # 第三个ControlNet相关参数
                "image3": ("IMAGE",),
                "controlnet3": (['None'] + folder_paths.get_filename_list("controlnet"),),
                "strength3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent3": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent3": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("context","positive", "negative",)
    CATEGORY = "Apt_Preset/chx_tool/controlnet"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, 
                        strength1,strength2, strength3, start_percent1=0.0, end_percent1=1.0,
                         start_percent2=0.0, end_percent2=1.0,
                        start_percent3=0.0, end_percent3=1.0,
                        context=None, 
                        controlnet1=None, controlnet2=None, controlnet3=None,
                        image1=None, image2=None, image3=None, vae=None,):


        positive = context.get("positive", [])
        negative = context.get("negative", [])
        vae = context.get("vae", None)

        # 处理第一个ControlNet
        if controlnet1 != "None" and image1 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet1)
            controlnet1 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet1, image1, 
                strength1, start_percent1, end_percent1, 
                vae, extra_concat=[]
            )
            positive = conditioning[0]
            negative = conditioning[1]

        # 处理第二个ControlNet
        if controlnet2 != "None" and image2 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet2)
            controlnet2 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet2, image2, 
                strength2, start_percent2, end_percent2, 
                vae, extra_concat=[]
            )
            positive = conditioning[0]
            negative = conditioning[1]

        # 处理第三个ControlNet
        if controlnet3 != "None" and image3 is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet3)
            controlnet3 = comfy.controlnet.load_controlnet(controlnet_path)
            conditioning = ControlNetApplyAdvanced().apply_controlnet(
                positive, negative, controlnet3, image3, 
                strength3, start_percent3, end_percent3, 
                vae, extra_concat=[]
            )
            positive = conditioning[0]
            negative = conditioning[1]

        context = new_context(context, positive=positive, negative=negative)
        return (context, positive, negative, )





class sum_lora:
    @classmethod
    def INPUT_TYPES(cls):  
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "lora_01": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_01":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_02": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_02":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_03": (['None'] + folder_paths.get_filename_list("loras"), ),
                "strength_03":("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {

                "model": ("MODEL",),
                "clip": ("CLIP",),
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "style": (["None"] + style_list()[0],{"default": "None"}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("context",  "positive", "negative",)
    CATEGORY = "Apt_Preset/chx_tool"
    FUNCTION = "load_lora"

    def load_lora(self, style, lora_01, strength_01, lora_02, strength_02, lora_03, strength_03, context=None, clip=None, model=None,  pos="", neg=""):

        if model is None:
            model=  context.get("model",None)
        
        if clip is None:
            clip=  context.get("clip",None)

        positive = context.get("positive")
        negative = context.get("negative") 

        if style != "None":
            pos += f"{pos}, {style_list()[1][style_list()[0].index(style)][1]}"
            neg += f"{neg}, {style_list()[1][style_list()[0].index(style)][2]}" if len(style_list()[1][style_list()[0].index(style)]) > 2 else ""

        if lora_01!= "None" and strength_01!= 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_01, strength_01, strength_01)
        if lora_02!= "None" and strength_02!= 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_02, strength_02, strength_02)
        if lora_03!= "None" and strength_03!= 0:
            model, clip = LoraLoader().load_lora(model, clip, lora_03, strength_03, strength_03)

        if pos != None and pos != '':          
            positive, = CLIPTextEncode().encode(clip, pos)
        if neg!= None and pos != '':  
            negative, = CLIPTextEncode().encode(clip, neg)
            
        context = new_context(context, model=model, clip=clip, positive=positive, negative=negative, )
        return (context, positive, negative,)




class sum_editor:
    ratio_sizes, ratio_dict = read_ratios()
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",)},
            "optional": {
                "model": ("MODEL",), 
                "clip": ("CLIP",), 
                "positive": ("CONDITIONING",), 
                "negative": ("CONDITIONING",),
                "vae": ("VAE",), 
                "latent": ("LATENT",), 
                "latent_image": ("IMAGE",),
                "latent_mask": ("MASK",),  # 添加latent_mask输入
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == None"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == None"}),
                "sampler": (['None'] + comfy.samplers.KSampler.SAMPLERS, {"default": "None"}),
                "scheduler": (['None'] + comfy.samplers.KSampler.SCHEDULERS, {"default": "None"}),
                "pos": ("STRING", {"default": "", "multiline": True}),
                "neg": ("STRING", {"default": "", "multiline": False}),
                "guidance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "style": (["None"] + style_list()[0], {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300, }),
                "ratio_selected": (['None', 'customer_WxH'] + s.ratio_sizes, {"default": "None"}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384, "step": 8}),
            }
        }
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE","CLIP",  "IMAGE", "MASK",)  # 添加MASK输出类型
    RETURN_NAMES = ("context", "model","positive", "negative", "latent", "vae","clip", "latent_image", "latent_mask",)  # 添加latent_mask输出名称
    FUNCTION = "text"
    CATEGORY = "Apt_Preset/chx_load"
    def generate(self, ratio_selected, batch_size=1, width=None, height=None):
        if ratio_selected == 'customer_WxH':
            used_width = width
            used_height = height
        else:
            used_width = self.ratio_dict[ratio_selected]["width"]
            used_height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, used_height // 8, used_width // 8])
        return ({"samples": latent}, )
    
    # 在方法参数中添加latent_mask
    def text(self, context=None, model=None, clip=None, positive=None, negative=None, 
            pos="", neg="", latent_image=None, vae=None, latent=None, steps=None, cfg=None, 
            sampler=None, scheduler=None, style=None, batch_size=1, ratio_selected=None, 
            guidance=0, width=512, height=512, latent_mask=None):  # 新增latent_mask参数
        
        # 获取上下文中的宽度和高度，默认为512
        ctx_width = context.get("width") if context else None
        ctx_height = context.get("height") if context else None
        final_width, final_height = ctx_width or 512, ctx_height or 512
        
        # 处理比例选择
        if ratio_selected and ratio_selected != "None":
            try:
                if ratio_selected == 'customer_WxH':
                    final_width, final_height = width, height
                else:
                    final_width = self.ratio_dict[ratio_selected]["width"]
                    final_height = self.ratio_dict[ratio_selected]["height"]
            except KeyError as e:
                print(f"[ERROR] Invalid ratio: {e}")
        
        # 从上下文中获取默认值
        if model is None: 
            model = context.get("model") if context else None
        if clip is None: 
            clip = context.get("clip") if context else None
        if vae is None: 
            vae = context.get("vae") if context else None
        if steps == 0: 
            steps = context.get("steps") if context else None
        if cfg == 0.0: 
            cfg = context.get("cfg") if context else None
        if sampler == "None": 
            sampler = context.get("sampler") if context else None
        if scheduler == "None": 
            scheduler = context.get("scheduler") if context else None
        if guidance == 0.0: 
            guidance = context.get("guidance", 3.5) if context else 3.5

        # 处理latent_mask：如果未提供则从上下文获取
        if latent_mask is None:
            latent_mask = context.get("mask") if context else None

#-------------latent------------------------------------                   
        if latent_image is not None: 
            latent = VAEEncode().encode(vae, latent_image)[0]
            latent = latentrepeat(latent, batch_size)[0]
            # 如果有latent_mask，将其应用到latent
            if latent_mask is not None:
                if isinstance(latent, dict) and "samples" in latent:
                    latent_copy = {"samples": latent["samples"].clone()}
                    latent = self.set_latent_mask2(latent_copy, latent_mask)
                else:
                    latent = self.set_latent_mask2(latent, latent_mask)
        elif latent is not None:
            latent = latentrepeat(latent, batch_size)[0]    
            # 应用latent_mask
            if latent_mask is not None:
                if isinstance(latent, dict) and "samples" in latent:
                    latent_copy = {"samples": latent["samples"].clone()}
                    latent = self.set_latent_mask2(latent_copy, latent_mask)
                else:
                    latent = self.set_latent_mask2(latent, latent_mask)
        elif ratio_selected != "None":
            latent = self.generate(ratio_selected, batch_size, final_width, final_height)[0]      
        else:
            latent = context.get("latent") if context else None
            if latent is not None:
                latent = latentrepeat(latent, batch_size)[0]
                # 应用latent_mask
                if latent_mask is not None:
                    if isinstance(latent, dict) and "samples" in latent:
                        latent_copy = {"samples": latent["samples"].clone()}
                        latent = self.set_latent_mask2(latent_copy, latent_mask)
                    else:
                        latent = self.set_latent_mask2(latent, latent_mask)

#--------------条件--------------------------------------------------------------------------------
        pos, neg = add_style_to_subject(style, pos, neg)
        if positive is not None: 
            pass
        elif pos and pos != "":
            positive, = CLIPTextEncode().encode(clip, pos)
        else: 
            positive = context.get("positive") if context else None     
        if negative is not None: 
            pass           
        elif neg and neg != "":
            negative, = CLIPTextEncode().encode(clip, neg)
        else: 
            negative = context.get("negative") if context else None  
        if positive is not None:            
           positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
       
        # 将latent_mask存入上下文
        context = new_context(
            context, 
            model=model, 
            latent=latent, 
            clip=clip, 
            vae=vae, 
            positive=positive, 
            negative=negative, 
            images=latent_image, 
            mask=latent_mask,  # 新增mask存入上下文
            steps=steps, 
            cfg=cfg, 
            sampler=sampler, 
            scheduler=scheduler, 
            guidance=guidance, 
            pos=pos, 
            neg=neg, 
            width=final_width, 
            height=final_height, 
            batch=batch_size
        )
    
        # 返回值中添加latent_mask
        return (context, model, positive, negative, latent, vae, clip, latent_image, latent_mask,)
    
    # 复用Data_basic中的set_latent_mask2方法处理mask
    def set_latent_mask2(self, latent, mask):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent 必须是包含 'samples' 键的字典")
        newlatent = {
            "samples": latent["samples"].clone()  # 只对张量部分调用 clone()
        }
        
        if mask is not None:
            newlatent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        
        return newlatent







class sum_latent:
    
    ratio_sizes, ratio_dict = read_ratios()
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {  "context": ("RUN_CONTEXT",),   },         
            "optional":  {
                "latent": ("LATENT", ),
                "pixels": ("IMAGE", ),
                "mask": ("MASK", ),
                "diff_difusion": ("BOOLEAN", {"default": True}), 
                "smoothness":("INT", {"default": 0,  "min":0, "max": 150, "step": 1,"display": "slider"}),
                "ratio_selected": (['None','customer_WxH'] + s.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300, }),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}), 
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),   
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","LATENT","MASK"  )
    RETURN_NAMES = ("context","latent","mask"  )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_tool"

    def generate(self, ratio_selected, batch_size=1, width=None, height=None):
        # 当选择customer_WxH模式时，使用输入的width和height
        if ratio_selected == 'customer_WxH':
            # 确保width和height有效
            if width is None or height is None:
                raise ValueError("When 'customer_WxH' is selected, 'width' and 'height' must be provided.")
            used_width = width
            used_height = height
        else:
            # 其他模式从ratio_dict获取宽高
            used_width = self.ratio_dict[ratio_selected]["width"]
            used_height = self.ratio_dict[ratio_selected]["height"]
        
        latent = torch.zeros([batch_size, 4, used_height // 8, used_width // 8])
        return ({"samples": latent}, )

    def process(self, ratio_selected, smoothness=1, batch_size=1, context=None, latent=None, pixels=None, mask=None, 
                diff_difusion=True, width=None, height=None):
        noise_mask = True
        model = context.get("model")
        if diff_difusion:
            model = DifferentialDiffusion().apply(model)[0]

        if latent is not None and pixels is not None:
            raise ValueError("Only one of 'latent', 'pixels' should be provided.")
        if latent is not None:
            latent = latentrepeat(latent, batch_size)[0]
            context = new_context(context, model=model,latent=latent)
            return (context, latent, None)
        if latent is None and pixels is None and ratio_selected == "None":
            latent = context.get("latent", None)
            latent = latentrepeat(latent, batch_size)[0]
            context = new_context(context, model=model,latent=latent)
            return (context, latent, None)

        vae = context.get("vae")
        positive = context.get("positive", None)
        negative = context.get("negative", None)

        if ratio_selected != "None":
            # 调用generate时传入width和height，用于customer_WxH模式
            latent = self.generate(ratio_selected, batch_size, width, height)[0]

        if pixels is not None:
            if mask is not None:
                if torch.all(mask == 0):
                    latent = VAEEncode().encode(vae, pixels)[0]
                else:
                    mask = tensor2pil(mask)
                    if not isinstance(mask, Image.Image):
                        raise TypeError("mask is not a valid PIL Image object")
                    feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
                    mask = pil2tensor(feathered_image)
                    
                    positive, negative, latent = InpaintModelConditioning().encode(positive, negative, pixels, vae, mask, noise_mask)
            else:
                latent = VAEEncode().encode(vae, pixels)[0]
            latent = latentrepeat(latent, batch_size)[0]
        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent)

        return (context, latent, mask)
    


class sum_create_chx:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae": (["None"] + available_vaes, ),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "batch": ("INT", {"default": 1, "min": 1, "max": 999999}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pos": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "a girl"}), 
                "neg": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": " worst quality, low quality"}),
            },
            
            "optional": { 
                "model": ("MODEL",),
                "clip": ("CLIP",),  
                "over_vae": ("VAE",),
                "over_positive": ("CONDITIONING",),
                "over_negative": ("CONDITIONING",),
                "over_latent": ("LATENT",),
                "lora_stack": ("LORASTACK",),
                "data":(ANY_TYPE,),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "ANY_TYPE")
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent", "vae", "clip", "data")
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"

    def process_settings(self, 
                        width, height, batch, steps, cfg, sampler, scheduler, data=None, guidance=3.5, lora_stack=None,over_latent=None,
                        vae=None, over_vae=None, clip=None, model=None, over_positive=None, over_negative=None, pos="default", neg="default"):

        # 分辨率修正
        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([1, 4, height // 8, width // 8])
        if latent.shape[1] != 16:
            latent = latent.repeat(1, 16 // 4, 1, 1)

        if over_latent is not None:
            latent = over_latent
        # 处理VAE
        if over_vae is not None:
            vae = over_vae
        elif over_vae is None and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        # 初始化条件为None
        positive = None
        negative = None
        
        # 处理LoRA和文本编码
        if clip is not None:
            # 如果提供了clip，可以处理文本条件
            # 只有当model和clip都提供时，才应用LoRA
            if model is not None and lora_stack is not None:
                model, clip = apply_lora_stack(model, clip, lora_stack)
                
            # 处理条件
            positive, = CLIPTextEncode().encode(clip, pos)
            negative, = CLIPTextEncode().encode(clip, neg)



        # 覆盖条件（如果提供）
        if over_positive:
            positive = over_positive
            if negative is None:
                negative = condi_zero_out(over_positive)[0]
                
        if over_negative:
            negative = over_negative


        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        # 确保latent格式正确
        latent_dict = {"samples": latent}

        # 创建上下文
        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": latent_dict,  # 使用正确格式的latent
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,
            "clip1": None,
            "clip2": None,
            "clip3": None,
            "clip4": None,
            "unet_name": None,
            "ckpt_name": None,
            "pos": pos, 
            "neg": neg, 
            "width": width,
            "height": height,
            "batch": batch,
            "data": data,
        }

        return (context, model, positive, negative, latent_dict, vae, clip, data)  # 返回正确格式的latent




class chx_input_data:   
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"model": ("MODEL",), },

            "optional": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","CONDITIONING","LATENT","VAE","CLIP",)
    RETURN_NAMES = ("context", "model","positive","negative","latent","vae","clip",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_load"

    def sample(self, model=None,positive=None,negative=None,latent=None,vae =None,clip =None,):

        if clip is None and positive is None :
            raise ValueError("clip or positive is required")
        if clip is None and negative is None and positive is not None:
            negative=self.zero_out(positive)[0]

        if vae is None:
            raise ValueError("vae is required")


        if clip is not None:
            positive, = CLIPTextEncode().encode(clip, "a girl")
            negative, = CLIPTextEncode().encode(clip, "worst quality, low quality")

    
        if latent is None:
            latent_tensor = torch.zeros([1, 4, 64, 64])
            latent = {"samples": latent_tensor}

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": latent,  
            "vae": vae,
            "clip": clip,
            "steps": 20,
            "cfg": 8,
            "sampler": "euler",
            "scheduler": "normal",
            "width": 512,
            "height": 512,
            "batch": 1,
        }
        return (context, model, positive, negative, latent, vae, clip,)

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return (c, )




#endregion---------加载器-----------------------------------------------------------------------------------#




#region-----------采样器---------------------------------------------------------------------------------------#

class basic_Ksampler_full:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 10000,"tooltip": "  -1  == None"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == None"}),
                "sampler": (['None'] + comfy.samplers.KSampler.SAMPLERS, {"default": "None"}),  
                "scheduler": (['None'] + comfy.samplers.KSampler.SCHEDULERS, {"default": "None"}), 
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "context": ("RUN_CONTEXT",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "latent_image": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }
        
    OUTPUT_NODE = True
    RETURN_TYPES = ("RUN_CONTEXT","IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT","VAE","CLIP", )
    RETURN_NAMES = ("context", "image", "model","positive", "negative",  "latent", "vae", "clip", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample"


    def sample(self,  seed, denoise, context=None, clip=None, model=None,vae=None, positive=None, negative=None, latent=None,steps=None, cfg=None, sampler=None,
                scheduler=None, latent_image=None, prompt=None, image_output=None, extra_pnginfo=None, ):
        if steps == -1:
            steps = context.get("steps")
        if cfg == 0.0:
            cfg = context.get("cfg")
        if sampler == "None":
            sampler = context.get("sampler")
        if scheduler == "None":
            scheduler = context.get("scheduler")

        if positive is None:
            positive = context.get("positive" )
        if negative is None:
            negative = context.get("negative" )
        if vae is None:
            vae= context.get("vae")
        if model is None:
            model= context.get("model")
        if clip is None:
            clip= context.get("clip")

        if latent_image is not None:
            latent = encode(vae, latent_image)[0]
        if latent is None:
            latent = context.get("latent",None)

        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]
        

        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=output_image, vae=vae,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )

        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}
            
        return {"ui": {"images": results},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}


class basic_Ksampler_mid:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            
            "optional": {
                "context": ("RUN_CONTEXT",),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},

        }
        
        
    OUTPUT_NODE = True
    RETURN_TYPES = ("RUN_CONTEXT","IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT","VAE","CLIP", )
    RETURN_NAMES = ("context", "image", "model","positive", "negative",  "latent", "vae", "clip", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"


    def sample(self,  seed, denoise, context=None, clip=None, model=None,vae=None, positive=None, negative=None, latent=None, image=None, prompt=None, image_output=None, extra_pnginfo=None, ):


        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")


        if positive is None:
            positive = context.get("positive" )
        if negative is None:
            negative = context.get("negative" )
        if vae is None:
            vae= context.get("vae")
        if model is None:
            model= context.get("model")
        if clip is None:
            clip= context.get("clip")


#------------------------latent四种处理方式-------------------------

        if image is not None:
            latent = encode(vae, image)[0]
        if latent is None:
            latent = context.get("latent",None)
#----------------------------------------------------------------  


        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]
        
        
        if image_output == "None":
            context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=None, vae=vae,steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )
            return(context, None, model, positive, negative, latent, vae, clip)


        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=output_image, vae=vae,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )

        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}
            
        return {"ui": {"images": results},
                "result": (context, output_image, model, positive, negative, latent, vae, clip)}


class basic_Ksampler_simple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": ([ "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                                
            },
            
            "optional": {
                "image": ("IMAGE",),
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }

    RETURN_TYPES = ("RUN_CONTEXT",  "IMAGE", )
    RETURN_NAMES = ("context",  "image", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample"


    def run(self,context, seed, denoise, image=None,  prompt=None, image_output=None, extra_pnginfo=None,):
        vae = context.get("vae",None)
        steps = context.get("steps",20)
        cfg = context.get("cfg",8)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)
        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)

        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = context.get("latent",None)


        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]


        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, latent=latent, images=output_image,  )
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}


class basic_Ksampler_custom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {

            },
            
            "optional":
                    {
                    "context": ("RUN_CONTEXT",),
                    "model": ("MODEL", ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent": ("LATENT", ),
                    "image": ("IMAGE", ),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "image_output": (["None","Hide", "Preview", "Save", "Hide/Save"], {"default": "None", "tooltip": "  output_image will take up CPU resources "}),
                    
                    },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            
                }
    OUTPUT_NODE = True
    RETURN_TYPES = ("RUN_CONTEXT","IMAGE", "MODEL","CONDITIONING","CONDITIONING","LATENT", "VAE", )
    RETURN_NAMES = ("context","image", "model","positive","negative","latent", "vae",  )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def sample(self, context=None,model=None,image=None,positive=None, negative=None, latent=None, noise=None, sampler=None, guider=None, sigmas=None, seed=1, denoise=1, prompt=None, image_output=None, extra_pnginfo=None,):
        
        vae=context.get("vae")
        steps = context.get("steps")
        cfg = context.get("cfg")
        scheduler = context.get("scheduler")



        if model is None:
            model=context.get("model")
            
        if positive is None:
            positive = context.get("positive",None)
            
        if negative is None:    
            negative = context.get("negative",None)
            
        if sampler is None:
            sampler_name = context.get("sampler",None)
            sampler = KSamplerSelect().get_sampler(sampler_name)[0]  
            
        if noise is None:
            noise = RandomNoise().get_noise(seed)[0] 

        if guider is None:
            guider = BasicGuider().get_guider(model, positive)[0] 

        if sigmas is None:
            sigmas = BasicScheduler().get_sigmas(model, scheduler, steps, denoise)[0]


#------------------------latent三种处理方式-------------------------
        if image is not None:
            latent = encode(vae, image)[0]
        elif latent is None:
            latent = latent
        else:
            latent = context.get("latent",None)
#----------------------------------------------------------------           
               
        out= SamplerCustomAdvanced().sample( noise, guider, sampler, sigmas, latent)
        latent= out[0]
        
        if image_output == "None":
            context = new_context(context, images=None, latent=latent, model=model, positive=positive, negative=negative,  )
            return(context, model, positive, negative, latent, vae, None, ) 
            
        output_image = VAEDecode().decode(vae, latent)[0]  
        context = new_context(context, images=output_image, latent=latent, model=model, positive=positive, negative=negative,  )   
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context,output_image, model, positive, negative, latent, vae, )}
            
        return {"ui": {"images": results},
                "result": (context,output_image, model, positive, negative, latent, vae,)}


class basic_Ksampler_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "context": ("RUN_CONTEXT",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 10000,"tooltip": "  0  == None"}),
                    "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "end_at_step": ("INT", {"default": 1000, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Hide"}),
                    },
                "optional": {
                    "latent": ("LATENT", ),
                    },
                
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
                
                }

    RETURN_TYPES = ("RUN_CONTEXT","IMAGE")
    RETURN_NAMES = ("context ", "image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def sample(self, context, add_noise, steps, noise_seed, start_at_step,end_at_step, return_with_leftover_noise, denoise=1.0, 
                pos="", neg="", latent=None, prompt=None, image_output=None, extra_pnginfo=None, ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        model = context.get("model", None)  
        vae = context.get("vae", None)
        cfg = context.get("cfg", 8.0)  
        sampler_name = context.get("sampler", None) 
        scheduler = context.get("scheduler", None) 
        latent = latent or context.get("latent", None)


        positive = context.get("positive", None)
        negative = context.get("negative", None)

        """guidance = context.get("guidance",None)
        if guidance is None:
            guidance = 3.5  # 默认值
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})"""


        latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, 
                                positive, negative, latent, denoise=denoise, 
                                disable_noise=disable_noise, 
                                start_step=start_at_step, 
                                last_step=end_at_step,
                                force_full_denoise=force_full_denoise)[0]      
        
        
        if image_output =="None":
            context = new_context(context, latent=latent,images=None)

            return (context, None,)
        
        output_image = VAEDecode().decode(vae, latent)[0]
        context = new_context(context, latent=latent, images=output_image)  
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)

        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}


class chx_Ksampler_mix:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "context": ("RUN_CONTEXT",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 3, "min": 0, "max": 0xffffffffffffffff}),
                    
                    "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "mid_denoise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,"tooltip": "percentage of denoising at mid step"}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    
                    },
                    
                "optional": {
                    "latent": ("LATENT", ),
                    "pos1": ("STRING", {"default": "a dog ", "multiline": True}),
                    "pos2": ("STRING", {"default": "a girl ", "multiline": True}),
                    "neg": ("STRING", {"default": "blur, blurry,", "multiline": False}),
                    "mix_method": (["None","combine", "concat", "average"],),

                    },
                
                }

    RETURN_TYPES = ("RUN_CONTEXT","IMAGE")
    RETURN_NAMES = ("context ", "image")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def sample(self, context, add_noise, noise_seed,  start_step,  steps,  return_with_leftover_noise, denoise=1.0, mid_denoise =0.3,
                pos1="", pos2="", neg="", latent=None, mix_method=None ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        model = context.get("model", None)  
        vae = context.get("vae", None)
        cfg = context.get("cfg", 8.0)  
        sampler_name = context.get("sampler", None) 
        scheduler = context.get("scheduler", None) 
        clip = context.get("clip", None)
        latent = latent or context.get("latent", None)
        
        mix_step = math.ceil(steps * mid_denoise) 

        positive1 = CLIPTextEncode().encode(clip, pos1)[0]
        positive2 = CLIPTextEncode().encode(clip, pos2)[0]
        negative = CLIPTextEncode().encode(clip, neg)[0]

        latent1 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive1, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=mix_step, force_full_denoise=force_full_denoise)[0]


        if mix_method == "combine":
            if isinstance(positive1, torch.Tensor) and isinstance(positive2, torch.Tensor):
                if positive1.shape != positive2.shape:
                    positive2 = torch.nn.functional.interpolate(positive2, size=positive1.shape[2:])
                positive2 = positive2 + positive1

        elif mix_method == "concat":
            positive2 = ConditioningConcat().concat(positive1, positive2)[0]

        elif mix_method == "average":
            positive2 = ConditioningAverage().addWeighted(positive1, positive2, 0.5,)[0]
        
        
        latent2 = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive2, negative, latent1, denoise=denoise, disable_noise=disable_noise, start_step=mix_step, last_step=steps, force_full_denoise=force_full_denoise)[0]

        output_image = VAEDecode().decode(vae, latent2)[0]
        
        context = new_context(context, latent=latent2, images=output_image, positive=positive2,negative=negative,)  
        
        return (context ,output_image )





class chx_Ksampler_highAndLow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {
                    "context": ("RUN_CONTEXT",),
                    "add_noise": (["enable", "disable"], ),
                    "noise_seed": ("INT", {"default": 3, "min": 0, "max": 0xffffffffffffffff}),                    
                    "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "mid_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                    "steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
                    "return_with_leftover_noise": (["disable", "enable"], ),
                    
                    },
                    
                "optional": {
                    "chx2": ("RUN_CONTEXT",),
                    "model2": ("MODEL",),
                    "positive2": ("CONDITIONING",),     
                    "image": ("IMAGE",),           
                    },               
                }


    RETURN_TYPES = ("RUN_CONTEXT","IMAGE")
    RETURN_NAMES = ("context ", "image")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def sample(self, context, add_noise, noise_seed,  start_step,  steps,  return_with_leftover_noise, denoise=1.0, mid_step =10,model2=None,positive2=None,
              latent=None,chx2=None,image=None ):
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False
        disable_noise = False
        if add_noise == "disable":
            disable_noise = True
 
        model = context.get("model", None)  
        vae = context.get("vae", None)
        cfg = context.get("cfg", 8.0)  
        sampler_name = context.get("sampler", None) 
        scheduler = context.get("scheduler", None) 
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        
        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]
        else:
            latent = context.get("latent", None)
         
        latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, 
                                  disable_noise=disable_noise, start_step=start_step, last_step=mid_step, force_full_denoise=force_full_denoise)[0]

#-------------------------第二次采样-------------------------
        if chx2 is not None:
            context = Data_chx_Merge().merge(context,chx2)
            model2 = context.get("model", None)  
            vae = context.get("vae", None)
            cfg = context.get("cfg", 8.0)  
            sampler_name = context.get("sampler", None) 
            scheduler = context.get("scheduler", None) 
            positive2 = context.get("positive", None)
            negative = context.get("negative", None)
        if model2 is not None:
            model = model2
        if positive2 is not None:
            positive = positive2

        latent = common_ksampler(model, noise_seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=denoise, 
                                  disable_noise=disable_noise, start_step=mid_step, last_step=steps, force_full_denoise=force_full_denoise)[0]


        output_image = VAEDecode().decode(vae, latent)[0]
        
        context = new_context(context,model=model, latent=latent, images=output_image, positive=positive,negative=negative,)  
        
        return (context ,output_image )


class texture_Ksampler:
    def __init__(self):
        pass

    @classmethod

    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "tileX": ("INT", {"default": 1, "min": 0, "max": 2}),
                "tileY": ("INT", {"default": 1, "min": 0, "max": 2}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/imgEffect"


    def apply_asymmetric_tiling(self, model, tileX, tileY):
        for layer in [layer for layer in model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            print(layer.paddingX, layer.paddingY)

    def __hijackConv2DMethods(self, model, tileX: bool, tileY: bool):
        for layer in [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer.padding_modeX = 'circular' if tileX else 'constant'
            layer.padding_modeY = 'circular' if tileY else 'constant'
            layer.paddingX = (layer._reversed_padding_repeated_twice[0], layer._reversed_padding_repeated_twice[1], 0, 0)
            layer.paddingY = (0, 0, layer._reversed_padding_repeated_twice[2], layer._reversed_padding_repeated_twice[3])
            
            def make_bound_method(method, current_layer):
                def bound_method(self, *args, **kwargs):  # Add 'self' here
                    return method(current_layer, *args, **kwargs)
                return bound_method
                
            bound_method = make_bound_method(self.__replacementConv2DConvForward, layer)
            layer._conv_forward = bound_method.__get__(layer, type(layer))

    def __replacementConv2DConvForward(self, layer, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        working = torch.nn.functional.pad(input, layer.paddingX, mode=layer.padding_modeX)
        working = torch.nn.functional.pad(working, layer.paddingY, mode=layer.padding_modeY)
        return torch.nn.functional.conv2d(working, weight, bias, layer.stride, (0, 0), layer.dilation, layer.groups)

    def __restoreConv2DMethods(self, model):
        for layer in [l for l in model.modules() if isinstance(l, torch.nn.Conv2d)]:
            layer._conv_forward = torch.nn.Conv2d._conv_forward.__get__(layer, torch.nn.Conv2d)
    
    
    def sample(self, context,  seed, tileX, tileY,  denoise=1.0):

        
        vae = context.get("vae")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler_name = context.get("sampler")
        scheduler = context.get("scheduler")

        positive = context.get("positive")
        negative = context.get("negative")
        model = context.get("model")
        latent_image = context.get("latent") 
        
        self.__hijackConv2DMethods(model.model, tileX == 1, tileY == 1)
        result = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise=denoise)[0]
        self.__restoreConv2DMethods(model.model)

        
        for layer in [layer for layer in vae.first_stage_model.modules() if isinstance(layer, torch.nn.Conv2d)]:
            layer.padding_mode = 'circular'

        out_image = vae.decode(result["samples"])

        return (out_image,)  


class chx_Ksampler_refine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "upscale_model": (["None"] +folder_paths.get_filename_list("upscale_models"), {"default": "1xDeJPG_OmniSR.pth"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "Add_img_scale": ("FLOAT", {"default": 2, "min": 1, "max": 16.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),                
                
            },
            
            "optional": {
                "image": ("IMAGE",),
                "lowCpu": ("VAEDecodeTiled",),  
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }


    RETURN_TYPES = ("RUN_CONTEXT",  "IMAGE", )
    RETURN_NAMES = ("context",  "image", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"
    def run(self,context, seed, denoise, upscale_model,upscale_method, image=None,  prompt=None, image_output=None, extra_pnginfo=None,Add_img_scale=1, lowCpu=None):

        vae = context.get("vae",None)
        steps = context.get("steps",None)
        cfg = context.get("cfg",None)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)

        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None) 

        if image is None:
            image = context.get("images",None)

        if upscale_model != "None":         
            up_model = load_upscale_model(upscale_model)
            image = upscale_with_model(up_model, image )

        if Add_img_scale != 1:
            image = image_upscale(image, upscale_method, Add_img_scale)[0]

        if image is not None:
            latent = encode(vae, image)[0]


        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]


        if lowCpu is not None:
            (tile_size, overlap, temporal_size, temporal_overlap)=lowCpu
            output_image = VAEDecodeTiled(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)[0]
        else:
            output_image = decode(vae, latent)[0]


        if image_output == "None":
            context = new_context(context, latent=latent, images=None, )
            return(context, None)

        
        context = new_context(context, latent=latent, images=output_image,  )
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}




class chx_Ksampler_dual_paint:    #双区采样 ksampler
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "smoothness":("INT", {"default": 0,  "min":0, "max": 150, "step": 1,"display": "slider"}),
                "mask_area_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_area_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine": ("BOOLEAN", {"default": True}),
                "refine_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE",)
    RETURN_NAMES = ("context", "image",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def execute(self,context, image, mask, smoothness, mask_area_denoise, image_area_denoise,refine,refine_denoise, seed,):
        
        vae = context.get("vae",None)
        steps = context.get("steps",None)
        cfg = context.get("cfg",None)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)

        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None) 


        phase_steps = math.ceil(steps / 2)
        device = model.model.device if hasattr(model, 'model') else model.device
        
        vae_encoder = VAEEncode()
        latent_dict = vae_encoder.encode(vae, image)[0]
        input_latent = latent_dict["samples"].to(device)


        if mask is not None :
            mask=tensor2pil(mask)
            if not isinstance(mask, Image.Image):
                raise TypeError("mask is not a valid PIL Image object")
            
            feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
            mask=pil2tensor(feathered_image)


        
        mask = 1-mask.float().to(device)
        
        mask_resized = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
            size=(input_latent.shape[2], input_latent.shape[3]), 
            mode='bilinear'
        )
        
        mask_strength = mask_resized * (image_area_denoise - mask_area_denoise) + mask_area_denoise
        
        noise_mask = SetLatentNoiseMask()
        latent_with_mask = noise_mask.set_mask({"samples": input_latent}, mask_strength)[0]
        
        advanced_sampler = KSamplerAdvanced()
        
        result = advanced_sampler.sample(
            model=model,
            add_noise=0.00,
            noise_seed=seed,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_with_mask,
            start_at_step=0,
            end_at_step=phase_steps,
            return_with_leftover_noise=False
        )[0]
        samples = result["samples"].to(device)
        binary_mask = (mask_resized >= 0.5).float()
        phase2_mask = binary_mask * 1.0 + (1 - binary_mask) * mask_area_denoise
        
        latent_phase2 = noise_mask.set_mask(
            {"samples": samples},
            phase2_mask
        )[0]
    
        result = advanced_sampler.sample(
            model=model,
            add_noise=0.00,
            noise_seed=seed + 1,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_phase2,
            start_at_step=phase_steps,
            end_at_step=steps,
            return_with_leftover_noise=False
        )[0]
        samples = result["samples"].to(device)
        
        if refine:

            sampler = KSampler()
            result = sampler.sample(
                model,
                seed + 1,
                steps,
                cfg,
                sampler,
                scheduler,
                positive,
                negative,
                {"samples": samples},
                refine_denoise
            )[0]
            samples = result["samples"].to(device)
        
        latent= {"samples": samples}
        images = VAEDecode().decode(vae, latent)[0]
        
        context = new_context(context,  latent=latent, images=images, )

        return (context,images,)



class basic_Ksampler_low_gpu:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."}),


                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                                
            },
            
            "optional": {
                "image": ("IMAGE",),
            },
            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }

    RETURN_TYPES = ("RUN_CONTEXT",  "IMAGE", )
    RETURN_NAMES = ("context",  "image", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"


    def run(self,context, seed, denoise, image=None,  prompt=None, image_output=None, extra_pnginfo=None,sampler=None,tile_size=512, overlap=64, temporal_size=64, temporal_overlap=8):
        vae = context.get("vae",None)
        steps = context.get("steps",20)
        cfg = context.get("cfg",8)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)
        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None)

        if image is not None:
            latent = VAEEncode().encode(vae, image)[0]

        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive, 
                negative, 
                latent, 
                denoise=denoise
                )[0]

        if image_output == "None":
            context = new_context(context, latent=latent, images=None,  )
            return(context, None)


        output_image = VAEDecodeTiled(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)[0]

        context = new_context(context, latent=latent, images=output_image,  )
        
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image,)}
            
        return {"ui": {"images": results},
                "result": (context, output_image,)}





class chx_ksampler_tile:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "context": ("RUN_CONTEXT",),
                    "model_name": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                    "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise_image": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                    "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                    },
                "optional": {"image_optional": ("IMAGE",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            }
            
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                    upscale_by, model_name, seed, vae, denoise_image,
                    steps, cfg, sampler_name, scheduler):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        start_step = int(steps - (steps * denoise_image))
        sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                start_step=start_step, last_step=steps, force_full_denoise=False)[0]
        pixels = vaedecoder.decode(vae, sample1)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, "bicubic", upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def run(self, seed, model_name, upscale_by=2.0, tile_size=512,prompt=None, image_output=None, extra_pnginfo=None,
            upscale_method='normal', denoise_image=1.0, image_optional=None, context=None):
        if image_output == "None":
            output_image = context.get("images",None)
            return (output_image,)
        
        vae = context.get("vae", None)
        steps = context.get("steps", 8)
        cfg = context.get("cfg", 7)
        sampler_name = context.get("sampler", "dpmpp_sde_gpu")
        scheduler = context.get("scheduler", "karras")
        positive_cond_base = context.get("positive", "")
        negative_cond_base = context.get("negative", "")
        base_model = context.get("model", None)
        samples = context.get("latent", None)

        tile_denoise = denoise_image

        if image_optional is not None:
            vaeencoder = VAEEncode()
            samples = vaeencoder.encode(vae, image_optional)[0]
        
        img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                                            upscale_by, model_name, seed, vae, denoise_image,
                                                            steps, cfg, sampler_name, scheduler)
        img= tensor2pil(img)

        tiled_image = run_tiler_for_steps(img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps, tile_denoise, tile_size)

        results = easySave(tiled_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": ( tiled_image,)}
            
        return {"ui": {"images": results},
                "result": ( tiled_image,)}




#region------------------------ksampler-tile------------------------

def split_image(img, tile_size=1024):
    if isinstance(img, list):
        print("Warning: img is a list, selecting the first element.")
        img = img[0]
    if not hasattr(img, 'width') or not hasattr(img, 'height'):
        raise TypeError("The input 'img' must be an image object (e.g., PIL Image or torch tensor).")

    tile_width, tile_height = tile_size, tile_size
    width, height = img.width, img.height

    num_tiles_x = ceil(width / tile_width)
    num_tiles_y = ceil(height / tile_height)

    if num_tiles_x < 2:
        num_tiles_x = 2
    if num_tiles_y < 2:
        num_tiles_y = 2

    if width % tile_width == 0:
        num_tiles_x += 1
    if height % tile_height == 0:
        num_tiles_y += 1

    if num_tiles_x > 1:
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    else:
        overlap_x = 0
    if num_tiles_y > 1:
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)
    else:
        overlap_y = 0

    if overlap_x < 256:
        num_tiles_x += 1
        overlap_x = (num_tiles_x * tile_width - width) / (num_tiles_x - 1)
    if overlap_y < 256:
        num_tiles_y += 1
        overlap_y = (num_tiles_y * tile_height - height) / (num_tiles_y - 1)

    tiles = []

    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            x_start = j * tile_width - j * overlap_x
            y_start = i * tile_height - i * overlap_y

            x_start = round(x_start)
            y_start = round(y_start)

            tile_img = img.crop((x_start, y_start, x_start + tile_width, y_start + tile_height))
            tiles.append(((x_start, y_start, x_start + tile_width, y_start + tile_height), tile_img))

    return tiles

def stitch_images(upscaled_size, tiles):
    if isinstance(upscaled_size, tuple):
        width, height = upscaled_size
    elif hasattr(upscaled_size, 'size'):
        width, height = upscaled_size.size
    elif hasattr(upscaled_size, 'shape'):
        _, height, width = upscaled_size.shape
    else:
        raise TypeError("upscaled_size should be a tuple, PIL.Image, or torch.Tensor.")
    
    result = torch.zeros((3, height, width))
    sorted_tiles = sorted(tiles, key=lambda x: (x[0][1], x[0][0]))
    current_row_upper = None

    for (left, upper, right, lower), tile in sorted_tiles:
        if current_row_upper != upper:
            current_row_upper = upper
            first_tile_in_row = True
        else:
            first_tile_in_row = False

        tile_width = right - left
        tile_height = lower - upper
        feather = tile_width // 8

        mask = torch.ones(tile.shape[0], tile.shape[1], tile.shape[2])

        if not first_tile_in_row:
            for t in range(feather):
                mask[:, :, t:t+1] *= (1.0 / feather) * (t + 1)

        if upper != 0:
            for t in range(feather):
                mask[:, t:t+1, :] *= (1.0 / feather) * (t + 1)

        tile = tile.squeeze(0).squeeze(0)
        tile_to_add = tile.permute(2, 0, 1)
        combined_area = tile_to_add * mask.unsqueeze(0) + result[:, upper:lower, left:right] * (1.0 - mask.unsqueeze(0))
        result[:, upper:lower, left:right] = combined_area

    tensor_expanded = result.unsqueeze(0)
    tensor_final = tensor_expanded.permute(0, 2, 3, 1)
    return tensor_final

def ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, start_step=11, end_step=20):
    vaedecoder = VAEDecode()
    vaeencoder = VAEEncode()
    tile = pil2tensor(tile)
    encoded_tile = vaeencoder.encode(vae, tile)[0]
    tile = common_ksampler(base_model, seed, end_step, cfg, sampler_name, scheduler,
                        positive_cond_base, negative_cond_base, encoded_tile,
                        start_step=start_step, force_full_denoise=True)[0]
    tile = vaedecoder.decode(vae, tile)[0]
    return tile

def run_tiler_for_steps(enlarged_img, base_model, vae, seed, cfg, sampler_name, scheduler,
                        positive_cond_base, negative_cond_base, steps=20, denoise=0.25, tile_size=1024):
    if isinstance(enlarged_img, list):
        print("Warning: enlarged_img is a list, selecting the first element.")
        enlarged_img = enlarged_img[0]
    if not hasattr(enlarged_img, 'size') and not hasattr(enlarged_img, 'shape'):
        raise TypeError("enlarged_img should be a valid image object (e.g., PIL.Image or torch tensor).")

    tiles = split_image(enlarged_img, tile_size=tile_size)

    start_step = int(steps - (steps * denoise))
    end_step = steps
    resampled_tiles = [(coords, ai_upscale_adv(tile, base_model, vae, seed, cfg, sampler_name, scheduler,
                                            positive_cond_base, negative_cond_base, start_step, end_step)) for coords, tile in tiles]

    result = stitch_images(enlarged_img.size, resampled_tiles)

    return result




class chx_ksampler_tile:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "context": ("RUN_CONTEXT",),
                    "model_name": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                    "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise_image": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                    "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                    },
                "optional": {"image_optional": ("IMAGE",),},
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            }
            
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = "Apt_Preset/chx_ksample/😺backup"

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                    upscale_by, model_name, seed, vae, denoise_image,
                    steps, cfg, sampler_name, scheduler):
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        start_step = int(steps - (steps * denoise_image))
        sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                start_step=start_step, last_step=steps, force_full_denoise=False)[0]
        pixels = vaedecoder.decode(vae, sample1)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, "bicubic", upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def run(self, seed, model_name, upscale_by=2.0, tile_size=512,prompt=None, image_output=None, extra_pnginfo=None,
            upscale_method='normal', denoise_image=1.0, image_optional=None, context=None):
        if image_output == "None":
            output_image = context.get("images",None)
            return (output_image,)
        
        vae = context.get("vae", None)
        steps = context.get("steps", 8)
        cfg = context.get("cfg", 7)
        sampler_name = context.get("sampler", "dpmpp_sde_gpu")
        scheduler = context.get("scheduler", "karras")
        positive_cond_base = context.get("positive", "")
        negative_cond_base = context.get("negative", "")
        base_model = context.get("model", None)
        samples = context.get("latent", None)

        tile_denoise = denoise_image

        if image_optional is not None:
            vaeencoder = VAEEncode()
            samples = vaeencoder.encode(vae, image_optional)[0]
        
        img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                    upscale_by, model_name, seed, vae, denoise_image,steps, cfg, sampler_name, scheduler)
        img= tensor2pil(img)

        tiled_image = run_tiler_for_steps(img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps, tile_denoise, tile_size)

        results = easySave(tiled_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": ( tiled_image,)}
            
        return {"ui": {"images": results},
                "result": ( tiled_image,)}







#endregion-----------tile采样器--------------------------------------------------------------------------------#








#region-----------风格组--------------------------------------------------------------------------------------#--



class chx_YC_LG_Redux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            
            "context": ("RUN_CONTEXT",),
            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),

            "crop": (["center", "mask_area", "none"], {
                "default": "none",
                "tooltip": "裁剪模式：center-中心裁剪, mask_area-遮罩区域裁剪, none-不裁剪"
            }),
            "sharpen": ("FLOAT", {
                "default": 0.0,
                "min": -5.0,
                "max": 5.0,
                "step": 0.1,
                "tooltip": "锐化强度：负值为模糊，正值为锐化，0为不处理"
            }),
            "patch_res": ("INT", {
                "default": 16,
                "min": 1,
                "max": 64,
                "step": 1,
                "tooltip": "patch分辨率，数值越大分块越细致"
            }),
            "style_strength": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "风格强度，越高越偏向参考图片"
            }),
            "prompt_strength": ("FLOAT", { 
                "default": 1.0,
                "min": 0.0,
                "max": 2.0,
                "step": 0.01,
                "tooltip": "文本提示词强度，越高文本特征越强"
            }),
            "blend_mode": (["lerp", "feature_boost", "frequency"], {
                "default": "lerp",
                "tooltip": "风格强度的计算方式：\n" +
                        "lerp - 线性混合 - 高度参考原图\n" +
                        "feature_boost - 特征增强 - 增强真实感\n" +
                        "frequency - 频率增强 - 增强高频细节"
            }),
            "noise_level": ("FLOAT", { 
                "default": 0.0,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "添加随机噪声的强度，可用于修复错误细节"
            }),
        },
        "optional": { 
            "image": ("IMAGE",),
            "mask": ("MASK", ), 
            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}
        
    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    
    FUNCTION = "apply_stylemodel"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def crop_to_mask_area(self, image, mask):
        if len(image.shape) == 4:
            B, H, W, C = image.shape
            image = image.squeeze(0)
        else:
            H, W, C = image.shape
        
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)
        
        nonzero_coords = torch.nonzero(mask)
        if len(nonzero_coords) == 0:
            return image, mask
        
        top = nonzero_coords[:, 0].min().item()
        bottom = nonzero_coords[:, 0].max().item()
        left = nonzero_coords[:, 1].min().item()
        right = nonzero_coords[:, 1].max().item()
        
        width = right - left
        height = bottom - top
        size = max(width, height)
        
        center_y = (top + bottom) // 2
        center_x = (left + right) // 2
        
        half_size = size // 2
        new_top = max(0, center_y - half_size)
        new_bottom = min(H, center_y + half_size)
        new_left = max(0, center_x - half_size)
        new_right = min(W, center_x + half_size)
        
        cropped_image = image[new_top:new_bottom, new_left:new_right]
        cropped_mask = mask[new_top:new_bottom, new_left:new_right]
        
        cropped_image = cropped_image.unsqueeze(0)
        cropped_mask = cropped_mask.unsqueeze(0)
        
        return cropped_image, cropped_mask
    
    def apply_image_preprocess(self, image, strength):
        original_shape = image.shape
        original_device = image.device
        
        if torch.is_tensor(image):
            if len(image.shape) == 4:
                image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
            else:
                image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        
        if strength < 0:
            abs_strength = abs(strength)
            kernel_size = int(3 + abs_strength * 12) // 2 * 2 + 1
            sigma = 0.3 + abs_strength * 2.7
            processed = cv2.GaussianBlur(image_np, (kernel_size, kernel_size), sigma)
        elif strength > 0:
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]]) * strength + np.array([[0,0,0],
                                                               [0,1,0],
                                                               [0,0,0]]) * (1 - strength)
            processed = cv2.filter2D(image_np, -1, kernel)
            processed = np.clip(processed, 0, 255)
        else:
            processed = image_np
        
        processed_tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).to(original_device)
        if len(original_shape) == 4:
            processed_tensor = processed_tensor.unsqueeze(0)
        
        return processed_tensor
    
    def apply_style_strength(self, cond, txt, strength, mode="lerp"):
        if mode == "lerp":
            if txt.shape[1] != cond.shape[1]:
                txt_mean = txt.mean(dim=1, keepdim=True)
                txt_expanded = txt_mean.expand(-1, cond.shape[1], -1)
                return torch.lerp(txt_expanded, cond, strength)
            return torch.lerp(txt, cond, strength)
        
        elif mode == "feature_boost":
            mean = torch.mean(cond, dim=-1, keepdim=True)
            std = torch.std(cond, dim=-1, keepdim=True)
            normalized = (cond - mean) / (std + 1e-6)
            boost = torch.tanh(normalized * (strength * 2.0))
            return cond * (1 + boost * 2.0)
    
        elif mode == "frequency":
            try:
                B, N, C = cond.shape
                x = cond.float()
                fft = torch.fft.rfft(x, dim=-1)
                magnitudes = torch.abs(fft)
                phases = torch.angle(fft)
                freq_dim = fft.shape[-1]
                freq_range = torch.linspace(0, 1, freq_dim, device=cond.device)
                alpha = 2.0 * strength
                beta = 0.5
                filter_response = 1.0 + alpha * torch.pow(freq_range, beta)
                filter_response = filter_response.view(1, 1, -1)
                enhanced_magnitudes = magnitudes * filter_response
                enhanced_fft = enhanced_magnitudes * torch.exp(1j * phases)
                enhanced = torch.fft.irfft(enhanced_fft, n=C, dim=-1)
                mean = enhanced.mean(dim=-1, keepdim=True)
                std = enhanced.std(dim=-1, keepdim=True)
                enhanced_norm = (enhanced - mean) / (std + 1e-6)
                mix_ratio = torch.sigmoid(torch.tensor(strength * 2 - 1))
                result = torch.lerp(cond, enhanced_norm.to(cond.dtype), mix_ratio)
                residual = (result - cond) * strength
                final = cond + residual
                return final
            except Exception as e:
                print(f"频率处理出错: {e}")
                print(f"输入张量形状: {cond.shape}")
                return cond
                
        return cond
    
    def apply_stylemodel(self, style_model, clip_vision,
                        patch_res=16, style_strength=1.0, prompt_strength=1.0, 
                        noise_level=0.0, crop="none", sharpen=0.0, guidance=30,
                        blend_mode="lerp", image=None,  mask=None, context=None):
        
        
        conditioning = context.get("positive", None)  
        if image is None:
            return (context,positive,)

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)


        
        processed_image = image.clone()
        if sharpen != 0:
            processed_image = self.apply_image_preprocess(processed_image, sharpen)
        if crop == "mask_area" and mask is not None:
            processed_image, mask = self.crop_to_mask_area(processed_image, mask)
            clip_vision_output = clip_vision.encode_image(processed_image, crop=False)
        else:
            crop_image = True if crop == "center" else False
            clip_vision_output = clip_vision.encode_image(processed_image, crop=crop_image)
        
        cond = style_model.get_cond(clip_vision_output)
        
        B = cond.shape[0]
        H = W = int(math.sqrt(cond.shape[1]))
        C = cond.shape[2]
        cond = cond.reshape(B, H, W, C)
        
        new_H = H * patch_res // 16
        new_W = W * patch_res // 16
        
        cond = torch.nn.functional.interpolate(
            cond.permute(0, 3, 1, 2),
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )
        
        cond = cond.permute(0, 2, 3, 1)
        cond = cond.reshape(B, -1, C)
        cond = cond.flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        c_out = []
        for t in conditioning:
            txt, keys = t
            keys = keys.copy()
            
            if prompt_strength != 1.0:
                txt_enhanced = txt * (prompt_strength ** 3)
                txt_repeated = txt_enhanced.repeat(1, 2, 1)
                txt = txt_repeated
            
            if style_strength != 1.0:
                processed_cond = self.apply_style_strength(
                    cond, txt, style_strength, blend_mode
                )
            else:
                processed_cond = cond
    
            if mask is not None:
                feature_size = int(math.sqrt(processed_cond.shape[1]))
                processed_mask = torch.nn.functional.interpolate(
                    mask.unsqueeze(1) if mask.dim() == 3 else mask,
                    size=(feature_size, feature_size),
                    mode='bilinear',
                    align_corners=False
                ).flatten(1).unsqueeze(-1)
                
                if txt.shape[1] != processed_cond.shape[1]:
                    txt_mean = txt.mean(dim=1, keepdim=True)
                    txt_expanded = txt_mean.expand(-1, processed_cond.shape[1], -1)
                else:
                    txt_expanded = txt
                
                processed_cond = processed_cond * processed_mask + \
                               txt_expanded * (1 - processed_mask)
    
            if noise_level > 0:
                noise = torch.randn_like(processed_cond)
                noise = (noise - noise.mean()) / (noise.std() + 1e-8)
                processed_cond = torch.lerp(processed_cond, noise, noise_level)
                processed_cond = processed_cond * (1.0 + noise_level)
                
            c_out.append([torch.cat((txt, processed_cond), dim=1), keys])
        
        
        
        positive = node_helpers.conditioning_set_values(c_out, {"guidance": guidance})

        context = new_context(context, positive=positive,)
        
        return (context,positive,)



class chx_StyleModelApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                            "context": ("RUN_CONTEXT",),
                            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
                            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),
                            "image": ("IMAGE",),
                            
                            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.001}),
                            "strength_type": (["multiply", "attn_bias"], ),
                            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
                            }}

    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    FUNCTION = "apply_stylemodel"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def apply_stylemodel(self, style_model, clip_vision, strength, strength_type, guidance=30, context=None, image=None):
        
        conditioning = context.get("positive", None)  

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)
        clip_vision_output = clip_vision.encode_image(image, crop="center") 
    
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        if strength_type == "multiply":
            cond *= strength

        n = cond.shape[1]
        c_out = []
        for t in conditioning:
            (txt, keys) = t
            keys = keys.copy()
            if strength_type == "attn_bias" and strength != 1.0:
                attn_bias = torch.log(torch.Tensor([strength]))
                mask_ref_size = keys.get("attention_mask_img_shape", (1, 1))
                n_ref = mask_ref_size[0] * mask_ref_size[1]
                n_txt = txt.shape[1]

                mask = keys.get("attention_mask", None)
                if mask is None:
                    mask = torch.zeros((txt.shape[0], n_txt + n_ref, n_txt + n_ref), dtype=torch.float16)
                if mask.dtype == torch.bool:

                    mask = torch.log(mask.to(dtype=torch.float16))
                new_mask = torch.zeros((txt.shape[0], n_txt + n + n_ref, n_txt + n + n_ref), dtype=torch.float16)

                new_mask[:, :n_txt, :n_txt] = mask[:, :n_txt, :n_txt]
                new_mask[:, :n_txt, n_txt+n:] = mask[:, :n_txt, n_txt:]
                new_mask[:, n_txt+n:, :n_txt] = mask[:, n_txt:, :n_txt]
                new_mask[:, n_txt+n:, n_txt+n:] = mask[:, n_txt:, n_txt:]

                new_mask[:, :n_txt, n_txt:n_txt+n] = attn_bias
                new_mask[:, n_txt+n:, n_txt:n_txt+n] = attn_bias
                keys["attention_mask"] = new_mask.to(txt.device)
                keys["attention_mask_img_shape"] = mask_ref_size

            c_out.append([torch.cat((txt, cond), dim=1), keys])
        
        positive = node_helpers.conditioning_set_values(c_out, {"guidance": guidance})
        context = new_context(context, positive=positive,)
        return (context,positive,)  



class chx_Style_Redux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),
            "style_model": (folder_paths.get_filename_list("style_models"), {"default": "flux1-redux-dev.safetensors"}),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "sigclip_vision_patch14_384.safetensors"}),
            "image": ("IMAGE",),
            "style_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制整体艺术风格的权重"
            }),
            "color_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制颜色特征的权重"
            }),
            "content_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制内容语义的权重"
            }),
            "structure_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制结构布局的权重"
            }),
            "texture_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制纹理细节的权重"
            }),
            "similarity_threshold": ("FLOAT", {
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "特征相似度阈值，超过此值的区域将被替换"
            }),
            "enhancement_base": ("FLOAT", {
                "default": 1.5,
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "tooltip": "文本特征替换的基础增强系数"
            })
        },
        
        "optional": { 
            "guidance": ("FLOAT", {"default": 30, "min": 0.0, "max": 100.0, "step": 0.1}),
        }}
        
    
    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING",)
    RETURN_NAMES = ("context", "positive",)
    
    FUNCTION = "apply_style"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def __init__(self):
        
        import comfy.ops
        ops = comfy.ops.manual_cast

        self.text_projector = ops.Linear(4096, 4096)  # 保持维度一致
        # 为不同类型特征设置增强系数
        self.enhancement_factors = {
            'style': 1.2,    # 风格特征增强系数
            'color': 1.0,    # 颜色特征增强系数
            'content': 1.1,  # 内容特征增强系数
            'structure': 1.3, # 结构特征增强系数
            'texture': 1.0   # 纹理特征增强系数
        }

    def compute_similarity(self, text_feat, image_feat):
        """计算多种相似度的组合"""
        # 1. 余弦相似度
        cos_sim = torch.cosine_similarity(text_feat, image_feat, dim=-1)
        
        l2_dist = torch.norm(text_feat - image_feat, p=2, dim=-1)
        l2_sim = 1 / (1 + l2_dist)  # 转换为相似度
        
        dot_sim = torch.sum(text_feat * image_feat, dim=-1)
        dot_sim = torch.tanh(dot_sim)  # 归一化到[-1,1]
        
        attn_weights = torch.softmax(torch.matmul(text_feat, image_feat.transpose(-2, -1)) / math.sqrt(text_feat.size(-1)), dim=-1)
        attn_sim = torch.mean(attn_weights, dim=-1)
        
        combined_sim = (
            0.4 * cos_sim +
            0.2 * l2_sim +
            0.2 * dot_sim +
            0.2 * attn_sim
        )
        
        return combined_sim.mean()

    def apply_style(self, style_weight=1.0, color_weight=1.0, content_weight=1.0,guidance=30,
                structure_weight=1.0, texture_weight=1.0, image=None, style_model=None, clip_vision=None,
                similarity_threshold=0.7, enhancement_base=1.5,context=None,):
        
        conditioning = context.get("positive", None)  

        style_model_path = folder_paths.get_full_path_or_raise("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)
        clip_vision_output = clip_vision.encode_image(image, crop="center") 
    
        image_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        
        text_features = conditioning[0][0]  # [batch_size, seq_len, 4096]
        text_features = text_features.mean(dim=1)  # [batch_size, 4096]
        
        text_features = self.text_projector(text_features)  # [batch_size, 4096]
        
        if text_features.shape[0] != image_cond.shape[0]:
            text_features = text_features.expand(image_cond.shape[0], -1)

        feature_size = image_cond.shape[-1]  # 4096
        splits = feature_size // 5  # 每部分约819维

        image_features = {
            'style': image_cond[..., :splits],
            'color': image_cond[..., splits:splits*2],
            'content': image_cond[..., splits*2:splits*3],
            'structure': image_cond[..., splits*3:splits*4],
            'texture': image_cond[..., splits*4:]
        }
        
        similarities = {}
        for key, region_features in image_features.items():
            region_text_features = text_features[..., :region_features.shape[-1]]
            similarities[key] = self.compute_similarity(region_text_features, region_features)
        final_features = {}
        weights = {
            'style': style_weight,
            'color': color_weight,
            'content': content_weight,
            'structure': structure_weight,
            'texture': texture_weight
        }
        
        for key in image_features:
            if similarities[key] > similarity_threshold:
                region_size = image_features[key].shape[-1]
                dynamic_factor = enhancement_base * self.enhancement_factors[key]
                final_features[key] = text_features[..., :region_size] * weights[key] * dynamic_factor
            else:
                final_features[key] = image_features[key] * weights[key]
        
        # 合并所有特征
        combined_cond = torch.cat([
            final_features['style'],
            final_features['color'],
            final_features['content'],
            final_features['structure'],
            final_features['texture']
        ], dim=-1).unsqueeze(dim=0)
        
        # 构建新的条件
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], combined_cond), dim=1), t[1].copy()]
            c.append(n)
            
        positive = node_helpers.conditioning_set_values(c, {"guidance": guidance})
        
        context = new_context(context, positive=positive,)
        return (context,positive,)  



#endregion-----------风格组--------------------------------------------------------------------------------------#--



#region------------nanchaku--------------------------
import sys
import os
import importlib

current_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(custom_nodes_dir)


try:
    nunchaku_module = importlib.import_module("ComfyUI-nunchaku.nodes.models.flux")
    NunchakuFluxDiTLoader = nunchaku_module.NunchakuFluxDiTLoader
except ImportError as e:
    print(f"导入ComfyUI-nunchaku.nodes.models.flux错误: {e}")
    NunchakuFluxDiTLoader = None
except AttributeError:
    NunchakuFluxDiTLoader = None

try:
    nunchaku_module_edit = importlib.import_module("ComfyUI-nunchaku.nodes.models.qwenimage")
    NunchakuQwenImageDiTLoader = nunchaku_module_edit.NunchakuQwenImageDiTLoader
except ImportError as e:
    print(f"导入ComfyUI-nunchaku.nodes.qwenimage错误: {e}")
    NunchakuQwenImageDiTLoader = None
except AttributeError:
    NunchakuQwenImageDiTLoader = None
def check_Nunchaku_installed():
    if NunchakuFluxDiTLoader is None:
        raise RuntimeError("Please install ComfyUI-nunchaku before using this function.")





class load_Nanchaku:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": (["None"] + preset_list, {"default": "None"}),
                "unet_name": (["None"] + available_unets,),
                "cache_threshold": ("FLOAT", {"default": 0, "min": 0, "max": 1, "step": 0.001,}),
                "attention": (["nunchaku-fp16", "flash-attention2"], {"default": "nunchaku-fp16",}),
                "cpu_offload": (["auto", "enable", "disable"], {"default": "auto",}),
                "clip1": (["None"] + available_clips, {"default": "clip_l.safetensors"}),
                "clip2": (["None"] + available_clips, {"default": "t5xxl_fp8_e4m3fn.safetensors"}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "vae": (available_vaes, {"default": "ae.safetensors"}),
                "lora": (["None"] + available_loras,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 1024, "min": 8, "max": 16384}),
                "steps": ("INT", {"default": 10, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "pos": ("STRING", {"multiline": False, "dynamicPrompts": True, "default": "beautiful detailed glow,a girl"}),
            },
            "optional": {
                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),
                "lora_stack": ("LORASTACK",),
            },
            "hidden": {
                "node_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "PDATA")
    RETURN_NAMES = ("context", "model", "preset_save")
    FUNCTION = "process_settings"
    CATEGORY = "Apt_Preset/chx_load"
    DESCRIPTION = """
- cache_threshold典型设置为0.12。越大速度越快，将其设为0会禁用该效果。
- nunchaku-fp16采用FP16注意力机制，可提供约1.2倍的速度提升。
- 请注意，20系列显卡只能使用nunchaku-fp16。
    """

    def _load_clip(self, clip1, clip2, clip_type, device):
        """加载CLIP模型"""
        clip = None
        
        # 双CLIP加载情况 (FluxDiT)
        if clip1 != "None" and clip2 != "None":
            if clip1.endswith(".gguf"):
                from .load_GGUF.nodes import DualCLIPLoaderGGUF2
                clip = DualCLIPLoaderGGUF2().load_clip(clip1, clip2, clip_type)[0]
            else:
                clip = DualCLIPLoader().load_clip(clip1, clip2, clip_type, device)[0]
        
        # 单CLIP加载情况
        elif (clip1 != "None" and clip2 == "None") or (clip1 == "None" and clip2 != "None"):
            # 确定使用的clip文件
            clip_file = clip1 if clip1 != "None" else clip2
            
            if clip_file.endswith(".gguf"):
                from .load_GGUF.nodes import CLIPLoaderGGUF2
                clip = CLIPLoaderGGUF2().load_clip(clip_file, clip_type)[0]
            else:
                clip = CLIPLoader().load_clip(clip_file, clip_type, device)[0]
                
        return clip

    def _load_model(self, unet_name, attention, cache_threshold, cpu_offload, device_id, data_type, clip1, clip2, **kwargs):
        """加载模型"""
        model = None
        
        # 双CLIP加载情况 (FluxDiT)
        if clip1 != "None" and clip2 != "None":
            if NunchakuFluxDiTLoader is not None:
                try:
                    if callable(NunchakuFluxDiTLoader):
                        model_loader = NunchakuFluxDiTLoader()
                        model_result = model_loader.load_model(unet_name, attention, cache_threshold, cpu_offload, device_id, data_type, **kwargs)
                        model = model_result[0] if isinstance(model_result, tuple) else model_result
                    else:
                        model_result = NunchakuFluxDiTLoader.load_model(unet_name, attention, cache_threshold, cpu_offload, device_id, data_type, **kwargs)
                        model = model_result[0] if isinstance(model_result, tuple) else model_result
                except Exception as e:
                    raise RuntimeError(f"Failed to load model with NunchakuFluxDiTLoader: {e}")
            else:
                raise RuntimeError("NunchakuFluxDiTLoader is not available")

        # 单CLIP加载情况 (QwenImageDiT)
        elif (clip1 != "None" and clip2 == "None") or (clip1 == "None" and clip2 != "None"):
            # 检查并加载QwenImageDiT模型
            if NunchakuQwenImageDiTLoader is not None:
                try:
                    if callable(NunchakuQwenImageDiTLoader):
                        model_loader = NunchakuQwenImageDiTLoader()
                        model_result = model_loader.load_model(unet_name, cpu_offload, **kwargs)
                        model = model_result[0] if isinstance(model_result, tuple) else model_result
                    else:
                        model_result = NunchakuQwenImageDiTLoader.load_model(unet_name, cpu_offload, **kwargs)
                        model = model_result[0] if isinstance(model_result, tuple) else model_result
                except Exception as e:
                    raise RuntimeError(f"Failed to load model with NunchakuQwenImageDiTLoader: {e}")
            else:
                raise RuntimeError("NunchakuQwenImageDiTLoader is not available")
                
        return model

    def process_settings(self, node_id, width, height, steps, cfg, sampler, scheduler, guidance, device="default", 
                         lora=None, lora_strength=1.0, cache_threshold=0, cpu_offload="auto", attention="nunchaku-fp16",
                         vae=None, clip1=None, unet_name=None, data_type=None, lora_stack=None, over_model=None, over_clip=None,
                         clip2=None, pos="default", preset=[], **kwargs):

        device_id = 0
        data_type = "bfloat16"
        neg = "worst quality, low quality"
        batch = 1
        clip_type = "flux"

        parameters_data = [{
            "run_Mode": "FLUX",
            "ckpt_name": None,
            "clipnum": None,
            "unet_name": unet_name,
            "unet_Weight_Dtype": None,
            "clip_type": None,
            "clip1": clip1,
            "clip2": clip2,
            "guidance": guidance,
            "clip3": None,
            "clip4": None,
            "vae": vae,
            "lora": lora,
            "lora_strength": lora_strength,
            "width": width,
            "height": height,
            "batch": batch,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "positive": pos,
            "negative": neg,
            "cache_threshold": cache_threshold,
            "attention": attention,
            "cpu_offload": cpu_offload,
        }]

        # 分辨率修正
        width, height = width - (width % 8), height - (height % 8)
        latent = torch.zeros([batch, 4, height // 8, width // 8])
        if latent.shape[1] != 16:
            latent = latent.repeat(1, 16 // 4, 1, 1)

        # 处理VAE
        if isinstance(vae, str) and vae != "None":
            vae_path = folder_paths.get_full_path("vae", vae)
            vae = comfy.sd.VAE(comfy.utils.load_torch_file(vae_path))

        # 初始化model和clip变量
        model = over_model
        clip = over_clip

        # 如果没有提供over_model，则加载模型
        if over_model is None:
            clip = self._load_clip(clip1, clip2, clip_type, device)
            model = self._load_model(unet_name, attention, cache_threshold, cpu_offload, device_id, data_type, clip1, clip2, **kwargs)
        # 如果提供了over_model但没有提供over_clip，则只加载clip
        elif over_clip is None:
            clip = self._load_clip(clip1, clip2, clip_type, device)

        # 应用LoRA
        if lora_stack is not None and model is not None and clip is not None:
            model, clip = apply_lora_stack(model, clip, lora_stack)
        if lora != "None" and lora_strength != 0 and model is not None and clip is not None:
            model, clip = LoraLoader().load_lora(model, clip, lora, lora_strength, lora_strength)

        # 编码文本
        positive = None
        negative = None
        if clip is not None:
            (positive,) = CLIPTextEncode().encode(clip, pos)
            (negative,) = CLIPTextEncode().encode(clip, neg)
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        context = {
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},
            "vae": vae,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": guidance,
            "clip1": clip1,
            "clip2": clip2,
            "clip3": None,
            "clip4": None,
            "unet_name": None,
            "ckpt_name": None,
            "pos": pos,
            "neg": neg,
            "width": width,
            "height": height,
            "batch": batch,
        }
        return (context, model, parameters_data,)

    def handle_my_message(d):
        
        preset_data = ""
        preset_path = os.path.join(presets_directory_path, d['message'])
        with open(preset_path, 'r', encoding='utf-8') as f:    
            preset_data = toml.load(f)
        PromptServer.instance.send_sync("my.custom.message", {"message":preset_data, "node":d['node_id']})






#endregion---------------------------nanchaku-------------------------------





#region-----------tool--------------------------------------------------------------------------------------#--



class pre_sample_data:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "steps": ("INT", {"default": 0, "min": 0, "max": 10000,"tooltip": "  0  == no change"}),
                "cfg": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "tooltip": "  0  == no change"}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),  
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}), 
                
                
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", )
    RETURN_NAMES = ("context", )
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/chx_tool/😺backup"

    def sample(self, context, steps, cfg, sampler, scheduler):
        
        if cfg == 0.0:
            cfg = context.get("cfg")
        if steps == 0:
            steps = context.get("steps")
        sampler = context.get("sampler","euler")
        scheduler = context.get("scheduler","normal")
        
        context = new_context(context, steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler)
        return (context, )


class pre_guide:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
            
        }

    RETURN_TYPES = ("RUN_CONTEXT", )
    RETURN_NAMES = ("context", )
    FUNCTION = "fluxguide"
    CATEGORY = "Apt_Preset/chx_tool/😺backup"

    def fluxguide(self,context, guidance, ):  

        positive = context.get("positive",2.5)
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})
        
        context = new_context(context, positive=positive, guidance=guidance )
        return (context, )


class pre_USO:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "context": ("RUN_CONTEXT",),

            "image": ("IMAGE", ),
            "reference_latents_method": (("uxo/uno","offset", "index" ), ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "smoothness":("INT", {"default": 0,  "min":0, "max": 10, "step": 0.1,}),

            "crop": (["center", "none"],),
            "clip_vision": (folder_paths.get_filename_list("clip_vision"),{"default": "sigclip_vision_patch14_384.safetensors"} ),
            "model_patch": (folder_paths.get_filename_list("model_patches"), {"default": "uso-flux1-projector-v1.safetensors"}),


                    },

        "optional": {
            "mask": ("MASK",),
            "ref_image": ("IMAGE", ),                 
                    }
               }


    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CONDITIONING","LATENT" )
    RETURN_NAMES = ("context","model","positive","latent")
    FUNCTION = "append"
    CATEGORY = "Apt_Preset/chx_tool/😺backup"


    def append(self,context, guidance, crop, clip_vision=None, model_patch=None, image=None, mask=None,smoothness=0, ref_image=None, reference_latents_method="uxo/uno"):
        vae = context.get("vae", None)
        conditioning = context.get("positive", None)
        negative = context.get("negative", None)



        if image is None:
           raise Exception("Please provide an input image.")


        latent = encode(vae, image)[0]

        conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [latent["samples"]]}, append=True)
        conditioning = FluxKontextMultiReferenceLatentMethod().append(conditioning, reference_latents_method)[0]
        conditioning = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})

        positive=conditioning
        

        if mask is not None:
            mask =smoothness_mask(mask, smoothness)
            positive, negative, latent = InpaintModelConditioning().encode(positive, negative, image, vae, mask, True)
        else:
            latent = encode(vae, image)[0]


        if ref_image is not None:
            model=context.get("model", None)
            model_patch= ModelPatchLoader().load_model_patch(model_patch)[0]
            clip_vision = CLIPVisionLoader().load_clip(clip_vision)[0]
            clip_vision_out = CLIPVisionEncode().encode(clip_vision, ref_image, crop)[0]

            model= USOStyleReference().apply_patch(model, model_patch, clip_vision_out)[0]
        else:
            model=context.get("model", None)

        context = new_context(context, positive=positive, latent=latent, model=model )

        return (context, model, positive, latent )




#endregion-----------tool--------------------------------------------------------------------------------------#--


class Data_preset_save:
    @classmethod
    def INPUT_TYPES(s):
        savetype_list = ["new save", "overwrite save"]
        return {
            "required": {
                "param": ("PDATA", ),
                "tomlname": ("STRING", {"default": "new_preset"}),
                "savetype": (savetype_list,),
            },
        }
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "saveparam"
    CATEGORY = "Apt_Preset/chx_load"

    def saveparam(self, param, tomlname, savetype):
        # 初始化 tomltext 为一个空字符串
        tomltext = ""

        def format_value(value):
            # 如果是数字类型，直接返回不带引号的字符串
            if isinstance(value, (int, float)):
                return str(value)
            # 如果是字符串类型，处理转义并加上引号
            elif isinstance(value, str):
                escaped_value = value.replace('\\', '\\\\')
                return f'"{escaped_value}"'
            # 其他类型（包括None）转换为字符串并加上引号
            else:
                escaped_value = str(value).replace('\\', '\\\\')
                return f'"{escaped_value}"'

        # 定义所有需要保存的参数键
        keys = [
            "run_Mode", "ckpt_name", "clipnum", "unet_name", "unet_Weight_Dtype",
            "clip_type", "clip1", "clip2", "guidance", "clip3", "clip4", "vae",
            "lora", "lora_strength", "width", "height", "batch", "steps", "cfg",
            "sampler", "scheduler", "positive", "negative", "cache_threshold",
            "attention", "cpu_offload"
        ]

        # 逐个处理参数并添加到tomltext中
        for key in keys:
            value = param[0].get(key, None)
            tomltext += f"{key} = {format_value(value)}\n"

        tomlnameExt = getNewTomlnameExt(tomlname, presets_directory_path, savetype)

        check_folder_path = os.path.dirname(f"{presets_directory_path}/{tomlnameExt}")
        os.makedirs(check_folder_path, exist_ok=True)

        with open(f"{presets_directory_path}/{tomlnameExt}", mode='w', encoding='utf-8') as f:
            f.write(tomltext)

        return ()






# region Ksampler_sum-----------------------------------

class Stack_VAEDecodeTiled:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to decode at a time."}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4, "tooltip": "Only used for video VAEs: Amount of frames to overlap."})
                },
                "optional": {}}

    RETURN_TYPES = ("VAEDecodeTiled",)
    RETURN_NAMES = ("vaeTile",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def encode(self, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
        pack = (tile_size, overlap, temporal_size, temporal_overlap)
        return (pack,)


class Stack_Ksampler_adv:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "steps": ("INT", {"default": 20, "min": 0, "max": 10000,}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0,}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 1000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],)
            },

        }

    RETURN_TYPES = ("KS_STACK",)
    RETURN_NAMES = ("ksample",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def encode(self, add_noise, steps, cfg, sampler, scheduler, start_at_step, end_at_step, return_with_leftover_noise):
        data = (add_noise, steps, cfg, sampler, scheduler, start_at_step, end_at_step, return_with_leftover_noise)
        return (data, )


class Stack_Ksampler_basic:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "steps": ("INT", {"default": 8, "min": 0, "max": 10000, }),
                "cfg": ("FLOAT", {"default": 8, "min": 0.0, "max": 100.0, }),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"})
            },

        }

    RETURN_TYPES = ("KS_STACK",)
    RETURN_NAMES = ("ksample",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def encode(self, steps, cfg, sampler, scheduler):
        data = (steps, cfg, sampler, scheduler)
        return (data, )


class Stack_Ksampler_custom:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {  # 保持所有参数为可选
                "steps": ("INT", {"default": 8, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "noise": ("NOISE",),  # 可选参数
                "guider": ("GUIDER",),  # 可选参数
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",)
            }
        }

    RETURN_TYPES = ("KS_STACK",)
    RETURN_NAMES = ("ksample",)
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def encode(self, steps=8, cfg=8.0, scheduler="normal", 
               noise=None, guider=None, sampler=None, sigmas=None):
        data = (steps, cfg, scheduler, noise, guider, sampler, sigmas)
        return (data, )



class Stack_Ksampler_refine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "resample": ("RESAMPLE",),
                "upscale_model": (["None"] +folder_paths.get_filename_list("upscale_models"), {"default": "1xDeJPG_OmniSR.pth"}),
                "upscale_method": (["nearest-exact", "bilinear", "area", "bicubic", "lanczos"], {"default": "bilinear" }),
                "pixls_scale": ("FLOAT", {"default": 2, "min": 1, "max": 16.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "over_image": ("IMAGE",),
                "lowCpu": ("VAEDecodeTiled",),  
            },            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
        }


    RETURN_TYPES = ( "IMAGE", )
    RETURN_NAMES = ( "image", )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"
    def run(self,resample, seed, denoise, upscale_model,upscale_method, prompt=None, over_image=None,image_output=None, extra_pnginfo=None,lowCpu=None,pixls_scale=1):

        (model, positive, negative, vae, steps, cfg, sampler, scheduler, image, clip, latent) = resample 

        if over_image is not None:
            image = over_image

        if upscale_model != "None":         
            up_model = load_upscale_model(upscale_model)
            image = upscale_with_model(up_model, image )
        if pixls_scale != 1:
            image = image_upscale(image, upscale_method, pixls_scale)[0]
        latent = encode(vae, image)[0]
        latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                positive,  negative, latent,  denoise=denoise )[0]

        if lowCpu is not None:
            (tile_size, overlap, temporal_size, temporal_overlap)=lowCpu
            output_image = VAEDecodeTiled(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)[0]
        else:
            output_image = decode(vae, latent)[0] 
        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (output_image,)}
            
        return {"ui": {"images": results},
                "result": (output_image,)}



class Stack_Ksampler_highAndLow:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "add_noise": (["enable", "disable"],),
                "noise_seed": ("INT", {"default": 3, "min": 0, "max": 0xffffffffffffffff}),
                "start_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "mid_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],)
            },
            "optional": {
                "model2": ("MODEL",),
                "positive2": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("FUNTION", )
    RETURN_NAMES = ("funtion", )
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def encode(self, add_noise, noise_seed, start_step, mid_step, steps, return_with_leftover_noise, model2=None,
               positive2=None):
        data = (add_noise, noise_seed, start_step, mid_step, steps, return_with_leftover_noise, model2, positive2)
        return (data,)


class Stack_Ksampler_dual_paint:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 500, "step": 1, "display": "slider"}),
                "mask_area_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_area_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refine": ("BOOLEAN", {"default": False}),
                "refine_denoise": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("FUNTION", )
    RETURN_NAMES = ("funtion", )
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def encode(self, image, mask, smoothness, mask_area_denoise, image_area_denoise, refine, refine_denoise):
        data = (image, mask, smoothness, mask_area_denoise, image_area_denoise, refine, refine_denoise)
        return (data,)


def Stack_dual_paint(image, mask, smoothness, mask_area_denoise, image_area_denoise,refine,refine_denoise, seed, 
            model, positive, negative, steps, cfg, sampler, scheduler,vae):
    

    phase_steps = math.ceil(steps / 2)
    device = model.model.device if hasattr(model, 'model') else model.device
    
    vae_encoder = VAEEncode()
    latent_dict = vae_encoder.encode(vae, image)[0]
    input_latent = latent_dict["samples"].to(device)


    if mask is not None :
        mask=tensor2pil(mask)
        if not isinstance(mask, Image.Image):
            raise TypeError("mask is not a valid PIL Image object")
        
        feathered_image = mask.filter(ImageFilter.GaussianBlur(smoothness))
        mask=pil2tensor(feathered_image)


    
    mask = 1-mask.float().to(device)
    
    mask_resized = torch.nn.functional.interpolate(
        mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), 
        size=(input_latent.shape[2], input_latent.shape[3]), 
        mode='bilinear'
    )
    
    mask_strength = mask_resized * (image_area_denoise - mask_area_denoise) + mask_area_denoise
    
    noise_mask = SetLatentNoiseMask()
    latent_with_mask = noise_mask.set_mask({"samples": input_latent}, mask_strength)[0]
    
    advanced_sampler = KSamplerAdvanced()
    
    result = advanced_sampler.sample(
        model=model,
        add_noise=0.00,
        noise_seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_with_mask,
        start_at_step=0,
        end_at_step=phase_steps,
        return_with_leftover_noise=False
    )[0]
    samples = result["samples"].to(device)
    binary_mask = (mask_resized >= 0.5).float()
    phase2_mask = binary_mask * 1.0 + (1 - binary_mask) * mask_area_denoise
    
    latent_phase2 = noise_mask.set_mask(
        {"samples": samples},
        phase2_mask
    )[0]

    result = advanced_sampler.sample(
        model=model,
        add_noise=0.00,
        noise_seed=seed + 1,
        steps=steps,
        cfg=cfg,
        sampler_name=sampler,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_phase2,
        start_at_step=phase_steps,
        end_at_step=steps,
        return_with_leftover_noise=False
    )[0]
    samples = result["samples"].to(device)
    
    if refine:

        sampler = KSampler()
        result = sampler.sample(
            model,
            seed + 1,
            steps,
            cfg,
            sampler,
            scheduler,
            positive,
            negative,
            {"samples": samples},
            refine_denoise
        )[0]
        samples = result["samples"].to(device)   
    latent= {"samples": samples}       
    return ( latent,)


class Stack_ksampler_tile:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                   "resample": ("RESAMPLE",),
                    "model_name": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                    "upscale_by": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "denoise_image": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "tile_size": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                    "image_output": (["Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
                    },
            "optional": {
                "lowCpu": ("VAEDecodeTiled",),  
                "over_image": ("IMAGE",),
            },              
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",},
            }
            
    OUTPUT_NODE = True
    RETURN_TYPES = ('IMAGE', )
    RETURN_NAMES = ('output_image', )
    FUNCTION = 'run'
    CATEGORY = "Apt_Preset/stack/ksample/😺backup"

    def phase_one(self, base_model, samples, positive_cond_base, negative_cond_base,
                    upscale_by, model_name, seed, vae, denoise_image, steps, cfg, sampler_name, scheduler):
        
        image_scaler = ImageScale()
        vaedecoder = VAEDecode()
        uml = UpscaleModelLoader()
        upscale_model = uml.load_model(model_name)[0]
        iuwm = ImageUpscaleWithModel()
        start_step = int(steps - (steps * denoise_image))
        
        # 确保 samples 是 latent 格式（字典）
        if isinstance(samples, torch.Tensor):
            if samples.ndim == 4 and samples.shape[-1] in [3, 4]:
                # 这是图像数据，需要编码为 latent
                samples = {"samples": VAEEncode().encode(vae, samples)[0]["samples"]}
            else:
                # 这已经是 latent 数据
                samples = {"samples": samples}
        elif not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("samples must be a tensor or a dict with 'samples' key")
        
        sample1 = common_ksampler(base_model, seed, steps, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, samples,
                                start_step=start_step, last_step=steps, force_full_denoise=False)[0]
        pixels = vaedecoder.decode(vae, sample1)[0]
        org_width, org_height = pixels.shape[2], pixels.shape[1]
        img = iuwm.upscale(upscale_model, image=pixels)[0]
        upscaled_width, upscaled_height = int(org_width * upscale_by // 8 * 8), int(org_height * upscale_by // 8 * 8)
        img = image_scaler.upscale(img, "bicubic", upscaled_width, upscaled_height, 'center')[0]
        return img, upscaled_width, upscaled_height

    def run(self, seed, model_name, upscale_by=2.0, tile_size=512, prompt=None, image_output=None, extra_pnginfo=None,
            upscale_method='normal', denoise_image=1.0, resample=None, lowCpu=None,over_image=None):

        (base_model, positive_cond_base, negative_cond_base, vae, steps, cfg, sampler_name, scheduler, image, clip, latent) = resample
        if over_image is not None:
            image = over_image
            
        tile_denoise = denoise_image

        # 处理输入数据
        if image is not None:
            # 如果提供了图像，先编码为 latent
            samples = VAEEncode().encode(vae, image)[0]
        elif latent is not None:
            # 如果提供了 latent，直接使用
            samples = latent
        else:
            raise ValueError("Either image or latent must be provided")

        img, upscaled_width, upscaled_height = self.phase_one(base_model, samples, positive_cond_base, negative_cond_base,
                                    upscale_by, model_name, seed, vae, denoise_image, steps, cfg, sampler_name, scheduler)
        img = tensor2pil(img)

        tiled_image = run_tiler_for_steps(img, base_model, vae, seed, cfg, sampler_name, scheduler, positive_cond_base, negative_cond_base, steps, tile_denoise, tile_size)

        results = easySave(tiled_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (tiled_image,)}
            
        return {"ui": {"images": results},
                "result": (tiled_image,)}


class sum_Ksampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),                                
            },           
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_stack": ("LATENT_STACK",),       
                "lowCpu": ("VAEDecodeTiled",),         
                "ksample_type":("KS_STACK",),
                "funtion":("FUNTION",),

            },            
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO",}, }

    RETURN_TYPES = ("RUN_CONTEXT","IMAGE","RESAMPLE", )
    RETURN_NAMES = ("context", "image","resample"  )
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_ksample"


    def run(self,context, seed, denoise, positive=None, negative=None, model=None, prompt=None, image_output=None, extra_pnginfo=None, 
            ksample_type=None,lowCpu=None,latent_stack=None,funtion=None):
        clip = context.get("clip",None)
        vae = context.get("vae",None)
        steps = context.get("steps",20)
        cfg = context.get("cfg",8)
        sampler = context.get("sampler",None)
        scheduler = context.get("scheduler",None)
        image2 = context.get("images",None)

        if positive is None:
            positive = context.get("positive","boy" )
        if negative is None:
            negative = context.get("negative","" )
        if model is None:
            model= context.get("model")

#-------------------------------------------------------------------------------------

        if latent_stack is not None:
            from .main_stack import Apply_latent
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)      
        else:   
            latent = context.get("latent",None)

#-------------------ksample_type------------------------------------------------------------------

        if ksample_type is not None:
            if len(ksample_type) == 8:
                (add_noise, steps, cfg, sampler, scheduler, start_at_step, end_at_step, return_with_leftover_noise) = ksample_type              
                disable_noise = False
                if add_noise == "disable":
                    disable_noise = True
                force_full_denoise = True
                if return_with_leftover_noise == "enable":
                    force_full_denoise = False
                if latent is None:
                    device = model.device if hasattr(model, 'device') else 'cpu'
                    latent = {"samples": torch.zeros([1, 4, 64, 64], device=device)}
                latent = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]

            elif len(ksample_type) == 7:
                (ks_steps, cfg, ks_scheduler, noise, guider, sampler, sigmas) = ksample_type    
                if sampler is None:
                    sampler_name = context.get("sampler", None)
                    if sampler_name:
                        sampler = KSamplerSelect().get_sampler(sampler_name)[0]
                    else:
                        sampler = KSamplerSelect().get_sampler("euler")[0]                       
                if noise is None:
                    noise = RandomNoise().get_noise(seed)[0]                 
                if guider is None:
                    guider = BasicGuider().get_guider(model, positive)[0]                    
                if sigmas is None:
                    sched = ks_scheduler if ks_scheduler != None else context.get("scheduler", "normal")
                    sigmas = BasicScheduler().get_sigmas(model, sched, ks_steps if ks_steps != None else steps, denoise)[0]      
                latent = SamplerCustomAdvanced().sample(noise, guider, sampler, sigmas, latent)[0]
            
            elif len(ksample_type) == 4:
                (steps, cfg, sampler, scheduler) = ksample_type              
                latent = common_ksampler(model, seed, steps, cfg, sampler, scheduler,
                        positive, negative,  latent,  denoise=denoise )[0]


        if funtion is not None:

            if len(funtion) == 8:
                (add_noise, noise_seed, start_step, mid_step, steps, return_with_leftover_noise, model2, positive2)=funtion 
                disable_noise = False 
                if add_noise == "disable":
                    disable_noise = True 
                force_full_denoise = True
                if return_with_leftover_noise == "enable":
                    force_full_denoise = False
                         
                latent1 = common_ksampler(model, noise_seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise, 
                                        disable_noise=disable_noise, start_step=start_step, last_step=mid_step, force_full_denoise=force_full_denoise)[0]
                
                if model2 is None:
                    model = context.get("model", None)  
                if positive2 is None:
                    positive = context.get("positive", None)
                latent = common_ksampler(model, noise_seed, steps, cfg, sampler, scheduler, positive, negative, latent1, denoise=denoise, 
                                        disable_noise=disable_noise, start_step=mid_step, last_step=steps, force_full_denoise=force_full_denoise)[0]

            elif len(funtion) == 7:
                (image, mask, smoothness, mask_area_denoise, image_area_denoise, refine, refine_denoise)=funtion
                latent= Stack_dual_paint(image, mask, smoothness, mask_area_denoise, image_area_denoise,refine,refine_denoise, seed, 
                                    model, positive, negative, steps, cfg, sampler, scheduler,vae)[0]


        else:
            latent = common_ksampler(model,seed, steps, cfg, sampler, scheduler,
                    positive,  negative, latent, denoise=denoise)[0]
            
#----------------vae-way----------------------------------------------------------------------

        if lowCpu is not None:
            (tile_size, overlap, temporal_size, temporal_overlap)=lowCpu
            output_image = VAEDecodeTiled(vae, latent, tile_size, overlap, temporal_size, temporal_overlap)[0]
        else:
            output_image = VAEDecode().decode(vae, latent)[0]

#-----------------------------------------------------------------------------------------------

        context = new_context(context, model=model, positive=positive, negative=negative,  clip=clip, latent=latent, images=output_image, vae=vae,
            steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler, )

        resample = (model, positive, negative, vae, steps, cfg, sampler, scheduler, output_image, clip, latent)


        if image_output == "None":
            context = new_context(context, latent=latent, images=None,  )
            return (context, None, None, )

        results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
        if image_output in ("Hide", "Hide/Save"):
            return {"ui": {},
                "result": (context, output_image, resample )}
            
        return {"ui": {"images": results},
                "result": (context, output_image, resample)}



#endregion-----------sum采样器--------------------------------------------------------------------------------#













#endregion-----------采样器--------------------------------------------------------------------------------#
























































































