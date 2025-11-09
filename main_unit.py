
#region------------------导入库----------------------------------------------------#
import numpy as np
import torch
import os
import comfy.utils
from comfy.cli_args import args
import comfy.samplers
import json
import csv
import re
import math
from spandrel import ModelLoader
import folder_paths
from comfy import model_management
from math import ceil
import torchvision.transforms.functional as TF
from typing import cast
import numbers
from io import BytesIO
import matplotlib.pyplot as plt
from typing import Any, Callable, Mapping
import logging
from PIL import Image, ImageChops, ImageFilter
from nodes import PreviewImage, SaveImage
import numpy.typing as npt
from PIL import Image, ImageDraw,  ImageFilter,  ImageChops, ImageDraw, ImageFont
import hashlib
import torch.nn.functional as F
from scipy.fft import fft
from dataclasses import dataclass
from comfy.utils import ProgressBar
from comfy.utils import common_upscale
import node_helpers
from enum import Enum




#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False




#endregion------------------导入库----------------------------------------------------#


from server import PromptServer

def flow_sch_control_auto_increment(json_data):
    prompt = json_data['prompt']
    for k, v in prompt.items():
        if v.get('class_type') == 'flow_sch_control':
            if v['inputs'].get('mode', False):
                count = v['inputs'].get('count', 0)
                total = v['inputs'].get('total', 1)
                if count < total - 1:
                    v['inputs']['count'] = count + 1
                else:
                    v['inputs']['count'] = 0
    return json_data

PromptServer.instance.add_on_prompt_handler(flow_sch_control_auto_increment)



#region-------------------路径、数据------------------------------------------------------------------------------#


MAX_RESOLUTION=10240

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class AnyType(str):
    def __eq__(self, _) -> bool:
        return True
    def __ne__(self, __value: object) -> bool:
        return False
ANY_TYPE = AnyType("*")
any_type = AnyType("*")
anyType = AnyType("*")
anytype = AnyType("*")
any_typ= AnyType("*")



MAX_RESOLUTION = 88888
CLIP_TYPE = ["sdxl", "sd3", "flux", "hunyuan_video", "stable_diffusion", "stable_audio", "mochi", "ltxv", "pixart", "cosmos", "lumina2", "wan"]



available_ckpt = folder_paths.get_filename_list("checkpoints")
available_loras = folder_paths.get_filename_list("loras")
available_vaes = folder_paths.get_filename_list("vae")
available_embeddings = folder_paths.get_filename_list("embeddings")
available_style_models = folder_paths.get_filename_list("style_models")
available_clip_visions = folder_paths.get_filename_list("clip_vision")
available_controlnet=folder_paths.get_filename_list("controlnet")
available_samplers = comfy.samplers.KSampler.SAMPLERS
available_schedulers = comfy.samplers.KSampler.SCHEDULERS

try:
    available_unets = list(set(folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf")))
except KeyError:
    available_unets = folder_paths.get_filename_list("unet")
try:
    available_clips = list(set(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip_gguf")))
except KeyError:
    available_clips = folder_paths.get_filename_list("text_encoders") 








def load_upscale_model(model_name):
    model_path = folder_paths.get_full_path("upscale_models", model_name)
    model = ModelLoader().load_from_file(model_path).eval()
    return model    



#endregion-----------路径、数据------------------------------------------------------------------------------#



#region-------------------context全局定义------------------------------------------------------------------------------#




_all_contextput_output_data = {
    "context": ("context", "RUN_CONTEXT", "context"),
    "model": ("model", "MODEL", "model"),
    "positive": ("positive", "CONDITIONING", "positive"),
    "negative": ("negative", "CONDITIONING", "negative"),
    "latent": ("latent", "LATENT", "latent"),
    "vae": ("vae", "VAE", "vae"),
    "clip": ("clip","CLIP", "clip"),
    "images": ("images", "IMAGE", "images"),
    "mask": ("mask", "MASK", "mask"),
    "guidance": ("guidance", "FLOAT", "guidance"),
    "steps": ("steps", "INT", "steps"),
    "cfg": ("cfg", "FLOAT", "cfg"),
    "sampler": ("sampler", available_samplers, "sampler"),
    "scheduler": ("scheduler", available_schedulers, "scheduler"),

    "clip1": ("clip1", available_clips, "clip1"),
    "clip2": ("clip2", available_clips, "clip2"),
    "clip3": ("clip3", available_clips, "clip3"),
    "clip4": ("clip4", available_clips, "clip3"),
    "unet_name": ("unet_name", available_unets, "unet_name"),
    "ckpt_name": ("ckpt_name", available_ckpt, "ckpt_name"),
    "pos": ("pos", "STRING", "pos"),
    "neg": ("neg", "STRING", "neg"),
    "width": ("width", "INT","width" ),
    "height": ("height", "INT","height"),
    "batch": ("batch", "INT","batch"),
    "data": ("data", ANY_TYPE, "data"),
    "data1": ("data1", ANY_TYPE, "data1"),
    "data2": ("data2", ANY_TYPE, "data2"),
    "data3": ("data3", ANY_TYPE, "data3"),
    "data4": ("data4", ANY_TYPE, "data4"),
    "data5": ("data5", ANY_TYPE, "data5"),
    "data6": ("data6", ANY_TYPE, "data6"),
    "data7": ("data7", ANY_TYPE, "data7"),
    "data8": ("data8", ANY_TYPE, "data8"),

}

force_input_types = ["INT", "STRING", "FLOAT"]
force_input_names = ["sampler", "scheduler","clip1", "clip2", "clip3","clip4", "unet_name", "ckpt_name"]


def _create_context_data(input_list=None):
    if input_list is None:
        input_list = _all_contextput_output_data.keys()
    list_ctx_return_types = []
    list_ctx_return_names = []
    ctx_optional_inputs = {}
    for inp in input_list:
        data = _all_contextput_output_data[inp]
        list_ctx_return_types.append(data[1])
        list_ctx_return_names.append(data[2])
        ctx_optional_inputs[data[0]] = tuple([data[1]] + (
            [{"forceInput": True}] if data[1] in force_input_types or data[0] in force_input_names else []
        ))
    ctx_return_types = tuple(list_ctx_return_types)
    ctx_return_names = tuple(list_ctx_return_names)
    return (ctx_optional_inputs, ctx_return_types, ctx_return_names)

ALL_CTX_OPTIONAL_INPUTS, ALL_CTX_RETURN_TYPES, ALL_CTX_RETURN_NAMES = _create_context_data()



_original_ctx_inputs_list = ["context", "model", "positive", "negative", "latent", "vae", "clip", "images", "mask",]
ORIG_CTX_OPTIONAL_INPUTS, ORIG_CTX_RETURN_TYPES, ORIG_CTX_RETURN_NAMES = _create_context_data(_original_ctx_inputs_list)


_load_ctx_inputs_list = ["context", "clip1", "clip2", "clip3","clip4", "ckpt_name","unet_name", "pos","neg" ,"width", "height", "batch"]
LOAD_CTX_OPTIONAL_INPUTS, LOAD_CTX_RETURN_TYPES, LOAD_CTX_RETURN_NAMES = _create_context_data(_load_ctx_inputs_list)



def new_context(context, **kwargs):
    context = context if context is not None else None
    new_ctx = {}
    for key in _all_contextput_output_data:
        if key == "context":
            continue
        v = kwargs[key] if key in kwargs else None
        new_ctx[key] = v if v is not None else (
            context[key] if context is not None and key in context else None
        )
    return new_ctx


def merge_new_context(*args):
    new_ctx = {}
    for key in _all_contextput_output_data:
        if key == "base_ctx":
            continue
        v = None
        for ctx in reversed(args):
            v = ctx[key] if not is_context_empty(ctx) and key in ctx else None
            if v is not None:
                break
        new_ctx[key] = v
    return new_ctx


def get_context_return_tuple(ctx, inputs_list=None):
    if inputs_list is None:
        inputs_list = _all_contextput_output_data.keys()
    tup_list = [ctx]
    for key in inputs_list:
        if key == "context":
            continue
        tup_list.append(ctx[key] if ctx is not None and key in ctx else None)
    return tuple(tup_list)


def get_orig_context_return_tuple(ctx):
    return get_context_return_tuple(ctx, _original_ctx_inputs_list)

def get_load_context_return_tuple(ctx):
    return get_context_return_tuple(ctx, _load_ctx_inputs_list)



def is_context_empty(ctx):
    return not ctx or all(v is None for v in ctx.values())



#endregion



# region-----------------缓动函数------------------------------------------------------
def easeInBack(t):
    s = 1.70158
    return t * t * ((s + 1) * t - s)

def easeInOutSinSquared(t):
    if t < 0.5:
        return 0.5 * (1 - math.cos(t * 2 * math.pi))
    else:
        return 0.5 * (1 + math.cos((t - 0.5) * 2 * math.pi))

def easeOutBack(t):
    s = 1.70158
    return ((t - 1) * t * ((s + 1) * t + s)) + 1

def easeInOutBack(t):
    s = 1.70158 * 1.525
    if t < 0.5:
        return (t * t * (t * (s + 1) - s)) * 2
    return ((t - 2) * t * ((s + 1) * t + s) + 2) * 2

# Elastic easing functions
def easeInElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return -(math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p))

def easeOutElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3
    s = p / 4
    return math.pow(2, -10 * t) * math.sin((t - s) * (2 * math.pi) / p) + 1

def easeInOutElastic(t):
    if t == 0:
        return 0
    if t == 1:
        return 1
    p = 0.3 * 1.5
    s = p / 4
    t = t * 2
    if t < 1:
        return -0.5 * (
            math.pow(2, 10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
        )
    return (
        0.5 * math.pow(2, -10 * (t - 1)) * math.sin((t - 1 - s) * (2 * math.pi) / p)
        + 1
    )

# Bounce easing functions
def easeInBounce(t):
    return 1 - easeOutBounce(1 - t)

def easeOutBounce(t):
    if t < (1 / 2.75):
        return 7.5625 * t * t
    elif t < (2 / 2.75):
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < (2.5 / 2.75):
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375

def easeInOutBounce(t):
    if t < 0.5:
        return easeInBounce(t * 2) * 0.5
    return easeOutBounce(t * 2 - 1) * 0.5 + 0.5

# Quart easing functions
def easeInQuart(t):
    return t * t * t * t

def easeOutQuart(t):
    t -= 1
    return -(t**2 * t * t - 1)

def easeInOutQuart(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t * t
    t -= 2
    return -0.5 * (t**2 * t * t - 2)

# Cubic easing functions
def easeInCubic(t):
    return t * t * t

def easeOutCubic(t):
    t -= 1
    return t**2 * t + 1

def easeInOutCubic(t):
    t *= 2
    if t < 1:
        return 0.5 * t * t * t
    t -= 2
    return 0.5 * (t**2 * t + 2)

# Circ easing functions
def easeInCirc(t):
    return -(math.sqrt(1 - t * t) - 1)

def easeOutCirc(t):
    t -= 1
    return math.sqrt(1 - t**2)

def easeInOutCirc(t):
    t *= 2
    if t < 1:
        return -0.5 * (math.sqrt(1 - t**2) - 1)
    t -= 2
    return 0.5 * (math.sqrt(1 - t**2) + 1)

# Sine easing functions
def easeInSine(t):
    return -math.cos(t * (math.pi / 2)) + 1

def easeOutSine(t):
    return math.sin(t * (math.pi / 2))

def easeInOutSine(t):
    return -0.5 * (math.cos(math.pi * t) - 1)

def easeLinear(t):
    return t

easing_functions = {
    "Linear": easeLinear,
    "Sine_In": easeInSine,
    "Sine_Out": easeOutSine,
    "Sine_InOut": easeInOutSine,
    "Sin_Squared": easeInOutSinSquared,
    "Quart_In": easeInQuart,
    "Quart_Out": easeOutQuart,
    "Quart_InOut": easeInOutQuart,
    "Cubic_In": easeInCubic,
    "Cubic_Out": easeOutCubic,
    "Cubic_InOut": easeInOutCubic,
    "Circ_In": easeInCirc,
    "Circ_Out": easeOutCirc,
    "Circ_InOut": easeInOutCirc,
    "Back_In": easeInBack,
    "Back_Out": easeOutBack,
    "Back_InOut": easeInOutBack,
    "Elastic_In": easeInElastic,
    "Elastic_Out": easeOutElastic,
    "Elastic_InOut": easeInOutElastic,
    "Bounce_In": easeInBounce,
    "Bounce_Out": easeOutBounce,
    "Bounce_InOut": easeInOutBounce,
}

EASING_TYPES = list(easing_functions.keys())

def apply_easing(value, easing_type):
    function_ease = easing_functions.get(easing_type)
    if function_ease:
        return function_ease(value)
    
    raise ValueError(f"Unknown easing type: {easing_type}")



# endregion---------------插值算法------------------------



#region  ----------------math 算法-------------------------------------------------------------#


DEFAULT_FLOAT = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 9999.0, "step": 0.001,})

FLOAT_UNARY_OPERATIONS: Mapping[str, Callable[[float], float]] = {
    "Neg": lambda a: -a,
    "Inc": lambda a: a + 1,
    "Dec": lambda a: a - 1,
    "Abs": lambda a: abs(a),
    "Sqr": lambda a: a * a,
    "Cube": lambda a: a * a * a,
    "Sqrt": lambda a: math.sqrt(a),
    "Exp": lambda a: math.exp(a),
    "Ln": lambda a: math.log(a),
    "Log10": lambda a: math.log10(a),
    "Log2": lambda a: math.log2(a),
    "Sin": lambda a: math.sin(a),
    "Cos": lambda a: math.cos(a),
    "Tan": lambda a: math.tan(a),
    "Asin": lambda a: math.asin(a),
    "Acos": lambda a: math.acos(a),
    "Atan": lambda a: math.atan(a),
    "Sinh": lambda a: math.sinh(a),
    "Cosh": lambda a: math.cosh(a),
    "Tanh": lambda a: math.tanh(a),
    "Asinh": lambda a: math.asinh(a),
    "Acosh": lambda a: math.acosh(a),
    "Atanh": lambda a: math.atanh(a),
    "Round": lambda a: round(a),
    "Floor": lambda a: math.floor(a),
    "Ceil": lambda a: math.ceil(a),
    "Trunc": lambda a: math.trunc(a),
    "Erf": lambda a: math.erf(a),
    "Erfc": lambda a: math.erfc(a),
    "Gamma": lambda a: math.gamma(a),
    "Radians": lambda a: math.radians(a),
    "Degrees": lambda a: math.degrees(a),
}

FLOAT_UNARY_CONDITIONS: Mapping[str, Callable[[float], bool]] = {
    "IsZero": lambda a: a == 0.0000,
    "IsPositive": lambda a: a > 0.000,
    "IsNegative": lambda a: a < 0.000,
    "IsNonZero": lambda a: a != 0.000,
    "IsPositiveInfinity": lambda a: math.isinf(a) and a > 0.000,
    "IsNegativeInfinity": lambda a: math.isinf(a) and a < 0.000,
    "IsNaN": lambda a: math.isnan(a),
    "IsFinite": lambda a: math.isfinite(a),
    "IsInfinite": lambda a: math.isinf(a),
    "IsEven": lambda a: a % 2 == 0.000,
    "IsOdd": lambda a: a % 2 != 0.000,
}

FLOAT_BINARY_OPERATIONS: Mapping[str, Callable[[float, float], float]] = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a / b,
    "Ceil": lambda a, b: math.ceil(a / b),
    "Mod": lambda a, b: a % b,
    "Pow": lambda a, b: a**b,
    "FloorDiv": lambda a, b: a // b,
    "Max": lambda a, b: max(a, b),
    "Min": lambda a, b: min(a, b),
    "Log": lambda a, b: math.log(a, b),
    "Atan2": lambda a, b: math.atan2(a, b),
}

FLOAT_BINARY_CONDITIONS: Mapping[str, Callable[[float, float], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Gte": lambda a, b: a >= b,
    "Lt": lambda a, b: a < b,
    "Lte": lambda a, b: a <= b,
}


DEFAULT_INT = ("INT", {"default": 0})

INT_UNARY_OPERATIONS: Mapping[str, Callable[[int], int]] = {
    "Abs": lambda a: abs(a),
    "Neg": lambda a: -a,
    "Inc": lambda a: a + 1,
    "Dec": lambda a: a - 1,
    "Sqr": lambda a: a * a,
    "Cube": lambda a: a * a * a,
    "Not": lambda a: ~a,
    "Factorial": lambda a: math.factorial(a),
}

INT_UNARY_CONDITIONS: Mapping[str, Callable[[int], bool]] = {
    "IsZero": lambda a: a == 0,
    "IsNonZero": lambda a: a != 0,
    "IsPositive": lambda a: a > 0,
    "IsNegative": lambda a: a < 0,
    "IsEven": lambda a: a % 2 == 0,
    "IsOdd": lambda a: a % 2 == 1,
}

INT_BINARY_OPERATIONS: Mapping[str, Callable[[int, int], int]] = {
    "Add": lambda a, b: a + b,
    "Sub": lambda a, b: a - b,
    "Mul": lambda a, b: a * b,
    "Div": lambda a, b: a // b,
    "Ceil": lambda a, b: math.ceil(a / b),
    "Mod": lambda a, b: a % b,
    "Pow": lambda a, b: a**b,
    "And": lambda a, b: a & b,
    "Nand": lambda a, b: ~a & b,
    "Or": lambda a, b: a | b,
    "Nor": lambda a, b: ~a & b,
    "Xor": lambda a, b: a ^ b,
    "Xnor": lambda a, b: ~a ^ b,
    "Shl": lambda a, b: a << b,
    "Shr": lambda a, b: a >> b,
    "Max": lambda a, b: max(a, b),
    "Min": lambda a, b: min(a, b),
}

INT_BINARY_CONDITIONS: Mapping[str, Callable[[int, int], bool]] = {
    "Eq": lambda a, b: a == b,
    "Neq": lambda a, b: a != b,
    "Gt": lambda a, b: a > b,
    "Lt": lambda a, b: a < b,
    "Geq": lambda a, b: a >= b,
    "Leq": lambda a, b: a <= b,
}

# endregion-------------------------math 算法--------------------------------------------------#



#region------------------调度数据-AD--sch prompt  def----------------------------------------------------#

def adapt_to_batch(value, frame):    # 适应批处理数据处理，value对应当前的帧数
    if not hasattr(value, '__iter__'):
        value = [value]
    if frame < len(value):
        value = value[:frame]
    elif frame > len(value):
        last_value = value[-1]
        value = value + [last_value] * (frame - len(value))
    return value




DefaultPromp = """0: a girl @Sine_In@
7: a boy
15: a dog
"""



DefaultValue = """0:0.5 @Sine_In@
30:1
60:0.5
90:1
120:0.5
"""



def lerp_tensors(tensor_from: torch.Tensor, tensor_to: torch.Tensor, weight: float):
    return tensor_from * (1.0 - weight) + tensor_to * weight


@dataclass
class PromptKeyframe:
    index: int
    prompt: str
    interp_method: str = "linear"


class ValueKeyframe:
    def __init__(self, index, value, interp_method):
        self.index = index
        self.value = value
        self.interp_method = interp_method


def parse_prompt_schedule(text: str, easing_type="Linear"):
    keyframes = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip()
        if not line or ':' not in line:
            continue
        idx_part, prompt_part = line.split(':', 1)
        idx = int(idx_part.strip())
        prompt = prompt_part.strip()

        interp_method = easing_type
        match = re.search(r'@([a-zA-Z0-9_ ]+)@', prompt)
        if match:
            custom_ease = match.group(1).strip()
            if custom_ease in easing_functions:
                interp_method = custom_ease
            prompt = re.sub(r'\s*@([a-zA-Z0-9_ ]+)@\s*', ' ', prompt).strip()  # 移除标签

        if not prompt:
            continue
        keyframes.append(PromptKeyframe(index=idx, prompt=prompt, interp_method=interp_method))

    return sorted(keyframes, key=lambda x: x.index)


def build_conditioning(keyframes, clip, max_length, f_text="", b_text=""):
    if len(keyframes) == 0:
        raise ValueError("No valid keyframes found.")

    if max_length <= 0:
        max_length = keyframes[-1].index + 1

    conds = [None] * max_length
    pooleds = [None] * max_length

    prev_idx, prev_cond, prev_pooled = None, None, None

    pbar = ProgressBar(max_length)

    all_weights = [None] * max_length  # 存储所有帧的权重用于绘图

    for i, kf in enumerate(keyframes):
        curr_idx, curr_prompt, curr_method = kf.index, kf.prompt, kf.interp_method
        if curr_idx >= max_length:
            break

        # 添加前后缀
        prefix = f"{f_text}, " if f_text else ""
        suffix = f", {b_text}" if b_text else ""
        full_prompt = f"{prefix}{curr_prompt}{suffix}".strip()

        # tokenize and encode prompt
        tokens = clip.tokenize(full_prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conds[curr_idx] = cond
        pooleds[curr_idx] = pooled
        pbar.update(1)

        # 插值逻辑：当前帧和上一帧之间
        if prev_idx is not None and prev_idx < curr_idx:
            diff_len = curr_idx - prev_idx
            weights = torch.linspace(0, 1, diff_len + 1)[1:-1]
            easing_weights = [apply_easing(w.item(), curr_method) for w in weights]

            print(f"Frame {prev_idx} to {curr_idx}, method: {curr_method}")
            print("Raw weights:     ", [round(w.item(), 3) for w in weights])
            print("Easing weights:  ", [round(w, 3) for w in easing_weights])

            for j, w in enumerate(easing_weights):
                idx = prev_idx + j + 1
                if idx >= max_length:
                    break
                conds[idx] = lerp_tensors(prev_cond, cond, w)
                pooleds[idx] = pooleds[idx - 1]
                all_weights[idx] = w  # 记录当前帧的权重

        # 更新 prev 变量
        prev_idx, prev_cond, prev_pooled = curr_idx, cond, pooled
        all_weights[curr_idx] = 1.0  # 当前帧权重设为1

    # 填充开头和结尾缺失帧
    first_valid = next((i for i in range(max_length) if conds[i] is not None), None)
    last_valid = None
    for i in range(max_length):
        if conds[i] is not None:
            last_valid = i
        elif last_valid is not None:
            conds[i] = conds[last_valid]
            pooleds[i] = pooleds[last_valid]
            all_weights[i] = all_weights[last_valid]

    if first_valid is not None:
        for i in range(first_valid):
            conds[i] = conds[first_valid]
            pooleds[i] = pooleds[first_valid]
            all_weights[i] = all_weights[first_valid]

    final_cond = torch.cat(conds, dim=0)
    final_pooled_dict = {"pooled_output": torch.cat(pooleds, dim=0)}

    return [[final_cond, final_pooled_dict]]


def generate_frame_weight_curve_image(keyframes, max_length):
    current_weights = [None] * max_length  # 当前帧影响权重
    previous_weights = [None] * max_length  # 上一帧影响权重

    prev_idx = None

    for kf in keyframes:
        curr_idx = kf.index
        if prev_idx is not None and prev_idx < curr_idx:
            diff_len = curr_idx - prev_idx
            weights = torch.linspace(0, 1, diff_len + 1)[1:-1]
            easing_weights = [apply_easing(w.item(), kf.interp_method) for w in weights]

            for j, w in enumerate(easing_weights):
                idx = prev_idx + j + 1
                if idx >= max_length:
                    break
                current_weights[idx] = w           # 当前帧权重 (curr)
                previous_weights[idx] = 1.0 - w    # 上一帧权重 (prev)

        prev_idx = curr_idx
        current_weights[curr_idx] = 1.0
        previous_weights[curr_idx] = 0.0

    # 补全缺失帧（保持连续性）
    def fill_weights(weights_list):
        last_valid = None
        for i in range(len(weights_list)):
            if weights_list[i] is not None:
                last_valid = i
            elif last_valid is not None:
                weights_list[i] = weights_list[last_valid]

    fill_weights(current_weights)
    fill_weights(previous_weights)

    # 如果开头没有值，则复制第一个有效值
    first_valid = next((i for i, w in enumerate(current_weights) if w is not None), None)
    if first_valid is not None:
        for i in range(first_valid):
            current_weights[i] = current_weights[first_valid]
            previous_weights[i] = previous_weights[first_valid]

    # 转换为浮点数
    y_current = [w if w is not None else 0.0 for w in current_weights]
    y_previous = [w if w is not None else 0.0 for w in previous_weights]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y_previous)), y_previous, marker='x', linestyle='--', color='blue', markersize=2, label='Prev Frame Weight')
    plt.plot(range(len(y_current)), y_current, marker='o', linestyle='-', color='green', markersize=2, label='Curr Frame Weight')
    plt.title("Interpolation Weights per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image = Image.open(buffer)
    return pil2tensor(image)


def generate_value_curve_image_with_data(values_seq, max_length, frame_methods=None):
    y = [v if v is not None else np.nan for v in values_seq]

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(y)), y, marker='o', linestyle='-', markersize=3, label="Value Curve")

    # 如果有插值区间信息，则绘制缓动标记
    if frame_methods:
        for start, end, method in frame_methods:
            plt.axvspan(start, end, alpha=0.1, color='gray', label=f"{method} Interpolation" if start == 0 else "")

    plt.title("Interpolated Value Curve per Frame")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend(loc="upper left")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()
    buffer.seek(0)
    image = Image.open(buffer)
    return pil2tensor(image)



#endregion---------------调度数据----------------------------------------------------#



#region------------------风格处理----------------------------------------------------#
def style_list():   # 读取风格csv文件
    current_dir = os.path.dirname(__file__)
    csv_dir = os.path.join(current_dir, "csv")
    file_path = os.path.join(csv_dir, "styles.csv")
    with open(file_path, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        data_list = [row for row in reader]
    my_path = os.path.join(csv_dir, "my_styles.csv")
    if os.path.exists(my_path):
        with open(my_path, mode="r", encoding="utf-8") as file:
            reader = csv.reader(file)
            my_data_list = [row for row in reader]
        data_list = my_data_list + data_list
    
    card_list = []
    for i in data_list:
        card_list += [i[0]]
    return (card_list, data_list)




def add_style_to_subject(style, positive, negative):
    if style != "None":
        style_info = style_list()
        style_index = style_info[0].index(style)
        style_text = style_info[1][style_index][1]

        if "{prompt}" in style_text:
            positive = style_text.replace("{prompt}", positive)
        else:
            positive += f", {style_text}"

        if len(style_info[1][style_index]) > 2:
            negative += f", {style_info[1][style_index][2]}"

    return positive, negative

#endregion------------------风格处理----------------------------------------------------#




#region------------------类型转换----------------------------------------------------#  





def to_numpy(image: torch.Tensor) -> npt.NDArray[np.uint8]:
    np_array = np.clip(255.0 * image.cpu().numpy(), 0, 255).astype(np.uint8)
    return np_array





def try_cast(x, dst_type: str): #按值的类型返回
    result = x
    if dst_type == "STRING":
        result = str(x)
    elif dst_type == "INT":
        result = int(x)
    elif dst_type == "FLOAT" or dst_type == "NUMBER":
        result = float(x)
    elif dst_type == "BOOLEAN":
        if isinstance(x, numbers.Number):
            if x > 0:
                result = True
            else:
                result = False
        elif isinstance(x, str):
            try:
                x = float(x)
                if x > 0:
                    result = True
                else:
                    result = False
            except:
                result = bool(x)
        else:
            result = bool(x)
    return result


def pil2tensor(image):  #多维度的图像也可以
    np_image = np.array(image).astype(np.float32) / 255.0
    if np_image.ndim == 2:
        np_image = np_image[None, None, ...]
    elif np_image.ndim == 3:
        np_image = np_image[None, ...]
    return torch.from_numpy(np_image)


def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def tensor_to_pillow(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pillow_to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)




def list_pil2tensor(images: Image.Image | list[Image.Image]) -> torch.Tensor:
    def single_pil2tensor(image: Image.Image) -> torch.Tensor:
        np_image = np.array(image).astype(np.float32) / 255.0
        if np_image.ndim == 2:  # Grayscale
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W)
        else:  # RGB or RGBA
            return torch.from_numpy(np_image).unsqueeze(0)  # (1, H, W, C)
    if isinstance(images, Image.Image):
        return single_pil2tensor(images)
    else:
        return torch.cat([single_pil2tensor(img) for img in images], dim=0)


def list_tensor2pil(tensor: torch.Tensor) -> list[Image.Image]: 
    def single_tensor2pil(t: torch.Tensor) -> Image.Image:
        np_array = to_numpy(t)
        if np_array.ndim == 2:  # (H, W) for masks
            return Image.fromarray(np_array, mode="L")
        elif np_array.ndim == 3:  # (H, W, C) for RGB/RGBA
            if np_array.shape[2] == 1:  # 处理 [H, W, 1] 形状的张量
                return Image.fromarray(np_array.squeeze(-1), mode="L")
            elif np_array.shape[2] == 3:
                return Image.fromarray(np_array, mode="RGB")
            elif np_array.shape[2] == 4:
                return Image.fromarray(np_array, mode="RGBA")
        raise ValueError(f"Invalid tensor shape: {t.shape}")
    return handle_batch(tensor, single_tensor2pil)


def handle_batch(       #输入张量进行批量处理，将指定的转换函数应用到张量的每个元素上，并返回处理结果的列表
    tensor: torch.Tensor,
    func: Callable[[torch.Tensor], Image.Image | npt.NDArray[np.uint8]],
) -> list[Image.Image] | list[npt.NDArray[np.uint8]]:
    """Handles batch processing for a given tensor and conversion function."""
    return [func(tensor[i]) for i in range(tensor.shape[0])]






#endregion------------------类型转换----------------------------------------------------#



#region------------------图像处理----------------------------------------------------#

def convert_pil_image(image):     
    batch_size = image.shape[0]
    converted_images = []       
    for i in range(batch_size):
        single_image = image[i]  # (H, W, C)     
        pil_image = tensor2pil(single_image)
        converted_image = pil2tensor(pil_image)       
        converted_images.append(converted_image)
    image = torch.cat(converted_images, dim=0) 
    return image


def latentrepeat(samples, amount):
    try:
        # 复制原始样本字典，避免修改原始数据
        s = samples.copy()
        s_in = samples["samples"]

        # 动态生成重复维度
        num_dims = s_in.dim()
        if num_dims < 4:
            raise ValueError(f"输入张量的维度为 {num_dims}，期望至少为 4 维。")
        repeat_dims = (amount,) + (1,) * (num_dims - 1)
        s["samples"] = s_in.repeat(repeat_dims)

        # 处理 noise_mask
        if "noise_mask" in samples and samples["noise_mask"].shape[0] > 1:
            masks = samples["noise_mask"]
            if masks.shape[0] < s_in.shape[0]:
                repeat_factor = math.ceil(s_in.shape[0] / masks.shape[0])
                masks = masks.repeat(repeat_factor, 1, 1, 1)[:s_in.shape[0]]
            # 动态生成 noise_mask 的重复维度
            mask_num_dims = masks.dim()
            if mask_num_dims < 4:
                raise ValueError(f"noise_mask 张量的维度为 {mask_num_dims}，期望至少为 4 维。")
            mask_repeat_dims = (amount,) + (1,) * (mask_num_dims - 1)
            s["noise_mask"] = masks.repeat(mask_repeat_dims)

        # 处理 batch_index
        if "batch_index" in s:
            offset = max(s["batch_index"]) - min(s["batch_index"]) + 1
            additional_indices = [x + (i * offset) for i in range(1, amount) for x in s["batch_index"]]
            s["batch_index"] += additional_indices

        return (s,)
    except Exception as e:
        print(f"在 latentrepeat 函数中发生错误: {str(e)}")
        return (samples,)  # 发生错误时返回原始样本


def read_ratios():           # 读取ratios.json文件,生成图像的尺寸
    current_dir = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(current_dir, 'web')
    file_path = os.path.join(p, 'ratios.json')
    
    # 显式指定使用 utf-8 编码打开文件
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    ratio_sizes = list(data['ratios'].keys())
    ratio_dict = data['ratios']
    return ratio_sizes, ratio_dict


def easySave(images, filename_prefix, output_type, prompt=None, extra_pnginfo=None):  # 预览、保存图片

    if output_type in ["Hide", "None"]:
        return list()
    elif output_type in ["Preview", "Preview&Choose"]:
        filename_prefix = 'easyPreview'
        results = PreviewImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']
    else:
        results = SaveImage().save_images(images, filename_prefix, prompt, extra_pnginfo)
        return results['ui']['images']



def append_helper(t, mask, c, set_area_to_bounds, strength): #多文本条件
    if mask is not None:  
        n = [t[0], t[1].copy()]
        _, h, w = mask.shape
        n[1]['mask'] = mask
        n[1]['set_area_to_bounds'] = set_area_to_bounds
        n[1]['mask_strength'] = strength
        c.append(n)


def rescale(samples, width, height, algorithm):

    if len(samples.shape) != 4:
        raise ValueError(f"Expected 4D tensor, but got shape {samples.shape}")
    if width <= 0 or height <= 0:
        raise ValueError(f"Width and height must be > 0, but got {width}, {height}")
    if samples.shape[2] == height and samples.shape[3] == width:
        return samples

    if algorithm == "nearest":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="nearest")
    elif algorithm == "bilinear":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bilinear")
    elif algorithm == "bicubic":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="bicubic")
    elif algorithm == "area":
        return torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
    elif algorithm == "bislerp":
        return comfy.utils.bislerp(samples, width, height)

    return samples



def upscale_with_model(upscale_model, image):
    device = model_management.get_torch_device()
    upscale_model.to(device)
    in_img = image.movedim(-1,-3).to(device)
    free_memory = model_management.get_free_memory(device)

    tile = 512
    overlap = 32

    oom = True
    while oom:
        try:
            steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
            pbar = comfy.utils.ProgressBar(steps)
            s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
            oom = False
        except model_management.OOM_EXCEPTION as e:
            tile //= 2
            if tile < 128:
                raise e

    upscale_model.cpu()
    s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
    return s        


def image_upscale(image, upscale_method, scale_by):
    samples = image.movedim(-1,1)
    width = round(samples.shape[3] * scale_by)
    height = round(samples.shape[2] * scale_by)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return (s,)


def upscale(image, upscale_method, width,height):
    samples = image.movedim(-1,1)
    s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
    s = s.movedim(1,-1)
    return (s,)


def image_2dtransform(             # 2D图像变换
        image,
        x,
        y,
        zoom,
        angle,
        shear=0,
        border_handling="edge",
    ):
        x = int(x)
        y = int(y)
        angle = int(angle)

        if image.size(0) == 0:
            return (torch.zeros(0),)
        frames_count, frame_height, frame_width, frame_channel_count = image.size()

        new_height, new_width = int(frame_height * zoom), int(frame_width * zoom)

        # - Calculate diagonal of the original image
        diagonal = math.sqrt(frame_width**2 + frame_height**2)
        max_padding = math.ceil(diagonal * zoom - min(frame_width, frame_height))
        # Calculate padding for zoom
        pw = int(frame_width - new_width)
        ph = int(frame_height - new_height)

        pw += abs(max_padding)
        ph += abs(max_padding)

        padding = [max(0, pw + x), max(0, ph + y), max(0, pw - x), max(0, ph - y)]

        img = tensor2pil(image)

        img = TF.pad(
            img,  # transformed_frame,
            padding=padding,
            padding_mode=border_handling,
        )

        img = cast(
            Image.Image,
            TF.affine(img, angle=angle, scale=zoom, translate=[x, y], shear=shear),
        )

        left = abs(padding[0])
        upper = abs(padding[1])
        right = img.width - abs(padding[2])
        bottom = img.height - abs(padding[3])

        img = img.crop((left, upper, right, bottom))

        return pil2tensor(img)





def get_image_resize(image, target_image=None):  # 中心裁切，输入和输出都是4维张量或都是PIL
    # 检查输入是否为PIL图像
    input_was_pil = isinstance(image, Image.Image)
    target_was_pil = isinstance(target_image, Image.Image) if target_image is not None else False   
    # 转换PIL图像为张量
    if input_was_pil:
        image = pil2tensor(image)
    
    if target_was_pil and target_image is not None:
        target_image = pil2tensor(target_image)  
    # 原始处理逻辑
    B, H, W, C = image.shape
    upscale_method = "bicubic"
    crop = "center"  
    if target_image is None:
        width = W - (W % 8)
        height = H - (H % 8)
        image = image.movedim(-1, 1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1, -1)
        result = image
    else:
        _, height, width, _ = target_image.shape
        image = image.movedim(-1, 1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1, -1)
        result = image
    # 如果原始输入是PIL图像，转换回PIL图像
    if input_was_pil:
        result = tensor2pil(result)
    
    return result








#endregion------------------图像处理----------------------------------------------------#



#region------------------颜色处理----------------------------------------------------#


def hex_to_float(color):
    if not isinstance(color, str):
        raise ValueError("Color must be a hex string")
    color = color.strip("#")
    return int(color, 16) / 255.0


def hex_to_rgba(hex_color, alpha): 
    r, g, b = hex_to_rgb(hex_color)
    return (r, g, b, int(alpha * 255 / 100))



def hex_to_rgb(inhex:str) -> tuple:
    if not inhex.startswith('#'):
        raise ValueError(f'Invalid Hex Code in {inhex}')
    else:
        if len(inhex) == 4:
            inhex = "#" + "".join([char * 2 for char in inhex[1:]])
        rval = inhex[1:3]
        gval = inhex[3:5]
        bval = inhex[5:]
        rgb = (int(rval, 16), int(gval, 16), int(bval, 16))
    return tuple(rgb)







def hex_to_rgb_tuple(color):   # 处理 3 位和 6 位十六进制颜色字符串作为输入，例如 "#F00" 和 "#FF0000"
    if isinstance(color, list) and len(color) == 3:
        return tuple(int(c * 255) for c in color)
    elif isinstance(color, str):
        color = color.lstrip('#')
        if len(color) == 6:
            return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        elif len(color) == 3:
            return tuple(int(c * 2, 16) for c in color)
    raise ValueError(f"不支持的颜色格式: {color}")


color_mapping = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "lightgray": (211, 211, 211),
    "darkgray": (169, 169, 169),
    "olive": (128, 128, 0),
    "lime": (0, 128, 0),
    "teal": (0, 128, 128),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "fuchsia": (255, 0, 128),
    "aqua": (0, 255, 128),
    "silver": (192, 192, 192),
    "gold": (255, 215, 0),
    "turquoise": (64, 224, 208),
    "lavender": (230, 230, 250),
    "violet": (238, 130, 238),
    "coral": (255, 127, 80),
    "indigo": (75, 0, 130),    
}


COLORSMAP = ["white", "black", "red", "green", "blue", "yellow",
        "cyan", "magenta", "orange", "purple", "pink", "brown", "gray",
        "lightgray", "darkgray", "olive", "lime", "teal", "navy", "maroon",
        "fuchsia", "aqua", "silver", "gold", "turquoise", "lavender",
        "violet", "coral", "indigo"]


STYLES = ["Accent","afmhot","autumn","binary","Blues","bone","BrBG","brg",
    "BuGn","BuPu","bwr","cividis","CMRmap","cool","coolwarm","copper","cubehelix","Dark2","flag",
    "gist_earth","gist_gray","gist_heat","gist_rainbow","gist_stern","gist_yarg","GnBu","gnuplot","gnuplot2","gray","Greens",
    "Greys","hot","hsv","inferno","jet","magma","nipy_spectral","ocean","Oranges","OrRd",
    "Paired","Pastel1","Pastel2","pink","PiYG","plasma","PRGn","prism","PuBu","PuBuGn",
    "PuOr","PuRd","Purples","rainbow","RdBu","RdGy","RdPu","RdYlBu","RdYlGn","Reds","seismic",
    "Set1","Set2","Set3","Spectral","spring","summer","tab10","tab20","tab20b","tab20c","terrain",
    "turbo","twilight","twilight_shifted","viridis","winter","Wistia","YlGn","YlGnBu","YlOrBr","YlOrRd"]


# endregion------------------颜色处理----------------------------------------------------#



#region------------------文本处理----------------------------------------------------#





def split_with_quotes(s):
    pattern = r'"([^"]*)"|\s*([^,]+)'
    matches = re.finditer(pattern, s)
    return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]

def single_replace(text, target, replacement):
    if not target or not replacement:
        return text
    target_clean = target.strip('"').strip()
    replacement_clean = replacement.strip('"').strip()
    pattern = re.escape(target_clean)
    return re.sub(pattern, replacement_clean, text)


def multi_replace(text, multi_targets, multi_replacements):
    if not multi_targets or not multi_replacements:
        return text
    targets = multi_targets.split('@')
    replacements = multi_replacements.split('@')
    min_len = min(len(targets), len(replacements))
    targets = targets[:min_len]
    replacements = replacements[:min_len]
    result = text
    for target, replacement in zip(targets, replacements):
        result = single_replace(result, target.strip(), replacement.strip())
    return result






def xxxxreplace_text(text, target, replace):
    def split_with_quotes(s):
        pattern = r'"([^"]*)"|\s*([^,]+)'
        matches = re.finditer(pattern, s)
        return [match.group(1) or match.group(2).strip() for match in matches if match.group(1) or match.group(2).strip()]
    
    targets = split_with_quotes(target)
    exchanges = split_with_quotes(replace)
    
    word_map = {}
    for target_item, exchange_item in zip(targets, exchanges):
        target_clean = target_item.strip('"').strip().lower()
        exchange_clean = exchange_item.strip('"').strip()
        word_map[target_clean] = exchange_clean

    sorted_targets = sorted(word_map.keys(), key=len, reverse=True)
    result = text
    
    for target_item in sorted_targets:
        if ' ' in target_item:
            pattern = re.escape(target_item)
        else:
            pattern = r'\b' + re.escape(target_item) + r'\b'
        
        result = re.sub(pattern, word_map[target_item], result, flags=re.IGNORECASE)
    return result  # 直接返回字符串



def clean_prompt(text, words_to_remove):
    remove_words = [word.strip().lower() for word in words_to_remove.split('@')]
    words = re.findall(r'\b\w+\b|[^\w\s]', text)

    cleaned_words = []
    for i, word in enumerate(words):
        word_lower = word.lower()
        if word_lower in remove_words:
            continue
        cleaned_words.append(word)
    cleaned_text = ' '.join(cleaned_words)
    return cleaned_text  # 直接返回字符串
#endregion------------------文本处理----------------------------------------------------#



#region------------------图层混合----------------------------------------------------#
# 定义混合模式的静态列表
BLEND_METHODS = [
    "normal", "multiply", "screen", "overlay", "darken", "lighten","color_dodge", "color_burn", "linear_dodge", "linear_burn",
    "hard_light", "soft_light", "vivid_light", "linear_light","pin_light", "difference", "exclusion", "subtract"]

BLEND_METHODS = [
    "normal", "multiply", "screen", "overlay", "darken", "lighten","color_dodge", "color_burn", "linear_dodge", "linear_burn",
    "hard_light", "soft_light", "vivid_light", "linear_light","pin_light", "difference", "exclusion", "subtract"]
def apply_blending_mode(bg, fg, mode, strength):
    bg_np = np.array(bg).astype(np.float32) / 255.0
    fg_np = np.array(fg).astype(np.float32) / 255.0
    if bg_np.shape[-1] == 3: bg_np = np.dstack([bg_np, np.ones(bg_np.shape[:2], dtype=np.float32)])
    if fg_np.shape[-1] == 3: fg_np = np.dstack([fg_np, np.ones(fg_np.shape[:2], dtype=np.float32)])
    bg_rgb, bg_a = bg_np[..., :3], bg_np[..., 3]
    fg_rgb, fg_a = fg_np[..., :3], fg_np[..., 3]
    if mode == "normal": blended_rgb = fg_rgb
    elif mode == "multiply": blended_rgb = bg_rgb * fg_rgb
    elif mode == "screen": blended_rgb = 1 - (1 - bg_rgb) * (1 - fg_rgb)
    elif mode == "overlay": blended_rgb = np.where(bg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
    elif mode == "darken": blended_rgb = np.minimum(bg_rgb, fg_rgb)
    elif mode == "lighten": blended_rgb = np.maximum(bg_rgb, fg_rgb)
    elif mode == "color_dodge": blended_rgb = np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (1 - fg_rgb)))
    elif mode == "color_burn": blended_rgb = np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / fg_rgb))
    elif mode == "linear_dodge": blended_rgb = np.minimum(1, bg_rgb + fg_rgb)
    elif mode == "linear_burn": blended_rgb = np.maximum(0, bg_rgb + fg_rgb - 1)
    elif mode == "hard_light": blended_rgb = np.where(fg_rgb <= 0.5, 2 * bg_rgb * fg_rgb, 1 - 2 * (1 - bg_rgb) * (1 - fg_rgb))
    elif mode == "soft_light": blended_rgb = np.where(fg_rgb <= 0.5, bg_rgb - (1 - 2 * fg_rgb) * bg_rgb * (1 - bg_rgb), bg_rgb + (2 * fg_rgb - 1) * (np.sqrt(bg_rgb) - bg_rgb))
    elif mode == "vivid_light": blended_rgb = np.where(fg_rgb <= 0.5, np.where(fg_rgb == 0, 0, np.maximum(0, 1 - (1 - bg_rgb) / (2 * fg_rgb))), np.where(fg_rgb == 1, 1, np.minimum(1, bg_rgb / (2 * (1 - fg_rgb)))))
    elif mode == "linear_light": blended_rgb = np.clip(bg_rgb + 2 * fg_rgb - 1, 0, 1)
    elif mode == "pin_light": blended_rgb = np.where(fg_rgb <= 0.5, np.minimum(bg_rgb, 2 * fg_rgb), np.maximum(bg_rgb, 2 * fg_rgb - 1))
    elif mode == "difference": blended_rgb = np.abs(bg_rgb - fg_rgb)
    elif mode == "exclusion": blended_rgb = bg_rgb + fg_rgb - 2 * bg_rgb * fg_rgb
    elif mode == "subtract": blended_rgb = np.maximum(0, bg_rgb - fg_rgb)
    else: blended_rgb = fg_rgb
    blended_rgb = blended_rgb * strength + bg_rgb * (1 - strength)
    blended_a = fg_a * strength + bg_a * (1 - strength)
    blended = np.dstack([blended_rgb, blended_a])
    out_img = Image.fromarray((blended * 255.0).astype(np.uint8))
    return out_img
#endregion------------------图像处理----------------------------------------------------#



#region------------------遮罩处理----------------------------------------------------#

def pil2mask(image, batch_dim=True, channel_dim=True):
    # 处理单张图像或批量图像
    if isinstance(image, Image.Image):
        images = [image]  # 转为列表处理
    else:
        images = image
    masks = []
    for img in images:
        img = img.convert("L")  # 转为灰度图
        arr = np.array(img).astype(np.float32) / 255.0
        masks.append(torch.from_numpy(arr))
    mask_tensor = torch.stack(masks, dim=0)  # 形状: (B, H, W)
    if channel_dim:
        mask_tensor = mask_tensor.unsqueeze(1)  # 形状: (B, 1, H, W)
    if not batch_dim and len(images) == 1:
        mask_tensor = mask_tensor.squeeze(0)  # 形状: (1, H, W) 或 (H, W)   
    return mask_tensor




def mask_crop(image, mask):              # 裁剪图片-PIL-------------------------------
    image_pil = tensor2pil(image)
    mask_pil = tensor2pil(mask)
    mask_array = np.array(mask_pil) > 0
    
    coords = np.where(mask_array)
    if coords[0].size == 0 or coords[1].size == 0:
        return (image, None)
    x0, y0, x1, y1 = coords[1].min(), coords[0].min(), coords[1].max(), coords[0].max()
    # 移除边界调整逻辑
    x0 = max(x0, 0)
    y0 = max(y0, 0)
    x1 = min(x1, image_pil.width)
    y1 = min(y1, image_pil.height)
    cropped_image_pil = image_pil.crop((x0, y0, x1, y1))
    cropped_mask_pil = mask_pil.crop((x0, y0, x1, y1))
    cropped_image_tensor = pil2tensor(cropped_image_pil)
    cropped_mask_tensor = pil2tensor(cropped_mask_pil)
    return (cropped_image_tensor, cropped_mask_tensor)




def set_mask(samples, mask):                   # 设置latent的mask
    s = samples.copy()
    s["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
    return s



def smoothness_mask(mask, smoothness=5):
    if isinstance(mask, torch.Tensor):
        input_shape = mask.shape
        input_device = mask.device
        mask_np = mask.squeeze().cpu().detach().numpy()
    elif isinstance(mask, np.ndarray):
        input_shape = mask.shape
        input_device = torch.device("cpu")
        mask_np = mask.squeeze()
    elif isinstance(mask, Image.Image):
        input_shape = (1,) + mask.size[::-1]
        input_device = torch.device("cpu")
        mask_np = np.array(mask).squeeze()
    else:
        raise TypeError(f"不支持的输入类型: {type(mask)}，仅支持 torch.Tensor、np.ndarray、PIL.Image")
    
    if mask_np.ndim == 3 and mask_np.shape[-1] in [3, 4]:
        mask_np = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY) if mask_np.shape[-1] == 3 else cv2.cvtColor(mask_np, cv2.COLOR_RGBA2GRAY)
    
    mask_np = mask_np / 255.0 if mask_np.max() > 1 else mask_np
    mask_np = np.clip(mask_np, 0, 1)
    H, W = mask_np.shape

    mask_binary = (mask_np > 0.5).astype(np.uint8)
    edges = cv2.Canny(mask_binary * 255, threshold1=100, threshold2=200)
    edges = (edges > 0).astype(np.uint8)
    
    if edges.sum() == 0:
        result = torch.tensor(mask_np[None, ...]).float().to(input_device)
        return result.reshape(input_shape) if isinstance(mask, torch.Tensor) else result

    distance = cv2.distanceTransform(1 - edges, distanceType=cv2.DIST_L2, maskSize=5)
    distance = distance / smoothness
    weight = np.clip(distance, 0, 1)[None, ...]

    mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
    blurred_np = np.array(mask_pil.filter(ImageFilter.GaussianBlur(smoothness))) / 255.0
    blurred_np = blurred_np[None, ...]

    feathered_np = blurred_np * (1 - weight) + mask_np[None, ...] * weight
    feathered_np = np.clip(feathered_np, 0, 1)

    result_tensor = torch.tensor(feathered_np).float().to(input_device)
    if isinstance(mask, torch.Tensor):
        if mask.ndim == 3 and mask.shape[0] > 1:
            result_tensor = result_tensor.repeat(mask.shape[0], 1, 1)
        result_tensor = result_tensor.reshape(input_shape)
    
    return result_tensor










def blur_and_expand_mask(mask_img, mask_blur, mask_expansion):   #图片遮罩膨胀、模糊处理---------------------
    mask = np.array(mask_img)
    if mask_expansion != 0:
        kernel = np.ones((abs(mask_expansion), abs(mask_expansion)), np.uint8)
        if mask_expansion > 0:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
    if mask_blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=mask_blur)
    mask_img = Image.fromarray(mask)
    return mask_img


def apply_extrant_to_block(mask_input):#输入兼容，输出总是图像
    # 统一转换为NumPy数组
    if isinstance(mask_input, Image.Image):
        mask_array = np.array(mask_input)
        if mask_array.ndim == 2:
            mask_array = np.expand_dims(mask_array, 0)  # (H, W) -> (1, H, W)
        else:
            # 如果是多通道图像，只取第一个通道
            mask_array = mask_array[:, :, 0] if mask_array.ndim == 3 else mask_array
            mask_array = np.expand_dims(mask_array, 0)  # (1, H, W)
    elif isinstance(mask_input, torch.Tensor):
        mask_tensor = mask_input
        if mask_tensor.dim() == 2:
            mask_tensor = mask_tensor.unsqueeze(0)  # (H, W) -> (1, H, W)
        mask_array = mask_tensor.cpu().numpy()
    else:  # 假设是np.ndarray
        if isinstance(mask_input, np.ndarray):
            mask_array = mask_input
            if mask_array.ndim == 2:
                mask_array = np.expand_dims(mask_array, 0)  # (H, W) -> (1, H, W)
        else:
            # 如果是其他类型，尝试转换
            mask_array = np.array(mask_input)
            if mask_array.ndim == 2:
                mask_array = np.expand_dims(mask_array, 0)
    
    # 处理批次中的每一张mask
    rectangular_masks = []
    for i in range(mask_array.shape[0]):
        single_mask = mask_array[i]  # (H, W)
        coords = np.argwhere(single_mask > 0)
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            rect_mask = np.zeros_like(single_mask)
            rect_mask[y_min:y_max+1, x_min:x_max+1] = 255
            rectangular_masks.append(Image.fromarray(rect_mask.astype(np.uint8)))
        else:
            rectangular_masks.append(Image.fromarray(single_mask.astype(np.uint8)))
    
    # 如果只有一个mask，直接返回该mask而不是包含一个元素的列表
    if len(rectangular_masks) == 1:
        return rectangular_masks[0]
    else:
        return rectangular_masks


#endregion------------------遮罩处理----------------------------------------------------#



#region------------------图像绘制形状----------------------------------------------------#
#该函数用于在图像上绘制指定形状。支持多种形状，如圆形、半圆形、正方形等
DRAW_SHAPE_LIST = ["circle", "semicircle", "quarter_circle", "ellipse", "square", "triangle","cross", "star", "radial"]

def draw_shape(shape, size=(200, 200), offset=(0, 0), scale=1.0, rotation=0, bg_color=(255, 255, 255),
               shape_color=(0, 0, 0), opacity=1.0, blur_radius=0, base_image=None):
    width, height = size
    offset_x, offset_y = offset
    center_x, center_y = width // 2 + offset_x, height // 2 + offset_y
    max_dim = min(width, height) * scale

    diagonal = int(math.sqrt(width ** 2 + height ** 2))
    img_tmp = Image.new('RGBA', (diagonal, diagonal), (0, 0, 0, 0))
    draw_tmp = ImageDraw.Draw(img_tmp)

    tmp_center = diagonal // 2

    alpha = int(opacity * 255)
    shape_color = shape_color + (alpha,)

    if shape == 'circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'semicircle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=180, fill=shape_color)

    elif shape == 'quarter_circle':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.pieslice(bbox, start=0, end=90, fill=shape_color)

    elif shape == 'ellipse':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 4, tmp_center + max_dim / 2, tmp_center + max_dim / 4)
        draw_tmp.ellipse(bbox, fill=shape_color)

    elif shape == 'square':
        bbox = (tmp_center - max_dim / 2, tmp_center - max_dim / 2, tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        draw_tmp.rectangle(bbox, fill=shape_color)

    elif shape == 'triangle':
        points = [
            (tmp_center, tmp_center - max_dim / 2),
            (tmp_center - max_dim / 2, tmp_center + max_dim / 2),
            (tmp_center + max_dim / 2, tmp_center + max_dim / 2)
        ]
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'cross':
        vertical = [(tmp_center - max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center - max_dim / 2),
                    (tmp_center + max_dim / 6, tmp_center + max_dim / 2),
                    (tmp_center - max_dim / 6, tmp_center + max_dim / 2)]
        horizontal = [(tmp_center - max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center - max_dim / 6),
                      (tmp_center + max_dim / 2, tmp_center + max_dim / 6),
                      (tmp_center - max_dim / 2, tmp_center + max_dim / 6)]
        draw_tmp.polygon(vertical, fill=shape_color)
        draw_tmp.polygon(horizontal, fill=shape_color)

    elif shape == 'star':
        points = []
        for i in range(10):
            angle = i * 36 * math.pi / 180
            radius = max_dim / 2 if i % 2 == 0 else max_dim / 4
            points.append((tmp_center + radius * math.sin(angle), tmp_center - radius * math.cos(angle)))
        draw_tmp.polygon(points, fill=shape_color)

    elif shape == 'radial':
        num_rays = 12
        for i in range(num_rays):
            angle = i * (360 / num_rays) * math.pi / 180
            x1 = tmp_center + max_dim / 4 * math.cos(angle)
            y1 = tmp_center + max_dim / 4 * math.sin(angle)
            x2 = tmp_center + max_dim / 2 * math.cos(angle)
            y2 = tmp_center + max_dim / 2 * math.sin(angle)
            draw_tmp.line([(x1, y1), (x2, y2)], fill=shape_color, width=int(max_dim / 20))

    img_tmp = img_tmp.rotate(rotation, resample=Image.BICUBIC, expand=True)
    if base_image is None:
        img = Image.new('RGBA', size, bg_color + (255,))
    else:
        img = base_image.copy()
        if img.mode != 'RGBA':
            img = img.convert('RGBA')

    paste_x = center_x - img_tmp.width // 2
    paste_y = center_y - img_tmp.height // 2

    img.alpha_composite(img_tmp, (paste_x, paste_y))

    if blur_radius > 0:
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    return img


#endregion------------------图像形状----------------------------------------------------#



#region------------------mask bridge----------------------------------------------------#

def tensor_to_hash(tensor):
    np_array = tensor.cpu().numpy()
    byte_data = np_array.tobytes()
    hash_value = hashlib.md5(byte_data).hexdigest()
    return hash_value


def create_temp_file(image):
    output_dir = folder_paths.get_temp_directory()
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path('material', output_dir)
    image = tensor2pil(image)
    image_file = f"{filename}_{counter:05}.png"
    image_path = os.path.join(full_output_folder, image_file)
    image.save(image_path, compress_level=4)
    return (image_path, [{"filename": image_file, "subfolder": subfolder, "type": "temp"}])



def generate_masked_black_image(image, mask):
    b, h, w = image.shape[0], image.shape[1], image.shape[2]
    if mask.dim() == 2:
        mask = mask.unsqueeze(0)
    if mask.shape != (b, h, w):
        mask = torch.zeros_like(image[:, :, :, 0])
    black_mask = torch.zeros_like(image)
    masked_image = image * (1 - mask.unsqueeze(-1)) + black_mask * mask.unsqueeze(-1)

    return {"ui": {"images": []}, "result": (masked_image,)}






#endregion------------------mask bridge----------------------------------------------------#




#region------------------lora----------------------------------------------------#

def apply_lora_stack( model, clip, lora_stack=None):
    if not lora_stack:
        return (model, clip,)

    model_lora = model
    clip_lora = clip

    for tup in lora_stack:
        lora_name, weight = tup  
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, weight, weight )

    return (model_lora, clip_lora,)


#endregion------------------lora----------------------------------------------------#




#region------------------vae----------------------------------------------------#
def decode(vae, samples):
    images = vae.decode(samples["samples"])
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return (images, )


def encode(vae, pixels):
    t = vae.encode(pixels[:,:,:,:3])
    return ({"samples":t}, )

#endregion------------------vae----------------------------------------------------#


#region------------------latent----------------------------------------------------#


def set_latent_mask2(latent, mask):
    newlatent = latent.clone()  # 克隆 latent
    if mask is not None:
        # 将 mask reshape 成合适的形状 (batch, 1, H, W)
        processed_mask = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        # 返回包含 samples 和 noise_mask 的字典
        return {"samples": newlatent, "noise_mask": processed_mask}
    else:
        return {"samples": newlatent}



def reshape_latent_to(target_shape, latent, repeat_batch=True):
    if latent.shape[1:] != target_shape[1:]:
        latent = comfy.utils.common_upscale(latent, target_shape[-1], target_shape[-2], "bilinear", "center")
    if repeat_batch:
        return comfy.utils.repeat_to_batch_size(latent, target_shape[0])
    else:
        return latent


def latent_inter_polate(samples1, samples2, ratio):
    samples_out = samples1.copy()

    s1 = samples1["samples"]
    s2 = samples2["samples"]

    s2 = reshape_latent_to(s1.shape, s2)

    m1 = torch.linalg.vector_norm(s1, dim=(1))
    m2 = torch.linalg.vector_norm(s2, dim=(1))

    s1 = torch.nan_to_num(s1 / m1)
    s2 = torch.nan_to_num(s2 / m2)

    t = (s1 * ratio + s2 * (1.0 - ratio))
    mt = torch.linalg.vector_norm(t, dim=(1))
    st = torch.nan_to_num(t / mt)

    samples_out["samples"] = st * (m1 * ratio + m2 * (1.0 - ratio))
    return samples_out



def set_last_layer(clip, clipnum):
    clip = clip.clone()
    clip.clip_layer(clipnum)
    return clip

def create_latent_tensor(width, height, batch):
    width = width - (width % 8)
    height = height - (height % 8)
    latent = torch.zeros([batch, 4, height // 8, width // 8])
    if latent.shape[1] != 16:
        latent = latent.repeat(1, 16// 4, 1, 1)
    return latent




def condi_zero_out(conditioning):
    c = []
    for t in conditioning:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        n = [torch.zeros_like(t[0]), d]
        c.append(n)
    return (c, )


#endregion------------------latent----------------------------------------------------#





#region------------------kontext---------------------------------------------------#

PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def resize_to_multiple_of_8(x):
    if isinstance(x, torch.Tensor):
        # Tensor 分支
        _, h, w, _ = x.shape
        new_w = w // 8 * 8
        new_h = h // 8 * 8
        if new_w != w or new_h != h:
            x = comfy.utils.common_upscale(
                x.movedim(-1, 1), new_w, new_h, "bilinear", "center"
            ).movedim(1, -1)
        return x
    elif isinstance(x, Image.Image):
        # PIL 分支
        width, height = x.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        return x.resize((new_width, new_height), Image.LANCZOS)
    else:
        raise TypeError("Input must be PIL.Image or torch.Tensor")


def kontext_adjust_image_resolution(image, auto_adjust_image):      
    if auto_adjust_image:
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, target_width, target_height = min(
            (abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS
        )

        scaled_image = comfy.utils.common_upscale(
            image.movedim(-1, 1),
            target_width,
            target_height,
            "bicubic",
            "center"
        ).movedim(1, -1)

        image = scaled_image[:, :, :, :3]
    else:
        image = image
    return (image,)



def condi_zero_out(negative): #条件零化
    c = []
    for t in negative:
        d = t[1].copy()
        pooled_output = d.get("pooled_output", None)
        if pooled_output is not None:
            d["pooled_output"] = torch.zeros_like(pooled_output)
        conditioning_lyrics = d.get("conditioning_lyrics", None)
        if conditioning_lyrics is not None:
            d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
        n = [torch.zeros_like(t[0]), d]
        c.append(n)
    negative = c
    return (negative, )


def Inpaint_encode(positive, negative, pixels, vae, mask, noise_mask=True):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

    orig_pixels = pixels
    pixels = orig_pixels.clone()
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
        mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

    m = (1.0 - mask.round()).squeeze(1)
    for i in range(3):
        pixels[:,:,:,i] -= 0.5
        pixels[:,:,:,i] *= m
        pixels[:,:,:,i] += 0.5
    concat_latent = vae.encode(pixels)
    orig_latent = vae.encode(orig_pixels)

    out_latent = {}

    out_latent["samples"] = orig_latent
    if noise_mask:
        out_latent["noise_mask"] = mask

    out = []
    for conditioning in [positive, negative]:
        c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                "concat_mask": mask})
        out.append(c)   
    positive=out[0]
    negative=out[1]  
    return (positive, negative, out_latent)



#endregion------------------kontext---------------------------------------------------#




#region---------------------------视觉标记--------------------------

BJ_MODE=["image", "transparent", "white", "black", "red", "green", "blue"] 

ALIGNMENT_REFERENCE_POINTS = [
    'center',
    'top_left',
    'top_center',
    'bottom_center',
    'left_center',
    'right_center'
]

def XXcreate_background(bg_mode, width, height, f_imge=None,):
    """
    创建指定背景模式的背景图像，统一返回4通道tensor
    """
    if bg_mode == "transparent":
        # 创建RGBA透明背景
        background = np.zeros((height, width, 4), dtype=np.float32)
        background[:, :, 3] = 0  # Alpha通道设为0（透明）
        return torch.from_numpy(background)
    elif bg_mode == "white":
        background = torch.ones((height, width, 4), dtype=torch.float32)
        background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "black":
        background = torch.zeros((height, width, 4), dtype=torch.float32)
        background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "red":
        background = torch.zeros((height, width, 4), dtype=torch.float32)
        background[:, :, 0] = 1.0  # Red channel
        background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "green":
        background = torch.zeros((height, width, 4), dtype=torch.float32)
        background[:, :, 1] = 1.0  # Green channel
        background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "blue":
        background = torch.zeros((height, width, 4), dtype=torch.float32)
        background[:, :, 2] = 1.0  # Blue channel
        background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "image" and f_imge is not None:
        # 如果f_imge是3通道，转换为4通道
        if f_imge.shape[-1] == 3:
            alpha_channel = torch.ones((*f_imge.shape[:-1], 1), dtype=f_imge.dtype)
            f_imge = torch.cat([f_imge, alpha_channel], dim=-1)
        return f_imge.clone()
    else:
        # 默认返回黑色背景
        background = torch.zeros((height, width, 4), dtype=torch.float32)
        background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background


def create_background(bg_mode, width, height, f_imge=None):
    """
    创建指定背景模式的背景图像，根据f_imge的通道数决定输出通道数
    """
    # 确定输出通道数
    channels = 3  # 默认3通道
    if f_imge is not None:
        if isinstance(f_imge, torch.Tensor):
            channels = f_imge.shape[-1] if len(f_imge.shape) == 3 else (f_imge.shape[-1] if len(f_imge.shape) == 4 else 3)
        elif isinstance(f_imge, np.ndarray):
            channels = f_imge.shape[-1] if len(f_imge.shape) >= 3 else 3
    elif bg_mode == "transparent":
        channels = 4  # 透明背景使用4通道
    
    if bg_mode == "transparent":
        # 创建RGBA透明背景
        background = np.zeros((height, width, 4), dtype=np.float32)
        background[:, :, 3] = 0  # Alpha通道设为0（透明）
        return background
    elif bg_mode == "white":
        background = np.ones((height, width, channels), dtype=np.float32)
        if channels == 4:
            background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        elif channels == 3:
            pass  # RGB已经设置为白色
        return background
    elif bg_mode == "black":
        background = np.zeros((height, width, channels), dtype=np.float32)
        if channels == 4:
            background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "red":
        background = np.zeros((height, width, channels), dtype=np.float32)
        background[:, :, 0] = 1.0  # Red channel
        if channels >= 3:
            background[:, :, 1] = 0.0  # Green channel
            background[:, :, 2] = 0.0  # Blue channel
        if channels == 4:
            background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "green":
        background = np.zeros((height, width, channels), dtype=np.float32)
        if channels >= 3:
            background[:, :, 0] = 0.0  # Red channel
            background[:, :, 1] = 1.0  # Green channel
            background[:, :, 2] = 0.0  # Blue channel
        if channels == 4:
            background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "blue":
        background = np.zeros((height, width, channels), dtype=np.float32)
        if channels >= 3:
            background[:, :, 0] = 0.0  # Red channel
            background[:, :, 1] = 0.0  # Green channel
            background[:, :, 2] = 1.0  # Blue channel
        if channels == 4:
            background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background
    elif bg_mode == "image" and f_imge is not None:
        # 如果f_imge是torch.Tensor，转换为numpy
        if isinstance(f_imge, torch.Tensor):
            f_imge = f_imge.cpu().numpy()
        
        # 调整f_imge大小以匹配目标尺寸
        if len(f_imge.shape) == 3:
            f_h, f_w, f_c = f_imge.shape
            if f_h != height or f_w != width:
                # 需要调整大小
                f_imge_resized = np.zeros((height, width, f_c), dtype=f_imge.dtype)
                # 简单的中心裁剪或填充逻辑
                h_start = max(0, (height - f_h) // 2)
                w_start = max(0, (width - f_w) // 2)
                h_end = min(height, h_start + f_h)
                w_end = min(width, w_start + f_w)
                src_h_start = max(0, (f_h - height) // 2)
                src_w_start = max(0, (f_w - width) // 2)
                src_h_end = src_h_start + (h_end - h_start)
                src_w_end = src_w_start + (w_end - w_start)
                f_imge_resized[h_start:h_end, w_start:w_end] = f_imge[src_h_start:src_h_end, src_w_start:src_w_end]
                f_imge = f_imge_resized
        elif len(f_imge.shape) == 4 and f_imge.shape[0] == 1:
            f_imge = f_imge[0]  # 去掉batch维度
            
        return f_imge.copy()
    else:
        # 默认返回黑色背景
        background = np.zeros((height, width, channels), dtype=np.float32)
        if channels == 4:
            background[:, :, 3] = 1.0  # Alpha通道设为1（不透明）
        return background



def visualize_mask_on_image(base_image, mask, ignore_threshold=100, opacity=0.8, outline_thickness=1, extrant_to_block=False, smoothness=1, out_color="colorful", fill=True):
    colors = {"white": (255, 255, 255), "black": (0, 0, 0), "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "yellow": (255, 255, 0), "cyan": (0, 255, 255), "magenta": (255, 0, 255)}
    
    def tensorMask2cv2img(tensor_mask):
        # 处理 None 情况
        if tensor_mask is None:
            return np.zeros((base_image.shape[1], base_image.shape[2]), dtype=np.uint8)
            
        if isinstance(tensor_mask, torch.Tensor):
            mask_np = tensor_mask.squeeze().cpu().numpy()
            # 确保数据类型为uint8
            if mask_np.dtype != np.uint8:
                # 检查是否为 None 或空数组
                if mask_np is not None and mask_np.size > 0:
                    mask_np = (mask_np * 255).astype(np.uint8)
                else:
                    mask_np = np.zeros((base_image.shape[1], base_image.shape[2]), dtype=np.uint8)
            return mask_np
        elif isinstance(tensor_mask, np.ndarray):
            # 确保数据类型为uint8
            if tensor_mask.dtype != np.uint8:
                # 检查是否为空数组
                if tensor_mask.size > 0:
                    return (tensor_mask * 255).astype(np.uint8)
                else:
                    return np.zeros((base_image.shape[1], base_image.shape[2]), dtype=np.uint8)
            return tensor_mask
        else:
            # 如果是PIL图像或其他类型，先转为numpy数组再转为uint8
            mask_np = np.array(tensor_mask)
            if mask_np.dtype != np.uint8:
                # 检查是否为空数组
                if mask_np.size > 0:
                    return (mask_np * 255).astype(np.uint8)
                else:
                    return np.zeros((base_image.shape[1], base_image.shape[2]), dtype=np.uint8)
            return mask_np
            
    opencv_gray_image = tensorMask2cv2img(mask)
    _, binary_mask = cv2.threshold(opencv_gray_image, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_with_positions = [(cv2.boundingRect(c)[0], cv2.boundingRect(c)[1], c) for c in contours]
    contours_with_positions.sort(key=lambda x: (x[1], x[0]))
    sorted_contours = [c[2] for c in contours_with_positions[:8]]
    fill_mask = np.zeros_like(binary_mask)
    for contour in sorted_contours:
        if cv2.contourArea(contour) < ignore_threshold: 
            continue
        if extrant_to_block:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(fill_mask, (x, y), (x+w, y+h), 255, cv2.FILLED)
        else: 
            cv2.drawContours(fill_mask, [contour], 0, 255, cv2.FILLED)
    
    # 确保fill_mask是正确的数据类型
    if fill_mask.dtype != np.uint8:
        fill_mask = (fill_mask * 255).astype(np.uint8)
        
    if smoothness > 0: 
        fill_mask_pil = Image.fromarray(fill_mask)
        fill_mask_pil = fill_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
        fill_mask = np.array(fill_mask_pil)
        
    base_image_np = base_image[0].cpu().numpy() * 255.0
    base_image_np = base_image_np.astype(np.float32)
    mask_color_layer = base_image_np.copy()
    final_mask = np.zeros_like(binary_mask)
    
    for i, contour in enumerate(sorted_contours):
        area = cv2.contourArea(contour)
        if area < ignore_threshold: 
            continue
        if out_color in ["white", "black", "red", "green", "blue"]: 
            color = np.array(colors[out_color], dtype=np.float32)
        elif out_color == "colorful": 
            color = np.array(colors[["white", "black", "red", "green", "blue", "yellow", "cyan", "magenta"][i % 8]], dtype=np.float32)
        else: 
            color = np.array(colors["white"], dtype=np.float32)
            
        temp_mask = np.zeros_like(binary_mask)
        if extrant_to_block:
            x, y, w, h = cv2.boundingRect(contour)
            thickness = cv2.FILLED if fill else outline_thickness
            cv2.rectangle(temp_mask, (x, y), (x+w, y+h), 255, thickness)
        else:
            thickness = cv2.FILLED if fill else outline_thickness
            cv2.drawContours(temp_mask, [contour], 0, 255, thickness)
            
        # 确保temp_mask是正确的数据类型
        if temp_mask.dtype != np.uint8:
            temp_mask = (temp_mask * 255).astype(np.uint8)
            
        if smoothness > 0: 
            temp_mask_pil = Image.fromarray(temp_mask)
            temp_mask_pil = temp_mask_pil.filter(ImageFilter.GaussianBlur(radius=smoothness))
            temp_mask = np.array(temp_mask_pil)
            
        final_mask = cv2.bitwise_or(final_mask, temp_mask)
        mask_float = temp_mask.astype(np.float32) / 255.0
        for c in range(3): 
            mask_color_layer[:, :, c] = mask_float * color[c] + (1 - mask_float) * mask_color_layer[:, :, c]
            
    mask_float_global = final_mask.astype(np.float32) / 255.0
    combined_image = opacity * mask_color_layer + (1 - opacity) * base_image_np
    combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)
    combined_image_tensor = torch.from_numpy(combined_image).float() / 255.0
    combined_image_tensor = combined_image_tensor.unsqueeze(0)
    final_mask_tensor = torch.from_numpy(final_mask).float() / 255.0
    final_mask_tensor = final_mask_tensor.unsqueeze(0)
    return combined_image_tensor, final_mask_tensor




def apply_position_transform(bj_img, img, mask, scale, x_position, y_position, rotation):
    # 确保所有输入都是正确的模式
    img = img.convert('RGBA')
    mask = mask.convert('L')
    bg_width, bg_height = bj_img.size
    bj_img = bj_img.convert('RGBA')
    
    # 计算mask的边界框和中心
    mask_array = np.array(mask)
    mask_coords = np.where(mask_array > 128)
    
    if len(mask_coords[0]) > 0:
        x_min, x_max = np.min(mask_coords[1]), np.max(mask_coords[1])
        y_min, y_max = np.min(mask_coords[0]), np.max(mask_coords[0])
        mask_center_x = (x_min + x_max) // 2
        mask_center_y = (y_min + y_max) // 2
    else:
        # 如果mask为空，使用图像中心
        mask_center_x, mask_center_y = img.width // 2, img.height // 2
    
    # 创建足够大的画布以容纳所有变换
    max_dim = int(max(img.width, img.height, bg_width, bg_height) * 3)
    canvas_width, canvas_height = max_dim, max_dim
    
    # 计算在画布上放置背景的位置
    bj_pos_x = (canvas_width - bg_width) // 2
    bj_pos_y = (canvas_height - bg_height) // 2
    
    # 创建画布并粘贴背景图像
    canvas = Image.new('RGBA', (canvas_width, canvas_height), (0, 0, 0, 0))
    canvas.paste(bj_img, (bj_pos_x, bj_pos_y))
    
    # 计算前景图像在画布上的初始位置
    img_pos_x = bj_pos_x + x_position
    img_pos_y = bj_pos_y + y_position
    
    # 应用变换
    if scale != 1.0 or rotation != 0:
        # 计算变换后的图像尺寸
        new_width, new_height = int(img.width * scale), int(img.height * scale)
        
        # 缩放图像和遮罩
        transformed_img = img.resize((new_width, new_height), Image.LANCZOS)
        transformed_mask = mask.resize((new_width, new_height), Image.NEAREST)
        
        # 计算变换后的mask中心（考虑缩放）
        new_mask_center_x = int(mask_center_x * scale)
        new_mask_center_y = int(mask_center_y * scale)
        
        # 应用旋转（围绕mask中心）
        if rotation != 0:
            # 旋转图像和遮罩
            transformed_img = transformed_img.rotate(
                -rotation,  # PIL的rotate函数使用逆时针旋转
                center=(new_mask_center_x, new_mask_center_y),
                expand=True,  # 允许图像扩展以包含所有旋转后的内容
                resample=Image.BICUBIC
            )
            
            transformed_mask = transformed_mask.rotate(
                -rotation,
                center=(new_mask_center_x, new_mask_center_y),
                expand=True,
                resample=Image.NEAREST
            )
        
        # 计算旋转后的图像中心偏移
        offset_x = (transformed_img.width - new_width) // 2
        offset_y = (transformed_img.height - new_height) // 2
        
        # 更新mask中心位置
        new_mask_center_x += offset_x
        new_mask_center_y += offset_y
        
        # 计算粘贴位置（左上角）
        paste_x = img_pos_x - new_mask_center_x + mask_center_x
        paste_y = img_pos_y - new_mask_center_y + mask_center_y
        
        # 将变换后的图像和遮罩粘贴到画布上
        canvas.paste(transformed_img, (paste_x, paste_y), transformed_mask)
        
        # 创建最终的遮罩图像
        result_mask = Image.new('L', (canvas_width, canvas_height), 0)
        result_mask.paste(transformed_mask, (paste_x, paste_y))
        
        # 计算最终的mask中心位置（相对于原始背景）
        final_x = x_position
        final_y = y_position
    else:
        # 如果没有变换，直接粘贴原图
        canvas.paste(img, (img_pos_x, img_pos_y), mask)
        
        # 创建最终的遮罩图像
        result_mask = Image.new('L', (canvas_width, canvas_height), 0)
        result_mask.paste(mask, (img_pos_x, img_pos_y))
        
        # 计算最终的mask中心位置（相对于原始背景）
        final_x = x_position
        final_y = y_position
    
    # === 关键修改：在输出前进行裁切 ===
    # 计算相对于原始背景的可见区域
    bg_visible_x1 = max(0, bj_pos_x)
    bg_visible_y1 = max(0, bj_pos_y)
    bg_visible_x2 = min(bj_pos_x + bg_width, canvas_width)
    bg_visible_y2 = min(bj_pos_y + bg_height, canvas_height)
    
    # 裁切画布和遮罩到背景大小
    cropped_img = canvas.crop((bg_visible_x1, bg_visible_y1, bg_visible_x2, bg_visible_y2))
    cropped_mask = result_mask.crop((bg_visible_x1, bg_visible_y1, bg_visible_x2, bg_visible_y2))
    
    # 如果裁切后的尺寸小于背景尺寸，创建与背景相同大小的新图像并居中粘贴
    if cropped_img.size != (bg_width, bg_height):
        final_img = Image.new('RGBA', (bg_width, bg_height), (0, 0, 0, 0))
        final_mask = Image.new('L', (bg_width, bg_height), 0)
        
        paste_x = (bg_width - cropped_img.width) // 2
        paste_y = (bg_height - cropped_img.height) // 2
        
        final_img.paste(cropped_img, (paste_x, paste_y))
        final_mask.paste(cropped_mask, (paste_x, paste_y))
    else:
        final_img = cropped_img
        final_mask = cropped_mask
    
    return final_img, final_mask, final_x, final_y



def center_transform_layer( align_mode, x_offset, y_offset, rotation, scale, edge_detection, edge_thickness, edge_color, opacity, bj_img=None, fj_img=None, mask=None):
    # 颜色映射表
    color_mapping = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 255, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
    }
    
    if fj_img is None: raise ValueError("前景图像(fj_img)是必需的输入")
    if bj_img is None: bj_img = fj_img.clone()
        
    bj_np = bj_img[0].cpu().numpy()
    fj_np = fj_img[0].cpu().numpy()
    
    bj_pil = Image.fromarray((bj_np * 255).astype(np.uint8)).convert("RGBA")
    fj_pil = Image.fromarray((fj_np * 255).astype(np.uint8)).convert("RGBA")
    
    canvas_width, canvas_height = bj_pil.size
    canvas_center_x, canvas_center_y = canvas_width // 2, canvas_height // 2
    
    # 记录原始mask尺寸
    original_mask_width, original_mask_height = 0, 0
    if mask is not None:
        mask_np = mask[0].cpu().numpy()
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8)).convert("L")
        original_mask_width, original_mask_height = mask_pil.size
        mask_pil = mask_pil.resize(fj_pil.size, Image.LANCZOS)
    else:
        mask_pil = Image.new("L", fj_pil.size, 255)
        original_mask_width, original_mask_height = mask_pil.size
    
    mask_center_x, mask_center_y = mask_pil.size[0] // 2, mask_pil.size[1] // 2
    
    # 计算布局调整后的尺寸和缩放比例
    scale_x, scale_y = 1.0, 1.0
    if align_mode == "height":
        height_ratio = canvas_height / fj_pil.size[1]
        new_width = int(fj_pil.size[0] * height_ratio)
        scale_x = height_ratio
        scale_y = height_ratio
        fj_pil = fj_pil.resize((new_width, canvas_height), Image.LANCZOS)
        mask_pil = mask_pil.resize((new_width, canvas_height), Image.LANCZOS)
        mask_center_x, mask_center_y = new_width // 2, canvas_height // 2
    elif align_mode == "width":
        width_ratio = canvas_width / fj_pil.size[0]
        new_height = int(fj_pil.size[1] * width_ratio)
        scale_x = width_ratio
        scale_y = width_ratio
        fj_pil = fj_pil.resize((canvas_width, new_height), Image.LANCZOS)
        mask_pil = mask_pil.resize((canvas_width, new_height), Image.LANCZOS)
        mask_center_x, mask_center_y = canvas_width // 2, new_height // 2
    
    # 保存调整后但未旋转缩放的尺寸
    adjusted_mask_width, adjusted_mask_height = mask_pil.size
    
    # 确保rotation是数值类型
    rotation = float(rotation)  # 新增的类型转换
    
    # 应用旋转和缩放
    if rotation != 0 or scale != 1.0:
        mask_pil = mask_pil.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
        fj_pil = fj_pil.rotate(rotation, center=(mask_center_x, mask_center_y), resample=Image.BICUBIC, expand=True)
        
        if scale != 1.0:
            new_size = (int(fj_pil.size[0] * scale), int(fj_pil.size[1] * scale))
            scale_x *= scale
            scale_y *= scale
            fj_pil = fj_pil.resize(new_size, Image.LANCZOS)
            mask_pil = mask_pil.resize(new_size, Image.LANCZOS)
        
        mask_center_x, mask_center_y = fj_pil.size[0] // 2, fj_pil.size[1] // 2
    
    # 确保composite_pil正确初始化
    composite_pil = bj_pil.copy()
    
    # 计算mask在画布上的位置（考虑偏移，不应用缩放）
    x_position = canvas_center_x - mask_center_x + x_offset
    y_position = canvas_center_y - mask_center_y + y_offset
    
    # 计算实际粘贴位置（相对于画布左上角）
    paste_x = max(0, x_position)
    paste_y = max(0, y_position)
    
    # 应用遮罩到前景图像
    fj_with_mask = fj_pil.copy()
    fj_with_mask.putalpha(mask_pil)
    
    # 调整透明度
    if opacity < 1.0:
        r, g, b, a = fj_with_mask.split()
        a = a.point(lambda p: p * opacity)
        fj_with_mask = Image.merge("RGBA", (r, g, b, a))
    
    # 裁切超出画布的部分
    cropped_fj = fj_with_mask.crop((
        max(0, -x_position),
        max(0, -y_position),
        min(fj_pil.size[0], canvas_width - x_position),
        min(fj_pil.size[1], canvas_height - y_position)
    ))
    
    # 合成图像
    composite_pil.paste(cropped_fj, (paste_x, paste_y), cropped_fj)
    
    # 重构完整尺寸的遮罩
    full_size_mask = Image.new("L", composite_pil.size, 0)
    full_size_mask.paste(mask_pil.crop((
        max(0, -x_position),
        max(0, -y_position),
        min(fj_pil.size[0], canvas_width - x_position),
        min(fj_pil.size[1], canvas_height - y_position)
    )), (paste_x, paste_y))
    
    # 计算mask的边界框（相对于mask自身）
    bbox = mask_pil.getbbox()  # 返回格式：(left, top, right, bottom)
    
    if bbox:
        bbox_left, bbox_top, bbox_right, bbox_bottom = bbox
        bbox_width = bbox_right - bbox_left
        bbox_height = bbox_bottom - bbox_top
        
        # 计算mask边界框中心点相对于mask的坐标
        bbox_center_x = bbox_left + bbox_width // 2
        bbox_center_y = bbox_top + bbox_height // 2
        
        # 计算mask有效区域中心点在画布上的实际坐标
        bbox_actual_x = x_position + bbox_center_x
        bbox_actual_y = y_position + bbox_center_y
        
        # 计算相对于画布中心的偏移（不考虑缩放）
        bbox_center_x_rel = bbox_actual_x - canvas_center_x
        bbox_center_y_rel = bbox_actual_y - canvas_center_y
    else:
        # 如果mask为空，返回默认值
        bbox_width = 0
        bbox_height = 0
        bbox_center_x_rel = 0
        bbox_center_y_rel = 0
    
    # 应用边缘检测
    if edge_detection:
        # 获取颜色值
        if edge_color in color_mapping:
            r, g, b = color_mapping[edge_color]
        else:
            r, g, b = 0, 0, 0  # 默认黑色
        
        # 确保遮罩为二值图像
        threshold = 128
        mask_array = np.array(full_size_mask)
        binary_mask = np.where(mask_array > threshold, 255, 0).astype(np.uint8)
        binary_mask_pil = Image.fromarray(binary_mask)
        
        # 应用边缘检测
        edge_image = Image.new("RGBA", composite_pil.size, (0, 0, 0, 0))
        edge_draw = ImageDraw.Draw(edge_image)
        
        # 转换为OpenCV格式进行轮廓检测
        mask_cv = np.array(binary_mask_pil)
        contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            for i in range(edge_thickness):
                points = [tuple(point[0]) for point in contour]
                edge_draw.line(points, fill=(r, g, b, int(opacity * 255)), width=edge_thickness-i+1)
        
        # 合并边缘到合成图
        composite_pil = Image.alpha_composite(composite_pil, edge_image)
        
        # 生成边缘遮罩
        edge_mask = np.zeros_like(mask_cv)
        cv2.drawContours(edge_mask, contours, -1, 255, edge_thickness)
        line_mask_pil = Image.fromarray(edge_mask)
    else:
        line_mask_pil = Image.new("L", composite_pil.size, 0)
    
    # 准备输出
    composite_np = np.array(composite_pil).astype(np.float32) / 255.0
    mask_np = np.array(full_size_mask).astype(np.float32) / 255.0
    line_mask_np = np.array(line_mask_pil).astype(np.float32) / 255.0
    
    if composite_np.shape[2] == 4: pass
    else: composite_np = np.dstack([composite_np, np.ones_like(composite_np[..., 0:1])])
    
    composite_tensor = torch.from_numpy(composite_np).unsqueeze(0)
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    line_mask_tensor = torch.from_numpy(line_mask_np).unsqueeze(0).unsqueeze(0)
    
    # box_info格式：(mask有效区域的宽度, mask有效区域的高度, 有效区域中心点相对于画布中心的X坐标, 有效区域中心点相对于画布中心的Y坐标)
    box_info = (bbox_width, bbox_height, bbox_center_x_rel, bbox_center_y_rel)
    
    return (composite_tensor, mask_tensor, line_mask_tensor, box_info)



#endregion---------------------------视觉标记--------------------------




































