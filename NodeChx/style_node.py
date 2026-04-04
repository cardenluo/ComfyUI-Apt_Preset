import numpy as np
import torch
import os
import folder_paths
from comfy.cli_args import args
import torch.nn as nn
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from comfy.ldm.modules.attention import optimized_attention
from comfy.ldm.modules.diffusionmodules.mmdit import (RMSNorm,JointBlock,)
import math

import comfy.model_patcher
import comfy.samplers
import torch

from ..main_unit import *


#----------------安全导入-------


#region-------------------------------prompt_injection-------------------------------------

def build_patch(patchedBlocks, weight=1.0, sigma_start=0.0, sigma_end=1.0, noise=0.0):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        (block, block_index) = extra_options.get('block', (None,None))
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
        batch_prompt = n.shape[0] // len(extra_options["cond_or_uncond"])

        if sigma <= sigma_start and sigma >= sigma_end:
            if (block and f'{block}:{block_index}' in patchedBlocks and patchedBlocks[f'{block}:{block_index}']):
                if context_attn1.dim() == 3:
                    c = context_attn1[0].unsqueeze(0)
                else:
                    c = context_attn1[0][0].unsqueeze(0)
                b = patchedBlocks[f'{block}:{block_index}'][0][0].repeat(c.shape[0], 1, 1).to(context_attn1.device)
                if noise != 0.0:
                    b = b + torch.randn_like(b) * noise

                padding = abs(c.shape[1] - b.shape[1])
                if c.shape[1] > b.shape[1]:
                    b = F.pad(b, (0, 0, 0, padding), mode='constant', value=0)
                elif c.shape[1] < b.shape[1]:
                    c = F.pad(c, (0, 0, 0, padding), mode='constant', value=0)

                out = torch.stack((c, b)).to(dtype=context_attn1.dtype)
                out = out.repeat(1, batch_prompt, 1, 1) * weight

                return n, out, out 

        return n, context_attn1, value_attn1
    return prompt_injection_patch

def build_patch_by_index(patchedBlocks, weight=1.0, sigma_start=0.0, sigma_end=1.0, noise=0.0):
    def prompt_injection_patch(n, context_attn1: torch.Tensor, value_attn1, extra_options):
        idx = extra_options["transformer_index"]
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9
        batch_prompt = n.shape[0] // len(extra_options["cond_or_uncond"])

        if sigma <= sigma_start and sigma >= sigma_end:
            if idx in patchedBlocks and patchedBlocks[idx] is not None:
                if context_attn1.dim() == 3:
                    c = context_attn1[0].unsqueeze(0)
                else:
                    c = context_attn1[0][0].unsqueeze(0)

                b = patchedBlocks[idx][0][0].repeat(c.shape[0], 1, 1).to(context_attn1.device)
                if noise != 0.0:
                    b = b + torch.randn_like(b) * noise

                padding = abs(c.shape[1] - b.shape[1])
                if c.shape[1] > b.shape[1]:
                    b = F.pad(b, (0, 0, 0, padding), mode='constant', value=0)
                elif c.shape[1] < b.shape[1]:
                    c = F.pad(c, (0, 0, 0, padding), mode='constant', value=0)

                out = torch.stack((c, b)).to(dtype=context_attn1.dtype)
                out = out.repeat(1, batch_prompt, 1, 1) * weight

                return n, out, out

        return n, context_attn1, value_attn1
    return prompt_injection_patch

class IPA_PromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "all":  ("CONDITIONING",),
                "input_4":  ("CONDITIONING",),
                "input_5":  ("CONDITIONING",),
                "input_7":  ("CONDITIONING",),
                "input_8":  ("CONDITIONING",),
                "middle_0": ("CONDITIONING",),
                "output_0": ("CONDITIONING",),
                "output_1": ("CONDITIONING",),
                "output_2": ("CONDITIONING",),
                "output_3": ("CONDITIONING",),
                "output_4": ("CONDITIONING",),
                "output_5": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "noise": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def patch(self, model: comfy.model_patcher.ModelPatcher, all=None, input_4=None, input_5=None, input_7=None, input_8=None, middle_0=None, output_0=None, output_1=None, output_2=None, output_3=None, output_4=None, output_5=None, weight=1.0, start_at=0.0, end_at=1.0, noise=0.0):
        if not any((all, input_4, input_5, input_7, input_8, middle_0, output_0, output_1, output_2, output_3, output_4, output_5)):
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        patchedBlocks = {}
        blocks = {'input': [4, 5, 7, 8], 'middle': [0], 'output': [0, 1, 2, 3, 4, 5]}

        for block in blocks:
            for index in blocks[block]:
                value = locals()[f"{block}_{index}"] if locals()[f"{block}_{index}"] is not None else all
                if value is not None:
                    patchedBlocks[f"{block}:{index}"] = value

        m.set_model_attn2_patch(build_patch(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end, noise=noise))

        return (m,)

class IPA_PromptInjectionIdx:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "all": ("CONDITIONING",),
                "idx_0": ("CONDITIONING",),
                "idx_1": ("CONDITIONING",),
                "idx_2": ("CONDITIONING",),
                "idx_3": ("CONDITIONING",),
                "idx_4": ("CONDITIONING",),
                "idx_5": ("CONDITIONING",),
                "idx_6": ("CONDITIONING",),
                "idx_7": ("CONDITIONING",),
                "idx_8": ("CONDITIONING",),
                "idx_9": ("CONDITIONING",),
                "idx_10": ("CONDITIONING",),
                "idx_11_sd15": ("CONDITIONING",),
                "idx_12_sd15": ("CONDITIONING",),
                "idx_13_sd15": ("CONDITIONING",),
                "idx_14_sd15": ("CONDITIONING",),
                "idx_15_sd15": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "noise": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def patch(self, model, all=None, idx_0=None, idx_1=None, idx_2=None, idx_3=None, idx_4=None, idx_5=None, idx_6=None, idx_7=None, idx_8=None, idx_9=None, idx_10=None, idx_11_sd15=None, idx_12_sd15=None, idx_13_sd15=None, idx_14_sd15=None, idx_15_sd15=None, weight=1.0, start_at=0.0, end_at=1.0, noise=0.0):
        if not any((all, idx_0, idx_1, idx_2, idx_3, idx_4, idx_5, idx_6, idx_7, idx_8, idx_9, idx_10, idx_11_sd15, idx_12_sd15, idx_13_sd15, idx_14_sd15, idx_15_sd15)):
            return (model,)
        
        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

        patchedBlocks = {
            0: idx_0 if idx_0 is not None else all,
            1: idx_1 if idx_1 is not None else all,
            2: idx_2 if idx_2 is not None else all,
            3: idx_3 if idx_3 is not None else all,
            4: idx_4 if idx_4 is not None else all,
            5: idx_5 if idx_5 is not None else all,
            6: idx_6 if idx_6 is not None else all,
            7: idx_7 if idx_7 is not None else all,
            8: idx_8 if idx_8 is not None else all,
            9: idx_9 if idx_9 is not None else all,
            10: idx_10 if idx_10 is not None else all,
            11: idx_11_sd15 if idx_11_sd15 is not None else all if not is_sdxl else None,
            12: idx_12_sd15 if idx_12_sd15 is not None else all if not is_sdxl else None,
            13: idx_13_sd15 if idx_13_sd15 is not None else all if not is_sdxl else None,
            14: idx_14_sd15 if idx_14_sd15 is not None else all if not is_sdxl else None,
            15: idx_15_sd15 if idx_15_sd15 is not None else all if not is_sdxl else None,
        }

        m.set_model_attn2_patch(build_patch_by_index(patchedBlocks, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end, noise=noise))

        return (m,)


class IPA_SimplePromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "block": (["input:4", "input:5", "input:7", "input:8", "middle:0", "output:0", "output:1", "output:2", "output:3", "output:4", "output:5"],),
                "conditioning": ("CONDITIONING",),
                "weight": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def patch(self, model: comfy.model_patcher.ModelPatcher, block, conditioning=None, weight=1.0, start_at=0.0, end_at=1.0):
        if conditioning is None:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        m.set_model_attn2_patch(build_patch({f"{block}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)

class IPA_AdvancedPromptInjection:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "locations": ("STRING", {"multiline": True, "default": "output:0,1.0\noutput:1,1.0"}),
                "conditioning": ("CONDITIONING",),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def patch(self, model: comfy.model_patcher.ModelPatcher, locations: str, conditioning=None, start_at=0.0, end_at=1.0):
        if not conditioning:
            return (model,)

        m = model.clone()
        sigma_start = m.get_model_object("model_sampling").percent_to_sigma(start_at)
        sigma_end = m.get_model_object("model_sampling").percent_to_sigma(end_at)

        for line in locations.splitlines():
            line = line.strip().strip('\n')
            weight = 1.0
            if ',' in line:
                line, weight = line.split(',')
                line = line.strip()
                weight = float(weight)
            if line:
                m.set_model_attn2_patch(build_patch({f"{line}": conditioning}, weight=weight, sigma_start=sigma_start, sigma_end=sigma_end))

        return (m,)



#endregion-------------------------------prompt_injection-------------------------------------