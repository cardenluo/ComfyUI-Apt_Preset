# 标准库
import math
import os
import sys
from dataclasses import dataclass
import enum
from enum import Enum



# 第三方库
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv2d
from torch.nn.modules.utils import _pair
import numpy as np
from tqdm import tqdm


import comfy
from comfy import samplers
import comfy.model_management as mm


import folder_paths
import node_helpers

import nodes
from nodes import CLIPTextEncode, common_ksampler, VAEDecode, VAEEncode, ImageScale, KSampler,InpaintModelConditioning

import matplotlib
from PIL import Image, ImageFilter, ImageDraw

#---------------------安全导入------
try:
    from scipy.stats import norm
    SCIPY_AVAILABLE = True
except ImportError:
    norm = None
    SCIPY_AVAILABLE = False
#---------------------安全导入------

from typing import Any, Dict, Optional, Tuple, Union, cast
from comfy.samplers import KSAMPLER
import torchvision.transforms as transforms

import types
from comfy.utils import load_torch_file
from comfy import lora
import functools
import comfy.model_management
import comfy.utils
from functools import partial
from comfy.model_base import Flux

from server import PromptServer

from typing import Optional
import math
import logging
import nodes

import latent_preview



from .main_stack import Apply_CN_union,Apply_Redux




from ..office_unit import *
from ..main_unit import *

sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))

#region--------------------------------临时收纳--------------------------------


#---------------------安全导入------
try:
    import cv2
    REMOVER_AVAILABLE = True  # 导入成功时设置为True
except ImportError:
    cv2 = None
    REMOVER_AVAILABLE = False  # 导入失败时设置为False





class texture_Offset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pixels": ("IMAGE",),
                "x_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
                "y_percent": (
                    "FLOAT",
                    {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/imgEffect/texture"

    def run(self, pixels, x_percent, y_percent):
        n, y, x, c = pixels.size()
        y = round(y * y_percent / 100)
        x = round(x * x_percent / 100)
        return (pixels.roll((y, x), (1, 2)),)





class DynamicTileSplit:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        upscale_models = ["None"] + folder_paths.get_filename_list("upscale_models")
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": (upscale_models, ),
                "width_factor": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "height_factor": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "overlap_rate": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 0.95, "step": 0.01}),
                "use_fixed_size": ("BOOLEAN", {"default": False}),
                "tile_width": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),
                "tile_height": ("INT", {"default": 512, "min": 32, "max": 4096, "step": 32}),

            }
        }
    
    RETURN_TYPES = ("IMAGE", "STICH3", "INT")
    RETURN_NAMES = ("image", "stich", "total")
    FUNCTION = "tile_image"
    CATEGORY = "Apt_Preset/chx_ksample"

    def image_width_height(self, image, width_factor, height_factor, overlap_rate, 
                           use_fixed_size, tile_width, tile_height):
        _, raw_H, raw_W, _ = image.shape
        
        if use_fixed_size:
            tile_width = ((tile_width + 7) // 8) * 8
            tile_height = ((tile_height + 7) // 8) * 8
            return (tile_width, tile_height)
        
        if overlap_rate == 0:
            if width_factor == 1:
                tile_width = raw_W
            else:
                tile_width = int(raw_W / width_factor)
                if tile_width % 8 != 0:
                    tile_width = ((tile_width + 7) // 8) * 8
            if height_factor == 1:
                tile_height = raw_H
            else:
                tile_height = int(raw_H / height_factor)
                if tile_height % 8 != 0:
                    tile_height = ((tile_height + 7) // 8) * 8
        else:
            if width_factor == 1:
                tile_width = raw_W
            else:
                tile_width = int(raw_W / (1 + (width_factor - 1) * (1 - overlap_rate)))
                if tile_width % 8 != 0:
                    tile_width = (tile_width // 8) * 8
            if height_factor == 1:
                tile_height = raw_H
            else:
                tile_height = int(raw_H / (1 + (height_factor - 1) * (1 - overlap_rate)))
                if tile_height % 8 != 0:
                    tile_height = (tile_height // 8) * 8
        return (tile_width, tile_height)

    def tile_image(self, image, upscale_model, use_fixed_size, tile_width, tile_height,
                   width_factor, height_factor, overlap_rate):
        if upscale_model != "None":
            up_model = load_upscale_model(upscale_model)
            image = upscale_with_model(up_model, image)

        tile_width, tile_height = self.image_width_height(
            image, width_factor, height_factor, overlap_rate,
            use_fixed_size, tile_width, tile_height
        )

        image = tensor2pil(image.squeeze(0))
        img_width, img_height = image.size

        if img_width <= tile_width and img_height <= tile_height:
            return (pil2tensor(image), [(0, 0, img_width, img_height)], 1)

        def calculate_step(size, tile_size):
            if size <= tile_size:
                return 1, 0
            else:
                num_tiles = (size + tile_size - 1) // tile_size
                overlap = (num_tiles * tile_size - size) // (num_tiles - 1)
                step = tile_size - overlap
                return num_tiles, step

        num_cols, step_x = calculate_step(img_width, tile_width)
        num_rows, step_y = calculate_step(img_height, tile_height)
        total = num_cols * num_rows

        tiles = []
        positions = []
        for y in range(num_rows):
            for x in range(num_cols):
                left = x * step_x
                upper = y * step_y
                right = min(left + tile_width, img_width)
                lower = min(upper + tile_height, img_height)

                if right - left < tile_width:
                    left = max(0, img_width - tile_width)
                if lower - upper < tile_height:
                    upper = max(0, img_height - tile_height)

                tile = image.crop((left, upper, right, lower))
                tile_tensor = pil2tensor(tile)
                tiles.append(tile_tensor)
                positions.append((left, upper, right, lower))

        image = torch.stack(tiles, dim=0).squeeze(1)
        orig_size = (img_width, img_height)
        grid_size = (num_cols, num_rows)
        stich = (positions, orig_size, grid_size)
        
        return (image, stich, total)



class DynamicTileMerge:
    def __init__(self, *args, **kwargs):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiles": ("IMAGE",),
                "stich": ("STICH3",),
                "padding": ("INT", {"default": 64, "min": 0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "assemble_image"
    CATEGORY = "Apt_Preset/chx_ksample"

    def create_gradient_mask(self, size, direction):
        """Create a gradient mask for blending."""
        mask = Image.new("L", size)
        for i in range(size[0] if direction == 'horizontal' else size[1]):
            value = int(255 * (1 - (i / size[0] if direction == 'horizontal' else i / size[1])))
            if direction == 'horizontal':
                mask.paste(value, (i, 0, i+1, size[1]))
            else:
                mask.paste(value, (0, i, size[0], i+1))
        return mask

    def blend_tiles(self, tile1, tile2, overlap_size, direction, padding):
        """Blend two tiles with a smooth transition."""
        blend_size = padding
        if blend_size > overlap_size:
            blend_size = overlap_size

        if blend_size == 0:
            # No blending, just concatenate the images at the correct overlap
            if direction == 'horizontal':
                result = Image.new("RGB", (tile1.width + tile2.width - overlap_size, tile1.height))
                # Paste the left part of tile1 excluding the overlap
                result.paste(tile1.crop((0, 0, tile1.width - overlap_size, tile1.height)), (0, 0))
                # Paste tile2 directly after tile1
                result.paste(tile2, (tile1.width - overlap_size, 0))
            else:
                # For vertical direction
                result = Image.new("RGB", (tile1.width, tile1.height + tile2.height - overlap_size))
                result.paste(tile1.crop((0, 0, tile1.width, tile1.height - overlap_size)), (0, 0))
                result.paste(tile2, (0, tile1.height - overlap_size))
            return result

        # 以下为原有的混合代码，当 blend_size > 0 时执行
        offset_total = overlap_size - blend_size
        offset_left = offset_total // 2
        offset_right = offset_total - offset_left

        size = (blend_size, tile1.height) if direction == 'horizontal' else (tile1.width, blend_size)
        mask = self.create_gradient_mask(size, direction)

        if direction == 'horizontal':
            crop_tile1 = tile1.crop((tile1.width - overlap_size + offset_left, 0, tile1.width - offset_right, tile1.height))
            crop_tile2 = tile2.crop((offset_left, 0, offset_left + blend_size, tile2.height))
            if crop_tile1.size != crop_tile2.size:
                raise ValueError(f"Crop sizes do not match: {crop_tile1.size} vs {crop_tile2.size}")

            blended = Image.composite(crop_tile1, crop_tile2, mask)
            result = Image.new("RGB", (tile1.width + tile2.width - overlap_size, tile1.height))
            result.paste(tile1.crop((0, 0, tile1.width - overlap_size + offset_left, tile1.height)), (0, 0))
            result.paste(blended, (tile1.width - overlap_size + offset_left, 0))
            result.paste(tile2.crop((offset_left + blend_size, 0, tile2.width, tile2.height)), (tile1.width - offset_right, 0))
        else:
            offset_total = overlap_size - blend_size
            offset_top = offset_total // 2
            offset_bottom = offset_total - offset_top

            size = (tile1.width, blend_size)
            mask = self.create_gradient_mask(size, direction)

            crop_tile1 = tile1.crop((0, tile1.height - overlap_size + offset_top, tile1.width, tile1.height - offset_bottom))
            crop_tile2 = tile2.crop((0, offset_top, tile2.width, offset_top + blend_size))
            if crop_tile1.size != crop_tile2.size:
                raise ValueError(f"Crop sizes do not match: {crop_tile1.size} vs {crop_tile2.size}")

            blended = Image.composite(crop_tile1, crop_tile2, mask)
            result = Image.new("RGB", (tile1.width, tile1.height + tile2.height - overlap_size))
            result.paste(tile1.crop((0, 0, tile1.width, tile1.height - overlap_size + offset_top)), (0, 0))
            result.paste(blended, (0, tile1.height - overlap_size + offset_top))
            result.paste(tile2.crop((0, offset_top + blend_size, tile2.width, tile2.height)), (0, tile1.height - offset_bottom))
        return result

    def assemble_image(self, tiles, stich,  padding):

        positions, original_size, grid_size = stich

        num_cols, num_rows = grid_size
        reconstructed_image = Image.new("RGB", original_size)

        # First, blend each row independently
        row_images = []
        for row in range(num_rows):
            row_image = tensor2pil(tiles[row * num_cols].unsqueeze(0))
            for col in range(1, num_cols):
                index = row * num_cols + col
                tile_image = tensor2pil(tiles[index].unsqueeze(0))
                prev_right = positions[index - 1][2]
                left = positions[index][0]
                overlap_width = prev_right - left
                if overlap_width > 0:
                    row_image = self.blend_tiles(row_image, tile_image, overlap_width, 'horizontal', padding)
                else:
                    # Adjust the size of row_image to accommodate the new tile
                    new_width = row_image.width + tile_image.width
                    new_height = max(row_image.height, tile_image.height)
                    new_row_image = Image.new("RGB", (new_width, new_height))
                    new_row_image.paste(row_image, (0, 0))
                    new_row_image.paste(tile_image, (row_image.width, 0))
                    row_image = new_row_image
            row_images.append(row_image)

        # Now, blend each row together vertically
        final_image = row_images[0]
        for row in range(1, num_rows):
            prev_lower = positions[(row - 1) * num_cols][3]
            upper = positions[row * num_cols][1]
            overlap_height = prev_lower - upper
            if overlap_height > 0:
                final_image = self.blend_tiles(final_image, row_images[row], overlap_height, 'vertical', padding)
            else:
                # Adjust the size of final_image to accommodate the new row image
                new_width = max(final_image.width, row_images[row].width)
                new_height = final_image.height + row_images[row].height
                new_final_image = Image.new("RGB", (new_width, new_height))
                new_final_image.paste(final_image, (0, 0))
                new_final_image.paste(row_images[row], (0, final_image.height))
                final_image = new_final_image

        return pil2tensor(final_image).unsqueeze(0)



#region---------sampler_enhance------------------------------------------------------------------------#


def get_dd_schedule(
    sigma: float,
    sigmas: torch.Tensor,
    dd_schedule: torch.Tensor,
) -> float:
    sched_len = len(dd_schedule)
    if (
        sched_len < 2
        or len(sigmas) < 2
        or sigma <= 0
        or not (sigmas[-1] <= sigma <= sigmas[0])
    ):
        return 0.0
    # First, we find the index of the closest sigma in the list to what the model was
    # called with.
    deltas = (sigmas[:-1] - sigma).abs()
    idx = int(deltas.argmin())
    if (
        (idx == 0 and sigma >= sigmas[0])
        or (idx == sched_len - 1 and sigma <= sigmas[-2])
        or deltas[idx] == 0
    ):
        # Either exact match or closest to head/tail of the DD schedule so we
        # can't interpolate to another schedule item.
        return dd_schedule[idx].item()
    # If we're here, that means the sigma is in between two sigmas in the
    # list.
    idxlow, idxhigh = (idx, idx - 1) if sigma > sigmas[idx] else (idx + 1, idx)
    # We find the low/high neighbor sigmas - our sigma is somewhere between them.
    nlow, nhigh = sigmas[idxlow], sigmas[idxhigh]
    if nhigh - nlow == 0:
        # Shouldn't be possible, but just in case... Avoid divide by zero.
        return dd_schedule[idxlow]
    # Ratio of how close we are to the high neighbor.
    ratio = ((sigma - nlow) / (nhigh - nlow)).clamp(0, 1)
    # Mix the DD schedule high/low items according to the ratio.
    return torch.lerp(dd_schedule[idxlow], dd_schedule[idxhigh], ratio).item()


def detail_daemon_sampler(
    model: object,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    dds_wrapped_sampler: object,
    dds_make_schedule: callable,
    dds_cfg_scale_override: float,
    **kwargs: dict,
) -> torch.Tensor:
    if dds_cfg_scale_override > 0:
        cfg_scale = dds_cfg_scale_override
    else:
        maybe_cfg_scale = getattr(model.inner_model, "cfg", None)
        cfg_scale = (
            float(maybe_cfg_scale) if isinstance(maybe_cfg_scale, (int, float)) else 1.0
        )
    dd_schedule = torch.tensor(
        dds_make_schedule(len(sigmas) - 1),
        dtype=torch.float32,
        device="cpu",
    )
    sigmas_cpu = sigmas.detach().clone().cpu()
    sigma_max, sigma_min = float(sigmas_cpu[0]), float(sigmas_cpu[-1]) + 1e-05

    def model_wrapper(x: torch.Tensor, sigma: torch.Tensor, **extra_args: dict):
        sigma_float = float(sigma.max().detach().cpu())
        if not (sigma_min <= sigma_float <= sigma_max):
            return model(x, sigma, **extra_args)
        dd_adjustment = get_dd_schedule(sigma_float, sigmas_cpu, dd_schedule) * 0.1
        adjusted_sigma = sigma * max(1e-06, 1.0 - dd_adjustment * cfg_scale)
        return model(x, adjusted_sigma, **extra_args)

    for k in (
        "inner_model",
        "sigmas",
    ):
        if hasattr(model, k):
            setattr(model_wrapper, k, getattr(model, k))
    return dds_wrapped_sampler.sampler_function(
        model_wrapper,
        x,
        sigmas,
        **kwargs,
        **dds_wrapped_sampler.extra_options,
    )


def make_detail_daemon_schedule(
    steps,
    start,
    end,
    bias,
    amount,
    exponent,
    start_offset,
    end_offset,
    fade,
    smooth,
):
    start = min(start, end)
    mid = start + bias * (end - start)
    multipliers = np.zeros(steps)

    start_idx, mid_idx, end_idx = [
        int(round(x * (steps - 1))) for x in [start, mid, end]
    ]

    start_values = np.linspace(0, 1, mid_idx - start_idx + 1)
    if smooth:
        start_values = 0.5 * (1 - np.cos(start_values * np.pi))
    start_values = start_values**exponent
    if start_values.any():
        start_values *= amount - start_offset
        start_values += start_offset

    end_values = np.linspace(1, 0, end_idx - mid_idx + 1)
    if smooth:
        end_values = 0.5 * (1 - np.cos(end_values * np.pi))
    end_values = end_values**exponent
    if end_values.any():
        end_values *= amount - end_offset
        end_values += end_offset

    multipliers[start_idx : mid_idx + 1] = start_values
    multipliers[mid_idx : end_idx + 1] = end_values
    multipliers[:start_idx] = start_offset
    multipliers[end_idx + 1 :] = end_offset
    multipliers *= 1 - fade

    return multipliers


class sampler_enhance:

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "sampler": ("SAMPLER",),
                "detail_amount": ("FLOAT", {"default": 0.1, "min": -5.0, "max": 5.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "smooth": ("BOOLEAN", {"default": True}),
                "cfg_scale_override": ("FLOAT", {"default": 0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                
            },
        }
    CATEGORY = "Apt_Preset/chx_ksample"
    RETURN_TYPES = ("SAMPLER",)
    FUNCTION = "go"
    
    @classmethod
    def go(
        cls,
        sampler: object,
        *,
        detail_amount=0.1,
        start=0.2,
        end=0.8,
        bias=0.5,
        exponent=1,
        start_offset=0,
        end_offset=0,
        fade=0,
        smooth="true",
        cfg_scale_override=0,
    ) -> tuple:
        def dds_make_schedule(steps):
            return make_detail_daemon_schedule(
                steps,
                start,
                end,
                bias,
                detail_amount,
                exponent,
                start_offset,
                end_offset,
                fade,
                smooth,
            )

        return (
            KSAMPLER(
                detail_daemon_sampler,
                extra_options={
                    "dds_wrapped_sampler": sampler,
                    "dds_make_schedule": dds_make_schedule,
                    "dds_cfg_scale_override": cfg_scale_override,
                },
            ),
        )


#endregion------sampler_enhance---------------------------------



#region-----------------------------pre-ic light------------------------

UNET_MAP_ATTENTIONS = {"proj_in.weight","proj_in.bias","proj_out.weight","proj_out.bias","norm.weight","norm.bias"}
TRANSFORMER_BLOCKS = {"norm1.weight","norm1.bias","norm2.weight","norm2.bias","norm3.weight","norm3.bias","attn1.to_q.weight","attn1.to_k.weight","attn1.to_v.weight","attn1.to_out.0.weight","attn1.to_out.0.bias","attn2.to_q.weight","attn2.to_k.weight","attn2.to_v.weight","attn2.to_out.0.weight","attn2.to_out.0.bias","ff.net.0.proj.weight","ff.net.0.proj.bias","ff.net.2.weight","ff.net.2.bias"}
UNET_MAP_RESNET = {"in_layers.2.weight": "conv1.weight","in_layers.2.bias": "conv1.bias","emb_layers.1.weight": "time_emb_proj.weight","emb_layers.1.bias": "time_emb_proj.bias","out_layers.3.weight": "conv2.weight","out_layers.3.bias": "conv2.bias","skip_connection.weight": "conv_shortcut.weight","skip_connection.bias": "conv_shortcut.bias","in_layers.0.weight": "norm1.weight","in_layers.0.bias": "norm1.bias","out_layers.0.weight": "norm2.weight","out_layers.0.bias": "norm2.bias"}
UNET_MAP_BASIC = {("label_emb.0.0.weight", "class_embedding.linear_1.weight"),("label_emb.0.0.bias", "class_embedding.linear_1.bias"),("label_emb.0.2.weight", "class_embedding.linear_2.weight"),("label_emb.0.2.bias", "class_embedding.linear_2.bias"),("label_emb.0.0.weight", "add_embedding.linear_1.weight"),("label_emb.0.0.bias", "add_embedding.linear_1.bias"),("label_emb.0.2.weight", "add_embedding.linear_2.weight"),("label_emb.0.2.bias", "add_embedding.linear_2.bias"),("input_blocks.0.0.weight", "conv_in.weight"),("input_blocks.0.0.bias", "conv_in.bias"),("out.0.weight", "conv_norm_out.weight"),("out.0.bias", "conv_norm_out.bias"),("out.2.weight", "conv_out.weight"),("out.2.bias", "conv_out.bias"),("time_embed.0.weight", "time_embedding.linear_1.weight"),("time_embed.0.bias", "time_embedding.linear_1.bias"),("time_embed.2.weight", "time_embedding.linear_2.weight"),("time_embed.2.bias", "time_embedding.linear_2.bias")}
TEMPORAL_TRANSFORMER_BLOCKS = {"norm_in.weight","norm_in.bias","ff_in.net.0.proj.weight","ff_in.net.0.proj.bias","ff_in.net.2.weight","ff_in.net.2.bias"}
TEMPORAL_TRANSFORMER_BLOCKS.update(TRANSFORMER_BLOCKS)
TEMPORAL_UNET_MAP_ATTENTIONS = {"time_mixer.mix_factor"}
TEMPORAL_UNET_MAP_ATTENTIONS.update(UNET_MAP_ATTENTIONS)
TEMPORAL_TRANSFORMER_MAP = {"time_pos_embed.0.weight": "time_pos_embed.linear_1.weight","time_pos_embed.0.bias": "time_pos_embed.linear_1.bias","time_pos_embed.2.weight": "time_pos_embed.linear_2.weight","time_pos_embed.2.bias": "time_pos_embed.linear_2.bias"}
TEMPORAL_RESNET = {"time_mixer.mix_factor"}
unet_config = {'use_checkpoint': False, 'image_size': 32, 'out_channels': 4, 'use_spatial_transformer': True, 'legacy': False, 'adm_in_channels': None,'in_channels': 8, 'model_channels': 320, 'num_res_blocks': [2, 2, 2, 2], 'transformer_depth': [1, 1, 1, 1, 1, 1, 0, 0],'channel_mult': [1, 2, 4, 4], 'transformer_depth_middle': 1, 'use_linear_in_transformer': False, 'context_dim': 768, 'num_heads': 8,'transformer_depth_output': [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],'use_temporal_attention': False, 'use_temporal_resblock': False}

def convert_iclight_unet(state_dict):
    num_res_blocks = unet_config["num_res_blocks"]
    channel_mult = unet_config["channel_mult"]
    transformer_depth = unet_config["transformer_depth"][:]
    transformer_depth_output = unet_config["transformer_depth_output"][:]
    num_blocks = len(channel_mult)
    transformers_mid = unet_config.get("transformer_depth_middle", None)
    diffusers_unet_map = {}
    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, b)] = "input_blocks.{}.0.{}".format(n, b)
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["down_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.time_stack.{}".format(n, b)
                diffusers_unet_map["down_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "input_blocks.{}.0.{}".format(n, b)
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, b)] = "input_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["down_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "input_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["down_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "input_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map["down_blocks.{}.downsamplers.0.conv.{}".format(x, k)] = "input_blocks.{}.0.op.{}".format(n, k)
    i = 0
    for b in TEMPORAL_UNET_MAP_ATTENTIONS:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, b)] = "middle_block.1.{}".format(b)
    for b in TEMPORAL_TRANSFORMER_MAP:
        diffusers_unet_map["mid_block.attentions.{}.{}".format(i, TEMPORAL_TRANSFORMER_MAP[b])] = "middle_block.1.{}".format(b)
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.transformer_blocks.{}.{}".format(t, b)
        for b in TEMPORAL_TRANSFORMER_BLOCKS:
            diffusers_unet_map["mid_block.attentions.{}.temporal_transformer_blocks.{}.{}".format(i, t, b)] = "middle_block.1.time_stack.{}.{}".format(t, b)
    for i, n in enumerate([0, 2]):
        for b in TEMPORAL_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, b)] = "middle_block.{}.{}".format(n, b)
        for b in UNET_MAP_RESNET:
            diffusers_unet_map["mid_block.resnets.{}.spatial_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.temporal_res_block.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.time_stack.{}".format(n, b)
            diffusers_unet_map["mid_block.resnets.{}.{}".format(i, UNET_MAP_RESNET[b])] = "middle_block.{}.{}".format(n, b)
    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        l = num_res_blocks[x] + 1
        for i in range(l):
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, b)] = "output_blocks.{}.0.{}".format(n, b)
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["up_blocks.{}.resnets.{}.spatial_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.{}".format(n, b)
                diffusers_unet_map["up_blocks.{}.resnets.{}.temporal_res_block.{}".format(x, i, UNET_MAP_RESNET[b])] = "output_blocks.{}.0.time_stack.{}".format(n, b)
            for b in TEMPORAL_RESNET:
                diffusers_unet_map["up_blocks.{}.resnets.{}".format(i, b)] = "output_blocks.{}.{}".format(n, b)
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_TRANSFORMER_MAP:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, TEMPORAL_TRANSFORMER_MAP[b])] = "output_blocks.{}.1.{}".format(n, b)
                for b in TEMPORAL_UNET_MAP_ATTENTIONS:
                    diffusers_unet_map["up_blocks.{}.attentions.{}.{}".format(x, i, b)] = "output_blocks.{}.1.{}".format(n, b)
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.transformer_blocks.{}.{}".format(n, t, b)
                    for b in TEMPORAL_TRANSFORMER_BLOCKS:
                        diffusers_unet_map["up_blocks.{}.attentions.{}.temporal_transformer_blocks.{}.{}".format(x, i, t, b)] = "output_blocks.{}.1.time_stack.{}.{}".format(n, t, b)
            if i == l - 1:
                for k in ["weight", "bias"]:
                    diffusers_unet_map["up_blocks.{}.upsamplers.0.conv.{}".format(x, k)] = "output_blocks.{}.{}.conv.{}".format(n, c, k)
            n += 1
    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]
    unet_state_dict = state_dict
    diffusers_keys = diffusers_unet_map
    new_sd = {}
    for k in diffusers_keys:
        if k in unet_state_dict:
            new_sd[diffusers_keys[k]] = unet_state_dict.pop(k)
    leftover_keys = unet_state_dict.keys()
    if len(leftover_keys) > 0:
        spatial_leftover_keys = []
        temporal_leftover_keys = []
        other_leftover_keys = []
        for key in leftover_keys:
            if "spatial" in key:
                spatial_leftover_keys.append(key)
            elif "temporal" in key:
                temporal_leftover_keys.append(key)
            else:
                other_leftover_keys.append(key)
        print("spatial_leftover_keys:")
        for key in spatial_leftover_keys:
            print(key)
        print("temporal_leftover_keys:")
        for key in temporal_leftover_keys:
            print(key)
        print("other_leftover_keys:")
        for key in other_leftover_keys:
            print(key)
    new_sd = {"diffusion_model." + k: v for k, v in new_sd.items()}
    return new_sd

class LightPosition(Enum):
    LEFT = "Left Light"
    RIGHT = "Right Light"
    TOP = "Top Light"
    BOTTOM = "Bottom Light"
    TOP_LEFT = "Top Left Light"
    TOP_RIGHT = "Top Right Light"
    BOTTOM_LEFT = "Bottom Left Light"
    BOTTOM_RIGHT = "Bottom Right Light"

def generate_gradient_image(width, height, start_color, end_color, multiplier, lightPosition):
    if lightPosition == LightPosition.LEFT:
        gradient = np.tile(np.linspace(0, 1, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.RIGHT:
        gradient = np.tile(np.linspace(1, 0, width)**multiplier, (height, 1))
    elif lightPosition == LightPosition.TOP:
        gradient = np.tile(np.linspace(0, 1, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM:
        gradient = np.tile(np.linspace(1, 0, height)**multiplier, (width, 1)).T
    elif lightPosition == LightPosition.BOTTOM_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.BOTTOM_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(1, 0, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_RIGHT:
        x = np.linspace(1, 0, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    elif lightPosition == LightPosition.TOP_LEFT:
        x = np.linspace(0, 1, width)**multiplier
        y = np.linspace(0, 1, height)**multiplier
        x_mesh, y_mesh = np.meshgrid(x, y)
        gradient = np.sqrt(x_mesh**2 + y_mesh**2) / np.sqrt(2.0)
    else:
        raise ValueError(f"Unsupported position. Choose from {', '.join([member.value for member in LightPosition])}.")
    gradient_img = np.zeros((height, width, 3), dtype=np.float32)
    for i in range(3):
        gradient_img[..., i] = start_color[i] + (end_color[i] - start_color[i]) * gradient
    gradient_img = np.clip(gradient_img, 0, 255).astype(np.uint8)
    return gradient_img

class LoadAndApplyICLightUnet:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),"model_path": (folder_paths.get_filename_list("unet"), )}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "IC-Light"
    def load(self, model, model_path):
        type_str = str(type(model.model.model_config).__name__)
        device = model_management.get_torch_device()
        dtype = model_management.unet_dtype()
        if "SD15" not in type_str:
            raise Exception(f"Attempted to load {type_str} model, IC-Light is only compatible with SD 1.5 models.")
        print("LoadAndApplyICLightUnet: Checking IC-Light Unet path")
        model_full_path = folder_paths.get_full_path("unet", model_path)
        if not os.path.exists(model_full_path):
            raise Exception("Invalid model path")
        else:
            print("LoadAndApplyICLightUnet: Loading IC-Light Unet weights")
            model_clone = model.clone()
            iclight_state_dict = load_torch_file(model_full_path)
            print("LoadAndApplyICLightUnet: Attempting to add patches with IC-Light Unet weights")
            try:
                if 'conv_in.weight' in iclight_state_dict:
                    iclight_state_dict = convert_iclight_unet(iclight_state_dict)
                    in_channels = iclight_state_dict["diffusion_model.input_blocks.0.0.weight"].shape[1]
                    prefix = ""
                else:
                    prefix = "diffusion_model."
                    in_channels = iclight_state_dict["input_blocks.0.0.weight"].shape[1]
                model_clone.model.model_config.unet_config["in_channels"] = in_channels
                patches={(prefix + key): ("diff",[value.to(dtype=dtype, device=device),{"pad_weight": key == "diffusion_model.input_blocks.0.0.weight" or key == "input_blocks.0.0.weight"},])for key, value in iclight_state_dict.items()}
                model_clone.add_patches(patches)
            except:
                raise Exception("Could not patch model")
            print("LoadAndApplyICLightUnet: Added LoadICLightUnet patches")
            def bound_extra_conds(self, **kwargs):
                 return ICLight.extra_conds(self, **kwargs)
            new_extra_conds = types.MethodType(bound_extra_conds, model_clone.model)
            model_clone.add_object_patch("extra_conds", new_extra_conds)
            return (model_clone, )

class ICLight:
    def extra_conds(self, **kwargs):
        out = {}
        image = kwargs.get("concat_latent_image", None)
        noise = kwargs.get("noise", None)
        device = kwargs["device"]
        model_in_channels = self.model_config.unet_config['in_channels']
        input_channels = image.shape[1] + 4
        if model_in_channels != input_channels:
            raise Exception(f"Input channels {input_channels} does not match model in_channels {model_in_channels}, 'opt_background' latent input should be used with the IC-Light 'fbc' model, and only with it")
        if image is None:
            image = torch.zeros_like(noise)
        if image.shape[1:] != noise.shape[1:]:
            image = comfy.utils.common_upscale(image.to(device), noise.shape[-1], noise.shape[-2], "bilinear", "center")
        image = comfy.utils.resize_to_batch_size(image, noise.shape[0])
        process_image_in = lambda image: image
        out['c_concat'] = comfy.conds.CONDNoiseShape(process_image_in(image))
        cross_attn = kwargs.get("cross_attn", None)
        if cross_attn is not None:
            out['c_crossattn'] = comfy.conds.CONDCrossAttn(cross_attn)
        adm = self.encode_adm(**kwargs)
        if adm is not None:
            out['y'] = comfy.conds.CONDRegular(adm)
        return out

class ICLightConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),"negative": ("CONDITIONING", ),"vae": ("VAE", ),"foreground": ("LATENT", ),"multiplier": ("FLOAT", {"default": 0.18215, "min": 0.0, "max": 1.0, "step": 0.001}),},"optional": {"opt_background": ("LATENT", ),}}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "empty_latent")
    FUNCTION = "encode"
    CATEGORY = "IC-Light"
    def encode(self, positive, negative, vae, foreground, multiplier, opt_background=None):
        samples_1 = foreground["samples"]
        if opt_background is not None:
            samples_2 = opt_background["samples"]
            repeats_1 = samples_2.size(0) // samples_1.size(0)
            repeats_2 = samples_1.size(0) // samples_2.size(0)
            if samples_1.shape[1:] != samples_2.shape[1:]:
                samples_2 = comfy.utils.common_upscale(samples_2, samples_1.shape[-1], samples_1.shape[-2], "bilinear", "disabled")
            if repeats_1 > 1:
                samples_1 = samples_1.repeat(repeats_1, 1, 1, 1)
            if repeats_2 > 1:
                samples_2 = samples_2.repeat(repeats_2, 1, 1, 1)
            concat_latent = torch.cat((samples_1, samples_2), dim=1)
        else:
            concat_latent = samples_1
        out_latent = torch.zeros_like(samples_1)
        out = []
        for conditioning in [positive, negative]:
            c = []
            for t in conditioning:
                d = t[1].copy()
                d["concat_latent_image"] = concat_latent * multiplier
                n = [t[0], d]
                c.append(n)
            out.append(c)
        return (out[0], out[1], {"samples": out_latent})

class LightSource:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "light_position": ([member.value for member in LightPosition],),
                "multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.001}),
                "start_color": ("STRING", {"default": "#FFFFFF"}),
                "end_color": ("STRING", {"default": "#000000"}),
                "width": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                "height": ("INT", { "default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 8, }),
                },
            "optional": {
                "batch_size": ("INT", { "default": 1, "min": 1, "max": 4096, "step": 1, }),
                "prev_image": ("IMAGE",),
                } 
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "IC-Light"
    def execute(self, light_position, multiplier, start_color, end_color, width, height, batch_size=1, prev_image=None):
        def toRgb(color):
            if color.startswith('#') and len(color) == 7:
                color_rgb =tuple(int(color[i:i+2], 16) for i in (1, 3, 5))
            else:
                color_rgb = tuple(int(i) for i in color.split(','))
            return color_rgb
        lightPosition = LightPosition(light_position)
        start_color_rgb = toRgb(start_color)
        end_color_rgb = toRgb(end_color)
        image = generate_gradient_image(width, height, start_color_rgb, end_color_rgb, multiplier, lightPosition)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        image = image.repeat(batch_size, 1, 1, 1)
        if prev_image is not None:
            image = torch.cat((prev_image, image), dim=0)
        return (image,)




#endregion-----------------------------pre-ic light------------------------




class pre_ic_light_sd15:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "bg_unet": (folder_paths.get_filename_list("unet"), {"default": "iclight_sd15_fbc_unet_ldm.safetensors"} ),
                "fo_unet": (folder_paths.get_filename_list("unet"), {"default": "iclight_sd15_fc_unet_ldm.safetensors"} ),
                "multiplier": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01})
            },
            
            "optional": {
                "fore_img":("IMAGE",),
                "bg_img":("IMAGE",),
                "light_img":("IMAGE",),
                
            },
            
        }


    RETURN_TYPES = ("RUN_CONTEXT","IMAGE" )
    RETURN_NAMES = ("context", "light_img" )
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    def run(self, context, bg_unet, fo_unet, multiplier, fore_img=None, light_img=None, bg_img=None):
        vae = context.get("vae",None)
        positive = context.get("positive",None)
        negative = context.get("negative",None)
        model = context.get("model",None)
        latent = context.get("latent",None)
        images = context.get("images",None)

        if light_img is None:
            outimg = decode(vae, latent)[0] if latent is not None else images
            return (context, outimg)

        if fore_img is None:
            if images is None:
                raise ValueError("fore_img 和 context 中的 images 都为 None，无法继续处理")
            fore_img = images

        if fore_img is None:
            raise ValueError("前景图像为 None，无法继续处理")

        foreground = encode(vae, fore_img)[0]

        opt_background = None
        if bg_img is not None:
            bg_img = get_image_resize(bg_img,fore_img)   #尺寸一致性
            opt_background = encode(vae, bg_img)[0]

        if bg_img is not None:
            unet = bg_unet
        else:
            unet = fo_unet

        model = LoadAndApplyICLightUnet().load(model, unet)[0]
        positive, negative, empty_latent = ICLightConditioning().encode(
            positive=positive,
            negative=negative,
            vae=vae,
            foreground=foreground,
            multiplier=multiplier,
            opt_background=opt_background
        )

        light_img = get_image_resize(light_img,fore_img)   #尺寸一致性
        context = new_context(context, positive=positive, negative=negative, model=model, images=light_img,)

        return(context, light_img)



class pre_latent_light:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "weigh": ("FLOAT", {"default": 0.3, "min": 0.1, "max": 0.7, "step": 0.01}),

            },
            "optional": {
                "img_targe": ("IMAGE",),
                "img_light": ("IMAGE",),
            },

        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE","LATENT" )
    RETURN_NAMES = ("context", "image", "latent")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/chx_tool/😺backup"

    def run(self,context, weigh, img_targe=None, img_light=None, ):

        latent = context.get("latent", None)
        vae = context.get("vae", None)

        if img_light is None:
            outimg = decode(vae, latent)[0]
            return (context,outimg)


        if img_targe is None:
            img_targe = context.get("images", None)

        img_light = get_image_resize(img_light,img_targe)   

        latent2 = encode(vae, img_targe)[0]
        latent1 = encode(vae, img_light)[0]
        latent = latent_inter_polate(latent1, latent2, weigh)

        output_image = decode(vae, latent)[0]
        context = new_context(context, latent=latent, images=output_image, )
        
        return  (context, output_image, latent)



#endregion-----------总------------------------------



class chx_Ksampler_Kontext_adv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "add_noise": (["enable", "disable"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip": "-1 means no change"}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 1000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"],),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "auto_adjust_image": ("BOOLEAN", {"default": False}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
                "redux_stack": ("REDUX_STACK",),
                "union_stack": ("UNION_STACK",),
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "LATENT")
    RETURN_NAMES = ("context", "image", "latent")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    OUTPUT_NODE = True

    def run(self, context, seed, steps=0, image_output="Preview",union_stack=None, image=None, denoise=1, mask=None,redux_stack=None, prompt_weight=0.5, pos="", auto_adjust_image=False, prompt=None, extra_pnginfo=None, add_noise="enable", start_at_step=0, end_at_step=0, return_with_leftover_noise="disable"):
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")
        if steps == 0 or steps==-1: steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        guidance = context.get("guidance", 3.5)
        force_full_denoise = False if return_with_leftover_noise == "enable" else True
        disable_noise = True if add_noise == "disable" else False  
        latent =context.get("latent", None)    
        negative = context.get("negative", None)
        positive = None
        if pos and pos.strip(): positive, = CLIPTextEncode().encode(clip, pos)
        else: positive = context.get("positive", None)        

        if image.dim() == 3: image = image.unsqueeze(0)
        all_output_images = []
        all_results = []

#------------------------
        if image is None: 
            image = context.get("images", None)
            if image is None: 
                final_output = torch.cat(all_output_images, dim=0) if all_output_images else None
                context = new_context(context, latent=samples_dict,pos=pos,positive=positive, negative=negative, images=final_output)
                if image_output == "None": return (context, None, samples_dict)
                elif image_output in ("Hide", "Hide/Save"): return {"ui": {}, "result": (context, final_output, samples_dict)}
                else: return {"ui": {"images": all_results}, "result": (context, final_output, samples_dict)}
#------------------------

        for i in range(image.shape[0]):
            img = image[i:i+1]
            pixels = img
            if auto_adjust_image:
                width = img.shape[2]
                height = img.shape[1]
                aspect_ratio = width / height
                _, target_width, target_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
                scaled_image = comfy.utils.common_upscale(img.movedim(-1, 1), target_width, target_height, "bicubic", "center").movedim(1, -1)
                pixels = scaled_image[:, :, :, :3].clamp(0, 1)
            
            encoded_latent = vae.encode(pixels)[0]
            if encoded_latent.dim() == 3: encoded_latent = encoded_latent.unsqueeze(0)
            latent = {"samples": encoded_latent}


            if positive is not None and prompt_weight > 0:
                influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
                scaled_latent = latent["samples"] * influence
                positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
                positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})


            if redux_stack is not None:
                positive, =  Apply_Redux().apply_redux_stack(positive, redux_stack,)



            if union_stack is not None:
                positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])


            if mask is not None: latent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))

            result_tuple = common_ksampler(model, seed + i, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise, disable_noise=disable_noise, start_step=start_at_step, last_step=end_at_step, force_full_denoise=force_full_denoise)[0]

            if isinstance(result_tuple, dict):
                result_latent = result_tuple["samples"] if "samples" in result_tuple else result_tuple
            elif isinstance(result_tuple, (list, tuple)):
                result_latent = result_tuple[0] if len(result_tuple) > 0 else None
            else: result_latent = result_tuple
            
            assert result_latent is not None, "Failed to get valid latent from common_ksampler"

            if isinstance(result_latent, dict):
                if "samples" in result_latent:
                    samples = result_latent["samples"]
                    if isinstance(samples, torch.Tensor):
                        if samples.dim() == 3: samples = samples.unsqueeze(0)
                        result_latent = samples
                    else: raise TypeError(f"Expected tensor but got {type(samples).__name__} in 'samples' key")
                else: raise KeyError("Result dictionary does not contain 'samples' key")
            elif isinstance(result_latent, torch.Tensor):
                if result_latent.dim() == 3: result_latent = result_latent.unsqueeze(0)
            else: raise TypeError(f"Unsupported result type: {type(result_latent).__name__}")

            samples_dict = {"samples": result_latent} if isinstance(result_latent, torch.Tensor) else result_latent
            output_image = VAEDecode().decode(vae, samples_dict)[0]
            all_output_images.append(output_image)
            results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
            all_results.extend(results)

        final_output = torch.cat(all_output_images, dim=0) if all_output_images else None
        context = new_context(context, latent=samples_dict,pos=pos,positive=positive, negative=negative, images=final_output)

        if image_output == "None": return (context, None, samples_dict)
        elif image_output in ("Hide", "Hide/Save"): return {"ui": {}, "result": (context, final_output, samples_dict)}
        else: return {"ui": {"images": all_results}, "result": (context, final_output, samples_dict)}




class chx_Ksampler_Kontext:   #0803仅遮罩数据处理
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "prompt_weight": ("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "steps": ("INT", {"default": -1, "min": -1, "max": 10000,  "tooltip": "-1 means no change"}),
                "denoise": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.01}),
                "auto_adjust_image": ("BOOLEAN", {"default": False}),
                "image_output": (["None", "Hide", "Preview", "Save", "Hide/Save"], {"default": "Preview"}),
            },
            "optional": {
                "image": ("IMAGE", ),
                "mask": ("MASK", ),
                "redux_stack": ("REDUX_STACK",),
                "union_stack": ("UNION_STACK",),
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "LATENT")
    RETURN_NAMES = ("context", "image", "latent")
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    OUTPUT_NODE = True

    def run(self, context, seed, image_output="Preview", image=None, mask=None, steps=0, denoise=1, prompt_weight=0.5,union_stack=None,redux_stack=None,
             auto_adjust_image=False, pos="", prompt=None, extra_pnginfo=None):
        vae = context.get("vae")
        model = context.get("model")
        clip = context.get("clip")
        if steps == 0 or steps==-1: steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        guidance = context.get("guidance", 2.5)
        negative = context.get("negative", None)
        latent =context.get("latent", None)
        positive = None
        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive", None)


        if image is None: 
            image = context.get("images", None)
            if image is None: 
                result = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise)
                latent_result = result[0]

                output_image = decode(vae, latent_result)[0]
                context = new_context(context, latent=latent_result, images=output_image)
                
                if image_output != "None":
                    results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
                    if image_output in ("Hide", "Hide/Save"):
                        return {"ui": {}, "result": (context, output_image, latent_result)}
                    else:
                        return {"ui": {"images": results}, "result": (context, output_image, latent_result)}
                else:
                    return (context, output_image, latent_result)


        pixels = image
        if auto_adjust_image:
            width = image.shape[2]
            height = image.shape[1]
            aspect_ratio = width / height
            _, target_width, target_height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
            scaled_image = comfy.utils.common_upscale(image.movedim(-1, 1), target_width, target_height, "bicubic", "center").movedim(1, -1)
            pixels = scaled_image[:, :, :, :3].clamp(0, 1)
        
        encoded_latent = vae.encode(pixels)[0]        
        if encoded_latent.dim() == 3:
            encoded_latent = encoded_latent.unsqueeze(0)
        elif encoded_latent.dim() != 4:
            raise ValueError(f"Unexpected latent dimensions: {encoded_latent.dim()}. Expected 4D tensor.")           
        latent = {"samples": encoded_latent}


        if positive is not None and prompt_weight > 0:
            influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
            scaled_latent = latent["samples"] * influence
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [scaled_latent]}, append=True)
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        if redux_stack is not None:
            positive, =  Apply_Redux().apply_redux_stack(positive, redux_stack,)

        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            if mask.dim() == 3:
                mask = mask.unsqueeze(0)
            
            if mask.shape[0] == 1 and latent["samples"].shape[0] > 1:
                mask = mask.repeat(latent["samples"].shape[0], 1, 1, 1)
            
            if mask.shape[1] != 1:
                if mask.shape[1] == 3 or mask.shape[1] == 4:
                    mask = mask.mean(dim=1, keepdim=True)
                else:
                    mask = mask[:, :1, :, :]
            
            latent_shape = latent["samples"].shape
            if len(latent_shape) >= 4 and mask.shape[-2:] != latent_shape[-2:]:
                try:
                    mask = torch.nn.functional.interpolate(
                        mask, 
                        size=(latent_shape[2], latent_shape[3]), 
                        mode='bicubic', 
                        align_corners=False
                    )
                except:
                    mask = torch.nn.functional.interpolate(
                        mask, 
                        size=(latent_shape[2], latent_shape[3]), 
                        mode='nearest'
                    )
            
            mask = torch.clamp(mask, 0, 1)
            latent["noise_mask"] = mask



        result = common_ksampler(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, denoise=denoise)
        latent_result = result[0]

        output_image = decode(vae, latent_result)[0]
        context = new_context(context, latent=latent_result, images=output_image)
        
        if image_output != "None":
            results = easySave(output_image, 'easyPreview', image_output, prompt, extra_pnginfo)
            if image_output in ("Hide", "Hide/Save"):
                return {"ui": {}, "result": (context, output_image, latent_result)}
            else:
                return {"ui": {"images": results}, "result": (context, output_image, latent_result)}
        else:
            return (context, output_image, latent_result)




#endregion--------------------------------临时收纳--------------------------------































