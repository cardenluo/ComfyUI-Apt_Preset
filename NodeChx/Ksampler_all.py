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








#endregion--------------------------------临时收纳--------------------------------































