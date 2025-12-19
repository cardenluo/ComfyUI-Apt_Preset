
from torch import Tensor
from math import cos, sin, pi
from random import random
import torch
import comfy.model_management
from typing import Callable
import comfy.latent_formats
import torch.nn as nn
from comfy.model_patcher import ModelPatcher
from typing import Union
import node_helpers
T = torch.Tensor
from comfy_extras.nodes_cfg import CFGNorm
from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow







#region---------------style
def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d


class StyleAlignedArgs:
    def __init__(self, share_attn: str) -> None:
        self.adain_keys = "k" in share_attn
        self.adain_values = "v" in share_attn
        self.adain_queries = "q" in share_attn

    share_attention: bool = True
    adain_queries: bool = True
    adain_keys: bool = True
    adain_values: bool = True


def expand_first(
    feat: T,
    scale=1.0,
) -> T:
    """
    Expand the first element so it has the same shape as the rest of the batch.
    """
    b = feat.shape[0]
    feat_style = torch.stack((feat[0], feat[b // 2])).unsqueeze(1)
    if scale == 1:
        feat_style = feat_style.expand(2, b // 2, *feat.shape[1:])
    else:
        feat_style = feat_style.repeat(1, b // 2, 1, 1, 1)
        feat_style = torch.cat([feat_style[:, :1], scale * feat_style[:, 1:]], dim=1)
    return feat_style.reshape(*feat.shape)


def concat_first(feat: T, dim=2, scale=1.0) -> T:
    """
    concat the the feature and the style feature expanded above
    """
    feat_style = expand_first(feat, scale=scale)
    return torch.cat((feat, feat_style), dim=dim)


def calc_mean_std(feat, eps: float = 1e-5) -> "tuple[T, T]":
    feat_std = (feat.var(dim=-2, keepdims=True) + eps).sqrt()
    feat_mean = feat.mean(dim=-2, keepdims=True)
    return feat_mean, feat_std

def adain(feat: T) -> T:
    feat_mean, feat_std = calc_mean_std(feat)
    feat_style_mean = expand_first(feat_mean)
    feat_style_std = expand_first(feat_std)
    feat = (feat - feat_mean) / feat_std
    feat = feat * feat_style_std + feat_style_mean
    return feat

class SharedAttentionProcessor:
    def __init__(self, args: StyleAlignedArgs, scale: float):
        self.args = args
        self.scale = scale

    def __call__(self, q, k, v, extra_options):
        if self.args.adain_queries:
            q = adain(q)
        if self.args.adain_keys:
            k = adain(k)
        if self.args.adain_values:
            v = adain(v)
        if self.args.share_attention:
            k = concat_first(k, -2, scale=self.scale)
            v = concat_first(v, -2)

        return q, k, v


def get_norm_layers(
    layer: nn.Module,
    norm_layers_: "dict[str, list[Union[nn.GroupNorm, nn.LayerNorm]]]",
    share_layer_norm: bool,
    share_group_norm: bool,
):
    if isinstance(layer, nn.LayerNorm) and share_layer_norm:
        norm_layers_["layer"].append(layer)
    if isinstance(layer, nn.GroupNorm) and share_group_norm:
        norm_layers_["group"].append(layer)
    else:
        for child_layer in layer.children():
            get_norm_layers(
                child_layer, norm_layers_, share_layer_norm, share_group_norm
            )


def register_norm_forward(
    norm_layer: Union[nn.GroupNorm, nn.LayerNorm],
) -> Union[nn.GroupNorm, nn.LayerNorm]:
    if not hasattr(norm_layer, "orig_forward"):
        setattr(norm_layer, "orig_forward", norm_layer.forward)
    orig_forward = norm_layer.orig_forward

    def forward_(hidden_states: T) -> T:
        n = hidden_states.shape[-2]
        hidden_states = concat_first(hidden_states, dim=-2)
        hidden_states = orig_forward(hidden_states)  # type: ignore
        return hidden_states[..., :n, :]

    norm_layer.forward = forward_  # type: ignore
    return norm_layer


def register_shared_norm(
    model: ModelPatcher,
    share_group_norm: bool = True,
    share_layer_norm: bool = True,
):
    norm_layers = {"group": [], "layer": []}
    get_norm_layers(model.model, norm_layers, share_layer_norm, share_group_norm)
    print(
        f"Patching {len(norm_layers['group'])} group norms, {len(norm_layers['layer'])} layer norms."
    )
    return [register_norm_forward(layer) for layer in norm_layers["group"]] + [
        register_norm_forward(layer) for layer in norm_layers["layer"]
    ]


SHARE_NORM_OPTIONS = ["both", "group", "layer", "disabled"]
SHARE_ATTN_OPTIONS = ["q+k", "q+k+v", "disabled"]

#endregion



class model_Regional:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"model": ("MODEL",),
                            "mask": ("MASK",),
                            }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "doit"
    CATEGORY = "Apt_Preset/model"

    @staticmethod
    def doit(model, mask):
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) == 3:
            mask = mask.unsqueeze(0)

        size = None

        def regional_cfg(args):
            nonlocal mask
            nonlocal size

            x = args['input']

            if mask.device != x.device:
                mask = mask.to(x.device)

            if size != (x.shape[2], x.shape[3]):
                size = (x.shape[2], x.shape[3])
                mask = torch.nn.functional.interpolate(mask, size=size, mode='bilinear', align_corners=False)

            cond_pred = args["cond_denoised"]
            uncond_pred = args["uncond_denoised"]
            cond_scale = args["cond_scale"]

            cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale * mask

            return x - cfg_result

        m = model.clone()
        m.set_model_sampler_cfg_function(regional_cfg)
        return (m,)
    


#region-------------------------------model 小组件----------------------------


from comfy_extras.nodes_freelunch import FreeU_V2
import comfy.model_patcher
import comfy.samplers
import torch
from torch import einsum
import torch.nn.functional as F
import math
from einops import rearrange, repeat
from comfy.ldm.modules.attention import optimized_attention
import comfy.samplers

# but modified to return attention scores as well as output
def attention_basic_with_sim(q, k, v, heads, mask=None, attn_precision=None):
    b, _, dim_head = q.shape
    dim_head //= heads
    scale = dim_head ** -0.5

    h = heads
    q, k, v = map(
        lambda t: t.unsqueeze(3)
        .reshape(b, -1, heads, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b * heads, -1, dim_head)
        .contiguous(),
        (q, k, v),
    )

    # force cast to fp32 to avoid overflowing
    if attn_precision == torch.float32:
        sim = einsum('b i d, b j d -> b i j', q.float(), k.float()) * scale
    else:
        sim = einsum('b i d, b j d -> b i j', q, k) * scale

    del q, k

    if mask is not None:
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim.to(v.dtype), v)
    out = (
        out.unsqueeze(0)
        .reshape(b, heads, -1, dim_head)
        .permute(0, 2, 1, 3)
        .reshape(b, -1, heads * dim_head)
    )
    return (out, sim)

def create_blur_map(x0, attn, sigma=3.0, threshold=1.0):
    # reshape and GAP the attention map
    _, hw1, hw2 = attn.shape
    b, _, lh, lw = x0.shape
    attn = attn.reshape(b, -1, hw1, hw2)
    # Global Average Pool
    mask = attn.mean(1, keepdim=False).sum(1, keepdim=False) > threshold

    total = mask.shape[-1]
    x = round(math.sqrt((lh / lw) * total))
    xx = None
    for i in range(0, math.floor(math.sqrt(total) / 2)):
        for j in [(x + i), max(1, x - i)]:
            if total % j == 0:
                xx = j
                break
        if xx is not None:
            break

    x = xx
    y = total // x

    # Reshape
    mask = (
        mask.reshape(b, x, y)
        .unsqueeze(1)
        .type(attn.dtype)
    )
    # Upsample
    mask = F.interpolate(mask, (lh, lw))

    blurred = gaussian_blur_2d(x0, kernel_size=9, sigma=sigma)
    blurred = blurred * mask + x0 * (1 - mask)
    return blurred

def gaussian_blur_2d(img, kernel_size, sigma):
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)

    pdf = torch.exp(-0.5 * (x / sigma).pow(2))

    x_kernel = pdf / pdf.sum()
    x_kernel = x_kernel.to(device=img.device, dtype=img.dtype)

    kernel2d = torch.mm(x_kernel[:, None], x_kernel[None, :])
    kernel2d = kernel2d.expand(img.shape[-3], 1, kernel2d.shape[0], kernel2d.shape[1])

    padding = [kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2]

    img = F.pad(img, padding, mode="reflect")
    img = F.conv2d(img, kernel2d, groups=img.shape[-3])
    return img

class SelfAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.01}),
                             "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                              }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "_for_testing"

    def patch(self, model, scale, blur_sigma):
        m = model.clone()

        attn_scores = None

        # TODO: make this work properly with chunked batches
        #       currently, we can only save the attn from one UNet call
        def attn_and_record(q, k, v, extra_options):
            nonlocal attn_scores
            # if uncond, save the attention scores
            heads = extra_options["n_heads"]
            cond_or_uncond = extra_options["cond_or_uncond"]
            b = q.shape[0] // len(cond_or_uncond)
            if 1 in cond_or_uncond:
                uncond_index = cond_or_uncond.index(1)
                # do the entire attention operation, but save the attention scores to attn_scores
                (out, sim) = attention_basic_with_sim(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])
                # when using a higher batch size, I BELIEVE the result batch dimension is [uc1, ... ucn, c1, ... cn]
                n_slices = heads * b
                attn_scores = sim[n_slices * uncond_index:n_slices * (uncond_index+1)]
                return out
            else:
                return optimized_attention(q, k, v, heads=heads, attn_precision=extra_options["attn_precision"])

        def post_cfg_function(args):
            nonlocal attn_scores
            uncond_attn = attn_scores

            sag_scale = scale
            sag_sigma = blur_sigma
            sag_threshold = 1.0
            model = args["model"]
            uncond_pred = args["uncond_denoised"]
            uncond = args["uncond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"]
            x = args["input"]
            if min(cfg_result.shape[2:]) <= 4: #skip when too small to add padding
                return cfg_result

            # create the adversarially blurred image
            degraded = create_blur_map(uncond_pred, uncond_attn, sag_sigma, sag_threshold)
            degraded_noised = degraded + x - uncond_pred
            # call into the UNet
            (sag,) = comfy.samplers.calc_cond_batch(model, [uncond], degraded_noised, sigma, model_options)
            return cfg_result + (degraded - sag) * sag_scale

        m.set_model_sampler_post_cfg_function(post_cfg_function, disable_cfg1_optimization=True)
        m.set_model_attn1_replace(attn_and_record, "middle", 0, 0)

        return (m, )



class PerturbedAttentionGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "model_patches/unet"

    def patch(self, model, scale):
        unet_block = "middle"
        unet_block_id = 0
        m = model.clone()

        def perturbed_attention(q, k, v, extra_options, mask=None):
            return v

        def post_cfg_function(args):
            model = args["model"]
            cond_pred = args["cond_denoised"]
            cond = args["cond"]
            cfg_result = args["denoised"]
            sigma = args["sigma"]
            model_options = args["model_options"].copy()
            x = args["input"]

            if scale == 0:
                return cfg_result

            # Replace Self-attention with PAG
            model_options = comfy.model_patcher.set_model_options_patch_replace(model_options, perturbed_attention, "attn1", unet_block, unet_block_id)
            (pag,) = comfy.samplers.calc_cond_batch(model, [cond], x, sigma, model_options)

            return cfg_result + (cond_pred - pag) * scale

        m.set_model_sampler_post_cfg_function(post_cfg_function)

        return (m,)



class model_tool_assy:   #小工具串联
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             
                            "switch_freeU": ("BOOLEAN", {"default": True}),
                            "b1": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 10.0, "step": 0.01}),
                            "b2": ("FLOAT", {"default": 1.4, "min": 0.0, "max": 10.0, "step": 0.01}),
                            "s1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 10.0, "step": 0.01}),
                            "s2": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                            "switch_PAG": ("BOOLEAN", {"default": False}),
                            "PAG_scale": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": 0.01}),
                            "switch_SAG": ("BOOLEAN", {"default": False}),
                            "SAG_scale": ("FLOAT", {"default": 0.5, "min": -2.0, "max": 5.0, "step": 0.01}),
                            "blur_sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                            "switch_CFGNorm": ("BOOLEAN", {"default": False}),
                            "strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 100.0, "step": 0.01}),
                            "switch_AuraFlow": ("BOOLEAN", {"default": False}),
                            "shift": ("FLOAT", {"default": 1.73, "min": 0.0, "max": 100.0, "step": 0.01}),                            

                              }}
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "Apt_Preset/chx_tool/Model"

    def patch(self, model, switch_freeU, b1, b2, s1, s2, switch_PAG, PAG_scale, switch_SAG, SAG_scale, blur_sigma, 
              switch_CFGNorm, strength, switch_AuraFlow, shift):

        patched_model = model        
        if switch_freeU:
            patched_model = FreeU_V2().patch(patched_model, b1, b2, s1, s2)[0]
        if switch_PAG:
            patched_model = PerturbedAttentionGuidance().patch(patched_model, PAG_scale)[0]
        if switch_SAG:
            patched_model = SelfAttentionGuidance().patch(patched_model, SAG_scale, blur_sigma)[0]

        if switch_CFGNorm:
            patched_model = CFGNorm().execute(patched_model, strength)[0]
        if switch_AuraFlow:
            patched_model = ModelSamplingAuraFlow().patch_aura( model, shift)[0]
            
        return (patched_model, )





#endregion-------------------------------model 小组件----------------------------



















