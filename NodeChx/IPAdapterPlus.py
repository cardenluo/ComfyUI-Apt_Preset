#region--------收纳------------

#region----------------jichu --------------------- 

import torch
import os
import math
import folder_paths
import comfy.model_management as model_management
from node_helpers import conditioning_set_values
from comfy.clip_vision import load as load_clip_vision
from comfy.sd import load_lora_for_models
import comfy.utils

import torch.nn as nn
from PIL import Image
try:
    import torchvision.transforms.v2 as T
except ImportError:
    import torchvision.transforms as T


from nodes import common_ksampler, CLIPTextEncode

import re
from comfy.clip_vision import clip_preprocess, Output
from ..main_unit import *





def get_clipvision_file(preset):
    preset = preset.lower()
    clipvision_list = folder_paths.get_filename_list("clip_vision")

    if preset.startswith("vit-g"):
        pattern = r'(ViT.bigG.14.*39B.b160k|ipadapter.*sdxl|sdxl.*model)\.(bin|safetensors)'
    elif preset.startswith("kolors"):
        pattern = r'clip.vit.large.patch14.336\.(bin|safetensors)'
    else:
        pattern = r'(ViT.H.14.*s32B.b79K|ipadapter.*sd15|sd1.?5.*model)\.(bin|safetensors)'
    clipvision_file = [e for e in clipvision_list if re.search(pattern, e, re.IGNORECASE)]

    clipvision_file = folder_paths.get_full_path("clip_vision", clipvision_file[0]) if clipvision_file else None

    return clipvision_file

def get_ipadapter_file(preset, is_sdxl):
    preset = preset.lower()
    ipadapter_list = folder_paths.get_filename_list("ipadapter")
    is_insightface = False
    lora_pattern = None

    if preset.startswith("light"):
        if is_sdxl:
            raise Exception("light model is not supported for SDXL")
        pattern = r'sd15.light.v11\.(safetensors|bin)$'
        # if v11 is not found, try with the old version
        if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
            pattern = r'sd15.light\.(safetensors|bin)$'
    elif preset.startswith("standard"):
        if is_sdxl:
            pattern = r'ip.adapter.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = r'ip.adapter.sd15\.(safetensors|bin)$'
    elif preset.startswith("vit-g"):
        if is_sdxl:
            pattern = r'ip.adapter.sdxl\.(safetensors|bin)$'
        else:
            pattern = r'sd15.vit.g\.(safetensors|bin)$'
    elif preset.startswith("plus ("):
        if is_sdxl:
            pattern = r'plus.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = r'ip.adapter.plus.sd15\.(safetensors|bin)$'
    elif preset.startswith("plus face"):
        if is_sdxl:
            pattern = r'plus.face.sdxl.vit.h\.(safetensors|bin)$'
        else:
            pattern = r'plus.face.sd15\.(safetensors|bin)$'
    elif preset.startswith("full"):
        if is_sdxl:
            raise Exception("full face model is not supported for SDXL")
        pattern = r'full.face.sd15\.(safetensors|bin)$'
    elif preset.startswith("faceid portrait ("):
        if is_sdxl:
            pattern = r'portrait.sdxl\.(safetensors|bin)$'
        else:
            pattern = r'portrait.v11.sd15\.(safetensors|bin)$'
            # if v11 is not found, try with the old version
            if not [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]:
                pattern = r'portrait.sd15\.(safetensors|bin)$'
        is_insightface = True
    elif preset.startswith("faceid portrait unnorm"):
        if is_sdxl:
            pattern = r'portrait.sdxl.unnorm\.(safetensors|bin)$'
        else:
            raise Exception("portrait unnorm model is not supported for SD1.5")
        is_insightface = True
    elif preset == "faceid":
        if is_sdxl:
            pattern = r'faceid.sdxl\.(safetensors|bin)$'
            lora_pattern = r'faceid.sdxl.lora\.safetensors$'
        else:
            pattern = r'faceid.sd15\.(safetensors|bin)$'
            lora_pattern = r'faceid.sd15.lora\.safetensors$'
        is_insightface = True
    elif preset.startswith("faceid plus -"):
        if is_sdxl:
            raise Exception("faceid plus model is not supported for SDXL")
        pattern = r'faceid.plus.sd15\.(safetensors|bin)$'
        lora_pattern = r'faceid.plus.sd15.lora\.safetensors$'
        is_insightface = True
    elif preset.startswith("faceid plus v2"):
        if is_sdxl:
            pattern = r'faceid.plusv2.sdxl\.(safetensors|bin)$'
            lora_pattern = r'faceid.plusv2.sdxl.lora\.safetensors$'
        else:
            pattern = r'faceid.plusv2.sd15\.(safetensors|bin)$'
            lora_pattern = r'faceid.plusv2.sd15.lora\.safetensors$'
        is_insightface = True
    # Community's models
    elif preset.startswith("composition"):
        if is_sdxl:
            pattern = r'plus.composition.sdxl\.safetensors$'
        else:
            pattern = r'plus.composition.sd15\.safetensors$'
    elif preset.startswith("kolors"):
        if is_sdxl:
            pattern = r'(ip_adapter_plus_general|kolors.ip.adapter.plus)\.(safetensors|bin)$'
        else:
            raise Exception("Only supported for Kolors model")
    else:
        raise Exception(f"invalid type '{preset}'")

    ipadapter_file = [e for e in ipadapter_list if re.search(pattern, e, re.IGNORECASE)]
    ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file[0]) if ipadapter_file else None

    return ipadapter_file, is_insightface, lora_pattern

def get_lora_file(pattern):
    lora_list = folder_paths.get_filename_list("loras")
    lora_file = [e for e in lora_list if re.search(pattern, e, re.IGNORECASE)]
    lora_file = folder_paths.get_full_path("loras", lora_file[0]) if lora_file else None

    return lora_file

def ipadapter_model_loader(file):
    model = comfy.utils.load_torch_file(file, safe_load=True)

    if file.lower().endswith(".safetensors"):
        st_model = {"image_proj": {}, "ip_adapter": {}}
        for key in model.keys():
            if key.startswith("image_proj."):
                st_model["image_proj"][key.replace("image_proj.", "")] = model[key]
            elif key.startswith("ip_adapter."):
                st_model["ip_adapter"][key.replace("ip_adapter.", "")] = model[key]
            elif key.startswith("adapter_modules."):
                st_model["ip_adapter"][key.replace("adapter_modules.", "")] = model[key]
        model = st_model
        del st_model
    elif "adapter_modules" in model.keys():
        model["ip_adapter"] = model.pop("adapter_modules")

    if not "ip_adapter" in model.keys() or not model["ip_adapter"]:
        raise Exception("invalid IPAdapter model {}".format(file))

    if 'plusv2' in file.lower():
        model["faceidplusv2"] = True
    
    if 'unnorm' in file.lower():
        model["portraitunnorm"] = True

    return model

def insightface_loader(provider, model_name='buffalo_l'):
    try:
        from insightface.app import FaceAnalysis
    except ImportError as e:
        raise Exception(e)

    path = os.path.join(folder_paths.models_dir, "insightface")
    model = FaceAnalysis(name=model_name, root=path, providers=[provider + 'ExecutionProvider',])
    model.prepare(ctx_id=0, det_size=(640, 640))
    return model

def split_tiles(embeds, num_split):
    _, H, W, _ = embeds.shape
    out = []
    for x in embeds:
        x = x.unsqueeze(0)
        h, w = H // num_split, W // num_split
        x_split = torch.cat([x[:, i*h:(i+1)*h, j*w:(j+1)*w, :] for i in range(num_split) for j in range(num_split)], dim=0)    
        out.append(x_split)
    
    x_split = torch.stack(out, dim=0)
    
    return x_split

def merge_hiddenstates(x, tiles):
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        tile_size = int((embeds.shape[1]-1) ** 0.5)
        grid_size = int(num_tiles ** 0.5)

        # Extract class tokens
        class_tokens = embeds[:, 0, :]  # Save class tokens: [num_tiles, embeds[-1]]
        avg_class_token = class_tokens.mean(dim=0, keepdim=True).unsqueeze(0)  # Average token, shape: [1, 1, embeds[-1]]

        patch_embeds = embeds[:, 1:, :]  # Shape: [num_tiles, tile_size^2, embeds[-1]]
        reshaped = patch_embeds.reshape(grid_size, grid_size, tile_size, tile_size, embeds.shape[-1])

        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size, embeds[-1]]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged.permute(0, 3, 1, 2), (tile_size, tile_size)).permute(0, 2, 3, 1)
        flattened = pooled.reshape(1, tile_size*tile_size, embeds.shape[-1])
        
        # Add back the class token
        with_class = torch.cat([avg_class_token, flattened], dim=1)  # Shape: original shape
        out.append(with_class)
    
    out = torch.cat(out, dim=0)

    return out

def merge_embeddings(x, tiles): # TODO: this needs so much testing that I don't even
    chunk_size = tiles*tiles
    x = x.split(chunk_size)

    out = []
    for embeds in x:
        num_tiles = embeds.shape[0]
        grid_size = int(num_tiles ** 0.5)
        tile_size = int(embeds.shape[1] ** 0.5)
        reshaped = embeds.reshape(grid_size, grid_size, tile_size, tile_size)
        
        # Merge the tiles
        merged = torch.cat([torch.cat([reshaped[i, j] for j in range(grid_size)], dim=1) 
                            for i in range(grid_size)], dim=0)
        
        merged = merged.unsqueeze(0)  # Shape: [1, grid_size*tile_size, grid_size*tile_size]
        
        # Pool to original size
        pooled = torch.nn.functional.adaptive_avg_pool2d(merged, (tile_size, tile_size))  # pool to [1, tile_size, tile_size]
        pooled = pooled.flatten(1)  # flatten to [1, tile_size^2]
        out.append(pooled)
    out = torch.cat(out, dim=0)
    
    return out

def encode_image_masked(clip_vision, image, mask=None, batch_size=0, tiles=1, ratio=1.0, clipvision_size=224):
    # full image embeds
    embeds = encode_image_masked_(clip_vision, image, mask, batch_size, clipvision_size=clipvision_size)
    tiles = min(tiles, 16)

    if tiles > 1:
        # split in tiles
        image_split = split_tiles(image, tiles)

        # get the embeds for each tile
        embeds_split = Output()
        for i in image_split:
            encoded = encode_image_masked_(clip_vision, i, mask, batch_size, clipvision_size=clipvision_size)
            if not hasattr(embeds_split, "image_embeds"):
                #embeds_split["last_hidden_state"] = encoded["last_hidden_state"]
                embeds_split["image_embeds"] = encoded["image_embeds"]
                embeds_split["penultimate_hidden_states"] = encoded["penultimate_hidden_states"]
            else:
                #embeds_split["last_hidden_state"] = torch.cat((embeds_split["last_hidden_state"], encoded["last_hidden_state"]), dim=0)
                embeds_split["image_embeds"] = torch.cat((embeds_split["image_embeds"], encoded["image_embeds"]), dim=0)
                embeds_split["penultimate_hidden_states"] = torch.cat((embeds_split["penultimate_hidden_states"], encoded["penultimate_hidden_states"]), dim=0)

        #embeds_split['last_hidden_state'] = merge_hiddenstates(embeds_split['last_hidden_state'])
        embeds_split["image_embeds"] = merge_embeddings(embeds_split["image_embeds"], tiles)
        embeds_split["penultimate_hidden_states"] = merge_hiddenstates(embeds_split["penultimate_hidden_states"], tiles)

        #embeds['last_hidden_state'] = torch.cat([embeds_split['last_hidden_state'], embeds['last_hidden_state']])
        if embeds['image_embeds'].shape[0] > 1: # if we have more than one image we need to average the embeddings for consistency
            embeds['image_embeds'] = embeds['image_embeds']*ratio + embeds_split['image_embeds']*(1-ratio)
            embeds['penultimate_hidden_states'] = embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']*(1-ratio)
            #embeds['image_embeds'] = (embeds['image_embeds']*ratio + embeds_split['image_embeds']) / 2
            #embeds['penultimate_hidden_states'] = (embeds['penultimate_hidden_states']*ratio + embeds_split['penultimate_hidden_states']) / 2
        else: # otherwise we can concatenate them, they can be averaged later
            embeds['image_embeds'] = torch.cat([embeds['image_embeds']*ratio, embeds_split['image_embeds']])
            embeds['penultimate_hidden_states'] = torch.cat([embeds['penultimate_hidden_states']*ratio, embeds_split['penultimate_hidden_states']])

    #del embeds_split

    return embeds

def encode_image_masked_(clip_vision, image, mask=None, batch_size=0, clipvision_size=224):
    model_management.load_model_gpu(clip_vision.patcher)
    outputs = Output()

    if batch_size == 0:
        batch_size = image.shape[0]
    elif batch_size > image.shape[0]:
        batch_size = image.shape[0]

    image_batch = torch.split(image, batch_size, dim=0)

    for img in image_batch:
        img = img.to(clip_vision.load_device)
        pixel_values = clip_preprocess(img, size=clipvision_size).float()

        # TODO: support for multiple masks
        if mask is not None:
            pixel_values = pixel_values * mask.to(clip_vision.load_device)

        out = clip_vision.model(pixel_values=pixel_values, intermediate_output=-2)

        if not hasattr(outputs, "last_hidden_state"):
            outputs["last_hidden_state"] = out[0].to(model_management.intermediate_device())
            outputs["image_embeds"] = out[2].to(model_management.intermediate_device())
            outputs["penultimate_hidden_states"] = out[1].to(model_management.intermediate_device())
        else:
            outputs["last_hidden_state"] = torch.cat((outputs["last_hidden_state"], out[0].to(model_management.intermediate_device())), dim=0)
            outputs["image_embeds"] = torch.cat((outputs["image_embeds"], out[2].to(model_management.intermediate_device())), dim=0)
            outputs["penultimate_hidden_states"] = torch.cat((outputs["penultimate_hidden_states"], out[1].to(model_management.intermediate_device())), dim=0)

    del img, pixel_values, out
    torch.cuda.empty_cache()

    return outputs

def tensor_to_size(source, dest_size):
    if isinstance(dest_size, torch.Tensor):
        dest_size = dest_size.shape[0]
    source_size = source.shape[0]

    if source_size < dest_size:
        shape = [dest_size - source_size] + [1]*(source.dim()-1)
        source = torch.cat((source, source[-1:].repeat(shape)), dim=0)
    elif source_size > dest_size:
        source = source[:dest_size]

    return source

def min_(tensor_list):
    # return the element-wise min of the tensor list.
    x = torch.stack(tensor_list)
    mn = x.min(axis=0)[0]
    return torch.clamp(mn, min=0)

def max_(tensor_list):
    # return the element-wise max of the tensor list.
    x = torch.stack(tensor_list)
    mx = x.max(axis=0)[0]
    return torch.clamp(mx, max=1)

# From https://github.com/Jamy-L/Pytorch-Contrast-Adaptive-Sharpening/
def contrast_adaptive_sharpening(image, amount):
    img = T.functional.pad(image, (1, 1, 1, 1)).cpu()

    a = img[..., :-2, :-2]
    b = img[..., :-2, 1:-1]
    c = img[..., :-2, 2:]
    d = img[..., 1:-1, :-2]
    e = img[..., 1:-1, 1:-1]
    f = img[..., 1:-1, 2:]
    g = img[..., 2:, :-2]
    h = img[..., 2:, 1:-1]
    i = img[..., 2:, 2:]

    # Computing contrast
    cross = (b, d, e, f, h)
    mn = min_(cross)
    mx = max_(cross)

    diag = (a, c, g, i)
    mn2 = min_(diag)
    mx2 = max_(diag)
    mx = mx + mx2
    mn = mn + mn2

    # Computing local weight
    inv_mx = torch.reciprocal(mx)
    amp = inv_mx * torch.minimum(mn, (2 - mx))

    # scaling
    amp = torch.sqrt(amp)
    w = - amp * (amount * (1/5 - 1/8) + 1/8)
    div = torch.reciprocal(1 + 4*w)

    output = ((b + d + f + h)*w + e) * div
    output = torch.nan_to_num(output)
    output = output.clamp(0, 1)

    return output

def tensor_to_image(tensor):
    image = tensor.mul(255).clamp(0, 255).byte().cpu()
    image = image[..., [2, 1, 0]].numpy()
    return image

def image_to_tensor(image):
    tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
    tensor = tensor[..., [2, 1, 0]]
    return tensor



import math
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        weight = (q * scale) @ (k * scale).transpose(-2, -1)  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        max_seq_len: int = 257,  # CLIP tokens + CLS token
        apply_pos_emb: bool = False,
        num_latents_mean_pooled: int = 0,  # number of latents derived from mean pooled representation of the sequence
    ):
        super().__init__()
        self.pos_emb = nn.Embedding(max_seq_len, embedding_dim) if apply_pos_emb else None

        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.to_latents_from_mean_pooled_seq = (
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * num_latents_mean_pooled),
                Rearrange("b (n d) -> b n d", n=num_latents_mean_pooled),
            )
            if num_latents_mean_pooled > 0
            else None
        )

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        if self.pos_emb is not None:
            n, device = x.shape[1], x.device
            pos_emb = self.pos_emb(torch.arange(n, device=device))
            x = x + pos_emb

        latents = self.latents.repeat(x.size(0), 1, 1)

        x = self.proj_in(x)

        if self.to_latents_from_mean_pooled_seq:
            meanpooled_seq = masked_mean(x, dim=1, mask=torch.ones(x.shape[:2], device=x.device, dtype=torch.bool))
            meanpooled_latents = self.to_latents_from_mean_pooled_seq(meanpooled_seq)
            latents = torch.cat((meanpooled_latents, latents), dim=-2)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        return self.norm_out(latents)


def masked_mean(t, *, dim, mask=None):
    if mask is None:
        return t.mean(dim=dim)

    denom = mask.sum(dim=dim, keepdim=True)
    mask = rearrange(mask, "b n -> b n 1")
    masked_t = t.masked_fill(~mask, 0.0)

    return masked_t.sum(dim=dim) / denom.clamp(min=1e-5)


class FacePerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim=768,
        depth=4,
        dim_head=64,
        heads=16,
        embedding_dim=1280,
        output_dim=768,
        ff_mult=4,
    ):
        super().__init__()

        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, latents, x):
        x = self.proj_in(x)
        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
        latents = self.proj_out(latents)
        return self.norm_out(latents)


class MLPProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(clip_embeddings_dim, clip_embeddings_dim),
            nn.GELU(),
            nn.Linear(clip_embeddings_dim, cross_attention_dim),
            nn.LayerNorm(cross_attention_dim)
        )

    def forward(self, image_embeds):
        clip_extra_context_tokens = self.proj(image_embeds)
        return clip_extra_context_tokens

class MLPProjModelFaceId(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, id_embeds):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x

class ProjModelFaceIdPlus(nn.Module):
    def __init__(self, cross_attention_dim=768, id_embeddings_dim=512, clip_embeddings_dim=1280, num_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.proj = nn.Sequential(
            nn.Linear(id_embeddings_dim, id_embeddings_dim*2),
            nn.GELU(),
            nn.Linear(id_embeddings_dim*2, cross_attention_dim*num_tokens),
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

        self.perceiver_resampler = FacePerceiverResampler(
            dim=cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=cross_attention_dim // 64,
            embedding_dim=clip_embeddings_dim,
            output_dim=cross_attention_dim,
            ff_mult=4,
        )

    def forward(self, id_embeds, clip_embeds, scale=1.0, shortcut=False):
        x = self.proj(id_embeds)
        x = x.reshape(-1, self.num_tokens, self.cross_attention_dim)
        x = self.norm(x)
        out = self.perceiver_resampler(x, clip_embeds)
        if shortcut:
            out = x + scale * out
        return out

class ImageProjModel(nn.Module):
    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        x = self.proj(embeds).reshape(-1, self.clip_extra_context_tokens, self.cross_attention_dim)
        x = self.norm(x)
        return x



import torch
import math
import torch.nn.functional as F
from comfy.ldm.modules.attention import optimized_attention


class Attn2Replace:
    def __init__(self, callback=None, **kwargs):
        self.callback = [callback]
        self.kwargs = [kwargs]

    def add(self, callback, **kwargs):
        self.callback.append(callback)
        self.kwargs.append(kwargs)

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, q, k, v, extra_options):
        dtype = q.dtype
        out = optimized_attention(q, k, v, extra_options["n_heads"])
        sigma = extra_options["sigmas"].detach().cpu()[0].item() if 'sigmas' in extra_options else 999999999.9

        for i, callback in enumerate(self.callback):
            if sigma <= self.kwargs[i]["sigma_start"] and sigma >= self.kwargs[i]["sigma_end"]:
                out = out + callback(out, q, k, v, extra_options, **self.kwargs[i])

        return out.to(dtype=dtype)

def ipadapter_attention(out, q, k, v, extra_options, module_key='', ipadapter=None, weight=1.0, cond=None, cond_alt=None, uncond=None, weight_type="linear", mask=None, sigma_start=0.0, sigma_end=1.0, unfold_batch=False, embeds_scaling='V only', **kwargs):
    dtype = q.dtype
    cond_or_uncond = extra_options["cond_or_uncond"]
    block_type = extra_options["block"][0]
    #block_id = extra_options["block"][1]
    t_idx = extra_options["transformer_index"]
    layers = 11 if '101_to_k_ip' in ipadapter.ip_layers.to_kvs else 16
    k_key = module_key + "_to_k_ip"
    v_key = module_key + "_to_v_ip"

    # extra options for AnimateDiff
    ad_params = extra_options['ad_params'] if "ad_params" in extra_options else None

    b = q.shape[0]
    seq_len = q.shape[1]
    batch_prompt = b // len(cond_or_uncond)
    _, _, oh, ow = extra_options["original_shape"]

    if weight_type == 'ease in':
        weight = weight * (0.05 + 0.95 * (1 - t_idx / layers))
    elif weight_type == 'ease out':
        weight = weight * (0.05 + 0.95 * (t_idx / layers))
    elif weight_type == 'ease in-out':
        weight = weight * (0.05 + 0.95 * (1 - abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'reverse in-out':
        weight = weight * (0.05 + 0.95 * (abs(t_idx - (layers/2)) / (layers/2)))
    elif weight_type == 'weak input' and block_type == 'input':
        weight = weight * 0.2
    elif weight_type == 'weak middle' and block_type == 'middle':
        weight = weight * 0.2
    elif weight_type == 'weak output' and block_type == 'output':
        weight = weight * 0.2
    elif weight_type == 'strong middle' and (block_type == 'input' or block_type == 'output'):
        weight = weight * 0.2
    elif isinstance(weight, dict):
        if t_idx not in weight:
            return 0

        if weight_type == "style transfer precise":
            if layers == 11 and t_idx == 3:
                uncond = cond
                cond = cond * 0
            elif layers == 16 and (t_idx == 4 or t_idx == 5):
                uncond = cond
                cond = cond * 0
        elif weight_type == "composition precise":
            if layers == 11 and t_idx != 3:
                uncond = cond
                cond = cond * 0
            elif layers == 16 and (t_idx != 4 and t_idx != 5):
                uncond = cond
                cond = cond * 0

        weight = weight[t_idx]

        if cond_alt is not None and t_idx in cond_alt:
            cond = cond_alt[t_idx]
            del cond_alt

    if unfold_batch:
        # Check AnimateDiff context window
        if ad_params is not None and ad_params["sub_idxs"] is not None:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, ad_params["full_length"])
                weight = torch.Tensor(weight[ad_params["sub_idxs"]])
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            # if image length matches or exceeds full_length get sub_idx images
            if cond.shape[0] >= ad_params["full_length"]:
                cond = torch.Tensor(cond[ad_params["sub_idxs"]])
                uncond = torch.Tensor(uncond[ad_params["sub_idxs"]])
            # otherwise get sub_idxs images
            else:
                cond = tensor_to_size(cond, ad_params["full_length"])
                uncond = tensor_to_size(uncond, ad_params["full_length"])
                cond = cond[ad_params["sub_idxs"]]
                uncond = uncond[ad_params["sub_idxs"]]
        else:
            if isinstance(weight, torch.Tensor):
                weight = tensor_to_size(weight, batch_prompt)
                if torch.all(weight == 0):
                    return 0
                weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
            elif weight == 0:
                return 0

            cond = tensor_to_size(cond, batch_prompt)
            uncond = tensor_to_size(uncond, batch_prompt)

        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond)
    else:
        # TODO: should we always convert the weights to a tensor?
        if isinstance(weight, torch.Tensor):
            weight = tensor_to_size(weight, batch_prompt)
            if torch.all(weight == 0):
                return 0
            weight = weight.repeat(len(cond_or_uncond), 1, 1) # repeat for cond and uncond
        elif weight == 0:
            return 0
        
        k_cond = ipadapter.ip_layers.to_kvs[k_key](cond).repeat(batch_prompt, 1, 1)
        k_uncond = ipadapter.ip_layers.to_kvs[k_key](uncond).repeat(batch_prompt, 1, 1)
        v_cond = ipadapter.ip_layers.to_kvs[v_key](cond).repeat(batch_prompt, 1, 1)
        v_uncond = ipadapter.ip_layers.to_kvs[v_key](uncond).repeat(batch_prompt, 1, 1)

    if len(cond_or_uncond) == 3: # TODO: cosxl, I need to check this
        ip_k = torch.cat([(k_cond, k_uncond, k_cond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond, v_cond)[i] for i in cond_or_uncond], dim=0)
    else:
        ip_k = torch.cat([(k_cond, k_uncond)[i] for i in cond_or_uncond], dim=0)
        ip_v = torch.cat([(v_cond, v_uncond)[i] for i in cond_or_uncond], dim=0)

    if embeds_scaling == 'K+mean(V) w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v_mean = torch.mean(ip_v, dim=1, keepdim=True)
        ip_v = (ip_v - ip_v_mean) + ip_v_mean * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        del ip_v_mean
    elif embeds_scaling == 'K+V w/ C penalty':
        scaling = float(ip_k.shape[2]) / 1280.0
        weight = weight * scaling
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    elif embeds_scaling == 'K+V':
        ip_k = ip_k * weight
        ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
    else:
        #ip_v = ip_v * weight
        out_ip = optimized_attention(q, ip_k, ip_v, extra_options["n_heads"])
        out_ip = out_ip * weight # I'm doing this to get the same results as before

    if mask is not None:
        mask_h = oh / math.sqrt(oh * ow / seq_len)
        mask_h = int(mask_h) + int((seq_len % int(mask_h)) != 0)
        mask_w = seq_len // mask_h

        # check if using AnimateDiff and sliding context window
        if (mask.shape[0] > 1 and ad_params is not None and ad_params["sub_idxs"] is not None):
            # if mask length matches or exceeds full_length, get sub_idx masks
            if mask.shape[0] >= ad_params["full_length"]:
                mask = torch.Tensor(mask[ad_params["sub_idxs"]])
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            else:
                mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
                mask = tensor_to_size(mask, ad_params["full_length"])
                mask = mask[ad_params["sub_idxs"]]
        else:
            mask = F.interpolate(mask.unsqueeze(1), size=(mask_h, mask_w), mode="bilinear").squeeze(1)
            mask = tensor_to_size(mask, batch_prompt)

        mask = mask.repeat(len(cond_or_uncond), 1, 1)
        mask = mask.view(mask.shape[0], -1, 1).repeat(1, 1, out.shape[2])

        # covers cases where extreme aspect ratios can cause the mask to have a wrong size
        mask_len = mask_h * mask_w
        if mask_len < seq_len:
            pad_len = seq_len - mask_len
            pad1 = pad_len // 2
            pad2 = pad_len - pad1
            mask = F.pad(mask, (0, 0, pad1, pad2), value=0.0)
        elif mask_len > seq_len:
            crop_start = (mask_len - seq_len) // 2
            mask = mask[:, crop_start:crop_start+seq_len, :]

        out_ip = out_ip * mask

    #out = out + out_ip

    return out_ip.to(dtype=dtype)





# set the models directory
if "ipadapter" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "ipadapter")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["ipadapter"]
folder_paths.folder_names_and_paths["ipadapter"] = (current_paths, folder_paths.supported_pt_extensions)

WEIGHT_TYPES = ["linear", "ease in", "ease out", 'ease in-out', 'reverse in-out', 'weak input', 'weak output', 'weak middle', 'strong middle', 'style transfer', 'composition', 'strong style transfer', 'style and composition', 'style transfer precise', 'composition precise']

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Main IPAdapter Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""



class IPAdapter(nn.Module):
    def __init__(self, ipadapter_model, cross_attention_dim=1024, output_cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4, is_sdxl=False, is_plus=False, is_full=False, is_faceid=False, is_portrait_unnorm=False, is_kwai_kolors=False, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        self.clip_embeddings_dim = clip_embeddings_dim
        self.cross_attention_dim = cross_attention_dim
        self.output_cross_attention_dim = output_cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.is_sdxl = is_sdxl
        self.is_full = is_full
        self.is_plus = is_plus
        self.is_portrait_unnorm = is_portrait_unnorm
        self.is_kwai_kolors = is_kwai_kolors

        if is_faceid and not is_portrait_unnorm:
            self.image_proj_model = self.init_proj_faceid()
        elif is_full:
            self.image_proj_model = self.init_proj_full()
        elif is_plus or is_portrait_unnorm:
            self.image_proj_model = self.init_proj_plus()
        else:
            self.image_proj_model = self.init_proj()

        self.image_proj_model.load_state_dict(ipadapter_model["image_proj"])
        self.ip_layers = To_KV(ipadapter_model["ip_adapter"], encoder_hid_proj=encoder_hid_proj, weight_kolors=weight_kolors)

    def init_proj(self):
        image_proj_model = ImageProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim,
            clip_extra_context_tokens=self.clip_extra_context_tokens
        )
        return image_proj_model

    def init_proj_plus(self):
        image_proj_model = Resampler(
            dim=self.cross_attention_dim,
            depth=4,
            dim_head=64,
            heads=20 if self.is_sdxl and not self.is_kwai_kolors else 12,
            num_queries=self.clip_extra_context_tokens,
            embedding_dim=self.clip_embeddings_dim,
            output_dim=self.output_cross_attention_dim,
            ff_mult=4
        )
        return image_proj_model

    def init_proj_full(self):
        image_proj_model = MLPProjModel(
            cross_attention_dim=self.cross_attention_dim,
            clip_embeddings_dim=self.clip_embeddings_dim
        )
        return image_proj_model

    def init_proj_faceid(self):
        if self.is_plus:
            image_proj_model = ProjModelFaceIdPlus(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                clip_embeddings_dim=self.clip_embeddings_dim,
                num_tokens=self.clip_extra_context_tokens,
            )
        else:
            image_proj_model = MLPProjModelFaceId(
                cross_attention_dim=self.cross_attention_dim,
                id_embeddings_dim=512,
                num_tokens=self.clip_extra_context_tokens,
            )
        return image_proj_model

    @torch.inference_mode()
    def get_image_embeds(self, clip_embed, clip_embed_zeroed, batch_size):
        torch_device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        clip_embed = torch.split(clip_embed, batch_size, dim=0)
        clip_embed_zeroed = torch.split(clip_embed_zeroed, batch_size, dim=0)
        
        image_prompt_embeds = []
        uncond_image_prompt_embeds = []

        for ce, cez in zip(clip_embed, clip_embed_zeroed):
            image_prompt_embeds.append(self.image_proj_model(ce.to(torch_device)).to(intermediate_device))
            uncond_image_prompt_embeds.append(self.image_proj_model(cez.to(torch_device)).to(intermediate_device))

        del clip_embed, clip_embed_zeroed

        image_prompt_embeds = torch.cat(image_prompt_embeds, dim=0)
        uncond_image_prompt_embeds = torch.cat(uncond_image_prompt_embeds, dim=0)

        torch.cuda.empty_cache()

        #image_prompt_embeds = self.image_proj_model(clip_embed)
        #uncond_image_prompt_embeds = self.image_proj_model(clip_embed_zeroed)
        return image_prompt_embeds, uncond_image_prompt_embeds

    @torch.inference_mode()
    def get_image_embeds_faceid_plus(self, face_embed, clip_embed, s_scale, shortcut, batch_size):
        torch_device = model_management.get_torch_device()
        intermediate_device = model_management.intermediate_device()

        if batch_size == 0:
            batch_size = clip_embed.shape[0]
            intermediate_device = torch_device
        elif batch_size > clip_embed.shape[0]:
            batch_size = clip_embed.shape[0]

        face_embed_batch = torch.split(face_embed, batch_size, dim=0)
        clip_embed_batch = torch.split(clip_embed, batch_size, dim=0)

        embeds = []
        for face_embed, clip_embed in zip(face_embed_batch, clip_embed_batch):
            embeds.append(self.image_proj_model(face_embed.to(torch_device), clip_embed.to(torch_device), scale=s_scale, shortcut=shortcut).to(intermediate_device))

        embeds = torch.cat(embeds, dim=0)
        del face_embed_batch, clip_embed_batch
        torch.cuda.empty_cache()
        #embeds = self.image_proj_model(face_embed, clip_embed, scale=s_scale, shortcut=shortcut)
        return embeds

class To_KV(nn.Module):
    def __init__(self, state_dict, encoder_hid_proj=None, weight_kolors=1.0):
        super().__init__()

        if encoder_hid_proj is not None:
            hid_proj = nn.Linear(encoder_hid_proj["weight"].shape[1], encoder_hid_proj["weight"].shape[0], bias=True)
            hid_proj.weight.data = encoder_hid_proj["weight"] * weight_kolors
            hid_proj.bias.data = encoder_hid_proj["bias"] * weight_kolors

        self.to_kvs = nn.ModuleDict()
        for key, value in state_dict.items():
            if encoder_hid_proj is not None:
                linear_proj = nn.Linear(value.shape[1], value.shape[0], bias=False)
                linear_proj.weight.data = value
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Sequential(hid_proj, linear_proj)
            else:
                self.to_kvs[key.replace(".weight", "").replace(".", "_")] = nn.Linear(value.shape[1], value.shape[0], bias=False)
                self.to_kvs[key.replace(".weight", "").replace(".", "_")].weight.data = value

def set_model_patch_replace(model, patch_kwargs, key):
    to = model.model_options["transformer_options"].copy()
    if "patches_replace" not in to:
        to["patches_replace"] = {}
    else:
        to["patches_replace"] = to["patches_replace"].copy()

    if "attn2" not in to["patches_replace"]:
        to["patches_replace"]["attn2"] = {}
    else:
        to["patches_replace"]["attn2"] = to["patches_replace"]["attn2"].copy()

    if key not in to["patches_replace"]["attn2"]:
        to["patches_replace"]["attn2"][key] = Attn2Replace(ipadapter_attention, **patch_kwargs)
        model.model_options["transformer_options"] = to
    else:
        to["patches_replace"]["attn2"][key].add(ipadapter_attention, **patch_kwargs)

def ipadapter_execute(model,
                      ipadapter,
                      clipvision,
                      insightface=None,
                      image=None,
                      image_composition=None,
                      image_negative=None,
                      weight=1.0,
                      weight_composition=1.0,
                      weight_faceidv2=None,
                      weight_kolors=1.0,
                      weight_type="linear",
                      combine_embeds="concat",
                      start_at=0.0,
                      end_at=1.0,
                      attn_mask=None,
                      pos_embed=None,
                      neg_embed=None,
                      unfold_batch=False,
                      embeds_scaling='V only',
                      layer_weights=None,
                      encode_batch_size=0,
                      style_boost=None,
                      composition_boost=None,
                      enhance_tiles=1,
                      enhance_ratio=1.0,):
    device = model_management.get_torch_device()
    dtype = model_management.unet_dtype()
    if dtype not in [torch.float32, torch.float16, torch.bfloat16]:
        dtype = torch.float16 if model_management.should_use_fp16() else torch.float32

    is_full = "proj.3.weight" in ipadapter["image_proj"]
    is_portrait_unnorm = "portraitunnorm" in ipadapter
    is_plus = (is_full or "latents" in ipadapter["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter["image_proj"]) and not is_portrait_unnorm
    output_cross_attention_dim = ipadapter["ip_adapter"]["1.to_k_ip.weight"].shape[1]
    is_sdxl = output_cross_attention_dim == 2048
    is_kwai_kolors_faceid = "perceiver_resampler.layers.0.0.to_out.weight" in ipadapter["image_proj"] and ipadapter["image_proj"]["perceiver_resampler.layers.0.0.to_out.weight"].shape[0] == 4096
    is_faceidv2 = "faceidplusv2" in ipadapter or is_kwai_kolors_faceid
    is_kwai_kolors = (is_sdxl and "layers.0.0.to_out.weight" in ipadapter["image_proj"] and ipadapter["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048) or is_kwai_kolors_faceid
    is_portrait = "proj.2.weight" in ipadapter["image_proj"] and not "proj.3.weight" in ipadapter["image_proj"] and not "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] and not is_kwai_kolors_faceid
    is_faceid = is_portrait or "0.to_q_lora.down.weight" in ipadapter["ip_adapter"] or is_portrait_unnorm or is_kwai_kolors_faceid

    if is_faceid and not insightface:
        raise Exception("insightface model is required for FaceID models")

    if is_faceidv2:
        weight_faceidv2 = weight_faceidv2 if weight_faceidv2 is not None else weight*2

    if is_kwai_kolors_faceid:
        cross_attention_dim = 4096
    elif is_kwai_kolors:
        cross_attention_dim = 2048
    elif (is_plus and is_sdxl and not is_faceid) or is_portrait_unnorm:
        cross_attention_dim = 1280
    else:
        cross_attention_dim = output_cross_attention_dim
    
    if is_kwai_kolors_faceid:
        clip_extra_context_tokens = 6
    elif (is_plus and not is_faceid) or is_portrait or is_portrait_unnorm:
        clip_extra_context_tokens = 16
    else:
        clip_extra_context_tokens = 4

    if image is not None and image.shape[1] != image.shape[2]:
        print("\033[33mINFO: the IPAdapter reference image is not a square, CLIPImageProcessor will resize and crop it at the center. If the main focus of the picture is not in the middle the result might not be what you are expecting.\033[0m")

    if isinstance(weight, list):
        weight = torch.tensor(weight).unsqueeze(-1).unsqueeze(-1).to(device, dtype=dtype) if unfold_batch else weight[0]

    if style_boost is not None:
        weight_type = "style transfer precise"
    elif composition_boost is not None:
        weight_type = "composition precise"

    # special weight types
    if layer_weights is not None and layer_weights != '':
        weight = { int(k): float(v)*weight for k, v in [x.split(":") for x in layer_weights.split(",")] }
        weight_type = weight_type if weight_type == "style transfer precise" or weight_type == "composition precise" else "linear"
    elif weight_type == "style transfer":
        weight = { 6:weight } if is_sdxl else { 0:weight, 1:weight, 2:weight, 3:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition":
        weight = { 3:weight } if is_sdxl else { 4:weight*0.25, 5:weight }
    elif weight_type == "strong style transfer":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style and composition":
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "strong style and composition":
        if is_sdxl:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight_composition, 4:weight, 5:weight, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition, 5:weight_composition, 6:weight, 7:weight, 8:weight, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "style transfer precise":
        weight_composition = style_boost if style_boost is not None else weight
        if is_sdxl:
            weight = { 3:weight_composition, 6:weight }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }
    elif weight_type == "composition precise":
        weight_composition = weight
        weight = composition_boost if composition_boost is not None else weight
        if is_sdxl:
            weight = { 0:weight*.1, 1:weight*.1, 2:weight*.1, 3:weight_composition, 4:weight*.1, 5:weight*.1, 6:weight, 7:weight*.1, 8:weight*.1, 9:weight*.1, 10:weight*.1 }
        else:
            weight = { 0:weight, 1:weight, 2:weight, 3:weight, 4:weight_composition*0.25, 5:weight_composition, 6:weight*.1, 7:weight*.1, 8:weight*.1, 9:weight, 10:weight, 11:weight, 12:weight, 13:weight, 14:weight, 15:weight }

    clipvision_size = 224 if not is_kwai_kolors else 336

    img_comp_cond_embeds = None
    face_cond_embeds = None
    if is_faceid:
        if insightface is None:
            raise Exception("Insightface model is required for FaceID models")

        from insightface.utils import face_align

        insightface.det_model.input_size = (640,640) # reset the detection size
        image_iface = tensor_to_image(image)
        face_cond_embeds = []
        image = []

        for i in range(image_iface.shape[0]):
            for size in [(size, size) for size in range(640, 256, -64)]:
                insightface.det_model.input_size = size # TODO: hacky but seems to be working
                face = insightface.get(image_iface[i])
                if face:
                    if not is_portrait_unnorm:
                        face_cond_embeds.append(torch.from_numpy(face[0].normed_embedding).unsqueeze(0))
                    else:
                        face_cond_embeds.append(torch.from_numpy(face[0].embedding).unsqueeze(0))
                    image.append(image_to_tensor(face_align.norm_crop(image_iface[i], landmark=face[0].kps, image_size=336 if is_kwai_kolors_faceid else 256 if is_sdxl else 224)))

                    if 640 not in size:
                        print(f"\033[33mINFO: InsightFace detection resolution lowered to {size}.\033[0m")
                    break
            else:
                raise Exception('InsightFace: No face detected.')
        face_cond_embeds = torch.stack(face_cond_embeds).to(device, dtype=dtype)
        image = torch.stack(image)
        del image_iface, face

    if image is not None:
        img_cond_embeds = encode_image_masked(clipvision, image, batch_size=encode_batch_size, tiles=enhance_tiles, ratio=enhance_ratio, clipvision_size=clipvision_size)
        if image_composition is not None:
            img_comp_cond_embeds = encode_image_masked(clipvision, image_composition, batch_size=encode_batch_size, tiles=enhance_tiles, ratio=enhance_ratio, clipvision_size=clipvision_size)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            image_negative = image_negative if image_negative is not None else torch.zeros([1, clipvision_size, clipvision_size, 3])
            img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size, clipvision_size=clipvision_size).penultimate_hidden_states
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds if not is_faceid else face_cond_embeds
            if image_negative is not None and not is_faceid:
                img_uncond_embeds = encode_image_masked(clipvision, image_negative, batch_size=encode_batch_size, clipvision_size=clipvision_size).image_embeds
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
            if image_composition is not None:
                img_comp_cond_embeds = img_comp_cond_embeds.image_embeds
        del image_negative, image_composition

        image = None if not is_faceid else image # if it's face_id we need the cropped face for later
    elif pos_embed is not None:
        img_cond_embeds = pos_embed

        if neg_embed is not None:
            img_uncond_embeds = neg_embed
        else:
            if is_plus:
                img_uncond_embeds = encode_image_masked(clipvision, torch.zeros([1, clipvision_size, clipvision_size, 3]), clipvision_size=clipvision_size).penultimate_hidden_states
            else:
                img_uncond_embeds = torch.zeros_like(img_cond_embeds)
        del pos_embed, neg_embed
    else:
        raise Exception("Images or Embeds are required")

    # ensure that cond and uncond have the same batch size
    img_uncond_embeds = tensor_to_size(img_uncond_embeds, img_cond_embeds.shape[0])

    img_cond_embeds = img_cond_embeds.to(device, dtype=dtype)
    img_uncond_embeds = img_uncond_embeds.to(device, dtype=dtype)
    if img_comp_cond_embeds is not None:
        img_comp_cond_embeds = img_comp_cond_embeds.to(device, dtype=dtype)

    # combine the embeddings if needed
    if combine_embeds != "concat" and img_cond_embeds.shape[0] > 1 and not unfold_batch:
        if combine_embeds == "add":
            img_cond_embeds = torch.sum(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.sum(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.sum(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "subtract":
            img_cond_embeds = img_cond_embeds[0] - torch.mean(img_cond_embeds[1:], dim=0)
            img_cond_embeds = img_cond_embeds.unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = face_cond_embeds[0] - torch.mean(face_cond_embeds[1:], dim=0)
                face_cond_embeds = face_cond_embeds.unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = img_comp_cond_embeds[0] - torch.mean(img_comp_cond_embeds[1:], dim=0)
                img_comp_cond_embeds = img_comp_cond_embeds.unsqueeze(0)
        elif combine_embeds == "average":
            img_cond_embeds = torch.mean(img_cond_embeds, dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds, dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds, dim=0).unsqueeze(0)
        elif combine_embeds == "norm average":
            img_cond_embeds = torch.mean(img_cond_embeds / torch.norm(img_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if face_cond_embeds is not None:
                face_cond_embeds = torch.mean(face_cond_embeds / torch.norm(face_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
            if img_comp_cond_embeds is not None:
                img_comp_cond_embeds = torch.mean(img_comp_cond_embeds / torch.norm(img_comp_cond_embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        img_uncond_embeds = img_uncond_embeds[0].unsqueeze(0) # TODO: better strategy for uncond could be to average them

    if attn_mask is not None:
        attn_mask = attn_mask.to(device, dtype=dtype)

    encoder_hid_proj = None

    if is_kwai_kolors_faceid and hasattr(model.model, "diffusion_model") and hasattr(model.model.diffusion_model, "encoder_hid_proj"):
        encoder_hid_proj = model.model.diffusion_model.encoder_hid_proj.state_dict()

    ipa = IPAdapter(
        ipadapter,
        cross_attention_dim=cross_attention_dim,
        output_cross_attention_dim=output_cross_attention_dim,
        clip_embeddings_dim=img_cond_embeds.shape[-1],
        clip_extra_context_tokens=clip_extra_context_tokens,
        is_sdxl=is_sdxl,
        is_plus=is_plus,
        is_full=is_full,
        is_faceid=is_faceid,
        is_portrait_unnorm=is_portrait_unnorm,
        is_kwai_kolors=is_kwai_kolors,
        encoder_hid_proj=encoder_hid_proj,
        weight_kolors=weight_kolors
    ).to(device, dtype=dtype)

    if is_faceid and is_plus:
        cond = ipa.get_image_embeds_faceid_plus(face_cond_embeds, img_cond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
        # TODO: check if noise helps with the uncond face embeds
        uncond = ipa.get_image_embeds_faceid_plus(torch.zeros_like(face_cond_embeds), img_uncond_embeds, weight_faceidv2, is_faceidv2, encode_batch_size)
    else:
        cond, uncond = ipa.get_image_embeds(img_cond_embeds, img_uncond_embeds, encode_batch_size)
        if img_comp_cond_embeds is not None:
            cond_comp = ipa.get_image_embeds(img_comp_cond_embeds, img_uncond_embeds, encode_batch_size)[0]

    cond = cond.to(device, dtype=dtype)
    uncond = uncond.to(device, dtype=dtype)

    cond_alt = None
    if img_comp_cond_embeds is not None:
        cond_alt = { 3: cond_comp.to(device, dtype=dtype) }

    del img_cond_embeds, img_uncond_embeds, img_comp_cond_embeds, face_cond_embeds

    sigma_start = model.get_model_object("model_sampling").percent_to_sigma(start_at)
    sigma_end = model.get_model_object("model_sampling").percent_to_sigma(end_at)

    patch_kwargs = {
        "ipadapter": ipa,
        "weight": weight,
        "cond": cond,
        "cond_alt": cond_alt,
        "uncond": uncond,
        "weight_type": weight_type,
        "mask": attn_mask,
        "sigma_start": sigma_start,
        "sigma_end": sigma_end,
        "unfold_batch": unfold_batch,
        "embeds_scaling": embeds_scaling,
    }

    number = 0
    if not is_sdxl:
        for id in [1,2,4,5,7,8]: # id of input_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("input", id))
            number += 1
        for id in [3,4,5,6,7,8,9,10,11]: # id of output_blocks that have cross attention
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("output", id))
            number += 1
        patch_kwargs["module_key"] = str(number*2+1)
        set_model_patch_replace(model, patch_kwargs, ("middle", 1))
    else:
        for id in [4,5,7,8]: # id of input_blocks that have cross attention
            block_indices = range(2) if id in [4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("input", id, index))
                number += 1
        for id in range(6): # id of output_blocks that have cross attention
            block_indices = range(2) if id in [3, 4, 5] else range(10) # transformer_depth
            for index in block_indices:
                patch_kwargs["module_key"] = str(number*2+1)
                set_model_patch_replace(model, patch_kwargs, ("output", id, index))
                number += 1
        for index in range(10):
            patch_kwargs["module_key"] = str(number*2+1)
            set_model_patch_replace(model, patch_kwargs, ("middle", 1, index))
            number += 1

    return (model, image)

"""
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 Loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

#endregion----------------jichu --------------------- 


class IPAdapterUnifiedLoader:
    def __init__(self):
        self.lora = None
        self.clipvision = { "file": None, "model": None }
        self.ipadapter = { "file": None, "model": None }
        self.insightface = { "provider": None, "model": None }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['LIGHT - SD1.5 only (low strength)', 'STANDARD (medium strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}

    RETURN_TYPES = ("MODEL", "IPADAPTER", )
    RETURN_NAMES = ("model", "ipadapter", )
    FUNCTION = "load_models"
    CATEGORY = "ipadapter"

    def load_models(self, model, preset, lora_strength=0.0, provider="CPU", ipadapter=None):
        pipeline = { "clipvision": { 'file': None, 'model': None }, "ipadapter": { 'file': None, 'model': None }, "insightface": { 'provider': None, 'model': None } }
        if ipadapter is not None:
            pipeline = ipadapter

        # 1. Load the clipvision model
        clipvision_file = get_clipvision_file(preset)
        if clipvision_file is None:
            raise Exception("ClipVision model not found.")

        if clipvision_file != self.clipvision['file']:
            if clipvision_file != pipeline['clipvision']['file']:
                self.clipvision['file'] = clipvision_file
                self.clipvision['model'] = load_clip_vision(clipvision_file)
                print(f"\033[33mINFO: Clip Vision model loaded from {clipvision_file}\033[0m")
            else:
                self.clipvision = pipeline['clipvision']

        # 2. Load the ipadapter model
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))
        ipadapter_file, is_insightface, lora_pattern = get_ipadapter_file(preset, is_sdxl)
        if ipadapter_file is None:
            raise Exception("IPAdapter model not found.")

        if ipadapter_file != self.ipadapter['file']:
            if pipeline['ipadapter']['file'] != ipadapter_file:
                self.ipadapter['file'] = ipadapter_file
                self.ipadapter['model'] = ipadapter_model_loader(ipadapter_file)
                print(f"\033[33mINFO: IPAdapter model loaded from {ipadapter_file}\033[0m")
            else:
                self.ipadapter = pipeline['ipadapter']

        # 3. Load the lora model if needed
        if lora_pattern is not None:
            lora_file = get_lora_file(lora_pattern)
            lora_model = None
            if lora_file is None:
                raise Exception("LoRA model not found.")

            if self.lora is not None:
                if lora_file == self.lora['file']:
                    lora_model = self.lora['model']
                else:
                    self.lora = None
                    torch.cuda.empty_cache()

            if lora_model is None:
                lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)
                self.lora = { 'file': lora_file, 'model': lora_model }
                print(f"\033[33mINFO: LoRA model loaded from {lora_file}\033[0m")

            if lora_strength > 0:
                model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

        # 4. Load the insightface model if needed
        if is_insightface:
            if provider != self.insightface['provider']:
                if pipeline['insightface']['provider'] != provider:
                    self.insightface['provider'] = provider
                    self.insightface['model'] = insightface_loader(provider)
                    print(f"\033[33mINFO: InsightFace model loaded with {provider} provider\033[0m")
                else:
                    self.insightface = pipeline['insightface']

        return (model, { 'clipvision': self.clipvision, 'ipadapter': self.ipadapter, 'insightface': self.insightface }, )


class IPAdapterUnifiedLoaderFaceID(IPAdapterUnifiedLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['FACEID', 'FACEID PLUS - SD1.5 only', 'FACEID PLUS V2', 'FACEID PORTRAIT (style transfer)', 'FACEID PORTRAIT UNNORM - SDXL only (strong)'], ),
            "lora_strength": ("FLOAT", { "default": 0.6, "min": 0, "max": 1, "step": 0.01 }),
            "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}

    RETURN_NAMES = ("MODEL", "ipadapter", )
    CATEGORY = "ipadapter/faceid"


class IPAdapterUnifiedLoaderCommunity(IPAdapterUnifiedLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['Composition', 'Kolors'], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}

    CATEGORY = "ipadapter/loaders"


class IPAdapterModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "ipadapter_file": (folder_paths.get_filename_list("ipadapter"), )}}

    RETURN_TYPES = ("IPADAPTER",)
    FUNCTION = "load_ipadapter_model"
    CATEGORY = "ipadapter/loaders"

    def load_ipadapter_model(self, ipadapter_file):
        ipadapter_file = folder_paths.get_full_path("ipadapter", ipadapter_file)
        return (ipadapter_model_loader(ipadapter_file),)


class IPAdapterInsightFaceLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (['buffalo_l', 'antelopev2'], )
            },
        }

    RETURN_TYPES = ("INSIGHTFACE",)
    FUNCTION = "load_insightface"
    CATEGORY = "ipadapter/loaders"

    def load_insightface(self, model_name):
        return (insightface_loader("CUDA", model_name=model_name),)



#region--------------------Main Apply Nodes------------------

class IPAdapterSimple:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "weight_type": (['standard', 'prompt is more important', 'style transfer'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, image, weight, start_at, end_at, weight_type, attn_mask=None):
        if weight_type.startswith("style"):
            weight_type = "style transfer"
        elif weight_type == "prompt is more important":
            weight_type = "ease out"
        else:
            weight_type = "linear"

        ipa_args = {
            "image": image,
            "weight": weight,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "weight_type": weight_type,
            "insightface": ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
        }

        if 'ipadapter' not in ipadapter:
            raise Exception("IPAdapter model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")
        if 'clipvision' not in ipadapter:
            raise Exception("CLIPVision model not present in the pipeline. Please load the models with the IPAdapterUnifiedLoader node.")

        return ipadapter_execute(model.clone(), ipadapter['ipadapter']['model'], ipadapter['clipvision']['model'], **ipa_args)


class IPAdapterAdvanced:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter"

    def apply_ipadapter(self, model, ipadapter, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0):
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        if image_style is not None: # we are doing style + composition transfer
            if not is_sdxl:
                raise Exception("Style + Composition transfer is only available for SDXL models at the moment.") # TODO: check feasibility for SD1.5 models

            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style

            weight_type = "strong style and composition" if expand_style else "style and composition"
        if ipadapter_params is not None: # we are doing batch processing
            image = ipadapter_params['image']
            attn_mask = ipadapter_params['attn_mask']
            weight = ipadapter_params['weight']
            weight_type = ipadapter_params['weight_type']
            start_at = ipadapter_params['start_at']
            end_at = ipadapter_params['end_at']
        else:
            # at this point weight can be a list from the batch-weight or a single float
            weight = [weight]

        image = image if isinstance(image, list) else [image]

        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
                "composition_boost": composition_boost,
                "enhance_tiles": enhance_tiles,
                "enhance_ratio": enhance_ratio,
                "weight_kolors": weight_kolors,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter_model, clip_vision, **ipa_args)

        del ipadapter
        return (work_model, face_image, )


class IPAdapterBatch(IPAdapterAdvanced):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "encode_batch_size": ("INT", { "default": 0, "min": 0, "max": 4096 }),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }


class IPAdapterStyleComposition(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image_style": ("IMAGE",),
                "image_composition": ("IMAGE",),
                "weight_style": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_composition": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "expand_style": ("BOOLEAN", { "default": False }),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"], {"default": "average"}),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    CATEGORY = "ipadapter/style_composition"


class IPAdapterStyleCompositionBatch(IPAdapterStyleComposition):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image_style": ("IMAGE",),
                "image_composition": ("IMAGE",),
                "weight_style": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_composition": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "expand_style": ("BOOLEAN", { "default": False }),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }


class IPAdapterFaceID(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    CATEGORY = "ipadapter/faceid"
    RETURN_TYPES = ("MODEL","IMAGE",)
    RETURN_NAMES = ("MODEL", "face_image", )


class IPAAdapterFaceIDBatch(IPAdapterFaceID):
    def __init__(self):
        self.unfold_batch = True


class IPAdapterFaceIDKolors(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_kolors": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    CATEGORY = "ipadapter/faceid"
    RETURN_TYPES = ("MODEL","IMAGE",)
    RETURN_NAMES = ("MODEL", "face_image", )


class IPAdapterTiled:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "sharpening": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL", "IMAGE", "MASK", )
    RETURN_NAMES = ("MODEL", "tiles", "masks", )
    FUNCTION = "apply_tiled"
    CATEGORY = "ipadapter/tiled"

    def apply_tiled(self, model, ipadapter, image, weight, weight_type, start_at, end_at, sharpening, combine_embeds="concat", image_negative=None, attn_mask=None, clip_vision=None, embeds_scaling='V only', encode_batch_size=0):
        # 1. Select the models
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        # 2. Extract the tiles
        tile_size = 256     # I'm using 256 instead of 224 as it is more likely divisible by the latent size, it will be downscaled to 224 by the clip vision encoder
        _, oh, ow, _ = image.shape
        if attn_mask is None:
            attn_mask = torch.ones([1, oh, ow], dtype=image.dtype, device=image.device)

        image = image.permute([0,3,1,2])
        attn_mask = attn_mask.unsqueeze(1)
        # the mask should have the same proportions as the reference image and the latent
        attn_mask = T.Resize((oh, ow), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        # if the image is almost a square, we crop it to a square
        if oh / ow > 0.75 and oh / ow < 1.33:
            # crop the image to a square
            image = T.CenterCrop(min(oh, ow))(image)
            resize = (tile_size*2, tile_size*2)

            attn_mask = T.CenterCrop(min(oh, ow))(attn_mask)
        # otherwise resize the smallest side and the other proportionally
        else:
            resize = (int(tile_size * ow / oh), tile_size) if oh < ow else (tile_size, int(tile_size * oh / ow))

         # using PIL for better results
        imgs = []
        for img in image:
            img = T.ToPILImage()(img)
            img = img.resize(resize, resample=Image.Resampling['LANCZOS'])
            imgs.append(T.ToTensor()(img))
        image = torch.stack(imgs)
        del imgs, img

        # we don't need a high quality resize for the mask
        attn_mask = T.Resize(resize[::-1], interpolation=T.InterpolationMode.BICUBIC, antialias=True)(attn_mask)

        # we allow a maximum of 4 tiles
        if oh / ow > 4 or oh / ow < 0.25:
            crop = (tile_size, tile_size*4) if oh < ow else (tile_size*4, tile_size)
            image = T.CenterCrop(crop)(image)
            attn_mask = T.CenterCrop(crop)(attn_mask)

        attn_mask = attn_mask.squeeze(1)

        if sharpening > 0:
            image = contrast_adaptive_sharpening(image, sharpening)

        image = image.permute([0,2,3,1])

        _, oh, ow, _ = image.shape

        # find the number of tiles for each side
        tiles_x = math.ceil(ow / tile_size)
        tiles_y = math.ceil(oh / tile_size)
        overlap_x = max(0, (tiles_x * tile_size - ow) / (tiles_x - 1 if tiles_x > 1 else 1))
        overlap_y = max(0, (tiles_y * tile_size - oh) / (tiles_y - 1 if tiles_y > 1 else 1))

        base_mask = torch.zeros([attn_mask.shape[0], oh, ow], dtype=image.dtype, device=image.device)

        # extract all the tiles from the image and create the masks
        tiles = []
        masks = []
        for y in range(tiles_y):
            for x in range(tiles_x):
                start_x = int(x * (tile_size - overlap_x))
                start_y = int(y * (tile_size - overlap_y))
                tiles.append(image[:, start_y:start_y+tile_size, start_x:start_x+tile_size, :])
                mask = base_mask.clone()
                mask[:, start_y:start_y+tile_size, start_x:start_x+tile_size] = attn_mask[:, start_y:start_y+tile_size, start_x:start_x+tile_size]
                masks.append(mask)
        del mask

        # 3. Apply the ipadapter to each group of tiles
        model = model.clone()
        for i in range(len(tiles)):
            ipa_args = {
                "image": tiles[i],
                "image_negative": image_negative,
                "weight": weight,
                "weight_type": weight_type,
                "combine_embeds": combine_embeds,
                "start_at": start_at,
                "end_at": end_at,
                "attn_mask": masks[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "encode_batch_size": encode_batch_size,
            }
            # apply the ipadapter to the model without cloning it
            model, _ = ipadapter_execute(model, ipadapter_model, clip_vision, **ipa_args)

        return (model, torch.cat(tiles), torch.cat(masks), )


class IPAdapterTiledBatch(IPAdapterTiled):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "sharpening": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "encode_batch_size": ("INT", { "default": 0, "min": 0, "max": 4096 }),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }



class IPAdapterEmbeds:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "pos_embed": ("EMBEDS",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "neg_embed": ("EMBEDS",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "ipadapter/embeds"

    def apply_ipadapter(self, model, ipadapter, pos_embed, weight, weight_type, start_at, end_at, neg_embed=None, attn_mask=None, clip_vision=None, embeds_scaling='V only'):
        ipa_args = {
            "pos_embed": pos_embed,
            "neg_embed": neg_embed,
            "weight": weight,
            "weight_type": weight_type,
            "start_at": start_at,
            "end_at": end_at,
            "attn_mask": attn_mask,
            "embeds_scaling": embeds_scaling,
            "unfold_batch": self.unfold_batch,
        }

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None and neg_embed is None:
            raise Exception("Missing CLIPVision model.")

        del ipadapter

        return ipadapter_execute(model.clone(), ipadapter_model, clip_vision, **ipa_args)


class IPAdapterEmbedsBatch(IPAdapterEmbeds):
    def __init__(self):
        self.unfold_batch = True


class IPAdapterMS(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "layer_weights": ("STRING", { "default": "", "multiline": True }),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
                "insightface": ("INSIGHTFACE",),
            }
        }

    CATEGORY = "ipadapter/dev"



class IPAdapterClipVisionEnhancer(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "enhance_tiles": ("INT", { "default": 2, "min": 1, "max": 16 }),
                "enhance_ratio": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    CATEGORY = "ipadapter/dev"


class IPAdapterClipVisionEnhancerBatch(IPAdapterClipVisionEnhancer):
    def __init__(self):
        self.unfold_batch = True

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
                "enhance_tiles": ("INT", { "default": 2, "min": 1, "max": 16 }),
                "enhance_ratio": ("FLOAT", { "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05 }),
                "encode_batch_size": ("INT", { "default": 0, "min": 0, "max": 4096 }),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }


class IPAdapterFromParams(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "ipadapter_params": ("IPADAPTER_PARAMS", ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    CATEGORY = "ipadapter/params"


class IPAdapterPreciseStyleTransfer(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "style_boost": ("FLOAT", { "default": 1.0, "min": -5, "max": 5, "step": 0.05 }),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }


class IPAdapterPreciseStyleTransferBatch(IPAdapterPreciseStyleTransfer):
    def __init__(self):
        self.unfold_batch = True


class IPAdapterPreciseComposition(IPAdapterAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter": ("IPADAPTER", ),
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "composition_boost": ("FLOAT", { "default": 0.0, "min": -5, "max": 5, "step": 0.05 }),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }


class IPAdapterPreciseCompositionBatch(IPAdapterPreciseComposition):
    def __init__(self):
        self.unfold_batch = True


#endregion--------------------Main Apply Nodes------------------



#region--------------------helper------------------

class IPAdapterEncoder:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ipadapter": ("IPADAPTER",),
            "image": ("IMAGE",),
            "weight": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 3.0, "step": 0.01 }),
            },
            "optional": {
                "mask": ("MASK",),
                "clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("EMBEDS", "EMBEDS",)
    RETURN_NAMES = ("pos_embed", "neg_embed",)
    FUNCTION = "encode"
    CATEGORY = "ipadapter/embeds"

    def encode(self, ipadapter, image, weight, mask=None, clip_vision=None):
        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter
            clip_vision = clip_vision

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        is_plus = "proj.3.weight" in ipadapter_model["image_proj"] or "latents" in ipadapter_model["image_proj"] or "perceiver_resampler.proj_in.weight" in ipadapter_model["image_proj"]
        is_kwai_kolors = is_plus and "layers.0.0.to_out.weight" in ipadapter_model["image_proj"] and ipadapter_model["image_proj"]["layers.0.0.to_out.weight"].shape[0] == 2048

        clipvision_size = 224 if not is_kwai_kolors else 336

        # resize and crop the mask to 224x224
        if mask is not None and mask.shape[1:3] != torch.Size([clipvision_size, clipvision_size]):
            mask = mask.unsqueeze(1)
            transforms = T.Compose([
                T.CenterCrop(min(mask.shape[2], mask.shape[3])),
                T.Resize((clipvision_size, clipvision_size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            ])
            mask = transforms(mask).squeeze(1)
            #mask = T.Resize((image.shape[1], image.shape[2]), interpolation=T.InterpolationMode.BICUBIC, antialias=True)(mask.unsqueeze(1)).squeeze(1)

        img_cond_embeds = encode_image_masked(clip_vision, image, mask, clipvision_size=clipvision_size)

        if is_plus:
            img_cond_embeds = img_cond_embeds.penultimate_hidden_states
            img_uncond_embeds = encode_image_masked(clip_vision, torch.zeros([1, clipvision_size, clipvision_size, 3]), clipvision_size=clipvision_size).penultimate_hidden_states
        else:
            img_cond_embeds = img_cond_embeds.image_embeds
            img_uncond_embeds = torch.zeros_like(img_cond_embeds)

        if weight != 1:
            img_cond_embeds = img_cond_embeds * weight

        return (img_cond_embeds, img_uncond_embeds, )

class IPAdapterCombineEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embed1": ("EMBEDS",),
            "method": (["concat", "add", "subtract", "average", "norm average", "max", "min"], ),
        },
        "optional": {
            "embed2": ("EMBEDS",),
            "embed3": ("EMBEDS",),
            "embed4": ("EMBEDS",),
            "embed5": ("EMBEDS",),
        }}

    RETURN_TYPES = ("EMBEDS",)
    FUNCTION = "batch"
    CATEGORY = "ipadapter/embeds"

    def batch(self, embed1, method, embed2=None, embed3=None, embed4=None, embed5=None):
        if method=='concat' and embed2 is None and embed3 is None and embed4 is None and embed5 is None:
            return (embed1, )

        embeds = [embed1, embed2, embed3, embed4, embed5]
        embeds = [embed for embed in embeds if embed is not None]
        embeds = torch.cat(embeds, dim=0)

        if method == "add":
            embeds = torch.sum(embeds, dim=0).unsqueeze(0)
        elif method == "subtract":
            embeds = embeds[0] - torch.mean(embeds[1:], dim=0)
            embeds = embeds.unsqueeze(0)
        elif method == "average":
            embeds = torch.mean(embeds, dim=0).unsqueeze(0)
        elif method == "norm average":
            embeds = torch.mean(embeds / torch.norm(embeds, dim=0, keepdim=True), dim=0).unsqueeze(0)
        elif method == "max":
            embeds = torch.max(embeds, dim=0).values.unsqueeze(0)
        elif method == "min":
            embeds = torch.min(embeds, dim=0).values.unsqueeze(0)

        return (embeds, )

class IPAdapterNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "type": (["fade", "dissolve", "gaussian", "shuffle"], ),
                "strength": ("FLOAT", { "default": 1.0, "min": 0, "max": 1, "step": 0.05 }),
                "blur": ("INT", { "default": 0, "min": 0, "max": 32, "step": 1 }),
            },
            "optional": {
                "image_optional": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "make_noise"
    CATEGORY = "ipadapter/utils"

    def make_noise(self, type, strength, blur, image_optional=None):
        if image_optional is None:
            image = torch.zeros([1, 224, 224, 3])
        else:
            transforms = T.Compose([
                T.CenterCrop(min(image_optional.shape[1], image_optional.shape[2])),
                T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
            ])
            image = transforms(image_optional.permute([0,3,1,2])).permute([0,2,3,1])

        seed = int(torch.sum(image).item()) % 1000000007 # hash the image to get a seed, grants predictability
        torch.manual_seed(seed)

        if type == "fade":
            noise = torch.rand_like(image)
            noise = image * (1 - strength) + noise * strength
        elif type == "dissolve":
            mask = (torch.rand_like(image) < strength).float()
            noise = torch.rand_like(image)
            noise = image * (1-mask) + noise * mask
        elif type == "gaussian":
            noise = torch.randn_like(image) * strength
            noise = image + noise
        elif type == "shuffle":
            transforms = T.Compose([
                T.ElasticTransform(alpha=75.0, sigma=(1-strength)*3.5),
                T.RandomVerticalFlip(p=1.0),
                T.RandomHorizontalFlip(p=1.0),
            ])
            image = transforms(image.permute([0,3,1,2])).permute([0,2,3,1])
            noise = torch.randn_like(image) * (strength*0.75)
            noise = image * (1-noise) + noise

        del image
        noise = torch.clamp(noise, 0, 1)

        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            noise = T.functional.gaussian_blur(noise.permute([0,3,1,2]), blur).permute([0,2,3,1])

        return (noise, )

class PrepImageForClipVision:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "interpolation": (["LANCZOS", "BICUBIC", "HAMMING", "BILINEAR", "BOX", "NEAREST"],),
            "crop_position": (["top", "bottom", "left", "right", "center", "pad"],),
            "sharpening": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prep_image"

    CATEGORY = "ipadapter/utils"

    def prep_image(self, image, interpolation="LANCZOS", crop_position="center", sharpening=0.0):
        size = (224, 224)
        _, oh, ow, _ = image.shape
        output = image.permute([0,3,1,2])

        if crop_position == "pad":
            if oh != ow:
                if oh > ow:
                    pad = (oh - ow) // 2
                    pad = (pad, 0, pad, 0)
                elif ow > oh:
                    pad = (ow - oh) // 2
                    pad = (0, pad, 0, pad)
                output = T.functional.pad(output, pad, fill=0)
        else:
            crop_size = min(oh, ow)
            x = (ow-crop_size) // 2
            y = (oh-crop_size) // 2
            if "top" in crop_position:
                y = 0
            elif "bottom" in crop_position:
                y = oh-crop_size
            elif "left" in crop_position:
                x = 0
            elif "right" in crop_position:
                x = ow-crop_size

            x2 = x+crop_size
            y2 = y+crop_size

            output = output[:, :, y:y2, x:x2]

        imgs = []
        for img in output:
            img = T.ToPILImage()(img) # using PIL for better results
            img = img.resize(size, resample=Image.Resampling[interpolation])
            imgs.append(T.ToTensor()(img))
        output = torch.stack(imgs, dim=0)
        del imgs, img

        if sharpening > 0:
            output = contrast_adaptive_sharpening(output, sharpening)

        output = output.permute([0,2,3,1])

        return (output, )

class IPAdapterSaveEmbeds:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "embeds": ("EMBEDS",),
            "filename_prefix": ("STRING", {"default": "IP_embeds"})
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "ipadapter/embeds"

    def save(self, embeds, filename_prefix):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir)
        file = f"{filename}_{counter:05}.ipadpt"
        file = os.path.join(full_output_folder, file)

        torch.save(embeds, file)
        return (None, )

class IPAdapterLoadEmbeds:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [os.path.relpath(os.path.join(root, file), input_dir) for root, dirs, files in os.walk(input_dir) for file in files if file.endswith('.ipadpt')]
        return {"required": {"embeds": [sorted(files), ]}, }

    RETURN_TYPES = ("EMBEDS", )
    FUNCTION = "load"
    CATEGORY = "ipadapter/embeds"

    def load(self, embeds):
        path = folder_paths.get_annotated_filepath(embeds)
        return (torch.load(path).cpu(), )

class IPAdapterWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weights": ("STRING", {"default": '1.0, 0.0', "multiline": True }),
            "timing": (["custom", "linear", "ease_in_out", "ease_in", "ease_out", "random"], { "default": "linear" } ),
            "frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "end_frame": ("INT", {"default": 9999, "min": 0, "max": 9999, "step": 1 }),
            "add_starting_frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "add_ending_frames": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1 }),
            "method": (["full batch", "shift batches", "alternate batches"], { "default": "full batch" }),
            }, "optional": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT", "INT", "IMAGE", "IMAGE", "WEIGHTS_STRATEGY")
    RETURN_NAMES = ("weights", "weights_invert", "total_frames", "image_1", "image_2", "weights_strategy")
    FUNCTION = "weights"
    CATEGORY = "ipadapter/weights"

    def weights(self, weights='', timing='custom', frames=0, start_frame=0, end_frame=9999, add_starting_frames=0, add_ending_frames=0, method='full batch', weights_strategy=None, image=None):
        import random

        frame_count = image.shape[0] if image is not None else 0
        if weights_strategy is not None:
            weights = weights_strategy["weights"]
            timing = weights_strategy["timing"]
            frames = weights_strategy["frames"]
            start_frame = weights_strategy["start_frame"]
            end_frame = weights_strategy["end_frame"]
            add_starting_frames = weights_strategy["add_starting_frames"]
            add_ending_frames = weights_strategy["add_ending_frames"]
            method = weights_strategy["method"]
            frame_count = weights_strategy["frame_count"]
        else:
            weights_strategy = {
                "weights": weights,
                "timing": timing,
                "frames": frames,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "add_starting_frames": add_starting_frames,
                "add_ending_frames": add_ending_frames,
                "method": method,
                "frame_count": frame_count,
            }

        # convert the string to a list of floats separated by commas or newlines
        weights = weights.replace("\n", ",")
        weights = [float(weight) for weight in weights.split(",") if weight.strip() != ""]

        if timing != "custom":
            frames = max(frames, 2)
            start = 0.0
            end = 1.0

            if len(weights) > 0:
                start = weights[0]
                end = weights[-1]

            weights = []

            end_frame = min(end_frame, frames)
            duration = end_frame - start_frame
            if start_frame > 0:
                weights.extend([start] * start_frame)

            for i in range(duration):
                n = duration - 1
                if timing == "linear":
                    weights.append(start + (end - start) * i / n)
                elif timing == "ease_in_out":
                    weights.append(start + (end - start) * (1 - math.cos(i / n * math.pi)) / 2)
                elif timing == "ease_in":
                    weights.append(start + (end - start) * math.sin(i / n * math.pi / 2))
                elif timing == "ease_out":
                    weights.append(start + (end - start) * (1 - math.cos(i / n * math.pi / 2)))
                elif timing == "random":
                    weights.append(random.uniform(start, end))

            weights[-1] = end if timing != "random" else weights[-1]
            if end_frame < frames:
                weights.extend([end] * (frames - end_frame))

        if len(weights) == 0:
            weights = [0.0]

        frames = len(weights)

        # repeat the images for cross fade
        image_1 = None
        image_2 = None

        # Calculate the min and max of the weights
        min_weight = min(weights)
        max_weight = max(weights)

        if image is not None:

            if "shift" in method:
                image_1 = image[:-1]
                image_2 = image[1:]

                weights = weights * image_1.shape[0]
                image_1 = image_1.repeat_interleave(frames, 0)
                image_2 = image_2.repeat_interleave(frames, 0)
            elif "alternate" in method:
                image_1 = image[::2].repeat_interleave(2, 0)
                image_1 = image_1[1:]
                image_2 = image[1::2].repeat_interleave(2, 0)

                # Invert the weights relative to their own range
                mew_weights = weights + [max_weight - (w - min_weight) for w in weights]

                mew_weights = mew_weights * (image_1.shape[0] // 2)
                if image.shape[0] % 2:
                    image_1 = image_1[:-1]
                else:
                    image_2 = image_2[:-1]
                    mew_weights = mew_weights + weights

                weights = mew_weights
                image_1 = image_1.repeat_interleave(frames, 0)
                image_2 = image_2.repeat_interleave(frames, 0)
            else:
                weights = weights * image.shape[0]
                image_1 = image.repeat_interleave(frames, 0)

            # add starting and ending frames
            if add_starting_frames > 0:
                weights = [weights[0]] * add_starting_frames + weights
                image_1 = torch.cat([image[:1].repeat(add_starting_frames, 1, 1, 1), image_1], dim=0)
                if image_2 is not None:
                    image_2 = torch.cat([image[:1].repeat(add_starting_frames, 1, 1, 1), image_2], dim=0)
            if add_ending_frames > 0:
                weights = weights + [weights[-1]] * add_ending_frames
                image_1 = torch.cat([image_1, image[-1:].repeat(add_ending_frames, 1, 1, 1)], dim=0)
                if image_2 is not None:
                    image_2 = torch.cat([image_2, image[-1:].repeat(add_ending_frames, 1, 1, 1)], dim=0)

        # reverse the weights array
        weights_invert = weights[::-1]

        frame_count = len(weights)

        return (weights, weights_invert, frame_count, image_1, image_2, weights_strategy,)

class IPAdapterWeightsFromStrategy(IPAdapterWeights):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weights_strategy": ("WEIGHTS_STRATEGY",),
            }, "optional": {
                "image": ("IMAGE",),
            }
        }

class IPAdapterPromptScheduleFromWeightsStrategy():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "weights_strategy": ("WEIGHTS_STRATEGY",),
            "prompt": ("STRING", {"default": "", "multiline": True }),
            }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt_schedule", )
    FUNCTION = "prompt_schedule"
    CATEGORY = "ipadapter/weights"

    def prompt_schedule(self, weights_strategy, prompt=""):
        frames = weights_strategy["frames"]
        add_starting_frames = weights_strategy["add_starting_frames"]
        add_ending_frames = weights_strategy["add_ending_frames"]
        frame_count = weights_strategy["frame_count"]

        out = ""

        prompt = [p for p in prompt.split("\n") if p.strip() != ""]

        if len(prompt) > 0 and frame_count > 0:
            # prompt_pos must be the same size as the image batch
            if len(prompt) > frame_count:
                prompt = prompt[:frame_count]
            elif len(prompt) < frame_count:
                prompt += [prompt[-1]] * (frame_count - len(prompt))

            if add_starting_frames > 0:
                out += f"\"0\": \"{prompt[0]}\",\n"
            for i in range(frame_count):
                out += f"\"{i * frames + add_starting_frames}\": \"{prompt[i]}\",\n"
            if add_ending_frames > 0:
                out += f"\"{frame_count * frames + add_starting_frames}\": \"{prompt[-1]}\",\n"

        return (out, )

class IPAdapterCombineWeights:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "weights_1": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
            "weights_2": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05 }),
        }}
    RETURN_TYPES = ("FLOAT", "INT")
    RETURN_NAMES = ("weights", "count")
    FUNCTION = "combine"
    CATEGORY = "ipadapter/utils"

    def combine(self, weights_1, weights_2):
        if not isinstance(weights_1, list):
            weights_1 = [weights_1]
        if not isinstance(weights_2, list):
            weights_2 = [weights_2]
        weights = weights_1 + weights_2

        return (weights, len(weights), )

class IPAdapterRegionalConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            #"set_cond_area": (["default", "mask bounds"],),
            "image": ("IMAGE",),
            "image_weight": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 3.0, "step": 0.05 }),
            "prompt_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05 }),
            "weight_type": (WEIGHT_TYPES, ),
            "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
        }, "optional": {
            "mask": ("MASK",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }}

    RETURN_TYPES = ("IPADAPTER_PARAMS", "CONDITIONING", "CONDITIONING", )
    RETURN_NAMES = ("IPADAPTER_PARAMS", "POSITIVE", "NEGATIVE")
    FUNCTION = "conditioning"

    CATEGORY = "ipadapter/params"

    def conditioning(self, image, image_weight, prompt_weight, weight_type, start_at, end_at, mask=None, positive=None, negative=None):
        set_area_to_bounds = False #if set_cond_area == "default" else True

        if mask is not None:
            if positive is not None:
                positive = conditioning_set_values(positive, {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": prompt_weight})
            if negative is not None:
                negative = conditioning_set_values(negative, {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": prompt_weight})

        ipadapter_params = {
            "image": [image],
            "attn_mask": [mask],
            "weight": [image_weight],
            "weight_type": [weight_type],
            "start_at": [start_at],
            "end_at": [end_at],
        }

        return (ipadapter_params, positive, negative, )

class IPAdapterCombineParams:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "params_1": ("IPADAPTER_PARAMS",),
            "params_2": ("IPADAPTER_PARAMS",),
        }, "optional": {
            "params_3": ("IPADAPTER_PARAMS",),
            "params_4": ("IPADAPTER_PARAMS",),
            "params_5": ("IPADAPTER_PARAMS",),
        }}

    RETURN_TYPES = ("IPADAPTER_PARAMS",)
    FUNCTION = "combine"
    CATEGORY = "ipadapter/params"

    def combine(self, params_1, params_2, params_3=None, params_4=None, params_5=None):
        ipadapter_params = {
            "image": params_1["image"] + params_2["image"],
            "attn_mask": params_1["attn_mask"] + params_2["attn_mask"],
            "weight": params_1["weight"] + params_2["weight"],
            "weight_type": params_1["weight_type"] + params_2["weight_type"],
            "start_at": params_1["start_at"] + params_2["start_at"],
            "end_at": params_1["end_at"] + params_2["end_at"],
        }

        if params_3 is not None:
            ipadapter_params["image"] += params_3["image"]
            ipadapter_params["attn_mask"] += params_3["attn_mask"]
            ipadapter_params["weight"] += params_3["weight"]
            ipadapter_params["weight_type"] += params_3["weight_type"]
            ipadapter_params["start_at"] += params_3["start_at"]
            ipadapter_params["end_at"] += params_3["end_at"]
        if params_4 is not None:
            ipadapter_params["image"] += params_4["image"]
            ipadapter_params["attn_mask"] += params_4["attn_mask"]
            ipadapter_params["weight"] += params_4["weight"]
            ipadapter_params["weight_type"] += params_4["weight_type"]
            ipadapter_params["start_at"] += params_4["start_at"]
            ipadapter_params["end_at"] += params_4["end_at"]
        if params_5 is not None:
            ipadapter_params["image"] += params_5["image"]
            ipadapter_params["attn_mask"] += params_5["attn_mask"]
            ipadapter_params["weight"] += params_5["weight"]
            ipadapter_params["weight_type"] += params_5["weight_type"]
            ipadapter_params["start_at"] += params_5["start_at"]
            ipadapter_params["end_at"] += params_5["end_at"]

        return (ipadapter_params, )


#endregion--------------------helper------------------


#endregion--------收纳------------







#region--------------------修改------------------

class chx_IPA_basic:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image": ("IMAGE",),
                "preset": ([ 'STANDARD (medium strength)','LIGHT - SD1.5 only (low strength)', 'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "image_negative": ("IMAGE",),
                #"clip_vision": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = ("MODEL","IMAGE","MASK")
    RETURN_NAMES = ("model","image","mask")
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def apply_ipadapter(self, model, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0,preset=None, ):
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))
        
        image0=image
        mask0=attn_mask 
        
        model,ipadapter =IPAdapterUnifiedLoader().load_models(model, preset, lora_strength=0.0, provider="CPU", ipadapter=None) #并入

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        if image_style is not None: # we are doing style + composition transfer
            if not is_sdxl:
                raise Exception("Style + Composition transfer is only available for SDXL models at the moment.") # TODO: check feasibility for SD1.5 models

            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style

            weight_type = "strong style and composition" if expand_style else "style and composition"
        if ipadapter_params is not None: # we are doing batch processing
            image = ipadapter_params['image']
            attn_mask = ipadapter_params['attn_mask']
            weight = ipadapter_params['weight']
            weight_type = ipadapter_params['weight_type']
            start_at = ipadapter_params['start_at']
            end_at = ipadapter_params['end_at']
        else:
            # at this point weight can be a list from the batch-weight or a single float
            weight = [weight]

        image = image if isinstance(image, list) else [image]

        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
                "composition_boost": composition_boost,
                "enhance_tiles": enhance_tiles,
                "enhance_ratio": enhance_ratio,
                "weight_kolors": weight_kolors,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter_model, clip_vision, **ipa_args)

        del ipadapter
        return (work_model, image0, mask0 )



class chx_IPA_XL(chx_IPA_basic):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image_style": ("IMAGE",),
                "image_composition": ("IMAGE",),
                "preset": (['STANDARD (medium strength)','LIGHT - SD1.5 only (low strength)',  'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),
                "weight_style": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_composition": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "expand_style": ("BOOLEAN", { "default": False }),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"], {"default": "average"}),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "image_negative": ("IMAGE",),
            }
        }

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"



class IPAdapterfaceIDload:
    def __init__(self):
        self.lora = None
        self.clipvision = { "file": None, "model": None }
        self.ipadapter = { "file": None, "model": None }
        self.insightface = { "provider": None, "model": None }

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ("MODEL", ),
            "preset": (['FACEID PLUS V2', 'FACEID', 'FACEID PLUS - SD1.5 only', 'FACEID PORTRAIT (style transfer)', 'FACEID PORTRAIT UNNORM - SDXL only (strong)'], ),
            "lora_strength": ("FLOAT", { "default": 0.6, "min": 0, "max": 1, "step": 0.01 }),
            "provider": (["CPU", "CUDA", "ROCM", "DirectML", "OpenVINO", "CoreML"], ),
        },
        "optional": {
            "ipadapter": ("IPADAPTER", ),
        }}


    RETURN_TYPES = ("MODEL", "IPADAPTER", )
    RETURN_NAMES = ("model", "ipadapter", )
    FUNCTION = "load_models"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def load_models(self, model, preset, lora_strength=0.0, provider="CPU", ipadapter=None):
        pipeline = { "clipvision": { 'file': None, 'model': None }, "ipadapter": { 'file': None, 'model': None }, "insightface": { 'provider': None, 'model': None } }
        if ipadapter is not None:
            pipeline = ipadapter

        # 1. Load the clipvision model
        clipvision_file = get_clipvision_file(preset)
        if clipvision_file is None:
            raise Exception("ClipVision model not found.")

        if clipvision_file != self.clipvision['file']:
            if clipvision_file != pipeline['clipvision']['file']:
                self.clipvision['file'] = clipvision_file
                self.clipvision['model'] = load_clip_vision(clipvision_file)
                print(f"\033[33mINFO: Clip Vision model loaded from {clipvision_file}\033[0m")
            else:
                self.clipvision = pipeline['clipvision']

        # 2. Load the ipadapter model
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))
        ipadapter_file, is_insightface, lora_pattern = get_ipadapter_file(preset, is_sdxl)
        if ipadapter_file is None:
            raise Exception("IPAdapter model not found.")

        if ipadapter_file != self.ipadapter['file']:
            if pipeline['ipadapter']['file'] != ipadapter_file:
                self.ipadapter['file'] = ipadapter_file
                self.ipadapter['model'] = ipadapter_model_loader(ipadapter_file)
                print(f"\033[33mINFO: IPAdapter model loaded from {ipadapter_file}\033[0m")
            else:
                self.ipadapter = pipeline['ipadapter']

        # 3. Load the lora model if needed
        if lora_pattern is not None:
            lora_file = get_lora_file(lora_pattern)
            lora_model = None
            if lora_file is None:
                raise Exception("LoRA model not found.")

            if self.lora is not None:
                if lora_file == self.lora['file']:
                    lora_model = self.lora['model']
                else:
                    self.lora = None
                    torch.cuda.empty_cache()

            if lora_model is None:
                lora_model = comfy.utils.load_torch_file(lora_file, safe_load=True)
                self.lora = { 'file': lora_file, 'model': lora_model }
                print(f"\033[33mINFO: LoRA model loaded from {lora_file}\033[0m")

            if lora_strength > 0:
                model, _ = load_lora_for_models(model, None, lora_model, lora_strength, 0)

        # 4. Load the insightface model if needed
        if is_insightface:
            if provider != self.insightface['provider']:
                if pipeline['insightface']['provider'] != provider:
                    self.insightface['provider'] = provider
                    self.insightface['model'] = insightface_loader(provider)
                    print(f"\033[33mINFO: InsightFace model loaded with {provider} provider\033[0m")
                else:
                    self.insightface = pipeline['insightface']

        return (model, { 'clipvision': self.clipvision, 'ipadapter': self.ipadapter, 'insightface': self.insightface }, )



class chx_IPA_faceID:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image": ("IMAGE",),
                "preset": (['FACEID PLUS V2', 'FACEID', 'FACEID PLUS - SD1.5 only', 'FACEID PORTRAIT (style transfer)', 'FACEID PORTRAIT UNNORM - SDXL only (strong)'], ),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "image_negative": ("IMAGE",),
                "insightface": ("INSIGHTFACE",),
            }
        }


    RETURN_TYPES = ("MODEL","IMAGE","MASK")
    RETURN_NAMES = ("model","image","mask")
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def apply_ipadapter(self, model, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0,preset=None, ):
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

        image0=image
        mask0=attn_mask 

        model,ipadapter =IPAdapterfaceIDload().load_models( model, preset, lora_strength=0.6, provider="CPU", ipadapter=None)

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        if image_style is not None: # we are doing style + composition transfer
            if not is_sdxl:
                raise Exception("Style + Composition transfer is only available for SDXL models at the moment.") # TODO: check feasibility for SD1.5 models

            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style

            weight_type = "strong style and composition" if expand_style else "style and composition"
        if ipadapter_params is not None: # we are doing batch processing
            image = ipadapter_params['image']
            attn_mask = ipadapter_params['attn_mask']
            weight = ipadapter_params['weight']
            weight_type = ipadapter_params['weight_type']
            start_at = ipadapter_params['start_at']
            end_at = ipadapter_params['end_at']
        else:
            # at this point weight can be a list from the batch-weight or a single float
            weight = [weight]

        image = image if isinstance(image, list) else [image]

        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
                "composition_boost": composition_boost,
                "enhance_tiles": enhance_tiles,
                "enhance_ratio": enhance_ratio,
                "weight_kolors": weight_kolors,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter_model, clip_vision, **ipa_args)

        del ipadapter
        return (work_model, image0, mask0 )



class chx_IPA_region_combine:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "context": ("RUN_CONTEXT",),

            "image_weight": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 3.0, "step": 0.05 }),
            "prompt_weight": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05 }),
            "weight_type": (WEIGHT_TYPES, ),
            "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
            }, 
                
                
            "optional": {
            
            "combine_params": ("IPADAPTER_PARAMS",),
            "image": ("IMAGE",),
            "mask": ("MASK",),
            "pos": ("STRING", {"default": "","multiline": True}),
            "neg": ("STRING", {"default": "", "multiline": False}),
        }
            }


    RETURN_TYPES = ("RUN_CONTEXT","IPADAPTER_PARAMS", )
    RETURN_NAMES = ("combine_chx","combine_params", )
    FUNCTION = "conditioning"

    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def conditioning(self, image, image_weight, prompt_weight, weight_type, start_at, end_at, mask=None, positive=None, negative=None, combine_params=None, pos=None, neg=None, context=None):
        set_area_to_bounds = False
        clip = context.get("clip")

        # 初始化 positive 和 negative 为空列表
        if positive is None:
            positive = []
        if negative is None:
            negative = []

        # 如果 pos 不为空，编码为正条件
        if pos is not None and pos != '':
            positive, = CLIPTextEncode().encode(clip, pos)
        # 如果 neg 不为空，编码为负条件
        if neg is not None and neg != '':
            negative, = CLIPTextEncode().encode(clip, neg)

        # 如果 mask 存在，设置条件值
        if mask is not None:
            if positive is not None:
                positive = conditioning_set_values(positive, {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": prompt_weight})
            if negative is not None:
                negative = conditioning_set_values(negative, {"mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": prompt_weight})

        # 构建 ipadapter_params 字典
        ipadapter_params = {
            "image": [image],
            "attn_mask": [mask],
            "weight": [image_weight],
            "weight_type": [weight_type],
            "start_at": [start_at],
            "end_at": [end_at],
        }

        # 如果 combine_params 存在，合并参数
        if combine_params is not None:
            ipadapter_params["image"] += combine_params.get("image", [])
            ipadapter_params["attn_mask"] += combine_params.get("attn_mask", [])
            ipadapter_params["weight"] += combine_params.get("weight", [])
            ipadapter_params["weight_type"] += combine_params.get("weight_type", [])
            ipadapter_params["start_at"] += combine_params.get("start_at", [])
            ipadapter_params["end_at"] += combine_params.get("end_at", [])

        # 获取 context 中的正负条件并合并
        combine_pos = context.get("positive", [])  # 如果不存在，返回空列表
        positive += combine_pos
        combine_neg = context.get("negative", [])  # 如果不存在，返回空列表
        negative += combine_neg

        # 更新 context
        context = new_context(context, positive=positive, negative=negative, any=ipadapter_params)

        return (context, ipadapter_params)



class chx_IPA_apply_combine:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):

        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "preset": (['STANDARD (medium strength)','LIGHT - SD1.5 only (low strength)',  'VIT-G (medium strength)', 'PLUS (high strength)', 'PLUS FACE (portraits)', 'FULL FACE - SD1.5 only (portraits stronger)'], ),

                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "ipadapter_params": ("IPADAPTER_PARAMS", ),
                "image_negative": ("IMAGE",),

            }
        }



    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CONDITIONING",)
    RETURN_NAMES = ("context","model","combine_pos",)
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def apply_ipadapter(self, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0,preset=None,context=None, ):
        
        model = context.get("model")
        
        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))
        
        model,ipadapter =IPAdapterUnifiedLoader().load_models(model, preset, lora_strength=0.0, provider="CPU", ipadapter=None) #并入

        if 'ipadapter' in ipadapter:
            ipadapter_model = ipadapter['ipadapter']['model']
            clip_vision = clip_vision if clip_vision is not None else ipadapter['clipvision']['model']
        else:
            ipadapter_model = ipadapter

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        if image_style is not None: # we are doing style + composition transfer
            if not is_sdxl:
                raise Exception("Style + Composition transfer is only available for SDXL models at the moment.") # TODO: check feasibility for SD1.5 models

            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style

            weight_type = "strong style and composition" if expand_style else "style and composition"
            
            
            
        if ipadapter_params is None: 
            ipadapter_params = context.get("any")

        image = ipadapter_params['image']
        attn_mask = ipadapter_params['attn_mask']
        weight = ipadapter_params['weight']
        weight_type = ipadapter_params['weight_type']
        start_at = ipadapter_params['start_at']
        end_at = ipadapter_params['end_at']

        image = image if isinstance(image, list) else [image]

        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
                "composition_boost": composition_boost,
                "enhance_tiles": enhance_tiles,
                "enhance_ratio": enhance_ratio,
                "weight_kolors": weight_kolors,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter_model, clip_vision, **ipa_args)

        del ipadapter
        
        combine_pos =context.get("positive")
        context = new_context(context,  model= work_model)
        
        return (context, work_model, combine_pos )



class chx_IPA_adv:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "ipadapter_file": (folder_paths.get_filename_list("ipadapter"),{"default": "kolor_ip_adapter_plus_general.bin"}),
                "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "kolors_clip_version.bin"}), 
                "image": ("IMAGE",),
                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 5, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "image_negative": ("IMAGE",),
                "attn_mask": ("MASK",),
            }
        }
    RETURN_TYPES = ("MODEL","IMAGE","MASK")
    RETURN_NAMES = ("model","image","mask")
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"



    def apply_ipadapter(self, model, ipadapter_file, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0):
        image0 = image
        # 加载 IP-Adapter 模型
        ipadapter = IPAdapterModelLoader().load_ipadapter_model(ipadapter_file)[0]

        # 加载 CLIP Vision 模型（包含 .patcher）
        if clip_vision == "None" or clip_vision is None:
            raise Exception("CLIP Vision 模型未选择")

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_model = comfy.clip_vision.load(clip_path)

        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

        if image_style is not None:
            if not is_sdxl:
                raise Exception("Style + Composition transfer 仅支持 SDXL 模型")
            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style
            weight_type = "strong style and composition" if expand_style else "style and composition"

        if ipadapter_params is not None:
            image = ipadapter_params['image']
            attn_mask = ipadapter_params['attn_mask']
            weight = ipadapter_params['weight']
            weight_type = ipadapter_params['weight_type']
            start_at = ipadapter_params['start_at']
            end_at = ipadapter_params['end_at']
        else:
            weight = [weight]

        image = image if isinstance(image, list) else [image]
        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface if insightface is not None else ipadapter['insightface']['model'] if 'insightface' in ipadapter else None,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
                "composition_boost": composition_boost,
                "enhance_tiles": enhance_tiles,
                "enhance_ratio": enhance_ratio,
                "weight_kolors": weight_kolors,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter, clip_model, **ipa_args)

        del ipadapter
        return (work_model, image0, attn_mask,)



class chx_IPA_faceID_adv:
    def __init__(self):
        self.unfold_batch = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "image": ("IMAGE",),

                "ipadapter_file": (folder_paths.get_filename_list("ipadapter"),{"default": "kolor_ipa-faceid-plus.bin"}),
                "clip_vision": (folder_paths.get_filename_list("clip_vision"), {"default": "kolors-faceid-clipversion.bin"}), 
                "insightface": (['buffalo_l', 'antelopev2'], ),

                "weight": ("FLOAT", { "default": 1.0, "min": -1, "max": 3, "step": 0.05 }),
                "weight_faceidv2": ("FLOAT", { "default": 1.0, "min": -1, "max": 5.0, "step": 0.05 }),
                "weight_type": (WEIGHT_TYPES, ),
                "combine_embeds": (["concat", "add", "subtract", "average", "norm average"],),
                "start_at": ("FLOAT", { "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "end_at": ("FLOAT", { "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001 }),
                "embeds_scaling": (['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'], ),
            },
            "optional": {
                "attn_mask": ("MASK",),
                "image_negative": ("IMAGE",),

            }
        }


    RETURN_TYPES = ("MODEL","IMAGE","MASK")
    RETURN_NAMES = ("model","iamge","mask")
    FUNCTION = "apply_ipadapter"
    CATEGORY = "Apt_Preset/chx_tool/chx_IPA"

    def apply_ipadapter(self, model,ipadapter_file, start_at=0.0, end_at=1.0, weight=1.0, weight_style=1.0, weight_composition=1.0, expand_style=False, weight_type="linear", combine_embeds="concat", weight_faceidv2=None, image=None, image_style=None, image_composition=None, image_negative=None, clip_vision=None, attn_mask=None, insightface=None, embeds_scaling='V only', layer_weights=None, ipadapter_params=None, encode_batch_size=0, style_boost=None, composition_boost=None, enhance_tiles=1, enhance_ratio=1.0, weight_kolors=1.0,):

        image0=image
        mask0=attn_mask 

        insightface =insightface_loader(provider="CUDA", model_name="buffalo_l")

        ipadapter = IPAdapterModelLoader().load_ipadapter_model(ipadapter_file)[0]
        if clip_vision == "None" or clip_vision is None:
            raise Exception("CLIP Vision 模型未选择")

        clip_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision)
        clip_model = comfy.clip_vision.load(clip_path)

        is_sdxl = isinstance(model.model, (comfy.model_base.SDXL, comfy.model_base.SDXLRefiner, comfy.model_base.SDXL_instructpix2pix))

        # 使用之前加载的 ipadapter_model
        if not ipadapter:
            raise Exception("IPAdapter model failed to load")

        if clip_vision is None:
            raise Exception("Missing CLIPVision model.")

        if image_style is not None: # we are doing style + composition transfer
            if not is_sdxl:
                raise Exception("Style + Composition transfer is only available for SDXL models at the moment.") # TODO: check feasibility for SD1.5 models

            image = image_style
            weight = weight_style
            if image_composition is None:
                image_composition = image_style

            weight_type = "strong style and composition" if expand_style else "style and composition"
        if ipadapter_params is not None: # we are doing batch processing
            image = ipadapter_params['image']
            attn_mask = ipadapter_params['attn_mask']
            weight = ipadapter_params['weight']
            weight_type = ipadapter_params['weight_type']
            start_at = ipadapter_params['start_at']
            end_at = ipadapter_params['end_at']
        else:
            weight = [weight]

        image = image if isinstance(image, list) else [image]
        work_model = model.clone()

        for i in range(len(image)):
            if image[i] is None:
                continue

            ipa_args = {
                "image": image[i],
                "image_composition": image_composition,
                "image_negative": image_negative,
                "weight": weight[i],
                "weight_composition": weight_composition,
                "weight_faceidv2": weight_faceidv2,
                "weight_type": weight_type if not isinstance(weight_type, list) else weight_type[i],
                "combine_embeds": combine_embeds,
                "start_at": start_at if not isinstance(start_at, list) else start_at[i],
                "end_at": end_at if not isinstance(end_at, list) else end_at[i],
                "attn_mask": attn_mask if not isinstance(attn_mask, list) else attn_mask[i],
                "unfold_batch": self.unfold_batch,
                "embeds_scaling": embeds_scaling,
                "insightface": insightface,
                "layer_weights": layer_weights,
                "encode_batch_size": encode_batch_size,
                "style_boost": style_boost,
                "composition_boost": composition_boost,
                "enhance_tiles": enhance_tiles,
                "enhance_ratio": enhance_ratio,
                "weight_kolors": weight_kolors,
            }

            work_model, face_image = ipadapter_execute(work_model, ipadapter, clip_model, **ipa_args)

        del ipadapter
        return (work_model, image0, mask0 )




#endregion-----------------改进------------------





