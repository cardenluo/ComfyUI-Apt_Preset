import inspect
import logging
import math
import os
import re
import time
from functools import partial

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F

import comfy.k_diffusion.sampling
import comfy.ldm.common_dit
from comfy.ldm.minimax.model import PackedLayout
from comfy.ldm.minimax.vae import LATENTS_MEAN, LATENTS_STD
import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy.sampler_helpers
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview


def log_memory(label, device):
    if os.environ.get("SELFLIFT_MEMORY_LOG", "0") != "1":
        return
    device = torch.device(device)
    fields = []
    try:
        fields.append(f"process_rss={psutil.Process().memory_info().rss / 2**20:.2f} MiB")
        fields.append(f"system_available={psutil.virtual_memory().available / 2**20:.2f} MiB")
    except (psutil.Error, OSError) as error:
        fields.append(f"host_memory_unavailable={error}")
    if device.type == "cuda" and torch.cuda.is_initialized():
        try:
            free, total = torch.cuda.mem_get_info(device)
            fields.extend([
                f"allocated={torch.cuda.memory_allocated(device) / 2**20:.2f} MiB",
                f"reserved={torch.cuda.memory_reserved(device) / 2**20:.2f} MiB",
                f"process_peak_allocated={torch.cuda.max_memory_allocated(device) / 2**20:.2f} MiB",
                f"device_used={((total - free) / 2**20):.2f} MiB",
                f"device_free={free / 2**20:.2f} MiB",
            ])
        except RuntimeError as error:
            fields.append(f"cuda_memory_unavailable={error}")
    logging.info("[SelfLift memory] %s device=%s; %s", label, device, "; ".join(fields))


def paired_lifts(z0_low, vae, out_hw, latent_mode="nearest", latent_lifter=None,
                 need_lat=True, need_pix=True):
    """Direct latent lift and pixel-VAE re-encode of the low-res clean endpoint (Eqs. 4-5).

    z0_low: [B, C, T, h, w] video or [B, C, h, w] image clean latent prediction in the
    VAE's native latent space. latent_lifter: optional learned upsampler callable
    (z0_low, out_hw) -> latent, used for the direct lift on MiniMax H3.
    need_lat/need_pix skip branches the correction weights would discard: rho=0 needs
    no pixel anchor, the pure anchor (rho=w=1) needs no direct lift.
    Returns (z_lat, z_pix) at the target resolution, None for skipped branches.
    """
    H, W = out_hw
    if z0_low.ndim == 4:  # image latent
        z_lat = None
        if need_lat:
            z_lat = torch.nn.functional.interpolate(z0_low.float(), size=(H, W),
                                                    mode="nearest" if latent_mode == "nearest" else "bilinear")
        z_pix = None
        if need_pix:
            img = vae.decode(z0_low)  # [B, h*ratio, w*ratio, 3]
            ratio = img.shape[1] // z0_low.shape[-2]
            up = comfy.utils.common_upscale(img.movedim(-1, 1), W * ratio, H * ratio, "lanczos", "disabled").movedim(1, -1)
            del img
            z_pix = vae.encode(up).float()
            if z_lat is not None:
                z_lat = z_lat.to(z_pix.device)
        return z_lat, z_pix

    T = z0_low.shape[2]
    z_lat = None
    if need_lat:
        if latent_lifter is not None:
            z_lat = latent_lifter(z0_low, (H, W))
        else:
            if latent_mode == "bilinear":
                latent_mode = "trilinear"  # same method family, 5D name
            z_lat = torch.nn.functional.interpolate(z0_low.float(), size=(T, H, W), mode=latent_mode)

    z_pix = None
    if need_pix:
        z_pix = _pixel_anchor_video(z0_low, vae, (H, W))
        if z_lat is not None:
            z_lat = z_lat.to(z_pix.device)
    return z_lat, z_pix


def _pixel_anchor_video(z0_low, vae, out_hw):
    """Build pixel anchors independently for each video in the latent batch."""
    if z0_low.shape[0] == 1:
        return _pixel_anchor_video_single(z0_low, vae, out_hw)
    return torch.cat([
        _pixel_anchor_video_single(sample, vae, out_hw)
        for sample in z0_low.split(1)
    ], dim=0)


def _pixel_anchor_video_single(z0_low, vae, out_hw):
    """Pixel-VAE anchor for video: decode low-res, upscale, re-encode (Eq. 5).

    The upscale runs in fp16 GPU frame chunks into a preallocated buffer: a naive
    whole-video lanczos pass materializes several fp32 [F, H*ratio, W*ratio, 3] copies
    on the CPU (tens of GB on long clips), which pushes an already full RAM into swap
    and stalls the transition for minutes.
    """
    H, W = out_hw
    frames = vae.decode(z0_low)
    if frames.ndim == 5:  # VAEDecode convention: video pixels are frames-as-batch [F, H, W, C]
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    ratio = frames.shape[1] // z0_low.shape[-2]
    Hp, Wp = H * ratio, W * ratio
    work_dtype = vae.vae_dtype if vae.vae_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
    if vae.device.type == "cpu":
        work_dtype = torch.float32
    n = frames.shape[0]
    up = torch.empty((n, Hp, Wp, frames.shape[-1]), dtype=work_dtype)
    for i in range(0, n, 32):
        chunk = frames[i:i + 32].movedim(-1, 1).to(device=vae.device, dtype=work_dtype)
        chunk = torch.nn.functional.interpolate(chunk, size=(Hp, Wp), mode="bicubic", antialias=True)
        up[i:i + 32] = chunk.movedim(1, -1).to(up.device)
        del chunk
    del frames
    return vae.encode(up).float()  # the wrapper turns the frame batch back into the time dim


def artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=None):
    """Selective correction of the direct lift toward the pixel-VAE anchor (Eqs. 6-9).

    mask: optional [B, 1, (T,) H, W] generate-region mask (1 = generate).
    Partial correction uses it for risk statistics and location selection.
    """
    if rho <= 0.0 or w_max <= 0.0:
        return z_lat
    if rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0:
        return z_pix
    delta = z_pix - z_lat
    if mask is not None:
        s = delta.abs().mean(dim=1) * mask.squeeze(1)  # per-location inconsistency, [B, (T,) H, W]
        region = (mask.squeeze(1) > 0).expand_as(s)
        view = (-1,) + (1,) * (s.ndim - 1)
        thr = torch.stack([
            torch.quantile(s[b][region[b]], 1.0 - rho) if region[b].any()
            else torch.full((), float("inf"), device=s.device, dtype=s.dtype)
            for b in range(s.shape[0])
        ]).view(view)
        selected = (s >= thr) & region
    else:
        s = delta.abs().mean(dim=1)
        view = (-1,) + (1,) * (s.ndim - 1)
        flat = s.flatten(1)
        thr = torch.quantile(flat, 1.0 - rho, dim=1).view(view)
        selected = s >= thr
    if not selected.any():
        return z_lat
    s_min = s.masked_fill(~selected, float("inf")).flatten(1).amin(dim=1).view(view)
    s_max = s.masked_fill(~selected, float("-inf")).flatten(1).amax(dim=1).view(view)
    w = w_min + (w_max - w_min) * (s - s_min) / (s_max - s_min + 1e-8)
    w = torch.where(selected, w, torch.zeros_like(w)).unsqueeze(1)
    return z_lat + w * delta


_FOLDER = "latent_upscale_models"

if _FOLDER not in folder_paths.folder_names_and_paths:
    folder_paths.add_model_folder_path(_FOLDER, os.path.join(folder_paths.models_dir, _FOLDER))


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _temporal_windows(length, chunk, overlap):
    for start in range(0, length, chunk):
        end = min(length, start + chunk)
        yield start, end, max(0, start - overlap), min(length, end + overlap)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels, emb_channels, out_channels=None):
        super().__init__()
        self.out_channels = out_channels or channels
        self.in_layers = nn.Sequential(
            _normalization(channels), nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = _normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(), nn.Identity(),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Conv3d(channels, self.out_channels, 1)
            if self.out_channels != channels else nn.Identity()
        )

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels,
                                kernel_size=(kernel_size, 1, 1),
                                padding=(padding, 0, 0),
                                groups=channels)
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        identity = x
        h = self.norm(x)
        h = F.silu(h)
        h = self.dwconv(h)
        h = self.pwconv(h)
        return identity + h


class LatentResizer3D(nn.Module):
    """Pure-3D upscaler backbone with optional temporal chunking (LBH-123-AI architecture)."""

    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12,
                 channels=512,
                 temporal_every=2, temporal_kernel=5):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        embed_dim = 64
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))

        self.in_blocks = nn.ModuleList()
        for b in range(in_blocks):
            self.in_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and b % temporal_every == 0:
                self.in_blocks.append(TemporalConv(channels, temporal_kernel))

        self.out_blocks = nn.ModuleList()
        for b in range(out_blocks):
            self.out_blocks.append(ResBlockEmb3D(channels, embed_dim))
            if temporal_every > 0 and b % temporal_every == 0:
                self.out_blocks.append(TemporalConv(channels, temporal_kernel))

        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    def temporal_chunk_settings(self):
        for block in self.in_blocks:
            if isinstance(block, TemporalConv):
                return 32, block.dwconv.weight.shape[2]
        return 32, 0

    def temporal_window_budget(self, length):
        chunk, overlap = self.temporal_chunk_settings()
        return length if length <= chunk else min(length + 2 * overlap, chunk + 4 * overlap)

    def forward(self, x, scale=None, target_size=None, enable_chunking=True):
        if target_size is not None:
            size = target_size
        elif scale is not None:
            size = tuple(int(round(s * scale)) for s in x.shape[-3:])
        else:
            return x

        if size == x.shape[-3:]:
            return x

        B, C, T, H, W = x.shape

        chunk, overlap = self.temporal_chunk_settings()

        if not enable_chunking or T <= chunk:
            return self._forward_seg(x, scale, size)

        x_padded = F.pad(x, (0, 0, 0, 0, overlap, overlap), mode='replicate')

        out_full = torch.zeros(B, C, T, size[-2], size[-1], device=x.device, dtype=x.dtype)
        weight_full = torch.zeros(1, 1, T, 1, 1, device=x.device, dtype=x.dtype)

        for seg_start, seg_end, out_start, out_end in _temporal_windows(T, chunk, overlap):
            lo = out_start
            hi = out_end + 2 * overlap

            seg = x_padded[:, :, lo:hi].contiguous()
            seg_size = (hi - lo, size[-2], size[-1])
            seg_out = self._forward_seg(seg, scale, seg_size)

            s0 = (out_start + overlap) - lo
            s1 = s0 + (out_end - out_start)
            valid_out = seg_out[:, :, s0:s1]
            n_valid = out_end - out_start

            weight = torch.ones(n_valid, device=x.device, dtype=x.dtype)
            if seg_start > out_start:
                blend_len = seg_start - out_start
                weight[:blend_len] = torch.arange(1, blend_len + 1, device=x.device, dtype=x.dtype) / (blend_len + 1)
            if out_end > seg_end:
                blend_len = out_end - seg_end
                weight[-blend_len:] = torch.arange(blend_len, 0, -1, device=x.device, dtype=x.dtype) / (blend_len + 1)

            out_full[:, :, out_start:out_end] += valid_out * weight.view(1, 1, n_valid, 1, 1)
            weight_full[:, :, out_start:out_end] += weight.view(1, 1, n_valid, 1, 1)

            del seg, seg_out, valid_out

        return out_full / weight_full.clamp(min=1e-8)

    def _forward_seg(self, x, scale, size):
        scale_emb = torch.tensor(
            [scale - 1 if scale is not None else 0.0],
            dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_emb)

        x = self.conv_in(x)
        for b in self.in_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = F.interpolate(x, size=size, mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            if isinstance(b, ResBlockEmb3D):
                x = b(x, emb.expand(x.shape[0], -1))
            else:
                x = b(x)

        x = self.norm_out(x)
        x = F.silu(x)
        return self.conv_out(x)


_model_cache = {}


def list_upscaler_models():
    try:
        paths = folder_paths.get_folder_paths(_FOLDER)
    except KeyError:
        return []
    names = []
    for p in paths:
        for root, _, files in os.walk(p):
            for f in files:
                if os.path.splitext(f)[1].lower() in (".pth", ".safetensors"):
                    names.append(os.path.relpath(os.path.join(root, f), p))
    return sorted(names)


def _detect_arch(sd):
    cfg = {"in_channels": 24, "in_blocks": 12, "out_blocks": 12, "channels": 512,
           "temporal_every": 2, "temporal_kernel": 5}
    if 'conv_in.weight' in sd:
        cfg["in_channels"] = sd['conv_in.weight'].shape[1]
        cfg["channels"] = sd['conv_in.weight'].shape[0]
    in_ids, out_ids, tin, tout = set(), set(), set(), set()
    for k in sd:
        m = re.match(r'in_blocks\.(\d+)\.in_layers\.', k)
        if m: in_ids.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.in_layers\.', k)
        if m: out_ids.add(int(m.group(1)))
        m = re.match(r'in_blocks\.(\d+)\.dwconv\.weight', k)
        if m: tin.add(int(m.group(1)))
        m = re.match(r'out_blocks\.(\d+)\.dwconv\.weight', k)
        if m: tout.add(int(m.group(1)))
    if in_ids: cfg["in_blocks"] = len(in_ids)
    if out_ids: cfg["out_blocks"] = len(out_ids)
    if tin or tout:
        cfg["temporal_every"] = 2
        for k in sd:
            if k.endswith('dwconv.weight'):
                cfg["temporal_kernel"] = sd[k].shape[2]
                break
    else:
        cfg["temporal_every"] = 0
    return cfg


def _normalize_checkpoint_dtype(state_dict):
    weight = state_dict.get("conv_in.weight")
    if weight is None:
        raise ValueError("SelfLift: upscaler checkpoint is missing conv_in.weight")
    dtype = weight.dtype
    if str(dtype).startswith("torch.float8_"):
        dtype = torch.bfloat16 if any(value.dtype == torch.bfloat16 for value in state_dict.values()) else torch.float16
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"SelfLift: unsupported upscaler weight dtype {dtype}")
    if any(value.is_floating_point() and value.dtype != dtype for value in state_dict.values()):
        logging.info("SelfLift: normalizing mixed upscaler floating-point tensors to %s", dtype)
    return {name: value.to(dtype=dtype) if value.is_floating_point() else value
            for name, value in state_dict.items()}


def _load_model(model_name, device):
    key = (model_name, str(device))
    if key in _model_cache:
        return _model_cache[key]

    path = folder_paths.get_full_path_or_raise(_FOLDER, model_name)
    if not any(folder_paths.is_within_directory(p, path) for p in folder_paths.get_folder_paths(_FOLDER)):
        raise ValueError(f"SelfLift: upscaler model path is outside {_FOLDER}: {model_name}")

    sd = comfy.utils.load_torch_file(path)
    if isinstance(sd, dict) and 'model' in sd:
        sd = sd['model']
    if any(k.startswith("upscaler.") for k in sd):
        sd = {k[len("upscaler."):]: v for k, v in sd.items() if k.startswith("upscaler.")}
    sd = _normalize_checkpoint_dtype(sd)

    cfg = _detect_arch(sd)
    with torch.device("meta"):
        model = LatentResizer3D(
            in_channels=cfg["in_channels"], in_blocks=cfg["in_blocks"], out_blocks=cfg["out_blocks"],
            channels=cfg["channels"],
            temporal_every=cfg["temporal_every"], temporal_kernel=cfg["temporal_kernel"],
        )
    model.load_state_dict(sd, strict=True, assign=True)
    model = model.eval()
    patcher = comfy.model_patcher.CoreModelPatcher(
        model, load_device=device,
        offload_device=comfy.model_management.unet_offload_device())
    _model_cache[key] = patcher
    return patcher


def _inference_memory_required(model, z0_low, out_hw):
    H, W = out_hw
    T = z0_low.shape[2]
    temporal_window = model.temporal_window_budget(T)
    feature_elements = z0_low.shape[0] * model.conv_in.out_channels * temporal_window * H * W
    return feature_elements * model.conv_in.weight.element_size() * 8


def learned_latent_lift(z0_low, out_hw, model_name, device=None, force_unload=False):
    """2D/3D learned upsample of the low-res clean endpoint to the target latent size.

    z0_low: [B, 24, T, h, w] H3 video latent in VAE space. Returns [B, 24, T, H, W].

    force_unload: unload the upscaler from VRAM when the lift finishes (also on error).
    The patcher stays in `_model_cache`, so the next lift reloads it without re-reading the file.
    """
    H, W = out_hw
    if device is None:
        device = comfy.model_management.get_torch_device()
    h, w = z0_low.shape[-2], z0_low.shape[-1]
    scale = (H / h + W / w) / 2.0

    log_memory("upscaler before_load", device)
    patcher = _load_model(model_name, device)
    model = patcher.model
    memory_required = _inference_memory_required(model, z0_low, (H, W))
    length = z0_low.shape[2]
    chunk, overlap = model.temporal_chunk_settings()
    identity = (H, W) == (h, w)
    chunked = not identity and length > chunk
    windows = list(_temporal_windows(length, chunk, overlap)) if chunked else []
    actual_window = max(end - start + 2 * overlap for _, _, start, end in windows) if chunked else length
    budget_window = model.temporal_window_budget(length)
    logging.info("[SelfLift upscaler] model=%s dtype=%s input=%s target_hw=%s "
                 "spatial_lift=(%.4f, %.4f) scale_embedding=%.4f mode=%s "
                 "chunk=%d overlap=%d windows=%d max_input_window=%d budget_window=%d "
                 "estimated_workspace=%.2f MiB",
                 model_name, model.conv_in.weight.dtype, tuple(z0_low.shape), (H, W),
                 H / h, W / w, scale - 1.0, "identity" if identity else "chunked" if chunked else "full",
                 chunk, overlap if chunked else 0, len(windows) if chunked else int(not identity),
                 actual_window if not identity else 0, budget_window, memory_required / 2**20)
    try:
        comfy.model_management.load_models_gpu([patcher], memory_required=memory_required)
        log_memory("upscaler after_load", device)
        dtype = model.conv_in.weight.dtype
        mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)

        x = z0_low.to(device=device, dtype=dtype)
        x = (x - mean) / std
        out = model(x, scale=scale, target_size=(z0_low.shape[2], H, W))
        out = (out * std + mean).float().to(comfy.model_management.intermediate_device())
        log_memory("upscaler end", device)
        return out
    finally:
        if force_unload:
            # The high-resolution stage runs right after the lift and never needs the upscaler again.
            comfy.model_management.unload_model_and_clones(patcher, unload_additional_models=False)
            comfy.model_management.soft_empty_cache()
            log_memory("upscaler after_unload", device)


def _regions(length, tile_count=2):
    patches = (length + 1) // 2
    tile_count = max(1, min(tile_count, max(1, patches // 2)))
    if patches < 4 or tile_count <= 1:
        return [(0, length)]
    overlap = min(4, max(1, patches // 8))
    regions = []
    for index in range(tile_count):
        start = max(0, index * patches // tile_count - overlap) * 2
        end = min(patches, (index + 1) * patches // tile_count + overlap) * 2
        end = min(end, length)
        regions.append((start, end))
    regions[-1] = (regions[-1][0], length)
    return regions


def _packed_layout(signature, payload):
    options = {"keyframes": payload.get("keyframes"), "refs": payload.get("refs")}
    if "frame_count" in inspect.signature(PackedLayout).parameters:
        options["frame_count"] = payload.get("frame_count")
    return PackedLayout(*signature, **options)


def _tile_payload(payload, context, video, audio, axis, start, end):
    height, width = video.shape[-2:]
    padded_height, padded_width = (height + 1) // 2 * 2, (width + 1) // 2 * 2
    full_layout = payload.get("layout")
    signature = (context.shape[1], video.shape[2], padded_height, padded_width, audio.shape[-1])
    if full_layout is None or full_layout.signature != signature:
        full_layout = _packed_layout(signature, payload)
    tiled = payload.copy()
    if payload.get("keyframes"):
        keyframes = []
        for keyframe in payload["keyframes"]:
            latent = keyframe.get("latent")
            if latent is None:
                keyframes.append(dict(keyframe))
                continue
            if latent.shape[-2:] != (height, width):
                raise ValueError("SelfLift: tiled H3 keyframes must match the target latent height and width")
            region = latent.narrow(axis, start, end - start)
            keyframes.append({**keyframe, "latent": comfy.ldm.common_dit.pad_to_patch_size(
                region, (1, 2, 2)).contiguous()})
        tiled["keyframes"] = keyframes
        tiled["cond_video_latents"] = [keyframe["latent"] for keyframe in keyframes if keyframe.get("latent") is not None]
        tiled["cond_video_latents"].extend(ref["latent"] for ref in payload.get("refs") or () if "latent" in ref)
    tile_height = end - start if axis == 3 else height
    tile_width = end - start if axis == 4 else width
    layout = _packed_layout((context.shape[1], video.shape[2], (tile_height + 1) // 2 * 2,
                             (tile_width + 1) // 2 * 2, audio.shape[-1]), tiled)
    for (source_start, source_end, kind), (target_start, target_end, _) in zip(full_layout.segments, layout.segments):
        positions = full_layout.position_ids[source_start:source_end]
        if kind in ("cond", "video"):
            positions = positions.reshape(-1, padded_height // 2, padded_width // 2, 3)
            positions = positions.narrow(axis - 2, start // 2, (end - start + 1) // 2).reshape(-1, 3)
        layout.position_ids[target_start:target_end].copy_(positions)
    tiled["layout"] = layout
    return tiled


def _tiled_forward(executor, streams, timestep, context, transformer_options, minimax_payload=None,
                   n_tiles=2, plan=None, **kwargs):
    video, audio = streams
    axis = 3 if (video.shape[3] + 1) // 2 >= (video.shape[4] + 1) // 2 else 4
    length = video.shape[axis]
    regions = _regions(length, plan['tiles'] if plan is not None else n_tiles)
    if len(regions) == 1:
        return executor(streams, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)
    if kwargs.get("control") is not None:
        raise ValueError("SelfLift: high-resolution H3 tiling does not support ControlNet")
    # Keep the stitched accumulator on CPU so previous tiles do not remain on
    # the accelerator while the next tile is evaluated.
    video_output = torch.zeros(video.shape, dtype=torch.float32, device="cpu")
    audio_output = None
    weights = torch.zeros(length, dtype=torch.float32, device="cpu")
    for index, (start, end) in enumerate(regions):
        tile = video.narrow(axis, start, end - start).contiguous()
        payload = _tile_payload(minimax_payload or {}, context, video, audio, axis, start, end)
        tile_kwargs = kwargs.copy()
        if tile_kwargs.get("denoise_mask") is not None:
            tile_kwargs["denoise_mask"] = tile_kwargs["denoise_mask"].narrow(axis, start, end - start)
        predicted_video, predicted_audio = executor(
            [tile, audio], timestep, context, transformer_options.copy(), minimax_payload=payload, **tile_kwargs)
        window = torch.ones(end - start, device=video.device, dtype=torch.float32)
        if index > 0:
            overlap = min(end, regions[index - 1][1]) - start
            window[:overlap] *= (torch.arange(overlap, device=video.device, dtype=torch.float32) + 0.5) / overlap
        if index + 1 < len(regions):
            overlap = end - regions[index + 1][0]
            window[-overlap:] *= 1.0 - (torch.arange(overlap, device=video.device, dtype=torch.float32) + 0.5) / overlap
        predicted_video = predicted_video.float().cpu()
        window = window.cpu()
        weights[start:end].add_(window)
        window_shape = [1] * video.ndim
        window_shape[axis] = end - start
        video_output.narrow(axis, start, end - start).addcmul_(predicted_video.float(), window.view(window_shape))
        if audio_output is None:
            audio_output = predicted_audio.float().clone()
        del tile, payload, predicted_video, predicted_audio, window
    window_shape[axis] = length
    video_output.div_(weights.view(window_shape))
    return [video_output.to(device=video.device, dtype=video.dtype), audio_output.to(audio.dtype)]


def _condition_elements(condition, tile_height, tile_width, channels):
    text = condition.get("cross_attn")
    elements = text.shape[-2] * channels * 4 if text is not None else 0
    for keyframe in condition.get("minimax_keyframes") or []:
        shape = keyframe["latent"].shape
        elements += shape[1] * shape[2] * tile_height * tile_width
    for reference in condition.get("minimax_refs") or []:
        latent = reference.get("latent")
        if latent is not None:
            shape = latent.shape
            elements += shape[1] * shape[2] * ((shape[3] + 1) // 2 * 2) * ((shape[4] + 1) // 2 * 2)
        audio = reference.get("audio_latent")
        if audio is not None:
            elements += math.prod(audio.shape[1:])
    return elements


def _budget(model, noise_shape, conds, latent_shapes, regions, axis):
    video_shape, audio_shape = latent_shapes
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    tile_shape = list(video_shape)
    tile_shape[axis] = max(end - start for start, end in regions)
    tile_shape[3] = (tile_shape[3] + 1) // 2 * 2
    tile_shape[4] = (tile_shape[4] + 1) // 2 * 2
    tile_elements = math.prod(tile_shape[1:]) + math.prod(audio_shape[1:])
    condition_elements = max((_condition_elements(condition, tile_shape[3], tile_shape[4], video_shape[1])
                              for group in conds.values() for condition in (group or [])), default=0)
    buffer_bytes = full_elements * 4 * 8
    bytes_per_element = model.model.memory_required((1, 1, 1))
    budget_elements = tile_elements + condition_elements + math.ceil(buffer_bytes / bytes_per_element)
    budget_shape = (noise_shape[0], 1, budget_elements)
    preferred, minimum = comfy.sampler_helpers.estimate_memory(model, budget_shape, conds)
    return budget_shape, tuple(tile_shape), buffer_bytes, preferred, minimum


def _prepare_tiled_sampling(executor, model, noise_shape, conds, model_options=None,
                            force_full_load=False, force_offload=False, *, latent_shapes, plan=None):
    video_shape, audio_shape = latent_shapes
    axis = 3 if (video_shape[3] + 1) // 2 >= (video_shape[4] + 1) // 2 else 4
    full_elements = math.prod(video_shape[1:]) + math.prod(audio_shape[1:])
    if tuple(noise_shape) != (video_shape[0], 1, full_elements):
        raise ValueError("SelfLift: tiled memory planning received a different latent shape than the sampling input")
    count = 2
    if plan is not None:
        available = _available_workspace(model)
        for count in range(1, 9):
            regions = _regions(video_shape[axis], count)
            _, _, _, _, minimum = _budget(model, noise_shape, conds, latent_shapes, regions, axis)
            if minimum <= available:
                break
        count = len(regions)
        plan['tiles'] = count
        logging.info("[SelfLift tiling plan] axis=%s tiles=%d target=%.2f MiB estimate_fits=%s",
                     'H' if axis == 3 else 'W', plan['tiles'], available / 2**20, minimum <= available)
    regions = _regions(video_shape[axis], count)
    if len(regions) == 1 or force_offload:
        return executor(model, noise_shape, conds, model_options=model_options,
                        force_full_load=force_full_load, force_offload=force_offload)
    budget_shape, tile_shape, buffer_bytes, preferred, minimum = _budget(
        model, noise_shape, conds, latent_shapes, regions, axis)
    logging.info("[SelfLift tiling memory] largest_tile=%s full_audio=%s full_state_buffers=%.2f MiB "
                 "minimum=%.2f MiB preferred=%.2f MiB (ComfyUI estimates; additional models and reserves excluded)",
                 tuple(tile_shape), audio_shape, buffer_bytes * noise_shape[0] / 2**20,
                 minimum / 2**20, preferred / 2**20)
    return executor(model, budget_shape, conds, model_options=model_options,
                    force_full_load=force_full_load, force_offload=force_offload)


def _available_workspace(model):
    manager = comfy.model_management
    free = manager.get_free_memory(model.load_device)
    reclaimable = 0
    seen = set()
    for loaded in manager.loaded_models():
        patcher = loaded if callable(getattr(loaded, "loaded_size", None)) else loaded.model
        load_device = getattr(patcher, "load_device", None)
        identity = id(patcher.model)
        if load_device == model.load_device and identity not in seen:
            seen.add(identity)
            size_fn = getattr(patcher, "loaded_size", None)
            if callable(size_fn):
                reclaimable += size_fn()
    pool = min(manager.get_total_memory(model.load_device), free + reclaimable)
    weights = min(model.model_size(), pool * manager.MIN_WEIGHT_MEMORY_RATIO)
    available = max(0, pool - weights - manager.minimum_inference_memory())
    logging.info("[SelfLift tiling capacity] free=%.2f MiB reclaimable_weights=%.2f MiB "
                 "weight_allowance=%.2f MiB workspace=%.2f MiB",
                 free / 2**20, reclaimable / 2**20, weights / 2**20, available / 2**20)
    return available


def tiled_model(model, latent_shapes):
    if not isinstance(model.model, comfy.model_base.MiniMaxH3):
        raise ValueError("SelfLift: high-resolution tiling requires a MiniMax H3 model")
    if len(latent_shapes) != 2 or len(latent_shapes[0]) != 5 or len(latent_shapes[1]) != 4:
        raise ValueError("SelfLift: high-resolution tiling requires H3 video and audio latent streams")
    plan = {}
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "selflift_high_resolution_tiling", partial(_tiled_forward, plan=plan))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
                                 "selflift_high_resolution_tiling",
                                 partial(_prepare_tiled_sampling, latent_shapes=tuple(tuple(shape) for shape in latent_shapes), plan=plan))
    return patched


class _StageTimer:
    def __init__(self, stage, device, resolution=None, pixel_scale=1):
        self.stage = stage
        self.resolution = resolution
        self.pixel_scale = pixel_scale
        self.device = torch.device(device)
        self.synchronize = os.environ.get("SELFLIFT_TIMING_SYNC", "0") == "1" and self.device.type == "cuda"
        self.started = self.previous = self._now()
        timing_mode = "CUDA-synchronized wall time" if self.synchronize else "wall time; no forced CUDA sync"
        suffix = ""
        if resolution is not None:
            pixels = tuple(int(value * self.pixel_scale) for value in resolution)
            suffix = f" latent={resolution} pixels={pixels}"
        logging.info("[SelfLift timing] %s start (%s)%s", stage, timing_mode, suffix)

    def _now(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, label):
        current = self._now()
        logging.info("[SelfLift timing] %s %s: %.3fs", self.stage, label, current - self.previous)
        self.previous = current

    def finish(self):
        current = self._now()
        logging.info("[SelfLift timing] %s total: %.3fs; after last mark: %.3fs",
                     self.stage, current - self.started, current - self.previous)


def _upscaler_input():
    models = ["none"] + list_upscaler_models()
    h3_models = [name for name in models[1:] if "h3" in name.lower()]
    default = h3_models[0] if h3_models else "none"
    return (models, {"default": default,
                     "tooltip": "潜空间放大模型；选 none 时需设置 rho > 0。"})


def _streams(samples):
    if samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested:
        return comfy.nested_tensor.NestedTensor(streams)
    return streams[0]


def _validate_sampling(model_sampling, sampler):
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift requires a rectified-flow model")
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("SelfLift requires the standard Euler sampler")
    if sampler.extra_options.get("s_churn", 0.0) != 0.0:
        raise ValueError("SelfLift requires Euler with s_churn=0")


def _validate_schedule(sigmas, transition_step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("SelfLift: sigmas must be a one-dimensional floating-point tensor")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("SelfLift: sigmas must be finite and nonnegative")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(f"SelfLift: transition_step {transition_step} out of range for {sigmas.numel() - 1} steps")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("SelfLift: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("SelfLift: only the final sigma may be zero")
    if sigmas[transition_step] >= 1:
        raise ValueError("SelfLift: the high-resolution starting sigma must be less than 1")


def _validate_latent_input(latent_image):
    """Validate the latent template; returns the normalized noise_mask or None.

    Video masks accept [B, H, W], [B, 1, H, W], or [B, C|1, T|1, H, W].
    Nested AV masks also carry an audio mask matching the audio latent grid.
    A mask value of 1 generates and 0 keeps the original content.
    """
    streams, _ = _streams(latent_image["samples"])
    if not streams or streams[0].ndim not in (4, 5):
        raise ValueError("SelfLift: expected a 4D image or 5D video latent size template")
    for stream in streams:
        if stream.ndim == 0 or any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("SelfLift: latent streams must be nonempty and have the same batch size")
    mask = latent_image.get("noise_mask")
    if mask is None:
        return None
    video = streams[0].ndim == 5
    b, H, W = streams[0].shape[0], streams[0].shape[-2], streams[0].shape[-1]
    nested_mask = mask.is_nested
    if nested_mask:
        if not latent_image["samples"].is_nested:
            raise ValueError("SelfLift: nested noise_mask requires nested latent streams")
        masks = list(mask.unbind())
        if not masks or len(masks) > len(streams):
            raise ValueError("SelfLift: noise_mask stream count does not match the latent")
        for stream, stream_mask in zip(streams[1:], masks[1:]):
            if (stream_mask.ndim != stream.ndim or stream_mask.shape[0] != b
                    or stream_mask.shape[1] not in (1, stream.shape[1])
                    or tuple(stream_mask.shape[2:]) != tuple(stream.shape[2:])):
                raise ValueError("SelfLift: auxiliary noise_mask shape does not match its latent stream")
            if not torch.isfinite(stream_mask).all():
                raise ValueError("SelfLift: noise_mask must be finite")
        mask = masks[0]
    if mask.ndim == 3:  # [B, H, W]
        mask = mask[:, None]
    if video and mask.ndim == 4:  # [B, 1, H, W] shared over time
        mask = mask[:, :, None]
    if mask.ndim != (5 if video else 4) or mask.shape[1] not in (1, streams[0].shape[1]):
        raise ValueError("SelfLift: noise_mask must have shape [B, H, W], [B, 1, H, W], or [B, 1, T, H, W]")
    if mask.shape[0] != b:
        raise ValueError(f"SelfLift: noise_mask batch {mask.shape[0]} does not match the latent batch {b}")
    if tuple(mask.shape[-2:]) != (H, W):
        # masks are accepted at any resolution and resized to the latent grid,
        # matching ComfyUI's Set Latent Noise Mask convention
        lead = mask.shape[:-2]
        mask = torch.nn.functional.interpolate(
            mask.reshape(-1, 1, *mask.shape[-2:]).float(), size=(H, W), mode="bilinear", align_corners=False
        ).reshape(*lead, H, W)
    if video and mask.shape[2] not in (1, streams[0].shape[2]):
        raise ValueError(f"SelfLift: noise_mask time length {mask.shape[2]} does not match the latent frames {streams[0].shape[2]}")
    if not torch.isfinite(mask).all():
        raise ValueError("SelfLift: noise_mask must be finite")
    mask = mask.float().clamp(0.0, 1.0)
    if nested_mask:
        masks = [mask] + [m.float().clamp(0.0, 1.0) for m in masks[1:]]
        masks.extend(torch.ones_like(streams[i], dtype=torch.float32) for i in range(len(masks), len(streams)))
        return comfy.nested_tensor.NestedTensor(masks)
    return mask


def _mask_blend_fn(anchor, mask):
    """post-CFG hook pinning x0 to the clean anchor outside the generate region.

    Exact under Euler: with the unmasked x0 pinned to the original latent each
    step, the unmasked state tracks original + the step's noise exactly.
    """
    def fn(args):
        d = args["denoised"]
        m = mask.to(device=d.device, dtype=d.dtype)
        return d * m + anchor.to(device=d.device, dtype=d.dtype) * (1.0 - m)
    return fn


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(cond, h, w):
    """Keyframe cond latents share the generation grid; resize them to the low-res one."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and (lat.shape[-2] != h or lat.shape[-1] != w):
                if lat.ndim == 5:
                    # spatial-only resize per frame; never interpolates across time
                    batch, channels, frames = lat.shape[:3]
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float().permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, lat.shape[-2], lat.shape[-1]),
                        size=(h, w), mode="bilinear", align_corners=False
                    )
                    resized_latent = resized_latent.reshape(batch, frames, channels, h, w).permute(0, 2, 1, 3, 4)
                else:
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float(), size=(h, w), mode="bilinear", align_corners=False
                    )
                # per-channel, per-frame mean match: fixes bilinear's color drift without
                # restoring variance the low-res grid cannot represent
                source_mean = lat.float().mean(dim=(-2, -1), keepdim=True)
                resized_mean = resized_latent.mean(dim=(-2, -1), keepdim=True)
                kf["latent"] = (resized_latent + (source_mean - resized_mean)).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _debug_dump(vae, latents):
    """Decode transition intermediates to PNGs when SELFLIFT_DEBUG=1."""
    if os.environ.get("SELFLIFT_DEBUG", "0") != "1":
        return
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(out_dir, exist_ok=True)
    logging.warning("SelfLift: debug decoding enabled; intermediate video decodes can substantially increase time and memory")
    from PIL import Image
    for name, lat in latents.items():
        if lat is None:
            continue
        img = vae.decode(lat)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        frame = (img[0].float().cpu().numpy().clip(0.0, 1.0) * 255).round().astype("uint8")
        Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))


def progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None,
                       highres_tiling=False, model_hires=None):
    _validate_schedule(sigmas, transition_step)
    if sigmas.numel() < 2:
        return latent_image
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("SelfLift: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("SelfLift: rho must be between 0 and 1")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("SelfLift: weights must satisfy 0 <= w_min <= w_max <= 1")
    noise_mask = _validate_latent_input(latent_image)

    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    hires_base = model_hires if model_hires is not None else model
    high_model = tiled_model(hires_base, [tuple(stream.shape) for stream in streams]) if highres_tiling else hires_base
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)
    if noise_mask is None:
        mask_shape = "none"
    elif noise_mask.is_nested:
        mask_shape = tuple(tuple(mask.shape) for mask in noise_mask.unbind())
    else:
        mask_shape = tuple(noise_mask.shape)
    logging.info("[SelfLift plan] low_latent=%s target_latent=%s spatial_lift=(%.4f, %.4f) "
                 "low_nfe=%d high_nfe=%d sigma_prediction=%.8g sigma_resume=%.8g "
                 "rho=%.4f weights=(%.4f, %.4f) direct_lift=%s pixel_anchor=%s cfg=%.4f mask=%s hires_model=%s",
                 low_shape, tuple(streams[0].shape), H / h, W / w,
                 transition_step, sigmas.numel() - 1 - transition_step,
                 sigmas[transition_step - 1].item(), sigmas[transition_step].item(),
                 rho, w_min, w_max,
                 "skipped" if rho == 1.0 and w_min == 1.0 else "external" if latent_lifter is not None else latent_upsample,
                 rho > 0.0 and w_max > 0.0, cfg,
                 mask_shape,
                 "custom" if model_hires is not None else "same")

    device = comfy.model_management.intermediate_device()
    source_video = streams[0].to(device)
    audio_streams = [s.to(device) for s in streams[1:]]

    def masked_model(base, video_anchor, m):
        # pin the unmasked region to the clean anchor via a post-CFG x0 blend;
        # the nested H3 path uses ComfyUI's native AV denoise_mask instead
        if m is None:
            return base
        patched = base.clone()
        hook = _mask_blend_fn(video_anchor, m)
        patched.model_options = comfy.model_patcher.set_model_options_post_cfg_function(patched.model_options, hook)
        return patched

    if video:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(t, h, w), mode="trilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    else:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(h, w), mode="bilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    m_full = m_low = low_mask = high_mask = None
    if noise_mask is not None:
        mask_streams = list(noise_mask.unbind()) if noise_mask.is_nested else [noise_mask]
        if nested and len(mask_streams) < len(audio_streams) + 1:
            mask_streams.extend(torch.ones_like(s, dtype=torch.float32) for s in audio_streams[len(mask_streams) - 1:])
        video_mask = mask_streams[0]
        if video and video_mask.shape[2] == 1 and t > 1:
            m_full = video_mask.expand(b, video_mask.shape[1], t, H, W)
        else:
            m_full = video_mask
        if video:
            mask_channels = m_full.shape[1]
            m_low = torch.nn.functional.interpolate(
                m_full.permute(0, 2, 1, 3, 4).reshape(b * t, mask_channels, H, W).float(),
                size=(h, w), mode="bilinear", align_corners=False
            ).reshape(b, t, mask_channels, h, w).permute(0, 2, 1, 3, 4)
        else:
            m_low = torch.nn.functional.interpolate(
                m_full.float(), size=(h, w), mode="bilinear", align_corners=False)
        m_low = m_low.clamp(0.0, 1.0)
        if nested:
            low_mask = _pack([m_low] + mask_streams[1:], True)
            high_mask = _pack([m_full] + mask_streams[1:], True)
    low_anchor = low_video
    video_anchor = source_video
    low_latent = _pack([low_video] + audio_streams, nested)
    low_model = model
    if not nested:
        low_model = masked_model(model, low_anchor, m_low)
        high_model = masked_model(high_model, video_anchor, m_full)
    del source_video, low_video
    del streams
    noise_low = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))

    total_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if video:  # keyframe cond latents share the generation grid on MiniMax H3
        positive_low = _resize_keyframes(positive, h, w)
        negative_low = _resize_keyframes(negative, h, w)
    else:
        positive_low, negative_low = positive, negative

    transition = {}
    low_evaluations = 0

    def callback_low(step, x0, x, total):
        nonlocal low_evaluations
        step = low_evaluations
        low_evaluations += 1
        if low_evaluations > transition_step:
            raise RuntimeError("SelfLift: too many low-resolution callbacks for the Euler schedule")
        if step == transition_step - 1:
            transition["state"] = x
            transition["x0"] = x0
        # Preview decoders expect the target grid.  Decode a temporary lifted
        # preview while keeping the sampler state at its true low resolution.
        preview_x0 = x0
        if video:
            preview_streams, preview_nested = _streams(x0)
            preview_video = torch.nn.functional.interpolate(
                preview_streams[0].float(), size=(t, H, W), mode="trilinear", align_corners=False
            ).to(preview_streams[0].dtype)
            preview_x0 = _pack([preview_video] + preview_streams[1:], preview_nested)
            del preview_video, preview_streams
        elif x0.ndim == 4:
            preview_x0 = torch.nn.functional.interpolate(
                x0.float(), size=(H, W), mode="bilinear", align_corners=False
            ).to(x0.dtype)
        result = callback(step, preview_x0, preview_x0, total_steps)
        del preview_x0
        low_timer.mark(f"step {step + 1}/{transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    # The final low-resolution model evaluation is the Eq. 3 prediction. Its Euler
    # update is discarded and rebuilt after the resolution transition, preserving NFE.
    resolution_scale = (getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1)
                        if video else getattr(model.get_model_object("latent_format"), "spacial_downscale_ratio", 1))
    low_timer = _StageTimer("low_resolution", low_model.load_device, (h, w), resolution_scale)
    log_memory("low_resolution start", low_model.load_device)
    comfy.samplers.sample(low_model, noise_low, positive_low, negative_low, cfg, low_model.load_device,
                          sampler, sigmas[:transition_step + 1], low_model.model_options,
                          latent_image=low_latent, denoise_mask=low_mask, callback=callback_low,
                          disable_pbar=disable_pbar, seed=seed)
    if low_evaluations != transition_step:
        raise RuntimeError(f"SelfLift: expected {transition_step} low-resolution callbacks, received {low_evaluations}; check sampler wrappers")
    low_timer.finish()
    log_memory("low_resolution end", model.load_device)
    transition_timer = _StageTimer("transition", model.load_device, (H, W), resolution_scale)
    low_streams, nested = _streams(transition.pop("state"))
    x0_streams, _ = _streams(transition.pop("x0"))
    sigma_k = sigmas[transition_step - 1]
    sigma_next = sigmas[transition_step]
    auxiliary_next = [_euler_step(state.to(device), denoised.to(device), sigma_k, sigma_next)
                      for state, denoised in zip(low_streams[1:], x0_streams[1:])]
    latent_format = model.get_model_object("latent_format")
    z0_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    del low_latent, noise_low, low_streams, x0_streams, positive_low, negative_low
    transition_timer.mark("prepare_endpoint")
    log_memory("transition endpoint_ready", model.load_device)

    # Artifact-Aware Consistency Lift (Eqs. 4-9); skip branches the weights discard
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    z_lat_vae, z_pix_vae = paired_lifts(
        z0_low_vae, vae, (H, W), latent_upsample, latent_lifter,
        need_lat=need_lat, need_pix=need_pix)
    transition_timer.mark("paired_lifts")
    log_memory("transition lifts_ready", model.load_device)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    lift_mask = m_full.amax(dim=1, keepdim=True) if m_full is not None else None
    z0_high = artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=lift_mask)
    if m_full is not None:
        # restore the keep-region with the true original instead of any lifted estimate
        m_cast = m_full.to(device=z0_high.device, dtype=z0_high.dtype)
        z0_high = z0_high * m_cast + video_anchor * (1.0 - m_cast)
    if os.environ.get("SELFLIFT_DEBUG", "0") == "1":
        _debug_dump(vae, {
            "z0_low": z0_low_vae,
            "z_lat": z_lat_vae,
            "z_pix": z_pix_vae,
            "z0_high": latent_format.process_out(z0_high),
        })
    z0_high = z0_high.to(device)
    del z0_low_vae, z_lat_vae, z_pix_vae, z_lat, z_pix
    transition_timer.mark("correction_and_debug")
    log_memory("transition correction_ready", model.load_device)

    # Re-noise the corrected video/image endpoint at the sigma of the reused model
    # evaluation, then complete that Euler interval without another denoiser call.
    video_noise = comfy.sample.prepare_noise(z0_high, (seed + 1) % (1 << 64),
                                             latent_image.get("batch_index", None)).to(z0_high)
    video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
    next_streams = [_euler_step(video_state, z0_high, sigma_k, sigma_next)] + auxiliary_next
    del z0_high, video_noise, video_state, auxiliary_next
    if high_mask is not None:
        # The high stage must start at the lifted Euler state while its inpaint
        # anchor remains the original clean AV latent (including locked audio).
        resume_latent = _pack([video_anchor] + audio_streams, nested)
        clean_streams, _ = _streams(model.model.process_latent_in(resume_latent))
        resume_sigma = sigma_next * getattr(model_sampling, "noise_scale", 1.0)
        resume_noise = _pack([
            (state - (1.0 - sigma_next.to(state)) * clean.to(state)) / resume_sigma.to(state)
            for state, clean in zip(next_streams, clean_streams)
        ], nested)
        del clean_streams
    else:
        resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, s) for s in next_streams]
        resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
        resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)
        del resume_streams
    del next_streams
    transition_timer.mark("renoise")
    transition_timer.finish()
    log_memory("transition end / high_resolution start", model.load_device)

    high_evaluations = 0

    def callback_high(step, x0, x, total):
        nonlocal high_evaluations
        step = high_evaluations
        high_evaluations += 1
        if high_evaluations > total_steps - transition_step:
            raise RuntimeError("SelfLift: too many high-resolution callbacks for the Euler schedule")
        result = callback(step + transition_step, x0, x, total_steps)
        high_timer.mark(f"step {step + 1}/{total_steps - transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    high_timer = _StageTimer("high_resolution", model.load_device, (H, W), resolution_scale)
    if highres_tiling:
        logging.info("[SelfLift plan] automatic high-resolution tiling enabled; preparation selects the tile count")
    out = comfy.samplers.sample(high_model, resume_noise, positive, negative, cfg, model.load_device,
                                sampler, sigmas[transition_step:], high_model.model_options,
                                latent_image=resume_latent, denoise_mask=high_mask, callback=callback_high,
                                disable_pbar=disable_pbar, seed=seed)
    del resume_latent, resume_noise
    if high_evaluations != total_steps - transition_step:
        raise RuntimeError(f"SelfLift: expected {total_steps - transition_step} high-resolution callbacks, received {high_evaluations}; check sampler wrappers")

    result = latent_image.copy()
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
    high_timer.finish()
    log_memory("high_resolution end", model.load_device)
    return result


class AD_selfliftH3_Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "分辨率切换时用于画面重编码的 VAE。"}),
            "latent_image": ("LATENT", {"tooltip": "确定目标尺寸和时长的 H3 音视频 latent；支持遮罩。"}),
            "sampler": ("SAMPLER", {"tooltip": "仅支持标准 Euler 采样器。"}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "低分辨率阶段的采样步数。"}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "低分辨率阶段的空间缩放比例。"}),
            "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "参与像素修正的区域比例；0 为关闭。"}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "像素修正强度下限。"}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "像素修正强度上限。"}),
            "upscaler_model": _upscaler_input(),
        }, "optional": {
            "model_hires": ("MODEL", {"tooltip": "可选的高分辨率阶段模型，需与主模型架构一致。"}),
            "highres_tiling": ("BOOLEAN", {"default": False, "label_on": "高分辨率分块：开启", "label_off": "高分辨率分块：关闭", "tooltip": "高分辨率阶段自动分块，降低显存占用。"}),
            "upscaler_unload": ("BOOLEAN", {"default": True, "label_on": "放大后卸载：开启", "label_off": "放大后卸载：关闭", "tooltip": "放大后卸载模型，节省显存。"}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/AD"


    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, model_hires=None, highres_tiling=False,
               upscaler_unload=True):
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError(
                "SelfLift H3: rho=0 with upscaler_model=none disables both SelfLift-zero correction "
                "and external latent upscaling; choose rho>0 or select an external H3 upscaler."
            )
        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning("SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; select upscaler_model=none to test the paper's SelfLift-zero direct route")
            lifter = lambda z, hw: learned_latent_lift(z, hw, upscaler_model, force_unload=upscaler_unload)
        return (progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                   latent_lifter=lifter, highres_tiling=highres_tiling, model_hires=model_hires),)
