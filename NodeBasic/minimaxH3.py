import contextlib
import copy
import collections.abc
import inspect
import logging
import math

import torch
import torch.nn.functional as F

import comfy.nested_tensor
import comfy.model_management
import comfy.quant_ops
import comfy.samplers
import comfy.utils
import node_helpers
import latent_preview
from comfy.k_diffusion.sampling import to_d
from comfy.ldm.minimax.model import PackedLayout
from comfy.utils import model_trange
from comfy.ldm.modules.attention import AttentionTensorContainer, optimized_attention

try:
    import torchaudio
except ImportError:
    torchaudio = None

_LOG = logging.getLogger("h3_motion_context")

MC_KEY = "motion_context_index"
MC_AUDIO_KEY = "motion_context_audio_end_frame"
MC_GENERATED_KEY = "motion_context_generated"

FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
VIDEO_RUN_GRID = (124, 107, 90, 73, 56, 39, 22, 5, 1)
FPS = 24
FRAME_RESCALE = 5.0 / 3.0
AUDIO_HZ = 40.0

ENCODE_MODE = "video"
ANCHOR_MODE = "head"
AUDIO_MODE = "timeline"
CROP = "disabled"


def _ad_h3_make_packed_layout(text_len, latent_t, latent_h, latent_w, audio_t,
                              keyframes=None, refs=None, frame_count=None):
    """Build a tile-local H3 layout across ComfyUI PackedLayout versions."""
    kwargs = {"keyframes": keyframes, "refs": refs}
    try:
        parameters = inspect.signature(PackedLayout.__init__).parameters
        supports_extra = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        if "frame_count" in parameters or supports_extra:
            kwargs["frame_count"] = frame_count
    except (TypeError, ValueError):
        pass
    return PackedLayout(text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)


def _ad_h3_extract_av(samples):
    if getattr(samples, "is_nested", False):
        streams = tuple(samples.unbind())
        if not streams or not isinstance(streams[0], torch.Tensor):
            raise TypeError("H3 tiled Euler: AV latent does not contain a video tensor")
        return streams[0], streams[1] if len(streams) > 1 else None, "nested"
    if isinstance(samples, torch.Tensor):
        return samples, None, "tensor"
    raise TypeError(
        "H3 tiled Euler requires a video tensor or NestedTensor(video, audio), "
        f"got {type(samples).__name__}"
    )


def _ad_h3_rebuild_av(video, audio, layout):
    if layout == "nested" and audio is not None:
        return comfy.nested_tensor.NestedTensor((video, audio))
    return video


def _ad_h3_aligned_tile_regions(total, tile_count, halo, alignment=2):
    """Partition an axis into aligned cores and expand each core by a halo."""
    total = max(1, int(total))
    alignment = max(1, int(alignment))
    max_tiles = max(1, total // alignment)
    tile_count = max(1, min(int(tile_count), max_tiles))
    halo = max(0, int(round(int(halo) / alignment)) * alignment)
    if tile_count == 1:
        return [(0, total, 0, total)]

    boundaries = [0]
    for index in range(1, tile_count):
        raw = float(total) * index / tile_count
        boundary = int(round(raw / alignment)) * alignment
        minimum = boundaries[-1] + alignment
        maximum = total - (tile_count - index) * alignment
        boundaries.append(min(max(boundary, minimum), maximum))
    boundaries.append(total)

    regions = []
    for index in range(tile_count):
        core_start = boundaries[index]
        core_end = boundaries[index + 1]
        start = max(0, core_start - halo)
        end = min(total, core_end + halo)
        regions.append((start, end, core_start, core_end))
    return regions


def _ad_h3_core_halo_window(start, end, core_start, core_end, device):
    """Keep the core at full weight and fade unreliable halo edges to zero."""
    length = int(end) - int(start)
    window = torch.ones(length, dtype=torch.float32, device=device)
    left = max(0, int(core_start) - int(start))
    right = max(0, int(end) - int(core_end))
    if left > 0:
        progress = torch.linspace(0, 1, left + 1, device=device, dtype=torch.float32)[:-1]
        window[:left] = 0.5 - 0.5 * torch.cos(progress * math.pi)
    if right > 0:
        progress = torch.linspace(0, 1, right + 1, device=device, dtype=torch.float32)[1:]
        window[-right:] = 0.5 + 0.5 * torch.cos(progress * math.pi)
    return window


def _ad_h3_crop_spatial(tensor, axis, start, end):
    if tensor is None:
        return None
    if axis == "H":
        return tensor[:, :, :, start:end, :].contiguous()
    return tensor[:, :, :, :, start:end].contiguous()


def _ad_h3_noise_masks(latent):
    masks = latent.get("noise_mask")
    if masks is None:
        return None
    if getattr(masks, "is_nested", False):
        return tuple(masks.unbind())
    if isinstance(masks, torch.Tensor):
        return (masks,)
    raise TypeError("H3 tiled Euler: unsupported noise_mask layout")


def h3_sample_tiled_euler(noise, guider, sigmas, latent, tile_count=2,
                          overlap_pixels=128):
    """Run H3 video refinement with synchronized long-edge Euler tiles.

    The long edge is partitioned into even, patch-aligned core regions. Each
    core is expanded by overlap_pixels as a context halo whose prediction fades
    to zero at internal tile edges. Every Euler step evaluates all tiles,
    blends their video predictions, averages the full audio prediction, and
    only then advances the shared AV latent. The returned LATENT is the
    denoised/x0 output; input audio is intentionally preserved.
    """
    if not isinstance(latent, collections.abc.Mapping) or "samples" not in latent:
        raise TypeError("H3 tiled Euler requires a LATENT mapping with samples")
    video, audio, av_layout = _ad_h3_extract_av(latent["samples"])
    if video.ndim != 5:
        raise ValueError(
            "H3 tiled Euler video latent must be [B,C,T,H,W], "
            f"got shape={tuple(video.shape)}"
        )

    device = comfy.model_management.get_torch_device()
    video = video.to(device=device)
    audio = audio.to(device=device) if audio is not None else None
    height, width = int(video.shape[-2]), int(video.shape[-1])
    axis = "H" if height >= width else "W"
    axis_total = height if axis == "H" else width
    halo = max(0, int(round(int(overlap_pixels) / 16.0)))
    regions = _ad_h3_aligned_tile_regions(
        axis_total, tile_count, halo, alignment=2
    )
    if len(regions) <= 1 or any(
        start == 0 and end == axis_total
        for start, end, _core_start, _core_end in regions
    ):
        regions = [(0, axis_total, 0, axis_total)]
        _LOG.info(
            "H3 tiled Euler bypassed spatial splitting because the requested "
            "tiles cover the full long edge"
        )

    weighted_regions = []
    for start, end, core_start, core_end in regions:
        window_1d = _ad_h3_core_halo_window(
            start, end, core_start, core_end, device
        )
        window = (
            window_1d.view(1, 1, 1, -1, 1)
            if axis == "H" else window_1d.view(1, 1, 1, 1, -1)
        )
        weighted_regions.append((start, end, core_start, core_end, window))
    regions = weighted_regions

    weight_shape = (
        (1, 1, 1, axis_total, 1)
        if axis == "H" else (1, 1, 1, 1, axis_total)
    )
    weights = torch.zeros(weight_shape, dtype=torch.float32, device=device)
    for start, end, _core_start, _core_end, window in regions:
        if axis == "H":
            weights[:, :, :, start:end, :] += window
        else:
            weights[:, :, :, :, start:end] += window
    weights.clamp_(min=1e-8)

    full_samples = _ad_h3_rebuild_av(video, audio, av_layout)
    full_latent = dict(latent)
    full_latent["samples"] = full_samples
    full_noise = noise.generate_noise(full_latent)
    full_shapes = [tuple(video.shape)]
    if audio is not None:
        full_shapes.append(tuple(audio.shape))
    mask_streams = _ad_h3_noise_masks(full_latent)
    full_mask = None
    if mask_streams is not None:
        full_mask = (
            comfy.nested_tensor.NestedTensor(mask_streams)
            if len(mask_streams) > 1 else mask_streams[0]
        )

    x0_output = {}
    callback = latent_preview.prepare_callback(
        guider.model_patcher, sigmas.shape[-1] - 1, x0_output
    )
    original_extra = {}
    original_inpaint = {}

    @torch.no_grad()
    def synchronized_euler(model, x, step_sigmas, extra_args=None, callback=None,
                           disable=None, s_churn=0.0, s_tmin=0.0,
                           s_tmax=float("inf"), s_noise=1.0, **_unused):
        extra_args = {} if extra_args is None else extra_args
        s_in = x.new_ones([x.shape[0]])
        prepared_model = model.inner_model.inner_model
        saved_model_shapes = getattr(prepared_model, "latent_shapes", None)
        saved_conds = {}
        payload_conds = []
        packed_mask = extra_args.get("denoise_mask")
        current_masks = (
            comfy.utils.unpack_latents(packed_mask, full_shapes)
            if packed_mask is not None else None
        )
        source_latents = (
            comfy.utils.unpack_latents(model.latent_image, full_shapes)
            if getattr(model, "latent_image", None) is not None else None
        )
        source_noise = (
            comfy.utils.unpack_latents(model.noise, full_shapes)
            if getattr(model, "noise", None) is not None else None
        )

        for cond_group in getattr(model.inner_model, "conds", {}).values():
            if cond_group is None:
                continue
            for cond in cond_group:
                model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
                payload_cond = model_conds.get("minimax_payload")
                payload = getattr(payload_cond, "cond", None)
                if isinstance(payload, dict):
                    payload_conds.append((payload_cond, payload))

        def set_tile_shapes(tile_shapes):
            prepared_model.latent_shapes = tile_shapes
            for cond_group in getattr(model.inner_model, "conds", {}).values():
                if cond_group is None:
                    continue
                for cond in cond_group:
                    model_conds = cond.get("model_conds", {}) if isinstance(cond, dict) else {}
                    shape_cond = model_conds.get("latent_shapes")
                    if shape_cond is not None and hasattr(shape_cond, "cond"):
                        saved_conds.setdefault(id(shape_cond), (shape_cond, shape_cond.cond))
                        shape_cond.cond = tile_shapes

        def crop_keyframe(value, start, end):
            if not isinstance(value, torch.Tensor) or value.ndim != 5:
                return value
            cropped = _ad_h3_crop_spatial(value, axis, start, end)
            pad_h = (-cropped.shape[-2]) % 2
            pad_w = (-cropped.shape[-1]) % 2
            if pad_h or pad_w:
                cropped = F.pad(cropped, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
            return cropped

        def install_payloads(tile_shapes, start, end):
            video_shape = tile_shapes[0]
            tile_h = (int(video_shape[3]) + 1) // 2 * 2
            tile_w = (int(video_shape[4]) + 1) // 2 * 2
            audio_t = int(tile_shapes[1][-1]) if len(tile_shapes) > 1 else 0
            restorations = []
            for payload_cond, original_payload in payload_conds:
                payload = dict(original_payload)
                keyframes = []
                for item in list(payload.get("keyframes") or []):
                    copied = dict(item)
                    copied["latent"] = crop_keyframe(item.get("latent"), start, end)
                    keyframes.append(copied)
                refs = list(payload.get("refs") or [])
                old_layout = payload.get("layout")
                text_tags = payload.get("text_token_tags")
                text_len = 0
                if old_layout is not None and getattr(old_layout, "segments", None):
                    text_len = int(old_layout.segments[0][1])
                elif text_tags is not None:
                    text_len = int(text_tags.shape[-1])
                if text_len <= 0:
                    raise RuntimeError("H3 tiled Euler could not determine text token length")
                payload["keyframes"] = keyframes or payload.get("keyframes")
                payload["layout"] = _ad_h3_make_packed_layout(
                    text_len, int(video_shape[2]), tile_h, tile_w, audio_t,
                    keyframes=keyframes or None, refs=refs or None,
                    frame_count=payload.get("frame_count"),
                )
                ref_latents = [
                    ref.get("latent") for ref in refs
                    if ref.get("latent") is not None
                    and ref.get("kind") != "t8_keyframe_latent"
                ]
                ordered = []
                keyframe_index = reference_index = 0
                for _begin, _finish, kind in payload["layout"].segments:
                    if kind == "cond" and keyframe_index < len(keyframes):
                        ordered.append(keyframes[keyframe_index]["latent"])
                        keyframe_index += 1
                    elif kind == "ref_img" and reference_index < len(ref_latents):
                        ordered.append(ref_latents[reference_index])
                        reference_index += 1
                payload["cond_video_latents"] = ordered
                restorations.append((payload_cond, payload_cond.cond))
                payload_cond.cond = payload
            return restorations

        try:
            for step_index in model_trange(len(step_sigmas) - 1, disable=disable):
                gamma = (
                    min(s_churn / (len(step_sigmas) - 1), math.sqrt(2.0) - 1.0)
                    if s_churn > 0 and s_tmin <= step_sigmas[step_index] <= s_tmax
                    else 0.0
                )
                sigma_hat = step_sigmas[step_index] * (gamma + 1.0)
                if gamma > 0:
                    x = x + torch.randn_like(x) * s_noise * (
                        sigma_hat ** 2 - step_sigmas[step_index] ** 2
                    ) ** 0.5
                streams = comfy.utils.unpack_latents(x, full_shapes)
                video_x = streams[0]
                audio_x = streams[1] if len(streams) > 1 else None
                video_denoised = torch.zeros_like(video_x, dtype=torch.float32)
                audio_denoised = (
                    torch.zeros_like(audio_x, dtype=torch.float32)
                    if audio_x is not None else None
                )

                for start, end, _core_start, _core_end, window in regions:
                    video_tile = _ad_h3_crop_spatial(video_x, axis, start, end)
                    tile_streams = [video_tile] + ([audio_x] if audio_x is not None else [])
                    tile_x, tile_shapes = comfy.utils.pack_latents(tile_streams)
                    tile_mask = None
                    if current_masks is not None:
                        parts = [_ad_h3_crop_spatial(current_masks[0], axis, start, end)]
                        if len(current_masks) > 1:
                            parts.append(current_masks[1])
                        tile_mask, _ = comfy.utils.pack_latents(parts)
                    tile_latent_image = None
                    if source_latents is not None:
                        parts = [_ad_h3_crop_spatial(source_latents[0], axis, start, end)]
                        if len(source_latents) > 1:
                            parts.append(source_latents[1])
                        tile_latent_image, _ = comfy.utils.pack_latents(parts)
                    tile_noise = None
                    if source_noise is not None:
                        parts = [_ad_h3_crop_spatial(source_noise[0], axis, start, end)]
                        if len(source_noise) > 1:
                            parts.append(source_noise[1])
                        tile_noise, _ = comfy.utils.pack_latents(parts)

                    set_tile_shapes(tile_shapes)
                    restorations = install_payloads(tile_shapes, start, end)
                    saved_latent_image = getattr(model, "latent_image", None)
                    saved_noise = getattr(model, "noise", None)
                    tile_args = dict(extra_args)
                    tile_args["denoise_mask"] = tile_mask
                    if tile_latent_image is not None:
                        model.latent_image = tile_latent_image
                    if tile_noise is not None:
                        model.noise = tile_noise
                    try:
                        prediction = model(tile_x, sigma_hat * s_in, **tile_args)
                    finally:
                        model.latent_image = saved_latent_image
                        model.noise = saved_noise
                        for payload_cond, original in restorations:
                            payload_cond.cond = original
                    predictions = comfy.utils.unpack_latents(prediction, tile_shapes)
                    if axis == "H":
                        video_denoised[:, :, :, start:end, :] += predictions[0].float() * window
                    else:
                        video_denoised[:, :, :, :, start:end] += predictions[0].float() * window
                    if audio_denoised is not None:
                        audio_denoised += predictions[1].float()

                video_denoised /= weights
                merged = [video_denoised.to(dtype=video_x.dtype)]
                if audio_denoised is not None:
                    audio_denoised /= float(len(regions))
                    merged.append(audio_denoised.to(dtype=audio_x.dtype))
                denoised, _ = comfy.utils.pack_latents(merged)
                prepared_model.latent_shapes = full_shapes
                if callback is not None:
                    callback({
                        "x": x, "i": step_index, "sigma": step_sigmas[step_index],
                        "sigma_hat": sigma_hat, "denoised": denoised,
                    })
                derivative = to_d(x, sigma_hat, denoised)
                x = x + derivative * (step_sigmas[step_index + 1] - sigma_hat)
            return x
        finally:
            prepared_model.latent_shapes = saved_model_shapes
            for cond_object, original in saved_conds.values():
                cond_object.cond = original

    sampler = comfy.samplers.KSAMPLER(
        synchronized_euler,
        extra_options=original_extra,
        inpaint_options=original_inpaint,
    )
    samples = guider.sample(
        full_noise, full_samples, sampler, sigmas,
        denoise_mask=full_mask,
        callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
        seed=noise.seed,
    )
    sampled_video, _sampled_audio, _ = _ad_h3_extract_av(samples)
    denoised_video = sampled_video
    if x0_output.get("x0") is not None:
        denoised_video, _denoised_audio, _ = _ad_h3_extract_av(x0_output["x0"])
    intermediate = comfy.model_management.intermediate_device()
    denoised_video = denoised_video.to(
        device=intermediate, dtype=video.dtype
    )
    preserved_audio = audio.to(intermediate) if audio is not None else None
    result = dict(latent)
    result["samples"] = _ad_h3_rebuild_av(
        denoised_video, preserved_audio, av_layout
    )
    del weights, regions
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


_AD_H3_SAMPLING_PROFILES = [
    "None",
    "auto",
    "Speed_first | QKV 16384 | MLP 8192",
    "balanced | QKV 8192 | MLP 4096",
    "low_vram | QKV 4096 | MLP 2048",
    "maximum_safety | QKV 1024 | MLP 1024",
]

_AD_H3_VAE_TILE_PROFILES = [
    "default",
    "balanced",
    "low_vram",
    "maximum_safety",
]


def _ad_h3_sampling_profile_input():
    return (_AD_H3_SAMPLING_PROFILES, {
        "default": "auto",
        "tooltip": "模型内部QKV/MLP 分块：token分块+层内算子分块",
    })


def _ad_h3_sampling_policy(sampling_profile):
    profile = str(sampling_profile or "None").strip()
    if profile.lower() == "none":
        return None
    return {"sampling_profile": profile}


def _ad_h3_vae_tile_input():
    options = {
        "default": "default",
        "tooltip": (
            "控制 MiniMax H3 VAE 空间分块。default 保持官方 256/64；"
            "balanced 使用 224/64；low_vram 使用 192/64；"
            "maximum_safety 使用 128/64。官方时间分块保持不变。"
        ),
    }
    options["tooltip"] = "只解决VAE 编码、解码阶段的爆显存，不解决主要采样显存"
    return (_AD_H3_VAE_TILE_PROFILES, options)


@contextlib.contextmanager
def _ad_h3_vae_tile_scope(vae, profile="default"):
    name = str(profile or "default").strip().lower()
    settings = {
        "official": (256, 64),
        "balanced": (224, 64),
        "low_vram": (192, 64),
        "maximum_safety": (128, 64),
    }
    if name == "default":
        yield
        return
    if name not in settings:
        raise ValueError(f"Unknown MiniMax H3 VAE_TILE profile: {profile}")

    first_stage = getattr(vae, "first_stage_model", None)
    required = ("tiling", "tile_size", "tile_overlap_min", "clip_length")
    if first_stage is None or not all(hasattr(first_stage, key) for key in required):
        raise RuntimeError(
            "VAE_TILE can only be used with the native MiniMax H3 video VAE"
        )

    tile_size, overlap = settings[name]
    previous = {key: getattr(first_stage, key) for key in required[:-1]}
    first_stage.tiling = True
    first_stage.tile_size = int(tile_size)
    first_stage.tile_overlap_min = int(overlap)
    _LOG.info(
        "AD H3 VAE tile: profile=%s, spatial=%d, overlap=%d, temporal_clip=%d (official)",
        name, tile_size, overlap, int(first_stage.clip_length),
    )
    try:
        yield
    finally:
        for key, value in previous.items():
            setattr(first_stage, key, value)


def _ad_h3_latent_tokens(latent):
    if not isinstance(latent, collections.abc.Mapping):
        return 0
    samples = latent.get("samples")
    streams = samples.unbind() if getattr(samples, "is_nested", False) else (samples,)
    total = 0
    for stream in streams:
        if not isinstance(stream, torch.Tensor) or stream.ndim < 2:
            continue
        total += math.prod(stream.shape[:1] + stream.shape[2:])
    return int(total)


def _ad_h3_resolve_sampling_policy(policy, latent):
    profile_value = str(policy.get("sampling_profile", "")).strip()
    presets = {
        "Speed_first": (16384, 8192),
        "balanced": (8192, 4096),
        "low_vram": (4096, 2048),
        "maximum_safety": (1024, 1024),
    }
    tokens = _ad_h3_latent_tokens(latent)
    for profile, chunks in presets.items():
        if profile_value.startswith(profile):
            return chunks[0], chunks[1], tokens

    if tokens >= 200000:
        qkv_chunk_tokens = 2048
    elif tokens >= 90000:
        qkv_chunk_tokens = 4096
    else:
        qkv_chunk_tokens = 8192
    if tokens >= 90000:
        mlp_chunk_tokens = 2048
    elif tokens >= 40000:
        mlp_chunk_tokens = 4096
    else:
        mlp_chunk_tokens = 8192
    return qkv_chunk_tokens, mlp_chunk_tokens, tokens


class _ADH3ProjectionMLPChunkPatch:
    """Chunk H3 linear projections while preserving full global attention."""

    def __init__(self, index, state, qkv_chunk_tokens, mlp_chunk_tokens):
        self.index = int(index)
        self.state = state
        self.qkv_chunk_tokens = max(256, int(qkv_chunk_tokens))
        self.mlp_chunk_tokens = max(256, int(mlp_chunk_tokens))

    @staticmethod
    def _extract_block(original_block):
        closure = getattr(original_block, "__closure__", None) or ()
        for cell in closure:
            try:
                candidate = cell.cell_contents
            except Exception:
                continue
            required = ("adaln_proj", "norm1", "attn", "norm2", "mlp")
            if all(hasattr(candidate, name) for name in required):
                return candidate
        return None

    @staticmethod
    def _mod_scale_shift(value, shift, scale, segments, offset=0):
        end = offset + int(value.shape[0])
        for start, stop, row in segments:
            local_start = max(int(start), offset) - offset
            local_stop = min(int(stop), end) - offset
            if local_start < local_stop:
                value[local_start:local_stop].mul_(
                    1.0 + scale[row].to(value.dtype)
                ).add_(shift[row].to(value.dtype))
        return value

    @staticmethod
    def _mod_gate_residual(value, gate, other, segments, offset=0):
        end = offset + int(value.shape[0])
        for start, stop, row in segments:
            local_start = max(int(start), offset) - offset
            local_stop = min(int(stop), end) - offset
            if local_start < local_stop:
                value[local_start:local_stop].addcmul_(
                    other[local_start:local_stop], gate[row].to(value.dtype)
                )
        return value

    @staticmethod
    def _chunked_linear(layer, value, chunk_tokens):
        token_count = int(value.shape[0])
        chunk_tokens = min(max(1, int(chunk_tokens)), token_count)
        first_stop = min(chunk_tokens, token_count)
        first = layer(value[:first_stop])
        output = first.new_empty((token_count, *first.shape[1:]))
        output[:first_stop].copy_(first)
        del first
        for start in range(first_stop, token_count, chunk_tokens):
            stop = min(token_count, start + chunk_tokens)
            part = layer(value[start:stop])
            output[start:stop].copy_(part)
            del part
        return output

    def _attention(self, attention, value, rope_freqs, transformer_options):
        token_count = int(value.shape[0])
        inner = int(attention.heads * attention.head_dim)
        chunk_tokens = min(self.qkv_chunk_tokens, max(1, token_count))

        if token_count <= chunk_tokens:
            q, k, v = attention.qkv_proj(value).split(inner, dim=-1)
            v = v.view(token_count, attention.heads, attention.head_dim).clone()
        else:
            first_stop = min(chunk_tokens, token_count)
            first = attention.qkv_proj(value[:first_stop])
            first_q, first_k, first_v = first.split(inner, dim=-1)
            q = first_q.new_empty((token_count, inner))
            k = first_k.new_empty((token_count, inner))
            v = first_v.new_empty((token_count, attention.heads, attention.head_dim))
            q[:first_stop].copy_(first_q)
            k[:first_stop].copy_(first_k)
            v[:first_stop].copy_(first_v.view(first_stop, attention.heads, attention.head_dim))
            del first, first_q, first_k, first_v
            for start in range(first_stop, token_count, chunk_tokens):
                stop = min(token_count, start + chunk_tokens)
                projected = attention.qkv_proj(value[start:stop])
                q_part, k_part, v_part = projected.split(inner, dim=-1)
                q[start:stop].copy_(q_part)
                k[start:stop].copy_(k_part)
                v[start:stop].copy_(v_part.view(stop - start, attention.heads, attention.head_dim))
                del projected, q_part, k_part, v_part

        if rope_freqs is not None:
            q = q.view(1, token_count, attention.heads, attention.head_dim)
            k = k.view(1, token_count, attention.heads, attention.head_dim)
            qw = comfy.model_management.cast_to(attention.q_norm.weight, device=value.device)
            kw = comfy.model_management.cast_to(attention.k_norm.weight, device=value.device)
            rot = rope_freqs.shape[-3] * 2
            if comfy.model_management.in_training:
                q, k = comfy.quant_ops.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps, rot_dim=rot
                )
            else:
                comfy.quant_ops.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attention.q_norm.eps, rot_dim=rot
                )
            q = q[0]
            k = k[0]
        else:
            q = attention.q_norm(q.view(token_count, attention.heads, attention.head_dim))
            k = attention.k_norm(k.view(token_count, attention.heads, attention.head_dim))

        q = AttentionTensorContainer(q.transpose(0, 1).unsqueeze(0))
        k = AttentionTensorContainer(k.transpose(0, 1).unsqueeze(0))
        v = AttentionTensorContainer(v.transpose(0, 1).unsqueeze(0))
        output = optimized_attention(
            q, k, v, attention.heads, mask=None, skip_reshape=True,
            transformer_options=transformer_options,
        ).squeeze(0)
        del q, k, v
        return self._chunked_linear(attention.out_proj, output, chunk_tokens)

    def __call__(self, args, extra_options):
        original_block = extra_options["original_block"]
        block = self._extract_block(original_block)
        if block is None:
            if not self.state.get("closure_fallback_logged"):
                logging.getLogger("AD_H3_sampling_policy").warning(
                    "AD H3 projection/MLP chunking could not locate the DiT block; using the stock block"
                )
                self.state["closure_fallback_logged"] = True
            return original_block(args)

        x = args["img"]
        t_emb = args["t_emb"]
        segments = args["mod_segments"]
        rope_freqs = args["rope_freqs"]
        transformer_options = args["transformer_options"]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)

        h = self._mod_scale_shift(block.norm1(x), shift_msa, scale_msa, segments)
        token_count = int(h.shape[0])
        qkv_chunk_tokens = min(self.qkv_chunk_tokens, max(1, token_count))
        if token_count > qkv_chunk_tokens and not self.state.get("projection_announced"):
            logging.getLogger("AD_H3_sampling_policy").info(
                "AD H3 projection chunking enabled: tokens=%d, QKV/out=%d, attention=global",
                token_count, qkv_chunk_tokens,
            )
            self.state["projection_announced"] = True
        attention = self._attention(block.attn, h, rope_freqs, transformer_options)
        x = self._mod_gate_residual(x, gate_msa, attention, segments)
        del h, attention

        token_count = int(x.shape[0])
        chunk_tokens = min(self.mlp_chunk_tokens, max(1, token_count))
        if token_count > chunk_tokens and not self.state.get("mlp_announced"):
            logging.getLogger("AD_H3_sampling_policy").info(
                "AD H3 MLP chunking enabled: tokens=%d, chunk=%d, attention=global",
                token_count, chunk_tokens,
            )
            self.state["mlp_announced"] = True

        for start in range(0, token_count, chunk_tokens):
            stop = min(token_count, start + chunk_tokens)
            x_chunk = x[start:stop]
            h_chunk = self._mod_scale_shift(
                block.norm2(x_chunk), shift_mlp, scale_mlp, segments, offset=start
            )
            mlp_chunk = block.mlp(h_chunk)
            self._mod_gate_residual(
                x_chunk, gate_mlp, mlp_chunk, segments, offset=start
            )
            del h_chunk, mlp_chunk
        return {"img": x}


def _ad_clone_model_options(model_options):
    try:
        import comfy.model_patcher
        return comfy.model_patcher.create_model_options_clone(model_options)
    except Exception:
        cloned = dict(model_options)
        transformer_options = dict(model_options.get("transformer_options", {}))
        cloned["transformer_options"] = transformer_options
        patches_replace = dict(transformer_options.get("patches_replace", {}))
        transformer_options["patches_replace"] = patches_replace
        patches_replace["dit"] = dict(patches_replace.get("dit", {}))
        return cloned


def _ad_h3_wrap_guider(guider, policy, latent):
    qkv_chunk_tokens, mlp_chunk_tokens, tokens = _ad_h3_resolve_sampling_policy(policy, latent)
    logging.getLogger("AD_H3_sampling_policy").info(
        "AD H3 sampling policy: tokens=%d, QKV/out=%d, MLP=%d, "
        "attention=global, implementation=apt-local",
        tokens, qkv_chunk_tokens, mlp_chunk_tokens,
    )

    wrapped = copy.copy(guider)
    wrapped.model_options = _ad_clone_model_options(
        getattr(guider, "model_options", {}) or {}
    )
    transformer_options = wrapped.model_options.setdefault("transformer_options", {})
    patches_replace = transformer_options.setdefault("patches_replace", {})
    dit = patches_replace.setdefault("dit", {})
    state = {
        "qkv_chunk_tokens": int(qkv_chunk_tokens),
        "mlp_chunk_tokens": int(mlp_chunk_tokens),
        "patched_blocks": [],
        "skipped_blocks": [],
    }
    for index in range(128):
        key = ("double_block", index)
        if key in dit:
            state["skipped_blocks"].append(index)
            continue
        dit[key] = _ADH3ProjectionMLPChunkPatch(
            index, state, qkv_chunk_tokens, mlp_chunk_tokens
        )
        state["patched_blocks"].append(index)
    return wrapped


class AptMiniMaxH3NativeAudioLock:
    """Lock exact user audio into an H3 AV latent and denoise video only."""

    def lock_audio(self, model, av_latent, audio_vae, audio):
        samples = av_latent.get("samples")
        if samples is None or not getattr(samples, "is_nested", False):
            raise ValueError("AptMiniMaxH3NativeAudioLock requires a joint MiniMax H3 AV latent")

        video_latent, target_audio_template = samples.unbind()[:2]
        waveform = audio["waveform"][:1]
        sample_rate = int(audio["sample_rate"])
        vae_rate = int(getattr(audio_vae, "audio_sample_rate", 32000))
        if sample_rate != vae_rate:
            if torchaudio is None:
                raise RuntimeError("AptMiniMaxH3NativeAudioLock needs torchaudio to resample audio")
            waveform = torchaudio.functional.resample(waveform, sample_rate, vae_rate)

        exact_audio_latent = audio_vae.encode(waveform.movedim(1, -1))
        target_t = target_audio_template.shape[-1]
        if exact_audio_latent.shape[-1] > target_t:
            exact_audio_latent = exact_audio_latent[..., :target_t]
        elif exact_audio_latent.shape[-1] < target_t:
            exact_audio_latent = F.pad(exact_audio_latent, (0, target_t - exact_audio_latent.shape[-1]))

        locked = dict(av_latent)
        locked["samples"] = comfy.nested_tensor.NestedTensor((video_latent, exact_audio_latent))
        locked["noise_mask"] = comfy.nested_tensor.NestedTensor(
            (torch.ones_like(video_latent), torch.zeros_like(exact_audio_latent))
        )

        patched_model = model.clone()
        transformer_options = patched_model.model_options["transformer_options"] = (
            patched_model.model_options.get("transformer_options", {}).copy()
        )
        transformer_options["minimax_h3_lock_audio_clean"] = True
        return patched_model, locked, audio


def _ensure_layout_patch():
    if _layout_patch_applied():
        return
    if not _apply_layout_patch():
        raise RuntimeError(
            "h3_motion_context: the layout patch could not be applied, so "
            "interior anchors would be rejected by ComfyUI. The reason was "
            "logged just above this error.")


def _ensure_payload_patch():
    if _payload_patch_applied():
        return
    if not _apply_payload_patch():
        raise RuntimeError(
            "h3_motion_context: the payload patch could not be applied. "
            "Without it the audio ref would overwrite the pinned video "
            "latents and the motion context would be lost. The reason was "
            "logged just above this error.")


def h3_keyframe_anchor(position):
    """Return a first/last anchor compatible with native and patched layouts."""
    _ensure_layout_patch()
    position = int(position)
    if _layout_native:
        return {"resolved_frame_index": position}
    return {"resolved_frame_index": 0, MC_KEY: position}


def _pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _step_offsets(latent_t):
    out, acc = [], 0
    for k in range(latent_t):
        out.append(acc)
        acc += FRAME_PER_TOKEN[k % 5]
    return out


def _resize(image, width, height, crop):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _encode_tail_audio(audio_vae, audio, seconds):
    waveform = audio["waveform"]
    sr = int(audio["sample_rate"])
    vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
    if sr != vae_sr:
        if torchaudio is None:
            raise RuntimeError(
                "h3_motion_context: context_audio is %d Hz but the VAE wants %d Hz "
                "and torchaudio is not available to resample." % (sr, vae_sr))
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    want = int(round(seconds * vae_sr))
    have = int(waveform.shape[-1])
    if have < want:
        _LOG.warning("h3_motion_context: context_audio is %.3fs, shorter than the "
                     "%.3fs of pinned video. Pinning what there is.",
                     have / vae_sr, seconds)
    else:
        waveform = waveform[..., have - want:]
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, int(z.shape[-1])


def _streams_from_latent(latent):
    samples = latent["samples"]
    if hasattr(samples, "unbind"):
        parts = list(samples.unbind())
    elif isinstance(samples, (tuple, list)):
        parts = list(samples)
    else:
        raise ValueError(
            "h3_motion_context: expected a MiniMax H3 AV latent (a nested "
            "video/audio pair), got %r" % type(samples))
    if not parts:
        raise ValueError("h3_motion_context: AV latent contains no streams")
    return parts


def _video_from_latent(latent):
    video = _streams_from_latent(latent)[0]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if video.ndim != 5:
        raise ValueError("h3_motion_context: expected video latent [B,C,T,H,W], "
                         "got shape %s" % (tuple(video.shape),))
    return video


def _steps_for_frames(n):
    k, covered = 0, 0
    while covered < n:
        covered += FRAME_PER_TOKEN[k % 5]
        k += 1
    return k if covered == n else None


def h3_export_video_tail(vae, latent, frames, end_frame):
    """Reuse only an unmodified export tail ending on the sampled latent grid."""
    if latent is not None:
        video = _video_from_latent(latent)
        total = int(video.shape[2])
        steps = _steps_for_frames(int(frames.shape[0]))
        if (steps is not None and steps <= total and (total - steps) % 5 == 0
                and int(end_frame) == _pixel_frames(total)
                and tuple(frames.shape[1:3]) == (int(video.shape[3]) * 16, int(video.shape[4]) * 16)):
            _LOG.info("h3_motion_context: reusing %d export-tail latent steps without VAE re-encoding", steps)
            return video[:1, :, total - steps:].clone()
    return vae.encode(frames)


def _video_tail_from_latent(latent, n):
    exported_tail = latent.get("apt_h3_export_tail_latent")
    exported_frames = int(latent.get("apt_h3_export_context_frames", 22))
    if exported_tail is not None and n == exported_frames:
        if exported_tail.ndim == 4:
            exported_tail = exported_tail.unsqueeze(0)
        steps = _steps_for_frames(n)
        if exported_tail.ndim != 5 or int(exported_tail.shape[2]) != steps:
            raise ValueError(
                "h3_motion_context: stored export tail does not match the "
                "%d-frame H3 latent grid" % n)
        blocks = [exported_tail[:1, :, k:k + 1].clone() for k in range(steps)]
        return blocks, _step_offsets(steps), n

    video = _video_from_latent(latent)
    total = int(video.shape[2])
    steps = _steps_for_frames(n)
    if steps is None:
        raise ValueError(
            "h3_motion_context: a %d frame window is not a whole number of "
            "latent steps, so it cannot be sliced from a latent. Use 5, 22, "
            "39 or 56, or unwire context_latent to encode pixels." % n)
    if steps > total:
        raise ValueError(
            "h3_motion_context: asked for %d latent steps, context_latent "
            "has %d." % (steps, total))
    start = total - steps
    if start % 5 != 0:
        raise RuntimeError(
            "h3_motion_context: the %d step tail of a %d step latent starts "
            "at cycle position %d, not 0, so its frame spans would not match "
            "the positions written for them. Clip lengths are meant to make "
            "this impossible; refusing rather than rendering a shifted join."
            % (steps, total, start % 5))
    covered = _pixel_frames(steps)
    if covered != n:
        raise RuntimeError(
            "h3_motion_context: %d steps cover %d frames, expected %d."
            % (steps, covered, n))
    blocks = [video[:1, :, start + k:start + k + 1].clone()
              for k in range(steps)]
    return blocks, _step_offsets(steps), covered


def _audio_tail_from_latent(latent, a_frames):
    exported_tail = latent.get("apt_h3_export_tail_audio_latent")
    if exported_tail is not None:
        if exported_tail.ndim == 3:
            exported_tail = exported_tail.unsqueeze(0)
        if exported_tail.ndim != 4:
            raise ValueError(
                "h3_motion_context: stored export audio tail has an invalid shape")
        return exported_tail[:1].clone(), int(exported_tail.shape[-1]), 0.0

    parts = _streams_from_latent(latent)
    if len(parts) < 2:
        raise ValueError(
            "h3_motion_context: context_latent has no audio stream. Wire the "
            "sampler output of an H3 AV graph, not a video-only latent.")
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    if audio.ndim != 4:
        raise ValueError("h3_motion_context: expected audio latent [B,C,2,T], "
                         "got shape %s" % (tuple(audio.shape),))
    total_t = int(audio.shape[-1])
    frames = _pixel_frames(int(video.shape[2]))
    overhang = total_t - FRAME_RESCALE * frames
    # H3 rounds the audio grid to the nearest step. Depending on the clip
    # length the latent may overrun or stop short by one third of a step.
    if not (-0.5 < overhang < 0.5):
        _LOG.warning(
            "h3_motion_context: context_latent audio grid is unexpected "
            "(%d steps for %d frames); assuming no overhang.", total_t, frames)
        overhang = 0.0
    rt = int(round(a_frames / float(FPS) * AUDIO_HZ))
    if rt > total_t:
        _LOG.warning("h3_motion_context: asked for %d audio steps, the latent "
                     "has %d. Pinning all of it.", rt, total_t)
        rt = total_t
    if rt < 1:
        raise ValueError("h3_motion_context: audio window is empty")
    tail = audio[:1, ..., total_t - rt:].clone()
    return tail, rt, float(overhang)


class AptMiniMaxH3MotionContext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "trim_frames": ("INT", {
                    "default": 22,
                    "min": 5,
                    "max": 56,
                    "step": 17}),
            },
            "optional": {
                "context_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "trim_frames")
    FUNCTION = "apply"
    CATEGORY = "Apt_Preset/MiniMax H3"

    def apply(self, conditioning, latent, trim_frames=22,
              context_latent=None, audio_context_length=24):
        if context_latent is None:
            return (conditioning, 0)

        context_length = int(trim_frames)
        # Video and audio use independent windows. 22 video frames gives a
        # whole H3 video-latent run; 24 audio frames is exactly one second
        # and lands exactly on the 40 Hz audio grid.
        audio_context_length = int(audio_context_length)
        vae = None
        context_frames = None
        audio_vae = None
        context_audio = None
        encode_mode, anchor_mode = ENCODE_MODE, ANCHOR_MODE
        audio_mode, crop = AUDIO_MODE, CROP
        _ensure_layout_patch()

        video = _video_from_latent(latent)
        latent_t = int(video.shape[2])
        width = int(video.shape[4]) * 16
        height = int(video.shape[3]) * 16
        frame_count = _pixel_frames(latent_t)

        if context_latent is not None:
            src_video = _video_from_latent(context_latent)
            src_w = int(src_video.shape[4]) * 16
            src_h = int(src_video.shape[3]) * 16
            if src_w != width or src_h != height:
                raise ValueError(
                    "h3_motion_context: context_latent is %dx%d but this "
                    "clip is %dx%d. A latent cannot be resized, so the "
                    "previous clip has to be regenerated at this "
                    "resolution, or the chain restarted here."
                    % (src_w, src_h, width, height))
            if int(src_video.shape[1]) != int(video.shape[1]):
                raise ValueError(
                    "h3_motion_context: context_latent has %d channels, "
                    "this clip has %d. That is not an H3 video latent from "
                    "the same model."
                    % (int(src_video.shape[1]), int(video.shape[1])))
            available = _pixel_frames(int(src_video.shape[2]))
            video_src = "latent"
        else:
            if context_frames is None:
                raise ValueError(
                    "h3_motion_context: nothing to pin. Wire context_latent "
                    "(preferred) or context_frames.")
            available = int(context_frames.shape[0])
            video_src = "pixels"

        n = min(int(context_length), available)
        if n < 1:
            raise ValueError("h3_motion_context: no frames available to pin")
        if n < context_length:
            _LOG.warning("h3_motion_context: only %d frames available, pinning %d",
                         available, n)

        if encode_mode == "video":
            run = next(g for g in VIDEO_RUN_GRID if g <= n)
            if run != n:
                _LOG.warning(
                    "h3_motion_context: %d frames is off the VAE grid; pinning "
                    "the last %d instead (usable runs: 1, 5, 22, 39, 56)", n, run)
            n = run

        if n >= frame_count:
            raise ValueError(
                "h3_motion_context: asked to pin %d frames into a %d frame clip. "
                "The pinned run must be a small fraction of the timeline."
                % (n, frame_count))

        if video_src == "latent" and _steps_for_frames(n) is None:
            raise RuntimeError(
                "h3_motion_context: a %d frame window is not a whole number "
                "of latent steps. VIDEO_RUN_GRID no longer matches the "
                "VAE; refusing rather than rendering a shifted join." % n)

        if video_src == "latent":
            blocks, offsets, covered = _video_tail_from_latent(
                context_latent, n)
            span = covered
        else:
            tail = _resize(context_frames[available - n:], width, height, crop)

        if video_src == "pixels" and encode_mode == "video":
            enc = vae.encode(tail)
            if getattr(enc, "ndim", 0) != 5:
                raise ValueError(
                    "h3_motion_context: video-mode encode returned shape %s, "
                    "expected [B,C,T,H,W]. Try encode_mode=frames."
                    % (tuple(getattr(enc, "shape", ())),))
            steps = int(enc.shape[2])
            offsets = _step_offsets(steps)
            covered = _pixel_frames(steps)
            if covered != n:
                raise RuntimeError(
                    "h3_motion_context: %d frames encoded to %d latent steps "
                    "covering %d frames; the VAE grid no longer matches "
                    "VIDEO_RUN_GRID. Upstream VAE change, refusing to run."
                    % (n, steps, covered))
            blocks = [enc[:, :, k:k + 1] for k in range(steps)]
            span = covered
        elif video_src == "pixels":
            blocks, offsets = [], []
            for i in range(n):
                blocks.append(vae.encode(tail[i:i + 1]))
                offsets.append(i)
            span = n

        if anchor_mode == "before":
            indices = [o - span for o in offsets]
        else:
            indices = list(offsets)

        keyframes = []
        for p, blk in zip(indices, blocks):
            kf = {"latent": blk, MC_GENERATED_KEY: True}
            if _layout_native:
                kf["resolved_frame_index"] = p
            else:
                kf["resolved_frame_index"] = 0
                kf[MC_KEY] = p
            keyframes.append(kf)

        ref_audio_t = 0
        audio_ref = None
        a_frames = 0
        audio_src = "off"
        if context_latent is not None or context_audio is not None:
            _ensure_payload_patch()
            a_frames = int(audio_context_length) or span
            if context_latent is not None:
                if context_audio is not None:
                    _LOG.info("h3_motion_context: both context_latent and "
                              "context_audio wired; using the latent (skips "
                              "one VAE round trip).")
                audio_latent, ref_audio_t, overhang = _audio_tail_from_latent(
                    context_latent, a_frames)
                audio_src = "latent"
            else:
                if audio_vae is None:
                    raise ValueError(
                        "h3_motion_context: context_audio supplied without "
                        "audio_vae. Wire the H3 audio VAE, or wire "
                        "context_latent instead.")
                audio_latent, ref_audio_t = _encode_tail_audio(
                    audio_vae, context_audio, a_frames / float(FPS))
                overhang = 0.0
                audio_src = "vae"
            ref = {
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
                MC_GENERATED_KEY: True,
            }
            if audio_mode == "timeline":
                if _layout_native:
                    _LOG.info("h3_motion_context: audio_mode=timeline requires "
                              "the wrapper patch; stock ref placement used "
                              "instead (audio still pinned, just on the ref path)")
                else:
                    end_frame = float(span if anchor_mode == "head" else 0)
                    end_frame += overhang / FRAME_RESCALE
                    end_coord = round(FRAME_RESCALE * end_frame)
                    end_frame = end_coord / FRAME_RESCALE
                    ref[MC_AUDIO_KEY] = end_frame
            audio_ref = ref

        # Merge with upstream keyframes instead of replacing them. In
        # particular, keep a valid last-frame target from the H3 guide while
        # dropping anchors that conflict with the pinned head.
        head_end = span if anchor_mode == "head" else 0
        out = []
        dropped = []
        for emb, extra in conditioning:
            d = extra.copy()
            prior = d.get("minimax_keyframes") or []
            prior_frame_count = d.get("minimax_frame_count")
            if (prior and prior_frame_count is not None
                    and int(prior_frame_count) != frame_count):
                raise ValueError(
                    "h3_motion_context: the conditioning carries keyframes "
                    "resolved for a %d frame clip, but the latent is %d "
                    "frames. Wire the conditioning and the latent from the "
                    "same node." % (int(prior_frame_count), frame_count))
            kept = []
            for keyframe in prior:
                position = int(keyframe.get(
                    MC_KEY, keyframe.get("resolved_frame_index", 0)))
                if position < head_end:
                    dropped.append(position)
                    continue
                keyframe = dict(keyframe)
                if _layout_native:
                    keyframe["resolved_frame_index"] = position
                    keyframe.pop(MC_KEY, None)
                else:
                    keyframe[MC_KEY] = position
                kept.append(keyframe)
            d["minimax_keyframes"] = kept + keyframes
            d["minimax_frame_count"] = frame_count
            out.append([emb, d])
        if dropped:
            _LOG.warning(
                "h3_motion_context: dropped %d keyframe anchor(s) at "
                "frame(s) %s: the pinned head already decides frames "
                "0..%d. A last_frame anchor is kept.",
                len(dropped), sorted(set(dropped)), head_end - 1)

        if audio_ref is not None:
            out = node_helpers.conditioning_set_values(
                out, {"minimax_refs": [audio_ref]}, append=True)

        trim = span if anchor_mode == "head" else 0
        audio_end_frame = (audio_ref.get(MC_AUDIO_KEY)
                           if ref_audio_t and audio_ref is not None else None)
        if audio_end_frame is not None:
            audio_place = "on the timeline ending at frame %.3f" % float(audio_end_frame)
        elif audio_mode == "timeline" and _layout_native:
            audio_place = "timeline mode requested but stock ref placement used (native interior anchors)"
        else:
            audio_place = "stock ref placement"
        _LOG.info("h3_motion_context: video from %s, %s/%s, %d frames -> %d "
                  "cond blocks at indices %d..%d, %d frame clip at %dx%d, "
                  "trim %d, audio %s",
                  video_src, encode_mode, anchor_mode, n, len(blocks),
                  indices[0], indices[-1], frame_count, width, height, trim,
                  ("%d frames -> %d latent steps (%.3fs) from %s, %s"
                   % (a_frames, ref_audio_t, ref_audio_t / AUDIO_HZ, audio_src,
                      audio_place))
                  if ref_audio_t else "off")
        return (out, trim)





import torch

import comfy.ldm.minimax.model as mm

_LAYOUT_PATCH_MARKER = "_h3_motion_context_layout_patch"

_layout_orig_init = None
_layout_applied = False
_layout_native = False

REF_SEGMENT_KINDS = ("ref_img", "ref_audio")


def _target_origin(layout):
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError(
            "h3_motion_context: expected the target video rows to be the "
            "last layout segment, found %r spanning %d rows. Upstream "
            "layout change; refusing to rewrite positions." % (kind, b - a))
    return float(layout.position_ids[a, 0])


def _expected_ref_segments(blk):
    kind = blk.get("kind")
    if kind == "image":
        return ("ref_img",)
    if kind == "audio":
        return ("ref_audio",) if int(blk.get("ref_audio_t", 0)) > 0 else ()
    if kind in ("video", "video_audio"):
        if int(blk.get("ref_audio_t", 0)) > 0:
            return ("ref_audio", "ref_img")
        return ("ref_img",)
    raise RuntimeError(
        "h3_motion_context: unknown reference kind %r; cannot tell which "
        "layout rows belong to it." % (kind,))


def _ref_segment_map(layout, refs):
    ref_segs = [(a, b, k) for a, b, k in layout.segments
                if k in REF_SEGMENT_KINDS]
    want = [(i, k) for i, blk in enumerate(refs or [])
            for k in _expected_ref_segments(blk)]
    if len(want) != len(ref_segs):
        raise RuntimeError(
            "h3_motion_context: %d reference blocks should have produced %d "
            "layout segments, the layout has %d. Upstream layout change; "
            "refusing to move rows." % (len(refs or []), len(want),
                                        len(ref_segs)))
    out = {}
    for (i, kind), (a, b, got) in zip(want, ref_segs):
        if got != kind:
            raise RuntimeError(
                "h3_motion_context: reference block %d (%r) should have "
                "emitted a %s segment, the layout has %s. Upstream layout "
                "change; refusing to move rows."
                % (i, refs[i].get("kind"), kind, got))
        out.setdefault(i, {})[kind] = (a, b)
    return out


def _cond_t(text_len, latent_t, frame_count, p):
    if p == 0:
        return float(text_len)
    if frame_count is not None and p == frame_count - 1:
        return float(text_len) + sum(mm._video_t_spans(latent_t)) - mm.FRAME_RESCALE
    return float(text_len) + mm.FRAME_RESCALE * float(p)


def _fixup(layout, text_len, latent_t, frame_count, keyframes, refs=None):
    offset = _target_origin(layout) - float(text_len)
    if offset and any(kf.get(MC_KEY) is None for kf in keyframes):
        raise RuntimeError(
            "h3_motion_context: stock and motion-context keyframes mixed in "
            "one graph alongside a ref; their coordinates would disagree. "
            "Give every keyframe a %s entry or remove the refs." % MC_KEY)
    cond_spans = [(a, b) for a, b, kind in layout.segments if kind == "cond"]
    if len(cond_spans) != len(keyframes):
        raise RuntimeError(
            "h3_motion_context: expected %d cond segments, layout has %d. "
            "Refusing to rewrite positions."
            % (len(keyframes), len(cond_spans)))
    for (a, b), kf in zip(cond_spans, keyframes):
        p = kf.get(MC_KEY)
        if p is None:
            continue
        layout.position_ids[a:b, 0] = _cond_t(text_len, latent_t, frame_count, p) + offset


def _fixup_audio(layout, text_len, refs):
    marked = [i for i, r in enumerate(refs or [])
              if r.get(MC_AUDIO_KEY) is not None]
    if len(marked) != 1:
        raise RuntimeError(
            "h3_motion_context: audio timeline placement needs exactly one "
            "reference marked with %s; the layout has %d references and %d "
            "marked. If this appeared during startup, check for more than "
            "one H3 Motion Context folder in custom_nodes."
            % (MC_AUDIO_KEY, len(refs or []), len(marked)))
    idx = marked[0]
    blk = refs[idx]
    if blk.get("kind") != "audio":
        raise RuntimeError(
            "h3_motion_context: %s set on a %r ref; only audio refs can be "
            "moved onto the timeline." % (MC_AUDIO_KEY, blk.get("kind")))
    rt = int(blk.get("ref_audio_t", 0))
    if rt <= 0:
        return

    seg = _ref_segment_map(layout, refs).get(idx, {}).get("ref_audio")
    if seg is None:
        raise RuntimeError(
            "h3_motion_context: the marked audio reference produced no "
            "ref_audio segment. Upstream layout change; refusing to move "
            "rows.")
    a, b = seg
    if b - a != 2 * rt:
        raise RuntimeError(
            "h3_motion_context: the marked audio reference has %d rows for "
            "%d latent steps, expected %d (stereo, channel-major). Upstream "
            "layout change; refusing to move rows." % (b - a, rt, 2 * rt))

    target_origin = _target_origin(layout)
    slot_start = float(layout.position_ids[a, 0])
    end_frame = float(blk[MC_AUDIO_KEY])
    desired_start = target_origin + mm.FRAME_RESCALE * end_frame - float(rt)
    layout.position_ids[a:b, 0] = (layout.position_ids[a:b, 0]
                                   + (desired_start - slot_start))


def _patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                  keyframes=None, refs=None, frame_count=None):
    _call_layout_init(_layout_orig_init, self, text_len, latent_t, latent_h,
                      latent_w, audio_t, keyframes=keyframes, refs=refs,
                      frame_count=frame_count)
    has_mc_kf = bool(keyframes) and any(
        kf.get(MC_KEY) is not None for kf in keyframes)
    has_mc_audio = bool(refs) and any(
        r.get(MC_AUDIO_KEY) is not None for r in refs)
    if has_mc_kf:
        _fixup(self, text_len, latent_t, frame_count, keyframes, refs)
    if has_mc_audio:
        _fixup_audio(self, text_len, refs)


def _layout_self_test():
    text_len, latent_t, lh, lw, audio_t = 7, 7, 22, 38, 16
    frame_count = sum(mm.FRAME_PER_TOKEN[k % 5] for k in range(latent_t))

    def build(keyframes=None, refs=None, fix=False, move=False):
        lay = mm.PackedLayout.__new__(mm.PackedLayout)
        _call_layout_init(_layout_orig_init, lay, text_len, latent_t, lh, lw,
                          audio_t, keyframes=keyframes, refs=refs,
                          frame_count=frame_count)
        if fix:
            _fixup(lay, text_len, latent_t, frame_count, keyframes, refs)
        if move:
            _fixup_audio(lay, text_len, refs)
        return lay

    def cond_ts(lay):
        return [float(lay.position_ids[a, 0])
                for a, _, k in lay.segments if k == "cond"]

    stock_kf = [{"resolved_frame_index": 0},
                {"resolved_frame_index": frame_count - 1}]
    ours_kf = [{"resolved_frame_index": 0, MC_KEY: 0},
               {"resolved_frame_index": 0, MC_KEY: frame_count - 1}]
    a = build(keyframes=stock_kf)
    b = build(keyframes=ours_kf, fix=True)
    if a.position_ids.shape != b.position_ids.shape:
        raise RuntimeError("position_ids shape mismatch in self-test")
    if not torch.equal(a.position_ids, b.position_ids):
        bad = (a.position_ids != b.position_ids).any(dim=1).nonzero().flatten()
        raise RuntimeError("position mismatch at rows %s" % bad[:8].tolist())

    run = [{"resolved_frame_index": 0, MC_KEY: i} for i in range(4)]
    c = build(keyframes=run, fix=True)
    ts = cond_ts(c)
    if len(ts) != len(run):
        raise RuntimeError("expected %d cond segments, got %d" % (len(run), len(ts)))
    if any(ts[i] >= ts[i + 1] for i in range(len(ts) - 1)):
        raise RuntimeError("consecutive anchors not strictly increasing: %s" % ts)
    t_last = float(text_len) + mm.FRAME_RESCALE * (frame_count - 1)
    if not (ts[0] == float(text_len) and ts[-1] < t_last):
        raise RuntimeError("run %s escapes the [%.4f, %.4f] span"
                           % (ts, float(text_len), t_last))

    ref = [{"kind": "audio", "ref_audio_t": 8}]
    d = build(keyframes=run, refs=ref, fix=True)
    ts_ref = cond_ts(d)
    if len(ts_ref) != len(ts):
        raise RuntimeError("cond segment count changed when a ref was added")
    tol = 1e-3
    gap = float(c.position_ids[:, 0].max()) - ts[0]
    gap_ref = float(d.position_ids[:, 0].max()) - ts_ref[0]
    if abs(gap - gap_ref) > tol:
        raise RuntimeError(
            "ref compensation off by %.6f: anchor-to-target gap %.6f without "
            "ref, %.6f with. The target origin read back from the layout no "
            "longer matches its cursor arithmetic." % (gap_ref - gap, gap, gap_ref))
    shifts = [y - x for x, y in zip(ts, ts_ref)]
    if any(abs(sh - shifts[0]) > tol for sh in shifts):
        raise RuntimeError("ref shifted anchors unevenly: %s" % shifts)

    end_frame, rt = 4, 8
    ref_mc = [{"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}]
    e = build(keyframes=run, refs=ref_mc, fix=True, move=True)
    _check_move(d, e, ref_mc, 0, "single-ref")

    r_lh, r_lw, r_vt = 8, 12, 3
    others = [
        {"kind": "image", "latent_h": r_lh, "latent_w": r_lw},
        {"kind": "video_audio", "latent_h": r_lh, "latent_w": r_lw,
         "latent_t": r_vt, "ref_audio_t": 5},
        {"kind": "audio", "ref_audio_t": 3},
    ]
    marked = {"kind": "audio", "ref_audio_t": rt, MC_AUDIO_KEY: end_frame}
    plain = {"kind": "audio", "ref_audio_t": rt}
    multi_plain = others[:2] + [plain] + others[2:]
    multi_marked = others[:2] + [marked] + others[2:]
    f = build(keyframes=run, refs=multi_plain, fix=True)
    g = build(keyframes=run, refs=multi_marked, fix=True, move=True)
    _check_move(f, g, multi_marked, 2, "multi-ref")

    smap = _ref_segment_map(f, multi_plain)
    prev_hi = float(text_len) - 1e-9
    origin = _target_origin(f)
    for i in range(len(multi_plain)):
        spans = smap.get(i)
        if not spans:
            continue
        rows = [r for a0, b0 in spans.values() for r in range(a0, b0)]
        lo = min(float(f.position_ids[r, 0]) for r in rows)
        hi = max(float(f.position_ids[r, 0]) for r in rows)
        if lo < prev_hi - 1e-9:
            raise RuntimeError(
                "reference block %d starts at %.6f, before block %d ended "
                "at %.6f. Reference blocks are not laid out in list order."
                % (i, lo, i - 1, prev_hi))
        if hi >= origin - 1e-9:
            raise RuntimeError(
                "reference block %d reaches %.6f, at or past the target "
                "origin %.6f. Reference rows should sit before the target."
                % (i, hi, origin))
        prev_hi = hi


def _check_move(before, after, refs, idx, label):
    if after.position_ids.shape != before.position_ids.shape:
        raise RuntimeError("%s: audio move changed the layout shape" % label)
    if not torch.equal(before.position_ids[:, 1:], after.position_ids[:, 1:]):
        raise RuntimeError(
            "%s: audio move touched a non-time coordinate column" % label)
    a, b = _ref_segment_map(before, refs)[idx]["ref_audio"]
    expect_moved = set(range(a, b))
    tb, ta = before.position_ids[:, 0], after.position_ids[:, 0]
    moved = set(i for i in range(len(tb)) if float(tb[i]) != float(ta[i]))
    if not moved:
        raise RuntimeError("%s: audio move moved no rows" % label)
    if moved != expect_moved:
        raise RuntimeError(
            "%s: audio move touched the wrong rows: %d moved, %d expected, "
            "e.g. %s" % (label, len(moved), len(expect_moved),
                         sorted(moved ^ expect_moved)[:8]))
    deltas = [float(ta[i]) - float(tb[i]) for i in sorted(moved)]
    if any(abs(dd - deltas[0]) > 1e-9 for dd in deltas):
        raise RuntimeError("%s: audio rows shifted non-uniformly: %s"
                           % (label, deltas[:4]))

    blk = refs[idx]
    rt = int(blk["ref_audio_t"])
    want_end = (_target_origin(after)
                + mm.FRAME_RESCALE * float(blk[MC_AUDIO_KEY]))
    got_end = float(after.position_ids[a, 0]) + float(rt)
    if abs(got_end - want_end) > 1e-9:
        raise RuntimeError(
            "%s: audio window ends at %.6f, should end at %.6f"
            % (label, got_end, want_end))


setattr(_patched_init, _LAYOUT_PATCH_MARKER, True)


def _call_layout_init(init, self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=None, refs=None, frame_count=None):
    """Call PackedLayout.__init__, forwarding frame_count only when the
    stock constructor accepts it.

    ComfyUI 0.33.0+ generalised the keyframe position formula and removed
    the frame_count parameter. Passing it on those builds raises
    TypeError on an unexpected keyword argument, so probe the signature
    (with a fallback test call) and strip it when unsupported.
    """
    import inspect
    kwargs = dict(keyframes=keyframes, refs=refs)
    send_fc = False
    try:
        sig = inspect.signature(init)
        send_fc = "frame_count" in sig.parameters
    except (ValueError, TypeError):
        pass
    if not send_fc:
        try:
            init(self, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)
            return
        except TypeError:
            send_fc = True
            # fall through to retry with frame_count (older build)
    if frame_count is not None:
        kwargs["frame_count"] = frame_count
    init(self, text_len, latent_t, latent_h, latent_w, audio_t, **kwargs)


_H3_INTERIOR_SOURCE_MARK = "FRAME_RESCALE * float(pixel_index)"
_H3_INTERIOR_CLASS_MARK = "_h3_interior_keyframes_patched"


def _layout_source_has_interior_form(cls):
    """Detect source-level (exec/monkey-patch) interior-anchor rewrites.

    Packs like H3-Multishot do ``inspect.getsource`` + string replace +
    ``exec`` instead of wrapping the constructor. Their output is invisible
    to wrapper markers but leaves a clear fingerprint in the source text.
    """
    init = getattr(cls, "__init__", None)
    if init is None:
        return False
    try:
        import inspect
        raw = inspect.getsource(init)
    except Exception:
        # frozen / built-in / bytecode-only: fall back to attribute marks
        return bool(getattr(cls, _H3_INTERIOR_CLASS_MARK, False))
    if _H3_INTERIOR_SOURCE_MARK in raw:
        return True
    if "only first/last keyframe anchors are supported" not in raw:
        # No stock rejection branch → someone already generalised it
        # (hand-patched core build, exec patch, different pack)
        return bool(getattr(cls, _H3_INTERIOR_CLASS_MARK, False))
    return False


def _layout_already_patched():
    cls = getattr(mm, "PackedLayout", None)
    init = getattr(cls, "__init__", None)
    if init is None:
        return None
    if getattr(init, _LAYOUT_PATCH_MARKER, False):
        return "same"
    if getattr(init, "__name__", "") == "_patched_init":
        return "other"
    if hasattr(init, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(init, "__module__", None)
    if home and where and where != home:
        return "foreign"
    if _layout_source_has_interior_form(cls):
        return "foreign"
    return None


def _apply_layout_patch():
    global _layout_orig_init, _layout_applied, _layout_native
    if _layout_applied:
        return True
    who = _layout_already_patched()

    def _class_has_native_interior():
        """Probe whether mm.PackedLayout actually supports interior anchors.

        Two paths count as ``native``: the stock constructor accepts an
        interior ``resolved_frame_index`` and does not require
        ``frame_count``, OR another pack has already installed a wrapper
        that our caller-side MC_KEY/MC_AUDIO_KEY paths can rely on.
        """
        if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
            return False
        import inspect
        stock_has_fc = False
        try:
            stock_sig = inspect.signature(mm.PackedLayout.__init__)
            stock_has_fc = "frame_count" in stock_sig.parameters
        except (ValueError, TypeError):
            try:
                probe = mm.PackedLayout.__new__(mm.PackedLayout)
                mm.PackedLayout.__init__(probe, 4, 3, 8, 8, 4,
                                         keyframes=None, refs=None)
                stock_has_fc = False
            except TypeError:
                try:
                    probe2 = mm.PackedLayout.__new__(mm.PackedLayout)
                    mm.PackedLayout.__init__(probe2, 4, 3, 8, 8, 4,
                                             keyframes=None, refs=None,
                                             frame_count=9)
                    stock_has_fc = True
                except TypeError:
                    stock_has_fc = False
        if not stock_has_fc:
            import torch
            try:
                probe = mm.PackedLayout(
                    4, 3, 8, 8, 4,
                    keyframes=[{"resolved_frame_index": 1,
                                "latent": torch.zeros(1, 16, 1, 8, 8)}],
                    refs=None)
                return True
            except Exception:
                return False
        return False

    if who == "foreign":
        _layout_applied = True
        # Source-level / exec-style patches generalise the keyframe formula
        # but still accept ``resolved_frame_index`` natively; they do NOT
        # consume MC_KEY markers, so our side must emit native anchors.
        _layout_native = _class_has_native_interior()
        _LOG.info(
            "h3_motion_context: another pack has already generalised H3's "
            "first/last keyframe restriction (source-level patch from %r). "
            "Apt_Preset is not installing its own wrapper; native-support "
            "probe: %s.",
            getattr(getattr(getattr(mm, "PackedLayout", None), "__init__",
                            None), "__module__", "?"),
            _layout_native)
        return True
    if who:
        _layout_applied = True
        # A wrapper-style patch is already present. The wrapper is expected
        # to consume MC_KEY/MC_AUDIO_KEY markers, so ``_layout_native`` must
        # stay False. But if the probe says native support is also present
        # (wrapper + native coexist because it fell back to a no-op), prefer
        # the native path on our side so keyframes and audio refs align.
        if _class_has_native_interior() and (
                who != "same" or not getattr(
                    getattr(mm.PackedLayout, "__init__", None),
                    "moves_audio_refs", False)):
            _layout_native = True
        if who == "same":
            _LOG.info("h3_motion_context: interior keyframe anchors already "
                      "enabled by another pack, standing down (native=%s)",
                      _layout_native)
        else:
            _LOG.warning(
                "h3_motion_context: the H3 layout patch is already installed "
                "by a DIFFERENT copy of this code (another version, or a "
                "fork). Standing down; that copy decides what the patch "
                "supports, so features added since it may be unavailable. "
                "If you have more than one H3 Motion Context folder in "
                "custom_nodes, keep one and remove the rest. Renaming a "
                "folder does not stop ComfyUI loading it. (native=%s)",
                _layout_native)
        return True
    if not hasattr(mm, "PackedLayout") or not hasattr(mm, "FRAME_RESCALE"):
        _LOG.warning("h3_motion_context: MiniMax H3 model module missing expected "
                     "attributes, patch not applied")
        return False

    # Newer ComfyUI builds already ship the general keyframe position formula
    # (cursor + FRAME_RESCALE * resolved_frame_index) and dropped the
    # frame_count constructor argument. Our wrapper and self-test both pass
    # frame_count=... so they cannot run against that stock. Detect it and
    # skip the wrapper entirely: stock already does what the wrapper did.
    import inspect
    stock_has_fc = False  # default: assume recent ComfyUI (post 0.33.0)
    try:
        stock_sig = inspect.signature(mm.PackedLayout.__init__)
        stock_has_fc = "frame_count" in stock_sig.parameters
    except (ValueError, TypeError):
        # Signature probe failed; fall back to a real call probe. Try to
        # construct a PackedLayout WITHOUT frame_count first: if it works the
        # build is recent and native-supporting.
        try:
            probe = mm.PackedLayout.__new__(mm.PackedLayout)
            mm.PackedLayout.__init__(probe, 4, 3, 8, 8, 4, keyframes=None, refs=None)
            stock_has_fc = False
        except TypeError:
            # Rejected because refs/kwargs shape, or because frame_count is
            # required as positional. Try the other way to disambiguate.
            try:
                probe2 = mm.PackedLayout.__new__(mm.PackedLayout)
                mm.PackedLayout.__init__(probe2, 4, 3, 8, 8, 4, keyframes=None,
                                         refs=None, frame_count=9)
                stock_has_fc = True
            except TypeError:
                stock_has_fc = False
    if not stock_has_fc:
        _layout_native = True
        _layout_applied = True
        _LOG.info("h3_motion_context: stock ComfyUI already supports interior "
                  "keyframe anchors; skipping wrapper, using resolved_frame_index "
                  "directly")
        return True

    _layout_orig_init = mm.PackedLayout.__init__
    try:
        _layout_self_test()
    except Exception as exc:
        _layout_orig_init = None
        # Self-test failure is not always terminal: newer ComfyUI builds
        # might have changed the layout internals while still supporting
        # resolved_frame_index natively. Before giving up, probe one anchor
        # at an interior position directly through stock.
        try:
            probe = mm.PackedLayout(4, 3, 8, 8, 4,
                                    keyframes=[{"resolved_frame_index": 1,
                                                "latent": torch.zeros(1, 16, 1, 8, 8)}],
                                    refs=None)
        except Exception as probe_exc:
            _LOG.warning("h3_motion_context: native interior probe also "
                         "failed (%s), patch not applied. Interior keyframe "
                         "anchors unavailable.", probe_exc)
            _LOG.warning("h3_motion_context: self-test failure was (%s)", exc)
            _LOG.warning(
                "h3_motion_context: if you have more than one H3 Motion Context "
                "folder in custom_nodes (a fork, a backup, a manual clone "
                "alongside a Manager install), that is the usual cause: each "
                "copy self-tests against whichever one loaded first. Keep one "
                "and remove the rest. Renaming a folder does not stop ComfyUI "
                "loading it. Otherwise this is an upstream ComfyUI change and "
                "the message above says what moved.")
            return False
        _layout_native = True
        _layout_applied = True
        _LOG.warning("h3_motion_context: wrapper self-test failed (%s) but "
                     "the stock constructor accepted an interior "
                     "resolved_frame_index; switching to the native path.",
                     exc)
        return True
    mm.PackedLayout.__init__ = _patched_init
    _layout_applied = True
    _LOG.info("h3_motion_context: interior keyframe anchors enabled")
    return True


def _layout_patch_applied():
    return _layout_applied


import comfy.model_base as model_base

_PAYLOAD_PATCH_MARKER = "_h3_motion_context_payload_patch"

_payload_orig_extra_conds = None
_payload_applied = False


def _patched_extra_conds(self, **kwargs):
    out = _payload_orig_extra_conds(self, **kwargs)

    keyframes = kwargs.get("minimax_keyframes", None)
    refs = kwargs.get("minimax_refs", None)
    if not keyframes or not refs:
        return out
    if not (any(MC_KEY in kf for kf in keyframes)
            or any(MC_AUDIO_KEY in r for r in refs)):
        return out

    cond = out.get("minimax_payload", None)
    payload = getattr(cond, "cond", None) if cond is not None else None
    if not isinstance(payload, dict):
        _LOG.warning("h3_motion_context: could not reach the H3 payload, "
                     "keyframe latents may have been overwritten by refs")
        return out

    kf_video = [kf["latent"] for kf in keyframes if "latent" in kf]
    ref_video = [r["latent"] for r in refs if "latent" in r]
    payload["cond_video_latents"] = kf_video + ref_video
    payload["cond_audio_latents"] = [r["audio_latent"] for r in refs
                                     if r.get("audio_latent") is not None]

    fc = kwargs.get("minimax_frame_count", None)
    if fc is not None:
        payload["frame_count"] = fc
    return out


setattr(_patched_extra_conds, _PAYLOAD_PATCH_MARKER, True)


def _payload_already_patched(cls):
    fn = getattr(cls, "extra_conds", None)
    if fn is None:
        return None
    if getattr(fn, _PAYLOAD_PATCH_MARKER, False):
        return "same"
    if getattr(fn, "__name__", "") == "_patched_extra_conds":
        return "other"
    if hasattr(fn, "__wrapped__"):
        return "foreign"
    home = getattr(cls, "__module__", None)
    where = getattr(fn, "__module__", None)
    if home and where and where != home:
        return "foreign"
    return None


def _apply_payload_patch():
    global _payload_orig_extra_conds, _payload_applied
    if _payload_applied:
        return True
    cls = getattr(model_base, "MiniMaxH3", None)
    if cls is None or not hasattr(cls, "extra_conds"):
        _LOG.warning("h3_motion_context: MiniMaxH3.extra_conds not found, "
                     "keyframes and refs cannot be combined")
        return False
    who = _payload_already_patched(cls)
    if who == "foreign":
        _payload_applied = True
        _LOG.info(
            "h3_motion_context: another pack has already patched "
            "MiniMaxH3.extra_conds (it now comes from %r). Apt_Preset is "
            "standing down instead of wrapping the wrapper; the installed "
            "patch should already handle keyframe+ref coexistence.",
            getattr(getattr(cls, "extra_conds", None), "__module__", "?"))
        return True
    if who:
        _payload_applied = True
        if who == "same":
            _LOG.info("h3_motion_context: keyframe/ref coexistence already "
                      "enabled by another pack, standing down")
        else:
            _LOG.warning(
                "h3_motion_context: the H3 payload patch is already "
                "installed by a DIFFERENT copy of this code (another "
                "version, or a fork). Standing down. If you have more than "
                "one H3 Motion Context folder in custom_nodes, keep one and "
                "remove the rest.")
        return True
    _payload_orig_extra_conds = cls.extra_conds
    cls.extra_conds = _patched_extra_conds
    _payload_applied = True
    _LOG.info("h3_motion_context: keyframe/ref coexistence enabled")
    return True


def _payload_patch_applied():
    return _payload_applied
