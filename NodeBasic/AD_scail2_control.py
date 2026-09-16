from fractions import Fraction

import comfy.model_management
import comfy.utils
import folder_paths
import nodes
import torch
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_post_processing import ColorTransfer
from comfy_extras.nodes_scail import WanSCAILToVideo
from comfy.patcher_extension import WrappersMP

from ..main_unit import load_upscale_model, new_context, upscale_with_model
from .C_AD import (
    _AD_H3_LATENT_TILE_PRESETS,
    _ad_repair_boundary_flash,
    _ad_stage_info,
    _ad_stage_output_dir,
    _ad_stage_video_outputs,
)
from .minimaxH3 import _ad_h3_aligned_tile_regions, _ad_h3_core_halo_window


_SCAIL_NO_LATENT_TILE = next(iter(_AD_H3_LATENT_TILE_PRESETS))
_SCAIL_TILED_APPLY_MODEL_KEY = "apt_scail2_tiled_apply_model"
_SCAIL_SPATIAL_CONDS = ("pose_latents", "sam_latents", "reference_latent", "ref_mask_latents")


def _scail_crop_spatial(value, axis, start, end):
    if not torch.is_tensor(value) or value.ndim != 5:
        return value
    if axis == "H":
        return value[:, :, :, start:end, :].contiguous()
    return value[:, :, :, :, start:end].contiguous()


def _scail_crop_scaled(value, axis, start, end, full_size):
    if not torch.is_tensor(value) or value.ndim != 5:
        return value
    cond_size = int(value.shape[-2] if axis == "H" else value.shape[-1])
    cond_start = start * cond_size // full_size
    cond_end = (end * cond_size + full_size - 1) // full_size
    return _scail_crop_spatial(value, axis, cond_start, cond_end)


class _SCAIL2TiledApplyModel:
    def __init__(self, tile_count, overlap_pixels):
        self.tile_count = int(tile_count)
        self.overlap_pixels = int(overlap_pixels)

    def apply_model_wrapper(self, executor, x, t, c_concat=None, c_crossattn=None,
                            control=None, transformer_options={}, **kwargs):
        if not torch.is_tensor(x) or x.ndim != 5:
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        height, width = int(x.shape[-2]), int(x.shape[-1])
        axis = "H" if height >= width else "W"
        full_size = height if axis == "H" else width
        halo = max(0, int(round(self.overlap_pixels / 8.0)))
        regions = _ad_h3_aligned_tile_regions(full_size, self.tile_count, halo, alignment=4)
        if len(regions) <= 1 or any(start == 0 and end == full_size for start, end, _, _ in regions):
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        output = torch.zeros_like(x, dtype=torch.float32)
        weight_shape = (1, 1, 1, full_size, 1) if axis == "H" else (1, 1, 1, 1, full_size)
        weights = torch.zeros(weight_shape, dtype=torch.float32, device=x.device)
        for start, end, core_start, core_end in regions:
            window = _ad_h3_core_halo_window(start, end, core_start, core_end, x.device)
            window = window.view(1, 1, 1, -1, 1) if axis == "H" else window.view(1, 1, 1, 1, -1)
            tile_x = _scail_crop_spatial(x, axis, start, end)
            tile_concat = _scail_crop_scaled(c_concat, axis, start, end, full_size)
            tile_kwargs = dict(kwargs)
            for name in _SCAIL_SPATIAL_CONDS:
                if name in tile_kwargs:
                    tile_kwargs[name] = _scail_crop_scaled(
                        tile_kwargs[name], axis, start, end, full_size
                    )
            prediction = executor(
                tile_x, t, tile_concat, c_crossattn, control, transformer_options,
                **tile_kwargs,
            )
            if axis == "H":
                output[:, :, :, start:end, :] += prediction.float() * window
                weights[:, :, :, start:end, :] += window
            else:
                output[:, :, :, :, start:end] += prediction.float() * window
                weights[:, :, :, :, start:end] += window
        return (output / weights.clamp_min(1e-8)).to(dtype=x.dtype)


def _scail_tiled_model(model, tile_count, overlap_pixels):
    tiled_model = model.clone()
    state = _SCAIL2TiledApplyModel(tile_count, overlap_pixels)
    tiled_model.add_wrapper_with_key(
        WrappersMP.APPLY_MODEL, _SCAIL_TILED_APPLY_MODEL_KEY, state.apply_model_wrapper
    )
    return tiled_model


def _scail_resize_video_tensor(value, height, width, channel_dim=1, mode="bilinear"):
    if not torch.is_tensor(value) or value.ndim != 5:
        return value
    if channel_dim == 2:
        value = value.movedim(2, 1)
    batch, channels, frames, source_height, source_width = value.shape
    if (source_height, source_width) != (height, width):
        images = value.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, source_height, source_width)
        images = comfy.utils.common_upscale(images, width, height, mode, "disabled")
        value = images.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)
    if channel_dim == 2:
        value = value.movedim(1, 2)
    return value.contiguous()


def _scail_resize_refine_conditioning(conditioning, latent_height, latent_width):
    output = []
    pose_height = max(1, latent_height // 2)
    pose_width = max(1, latent_width // 2)
    for embedding, extra in conditioning:
        values = extra.copy()
        reference_latents = values.get("reference_latents")
        if isinstance(reference_latents, (list, tuple)):
            values["reference_latents"] = [
                _scail_resize_video_tensor(latent, latent_height, latent_width)
                for latent in reference_latents
            ]
        if values.get("pose_video_latent") is not None:
            values["pose_video_latent"] = _scail_resize_video_tensor(
                values["pose_video_latent"], pose_height, pose_width
            )
        if values.get("driving_mask_28ch") is not None:
            values["driving_mask_28ch"] = _scail_resize_video_tensor(
                values["driving_mask_28ch"], pose_height, pose_width,
                channel_dim=2, mode="nearest-exact",
            )
        if values.get("ref_mask_28ch") is not None:
            values["ref_mask_28ch"] = _scail_resize_video_tensor(
                values["ref_mask_28ch"], latent_height, latent_width,
                channel_dim=2, mode="nearest-exact",
            )
        output.append((embedding, values))
    return output


def _scail_file_video(value):
    video_type = getattr(InputImpl, "VideoFromFile", None)
    return video_type is not None and isinstance(value, video_type)


def _scail_source_frame_count(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return int(value.shape[0])
    if _scail_file_video(value):
        return int(value.get_frame_count())
    raise TypeError("AD_scail2_generate: pose inputs must be IMAGE tensors or file-backed VIDEO objects")


def _scail_segment_frames(value, start_frame, wanted_frames, padded_frames):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        segment = value[start_frame:]
    elif _scail_file_video(value):
        segment = value.get_components().images[start_frame:]
    else:
        raise TypeError("AD_scail2_generate: pose inputs must be IMAGE tensors or file-backed VIDEO objects")
    if segment.shape[0] == 0:
        return None
    if segment.shape[0] > wanted_frames:
        if wanted_frames == 1:
            segment = segment[:1]
        else:
            last = int(segment.shape[0]) - 1
            indices = [round(index * last / (wanted_frames - 1)) for index in range(wanted_frames)]
            segment = segment[indices]
    if segment.shape[0] < padded_frames:
        segment = torch.cat(
            (segment, segment[-1:].expand(padded_frames - segment.shape[0], -1, -1, -1)),
            dim=0,
        )
    return segment[:padded_frames]


def _scail_stage_frames(value, wanted_frames, context_frames, padded_frames, previous_frames=None):
    segment = _scail_segment_frames(value, 0, wanted_frames, wanted_frames)
    if segment is None:
        return None
    if context_frames > 0:
        context = None
        if isinstance(previous_frames, torch.Tensor) and previous_frames.shape[0] >= context_frames:
            context = previous_frames[-context_frames:]
            if tuple(context.shape[1:3]) != tuple(segment.shape[1:3]):
                context = torch.nn.functional.interpolate(
                    context.movedim(-1, 1),
                    size=tuple(segment.shape[1:3]),
                    mode="bilinear",
                    align_corners=False,
                ).movedim(1, -1)
            context = context.to(device=segment.device, dtype=segment.dtype)
        if context is None:
            context = segment[:1].expand(context_frames, -1, -1, -1)
        segment = torch.cat((context, segment), dim=0)
    if segment.shape[0] < padded_frames:
        segment = torch.cat(
            (segment, segment[-1:].expand(padded_frames - segment.shape[0], -1, -1, -1)),
            dim=0,
        )
    return segment[:padded_frames]


class AD_scail2_generate:
    @classmethod
    def INPUT_TYPES(cls):
        clip_vision_names = nodes.CLIPVisionLoader.INPUT_TYPES()["required"]["clip_name"][0]
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "positive": ("STRING", {"default": "reference motion  ", "multiline": True, "dynamicPrompts": True}),
                "negative": ("STRING", {"default": "bad", "multiline": False, "dynamicPrompts": True}),
                "length": ("INT", {
                    "default": 81,
                    "min": 1,
                    "max": nodes.MAX_RESOLUTION,
                    "step": 4,
                    "tooltip": "Final output frames per segment. The last segment uses its actual remaining source frames.",
                }),
                "segment_count": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 1,
                    "hidden": True,
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "clip_vision_name": (clip_vision_names, {"default": "clip_vision_h.safetensors"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "transfer_color": (["none", "reinhard_lab", "mkl_lab", "histogram"], {"default": "reinhard_lab"}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the pose latent."}),
                "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4, "tooltip": "Tail frames of previous_frames to anchor. SCAIL-2 was trained with 5."}),
                "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "False: Animation Mode with a black pose mask background. True: Replacement Mode with a white pose mask background."}),
                "latent_sample_tile": (
                    list(_AD_H3_LATENT_TILE_PRESETS),
                    {"default": _SCAIL_NO_LATENT_TILE, "tooltip": "Spatially tile each model prediction while keeping the sampler on the complete latent."},
                ),
            },
            "optional": {
                "stage_info_data1": ("FLOW_STAGE_INFO",),
                "pose_video": ("IMAGE,VIDEO", {"tooltip": "The clean pose-video segment for the current flow stage."}),
                "reference_image": ("IMAGE", {"tooltip": "Primary reference followed by optional additional identity views."}),
                "pose_video_mask": ("IMAGE,VIDEO", {"tooltip": "The clean SCAIL-2 mask-video segment for the current flow stage."}),
                "reference_image_mask": ("IMAGE", {"tooltip": "Colored reference masks matching reference_image."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"},
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE", "VIDEO", "VIDEO")
    RETURN_NAMES = ("context", "bridge_image", "segment_video", "merged_video")
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"
    DESCRIPTION = "Generate one Wan SCAIL-2 segment per flow stage, carry only tail frames, and concatenate saved MP4 segments."

    @classmethod
    def execute(cls, context, positive, negative, length, segment_count, fps, clip_vision_name, seed,
                transfer_color, pose_strength, previous_frame_count, replacement_mode=False,
                latent_sample_tile=_SCAIL_NO_LATENT_TILE,
                stage_info_data1=None, pose_video=None, pose_video_mask=None,
                reference_image=None, reference_image_mask=None, unique_id=None, workflow_prompt=None):
        clip = context.get("clip")
        vae = context.get("vae")
        width = context.get("width")
        height = context.get("height")
        model = context.get("model")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        missing = [name for name, value in (
            ("clip", clip), ("vae", vae), ("width", width), ("height", height),
            ("model", model), ("steps", steps), ("cfg", cfg), ("sampler", sampler), ("scheduler", scheduler),
        ) if value is None]
        if missing:
            raise ValueError(f"AD_scail2_generate context is missing: {', '.join(missing)}")
        if latent_sample_tile not in _AD_H3_LATENT_TILE_PRESETS:
            raise ValueError(f"AD_scail2_generate: unknown latent tile preset: {latent_sample_tile}")
        tile_count, overlap_pixels = _AD_H3_LATENT_TILE_PRESETS[latent_sample_tile]
        if segment_count != 1:
            raise ValueError("AD_scail2_generate: segment_count is fixed at 1")
        if stage_info_data1 is None:
            stage_index = 0
            total = 1
            run_id = None
        else:
            run_id = str(stage_info_data1.get("run_id") or "").strip()
            stage_index = int(stage_info_data1.get("stage_index", -1))
            total = int(stage_info_data1.get("total", 0))
            if not run_id or stage_index < 0 or total < 1 or stage_index >= total:
                raise ValueError("AD_scail2_generate: invalid stage_info_data1")

        if total > 1 and length < previous_frame_count:
            raise ValueError(
                "AD_scail2_generate: length must be at least previous_frame_count when using multiple segments"
            )
        source_lengths = [
            _scail_source_frame_count(value)
            for value in (pose_video, pose_video_mask)
            if value is not None
        ]
        source_frames = min(source_lengths) if source_lengths else length
        output_length = min(length, source_frames)
        if output_length <= 0:
            raise ValueError(f"AD_scail2_generate: stage {stage_index + 1} input has no source frames")
        context_frames = previous_frame_count if stage_index > 0 else 0
        conditioned_length = context_frames + output_length
        generation_length = ((conditioned_length - 1 + 3) // 4) * 4 + 1

        previous_frames = None
        if stage_index > 0:
            previous_stage = stage_info_data1.get("stage_data_1", stage_info_data1.get("stage_data"))
            if not isinstance(previous_stage, torch.Tensor) or previous_stage.ndim != 4:
                raise TypeError("AD_scail2_generate: the previous flow stage data_1 must be an IMAGE tensor")
            if previous_stage.shape[0] < previous_frame_count:
                raise ValueError("AD_scail2_generate: previous stage has fewer frames than previous_frame_count")
            previous_frames = previous_stage[-previous_frame_count:].clone()

        pose_video = _scail_stage_frames(
            pose_video, output_length, context_frames, generation_length, previous_frames
        )
        pose_video_mask = _scail_stage_frames(
            pose_video_mask, output_length, context_frames, generation_length
        )

        base_positive = nodes.CLIPTextEncode().encode(clip, positive)[0]
        base_negative = nodes.CLIPTextEncode().encode(clip, negative)[0]
        clip_vision_output = None
        if reference_image is not None:
            clip_vision = nodes.CLIPVisionLoader().load_clip(clip_vision_name)[0]
            clip_vision_output = nodes.CLIPVisionEncode().encode(clip_vision, reference_image, "none")[0]

        video_frame_offset = context_frames

        positive, negative, latent, _next_offset = WanSCAILToVideo.execute(
            positive=base_positive,
            negative=base_negative,
            vae=vae,
            width=width,
            height=height,
            length=generation_length,
            batch_size=1,
            pose_strength=pose_strength,
            pose_start=0.0,
            pose_end=1.0,
            video_frame_offset=video_frame_offset,
            previous_frame_count=previous_frame_count,
            replacement_mode=replacement_mode,
            reference_image=reference_image,
            clip_vision_output=clip_vision_output,
            pose_video=pose_video,
            pose_video_mask=pose_video_mask,
            reference_image_mask=reference_image_mask,
            previous_frames=previous_frames,
        ).result
        segment_seed = (seed + stage_index) & 0xffffffffffffffff
        sample_model = (
            _scail_tiled_model(model, tile_count, overlap_pixels)
            if tile_count > 1 else model
        )
        latent = nodes.common_ksampler(
            sample_model, segment_seed, steps, cfg, sampler, scheduler,
            positive, negative, latent, denoise=1.0,
        )[0]
        stage_image = nodes.VAEDecode().decode(vae, latent)[0]
        if transfer_color != "none":
            if reference_image is None:
                raise ValueError("AD_scail2_generate: reference_image is required when transfer_color is enabled")
            stage_image = ColorTransfer.execute(
                image_target=stage_image,
                image_ref=reference_image,
                method=transfer_color,
                source_stats={"source_stats": "per_frame"},
                strength=1.0,
            ).result[0]

        if stage_image.shape[0] != generation_length:
            raise ValueError(
                f"AD_scail2_generate: generated segment has {stage_image.shape[0]} frames, expected {generation_length}"
            )
        segment_images = stage_image[context_frames:context_frames + output_length]
        bridge_image = segment_images[-previous_frame_count:].clone()
        segment_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=segment_images,
                audio=None,
                frame_rate=Fraction(str(fps)),
            )
        )
        overlap_images = stage_image[:previous_frame_count] if stage_index > 0 else None
        segment_video, merged_video = _ad_stage_video_outputs(
            segment_video,
            run_id,
            stage_index,
            total,
            workflow_prompt,
            unique_id,
            "AD_scail2_generate",
            overlap_images=overlap_images,
            merged_output_slot=3,
        )
        if run_id is None:
            merged_video = segment_video
        else:
            segment_path = _ad_stage_output_dir(run_id)
            segment_video = InputImpl.VideoFromFile(
                f"{segment_path}/segments/{stage_index + 1:05d}.mp4"
            )

        output_context = new_context(
            context,
            positive=base_positive,
            negative=base_negative,
            images=bridge_image,
            vae=vae,
            width=width,
            height=height,
            batch=1,
        )
        output_context["latent"] = None
        output_context["apt_scail2_refine_state"] = {
            "positive": positive,
            "negative": negative,
            "full_images": stage_image.detach().to(comfy.model_management.intermediate_device()),
            "context_frames": context_frames,
            "bridge_frames": previous_frame_count,
            "output_length": output_length,
        }
        return output_context, bridge_image, segment_video, merged_video


class AD_scail2_generate_refine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "seed": AD_scail2_generate.INPUT_TYPES()["required"]["seed"],
                "pixel_refine_model": (
                    ["None"] + folder_paths.get_filename_list("upscale_models"),
                    {"default": "None", "tooltip": "Optional pixel upscaler before the refine pass."},
                ),
                "upscale_output_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Scale applied after pixel upscaling. Ignored when pixel_refine_model is None.",
                }),
                "refine_denoise": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "latent_sample_tile": (
                    list(_AD_H3_LATENT_TILE_PRESETS),
                    {"default": _SCAIL_NO_LATENT_TILE, "tooltip": "Spatially tile each refine prediction while keeping the complete latent sampler state."},
                ),
            },
            "optional": {
                "stage_info_data2": ("FLOW_STAGE_INFO",),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "VIDEO")
    RETURN_NAMES = ("bridge_image", "segment_video", "merged_video")
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"
    DESCRIPTION = "Pixel-refine a SCAIL-2 stage and carry its refined tail through flow data2."

    @classmethod
    def execute(cls, context, fps, seed, pixel_refine_model, upscale_output_scale,
                refine_denoise, latent_sample_tile, stage_info_data2=None,
                unique_id=None, workflow_prompt=None):
        if stage_info_data2 is None:
            run_id, stage_index, total = None, 0, 1
        else:
            run_id, stage_index, total = _ad_stage_info(stage_info_data2)

        state = context.get("apt_scail2_refine_state") if isinstance(context, dict) else None
        if not isinstance(state, dict):
            raise ValueError("AD_scail2_generate_refine needs context produced by AD_scail2_generate")
        full_images = state.get("full_images")
        positive = state.get("positive")
        negative = state.get("negative")
        if not torch.is_tensor(full_images) or full_images.ndim != 4:
            raise ValueError("AD_scail2_generate_refine context is missing the first-pass video")
        if positive is None or negative is None:
            raise ValueError("AD_scail2_generate_refine context is missing SCAIL-2 conditioning")

        model = context.get("model")
        vae = context.get("vae")
        steps = context.get("steps")
        cfg = context.get("cfg")
        sampler = context.get("sampler")
        scheduler = context.get("scheduler")
        missing = [name for name, value in (
            ("model", model), ("vae", vae), ("steps", steps), ("cfg", cfg),
            ("sampler", sampler), ("scheduler", scheduler),
        ) if value is None]
        if missing:
            raise ValueError(f"AD_scail2_generate_refine context is missing: {', '.join(missing)}")
        if latent_sample_tile not in _AD_H3_LATENT_TILE_PRESETS:
            raise ValueError(f"AD_scail2_generate_refine: unknown latent tile preset: {latent_sample_tile}")
        if pixel_refine_model != "None" and pixel_refine_model not in folder_paths.get_filename_list("upscale_models"):
            raise ValueError(f"AD_scail2_generate_refine: invalid pixel_refine_model: {pixel_refine_model}")

        work_images = full_images
        if pixel_refine_model != "None":
            work_images = upscale_with_model(load_upscale_model(pixel_refine_model), work_images)
            image_height, image_width = work_images.shape[1:3]
            target_width = max(32, round(image_width * float(upscale_output_scale) / 32) * 32)
            target_height = max(32, round(image_height * float(upscale_output_scale) / 32) * 32)
            if (target_height, target_width) != (image_height, image_width):
                work_images = comfy.utils.common_upscale(
                    work_images.movedim(-1, 1), target_width, target_height,
                    "lanczos", "disabled",
                ).movedim(1, -1)

        context_frames = int(state.get("context_frames", 0))
        bridge_frames = int(state.get("bridge_frames", 5))
        previous_refined = (
            stage_info_data2.get("stage_data_2")
            if stage_info_data2 is not None and stage_index > 0 and context_frames > 0
            else None
        )
        if stage_index > 0 and context_frames > 0:
            if not torch.is_tensor(previous_refined) or previous_refined.ndim != 4:
                raise ValueError("AD_scail2_generate_refine: stage_info_data2 does not contain refined bridge images")
            if int(previous_refined.shape[0]) < context_frames:
                raise ValueError("AD_scail2_generate_refine: refined bridge has too few frames")
            previous_refined = previous_refined[-context_frames:]
            if tuple(previous_refined.shape[1:3]) != tuple(work_images.shape[1:3]):
                previous_refined = comfy.utils.common_upscale(
                    previous_refined.movedim(-1, 1),
                    int(work_images.shape[2]), int(work_images.shape[1]),
                    "lanczos", "disabled",
                ).movedim(1, -1)
            work_images = work_images.clone()
            work_images[:context_frames] = previous_refined.to(
                device=work_images.device, dtype=work_images.dtype
            )

        samples = vae.encode(work_images[..., :3])
        latent = {"samples": samples}
        if previous_refined is not None:
            prefix_steps = min(int(samples.shape[2]), (context_frames - 1) // 4 + 1)
            noise_mask = torch.ones(
                (1, 1, int(samples.shape[2]), int(samples.shape[-2]), int(samples.shape[-1])),
                device=samples.device, dtype=samples.dtype,
            )
            noise_mask[:, :, :prefix_steps] = 0.0
            latent["noise_mask"] = noise_mask

        latent_height, latent_width = int(samples.shape[-2]), int(samples.shape[-1])
        positive = _scail_resize_refine_conditioning(positive, latent_height, latent_width)
        negative = _scail_resize_refine_conditioning(negative, latent_height, latent_width)
        tile_count, overlap_pixels = _AD_H3_LATENT_TILE_PRESETS[latent_sample_tile]
        sample_model = (
            _scail_tiled_model(model, tile_count, overlap_pixels)
            if tile_count > 1 else model
        )
        refined_latent = nodes.common_ksampler(
            sample_model, int(seed), int(steps), float(cfg), sampler, scheduler,
            positive, negative, latent, denoise=float(refine_denoise),
        )[0]
        refined_images = nodes.VAEDecode().decode(vae, refined_latent)[0]

        output_length = int(state.get("output_length", 0))
        trim_frames = context_frames if previous_refined is not None else 0
        if output_length < 1 or int(refined_images.shape[0]) < trim_frames + output_length:
            raise ValueError("AD_scail2_generate_refine did not decode a complete video")
        if trim_frames:
            _ad_repair_boundary_flash(refined_images, trim_frames)
        overlap_images = refined_images[:trim_frames].detach().cpu() if trim_frames else None
        segment_images = refined_images[trim_frames:trim_frames + output_length]
        bridge_image = segment_images[-bridge_frames:].clone()
        segment_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=segment_images,
                audio=None,
                frame_rate=Fraction(str(fps)),
            )
        )
        refine_run_id = f"{run_id}_scail2_refine_{unique_id or 'node'}" if run_id is not None else None
        segment_video, merged_video = _ad_stage_video_outputs(
            segment_video,
            refine_run_id,
            stage_index,
            total,
            workflow_prompt,
            unique_id,
            "AD_scail2_generate_refine",
            overlap_images=overlap_images,
            merged_output_slot=2,
            color_match=True,
        )
        if refine_run_id is None:
            merged_video = segment_video
        return bridge_image, segment_video, merged_video
