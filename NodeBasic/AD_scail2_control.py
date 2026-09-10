from fractions import Fraction

import nodes
import torch
from comfy_api.latest import InputImpl, Types
from comfy_extras.nodes_post_processing import ColorTransfer
from comfy_extras.nodes_scail import WanSCAILToVideo

from ..main_unit import new_context
from .C_AD import _ad_stage_output_dir, _ad_stage_video_outputs


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
        segment = value[start_frame:start_frame + wanted_frames]
    elif _scail_file_video(value):
        source_fps = float(value.get_frame_rate())
        if source_fps <= 0:
            raise ValueError("AD_scail2_generate: source VIDEO has an invalid frame rate")
        base_start, base_duration = value.get_active_trim_window()
        duration = wanted_frames / source_fps
        if float(base_duration) > 0:
            remaining = float(base_duration) - start_frame / source_fps
            if remaining <= 0:
                return None
            duration = min(duration, remaining)
        window = InputImpl.VideoFromFile(
            value.get_stream_source(),
            start_time=float(base_start) + start_frame / source_fps,
            duration=max(0.0, duration),
        )
        segment = window.get_components().images[:wanted_frames]
    else:
        raise TypeError("AD_scail2_generate: pose inputs must be IMAGE tensors or file-backed VIDEO objects")
    if segment.shape[0] == 0:
        return None
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
                    "max": 4096,
                    "tooltip": "1 uses flow_stage_begin.total automatically. Values above 1 must match total.",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "clip_vision_name": (clip_vision_names, {"default": "clip_vision_h.safetensors"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "transfer_color": (["none", "reinhard_lab", "mkl_lab", "histogram"], {"default": "reinhard_lab"}),
                "pose_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Strength of the pose latent."}),
                "previous_frame_count": ("INT", {"default": 5, "min": 1, "max": nodes.MAX_RESOLUTION, "step": 4, "tooltip": "Tail frames of previous_frames to anchor. SCAIL-2 was trained with 5."}),
                "replacement_mode": ("BOOLEAN", {"default": False, "tooltip": "False: Animation Mode with a black pose mask background. True: Replacement Mode with a white pose mask background."}),
            },
            "optional": {
                "stage_info_data1": ("FLOW_STAGE_INFO",),
              
                "pose_video": ("IMAGE,VIDEO", {"tooltip": "Pose conditioning. Connect Load Video's VIDEO output for per-stage window decoding; IMAGE batches are sliced after upstream decoding."}),
                "reference_image": ("IMAGE", {"tooltip": "Primary reference followed by optional additional identity views."}),
                "pose_video_mask": ("IMAGE,VIDEO", {"tooltip": "SCAIL-2 colored mask video. File-backed VIDEO is decoded per stage; IMAGE batches are sliced after upstream decoding."}),
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
        if stage_info_data1 is None:
            if segment_count != 1:
                raise ValueError("AD_scail2_generate: connect flow_stage_begin.stage_info when segment_count is greater than 1")
            stage_index = 0
            total = 1
            run_id = None
        else:
            run_id = str(stage_info_data1.get("run_id") or "").strip()
            stage_index = int(stage_info_data1.get("stage_index", -1))
            total = int(stage_info_data1.get("total", 0))
            if not run_id or stage_index < 0 or total < 1 or stage_index >= total:
                raise ValueError("AD_scail2_generate: invalid stage_info_data1")
            if segment_count > 1 and total != segment_count:
                raise ValueError(
                    f"AD_scail2_generate: segment_count ({segment_count}) must match flow_stage_begin total ({total})"
                )

        if total > 1 and length < previous_frame_count:
            raise ValueError(
                "AD_scail2_generate: length must be at least previous_frame_count when using multiple segments"
            )
        source_lengths = [
            _scail_source_frame_count(value)
            for value in (pose_video, pose_video_mask)
            if value is not None
        ]
        source_frames = min(source_lengths) if source_lengths else length * total
        usable_frames = min(source_frames, length * total)
        output_start = stage_index * length
        output_length = min(length, usable_frames - output_start)
        if output_length <= 0:
            raise ValueError(
                f"AD_scail2_generate: stage {stage_index + 1} has no source frames; "
                "reduce flow_stage_begin.total"
            )
        context_frames = previous_frame_count if stage_index > 0 else 0
        conditioned_length = context_frames + output_length
        generation_length = ((conditioned_length - 1 + 3) // 4) * 4 + 1

        segment_start = max(0, output_start - context_frames)
        segment_source_frames = output_start + output_length - segment_start
        pose_video = _scail_segment_frames(
            pose_video, segment_start, segment_source_frames, generation_length
        )
        pose_video_mask = _scail_segment_frames(
            pose_video_mask, segment_start, segment_source_frames, generation_length
        )

        base_positive = nodes.CLIPTextEncode().encode(clip, positive)[0]
        base_negative = nodes.CLIPTextEncode().encode(clip, negative)[0]
        clip_vision_output = None
        if reference_image is not None:
            clip_vision = nodes.CLIPVisionLoader().load_clip(clip_vision_name)[0]
            clip_vision_output = nodes.CLIPVisionEncode().encode(clip_vision, reference_image, "none")[0]

        video_frame_offset = context_frames
        previous_frames = None
        previous_stage = None
        if stage_index > 0:
            previous_stage = stage_info_data1.get("stage_data_1", stage_info_data1.get("stage_data"))
            if not isinstance(previous_stage, torch.Tensor) or previous_stage.ndim != 4:
                raise TypeError("AD_scail2_generate: the previous flow stage data_1 must be an IMAGE tensor")
            if previous_stage.shape[0] < previous_frame_count:
                raise ValueError("AD_scail2_generate: previous stage has fewer frames than previous_frame_count")
            previous_frames = previous_stage[-previous_frame_count:].clone()

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
        latent = nodes.common_ksampler(
            model, segment_seed, steps, cfg, sampler, scheduler,
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
        return output_context, bridge_image, segment_video, merged_video
