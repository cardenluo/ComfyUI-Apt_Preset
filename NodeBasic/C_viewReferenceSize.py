import math

import torch

import comfy.utils
from comfy_api.latest import InputImpl, Types
from comfy_execution.graph_utils import ExecutionBlocker
from nodes import MAX_RESOLUTION


CANVAS_MULTIPLE = 32
BASE_SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
REF_IMAGE_SHORT_EDGE = 2048


def _round_to_canvas(value):
    return max(
        CANVAS_MULTIPLE,
        round(value / CANVAS_MULTIPLE) * CANVAS_MULTIPLE,
    )


def _reference_image_size(width, height, generation_width, generation_height, mode):
    if mode == "minimax_match":
        scale = min(
            1.0,
            math.sqrt((generation_width * generation_height) / (width * height)),
        )
    elif mode == "minimax_max":
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(width, height))
    else:
        raise ValueError(f"Unsupported reference size mode: {mode}")
    return _round_to_canvas(width * scale), _round_to_canvas(height * scale)


def _reference_video_size(width, height):
    ratio = width / height
    if ratio >= 1.0:
        nominal_width, nominal_height = BASE_SHORT_EDGE * ratio, BASE_SHORT_EDGE
    else:
        nominal_width, nominal_height = BASE_SHORT_EDGE, BASE_SHORT_EDGE / ratio
    if nominal_width * nominal_height > MAX_PIXELS:
        scale = math.sqrt(MAX_PIXELS / (nominal_width * nominal_height))
        nominal_width *= scale
        nominal_height *= scale

    target_width = _round_to_canvas(nominal_width)
    target_height = _round_to_canvas(nominal_height)
    if width * height < target_width * target_height:
        target_width = _round_to_canvas(width)
        target_height = _round_to_canvas(height)
    return target_width, target_height


def _resize_frames(frames, width, height):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("view_Reference_Size expects IMAGE frames in [B, H, W, C] format")
    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
    return samples.movedim(1, -1)


def _resize_frames_center(frames, width, height, method):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("view_Reference_Size expects IMAGE frames in [B, H, W, C] format")
    samples = comfy.utils.common_upscale(
        frames[..., :3].movedim(-1, 1), width, height, method, "center"
    )
    return samples.movedim(1, -1)


def _center_crop_to_multiple(frames, multiple):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("view_Reference_Size expects IMAGE frames in [B, H, W, C] format")
    height, width = int(frames.shape[1]), int(frames.shape[2])
    target_height = (height // multiple) * multiple
    target_width = (width // multiple) * multiple
    if target_height < multiple or target_width < multiple:
        raise ValueError(
            f"view_Reference_Size needs image dimensions of at least {multiple} pixels "
            f"for this mode, got {width} x {height}"
        )
    offset_y = (height - target_height) // 2
    offset_x = (width - target_width) // 2
    return frames[
        :, offset_y:offset_y + target_height, offset_x:offset_x + target_width, :3
    ]


def _qwen_edit_resize(frames, generation_width, generation_height):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("view_Reference_Size expects IMAGE frames in [B, H, W, C] format")

    target_height = max(int(generation_height), 64)
    target_width = max(int(generation_width), 64)
    original_height = max(int(frames.shape[1]), 64)
    original_width = max(int(frames.shape[2]), 64)
    scale = max(target_width / original_width, target_height / original_height)
    scaled_width = max(int(original_width * scale), target_width)
    scaled_height = max(int(original_height * scale), target_height)
    samples = comfy.utils.common_upscale(
        frames[..., :3].movedim(-1, 1),
        scaled_width,
        scaled_height,
        "bicubic",
        "disabled",
    )
    offset_x = (scaled_width - target_width) // 2
    offset_y = (scaled_height - target_height) // 2
    samples = samples[
        :, :, offset_y:offset_y + target_height, offset_x:offset_x + target_width
    ]

    vae_width = max((target_width // 8) * 8, 64)
    vae_height = max((target_height // 8) * 8, 64)
    if vae_width != target_width or vae_height != target_height:
        samples = comfy.utils.common_upscale(
            samples, vae_width, vae_height, "bicubic", "disabled"
        )
    return samples.movedim(1, -1)


def _resize_for_mode(frames, mode, generation_width, generation_height, is_video=False):
    source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
    if mode in {"minimax_match", "minimax_max"}:
        if is_video:
            target_width, target_height = _reference_video_size(source_width, source_height)
        else:
            target_width, target_height = _reference_image_size(
                source_width,
                source_height,
                int(generation_width),
                int(generation_height),
                mode,
            )
        return _resize_frames(frames, target_width, target_height)
    if mode == "klein_image":
        return _center_crop_to_multiple(frames, 16)
    if mode == "qwenEdit_image":
        return _qwen_edit_resize(frames, generation_width, generation_height)
    if mode == "scail2":
        if is_video:
            return _resize_frames_center(
                frames,
                max(1, int(generation_width) // 2),
                max(1, int(generation_height) // 2),
                "area",
            )
        return _resize_frames_center(
            frames,
            int(generation_width),
            int(generation_height),
            "bicubic",
        )
    raise ValueError(f"Unsupported reference size mode: {mode}")


class view_Reference_Size:
    """Resize image/video references as MiniMax H3, Flux2 Klein, or Qwen Edit consumes them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_type": ([
                    "minimax_match",
                    "minimax_max",
                    "klein_image",
                    "qwenEdit_image",
                    "scail2",
                ], {
                    "default": "minimax_match",
                    "tooltip": (
                        "图片：match 按生成画布像素面积等比缩小，max 按 2048 短边等比缩小；"
                        "MiniMax 视频使用 H3 参考视频画布；Klein 居中裁到 16 倍数；"
                        "Qwen Edit 按生成宽高中心覆盖裁剪并对齐到 8；"
                        "SCAIL-2 图片使用完整生成尺寸，视频使用一半生成尺寸。"
                    ),
                }),
                "generation_width": ("INT", {
                    "default": 1344,
                    "min": 32,
                    "max": MAX_RESOLUTION,
                    "step": 4,
                }),
                "generation_height": ("INT", {
                    "default": 768,
                    "min": 32,
                    "max": MAX_RESOLUTION,
                    "step": 4,
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO")
    RETURN_NAMES = ("image", "video")
    FUNCTION = "resize_reference"
    CATEGORY = "Apt_Preset/PreView"
    DESCRIPTION = (
        "按 MiniMax H3、Flux2 Klein、Qwen Edit 或 SCAIL-2 的实际参考管线调整媒体尺寸，"
        "并在节点底部显示真实输出宽高。"
    )

    def resize_reference(
        self,
        input_type,
        generation_width,
        generation_height,
        image=None,
        video=None,
    ):
        image_output = ExecutionBlocker(None)
        video_output = ExecutionBlocker(None)
        size_preview = {}

        if image is not None:
            source_height, source_width = int(image.shape[1]), int(image.shape[2])
            image_output = _resize_for_mode(
                image,
                input_type,
                int(generation_width),
                int(generation_height),
            )
            target_height, target_width = int(image_output.shape[1]), int(image_output.shape[2])
            size_preview["image"] = {
                "source_width": source_width,
                "source_height": source_height,
                "width": target_width,
                "height": target_height,
            }

        if video is not None:
            components = video.get_components()
            frames = components.images
            source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
            resized_frames = _resize_for_mode(
                frames,
                input_type,
                int(generation_width),
                int(generation_height),
                is_video=True,
            )
            target_height, target_width = (
                int(resized_frames.shape[1]),
                int(resized_frames.shape[2]),
            )
            bit_depth = video.get_bit_depth() if hasattr(video, "get_bit_depth") else 8
            video_output = InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=resized_frames,
                    audio=components.audio,
                    frame_rate=components.frame_rate,
                    metadata=components.metadata,
                ),
                bit_depth=bit_depth,
            )
            size_preview["video"] = {
                "source_width": source_width,
                "source_height": source_height,
                "width": target_width,
                "height": target_height,
            }

        return {
            "ui": {"reference_sizes": [size_preview]},
            "result": (image_output, video_output),
        }
