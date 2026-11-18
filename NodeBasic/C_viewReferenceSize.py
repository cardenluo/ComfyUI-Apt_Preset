import math

import torch

import comfy.utils
from comfy_api.latest import InputImpl, Types
from comfy_execution.graph_utils import ExecutionBlocker
from nodes import MAX_RESOLUTION


ASPECT_RATIOS = {
    "1:1（正方形）": (1, 1),
    "1:2（超长竖屏）": (1, 2),
    "2:1（超宽横屏）": (2, 1),
    "2:3（竖版照片）": (2, 3),
    "3:2（横版照片）": (3, 2),
    "3:4（竖版标准）": (3, 4),
    "4:3（横版标准）": (4, 3),
    "4:5（人像）": (4, 5),
    "5:4（横版人像）": (5, 4),
    "5:7（竖版照片）": (5, 7),
    "7:5（横版照片）": (7, 5),
    "9:16（竖屏）": (9, 16),
    "16:9（横屏）": (16, 9),
    "10:16（竖版宽屏）": (10, 16),
    "16:10（横版宽屏）": (16, 10),
    "9:21（超宽屏竖版）": (9, 21),
    "21:9（超宽屏）": (21, 9),
}


def _target_size(source_width, source_height, size_mode, width, height, aspect_ratio, megapixels, resolution, size_multiple, target_size):
    if size_mode == "像素量 + 比例":
        ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
        scale = math.sqrt(float(megapixels) * 1024 * 1024 / (ratio_width * ratio_height))
        width, height = ratio_width * scale, ratio_height * scale
    elif size_mode == "分辨率":
        if resolution:
            ratio = source_width / source_height
            width = math.sqrt(resolution * resolution * ratio)
            height = math.sqrt(resolution * resolution / ratio)
        else:
            width, height = source_width, source_height
    elif size_mode in ("长边缩放", "短边缩放", "按宽缩放", "按高缩放"):
        if size_mode == "长边缩放":
            scale = target_size / max(source_width, source_height)
        elif size_mode == "短边缩放":
            scale = target_size / min(source_width, source_height)
        elif size_mode == "按宽缩放":
            scale = target_size / source_width
        else:
            scale = target_size / source_height
        width, height = source_width * scale, source_height * scale
    multiple = int(size_multiple)
    width = max(multiple, round(width / multiple) * multiple)
    height = max(multiple, round(height / multiple) * multiple)
    if size_mode in ("按宽缩放", "长边缩放", "短边缩放") and (size_mode == "按宽缩放" or (size_mode == "长边缩放") == (source_width >= source_height)):
        width = target_size
    elif size_mode in ("按高缩放", "长边缩放", "短边缩放"):
        height = target_size
    return width, height


def _resize_frames(frames, width, height, crop):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
        raise TypeError("IO_Media_Size expects IMAGE frames in [B, H, W, C] format")
    samples = frames[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


def _default_source_size(aspect_ratio):
    ratio_width, ratio_height = ASPECT_RATIOS[aspect_ratio]
    return ratio_width * 1024, ratio_height * 1024


class IO_Media_Size:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "size_mode": (["指定宽高", "像素量 + 比例", "分辨率", "长边缩放", "短边缩放", "按宽缩放", "按高缩放"], {"default": "指定宽高"}),
                "target_size": ("INT", {"default": 1024, "min": 2, "max": MAX_RESOLUTION, "step": 2}),
                "width": ("INT", {"default": 1344, "min": 2, "max": MAX_RESOLUTION, "step": 2}),
                "height": ("INT", {"default": 768, "min": 2, "max": MAX_RESOLUTION, "step": 2}),
                "aspect_ratio": (list(ASPECT_RATIOS), {"default": "16:9（横屏）"}),
                "megapixels": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.1}),
                "resolution": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 32,
                                      "tooltip": "以平方值为目标像素量，保持输入媒体的宽高比；0 保持原尺寸。"}),
                "size_multiple": ("INT", {"default": 16, "min": 1, "max": 1024, "step": 1,
                                        "tooltip": "输出宽高取到最接近的整数倍。"}),
            },
            "optional": {"image": ("IMAGE",), "video": ("VIDEO",)},
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "INT", "INT")
    RETURN_NAMES = ("image", "video", "width", "height")
    FUNCTION = "resize_reference"
    CATEGORY = "Apt_Preset/IO_Port"
    DESCRIPTION = "按指定宽高、像素量与比例，或输入媒体的原始比例计算目标尺寸。"

    def resize_reference(self, size_mode, width, height, aspect_ratio, megapixels, resolution, size_multiple, image=None, video=None, target_size=1024):
        image_output = ExecutionBlocker(None)
        video_output = ExecutionBlocker(None)
        output_width = ExecutionBlocker(None)
        output_height = ExecutionBlocker(None)
        size_preview = {}
        crop = "center" if size_mode in ("指定宽高", "像素量 + 比例") else "disabled"

        if image is None and video is None:
            source_width, source_height = _default_source_size(aspect_ratio)
            target_width, target_height = _target_size(
                source_width, source_height, size_mode, width, height, aspect_ratio,
                megapixels, resolution, size_multiple, target_size,
            )
            output_width, output_height = target_width, target_height
            size_preview["default"] = {"source_width": source_width, "source_height": source_height,
                                        "width": target_width, "height": target_height}

        if image is not None:
            source_height, source_width = int(image.shape[1]), int(image.shape[2])
            target_width, target_height = _target_size(source_width, source_height, size_mode, width, height, aspect_ratio, megapixels, resolution, size_multiple, target_size)
            image_output = _resize_frames(image, target_width, target_height, crop)
            output_width, output_height = target_width, target_height
            size_preview["image"] = {"source_width": source_width, "source_height": source_height,
                                     "width": target_width, "height": target_height}

        if video is not None:
            components = video.get_components()
            frames = components.images
            source_height, source_width = int(frames.shape[1]), int(frames.shape[2])
            target_width, target_height = _target_size(source_width, source_height, size_mode, width, height, aspect_ratio, megapixels, resolution, size_multiple, target_size)
            resized_frames = _resize_frames(frames, target_width, target_height, crop)
            if image is None:
                output_width, output_height = target_width, target_height
            bit_depth = video.get_bit_depth() if hasattr(video, "get_bit_depth") else 8
            video_output = InputImpl.VideoFromComponents(
                Types.VideoComponents(images=resized_frames, audio=components.audio,
                                      frame_rate=components.frame_rate, metadata=components.metadata),
                bit_depth=bit_depth,
            )
            size_preview["video"] = {"source_width": source_width, "source_height": source_height,
                                     "width": target_width, "height": target_height}

        return {"ui": {"reference_sizes": [size_preview]}, "result": (image_output, video_output, output_width, output_height)}
