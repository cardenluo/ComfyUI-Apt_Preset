import os
import re
import uuid

import folder_paths
from comfy_api.latest import VideoCodec, VideoContainer, VideoFromFile
from nodes import PreviewImage


def _served_video(path):
    resolved = os.path.realpath(path)
    roots = (
        ("input", folder_paths.get_input_directory()),
        ("output", folder_paths.get_output_directory()),
        ("temp", folder_paths.get_temp_directory()),
    )
    for folder_type, root in roots:
        root = os.path.realpath(root)
        try:
            is_served = os.path.normcase(os.path.commonpath((resolved, root))) == os.path.normcase(root)
        except ValueError:
            is_served = False
        if not is_served:
            continue
        relative = os.path.relpath(resolved, root)
        subfolder, filename = os.path.split(relative)
        return {
            "filename": filename,
            "subfolder": subfolder.replace(os.sep, "/"),
            "type": folder_type,
        }
    return None


def _preview_video(video, node_id, side):
    if isinstance(video, VideoFromFile):
        source = video.get_stream_source()
        start_time, duration = video.get_active_trim_window()
        if isinstance(source, (str, os.PathLike)) and start_time == 0 and duration == 0:
            source = os.fspath(source)
            if os.path.splitext(source)[1].lower() in {".mp4", ".webm"}:
                served = _served_video(source)
                if served is not None:
                    return served

    filename = f"apt_video_compare_{node_id}_{side}_{uuid.uuid4().hex[:8]}.mp4"
    video.save_to(
        os.path.join(folder_paths.get_temp_directory(), filename),
        format=VideoContainer.MP4,
        codec=VideoCodec.H264,
    )
    return {"filename": filename, "subfolder": "", "type": "temp"}


class View_video_compare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("VIDEO",),
                "video_b": ("VIDEO",),
                "direction": (["左右", "上下"],),
                "split_position": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "autoplay": ("BOOLEAN", {"default": False}),
                "loop": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True

    def compare(self, video_a, video_b, direction, split_position, autoplay, loop, unique_id):
        node_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(unique_id))
        return {
            "ui": {
                "video_compare": [{
                    "video_a": _preview_video(video_a, node_id, "a"),
                    "video_b": _preview_video(video_b, node_id, "b"),
                    "direction": direction,
                    "split_position": split_position,
                    "autoplay": autoplay,
                    "loop": loop,
                }]
            }
        }


class View_image_compare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
                "direction": (["左右", "上下"],),
                "split_position": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True

    def compare(self, image_a, image_b, direction, split_position, opacity):
        preview = PreviewImage()
        image_a_info = preview.save_images(image_a[:1])["ui"]["images"][0]
        image_b_info = preview.save_images(image_b[:1])["ui"]["images"][0]
        return {
            "ui": {
                "image_compare": [{
                    "image_a": image_a_info,
                    "image_b": image_b_info,
                    "direction": direction,
                    "split_position": split_position,
                    "opacity": opacity,
                }]
            }
        }
