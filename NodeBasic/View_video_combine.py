import os
import re

import folder_paths
import nodes
from comfy_api.latest import VideoCodec, VideoContainer, VideoFromFile

from .View_video_compare import _served_video


def _vhs_video_combine():
    node_class = nodes.NODE_CLASS_MAPPINGS.get("VHS_VideoCombine")
    if node_class is None:
        raise RuntimeError("View_video_combine requires ComfyUI-VideoHelperSuite")
    return node_class


def _video_info(video):
    width, height = video.get_dimensions()
    return {
        "width": width,
        "height": height,
        "frames": video.get_frame_count(),
    }


def _format_defaults(format_name):
    format_input = _vhs_video_combine().INPUT_TYPES()["required"]["format"]
    format_widgets = format_input[1].get("formats", {}).get(format_name, ())
    defaults = {}
    for name, widget_type, *options in format_widgets:
        config = options[0] if options and isinstance(options[0], dict) else {}
        if "default" in config:
            defaults[name] = config["default"]
        elif isinstance(widget_type, list):
            defaults[name] = widget_type[0]
        else:
            defaults[name] = {"BOOLEAN": False, "INT": 0, "FLOAT": 0, "STRING": ""}[widget_type]
    return defaults


class View_video_combine:
    @classmethod
    def INPUT_TYPES(cls):
        vhs_inputs = _vhs_video_combine().INPUT_TYPES()
        required = vhs_inputs["required"]
        return {
            "required": {
                "frame_rate": (required["frame_rate"][0], {**required["frame_rate"][1], "default": 24}),
                "filename_prefix": (required["filename_prefix"][0], {**required["filename_prefix"][1], "default": "video"}),
                "format": required["format"],
                "save_output": required["save_output"],
            },
            "optional": {
                "video": ("VIDEO",),
                "images": required["images"],
                "audio": vhs_inputs["optional"]["audio"],
            },
            "hidden": vhs_inputs["hidden"],
        }

    RETURN_TYPES = ("VIDEO", "VHS_FILENAMES")
    RETURN_NAMES = ("video", "filenames")
    FUNCTION = "combine"
    CATEGORY = "Apt_Preset/PreView"
    OUTPUT_NODE = True

    def combine(self, frame_rate, filename_prefix, format, save_output, video=None, images=None,
                audio=None, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        node_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(unique_id))

        if video is not None:
            width, height = video.get_dimensions()
            output_dir = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, output_dir, width, height
            )
            file = f"{filename}_{counter:05}.mp4"
            output_path = os.path.join(full_output_folder, file)
            video.save_to(output_path, format=VideoContainer.MP4, codec=VideoCodec.H264)
            output_video = VideoFromFile(output_path)
            filenames = (save_output, [output_path])
            preview = {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
            }
        else:
            if images is None:
                raise ValueError("View_video_combine needs either video or images")
            format_values = _format_defaults(format)
            format_values.update(kwargs)
            combined = _vhs_video_combine()().combine_video(
                images=images,
                audio=audio,
                frame_rate=frame_rate,
                loop_count=0,
                filename_prefix=filename_prefix,
                format=format,
                pingpong=False,
                save_output=save_output,
                prompt=prompt,
                extra_pnginfo=extra_pnginfo,
                unique_id=unique_id,
                **format_values,
            )
            filenames = combined["result"][0]
            if not filenames[1]:
                return {"ui": combined.get("ui", {}), "result": (None, filenames)}
            output_path = filenames[1][-1]
            output_video = VideoFromFile(output_path)
            preview = _served_video(output_path)
            if preview is None or os.path.splitext(output_path)[1].lower() not in {".mp4", ".webm"}:
                preview_file = f"apt_video_combine_{node_id}.mp4"
                preview_path = os.path.join(folder_paths.get_temp_directory(), preview_file)
                output_video.save_to(preview_path, format=VideoContainer.MP4, codec=VideoCodec.H264)
                preview = {"filename": preview_file, "subfolder": "", "type": "temp"}

        return {
            "ui": {"video_combine": [{"video": preview, **_video_info(output_video)}]},
            "result": (output_video, filenames),
        }
