## ComfyUI/custom_nodes/CCNotes/py/utils.py
import torch
import numpy as np
import folder_paths
import os
import random
import server
import threading
from aiohttp import web
from nodes import SaveImage





import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import Tuple, List, NamedTuple, Any



PAUSE_REQUESTS = {}
MAX_FLOW_PORTS = 10






def generate_preview_images(input_values: List[Any]) -> List[torch.Tensor]:
    preview_images_list = []
    i = 0
    while i < len(input_values):
        val = input_values[i]
        is_img = isinstance(val, torch.Tensor) and val.ndim == 4 
        is_mask = isinstance(val, torch.Tensor) and val.ndim in (2, 3) 
        if is_img:
            next_val = input_values[i+1] if i + 1 < len(input_values) else None
            next_is_mask = isinstance(next_val, torch.Tensor) and next_val.ndim in (2, 3)
            if next_is_mask:
                try:
                    img_tensor = val
                    mask_tensor = next_val
                    if mask_tensor.ndim == 2:
                        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1)
                    elif mask_tensor.ndim == 3:
                        mask_tensor = mask_tensor.unsqueeze(-1)
                    if mask_tensor.shape[-3:-1] != img_tensor.shape[-3:-1]:
                        mask_for_resize = mask_tensor.permute(0, 3, 1, 2)
                        mask_resized = F.interpolate(
                            mask_for_resize, 
                            size=(img_tensor.shape[1], img_tensor.shape[2]), 
                            mode="nearest"
                        )
                        mask_tensor = mask_resized.permute(0, 2, 3, 1)
                    mask_opacity = 0.5
                    red_overlay = torch.zeros_like(img_tensor)
                    red_overlay[:, :, :, 0] = 1.0 # R channel = 1
                    alpha = mask_tensor * mask_opacity
                    if alpha.shape[0] != img_tensor.shape[0]:
                        alpha = alpha.repeat(img_tensor.shape[0], 1, 1, 1)
                    composite = img_tensor * (1 - alpha) + red_overlay * alpha
                    preview_images_list.append(composite)
                except Exception as e:
                    handle_error(e, "Overlay utility error")
                    preview_images_list.append(val)
            else:
                preview_images_list.append(val)
        elif is_mask:
            prev_val = input_values[i-1] if i > 0 else None
            prev_is_img = isinstance(prev_val, torch.Tensor) and prev_val.ndim == 4
            
            if not prev_is_img:
                mask_preview = val
                if mask_preview.ndim == 2:
                    mask_preview = mask_preview.unsqueeze(0) # [B, H, W]
                mask_preview = mask_preview.reshape((-1, 1, mask_preview.shape[-2], mask_preview.shape[-1]))
                mask_preview = mask_preview.movedim(1, -1).expand(-1, -1, -1, 3)
                preview_images_list.append(mask_preview)
        i += 1
    return preview_images_list

def flatten_input_values(input_values: List[Any]) -> List[Any]:
    flat_values = []
    for val in input_values:
        if isinstance(val, list):
            flat_values.extend(val)
        else:
            flat_values.append(val)
    return flat_values

def generate_text_previews(input_values: List[Any], max_length: int = 500) -> List[str]:
    text_previews = []
    for port_vals in input_values:
        if not isinstance(port_vals, list):
            port_vals = [port_vals]
        for val in port_vals:
            if isinstance(val, str):
                text_preview = val[:max_length] if len(val) > max_length else val
                if len(val) > max_length:
                    text_preview += "..."
                text_previews.append(text_preview)
            elif isinstance(val, bool):
                text_previews.append(f"Boolean: {val}")
            elif isinstance(val, (int, float)):
                text_previews.append(f"Number: {val}")
            elif isinstance(val, list):
                text_previews.append(f"List: {len(val)} items")
    return text_previews

def save_images_for_preview(save_image_instance, images_list: List[torch.Tensor], 
                            filename_prefix: str = "CCNotes_preview",
                            collect_filenames: bool = False) -> Tuple[List[dict], List[Tuple[str, str]]]:
    all_saved_images = []
    saved_filenames = []
    
    for img_tensor in images_list:
        res = save_image_instance.save_images(img_tensor, filename_prefix=filename_prefix)
        if 'ui' in res and 'images' in res['ui']:
            for img_data in res['ui']['images']:
                img_data['url'] = f"{img_data.get('subfolder', '')}/{img_data['filename']}"
                img_data['thumbnail'] = f"/api/view?type={img_data['type']}&filename={img_data['filename']}&subfolder={img_data.get('subfolder', '')}"
                all_saved_images.append(img_data)
                if collect_filenames:
                    saved_filenames.append((img_data.get('subfolder', ''), img_data['filename']))
    return all_saved_images, saved_filenames

def send_preview_event(unique_id_str: str, frontend_data: dict, node_type: str = "preview", 
                       action: str = None, extra_data: dict = None):
    import server
    event_data = {
        "node_id": unique_id_str,
        "node_type": node_type,
        "data": frontend_data
    }
    if action is not None:
        event_data["action"] = action
    if extra_data is not None:
        event_data.update(extra_data)
    server.PromptServer.instance.send_sync("ccnotes.node_event", event_data)



class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

def handle_error(e: Exception, msg: str = "Operation failed"):
    raise type(e)(f"[CCNotes] {msg}: {e}").with_traceback(e.__traceback__)

def handle_error_safe(e: Exception, msg: str = "Operation failed", port_count: int = 1):
    print(f"[CCNotes] {msg}: {e}")
    return tuple([[""] for _ in range(port_count)])









@server.PromptServer.instance.routes.post("/ccnotes/resume_pause")
async def resume_pause(request):
    post = await request.json()
    node_id = str(post.get("node_id"))
    action = post.get("action", "continue")
    edited_text = post.get("edited_text", None)

    images = post.get("images", None)
    if node_id in PAUSE_REQUESTS:
        PAUSE_REQUESTS[node_id]["action"] = action
        if edited_text is not None:
            PAUSE_REQUESTS[node_id]["edited_text"] = edited_text
        if images is not None:
            PAUSE_REQUESTS[node_id]["images"] = images
        PAUSE_REQUESTS[node_id]["event"].set()
        return web.json_response({"status": "success"})
    return web.json_response({"status": "ignored"}, status=200)



class AnyPreview(SaveImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 4

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "preview_text": ("STRING", {"default": "Text Preview", "multiline": True, "forceInput": False}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "unique_id": "UNIQUE_ID"},
        }

    NAME="AnyPreview"
    RETURN_TYPES = (any_type,) * MAX_FLOW_PORTS
    RETURN_NAMES = tuple(f"output_{i}" for i in range(1, MAX_FLOW_PORTS + 1))
    FUNCTION = "process_preview"
    OUTPUT_NODE = True
    CATEGORY = "Apt_Preset/PreView"
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,) * MAX_FLOW_PORTS

    @classmethod
    def IS_CHANGED(s, **kwargs):
        return 0

    def process_preview(self, prompt=None, extra_pnginfo=None, unique_id=None, **kwargs):
        unique_id_str = str(unique_id[0] if isinstance(unique_id, list) and unique_id else unique_id)

        input_keys = sorted([k for k in kwargs.keys() if k.startswith("input_")], key=lambda x: int(x.split("_")[1]))
        input_values = [kwargs[k] for k in input_keys]
        actual_returns = list(input_values)

        # Flatten input values using utility function
        flat_input_values = flatten_input_values(input_values)

        # Generate preview images and save using utility function
        preview_images_list = generate_preview_images(flat_input_values)
        frontend_data = {}

        if preview_images_list:
            all_saved_images, _ = save_images_for_preview(self, preview_images_list, "CCNotes_preview")
            frontend_data["images"] = all_saved_images

        # Generate text previews using utility function
        text_previews = generate_text_previews(input_values)
        if text_previews:
            frontend_data["text"] = text_previews

        padding_count = MAX_FLOW_PORTS - len(actual_returns)
        if padding_count > 0:
            result_tuple = tuple(actual_returns) + ([],) * padding_count
        else:
            result_tuple = tuple(actual_returns[:MAX_FLOW_PORTS])

        # Send preview event using utility function
        send_preview_event(unique_id_str, frontend_data, "preview")

        return {"ui": frontend_data, "result": result_tuple}













