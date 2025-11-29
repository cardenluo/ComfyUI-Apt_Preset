import os
import torch
import numpy as np
from PIL import Image
from server import PromptServer
import folder_paths
import threading
import sys
import base64
from io import BytesIO
import queue
import time
import ctypes
import functools
import comfy.model_management as model_management
from aiohttp import web



parent_module = sys.modules.get('.'.join(__name__.split('.')[:-2]))

active_canvas = None
@PromptServer.instance.routes.post("/preset_canvas")
async def save_canvas(request):
    global active_canvas
    try:
        parent_module = sys.modules.get('.'.join(__name__.split('.')[:-2]))
        if parent_module.active_canvas is None:
            return web.Response(status=400, text="No active canvas instance")
        data = await request.json()
        parent_module.active_canvas.edited_data = data
        print("Edited data saved")
        parent_module.active_canvas.save_event.set()
        print("Save event set")
        return web.json_response({"status": "success"})
    except Exception as e:
        return web.Response(status=500, text=str(e))
def interruptible(timeout=60.0):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result_queue = queue.Queue()
            def worker():
                try:
                    result = func(self, *args, **kwargs)
                    result_queue.put(("success", result))
                except Exception as e:
                    result_queue.put(("error", e))
            thread = threading.Thread(target=worker)
            thread.daemon = True
            thread.start()
            def terminate_thread():
                if not thread.is_alive():
                    return
                exc = ctypes.py_object(SystemExit)
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
                    ctypes.c_long(thread.ident), exc)
                if res > 1:
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(
                        ctypes.c_long(thread.ident), None)
            try:
                start_time = time.time()
                while thread.is_alive():
                    try:
                        status, result = result_queue.get(timeout=0.1)
                        if status == "success":
                            return result
                        else:
                            raise result
                    except queue.Empty:
                        if time.time() - start_time > timeout:
                            terminate_thread()
                            return None
                        model_management.throw_exception_if_processing_interrupted()
            finally:
                terminate_thread()
        return wrapper
    return decorator


class lay_imgCanvasNode:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "web_canvas")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.save_event = threading.Event()
        self.edited_data = None
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "back_image": ("IMAGE",),
                "fore_image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "fore_mask": ("MASK",),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }
    @interruptible(timeout=60.0)
    def process(self, back_image,  prompt,extra_pnginfo,fore_image, seed, fore_mask=None):
        try:
            try:
                nodes = extra_pnginfo.get("workflow").get("nodes")
                webpro = [node for node in nodes if node.get("type") == "lay_imgCanvas" ]
                if webpro and len(webpro) > 0:
                    widgets_values = webpro[0].get("widgets_values", [])
                    if len(widgets_values) >= 2:
                        window_seed = widgets_values[1]
                    else:
                        window_seed = seed
                else:
                    window_seed = seed
            except Exception as e:
                window_seed = seed
                print(f"Warning: Could not get window_seed, using seed instead: {str(e)}")
            if len(back_image.shape) != 4 or back_image.shape[-1] != 3:
                raise ValueError(f"Expected back_image shape (B,H,W,3), got {back_image.shape}")
            if len(fore_image.shape) != 4 or fore_image.shape[-1] != 3:
                raise ValueError(f"Expected fore_image shape (B,H,W,3), got {fore_image.shape}")
            back_image = back_image.float() if back_image.dtype != torch.float32 else back_image
            fore_image = fore_image.float() if fore_image.dtype != torch.float32 else fore_image
            if fore_mask is None:
                fore_mask = torch.ones((1, fore_image.shape[1], fore_image.shape[2]), 
                                     dtype=torch.float32, 
                                     device=fore_image.device)
            else:
                if len(fore_mask.shape) != 3:
                    raise ValueError(f"Expected fore_mask shape (B,H,W), got {fore_mask.shape}")
                fore_mask = fore_mask.float() if fore_mask.dtype != torch.float32 else fore_mask
            canvas_width = back_image.shape[2]
            canvas_height = back_image.shape[1]
            self.save_event.clear()
            self.edited_data = None
            parent_module.active_canvas = self
            back_filename = f"back_{seed}.png"
            fore_filename = f"fore_{seed}.png"
            back_path = os.path.join(self.output_dir, back_filename)
            fore_path = os.path.join(self.output_dir, fore_filename)
            for file in os.listdir(self.output_dir):
                os.remove(os.path.join(self.output_dir, file))
            back_img = Image.fromarray(np.clip(255. * back_image[0].cpu().numpy(), 0, 255).astype(np.uint8))
            back_img.save(back_path, 'PNG')
            fore_img = Image.fromarray(np.clip(255. * fore_image[0].cpu().numpy(), 0, 255).astype(np.uint8))
            fore_img = fore_img.convert('RGBA')
            alpha = Image.fromarray(np.clip(255. * fore_mask[0].cpu().numpy(), 0, 255).astype(np.uint8))
            mask_min = fore_mask.min().item()
            mask_max = fore_mask.max().item()
            if not (0 <= mask_min <= 1 and 0 <= mask_max <= 1):
                pass
            fore_img.putalpha(alpha)
            fore_img.save(fore_path, 'PNG')
            PromptServer.instance.send_sync("show_canvas", {
                "back_image": f"/view?filename={back_filename}&subfolder=web_canvas",
                "fore_image": f"/view?filename={fore_filename}&subfolder=web_canvas",
                "seed": seed,
                "canvas_width": canvas_width,
                "canvas_height": canvas_height,
                "mask_value": fore_mask[0].mean().item(),
                "window_id": window_seed
            })
            if not self.save_event.wait(timeout=300):
                return (back_image, fore_mask)
            if not self.edited_data:
                return (back_image, fore_mask)
            if not self.edited_data.get('confirmed', False):
                return (back_image, fore_mask)
            if 'image' in self.edited_data and 'mask' in self.edited_data:
                image_data = self.edited_data['image'].split(',')[1]
                mask_data = self.edited_data['mask'].split(',')[1]
                image_bytes = base64.b64decode(image_data)
                mask_bytes = base64.b64decode(mask_data)
                final_image = Image.open(BytesIO(image_bytes))
                mask_image = Image.open(BytesIO(mask_bytes))
                if mask_image.mode != 'L':
                    mask_image = mask_image.convert('L')
                mask_array = np.array(mask_image)
                mask_tensor = torch.from_numpy(mask_array).float() / 255.0 
                mask_tensor = mask_tensor.unsqueeze(0)
                transform = self.edited_data.get('transform', {})
                x = float(transform.get('x', 0))
                y = float(transform.get('y', 0))
                scale = float(transform.get('scale', 1.0))
                image_array = np.array(final_image)
                image_tensor = torch.from_numpy(image_array[:,:,:3]).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0)
                return (image_tensor, mask_tensor)
            return (back_image, fore_mask)
        except Exception as e:
            print(f"Error in process: {str(e)}")
            return (back_image, fore_mask)
        finally:
            parent_module.active_canvas = None
            self.save_event.clear()
    def web_canvas(self, image, x: float, y: float, scale: float = 1.0, rotation: float = 0.0):
        with torch.no_grad():
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            input_height, input_width = image.shape[1:3]
            canvas_size = (input_width, input_height)
            img = Image.fromarray(np.clip(255. * image[0].cpu().numpy(), 0, 255).astype(np.uint8))
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            canvas = Image.new('RGB', canvas_size, (255, 255, 255))
            mask = Image.new('L', canvas_size, 0)
            if scale != 1.0:
                new_size = (int(img.width * scale), int(img.height * scale))
                img = img.resize(new_size, Image.LANCZOS)
            if rotation != 0:
                img = img.rotate(-rotation, expand=True, resample=Image.BICUBIC)
            paste_x = int(canvas_size[0]//2 + x - img.width//2)
            paste_y = int(canvas_size[1]//2 + y - img.height//2)
            alpha = img.split()[3]
            temp_mask = Image.new('L', img.size, 0)
            temp_mask.paste(1, mask=alpha)
            canvas.paste(img, (paste_x, paste_y), alpha)
            mask.paste(temp_mask, (paste_x, paste_y))
            canvas_tensor = torch.from_numpy(np.array(canvas)).float() / 255.0
            mask_tensor = torch.from_numpy(np.array(mask)).float()
            canvas_tensor = canvas_tensor.unsqueeze(0)
            mask_tensor = mask_tensor.unsqueeze(0)
            return canvas_tensor, mask_tensor
