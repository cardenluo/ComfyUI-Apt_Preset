
import torch
import comfy.nested_tensor
import comfy.model_management
import numpy as np
from typing import Any
import math
from PIL import Image, ImageDraw
from scipy.signal import savgol_filter
import random
import torch.nn.functional as F
from io import BytesIO
import hashlib
import re
import json
import collections.abc
import folder_paths
from ..main_unit import *
from .C_mask import mask_sam_detctor



#region-----------------收纳--------------------

try:
    from pydub import AudioSegment
    REMOVER_AVAILABLE = True  
except ImportError:
    AudioSegment = None
    REMOVER_AVAILABLE = False  


try:
    from scipy.fft import fft
    REMOVER_AVAILABLE = True  
except ImportError:
    fft = None
    REMOVER_AVAILABLE = False  


try:
    import pandas as pd
    REMOVER_AVAILABLE = True  
except ImportError:
    pd = None
    REMOVER_AVAILABLE = False  



try:
    import matplotlib.pyplot as plt
    REMOVER_AVAILABLE = True  
except ImportError:
    plt = None
    REMOVER_AVAILABLE = False  











class AD_ImageExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def execute(self, image, size, method):
        orig_size = image.shape[0]

        if orig_size == size:
            return (image,)

        if size <= 1:
            return (image[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(image.shape)[1:], dtype=image.dtype, device=image.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = image[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = image[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = image.repeat([math.ceil(size / image.shape[0])] + [1] * (len(image.shape) - 1))[:size]
        elif 'first' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat([image[:1].repeat(size-image.shape[0], 1, 1, 1), image], dim=0)
        elif 'last' in method:
            if size < image.shape[0]:
                out = image[:size]
            else:
                out = torch.cat((image, image[-1:].repeat((size-image.shape[0], 1, 1, 1))), dim=0)

        return (out,)


class AD_MaskExpandBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "size": ("INT", { "default": 16, "min": 1, "step": 1, }),
                "method": (["expand", "repeat all", "repeat first", "repeat last"],)
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def execute(self, mask, size, method):
        orig_size = mask.shape[0]

        if orig_size == size:
            return (mask,)

        if size <= 1:
            return (mask[:size],)

        if 'expand' in method:
            out = torch.empty([size] + list(mask.shape)[1:], dtype=mask.dtype, device=mask.device)
            if size < orig_size:
                scale = (orig_size - 1) / (size - 1)
                for i in range(size):
                    out[i] = mask[min(round(i * scale), orig_size - 1)]
            else:
                scale = orig_size / size
                for i in range(size):
                    out[i] = mask[min(math.floor((i + 0.5) * scale), orig_size - 1)]
        elif 'all' in method:
            out = mask.repeat([math.ceil(size / mask.shape[0])] + [1] * (len(mask.shape) - 1))[:size]
        elif 'first' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat([mask[:1].repeat(size-mask.shape[0], 1, 1), mask], dim=0)
        elif 'last' in method:
            if size < mask.shape[0]:
                out = mask[:size]
            else:
                out = torch.cat((mask, mask[-1:].repeat((size-mask.shape[0], 1, 1))), dim=0)

        return (out,)


class AD_frame_replace:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "start_index": ("INT", {"default": 0,"min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1,"min": 1, "max": 4096, "step": 1}),
                # 添加节点工作类型选择
                "type": (["choose frame output", "replace  frame and  output all"], {"default": "choose frame output"}),
            },
            "optional": {
                "images": ("IMAGE",),
                "masks": ("MASK",),
                "replace_img": ("IMAGE",),
                "replace_mask": ("MASK",),
            }
        } 
    
    RETURN_TYPES = ("IMAGE", "MASK", )
    FUNCTION = "imagesfrombatch"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def imagesfrombatch(self, start_index, num_frames, type, images=None, masks=None, replace_img=None, replace_mask=None):
        chosen_images = None
        chosen_masks = None

        # Process images if provided
        if images is not None:
            if start_index == -1:
                start_index = max(0, len(images) - num_frames)
            if start_index < 0 or start_index >= len(images):
                raise ValueError("Start index is out of range")
            end_index = min(start_index + num_frames, len(images))

            if replace_img is not None:
                # 尺寸处理
                processed_input_img = []
                for img in replace_img:
                    if img.shape != images[0].shape:
                        # 中心对齐裁切逻辑
                        img_height, img_width = img.shape[0], img.shape[1]
                        target_height, target_width = images[0].shape[0], images[0].shape[1]
                        y_start = (img_height - target_height) // 2
                        x_start = (img_width - target_width) // 2
                        cropped_img = img[y_start:y_start + target_height, x_start:x_start + target_width]
                        processed_input_img.append(cropped_img)
                    else:
                        processed_input_img.append(img)
                processed_input_img = torch.stack(processed_input_img)

                # 补齐或舍弃图像
                if len(processed_input_img) < num_frames:
                    last_img = processed_input_img[-1:]
                    repeat_times = num_frames - len(processed_input_img)
                    padded_img = last_img.repeat(repeat_times, 1, 1, 1)
                    processed_input_img = torch.cat([processed_input_img, padded_img], dim=0)
                elif len(processed_input_img) > num_frames:
                    processed_input_img = processed_input_img[:num_frames]

                # 替换对应位置的图像
                images = torch.cat([images[:start_index], processed_input_img, images[end_index:]], dim=0)

            if type == "choose frame output":
                chosen_images = images[start_index:end_index]
            elif type == "replace  frame and  output all":
                chosen_images = images

        # Process masks if provided
        if masks is not None:
            if start_index == -1:
                start_index = max(0, len(masks) - num_frames)
            if start_index < 0 or start_index >= len(masks):
                raise ValueError("Start index is out of range for masks")
            end_index = min(start_index + num_frames, len(masks))

            if replace_mask is not None:
                if len(replace_mask) < num_frames:
                    last_mask = replace_mask[-1:]
                    repeat_times = num_frames - len(replace_mask)
                    padded_mask = last_mask.repeat(repeat_times, 1, 1)
                    replace_mask = torch.cat([replace_mask, padded_mask], dim=0)
                elif len(replace_mask) > num_frames:
                    replace_mask = replace_mask[:num_frames]
                masks = torch.cat([masks[:start_index], replace_mask, masks[end_index:]], dim=0)

            if type == "choose frame output":
                chosen_masks = masks[start_index:end_index]
            elif type == "replace  frame and  output all":
                chosen_masks = masks

        return (chosen_images, chosen_masks,)

#endregion-----------------收纳--------------------



#region---------------------Audio----def----------------------



class AudioData:
    def __init__(self, audio_file) -> None:
        
        # Extract the sample rate
        sample_rate = audio_file.frame_rate

        # Get the number of audio channels
        num_channels = audio_file.channels

        # Extract the audio data as a NumPy array
        audio_data = np.array(audio_file.get_array_of_samples())
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.num_channels = num_channels
    
    def get_channel_audio_data(self, channel: int):
        if channel < 0 or channel >= self.num_channels:
            raise IndexError(f"Channel '{channel}' out of range. total channels is '{self.num_channels}'.")
        return self.audio_data[channel::self.num_channels]
    
    def get_channel_fft(self, channel: int):
        audio_data = self.get_channel_audio_data(channel)
        return fft(audio_data)


class AudioFFTData:
    def __init__(self, audio_data, sample_rate) -> None:

        self.fft = fft(audio_data)
        self.length = len(self.fft)
        self.frequency_bins = np.fft.fftfreq(self.length, 1 / sample_rate)
    
    def get_max_amplitude(self):
        return np.max(np.abs(self.fft))
    
    def get_normalized_fft(self) -> float:
        max_amplitude = self.get_max_amplitude()
        return np.abs(self.fft) / max_amplitude

    def get_indices_for_frequency_bands(self, lower_band_range: int, upper_band_range: int):
        return np.where((self.frequency_bins >= lower_band_range) & (self.frequency_bins < upper_band_range))

    def __len__(self):
        return self.length


defaultText="""Rabbit
Dog
Cat
One prompt per line
"""


#endregion-------------------Audio----def-------------------------------------------------



class Amp_drive_value:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normalized_amp": ("FLOAT", {"forceInput": True}),
                "add_to": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "threshold_for_add": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "add_ceiling": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.1}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            },
        }

    CATEGORY = "Apt_Preset/AD/Amp"

    RETURN_TYPES = ("FLOAT", "INT", "IMAGE")
    RETURN_NAMES = ("float", "int", "graph")
    FUNCTION = "convert_and_graph"

    def convert(self, normalized_amp, add_to, threshold_for_add, add_ceiling, scale):
        normalized_amp[np.isnan(normalized_amp)] = 0.0
        normalized_amp[np.isinf(normalized_amp)] = 1.0
        modified_values = np.where(normalized_amp > threshold_for_add, normalized_amp + add_to, normalized_amp)
        modified_values = np.clip(modified_values, 0.0, add_ceiling)
        # 使用 scale 放大 modified_values
        scaled_values = modified_values * scale
        return scaled_values, scaled_values.astype(int)

    def graph(self, normalized_amp):
        width = int(len(normalized_amp) / 10)
        if width < 10:
            width = 10
        if width > 100:
            width = 100
        plt.figure(figsize=(width, 6))
        plt.plot(normalized_amp)
        plt.xlabel("Frame(s)")
        plt.ylabel("Amplitude")
        plt.grid()
        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        plt.close()  
        buffer.seek(0)
        image = Image.open(buffer)
        print(f"Image mode: {image.mode}, Image size: {image.size}")
        return (pil2tensor(image),)


    def convert_and_graph(self, normalized_amp, add_to, threshold_for_add, add_ceiling, scale):
        float_value, int_value = self.convert(normalized_amp, add_to, threshold_for_add, add_ceiling, scale)
        graph_image = self.graph(float_value)[0]
        return float_value, int_value, graph_image


class Amp_drive_String:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultText}),
                    "normalized_amp": ("FLOAT", {"forceInput": True}),
                    "triggering_threshold": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
                     },                          
               "optional": {
                    "loop": ("BOOLEAN", {"default": True},),
                    "shuffle": ("BOOLEAN", {"default": False},),
                    }
                }

    @classmethod
    def IS_CHANGED(self, text, normalized_amp, triggering_threshold, loop, shuffle):
        if shuffle:
            return float("nan")
        m = hashlib.sha256()
        m.update(text)
        m.update(normalized_amp)
        m.update(triggering_threshold)
        m.update(loop)
        return m.digest().hex()


    CATEGORY = "Apt_Preset/AD/Amp"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    FUNCTION = "convert"
        

    def convert(self, text, normalized_amp, triggering_threshold, loop, shuffle):
        prompts = text.splitlines()

        keyframes = self.get_keyframes(normalized_amp, triggering_threshold)

        if loop and len(prompts) < len(keyframes): # Only loop if there's more prompts than keyframes
            i = 0
            result = []
            for _ in range(len(keyframes) // len(prompts)):
                if shuffle:
                    random.shuffle(prompts)
                for prompt in prompts:
                    result.append('"{}": "{}"'.format(keyframes[i], prompt))
                    i += 1
        else: # normal
            if shuffle:
                random.shuffle(prompts)
            result = ['"{}": "{}"'.format(keyframe, prompt) for keyframe, prompt in zip(keyframes, prompts)]

        result_string = ',\n'.join(result)

        return (result_string,)

    def get_keyframes(self, normalized_amp, triggering_threshold):
        above_threshold = normalized_amp >= triggering_threshold
        above_threshold = np.insert(above_threshold, 0, False)  # Add False to the beginning
        transition = np.diff(above_threshold.astype(int))
        keyframes = np.where(transition == 1)[0]
        return keyframes


class Amp_audio_Normalized:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "audio": ("AUDIO",),
                    "frame_rate": ("INT", {"default": 12, "min": 0, "max": 240, "step": 1}),
                    "operation": (["avg","max","sum"], {"default": "max"}),
                    },                            
                "optional": {
                    "start_frame": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                    "limit_frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                    }
                }

    CATEGORY = "Apt_Preset/AD/Amp"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("normalized_amp",)
    FUNCTION = "process_audio"

    def load_audio(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        # 兼容 Tensor 和 numpy array
        if isinstance(waveform, torch.Tensor):
            waveform_np = waveform.squeeze().cpu().numpy()
        else:
            waveform_np = np.asarray(waveform).squeeze()
        
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
        audio_segment = AudioSegment(
            waveform_int16.tobytes(), 
            frame_rate=sample_rate, 
            sample_width=waveform_int16.dtype.itemsize, 
            channels=1
        )
        audio_data = AudioData(audio_segment)
        return (audio_data,)

    def get_ffts(self, audio, frame_rate:int, start_frame:int=0, limit_frames:int=0):
        audio = self.load_audio(audio)[0]

        audio_data = audio.get_channel_audio_data(0)
        total_samples = len(audio_data)
        
        samples_per_frame = audio.sample_rate / frame_rate
        total_frames = int(np.ceil(total_samples / samples_per_frame))

        if (np.abs(start_frame) > total_frames):
            raise IndexError(f"Absolute value of start_frame '{start_frame}' cannot exceed the total_frames '{total_frames}'")
        if (start_frame < 0):
            start_frame = total_frames + start_frame

        ffts = []
        if (limit_frames > 0 and start_frame + limit_frames < total_frames):
            end_at_frame = start_frame + limit_frames
            total_frames = limit_frames
        else:
            end_at_frame = total_frames
        
        for i in range(start_frame, end_at_frame):
            i_next = (i + 1) * samples_per_frame

            if i_next >= total_samples:
                i_next = total_samples
            i_current = i * samples_per_frame
            frame = audio_data[round(i_current) : round(i_next)]
            ffts.append(AudioFFTData(frame, audio.sample_rate))

        return ffts

    def process_amplitude(self, audio_fft, operation):
        lower_band_range =100
        upper_band_range = 20000

        max_frames = len(audio_fft)
        # 修复未存取变量 a 的问题
        key_frame_series = pd.Series([np.nan for _ in range(max_frames)])
        
        for i in range(0, max_frames):
            fft = audio_fft[i]
            indices = fft.get_indices_for_frequency_bands(lower_band_range, upper_band_range)
            amplitude = (2 / len(fft)) * np.abs(fft.fft[indices])

            if "avg" in operation:
                key_frame_series[i] = np.mean(amplitude)
            elif "max" in operation:
                key_frame_series[i] = np.max(amplitude)
            elif "sum" in operation:
                key_frame_series[i] = np.sum(amplitude)

        normalized_amplitude =  key_frame_series / np.max( key_frame_series)
        return normalized_amplitude

    def process_audio(self, audio, frame_rate:int, operation, start_frame:int=0, limit_frames:int=0):
        ffts = self.get_ffts(audio, frame_rate, start_frame, limit_frames)
        normalized_amplitude = self.process_amplitude(ffts, operation)
        return (normalized_amplitude,)


class Amp_drive_mask:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "normalized_amp": ("FLOAT", {"forceInput": True}),
                    "width": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "height": ("INT", {"default": 512,"min": 16, "max": 4096, "step": 1}),
                    "frame_offset": ("INT", {"default": 0,"min": -255, "max": 255, "step": 1}),
                    "location_x": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "location_y": ("INT", {"default": 256,"min": 0, "max": 4096, "step": 1}),
                    "size": ("INT", {"default": 128,"min": 8, "max": 4096, "step": 1}),
                    "shape": (
                        [   
                            'none',
                            'circle',
                            'square',
                            'triangle',
                        ],
                        {
                        "default": 'none'
                        }),
                    "color": (
                        [   
                            'white',
                            'amplitude',
                        ],
                        {
                        "default": 'amplitude'
                        }),
                    },}

    CATEGORY = "Apt_Preset/AD/Amp"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"

    def convert(self, normalized_amp, width, height, frame_offset, shape, location_x, location_y, size, color):
        normalized_amp = np.clip(normalized_amp, 0.0, 1.0)
        normalized_amp = np.roll(normalized_amp, frame_offset)
        out = []
        for amp in normalized_amp:
            if color == 'amplitude':
                grayscale_value = int(amp * 255)
            elif color == 'white':
                grayscale_value = 255
            gray_color = (grayscale_value, grayscale_value, grayscale_value)
            finalsize = size * amp
            
            if shape == 'none':
                shapeimage = Image.new("RGB", (width, height), gray_color)
            else:
                shapeimage = Image.new("RGB", (width, height), "black")

            draw = ImageDraw.Draw(shapeimage)
            if shape == 'circle' or shape == 'square':
                left_up_point = (location_x - finalsize, location_y - finalsize)
                right_down_point = (location_x + finalsize,location_y + finalsize)
                two_points = [left_up_point, right_down_point]

                if shape == 'circle':
                    draw.ellipse(two_points, fill=gray_color)
                elif shape == 'square':
                    draw.rectangle(two_points, fill=gray_color)
                    
            elif shape == 'triangle':
                left_up_point = (location_x - finalsize, location_y + finalsize)
                right_down_point = (location_x + finalsize, location_y + finalsize)
                top_point = (location_x, location_y)
                draw.polygon([top_point, left_up_point, right_down_point], fill=gray_color)
            
            shapeimage = pil2tensor(shapeimage)
            mask = shapeimage[:, :, :, 0]
            out.append(mask)
        
        return (torch.cat(out, dim=0),)


class AD_sch_mask_weigh:
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "points_string": ("STRING", {"default": "0:(0.0),\n7:(1.0),\n15:(0.0)\n", "multiline": True}),
                "invert": ("BOOLEAN", {"default": False}),
                "frames": ("INT", {"default": 16,"min": 2, "max": 255, "step": 1}),
                "width": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "height": ("INT", {"default": 512,"min": 1, "max": 4096, "step": 1}),
                "easing_type": (list(easing_functions.keys()), ),
        },
    } 
    
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "createfademask"
    CATEGORY = "Apt_Preset/AD/😺backup"
    def createfademask(self, frames, width, height, invert, points_string, easing_type):
        points = []
        points_string = points_string.rstrip(',\n')
        for point_str in points_string.split(','):
            frame_str, color_str = point_str.split(':')
            frame = int(frame_str.strip())
            color = float(color_str.strip()[1:-1])
            points.append((frame, color))

        if len(points) == 0 or points[-1][0] != frames - 1:
            points.append((frames - 1, points[-1][1] if points else 0))

        points.sort(key=lambda x: x[0])

        batch_size = frames
        out = []
        image_batch = np.zeros((batch_size, height, width), dtype=np.float32)

        next_point = 1

        for i in range(batch_size):
            while next_point < len(points) and i > points[next_point][0]:
                next_point += 1

            prev_point = next_point - 1
            t = (i - points[prev_point][0]) / (points[next_point][0] - points[prev_point][0])

            easing_function = easing_functions.get(easing_type)
            if easing_function:
                t = easing_function(t)

            color = points[prev_point][1] - t * (points[prev_point][1] - points[next_point][1])
            color = np.clip(color, 0, 255)
            image = np.full((height, width), color, dtype=np.float32)
            image_batch[i] = image

        output = torch.from_numpy(image_batch)
        mask = output
        out.append(mask)

        if invert:
            return (1.0 - torch.cat(out, dim=0),)
        return (torch.cat(out, dim=0),)


class AD_sch_prompt_basic:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompts": ("STRING", {"multiline": True, "default": DefaultPromp}),
                "easing_type": (list(easing_functions.keys()), {"default": "Linear"}),
            },
            "optional": {
                "max_length": ("INT", {"default": 120, "min": 0, "max": 100000}),
                "f_text": ("STRING", {"default": "", "multiline": False}),
                "b_text": ("STRING", {"default": "", "multiline": False}),

            }
        }

    RETURN_TYPES = ("CONDITIONING","IMAGE")
    RETURN_NAMES = ("positive","graph")
    FUNCTION = "create_schedule"
    CATEGORY = "Apt_Preset/AD/😺backup"
    DESCRIPTION = """
    - 插入缓动函数举例Examples functions：
    - 0:0.5 @Sine_In@
    - 30:1 @Linear@
    - 60:0.5
    - 90:1
    - 支持的缓动函数Supported easing functions:
    - Linear,
    - Sine_In,Sine_Out,Sine_InOut,Sin_Squared,
    - Quart_In,Quart_Out,Quart_InOut,
    - Cubic_In,Cubic_Out,Cubic_InOut,
    - Circ_In,Circ_Out,Circ_InOut,
    - Back_In,Back_Out,Back_InOut,
    - Elastic_In,Elastic_Out,Elastic_InOut,
    - Bounce_In,Bounce_Out,Bounce_InOut"
    """
    def create_schedule(self,clip, prompts: str, max_length=0, easing_type="Linear", f_text="", b_text="", ):

        frames = parse_prompt_schedule(prompts.strip(), easing_type=easing_type)
        curve_img = generate_frame_weight_curve_image(frames, max_length)
        positive = build_conditioning(frames, clip, max_length, f_text=f_text, b_text=b_text)

        return ( positive, curve_img)


class AD_sch_value:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "values": ("STRING", {"multiline": True, "default": DefaultValue}),
                "easing_type": (list(easing_functions.keys()), {"default": "Linear"}),
            },
            "optional": {
                "max_length": ("INT", {"default": 120, "min": 0, "max": 100000}),
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.001, "max": 1000.0, "step": 0.01}),
                "offset": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.01}),
            }
        }

    # 修改返回类型，添加 INT
    RETURN_TYPES = (ANY_TYPE, "IMAGE")
    RETURN_NAMES = ("data",  "graph")
    FUNCTION = "create_schedule"
    CATEGORY = "Apt_Preset/AD/😺backup"
    DESCRIPTION = """
    - 插入缓动函数举例Examples functions：
    - 0:0.5 @Sine_In@
    - 30:1 @Linear@
    - 60:0.5
    - 90:1
    - 支持的缓动函数Supported easing functions:
    - Linear,
    - Sine_In,Sine_Out,Sine_InOut,Sin_Squared,
    - Quart_In,Quart_Out,Quart_InOut,
    - Cubic_In,Cubic_Out,Cubic_InOut,
    - Circ_In,Circ_Out,Circ_InOut,
    - Back_In,Back_Out,Back_InOut,
    - Elastic_In,Elastic_Out,Elastic_InOut,
    - Bounce_In,Bounce_Out,Bounce_InOut"
    """
    def create_schedule(self, values: str, easing_type="Linear", max_length=0, scale_factor=1.0, offset=0.0, ):
        keyframes = parse_prompt_schedule(values.strip(), easing_type=easing_type)
        if not keyframes:
            raise ValueError("No valid keyframes found.")

        if max_length <= 0:
            max_length = keyframes[-1].index + 1

        values_seq = [None] * max_length
        frame_methods = []  # 用于记录每段使用的插值方法

        # 遍历所有关键帧，为每个帧设置值并处理与下一个关键帧之间的插值
        for i in range(len(keyframes)):
            curr_kf = keyframes[i]
            curr_idx = curr_kf.index

            try:
                curr_val = float(curr_kf.prompt)
            except ValueError:
                continue

            if curr_idx >= max_length:
                break

            # 设置当前帧数值
            values_seq[curr_idx] = curr_val

            # 如果不是最后一帧，则处理与下一帧之间的插值
            if i + 1 < len(keyframes):
                next_kf = keyframes[i + 1]
                next_idx = next_kf.index
                next_val = float(next_kf.prompt)

                if next_idx >= max_length:
                    continue

                diff_len = next_idx - curr_idx
                weights = torch.linspace(0, 1, diff_len + 1)[1:-1]
                easing_weights = [apply_easing(w.item(), curr_kf.interp_method) for w in weights]
                transformed_weights = [min(max(w * scale_factor + offset, 0.0), 1.0) for w in easing_weights]

                for j, w in enumerate(transformed_weights):
                    idx = curr_idx + j + 1
                    if idx >= max_length:
                        break
                    values_seq[idx] = curr_val * (1.0 - w) + next_val * w

                # 记录插值区间及使用的 interp_method（用于绘图）
                frame_methods.append((curr_idx, next_idx, curr_kf.interp_method))

        # 填充首尾缺失帧
        first_valid = next((i for i in range(max_length) if values_seq[i] is not None), None)
        last_valid = None
        for i in range(max_length):
            if values_seq[i] is not None:
                last_valid = i
            elif last_valid is not None:
                values_seq[i] = values_seq[last_valid]

        if first_valid is not None:
            for i in range(first_valid):
                values_seq[i] = values_seq[first_valid]

        # 构建输出 tensor
        value_tensor = torch.tensor(values_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)

        # 将 value_tensor 转换为 np.array
        value_array = np.array(value_tensor.squeeze().tolist(), dtype=np.float32)

        # 转换为 int 类型的 np.array
        values_int_array = np.array([int(val) if val is not None else 0 for val in values_seq], dtype=np.int32)

        # 绘图使用实际数值
        curve_img = generate_value_curve_image_with_data(values_seq, max_length, frame_methods)

        # 修改返回值，使用 np.array
        return (value_array, curve_img)




COLOR_CHOICES = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "gray"]

class AD_sch_image_merge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data1": ("FLOAT", {"forceInput": True}),
                "data2": ("FLOAT", {"forceInput": True}),
                "color1": (COLOR_CHOICES, {"default": "red"}),
                "color2": (COLOR_CHOICES, {"default": "green"})
            },
            "optional": {
                "data3": ("FLOAT", {"forceInput": True}),
                "data4": ("FLOAT", {"forceInput": True}),
                "color3": (COLOR_CHOICES, {"default": "blue"}),
                "color4": (COLOR_CHOICES, {"default": "yellow"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("merged_graph",)
    FUNCTION = "generate_multi_value_image"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def generate_multi_value_image(self, data1, data2, color1, color2, data3=None, data4=None, color3=None, color4=None):


        # 存储所有输入数据和对应颜色
        data_list = [data1, data2]
        color_list = [color1, color2]

        if data3 is not None:
            data_list.append(data3)
            color_list.append(color3)
        if data4 is not None:
            data_list.append(data4)
            color_list.append(color4)

        # 过滤出可迭代对象并计算最大长度
        iterable_data = [data for data in data_list if isinstance(data, collections.abc.Iterable) and not isinstance(data, (str, bytes))]
        if iterable_data:
            max_length = max(len(data) for data in iterable_data)
        else:
            max_length = 1  # 如果没有可迭代对象，设置默认长度为 1

        plt.figure(figsize=(12, 6))

        # 绘制每条曲线
        for i, data in enumerate(data_list):
            if isinstance(data, collections.abc.Iterable) and not isinstance(data, (str, bytes)):
                y = [v if v is not None else 0.0 for v in data]
                plt.plot(range(len(y)), y, marker='o', linestyle='-', markersize=3, color=color_list[i], label=f"Data {i + 1}")
            else:
                # 处理单个数值的情况
                plt.axhline(y=data, color=color_list[i], label=f"Data {i + 1}")

        plt.title("Multiple Interpolated Value Curves per Frame")
        plt.xlabel("Frame Index")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend(loc="upper left")

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()
        buffer.seek(0)
        image = Image.open(buffer)

        def pil2tensor(image):
            return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

        return (pil2tensor(image),)



class AD_pingpong_vedio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                "startOffset": ("INT", {"default": 0, "min": 0, "max": 100}),
                "endOffset": ("INT", {"default": 0, "min": 0, "max": 100}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "loop_video"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def loop_video(self, images, startOffset=0, endOffset=0):
        total_frames = len(images)

        if total_frames < 2:
            return (images,)

        # 计算偏移后的起始和结束索引
        new_start = min(max(0, startOffset), total_frames - 1)
        new_end = max(min(total_frames - 1, total_frames - 1 - endOffset), new_start)

        # 确保总帧数不少于6帧
        if new_end - new_start + 1 < 6:
            new_start = max(0, new_end - 5)

        original_sequence = images[new_start : new_end + 1]

        if len(original_sequence) == 1:
            return (original_sequence,)
        elif len(original_sequence) == 2:
            return (torch.cat([original_sequence, original_sequence[0].unsqueeze(0)], dim=0),)

        reversed_middle = original_sequence[1:-1].flip(dims=[0])
        outimage = torch.cat([original_sequence, reversed_middle], dim=0)

        return (outimage,)









import os
import av
import torch
from typing import Optional, List
from fractions import Fraction
from comfy_api.latest import io, Input, InputImpl, Types
from comfy_execution.graph import ExecutionBlocker
from comfy_extras.nodes_custom_sampler import SplitSigmas
from ..office_unit import BasicScheduler
from ..NodeChx.main_nodes import basic_Ksampler_custom, AD_CreateVideo, _apt_replace_av_video_latent, _apt_second_pass_positive
from .C_latent import latent_minimaxH3_scale


class AD_In_VideoSplit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_In_VideoSplit",
            display_name="AD_In_VideoSplit",
            category="Apt_Preset/AD",
            essentials_category="Video Tools",
            inputs=[
                io.Video.Input("video"),
                io.Float.Input("force_rate", default=0.0, min=0.0, max=60.0, step=1.0, display_mode=io.NumberDisplay.number),
                io.Int.Input("custom_width", default=0, min=0, max=8192, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("custom_height", default=0, min=0, max=8192, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("frame_load_cap", default=0, min=0, max=2**53 - 1, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("skip_first_frames", default=0, min=0, max=2**53 - 1, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("select_every_nth", default=1, min=1, max=2**53 - 1, step=1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Video.Output(display_name="VIDEO"),
                io.Image.Output(display_name="IMAGE"),
                io.Audio.Output(display_name="audio"),
            ],
        )

    @classmethod
    def execute(cls, video, force_rate=0.0, custom_width=0, custom_height=0,
                frame_load_cap=0, skip_first_frames=0, select_every_nth=1):
        components = video.get_components()
        images = components.images
        source_fps = float(components.frame_rate)
        loaded_fps = float(force_rate) if force_rate > 0 else source_fps

        if force_rate > 0 and images.shape[0] > 0:
            frame_count = int(images.shape[0] / source_fps * loaded_fps)
            indexes = (torch.arange(frame_count, device=images.device) * source_fps / loaded_fps).long()
            images = images[indexes.clamp_max(images.shape[0] - 1)]

        images = images[skip_first_frames::select_every_nth]
        if frame_load_cap > 0:
            images = images[:frame_load_cap]
        if images.shape[0] == 0:
            raise RuntimeError("No frames generated")

        height, width = images.shape[1:3]
        if custom_width > 0 or custom_height > 0:
            if custom_width == 0:
                custom_width = round(width * custom_height / height)
            elif custom_height == 0:
                custom_height = round(height * custom_width / width)
            images = comfy.utils.common_upscale(
                images.movedim(-1, 1), custom_width, custom_height, "lanczos", "center"
            ).movedim(1, -1)

        audio = components.audio
        if audio is not None and (skip_first_frames > 0 or frame_load_cap > 0):
            sample_rate = int(audio["sample_rate"])
            start = round(skip_first_frames / loaded_fps * sample_rate)
            end = None
            if frame_load_cap > 0:
                end = start + round(frame_load_cap * select_every_nth / loaded_fps * sample_rate)
            audio = dict(audio)
            audio["waveform"] = audio["waveform"][..., start:end]

        output_fps = Fraction(loaded_fps / select_every_nth).limit_denominator(1000)
        output_video = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images, audio=audio, frame_rate=output_fps),
            bit_depth=video.get_bit_depth(),
        )
        return io.NodeOutput(output_video, images, audio)


def normalize_audio(audio_data, default_sample_rate=44100):
    if audio_data is None:
        return None
    waveform = None
    sample_rate = int(default_sample_rate)
    if isinstance(audio_data, dict):
        if "waveform" in audio_data and "sample_rate" in audio_data:
            waveform = audio_data["waveform"]
            sample_rate = int(audio_data["sample_rate"])
        elif "tensor" in audio_data:
            waveform = audio_data["tensor"]
    elif isinstance(audio_data, torch.Tensor):
        waveform = audio_data
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        return None
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0).unsqueeze(0)
    elif waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 3:
        waveform = waveform.reshape(1, waveform.shape[-2], waveform.shape[-1])
    elif waveform.ndim == 3 and waveform.shape[0] != 1:
        waveform = waveform[:1]
    channels = waveform.shape[1]
    if channels > 2:
        waveform = waveform[:, :2, :]
    return {"waveform": waveform, "sample_rate": sample_rate}

def resample_audio_waveform(waveform, current_sample_rate, target_sample_rate):
    if current_sample_rate == target_sample_rate:
        return waveform
    if waveform.shape[-1] == 0:
        return waveform
    try:
        import torchaudio
        batch, channels, _ = waveform.shape
        flattened = waveform.reshape(batch * channels, -1)
        resampled = torchaudio.functional.resample(flattened, current_sample_rate, target_sample_rate)
        return resampled.reshape(batch, channels, -1)
    except Exception:
        target_len = max(1, int(round(waveform.shape[-1] * float(target_sample_rate) / float(current_sample_rate))))
        return torch.nn.functional.interpolate(
            waveform,
            size=target_len,
            mode="linear",
            align_corners=False,
        )

def concat_audio_segments(audio_segments, preferred_sample_rate):
    if len(audio_segments) == 0:
        return None
    normalized = []
    for audio in audio_segments:
        item = normalize_audio(audio)
        if item is None:
            continue
        normalized.append(item)
    if len(normalized) == 0:
        return None
    sample_rates = {int(item["sample_rate"]) for item in normalized}
    if len(sample_rates) == 1:
        target_sample_rate = int(normalized[0]["sample_rate"])
    else:
        target_sample_rate = int(preferred_sample_rate) if preferred_sample_rate and preferred_sample_rate > 0 else max(sample_rates)
    for i, item in enumerate(normalized):
        if int(item["sample_rate"]) != target_sample_rate:
            normalized[i] = {
                "waveform": resample_audio_waveform(item["waveform"], int(item["sample_rate"]), target_sample_rate),
                "sample_rate": target_sample_rate,
            }
    target_channels = max(item["waveform"].shape[1] for item in normalized)
    waveforms = []
    for item in normalized:
        waveform = item["waveform"]
        if waveform.shape[1] == 1 and target_channels == 2:
            waveform = waveform.repeat(1, 2, 1)
        elif waveform.shape[1] > target_channels:
            waveform = waveform[:, :target_channels, :]
        waveforms.append(waveform)
    return {"waveform": torch.cat(waveforms, dim=2), "sample_rate": target_sample_rate}

class AD_video_merge(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_video_merge",
            display_name="AD_video_merge",
            search_aliases=["combine videos", "join videos", "concatenate videos", "merge videos horizontally", "merge videos vertically"],
            category="Apt_Preset/AD",
            essentials_category="Video Tools",
            description="Merge videos with audio (like Jianying)",
            inputs=[
                io.Video.Input("video1", optional=True),
                io.Video.Input("video2", optional=True),
                io.Video.Input("video3", optional=True),
                io.Video.Input("video4", optional=True),
                io.Video.Input("video5", optional=True),
                io.Video.Input("video6", optional=True),
                io.Video.Input("video7", optional=True),
                io.Video.Input("video8", optional=True),
                io.Video.Input("video9", optional=True),
                io.Video.Input("video10", optional=True),
                io.Combo.Input("merge_mode", options=[ "sequential", "horizontal", "vertical"], default="sequential"),
                io.Float.Input("target_fps", default=24.0, min=1.0, max=120.0, step=1.0),
                io.Int.Input("audio_sample_rate", default=44100, min=16000, max=48000),
                io.Boolean.Input("force_audio_merge", default=True)
            ],
            outputs=[io.Video.Output()]
        )

    @classmethod
    def execute(cls, video1=None, video2=None, video3=None, video4=None, video5=None, video6=None, video7=None, video8=None, video9=None, video10=None, merge_mode="sequential", target_fps=24.0, audio_sample_rate=44100, force_audio_merge=True):
        videos = [v for v in [video1, video2, video3, video4, video5, video6, video7, video8, video9, video10] if v is not None]
        if len(videos) == 0:
            raise ValueError("At least one video input must be connected")
        if len(videos) == 1:
            return io.NodeOutput(videos[0])
        all_components = [v.get_components() for v in videos]
        fps = target_fps if target_fps > 0 else float(all_components[0].frame_rate)
        merged_images = None
        merged_audio = None

        if merge_mode == "sequential":
            all_images = []
            all_audio_items = []
            for comp in all_components:
                video_fps = float(comp.frame_rate)
                num_frames = comp.images.shape[0]
                if video_fps != fps:
                    target_frames = max(1, int(round(num_frames * fps / video_fps)))
                    indices = torch.linspace(0, num_frames - 1, target_frames).long()
                    frames = comp.images[indices]
                else:
                    frames = comp.images
                all_images.append(frames)
                if force_audio_merge and comp.audio is not None:
                    audio = normalize_audio(comp.audio)
                    if audio is not None:
                        all_audio_items.append(audio)
            merged_images = torch.cat(all_images, dim=0)
            if force_audio_merge and len(all_audio_items) > 0:
                merged_audio = concat_audio_segments(all_audio_items, audio_sample_rate)
        else:
            min_frames = min(comp.images.shape[0] for comp in all_components)
            resampled_images = []
            main_audio = None
            for idx, comp in enumerate(all_components):
                num_frames = comp.images.shape[0]
                if num_frames != min_frames:
                    indices = torch.linspace(0, num_frames - 1, min_frames).long()
                    frames = comp.images[indices]
                else:
                    frames = comp.images[:min_frames]
                resampled_images.append(frames)
                if force_audio_merge and main_audio is None and comp.audio is not None:
                    audio = normalize_audio(comp.audio)
                    if audio is not None:
                        main_audio = audio
            if merge_mode == "horizontal":
                merged_images = torch.cat(resampled_images, dim=2)
            else:
                merged_images = torch.cat(resampled_images, dim=1)
            merged_audio = main_audio

        return io.NodeOutput(InputImpl.VideoFromComponents(Types.VideoComponents(images=merged_images, audio=merged_audio, frame_rate=Fraction(fps))))











import os
import sys
import cv2
import zipfile
import traceback
import datetime
import subprocess
import base64
import json
import numpy as np
import folder_paths
from PIL import Image
from aiohttp import web
from server import PromptServer

# ========== 安全导入 ==========
try:
    import requests
except ImportError:
    requests = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from scenedetect import open_video, SceneManager, FrameTimecode
    from scenedetect.detectors import ContentDetector, AdaptiveDetector, HashDetector, ThresholdDetector
    from scenedetect.video_splitter import split_video_ffmpeg
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False

# ==================================

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

ANY = AnyType("*")

def register_node(cls):
    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.__name__] = cls.DISPLAY_NAME
    return cls

def get_ffmpeg_path():
    comfy_root = os.path.dirname(os.path.abspath(sys.argv[0]))
    ffmpeg_dir = os.path.join(comfy_root, "models", "Apt_File")
    ffmpeg_path = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    return ffmpeg_dir, ffmpeg_path

def auto_install_ffmpeg():
    ffmpeg_dir, ffmpeg_path = get_ffmpeg_path()
    os.makedirs(ffmpeg_dir, exist_ok=True)
    if os.path.exists(ffmpeg_path):
        return True, ffmpeg_path
    if not requests:
        return False, ffmpeg_path
    try:
        zip_url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-win64-gpl.zip"
        zip_path = os.path.join(ffmpeg_dir, "ffmpeg.zip")
        with requests.get(zip_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            extracted_ffmpeg = None
            for f in zip_ref.namelist():
                if f.endswith("ffmpeg.exe"):
                    zip_ref.extract(f, ffmpeg_dir)
                    extracted_ffmpeg = os.path.join(ffmpeg_dir, f.replace("/", os.sep))
                    break
            if not extracted_ffmpeg or not os.path.exists(extracted_ffmpeg):
                for root, _, files in os.walk(ffmpeg_dir):
                    if "ffmpeg.exe" in files:
                        extracted_ffmpeg = os.path.join(root, "ffmpeg.exe")
                        break
            if not extracted_ffmpeg or not os.path.exists(extracted_ffmpeg):
                return False, ffmpeg_path
            if os.path.exists(ffmpeg_path):
                os.remove(ffmpeg_path)
            os.replace(extracted_ffmpeg, ffmpeg_path)
        os.remove(zip_path)
        return True, ffmpeg_path
    except Exception:
        return False, ffmpeg_path

def check_ffmpeg():
    _, ffmpeg_path = get_ffmpeg_path()
    if os.path.exists(ffmpeg_path):
        return True, ffmpeg_path
    return auto_install_ffmpeg()


def get_ffprobe_path():
    _, ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = os.path.join(os.path.dirname(ffmpeg_path), "ffprobe.exe")
    if os.path.exists(ffprobe_path):
        return ffprobe_path
    return None


def _run_process(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or "").strip()
        # Prefer tail lines so ffmpeg banner doesn't hide the actual reason.
        if err:
            lines = [ln for ln in err.splitlines() if ln.strip()]
            err = "\n".join(lines[-15:]) if lines else err
        raise RuntimeError(err[:1600] if err else f"Command failed: {' '.join(cmd)}")
    return result.stdout


def _run_process_bytes(cmd):
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = (result.stderr or result.stdout or b"").decode("utf-8", errors="replace").strip()
        raise RuntimeError(err[:1200] if err else f"Command failed: {' '.join(cmd)}")
    return result.stdout


def _resolve_media_input_path(raw_path: str):
    if not raw_path:
        return None
    candidate = str(raw_path).strip().strip('"').strip("'")
    if candidate.lower().startswith("file://"):
        candidate = candidate[7:]
    # Remove url query/hash parts if user pasted browser-style path.
    candidate = candidate.split("?", 1)[0].split("#", 1)[0]
    # Normalize common slash variants.
    candidate = candidate.replace("\\\\", "\\")
    if not candidate:
        return None
    if os.path.exists(candidate):
        return os.path.abspath(candidate)
    try:
        annotated = folder_paths.get_annotated_filepath(candidate)
        if annotated and os.path.exists(annotated):
            return os.path.abspath(annotated)
    except Exception:
        pass
    in_dir = folder_paths.get_input_directory()
    p1 = os.path.join(in_dir, candidate)
    if os.path.exists(p1):
        return os.path.abspath(p1)
    p2 = os.path.join(in_dir, os.path.basename(candidate))
    if os.path.exists(p2):
        return os.path.abspath(p2)
    return None


def _resolve_media_from_video_input(video):
    if video is None:
        return None
    visited = set()

    def _iter_strings(obj, depth=0):
        if depth > 6:
            return
        oid = id(obj)
        if oid in visited:
            return
        visited.add(oid)

        if isinstance(obj, str):
            s = obj.strip()
            if s:
                yield s
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("{") and s.endswith("}")):
                try:
                    parsed = json.loads(s)
                    yield from _iter_strings(parsed, depth + 1)
                except Exception:
                    pass
            return

        if isinstance(obj, dict):
            for v in obj.values():
                yield from _iter_strings(v, depth + 1)
            return

        if isinstance(obj, (list, tuple, set)):
            for v in obj:
                yield from _iter_strings(v, depth + 1)
            return

        if hasattr(obj, "video_info") and isinstance(getattr(obj, "video_info", None), dict):
            yield from _iter_strings(obj.video_info, depth + 1)

        for attr in ("path", "video_path", "file_path", "filepath", "url", "name"):
            if hasattr(obj, attr):
                try:
                    v = getattr(obj, attr)
                except Exception:
                    continue
                yield from _iter_strings(v, depth + 1)
        if hasattr(obj, "__dict__"):
            try:
                yield from _iter_strings(vars(obj), depth + 1)
            except Exception:
                pass

    for candidate in _iter_strings(video):
        resolved = _resolve_media_input_path(candidate)
        if resolved:
            return resolved

    # Last-resort: parse repr/str for path-like substrings
    try:
        text = str(video)
    except Exception:
        text = ""
    if text:
        path_like = re.findall(
            r"[A-Za-z]:[\\/][^\s'\"<>|]+?\.(?:mp4|mov|mkv|webm|avi|m4v|wav|mp3|flac|ogg|m4a|aac)|"
            r"[^\\/:*?\"<>|\r\n]+?\.(?:mp4|mov|mkv|webm|avi|m4v|wav|mp3|flac|ogg|m4a|aac)",
            text,
            flags=re.IGNORECASE,
        )
        for candidate in path_like:
            resolved = _resolve_media_input_path(candidate)
            if resolved:
                return resolved
    return None


def _probe_media_info(path: str):
    # Prefer ffprobe; fall back to ffmpeg stderr parsing when ffprobe is unavailable.
    ffprobe_path = get_ffprobe_path()
    if ffprobe_path:
        try:
            cmd = [
                ffprobe_path, "-v", "error",
                "-show_entries", "format=duration:stream=codec_type",
                "-of", "json", path,
            ]
            out = _run_process(cmd)
            data = json.loads(out) if out else {}
            duration = float((data.get("format") or {}).get("duration") or 0.0)
            has_video = False
            has_audio = False
            for s in data.get("streams", []):
                codec_type = (s.get("codec_type") or "").lower()
                if codec_type == "video":
                    has_video = True
                elif codec_type == "audio":
                    has_audio = True
            if duration > 0 or has_video or has_audio:
                return {"duration": duration, "has_video": has_video, "has_audio": has_audio}
        except Exception:
            pass

    ffmpeg_ok, ffmpeg_path = check_ffmpeg()
    if not ffmpeg_ok:
        raise RuntimeError("未找到 ffmpeg，无法探测媒体信息。")

    probe_cmd = [ffmpeg_path, "-hide_banner", "-i", path]
    proc = subprocess.run(probe_cmd, capture_output=True, text=True)
    probe_text = f"{proc.stderr or ''}\n{proc.stdout or ''}"
    text_lower = probe_text.lower()

    has_video = "video:" in text_lower
    has_audio = "audio:" in text_lower

    duration = 0.0
    marker = "Duration:"
    idx = probe_text.find(marker)
    if idx >= 0:
        # Example: Duration: 00:01:23.45, start: 0.000000, bitrate: ...
        tail = probe_text[idx + len(marker):].strip()
        clock = tail.split(",", 1)[0].strip()
        parts = clock.split(":")
        if len(parts) == 3:
            try:
                h = float(parts[0])
                m = float(parts[1])
                s = float(parts[2])
                duration = h * 3600 + m * 60 + s
            except Exception:
                duration = 0.0

    if duration <= 0 and not has_video and not has_audio:
        raise RuntimeError("ffmpeg/ffprobe 均未能识别媒体信息，请确认文件可播放且路径有效。")

    return {"duration": duration, "has_video": has_video, "has_audio": has_audio}


def _extract_waveform_peaks(path: str, bins: int = 1400):
    ffmpeg_ok, ffmpeg_path = check_ffmpeg()
    if not ffmpeg_ok:
        raise RuntimeError("未找到 ffmpeg，无法生成波形。")
    bins = max(64, min(int(bins), 4096))
    cmd = [
        ffmpeg_path, "-v", "error",
        "-i", path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "f32le",
        "-",
    ]
    raw = _run_process_bytes(cmd)
    if not raw or not isinstance(raw, (bytes, bytearray, memoryview)):
        return []
    # 确保是 bytes 类型
    if isinstance(raw, memoryview):
        raw = bytes(raw)
    samples = np.frombuffer(raw, dtype=np.float32)
    if samples.size == 0:
        return []
    abs_samples = np.abs(samples)
    edges = np.linspace(0, abs_samples.size, num=bins + 1, dtype=np.int64)
    peaks = []
    for i in range(bins):
        s = edges[i]
        e = edges[i + 1]
        if e <= s:
            peaks.append(0.0)
        else:
            peaks.append(float(np.max(abs_samples[s:e])))
    return peaks


def _parse_marker_seconds(markers_json: str, duration: float):
    text = (markers_json or "").strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    if isinstance(parsed, dict):
        parsed = parsed.get("markers", [])
    if not isinstance(parsed, list):
        return []
    seen = set()
    markers = []
    for item in parsed:
        try:
            sec = float(item)
        except Exception:
            continue
        sec = max(0.0, min(sec, max(0.0, duration)))
        key = int(round(sec * 1000))
        if key in seen:
            continue
        seen.add(key)
        markers.append(sec)
    markers.sort()
    return markers


def _build_segments_by_markers(markers, duration):
    points = [0.0] + list(markers) + [max(0.0, float(duration))]
    segments = []
    for i in range(len(points) - 1):
        s = float(points[i])
        e = float(points[i + 1])
        if e - s >= 0.01:
            segments.append((s, e))
    return segments


def _load_image_tensor(image_path: str):
    with Image.open(image_path).convert("RGB") as img:
        arr = np.array(img).astype(np.float32) / 255.0
    # Comfy IMAGE standard shape: [B, H, W, C]
    return torch.from_numpy(arr).unsqueeze(0)


def _encode_media_token(path: str) -> str:
    return base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii")


def _decode_media_token(token: str):
    if not token:
        return None
    try:
        padded = token + "=" * (-len(token) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        return decoded
    except Exception:
        return None


@PromptServer.instance.routes.post("/apt_preset/media_trim/resolve")
async def apt_media_trim_resolve(request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    media_path = _resolve_media_input_path(payload.get("path", ""))
    if not media_path or not os.path.exists(media_path):
        return web.json_response({"ok": False, "error": "未找到媒体文件，请检查路径。"}, status=400)
    try:
        info = _probe_media_info(media_path)
        peaks = _extract_waveform_peaks(media_path, bins=1400)
    except Exception as e:
        return web.json_response({"ok": False, "error": f"探测媒体信息失败: {e}"}, status=500)
    token = _encode_media_token(media_path)
    media_type = "video" if info.get("has_video") else "audio"
    return web.json_response(
        {
            "ok": True,
            "media_url": f"/apt_preset/media_trim/file?token={token}",
            "duration": float(info.get("duration", 0.0)),
            "media_type": media_type,
            "peaks": peaks,
        }
    )


@PromptServer.instance.routes.get("/apt_preset/media_trim/file")
async def apt_media_trim_file(request):
    token = request.query.get("token", "")
    media_path = _decode_media_token(token)
    if not media_path:
        return web.Response(status=400, text="invalid token")
    media_path = os.path.abspath(media_path)
    if not os.path.exists(media_path):
        return web.Response(status=404, text="file not found")
    return web.FileResponse(media_path)


class AD_media_trim_visual:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "media_path": ("STRING", {"default": "", "tooltip": "可留空。若连接 video 端口，优先使用端口视频路径。"}),
                "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "end_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "step": 0.01}),
                "markers_json": ("STRING", {"default": "[]", "multiline": False, "tooltip": "前端打标记后自动写入。格式: [1.2, 3.4]"}),
                "output_name": ("STRING", {"default": "trim"}),
                "reencode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "video": (ANY, {"default": None}),
                "audio": (ANY, {"default": None}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("audio_list", "video_list", "image_list")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD/😺backup"
    name="AD_media_trim_visual"
    @staticmethod
    def _safe_name(name: str):
        name = (name or "trim").strip()
        name = re.sub(r"[^a-zA-Z0-9_\-\u4e00-\u9fff]+", "_", name)
        return name[:80] or "trim"

    def execute(self, media_path, start_sec, end_sec, markers_json, output_name, reencode, video=None, audio=None):
        input_path = _resolve_media_from_video_input(video)
        if not input_path:
            input_path = _resolve_media_from_video_input(audio)
        if not input_path:
            input_path = _resolve_media_input_path(media_path)
        if not input_path or not os.path.exists(input_path):
            raise ValueError("未找到媒体文件。请优先连接 video/audio 端口，或填写可访问的 media_path。")

        ffmpeg_ok, ffmpeg_path = check_ffmpeg()
        if not ffmpeg_ok:
            raise RuntimeError("缺少 FFmpeg，请先安装或检查 models/Apt_File/ffmpeg.exe。")

        info = _probe_media_info(input_path)
        duration = float(info.get("duration", 0.0))
        has_video = bool(info.get("has_video"))
        has_audio = bool(info.get("has_audio"))
        if duration <= 0:
            raise RuntimeError("无法读取媒体时长，可能是格式不支持或文件损坏。")

        markers = _parse_marker_seconds(markers_json, duration)
        if markers:
            segments = _build_segments_by_markers(markers, duration)
        else:
            start = max(0.0, min(float(start_sec), duration))
            end = max(0.0, min(float(end_sec), duration))
            if end <= start:
                raise ValueError(f"结束时间必须大于起始时间。当前 start={start:.3f}, end={end:.3f}")
            segments = [(start, end)]
        if not segments:
            # Keep behavior stable even when markers are out of range or duplicated.
            segments = [(0.0, max(0.01, duration))]

        out_dir = os.path.join(folder_paths.get_output_directory(), "apt_media_trim")
        os.makedirs(out_dir, exist_ok=True)
        # Use stable containers for trimming to avoid codec/container mismatch.
        ext = ".mp4" if has_video else ".wav"
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self._safe_name(output_name)}_{stamp}"
        audio_list = []
        video_list = []
        image_tensors = []

        for idx, (seg_start, seg_end) in enumerate(segments):
            seg_tag = f"{base}_{idx:03d}"
            clip_path = os.path.join(out_dir, f"{seg_tag}_clip{ext}")

            # Stable trim: try stream-copy first for speed, then fall back to re-encode.
            copy_cmd = [
                ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-y", "-ss", f"{seg_start:.3f}", "-to", f"{seg_end:.3f}",
                "-i", input_path, "-c", "copy", "-avoid_negative_ts", "make_zero", clip_path,
            ]
            reencode_cmd = [
                ffmpeg_path, "-hide_banner", "-loglevel", "error",
                "-y", "-ss", f"{seg_start:.3f}", "-to", f"{seg_end:.3f}", "-i", input_path,
            ]
            if has_video:
                reencode_cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-b:a", "192k"]
            else:
                reencode_cmd += ["-c:a", "pcm_s16le", "-ar", "44100", "-ac", "2"]
            reencode_cmd.append(clip_path)

            if reencode:
                _run_process(reencode_cmd)
            else:
                try:
                    _run_process(copy_cmd)
                except Exception:
                    _run_process(reencode_cmd)

            if has_audio:
                audio_path = os.path.join(out_dir, f"{seg_tag}_audio.wav")
                _run_process([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", clip_path, "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2", audio_path])
                audio_list.append(audio_path)
            else:
                audio_list.append("")

            if has_video:
                video_path = os.path.join(out_dir, f"{seg_tag}_video.mp4")
                try:
                    _run_process([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", clip_path, "-an", "-c:v", "copy", video_path])
                except Exception:
                    _run_process([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-i", clip_path, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", video_path])
                video_list.append(video_path)

                shot_time = seg_start + max(0.0, (seg_end - seg_start) * 0.5)
                img_path = os.path.join(out_dir, f"{seg_tag}_shot.png")
                _run_process([ffmpeg_path, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{shot_time:.3f}", "-i", input_path, "-frames:v", "1", img_path])
                if os.path.exists(img_path):
                    image_tensors.append(_load_image_tensor(img_path))
                else:
                    image_tensors.append(torch.zeros((1, 64, 64, 3), dtype=torch.float32))
            else:
                video_list.append("")
                image_tensors.append(torch.zeros((1, 64, 64, 3), dtype=torch.float32))

        # Ensure list outputs have identical lengths to avoid list-mapping index errors.
        target_len = max(len(audio_list), len(video_list), len(image_tensors), len(segments), 1)
        while len(audio_list) < target_len:
            audio_list.append("")
        while len(video_list) < target_len:
            video_list.append("")
        while len(image_tensors) < target_len:
            image_tensors.append(torch.zeros((1, 64, 64, 3), dtype=torch.float32))

        return (
            audio_list,
            video_list,
            image_tensors,
        )

def pil2tensor(img):
    return np.array(img).astype(np.float32) / 255.0

@register_node
class AD_VideoSeg:
    CATEGORY = "Apt_Preset/AD"
    DISPLAY_NAME = "AD_VideoSeg"

    INPUT_IS_LIST = False

    INPUT_TYPES = lambda: {
        "required": {
            "video_path": ("STRING", {"default": ""}), # 路径输入（手动填）
            "detector_mode": (["内容检测", "自适应检测", "哈希检测"], {"default": "自适应检测"}),
            "enable_fade_black": ("BOOLEAN", {"default": True}),
            "sensitivity": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 200.0, "step": 1}),
            "black_threshold": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1}),
            "min_scene_seconds": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 10.0, "step": 0.1}),
            "frame_skip": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
            "Seg_mold": ("BOOLEAN", {"default": True, "label_on": "按数量分割", "label_off": "自动分割"}),
            "target_scene_count": ("INT", {"default": 5, "min": 1, "max": 30}),
            "save_folder": ("STRING", {"default": "output/scene_ultimate"}),
        },
        "optional": {

            "video": (ANY, {"default": None}),       # 视频输入（连线用）
        }
    }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("imagelsit", "status")
    FUNCTION = "process"
    OUTPUT_NODE = True
    DESCRIPTION = """
    视频场景分割工具：自动检测镜头切换，分割视频并提取每段首尾帧。
    支持三种检测算法 + 黑场/淡入淡出检测，可精准控制分割效果。

    【三种检测模式】
    • 内容检测：基础画面突变检测，适合普通硬切镜头
    • 自适应检测：抗抖动、抗快速运动，最稳定，推荐默认
    • 哈希检测：画面感知哈希对比，抗水印、抗光影变化

    【关键参数说明】
    • 灵敏度：数值越小越灵敏，切分越细（1-200）
    • 黑场阈值：画面亮度低于该值判定为黑场/淡入淡出
    • 最小场景时长：防止切出过短碎片镜头（秒）
    • 跳帧检测：数值越大速度越快，1=不跳帧，最高4
    • 目标分割数量：自动合并/均分，强制输出N段视频
    • 跳过片头/裁剪片尾：忽略视频开头结尾不参与分割
    """

    def _resolve_video_path(self, video, video_path):
        final_path = None

        if video is not None:
            if hasattr(video, "video_info") and isinstance(video.video_info, dict):
                final_path = video.video_info.get("filepath", None)

            if not final_path:
                if isinstance(video, str):
                    final_path = video
                elif isinstance(video, (list, tuple)) and len(video) > 0 and isinstance(video[0], str):
                    final_path = video[0]
                elif isinstance(video, dict):
                    for val in video.values():
                        if isinstance(val, str) and val.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")):
                            final_path = val
                            break
                else:
                    for attr in ["path", "video_path", "file_path", "filepath", "url"]:
                        if hasattr(video, attr):
                            val = getattr(video, attr)
                            if isinstance(val, str):
                                final_path = val
                                break
                    if not final_path:
                        try:
                            for attr in dir(video):
                                if not attr.startswith("__"):
                                    val = getattr(video, attr)
                                    if isinstance(val, str) and val.lower().endswith((".mp4", ".mov", ".webm", ".avi", ".mkv")):
                                        final_path = val
                                        break
                        except Exception:
                            pass

        if not final_path and video_path:
            final_path = video_path

        if final_path:
            final_path = str(final_path).strip('"').strip("'")
            if not os.path.exists(final_path):
                try_path = os.path.join(folder_paths.get_input_directory(), final_path)
                if os.path.exists(try_path):
                    final_path = try_path
                else:
                    basename_path = os.path.join(folder_paths.get_input_directory(), os.path.basename(final_path))
                    if os.path.exists(basename_path):
                        final_path = basename_path

        return final_path

    def _safe_release(self, video_obj):
        if hasattr(video_obj, "release") and callable(getattr(video_obj, "release")):
            video_obj.release()
            return
        if hasattr(video_obj, "reset") and callable(getattr(video_obj, "reset")):
            video_obj.reset()

    def process(self, **kwargs):
        video = kwargs.get("video")
        video_path = kwargs.get("video_path", "").strip()

        final_path = self._resolve_video_path(video, video_path)
        if not final_path or not os.path.exists(final_path):
            raise ValueError("❌ 未找到视频，请连接视频输入或填写有效路径")

        # ----------------------
        # 依赖检查
        # ----------------------
        if not SCENEDETECT_AVAILABLE:
            raise ImportError("❌ 请安装：pip install scenedetect opencv-python-headless")
        if not cv2:
            raise ImportError("❌ 请安装 opencv")

        ffmpeg_ok, ffmpeg_path = check_ffmpeg()
        if not ffmpeg_ok:
            raise RuntimeError("❌ 缺少 FFmpeg，手动下载https://github.com/BtbN/FFmpeg-Builds/releases")

        # ----------------------
        # 场景检测
        # ----------------------
        try:
            video_obj = open_video(final_path)
            fps = video_obj.frame_rate
            total_frames = video_obj.duration.get_frames()
            min_scene_len = int(kwargs["min_scene_seconds"] * fps)

            scene_manager = SceneManager()
            mode = kwargs["detector_mode"]
            sens = kwargs["sensitivity"]

            if mode == "内容检测":
                scene_manager.add_detector(ContentDetector(threshold=sens))
            elif mode == "自适应检测":
                scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=sens))
            elif mode == "哈希检测":
                scene_manager.add_detector(HashDetector(threshold=sens))

            if kwargs["enable_fade_black"]:
                scene_manager.add_detector(ThresholdDetector(threshold=kwargs["black_threshold"], min_scene_len=min_scene_len))

            # 全版本兼容
            scene_manager.detect_scenes(video_obj, frame_skip=kwargs["frame_skip"])
            scenes = scene_manager.get_scene_list()

            if not scenes:
                raise ValueError("❌ 未检测到场景，请降低敏感度或关闭淡入淡出检测")

            split_mode = "按数量分割" if kwargs["Seg_mold"] else "自动分割"
            if kwargs["Seg_mold"]:
                scenes = self._adjust_to_target(scenes, kwargs["target_scene_count"], total_frames, fps)

            source_name = os.path.splitext(os.path.basename(final_path))[0]
            run_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir = os.path.join(kwargs["save_folder"], f"{source_name}_{run_tag}")
            video_out_dir = os.path.join(run_dir, "videos")
            image_out_dir = os.path.join(run_dir, "images")
            os.makedirs(video_out_dir, exist_ok=True)
            os.makedirs(image_out_dir, exist_ok=True)

            split_video_ffmpeg(final_path, scenes, output_dir=video_out_dir)

            cap = cv2.VideoCapture(final_path)
            if not cap.isOpened():
                raise RuntimeError("❌ 无法打开视频读取帧")
            max_frame_idx = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1, 0)
            images = []
            for i, (s_tc, e_tc) in enumerate(scenes):
                s = min(max(s_tc.get_frames(), 0), max_frame_idx)
                e = min(max(e_tc.get_frames() - 1, s), max_frame_idx)

                cap.set(cv2.CAP_PROP_POS_FRAMES, s)
                ret1, frm1 = cap.read()
                if ret1:
                    rgb1 = cv2.cvtColor(frm1, cv2.COLOR_BGR2RGB)
                    images.append(torch.from_numpy(pil2tensor(rgb1)))
                    Image.fromarray(rgb1).save(os.path.join(image_out_dir, f"scene_{i:03d}_start_{s:06d}.png"))

                cap.set(cv2.CAP_PROP_POS_FRAMES, e)
                ret2, frm2 = cap.read()
                if ret2:
                    rgb2 = cv2.cvtColor(frm2, cv2.COLOR_BGR2RGB)
                    images.append(torch.from_numpy(pil2tensor(rgb2)))
                    Image.fromarray(rgb2).save(os.path.join(image_out_dir, f"scene_{i:03d}_end_{e:06d}.png"))

            cap.release()
            self._safe_release(video_obj)

            if not images:
                raise ValueError("❌ 场景已检测到，但未成功提取关键帧")

            return (torch.stack(images), f"✅ 完成！模式：{split_mode}，分割 {len(scenes)} 段，预览帧 {len(images)} 张，输出目录：{run_dir}")

        except Exception as e:
            traceback.print_exc()
            raise RuntimeError(f"❌ 错误：{str(e)}")

    def _adjust_to_target(self, scenes, target, total_frames, fps=24):
        n = len(scenes)
        if n == target:
            return scenes
        if n > target:
            while len(scenes) > target:
                pairs = list(zip(scenes, scenes[1:]))
                gaps = [p[1][0].get_frames() - p[0][1].get_frames() for p in pairs]
                idx = gaps.index(min(gaps))
                merged = (scenes[idx][0], scenes[idx+1][1])
                scenes = scenes[:idx] + [merged] + scenes[idx+2:]
            return scenes
        else:
            new_scenes = []
            step = total_frames / target
            for i in range(target):
                s = FrameTimecode(int(i * step), fps)
                e = FrameTimecode(int((i+1) * step), fps)
                new_scenes.append((s, e))
            return new_scenes


class AD_AutoTileVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 32768, "step": 8}),
                "total_frames": ("INT", {"default": 64, "min": 1, "max": 4096, "step": 1}),
                "lowGpu_mode": ("BOOLEAN", {"default": False, "display_name": "Low GPU Mode"}),
                "temporal_compression": ("INT", {"default": 8, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("tile_size", "overlap", "temporal_size", "temporal_overlap", "temporal_compression")
    FUNCTION = "recommend"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def recommend(self, width, height, total_frames, lowGpu_mode, temporal_compression, vae=None):
        width = max(64, int(width))
        height = max(64, int(height))
        total_frames = max(1, int(total_frames))
        temporal_compression = int(temporal_compression)
        if temporal_compression <= 0 and vae is not None:
            read_tc = vae.temporal_compression_decode()
            if read_tc is not None:
                temporal_compression = int(read_tc)
        temporal_compression = max(1, temporal_compression)
        short_edge = max(64, min(width, height))
        megapixels = (width * height) / 1000000.0

        mode = "显存优先" if lowGpu_mode else "质量优先"
        vram_level = "12GB" if lowGpu_mode else "≥24GB"

        if vram_level == "12GB":
            base_tile = 512
            if megapixels >= 24:
                base_tile = 448
            base_effective_temporal = 8
        elif vram_level == "16GB":
            base_tile = 640
            if megapixels >= 24:
                base_tile = 576
            base_effective_temporal = 10
        elif vram_level == "20GB":
            base_tile = 768
            if megapixels >= 24:
                base_tile = 640
            base_effective_temporal = 12
        else:
            base_tile = 1024
            if megapixels >= 24:
                base_tile = 896
            if megapixels >= 48:
                base_tile = 768
            base_effective_temporal = 16

        if megapixels >= 48:
            base_effective_temporal = max(4, base_effective_temporal - 4)
        elif megapixels >= 24:
            base_effective_temporal = max(4, base_effective_temporal - 2)

        if mode == "显存优先":
            max_effective_frames = max(2, total_frames // temporal_compression)
            effective_temporal = min(base_effective_temporal, max_effective_frames)
            temporal_size = max(8, effective_temporal * temporal_compression)
            temporal_size = (temporal_size // 4) * 4
            max_temporal_size_by_frames = max(8, (total_frames // 4) * 4)
            temporal_size = min(temporal_size, max_temporal_size_by_frames)
            temporal_size = max(8, temporal_size)
            effective_temporal = max(2, temporal_size // temporal_compression)
            temporal_pressure = math.sqrt(max(1.0, effective_temporal / 8.0))
            base_tile = int(base_tile / temporal_pressure)
            temporal_overlap = ((temporal_size // 8) // 4) * 4
            if temporal_size >= 16:
                temporal_overlap = max(4, temporal_overlap)
            temporal_overlap = min(temporal_overlap, 64)
        else:
            if vram_level == "12GB":
                temporal_size = 512
            elif vram_level == "16GB":
                temporal_size = 1024
            elif vram_level == "20GB":
                temporal_size = 2048
            else:
                temporal_size = 4096
            temporal_size = max(8, min(4096, temporal_size))
            temporal_size = (temporal_size // 4) * 4
            temporal_overlap = 64

        tile_size = min(base_tile, short_edge)
        tile_size = max(64, (tile_size // 32) * 32)

        overlap = ((tile_size // 8) // 32) * 32
        if tile_size >= 128:
            overlap = max(32, overlap)
        overlap = min(overlap, 160)

        max_overlap = (tile_size // 4 // 32) * 32
        if overlap > max_overlap:
            overlap = max_overlap
        overlap = max(0, overlap)

        max_temporal_overlap = (temporal_size // 2 // 4) * 4
        if temporal_overlap > max_temporal_overlap:
            temporal_overlap = max_temporal_overlap
        temporal_overlap = max(4, temporal_overlap)

        return (tile_size, overlap, temporal_size, temporal_overlap, temporal_compression)






#region----------MiniMax H3---------------

try:
    import torchaudio as _torchaudio
except ImportError:
    _torchaudio = None

try:
    import node_helpers as _node_helpers
except ImportError:
    _node_helpers = None

from .minimaxH3 import AptMiniMaxH3MotionContext, AptMiniMaxH3NativeAudioLock, h3_keyframe_anchor

# 复用 H3 节点工具函数（若 comfy_extras 中不存在 H3 模块则全部置空，节点将在执行时报错）
try:
    from comfy_extras.nodes_minimax_h3 import (
        _empty_av_latent as _h3_empty_av_latent,
        _resize as _h3_resize,
        adapt_canvas as _h3_adapt_canvas,
        REF_IMAGE_SHORT_EDGE as _H3_REF_IMAGE_SHORT_EDGE,
        CANVAS_MULTIPLE as _H3_CANVAS_MULTIPLE,
        FPS as _H3_FPS,
    )
except ImportError:
    _h3_empty_av_latent = None
    _h3_resize = None
    _h3_adapt_canvas = None
    _H3_REF_IMAGE_SHORT_EDGE = 2048
    _H3_CANVAS_MULTIPLE = 32
    _H3_FPS = 24


class AD_sam_Crop:
    SMOOTHING_PRESETS = {
        "balanced": (21, 51, "gaussian"),
        "stable_max": (41, 91, "gaussian"),
        "stable_extreme": (71, 131, "gaussian"),
        "cinematic_push": (17, 35, "savgol"),
        "responsive": (9, 21, "savgol"),
        "static_shot": (51, 151, "gaussian"),
        "cg_animation": (5, 11, "gaussian"),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "detection_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_objects": ("INT", {"default": 4, "min": 0, "max": 64, "step": 1}),
                "detect_interval": ("INT", {"default": 1, "min": 1, "max": 10000, "step": 1}),
                "ckpt_name": (folder_paths.get_filename_list("checkpoints"), {"default": "sam3.1_multiplex_fp16.safetensors"}),
                "pos": ("STRING", {"default": "", "multiline": True}),
                "crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "crop_width": ("INT", {"default": 512, "min": 128, "max": 1344, "step": 32}),
                "crop_height": ("INT", {"default": 384, "min": 128, "max": 1344, "step": 32}),
                "smoothing_preset": (list(cls.SMOOTHING_PRESETS), {
                    "default": "balanced",
                    "tooltip":
                        "  balanced       : 通用平衡档，适合 80% 的素材。\n"
                        "  stable_max     : 强抗抖，适合三脚架/稳定器/采访镜头。\n"
                        "  stable_extreme : 极端抗抖，适合夜景、低码率、720p 以下、老手机这类检测框抖动严重的素材。\n"
                        "  cinematic_push : 保留推镜/拉镜节奏，适合广告、MV、电影感镜头（savgol 中窗口）。\n"
                        "  responsive     : 高灵敏度跟随，适合手持快速转头、快速运镜、动作幅度大的素材。\n"
                        "  static_shot    : 锁死三脚架/产品照，size 平滑窗口开到最大（151 帧），轨迹最稳。\n"
                        "  cg_animation   : 最小平滑，用于 CG 动画 / 游戏录屏这类本身无像素噪声、检测框极稳的渲染素材。",
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "H3FACEXFORM", "SAM3_TRACK_DATA", "MASK")
    RETURN_NAMES = ("crop_img", "transform", "track_data", "masks")
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"
    DESCRIPTION = "SAM3 video tracking with fixed-size per-frame crops and an H3 transform."

    @staticmethod
    def _interpolate(values, valid):
        indices = np.arange(len(values))
        return np.interp(indices, indices[valid], values[valid])

    @staticmethod
    def _smooth(values, window, method="gaussian"):
        if window <= 1 or len(values) < 3:
            return values
        window = min(int(window), len(values))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return values
        padding = window // 2
        padded = np.pad(values, padding, mode="reflect")
        if method == "savgol":
            polyorder = 2 if window > 3 else 1
            return np.asarray(savgol_filter(padded, window, polyorder))[padding:padding + len(values)]
        if method == "gaussian":
            x = np.arange(window, dtype=np.float64) - padding
            sigma = max(window / 6.0, 0.5)
            kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
            kernel /= kernel.sum()
        return np.convolve(padded, kernel, mode="valid")

    @classmethod
    def _build_transform(cls, masks, image, crop_factor, canvas_width, canvas_height,
                         smooth_window, size_smooth_window, smooth_method):
        frames, src_height, src_width, _ = image.shape
        binary = masks > 0.5
        rows = binary.any(dim=2).detach().cpu().numpy()
        columns = binary.any(dim=1).detach().cpu().numpy()

        center_x = np.zeros(frames, dtype=np.float64)
        center_y = np.zeros(frames, dtype=np.float64)
        object_width = np.zeros(frames, dtype=np.float64)
        object_height = np.zeros(frames, dtype=np.float64)
        detected = np.zeros(frames, dtype=bool)

        for i in range(frames):
            ys = np.flatnonzero(rows[i])
            xs = np.flatnonzero(columns[i])
            if len(xs) == 0 or len(ys) == 0:
                continue
            x0, x1 = float(xs[0]), float(xs[-1] + 1)
            y0, y1 = float(ys[0]), float(ys[-1] + 1)
            center_x[i] = (x0 + x1) * 0.5
            center_y[i] = (y0 + y1) * 0.5
            object_width[i] = x1 - x0
            object_height[i] = y1 - y0
            detected[i] = True

        if not detected.any():
            raise ValueError("SAM3 did not detect the requested object in any frame. Lower detection_threshold or change pos.")

        center_x = cls._interpolate(center_x, detected)
        center_y = cls._interpolate(center_y, detected)
        object_width = cls._interpolate(object_width, detected)
        object_height = cls._interpolate(object_height, detected)
        center_x = cls._smooth(center_x, smooth_window, smooth_method)
        center_y = cls._smooth(center_y, smooth_window, smooth_method)
        object_width = cls._smooth(object_width, size_smooth_window, smooth_method)
        object_height = cls._smooth(object_height, size_smooth_window, smooth_method)

        valid_weights = detected.astype(np.float64)
        weight_window = max(9, int(smooth_window) // 2)
        weights = np.clip(cls._smooth(valid_weights, weight_window, "gaussian"), 0.0, 1.0)

        aspect = canvas_width / float(canvas_height)
        boxes = []
        object_rects = []
        for i in range(frames):
            crop_height = max(object_height[i], object_width[i] / aspect) * crop_factor
            crop_width = crop_height * aspect
            if crop_width > src_width:
                crop_width = float(src_width)
                crop_height = crop_width / aspect
            if crop_height > src_height:
                crop_height = float(src_height)
                crop_width = crop_height * aspect

            x = min(max(center_x[i] - crop_width * 0.5, 0.0), max(0.0, src_width - crop_width))
            y = min(max(center_y[i] - crop_height * 0.5, 0.0), max(0.0, src_height - crop_height))
            boxes.append((float(x), float(y), float(crop_width), float(crop_height)))

            object_x = center_x[i] - object_width[i] * 0.5
            object_y = center_y[i] - object_height[i] * 0.5
            object_rects.append((
                float((object_x - x) / crop_width * canvas_width),
                float((object_y - y) / crop_height * canvas_height),
                float(object_width[i] / crop_width * canvas_width),
                float(object_height[i] / crop_height * canvas_height),
            ))

        return {
            "boxes": boxes,
            "canvas": (int(canvas_width), int(canvas_height)),
            "src_size": (int(src_width), int(src_height)),
            "frames": int(frames),
            "source_img": image,
            "weights": [float(value) for value in weights],
            "detected": [bool(value) for value in detected],
            "face_rect": object_rects,
            "object_rect": object_rects,
            "crop_factor": float(crop_factor),
        }

    @staticmethod
    def _crop_images(image, transform):
        frames, src_height, src_width, _ = image.shape
        canvas_width, canvas_height = transform["canvas"]
        theta = torch.empty((frames, 2, 3), dtype=torch.float32, device=image.device)
        for i, (x, y, crop_width, crop_height) in enumerate(transform["boxes"]):
            theta[i, 0, 0] = crop_width / src_width
            theta[i, 0, 1] = 0.0
            theta[i, 0, 2] = (2.0 * x + crop_width) / src_width - 1.0
            theta[i, 1, 0] = 0.0
            theta[i, 1, 1] = crop_height / src_height
            theta[i, 1, 2] = (2.0 * y + crop_height) / src_height - 1.0

        source = image[..., :3].movedim(-1, 1).float()
        grid = F.affine_grid(theta, (frames, 3, canvas_height, canvas_width), align_corners=False)
        crops = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return crops.movedim(1, -1).to(image.dtype)

    def execute(self, image, detection_threshold, max_objects, detect_interval, ckpt_name, pos,
                crop_factor, crop_width, crop_height, smoothing_preset):
        model, conditioning = mask_sam_detctor._load_model_cached(ckpt_name, pos)
        if model is None:
            raise RuntimeError(f"Unable to load SAM3 checkpoint: {ckpt_name}")
        if conditioning is None:
            raise ValueError("pos must contain the object description used for SAM3 video tracking.")

        from comfy_extras.nodes_sam3 import SAM3_TrackToMask, SAM3_VideoTrack

        track_data = SAM3_VideoTrack.execute(
            images=image,
            model=model,
            conditioning=conditioning,
            detection_threshold=float(detection_threshold),
            max_objects=int(max_objects),
            detect_interval=int(detect_interval),
        )[0]
        masks = SAM3_TrackToMask.execute(track_data=track_data)[0]
        smooth_window, size_smooth_window, smooth_method = self.SMOOTHING_PRESETS[smoothing_preset]
        transform = self._build_transform(
            masks, image, crop_factor, crop_width, crop_height,
            smooth_window, size_smooth_window, smooth_method
        )
        crop_img = self._crop_images(image, transform)
        return (crop_img, transform, track_data, masks)


class AD_sam_stitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stitch_img": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
                "paste_region": (["obj_only", "obj_ellipse", "full_crop"], {"default": "full_crop"}),
                "mask_dilation": ("INT", {"default": 16, "min": 0, "max": 256, "step": 2}),
                "feather": ("INT", {"default": 6, "min": 0, "max": 256, "step": 2}),
                "colour_match": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, }),
                "undetected_frames": (["fade_out", "skip", "composite_anyway"], {"default": "fade_out"}),

            },
            "optional": {
                "masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"
    DESCRIPTION = "Paste full refined crops back into their tracked source-frame positions."

    @staticmethod
    def _feather_mask(height, width, feather, device):
        mask = torch.ones((height, width), device=device, dtype=torch.float32)
        size = int(max(0, min(feather, min(height, width) // 2 - 1)))
        if size <= 0:
            return mask
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, np.pi, size + 2, device=device, dtype=torch.float32)[1:-1]
        )
        mask[:size, :] *= ramp.view(-1, 1)
        mask[height - size:, :] *= ramp.flip(0).view(-1, 1)
        mask[:, :size] *= ramp.view(1, -1)
        mask[:, width - size:] *= ramp.flip(0).view(1, -1)
        return mask

    @staticmethod
    def _blur_mask(mask, feather):
        if feather <= 0:
            return mask
        kernel_size = 2 * int(feather) + 1
        shortest = min(mask.shape[-2], mask.shape[-1])
        if shortest <= kernel_size:
            kernel_size = max(3, int(shortest / 2) | 1)
        sigma = max(kernel_size / 6.0, 0.5)
        x = torch.arange(kernel_size, device=mask.device, dtype=torch.float32) - kernel_size // 2
        kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
        kernel = (kernel / kernel.sum()).to(mask.dtype)
        padding = kernel_size // 2
        mask = F.conv2d(F.pad(mask, (padding, padding, 0, 0), mode="replicate"), kernel.view(1, 1, 1, kernel_size))
        return F.conv2d(F.pad(mask, (0, 0, padding, padding), mode="replicate"), kernel.view(1, 1, kernel_size, 1))

    @classmethod
    def _region_mask(cls, height, width, rect, dilation, feather, ellipse, device):
        mask = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
        x, y, rect_width, rect_height = rect
        x -= dilation
        y -= dilation
        rect_width += 2 * dilation
        rect_height += 2 * dilation
        if ellipse:
            yy = torch.arange(height, device=device, dtype=torch.float32).view(-1, 1)
            xx = torch.arange(width, device=device, dtype=torch.float32).view(1, -1)
            center_x, center_y = x + rect_width * 0.5, y + rect_height * 0.5
            radius_x, radius_y = max(rect_width * 0.5, 1.0), max(rect_height * 0.5, 1.0)
            mask[0, 0] = (((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1.0).float()
        else:
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1 = min(width, int(round(x + rect_width)))
            y1 = min(height, int(round(y + rect_height)))
            if x1 > x0 and y1 > y0:
                mask[0, 0, y0:y1, x0:x1] = 1.0
        return cls._blur_mask(mask, feather).clamp(0, 1)

    def execute(self, stitch_img, transform, paste_region, mask_dilation, feather,
                colour_match, undetected_frames, masks=None, blend=1.0):
        source_img = transform["source_img"]
        boxes = transform["boxes"]
        canvas_width, canvas_height = transform["canvas"]
        src_width, src_height = transform["src_size"]
        if (source_img.shape[2], source_img.shape[1]) != (src_width, src_height):
            raise ValueError(
                f"source_img is {source_img.shape[2]}x{source_img.shape[1]}, but transform expects "
                f"{src_width}x{src_height}."
            )
        if (stitch_img.shape[2], stitch_img.shape[1]) != (canvas_width, canvas_height):
            raise ValueError(
                f"stitch_img is {stitch_img.shape[2]}x{stitch_img.shape[1]}, but transform expects "
                f"{canvas_width}x{canvas_height}."
            )

        if undetected_frames == "composite_anyway":
            weights = None
        elif undetected_frames == "skip":
            weights = [1.0 if value else 0.0 for value in transform.get("detected", [])] or None
        else:
            weights = transform.get("weights")

        frames = min(len(boxes), source_img.shape[0], stitch_img.shape[0])
        if masks is not None:
            frames = min(frames, masks.shape[0])
        output = source_img[..., :3].clone()
        device = comfy.model_management.get_torch_device()
        per_frame_mb = src_height * src_width * 3 * 4 / 2 ** 20
        chunk_size = max(1, min(32, int(1024 / max(per_frame_mb, 1e-6))))

        for start in range(0, frames, chunk_size):
            comfy.model_management.throw_exception_if_processing_interrupted()
            end = min(start + chunk_size, frames)
            count = end - start
            crop_height = float(boxes[(start + end - 1) // 2][3])
            canvas_feather = int(round(feather * canvas_height / max(crop_height, 1.0)))
            if feather > 0:
                canvas_feather = max(1, min(canvas_feather, canvas_height // 3))

            theta = torch.empty((count, 2, 3), dtype=torch.float32, device=device)
            crop_theta = torch.empty((count, 2, 3), dtype=torch.float32, device=device)
            for local_index, frame_index in enumerate(range(start, end)):
                x, y, width, height = (float(value) for value in boxes[frame_index])
                theta[local_index, 0, 0] = src_width / width
                theta[local_index, 0, 1] = 0.0
                theta[local_index, 0, 2] = (src_width - 2.0 * x) / width - 1.0
                theta[local_index, 1, 0] = 0.0
                theta[local_index, 1, 1] = src_height / height
                theta[local_index, 1, 2] = (src_height - 2.0 * y) / height - 1.0
                crop_theta[local_index, 0, 0] = width / src_width
                crop_theta[local_index, 0, 1] = 0.0
                crop_theta[local_index, 0, 2] = (2.0 * x + width) / src_width - 1.0
                crop_theta[local_index, 1, 0] = 0.0
                crop_theta[local_index, 1, 1] = height / src_height
                crop_theta[local_index, 1, 2] = (2.0 * y + height) / src_height - 1.0

            if masks is not None:
                source_mask = masks[start:end].to(device).float().unsqueeze(1)
                if source_mask.shape[-2:] != (src_height, src_width):
                    source_mask = F.interpolate(source_mask, size=(src_height, src_width), mode="bilinear", align_corners=False)
                crop_grid = F.affine_grid(
                    crop_theta, (count, 1, canvas_height, canvas_width), align_corners=False
                )
                canvas_mask = F.grid_sample(
                    source_mask, crop_grid, mode="bilinear", padding_mode="zeros", align_corners=False
                )
                if mask_dilation > 0:
                    kernel_size = 2 * int(mask_dilation) + 1
                    canvas_mask = F.max_pool2d(canvas_mask, kernel_size, stride=1, padding=kernel_size // 2)
                canvas_mask = self._blur_mask(canvas_mask, canvas_feather).clamp(0, 1)
            elif paste_region == "full_crop":
                one = self._feather_mask(canvas_height, canvas_width, canvas_feather, device)
                canvas_mask = one.view(1, 1, canvas_height, canvas_width).expand(count, 1, -1, -1)
            else:
                object_rects = transform.get("object_rect", transform.get("face_rect"))
                canvas_mask = torch.cat([
                    self._region_mask(
                        canvas_height,
                        canvas_width,
                        object_rects[frame_index] if object_rects and frame_index < len(object_rects)
                        else (canvas_width * 0.25, canvas_height * 0.25, canvas_width * 0.5, canvas_height * 0.5),
                        int(mask_dilation),
                        canvas_feather,
                        paste_region == "obj_ellipse",
                        device,
                    )
                    for frame_index in range(start, end)
                ], dim=0)

            grid = F.affine_grid(theta, (count, 3, src_height, src_width), align_corners=False)
            patch = stitch_img[start:end, ..., :3].to(device).movedim(-1, 1).float()
            patch = F.grid_sample(patch, grid, mode="bilinear", padding_mode="border", align_corners=False)
            mask = F.grid_sample(canvas_mask, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
            patch = patch.movedim(1, -1)
            mask = mask.clamp(0, 1).movedim(1, -1)
            base = output[start:end].to(device).float()

            if colour_match > 0.0:
                weight_sum = mask.sum(dim=(1, 2), keepdim=True).clamp_min(1e-6)
                base_mean = (base * mask).sum(dim=(1, 2), keepdim=True) / weight_sum
                patch_mean = (patch * mask).sum(dim=(1, 2), keepdim=True) / weight_sum
                base_std = (((base - base_mean) ** 2 * mask).sum(dim=(1, 2), keepdim=True) / weight_sum).sqrt().clamp_min(1e-6)
                patch_std = (((patch - patch_mean) ** 2 * mask).sum(dim=(1, 2), keepdim=True) / weight_sum).sqrt().clamp_min(1e-6)
                matched = (patch - patch_mean) * (base_std / patch_std) + base_mean
                patch = (patch + (matched - patch) * float(colour_match)).clamp(0, 1)

            frame_weights = torch.full(
                (count, 1, 1, 1), float(blend), device=device, dtype=torch.float32
            )
            if weights is not None:
                for local_index, frame_index in enumerate(range(start, end)):
                    if frame_index < len(weights):
                        frame_weights[local_index] *= float(weights[frame_index])
            mask = mask * frame_weights

            output[start:end] = ((1.0 - mask) * base + mask * patch).to(output.device, output.dtype)

        return (output,)


class AD_Inject_Latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "av_latent": ("LATENT",),
                "vae": ("VAE",),
                "crop_img": ("IMAGE",),
                "transform": ("H3FACEXFORM",),
                "smooth_frames": ("INT", {"default": 9, "min": 1, "max": 61, "step": 2,
                    "tooltip": "Temporally smooths the per-frame strength curve to avoid visible texture pops."}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("av_latent",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/AD"
    DESCRIPTION = "Inject tracked object crops and adapt per-frame denoise to object size."

    @staticmethod
    def _smooth_weights(values, window):
        values = np.asarray(values, dtype=np.float64)
        if window <= 1 or len(values) < 3:
            return values
        window = min(int(window), len(values))
        if window % 2 == 0:
            window += 1
        padding = window // 2
        padded = np.pad(values, padding, mode="reflect")
        x = np.arange(window, dtype=np.float64) - padding
        sigma = max(window / 6.0, 0.5)
        kernel = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
        kernel /= kernel.sum()
        return np.convolve(padded, kernel, mode="valid")[:len(values)]

    def execute(self, av_latent, vae, crop_img, transform, smooth_frames):
        if not isinstance(transform, collections.abc.Mapping):
            raise TypeError("transform must be an H3FACEXFORM mapping")
        canvas_width, canvas_height = transform["canvas"]
        if (crop_img.shape[2], crop_img.shape[1]) != (canvas_width, canvas_height):
            raise ValueError(
                f"crop_img is {crop_img.shape[2]}x{crop_img.shape[1]}, but transform expects "
                f"{canvas_width}x{canvas_height}."
            )
        if int(transform.get("frames", crop_img.shape[0])) != crop_img.shape[0]:
            raise ValueError(
                f"crop_img has {crop_img.shape[0]} frames, but transform expects "
                f"{transform['frames']}."
            )

        samples = av_latent.get("samples")
        if samples is None or not (
            isinstance(samples, comfy.nested_tensor.NestedTensor)
            or getattr(samples, "is_nested", False)
        ):
            raise ValueError("Expected a MiniMax H3 joint AV latent (NestedTensor).")

        members = list(samples.unbind())
        video_template = members[0]
        encoded = vae.encode(crop_img[..., :3])
        if encoded.ndim == 4:
            encoded = encoded.unsqueeze(0).movedim(1, 2)

        target_t, target_h, target_w = video_template.shape[-3:]
        encoded_t, encoded_h, encoded_w = encoded.shape[-3:]
        if (encoded_h, encoded_w) != (target_h, target_w):
            raise ValueError(
                f"Encoded crop latent is {encoded_w}x{encoded_h}, but the AV latent expects "
                f"{target_w}x{target_h}. Ensure the H3 width/height matches crop_img."
            )
        if encoded_t > target_t:
            encoded = encoded[..., :target_t, :, :]
        elif encoded_t < target_t:
            padding = video_template[..., : target_t - encoded_t, :, :].to(encoded.device, encoded.dtype)
            encoded = torch.cat((encoded, padding), dim=-3)

        members[0] = encoded.to(video_template.device, video_template.dtype)

        weights = transform.get("weights")
        if weights is None:
            detected = transform.get("detected")
            weights = detected if detected is not None else [1.0] * crop_img.shape[0]
        if len(weights) != crop_img.shape[0]:
            raise ValueError(
                f"transform has {len(weights)} frame weights, but crop_img has {crop_img.shape[0]} frames."
            )

        boxes = transform["boxes"]
        object_rects = transform.get("face_rect")
        if object_rects is not None and len(object_rects) == len(boxes):
            object_size = np.array([
                max(
                    rect[2] / canvas_width * box[2],
                    rect[3] / canvas_height * box[3],
                )
                for rect, box in zip(object_rects, boxes)
            ], dtype=np.float64)
        else:
            crop_factor = float(transform.get("crop_factor", 3.0)) or 3.0
            object_size = np.array([box[3] / crop_factor for box in boxes], dtype=np.float64)
        if len(object_size) != crop_img.shape[0]:
            raise ValueError(
                f"transform has {len(object_size)} object sizes, but crop_img has {crop_img.shape[0]} frames."
            )

        size_ratio = np.clip((object_size - 30.0) / 90.0, 0.0, 1.0)
        strength_values = 1.0 - 0.65 * size_ratio
        strength_values *= np.clip(np.asarray(weights, dtype=np.float64), 0.0, 1.0)
        strength_values = self._smooth_weights(strength_values, smooth_frames)
        strength_values = np.clip(strength_values, 0.0, 1.0)
        strength = torch.from_numpy(strength_values).float().view(1, 1, -1)
        strength = F.interpolate(strength, size=int(target_t), mode="linear", align_corners=True)
        strength = strength.view(1, 1, int(target_t), 1, 1).to(video_template.device)
        video_mask = strength.expand(
            video_template.shape[0], video_template.shape[1], target_t,
            video_template.shape[-2], video_template.shape[-1],
        ).contiguous()

        previous_mask = av_latent.get("noise_mask")
        if previous_mask is not None and (
            isinstance(previous_mask, comfy.nested_tensor.NestedTensor)
            or getattr(previous_mask, "is_nested", False)
        ):
            mask_members = list(previous_mask.unbind())
            mask_members[0] = video_mask.to(mask_members[0].dtype)
        else:
            mask_members = [video_mask.to(video_template.dtype)]
            mask_members.extend(torch.zeros_like(member) for member in members[1:])

        output = dict(av_latent)
        output["samples"] = comfy.nested_tensor.NestedTensor(tuple(members))
        output["noise_mask"] = comfy.nested_tensor.NestedTensor(tuple(mask_members))
        return (output,)


class AD_MiniMax_Ref2V:
 
    CATEGORY = "Apt_Preset/AD/😺backup"
    FUNCTION = "execute"

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "clip": ("CLIP",),
            "vae": ("VAE",),
            "audio_vae": ("VAE",),
            "prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "dynamicPrompts": True,
                "socketless": True,
            }),
            "width": ("INT", {
                "default": 512, "min": 32, "max": 4096, "step": 32,
            }),
            "height": ("INT", {
                "default": 768, "min": 32, "max": 4096, "step": 32,
            }),
            "length": ("INT", {
                "default": 124, "min": 5, "max": 3600, "step": 17,
                "tooltip": "帧数（24 fps），自动 snap 到 17k+5 网格（124 ≈ 5s）",
            }),
            "ref_image_size": (["match", "max"], {
                "default": "match",
                "tooltip": "'match' 缩放到画布面积；'max' 对齐 2048 短边（身份更好但更慢）",
            }),
        }
        optional = {}
        # 9 个参考图片
        for i in range(9):
            optional[f"ref_image_{i}"] = ("IMAGE",)
        # 3 个参考视频
        for i in range(3):
            optional[f"ref_video_{i}"] = ("IMAGE",)
        # 3 个视频配对音轨
        for i in range(3):
            optional[f"ref_video_audio_{i}"] = ("AUDIO",)
        # 3 个独立参考音频
        for i in range(3):
            optional[f"ref_audio_{i}"] = ("AUDIO",)
        return {"required": required, "optional": optional}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _encode_ref_audio(audio_vae, audio):
        waveform = audio["waveform"]  # [B, C, L]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = _torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))  # [1, 32, 2, T]
        return z, z.shape[-1]

    # ----------------------------------------------------------------- execute
    def execute(
        self,
        clip, vae, audio_vae, prompt,
        width, height, length, ref_image_size="match",
        **kwargs,
    ):
        # 收集固定端口为 dict（与 Autogrow 模板输出同构）
        ref_images = {}
        ref_videos = {}
        ref_video_audios = {}
        ref_audios = {}
        for i in range(9):
            v = kwargs.get(f"ref_image_{i}")
            if v is not None:
                ref_images[f"ref_image_{i}"] = v
        for i in range(3):
            v = kwargs.get(f"ref_video_{i}")
            if v is not None:
                ref_videos[f"ref_video_{i}"] = v
        for i in range(3):
            v = kwargs.get(f"ref_video_audio_{i}")
            if v is not None:
                ref_video_audios[f"ref_video_audio_{i}"] = v
        for i in range(3):
            v = kwargs.get(f"ref_audio_{i}")
            if v is not None:
                ref_audios[f"ref_audio_{i}"] = v

        latent, frame_count = _h3_empty_av_latent(width, height, length)

        ref_items = []   # 给 tokenizer 用，按请求顺序
        ref_blocks = []  # 给 DiT payload 用，同顺序

        # ---- 参考图片 ----
        for img in ref_images.values():
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, _H3_REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(_H3_CANVAS_MULTIPLE, round(w * scale / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
            th = max(_H3_CANVAS_MULTIPLE, round(h * scale / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
            resized = _h3_resize(img[:1], tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        # ---- 参考视频（含配对音轨）----
        for name, video_frames in ref_videos.items():
            # 通过名称后缀配对：ref_video_audio_N ↔ ref_video_N
            soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
            vh, vw = video_frames.shape[1], video_frames.shape[2]
            cw, ch = _h3_adapt_canvas(vw, vh)
            if vw * vh < cw * ch:
                cw = max(_H3_CANVAS_MULTIPLE, round(vw / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
                ch = max(_H3_CANVAS_MULTIPLE, round(vh / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
            frames = _h3_resize(video_frames, cw, ch, "disabled")
            if frames.shape[0] > frame_count:
                frames = frames[:frame_count]
            n = frames.shape[0]
            if n < 5:
                raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
            while n % 17 != 5:
                n -= 1
            frames = frames[:n]
            z = vae.encode(frames)
            audio_latent, ref_audio_t = (None, 0)
            if soundtrack is not None:
                audio_latent, ref_audio_t = self._encode_ref_audio(audio_vae, soundtrack)
                ref_items.append({"type": "audio"})
            # Qwen 以 2 fps 采样视频并附加时间戳
            sample_idx = list(range(0, frames.shape[0], _H3_FPS // 2))
            qwen_frames = frames[sample_idx]
            ref_items.append({"type": "video", "data": qwen_frames,
                              "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
            ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                               "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                               "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

        # ---- 独立参考音频 ----
        for audio in ref_audios.values():
            audio_latent, ref_audio_t = self._encode_ref_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t, "audio_latent": audio_latent})

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        if ref_blocks:
            cond = _node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})

        return (cond, latent)


#endregion----------MiniMax H3---------------


#region----------MiniMax H3 Guide---------------

_AD_GUIDE_MAX_MEDIA = 16
_AD_GUIDE_MAX_STAGE_TEXT = 64
_AD_GUIDE_MAX_REFERENCES = 15
_AD_GUIDE_MAX_IMAGES = 9
_AD_GUIDE_MAX_VIDEOS = 3
_AD_GUIDE_MAX_AUDIOS = 3
_AD_GUIDE_CONTEXT_LENGTH = 22
_AD_GUIDE_AUDIO_CONTEXT_LENGTH = 24
_AD_GUIDE_TRIM_FRAMES = _AD_GUIDE_CONTEXT_LENGTH
_AD_GUIDE_PLACEHOLDER_RE = re.compile(r"__AD_MINIMAX_GUIDE_REF_(\d+)__")
_AD_GUIDE_UNRESOLVED_RE = re.compile(r"__AD_MINIMAX_GUIDE_UNRESOLVED_REF_[^_]+__")
_AD_FL2_PICTURE_RE = re.compile(r"(?<!\w)(?:<\s*)?Pic(?:ture)?\s+([12])(?:\s*>)?(?!\w)", re.IGNORECASE)
_AD_FL2_SHOT_RE = re.compile(r"\[Shot\s+(\d+)\]", re.IGNORECASE)
_AD_GUIDE_TAG_SEPARATOR = r"[\t \u3000#＃_\-－–—?？]*"
_AD_GUIDE_MEDIA_ALIASES = {
    "p": "image", "pic": "image", "picture": "image", "image": "image", "img": "image", "refimg": "image", "refpic": "image",
    "图": "image", "图片": "image", "图像": "image",
    "v": "video", "vid": "video", "video": "video", "clip": "video", "movie": "video", "refvid": "video", "视频": "video", "影片": "video",
    "a": "audio", "aud": "audio", "audio": "audio", "sound": "audio", "bgm": "audio", "refaud": "audio", "音频": "audio", "声音": "audio", "语音": "audio",
}


def _ad_guide_marked_tag_re(aliases):
    keywords = "|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True))
    return re.compile(
        rf"(?:(?P<at>[@＠])|(?P<open>[<\[({{]))[\t \u3000]*(?P<kind>{keywords}){_AD_GUIDE_TAG_SEPARATOR}"
        rf"(?P<number>[0-9０-９]+)(?(at)|[\t \u3000]*(?P<close>[>\])}}]))(?![0-9０-９A-Za-z_])",
        re.IGNORECASE,
    )


_AD_GUIDE_MEDIA_TAG_RE = _ad_guide_marked_tag_re(_AD_GUIDE_MEDIA_ALIASES)
_AD_GUIDE_SHOT_TAG_RE = _ad_guide_marked_tag_re({"shot": "shot", "镜头": "shot", "分镜": "shot", "镜": "shot"})
_AD_GUIDE_SUBJECT_TAG_RE = _ad_guide_marked_tag_re({"subject": "subject", "主体": "subject", "角色": "subject", "人物": "subject"})


def _ad_guide_media_type(value):
    if value is None:
        return ""
    if isinstance(value, torch.Tensor):
        return "image"
    if isinstance(value, collections.abc.Mapping) and "samples" in value:
        return "latent"
    if isinstance(value, collections.abc.Mapping) and "waveform" in value:
        return "audio"
    if hasattr(value, "get_components"):
        return "video"
    if isinstance(value, collections.abc.Mapping) and ("images" in value or "frames" in value):
        return "video"
    return ""


def _ad_guide_video_parts(value):
    if hasattr(value, "get_components"):
        components = value.get_components()
        return components.images, components.audio, float(components.frame_rate or _H3_FPS)
    if isinstance(value, collections.abc.Mapping):
        frames = value.get("images")
        if frames is None:
            frames = value.get("frames")
        if isinstance(frames, torch.Tensor):
            return frames, value.get("audio"), float(value.get("fps") or value.get("frame_rate") or _H3_FPS)
    if isinstance(value, torch.Tensor) and value.ndim == 4:
        return value, None, float(_H3_FPS)
    raise ValueError("AD_MiniMax_guide received an unsupported reference video payload")


def _ad_guide_resample_video(frames, source_fps):
    if not source_fps or abs(float(source_fps) - float(_H3_FPS)) < 0.01:
        return frames
    count = max(1, round(frames.shape[0] * float(_H3_FPS) / float(source_fps)))
    indexes = torch.linspace(0, frames.shape[0] - 1, count, device=frames.device).round().long()
    return frames[indexes]


def _ad_split_audio(audio, start_seconds, duration_seconds):
    if not isinstance(audio, collections.abc.Mapping) or "waveform" not in audio:
        raise ValueError("AD_MinMax_Ref2_mul: split audio must be an AUDIO payload")
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    start = round(float(start_seconds) * sample_rate)
    count = round(float(duration_seconds) * sample_rate)
    end = min(start + count, int(waveform.shape[-1]))
    if start >= end:
        return None
    result = dict(audio)
    result["waveform"] = waveform[..., start:end].clone()
    result["sample_rate"] = sample_rate
    return result


def _ad_split_video(video, start_seconds, duration_seconds):
    frames, audio, source_fps = _ad_guide_video_parts(video)
    start = round(float(start_seconds) * source_fps)
    count = round(float(duration_seconds) * source_fps)
    end = min(start + count, int(frames.shape[0]))
    available = end - start
    resampled_count = round(available * float(_H3_FPS) / float(source_fps))
    if start >= end or resampled_count < 5:
        return None
    split_audio = None if audio is None else _ad_split_audio(audio, start_seconds, duration_seconds)
    return {
        "images": frames[start:end].clone(),
        "audio": split_audio,
        "fps": source_fps,
    }


def _ad_guide_normalize_marked_tags(text, image_count, video_count, audio_count):
    closing = {"<": ">", "[": "]", "(": ")", "{": "}"}
    limits = {"image": image_count, "video": video_count, "audio": audio_count}
    names = {"image": "Pic", "video": "Video", "audio": "Audio"}

    def valid_match(match):
        if match.group("at"):
            return True
        opener = match.group("open")
        closer = match.group("close")
        return closing.get(opener) == closer

    def replace_media(match):
        if not valid_match(match):
            return match.group(0)
        kind = _AD_GUIDE_MEDIA_ALIASES[match.group("kind").lower()]
        ordinal = int(match.group("number"))
        if ordinal <= 0 or ordinal > limits[kind]:
            return ""
        return f"<{names[kind]} {ordinal}>"

    def replace_shot(match):
        if not valid_match(match):
            return match.group(0)
        return f"[Shot {int(match.group('number'))}]"

    def replace_subject(match):
        if not valid_match(match):
            return match.group(0)
        return f"<Subject {int(match.group('number'))}>"

    text = _AD_GUIDE_MEDIA_TAG_RE.sub(replace_media, text)
    text = _AD_GUIDE_SHOT_TAG_RE.sub(replace_shot, text)
    return _AD_GUIDE_SUBJECT_TAG_RE.sub(replace_subject, text)


def _ad_guide_resolve_prompt(prompt, tag_by_input, image_count, video_count, audio_count):
    text = str(prompt or "")
    text = _AD_GUIDE_UNRESOLVED_RE.sub("", text)
    text = _AD_GUIDE_PLACEHOLDER_RE.sub(lambda match: tag_by_input.get(int(match.group(1)), ""), text)
    return _ad_guide_normalize_marked_tags(text, image_count, video_count, audio_count)


def _ad_prompt_media_references(prompt):
    references = []
    for match in _AD_GUIDE_PLACEHOLDER_RE.finditer(str(prompt or "")):
        index = int(match.group(1))
        if index not in references:
            references.append(index)
    return references


def _ad_collect_media(kwargs):
    """从 kwargs 收集所有连接的素材，返回 (input_index, media_type, value) 列表。

    列表保持按 input_index 升序（_ad_select_prompt_media 重映射后即按 prompt 引用顺序）。
    """
    items = []
    direct = kwargs.get("media")
    if direct is not None and not isinstance(direct, str):
        detected = _ad_guide_media_type(direct)
        if not detected:
            raise ValueError("Media only accepts image, video, audio, latent or text inputs")
        items.append((0, detected, direct))
    for index in range(1, _AD_GUIDE_MAX_MEDIA + 1):
        value = kwargs.get(f"media_{index}")
        if value is None:
            continue
        detected = _ad_guide_media_type(value)
        if not detected:
            raise ValueError(f"media_{index} only accepts image, video, audio or latent inputs")
        declared = str(kwargs.get(f"media_type_{index}") or "").strip().lower()
        media_type = declared if declared in {"image", "video", "audio", "latent"} else detected
        items.append((index, media_type, value))
    return items


def _ad_build_reference_plan(items):
    """从 _ad_collect_media 输出构建 tag_by_input 映射 + image/video/audio count。

    items 必须按 prompt 引用顺序排列（local_index 1..N），ordinal 即为该段新编号。
    返回 (tag_by_input, image_count, video_count, audio_count)。
    静默截断每类（图片 9 / 视频 3 / 音频 3）。
    """
    images = [item for item in items if item[1] == "image"][:_AD_GUIDE_MAX_IMAGES]
    videos = [item for item in items if item[1] == "video"][:_AD_GUIDE_MAX_VIDEOS]
    audios = [item for item in items if item[1] == "audio"][:_AD_GUIDE_MAX_AUDIOS]

    tag_by_input = {}
    audio_ordinal = 0
    for ordinal, (input_index, _kind, _) in enumerate(images, start=1):
        tag_by_input[input_index] = f"<Picture {ordinal}>"
    for ordinal, (input_index, _kind, _) in enumerate(videos, start=1):
        tag_by_input[input_index] = f"<Video {ordinal}>"
    for input_index, _kind, _ in audios:
        audio_ordinal += 1
        tag_by_input[input_index] = f"<Audio {audio_ordinal}>"
    return tag_by_input, len(images), len(videos), audio_ordinal


def _ad_resolve_segment_text(prompt, kwargs, node_name):
    """对单段 prompt 计算 resolved_prompt（不执行 sampler，只算文本）。

    该段独立按 prompt 引用顺序编号，返回 (resolved_prompt, items, tag_by_input, counts)。
    """
    selected_prompt, selected_kwargs, _refs = _ad_select_prompt_media(
        str(prompt or ""), dict(kwargs), node_name
    )
    items = _ad_collect_media(selected_kwargs)
    tag_by_input, image_count, video_count, audio_count = _ad_build_reference_plan(items)
    resolved = _ad_guide_resolve_prompt(
        selected_prompt, tag_by_input, image_count, video_count, audio_count
    )
    return resolved, items, tag_by_input, (image_count, video_count, audio_count)


def _ad_format_segmented_text_ref2(prompts, kwargs, node_name, stage_index=None,
                                   stage_text_overrides=None):
    """参考生成类多段合并输出。

    对每段独立按 prompt 引用顺序编号，用 #segmentN--------- 行区分。
    stage_text_overrides: {1: "stage 1 prompt", 2: "stage 2 prompt", ...}，覆盖对应段。
    """
    parts = []
    for i, prompt in enumerate(prompts, start=1):
        effective = prompt
        if stage_text_overrides and i in stage_text_overrides and stage_text_overrides[i] is not None:
            effective = stage_text_overrides[i]
        try:
            resolved, _, _, _ = _ad_resolve_segment_text(effective, kwargs, node_name)
        except ValueError:
            # 单段解析失败时使用空字符串（避免单段错误中断整个合并）
            resolved = ""
        parts.append(f"#segment{i}---------")
        parts.append(resolved)
    return "\n".join(parts)


def _ad_format_segmented_text_fl2(prompts, kwargs, node_name, length,
                                  stage_index=None, stage_text_overrides=None):
    """FL2（首尾帧）多段合并输出。

    每段独立用 _ad_fl2_prompt 处理：把 Pic N 转成 Picture N（裸的）。
    段 N+1 时使用 has_context_latent=True（FL2 单图隐式补充首帧）。
    """
    parts = []
    frame_count = _ad_h3_frame_count(length)
    for i, prompt in enumerate(prompts, start=1):
        effective = prompt
        if stage_text_overrides and i in stage_text_overrides and stage_text_overrides[i] is not None:
            effective = stage_text_overrides[i]
        try:
            preview = _ad_preview_prompt(effective, dict(kwargs))
            references = _ad_prompt_media_references(effective)
            image_count = sum(
                1 for index in references
                if str(kwargs.get(f"media_type_{index}") or "").strip().lower() == "image"
            )
            if i > 1 and image_count == 1:
                preview = _AD_FL2_PICTURE_RE.sub(
                    lambda match: "Picture 2" if match.group(1) == "1" else match.group(0),
                    preview,
                )
                image_count = 2
            text = _ad_fl2_prompt(preview, image_count, "first", frame_count, i > 1)
        except ValueError:
            text = ""
        parts.append(f"#segment{i}---------")
        parts.append(text)
    return "\n".join(parts)


def _ad_select_prompt_media(prompt, values, node_name):
    references = _ad_prompt_media_references(prompt)
    if any(index < 1 or index > _AD_GUIDE_MAX_MEDIA for index in references):
        raise ValueError(f"{node_name}: prompt references a material outside the supported range")

    selected = {}
    if values.get("media") is not None:
        selected["media"] = values["media"]
    global_to_local = {}
    for local_index, global_index in enumerate(references, start=1):
        value = values.get(f"media_{global_index}")
        if value is None:
            raise ValueError(f"{node_name}: material {global_index} is not connected")
        media_type = _ad_guide_media_type(value)
        if not media_type:
            raise ValueError(f"{node_name}: material {global_index} has an unsupported type")
        selected[f"media_{local_index}"] = value
        selected[f"media_type_{local_index}"] = media_type
        global_to_local[global_index] = local_index

    selected_prompt = _AD_GUIDE_PLACEHOLDER_RE.sub(
        lambda match: f"__AD_MINIMAX_GUIDE_REF_{global_to_local[int(match.group(1))]}__",
        str(prompt or ""),
    )
    return selected_prompt, selected, references


def _ad_preview_prompt(prompt, values):
    counters = {"image": 0, "video": 0, "audio": 0}
    labels = {"image": "Pic", "video": "Video", "audio": "Audio"}
    tag_by_input = {}
    for index in _ad_prompt_media_references(prompt):
        media_type = str(values.get(f"media_type_{index}") or "").strip().lower()
        if media_type not in counters:
            media_type = _ad_guide_media_type(values.get(f"media_{index}"))
        if media_type in counters:
            counters[media_type] += 1
            tag_by_input[index] = f"{labels[media_type]} {counters[media_type]}"
        else:
            tag_by_input[index] = ""

    text = _AD_GUIDE_UNRESOLVED_RE.sub("", str(prompt or ""))
    return _AD_GUIDE_PLACEHOLDER_RE.sub(
        lambda match: tag_by_input.get(int(match.group(1)), ""),
        text,
    )


def _ad_preview_media_types(values):
    media_types = []
    direct = values.get("media")
    if direct is not None and not isinstance(direct, str):
        media_type = _ad_guide_media_type(direct)
        if media_type:
            media_types.append(media_type)
    for index in range(1, _AD_GUIDE_MAX_MEDIA + 1):
        media_type = str(values.get(f"media_type_{index}") or "").strip().lower()
        if media_type not in {"image", "video", "audio", "latent"}:
            media_type = _ad_guide_media_type(values.get(f"media_{index}"))
        if media_type:
            media_types.append(media_type)
    return media_types


def _ad_last_frame(value):
    if isinstance(value, torch.Tensor) and value.ndim == 4 and value.shape[0] > 0:
        return value[-1:]
    if hasattr(value, "get_components"):
        images = value.get_components().images
        return images[-1:] if isinstance(images, torch.Tensor) and images.shape[0] > 0 else None
    if isinstance(value, collections.abc.Mapping):
        for name in ("images", "frames"):
            images = value.get(name)
            if isinstance(images, torch.Tensor) and images.ndim == 4 and images.shape[0] > 0:
                return images[-1:]
    return None


def _ad_h3_frame_count(length):
    frame_count = max(5, int(length))
    while frame_count % 17 != 5:
        frame_count += 1
    return frame_count


def _ad_fl2_prompt(prompt, image_count, single_image_position, frame_count, has_context_latent=False):
    text = str(prompt or "").strip()
    lines = text.splitlines()
    if lines and (
        lines[0].strip().lower().startswith("how the reference pictures align with the target video")
        or lines[0].strip().lower().startswith("for the target video, at 0.00 seconds into the target video")
    ):
        body_index = next(
            (
                index for index, line in enumerate(lines[1:], start=1)
                if line.strip().lower().startswith("integrated_multimodal_description:")
            ),
            1,
        )
        lines = lines[body_index:]
        while lines and not lines[0].strip():
            lines.pop(0)
        text = "\n".join(lines).strip()

    if image_count <= 0:
        return text

    position = str(single_image_position or "auto").strip().lower()
    position = {
        "\u81ea\u52a8": "auto", "\u9996\u5e27": "first", "\u5c3e\u5e27": "last",
    }.get(position, position)
    if position not in {"auto", "first", "last"}:
        position = "auto"
    if image_count == 1 and position == "auto":
        position = "last" if has_context_latent else "first"

    shot_numbers = [int(match.group(1)) for match in _AD_FL2_SHOT_RE.finditer(text)]
    final_shot = max(shot_numbers, default=1)
    duration = frame_count / float(_H3_FPS)
    if image_count >= 2:
        text = _AD_FL2_PICTURE_RE.sub(lambda match: f"Picture {match.group(1)}", text)
        header = (
            "How the reference pictures align with the target video — "
            "Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; "
            f"Picture 2 (from Shot {final_shot}) aligns with the {duration:.2f}-second mark of the target video."
        )
    elif position == "last":
        text = _AD_FL2_PICTURE_RE.sub(lambda match: f"Picture {match.group(1)}", text)
        header = (
            "How the reference pictures align with the target video — "
            f"Picture 1 (from [Shot {final_shot}]) aligns with the {duration:.2f}-second mark of the target video."
        )
    else:
        text = _AD_FL2_PICTURE_RE.sub(lambda match: f"Picture {match.group(1)}", text)
        header = "For the target video, at 0.00 seconds into the target video, Picture 1 (from [Shot 1]) is fully referenced."
    return f"{header}\n\n{text}" if text else header




class AD_MiniMax_guide(AD_MiniMax_Ref2V):
    """MiniMax H3 reference guide with ordered virtual media inputs."""

    CATEGORY = "Apt_Preset/AD/😺backup"
    FUNCTION = "execute"
    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("positive", "latent", "trim_frames", "text")

    @classmethod
    def INPUT_TYPES(cls):
        inherited = super().INPUT_TYPES()
        required = dict(inherited["required"])
        encoders = {
            name: required.pop(name) for name in ("clip", "vae", "audio_vae")
        }
        media_input = ("IMAGE,VIDEO,AUDIO,LATENT", {"lazy": True})
        optional = {**encoders, "media": ("IMAGE,VIDEO,AUDIO,LATENT,STRING",)}
        for index in range(1, _AD_GUIDE_MAX_MEDIA + 1):
            optional[f"media_{index}"] = media_input
            optional[f"media_type_{index}"] = ("STRING", {"default": ""})
        return {
            "required": required,
            "optional": optional,
        }

    def check_lazy_status(self, prompt="", **kwargs):
        if any(kwargs.get(name) is None for name in ("clip", "vae", "audio_vae")):
            return []
        references = _ad_prompt_media_references(prompt)
        return [f"media_{index}" for index in references if kwargs.get(f"media_{index}") is None]

    @staticmethod
    def _collect_media(kwargs):
        return _ad_collect_media(kwargs)

    def execute(self, prompt, width, height, length, ref_image_size="match",
                clip=None, vae=None, audio_vae=None, _context_latent=None, **kwargs):
        if isinstance(kwargs.get("media"), str):
            prompt = kwargs["media"]
        if clip is None or vae is None or audio_vae is None:
            blocker = ExecutionBlocker(None)
            return blocker, blocker, blocker, _ad_preview_prompt(prompt, kwargs)
        if _h3_empty_av_latent is None or _h3_resize is None or _node_helpers is None:
            raise RuntimeError("This ComfyUI build does not provide MiniMax H3 support")
        prompt, kwargs, _references = _ad_select_prompt_media(prompt, kwargs, "AD_MiniMax_guide")
        latent, frame_count = _h3_empty_av_latent(width, height, length)
        items = _ad_collect_media(kwargs)
        if _context_latent is not None and not any(item[1] == "latent" for item in items):
            items.append((_AD_GUIDE_MAX_MEDIA + 1, "latent", _context_latent))
        if not items:
            raise ValueError("AD_MiniMax_guide needs at least one image or video")
        context_latents = [item for item in items if item[1] == "latent"]
        if len(context_latents) > 1:
            raise ValueError("AD_MiniMax_guide accepts only one context latent")

        # items 已按 prompt 出现顺序排列（_ad_select_prompt_media 重映射为 local_index 1..N）
        # 静默截断每类（图片 9 / 视频 3 / 音频 3）
        images = [item for item in items if item[1] == "image"][:_AD_GUIDE_MAX_IMAGES]
        videos = [item for item in items if item[1] == "video"][:_AD_GUIDE_MAX_VIDEOS]
        audios = [item for item in items if item[1] == "audio"][:_AD_GUIDE_MAX_AUDIOS]

        if not images and not videos and not context_latents:
            raise ValueError("AD_MiniMax_guide needs an image or video in addition to audio")

        ref_items = []
        ref_blocks = []
        tag_by_input = {}
        audio_ordinal = 0

        # images / videos / audios 已是按 prompt 引用顺序排列，ordinal 即为该段新编号
        for ordinal, (input_index, _kind, image) in enumerate(images, start=1):
            if not isinstance(image, torch.Tensor) or image.ndim != 4:
                raise ValueError("Image references must be IMAGE tensors")
            image_h, image_w = image.shape[1], image.shape[2]
            if str(ref_image_size) == "match":
                scale = min(1.0, math.sqrt((width * height) / max(1, image_w * image_h)))
            else:
                scale = min(1.0, _H3_REF_IMAGE_SHORT_EDGE / max(1, min(image_w, image_h)))
            target_w = max(_H3_CANVAS_MULTIPLE, round(image_w * scale / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
            target_h = max(_H3_CANVAS_MULTIPLE, round(image_h * scale / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
            resized = _h3_resize(image[:1], target_w, target_h, "disabled")
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": target_h // 16, "latent_w": target_w // 16, "latent": vae.encode(resized)})
            tag_by_input[input_index] = f"<Picture {ordinal}>"

        for ordinal, (input_index, _kind, video) in enumerate(videos, start=1):
            frames, soundtrack, source_fps = _ad_guide_video_parts(video)
            frames = _ad_guide_resample_video(frames, source_fps)
            video_h, video_w = frames.shape[1], frames.shape[2]
            canvas_w, canvas_h = _h3_adapt_canvas(video_w, video_h)
            if video_w * video_h < canvas_w * canvas_h:
                canvas_w = max(_H3_CANVAS_MULTIPLE, round(video_w / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
                canvas_h = max(_H3_CANVAS_MULTIPLE, round(video_h / _H3_CANVAS_MULTIPLE) * _H3_CANVAS_MULTIPLE)
            frames = _h3_resize(frames, canvas_w, canvas_h, "disabled")[:frame_count]
            count = frames.shape[0]
            if count < 5:
                raise ValueError("Reference videos need at least 5 frames")
            while count % 17 != 5:
                count -= 1
            frames = frames[:count]
            video_latent = vae.encode(frames)
            audio_latent, audio_t = None, 0
            if soundtrack is not None:
                audio_latent, audio_t = self._encode_ref_audio(audio_vae, soundtrack)
                audio_ordinal += 1
                ref_items.append({"type": "audio"})
            sample_indexes = list(range(0, frames.shape[0], max(1, _H3_FPS // 2)))
            ref_items.append({"type": "video", "data": frames[sample_indexes], "timestamps": [i / 2.0 for i in range(len(sample_indexes))]})
            ref_blocks.append({
                "kind": "video_audio" if audio_t else "video", "latent_t": video_latent.shape[2],
                "latent_h": canvas_h // 16, "latent_w": canvas_w // 16, "ref_audio_t": audio_t,
                "latent": video_latent, "audio_latent": audio_latent,
            })
            tag_by_input[input_index] = f"<Video {ordinal}>"

        for input_index, _kind, audio in audios:
            if not isinstance(audio, collections.abc.Mapping) or "waveform" not in audio:
                raise ValueError("Audio references must be AUDIO payloads")
            audio_latent, audio_t = self._encode_ref_audio(audio_vae, audio)
            audio_ordinal += 1
            ref_items.append({"type": "audio"})
            ref_blocks.append({"kind": "audio", "ref_audio_t": audio_t, "audio_latent": audio_latent})
            tag_by_input[input_index] = f"<Audio {audio_ordinal}>"

        resolved_prompt = _ad_guide_resolve_prompt(prompt, tag_by_input, len(images), len(videos), audio_ordinal)
        tokens = clip.tokenize(resolved_prompt, minimax_ref_items=ref_items)
        positive = clip.encode_from_tokens_scheduled(tokens)
        positive = _node_helpers.conditioning_set_values(positive, {"minimax_refs": ref_blocks})
        if context_latents:
            positive, _ = AptMiniMaxH3MotionContext().apply(
                positive,
                latent,
                trim_frames=_AD_GUIDE_CONTEXT_LENGTH,
                context_latent=context_latents[0][2],
                audio_context_length=_AD_GUIDE_AUDIO_CONTEXT_LENGTH,
            )
        return positive, latent, _AD_GUIDE_TRIM_FRAMES, resolved_prompt


class _AD_MinMaxBase:
    CATEGORY = "Apt_Preset/AD"
    FUNCTION = "execute"
    RETURN_TYPES = ("RUN_CONTEXT", "VIDEO", "STRING")
    RETURN_NAMES = ("context", "video", "text")

    @staticmethod
    def _custom_sample(context, seed, denoise=1.0, latent=None, sigmas=None):
        result = basic_Ksampler_custom().sample(
            context=context,
            latent=latent,
            sigmas=sigmas,
            seed=seed,
            denoise=denoise,
            image_output="Hide",
        )["result"]
        return result[0], result[7]

    @staticmethod
    def _second_pass_positive(positive, source_latent, target_latent):
        return _apt_second_pass_positive(positive, source_latent, target_latent)

    def _sample_video(self, context, model, positive, latent, seed, fps,
                      has_context_latent, text, second_pass_mode="None",
                      refine_model="None", refine_denoise=0.3, refine_steps=8,
                      latent_model=None, latent_scale=1.3, split_step=4,
                      exact_audio=None):
        node_name = type(self).__name__
        guide_context = new_context(
            context, model=model, positive=positive, latent=latent
        )
        if second_pass_mode not in ("None", "refine", "latent_scale"):
            raise ValueError(f"{node_name}: invalid second_pass_mode: {second_pass_mode}")
        if second_pass_mode == "latent_scale":
            active_model = guide_context.get("model")
            steps = int(guide_context.get("steps"))
            if split_step <= 0 or split_step >= steps:
                raise ValueError(f"{node_name}: split_step must be between 1 and {steps - 1}")
            scheduler = guide_context.get("scheduler")
            sigmas = BasicScheduler().get_sigmas(active_model, scheduler, steps, 1.0)[0]
            high_sigmas, low_sigmas = SplitSigmas.execute(sigmas, split_step).result
            first_context, first_denoise_latent = self._custom_sample(
                guide_context, seed, latent=latent, sigmas=high_sigmas
            )
            scaled_latent = latent_minimaxH3_scale().execute(
                first_denoise_latent, latent_model, latent_scale
            )[0]
            second_positive = self._second_pass_positive(
                first_context.get("positive"), first_denoise_latent, scaled_latent
            )
            second_context = new_context(
                first_context, positive=second_positive, latent=scaled_latent
            )
            sampled_context, _final_denoise_latent = self._custom_sample(
                second_context, seed, latent=scaled_latent, sigmas=low_sigmas
            )
        else:
            sampled_context, first_denoise_latent = self._custom_sample(
                guide_context, seed, latent=latent
            )

        if second_pass_mode == "refine":
            if refine_model != "None" and refine_model not in folder_paths.get_filename_list("upscale_models"):
                raise ValueError(f"{node_name}: invalid refine_model: {refine_model}")
            refine_latent = sampled_context.get("latent")
            if refine_model != "None":
                up_model = load_upscale_model(refine_model)
                upscaled_image = upscale_with_model(up_model, sampled_context.get("images"))
                video_latent = encode(sampled_context.get("vae"), upscaled_image)[0]
                refine_latent = _apt_replace_av_video_latent(refine_latent, video_latent)
            refine_positive = self._second_pass_positive(
                sampled_context.get("positive"), first_denoise_latent, refine_latent
            )
            refine_context = new_context(
                sampled_context, steps=refine_steps, positive=refine_positive, latent=refine_latent
            )
            sampled_context, _final_denoise_latent = self._custom_sample(
                refine_context, seed, denoise=refine_denoise, latent=refine_latent
            )
        video = AD_CreateVideo.execute(
            context=sampled_context,
            audio=exact_audio,
            fps=fps,
            trim_frames=_AD_GUIDE_TRIM_FRAMES if has_context_latent else 0,
        )[0]
        return first_denoise_latent, video, text


class _AD_MinMax_Ref2Base(_AD_MinMaxBase, AD_MiniMax_guide):
    """Shared Ref2VA input contract."""

    @classmethod
    def INPUT_TYPES(cls):
        inherited = AD_MiniMax_guide.INPUT_TYPES()
        required = {
            name: value for name, value in inherited["required"].items()
            if name not in {"clip", "vae", "audio_vae"}
        }
        required["seed"] = ("INT", {
            "default": 0,
            "min": 0,
            "max": 0xffffffffffffffff,
            "control_after_generate": True,
        })
        optional = {
            "context": ("RUN_CONTEXT",),
            "model": ("MODEL", {"lazy": True}),
            "fps": ("FLOAT", {
            "default": 24.0,
            "min": 1.0,
            "max": 120.0,
            "step": 1.0,
            }),
        }
        optional.update({
            name: value for name, value in inherited["optional"].items()
            if name not in {"clip", "vae", "audio_vae"}
        })
        return {
            "required": required,
            "optional": optional,
        }

    def check_lazy_status(self, prompt="", context=None, **kwargs):
        if context is None:
            return []
        references = _ad_prompt_media_references(prompt)
        required = ["model"] if "model" in kwargs and kwargs.get("model") is None else []
        required.extend(f"media_{index}" for index in references if kwargs.get(f"media_{index}") is None)
        return required

    def execute(self, prompt, width, height, length, seed, ref_image_size="match",
                context=None, model=None, fps=24.0, **kwargs):
        if isinstance(kwargs.get("media"), str):
            prompt = kwargs["media"]
        if context is None:
            blocker = ExecutionBlocker(None)
            return blocker, blocker, _ad_preview_prompt(prompt, kwargs)
        clip = context.get("clip")
        vae = context.get("vae")
        audio_vae = context.get("audio_vae")
        missing = [name for name, value in (("clip", clip), ("vae", vae), ("audio_vae", audio_vae)) if value is None]
        if missing:
            raise ValueError(f"Ref2VA context is missing: {', '.join(missing)}")

        # An explicitly connected Media latent always wins. Otherwise carry
        # the previous sampler latent forward through context.
        has_context_latent = any(
            item[1] == "latent" for item in _ad_collect_media(kwargs)
        )
        upstream_latent = None if has_context_latent else context.get("latent")
        has_context_latent = has_context_latent or upstream_latent is not None

        positive, latent, _trim_frames, text = AD_MiniMax_guide.execute(
            self,
            prompt,
            width,
            height,
            length,
            ref_image_size,
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            _context_latent=upstream_latent,
            **kwargs,
        )
        return self._sample_video(
            context, model, positive, latent, seed, fps,
            has_context_latent, text,
        )


_AD_STAGE_INFO_VERSION = 1
_AD_STAGE_VIDEO_CRF = 23.0


def _ad_stage_info(stage_info):
    if not isinstance(stage_info, collections.abc.Mapping):
        raise TypeError("AD_MinMax_Ref2_mul: stage_info must come from flow_stage_begin")
    if int(stage_info.get("version", -1)) != _AD_STAGE_INFO_VERSION:
        raise ValueError("AD_MinMax_Ref2_mul: unsupported stage_info version")
    run_id = str(stage_info.get("run_id") or "").strip()
    stage_index = int(stage_info.get("stage_index", -1))
    total = int(stage_info.get("total", 0))
    if not run_id or total < 1 or stage_index < 0 or stage_index >= total:
        raise ValueError("AD_MinMax_Ref2_mul: invalid stage_info")
    return run_id, stage_index, total


def _ad_stage_prompts(value, fallback=""):
    try:
        prompts = json.loads(str(value or "[]"))
    except json.JSONDecodeError as exc:
        raise ValueError("AD_MinMax_Ref2_mul: stage prompts are invalid") from exc
    if not isinstance(prompts, list) or not all(isinstance(item, str) for item in prompts):
        raise ValueError("AD_MinMax_Ref2_mul: stage prompts must be a list of strings")
    if not prompts and fallback:
        prompts = [str(fallback)]
    return prompts


def _ad_stage_prompt_plan(stage_prompts, prompt, stage_info=None):
    stage_index = 0 if stage_info is None else _ad_stage_info(stage_info)[1]
    prompts = _ad_stage_prompts(stage_prompts, prompt)
    # The scheduler total is authoritative. Extra prompts are intentionally
    # ignored; if there are fewer prompts than stages, reuse the final prompt.
    # This keeps prompt editing independent from queue length.
    if not prompts:
        prompts = [str(prompt or "")]
    selected = prompts[min(stage_index, len(prompts) - 1)]
    references = []
    for match in _AD_GUIDE_PLACEHOLDER_RE.finditer(selected):
        index = int(match.group(1))
        if index not in references:
            references.append(index)
    if any(index < 1 or index > _AD_GUIDE_MAX_MEDIA for index in references):
        raise ValueError("AD_MinMax_Ref2_mul: prompt references a material outside the supported range")
    return selected, references


def _ad_single_split_material(values, media_type):
    matches = []
    for name, value in values.items():
        if name == "media" or re.fullmatch(r"media_\d+", name):
            if _ad_guide_media_type(value) == media_type:
                matches.append((name, value))
    if len(matches) != 1:
        raise ValueError(
            f"AD_MinMax_Ref2_mul: single_long_{media_type}_split needs exactly one referenced {media_type} material"
        )
    return matches[0]


def _ad_remove_split_material(values, name):
    values.pop(name, None)
    if name.startswith("media_"):
        values.pop(f"media_type_{name.removeprefix('media_')}", None)


def _ad_output_is_connected(workflow_prompt, unique_id, output_slot):
    if not isinstance(workflow_prompt, collections.abc.Mapping) or unique_id is None:
        return True
    node_id = str(unique_id)
    for node in workflow_prompt.values():
        if not isinstance(node, collections.abc.Mapping):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, collections.abc.Mapping):
            continue
        for value in inputs.values():
            if isinstance(value, (list, tuple)) and len(value) == 2:
                if str(value[0]) == node_id and str(value[1]) == str(output_slot):
                    return True
    return False


def _ad_stage_output_dir(run_id):
    readable = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_id)[:48]
    safe_name = f"{readable or 'stage'}_{hashlib.sha256(run_id.encode('utf-8')).hexdigest()[:10]}"
    root = os.path.abspath(os.path.join(folder_paths.get_output_directory(), "apt_stage_video"))
    path = os.path.abspath(os.path.join(root, safe_name))
    if os.path.commonpath((root, path)) != root:
        raise ValueError("AD_MinMax_Ref2_mul: invalid run_id")
    os.makedirs(os.path.join(path, "segments"), exist_ok=True)
    return path


def _ad_stage_save_video(video, run_id, stage_index):
    output_dir = _ad_stage_output_dir(run_id)
    path = os.path.join(output_dir, "segments", f"{stage_index + 1:05d}.mp4")
    temp_path = path + ".tmp.mp4"
    try:
        video.save_to(
            temp_path,
            format=Types.VideoContainer.MP4,
            codec=Types.VideoCodec.H264,
            crf=_AD_STAGE_VIDEO_CRF,
        )
        os.replace(temp_path, path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)
    return path


def _ad_stage_concat_mp4(paths, output_path, continuous_audio=None):
    audio = normalize_audio(continuous_audio) if continuous_audio is not None else None
    if continuous_audio is not None and audio is None:
        raise ValueError("AD_MinMax_Ref2_mul: original continuous audio is invalid")
    temp_path = output_path + ".tmp.mp4"
    try:
        with av.open(paths[0], mode="r") as first:
            stream_types = ("video",) if audio is not None else ("video", "audio")
            templates = [stream for stream in first.streams if stream.type in stream_types]
            if not templates:
                raise ValueError("AD_MinMax_Ref2_mul: saved segment has no usable stream")
            with av.open(temp_path, mode="w", format="mp4", options={"movflags": "use_metadata_tags+faststart"}) as output:
                output_streams = [output.add_stream_from_template(stream, opaque=True) for stream in templates]
                audio_stream = None
                if audio is not None:
                    sample_rate = int(audio["sample_rate"])
                    channels = int(audio["waveform"].shape[1])
                    layout = "mono" if channels == 1 else "stereo"
                    audio_stream = output.add_stream("aac", rate=sample_rate, layout=layout)
                timeline = Fraction(0)
                for path in paths:
                    with av.open(path, mode="r") as source:
                        streams = [stream for stream in source.streams if stream.type in stream_types]
                        if len(streams) != len(templates):
                            raise ValueError("AD_MinMax_Ref2_mul: segment stream layouts do not match")
                        bases = {}
                        segment_end = Fraction(0)
                        for packet in source.demux(streams):
                            if packet.dts is None and packet.pts is None:
                                continue
                            stream_index = streams.index(packet.stream)
                            template = templates[stream_index]
                            if packet.stream.type != template.type or packet.stream.codec_context.name != template.codec_context.name:
                                raise ValueError("AD_MinMax_Ref2_mul: segment codecs do not match")
                            time_base = Fraction(packet.time_base or packet.stream.time_base)
                            timestamps = [value for value in (packet.dts, packet.pts) if value is not None]
                            base = bases.setdefault(stream_index, min(timestamps))
                            offset = int(timeline / time_base)
                            if packet.dts is not None:
                                packet.dts = packet.dts - base + offset
                            if packet.pts is not None:
                                packet.pts = packet.pts - base + offset
                            packet.time_base = time_base
                            packet.stream = output_streams[stream_index]
                            end_value = max(value for value in (packet.dts, packet.pts) if value is not None)
                            segment_end = max(segment_end, (end_value + int(packet.duration or 0)) * time_base - timeline)
                            output.mux(packet)
                        if segment_end <= 0:
                            raise ValueError("AD_MinMax_Ref2_mul: saved segment is empty")
                        timeline += segment_end
                if audio_stream is not None:
                    sample_rate = int(audio["sample_rate"])
                    wanted = min(
                        int(audio["waveform"].shape[-1]),
                        max(1, int(round(float(timeline) * sample_rate))),
                    )
                    waveform = audio["waveform"][0, :, :wanted].float().cpu()
                    for offset in range(0, wanted, 32768):
                        samples = waveform[:, offset:offset + 32768].contiguous().numpy()
                        frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout=layout)
                        frame.sample_rate = sample_rate
                        frame.pts = offset
                        frame.time_base = Fraction(1, sample_rate)
                        for packet in audio_stream.encode(frame):
                            output.mux(packet)
                    for packet in audio_stream.encode(None):
                        output.mux(packet)
        os.replace(temp_path, output_path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)


def _ad_stage_video_outputs(video, run_id, stage_index, total, workflow_prompt, unique_id,
                            node_name, continuous_audio=None):
    merged_video = ExecutionBlocker(None)
    if run_id is None:
        return video, merged_video
    _ad_stage_save_video(video, run_id, stage_index)
    if stage_index != total - 1 or not _ad_output_is_connected(workflow_prompt, unique_id, 2):
        return video, merged_video
    output_dir = _ad_stage_output_dir(run_id)
    paths = [os.path.join(output_dir, "segments", f"{index + 1:05d}.mp4") for index in range(total)]
    missing_paths = [path for path in paths if not os.path.isfile(path)]
    if missing_paths:
        raise FileNotFoundError(f"{node_name}: missing segment {os.path.basename(missing_paths[0])}")
    final_path = os.path.join(output_dir, f"{os.path.basename(output_dir)}_final.mp4")
    _ad_stage_concat_mp4(paths, final_path, continuous_audio)
    return video, InputImpl.VideoFromFile(final_path)


def _ad_add_second_pass_inputs(required):
    required["second_pass_mode"] = (
        ["None", "refine", "latent_scale"],
        {"default": "None", "tooltip": "Select one mutually exclusive second-pass workflow."},
    )
    required["refine_model"] = (
        ["None"] + folder_paths.get_filename_list("upscale_models"),
        {
            "default": "None",
            "tooltip": "Optional image upscaler before the second pass. None resamples the first-pass latent directly. Use None or a 1x model for multi-stage latent continuity.",
        },
    )
    required["refine_denoise"] = (
        "FLOAT",
        {
            "default": 0.3,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "tooltip": "Denoise strength for the second sampling pass.",
        },
    )
    required["refine_steps"] = (
        "INT",
        {
            "default": 8,
            "min": 1,
            "max": 10000,
            "step": 1,
            "tooltip": "Number of steps for the second sampling pass.",
        },
    )
    required["latent_model"] = latent_minimaxH3_scale.INPUT_TYPES()["required"]["model"]
    required["latent_scale"] = (
        "FLOAT",
        {
            "default": 1.3,
            "min": 1.0,
            "max": 4.0,
            "step": 0.05,
            "tooltip": "MiniMax H3 latent upscale multiplier between the two sigma ranges.",
        },
    )
    required["split_step"] = (
        "INT",
        {
            "default": 4,
            "min": 0,
            "max": 10000,
            "step": 1,
            "tooltip": "SplitSigmas step: high sigmas run before latent scaling and low sigmas run after it.",
        },
    )


class AD_MinMax_Ref2_generate(_AD_MinMaxBase, AD_MiniMax_guide):
    """Queue-stage Ref2VA sampler with stage prompts and local media numbering."""

    RETURN_TYPES = ("LATENT", "VIDEO", "VIDEO", "STRING")
    RETURN_NAMES = ("denoise_latent1", "segment_video", "merged_video", "text")

    @classmethod
    def INPUT_TYPES(cls):
        inherited = _AD_MinMax_Ref2Base.INPUT_TYPES()
        required = dict(inherited["required"])
        required["stage_prompts"] = ("STRING", {"default": "[]", "multiline": True})
        required["single_long_video_split"] = (
            "BOOLEAN",
            {"default": False, "tooltip": "Split one referenced long video by stage. Its embedded audio drives H3; the final merge uses the original continuous soundtrack."},
        )
        required["single_long_audio_split"] = (
            "BOOLEAN",
            {"default": False, "tooltip": "Split one referenced long audio clip by stage and lock it into H3; the final merge uses the original continuous audio."},
        )
        _ad_add_second_pass_inputs(required)
        optional = dict(inherited["optional"])
        optional["stage_info"] = ("FLOW_STAGE_INFO",)
        optional["stage_data"] = ("IMAGE,VIDEO,AUDIO,LATENT",)
        for index in range(1, _AD_GUIDE_MAX_STAGE_TEXT + 1):
            optional[f"stage_text_{index}"] = ("STRING", {"forceInput": True})
        for index in range(1, _AD_GUIDE_MAX_MEDIA + 1):
            name = f"media_{index}"
            if name in optional:
                input_type = optional[name][0]
                options = optional[name][1] if len(optional[name]) > 1 else {}
                optional[name] = (input_type, {**options, "lazy": True})
        hidden = dict(inherited.get("hidden", {}))
        hidden.update({"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"})
        return {"required": required, "optional": optional, "hidden": hidden}

    def check_lazy_status(self, stage_prompts, prompt="", stage_info=None, **kwargs):
        if kwargs.get("context") is None:
            return []
        _selected, references = _ad_stage_prompt_plan(stage_prompts, prompt, stage_info)
        required = ["model"] if "model" in kwargs and kwargs.get("model") is None else []
        required.extend(f"media_{index}" for index in references if kwargs.get(f"media_{index}") is None)
        return required

    def execute(self, prompt, width, height, length, seed, stage_prompts,
                single_long_video_split, single_long_audio_split,
                second_pass_mode, refine_model, refine_denoise, refine_steps,
                latent_model, latent_scale, split_step,
                ref_image_size="match", context=None, model=None, fps=24.0, stage_info=None, stage_data=None,
                unique_id=None, workflow_prompt=None, **kwargs):
        if stage_info is None:
            run_id, stage_index, total = None, 0, 1
        else:
            run_id, stage_index, total = _ad_stage_info(stage_info)
        selected_prompt, _references = _ad_stage_prompt_plan(stage_prompts, prompt, stage_info)
        stage_text = kwargs.get(f"stage_text_{stage_index + 1}")
        if stage_text is not None:
            selected_prompt = str(stage_text)

        if stage_data is None and stage_info is not None and stage_index > 0:
            stage_data = stage_info.get("stage_data")

        media_values = dict(kwargs)
        if stage_data is not None:
            media_values["media"] = stage_data
        if context is None:
            blocker = ExecutionBlocker(None)
            return blocker, blocker, blocker, _ad_preview_prompt(selected_prompt, media_values)
        selected_prompt, selected_kwargs, references = _ad_select_prompt_media(
            selected_prompt,
            media_values,
            "AD_MinMax_Ref2_generate",
        )
        clip = context.get("clip")
        vae = context.get("vae")
        audio_vae = context.get("audio_vae")
        missing = [name for name, value in (("clip", clip), ("vae", vae), ("audio_vae", audio_vae)) if value is None]
        if missing:
            raise ValueError(f"AD_MinMax_Ref2_generate context is missing: {', '.join(missing)}")

        exact_audio = None
        continuous_audio = None
        if single_long_video_split or single_long_audio_split:
            segment_frames = int(length)
            stride_frames = max(1, segment_frames - _AD_GUIDE_TRIM_FRAMES)
            start_seconds = stage_index * stride_frames / float(fps)
            duration_seconds = segment_frames / float(fps)

            if single_long_video_split:
                video_name, long_video = _ad_single_split_material(selected_kwargs, "video")
                _frames, continuous_audio, _source_fps = _ad_guide_video_parts(long_video)
                split_video = _ad_split_video(long_video, start_seconds, duration_seconds)
                if split_video is None:
                    _ad_remove_split_material(selected_kwargs, video_name)
                else:
                    selected_kwargs[video_name] = split_video
                    exact_audio = split_video["audio"]

            if single_long_audio_split:
                audio_name, long_audio = _ad_single_split_material(selected_kwargs, "audio")
                continuous_audio = long_audio
                exact_audio = _ad_split_audio(long_audio, start_seconds, duration_seconds)
                if exact_audio is None:
                    _ad_remove_split_material(selected_kwargs, audio_name)
                else:
                    selected_kwargs[audio_name] = exact_audio

        active_model = model
        if exact_audio is not None:
            active_model = model if model is not None else context.get("model")
            if active_model is None:
                raise ValueError("AD_MinMax_Ref2_generate: audio lock needs a model in context or the model input")

        items = _ad_collect_media(selected_kwargs)
        has_context_latent = any(item[1] == "latent" for item in items)
        upstream_latent = None if has_context_latent else context.get("latent")
        has_context_latent = has_context_latent or upstream_latent is not None
        positive, latent, _trim_frames, text = AD_MiniMax_guide.execute(
            self,
            selected_prompt,
            width,
            height,
            length,
            ref_image_size,
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            _context_latent=upstream_latent,
            **selected_kwargs,
        )
        if exact_audio is not None:
            active_model, latent, exact_audio = AptMiniMaxH3NativeAudioLock().lock_audio(
                active_model, latent, audio_vae, exact_audio
            )
        denoise_latent1, video, text = self._sample_video(
            context, active_model, positive, latent, seed, fps, has_context_latent, text,
            second_pass_mode, refine_model, refine_denoise, refine_steps,
            latent_model, latent_scale, split_step, exact_audio,
        )

        video, merged_video = _ad_stage_video_outputs(
            video, run_id, stage_index, total, workflow_prompt, unique_id,
            "AD_MinMax_Ref2_generate", continuous_audio,
        )

        # 多段合并 text 输出（每段用 #segmentN--------- 分隔）
        all_prompts = _ad_stage_prompts(stage_prompts, prompt)
        stage_text_overrides = {}
        for index in range(1, _AD_GUIDE_MAX_STAGE_TEXT + 1):
            override = kwargs.get(f"stage_text_{index}")
            if override is not None:
                stage_text_overrides[index] = str(override)
        merged_text = _ad_format_segmented_text_ref2(
            all_prompts, media_values, "AD_MinMax_Ref2_generate",
            stage_index, stage_text_overrides,
        )
        return denoise_latent1, video, merged_video, merged_text


class AD_MinMax_Ref2_mul(AD_MinMax_Ref2_generate):
    """Queue-stage Ref2VA conditioning for an external sampler and video output."""

    RETURN_TYPES = ("RUN_CONTEXT", "AUDIO", "INT", "STRING")
    RETURN_NAMES = ("context", "exact_audio", "trim_frames", "text")

    @classmethod
    def INPUT_TYPES(cls):
        inherited = AD_MinMax_Ref2_generate.INPUT_TYPES()
        required = dict(inherited["required"])
        for name in (
            "seed", "second_pass_mode", "refine_model", "refine_denoise", "refine_steps",
            "latent_model", "latent_scale", "split_step",
        ):
            required.pop(name, None)
        required["long_split_FPS"] = (
            "FLOAT",
            {
                "default": 24.0,
                "min": 1.0,
                "max": 120.0,
                "step": 1.0,
                "tooltip": "FPS used only to calculate long video/audio split positions.",
            },
        )
        required["default_trim_frames"] = (
            "INT",
            {
                "default": 22,
                "min": 0,
                "max": 4096,
                "step": 1,
                "tooltip": "Leading frames removed from continuation clips. The first clip still outputs 0.",
            },
        )
        optional = dict(inherited["optional"])
        optional.pop("fps", None)
        hidden = {
            name: value for name, value in inherited.get("hidden", {}).items()
            if name not in {"unique_id", "workflow_prompt"}
        }
        return {"required": required, "optional": optional, "hidden": hidden}

    def execute(self, prompt, width, height, length, stage_prompts,
                single_long_video_split, single_long_audio_split, long_split_FPS, default_trim_frames,
                ref_image_size="match", context=None, model=None, stage_info=None, stage_data=None,
                **kwargs):
        if stage_info is None:
            stage_index = 0
        else:
            _run_id, stage_index, _total = _ad_stage_info(stage_info)
        selected_prompt, _references = _ad_stage_prompt_plan(stage_prompts, prompt, stage_info)
        stage_text = kwargs.get(f"stage_text_{stage_index + 1}")
        if stage_text is not None:
            selected_prompt = str(stage_text)

        if stage_data is None and stage_info is not None and stage_index > 0:
            stage_data = stage_info.get("stage_data")

        media_values = dict(kwargs)
        if stage_data is not None:
            media_values["media"] = stage_data
        if context is None:
            blocker = ExecutionBlocker(None)
            return blocker, blocker, blocker, _ad_preview_prompt(selected_prompt, media_values)

        selected_prompt, selected_kwargs, _references = _ad_select_prompt_media(
            selected_prompt,
            media_values,
            "AD_MinMax_Ref2_mul",
        )
        clip = context.get("clip")
        vae = context.get("vae")
        audio_vae = context.get("audio_vae")
        missing = [name for name, value in (("clip", clip), ("vae", vae), ("audio_vae", audio_vae)) if value is None]
        if missing:
            raise ValueError(f"AD_MinMax_Ref2_mul context is missing: {', '.join(missing)}")

        exact_audio = None
        if single_long_video_split or single_long_audio_split:
            segment_frames = int(length)
            stride_frames = max(1, segment_frames - int(default_trim_frames))
            start_seconds = stage_index * stride_frames / float(long_split_FPS)
            duration_seconds = segment_frames / float(long_split_FPS)

            if single_long_video_split:
                video_name, long_video = _ad_single_split_material(selected_kwargs, "video")
                split_video = _ad_split_video(long_video, start_seconds, duration_seconds)
                if split_video is None:
                    _ad_remove_split_material(selected_kwargs, video_name)
                else:
                    selected_kwargs[video_name] = split_video
                    exact_audio = split_video["audio"]

            if single_long_audio_split:
                audio_name, long_audio = _ad_single_split_material(selected_kwargs, "audio")
                exact_audio = _ad_split_audio(long_audio, start_seconds, duration_seconds)
                if exact_audio is None:
                    _ad_remove_split_material(selected_kwargs, audio_name)
                else:
                    selected_kwargs[audio_name] = exact_audio

        active_model = model
        if exact_audio is not None:
            active_model = model if model is not None else context.get("model")
            if active_model is None:
                raise ValueError("AD_MinMax_Ref2_mul: audio lock needs a model in context or the model input")

        items = _ad_collect_media(selected_kwargs)
        has_context_latent = any(item[1] == "latent" for item in items)
        upstream_latent = None if has_context_latent else context.get("latent")
        has_context_latent = has_context_latent or upstream_latent is not None
        positive, latent, _trim_frames, text = AD_MiniMax_guide.execute(
            self,
            selected_prompt,
            width,
            height,
            length,
            ref_image_size,
            clip=clip,
            vae=vae,
            audio_vae=audio_vae,
            _context_latent=upstream_latent,
            **selected_kwargs,
        )
        if exact_audio is not None:
            active_model, latent, exact_audio = AptMiniMaxH3NativeAudioLock().lock_audio(
                active_model, latent, audio_vae, exact_audio
            )

        output_context = new_context(
            context,
            model=active_model,
            positive=positive,
            latent=latent,
            pos=text,
        )
        audio_output = exact_audio if exact_audio is not None else ExecutionBlocker(None)
        trim_frames = int(default_trim_frames) if has_context_latent else 0

        # 多段合并 text 输出（每段用 #segmentN--------- 分隔）；pos 保留当前段供 sampler 使用
        all_prompts = _ad_stage_prompts(stage_prompts, prompt)
        stage_text_overrides = {}
        for index in range(1, _AD_GUIDE_MAX_STAGE_TEXT + 1):
            override = kwargs.get(f"stage_text_{index}")
            if override is not None:
                stage_text_overrides[index] = str(override)
        merged_text = _ad_format_segmented_text_ref2(
            all_prompts, media_values, "AD_MinMax_Ref2_mul",
            stage_index, stage_text_overrides,
        )
        return output_context, audio_output, trim_frames, merged_text


class _AD_MinMax_FL2Base(_AD_MinMaxBase):
    """Shared FL2VA input and media handling."""

    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "dynamicPrompts": True,
                "socketless": True,
            }),
            "width": ("INT", {
                "default": 512, "min": 32, "max": 4096, "step": 32,
            }),
            "height": ("INT", {
                "default": 768, "min": 32, "max": 4096, "step": 32,
            }),
            "length": ("INT", {
                "default": 124, "min": 5, "max": 3600, "step": 17,
            }),
            "single_image_position": (["auto", "first", "last"], {
                "default": "auto",
            }),
            "seed": ("INT", {
                "default": 0,
                "min": 0,
                "max": 0xffffffffffffffff,
                "control_after_generate": True,
            }),
        }
        media_input = ("IMAGE,LATENT", {"lazy": True})
        optional = {
            "context": ("RUN_CONTEXT",),
            "model": ("MODEL", {"lazy": True}),
            "fps": ("FLOAT", {
                "default": 24.0,
                "min": 1.0,
                "max": 120.0,
                "step": 1.0,
            }),
            "media": ("IMAGE,LATENT,STRING",),
        }
        for index in range(1, _AD_GUIDE_MAX_MEDIA + 1):
            optional[f"media_{index}"] = media_input
            optional[f"media_type_{index}"] = ("STRING", {"default": ""})
        return {"required": required, "optional": optional}

    def check_lazy_status(self, context=None, **kwargs):
        if context is None:
            return []
        required = ["model"] if "model" in kwargs and kwargs.get("model") is None else []
        required.extend(
            f"media_{index}"
            for index in range(1, _AD_GUIDE_MAX_MEDIA + 1)
            if str(kwargs.get(f"media_type_{index}") or "").strip().lower() in {"image", "latent"}
            and kwargs.get(f"media_{index}") is None
        )
        return required

    @staticmethod
    def _collect_media(kwargs):
        items = AD_MiniMax_guide._collect_media(kwargs)
        unsupported = [item[1] for item in items if item[1] not in {"image", "latent"}]
        if unsupported:
            raise ValueError("AD_MinMax_FL2_mul Media only accepts image, latent or text")
        images = [item for item in items if item[1] == "image"]
        latents = [item for item in items if item[1] == "latent"]
        if len(images) > 2:
            raise ValueError("AD_MinMax_FL2_mul accepts at most two ordered images")
        if len(latents) > 1:
            raise ValueError("AD_MinMax_FL2_mul accepts only one context latent")
        return items

    def execute(self, prompt, width, height, length, seed,
                single_image_position="auto", context=None, model=None, fps=24.0, **kwargs):
        if isinstance(kwargs.get("media"), str):
            prompt = kwargs["media"]
        if context is None:
            media_types = _ad_preview_media_types(kwargs)
            resolved_prompt = _ad_fl2_prompt(
                prompt,
                media_types.count("image"),
                single_image_position,
                _ad_h3_frame_count(length),
                "latent" in media_types,
            )
            blocker = ExecutionBlocker(None)
            return blocker, blocker, resolved_prompt
        if _h3_empty_av_latent is None or _h3_resize is None or _node_helpers is None:
            raise RuntimeError("This ComfyUI build does not provide MiniMax H3 support")
        clip = context.get("clip")
        vae = context.get("vae")
        audio_vae = context.get("audio_vae")
        missing = [
            name for name, value in (
                ("clip", clip), ("vae", vae), ("audio_vae", audio_vae)
            ) if value is None
        ]
        if missing:
            raise ValueError(f"AD_MinMax_FL2_mul context is missing: {', '.join(missing)}")

        items = _ad_collect_media(kwargs)
        images = [item for item in items if item[1] == "image"]
        explicit_latents = [item for item in items if item[1] == "latent"]
        context_latent = explicit_latents[0][2] if explicit_latents else context.get("latent")
        has_context_latent = context_latent is not None

        latent, frame_count = _h3_empty_av_latent(width, height, length)
        keyframe_images = []
        keyframes = []
        position_aliases = {
            "auto": "auto", "\u81ea\u52a8": "auto",
            "first": "first", "\u9996\u5e27": "first",
            "last": "last", "\u5c3e\u5e27": "last",
        }
        single_position = position_aliases.get(
            str(single_image_position or "auto").strip().lower(), "auto"
        )
        if len(images) == 1 and single_position == "auto":
            single_position = "last" if has_context_latent else "first"

        if images and (len(images) > 1 or single_position == "first"):
            first = _h3_resize(images[0][2][:1], width, height, "disabled")
            keyframe_images.append(first)
            keyframes.append({**h3_keyframe_anchor(0), "image": first})
        if images and (len(images) > 1 or single_position == "last"):
            last_source = images[1][2] if len(images) > 1 else images[0][2]
            last = _h3_resize(last_source[:1], width, height, "center")
            keyframe_images.append(last)
            keyframes.append({
                **h3_keyframe_anchor(frame_count - 1),
                "image": last,
            })

        resolved_prompt = _ad_fl2_prompt(
            prompt,
            len(images),
            single_position,
            frame_count,
            has_context_latent,
        )
        tokens = clip.tokenize(resolved_prompt, images=keyframe_images)
        positive = clip.encode_from_tokens_scheduled(tokens)
        if keyframes:
            for keyframe in keyframes:
                keyframe["latent"] = vae.encode(keyframe.pop("image"))
            positive = _node_helpers.conditioning_set_values(positive, {
                "minimax_keyframes": keyframes,
                "minimax_frame_count": frame_count,
            })
        if context_latent is not None:
            positive, _ = AptMiniMaxH3MotionContext().apply(
                positive,
                latent,
                trim_frames=_AD_GUIDE_CONTEXT_LENGTH,
                context_latent=context_latent,
                audio_context_length=_AD_GUIDE_AUDIO_CONTEXT_LENGTH,
            )
        return self._sample_video(
            context, model, positive, latent, seed, fps,
            has_context_latent, resolved_prompt,
        )


class AD_MinMax_FL2_mul(_AD_MinMax_FL2Base):
    """Queue-stage FL2VA conditioning for an external sampler and video output."""

    RETURN_TYPES = ("RUN_CONTEXT", "INT", "STRING")
    RETURN_NAMES = ("context", "trim_frames", "text")

    @classmethod
    def INPUT_TYPES(cls):
        inherited = _AD_MinMax_FL2Base.INPUT_TYPES()
        required = dict(inherited["required"])
        required.pop("single_image_position", None)
        required.pop("seed", None)
        required["stage_prompts"] = ("STRING", {"default": "[]", "multiline": True})
        required["default_trim_frames"] = (
            "INT",
            {
                "default": 22,
                "min": 1,
                "max": 56,
                "step": 1,
                "tooltip": "Motion-context frames reused and removed from continuation clips.",
            },
        )
        optional = dict(inherited["optional"])
        optional.pop("fps", None)
        optional["stage_info"] = ("FLOW_STAGE_INFO",)
        optional["stage_data"] = ("IMAGE,VIDEO,LATENT",)
        for index in range(1, _AD_GUIDE_MAX_STAGE_TEXT + 1):
            optional[f"stage_text_{index}"] = ("STRING", {"forceInput": True})
        return {"required": required, "optional": optional}

    def check_lazy_status(self, stage_prompts, prompt="", stage_info=None, context=None, **kwargs):
        if context is None:
            return []
        _selected, references = _ad_stage_prompt_plan(stage_prompts, prompt, stage_info)
        required = ["model"] if "model" in kwargs and kwargs.get("model") is None else []
        required.extend(f"media_{index}" for index in references if kwargs.get(f"media_{index}") is None)
        return required

    def execute(self, prompt, width, height, length, stage_prompts, default_trim_frames,
                context=None, model=None, stage_info=None, stage_data=None, **kwargs):
        node_name = self.__class__.__name__
        if stage_info is None:
            stage_index = 0
        else:
            _run_id, stage_index, _total = _ad_stage_info(stage_info)
        selected_prompt, references = _ad_stage_prompt_plan(stage_prompts, prompt, stage_info)
        stage_text = kwargs.get(f"stage_text_{stage_index + 1}")
        if stage_text is not None:
            selected_prompt = str(stage_text)

        if stage_data is None and stage_info is not None and stage_index > 0:
            stage_data = stage_info.get("stage_data")

        media_values = dict(kwargs)
        if context is None:
            preview_prompt = _ad_preview_prompt(selected_prompt, media_values)
            image_count = sum(
                1 for index in references
                if str(media_values.get(f"media_type_{index}") or "").strip().lower() == "image"
            )
            if stage_index > 0 and image_count == 1:
                preview_prompt = _AD_FL2_PICTURE_RE.sub(
                    lambda match: "Picture 2" if match.group(1) == "1" else match.group(0),
                    preview_prompt,
                )
                image_count = 2
            text = _ad_fl2_prompt(
                preview_prompt,
                image_count,
                "first",
                _ad_h3_frame_count(length),
                stage_index > 0,
            )
            blocker = ExecutionBlocker(None)
            return blocker, blocker, text

        selected_prompt, selected_kwargs, _references = _ad_select_prompt_media(
            selected_prompt,
            media_values,
            node_name,
        )
        clip = context.get("clip")
        vae = context.get("vae")
        missing = [name for name, value in (("clip", clip), ("vae", vae)) if value is None]
        if missing:
            raise ValueError(f"{node_name} context is missing: {', '.join(missing)}")

        items = _ad_collect_media(selected_kwargs)
        images = [item for item in items if item[1] == "image"]
        explicit_latents = [item for item in items if item[1] == "latent"]
        if not images:
            raise ValueError(f"{node_name} stage {stage_index + 1} needs at least one image")

        stage_latent = stage_data if isinstance(stage_data, collections.abc.Mapping) and "samples" in stage_data else None
        context_latent = explicit_latents[0][2] if explicit_latents else stage_latent
        if context_latent is None:
            context_latent = context.get("latent")
        has_context_latent = context_latent is not None

        image_sources = [item[2] for item in images]
        if stage_index > 0 and len(image_sources) == 1:
            previous_last = _ad_last_frame(stage_data)
            if previous_last is None:
                previous_last = _ad_last_frame(context.get("images"))
            if previous_last is None:
                raise ValueError(
                    f"{node_name} stage {stage_index + 1} has one image but cannot find the previous stage's last frame"
                )
            image_sources.insert(0, previous_last)

        latent, frame_count = _h3_empty_av_latent(width, height, length)
        keyframe_images = []
        keyframes = []
        first = _h3_resize(image_sources[0][:1], width, height, "disabled")
        keyframe_images.append(first)
        keyframes.append({**h3_keyframe_anchor(0), "image": first})
        if len(image_sources) > 1:
            last = _h3_resize(image_sources[1][:1], width, height, "center")
            keyframe_images.append(last)
            keyframes.append({**h3_keyframe_anchor(frame_count - 1), "image": last})

        resolved_input = _ad_preview_prompt(selected_prompt, selected_kwargs)
        if stage_index > 0 and len(images) == 1:
            resolved_input = _AD_FL2_PICTURE_RE.sub(
                lambda match: "Picture 2" if match.group(1) == "1" else match.group(0),
                resolved_input,
            )
        resolved_prompt = _ad_fl2_prompt(
            resolved_input,
            len(image_sources),
            "first",
            frame_count,
            has_context_latent,
        )
        tokens = clip.tokenize(resolved_prompt, images=keyframe_images)
        positive = clip.encode_from_tokens_scheduled(tokens)
        for keyframe in keyframes:
            keyframe["latent"] = vae.encode(keyframe.pop("image"))
        positive = _node_helpers.conditioning_set_values(positive, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": frame_count,
        })

        trim_frames = 0
        if context_latent is not None:
            positive, trim_frames = AptMiniMaxH3MotionContext().apply(
                positive,
                latent,
                trim_frames=default_trim_frames,
                context_latent=context_latent,
                audio_context_length=_AD_GUIDE_AUDIO_CONTEXT_LENGTH,
            )
        output_context = new_context(
            context,
            model=model,
            positive=positive,
            latent=latent,
            pos=resolved_prompt,
        )

        # 多段合并 text 输出（每段用 #segmentN--------- 分隔）
        all_prompts = _ad_stage_prompts(stage_prompts, prompt)
        stage_text_overrides = {}
        for index in range(1, _AD_GUIDE_MAX_STAGE_TEXT + 1):
            override = kwargs.get(f"stage_text_{index}")
            if override is not None:
                stage_text_overrides[index] = str(override)
        merged_text = _ad_format_segmented_text_fl2(
            all_prompts, media_values, node_name, length,
            stage_index, stage_text_overrides,
        )
        return output_context, int(trim_frames), merged_text


class AD_MinMax_FL2_generate(AD_MinMax_FL2_mul):
    """Queue-stage FL2VA sampler with optional second pass and video merging."""

    RETURN_TYPES = ("LATENT", "VIDEO", "VIDEO", "STRING")
    RETURN_NAMES = ("denoise_latent1", "segment_video", "merged_video", "text")

    @classmethod
    def INPUT_TYPES(cls):
        inherited = _AD_MinMax_FL2Base.INPUT_TYPES()
        required = dict(inherited["required"])
        required.pop("single_image_position", None)
        required["stage_prompts"] = ("STRING", {"default": "[]", "multiline": True})
        _ad_add_second_pass_inputs(required)
        optional = dict(inherited["optional"])
        optional["stage_info"] = ("FLOW_STAGE_INFO",)
        optional["stage_data"] = ("IMAGE,VIDEO,LATENT",)
        for index in range(1, _AD_GUIDE_MAX_STAGE_TEXT + 1):
            optional[f"stage_text_{index}"] = ("STRING", {"forceInput": True})
        hidden = {"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"}
        return {"required": required, "optional": optional, "hidden": hidden}

    def execute(self, prompt, width, height, length, seed, stage_prompts,
                second_pass_mode, refine_model, refine_denoise, refine_steps,
                latent_model, latent_scale, split_step,
                context=None, model=None, fps=24.0, stage_info=None, stage_data=None,
                unique_id=None, workflow_prompt=None, **kwargs):
        if stage_info is None:
            run_id, stage_index, total = None, 0, 1
        else:
            run_id, stage_index, total = _ad_stage_info(stage_info)
        prepared_context, trim_frames, text = super().execute(
            prompt, width, height, length, stage_prompts, _AD_GUIDE_TRIM_FRAMES,
            context=context, model=model, stage_info=stage_info, stage_data=stage_data,
            **kwargs,
        )
        if context is None:
            blocker = ExecutionBlocker(None)
            return blocker, blocker, blocker, text

        denoise_latent1, video, text = self._sample_video(
            context,
            prepared_context.get("model"),
            prepared_context.get("positive"),
            prepared_context.get("latent"),
            seed,
            fps,
            int(trim_frames) > 0,
            text,
            second_pass_mode,
            refine_model,
            refine_denoise,
            refine_steps,
            latent_model,
            latent_scale,
            split_step,
        )
        video, merged_video = _ad_stage_video_outputs(
            video, run_id, stage_index, total, workflow_prompt, unique_id,
            "AD_MinMax_FL2_generate",
        )
        return denoise_latent1, video, merged_video, text


#endregion----------MiniMax H3 Guide---------------














