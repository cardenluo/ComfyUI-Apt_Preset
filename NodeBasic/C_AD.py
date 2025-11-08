
import torch
import numpy as np
from typing import Any
import math
from PIL import Image, ImageDraw, ImageFilter, ImageOps, ImageChops, ImageEnhance
import random

import torch.nn.functional as F
import pandas as pd
from io import BytesIO
import hashlib
import matplotlib.pyplot as plt

from scipy.fft import fft
import collections.abc

from ..main_unit import *



#region-----------------收纳--------------------

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None
    print("Warning: pydub not installed. Audio-related nodes will not be available.")









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


class AD_sch_prompt_preset:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keyframe_interval": ("INT", {"default": 30, "min": 0, "max": 999, "step": 1}),
                "loops": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "prompt_1": ("STRING", {"multiline": True, "default": "dancing,(neon:1.3)"}),
            },
            "optional": {
                "prompt_2": ("STRING", {"multiline": True, "default": "(dancing:`pw_a`)"}),
                "prompt_3": ("STRING", {"multiline": True, "default": "(smiling:`(0.5+0.5*sin(t/max_f))`"}),
                "prompt_4": ("STRING", {"multiline": True, "default": "(b:`(1.5*cos(1.57*(t-30)/30)*cos(1.57*(t-30)/30))`)"}),
                "prompt_5": ("STRING", {"multiline": True}),
                "prompt_6": ("STRING", {"multiline": True}),
                "prompt_7": ("STRING", {"multiline": True}),
                "prompt_8": ("STRING", {"multiline": True}),
                "prompt_9": ("STRING", {"multiline": True}),
                "prompt_10": ("STRING", {"multiline": True}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("sch_prompt",)
    FUNCTION = "make_keyframes"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def make_keyframes(self, keyframe_interval, loops, prompt_1, prompt_2="", prompt_3="", prompt_4="", prompt_5="", prompt_6="", prompt_7="", prompt_8="", prompt_9="", prompt_10=""):
        prompts = []

        if prompt_1 != "":
            prompts.append(prompt_1)

        if prompt_2 != "":
            prompts.append(prompt_2)

        if prompt_3 != "":
            prompts.append(prompt_3)

        if prompt_4 != "":
            prompts.append(prompt_4)

        if prompt_5 != "":
            prompts.append(prompt_5)

        if prompt_6 != "":
            prompts.append(prompt_6)

        if prompt_7 != "":
            prompts.append(prompt_7)

        if prompt_8 != "":
            prompts.append(prompt_8)

        if prompt_9 != "":
            prompts.append(prompt_9)

        if prompt_10 != "":
            prompts.append(prompt_10)

        keyframe_list = []
        
        i = 0
        for j in range(1, loops + 1): 
            for index, prompt in enumerate(prompts):
                if i == 0:
                    keyframe_list.append(f"\"0\": \"{prompt}\",\n")
                    i += keyframe_interval  
                    continue
                
                new_keyframe = f"\"{i}\": \"{prompt}\",\n"
                keyframe_list.append(new_keyframe)
                i += keyframe_interval 
        
        sch_prompt = " ".join(keyframe_list)[:-2]
        
        return (sch_prompt,)


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
    CATEGORY = "Apt_Preset/AD"

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

    CATEGORY = "Apt_Preset/AD"

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


    CATEGORY = "Apt_Preset/AD"

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

    CATEGORY = "Apt_Preset/AD"
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("normalized_amp",)
    FUNCTION = "process_audio"

    def load_audio(self, audio):
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        waveform_np = waveform.squeeze().numpy()
        
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

    CATEGORY = "Apt_Preset/AD"
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
    CATEGORY = "Apt_Preset/AD"

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


NODE_CLASS_MAPPINGS = {
    "Amp_drive_value": Amp_drive_value,
    "Amp_drive_String": Amp_drive_String,
    "Amp_audio_Normalized": Amp_audio_Normalized,
    "Amp_drive_mask": Amp_drive_mask,
}

NODE_DISPLAY_NAME_MAPPINGS = {

}

