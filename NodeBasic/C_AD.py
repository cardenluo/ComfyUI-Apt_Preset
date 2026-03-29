
import torch
import numpy as np
from typing import Any
import math
from PIL import Image, ImageDraw
import random
import torch.nn.functional as F
from io import BytesIO
import hashlib
import collections.abc
from ..main_unit import *



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









import os
import av
import torch
from typing import Optional, List
from fractions import Fraction
from comfy_api.latest import ComfyExtension, io, Input, InputImpl, Types

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
import numpy as np
import folder_paths
from PIL import Image

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

    def _adjust_to_target(self, scenes, target, total_frames, fps=30):
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





