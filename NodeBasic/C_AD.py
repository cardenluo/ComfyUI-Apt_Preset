
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
from comfy_api.latest import io, Input, InputImpl, Types

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
    CATEGORY = "Apt_Preset/AD"
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
    CATEGORY = "Apt_Preset/AD"

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





#region----------LTX---------------
# 导入需要的节点

import folder_paths
from comfy_extras.nodes_video import CreateVideo
from nodes import CheckpointLoader, CLIPTextEncode, VAEDecodeTiled,CheckpointLoaderSimple, LoraLoaderModelOnly
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader


from comfy_extras.nodes_custom_sampler import (
    RandomNoise,
    CFGGuider,
    SamplerCustomAdvanced
)

from comfy_extras.nodes_lt_audio import LTXAVTextEncoderLoader, LTXVEmptyLatentAudio,LTXVAudioVAELoader
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler


from comfy_extras.nodes_lt import (
    EmptyLTXVLatentVideo,
    LTXVConcatAVLatent,
    LTXVSeparateAVLatent,
    LTXVCropGuides,
    LTXVAddGuide,
    LTXVConditioning,
    get_noise_mask,
    _append_guide_attention_entry
)

from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode
from comfy_api.latest import io
import torch
import numpy as np
from PIL import Image, ImageDraw


HISTORY = io.Custom("HISTORY")
KEYFRAME_TREND = io.Custom("KEYFRAME_TREND")
LTX_CONFIG = io.Custom("LTX_CONFIG")
RUN_CONTEXT = io.Custom("RUN_CONTEXT")


class AD_Latent_Diffusion_Keyframe(LTXVAddGuide):
    DESCRIPTION = """
    多段衔接设置要点：
    - 适用场景：第一段生成后，把 history 输出接到下一段 history 输入
    - 第一段建议：num_guides=2，frame_idx_1=0，frame_idx_2 设在本段末尾附近（避免越界）
    - 第二段建议：num_guides=1，history_strength 先用 0.30~0.45，history_fade_frames 先用 3~6
    - 为减少跳变：第二段 image_1 的 frame_idx_1 放在 16~32，再按效果微调
    - 为避免首帧混入：不要把第二段目标图放在 frame 0 且强度过高
    - 调参顺序：先调 frame_idx_1，再调 history_strength，最后调 history_fade_frames
    """
    @classmethod
    def define_schema(cls):
        options = []
        for num_guides in range(1, 11):
            guide_inputs = []
            for i in range(1, num_guides + 1):
                guide_inputs.extend([
                    io.Image.Input(f"image_{i}"),
                    io.Int.Input(f"frame_idx_{i}", default=0, min=0, max=9999),
                    io.Float.Input(f"strength_{i}", default=0.85, min=0.0, max=1.0, step=0.01),
                ])
            options.append(io.DynamicCombo.Option(key=str(num_guides), inputs=guide_inputs))

        return io.Schema(
            node_id="AD_Latent_Diffusion_Keyframe",
            category= "Apt_Preset/🚫Deprecated/🚫",
            description="AD video keyframe auto relay",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("img_latent"),
                io.Vae.Input("img_vae"),
                HISTORY.Input("history", optional=True),
                io.Float.Input("history_strength", default=0.5, min=0.0, max=1.0, step=0.01, display_name="History Strength"),
                io.Int.Input("history_fade_frames", default=16, min=1, max=100, step=1, display_name="Fade Frames"),
                io.DynamicCombo.Input("num_guides", options=options, display_name="Guides"),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("img_latent"),
                io.Vae.Output(display_name="img_vae"),
                HISTORY.Output("history"),
                KEYFRAME_TREND.Output("trend_data"),
            ],
        )

    @classmethod
    def _build_trend_data(cls, points, frame_length):
        cleaned = []
        for item in sorted(points, key=lambda x: x["frame"]):
            frame = int(max(0, item["frame"]))
            strength = float(max(0.0, min(1.0, item["strength"])))
            source = str(item.get("source", "guide"))
            if cleaned and cleaned[-1]["frame"] == frame:
                cleaned[-1] = {"frame": frame, "strength": strength, "source": source}
            else:
                cleaned.append({"frame": frame, "strength": strength, "source": source})
        if not cleaned:
            cleaned = [
                {"frame": 0, "strength": 0.0, "source": "none"},
                {"frame": max(0, frame_length - 1), "strength": 0.0, "source": "none"},
            ]
        frame_length = max(int(frame_length), int(cleaned[-1]["frame"]) + 1)
        if len(cleaned) == 1:
            cleaned.append({"frame": max(0, frame_length - 1), "strength": cleaned[0]["strength"], "source": cleaned[0]["source"]})
        segments = []
        frame_span = max(1, frame_length - 1)
        for i in range(len(cleaned) - 1):
            p0 = cleaned[i]
            p1 = cleaned[i + 1]
            span = max(1, p1["frame"] - p0["frame"])
            delta = abs(p1["strength"] - p0["strength"])
            span_ratio = min(1.0, span / frame_span)
            curvature = max(0.05, min(0.95, 0.15 + 0.55 * delta + 0.30 * span_ratio))
            bend = 1.0 if p1["strength"] >= p0["strength"] else -1.0
            segments.append({
                "start_frame": p0["frame"],
                "end_frame": p1["frame"],
                "start_strength": p0["strength"],
                "end_strength": p1["strength"],
                "curvature": curvature,
                "bend": bend,
                "source": "mixed" if p0["source"] != p1["source"] else p1["source"],
            })
        return {
            "frame_length": int(frame_length),
            "points": cleaned,
            "segments": segments,
        }

    @classmethod
    def execute(cls, positive, negative, img_vae, img_latent, history=None, history_strength=0.5, history_fade_frames=16, **kwargs):
        vae = img_vae
        latent = img_latent
        keep_history_keyframes = True
        continuity_tail_frames = max(1, history_fade_frames)
        trend_points = []
        trend_max_frame = 0

        base_frame_offset = 0
        history_latent = None
        if history is not None:
            base_frame_offset = history.get("base_frame_offset", 0)
            history_latent = history.get("latent")
            continuity_tail_frames = max(1, int(history.get("tail_frames", continuity_tail_frames)))
        trend_max_frame = max(0, base_frame_offset)

        guide_payload = kwargs.get("num_guides", {})
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)
        _, _, latent_length, _, _ = latent_image.shape

        def _fade_ratio(step, fade_steps):
            x = step / fade_steps
            y = x*x*(3-2*x)
            return max(0.0, min(1.0, 1-y))

        history_tail_source = None
        if history_latent is not None and "samples" in history_latent:
            h_samples = history_latent["samples"]
            if isinstance(h_samples, torch.Tensor) and h_samples.ndim == 5:
                frames = min(max(1, continuity_tail_frames), h_samples.shape[2])
                h_clip = h_samples[:, :, -frames:]
                if h_clip.shape[0] != latent_image.shape[0]:
                    h_clip = h_clip[:1].repeat(latent_image.shape[0], 1,1,1,1)
                h_clip = h_clip.to(device=latent_image.device, dtype=latent_image.dtype)
                if h_clip.shape[3:] != latent_image.shape[3:]:
                    n,c,t,h,w = h_clip.shape
                    resized = torch.nn.functional.interpolate(
                        h_clip.permute(0,2,1,3,4).reshape(-1,c,h,w),
                        size=latent_image.shape[3:], mode="bilinear", align_corners=False
                    )
                    h_clip = resized.reshape(n,t,c,latent_image.shape[3],latent_image.shape[4]).permute(0,2,1,3,4)
                history_tail_source = h_clip
                fade_steps = min(max(1, history_fade_frames), latent_length)
                src_len = h_clip.shape[2]
                for step in range(fade_steps):
                    s = history_strength * _fade_ratio(step, fade_steps)
                    if s <= 0: continue
                    src_idx = max(0, src_len - 1 - step)
                    step_guide = h_clip[:,:,src_idx:src_idx+1]
                    f_idx, l_idx = cls.get_latent_index(positive, latent_length, 1, step, scale_factors)
                    if l_idx+1 > latent_length: continue
                    history_frame = base_frame_offset + int(step)
                    trend_max_frame = max(trend_max_frame, history_frame)
                    trend_points.append({"frame": history_frame, "strength": float(s), "source": "history"})
                    positive, negative, latent_image, noise_mask = cls.append_keyframe(
                        positive, negative, f_idx, latent_image, noise_mask, step_guide, s, scale_factors
                    )

        guides = []
        for k in guide_payload:
            if not k.startswith("image_"): continue
            idx = k.split("_",1)[1]
            img = guide_payload.get(k)
            if img is None: continue
            f = int(guide_payload.get(f"frame_idx_{idx}",0))
            s = float(guide_payload.get(f"strength_{idx}",0.85))
            guides.append((f, img, max(0.0, min(1.0, s))))
        guides.sort(key=lambda x:x[0])

        latest_guide_source = None
        for f_idx, img, s in guides:
            _, g_latent = cls.encode(vae, latent_image.shape[4], latent_image.shape[3], img, scale_factors)
            latest_guide_source = g_latent.to(device=latent_image.device, dtype=latent_image.dtype)
            fi, li = cls.get_latent_index(positive, latent_length, g_latent.shape[2], f_idx, scale_factors)
            if li + g_latent.shape[2] > latent_length:
                time_scale_factor = scale_factors[0]
                max_latent_idx = max(0, latent_length - g_latent.shape[2])
                if max_latent_idx == 0:
                    clamped_frame_idx = 0
                elif g_latent.shape[2] > 1:
                    clamped_frame_idx = (max_latent_idx - 1) * time_scale_factor + 1
                else:
                    clamped_frame_idx = max_latent_idx * time_scale_factor
                fi, li = cls.get_latent_index(positive, latent_length, g_latent.shape[2], clamped_frame_idx, scale_factors)
            if li + g_latent.shape[2] > latent_length: continue
            guide_frame = base_frame_offset + int(f_idx)
            trend_max_frame = max(trend_max_frame, guide_frame)
            trend_points.append({"frame": guide_frame, "strength": float(s), "source": "guide"})
            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive, negative, fi, latent_image, noise_mask, g_latent, s, scale_factors
            )
        if latest_guide_source is not None:
            history_tail_source = latest_guide_source

        source_limit = min(latent_length, latent_image.shape[2])
        source_samples = latent_image[:, :, :source_limit]
        source_noise_mask = noise_mask[:, :, :source_limit]
        if history_tail_source is not None:
            if history_tail_source.shape[0] != latent_image.shape[0]:
                history_tail_source = history_tail_source[:1].repeat(latent_image.shape[0], 1,1,1,1)
            if history_tail_source.shape[3:] != latent_image.shape[3:]:
                n,c,t,h,w = history_tail_source.shape
                resized = torch.nn.functional.interpolate(
                    history_tail_source.permute(0,2,1,3,4).reshape(-1,c,h,w),
                    size=latent_image.shape[3:], mode="bilinear", align_corners=False
                )
                history_tail_source = resized.reshape(n,t,c,latent_image.shape[3],latent_image.shape[4]).permute(0,2,1,3,4)
            source_samples = history_tail_source
            source_noise_mask = torch.ones_like(source_samples[:, :1])
        cont_frames = min(max(1, continuity_tail_frames), source_samples.shape[2])
        continuity_latent = {
            "samples": source_samples[:, :, -cont_frames:].clone(),
            "noise_mask": source_noise_mask[:, :, -cont_frames:].clone()
        }

        output_history = {
            "base_frame_offset": base_frame_offset + latent_length,
            "latent": continuity_latent,
            "tail_frames": cont_frames
        }
        trend_max_frame = max(trend_max_frame, base_frame_offset + latent_length - 1)
        total_frame_length = max(base_frame_offset + latent_length, trend_max_frame + 1)
        trend_data = cls._build_trend_data(trend_points, total_frame_length)
        trend_data["segment_start"] = int(base_frame_offset)
        trend_data["segment_length"] = int(latent_length)
        trend_data["total_frame_length"] = int(total_frame_length)

        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, vae, output_history, trend_data)



class AD_latent_history(io.ComfyNode):
    DESCRIPTION = """
    将采样后的 LATENT 打包为 HISTORY，用于下一段 AD_扩散关键帧衔接。
    """
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_latent_history",
            category= "Apt_Preset/🚫Deprecated/🚫",
            description="Pack sampled latent tail to HISTORY",
            inputs=[
                io.Latent.Input("latent"),
            ],
            outputs=[
                HISTORY.Output("history"),
            ],
        )

    @classmethod
    def execute(cls, latent):
        def _safe_copy(x):
            if hasattr(x, "clone"):
                return x.clone()
            if hasattr(x, "_copy"):
                return x._copy()
            return x

        latent_image = latent["samples"]
        _, _, latent_length, _, _ = latent_image.shape
        keep_frames = latent_length
        batch_size = latent_image.shape[0]
        noise_mask = torch.ones(
            (batch_size, 1, latent_length, 1, 1),
            dtype=torch.float32,
            device=latent_image.device,
        )
        sample_tail = latent_image[:, :, -keep_frames:]
        mask_tail = noise_mask[:, :, -keep_frames:]
        continuity_latent = {
            "samples": _safe_copy(sample_tail),
            "noise_mask": _safe_copy(mask_tail),
        }
        output_history = {
            "base_frame_offset": latent_length,
            "latent": continuity_latent,
            "tail_frames": keep_frames,
        }
        return io.NodeOutput(output_history)


class AD_LTX_MulGuide(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        options = []
        for num_guides in range(1, 11):
            guide_inputs = []
            for i in range(1, num_guides + 1):
                guide_inputs.extend([
                    io.Image.Input(f"image_{i}"),
                    io.Int.Input(f"frame_idx_{i}", default=0, min=0, max=9999),
                    io.Float.Input(f"strength_{i}", default=0.85, min=0.0, max=1.0, step=0.01),
                ])
            options.append(io.DynamicCombo.Option(key=str(num_guides), inputs=guide_inputs))
        return io.Schema(
            node_id="AD_LTX_MulGuide",
            category="Apt_Preset/AD/LTX_video",
            description="LTX multi-guide simplified",
            inputs=[
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("img_latent"),
                io.Vae.Input("img_vae"),
                io.DynamicCombo.Input("num_guides", options=options, display_name="Guides"),
            ],
            outputs=[
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("img_latent"),
                io.Vae.Output(display_name="img_vae"),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, img_vae, img_latent, **kwargs):
        scale_factors = img_vae.downscale_index_formula
        latent_image = img_latent["samples"]
        noise_mask = get_noise_mask(img_latent)
        _, _, latent_length, _, _ = latent_image.shape
        guide_payload = kwargs.get("num_guides", {})
        guides = []
        for k in guide_payload:
            if not k.startswith("image_"):
                continue
            idx = k.split("_", 1)[1]
            img = guide_payload.get(k)
            if img is None:
                continue
            f = int(guide_payload.get(f"frame_idx_{idx}", 0))
            s = float(guide_payload.get(f"strength_{idx}", 0.85))
            guides.append((f, img, max(0.0, min(1.0, s))))
        guides.sort(key=lambda x: x[0])

        for f_idx, img, s in guides:
            _, g_latent = cls.encode(img_vae, latent_image.shape[4], latent_image.shape[3], img, scale_factors)
            fi, li = cls.get_latent_index(positive, latent_length, g_latent.shape[2], f_idx, scale_factors)
            if li + g_latent.shape[2] > latent_length:
                time_scale_factor = scale_factors[0]
                max_latent_idx = max(0, latent_length - g_latent.shape[2])
                if max_latent_idx == 0:
                    clamped_frame_idx = 0
                elif g_latent.shape[2] > 1:
                    clamped_frame_idx = (max_latent_idx - 1) * time_scale_factor + 1
                else:
                    clamped_frame_idx = max_latent_idx * time_scale_factor
                fi, li = cls.get_latent_index(positive, latent_length, g_latent.shape[2], clamped_frame_idx, scale_factors)
            if li + g_latent.shape[2] > latent_length:
                continue
            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive, negative, fi, latent_image, noise_mask, g_latent, s, scale_factors
            )
        return io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, img_vae)




class AD_keyframe_trend_preview(io.ComfyNode):
    INPUT_IS_LIST = True
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_keyframe_trend_preview",
            category="Apt_Preset/AD/LTX_video",
            description="Preview keyframe trend graph",
            inputs=[
                KEYFRAME_TREND.Input("trend_data"),
                io.Int.Input("width", default=1024, min=256, max=4096, step=1),
                io.Int.Input("height", default=320, min=128, max=2048, step=1),
            ],
            outputs=[
                io.Image.Output("graph"),
            ],
        )

    @classmethod
    def _draw_line(cls, canvas, x0, y0, x1, y1, color):
        steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
        if steps <= 1:
            xi = int(max(0, min(canvas.shape[1] - 1, round(x0))))
            yi = int(max(0, min(canvas.shape[0] - 1, round(y0))))
            canvas[yi, xi] = color
            return
        for i in range(steps + 1):
            t = i / steps
            x = int(max(0, min(canvas.shape[1] - 1, round(x0 + (x1 - x0) * t))))
            y = int(max(0, min(canvas.shape[0] - 1, round(y0 + (y1 - y0) * t))))
            canvas[y, x] = color

    @classmethod
    def _annotate_axes(cls, canvas, frame_length, left, right, top, bottom):
        img = Image.fromarray((canvas.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img)
        h, w = canvas.shape[0], canvas.shape[1]
        draw.text((6, top - 6), "Y(strength)", fill=(0, 0, 0))
        draw.text((w - right - 86, top - 6), "Y2(visibility%)", fill=(0, 0, 0))
        draw.text((w // 2 - 24, h - 18), "X(frame)", fill=(0, 0, 0))
        for gy in range(0, 5):
            ratio = gy / 4.0
            y = top + int((h - top - bottom - 1) * ratio)
            label = f"{1.0 - ratio:.2f}"
            draw.text((6, y - 6), label, fill=(0, 0, 0))
            vis_label = f"{int(round((1.0 - ratio) * 100))}%"
            draw.text((w - right - 34, y - 6), vis_label, fill=(0, 0, 0))
        for gx in range(0, 6):
            ratio = gx / 5.0
            x = left + int((w - left - right - 1) * ratio)
            frame_label = int(round((frame_length - 1) * ratio))
            draw.text((x - 8, h - bottom + 6), str(frame_label), fill=(0, 0, 0))
        draw.text((left + 8, top + 4), "Light Red/Blue: visibility%", fill=(0, 0, 0))
        return torch.from_numpy(np.array(img)).float() / 255.0

    @classmethod
    def execute(cls, trend_data, width=1024, height=320):
        if isinstance(width, list):
            width = width[0] if width else 1024
        if isinstance(height, list):
            height = height[0] if height else 320
        w = int(max(256, width))
        h = int(max(128, height))
        trend_items = trend_data if isinstance(trend_data, list) else [trend_data]
        normalized_items = []
        seen_signatures = set()
        for item in trend_items:
            if not isinstance(item, dict):
                continue
            points = item.get("points", [])
            sig_points = tuple(
                (int(p.get("frame", 0)), round(float(p.get("strength", 0.0)), 6), str(p.get("source", "")))
                for p in points
            )
            signature = (int(item.get("frame_length", 0)), sig_points)
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)
            normalized_items.append(item)
        trend_items = normalized_items if normalized_items else [{}]
        merged_points = []
        inferred_total_frame_length = 1
        for item in trend_items:
            if not isinstance(item, dict):
                continue
            frame_length_local = int(max(1, item.get("total_frame_length", item.get("frame_length", 1))))
            inferred_total_frame_length = max(inferred_total_frame_length, frame_length_local)
            points = item.get("points", [])
            render_points = [p for p in points if p.get("source") == "guide"]
            if len(render_points) < 2:
                render_points = points
            local_points = []
            for p in sorted(render_points, key=lambda x: x["frame"]):
                frame = int(max(0, min(frame_length_local - 1, p["frame"])))
                strength = float(max(0.0, min(1.0, p["strength"])))
                if local_points and local_points[-1]["frame"] == frame:
                    local_points[-1]["strength"] = strength
                else:
                    local_points.append({"frame": frame, "strength": strength})
            if not local_points:
                continue
            for p in local_points:
                merged_points.append({"frame": p["frame"], "strength": p["strength"]})
        if not merged_points:
            merged_points = [{"frame": 0, "strength": 1.0}, {"frame": 1, "strength": 0.0}]
            frame_length = 2
        else:
            merged_points = sorted(merged_points, key=lambda x: x["frame"])
            if len(merged_points) == 1:
                merged_points.append({"frame": merged_points[0]["frame"] + 1, "strength": merged_points[0]["strength"]})
            frame_length = int(max(inferred_total_frame_length, merged_points[-1]["frame"] + 1))
        canvas = torch.ones((h, w, 3), dtype=torch.float32)
        left, right, top, bottom = 64, 16, 16, 34
        plot_w = max(10, w - left - right)
        plot_h = max(10, h - top - bottom)
        grid_color = torch.tensor([0.88, 0.88, 0.88], dtype=torch.float32)
        axis_color = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32)
        vis_a_color = torch.tensor([0.75, 0.30, 0.30], dtype=torch.float32)
        vis_b_color = torch.tensor([0.30, 0.45, 0.85], dtype=torch.float32)
        for gy in range(1, 5):
            y = top + int((plot_h - 1) * gy / 5)
            canvas[y:y + 1, left:left + plot_w, :] = grid_color
        for gx in range(1, 10):
            x = left + int((plot_w - 1) * gx / 10)
            canvas[top:top + plot_h, x:x + 1, :] = grid_color
        canvas[top:top + plot_h, left:left + 1, :] = axis_color
        canvas[top + plot_h - 1:top + plot_h, left:left + plot_w, :] = axis_color
        frame_span = max(1, frame_length - 1)
        if len(merged_points) >= 2:
            for i in range(len(merged_points) - 1):
                p0 = merged_points[i]
                p1 = merged_points[i + 1]
                start = int(max(0, min(frame_span, p0["frame"])))
                end = int(max(start + 1, min(frame_span, p1["frame"])))
                span = max(1, end - start)
                delta = abs(p1["strength"] - p0["strength"])
                span_ratio = min(1.0, span / frame_span)
                curvature = max(0.05, min(0.95, 0.15 + 0.55 * delta + 0.30 * span_ratio))
                gamma = max(0.45, min(2.2, 1.8 - curvature * 1.2))
                s0 = float(max(0.0, min(1.0, p0["strength"])))
                s1 = float(max(0.0, min(1.0, p1["strength"])))
                prev_va = prev_vb = None
                sample_count = int(max(12, (end - start) * 4))
                for si in range(sample_count + 1):
                    t = si / max(1, sample_count)
                    eased = t ** gamma
                    w0 = s0 * (1.0 - eased)
                    w1 = s1 * eased
                    vis_sum = max(1e-6, w0 + w1)
                    v0 = w0 / vis_sum
                    v1 = w1 / vis_sum
                    frame = start + (end - start) * t
                    x = left + (frame / frame_span) * (plot_w - 1)
                    yv0 = top + (1.0 - v0) * (plot_h - 1)
                    yv1 = top + (1.0 - v1) * (plot_h - 1)
                    if prev_va is not None:
                        cls._draw_line(canvas, prev_va[0], prev_va[1], x, yv0, vis_a_color)
                        cls._draw_line(canvas, prev_vb[0], prev_vb[1], x, yv1, vis_b_color)
                    prev_va = (x, yv0)
                    prev_vb = (x, yv1)
        canvas = cls._annotate_axes(canvas, frame_length, left, right, top, bottom)
        return io.NodeOutput(canvas.unsqueeze(0))




class AD_LTX_audio_input(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_LTX_audio_input",
            category="Apt_Preset/AD/LTX_video",
            inputs=[
                io.Audio.Input("audio"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("start_index", default=0, min=-0xffffffffffffffff, max=0xffffffffffffffff, step=1),
                io.Int.Input("frame_rate", default=24, min=1, max=120),
                io.Int.Input("frames_number", default=97, min=1, max=2000, step=1),
                io.Float.Input("audio_denoise_scale", default=0.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(display_name="audio_latent"),
                io.Vae.Output(display_name="audio_vae"),
            ],
        )

    @classmethod
    def execute(cls, audio, audio_vae, start_index, frame_rate, frames_number, audio_denoise_scale) -> io.NodeOutput:
        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        audio_length = waveform.shape[-1]

        start_offset = int(round((float(start_index) * float(sample_rate)) / float(frame_rate)))
        if start_index < 0:
            start_frame = audio_length + start_offset
        else:
            start_frame = start_offset
        start_frame = max(0, min(start_frame, audio_length - 1))

        end_frame = start_frame + int(round((float(frames_number) * float(sample_rate)) / float(frame_rate)))
        end_frame = max(0, min(end_frame, audio_length))

        if start_frame >= end_frame:
            raise ValueError("AudioTrim: Start time must be less than end time and be within the audio length.")

        trimmed_audio = {"waveform": waveform[..., start_frame:end_frame], "sample_rate": sample_rate}
        audio_latents = audio_vae.encode(trimmed_audio)
        audio_latent = {
            "samples": audio_latents,
            "sample_rate": int(audio_vae.sample_rate),
            "type": "audio",
        }

        samples = audio_latent["samples"]
        if not isinstance(samples, torch.Tensor) or samples.ndim < 3:
            raise ValueError("audio_latent samples format is invalid.")
        mask = torch.full(
            (int(samples.shape[0]), int(samples.shape[-2]), int(samples.shape[-1])),
            float(audio_denoise_scale),
            dtype=torch.float32,
            device=samples.device
        )
        output = audio_latent.copy()
        output["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return io.NodeOutput(output, audio_vae)



class AD_LTX_vae_combine(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_LTX_vae_combine",
            category="Apt_Preset/AD/LTX_video",
            inputs=[
                io.Latent.Input("img_latent"),
                io.Vae.Input("img_vae"),
                io.Latent.Input("audio_latent", optional=True),
                io.Vae.Input("audio_vae", optional=True),
                io.Int.Input("frame_rate", default=24, min=1, max=120),
                io.Boolean.Input("lowGpu_mode", default=False,),
                io.Int.Input("trim_latent_end", default=0, min=0, max=4096, step=1),

            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Audio.Output(display_name="audio"),
                io.Video.Output(display_name="video"),
                io.Image.Output(display_name="last_image"),
            ]
        )

    @classmethod
    def execute(cls, img_latent, img_vae, audio_latent, audio_vae, frame_rate, lowGpu_mode, trim_latent_end) -> io.NodeOutput:
        if img_vae is None:
            raise ValueError("img_vae is None. Please connect a valid VAE to AD_LTX_vae_combine.")

        decode_latent = img_latent
        samples = img_latent.get("samples", None) if isinstance(img_latent, dict) else None
        trim_latent_end = int(trim_latent_end)
        if isinstance(samples, torch.Tensor) and samples.ndim >= 3 and trim_latent_end > 0:
            frame_count = int(samples.shape[2])
            if frame_count > 1:
                keep_frames = max(1, frame_count - trim_latent_end)
                trimmed = img_latent.copy()
                trimmed["samples"] = samples[:, :, :keep_frames, ...]
                if "noise_mask" in trimmed and isinstance(trimmed["noise_mask"], torch.Tensor):
                    mask = trimmed["noise_mask"]
                    if mask.ndim == 5:
                        trimmed["noise_mask"] = mask[:, :, :keep_frames, ...]
                    elif mask.ndim >= 4:
                        trimmed["noise_mask"] = mask[:, :keep_frames, ...]
                decode_latent = trimmed

        temporal_compression = 0
        if hasattr(img_vae, "temporal_compression_decode"):
            read_tc = img_vae.temporal_compression_decode()
            if read_tc is not None:
                temporal_compression = int(read_tc)

        rec_width = 1024
        rec_height = 1024
        rec_total_frames = 1
        decode_samples = decode_latent.get("samples", None) if isinstance(decode_latent, dict) else None
        if isinstance(decode_samples, torch.Tensor):
            if decode_samples.ndim == 5:
                rec_total_frames = max(1, int(decode_samples.shape[2]))
                latent_h = max(1, int(decode_samples.shape[3]))
                latent_w = max(1, int(decode_samples.shape[4]))
            elif decode_samples.ndim == 4:
                rec_total_frames = 1
                latent_h = max(1, int(decode_samples.shape[2]))
                latent_w = max(1, int(decode_samples.shape[3]))
            else:
                latent_h = 128
                latent_w = 128

            scale_factors = getattr(img_vae, "downscale_index_formula", None)
            spatial_h_factor = 8
            spatial_w_factor = 8
            if isinstance(scale_factors, (list, tuple)) and len(scale_factors) >= 3:
                try:
                    spatial_h_factor = max(1, int(scale_factors[1]))
                    spatial_w_factor = max(1, int(scale_factors[2]))
                except Exception:
                    spatial_h_factor = 8
                    spatial_w_factor = 8

            rec_width = max(64, latent_w * spatial_w_factor)
            rec_height = max(64, latent_h * spatial_h_factor)

        tile_size, overlap, temporal_size, temporal_overlap, _ = AD_AutoTileVAEDecode().recommend(
            int(rec_width),
            int(rec_height),
            int(rec_total_frames),
            lowGpu_mode,
            int(temporal_compression),
            img_vae,
        )

        vae_decode_node = VAEDecodeTiled().decode(
            img_vae,
            decode_latent,
            int(tile_size),
            int(overlap),
            int(temporal_size),
            int(temporal_overlap),
        )
        image = vae_decode_node[0]
        if isinstance(image, torch.Tensor) and image.ndim >= 4 and image.shape[0] > 0:
            last_image = image[-1:].clone()
        elif isinstance(image, list) and len(image) > 0:
            last_image = image[-1]
        else:
            last_image = image

        audio = None
        if audio_latent is not None and audio_vae is not None:
            audio_decode_node = LTXVAudioVAEDecode.execute(audio_latent, audio_vae)
            audio = audio_decode_node[0]

        video_node = CreateVideo.execute(image, frame_rate, audio)
        video = video_node[0]

        return io.NodeOutput(image, audio, video,last_image)




class AD_LTX_latent_scale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_LTX_latent_scale",
            category="Apt_Preset/AD/LTX_video",

            inputs=[
                io.Combo.Input("model_name", options=folder_paths.get_filename_list("latent_upscale_models")),
                io.Latent.Input("img_latent"),
                io.Vae.Input("img_vae"),
            ],
            outputs=[
                io.Latent.Output(display_name="img_latent"),
            ]
        )

    @classmethod
    def execute(cls, model_name, img_latent, img_vae) -> io.NodeOutput:
        available_models = folder_paths.get_filename_list("latent_upscale_models")
        if len(available_models) == 0:
            raise FileNotFoundError("No latent upscale model found in models/latent_upscale_models.")
        if model_name not in available_models:
            model_name = available_models[0]
        upscale_model_node = LatentUpscaleModelLoader.execute(model_name)
        upscale_model = upscale_model_node[0]

        upsampler_node = LTXVLatentUpsampler().upsample_latent(img_latent, upscale_model, img_vae)
        latent = upsampler_node[0]

        return io.NodeOutput(latent)




class AD_LTX_sampler(io.ComfyNode):
    _SIGMAS_CACHE: dict[int, torch.Tensor] = {}

    @classmethod
    def define_schema(cls):
        return io.Schema (
            node_id="AD_LTX_sampler",
            category="Apt_Preset/AD/LTX_video",

            inputs=[

                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("img_latent"),           
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01),
                io.Int.Input("frame_rate", default=24, min=4, max=120),
                io.Vae.Input("img_vae" ),
                io.Latent.Input("audio_latent",),
                io.Vae.Input("audio_vae"),
                io.Sampler.Input("sampler"),
                io.Sigmas.Input("sigmas"),

            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="crop_positive"),
                io.Conditioning.Output(display_name="crop_negative"),
                io.Latent.Output(display_name="crop_img_latent"),
                io.Vae.Output(display_name="img_vae"),
                io.Latent.Output(display_name="audio_latent"),
                io.Vae.Output(display_name="audio_VAE"),
                io.Int.Output(display_name="frame_rate"),
            ]
        )

    @classmethod
    def _get_sigmas(cls, steps: int) -> torch.Tensor:
        steps = int(max(4, min(400, steps)))
        cached = cls._SIGMAS_CACHE.get(steps)
        if cached is None:
            cached = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
            cls._SIGMAS_CACHE[steps] = cached
        return cached.clone()

    @classmethod
    def execute(cls, sampler, sigmas, model, positive, negative, img_latent, seed, cfg, frame_rate, img_vae, audio_latent, audio_vae) -> io.NodeOutput:
        noise_node = RandomNoise.execute(seed)
        noise = noise_node[0]


        LTXVConditioning_node= LTXVConditioning.execute(positive, negative, frame_rate)
        positive = LTXVConditioning_node[0]
        negative =  LTXVConditioning_node[1]


        guider_node = CFGGuider.execute(model, positive, negative, cfg)
        guider = guider_node[0]


        if audio_latent is not None:
            av_latent_node = LTXVConcatAVLatent.execute(img_latent, audio_latent)
            av_latent = av_latent_node[0]
        else:
            av_latent = img_latent


        result = SamplerCustomAdvanced.execute(noise, guider, sampler, sigmas, av_latent)
        denoised_output = result[1]

        if audio_latent is not None:
            separate_node = LTXVSeparateAVLatent.execute(denoised_output)
            separated_video_latent = separate_node[0]
            separated_audio_latent = separate_node[1]
        else:
            separated_video_latent = denoised_output
            separated_audio_latent = None

        crop_node = LTXVCropGuides.execute(positive, negative, separated_video_latent)
        crop_positive = crop_node[0]
        crop_negative = crop_node[1]
        crop_latent = crop_node[2]

        return io.NodeOutput(model, crop_positive, crop_negative, crop_latent, img_vae, separated_audio_latent, audio_vae, frame_rate)





class AD_LTX_IC_lora(io.ComfyNode):
    _LORA_LOADER = LoraLoaderModelOnly()

    @classmethod
    def _encode_guide(cls, img_vae, latent_width, latent_height, image, scale_factors, tile_size, tile_overlap):
        time_scale_factor, width_scale_factor, height_scale_factor = scale_factors
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        image = image[:num_frames_to_keep]
        with torch.inference_mode():
            pixels = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                latent_width * width_scale_factor,
                latent_height * height_scale_factor,
                "bilinear",
                crop="disabled",
            ).movedim(1, -1)
            encode_pixels = pixels[:, :, :, :3]
            guide_latent = img_vae.encode_tiled(
                encode_pixels,
                tile_x=tile_size,
                tile_y=tile_size,
                overlap=tile_overlap,
            )
        return guide_latent

    @classmethod
    def _build_segment_plan(
        cls,
        guide_preset,
        latent_length,
        base_frame_idx,
        guide_len,
        base_strength,
        max_segments,
    ):
        latent_length = max(1, int(latent_length))
        guide_len = max(1, int(guide_len))
        start_idx = max(0, min(latent_length - 1, int(base_frame_idx)))
        max_segments = max(1, int(max_segments))
        base_strength = float(base_strength)

        # 在 [start_idx, end] 范围内按比例落点
        tail_span = max(0, (latent_length - 1) - start_idx)
        def at_ratio(r):
            return max(0, min(latent_length - 1, int(round(start_idx + tail_span * float(r)))))

        if guide_preset == "single_custom":
            return [(start_idx, max(0.0, min(1.0, base_strength)))]

        if guide_preset == "auto_balanced":
            if latent_length < 96:
                ratios = [0.0, 0.55]
                weights = [1.0, 0.78]
            elif latent_length < 192:
                ratios = [0.0, 0.35, 0.68]
                weights = [1.0, 0.82, 0.68]
            else:
                ratios = [0.0, 0.28, 0.56, 0.82]
                weights = [1.0, 0.84, 0.72, 0.62]
        elif guide_preset == "auto_stable_subject":
            ratios = [0.0, 0.22, 0.50, 0.78]
            weights = [1.0, 0.90, 0.78, 0.66]
        elif guide_preset == "auto_full_timeline":
            seg_n = min(max_segments, 12)
            seg_n = max(2, seg_n)
            ratios = [i / float(seg_n - 1) for i in range(seg_n)]
            weights = [max(0.45, 1.0 - i * 0.08) for i in range(seg_n)]
        elif guide_preset == "auto_low_vram_long":
            # 长视频低显存：段数适中、强度更温和，避免过约束
            if latent_length < 160:
                seg_n = min(max_segments, 3)
            elif latent_length < 320:
                seg_n = min(max_segments, 4)
            else:
                seg_n = min(max_segments, 5)
            seg_n = max(2, seg_n)
            ratios = [i / float(seg_n - 1) for i in range(seg_n)]
            weights = [max(0.40, 0.92 - i * 0.10) for i in range(seg_n)]
        else:
            ratios = [0.0]
            weights = [1.0]

        plan = []
        seen = set()
        for r, w in zip(ratios, weights):
            idx = at_ratio(r)
            if idx in seen:
                continue
            seen.add(idx)
            s = max(0.0, min(1.0, base_strength * float(w)))
            plan.append((idx, s))
        if not plan:
            plan = [(start_idx, max(0.0, min(1.0, base_strength)))]
        return plan

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_LTX_IC_lora",
            category="Apt_Preset/AD/LTX_video",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("img_latent"),
                io.Vae.Input("img_vae"),
                io.Image.Input("image"),
                io.Combo.Input("lora_name", options=["None"] + folder_paths.get_filename_list("loras")),
                io.Float.Input("strength_model", default=1.0, min=-100.0, max=100.0, step=0.01),
                io.Int.Input("frame_idx", default=0, min=0, max=9999, step=1),
                io.Float.Input("guide_strength", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Boolean.Input("lowGpu_mode", default=False,),
                io.Combo.Input(
                    "guide_preset",
                    options=["single_custom", "auto_balanced", "auto_stable_subject", "auto_full_timeline", "auto_low_vram_long"],
                    default="auto_balanced",
                ),
                io.Int.Input("max_segments", default=4, min=1, max=12, step=1),
                io.Int.Input("max_guide_frames", default=17, min=1, max=257, step=1),
            ],
            outputs=[
                io.Model.Output("model"),
                io.Conditioning.Output("positive"),
                io.Conditioning.Output("negative"),
                io.Latent.Output("img_latent"),
                io.Vae.Output("img_vae"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        img_latent,
        img_vae,
        image,
        lora_name,
        strength_model,
        frame_idx,
        guide_strength,
        lowGpu_mode,
        guide_preset,
        max_segments,
        max_guide_frames,
    ) -> io.NodeOutput:
        if lora_name != "None" and float(strength_model) != 0.0:
            model = cls._LORA_LOADER.load_lora_model_only(model, lora_name, strength_model)[0]

        scale_factors = img_vae.downscale_index_formula
        latent_image = img_latent["samples"]
        noise_mask = get_noise_mask(img_latent)
        _, _, latent_length, latent_height, latent_width = latent_image.shape
        time_scale_factor = scale_factors[0]
        num_frames_to_keep = ((image.shape[0] - 1) // time_scale_factor) * time_scale_factor + 1
        if max_guide_frames is not None:
            aligned_max_frames = ((int(max_guide_frames) - 1) // time_scale_factor) * time_scale_factor + 1
            num_frames_to_keep = min(num_frames_to_keep, aligned_max_frames)
        image = image[:num_frames_to_keep]
        causal_fix = num_frames_to_keep == 1

        temporal_compression = 0
        if hasattr(img_vae, "temporal_compression_decode"):
            read_tc = img_vae.temporal_compression_decode()
            if read_tc is not None:
                temporal_compression = int(read_tc)
        rec_tile_size, rec_overlap, _, _, _ = AD_AutoTileVAEDecode().recommend(
            int(image.shape[2]),
            int(image.shape[1]),
            int(image.shape[0]),
            lowGpu_mode,
            int(temporal_compression),
            img_vae,
        )
        if lowGpu_mode:
            tile_size = max(64, min(128, int(rec_tile_size)))
            tile_overlap = max(16, min(48, int(rec_overlap)))
        else:
            tile_size = max(96, min(256, int(rec_tile_size)))
            tile_overlap = max(24, min(96, int(rec_overlap)))

        if not causal_fix:
            image = torch.cat([image[:1], image], dim=0)

        guide_latent = cls._encode_guide(
            img_vae, latent_width, latent_height, image, scale_factors, tile_size, tile_overlap
        )

        if not causal_fix:
            guide_latent = guide_latent[:, :, 1:, :, :]

        guide_orig_shape = list(guide_latent.shape[2:])
        pre_filter_count = guide_latent.shape[2] * guide_latent.shape[3] * guide_latent.shape[4]
        segment_plan = cls._build_segment_plan(
            guide_preset=guide_preset,
            latent_length=latent_length,
            base_frame_idx=frame_idx,
            guide_len=int(guide_latent.shape[2]),
            base_strength=float(guide_strength),
            max_segments=max_segments,
        )
        appended_count = 0
        for seg_frame_idx, seg_strength in segment_plan:
            resolved_frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_length, guide_latent.shape[2], int(seg_frame_idx), scale_factors
            )
            if latent_idx + guide_latent.shape[2] > latent_length:
                continue
            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative, resolved_frame_idx,
                latent_image, noise_mask, guide_latent,
                float(seg_strength), scale_factors,
                guide_mask=None,
                latent_downscale_factor=1.0,
                causal_fix=causal_fix,
            )
            positive, negative = _append_guide_attention_entry(
                positive, negative, pre_filter_count, guide_orig_shape, strength=float(seg_strength)
            )
            appended_count += 1
        if appended_count == 0:
            raise ValueError("No valid guide segment could be appended (segment out of range).")

        return io.NodeOutput(
            model, positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, img_vae
        )





from ..main_unit import *



class xxxAD_LTX_load_model:
    CATEGORY = "Apt_Preset/AD/LTX_video"
    name = "AD_LTX_load_model(test)"
    _MODEL_CACHE = {}
    _LORA_MODEL_CACHE = {}
    _LORA_LOADER = LoraLoaderModelOnly()
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"),),
                "text_encoder": (["None"] + folder_paths.get_filename_list("text_encoders") ,),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120}),
                "frames_number": ("INT", {"default": 97, "min": 1, "max": 2000, "step": 8}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "LTX_CONFIG", )
    RETURN_NAMES = ("context", "audio_config", )
    FUNCTION = "run"

    @classmethod
    def _compute_model_cache_key(cls, ckpt_name, text_encoder):
        return hashlib.md5(f"{ckpt_name}|{text_encoder}".encode()).hexdigest()

    @classmethod
    def _compute_lora_cache_key(cls, ckpt_name, text_encoder, lora_name, strength_model):
        return hashlib.md5(f"{ckpt_name}|{text_encoder}|{lora_name}|{round(float(strength_model), 6)}".encode()).hexdigest()

    def run(self, ckpt_name, text_encoder, lora_name, strength_model, width, height, frames_number, frame_rate):
        cached_model = None
        cached_vae = None
        cached_audio_clip = None
        cached_audio_vae = None
        if ckpt_name != "None":
            cache_key = type(self)._compute_model_cache_key(ckpt_name, text_encoder)
            cached = type(self)._MODEL_CACHE.get(cache_key)
            if cached is None:
                base_model, base_clip, base_vae = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
                if text_encoder != "None":
                    audio_clip_output = LTXAVTextEncoderLoader.execute(text_encoder, ckpt_name, "default")
                    base_audio_clip = audio_clip_output[0] if audio_clip_output is not None else base_clip
                else:
                    base_audio_clip = base_clip
                base_audio_vae = LTXVAudioVAELoader.execute(ckpt_name)[0]
                cached = (base_model, base_audio_clip, base_vae, base_audio_vae)
                type(self)._MODEL_CACHE[cache_key] = cached
            cached_model, cached_audio_clip, cached_vae, cached_audio_vae = cached

        model = cached_model
        vae = cached_vae
        audio_clip = cached_audio_clip
        audio_vae = cached_audio_vae

        if model is None or vae is None:
            raise ValueError("AD_LTX_load_model: model/vae 为空，请提供有效 ckpt_name。")
        if audio_vae is None:
            raise ValueError("AD_LTX_load_model: audio_vae 为空，请提供有效 ckpt_name。")

        use_lora_cache = (ckpt_name != "None")
        if lora_name != "None" and float(strength_model) != 0.0:
            if use_lora_cache:
                lora_cache_key = type(self)._compute_lora_cache_key(ckpt_name, text_encoder, lora_name, strength_model)
                lora_cached_model = type(self)._LORA_MODEL_CACHE.get(lora_cache_key)
                if lora_cached_model is None:
                    lora_cached_model = type(self)._LORA_LOADER.load_lora_model_only(model, lora_name, strength_model)[0]
                    type(self)._LORA_MODEL_CACHE[lora_cache_key] = lora_cached_model
                model = lora_cached_model
            else:
                model = type(self)._LORA_LOADER.load_lora_model_only(model, lora_name, strength_model)[0]

        video_latent = EmptyLTXVLatentVideo.execute(width, height, frames_number, 1)[0]
        audio_latent = LTXVEmptyLatentAudio.execute(frames_number, frame_rate, 1, audio_vae)[0]

        audio_config = {
            "audio_latent": audio_latent,
            "audio_vae": audio_vae,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "frames_number": frames_number,
        }

        context = {
            "model": model,
            "latent": video_latent,
            "vae": vae,
            "clip": audio_clip,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "frames_number": frames_number,
            "batch": 1,
        }
        return ( context, audio_config, )



class xxxAD_LTX_config(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_LTX_config",
            category="Apt_Preset/AD/LTX_video",
            inputs=[
                LTX_CONFIG.Input("audio_config"),
            ],
            outputs=[
                LTX_CONFIG.Output("audio_config"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Latent.Output(display_name="audio_latent"),
                io.Vae.Output(display_name="audio_vae"),
                io.Int.Output(display_name="frame_rate"),
                io.Int.Output(display_name="frames_number"),
            ],
        )

    @classmethod
    def execute(cls, audio_config) -> io.NodeOutput:
        if not isinstance(audio_config, dict):
            raise ValueError("audio_config must be dict.")

        merged_config = dict(audio_config)
        if "audio_latent" not in merged_config:
            raise ValueError("audio_config 缺少 audio_latent。")
        if "audio_vae" not in merged_config:
            raise ValueError("audio_config 缺少 audio_vae。")

        width = int(merged_config.get("width", 0))
        height = int(merged_config.get("height", 0))
        frame_rate = int(merged_config.get("frame_rate", 24))
        frames_number = int(merged_config.get("frames_number", 1))
        return io.NodeOutput(
            merged_config,
            width,
            height,
            merged_config["audio_latent"],
            merged_config["audio_vae"],
            frame_rate,
            frames_number,
        )






class AD_LTX_load_model:
    CATEGORY = "Apt_Preset/AD/LTX_video"
    name = "AD_LTX_load_model(test)"
    _MODEL_CACHE = {}
    _LORA_MODEL_CACHE = {}
    _LORA_LOADER = LoraLoaderModelOnly()
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"),),
                "text_encoder": (["None"] + folder_paths.get_filename_list("text_encoders") ,),
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "width": ("INT", {"default": 768, "min": 64, "max": 4096, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120}),
                "frames_number": ("INT", {"default": 97, "min": 1, "max": 2000, "step": 8}),
            },
            "optional": {
                "over_model": ("MODEL",),
                "over_vae": ("VAE",),
                "over_clip": ("CLIP",),
                "over_audio_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LTX_CONFIG", "MODEL", "CLIP", "LATENT", "VAE", "LATENT", "VAE")
    RETURN_NAMES = ("config", "model", "clip", "img_latent", "img_vae", "audio_latent", "audio_vae")
    FUNCTION = "run"

    @classmethod
    def _compute_model_cache_key(cls, ckpt_name, text_encoder):
        return hashlib.md5(f"{ckpt_name}|{text_encoder}".encode()).hexdigest()

    @classmethod
    def _compute_lora_cache_key(cls, ckpt_name, text_encoder, lora_name, strength_model):
        return hashlib.md5(f"{ckpt_name}|{text_encoder}|{lora_name}|{round(float(strength_model), 6)}".encode()).hexdigest()

    def run(self, ckpt_name, text_encoder, lora_name, strength_model, width, height, frames_number, frame_rate, over_model=None, over_vae=None, over_clip=None, over_audio_vae=None):
        cached_model = None
        cached_vae = None
        cached_audio_clip = None
        cached_audio_vae = None
        if ckpt_name != "None":
            cache_key = type(self)._compute_model_cache_key(ckpt_name, text_encoder)
            cached = type(self)._MODEL_CACHE.get(cache_key)
            if cached is None:
                base_model, base_clip, base_vae = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
                if text_encoder != "None":
                    audio_clip_output = LTXAVTextEncoderLoader.execute(text_encoder, ckpt_name, "default")
                    base_audio_clip = audio_clip_output[0] if audio_clip_output is not None else base_clip
                else:
                    base_audio_clip = base_clip
                base_audio_vae = LTXVAudioVAELoader.execute(ckpt_name)[0]
                cached = (base_model, base_audio_clip, base_vae, base_audio_vae)
                type(self)._MODEL_CACHE[cache_key] = cached
            cached_model, cached_audio_clip, cached_vae, cached_audio_vae = cached

        model = over_model if over_model is not None else cached_model
        vae = over_vae if over_vae is not None else cached_vae
        audio_clip = over_clip if over_clip is not None else cached_audio_clip
        audio_vae = over_audio_vae if over_audio_vae is not None else cached_audio_vae

        if model is None or vae is None or audio_clip is None or audio_vae is None:
            raise ValueError("AD_LTX_load_model: override inputs not complete and ckpt/text_encoder fallback unavailable.")

        use_lora_cache = (over_model is None and ckpt_name != "None")
        if lora_name != "None" and float(strength_model) != 0.0:
            if use_lora_cache:
                lora_cache_key = type(self)._compute_lora_cache_key(ckpt_name, text_encoder, lora_name, strength_model)
                lora_cached_model = type(self)._LORA_MODEL_CACHE.get(lora_cache_key)
                if lora_cached_model is None:
                    lora_cached_model = type(self)._LORA_LOADER.load_lora_model_only(model, lora_name, strength_model)[0]
                    type(self)._LORA_MODEL_CACHE[lora_cache_key] = lora_cached_model
                model = lora_cached_model
            else:
                model = type(self)._LORA_LOADER.load_lora_model_only(model, lora_name, strength_model)[0]

        video_latent = EmptyLTXVLatentVideo.execute(width, height, frames_number, 1)[0]
        audio_latent = LTXVEmptyLatentAudio.execute(frames_number, frame_rate, 1, audio_vae)[0]

        config = {
            "model": model,
            "clip": audio_clip,
            "img_latent": video_latent,
            "img_vae": vae,
            "audio_latent": audio_latent,
            "audio_vae": audio_vae,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "frames_number": frames_number,
        }
        return (config, model, audio_clip, video_latent, vae, audio_latent, audio_vae)


class AD_LTX_config(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AD_LTX_config",
            category="Apt_Preset/AD/LTX_video",
            inputs=[
                LTX_CONFIG.Input("config"),
            ],
            outputs=[
                LTX_CONFIG.Output("config"),
                io.Model.Output(display_name="model"),
                io.Clip.Output(display_name="clip"),
                io.Latent.Output(display_name="img_latent"),
                io.Vae.Output(display_name="img_vae"),
                io.Latent.Output(display_name="audio_latent"),
                io.Vae.Output(display_name="audio_vae"),
                io.Int.Output(display_name="width"),
                io.Int.Output(display_name="height"),
                io.Int.Output(display_name="frame_rate"),
                io.Int.Output(display_name="frames_number"),
            ],
        )

    @classmethod
    def execute(cls, config) -> io.NodeOutput:
        if not isinstance(config, dict):
            raise ValueError("config must be dict.")
        width = config.get("width", None)
        height = config.get("height", None)
        if width is None or height is None:
            img_latent = config["img_latent"]["samples"]
            scale_factors = config["img_vae"].downscale_index_formula
            if width is None:
                width = int(img_latent.shape[-1] * scale_factors[1])
            if height is None:
                height = int(img_latent.shape[-2] * scale_factors[2])
        return io.NodeOutput(
            config,
            config["model"],
            config["clip"],
            config["img_latent"],
            config["img_vae"],
            config["audio_latent"],
            config["audio_vae"],
            int(width),
            int(height),
            int(config["frame_rate"]),
            int(config["frames_number"]),
        )









#endregion----------LTX---------------













