import comfy
from ..main_unit import *










class basicIn_input:
    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    RETURN_TYPES = ("INT", "FLOAT", "STRING")
    RETURN_NAMES = ("int", "float", "string")
    FUNCTION = "convert_number_types"
    CATEGORY = "Apt_Preset/View_IO/InputData"
    
    def convert_number_types(self, input):
        try:
            float_num = float(input)
            int_num = int(float_num)
            str_num = input
        except ValueError:
            return (None, None, input)
        return (int_num, float_num, str_num)









class basicIn_Seed:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "pass_seed"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def pass_seed(self, seed):
        return (seed,)


class basicIn_float:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("STRING", {"default": "", "multiline": False})
            }
        }
    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "convert_to_float"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def convert_to_float(self, input):
        try:
            return (float(input),)
        except (ValueError, TypeError):
            raise ValueError("请输入有效的数字")


class basicIn_Sampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sampler": ( comfy.samplers.KSampler.SAMPLERS, ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS,)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "pass_sampler"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def pass_sampler(self, sampler):
        return (sampler,)


class basicIn_Scheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            }
        }

    RETURN_TYPES = (comfy.samplers.KSampler.SCHEDULERS,)
    RETURN_NAMES = ("scheduler",)
    FUNCTION = "pass_scheduler"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def pass_scheduler(self, scheduler):
        return (scheduler,)


class basicIn_string:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"default": "", "multiline": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "pass_text"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def pass_text(self, input_text):
        return (input_text,)


class basicIn_Remap_slide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_min": ("FLOAT", {"default": 0.0, "min": -9999, "max": 9999, "step": 0.001}),
                "source_max": ("FLOAT", {"default": 1.0, "min": -9999, "max": 9999, "step": 0.001}),
                "slide": ("FLOAT", {"default": 0.0, "min": 0, "max": 1, "step": 0.001, "display": "slider"}),
                "precision": ("FLOAT", {"default": 0.001, "min": 0.001, "max": 1000, "step": 0.001}),
            },
            "optional": {
            },
        }

    FUNCTION = "set_range"
    RETURN_TYPES = ("FLOAT", "FLOAT", )
    RETURN_NAMES = ("source_value", "slide_value", )
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def set_range(self, source_min, source_max, precision, slide):

        step = max(0.0001, precision)           
        slide_rounded = round(slide / step) * step
        
        source_value = source_min + (source_max - source_min) * slide_rounded        
        slide_value = slide_rounded
        
        return (source_value, slide_value)



class basicIn_int:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": ("INT", { "min": 0, "max": 16384,  "step": 1,})
            }
        }
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("int",)
    FUNCTION = "convert_to_int"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def convert_to_int(self, input):
        try:
            return (int(input),)
        except (ValueError, TypeError):
            return (None,)





class basicIn_color:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "hex_str": ("STRING", {"multiline": False, "default": "#0E0B0B"}),
            },
            "optional": {
                "image": ("IMAGE",),
            }
        }
    NAME = "basicIn_color"
    RETURN_TYPES = ("COLOR", "STRING")  
    RETURN_NAMES = ("color", "hex_str")
    FUNCTION = "output_color"
    CATEGORY = "Apt_Preset/View_IO/InputData"

    def output_color(self, hex_str, image=None):
        hex_str = hex_str.strip().lower()
        if not hex_str.startswith("#"):
            hex_str = f"#{hex_str}"
        if len(hex_str) == 4:
            hex_str = f"#{hex_str[1]}{hex_str[1]}{hex_str[2]}{hex_str[2]}{hex_str[3]}{hex_str[3]}"
        elif len(hex_str) != 7:
            hex_str = "#ffffff"
        return (hex_str, hex_str)

































