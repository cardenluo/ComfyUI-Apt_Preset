
import matplotlib
import torch

import node_helpers
from io import BytesIO
from math import cos, pi, sin, sqrt

import comfy.samplers
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import norm


matplotlib.use("Agg")




try:
    from asteval import Interpreter
    REMOVER_AVAILABLE = True  
except ImportError:
    Interpreter = None
    REMOVER_AVAILABLE = False 


try:
    import enum
    REMOVER_AVAILABLE = True  
except ImportError:
    enum = None
    REMOVER_AVAILABLE = False 





def loglinear_interp(t_steps, num_steps):
    if len(t_steps) == 0 or num_steps <= 0:
        return np.array([])
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])
    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)
    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys

class GraphScale(enum.StrEnum):
    linear = "linear"
    log = "log"

def tensor_to_graph_image(tensor, color="blue", scale: GraphScale = GraphScale.linear):
    plt.figure(figsize=(10, 6), dpi=100)
    plt.plot(tensor.numpy(), marker="o", linestyle="-", color=color)
    plt.title("Graph from Tensor")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.yscale(str(scale))
    plt.tight_layout()
    with BytesIO() as buf:
        plt.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        image = Image.open(buf).copy()
    plt.close()
    return image

def fibonacci_normalized_descending(n):
    if n <= 0:
        return []
    if n == 1:
        fib_sequence = [1.0]
    else:
        fib_sequence = [1.0, 1.0]
        for _ in range(2, n):
            next_val = fib_sequence[-1] + fib_sequence[-2]
            fib_sequence.append(next_val)
    max_value = fib_sequence[-1] if fib_sequence else 1.0
    if max_value == 0:
        max_value = 1.0
    normalized_sequence = [x / max_value for x in fib_sequence]
    descending_sequence = normalized_sequence[::-1]
    return descending_sequence



class scheduler_sigmas2Graph:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        col = [
            "black", "red", "green", "blue", "cyan", "magenta", "yellow",
            "purple", "lime", "navy", "teal", "orange", "maroon", "lavender", "olive",
        ]
        scale_options = [option.value for option in GraphScale]
        return {
            "required": {
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "color": (col, {"default": "blue"}),
                "print_as_list": ("BOOLEAN", {"default": False}),
                "scale": (scale_options, {"default": GraphScale.linear.value}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("IMAGE",)
    CATEGORY = "Apt_Preset/chx_ksample/Scheduler"

    def simple_output(self, sigmas, color, print_as_list, scale):
        try:
            scale_enum = GraphScale(scale)
        except ValueError:
            scale_enum = GraphScale.linear
        if print_as_list and sigmas is not None and len(sigmas) > 0:
            print("Raw sigmas:", sigmas.tolist())
            if sigmas.max() != sigmas.min():
                sigmas_percentages = (sigmas - sigmas.min()) / (sigmas.max() - sigmas.min())
            else:
                sigmas_percentages = np.zeros_like(sigmas)
            sigmas_percentages_w_steps = [(i, round(float(s), 4)) for i, s in enumerate(sigmas_percentages)]
            print("Sigma percentages (index, value):", sigmas_percentages_w_steps)
        if sigmas is None or len(sigmas) == 0:
            img = Image.new('RGB', (500, 300), color='white')
        else:
            img = tensor_to_graph_image(sigmas.cpu(), color=color, scale=scale_enum)
        numpy_image = np.array(img).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(numpy_image).unsqueeze(0)
        if tensor_image.ndim == 4 and tensor_image.shape[-1] == 3:
            tensor_image = tensor_image.permute(0, 3, 1, 2)
        return (tensor_image,)



class scheduler_ModelAligned:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "steps": ("INT", {"default": 10, "min": 1, "max": 10000, "step": 1}),
                "model_type": (["SD1", "SDXL", "FLUX", "Qwen", "wan2.1", "wan2.2"],),
                "force_sigma_min": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "Apt_Preset/chx_ksample/Scheduler"

    def simple_output(self, model, steps, model_type, force_sigma_min):
        timestep_indices = {
            "SD1": [999, 850, 736, 645, 545, 455, 343, 233, 124, 24, 0],
            "SDXL": [999, 845, 730, 587, 443, 310, 193, 116, 53, 13, 0],
            "FLUX": [999, 890, 780, 670, 560, 450, 340, 230, 120, 30, 0],
            "Qwen": [999, 850, 740, 620, 490, 370, 250, 140, 60, 15, 0],
            "wan2.1": [999, 860, 750, 635, 520, 405, 290, 185, 80, 20, 0],
            "wan2.2": [999, 855, 745, 625, 505, 385, 270, 165, 70, 15, 0]
        }
        indices = timestep_indices.get(model_type, timestep_indices["SD1"])
        indices = [999 - i for i in indices]
        try:
            sampling = model.get_model_object("model_sampling")
            base_sigmas = comfy.samplers.calculate_sigmas(sampling, "simple", 1000)
            max_idx = len(base_sigmas) - 1
            safe_indices = [i if i <= max_idx else max_idx for i in indices]
            sigmas = base_sigmas[safe_indices]
        except Exception as e:
            print(f"[aligned_scheduler] Error calculating base sigmas: {e}")
            sigmas = np.linspace(10.0, 0.0, 11)
        target_steps = steps + 1 if not force_sigma_min else steps
        sigmas = loglinear_interp(sigmas.tolist(), target_steps)
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(model, "get_dtype", lambda: torch.float32)()
        sigmas_tensor = torch.FloatTensor(sigmas).to(device=device, dtype=dtype)
        if not force_sigma_min:
            sigmas_tensor = torch.cat([sigmas_tensor[:-1], torch.tensor([0.0], device=device, dtype=dtype)])
        else:
            sigmas_tensor = torch.cat([sigmas_tensor, torch.tensor([0.0], device=device, dtype=dtype)])
        return (sigmas_tensor.cpu(),)



class scheduler_manual_sigmas:
    def __init__(self):
        self.asteval = Interpreter()
        self.asteval.symtable.update({
            "pi": pi, "cos": cos, "sin": sin, "sqrt": sqrt,
            "norm": norm, "phi": (1 + sqrt(5)) / 2
        })

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "custom_sigmas": (
                    "STRING",
                    {
                        "default": "((1 - cos(2 * pi * (1-y**0.5) * 0.5)) / 2)*sigmax+((1 - cos(2 * pi * y**0.5 * 0.5)) / 2)*sigmin"
                    },
                ),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100000, "step": 1}),
                "sgm": ("BOOLEAN", {"default": False}),
            }
        }

    FUNCTION = "simple_output"
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "Apt_Preset/chx_ksample/Scheduler"

    def simple_output(self, model, custom_sigmas, steps, sgm):
        total_steps = steps + 1 if sgm else steps
        try:
            sampling = model.get_model_object("model_sampling")
            sigmin = float(sampling.sigma(sampling.timestep(sampling.sigma_min)))
            sigmax = float(sampling.sigma(sampling.timestep(sampling.sigma_max)))
        except Exception as e:
            print(f"[manual_scheduler] Error getting sigma bounds: {e}")
            sigmin = 0.0291
            sigmax = 14.6146
        sigmas = []
        fibo = fibonacci_normalized_descending(total_steps)
        if len(fibo) < total_steps:
            fibo += [fibo[-1]] * (total_steps - len(fibo))
        self.asteval.symtable.update({
            "sigmin": sigmin,
            "sigmax": sigmax,
            "s": total_steps,
            "phi": (1 + sqrt(5)) / 2,
        })
        error_occurred = False
        for j in range(total_steps):
            if total_steps == 1:
                y = 0.0
                x = 1.0
            else:
                y = j / (total_steps - 1)
                x = 1 - y
            self.asteval.symtable.update({
                "y": y,
                "x": x,
                "f": fibo[j] if j < len(fibo) else 0.0,
                "j": j,
            })
            try:
                val = self.asteval(custom_sigmas)
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    val = sigmin + (sigmax - sigmin) * (1 - y)
                sigmas.append(val)
            except Exception as e:
                print(f"[manual_scheduler] Error at step {j}: {e}")
                error_occurred = True
                sigmas.append(sigmin)
        if sgm and len(sigmas) > 1:
            sigmas = sigmas[:-1]
        sigmas.append(0.0)
        device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        dtype = getattr(model, "get_dtype", lambda: torch.float32)()
        sigmas_tensor = torch.tensor(sigmas, device=device, dtype=dtype)
        return (sigmas_tensor,)





#region-------------------------------Scheduler_MixScheduler-----------------------------------------


from comfy.samplers import KSampler
import matplotlib.pyplot as plt
from comfy.model_management import get_torch_device


plt.switch_backend('Agg')


class Scheduler_MixScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "total_steps": ("INT", {"default": 20, "min": 2, "max": 10000}),
                "split_step": ("INT", {"default": 10, "min": 1, "max": 9999}),
                "scheduler_1": (list(KSampler.SCHEDULERS),),
                "scheduler_2": (list(KSampler.SCHEDULERS),),
                "smooth_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }
    
    RETURN_TYPES = ("SIGMAS",)
    RETURN_NAMES = ("Sigma",)
    FUNCTION = "compute_segmented_scheduler"
    CATEGORY = "Apt_Preset/chx_ksample/Scheduler"

    def get_sigmas_from_ksampler(self, model, steps, scheduler):
        device = get_torch_device()
        temp_sampler = KSampler(
            model=model,
            steps=steps,
            device=device,
            sampler=KSampler.SAMPLERS[0],
            scheduler=scheduler
        )
        sigmas = temp_sampler.sigmas.to(device) if hasattr(temp_sampler, 'sigmas') else None
        if sigmas is None:
            sigmas = torch.linspace(14.6, 0.03, steps + 1, dtype=torch.float32, device=device)
        return sigmas

    def compute_segmented_scheduler(self, model, total_steps, split_step, scheduler_1, scheduler_2, smooth_strength=0.5):
        split_step = min(max(split_step, 1), total_steps - 1)
        device = get_torch_device()

        sigmas_full_1 = self.get_sigmas_from_ksampler(model, total_steps, scheduler_1)
        sigmas_full_2 = self.get_sigmas_from_ksampler(model, total_steps, scheduler_2)

        sigmas_1_segment = sigmas_full_1[:split_step + 1]
        sigmas_2_segment = sigmas_full_2[split_step:]
        sigmas_combined = torch.cat([sigmas_1_segment[:-1], sigmas_2_segment])

        if split_step > 0 and split_step < total_steps:
            prev_sigma = sigmas_combined[split_step - 1]
            curr_sigma = sigmas_combined[split_step]
            sigmas_combined[split_step] = (1 - smooth_strength) * prev_sigma + smooth_strength * curr_sigma
            if split_step > 1:
                sigmas_combined[split_step - 1] = (sigmas_combined[split_step - 2] + sigmas_combined[split_step]) / 2

        return (sigmas_combined, )


#endregion-------------------------------Scheduler_MixScheduler-----------------------------------------





class Scheduler_CondNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "strength": ("FLOAT", {"default": 20, "min": 0.0, "max": 50, "step": 0.1}),
                "start_step_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1}),
                "end_step_percent": ("FLOAT", {"default": 100.0, "min": 0.0, "max": 100.0, "step": 1}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "randomize_conditioning"
    CATEGORY = "Apt_Preset/chx_ksample/Scheduler"
    DESCRIPTION = """
    strength强度:
    轻微风格 / 细节扰动  0.1 ~ 5.0
    中度风格 / 构图调整  5.0 ~ 20.0
    重度风格 / 抽象化改造 20.0 ~ 50.0
    start_step_percent ~ end_step_percent:噪声应用范围
    """


    def randomize_conditioning(self, conditioning, strength, start_step_percent, end_step_percent):
        if strength <= 0:
            return (conditioning,)

        start_pct = start_step_percent / 100
        end_pct = end_step_percent / 100
        
        if start_pct >= end_pct:
            return (conditioning,)

        base_cond = conditioning[0][0]
        if not isinstance(base_cond, torch.Tensor):
            return (conditioning,)
        
        noise = (torch.rand_like(base_cond) * 2 - 1) * strength
        noisy_cond = base_cond + noise
        noisy_conditioning = [[noisy_cond, conditioning[0][1]]]

        cond_pre = node_helpers.conditioning_set_values(conditioning, {"start_percent": 0.0, "end_percent": start_pct})
        cond_noise = node_helpers.conditioning_set_values(noisy_conditioning, {"start_percent": start_pct, "end_percent": end_pct})
        cond_post = node_helpers.conditioning_set_values(conditioning, {"start_percent": end_pct, "end_percent": 1.0})
        final_cond = cond_pre + cond_noise + cond_post

        return (final_cond,)









