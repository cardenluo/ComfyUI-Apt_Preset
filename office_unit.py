
import numpy as np
import torch

#region------------------------------------------------

class DifferentialDiffusion():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
            },
            "optional": {
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "_for_testing"
    INIT = False

    def apply(self, model, strength=1.0):
        model = model.clone()
        model.set_model_denoise_mask_function(lambda *args, **kwargs: self.forward(*args, **kwargs, strength=strength))
        return (model, )

    def forward(self, sigma: torch.Tensor, denoise_mask: torch.Tensor, extra_options: dict, strength: float):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)

        # Generate the binary mask based on the threshold
        binary_mask = (denoise_mask >= threshold).to(denoise_mask.dtype)

        # Blend binary mask with the original denoise_mask using strength
        if strength and strength < 1:
            blended_mask = strength * binary_mask + (1 - strength) * denoise_mask
            return blended_mask
        else:
            return binary_mask



def loglinear_interp(t_steps, num_steps):
    """
    Performs log-linear interpolation of a given array of decreasing numbers.
    """
    xs = np.linspace(0, 1, len(t_steps))
    ys = np.log(t_steps[::-1])

    new_xs = np.linspace(0, 1, num_steps)
    new_ys = np.interp(new_xs, xs, ys)

    interped_ys = np.exp(new_ys)[::-1].copy()
    return interped_ys


NOISE_LEVELS = {"FLUX": [0.9968, 0.9886, 0.9819, 0.975, 0.966, 0.9471, 0.9158, 0.8287, 0.5512, 0.2808, 0.001],
"Wan":[1.0, 0.997, 0.995, 0.993, 0.991, 0.989, 0.987, 0.985, 0.98, 0.975, 0.973, 0.968, 0.96, 0.946, 0.927, 0.902, 0.864, 0.776, 0.539, 0.208, 0.001],
"Chroma": [0.992, 0.99, 0.988, 0.985, 0.982, 0.978, 0.973, 0.968, 0.961, 0.953, 0.943, 0.931, 0.917, 0.9, 0.881, 0.858, 0.832, 0.802, 0.769, 0.731, 0.69, 0.646, 0.599, 0.55, 0.501, 0.451, 0.402, 0.355, 0.311, 0.27, 0.232, 0.199, 0.169, 0.143, 0.12, 0.101, 0.084, 0.07, 0.058, 0.048, 0.001],
}

class OptimalStepsScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model_type": (["FLUX", "Wan", "Chroma"], ),
                     "steps": ("INT", {"default": 20, "min": 3, "max": 1000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model_type, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = round(steps * denoise)

        sigmas = NOISE_LEVELS[model_type][:]
        if (steps + 1) != len(sigmas):
            sigmas = loglinear_interp(sigmas, steps + 1)

        sigmas = sigmas[-(total_steps + 1):]
        sigmas[-1] = 0
        return (torch.FloatTensor(sigmas), )


#------------------------------------------------------------------------------------------------

import logging
from spandrel import ModelLoader, ImageModelDescriptor
from comfy import model_management
import torch
import comfy.utils
import folder_paths

try:
    from spandrel_extra_arches import EXTRA_REGISTRY
    from spandrel import MAIN_REGISTRY
    MAIN_REGISTRY.add(*EXTRA_REGISTRY)
    logging.info("Successfully imported spandrel_extra_arches: support for non commercial upscale models.")
except:
    pass



class UpscaleModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"), ),
                             }}
    RETURN_TYPES = ("UPSCALE_MODEL",)
    FUNCTION = "load_model"

    CATEGORY = "loaders"

    def load_model(self, model_name):
        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        out = ModelLoader().load_from_state_dict(sd).eval()

        if not isinstance(out, ImageModelDescriptor):
            raise Exception("Upscale model must be a single-image model.")

        return (out, )


class ImageUpscaleWithModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "upscale_model": ("UPSCALE_MODEL",),
                              "image": ("IMAGE",),
                              }}
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"

    CATEGORY = "image/upscaling"

    def upscale(self, upscale_model, image):
        device = model_management.get_torch_device()

        memory_required = model_management.module_size(upscale_model.model)
        memory_required += (512 * 512 * 3) * image.element_size() * max(upscale_model.scale, 1.0) * 384.0 #The 384.0 is an estimate of how much some of these models take, TODO: make it more accurate
        memory_required += image.nelement() * image.element_size()
        model_management.free_memory(memory_required, device)

        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        upscale_model.to("cpu")
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)
        return (s,)


#------------------------------------------------------------------------------------------------

import node_helpers
import comfy.utils

class CLIPTextEncodeFlux:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "clip_l": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "t5xxl": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning/flux"

    def encode(self, clip, clip_l, t5xxl, guidance):
        tokens = clip.tokenize(clip_l)
        tokens["t5xxl"] = clip.tokenize(t5xxl)["t5xxl"]

        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}), )

class FluxGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"

    def append(self, conditioning, guidance):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )


class FluxDisableGuidance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"

    CATEGORY = "advanced/conditioning/flux"
    DESCRIPTION = "This node completely disables the guidance embed on Flux and Flux like models"

    def append(self, conditioning):
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": None})
        return (c, )


PREFERED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


class FluxKontextImageScale:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"image": ("IMAGE", ),
                            },
               }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "scale"

    CATEGORY = "advanced/conditioning/flux"
    DESCRIPTION = "This node resizes the image to one that is more optimal for flux kontext."

    def scale(self, image):
        width = image.shape[2]
        height = image.shape[1]
        aspect_ratio = width / height
        _, width, height = min((abs(aspect_ratio - w / h), w, h) for w, h in PREFERED_KONTEXT_RESOLUTIONS)
        image = comfy.utils.common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)
        return (image, )


class FluxKontextMultiReferenceLatentMethod:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING", ),
            "reference_latents_method": (("offset", "index", "uxo/uno"), ),
            }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "append"
    EXPERIMENTAL = True

    CATEGORY = "advanced/conditioning/flux"

    def append(self, conditioning, reference_latents_method):
        if "uxo" in reference_latents_method or "uso" in reference_latents_method:
            reference_latents_method = "uxo"
        c = node_helpers.conditioning_set_values(conditioning, {"reference_latents_method": reference_latents_method})
        return (c, )


#------------------------------------------------------------------------------------------------
import folder_paths
import comfy.sd
import torch


class TripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "clip_name1": (folder_paths.get_filename_list("text_encoders"), ), "clip_name2": (folder_paths.get_filename_list("text_encoders"), ), "clip_name3": (folder_paths.get_filename_list("text_encoders"), )
                             }}
    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"

    CATEGORY = "advanced/loaders"

    DESCRIPTION = "[Recipes]\n\nsd3: clip-l, clip-g, t5"

    def load_clip(self, clip_name1, clip_name2, clip_name3):
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
        clip_path3 = folder_paths.get_full_path_or_raise("text_encoders", clip_name3)
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2, clip_path3], embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return (clip,)



#------------------------------------------------------------------------------------------------




class InpaintModelConditioning:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "vae": ("VAE", ),
                             "pixels": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "noise_mask": ("BOOLEAN", {"default": True, "tooltip": "Add a noise mask to the latent so sampling will only happen within the mask. Might improve results or completely break things depending on the model."}),
                             }}

    RETURN_TYPES = ("CONDITIONING","CONDITIONING","LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "encode"

    CATEGORY = "conditioning/inpaint"

    def encode(self, positive, negative, pixels, vae, mask, noise_mask=True):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")

        orig_pixels = pixels
        pixels = orig_pixels.clone()
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
            mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

        m = (1.0 - mask.round()).squeeze(1)
        for i in range(3):
            pixels[:,:,:,i] -= 0.5
            pixels[:,:,:,i] *= m
            pixels[:,:,:,i] += 0.5
        concat_latent = vae.encode(pixels)
        orig_latent = vae.encode(orig_pixels)

        out_latent = {}

        out_latent["samples"] = orig_latent
        if noise_mask:
            out_latent["noise_mask"] = mask

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent,
                                                                    "concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)






from comfy.cldm.control_types import UNION_CONTROLNET_TYPES
import nodes
import comfy.utils

class SetUnionControlNetType:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"control_net": ("CONTROL_NET", ),
                             "type": (["auto"] + list(UNION_CONTROLNET_TYPES.keys()),)
                             }}

    CATEGORY = "conditioning/controlnet"
    RETURN_TYPES = ("CONTROL_NET",)

    FUNCTION = "set_controlnet_type"

    def set_controlnet_type(self, control_net, type):
        control_net = control_net.copy()
        type_number = UNION_CONTROLNET_TYPES.get(type, -1)
        if type_number >= 0:
            control_net.set_extra_arg("control_type", [type_number])
        else:
            control_net.set_extra_arg("control_type", [])

        return (control_net,)

class ControlNetInpaintingAliMamaApply(nodes.ControlNetApplyAdvanced):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"positive": ("CONDITIONING", ),
                             "negative": ("CONDITIONING", ),
                             "control_net": ("CONTROL_NET", ),
                             "vae": ("VAE", ),
                             "image": ("IMAGE", ),
                             "mask": ("MASK", ),
                             "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                             "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                             "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001})
                             }}

    FUNCTION = "apply_inpaint_controlnet"

    CATEGORY = "conditioning/controlnet"

    def apply_inpaint_controlnet(self, positive, negative, control_net, vae, image, mask, strength, start_percent, end_percent):
        extra_concat = []
        if control_net.concat_mask:
            mask = 1.0 - mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
            mask_apply = comfy.utils.common_upscale(mask, image.shape[2], image.shape[1], "bilinear", "center").round()
            image = image * mask_apply.movedim(1, -1).repeat(1, 1, 1, image.shape[3])
            extra_concat = [mask]

        return self.apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent, vae=vae, extra_concat=extra_concat)











import math
import comfy.samplers
import comfy.sample
from comfy.k_diffusion import sampling as k_diffusion_sampling
from comfy.k_diffusion import sa_solver
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import latent_preview
import torch
import comfy.utils
import node_helpers


class BasicScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "scheduler": (comfy.samplers.SCHEDULER_NAMES, ),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, scheduler, steps, denoise):
        total_steps = steps
        if denoise < 1.0:
            if denoise <= 0.0:
                return (torch.FloatTensor([]),)
            total_steps = int(steps/denoise)

        sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1):]
        return (sigmas, )


class KarrasScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return (sigmas, )

class ExponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min):
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)
        return (sigmas, )

class PolyexponentialScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "rho": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, rho):
        sigmas = k_diffusion_sampling.get_sigmas_polyexponential(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        return (sigmas, )

class LaplaceScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "sigma_max": ("FLOAT", {"default": 14.614642, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "sigma_min": ("FLOAT", {"default": 0.0291675, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "mu": ("FLOAT", {"default": 0.0, "min": -10.0, "max": 10.0, "step":0.1, "round": False}),
                     "beta": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step":0.1, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, mu, beta):
        sigmas = k_diffusion_sampling.get_sigmas_laplace(n=steps, sigma_min=sigma_min, sigma_max=sigma_max, mu=mu, beta=beta)
        return (sigmas, )


class SDTurboScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 1, "min": 1, "max": 10}),
                     "denoise": ("FLOAT", {"default": 1.0, "min": 0, "max": 1.0, "step": 0.01}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, denoise):
        start_step = 10 - int(10 * denoise)
        timesteps = torch.flip(torch.arange(1, 11) * 100 - 1, (0,))[start_step:start_step + steps]
        sigmas = model.get_model_object("model_sampling").sigma(timesteps)
        sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
        return (sigmas, )

class BetaSamplingScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "alpha": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 50.0, "step":0.01, "round": False}),
                     "beta": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 50.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, model, steps, alpha, beta):
        sigmas = comfy.samplers.beta_scheduler(model.get_model_object("model_sampling"), steps, alpha=alpha, beta=beta)
        return (sigmas, )

class VPScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "beta_d": ("FLOAT", {"default": 19.9, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}), #TODO: fix default values
                     "beta_min": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 5000.0, "step":0.01, "round": False}),
                     "eps_s": ("FLOAT", {"default": 0.001, "min": 0.0, "max": 1.0, "step":0.0001, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, beta_d, beta_min, eps_s):
        sigmas = k_diffusion_sampling.get_sigmas_vp(n=steps, beta_d=beta_d, beta_min=beta_min, eps_s=eps_s)
        return (sigmas, )

class SplitSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                    "step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                     }
                }
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, step):
        sigmas1 = sigmas[:step + 1]
        sigmas2 = sigmas[step:]
        return (sigmas1, sigmas2)

class SplitSigmasDenoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                     }
                }
    RETURN_TYPES = ("SIGMAS","SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, denoise):
        steps = max(sigmas.shape[-1] - 1, 0)
        total_steps = round(steps * denoise)
        sigmas1 = sigmas[:-(total_steps)]
        sigmas2 = sigmas[-(total_steps + 1):]
        return (sigmas1, sigmas2)

class FlipSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas):
        if len(sigmas) == 0:
            return (sigmas,)

        sigmas = sigmas.flip(0)
        if sigmas[0] == 0:
            sigmas[0] = 0.0001
        return (sigmas,)

class SetFirstSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     "sigma": ("FLOAT", {"default": 136.0, "min": 0.0, "max": 20000.0, "step": 0.001, "round": False}),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "set_first_sigma"

    def set_first_sigma(self, sigmas, sigma):
        sigmas = sigmas.clone()
        sigmas[0] = sigma
        return (sigmas, )

class ExtendIntermediateSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     "steps": ("INT", {"default": 2, "min": 1, "max": 100}),
                     "start_at_sigma": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 20000.0, "step": 0.01, "round": False}),
                     "end_at_sigma": ("FLOAT", {"default": 12.0, "min":  0.0, "max": 20000.0, "step": 0.01, "round": False}),
                     "spacing": (['linear', 'cosine', 'sine'],),
                    }
               }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "extend"

    def extend(self, sigmas: torch.Tensor, steps: int, start_at_sigma: float, end_at_sigma: float, spacing: str):
        if start_at_sigma < 0:
            start_at_sigma = float("inf")

        interpolator = {
            'linear': lambda x: x,
            'cosine': lambda x: torch.sin(x*math.pi/2),
            'sine':   lambda x: 1 - torch.cos(x*math.pi/2)
        }[spacing]

        # linear space for our interpolation function
        x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
        computed_spacing = interpolator(x)

        extended_sigmas = []
        for i in range(len(sigmas) - 1):
            sigma_current = sigmas[i]
            sigma_next = sigmas[i+1]

            extended_sigmas.append(sigma_current)

            if end_at_sigma <= sigma_current <= start_at_sigma:
                interpolated_steps = computed_spacing * (sigma_next - sigma_current) + sigma_current
                extended_sigmas.extend(interpolated_steps.tolist())

        # Add the last sigma value
        if len(sigmas) > 0:
            extended_sigmas.append(sigmas[-1])

        extended_sigmas = torch.FloatTensor(extended_sigmas)

        return (extended_sigmas,)


class SamplingPercentToSigma:
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "sampling_percent": (IO.FLOAT, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.0001}),
                "return_actual_sigma": (IO.BOOLEAN, {"default": False, "tooltip": "Return the actual sigma value instead of the value used for interval checks.\nThis only affects results at 0.0 and 1.0."}),
            }
        }

    RETURN_TYPES = (IO.FLOAT,)
    RETURN_NAMES = ("sigma_value",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigma"

    def get_sigma(self, model, sampling_percent, return_actual_sigma):
        model_sampling = model.get_model_object("model_sampling")
        sigma_val = model_sampling.percent_to_sigma(sampling_percent)
        if return_actual_sigma:
            if sampling_percent == 0.0:
                sigma_val = model_sampling.sigma_max.item()
            elif sampling_percent == 1.0:
                sigma_val = model_sampling.sigma_min.item()
        return (sigma_val,)


class KSamplerSelect:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sampler_name": (comfy.samplers.SAMPLER_NAMES, ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, sampler_name):
        sampler = comfy.samplers.sampler_object(sampler_name)
        return (sampler, )

class SamplerDPMPP_3M_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, noise_device):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_3m_sde"
        else:
            sampler_name = "dpmpp_3m_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerDPMPP_2M_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"solver_type": (['midpoint', 'heun'], ),
                     "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, solver_type, eta, s_noise, noise_device):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_2m_sde"
        else:
            sampler_name = "dpmpp_2m_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "solver_type": solver_type})
        return (sampler, )


class SamplerDPMPP_SDE:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "r": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "noise_device": (['gpu', 'cpu'], ),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise, r, noise_device):
        if noise_device == 'cpu':
            sampler_name = "dpmpp_sde"
        else:
            sampler_name = "dpmpp_sde_gpu"
        sampler = comfy.samplers.ksampler(sampler_name, {"eta": eta, "s_noise": s_noise, "r": r})
        return (sampler, )

class SamplerDPMPP_2S_Ancestral:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler("dpmpp_2s_ancestral", {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerEulerAncestral:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler("euler_ancestral", {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerEulerAncestralCFGPP:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "eta": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step":0.01, "round": False}),
                "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step":0.01, "round": False}),
            }}
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, eta, s_noise):
        sampler = comfy.samplers.ksampler(
            "euler_ancestral_cfg_pp",
            {"eta": eta, "s_noise": s_noise})
        return (sampler, )

class SamplerLMS:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"order": ("INT", {"default": 4, "min": 1, "max": 100}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, order):
        sampler = comfy.samplers.ksampler("lms", {"order": order})
        return (sampler, )

class SamplerDPMAdaptative:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"order": ("INT", {"default": 3, "min": 2, "max": 3}),
                     "rtol": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "atol": ("FLOAT", {"default": 0.0078, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "h_init": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "pcoeff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "icoeff": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "dcoeff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "accept_safety": ("FLOAT", {"default": 0.81, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "eta": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                     "s_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.01, "round": False}),
                      }
               }
    RETURN_TYPES = ("SAMPLER",)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, order, rtol, atol, h_init, pcoeff, icoeff, dcoeff, accept_safety, eta, s_noise):
        sampler = comfy.samplers.ksampler("dpm_adaptive", {"order": order, "rtol": rtol, "atol": atol, "h_init": h_init, "pcoeff": pcoeff,
                                                              "icoeff": icoeff, "dcoeff": dcoeff, "accept_safety": accept_safety, "eta": eta,
                                                              "s_noise":s_noise })
        return (sampler, )


class SamplerER_SDE(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "solver_type": (IO.COMBO, {"options": ["ER-SDE", "Reverse-time SDE", "ODE"]}),
                "max_stage": (IO.INT, {"default": 3, "min": 1, "max": 3}),
                "eta": (
                    IO.FLOAT,
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False, "tooltip": "Stochastic strength of reverse-time SDE.\nWhen eta=0, it reduces to deterministic ODE. This setting doesn't apply to ER-SDE solver type."},
                ),
                "s_noise": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, solver_type, max_stage, eta, s_noise):
        if solver_type == "ODE" or (solver_type == "Reverse-time SDE" and eta == 0):
            eta = 0
            s_noise = 0

        def reverse_time_sde_noise_scaler(x):
            return x ** (eta + 1)

        if solver_type == "ER-SDE":
            # Use the default one in sample_er_sde()
            noise_scaler = None
        else:
            noise_scaler = reverse_time_sde_noise_scaler

        sampler_name = "er_sde"
        sampler = comfy.samplers.ksampler(sampler_name, {"s_noise": s_noise, "noise_scaler": noise_scaler, "max_stage": max_stage})
        return (sampler,)


class SamplerSASolver(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "eta": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "round": False},),
                "sde_start_percent": (IO.FLOAT, {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.001},),
                "sde_end_percent": (IO.FLOAT, {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.001},),
                "s_noise": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01, "round": False},),
                "predictor_order": (IO.INT, {"default": 3, "min": 1, "max": 6}),
                "corrector_order": (IO.INT, {"default": 4, "min": 0, "max": 6}),
                "use_pece": (IO.BOOLEAN, {}),
                "simple_order_2": (IO.BOOLEAN, {}),
            }
        }

    RETURN_TYPES = (IO.SAMPLER,)
    CATEGORY = "sampling/custom_sampling/samplers"

    FUNCTION = "get_sampler"

    def get_sampler(self, model, eta, sde_start_percent, sde_end_percent, s_noise, predictor_order, corrector_order, use_pece, simple_order_2):
        model_sampling = model.get_model_object("model_sampling")
        start_sigma = model_sampling.percent_to_sigma(sde_start_percent)
        end_sigma = model_sampling.percent_to_sigma(sde_end_percent)
        tau_func = sa_solver.get_tau_interval_func(start_sigma, end_sigma, eta=eta)

        sampler_name = "sa_solver"
        sampler = comfy.samplers.ksampler(
            sampler_name,
            {
                "tau_func": tau_func,
                "s_noise": s_noise,
                "predictor_order": predictor_order,
                "corrector_order": corrector_order,
                "use_pece": use_pece,
                "simple_order_2": simple_order_2,
            },
        )
        return (sampler,)


class Noise_EmptyNoise:
    def __init__(self):
        self.seed = 0

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        return torch.zeros(latent_image.shape, dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")


class Noise_RandomNoise:
    def __init__(self, seed):
        self.seed = seed

    def generate_noise(self, input_latent):
        latent_image = input_latent["samples"]
        batch_inds = input_latent["batch_index"] if "batch_index" in input_latent else None
        return comfy.sample.prepare_noise(latent_image, self.seed, batch_inds)

class SamplerCustom:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "add_noise": ("BOOLEAN", {"default": True}),
                    "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, add_noise, noise_seed, cfg, positive, negative, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)
        latent["samples"] = latent_image

        if not add_noise:
            noise = Noise_EmptyNoise().generate_noise(latent)
        else:
            noise = Noise_RandomNoise(noise_seed).generate_noise(latent)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise_seed)

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class Guider_Basic(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

class BasicGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "conditioning": ("CONDITIONING", ),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, conditioning):
        guider = Guider_Basic(model)
        guider.set_conds(conditioning)
        return (guider,)

class CFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, positive, negative, cfg):
        guider = comfy.samplers.CFGGuider(model)
        guider.set_conds(positive, negative)
        guider.set_cfg(cfg)
        return (guider,)

class Guider_DualCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, cfg1, cfg2, nested=False):
        self.cfg1 = cfg1
        self.cfg2 = cfg2
        self.nested = nested

    def set_conds(self, positive, middle, negative):
        middle = node_helpers.conditioning_set_values(middle, {"prompt_type": "negative"})
        self.inner_set_conds({"positive": positive, "middle": middle, "negative": negative})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        negative_cond = self.conds.get("negative", None)
        middle_cond = self.conds.get("middle", None)
        positive_cond = self.conds.get("positive", None)

        if self.nested:
            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            pred_text = comfy.samplers.cfg_function(self.inner_model, out[2], out[1], self.cfg1, x, timestep, model_options=model_options, cond=positive_cond, uncond=middle_cond)
            return out[0] + self.cfg2 * (pred_text - out[0])
        else:
            if model_options.get("disable_cfg1_optimization", False) == False:
                if math.isclose(self.cfg2, 1.0):
                    negative_cond = None
                    if math.isclose(self.cfg1, 1.0):
                        middle_cond = None

            out = comfy.samplers.calc_cond_batch(self.inner_model, [negative_cond, middle_cond, positive_cond], x, timestep, model_options)
            return comfy.samplers.cfg_function(self.inner_model, out[1], out[0], self.cfg2, x, timestep, model_options=model_options, cond=middle_cond, uncond=negative_cond) + (out[2] - out[1]) * self.cfg1

class DualCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                    "cond1": ("CONDITIONING", ),
                    "cond2": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "cfg_conds": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "cfg_cond2_negative": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                    "style": (["regular", "nested"],),
                     }
                }

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(self, model, cond1, cond2, negative, cfg_conds, cfg_cond2_negative, style):
        guider = Guider_DualCFG(model)
        guider.set_conds(cond1, cond2, negative)
        guider.set_cfg(cfg_conds, cfg_cond2_negative, nested=(style == "nested"))
        return (guider,)

class DisableNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":{
                     }
                }

    RETURN_TYPES = ("NOISE",)
    FUNCTION = "get_noise"
    CATEGORY = "sampling/custom_sampling/noise"

    def get_noise(self):
        return (Noise_EmptyNoise(),)


class RandomNoise(DisableNoise):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "noise_seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
            }
        }

    def get_noise(self, noise_seed):
        return (Noise_RandomNoise(noise_seed),)


class SamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"noise": ("NOISE", ),
                    "guider": ("GUIDER", ),
                    "sampler": ("SAMPLER", ),
                    "sigmas": ("SIGMAS", ),
                    "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT","LATENT")
    RETURN_NAMES = ("output", "denoised_output")

    FUNCTION = "sample"

    CATEGORY = "sampling/custom_sampling"

    def sample(self, noise, guider, sampler, sigmas, latent_image):
        latent = latent_image
        latent_image = latent["samples"]
        latent = latent.copy()
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent_image)
        latent["samples"] = latent_image

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        x0_output = {}
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0_output)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        samples = guider.sample(noise.generate_noise(latent), latent_image, sampler, sigmas, denoise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=noise.seed)
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = guider.model_patcher.model.process_latent_out(x0_output["x0"].cpu())
        else:
            out_denoised = out
        return (out, out_denoised)

class AddNoise:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"model": ("MODEL",),
                     "noise": ("NOISE", ),
                     "sigmas": ("SIGMAS", ),
                     "latent_image": ("LATENT", ),
                     }
                }

    RETURN_TYPES = ("LATENT",)

    FUNCTION = "add_noise"

    CATEGORY = "_for_testing/custom_sampling/noise"

    def add_noise(self, model, noise, sigmas, latent_image):
        if len(sigmas) == 0:
            return latent_image

        latent = latent_image
        latent_image = latent["samples"]

        noisy = noise.generate_noise(latent)

        model_sampling = model.get_model_object("model_sampling")
        process_latent_out = model.get_model_object("process_latent_out")
        process_latent_in = model.get_model_object("process_latent_in")

        if len(sigmas) > 1:
            scale = torch.abs(sigmas[0] - sigmas[-1])
        else:
            scale = sigmas[0]

        if torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = process_latent_in(latent_image)
        noisy = model_sampling.noise_scaling(scale, noisy, latent_image)
        noisy = process_latent_out(noisy)
        noisy = torch.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)

        out = latent.copy()
        out["samples"] = noisy
        return (out,)




#endregion------------------------------------------------




def VAEDecodeTiled(vae, samples, tile_size, overlap=64, temporal_size=64, temporal_overlap=8):
    if tile_size < overlap * 4:
        overlap = tile_size // 4
    if temporal_size < temporal_overlap * 2:
        temporal_overlap = temporal_overlap // 2
    temporal_compression = vae.temporal_compression_decode()
    if temporal_compression is not None:
        temporal_size = max(2, temporal_size // temporal_compression)
        temporal_overlap = max(1, min(temporal_size // 2, temporal_overlap // temporal_compression))
    else:
        temporal_size = None
        temporal_overlap = None

    compression = vae.spacial_compression_decode()
    images = vae.decode_tiled(samples["samples"], tile_x=tile_size // compression, tile_y=tile_size // compression, overlap=overlap // compression, tile_t=temporal_size, overlap_t=temporal_overlap)
    if len(images.shape) == 5: #Combine batches
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return (images, )

