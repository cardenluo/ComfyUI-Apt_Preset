import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as T
import math
import sys
import nodes
import comfy.samplers
import comfy.k_diffusion.sampling
import comfy.model_management
from ..main_unit import *




#region---------------------------latent def-------------------------------

def channels_layer(images, channels, function):
    if channels == "rgb":
        img = images[:, :, :, :3]
    elif channels == "rgba":
        img = images[:, :, :, :4]
    elif channels == "rg":
        img = images[:, :, :, [0, 1]]
    elif channels == "rb":
        img = images[:, :, :, [0, 2]]
    elif channels == "ra":
        img = images[:, :, :, [0, 3]]
    elif channels == "gb":
        img = images[:, :, :, [1, 2]]
    elif channels == "ga":
        img = images[:, :, :, [1, 3]]
    elif channels == "ba":
        img = images[:, :, :, [2, 3]]
    elif channels == "r":
        img = images[:, :, :, 0]
    elif channels == "g":
        img = images[:, :, :, 1]
    elif channels == "b":
        img = images[:, :, :, 2]
    elif channels == "a":
        img = images[:, :, :, 3]
    else:
        raise ValueError("Unsupported channels")

    result = torch.from_numpy(function(img.numpy()))

    if channels == "rgb":
        images[:, :, :, :3] = result
    elif channels == "rgba":
        images[:, :, :, :4] = result
    elif channels == "rg":
        images[:, :, :, [0, 1]] = result
    elif channels == "rb":
        images[:, :, :, [0, 2]] = result
    elif channels == "ra":
        images[:, :, :, [0, 3]] = result
    elif channels == "gb":
        images[:, :, :, [1, 2]] = result
    elif channels == "ga":
        images[:, :, :, [1, 3]] = result
    elif channels == "ba":
        images[:, :, :, [2, 3]] = result
    elif channels == "r":
        images[:, :, :, 0] = result
    elif channels == "g":
        images[:, :, :, 1] = result
    elif channels == "b":
        images[:, :, :, 2] = result
    elif channels == "a":
        images[:, :, :, 3] = result

    return images

def normalize(latent, target_min=None, target_max=None):
    min_val = latent.min()
    max_val = latent.max()
    
    if target_min is None:
        target_min = min_val
    if target_max is None:
        target_max = max_val
        
    normalized = (latent - min_val) / (max_val - min_val)
    scaled = normalized * (target_max - target_min) + target_min
    return scaled

def latent_to_image(latents, l2rgb=False):
    if l2rgb:
        # 默认使用 FLUX 风格的 5 通道 weight
        l2rgb_weight = torch.tensor([
            [0.298, 0.207, 0.208],   # L1
            [0.187, 0.286, 0.173],   # L2
            [-0.158, 0.189, 0.264],  # L3
            [-0.184, -0.271, -0.473],# L4
            [0.0, 0.0, 0.0],         # 第5个通道，默认为 0
        ], device=latents.device)

        # 如果输入是 4 通道，则补一个零通道
        if latents.shape[1] == 4:
            pad = torch.zeros_like(latents[:, :1])  # (B, 1, H, W)
            latents = torch.cat([latents, pad], dim=1)

        tensors = torch.einsum('...lhw,lr->...rhw', latents.float(), l2rgb_weight)
        tensors = ((tensors + 1) / 2).clamp(0, 1)
        tensors = tensors.movedim(1, -1)
    else:
        tensors = latents.permute(0, 2, 3, 1)
    return tensors

def sharpen_latents(latent, alpha=1.5):
    device = latent.device
    sharpen_kernel = torch.tensor([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=torch.float32, device=device).view(1,1,3,3)
    sharpen_kernel /= sharpen_kernel.sum()
    sharpened_tensors = []
    for channel in range(latent.size(1)):
        channel_tensor = latent[:, channel, :, :].unsqueeze(1)
        sharpened_channel = F.conv2d(channel_tensor, sharpen_kernel, padding=1)
        sharpened_tensors.append(sharpened_channel)
    sharpened_tensor = torch.cat(sharpened_tensors, dim=1)
    sharpened_tensor = latent + alpha * sharpened_tensor
    padding_size = (sharpen_kernel.shape[-1] - 1) // 2
    sharpened_tensor = sharpened_tensor[:, :, padding_size:-padding_size, padding_size:-padding_size]
    sharpened_tensor = torch.clamp(sharpened_tensor, 0, 1)
    sharpened_tensor = F.interpolate(sharpened_tensor, size=(latent.size(2), latent.size(3)), mode='nearest')
    return sharpened_tensor

def high_pass_latents(latent, radius=3, strength=1.0):
    sigma = radius / 3.0
    kernel_size = radius * 2 + 1
    x = torch.arange(-radius, radius+1).float().to(latent.device)
    gaussian_kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    high_pass_overlays = []
    for channel in range(latent.size(1)):
        channel_tensor = latent[:, channel, :, :].unsqueeze(1)
        weight_h = gaussian_kernel.view(1, 1, 1, -1)
        weight_v = gaussian_kernel.view(1, 1, -1, 1)
        input_blur_h = F.conv2d(channel_tensor, weight_h, padding=0)
        input_blur_v = F.conv2d(input_blur_h, weight_v, padding=0)
        input_blur_h = F.interpolate(input_blur_h, size=channel_tensor.shape[-2:], mode='bilinear', align_corners=False)
        input_blur_v = F.interpolate(input_blur_v, size=channel_tensor.shape[-2:], mode='bilinear', align_corners=False)
        high_pass_component = channel_tensor - input_blur_v
        high_pass_channel = channel_tensor + strength * high_pass_component
        high_pass_channel = torch.clamp(high_pass_channel, 0, 1)
        high_pass_overlays.append(high_pass_channel)
    high_pass_overlay = torch.cat(high_pass_overlays, dim=1)
    return high_pass_overlay

def hslerp(a, b, t):
    if a.shape != b.shape:
        raise ValueError("Input tensors a and b must have the same shape.")
    num_channels = a.size(1)
    interpolation_tensor = torch.zeros(1, num_channels, 1, 1, device=a.device, dtype=a.dtype)
    interpolation_tensor[0, 0, 0, 0] = 1.0
    result = (1 - t) * a + t * b
    if t < 0.5:
        result += (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    else:
        result -= (torch.norm(b - a, dim=1, keepdim=True) / 6) * interpolation_tensor
    return result


blending_modes = {
    'add': lambda a, b, t: (a * t + b * t),
    'bislerp': lambda a, b, t: normalize((1 - t) * a + t * b),
    'color dodge': lambda a, b, t: a / (1 - b + 1e-6),
    'colorize': lambda a, b, t: a + (b - a) * t,
    'cosine interp': lambda a, b, t: (a + b - (a - b) * torch.cos(t * torch.tensor(math.pi))) / 2,
    'cuberp': lambda a, b, t: a + (b - a) * (3 * t ** 2 - 2 * t ** 3),
    'difference': lambda a, b, t: normalize(abs(a - b) * t),
    'exclusion': lambda a, b, t: normalize((a + b - 2 * a * b) * t),
    'hslerp': hslerp,
    'glow': lambda a, b, t: torch.where(a <= 1, a ** 2 / (1 - b + 1e-6), b * (a - 1) / (a + 1e-6)),
    'hard light': lambda a, b, t: (2 * a * b * (a < 0.5).float() + (1 - 2 * (1 - a) * (1 - b)) * (a >= 0.5).float()) * t,
    'inject': lambda a, b, t: a + b * t,
    'lerp': lambda a, b, t: (1 - t) * a + t * b,
    'linear dodge': lambda a, b, t: normalize(a + b * t),
    'linear light': lambda a, b, t: torch.where(b <= 0.5, a + 2 * b - 1, a + 2 * (b - 0.5)),
    'multiply': lambda a, b, t: normalize(a * t * b * t),
    'overlay': lambda a, b, t: torch.where(b < 0.5, (2 * a * b + a**2 - 2 * a**2 * b) * t, (1 - 2 * (1 - a) * (1 - b)) * t),
    'pin light': lambda a, b, t: torch.where(b <= 0.5, torch.min(a, 2 * b), torch.max(a, 2 * b - 1)),
    'random': lambda a, b, t: normalize(torch.rand_like(a) * a * t + torch.rand_like(b) * b * t),
    'reflect': lambda a, b, t: torch.where(b <= 1, b ** 2 / (1 - a + 1e-6), a * (b - 1) / (b + 1e-6)),
    'screen': lambda a, b, t: normalize(1 - (1 - a) * (1 - b) * (1 - t)),
    'slerp': lambda a, b, t: normalize((a * torch.sin((1 - t) * torch.acos(torch.clamp(torch.sum(a * b, dim=1, keepdim=True), -1.0, 1.0))) + 
                                         b * torch.sin(t * torch.acos(torch.clamp(torch.sum(a * b, dim=1, keepdim=True), -1.0, 1.0)))) / 
                                        torch.sin(torch.acos(torch.clamp(torch.sum(a * b, dim=1, keepdim=True), -1.0, 1.0)))),
    'subtract': lambda a, b, t: (a * t - b * t),
    'vivid light': lambda a, b, t: torch.where(b <= 0.5, a / (1 - 2 * b + 1e-6), (a + 2 * b - 1) / (2 * (1 - b) + 1e-6)),
}


#endregion-----------------------latent def-------------------------------



class latent_chx_noise:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {
                    "default": 0.5,
                    "step": 0.01
                }),
                "monochromatic": (["false", "true"],),
                "invert": (["false", "true"],),
                "channels": (["rgb", "rgba", "rg", "rb", "ra", "gb", "ga", "ba", "r", "g", "b", "a"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "node"
    CATEGORY = "Apt_Preset/chx_tool/latent"

    def noise(self, images, strength, monochromatic, invert):
        if monochromatic and images.shape[3] > 1:
            noise = np.random.normal(0, 1, images.shape[:3])
        else:
            noise = np.random.normal(0, 1, images.shape)

        noise = np.abs(noise)
        noise /= noise.max()

        if monochromatic and images.shape[3] > 1:
            noise = noise[..., np.newaxis].repeat(images.shape[3], -1)

        if invert:
            noise = images - noise * strength
        else:
            noise = images + noise * strength

        noise = np.clip(noise, 0.0, 1.0)
        noise = noise.astype(images.dtype)

        return noise

    def node(self, images, strength, monochromatic, invert, channels):
        tensor = images.clone().detach()

        monochromatic = True if monochromatic == "true" else False
        invert = True if invert == "true" else False

        return (channels_layer(tensor, channels, lambda x: self.noise(
            x, strength, monochromatic, invert
        )),)


class latent_Image2Noise:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strenght": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "noise_size": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "color_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "saturation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "contrast": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.1 }),
                "mask_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_scale_diff": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01 }),
                "mask_contrast": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1 }),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "Apt_Preset/chx_tool/latent"

    def execute(self, image, noise_size, color_noise, mask_strength, mask_scale_diff, mask_contrast, noise_strenght, saturation, contrast, blur, mask=None):
        torch.manual_seed(0)

        elastic_alpha = max(image.shape[1], image.shape[2])# * noise_size
        elastic_sigma = elastic_alpha / 400 * noise_size

        blur_size = int(6 * blur+1)
        if blur_size % 2 == 0:
            blur_size+= 1

        if mask is None:
            noise_mask = image
        else:
            noise_mask=mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3)

        
        # increase contrast of the mask
        if mask_contrast != 1:
            noise_mask = T.ColorJitter(contrast=(mask_contrast,mask_contrast))(noise_mask.permute([0, 3, 1, 2])).permute([0, 2, 3, 1])

        # Ensure noise mask is the same size as the image
        if noise_mask.shape[1:] != image.shape[1:]:
            noise_mask = F.interpolate(noise_mask.permute([0, 3, 1, 2]), size=(image.shape[1], image.shape[2]), mode='bicubic', align_corners=False)
            noise_mask = noise_mask.permute([0, 2, 3, 1])
        # Ensure we have the same number of masks and images
        if noise_mask.shape[0] > image.shape[0]:
            noise_mask = noise_mask[:image.shape[0]]
        else:
            noise_mask = torch.cat((noise_mask, noise_mask[-1:].repeat((image.shape[0]-noise_mask.shape[0], 1, 1, 1))), dim=0)

        # Convert mask to grayscale mask
        noise_mask = noise_mask.mean(dim=3).unsqueeze(-1)

        # add color noise
        imgs = image.clone().permute([0, 3, 1, 2])
        if color_noise > 0:
            color_noise = torch.normal(torch.zeros_like(imgs), std=color_noise)
            color_noise *= (imgs - imgs.min()) / (imgs.max() - imgs.min())

            imgs = imgs + color_noise
            imgs = imgs.clamp(0, 1)

        # create fine and coarse noise
        fine_noise = []
        for n in imgs:
            avg_color = n.mean(dim=[1,2])

            tmp_noise = T.ElasticTransform(alpha=elastic_alpha, sigma=elastic_sigma, fill=avg_color.tolist())(n)
            if blur > 0:
                tmp_noise = T.GaussianBlur(blur_size, blur)(tmp_noise)
            tmp_noise = T.ColorJitter(contrast=(contrast,contrast), saturation=(saturation,saturation))(tmp_noise)
            fine_noise.append(tmp_noise)

        imgs = None
        del imgs

        fine_noise = torch.stack(fine_noise, dim=0)
        fine_noise = fine_noise.permute([0, 2, 3, 1])
        #fine_noise = torch.stack(fine_noise, dim=0)
        #fine_noise = pb(fine_noise)
        mask_scale_diff = min(mask_scale_diff, 0.99)
        if mask_scale_diff > 0:
            coarse_noise = F.interpolate(fine_noise.permute([0, 3, 1, 2]), scale_factor=1-mask_scale_diff, mode='area')
            coarse_noise = F.interpolate(coarse_noise, size=(fine_noise.shape[1], fine_noise.shape[2]), mode='bilinear', align_corners=False)
            coarse_noise = coarse_noise.permute([0, 2, 3, 1])
        else:
            coarse_noise = fine_noise

        output = (1 - noise_mask) * coarse_noise + noise_mask * fine_noise

        if mask_strength < 1:
            noise_mask = noise_mask.pow(mask_strength)
            noise_mask = torch.nan_to_num(noise_mask).clamp(0, 1)
        output = noise_mask * output + (1 - noise_mask) * image

        # apply noise to image
        output = output * noise_strenght + image * (1 - noise_strenght)
        output = output.clamp(0, 1)

        return (output, )


class latent_ratio:
    
    @classmethod
    def INPUT_TYPES(s):
        s.ratio_sizes, s.ratio_dict = read_ratios()
        return {'required': {'ratio_selected': (s.ratio_sizes,),
                            "batch_size": ("INT", {"default": 1, "min": 1, "max": 64})}}

    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'generate'
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def generate(self, ratio_selected, batch_size=1):
        width = self.ratio_dict[ratio_selected]["width"]
        height = self.ratio_dict[ratio_selected]["height"]
        latent = torch.zeros([batch_size, 4, height // 8, width // 8])
        return ({"samples": latent}, )


class chx_latent_adjust:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {

                "brightness": ("FLOAT", {"default": 1.0, "max": 2.0, "min": -1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "max": 2.0, "min": -1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.0, "step": 0.01}),
                "exposure": ("FLOAT", {"default": 0.0, "max": 2.0, "min": -1.0, "step": 0.01}),
                "alpha_sharpen": ("FLOAT", {"default": 0.0, "max": 10.0, "min": 0.0, "step": 0.01}),
                "high_pass_radius": ("FLOAT", {"default": 0.0, "max": 1024, "min": 0.0, "step": 0.01}),
                "high_pass_strength": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.0, "step": 0.01}),
            },

            "optional": {
                "context": ("RUN_CONTEXT",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("RUN_CONTEXT","LATENT")
    RETURN_NAMES = ("context","latent")
    FUNCTION = "adjust_latent"
    CATEGORY = "Apt_Preset/chx_tool/latent"
    def adjust_latent(self, context=None, brightness=1.0, contrast=1.0, saturation=1.0, exposure=0.0,
                    alpha_sharpen=0.0, high_pass_radius=0.0, high_pass_strength=1.0,latent=None):
        
        if latent is None:
            latent = context.get("latent", None)
        original_latent = latent['samples']
        
        r, g, b, a = original_latent[:, 0:1], original_latent[:, 1:2], original_latent[:, 2:3], original_latent[:, 3:4]

        r = (r - 0.5) * contrast + 0.5 + (brightness - 1.0)
        g = (g - 0.5) * contrast + 0.5 + (brightness - 1.0)
        b = (b - 0.5) * contrast + 0.5 + (brightness - 1.0)

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        r = (r - gray) * saturation + gray
        g = (g - gray) * saturation + gray
        b = (b - gray) * saturation + gray

        r = r * (2 ** exposure)
        g = g * (2 ** exposure)
        b = b * (2 ** exposure)

        latent_tensor = torch.cat((r, g, b, a), dim=1)
        if alpha_sharpen > 0:
            latent_tensor = sharpen_latents(latent_tensor, alpha_sharpen)
        if high_pass_radius > 0:
            latent_tensor = high_pass_latents(latent_tensor, high_pass_radius, high_pass_strength)
        tensors = latent_to_image(latent_tensor, l2rgb=True)
        new_latent = {'samples': latent_tensor}
        context = new_context(context, latent=new_latent)
        return context, new_latent



class latent_blend:      #未启用
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent_a": ("LATENT",),
                "latent_b": ("LATENT",),
                "operation": (sorted(list(blending_modes.keys())),),
                "blend_ratio": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "op_strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            },
            "optional": {
                "mask": ("MASK",),
                "set_noise_mask": (["false", "true"],),
                "normalize": (["false", "true"],),
                "clamp_min": ("FLOAT", {"default": 0.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "clamp_max": ("FLOAT", {"default": 1.0, "max": 10.0, "min": -10.0, "step": 0.01}),
                "latent2rgb_preview": (["false", "true"],),
            }
        }

    RETURN_TYPES = ("LATENT","IMAGE",)
    RETURN_NAMES = ("latents", "previews")
    FUNCTION = "latent_blend"
    CATEGORY = "Apt_Preset/chx_tool/latent"

    def latent_blend(self, latent_a, latent_b, operation, blend_ratio, op_strength, mask=None, set_noise_mask=None, normalize=None, clamp_min=None, clamp_max=None, latent2rgb_preview=None):
        
        latent_a = latent_a["samples"][:, :-1]
        latent_b = latent_b["samples"][:, :-1]

        assert latent_a.shape == latent_b.shape, f"Input latents must have the same shape, but got: a {latent_a.shape}, b {latent_b.shape}"

        alpha_a = latent_a[:, -1:]
        alpha_b = latent_b[:, -1:]
        
        blended_rgb = self.blend_latents(latent_a, latent_b, operation, blend_ratio, op_strength, clamp_min, clamp_max)
        blended_alpha = torch.ones_like(blended_rgb[:, :1])
        blended_latent = torch.cat((blended_rgb, blended_alpha), dim=1)
        
        tensors = latent_to_image(blended_latent, (True if latent2rgb_preview and latent2rgb_preview == "true" else False))

        if mask is not None:
            blend_mask = self.transform_mask(mask, latent_a["samples"].shape)
            blended_latent = blend_mask * blended_latent + (1 - blend_mask) * latent_a["samples"]
            if set_noise_mask == 'true':
                return ({"samples": blended_latent, "noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))}, tensors)
            else:
                return ({"samples": blended_latent}, tensors)
        else:
            return ({"samples": blended_latent}, tensors)
            
    def blend_latents(self, latent1, latent2, mode='add', blend_percentage=0.5, op_strength=0.5, mask=None, clamp_min=0.0, clamp_max=1.0):
        blend_func = blending_modes.get(mode)
        if blend_func is None:
            raise ValueError(f"Unsupported blending mode. Please choose from the supported modes: {', '.join(list(blending_modes.keys()))}")
        
        blend_factor1 = blend_percentage
        blend_factor2 = 1 - blend_percentage
        blended_latent = blend_func(latent1, latent2, op_strength * blend_factor1)

        if normalize and normalize == "true":
            blended_latent = normalize(blended_latent, clamp_min, clamp_max)
        return blended_latent

    def transform_mask(self, mask, shape):
        mask = mask.view(-1, 1, mask.shape[-2], mask.shape[-1])
        resized_mask = torch.nn.functional.interpolate(mask, size=(shape[2], shape[3]), mode="bilinear")
        expanded_mask = resized_mask.expand(-1, shape[1], -1, -1)
        if expanded_mask.shape[0] < shape[0]:
            expanded_mask = expanded_mask.repeat((shape[0] - 1) // expanded_mask.shape[0] + 1, 1, 1, 1)[:shape[0]]
        del mask, resized_mask
        return expanded_mask





















