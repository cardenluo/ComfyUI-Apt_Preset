import torch
import comfy.utils

class Test_CN_ImgPreview:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 99999, "step": 64}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 99999, "step": 64}),
                "upscale_algorithm": (["nearest-exact", "bilinear", "bicubic", "lanczos"], {"default": "nearest-exact"}),
            },
            "optional": {
                "target_noise_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("processed_reference",)
    FUNCTION = "process_reference"
    CATEGORY = "Apt_Preset/flow"
 

    def process_reference(self, reference_image, target_width, target_height, upscale_algorithm, target_noise_image=None):
        if target_noise_image is not None and len(target_noise_image) > 0:
            tgt_h = target_noise_image.shape[1]
            tgt_w = target_noise_image.shape[2]
        else:
            tgt_w = target_width
            tgt_h = target_height

        ref_img_model = reference_image[0].permute(2, 0, 1).unsqueeze(0)
        processed_img = comfy.utils.common_upscale(
            samples=ref_img_model,
            width=tgt_w,
            height=tgt_h,
            upscale_method=upscale_algorithm,
            crop="center"
        )

        processed_img = processed_img.squeeze(0).permute(1, 2, 0).unsqueeze(0)
        processed_img = torch.clamp(processed_img, 0.0, 1.0)

        return (processed_img,)
    























































