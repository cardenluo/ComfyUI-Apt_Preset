import comfy.model_management
import comfy.samplers
import comfy.utils
import folder_paths
import node_helpers
import torch
from PIL import Image
from comfy.patcher_extension import WrappersMP
from comfy_extras.nodes_audio import vae_decode_audio
from nodes import (
    CheckpointLoaderSimple, CLIPLoader, CLIPTextEncode, DualCLIPLoader,
    UNETLoader, VAEDecode, VAEEncode, VAELoader, common_ksampler,
)

from ..main_unit import CLIP_TYPE, apply_lora_stack, load_upscale_model, new_context, pil2tensor, read_ratios, upscale_with_model
from ..NodeBasic.C_AD import _AD_H3_LATENT_TILE_PRESETS
from ..NodeBasic.minimaxH3 import _ad_h3_aligned_tile_regions, _ad_h3_core_halo_window

try:
    from .load_GGUF.nodes import CLIPLoaderGGUF2, DualCLIPLoaderGGUF2, UnetLoaderGGUF2
    GGUF_AVAILABLE = True
except (ImportError, RuntimeError):
    CLIPLoaderGGUF2 = None
    DualCLIPLoaderGGUF2 = None
    UnetLoaderGGUF2 = None
    GGUF_AVAILABLE = False


_UC_NO_LATENT_TILE = next(iter(_AD_H3_LATENT_TILE_PRESETS))
_UC_TILED_APPLY_MODEL_KEY = "apt_uc_tiled_apply_model"


def _uc_conditioning_is_empty(conditioning):
    if not conditioning:
        return True
    for item in conditioning:
        if not isinstance(item, (list, tuple)) or not item:
            return False
        value = item[0]
        if not torch.is_tensor(value) or value.numel() > 0:
            return False
    return True


def _uc_crop_spatial(value, axis, start, end):
    if not torch.is_tensor(value) or value.ndim != 4:
        return value
    if axis == "H":
        return value[:, :, start:end, :].contiguous()
    return value[:, :, :, start:end].contiguous()


def _uc_crop_scaled(value, axis, start, end, full_size):
    if not torch.is_tensor(value) or value.ndim != 4:
        return value
    cond_size = int(value.shape[-2] if axis == "H" else value.shape[-1])
    cond_start = start * cond_size // full_size
    cond_end = (end * cond_size + full_size - 1) // full_size
    return _uc_crop_spatial(value, axis, cond_start, cond_end)


class _UCTiledApplyModel:
    def __init__(self, tile_count, overlap_pixels):
        self.tile_count = int(tile_count)
        self.overlap_pixels = int(overlap_pixels)

    def apply_model_wrapper(self, executor, x, t, c_concat=None, c_crossattn=None,
                            control=None, transformer_options={}, **kwargs):
        if not torch.is_tensor(x) or x.ndim != 4:
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        height, width = int(x.shape[-2]), int(x.shape[-1])
        axis = "H" if height >= width else "W"
        full_size = height if axis == "H" else width
        halo = max(0, int(round(self.overlap_pixels / 8.0)))
        regions = _ad_h3_aligned_tile_regions(full_size, self.tile_count, halo, alignment=4)
        if len(regions) <= 1 or any(start == 0 and end == full_size for start, end, _, _ in regions):
            return executor(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

        output = torch.zeros_like(x, dtype=torch.float32)
        weight_shape = (1, 1, full_size, 1) if axis == "H" else (1, 1, 1, full_size)
        weights = torch.zeros(weight_shape, dtype=torch.float32, device=x.device)
        for start, end, core_start, core_end in regions:
            window = _ad_h3_core_halo_window(start, end, core_start, core_end, x.device)
            window = window.view(1, 1, -1, 1) if axis == "H" else window.view(1, 1, 1, -1)
            prediction = executor(
                _uc_crop_spatial(x, axis, start, end), t,
                _uc_crop_scaled(c_concat, axis, start, end, full_size),
                c_crossattn, control, transformer_options, **kwargs,
            )
            if axis == "H":
                output[:, :, start:end, :] += prediction.float() * window
                weights[:, :, start:end, :] += window
            else:
                output[:, :, :, start:end] += prediction.float() * window
                weights[:, :, :, start:end] += window
        return (output / weights.clamp_min(1e-8)).to(dtype=x.dtype)


def _uc_tiled_model(model, tile_count, overlap_pixels):
    tiled_model = model.clone()
    state = _UCTiledApplyModel(tile_count, overlap_pixels)
    tiled_model.add_wrapper_with_key(
        WrappersMP.APPLY_MODEL, _UC_TILED_APPLY_MODEL_KEY, state.apply_model_wrapper
    )
    return tiled_model


class UC_create_context:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "latent": ("LATENT",),
                "audio_vae": ("VAE",),
                "collapse_sampling_parameters": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅折叠参数，如果后端缺这些参数，则会被采用",
                }),
                "steps": ("INT", {"default": 20, "min": 0, "max": 10000, "tooltip": "0 = None"}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "tooltip": "0 = None"}),
                "sampler": (["None"] + comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (["None"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CLIP")
    RETURN_NAMES = ("context", "model", "clip")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/unit_context"

    def sample(self, model, latent=None, vae=None, clip=None, audio_vae=None, steps=20, cfg=8.0,
               sampler="euler", scheduler="normal", collapse_sampling_parameters=False):
        del collapse_sampling_parameters
        context = {
            "model": model,
            "latent": latent,
            "vae": vae,
            "clip": clip,
            "audio_vae": audio_vae,
            "steps": None if steps == 0 else int(steps),
            "cfg": None if cfg == 0.0 else float(cfg),
            "sampler": None if sampler == "None" else sampler,
            "scheduler": None if scheduler == "None" else scheduler,
        }
        return (context, model, clip)


class UC_ksampler:
    ratio_sizes, ratio_dict = read_ratios()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "denoise": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "VAE_Decode": ("BOOLEAN", {"default": True, "tooltip": "关闭后不生成图片和音频，速度提升",},),
                "sample_parameters": ("BOOLEAN", {
                    "default": False,
                    "label_on": "自定义",
                    "label_off": "继承context",
                    "tooltip": "继承context时使用上游参数,自定义时使用下方参数",
                }),
                "steps": ("INT", {
                    "default": 8,
                    "min": 0,
                    "max": 10000,
          
                }),
                "cfg": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
  
                }),
                "sampler": (["None"] + comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (["None"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),

                "latent_size": ("BOOLEAN", {
                    "default": False,
                    "label_on": "自定义",
                    "label_off": "继承context",
                    "tooltip": "继承context时优先使用上游latent尺寸",
                }),
                "ratio_selected": (["None", "customer_WxH"] + cls.ratio_sizes, {"default": "None"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 300}),
                "width": ("INT", {"default": 512, "min": 8, "max": 16384}),
                "height": ("INT", {"default": 512, "min": 8, "max": 16384}),
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent": ("LATENT",),
                "latent_image": ("IMAGE",),
                "latent_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "LATENT", "IMAGE", "AUDIO")
    RETURN_NAMES = ("context", "latent", "image", "audio")
    FUNCTION = "sample"
    CATEGORY = "Apt_Preset/unit_context"
    NAME = "UC_ksampler"

    def set_latent_mask2(self, latent, mask):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("latent 必须是包含 'samples' 键的字典")
        newlatent = {"samples": latent["samples"].clone()}
        if mask is not None:
            newlatent["noise_mask"] = mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1]))
        return newlatent

    def sample(self, context, seed, denoise, steps, cfg, sampler, scheduler,
               VAE_Decode, model=None, positive=None, negative=None, latent=None,
               latent_image=None, latent_mask=None, sample_parameters=False,
               latent_size=False, ratio_selected="None", batch_size=1,
               width=512, height=512):
        model = model if model is not None else context.get("model")
        positive = positive if positive is not None else context.get("positive")
        negative = negative if negative is not None else context.get("negative")
        vae = context.get("vae")
        audio_vae = context.get("audio_vae")
        clip = context.get("clip")

        if sample_parameters is False:
            steps = context.get("steps")
            cfg = context.get("cfg")
            sampler = context.get("sampler")
            scheduler = context.get("scheduler")

        if steps is None or steps <= 0:
            steps = 8
        if cfg is None:
            cfg = 1.0
        if sampler in (None, "None"):
            sampler = "euler"
        if scheduler in (None, "None"):
            scheduler = "simple"

        if _uc_conditioning_is_empty(positive):
            if clip is None:
                raise ValueError("Cannot encode the default positive prompt without CLIP in context")
            positive = CLIPTextEncode().encode(clip, "")[0]
        if _uc_conditioning_is_empty(negative):
            if clip is None:
                raise ValueError("Cannot encode the default negative prompt 'blur' without CLIP in context")
            negative = CLIPTextEncode().encode(clip, "blur")[0]

        if latent_image is not None:
            if vae is None:
                raise ValueError("UC_ksampler needs VAE in context to encode latent_image")
            latent = VAEEncode().encode(vae, latent_image)[0]
        elif latent is None and latent_size is False:
            latent = context.get("latent")

        if latent is None:
            if vae is None:
                raise ValueError("UC_ksampler needs VAE in context to create the fallback latent")
            if latent_size is False:
                ratio_selected = "None"
                batch_size = 1
                width = 512
                height = 512
            if ratio_selected not in ("None", "customer_WxH"):
                width = self.ratio_dict[ratio_selected]["width"]
                height = self.ratio_dict[ratio_selected]["height"]
            width = max(int(width / 8) * 8, 64)
            height = max(int(height / 8) * 8, 64)
            black_tensor = pil2tensor(Image.new("RGB", (width, height), color=(0, 0, 0)))
            latent = VAEEncode().encode(vae, black_tensor)[0]
            latent["samples"] = comfy.utils.repeat_to_batch_size(latent["samples"], batch_size)

        if latent_mask is None:
            latent_mask = context.get("mask")
        if latent is not None and latent_mask is not None:
            latent = self.set_latent_mask2(latent, latent_mask)

        latent = common_ksampler(
            model, seed, steps, cfg, sampler, scheduler,
            positive, negative, latent, denoise=denoise,
        )[0]

        output_image = None
        output_audio = None
        if VAE_Decode:
            if vae is None:
                raise ValueError("UC_ksampler needs VAE in context when VAE_Decode is enabled")
            output_image = VAEDecode().decode(vae, latent)[0]
            if audio_vae is not None:
                output_audio = vae_decode_audio(audio_vae, latent)

        output_context = new_context(
            context, model=model, positive=positive, negative=negative,
            latent=latent, mask=latent_mask, steps=steps, cfg=cfg,
            sampler=sampler, scheduler=scheduler,
        )
        output_context["images"] = output_image
        return (output_context, latent, output_image, output_audio)


class UC_Ksampler_refine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
                "model_name": (folder_paths.get_filename_list("upscale_models"), {"default": "RealESRGAN_x2.pth"}),
                "upscale_output_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 10.0,
                    "step": 0.05,
                    "tooltip": "最终放大倍数 = 模型倍数X系数",
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "denoise": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                }),
                "tile_size": (
                    list(_AD_H3_LATENT_TILE_PRESETS),
                    {"default": _UC_NO_LATENT_TILE, "tooltip": "按分块数量和重叠像素进行 latent 分块采样。"},
                ),
                "sample_parameters": ("BOOLEAN", {
                    "default": False,
                    "label_on": "自定义",
                    "label_off": "继承context",
                    "tooltip": "继承context时使用上游采样参数；自定义时使用下方四项参数",
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "steps": ("INT", {"default": 8, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "sampler": (["None"] + comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (["None"] + comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT", "IMAGE")
    RETURN_NAMES = ("context", "image")
    FUNCTION = "refine"
    CATEGORY = "Apt_Preset/unit_context"
    NAME = "UC_Ksampler_refine"

    def refine(self, context, model_name, upscale_output_scale, seed, denoise, tile_size,
               image=None, steps=8, cfg=1.0, sampler="euler", scheduler="simple",
               sample_parameters=False):
        model = context.get("model")
        vae = context.get("vae")
        clip = context.get("clip")
        positive = context.get("positive")
        negative = context.get("negative")
        latent = context.get("latent")

        missing = [name for name, value in (("model", model), ("vae", vae)) if value is None]
        if missing:
            raise ValueError(f"UC_Ksampler_refine context is missing: {', '.join(missing)}")
        if _uc_conditioning_is_empty(positive):
            if clip is None:
                raise ValueError("UC_Ksampler_refine needs CLIP to encode the default positive prompt")
            positive = CLIPTextEncode().encode(clip, "")[0]
        if _uc_conditioning_is_empty(negative):
            if clip is None:
                raise ValueError("UC_Ksampler_refine needs CLIP to encode the default negative prompt 'blur'")
            negative = CLIPTextEncode().encode(clip, "blur")[0]
        if image is None:
            image = context.get("images")
        if image is None and latent is not None:
            image = VAEDecode().decode(vae, latent)[0]
        if image is None:
            raise ValueError("UC_Ksampler_refine needs an image or latent in context")

        if sample_parameters is False:
            steps = context.get("steps")
            cfg = context.get("cfg")
            sampler = context.get("sampler")
            scheduler = context.get("scheduler")
        if steps is None or steps <= 0:
            steps = 8
        if cfg is None:
            cfg = 1.0
        if sampler in (None, "None"):
            sampler = "euler"
        if scheduler in (None, "None"):
            scheduler = "simple"
        if tile_size not in _AD_H3_LATENT_TILE_PRESETS:
            raise ValueError(f"UC_Ksampler_refine: unknown tile preset: {tile_size}")

        upscale_model = load_upscale_model(model_name)
        image = upscale_with_model(upscale_model, image)
        image_height, image_width = image.shape[1:3]
        target_width = max(8, round(image_width * float(upscale_output_scale) / 8) * 8)
        target_height = max(8, round(image_height * float(upscale_output_scale) / 8) * 8)
        if (target_height, target_width) != (image_height, image_width):
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1), target_width, target_height, "lanczos", "disabled"
            ).movedim(1, -1)
        latent = VAEEncode().encode(vae, image)[0]

        tile_count, overlap_pixels = _AD_H3_LATENT_TILE_PRESETS[tile_size]
        sample_model = _uc_tiled_model(model, tile_count, overlap_pixels) if tile_count > 1 else model
        latent = common_ksampler(
            sample_model, int(seed), int(steps), float(cfg), sampler, scheduler,
            positive, negative, latent, denoise=float(denoise),
        )[0]
        output_image = VAEDecode().decode(vae, latent)[0]
        output_context = new_context(
            context, model=model, positive=positive, negative=negative, latent=latent,
            images=output_image, steps=steps, cfg=cfg, sampler=sampler, scheduler=scheduler,
        )
        return (output_context, output_image)


class UC_load_model:
    CATEGORY = "Apt_Preset/unit_context"
    @classmethod
    def INPUT_TYPES(cls):
        available_ckpt = folder_paths.get_filename_list("checkpoints")
        available_unets = sorted(set(
            folder_paths.get_filename_list("unet")
            + folder_paths.get_filename_list("unet_gguf")
        ))
        available_clips = sorted(set(
            folder_paths.get_filename_list("text_encoders")
            + folder_paths.get_filename_list("clip_gguf")
        ))
        available_vaes = folder_paths.get_filename_list("vae")

        return {
            "optional": {
                "ckpt_name": (["None"] + available_ckpt,),
                "unet_name": (["None"] + available_unets,),
                "unet_Weight_Dtype": (["None", "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_type": (["None"] + CLIP_TYPE,),
                "clip1": (["None"] + available_clips,),
                "clip2": (["None"] + available_clips,),
                "vae": (["None"] + available_vaes,),
                "audio_vae": (["None"] + available_vaes,),
                "collapse_sampling_parameters": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "仅折叠参数，如果后端缺这些参数，则会被采用",
                }),
                "steps": ("INT", {"default": 20, "min": 1, "max": 999999}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.5, "round": 0.01}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "over_model": ("MODEL",),
                "over_clip": ("CLIP",),
                "lora_stack": ("LORASTACK",),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CLIP")
    RETURN_NAMES = ("context", "model", "clip")
    FUNCTION = "process_settings"
    NAME = "UC_load_model"

    @staticmethod
    def _load_unet(unet_name, weight_dtype):
        if unet_name == "None":
            return None
        if unet_name.endswith(".gguf"):
            if not GGUF_AVAILABLE or UnetLoaderGGUF2 is None:
                raise RuntimeError("GGUF not installed. Please install gguf and protobuf")
            return UnetLoaderGGUF2().load_unet(
                unet_name, dequant_dtype=None, patch_dtype=None,
                patch_on_device=None,
            )[0]
        return UNETLoader().load_unet(unet_name, weight_dtype)[0]

    @staticmethod
    def _load_clip(clip1, clip2, clip_type):
        if clip1 == "None" and clip2 == "None":
            return None
        if clip1 == "None":
            raise ValueError("clip2 requires clip1")
        if clip2 == "None":
            if clip1.endswith(".gguf"):
                if not GGUF_AVAILABLE or CLIPLoaderGGUF2 is None:
                    raise RuntimeError("GGUF not installed. Please install gguf and protobuf")
                return CLIPLoaderGGUF2().load_clip(clip1, clip_type)[0]
            return CLIPLoader().load_clip(clip1, clip_type, "default")[0]
        if clip1.endswith(".gguf"):
            if not GGUF_AVAILABLE or DualCLIPLoaderGGUF2 is None:
                raise RuntimeError("GGUF not installed. Please install gguf and protobuf")
            return DualCLIPLoaderGGUF2().load_clip(clip1, clip2, clip_type)[0]
        return DualCLIPLoader().load_clip(clip1, clip2, clip_type, "default")[0]

    def process_settings(self, node_id=None, steps=20, cfg=8.0,
                         sampler="euler", scheduler="normal",
                         unet_Weight_Dtype="default", clip_type=None,
                         vae=None, audio_vae=None, unet_name=None,
                         ckpt_name=None, clip1=None, clip2=None,
                         over_model=None, over_clip=None, lora_stack=None,
                         collapse_sampling_parameters=False):
        del node_id, collapse_sampling_parameters
        ckpt_name = ckpt_name or "None"
        unet_name = unet_name or "None"
        clip_type = clip_type or "None"
        clip1 = clip1 or "None"
        clip2 = clip2 or "None"
        if ckpt_name != "None" and unet_name != "None":
            raise ValueError("ckpt_name and unet_name cannot be used at the same time")

        model = over_model
        clip = over_clip
        checkpoint_vae = None
        if model is None and ckpt_name != "None":
            model, checkpoint_clip, checkpoint_vae = CheckpointLoaderSimple().load_checkpoint(ckpt_name)
            if clip is None:
                clip = checkpoint_clip
        elif model is None:
            model = self._load_unet(unet_name, unet_Weight_Dtype)

        if over_clip is None and (clip1 != "None" or clip2 != "None"):
            clip = self._load_clip(clip1, clip2, clip_type)
        if lora_stack is not None:
            model, clip = apply_lora_stack(model, clip, lora_stack)

        positive = None
        negative = None
        if clip is not None:
            positive = CLIPTextEncode().encode(clip, "a girl")[0]
            negative = CLIPTextEncode().encode(clip, "worst quality, low quality")[0]
            if clip1 != "None" and clip2 != "None":
                positive = node_helpers.conditioning_set_values(positive, {"guidance": 3.5})

        vae_obj = checkpoint_vae
        if isinstance(vae, str) and vae != "None":
            vae_obj = VAELoader().load_vae(vae)[0]
        audio_vae_obj = None
        if isinstance(audio_vae, str) and audio_vae != "None":
            audio_vae_obj = VAELoader().load_vae(audio_vae)[0]

        if clip_type == "flux2":
            latent = torch.zeros(
                [1, 128, 32, 32],
                device=comfy.model_management.intermediate_device(),
            )
        else:
            latent = torch.zeros([1, 16, 64, 64])

        context = new_context(None, **{
            "model": model,
            "positive": positive,
            "negative": negative,
            "latent": {"samples": latent},
            "vae": vae_obj,
            "audio_vae": audio_vae_obj,
            "clip": clip,
            "steps": steps,
            "cfg": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "guidance": 3.5,
            "clip_type": clip_type,
            "clip1": clip1,
            "clip2": clip2,
            "unet_name": unet_name,
            "ckpt_name": ckpt_name,
            "pos": "a girl",
            "neg": "worst quality, low quality",
            "width": 512,
            "height": 512,
            "batch": 1,
        })
        return (context, model, clip)
