#region-------------------------------import-----------------------
import json
import re
import math
import torch
import comfy
import comfy.utils
import node_helpers
import comfy.model_management
from comfy_api.latest import io
from comfy_extras.nodes_qwen import TextEncodeQwenImage21

from ..office_unit import *
from ..main_unit import *
from ..NodeBasic.C_promp import text_clip_Generate
#endregion-----------------------------import----------------------------


class sum_qwenImage21(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="sum_qwenImage21",
            display_name="sum_qwenImage21",
            category="Apt_Preset/chx_tool",
            inputs=[
                io.Custom("RUN_CONTEXT").Input("context"),
                io.Model.Input("model", optional=True),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
                io.String.Input("negative_prompt", default="blur", multiline=False, dynamic_prompts=True),
                io.Boolean.Input("prompt_enhance", default=False),
                io.DynamicCombo.Input("resolution", tooltip="keep resolution：保持原尺寸；resize：统一分辨率。", options=[
                    io.DynamicCombo.Option("keep resolution", []),
                    io.DynamicCombo.Option("resize", [
                        io.Float.Input("megapixels", default=1.0, min=0.01, max=16.0, step=0.01),
                    ]),
                ]),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    io.Image.Input("image", optional=True), prefix="image", min=1, max=100,
                ), optional=True),
            ],
            outputs=[
                io.Custom("RUN_CONTEXT").Output(display_name="context"),
                io.Conditioning.Output(display_name="positive"),
                io.Conditioning.Output(display_name="negative"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="text"),
            ],
        )

    @classmethod
    def execute(cls, context, prompt, negative_prompt, prompt_enhance, resolution: io.DynamicCombo.Type,
                model=None, images: io.Autogrow.Type = None) -> io.NodeOutput:
        clip = context.get("clip")
        vae = context.get("vae")
        if model is None:
            model = context.get("model")
        prompt = prompt or context.get("pos") or ""
        negative_prompt = negative_prompt or "blur"

        source_images = [images[name] for name in sorted(images or {}, key=lambda name: int(name.removeprefix("image")))
                         if images[name] is not None]
        if prompt_enhance:
            enhance_images = (source_images[:5] + [None] * 5)[:5]
            enhance_prompt = "\n\n".join((prompt.rstrip(), _QWEN2_ENHANCE_INSTRUCTION)).strip("\n")
            enhanced_prompt = text_clip_Generate().generate(
                clip=clip, prompt=enhance_prompt,
                image1=enhance_images[0], image2=enhance_images[1], image3=enhance_images[2],
                image4=enhance_images[3], image5=enhance_images[4],
            )[0]
            if enhanced_prompt:
                prompt = "\n".join((prompt.rstrip(), enhanced_prompt.lstrip())).strip("\n")

        qwen_images = {f"image_{index}": image for index, image in enumerate(source_images, start=1)}
        image_resolution = 0 if resolution["resolution"] == "keep resolution" else round(math.sqrt(resolution["megapixels"]) * 1024)
        positive, negative, latent = TextEncodeQwenImage21.execute(
            clip, prompt, negative_prompt, vae=vae, resolution=image_resolution, images=qwen_images,
        ).args
        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent,
                              pos=prompt, neg=negative_prompt)
        return io.NodeOutput(context, positive, negative, latent, prompt)


_QWEN2_MAX_MEDIA = 16
_QWEN2_ENHANCE_INSTRUCTION = "Add detailed descriptions"
# 标签格式：@<image1> / <image1> / @<img2> / <图片3> 均可，统一解析为 image 序号
_QWEN2_TAG_RE = re.compile(
    r"@?<\s*(?:image|img|picture|pic|图片|图像)\s*(\d+)\s*>",
    re.IGNORECASE,
)


def _qwen2_parse_refs(prompt):
    """按首次出现顺序返回 prompt 中引用的 media 序号（1-based）。"""
    refs = []
    for match in _QWEN2_TAG_RE.finditer(str(prompt or "")):
        index = int(match.group(1))
        if 1 <= index <= _QWEN2_MAX_MEDIA and index not in refs:
            refs.append(index)
    return refs


def _qwen2_normalize_tags(prompt, refs=None):
    """按首次引用顺序转换为模型正文中的标准 <imageN> 指代。"""
    refs = refs or _qwen2_parse_refs(prompt)
    local_indices = {global_index: local_index for local_index, global_index in enumerate(refs, start=1)}
    return _QWEN2_TAG_RE.sub(
        lambda match: f"<image{local_indices.get(int(match.group(1)), int(match.group(1)))}>",
        str(prompt or ""),
    )


def _qwen2_select_stage_prompt(stage_prompts, prompt, stage_index):
    """根据 stage_index（0-based）从 stage_prompts 中选取当前 prompt。

    stage_prompts 是 JSON 数组，每项是字符串或 {"prompt": str, "single_stage_time": float}。
    不传 / 解析失败时回退到顶层 prompt。
    """
    try:
        raw = json.loads(str(stage_prompts or "[]"))
    except (ValueError, TypeError):
        raw = []
    if not isinstance(raw, list):
        raw = []

    texts = []
    for item in raw:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("prompt"), str):
            texts.append(item["prompt"])

    if not texts:
        return str(prompt or "")

    idx = max(0, min(len(texts) - 1, int(stage_index or 0) - 1))
    return texts[idx]


class sum_QwenImage2:

    @classmethod
    def INPUT_TYPES(s):
        optional = {
            "context": ("RUN_CONTEXT",),
            "model": ("MODEL", {"lazy": True}),
            "stage_index": ("INT", {"forceInput": True, "default": 1,
                                     "min": 0, "max": 1024, "step": 1,
                                     "tooltip": "不传或 0 时取第 1 段。"}),
            "latent_image": ("IMAGE", {
                "forceInput": True,
                "tooltip": "可选。接入后使用该图片宽高创建输出 latent；未接入时使用 width/height。",
            }),
            "media": ("IMAGE,VIDEO,AUDIO,LATENT,STRING",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "stage_prompts": ("STRING", {
                "multiline": True, "default": "[]",
          }),
            "negative_prompt": ("STRING", {"default": "blur", "tooltip": "默认 blur，单行；不建议修改"}),
            "ref_size_mode": ("BOOLEAN", {
                "default": False,
                "label_on": "各自原尺寸",
                "label_off": "统一分辨率",
                "tooltip": "参考图尺寸策略。统一分辨率（按 resolution 算法的面积缩放并保持长宽比）；"
                            "各自原尺寸（直接使用每张参考图的宽高，只做 32 倍数取整）。",
            }),
            "resolution": ("INT", {
                "default": 1024, "min": 0, "max": 4096, "step": 32,
                "tooltip": "把每张参考图缩放到 resolution x resolution 像素面积，"

            }),
            "width": ("INT", {
                "default": 1024, "min": 32, "max": 4096, "step": 32,
                "tooltip": "输出图的宽",
           }),
            "height": ("INT", {
                "default": 1024, "min": 32, "max": 4096, "step": 32,
                "tooltip": "输出图的高",
            }),
        }
        optional["prompt_enhance"] = ("BOOLEAN", {
            "default": False,
            "label_on": "开启",
            "label_off": "关闭",
            "tooltip": "开启后，自动添加 Add detailed descriptions，再结合当前提示词和最多 5 张参考图生成增强文本。",
        })
        for index in range(1, _QWEN2_MAX_MEDIA + 1):
            optional[f"media_{index}"] = ("IMAGE", {"lazy": True})
        return {
            "required": {},
            "optional": optional,
            "hidden": {},
        }

    RETURN_TYPES = ("RUN_CONTEXT", "CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("context", "positive", "negative", "latent", "text")
    FUNCTION = "encode"
    CATEGORY = "Apt_Preset/chx_tool/😺backup"
    DESCRIPTION = (
    )

    def check_lazy_status(self, prompt="", stage_index=1, stage_prompts="[]", **kwargs):
        required = []
        if kwargs.get("model") is None and "model" in kwargs:
            required.append("model")
        selected_prompt = _qwen2_select_stage_prompt(stage_prompts, prompt, stage_index)
        for index in _qwen2_parse_refs(selected_prompt):
            if kwargs.get(f"media_{index}") is None:
                required.append(f"media_{index}")
        return required

    @staticmethod
    def _process_image_channels(image):
        if image is None:
            return None
        if image.ndim == 4:
            b, h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                image = rgb * alpha + torch.zeros_like(rgb) * (1.0 - alpha)
            elif c != 3:
                image = image[..., :3]
        elif image.ndim == 3:
            h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                image = rgb * alpha + torch.zeros_like(rgb) * (1.0 - alpha)
            elif c != 3:
                image = image[..., :3]
        return image.clamp(0.0, 1.0)

    def encode(self, context=None, model=None, stage_index=1, latent_image=None, media=None,
               prompt="", stage_prompts="[]", negative_prompt="blur", ref_size_mode=False,
               resolution=1024, width=1024, height=1024, prompt_enhance=False, **kwargs):
        clip = context.get("clip", None) if context else None
        vae = context.get("vae", None) if context else None
        if clip is None:
            raise ValueError("sum_QwenImage2: context 缺少 clip，请先接入模型加载节点")

        if model is None and context:
            model = context.get("model", None)
        if prompt == "" and context:
            prompt = context.get("pos", "")
        # negative prompt 默认 "blur"，单行；用户可改但不建议
        if not negative_prompt:
            negative_prompt = "blur"
        if isinstance(media, str):
            prompt = media

        active_prompt = _qwen2_select_stage_prompt(stage_prompts, prompt, stage_index)
        refs = _qwen2_parse_refs(active_prompt)

        images_vl = []
        ref_latents = []
        # 默认 latent 尺寸：直接用 width/height，不从参考图推断
        latent_w = int(width) if width else 1024
        latent_h = int(height) if height else 1024

        direct = media if (isinstance(media, torch.Tensor) and media.ndim == 4) else None
        for global_index in refs:
            image = kwargs.get(f"media_{global_index}")
            if image is None and global_index == 1:
                image = direct
            if image is None:
                raise ValueError(
                    f"sum_QwenImage2: prompt 引用了 @<image{global_index}>，但 media_{global_index} 未连接"
                )
            image = self._process_image_channels(image)
            # 以下缩放与编码逻辑完全复刻官方 TextEncodeQwenImage21
            samples = image[:1].movedim(-1, 1)
            if ref_size_mode:
                # True = 各自原尺寸：每张图按自己的宽高取 32 倍数，resolution 控件被忽略（前端也已灰化）
                ref_w = round(samples.shape[3] / 32) * 32
                ref_h = round(samples.shape[2] / 32) * 32
            elif resolution > 0:
                ratio = samples.shape[3] / samples.shape[2]
                ref_w = round(math.sqrt(resolution * resolution * ratio) / 32) * 32
                ref_h = round(math.sqrt(resolution * resolution / ratio) / 32) * 32
            else:
                # False=统一分辨率 但 resolution=0 的退化情况：保持原图（取 32 倍数），与官方一致
                ref_w = round(samples.shape[3] / 32) * 32
                ref_h = round(samples.shape[2] / 32) * 32
            ref_w, ref_h = max(32, ref_w), max(32, ref_h)
            if (ref_w, ref_h) == (samples.shape[3], samples.shape[2]):
                s = image[:1]
            else:
                s = comfy.utils.common_upscale(samples, ref_w, ref_h, "lanczos", "disabled").movedim(1, -1)
            rgb = s[:, :, :, :3]
            if s.shape[-1] > 3:
                rgb = rgb * s[:, :, :, 3:] + (1.0 - s[:, :, :, 3:])
            images_vl.append(rgb)
            if vae is not None:
                ref_latents.append(vae.encode(s))

        # latent_image 是唯一的图片尺寸控制入口。接入时按首帧宽高创建
        # 输出 latent；未接入时严格使用 width/height，避免参考素材污染尺寸。
        if isinstance(latent_image, torch.Tensor) and latent_image.ndim == 4:
            source = self._process_image_channels(latent_image)
            h, w = source.shape[1], source.shape[2]
            # 与官方一致：宽高按 32 倍数向下对齐（VAE 8 × DiT patch 2）。
            latent_h = max((h // 32) * 32, 32)
            latent_w = max((w // 32) * 32, 32)
        else:
            latent_w = max(32, int(width) if width else 1024)
            latent_h = max(32, int(height) if height else 1024)

        if prompt_enhance:
            enhance_images = images_vl[:5] + [None] * max(0, 5 - len(images_vl))
            enhance_prompt = "\n\n".join((active_prompt.rstrip(), _QWEN2_ENHANCE_INSTRUCTION)).strip("\n")
            enhanced_prompt = text_clip_Generate().generate(
                clip=clip,
                prompt=enhance_prompt,
                image1=enhance_images[0],
                image2=enhance_images[1],
                image3=enhance_images[2],
                image4=enhance_images[3],
                image5=enhance_images[4],
            )[0]
            if enhanced_prompt:
                active_prompt = "\n".join((active_prompt.rstrip(), enhanced_prompt.lstrip())).strip("\n")

        model_prompt = _qwen2_normalize_tags(active_prompt, refs)
        keep_vision = len(ref_latents) == 0
        positive = clip.encode_from_tokens_scheduled(
            clip.tokenize(model_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
        )
        negative = clip.encode_from_tokens_scheduled(
            clip.tokenize(negative_prompt, images=images_vl, keep_vision=keep_vision, prevent_empty_text=True)
        )
        if len(ref_latents) > 0:
            positive = node_helpers.conditioning_set_values(
                positive, {"reference_latents": ref_latents}, append=True
            )
            negative = node_helpers.conditioning_set_values(
                negative, {"reference_latents": ref_latents}, append=True
            )

        latent = torch.zeros(
            [1, 64, latent_h // 16, latent_w // 16],
            device=comfy.model_management.intermediate_device(),
        )

        context = new_context(
            context,
            model=model,
            positive=positive,
            negative=negative,
            latent={"samples": latent},
            clip=clip,
            vae=vae,
        )
        return (context, positive, negative, {"samples": latent}, model_prompt)
