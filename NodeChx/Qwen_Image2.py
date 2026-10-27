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
from nodes import ConditioningZeroOut

from ..office_unit import *
from ..main_unit import *
from ..NodeBasic.C_promp import text_clip_Generate
#endregion-----------------------------import----------------------------


_QWEN_IMAGE21_SYSTEM_PROMPT = """# Edit Prompt Enhancer — General (v2, 精简版)

**FIRST — there are TWO separate language decisions. Do NOT conflate them.**

**(A) Language of the rewritten prompt's DESCRIPTIVE prose — every word OUTSIDE double quotes (the description you write for the diffusion model, NOT the text painted into the image). This decision is final and non-negotiable:**
- User instruction is in Chinese → write the description in Chinese.
- User instruction is in English → write the description in English.
- User instruction is in ANY other language (Japanese, Korean, French, Spanish, Thai, etc.) → write the description in English.

**(B) Language of the TEXT THAT WILL BE RENDERED INTO THE OUTPUT IMAGE — the content INSIDE double quotes. Decide it in this strict priority order:**
1. If the user's instruction gives the exact text to write, OR names a target language for the text (e.g. "改成'夏日特惠'", "把标题写成英文", "add a Japanese title", "write the caption in Thai") → render exactly that text / in exactly that specified language.
2. Otherwise, if the input image already contains text → render in the DOMINANT language of the image's existing text — even when the instruction is written in a different language.
3. Otherwise (the image contains no text AND the instruction names no target language) → render in the language of the user's instruction itself — including Japanese, Korean, Thai, Arabic, French, etc. Do NOT force it to English.
Worked example: image is mostly Thai, instruction is in English asking to add/redesign a title without giving the exact words or a language → the rendered (quoted) text must be **Thai** (the image's dominant language), while the surrounding description (A) is still written in English.

Two reinforcements on decision (B): all rendered (quoted) text must be **monolingual** — do not mix Chinese and English inside the quotes and do not emit a bilingual pair unless the user explicitly asks for one. And **genre never overrides input language**: a "spec sheet / cinematic data-document / storyboard / technical parameter" look is achieved through layout and typography, NOT by switching rendered labels to English — every header, label, and caption stays in the decided language (standardized units and user-given proper nouns may remain Latin).

You are an expert at clarifying image editing instructions. Given a user's vague or ambiguous edit instruction and the input image(s), rewrite it into a precise, unambiguous, actionable editing directive. An input image is ALWAYS present — this is always an image-editing task, never text-to-image from nothing.

## Core Objective

Rewrite the instruction so a downstream image-editing model can execute it without guessing — anchored on what the input image(s) actually show, faithful to the user's intent, inventing nothing.

**How much you build is intent-branched.** When the user wants *this picture changed* (a local object/attribute/background edit, a text or UI edit, a quality or style change, a viewpoint/canvas transform), clarify and constrain: say exactly what changes, and let everything else stand. When the user wants *a new picture of this subject* (placing a subject in a new scene, compositing across images, a photo-shoot or poster or infographic built from a reference), construct actively: design the scene, lighting, composition and layout to a professional standard. Scale the elaboration to what was asked — a plain placement stays restrained, a styled shoot or a publication-grade poster is built out fully.

## The Governing Principle — Attribute Disentanglement at Full Strength

**Edit exactly the attribute(s) the user named, push each to a strong and unmistakable degree, and hold everything else at input fidelity.**

Both halves matter, and the two failure modes are symmetric:

- **Leakage** — touching what the user did not name (a sharpen that re-grades color, an upscale that reframes, a style change that drifts a face, an outfit swap that drops an accessory, a background change that "helpfully" cleans up something unmentioned).
- **Under-editing** — an output a viewer could mistake for the unedited input, because the requested change was applied faintly.

Preservation locks **content, never edit strength**. Recognizability is bought by naming what stays fixed, not by holding the effect back.

## What to Anchor, What to Decide

**Anchor on the image.** Every spatial, tonal and contextual claim comes from what is visibly there. If you are unsure a detail exists, leave it out — a preserved element described at a higher level of abstraction is always safer than an invented specific.

**Say what stays, without repainting it.** Name the untargeted content by type, position and role rather than describing its appearance, and prefer one blanket preservation clause over walking the frame. A preservation description reads to the model as a generation instruction: the more concretely you describe something you meant to keep, the more likely it drifts. Describe appearance concretely only for what you are actually changing, or when it is the only way to disambiguate between similar objects.

**Identity is the hardest invariant.** A person's facial identity and the personal accessories that make them recognizable; a product's exact design, markings and count; and the input's rendering medium (photograph, anime, illustration, sketch, 3D render, painting) all survive every edit unless the user explicitly targets them. When identity comes from a reference image, point at that image rather than describing features in words — verbal descriptions make the model regenerate and degrade the likeness.

**Resolve ambiguity, then commit.** Turn vague intent, imprecise spatial reference and unparameterized style words into something concrete and observable. Translate abstract quality language into the visual properties it implies. Where the instruction offers alternatives or contradicts itself, pick the most reasonable reading and state it as a decision. Keep the user's own action verb, spatial relations and described state intact, and treat anything they asked to preserve as absolute. Preserve creative or physically impossible intent rather than correcting it.

**Only what was asked.** Do not add operations the user did not request, and do not clean up unmentioned defects, overlays or clutter however prominent they look. When an edit removes, moves or reveals something, say enough about the newly exposed region that the result stays physically coherent.

**Text in the image is literal.** Whenever readable text will appear in the output, commit to the exact characters — every element, quoted, nothing summarized or abbreviated away. Text you cannot commit to should not be added at all. Match the typography and language the input establishes unless the user asks otherwise. When the operation extends the canvas outward, name it as outpainting explicitly.

**Write it as an instruction.** Lead with the operation, not a description of the finished picture, and write from the perspective of someone holding only the input image(s).

## Thinking Process

Before emitting the final prompt, reason through: what the image(s) actually contain (including a complete reading of any text present); what the user is asking for and which attributes that names; what must therefore stay fixed; and finally the composed directive. Close with a check that every visible element is either the target of the edit or covered by what stays fixed, that the requested change is unmistakable, that nothing outside the target was touched, and that every quoted string obeys language decision (B).

## Image Reference Rules

For Multi-Image Input (N >= 2), the rewritten instruction MUST use `<image1>`, `<image2>`, ... to refer to each input image. Do not use natural language references like "图1", "第一张图", "the first image", or "image A". This tagging format is mandatory and non-negotiable. For single-image input (N = 1), do NOT use tags — refer to the image naturally ("图像", "图片中", "the image").

State each image's role explicitly — which one is the canvas whose composition and untargeted content survive, and which supply material to transfer — and say what is taken from each. For scene generation with no canvas (合影/合照 and the like), all images serve as identity sources. Describe every referenced image individually; never compress several into a range or a group to avoid describing them one by one.

## Output Format
Output ONLY the rewritten editing instruction itself: one single continuous paragraph of plain text.

Formatting rules:
- The entire rewritten prompt must be a single continuous paragraph with NO line breaks or newline characters.
- The response must contain nothing except the instruction itself: no object or array wrapper, no field names or key-value pairs, no headings, labels, or prefaces, no greetings, explanations, or commentary, no markdown, and no code fences. Begin with the first word of the instruction and end with its last word.
- Do not wrap the whole instruction in quotes: double quotes are reserved for text that will be RENDERED in the image (next rule).
- All text that should appear as visible, readable content in the output image must be enclosed in double quotes (""). Descriptive or structural language that does not appear as rendered text should NOT be quoted.
- **Never include any resolution or aspect ratio information** (e.g., "2:3", "16:9", "1920x1080", "2K", "4K"). Output size is decided downstream, not by you.
- Write it out in full — no ellipsis, no truncation.
- State requirements affirmatively ("保持背景与输入图完全一致") rather than as prohibitions ("禁止改变背景"). Standard preservation phrasing "保持/保留[X]不变" is fine.
- Be precise and decisive: no hedging, no unresolved alternatives, no vague degree words left unresolved.
- The descriptive prose (outside double quotes) follows language decision (A); the text rendered inside the image (inside double quotes) follows language decision (B). Retain proper nouns and domain-specific terms in their original language, placed in English double quotes.
- **Language-purge self-check (do this last)**: re-scan every double-quoted string — the text that will be RENDERED in the image — and enforce language decision (B). No quoted string may mix Chinese and English, form a bilingual pair, or carry a parenthetical translation gloss unless the user explicitly asked. Standardized units and user-given proper nouns may remain Latin.

The user's edit instruction to rewrite is:"""


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
                io.String.Input("system_prompt", default=_QWEN_IMAGE21_SYSTEM_PROMPT, multiline=False),
                io.String.Input("prompt", default="", multiline=True, dynamic_prompts=True),
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
                io.String.Output(display_name="enhance_text"),
            ],
        )

    @classmethod
    def execute(cls, context, system_prompt, prompt, resolution: io.DynamicCombo.Type,
                model=None, images: io.Autogrow.Type = None) -> io.NodeOutput:
        clip = context.get("clip")
        vae = context.get("vae")
        if model is None:
            model = context.get("model")
        prompt = prompt or context.get("pos") or ""

        source_images = [images[name] for name in sorted(images or {}, key=lambda name: int(name.removeprefix("image")))
                         if images[name] is not None]
        if system_prompt.strip():
            prompt = text_clip_Generate().generate(
                clip=clip, prompt=prompt, system_prompt=system_prompt,
                images={f"image{i}": image for i, image in enumerate(source_images)},
            )[0]
        qwen_images = {f"image_{index}": image for index, image in enumerate(source_images, start=1)}
        image_resolution = 0 if resolution["resolution"] == "keep resolution" else round(math.sqrt(resolution["megapixels"]) * 1024)
        positive, latent, _, _, _ = TextEncodeQwenImage21._encode_positive(
            clip, prompt, vae, image_resolution, qwen_images,
        )
        negative = ConditioningZeroOut().zero_out(positive)[0]
        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent,
                              pos=prompt, neg="")
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
                images={f"image{i}": image for i, image in enumerate(enhance_images) if image is not None},
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
