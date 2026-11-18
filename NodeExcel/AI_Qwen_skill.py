import base64
import io as python_io
import json
import os

import numpy as np
import requests
import torch
from PIL import Image

import folder_paths
from comfy_api.latest import io


_TEXT_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
_MULTIMODAL_API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
_API_KEY_FILE = os.path.join(os.path.dirname(__file__), "ApiKey_AI_Qwen.txt")
_SKILLS_DIR = os.path.join(folder_paths.models_dir, "skills")
_NO_SKILL = "不使用技能"
_MODEL_NAMES = {"Qwen3.8-Max": "qwen3.8-max"}
_SKILL_EXTENSIONS = {".md", ".txt", ".json", ".yaml", ".yml", ".rst"}
_REQUEST_TIMEOUT = 300
_VIDEO_SAMPLE_FRAMES = 8
_MEDIA_INPUT_LIMIT = 20
_HISTORY_MAX_TURNS = 12
_OUTPUT_REALTIME = "实时输出"
_OUTPUT_MANUAL = "仅对话"


def _api_key(value):
    value = str(value or "").strip()
    if value:
        return value
    for name in ("DASHSCOPE_API_KEY", "aliyun_API_KEY"):
        value = str(os.getenv(name) or "").strip()
        if value:
            return value
    if os.path.isfile(_API_KEY_FILE):
        with open(_API_KEY_FILE, "r", encoding="utf-8") as file:
            return file.read().strip()
    return ""


def _skill_files(path):
    if os.path.isfile(path):
        return [path] if os.path.splitext(path)[1].lower() in _SKILL_EXTENSIONS else []

    files = []
    for root, directories, names in os.walk(path):
        directories[:] = sorted(name for name in directories if not name.startswith(".") and name != "__pycache__")
        for name in sorted(names):
            if name.startswith(".") or os.path.splitext(name)[1].lower() not in _SKILL_EXTENSIONS:
                continue
            files.append(os.path.join(root, name))
    return sorted(files, key=lambda value: (os.path.basename(value).lower() != "skill.md", value.lower()))


def _skill_entries():
    os.makedirs(_SKILLS_DIR, exist_ok=True)
    entries = {_NO_SKILL: None}
    with os.scandir(_SKILLS_DIR) as items:
        for item in sorted(items, key=lambda value: value.name.lower()):
            if item.name.startswith("."):
                continue
            path = os.path.realpath(item.path)
            if item.is_dir(follow_symlinks=False) and _skill_files(path):
                entries[f"{item.name}/"] = path
            elif item.is_file(follow_symlinks=False) and os.path.splitext(item.name)[1].lower() in _SKILL_EXTENSIONS:
                entries[item.name] = path
    return entries


def _read_skill(selection):
    selection = str(selection or _NO_SKILL)
    if selection == _NO_SKILL:
        return ""
    path = _skill_entries().get(selection)
    if not path:
        raise ValueError(f"技能“{selection}”不存在，请刷新页面后重新选择")

    skills_root = os.path.realpath(_SKILLS_DIR)
    if os.path.commonpath((skills_root, path)) != skills_root:
        raise ValueError("技能路径超出 ComfyUI/models/skills")

    sections = []
    for filename in _skill_files(path):
        resolved = os.path.realpath(filename)
        if os.path.commonpath((skills_root, resolved)) != skills_root:
            continue
        with open(resolved, "r", encoding="utf-8-sig", errors="replace") as file:
            content = file.read().strip()
        if content:
            relative = os.path.relpath(resolved, _SKILLS_DIR).replace(os.sep, "/")
            sections.append(f"## {relative}\n\n{content}")
    if not sections:
        raise ValueError(f"技能“{selection}”中没有可读取的文本内容")
    return "\n\n".join(sections)


def _media_values(media):
    if isinstance(media, dict):
        def order(item):
            suffix = str(item[0]).rsplit("_", 1)[-1]
            return int(suffix) if suffix.isdigit() else 0
        return [value for _, value in sorted(media.items(), key=order) if value is not None]
    if isinstance(media, (list, tuple)):
        return [value for value in media if value is not None]
    return [media] if media is not None else []


def _media_type(value):
    if isinstance(value, str):
        return "text"
    if isinstance(value, torch.Tensor):
        return "image"
    if hasattr(value, "get_components"):
        return "video"
    if isinstance(value, dict) and ("images" in value or "frames" in value):
        return "video"
    return ""


def _image_data_url(image):
    tensor = image.detach().cpu().float()
    if tensor.ndim == 4:
        tensor = tensor[0]
    array = np.clip(tensor.numpy() * 255.0, 0, 255).astype(np.uint8)
    if array.ndim == 2:
        pil_image = Image.fromarray(array).convert("RGB")
    else:
        pil_image = Image.fromarray(array[..., :3]).convert("RGB")
    if max(pil_image.size) > 1536:
        pil_image.thumbnail((1536, 1536), Image.Resampling.LANCZOS)
    buffer = python_io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=88, optimize=True)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _video_frames(value):
    if hasattr(value, "get_components"):
        frames = value.get_components().images
    else:
        frames = value.get("images") if isinstance(value, dict) else None
        if frames is None and isinstance(value, dict):
            frames = value.get("frames")
    if not isinstance(frames, torch.Tensor) or frames.ndim < 4 or frames.shape[0] == 0:
        return []
    count = int(frames.shape[0])
    sample_count = min(count, _VIDEO_SAMPLE_FRAMES)
    if sample_count == 1:
        indices = [0]
    else:
        indices = [round(index * (count - 1) / (sample_count - 1)) for index in range(sample_count)]
    return [frames[index] for index in indices]


def _media_content(media):
    content = []
    counts = {"image": 0, "video": 0, "text": 0}
    has_visual = False
    for value in _media_values(media):
        media_type = _media_type(value)
        if not media_type:
            raise ValueError(f"media 不支持输入类型：{type(value).__name__}")
        counts[media_type] += 1
        ordinal = counts[media_type]
        if media_type == "text":
            content.append({"text": f"[Text {ordinal}]\n{value}"})
            continue
        if media_type == "image":
            has_visual = True
            images = value if value.ndim == 4 else value.unsqueeze(0)
            for index, image in enumerate(images):
                suffix = f"，批次图 {index + 1}/{len(images)}" if len(images) > 1 else ""
                content.append({"text": f"[Image {ordinal}{suffix}]"})
                content.append({"image": _image_data_url(image)})
            continue
        frames = _video_frames(value)
        if not frames:
            raise ValueError(f"Video {ordinal} 没有可读取的视频帧")
        has_visual = True
        for index, frame in enumerate(frames):
            content.append({"text": f"[Video {ordinal}，代表帧 {index + 1}/{len(frames)}]"})
            content.append({"image": _image_data_url(frame)})
    return content, has_visual


def _response_text(data):
    try:
        content = data["output"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"API 返回格式异常：{data}") from error
    if isinstance(content, list):
        return "\n".join(str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content).strip()
    return str(content or "").strip()


def _conversation_state(value, skill, model):
    try:
        state = json.loads(str(value or ""))
    except (TypeError, ValueError):
        state = {}
    if not isinstance(state, dict) or state.get("skill") != skill or state.get("model") != model:
        state = {"version": 1, "skill": skill, "model": model, "turns": []}
    turns = state.get("turns")
    if not isinstance(turns, list):
        turns = []
    state["turns"] = [
        {"user": str(turn.get("user") or ""), "assistant": str(turn.get("assistant") or "")}
        for turn in turns
        if isinstance(turn, dict) and (turn.get("user") or turn.get("assistant"))
    ][-_HISTORY_MAX_TURNS:]
    return state


def _conversation_ui(state, committed_result, answer="", error=""):
    payload = {
        "history": state,
        "committed_result": str(committed_result or ""),
        "answer": str(answer or ""),
        "error": str(error or ""),
    }
    return {"conversation": [json.dumps(payload, ensure_ascii=False)]}


class AI_Qwen_skill(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        skills = list(_skill_entries())
        media_input = lambda name: io.MultiType.Input(name, [io.Image, io.Video, io.String], optional=True)
        return io.Schema(
            node_id="AI_Qwen_skill",
            display_name="AI_Qwen_skill",
            category="Apt_Preset/AI_tool",
            description="使用杭州区百炼模型、本地技能及图片/视频/文本素材。",
            inputs=[
                io.String.Input("text", multiline=True, placeholder="输入指令；输入 @ 可引用已连接素材", default=""),
                io.Combo.Input("skill", options=skills, default=skills[0]),
                io.Combo.Input("llm_model", options=["Qwen3.8-Max", "自定义"], default="Qwen3.8-Max"),
                io.String.Input("custom_model", placeholder="选择“自定义”时填写模型名称", default=""),
                io.String.Input("api_key", default=""),
                io.Combo.Input("output_mode", options=[_OUTPUT_REALTIME, _OUTPUT_MANUAL], default=_OUTPUT_REALTIME),
                io.String.Input("conversation_history", default=""),
                io.String.Input("committed_result", default=""),
                io.String.Input("conversation_action", default="send"),
                media_input("media"),
                *(media_input(f"media_{index}") for index in range(1, _MEDIA_INPUT_LIMIT + 1)),
            ],
            outputs=[io.String.Output("result")],
        )

    @classmethod
    def execute(cls, text, skill, llm_model, custom_model, api_key, output_mode,
                conversation_history, committed_result, conversation_action,
                media=None, **kwargs):
        try:
            key = _api_key(api_key)
            if not key:
                raise ValueError("请填写 API Key")

            if llm_model == "自定义":
                model = str(custom_model or "").strip()
                if not model:
                    raise ValueError("选择自定义模型后，请填写自定义模型名称")
            else:
                model = _MODEL_NAMES.get(llm_model, str(llm_model or "").strip())

            state = _conversation_state(conversation_history, str(skill or ""), model)
            if str(conversation_action or "") == "adopt":
                latest = state["turns"][-1]["assistant"] if state["turns"] else str(committed_result or "")
                return io.NodeOutput(latest, ui=_conversation_ui(state, latest, latest))

            messages = []
            skill_content = _read_skill(skill)
            if skill_content:
                messages.append({
                    "role": "system",
                    "content": (
                        "你已加载以下本地技能。必须遵循该技能的工作流程和约束完成用户请求；"
                        "结合用户引用的素材制作结果，技能没有规定的部分按正常对话方式处理。\n\n"
                        f"{skill_content}"
                    ),
                })

            for turn in state["turns"]:
                messages.append({"role": "user", "content": turn["user"]})
                messages.append({"role": "assistant", "content": turn["assistant"]})

            media_values = {"media_0": media}
            media_values.update(
                (f"media_{index}", kwargs.get(f"media_{index}"))
                for index in range(1, _MEDIA_INPUT_LIMIT + 1)
            )
            media_content, has_visual = _media_content(media_values)
            user_text = str(text or "")
            if has_visual:
                messages.append({"role": "user", "content": [{"text": user_text}, *media_content]})
                api_url = _MULTIMODAL_API_URL
            else:
                text_materials = "\n\n".join(item["text"] for item in media_content)
                if text_materials:
                    user_text = f"{user_text}\n\n已连接素材：\n{text_materials}"
                messages.append({"role": "user", "content": user_text})
                api_url = _TEXT_API_URL

            response = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "input": {"messages": messages},
                    "parameters": {
                        "max_tokens": 8192,
                        "temperature": 0.7,
                        "top_p": 0.8,
                        "result_format": "message",
                    },
                },
                timeout=_REQUEST_TIMEOUT,
            )
            try:
                data = response.json()
            except ValueError as error:
                raise RuntimeError(f"服务返回了非 JSON 数据：{response.text[:500]}") from error
            if not response.ok:
                if isinstance(data, dict):
                    detail = data.get("message") or data.get("error") or data
                else:
                    detail = data
                raise RuntimeError(f"API 请求失败（HTTP {response.status_code}）：{detail}")
            answer = _response_text(data)
            state["turns"].append({"user": str(text or ""), "assistant": answer})
            state["turns"] = state["turns"][-_HISTORY_MAX_TURNS:]
            committed = answer if output_mode == _OUTPUT_REALTIME else str(committed_result or "")
            return io.NodeOutput(committed, ui=_conversation_ui(state, committed, answer))
        except Exception as error:
            failure = f"执行失败：{error}"
            if "state" not in locals():
                state = {"version": 1, "skill": str(skill or ""), "model": "", "turns": []}
            output = str(committed_result or "") if output_mode == _OUTPUT_MANUAL else failure
            return io.NodeOutput(output, ui=_conversation_ui(state, committed_result, error=failure))
