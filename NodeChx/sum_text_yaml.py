from ..main_unit import *
from nodes import CLIPTextEncode
import os, yaml, json, asyncio, shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from server import PromptServer
from aiohttp import web

class YAMLFileHandler(FileSystemEventHandler):
    def __init__(self, node_instance): self.node_instance = node_instance
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.yaml'): self.node_instance.refresh_enums()
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.yaml'): self.node_instance.refresh_enums()
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.yaml'): self.node_instance.refresh_enums()

class text_sum:
    def __init__(self):
        self.yaml_dir = Path(__file__).parent / "yaml"
        self.yaml_dir.mkdir(exist_ok=True)
        self.write_lock = asyncio.Lock()
        self.observer = None
        self.file_handler = YAMLFileHandler(self)
        self.server = PromptServer.instance
        self._start_watchdog()
        self._register_socket_handlers()
        self.refresh_enums()

    @classmethod
    def INPUT_TYPES(cls):
        yaml_dir = Path(__file__).parent / "yaml"
        yaml_dir.mkdir(exist_ok=True)
        yaml_files = [f.name for f in yaml_dir.glob("*.yaml")]
        if not yaml_files:
            default_yaml = yaml_dir / "default.yaml"
            default_yaml.write_text("example:\n  prompt: 'Enter your prompt here'\n")
            yaml_files = ["default.yaml"]
        all_titles = set([""])
        for yaml_file in yaml_files:
            yaml_path = yaml_dir / yaml_file
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f) or {}
                    all_titles.update(data.keys())
            except: pass
        return {
            "required": {
                "select_yaml": (sorted(yaml_files), {"default": yaml_files[0] if yaml_files else ""}),
                "select": (sorted(list(all_titles)), {"default": ""}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": False}),
                "add_pos": ("STRING", {"default": "", "multiline": False}),
                "add_neg": ("STRING", {"default": "", "multiline": False}),
                "style": (["None"] + style_list()[0], {"default": "None"}),
                "remove": ("STRING", {"multiline": False,"default": ""}),
                "replace_target": ("STRING", {"multiline": False,"default": ""}),
                "replace": ("STRING", {"multiline": False,"default": ""}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("pos", "neg",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/prompt"
    DESCRIPTION = """
- 替换内容，移除内容：多个词汇同时进行，用@隔开。例如："a@b@c"
- """


    @classmethod
    def _get_instance(cls):
        if not hasattr(cls, '_instance'): cls._instance = cls()
        return cls._instance

    def _get_yaml_files(self) -> List[str]:
        if not self.yaml_dir.exists(): return ["default.yaml"]
        yaml_files = [f.name for f in self.yaml_dir.glob("*.yaml")]
        if not yaml_files:
            default_yaml = self.yaml_dir / "default.yaml"
            default_yaml.write_text("example:\n  prompt: 'Enter your prompt here'\n")
            yaml_files = ["default.yaml"]
        return sorted(yaml_files)

    def _get_titles_from_yaml(self, yaml_file: str) -> List[str]:
        yaml_path = self.yaml_dir / yaml_file
        if not yaml_path.exists(): return [""]
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            return list(data.keys())
        except Exception as e:
            print(f"Error reading YAML {yaml_file}: {e}")
            self._handle_corrupt_yaml(yaml_path)
            return [""]

    def _handle_corrupt_yaml(self, yaml_path: Path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        bad_path = yaml_path.parent / f"bad_{timestamp}_{yaml_path.name}"
        shutil.move(str(yaml_path), str(bad_path))
        yaml_path.write_text("# Recovered from corrupt file\n")

    def _start_watchdog(self):
        if self.observer is None:
            self.observer = Observer()
            self.observer.schedule(self.file_handler, str(self.yaml_dir), recursive=False)
            self.observer.start()

    def refresh_enums(self):
        yaml_files = self._get_yaml_files()
        enum_data = {"yaml_files": yaml_files, "titles_by_yaml": {}}
        for yaml_file in yaml_files:
            titles = self._get_titles_from_yaml(yaml_file)
            enum_data["titles_by_yaml"][yaml_file] = titles
        self._broadcast_enum(enum_data)

    def _broadcast_enum(self, enum_data: Dict):
        if hasattr(self.server, 'send_sync'):
            self.server.send_sync("sum_text_list_enum", enum_data)
        else:
            try:
                if hasattr(self.server, 'socketio'):
                    self.server.socketio.emit("sum_text_list_enum", enum_data)
            except: pass

    def _register_socket_handlers(self):
        server = PromptServer.instance

        @server.routes.post("/sum_text_list/get_prompt")
        async def get_prompt(request):
            data = await request.json()
            yaml_file = data.get("yaml", "")
            title = data.get("title", "")
            node_id = data.get("node_id", None)
            response_data = self._get_prompt_data(yaml_file, title)
            response_data["node_id"] = node_id
            if hasattr(server, 'send_sync'):
                server.send_sync("sum_text_list_set_widgets", response_data)
            return web.json_response({"success": True})

        @server.routes.post("/sum_text_list/reload_yamls")
        async def reload_yamls(request):
            self.refresh_enums()
            return web.json_response({"success": True})

    def _get_prompt_data(self, yaml_file: str, title: str) -> Dict[str, str]:
        yaml_path = self.yaml_dir / yaml_file
        if not yaml_path.exists(): return {"title": title, "prompt": "", "negative": ""}
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if title in data and isinstance(data[title], dict):
                prompt = data[title].get("prompt", "")
                negative = data[title].get("negative", "")
            else: prompt, negative = "", ""
            return {"title": title, "prompt": prompt, "negative": negative}
        except Exception as e:
            print(f"Error reading prompt: {e}")
            return {"title": title, "prompt": "", "negative": ""}

    async def _save_yaml(self, yaml_file: str, data: Dict):
        async with self.write_lock:
            yaml_path = self.yaml_dir / yaml_file
            with tempfile.NamedTemporaryFile(mode='w', dir=str(self.yaml_dir), delete=False, encoding='utf-8') as tmp:
                yaml.dump(data, tmp, default_flow_style=False, allow_unicode=True, sort_keys=True)
                tmp_path = tmp.name
            os.replace(tmp_path, str(yaml_path))

    def delete_title(self, yaml_file: str, title: str):
        yaml_path = self.yaml_dir / yaml_file
        if not yaml_path.exists(): return
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
            if title in data:
                del data[title]
                asyncio.run(self._save_yaml(yaml_file, data))
                self.refresh_enums()
        except Exception as e:
            print(f"Error deleting title: {e}")

    def run(self, style="default", negative="", replace_target="", replace="",remove="", add_pos="", add_neg="",  select_yaml: str = "", select: str = "", prompt: str = "", unique_id: str = "") -> Tuple[str]:
        if len(prompt) > 4096: print(f"Warning: Prompt length ({len(prompt)}) exceeds recommended 4096 chars")
        if select and prompt:
            yaml_path = self.yaml_dir / select_yaml
            try:
                if yaml_path.exists():
                    with open(yaml_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f) or {}
                else: data = {}
                if select not in data: data[select] = {}
                data[select]["prompt"] = prompt
                if negative: data[select]["negative"] = negative
                asyncio.run(self._save_yaml(select_yaml, data))
            except Exception as e: print(f"Error saving prompt: {e}")
        if add_pos is not None and add_pos != "": prompt = prompt + "," + add_pos
        if add_neg is not None and add_neg != "": negative = negative + "," + add_neg
        if isinstance(prompt, tuple): prompt = ", ".join(str(x) for x in prompt if x is not None)
        elif not isinstance(prompt, str): prompt = str(prompt)
        if isinstance(negative, tuple): negative = ", ".join(str(x) for x in negative if x is not None)
        elif not isinstance(negative, str): negative = str(negative)
        prompt, negative = add_style_to_subject(style,  prompt, negative)
        if remove is not None and remove!= "": prompt = clean_prompt(prompt, remove)
        if replace_target is not None and replace_target!= "": 
           prompt = multi_replace(prompt, replace_target, replace)

        pos= prompt
        neg = negative
        return (pos, neg)

    @classmethod
    def IS_CHANGED(cls, **kwargs): return float("NaN")
    @classmethod
    def VALIDATE_INPUTS(cls, select_yaml, select, prompt, unique_id): return True

_instance = None
def get_instance():
    global _instance
    if _instance is None: _instance = text_sum()
    return _instance

get_instance()