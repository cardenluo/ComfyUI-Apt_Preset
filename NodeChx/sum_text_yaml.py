from ..main_unit import *
import os, json, asyncio, shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tempfile
from server import PromptServer
from aiohttp import web

#---------------------安全导入------
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileSystemEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = object
    WATCHDOG_AVAILABLE = False

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False
#---------------------安全导入------













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
            },
            "hidden": {"unique_id": "UNIQUE_ID"}
        }

    RETURN_TYPES = ("STRING","STRING",)
    RETURN_NAMES = ("pos", "neg",)
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/prompt"
    NAME="text_sum"


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
        if WATCHDOG_AVAILABLE and self.observer is None:
            self.observer = Observer()
            self.observer.schedule(self.file_handler, str(self.yaml_dir), recursive=False)
            self.observer.start()
        elif not WATCHDOG_AVAILABLE:
            print("Warning: watchdog not available, file monitoring disabled")

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

    def run(self, negative="", select_yaml: str = "", select: str = "", prompt: str = "", unique_id: str = "") -> Tuple[str]:
        if len(prompt) > 4096: print(f"Warning: Prompt length ({len(prompt)}) exceeds recommended 4096 chars")
        if isinstance(prompt, tuple): prompt = ", ".join(str(x) for x in prompt if x is not None)
        elif not isinstance(prompt, str): prompt = str(prompt)
        if isinstance(negative, tuple): negative = ", ".join(str(x) for x in negative if x is not None)
        elif not isinstance(negative, str): negative = str(negative)

        pos= prompt
        neg = negative
        return (pos, neg)

    #@classmethod
    #def IS_CHANGED(cls, **kwargs): return float("NaN")
    @classmethod
    def VALIDATE_INPUTS(cls, select_yaml, select, prompt, unique_id): return True


class text_sum_edit:
    CUSTOM_YAML = "custom_yaml"

    @classmethod
    def INPUT_TYPES(cls):
        yaml_dir = Path(__file__).parent / "yaml"
        yaml_dir.mkdir(exist_ok=True)
        yaml_files = sorted(path.name for path in yaml_dir.glob("*.yaml"))
        return {
            "required": {
                "select_yaml": (yaml_files + [cls.CUSTOM_YAML], {
                    "default": yaml_files[0] if yaml_files else cls.CUSTOM_YAML,
                }),
                "custom_yaml": ("STRING", {"default": "", "multiline": False}),
                "edit_item": ("STRING", {"default": "", "multiline": False}),
                "edit_mode": ("BOOLEAN", {
                    "default": True,
                    "label_on": "save",
                    "label_off": "delete",
                }),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "negative": ("STRING", {"default": "", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "edit"
    CATEGORY = "Apt_Preset/prompt"
    OUTPUT_NODE = True

    @classmethod
    def _target_path(cls, select_yaml, custom_yaml):
        yaml_dir = (Path(__file__).parent / "yaml").resolve()
        if select_yaml == cls.CUSTOM_YAML:
            filename = str(custom_yaml or "").strip()
            if not filename:
                raise ValueError("custom_yaml needs a file name")
            if not filename.lower().endswith(".yaml"):
                filename += ".yaml"
        else:
            filename = str(select_yaml or "").strip()
            existing = {path.name for path in yaml_dir.glob("*.yaml")}
            if filename not in existing:
                raise ValueError("select_yaml must be an existing YAML file or custom_yaml")

        if Path(filename).name != filename or Path(filename).suffix.lower() != ".yaml":
            raise ValueError("YAML file name must be a local .yaml file name")
        target = (yaml_dir / filename).resolve()
        if target.parent != yaml_dir:
            raise ValueError("YAML file must stay inside the text_sum yaml directory")
        return target

    async def edit(self, select_yaml, custom_yaml, edit_item, edit_mode, prompt, negative):
        item = str(edit_item or "").strip()
        if not item:
            raise ValueError("edit_item needs a name")
        if not isinstance(edit_mode, bool):
            raise ValueError("edit_mode must be save or delete")
        target = self._target_path(select_yaml, custom_yaml)
        if target.exists():
            with open(target, "r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
            if not isinstance(data, dict):
                raise ValueError(f"{target.name} must contain a YAML mapping")
        else:
            data = {}
        if not edit_mode:
            if item not in data:
                raise ValueError(f"{item} does not exist in {target.name}")
            del data[item]
            action = "Deleted"
        else:
            data[item] = {
                "prompt": str(prompt or ""),
                "negative": str(negative or ""),
            }
            action = "Saved"
        await get_instance()._save_yaml(target.name, data)
        get_instance().refresh_enums()
        return (f"{action} {item} in {target.name}",)

_instance = None
def get_instance():
    global _instance
    if _instance is None: _instance = text_sum()
    return _instance

get_instance()
