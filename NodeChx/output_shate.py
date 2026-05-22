import json
from pathlib import Path


class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False


ANY_TYPE = AnyType("*")


def get_node_class(source_class):
    class InstanceNode(source_class):
        def __init__(self):
            super().__init__()
            self.is_instance = True
            self.source_node_id = getattr(self, "source_node_id", None)

    return InstanceNode

def _iter_graph_nodes(graph):
    nodes = getattr(graph, "nodes", None)
    if isinstance(nodes, dict):
        return list(nodes.values())
    if isinstance(nodes, (list, tuple)):
        return list(nodes)
    return []

def _build_node_index(graph):
    index = {}
    for node in _iter_graph_nodes(graph):
        node_id = getattr(node, "id", None)
        if node_id is not None:
            index[node_id] = node
    return index

def _get_widgets(node):
    widgets = getattr(node, "widgets", None)
    if isinstance(widgets, dict):
        return widgets
    if isinstance(widgets, (list, tuple)):
        out = {}
        for widget in widgets:
            name = getattr(widget, "name", None)
            if name:
                out[name] = widget
        return out
    return {}

def sync_instance_values(graph):
    node_index = _build_node_index(graph)
    if not node_index:
        return
    for node in node_index.values():
        if not getattr(node, "is_instance", False):
            continue
        source_id = getattr(node, "source_node_id", None)
        if source_id is None:
            continue
        source_node = node_index.get(source_id)
        if source_node is None:
            continue
        source_widgets = _get_widgets(source_node)
        target_widgets = _get_widgets(node)
        for name, source_widget in source_widgets.items():
            target_widget = target_widgets.get(name)
            if target_widget is None:
                continue
            if not hasattr(source_widget, "value") or not hasattr(target_widget, "value"):
                continue
            target_widget.value = source_widget.value

_INPUT_SHARE_MAX_OUTPUTS = 20
_NODE_DEFS_ZH_CACHE = None


def _load_node_defs_zh():
    global _NODE_DEFS_ZH_CACHE
    if _NODE_DEFS_ZH_CACHE is not None:
        return _NODE_DEFS_ZH_CACHE
    node_defs_path = Path(__file__).resolve().parents[1] / "locales" / "zh" / "nodeDefs.json"
    try:
        with open(node_defs_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            _NODE_DEFS_ZH_CACHE = loaded if isinstance(loaded, dict) else {}
    except Exception:
        _NODE_DEFS_ZH_CACHE = {}
    return _NODE_DEFS_ZH_CACHE


def _resolve_zh_input_name(node_type, key):
    if not isinstance(node_type, str) or not isinstance(key, str):
        return str(key)
    node_defs = _load_node_defs_zh()
    node_def = node_defs.get(node_type, {})
    if not isinstance(node_def, dict):
        return key
    inputs = node_def.get("inputs", {})
    if not isinstance(inputs, dict):
        return key
    input_def = inputs.get(key, {})
    if not isinstance(input_def, dict):
        return key
    name = input_def.get("name")
    if isinstance(name, str) and len(name.strip()) > 0:
        return name
    return key


def _resolve_source_node_id(cur_node, links, fallback_id):
    inputs = cur_node.get("inputs", []) if isinstance(cur_node, dict) else []
    if isinstance(inputs, list):
        for input_info in inputs:
            if not isinstance(input_info, dict):
                continue
            link_id = input_info.get("link")
            if not link_id:
                continue
            link = next((item for item in links if isinstance(item, (list, tuple)) and len(item) >= 3 and item[0] == link_id), None)
            if link is not None:
                return link[1]
    return fallback_id


def _get_dynamic_values_from_prompt(prompt, source_node_data, source_node_id):
    prompt_data = prompt if isinstance(prompt, dict) else {}
    prompt_node = prompt_data.get(str(source_node_id), {})
    prompt_inputs = prompt_node.get("inputs", {}) if isinstance(prompt_node, dict) else {}
    node_type = str(source_node_data.get("type", ""))
    linked_input_names = {
        input_info.get("name")
        for input_info in source_node_data.get("inputs", [])
        if isinstance(input_info, dict) and input_info.get("name")
    }
    dynamic_items = []
    if isinstance(prompt_inputs, dict):
        for key, value in prompt_inputs.items():
            if key in linked_input_names:
                continue
            zh_key = _resolve_zh_input_name(node_type, str(key))
            dynamic_items.append((zh_key, value))
    return [item[1] for item in dynamic_items]

class InputShareNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_node": (ANY_TYPE, {}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    NAME = "Input Share Node"
    RETURN_TYPES = tuple([ANY_TYPE for _ in range(_INPUT_SHARE_MAX_OUTPUTS)])
    RETURN_NAMES = tuple([f"value_{i + 1}" for i in range(_INPUT_SHARE_MAX_OUTPUTS)])
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "Apt_Preset/stack/register"

    def run(self, source_node, unique_id, prompt, extra_pnginfo):
        workflow = extra_pnginfo.get("workflow", {}) if isinstance(extra_pnginfo, dict) else {}
        node_list = workflow.get("nodes", [])
        links = workflow.get("links", [])
        if not isinstance(node_list, list):
            node_list = []
        if not isinstance(links, list):
            links = []
        cur_node = next((n for n in node_list if str(n.get("id")) == unique_id), None)
        if not cur_node:
            return {"result": tuple([None] * _INPUT_SHARE_MAX_OUTPUTS)}

        source_node_id = getattr(self, "source_node_id", None)
        if not source_node_id:
            properties = getattr(self, "properties", {})
            if isinstance(properties, dict):
                source_node_id = properties.get("source_node_id")
        if not source_node_id and isinstance(cur_node, dict):
            node_props = cur_node.get("properties", {})
            if isinstance(node_props, dict):
                source_node_id = node_props.get("source_node_id")
        source_node_id = _resolve_source_node_id(cur_node, links, source_node_id)

        if not source_node_id:
            return {"result": tuple([None] * _INPUT_SHARE_MAX_OUTPUTS)}

        source_node_data = next((n for n in node_list if str(n.get("id")) == str(source_node_id)), None)
        if not source_node_data:
            return {"result": tuple([None] * _INPUT_SHARE_MAX_OUTPUTS)}

        dynamic_values = _get_dynamic_values_from_prompt(prompt, source_node_data, source_node_id)
        widget_values = source_node_data.get("widgets_values", [])
        if not isinstance(widget_values, list):
            widget_values = [widget_values]
        values = dynamic_values if len(dynamic_values) > 0 else widget_values
        values = values[:_INPUT_SHARE_MAX_OUTPUTS]
        padded_values = values + [None] * (_INPUT_SHARE_MAX_OUTPUTS - len(values))

        return {"result": tuple(padded_values)}

NODE_CLASS_MAPPINGS = {
    "InputShareNode": InputShareNode,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InputShareNode": "Input Share Node",
}
