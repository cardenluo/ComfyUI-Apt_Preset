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

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
