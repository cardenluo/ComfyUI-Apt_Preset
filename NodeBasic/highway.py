class AnyType(str):
    def __eq__(self, _) -> bool: return True
    def __ne__(self, __value: object) -> bool: return False
ANY_TYPE = AnyType("*")

class RevisionDict:
    def __init__(self, *args, **kwargs): self._data = dict(*args, **kwargs)
    def __getitem__(self, key): return self._data.get(key, None)
    def __setitem__(self, key, value): self._data[key] = value
    def get(self, key, default=None): return self._data.get(key, default)
    def path_iter(self, prefix):
        for k in self._data:
            if isinstance(k, tuple) and k[:len(prefix)] == prefix:
                yield k
    def path_count(self, prefix): return sum(1 for _ in self.path_iter(prefix))



class Data_Highway:
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_load"
    # 设置较大默认输出数量
    MAX_OUTPUTS = 20
    RETURN_TYPES = ("RUN_CONTEXT2", *[ANY_TYPE] * MAX_OUTPUTS)
    RETURN_NAMES = ("bus", *[f"output_{i}" for i in range(MAX_OUTPUTS)])
    DESCRIPTION = """
    -添加新端口 add new ports math: >H1>H2
    -点击更新 Then Click to update port
    -必须是同名端口，才能连接
    -It must be the same-named port to connect, 
    """


    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"port_config": ("STRING", {"multiline": False, "default": ""})},
            "optional": {"bus": ("RUN_CONTEXT2", {"default": None})}
        }

    def __init__(self):
        self.port_config = ""
        self.input_ports = []
        self.output_ports = []

    def process(self, port_config, bus=None, **kwargs):
        sections = port_config.split(";")
        input_section = sections[0].strip()
        output_section = sections[1].strip() if len(sections) > 1 else ""

        input_names = [p.strip() for p in input_section.split(">") if p.strip() and p.strip() != "bus"]
        output_names = [p.strip().lstrip("<") for p in output_section.split("<") if p.strip() and p.strip() != "bus"] or input_names.copy()

        packed = RevisionDict()
        packed[("kind")] = "Data_Highway"
        packed[("bus")] = bus

        if bus is not None and isinstance(bus, RevisionDict):
            for name in input_names:
                if ("data", name) in bus._data:
                    if name not in kwargs or kwargs[name] is None:
                        kwargs[name] = bus._data[("data", name)]

        for name in input_names:
            if name in kwargs:
                packed[("data", name)] = kwargs[name]
                packed[("type", name)] = type(kwargs[name]).__name__

        outputs = []
        for name in output_names:
            if name in kwargs:
                outputs.append(kwargs[name])
            elif ("data", name) in packed._data:
                outputs.append(packed._data[("data", name)])
            else:
                outputs.append(None)

        # 补齐输出到 MAX_OUTPUTS 数量
        outputs += [None] * (self.MAX_OUTPUTS - len(outputs))
        return (packed, *outputs)

    def _data(self, rev_dict):
        return rev_dict._data if hasattr(rev_dict, "_data") else rev_dict

    def parse_ports(self, config):
        sections = config.split(';')
        self.input_ports = []
        self.output_ports = []

        if sections[0].strip():
            self.input_ports = [p.strip() for p in sections[0].split('>') if p.strip() and p.strip() != "bus"]

        if len(sections) > 1 and sections[1].strip():
            self.output_ports = [p.strip().lstrip('<') for p in sections[1].split('<') if p.strip() and p.strip() != "bus"]
        else:
            self.output_ports = self.input_ports.copy()