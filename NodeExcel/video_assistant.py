

import os
import re
import csv


class excel_video_assistant:
    """
    视频助手节点 - 通过变量替换生成编剧模板
    从 CSV 文件加载编剧模板和各种配置选项，自动替换模板中的变量占位符
    """
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_PATH = os.path.join(BASE_DIR, "video", "video_assistant.csv")

    @staticmethod
    def load_csv_data(csv_path: str) -> dict:
        """加载 CSV 文件，返回分类到标题到内容的字典"""
        if not os.path.exists(csv_path):
            print(f"CSV 文件不存在：{csv_path}")
            return {}
        
        try:
            data = {}
            
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                reader = csv.reader(f)
                headers = next(reader, [])  # 读取表头
                
                # 确定列索引
                try:
                    idx_title = headers.index('标题')
                    idx_type = headers.index('类型')
                    idx_category = headers.index('分类')
                    idx_content = headers.index('内容')
                    idx_variables = headers.index('变量')
                except ValueError as e:
                    print(f"CSV 文件列名错误：{e}")
                    return {}
                
                for row in reader:
                    if len(row) > max(idx_title, idx_type, idx_category, idx_content):
                        item_type = str(row[idx_type]).strip() if idx_type < len(row) else ""
                        item_category = str(row[idx_category]).strip() if idx_category < len(row) else ""
                        title = str(row[idx_title]).strip() if idx_title < len(row) else ""
                        content = str(row[idx_content]) if idx_content < len(row) else ""
                        variables = str(row[idx_variables]) if idx_variables < len(row) else ""
                        
                        # 只处理"视频编辑"类型的数据
                        if item_type == "视频编辑" and title and item_category:
                            if item_category not in data:
                                data[item_category] = {}
                            data[item_category][title] = {
                                'content': content,
                                'variables': variables
                            }
            
            return data
        except Exception as e:
            print(f"加载 CSV 文件时出错：{e}")
            return {}

    def single_replace(self, text, target, replacement):
        """替换单个变量"""
        if not target or not replacement:
            return text
        target_clean = target.strip('"').strip()
        replacement_clean = replacement.strip('"').strip()
        return re.sub(re.escape(target_clean), replacement_clean, text)

    @classmethod
    def INPUT_TYPES(cls):
        # 加载 CSV 数据
        cls.video_data = cls.load_csv_data(cls.CSV_PATH)
        
        # 为各分类设置默认值
        if "编剧模板" not in cls.video_data:
            cls.video_data["编剧模板"] = {"默认模板": {'content': '', 'variables': ''}}
        if "视频主题" not in cls.video_data:
            cls.video_data["视频主题"] = {"默认主题": {'content': '普通场景', 'variables': ''}}
        if "镜头运镜" not in cls.video_data:
            cls.video_data["镜头运镜"] = {"默认运镜": {'content': '平稳镜头', 'variables': ''}}
        if "光影氛围" not in cls.video_data:
            cls.video_data["光影氛围"] = {"默认光影": {'content': '自然光', 'variables': ''}}
        if "视觉风格" not in cls.video_data:
            cls.video_data["视觉风格"] = {"默认风格": {'content': '写实风格', 'variables': ''}}
        if "音效" not in cls.video_data:
            cls.video_data["音效"] = {"默认音效": {'content': '背景音乐', 'variables': ''}}
        if "安全机制" not in cls.video_data:
            cls.video_data["安全机制"] = {"默认机制": {'content': '无特殊要求', 'variables': ''}}
        if "导演风格" not in cls.video_data:
            cls.video_data["导演风格"] = {"默认导演": {'content': '常规导演风格', 'variables': ''}}
        if "灵动变量" not in cls.video_data:
            cls.video_data["灵动变量"] = {"默认变量": {'content': '', 'variables': ''}}
        if "输出设置" not in cls.video_data:
            cls.video_data["输出设置"] = {"默认输出": {'content': '标准输出', 'variables': ''}}
        
        return {
            "required": {
                "script_template": (list(cls.video_data.get("编剧模板", {}).keys()), {"label": "编剧模板"}),
                "video_theme": (list(cls.video_data.get("视频主题", {}).keys()), {"label": "视频主题"}),
                "camera_movement": (list(cls.video_data.get("镜头运镜", {}).keys()), {"label": "镜头运镜"}),
                "lighting": (list(cls.video_data.get("光影氛围", {}).keys()), {"label": "光影氛围"}),
                "visual_style": (list(cls.video_data.get("视觉风格", {}).keys()), {"label": "视觉风格"}),
                "sound": (list(cls.video_data.get("音效", {}).keys()), {"label": "音效"}),
                "safety": (list(cls.video_data.get("安全机制", {}).keys()), {"label": "安全机制"}),
                "director": (list(cls.video_data.get("导演风格", {}).keys()), {"label": "导演风格"}),
                "output_settings": (list(cls.video_data.get("输出设置", {}).keys()), {"label": "输出设置"}),
            },
            "optional": {
                "dynamic_variable": (list(cls.video_data.get("灵动变量", {}).keys()), {"label": "灵动变量"}),
                "custom_template": ("STRING", {"default": "", "multiline": True, "placeholder": "输入自定义模板（留空则使用选中模板）"}),
                "video_theme_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义视频主题"}),
                "camera_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义镜头运镜"}),
                "lighting_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义光影氛围"}),
                "style_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义视觉风格"}),
                "sound_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义音效"}),
                "safety_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义安全机制"}),
                "director_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义导演风格"}),
                "output_custom": ("STRING", {"default": "", "multiline": False, "placeholder": "自定义输出设置"}),
                "dynamic_custom": ("STRING", {"default": "", "multiline": True, "placeholder": "自定义灵动变量"}),
            }
        }
        
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("script",)
    FUNCTION = "generate_script"
    CATEGORY = "Apt_Preset/prompt"

    DESCRIPTION = """
    视频助手 - 通过变量替换生成编剧模板
    
    功能说明：
    - 从 CSV 文件加载编剧模板和各种配置选项
    - 支持选择编剧模板、视频主题、镜头运镜、光影氛围等参数
    - 自动替换模板中的变量占位符
    - 支持自定义模板和各个参数的自定义内容
    
    变量占位符：
    {视频主题} - 视频主题内容
    {镜头运镜} - 镜头运镜方式
    {光影氛围} - 光影氛围描述
    {视觉风格} - 视觉风格设定
    {音效} - 音效描述
    {导演风格} - 导演风格描述
    {输出设置} - 输出设置参数
    {安全机制} - 安全机制设定
    {灵动变量} - 额外的灵动变量内容
    
    使用示例：
    1. 选择一个编剧模板（如：编剧 - 武术短片）
    2. 选择或自定义各项参数（视频主题、镜头运镜、光影氛围等）
    3. 节点会自动替换模板中的所有变量占位符
    4. 输出完整的编剧脚本
    
    自定义功能：
    - 可以在 custom_template 字段输入自定义模板
    - 每个参数都支持自定义值（如 video_theme_custom）
    - 自定义值优先于下拉框选择的值
    """

    def generate_script(self, script_template, video_theme, camera_movement, lighting, 
                        visual_style, sound, safety, director, output_settings,
                        dynamic_variable="默认变量", custom_template="",
                        video_theme_custom="", camera_custom="", lighting_custom="",
                        style_custom="", sound_custom="", safety_custom="",
                        director_custom="", output_custom="", dynamic_custom=""):
        
        # 获取各项内容，优先使用自定义值
        theme_content = video_theme_custom.strip() if video_theme_custom.strip() else \
                        self.video_data.get("视频主题", {}).get(video_theme, {}).get('content', '')
        
        camera_content = camera_custom.strip() if camera_custom.strip() else \
                         self.video_data.get("镜头运镜", {}).get(camera_movement, {}).get('content', '')
        
        lighting_content = lighting_custom.strip() if lighting_custom.strip() else \
                           self.video_data.get("光影氛围", {}).get(lighting, {}).get('content', '')
        
        style_content = style_custom.strip() if style_custom.strip() else \
                        self.video_data.get("视觉风格", {}).get(visual_style, {}).get('content', '')
        
        sound_content = sound_custom.strip() if sound_custom.strip() else \
                        self.video_data.get("音效", {}).get(sound, {}).get('content', '')
        
        safety_content = safety_custom.strip() if safety_custom.strip() else \
                         self.video_data.get("安全机制", {}).get(safety, {}).get('content', '')
        
        director_content = director_custom.strip() if director_custom.strip() else \
                           self.video_data.get("导演风格", {}).get(director, {}).get('content', '')
        
        output_content = output_custom.strip() if output_custom.strip() else \
                         self.video_data.get("输出设置", {}).get(output_settings, {}).get('content', '')
        
        dynamic_content = dynamic_custom.strip() if dynamic_custom.strip() else \
                          self.video_data.get("灵动变量", {}).get(dynamic_variable, {}).get('content', '')
        
        # 获取模板内容
        if custom_template.strip():
            # 使用自定义模板
            template_content = custom_template.strip()
        else:
            # 使用选中的模板
            template_content = self.video_data.get("编剧模板", {}).get(script_template, {}).get('content', '')
            if not template_content:
                template_content = "❌ 未找到模板内容"
        
        # 如果模板内容为空，使用默认模板
        if not template_content or template_content == "❌ 未找到模板内容":
            template_content = "请选择有效的编剧模板或输入自定义模板。"
        
        # 执行变量替换
        replacements = {
            "{视频主题}": theme_content,
            "{镜头运镜}": camera_content,
            "{光影氛围}": lighting_content,
            "{视觉风格}": style_content,
            "{音效}": sound_content,
            "{导演风格}": director_content,
            "{输出设置}": output_content,
            "{安全机制}": safety_content,
            "{灵动变量}": dynamic_content
        }
        
        # 依次替换所有变量
        result = template_content
        for placeholder, value in replacements.items():
            if value:  # 只替换有内容的变量
                result = self.single_replace(result, placeholder, value)
        
        return (result,)



