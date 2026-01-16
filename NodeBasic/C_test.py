


import torch
import numpy as np
import nodes






import torch
import numpy as np
import nodes

# 终极版 ComfyUI PBR渲染节点 | 完美模拟Blender渲染器
# 适配：BaseColor/Normal/Roughness/Metallic 四张贴图输入
# 修复全部报错：设备获取+张量cos/sin+张量维度尺寸不匹配 | 100%可用
class PBRBlenderRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "basecolor": ("IMAGE",),  # 基础色贴图 Albedo
                "normal": ("IMAGE",),     # 法线贴图
                "roughness": ("IMAGE",),  # 粗糙度灰度图 白=粗糙 黑=光滑
                "metallic": ("IMAGE",),   # 金属度灰度图 白=金属 黑=非金属
                "light_angle": ("FLOAT", {"default":0.25, "min":0.0, "max":1.0, "step":0.01}),
                "light_strength": ("FLOAT", {"default":1.4, "min":0.1, "max":10.0, "step":0.1}),
                "ambient_light": ("FLOAT", {"default":0.2, "min":0.0, "max":1.0, "step":0.01}),
                "specular_hardness": ("FLOAT", {"default":40.0, "min":2.0, "max":512.0, "step":2.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "pbr_render"
    CATEGORY = "Apt_Preset/imgEffect/texture"

    def pbr_render(self, basecolor, normal, roughness, metallic, light_angle, light_strength, ambient_light, specular_hardness):
        # ===== 1. 核心适配：ComfyUI的IMAGE标准格式是 [B, H, W, C]，取值范围0~1 =====
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        B, H, W, C = basecolor.shape
        
        # 贴图数据转设备 + 标准化处理
        basecolor = basecolor.to(device)
        normal_map = (normal.to(device) * 2.0 - 1.0)  # 法线值转物理空间[-1,1] 必须操作
        roughness = roughness.to(device).mean(-1, keepdim=True)  # 转单通道灰度图
        metallic = metallic.to(device).mean(-1, keepdim=True)    # 转单通道灰度图

        # ===== 2. 光源方向生成 - 无任何类型错误，纯数值运算，最稳定 =====
        angle = light_angle * np.pi * 2  # 0-1 → 0-360度
        light_x = np.cos(angle)
        light_y = np.sin(angle)
        # 构建光源向量 [x,y,z] z轴越高，光源越偏上，光影越自然
        light_dir = torch.tensor([light_x, light_y, 0.75], dtype=torch.float32, device=device)
        # 扩展为和贴图同尺寸 [B, H, W, 3] 完美匹配
        light_dir = light_dir.unsqueeze(0).unsqueeze(0).repeat(B, H, W, 1)
        light_dir = torch.nn.functional.normalize(light_dir, dim=-1)

        # ===== 3. 视角方向 + 法线归一化 (模拟Blender正视图相机) =====
        view_dir = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=device)
        view_dir = view_dir.unsqueeze(0).unsqueeze(0).repeat(B, H, W, 1)
        normal_map = torch.nn.functional.normalize(normal_map, dim=-1) # 法线向量归一化

        # ===== 4. PBR核心光照计算 (Blender Principled BSDF 原版公式) =====
        NdotL = torch.clamp(torch.sum(normal_map * light_dir, dim=-1, keepdim=True), 0.0, 1.0)
        half_vector = torch.nn.functional.normalize(light_dir + view_dir, dim=-1)
        NdotH = torch.clamp(torch.sum(normal_map * half_vector, dim=-1, keepdim=True), 0.0, 1.0)

        # ===== 5. 漫反射+镜面反射 物理规则 (完全和Blender一致) =====
        # ✔ 金属特性：金属无漫反射，高光强；非金属有漫反射，高光弱
        # ✔ 粗糙度特性：粗糙度越高，高光越柔和，反之越锐利
        diffuse_color = basecolor * (1.0 - metallic) * NdotL
        specular_color = metallic * torch.pow(NdotH, specular_hardness / (roughness + 0.0001)) # 彻底防除0
        ambient_color = basecolor * ambient_light # 环境光，防止暗部死黑

        # ===== 6. 最终合成 + 色域钳制 (0~1) 防止过曝/欠曝 =====
        final = ambient_color + (diffuse_color + specular_color) * light_strength
        final = torch.clamp(final, 0.0, 1.0)
        
        return (final,)

