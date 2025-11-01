
import sys
import os
from collections.abc import Iterable
import folder_paths
import comfy
from ..main_unit import *


defaultValue = """0:(0),
30:(1),
60:(0),
90:(0),
120:(0)
"""


#region---------------------------从外部导入模块，加载检查-------------------------------

#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Apt_Preset\NodeChx\AdvancedCN.py"
#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Advanced-ControlNet"
#"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Advanced-ControlNet\adv_control\nodes_deprecated.py"





current_dir = os.path.dirname(os.path.abspath(__file__))
custom_nodes_dir = os.path.dirname(os.path.dirname(current_dir))
target_dir = os.path.join(custom_nodes_dir, 'ComfyUI-Advanced-ControlNet')
#target_dir = r"F:\ComfyUI-aki-v1.6\ComfyUI\custom_nodes\ComfyUI-Advanced-ControlNet"


# 初始化可能从 ComfyUI-Advanced-ControlNet 导入的模块和类
AdvancedControlNetApply = None
TimestepKeyframeGroup = None
ControlWeights = None
LatentKeyframeGroup = None
LatentKeyframe = None


if not os.path.exists(target_dir):
    pass
else:
    sys.path.append(target_dir)
    try:
        from adv_control.utils import (
            TimestepKeyframeGroup,
            ControlWeights,
            LatentKeyframeGroup,
            LatentKeyframe
            
        )
    except ImportError as e:
        pass
    try:
        from adv_control.nodes_main import AdvancedControlNetApply
    except ImportError:
        pass
    try:
        from adv_control.nodes_deprecated import ScaledSoftUniversalWeightsDeprecated
    except ImportError:
        pass




# 定义一个函数用于检查插件是否安装
def check_advanced_controlnet_installed():
    if AdvancedControlNetApply is None:
        raise RuntimeError(" Please install ComfyUI-Advanced-ControlNet  before using this function.。")


#endregion---------------------------加载检查-------------------------------



class AD_sch_adv_CN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "conditioning": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "mask_optional": ("MASK", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                #"timestep_kf": ("TIMESTEP_KEYFRAME", ),
                #"weights_override": ("CONTROL_NET_WEIGHTS", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    DEPRECATED = True
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("CONDITIONING",)
    FUNCTION = "apply_controlnet"

    CATEGORY = "Apt_Preset/AD/😺backup"

    def apply_controlnet(self, conditioning, control_net, image, strength, 
                        mask_optional=None, model_optional=None, vae_optional=None,
                        timestep_kf: TimestepKeyframeGroup=None, latent_kf_override=None,
                        weights_override: ControlWeights=None):
        check_advanced_controlnet_installed()
        values = AdvancedControlNetApply.apply_controlnet(self, positive=conditioning, negative=None, control_net=control_net, image=image,
                                                        strength=strength, start_percent=0, end_percent=1,
                                                        mask_optional=mask_optional, vae_optional=vae_optional,
                                                        timestep_kf=timestep_kf, latent_kf_override=latent_kf_override, weights_override=weights_override,
                                                        control_apply_to_uncond=True)
        return (values[0], model_optional)



class AD_sch_latent:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": DefaultValue}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 999999, "step": 1}),
                "easing_type": (list(easing_functions.keys()), {"default": "Linear"}),
            },
            "optional": {
                "prev_latent_kf": ("LATENT_KEYFRAME", ),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }


    RETURN_TYPES = ("LATENT_KEYFRAME","IMAGE" )
    RETURN_NAMES = ("LATENT_KF", "graph", )
    FUNCTION = "load_keyframe"
    CATEGORY = "Apt_Preset/AD/😺backup"

    def load_keyframe(self, text, max_frames, easing_type=None,
                    prev_latent_kf: LatentKeyframeGroup = None,
                    prev_latent_keyframe: LatentKeyframeGroup = None,  # old name
                    ):
        check_advanced_controlnet_installed()
        print_output = None

        # 获取动画数据和绘图所需数据
        t, _, values_seq, frame_methods = self.animate(text, max_frames, easing_type=easing_type)
        float_strengths = t

        # 生成图像
        curve_img = generate_value_curve_image_with_data(values_seq, max_frames, frame_methods)

        # 处理 latent keyframe
        prev_latent_keyframe = prev_latent_keyframe if prev_latent_keyframe else prev_latent_kf
        if not prev_latent_keyframe:
            prev_latent_keyframe = LatentKeyframeGroup()
        else:
            prev_latent_keyframe = prev_latent_keyframe.clone()
        curr_latent_keyframe = LatentKeyframeGroup()

        if type(float_strengths) in (float, int):
            logger.info("No batched float_strengths passed into Latent Keyframe Batch Group node; will not create any new keyframes.")
        elif isinstance(float_strengths, Iterable):
            for idx, strength in enumerate(float_strengths):
                keyframe = LatentKeyframe(idx, strength)
                curr_latent_keyframe.add(keyframe)
        else:
            raise ValueError(f"Expected strengths to be an iterable input, but was {type(float_strengths).__repr__}.")

        # 合并已有的 keyframes
        for latent_keyframe in prev_latent_keyframe.keyframes:
            curr_latent_keyframe.add(latent_keyframe)

        return (curr_latent_keyframe, curve_img)
    def animate(self, text, max_frames, easing_type="Linear"):
        keyframes = []
        lines = text.strip().split('\n')
        for line in lines:
            if ':' not in line:
                continue
            idx_str, val_str = map(str.strip, line.split(':', 1))
            try:
                idx = int(idx_str)
                # 提取 value 和 interp_method（支持 @xxx@ 格式）
                val_match = re.match(r'^$$(.*?)$$$', val_str.strip())  # 支持括号包裹的数值
                val_source = val_match.group(1) if val_match else val_str

                # 解析插值方法
                interp_method = easing_type
                match = re.search(r'@([a-zA-Z0-9_ ]+)@', val_source)
                if match:
                    custom_ease = match.group(1).strip()
                    if custom_ease in easing_functions:
                        interp_method = custom_ease
                    val_source = re.sub(r'\s*@([a-zA-Z0-9_ ]+)@\s*', ' ', val_source).strip()

                val = float(val_source)
                keyframes.append(ValueKeyframe(index=idx, value=val, interp_method=interp_method))
            except (ValueError, IndexError):
                continue

        if not keyframes:
            t = torch.zeros(max_frames)
            values_seq = [0.0] * max_frames
            frame_methods = []
            return (t, list(map(int, t)), values_seq, frame_methods)

        values_seq = [None] * max_frames
        frame_methods = []

        # 插值处理
        for i in range(len(keyframes)):
            curr = keyframes[i]
            if curr.index >= max_frames:
                continue
            values_seq[curr.index] = curr.value

            if i + 1 < len(keyframes):
                next_kf = keyframes[i + 1]
                diff_len = next_kf.index - curr.index
                weights = torch.linspace(0, 1, diff_len + 1)[1:-1]
                easing_weights = [apply_easing(w.item(), curr.interp_method) for w in weights]

                for j, w in enumerate(easing_weights):
                    idx = curr.index + j + 1
                    if idx >= max_frames:
                        break
                    values_seq[idx] = curr.value * (1.0 - w) + next_kf.value * w

                # 记录插值区间
                frame_methods.append((curr.index, next_kf.index, curr.interp_method))

        # 填充缺失值
        last_val = None
        for i in range(max_frames):
            if values_seq[i] is not None:
                last_val = values_seq[i]
            elif last_val is not None:
                values_seq[i] = last_val

        # 构建输出
        t = torch.tensor(values_seq, dtype=torch.float32)
        values_seq = [v if v is not None else 0.0 for v in values_seq]

        return (t, list(map(int, t)), values_seq, frame_methods)





class AD_ScaledSoft:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_multiplier": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 1.0, "step": 0.001}, ),
            },
            "optional": {
                "uncond_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}, ),
                "cn_extras": ("CN_WEIGHTS_EXTRAS",),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }
    
    RETURN_TYPES = ("CONTROL_NET_WEIGHTS",)
    RETURN_NAMES = ("CN_WEIGHTS", )
    FUNCTION = "load_weights"
    CATEGORY = ""

    def load_weights(self, base_multiplier,  uncond_multiplier: float=1.0, cn_extras: dict[str]={}):
        weights = ControlWeights.universal(base_multiplier=base_multiplier, uncond_multiplier=uncond_multiplier, extras=cn_extras)
        return (weights, )






class Stack_adv_CN:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "controlnet": (folder_paths.get_filename_list("controlnet"),),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),

            },
            "optional": {
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 10.0, "step": 0.01}),
                "base_multiplier": ("FLOAT", {"default": 0.825, "min": 0.0, "max": 1.0, "step": 0.001}),
                "image": ("IMAGE", ),
                "mask_optional": ("MASK", ),
                "latent_kf_override": ("LATENT_KEYFRAME", ),
                "cn_stack": ("CN_STACK",),
            },
            "hidden": {
                "autosize": ("ACNAUTOSIZE", {"padding": 0}),
            }
        }

    RETURN_TYPES = ("CN_STACK","CONTROL_NET",)
    RETURN_NAMES = ("cn_stack","controlnet")
    FUNCTION = "controlnet_stacker"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def controlnet_stacker(self, controlnet, image, strength, start_percent, end_percent, base_multiplier,
                          cn_stack=None, mask_optional=None, latent_kf_override=None):
        check_advanced_controlnet_installed()
        controlnet_list = []

        if cn_stack is not None:
            controlnet_list.extend([cn for cn in cn_stack if cn[0] != "None"])

        if controlnet != "None" and image is not None:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
            
            # 将base_multiplier添加到控制网列表中
            controlnet_list.append(( controlnet, image, strength, mask_optional, 
                                   latent_kf_override, start_percent, end_percent, base_multiplier ))

        return (controlnet_list, controlnet)



class Apply_adv_CN:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "positive": ("CONDITIONING", ),
            },
            "optional": {
                "cn_stack": ("CN_STACK",),
            },


        }

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply_controlnet"
    CATEGORY = "Apt_Preset/stack"

    def apply_controlnet(self, positive, cn_stack=None):
        check_advanced_controlnet_installed()
        if cn_stack is not None:
            for cn in cn_stack:
                controlnet, image, strength, mask_optional, latent_kf_override, start_percent, end_percent, base_multiplier = cn             
                weights = ControlWeights.universal(base_multiplier=base_multiplier, uncond_multiplier=1.0, extras={})
                positive = AdvancedControlNetApply.apply_controlnet(
                    self, 
                    positive=positive, 
                    negative=None, 
                    control_net=controlnet, 
                    image=image,
                    mask_optional=mask_optional, 
                    strength=strength, 
                    # 使用从stack传递过来的参数
                    start_percent=start_percent, 
                    end_percent=end_percent,
                    timestep_kf=None,  
                    latent_kf_override=latent_kf_override, 
                    weights_override=weights,
                    control_apply_to_uncond=True
                )[0]
        return (positive,)






































