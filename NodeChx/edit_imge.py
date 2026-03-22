
#region-------------------------------import-----------------------
import folder_paths
import torch
import logging
import math
import comfy
import node_helpers
import comfy.utils
from nodes import common_ksampler, CLIPTextEncode
import torch.nn.functional as F





from ..office_unit import *
from .main_stack import Apply_LoRAStack,Apply_CN_union,Apply_latent,Apply_Redux
from ..main_unit import *



#---------------------安全导入------
try:
    from comfy_extras.nodes_model_patch import ModelPatchLoader, QwenImageDiffsynthControlnet
    REMOVER_AVAILABLE = True  
except ImportError:
    ModelPatchLoader = None
    QwenImageDiffsynthControlnet = None
    REMOVER_AVAILABLE = False  


try:
    from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow
    REMOVER_AVAILABLE = True  
except ImportError:
    ModelSamplingAuraFlow= None
    REMOVER_AVAILABLE = False  


#endregion-----------------------------import----------------------------





#region---------------------kontext------------------






class pre_advanced_condi_merge:
    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "main_cond": ("CONDITIONING", ),
                "mode": (["combine", "average", "concat", "attention", "cross_attention", "adaptive", "modal_sep", "adversarial"],
                         {"default": "attention"} ),
            },
            "optional": {
                "aux_cond": ("CONDITIONING", ),
                "main_cond_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001}),  # 前期主条件占比（结构）
                "aux_mask": ("MASK", ),
                "mask_area": (["default", "bounds"], {"default": "default"}),
                "attention_temp": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "cross_weight": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # cross_attention：模态对齐强度
                "adaptive_weight": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),  # adaptive：后期辅条件占比（细节）
                "adv_weight": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
        return inputs
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool/conditioning"
        
    DESCRIPTION = """
            main_cond：主控条件，可包括多个CN（controlnet）等控制核心结构（侧重结构）
            aux_cond：辅控条件（可选），适配主控维度（侧重细节/风格）
                    
            concat 连接：所有参数不可控制
            相同特征替换，不同则追加（若有CN,则集中在控制区呈混合结构)    

            combine合并: 所有参数可控，主条件0~1分配占比 (主强则辅弱,0.5平均)
            主控特征+辅助特征，两特征并存（重叠区域或构成混合：蓝衣服+红衣服=紫衣服)  
            
            average平均: 主条件占比可控, 0~1分配强度 (主强则辅弱,0.5平均)
            相同特征按权重分配，不同则追加（若有CN则呈混合效果)  

            attention动态注意力融合：main_cond_ratio 为基础偏向
            自动学习特征重要性（注意力动态分配权重）局部特征精准交互
            attention_temp（注意力集中程度，0.1~10.0）

            cross_attention 跨条件交叉注意力：主条件占比-控制原始特征占比
            模态对齐（如文本+CN/图像+风格）,解决不同模态条件错位问题，生成更连贯
            cross_weight（模态对齐强度，0.0~1.0）→ 越大对齐越紧密
            
            adaptive 时序自适应融合：分阶段精准控制
            前期（0~50%生成步骤）：侧重结构，主条件占比（默认0.9，强结构）
            后期（50~100%生成步骤）：侧重细节，adaptive_weight辅条件占比（默认0.7，强细节）
            
            modal_sep 模态分离融合：主条件占比-为内容占比
            分离内容（主成分）/风格（次要成分）分别融合,避免内容与风格冲突
            
            adversarial 对抗性融合：主条件占比-为无冲突区域基础权重
            抑制冲突特征（如颜色/风格冲突）,冲突条件下生成更协调，避免混乱结果
            adv_weight（冲突抑制强度，0.0~1.0）
            """

    def ConditioningCombine(self, conditioning1, conditioning2):
        return (conditioning1 + conditioning2, )

    def ConditioningAverage(self, conditioning_to, conditioning_from, conditioning_to_strength):
        out = []
        if len(conditioning_from) > 1:
            logging.warning("ConditioningAverage: Only first aux_cond is used.")
        cond_from = conditioning_from[0][0]
        pooled_output_from = conditioning_from[0][1].get("pooled_output", None)
        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2])).to(t1.device)], dim=1)
            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            t_to = conditioning_to[i][1].copy()
            if pooled_output_from is not None and pooled_output_to is not None:
                t_to["pooled_output"] = torch.mul(pooled_output_to, conditioning_to_strength) + torch.mul(pooled_output_from, (1.0 - conditioning_to_strength))
            elif pooled_output_from is not None:
                t_to["pooled_output"] = pooled_output_from
            out.append([tw, t_to])
        return (out, )

    def ConditioningConcat(self, conditioning_to, conditioning_from):
        out = []
        if len(conditioning_from) > 1:
            logging.warning("ConditioningConcat: Only first aux_cond is used.")
        cond_from = conditioning_from[0][0]
        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            tw = torch.cat((t1, cond_from), 1)
            out.append([tw, conditioning_to[i][1].copy()])
        return (out, )

    def ConditioningSetAreaStrength(self, conditioning, strength):
        return (node_helpers.conditioning_set_values(conditioning, {"strength": strength}), )

    def ConditioningSetMask(self, conditioning, mask, set_cond_area, strength):
        set_area_to_bounds = set_cond_area != "default"
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
        return (node_helpers.conditioning_set_values(conditioning, {
            "mask": mask, "set_area_to_bounds": set_area_to_bounds, "mask_strength": strength
        }), )

    def ConditioningSetTimestepRange(self, conditioning, start, end):
        return (node_helpers.conditioning_set_values(conditioning, {"start_percent": start, "end_percent": end}), )

    def ConditioningAttentionFusion(self, main_cond, aux_cond, temp=1.0, main_cond_ratio=0.7):
        out = []
        main_feat = main_cond[0][0]
        aux_feat = aux_cond[0][0]
        pooled_main = main_cond[0][1].get("pooled_output")
        pooled_aux = aux_cond[0][1].get("pooled_output")

        min_seq_len = min(main_feat.shape[1], aux_feat.shape[1])
        main_feat = main_feat[:, :min_seq_len, :]
        aux_feat = aux_feat[:, :min_seq_len, :]

        similarity = torch.matmul(main_feat, aux_feat.transpose(1, 2))
        main_weight = F.softmax(similarity / temp, dim=-1)
        aux_weight = F.softmax(similarity.transpose(1, 2) / temp, dim=-1)

        main_attended = torch.matmul(main_weight, aux_feat)
        aux_attended = torch.matmul(aux_weight, main_feat)
        fused_feat = (
            main_feat * main_cond_ratio +
            main_attended * main_cond_ratio * 0.5 +
            aux_feat * (1 - main_cond_ratio) +
            aux_attended * (1 - main_cond_ratio) * 0.5
        )

        for i in range(len(main_cond)):
            t_to = main_cond[i][1].copy()
            if pooled_main is not None and pooled_aux is not None:
                t_to["pooled_output"] = pooled_main * main_cond_ratio + pooled_aux * (1 - main_cond_ratio)
            elif pooled_main is not None:
                t_to["pooled_output"] = pooled_main
            elif pooled_aux is not None:
                t_to["pooled_output"] = pooled_aux
            out.append([fused_feat, t_to])
        return (out, )

    def ConditioningCrossAttention(self, main_cond, aux_cond, main_cond_ratio=0.7, cross_weight=0.5):
        out = []
        main_feat = main_cond[0][0]
        aux_feat = aux_cond[0][0]
        dim = main_feat.shape[-1]
        pooled_main = main_cond[0][1].get("pooled_output")
        pooled_aux = aux_cond[0][1].get("pooled_output")

        min_seq_len = min(main_feat.shape[1], aux_feat.shape[1])
        main_feat = main_feat[:, :min_seq_len, :]
        aux_feat = aux_feat[:, :min_seq_len, :]

        def cross_attend(q, k, v):
            q = q / torch.sqrt(torch.tensor(dim, dtype=q.dtype).to(q.device))
            attn = torch.matmul(q, k.transpose(1, 2))
            attn = F.softmax(attn, dim=-1)
            return torch.matmul(attn, v)

        main_fused = cross_attend(main_feat, aux_feat, aux_feat)
        aux_fused = cross_attend(aux_feat, main_feat, main_feat)

        main_bias = main_cond_ratio
        aux_bias = 1 - main_cond_ratio
        fused_feat = (
            main_feat * main_bias +
            aux_feat * aux_bias +
            (main_fused + aux_fused) * cross_weight
        ) / (1 + cross_weight)

        for i in range(len(main_cond)):
            t_to = main_cond[i][1].copy()
            if pooled_main is not None and pooled_aux is not None:
                t_to["pooled_output"] = pooled_main * main_bias + pooled_aux * aux_bias
            elif pooled_main is not None:
                t_to["pooled_output"] = pooled_main
            elif pooled_aux is not None:
                t_to["pooled_output"] = pooled_aux
            out.append([fused_feat, t_to])
        return (out, )

    # 核心修复：重新设计adaptive融合逻辑
    def ConditioningAdaptiveFusion(self, main_cond, aux_cond, main_cond_ratio=0.9, adaptive_weight=0.7):
        out = []
        main_feat = main_cond[0][0]
        aux_feat = aux_cond[0][0]
        pooled_main = main_cond[0][1].get("pooled_output")
        pooled_aux = aux_cond[0][1].get("pooled_output")

        # 确保两个条件的序列长度一致
        min_seq_len = min(main_feat.shape[1], aux_feat.shape[1])
        main_feat = main_feat[:, :min_seq_len, :]
        aux_feat = aux_feat[:, :min_seq_len, :]

        # 时序自适应权重函数：明确参数职责，线性过渡
        def adaptive_weight_fn(timestep_percent):
            """
            timestep_percent：生成步骤占比（0.0=开始，1.0=结束）
            - 0.0~0.5（前期）：主条件权重从 main_cond_ratio 线性降到 0.5
            - 0.5~1.0（后期）：辅条件权重从 0.5 线性升到 adaptive_weight
            主条件权重 = 1 - 辅条件权重，确保权重和为1
            """
            if timestep_percent < 0.5:
                # 前期：主条件主导（结构），逐步降低
                main_weight = main_cond_ratio - (main_cond_ratio - 0.5) * (timestep_percent / 0.5)
            else:
                # 后期：辅条件主导（细节），逐步提升
                main_weight = 0.5 - (0.5 - (1 - adaptive_weight)) * ((timestep_percent - 0.5) / 0.5)
            
            # 辅条件权重 = 1 - 主条件权重，确保权重和为1（避免特征稀释）
            aux_weight = 1.0 - main_weight
            return main_weight, aux_weight

        # 初始融合（使用前期初始权重）
        initial_main_weight, initial_aux_weight = adaptive_weight_fn(0.0)
        initial_fused = main_feat * initial_main_weight + aux_feat * initial_aux_weight

        for i in range(len(main_cond)):
            t_to = main_cond[i][1].copy()
            # 存储权重函数，供后续生成过程调用（关键：让权重随步骤动态变化）
            t_to["adaptive_weight_fn"] = adaptive_weight_fn
            
            # 池化输出也按初始权重融合
            if pooled_main is not None and pooled_aux is not None:
                t_to["pooled_output"] = pooled_main * initial_main_weight + pooled_aux * initial_aux_weight
            elif pooled_main is not None:
                t_to["pooled_output"] = pooled_main
            elif pooled_aux is not None:
                t_to["pooled_output"] = pooled_aux
            
            out.append([initial_fused, t_to])
        
        # 关键补充：返回时需要将动态权重逻辑注入到conditioning中（ComfyUI兼容）
        # 确保生成过程中会调用adaptive_weight_fn更新权重
        for cond in out:
            cond[1]["dynamic_weighting"] = True
        return (out, )

    def ConditioningModalSeparation(self, main_cond, aux_cond, main_cond_ratio=0.7):
        out = []
        main_feat = main_cond[0][0].squeeze(0)
        aux_feat = aux_cond[0][0].squeeze(0)
        pooled_main = main_cond[0][1].get("pooled_output")
        pooled_aux = aux_cond[0][1].get("pooled_output")

        min_seq_len = min(main_feat.shape[0], aux_feat.shape[0])
        main_feat = main_feat[:min_seq_len, :]
        aux_feat = aux_feat[:min_seq_len, :]

        def pca_separate(feat, top_k=5):
            mean = feat.mean(dim=0, keepdim=True)
            feat_centered = feat - mean
            cov = torch.matmul(feat_centered.T, feat_centered) / (feat_centered.shape[0] - 1)
            eigvals, eigvecs = torch.linalg.eig(cov)
            eigvals = eigvals.real
            eigvecs = eigvecs.real
            content_idx = eigvals.argsort(descending=True)[:top_k]
            style_idx = eigvals.argsort(descending=True)[top_k:]
            content_vecs = eigvecs[:, content_idx]
            style_vecs = eigvecs[:, style_idx]
            content = torch.matmul(feat_centered, content_vecs)
            style = torch.matmul(feat_centered, style_vecs)
            return content, style, content_vecs, style_vecs, mean

        main_content, main_style, main_c_vecs, main_s_vecs, main_mean = pca_separate(main_feat)
        aux_content, aux_style, aux_c_vecs, aux_s_vecs, aux_mean = pca_separate(aux_feat)

        fused_content = main_cond_ratio * main_content + (1 - main_cond_ratio) * aux_content
        fused_style = (1 - main_cond_ratio) * main_style + main_cond_ratio * aux_style

        fused_feat = (
            torch.matmul(fused_content, main_c_vecs.T) +
            torch.matmul(fused_style, main_s_vecs.T) +
            main_mean
        ).unsqueeze(0)

        for i in range(len(main_cond)):
            t_to = main_cond[i][1].copy()
            if pooled_main is not None and pooled_aux is not None:
                t_to["pooled_output"] = pooled_main * main_cond_ratio + pooled_aux * (1 - main_cond_ratio)
            elif pooled_main is not None:
                t_to["pooled_output"] = pooled_main
            elif pooled_aux is not None:
                t_to["pooled_output"] = pooled_aux
            out.append([fused_feat, t_to])
        return (out, )

    def ConditioningAdversarialFusion(self, main_cond, aux_cond, main_cond_ratio=0.7, adv_weight=0.1):
        out = []
        main_feat = main_cond[0][0]
        aux_feat = aux_cond[0][0]
        pooled_main = main_cond[0][1].get("pooled_output")
        pooled_aux = aux_cond[0][1].get("pooled_output")

        min_seq_len = min(main_feat.shape[1], aux_feat.shape[1])
        main_feat = main_feat[:, :min_seq_len, :]
        aux_feat = aux_feat[:, :min_seq_len, :]

        conflict = torch.norm(main_feat - aux_feat, dim=-1, keepdim=True)
        conflict = (conflict - conflict.min()) / (conflict.max() - conflict.min() + 1e-8)

        aux_base_weight = 1 - main_cond_ratio
        aux_weight = aux_base_weight * (1.0 - adv_weight * conflict)
        aux_weight = torch.clamp(aux_weight, 0.0, 1.0)

        fused_feat = main_feat * main_cond_ratio + aux_feat * aux_weight

        for i in range(len(main_cond)):
            t_to = main_cond[i][1].copy()
            if pooled_main is not None and pooled_aux is not None:
                pooled_conflict = torch.norm(pooled_main - pooled_aux) / (torch.norm(pooled_main + pooled_aux) + 1e-8)
                pooled_aux_weight = aux_base_weight * (1.0 - adv_weight * pooled_conflict)
                t_to["pooled_output"] = pooled_main * main_cond_ratio + pooled_aux * pooled_aux_weight
            elif pooled_main is not None:
                t_to["pooled_output"] = pooled_main
            elif pooled_aux is not None:
                t_to["pooled_output"] = pooled_aux
            out.append([fused_feat, t_to])
        return (out, )

    def merge(self, main_cond, mode, 
              aux_cond=None, main_cond_ratio=0.9, aux_mask=None,
              mask_area="default", attention_temp=1.0, cross_weight=0.5, adaptive_weight=0.7, adv_weight=0.1):
        
        if aux_cond is None:
            main_cond = self.ConditioningSetAreaStrength(main_cond, main_cond_ratio)[0]
            return (main_cond, )
        
        aux_base_strength = max(0.0, 1.0 - main_cond_ratio)
        if aux_mask is not None:
            aux_cond = self.ConditioningSetMask(aux_cond, aux_mask, mask_area, aux_base_strength)[0]
        else:
            aux_cond = self.ConditioningSetAreaStrength(aux_cond, aux_base_strength)[0]

        if mode == "attention":
            conditioning = self.ConditioningAttentionFusion(main_cond, aux_cond, attention_temp, main_cond_ratio)[0]
        elif mode == "cross_attention":
            conditioning = self.ConditioningCrossAttention(main_cond, aux_cond, main_cond_ratio, cross_weight)[0]
        elif mode == "adaptive":
            conditioning = self.ConditioningAdaptiveFusion(main_cond, aux_cond, main_cond_ratio, adaptive_weight)[0]
        elif mode == "modal_sep":
            conditioning = self.ConditioningModalSeparation(main_cond, aux_cond, main_cond_ratio)[0]
        elif mode == "adversarial":
            conditioning = self.ConditioningAdversarialFusion(main_cond, aux_cond, main_cond_ratio, adv_weight)[0]
        elif mode == "combine":
            main_cond = self.ConditioningSetAreaStrength(main_cond, main_cond_ratio)[0]
            conditioning = self.ConditioningCombine(main_cond, aux_cond)[0]
        elif mode == "average":
            conditioning = self.ConditioningAverage(main_cond, aux_cond, main_cond_ratio)[0]
        elif mode == "concat":
            main_cond = self.ConditioningSetAreaStrength(main_cond, main_cond_ratio)[0]
            conditioning = self.ConditioningConcat(main_cond, aux_cond)[0]
        else:
            main_cond = self.ConditioningSetAreaStrength(main_cond, main_cond_ratio)[0]
            conditioning = main_cond
        
        return (conditioning,)





class pre_Kontext:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image": ("IMAGE", ),
                "mask": ("MASK",),
                "prompt_weight":("FLOAT", {"default": 0.5, "min": 0, "max": 1, "step": 0.01}),
                "smoothness":("INT", {"default": 0,  "min":0, "max": 10, "step": 0.1,}),
                "auto_adjust_image": ("BOOLEAN", {"default": False}),  # 新增的输入开关
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","LATENT" )
    RETURN_NAMES = ("context","positive","latent" )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def process(self, context=None, image=None, mask=None, prompt_weight=0.5, pos="", smoothness=0, auto_adjust_image=True):  # 添加参数


        vae = context.get("vae", None)
        clip = context.get("clip", None)
        guidance = context.get("guidance", 2.5)

        if pos and pos.strip(): 
            positive, = CLIPTextEncode().encode(clip, pos)
        else:
            positive = context.get("positive", None)



        if image is None:
            image = context.get("images", None)
            if  image is None:
                return (context,positive,None)


        image=kontext_adjust_image_resolution(image, auto_adjust_image)[0]

        encoded_latent = vae.encode(image)  #
        latent = {"samples": encoded_latent}

        if positive is not None:
            influence = 8 * prompt_weight * (prompt_weight - 1) - 6 * prompt_weight + 6
            scaled_latent = latent["samples"] * influence
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [scaled_latent]},  append=True)
            positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        if mask is not None:
            
            mask =smoothness_mask(mask, smoothness)
            latent = {"samples": encoded_latent,"noise_mask": mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])) }

        context = new_context(context, positive=positive, latent=latent)

        return (context,positive,latent)



class pre_Kontext_mul_Image:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "context": ("RUN_CONTEXT",),
            "reference_latents_method": (("offset", "index","uxo/uno" ), ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                    },

        "optional": {
            "image1": ("IMAGE", ),
            "image2": ("IMAGE", ),
            "image3": ("IMAGE", ),
            "image4": ("IMAGE", ),
                    }
               }


    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", )
    RETURN_NAMES = ("context","positive",)
    FUNCTION = "append"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"


    def append(self,context, guidance, reference_latents_method="uxo/uno",image1=None, image2=None, image3=None, image4=None, ):
        vae = context.get("vae", None)
        positive = context.get("positive", None)
        

        if image1 is not None:
          latent = encode(vae, image1)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)

        if image2 is not None:
          latent = encode(vae, image2)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)    

        if image3 is not None:
          latent = encode(vae, image3)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)
        
        if image4 is not None:
          latent = encode(vae, image4)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)
  
        positive = FluxKontextMultiReferenceLatentMethod().append(positive, reference_latents_method)[0]
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        context = new_context(context, positive=positive, )

        return (context, positive,  )



class pre_Kontext_mul:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "context": ("RUN_CONTEXT",),
                "image": ("IMAGE",),
                "mask": ("MASK", ),
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos4": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos5": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "mask1": ("MASK", ),
                "mask2": ("MASK", ),
                "mask3": ("MASK", ),
                "mask4": ("MASK", ),
                "mask5": ("MASK", ),
                "prompt_weight1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  
                "prompt_weight2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_weight3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_weight4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "prompt_weight5": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                
            }
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING",)
    RETURN_NAMES = ("context","positive",)

    FUNCTION = "Mutil_Clip"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def Mutil_Clip(self, pos1, pos2, pos3, pos4, pos5, image, mask,  prompt_weight1, prompt_weight2, prompt_weight3, prompt_weight4,prompt_weight5,
                    mask1=None, mask2=None, mask3=None, mask4=None, mask5=None, context=None):
        
        set_cond_area = "default" 
        if mask is not None and image is not None:
            vae = context.get("vae", None)
            latent = encode(vae, image)[0]
            # 确保 latent 是张量
            if isinstance(latent, dict):
                latent_tensor = latent["samples"]
            else:
                latent_tensor = latent
            result = set_latent_mask2(latent_tensor, mask)
            Flatent = result  

        else:
            raise Exception("pls input image and mask")

        clip = context.get("clip")

        positive_1, = CLIPTextEncode().encode(clip, pos1)
        positive_2, = CLIPTextEncode().encode(clip, pos2)
        positive_3, = CLIPTextEncode().encode(clip, pos3)
        positive_4, = CLIPTextEncode().encode(clip, pos4)
        positive_5, = CLIPTextEncode().encode(clip, pos5)

        c = []
        set_area_to_bounds = False
        if set_cond_area!= "default":
            set_area_to_bounds = True


        # 处理 mask 维度
        if mask1 is not None and len(mask1.shape) < 3:
            mask1 = mask1.unsqueeze(0)
        if mask2 is not None and len(mask2.shape) < 3:
            mask2 = mask2.unsqueeze(0)
        if mask3 is not None and len(mask3.shape) < 3:
            mask3 = mask3.unsqueeze(0)
        if mask4 is not None and len(mask4.shape) < 3:
            mask4 = mask4.unsqueeze(0)
        if mask5 is not None and len(mask5.shape) < 3:
            mask5 = mask5.unsqueeze(0)

        # 添加条件权重
        if mask1 is not None:
            for t in positive_1:
                append_helper(t, mask1, c, set_area_to_bounds, 1)
        if mask2 is not None:
            for t in positive_2:
                append_helper(t, mask2, c, set_area_to_bounds, 1)
        if mask3 is not None:
            for t in positive_3:
                append_helper(t, mask3, c, set_area_to_bounds, 1)
        if mask4 is not None:
            for t in positive_4:
                append_helper(t, mask4, c, set_area_to_bounds, 1)
        if mask5 is not None:
            for t in positive_5:
                append_helper(t, mask5, c, set_area_to_bounds, 1)
        
        b = c
        # 创建一个原始 latent 的副本，避免重复修改
        original_latent = latent_tensor  # 使用确保的张量

        if mask1 is not None:
            influence = 8 * prompt_weight1 * (prompt_weight1 - 1) - 6 * prompt_weight1 + 6
            result = set_latent_mask2(original_latent, mask1)
            masked_latent = result["samples"]  # 提取 samples 部分进行计算
            latent_samples = masked_latent * influence
            b1 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b1
        if mask2 is not None:
            influence = 8 * prompt_weight2 * (prompt_weight2 - 1) - 6 * prompt_weight2 + 6
            result = set_latent_mask2(original_latent, mask2)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b2 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b2
        if mask3 is not None:
            influence = 8 * prompt_weight3 * (prompt_weight3 - 1) - 6 * prompt_weight3 + 6
            result = set_latent_mask2(original_latent, mask3)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b3 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b3
        if mask4 is not None:
            influence = 8 * prompt_weight4 * (prompt_weight4 - 1) - 6 * prompt_weight4 + 6
            result = set_latent_mask2(original_latent, mask4)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b4 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b4

        if mask5 is not None:
            influence = 8 * prompt_weight5 * (prompt_weight5 - 1) - 6 * prompt_weight5 + 6
            result = set_latent_mask2(original_latent, mask5)
            masked_latent = result["samples"]
            latent_samples = masked_latent * influence
            b5 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b5  

        # 返回张量而不是字典
        context = new_context(context, positive=b, latent=Flatent)
        return (context, b,)






class pre_Kontext_mulCondi:  #隐藏
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "clip":("CLIP",),
                "vae":("VAE",),   
                "image": ("IMAGE",),
                "mask": ("MASK", ),
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True,"default": "" }),
                "mask1": ("MASK", ),
                "mask2": ("MASK", ),
                "mask3": ("MASK", ),
            }
        }
        
    RETURN_TYPES = ("CONDITIONING")
    RETURN_NAMES = ("positive")

    FUNCTION = "Mutil_Clip"
    CATEGORY = "Apt_Preset/chx_tool/Kontext"

    def Mutil_Clip(self, clip=None, vae=None, image=None, mask=None,
                   pos1="", pos2="", pos3="",
                   mask1=None, mask2=None, mask3=None):
        
        set_cond_area = "default" 

        if mask is not None and image is not None:
            latent = encode(vae, image)[0]
            # 确保 latent 是张量
            if isinstance(latent, dict):
                latent_tensor = latent["samples"]
            else:
                latent_tensor = latent
            result = set_latent_mask2(latent_tensor, mask)
            Flatent = result  

        else:
            raise Exception("pls input image and mask")

        positive_1, = CLIPTextEncode().encode(clip, pos1)
        positive_2, = CLIPTextEncode().encode(clip, pos2)
        positive_3, = CLIPTextEncode().encode(clip, pos3)

        c = []
        set_area_to_bounds = False
        if set_cond_area!= "default":
            set_area_to_bounds = True

        if mask1 is not None and len(mask1.shape) < 3:
            mask1 = mask1.unsqueeze(0)
        if mask2 is not None and len(mask2.shape) < 3:
            mask2 = mask2.unsqueeze(0)
        if mask3 is not None and len(mask3.shape) < 3:
            mask3 = mask3.unsqueeze(0)


        # 添加条件权重
        if mask1 is not None:
            for t in positive_1:
                append_helper(t, mask1, c, set_area_to_bounds, 1)
        if mask2 is not None:
            for t in positive_2:
                append_helper(t, mask2, c, set_area_to_bounds, 1)
        if mask3 is not None:
            for t in positive_3:
                append_helper(t, mask3, c, set_area_to_bounds, 1)

        
        b = c
        original_latent = latent_tensor  # 使用确保的张量

        if mask1 is not None:
            result = set_latent_mask2(original_latent, mask1)
            masked_latent = result["samples"]  
            latent_samples = masked_latent 
            b1 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b1
        if mask2 is not None:
            result = set_latent_mask2(original_latent, mask2)
            masked_latent = result["samples"]
            latent_samples = masked_latent 
            b2 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b2
        if mask3 is not None:
            result = set_latent_mask2(original_latent, mask3)
            masked_latent = result["samples"]
            latent_samples = masked_latent 
            b3 = node_helpers.conditioning_set_values(b, {"reference_latents": [latent_samples]}, append=True)
            b = b3

        return (b,)



class Stack_Kontext_MulCondi:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": { 
                "image": ("IMAGE",),
                "mask": ("MASK", ),
                "pos1": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "" }),
                "pos2": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "" }),
                "pos3": ("STRING", {"multiline": True, "dynamicPrompts": True, "default": "" }),
                "mask1": ("MASK", ),
                "mask2": ("MASK", ),
                "mask3": ("MASK", ),
            }
        }

    RETURN_TYPES = ("KONTEXT_MUL_PACK",)
    RETURN_NAMES = ("kontext_MulCondi",)
    FUNCTION = "pack_params"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def pack_params(self, image=None, mask=None,
                   pos1="", pos2="", pos3="",
                   mask1=None, mask2=None, mask3=None):
        kontext_mul_pack = (
            image, mask,
            pos1, mask1,
            pos2, mask2,
            pos3, mask3
        )
        
        return (kontext_mul_pack,)
    





class pre_Kontext_MulImg:#隐藏
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "clip":("CLIP",),
            "vae":("VAE",),           
            "reference_latents_method": (("offset", "index","uxo/uno" ), ),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "pos": ("STRING", {"multiline": True, "default": ""}),
                    },

        "optional": {
            "image1": ("IMAGE", ),
            "image2": ("IMAGE", ),
            "image3": ("IMAGE", ),

                    }
               }


    RETURN_TYPES = ("CONDITIONING", )
    RETURN_NAMES = ("positive",)
    FUNCTION = "append"
    CATEGORY = "Apt_Preset/chx_tool/Kontext"


    def append(self, clip, vae, guidance, reference_latents_method="uxo/uno",image1=None, image2=None, image3=None, pos="", ):
  
        
        positive, = CLIPTextEncode().encode(clip, pos)
        

        if image1 is not None:
          latent = encode(vae, image1)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)

        if image2 is not None:
          latent = encode(vae, image2)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)    

        if image3 is not None:
          latent = encode(vae, image3)[0]
          positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [latent["samples"]]}, append=True)
        
  
        positive = FluxKontextMultiReferenceLatentMethod().append(positive, reference_latents_method)[0]
        positive = node_helpers.conditioning_set_values(positive, {"guidance": guidance})

        return (positive, )



class Stack_Kontext_MulImg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
        
                "reference_latents_method": (("offset", "index", "uxo/uno"), ),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
            }
        }

    RETURN_TYPES = ("KONTEXT_MUL_IMAGE",)
    RETURN_NAMES = ("kontext_Mul_img",)
    FUNCTION = "pack_params"
    CATEGORY = "Apt_Preset/stack/😺backup"

    def pack_params(self, reference_latents_method="uxo/uno", guidance=3.5, pos="", 
                   image1=None, image2=None, image3=None):
        
        kontext_mul_image_pack = (reference_latents_method, guidance, pos, 
                                 image1, image2, image3)
        
        return (kontext_mul_image_pack,)








class sum_stack_Kontext:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model":("MODEL", ),
                "lora_stack": ("LORASTACK",),
                "redux_stack": ("REDUX_STACK",),

                "kontext_MulCond":("KONTEXT_MUL_PACK",),
                "kontext_Mul_img": ("KONTEXT_MUL_IMAGE",),

                "union_stack": ("UNION_STACK",),
                "latent_stack": ("LATENT_STACK",),
            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","LATENT","CLIP","VAE")
    RETURN_NAMES = ("context", "model","positive","latent","clip","vae")
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def merge(self, context=None, model=None, 
              redux_stack=None, lora_stack=None, kontext_MulCond=None,
              union_stack=None, kontext_Mul_img=None, latent_stack=None):
         
        clip = context.get("clip")
        latent = context.get("latent", None)
        vae = context.get("vae", None)

        positive = context.get("positive", None)
        negative = context.get("negative", None)


        if model is None:
            model = context.get("model", None)

        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

#-----------------二选一--------------------------

        if kontext_MulCond is not None :
            if len(kontext_MulCond) >= 8:
                image, mask, pos1, mask1, pos2, mask2, pos3, mask3 = kontext_MulCond[:8]

                positive = pre_Kontext_mulCondi().Mutil_Clip(
                    clip=clip, vae=vae, 
                    pos1=pos1, pos2=pos2, pos3=pos3, 
                    image=image, mask=mask,
                    mask1=mask1, mask2=mask2, mask3=mask3
                )[0]  # 只取返回的第一个值(CONDITIONING)
            else:
                raise ValueError(f"kontext_MulCond 需要 8 个元素，但只提供了 {len(kontext_MulCond)} 个")

        elif kontext_Mul_img is not None :
            if len(kontext_Mul_img) >= 6:  # 修正检查条件
                reference_latents_method, guidance, pos, image1, image2, image3 = kontext_Mul_img[:6]  # 正确解包6个值
                if pos == "":
                    pos = context.get("pos", None)

                positive = pre_Kontext_MulImg().append(
                    clip=clip, vae=vae,
                    reference_latents_method=reference_latents_method,
                    guidance=guidance,
                    pos=pos,
                    image1=image1,
                    image2=image2,
                    image3=image3
                )[0]  
            else:
                raise ValueError(f"kontext_Mul_img 需要 6 个元素，但只提供了 {len(kontext_Mul_img)} 个")
            
#-------------------------------------------

        if redux_stack is not None:
            positive, = Apply_Redux().apply_redux_stack(positive, redux_stack,)

        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])

        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)

        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent=latent,)
        return (context, model, positive, latent, clip, vae )






#endregion---------------------kontext------------------



#region---------------------qwen------------------



class pre_qwen_controlnet:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image1": ("IMAGE",),
                "controlnet1": (['None'] + folder_paths.get_filename_list("model_patches"),),
                "strength1": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                
                "image2": ("IMAGE",),
                "controlnet2": (['None'] + folder_paths.get_filename_list("model_patches"),),
                "strength2": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),
                
                "image3": ("IMAGE",),
                "controlnet3": (['None'] + folder_paths.get_filename_list("model_patches"),),
                "strength3": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),

                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),


            },

        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CONDITIONING","CONDITIONING","LATENT" )
    RETURN_NAMES = ("context","model","positive","negative","latent" )
    CATEGORY = "Apt_Preset/chx_tool/controlnet"
    FUNCTION = "load_controlnet"


    def addConditioning(self,positive, negative, pixels, vae, mask=None):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        # 如果提供了 mask，则进行相关处理
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
            
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
                mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:,:,:,i] -= 0.5
                pixels[:,:,:,i] *= m
                pixels[:,:,:,i] += 0.5
                
            concat_latent = vae.encode(pixels)
            
            out_latent = {}
            out_latent["samples"] = vae.encode(orig_pixels)
            out_latent["noise_mask"] = mask
        else:
            # 如果没有提供 mask，直接编码原始像素
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            # 只有当 mask 存在时才添加 concat_mask
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent)


    def load_controlnet(self, 
                        strength1, 
                        strength2,
                        strength3,  
                        context=None, 
                        controlnet1=None, controlnet2=None, controlnet3=None,
                        image1=None, image2=None, image3=None, vae=None,latent_image=None, latent_mask=None,):



        vae = context.get("vae", None)
        model = context.get("model", None)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        latent = context.get("latent", None)

        if controlnet1 != "None" and image1 is not None:
            cn1=ModelPatchLoader().load_model_patch(controlnet1)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn1, vae, image1, strength1, latent_image, latent_mask)[0]


        if controlnet2 != "None" and image2 is not None:
            cn2=ModelPatchLoader().load_model_patch(controlnet2)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn2, vae, image2, strength2, latent_image, latent_mask)[0]


        if controlnet3 != "None" and image3 is not None:
            cn3=ModelPatchLoader().load_model_patch(controlnet3)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn3, vae, image3, strength3, latent_image, latent_mask)[0]


        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)

        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent)
        return (context, model, positive, negative, latent)





class ZImageFun(QwenImageDiffsynthControlnet):
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "model_patch": ("MODEL_PATCH",),
                              "vae": ("VAE",),
                              "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                              },
                "optional": {"image": ("IMAGE",), "inpaint_image": ("IMAGE",), "mask": ("MASK",)}}

    CATEGORY = "advanced/loaders/zimage"








class pre_ZImageInpaint_patch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {"context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image": ("IMAGE",),
                "controlnet": (['None'] + folder_paths.get_filename_list("model_patches"),{"default":"Z-Image-Turbo-Fun-Controlnet-Union-2.1.safetensors"}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 2.0, "step": 0.01}),         
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),
                "diffDiffusion": ("BOOLEAN", {"default": True}),
                "smoothness": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1, }),

            },

        }

    RETURN_TYPES = ("RUN_CONTEXT","MODEL","CONDITIONING","CONDITIONING","LATENT" )
    RETURN_NAMES = ("context","model","positive","negative","latent" )
    CATEGORY = "Apt_Preset/chx_tool/controlnet"
    FUNCTION = "load_controlnet"




    def addConditioning(self,positive, negative, pixels, vae, mask=None):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        orig_pixels = pixels
        pixels = orig_pixels.clone()
        
        # 如果提供了 mask，则进行相关处理
        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
            
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:,x_offset:x + x_offset, y_offset:y + y_offset,:]
                mask = mask[:,:,x_offset:x + x_offset, y_offset:y + y_offset]

            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:,:,:,i] -= 0.5
                pixels[:,:,:,i] *= m
                pixels[:,:,:,i] += 0.5
                
            concat_latent = vae.encode(pixels)
            
            out_latent = {}
            out_latent["samples"] = vae.encode(orig_pixels)
            out_latent["noise_mask"] = mask
        else:
            # 如果没有提供 mask，直接编码原始像素
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            # 只有当 mask 存在时才添加 concat_mask
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent)





    def load_controlnet(self, 
                        strength,  
                        context=None, 
                        controlnet=None,  smoothness=0, diffDiffusion=True,
                        image=None, vae=None,latent_image=None, latent_mask=None,):


        vae = context.get("vae", None)
        model = context.get("model", None)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        latent = context.get("latent", None)

        if latent_mask is not None:
            if smoothness > 0:
               latent_mask = smoothness_mask(latent_mask, smoothness)
            latent = set_mask(latent, latent_mask)[0]

        if controlnet != "None":
            cn1=ModelPatchLoader().load_model_patch(controlnet)[0]
            model=ZImageFun().diffsynth_controlnet(model, cn1, vae, image, strength, latent_image, latent_mask)[0]

 
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)
            
            
        if diffDiffusion:
            model = DifferentialDiffusion().apply(model)[0]


        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent)
        return (context, model, positive, negative, latent)





class pre_QwenEdit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "image": ("IMAGE", ),
                "mask": ("MASK",),
                "ref_edit": ("BOOLEAN", {"default": True}),
                "mask_condi": ("BOOLEAN", {"default": True}),                
                "model_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step":0.01}),
                "smoothness":("FLOAT", {"default": 0.0,  "min":0.0, "max": 10.0, "step": 0.1,}),
                "pos": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING", "LATENT" )
    RETURN_NAMES = ("context","positive","latent" )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"

    def qwen_encode(self, clip, prompt, vae=None, image=None):
        ref_latent = None
        processed_image = None
        
        if image is None:
            images = []
        else:
            if image.dtype != torch.float32:
                image = image.to(torch.float32)
                
            samples = image.movedim(-1, 1)
            total = int(1024 * 1024)

            scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))

            width = math.floor(samples.shape[3] * scale_by / 8) * 8
            height = math.floor(samples.shape[2] * scale_by / 8) * 8

            original_width = samples.shape[3]
            original_height = samples.shape[2]
            
            if width < original_width or height < original_height:
                upscale_method = "area"
            else:
                upscale_method = "lanczos"
            
            s = common_upscale(samples, width, height, upscale_method, "disabled")
            processed_image = s.movedim(1, -1)
            images = [processed_image[:, :, :, :3]]
            
            if vae is not None:
                ref_latent = vae.encode(processed_image[:, :, :, :3])

        tokens = clip.tokenize(prompt, images=images)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if ref_latent is not None:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": [ref_latent]}, append=True)
        
        return (conditioning, processed_image, ref_latent)

    def process(self, context=None, image=None, mask=None, ref_edit=True, mask_condi=True, pos="", smoothness=0, model_shift=3.0):  
        vae = context.get("vae", None)
        clip = context.get("clip", None)
        model = context.get("model", None)
        
        if model is not None:
            model, = ModelSamplingAuraFlow().patch_aura(model, model_shift)

        if image is None:
            image = context.get("images", None)
            if image is None:
                return (context, None, None)

        if image.dtype != torch.float32:
            image = image.to(torch.float32)

        encoded_latent = vae.encode(image) if vae is not None else None
        latent = {"samples": encoded_latent} if encoded_latent is not None else None

        if pos is None or (isinstance(pos, str) and pos.strip() == ""):
            pos = context.get("pos", "")

        processed_image_for_conditioning = image
        if mask is not None:
            smoothed_mask = smoothness_mask(mask, smoothness)
            
            latent_with_mask = {
                "samples": encoded_latent,
                "noise_mask": smoothed_mask.reshape((-1, 1, smoothed_mask.shape[-2], smoothed_mask.shape[-1]))
            }
            
            if mask_condi and vae is not None:
                conditioned_image = decode(vae, latent_with_mask)[0]
                processed_image_for_conditioning = conditioned_image

        vae_for_encoding = vae if ref_edit else None
        
        positive, _, _ = self.qwen_encode(clip, pos, vae_for_encoding, processed_image_for_conditioning)
        negative, _, _ = self.qwen_encode(clip, "", vae_for_encoding, processed_image_for_conditioning)
        
        if mask is not None and vae is not None and latent is not None:
            positive, negative, latent = InpaintModelConditioning().encode(
                positive, negative, image, vae, mask, True
            )
        elif encoded_latent is not None:
            latent = {"samples": encoded_latent}

        context = new_context(context, positive=positive, latent=latent, model=model)

        return (context, positive, latent)






#endregion-------------------qwen------------------


class pre_mul_ref_latent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "ref_latent_img1": ("IMAGE", ),
                "ref_latent_img2": ("IMAGE", ),
                "ref_latent_img3": ("IMAGE", ),
                "ref_latent_img4": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING",)
    RETURN_NAMES = ("context","positive", )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_tool/conditioning"

    def process(self, context=None, ref_latent_img1=None, ref_latent_img2=None, ref_latent_img3=None, ref_latent_img4=None):  
        vae = context.get("vae", None)
        positive = context.get("positive", None)

        if ref_latent_img1 is not None:
            encoded_latent = vae.encode(ref_latent_img1)
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent]},  append=True)

        if ref_latent_img2 is not None:
            encoded_latent2 = vae.encode(ref_latent_img2)
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent2]},  append=True)

        if ref_latent_img3 is not None:
            encoded_latent3 = vae.encode(ref_latent_img3)
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent3]},  append=True)
            
        if ref_latent_img4 is not None:
            encoded_latent4 = vae.encode(ref_latent_img4)
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent4]},  append=True)

        context = new_context(context, positive=positive)
        return (context,positive,)





class Easy_QwenEdit2509:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
                "vae": ("VAE",),
            },
            "optional": {
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "auto_resize": (["crop", "pad", "stretch"], {"default": "crop"}), 
                "vl_size": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64}),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),

                "system_prompt": ("STRING", {"multiline": False, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
            }
        }
    
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "zero_negative", "latent",)
    FUNCTION = "QWENencode"
    CATEGORY = "conditioning"
    DESCRIPTION = """
    vl_size:视觉尺寸，会影响细节 
    latent_image: 生成图尺寸。Generate the size of the figure. 
    latent_mask: 生成图遮罩 
    system_prompt:系统提示词，指导图像特征描述与修改逻辑（默认提供基础配置） 
    auto_resize:尺寸适配模式（crop-中心裁剪/pad-黑色填充/stretch-强制拉伸）"""

    def _process_image_channels(self, image):
        if image is None:
            return None
        if len(image.shape) == 4:
            b, h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        image = image.clamp(0.0, 1.0)
        return image

    def _auto_resize(self, image: torch.Tensor, target_h: int, target_w: int, auto_resize: str) -> torch.Tensor:
        batch, ch, orig_h, orig_w = image.shape
        
        # 强制最小尺寸≥32（适配VAE 3×3卷积核）
        target_h = max(target_h, 32)
        target_w = max(target_w, 32)
        orig_h = max(orig_h, 32)
        orig_w = max(orig_w, 32)
        
        if auto_resize == "crop":
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            # 强制新尺寸≥目标尺寸，避免裁剪后不足
            new_w = max(new_w, target_w)
            new_h = max(new_h, target_h)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            # 裁剪后强制宽高≥32，避免过小
            crop_h = min(target_h, new_h - y_offset)
            crop_w = min(target_w, new_w - x_offset)
            crop_h = max(crop_h, 32)
            crop_w = max(crop_w, 32)
            result = scaled[:, :, y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
            
        elif auto_resize == "pad":
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            black_bg = torch.zeros((batch, ch, target_h, target_w), dtype=image.dtype, device=image.device)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            black_bg[:, :, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled
            result = black_bg
            
        elif auto_resize == "stretch":
            result = comfy.utils.common_upscale(image, target_w, target_h, "bicubic", "disabled")
            
        else:
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            result = scaled[:, :, y_offset:y_offset + target_h, x_offset:x_offset + target_w]
        
        # 最终尺寸确保是8的倍数且≥32
        final_w = max(32, (result.shape[3] // 8) * 8)
        final_h = max(32, (result.shape[2] // 8) * 8)
        
        if final_w != result.shape[3] or final_h != result.shape[2]:
            x_offset = (result.shape[3] - final_w) // 2
            y_offset = (result.shape[2] - final_h) // 2
            result = result[:, :, y_offset:y_offset + final_h, x_offset:x_offset + final_w]
        
        return result

    def QWENencode(self, prompt="", image1=None, image2=None, image3=None, vae=None, clip=None, vl_size=384, 
                   latent_image=None, latent_mask=None, system_prompt="", auto_resize="crop"):
        
        if latent_image is None:
            raise ValueError("latent_image Must be input to determine the size of the generated image；latent_image 必须输入以确定生成图像的尺寸")
        
        image1 = self._process_image_channels(image1)
        image2 = self._process_image_channels(image2)
        image3 = self._process_image_channels(image3)
        orig_images = [image1, image2, image3]
        images_vl = []
        llama_template = self.get_system_prompt(system_prompt)
        image_prompt = ""

        for i, image in enumerate(orig_images):
            if image is not None:
                samples = image.movedim(-1, 1)
                current_total = samples.shape[3] * samples.shape[2]
                scale_by = math.sqrt(vl_size * vl_size / current_total) if current_total > 0 else 1.0
                width = max(64, round(samples.shape[3] * scale_by))
                height = max(64, round(samples.shape[2] * scale_by))
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"

        if latent_image is not None:
            latent_image = self._process_image_channels(latent_image)
            getsamples = latent_image.movedim(-1, 1)
            target_h, target_w = getsamples.shape[2], getsamples.shape[3]
            
            for i in range(3):
                if orig_images[i] is not None:
                    img_bchw = orig_images[i].movedim(-1, 1)
                    resized_img_bchw = self._auto_resize(img_bchw, target_h, target_w, auto_resize)
                    orig_images[i] = resized_img_bchw.movedim(1, -1)

        ref_latents = []
        for i, image in enumerate(orig_images):
            if image is not None and vae is not None:
                samples = image.movedim(-1, 1)
                # 强制尺寸≥32，避免VAE卷积报错
                orig_sample_h = max(samples.shape[2], 32)
                orig_sample_w = max(samples.shape[3], 32)
                if samples.shape[2] != orig_sample_h or samples.shape[3] != orig_sample_w:
                    samples = comfy.utils.common_upscale(samples, orig_sample_w, orig_sample_h, "bicubic", "disabled")
                # 计算8的倍数尺寸，仍强制≥32
                width = (orig_sample_w // 8) * 8
                height = (orig_sample_h // 8) * 8
                width = max(width, 32)
                height = max(height, 32)
                scaled_img = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")
                ref_latents.append(vae.encode(scaled_img.movedim(1, -1)[:, :, :, :3]))

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        
        
        
        
        positive = conditioning
        negative = self.zero_out(positive)

        latent = {"samples": torch.zeros(1, 4, 64, 64)}
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(positive, negative, latent_image, vae, mask=latent_mask if latent_mask is not None else None)

        return (positive, negative, latent)

    def addConditioning(self, positive, negative, pixels, vae, mask=None):
        pixels = self._process_image_channels(pixels)
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        orig_pixels = pixels
        pixels = orig_pixels.clone()

        if mask is not None:
            mask = torch.nn.functional.interpolate(mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])), size=(pixels.shape[1], pixels.shape[2]), mode="bilinear")
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
                mask = mask[:, :, x_offset:x + x_offset, y_offset:y + y_offset]
            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] = pixels[:, :, :, i] * m + 0.5 * (1 - m)
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": vae.encode(orig_pixels), "noise_mask": mask}
        else:
            if pixels.shape[1] != x or pixels.shape[2] != y:
                x_offset = (pixels.shape[1] % 8) // 2
                y_offset = (pixels.shape[2] % 8) // 2
                pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}

        out = []
        for conditioning in [positive, negative]:
            c = node_helpers.conditioning_set_values(conditioning, {"concat_latent_image": concat_latent})
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        return (out[0], out[1], out_latent)

    def zero_out(self, conditioning):
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return c

    def get_system_prompt(self, instruction):
        template_prefix = "<|im_start|>system\n"
        template_suffix = "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        if instruction == "":
            instruction_content = "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."
        else:
            if template_prefix in instruction:
                instruction = instruction.split(template_prefix)[1]
            if template_suffix in instruction:
                instruction = instruction.split(template_suffix)[0]
            if "{}" in instruction:
                instruction = instruction.replace("{}", "")
            instruction_content = instruction
        return template_prefix + instruction_content + template_suffix







class sum_stack_QwenEditPlus:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model": ("MODEL", ),
                "lora_stack": ("LORASTACK",),
                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                "union_controlnet": ("UNION_STACK",),
                "vl_size": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 64}),
                "auto_resize": (["crop", "pad", "stretch"], {"default": "crop", }),
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "system_prompt": ("STRING", {"multiline": False, "default": "Describe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate."}),
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),
                "image1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "image2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "image3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
            "hidden": {},
        }
    
    RETURN_TYPES = ("RUN_CONTEXT", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "CLIP", "VAE")
    RETURN_NAMES = ("context", "model", "positive", "negative", "latent", "clip", "vae")
    FUNCTION = "QWENencode"
    CATEGORY = "Apt_Preset/chx_tool"
    DESCRIPTION = """
    vl_size:视觉尺寸越大，提取的图像特征越丰富。
    auto_resize: 缩放模式（crop=中心裁剪，pad=中心黑色填充，stretch=强制拉伸） 
    system_prompt:系统提示词，用于指导图像特征描述与修改逻辑（默认提供基础配置） 
    latent_image: 生成图尺寸。
    latent_mask: 生成图遮罩"""
    
    def _process_image_channels(self, image):
        if image is None:
            return None
        if len(image.shape) == 4:
            b, h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        elif len(image.shape) == 3:
            h, w, c = image.shape
            if c == 4:
                rgb = image[..., :3]
                alpha = image[..., 3:4]
                black_bg = torch.zeros_like(rgb)
                image = rgb * alpha + black_bg * (1 - alpha)
                image = image[..., :3]
            elif c != 3:
                image = image[..., :3]
        image = image.clamp(0.0, 1.0)
        return image
    
    def QWENencode(self, context=None, prompt="", model=None, lora_stack=None, union_controlnet=None,
                   image1=None, image2=None, image3=None, vl_size=384, latent_image=None, latent_mask=None,
                   auto_resize="crop", union_stack=None, system_prompt="",
                   image1_strength=1.0, image2_strength=1.0, image3_strength=1.0):
        
        clip = context.get("clip", None)
        negative = context.get("negative", None)
        vae = context.get("vae", None)
        latent = context.get("latent", None)
        
        if union_stack is not None and union_controlnet is None:
            union_controlnet = union_stack
        if model is None:
            model = context.get("model", None)

        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        if prompt == "":
            prompt = context.get("pos", "")
        

        if latent_image is None:
            raise ValueError("Need to input the latent_image, refer to the image size需要输入latent_image，以此图片作为生成图的尺寸")
        
        image1 = self._process_image_channels(image1)
        image2 = self._process_image_channels(image2)
        image3 = self._process_image_channels(image3)
        images = [image1, image2, image3]
        
        target_h, target_w = 1024, 1024
        if latent_image is not None:
            latent_image = self._process_image_channels(latent_image)
            target_h, target_w = latent_image.shape[1], latent_image.shape[2]
        
        images_vl = []
        min_size = 64
        for image in images:
            if image is not None:
                samples = image.movedim(-1, 1)
                orig_h, orig_w = samples.shape[2], samples.shape[3]
                total_pixels = vl_size * vl_size
                orig_pixels = orig_h * orig_w
                scale_by = math.sqrt(total_pixels / orig_pixels) if orig_pixels > 0 else 1.0
                width = max(round(orig_w * scale_by), min_size)
                height = max(round(orig_h * scale_by), min_size)
                max_ratio = 10.0
                if width / height > max_ratio:
                    width = int(height * max_ratio)
                elif height / width > max_ratio:
                    height = int(width * max_ratio)
                scaled_img = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")
                images_vl.append(scaled_img.movedim(1, -1))
        
        for i in range(len(images)):
            if images[i] is not None:
                if auto_resize == "stretch":
                    images[i] = self.stretch_resize(images[i], target_h, target_w)
                else:
                    images[i] = self.auto_resize(images[i], target_h, target_w, auto_resize)
        
        ref_latents = []
        image_prompt = ""
        strengths = [image1_strength, image2_strength, image3_strength]
        for i, image in enumerate(images):
            if image is not None:
                image_prompt += f"Picture {i + 1}: <|vision_start|><|image_pad|><|vision_end|>"
                if vae is not None:
                    samples = image.movedim(-1, 1)
                    orig_sample_h = max(samples.shape[2], 64)
                    orig_sample_w = max(samples.shape[3], 64)
                    if samples.shape[2] != orig_sample_h or samples.shape[3] != orig_sample_w:
                        samples = comfy.utils.common_upscale(samples, orig_sample_w, orig_sample_h, "bicubic", "disabled")
                    width = (orig_sample_w // 8) * 8
                    height = (orig_sample_h // 8) * 8
                    width = max(width, 64)
                    height = max(height, 64)
                    if width != orig_sample_w or height != orig_sample_h:
                        samples = comfy.utils.common_upscale(samples, width, height, "bicubic", "disabled")
                    latent = vae.encode(samples.movedim(1, -1))
                    if i < len(strengths):
                        latent = latent * strengths[i]
                    ref_latents.append(latent)
        
        llama_template = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{{}}<|im_end|>\n<|im_start|>assistant\n"
        full_prompt = image_prompt + prompt
        tokens = clip.tokenize(full_prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(
                conditioning, {"reference_latents": ref_latents}, append=True
            )
        positive = conditioning
        negative = self.zero_out_simple(positive)
        
        if union_controlnet is not None:
            positive, negative = Apply_CN_union().apply_union_stack(
                positive, negative, vae, union_controlnet, extra_concat=[]
            )
        
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(positive, negative, latent_image, vae, latent_mask)
        
        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent=latent)
        return (context, model, positive, negative, latent, clip, vae)
    
    def addConditioning(self, positive, negative, pixels, vae, mask=None):
        pixels = self._process_image_channels(pixels)
        orig_pixels = pixels.clone()
        
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] - x) // 2
            y_offset = (pixels.shape[2] - y) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        
        if mask is not None:
            mask = torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                size=(pixels.shape[1], pixels.shape[2]),
                mode="bilinear",
                align_corners=False
            )
            m = (1.0 - mask.round()).squeeze(1)
            for i in range(3):
                pixels[:, :, :, i] = pixels[:, :, :, i] * m + 0.5 * (1 - m)
            
            pixels = pixels.clamp(0.0, 1.0)
            
            concat_latent = vae.encode(pixels)
            out_latent = {
                "samples": vae.encode(self._process_image_channels(orig_pixels)),
                "noise_mask": mask
            }
        else:
            concat_latent = vae.encode(pixels)
            out_latent = {"samples": concat_latent}
        
        out = []
        for cond in [positive, negative]:
            c = node_helpers.conditioning_set_values(cond, {"concat_latent_image": concat_latent})
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent)





    def auto_resize(self, image: torch.Tensor, target_h: int, target_w: int, auto_resize: str) -> torch.Tensor:
        batch, orig_h, orig_w, ch = image.shape
        target_h = max(target_h, 64)
        target_w = max(target_w, 64)
        orig_h = max(orig_h, 64)
        orig_w = max(orig_w, 64)
        
        image_bchw = image.movedim(-1, 1)
        
        if auto_resize == "crop":
            scale = max(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            new_w = max(new_w, target_w)
            new_h = max(new_h, target_h)
            scaled = comfy.utils.common_upscale(image_bchw, new_w, new_h, "bicubic", "disabled")
            x_offset = (new_w - target_w) // 2
            y_offset = (new_h - target_h) // 2
            crop_h = min(target_h, new_h - y_offset)
            crop_w = min(target_w, new_w - x_offset)
            crop_h = max(crop_h, 64)
            crop_w = max(crop_w, 64)
            result = scaled[:, :, y_offset:y_offset + crop_h, x_offset:x_offset + crop_w]
        elif auto_resize == "pad":
            scale = min(target_w / orig_w, target_h / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            scaled = comfy.utils.common_upscale(image_bchw, new_w, new_h, "bicubic", "disabled")
            black_bg = torch.zeros((batch, ch, target_h, target_w), dtype=image.dtype, device=image.device)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            black_bg[:, :, y_offset:y_offset + new_h, x_offset:x_offset + new_w] = scaled
            result = black_bg
        else:
            result = comfy.utils.common_upscale(image_bchw, target_w, target_h, "bicubic", "disabled")
        
        result_bhwc = result.movedim(1, -1)
        return result_bhwc
    
    def stretch_resize(self, image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        target_h = max(target_h, 64)
        target_w = max(target_w, 64)
        image_bchw = image.movedim(-1, 1)
        scaled = comfy.utils.common_upscale(image_bchw, target_w, target_h, "bicubic", "disabled")
        return scaled.movedim(1, -1)
    
    def zero_out_simple(self, conditioning):
        if not conditioning:
            return []
        c = []
        for t in conditioning:
            d = t[1].copy()
            pooled_output = d.get("pooled_output", None)
            if pooled_output is not None:
                d["pooled_output"] = torch.zeros_like(pooled_output)
            conditioning_lyrics = d.get("conditioning_lyrics", None)
            if conditioning_lyrics is not None:
                d["conditioning_lyrics"] = torch.zeros_like(conditioning_lyrics)
            n = [torch.zeros_like(t[0]), d]
            c.append(n)
        return c






class sum_stack_flux2_Klein:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "model": ("MODEL", ),
                "lora_stack": ("LORASTACK",),
                "ref_latent_img1": ("IMAGE", ),
                "ref_latent_img2": ("IMAGE", ),
                "ref_latent_img3": ("IMAGE", ),
                "ref_latent_img4": ("IMAGE", ),
                "ref_latent_img5": ("IMAGE", ),
                "union_stack": ("UNION_STACK",),

                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),

                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "img1_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "img2_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "img3_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "img4_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "img5_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "main_prompt_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),

            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING","CONDITIONING","STRING")
    RETURN_NAMES = ("context","positive","negative-zero","prompt_info" )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_tool"
    DESCRIPTION = """
    1-特征权重追加：[语义@权重数值0~10] 
    例如：个性特征默认权重1.0 ，<1则减弱，>1则加强
    风格减弱： 3d风格，[女孩打伞，瀑布@0.2]
    风格增强：[3d风格@0.2]，女孩打伞，瀑布
    main_prompt_ratio：整体权重占比，越高，语义越偏向整体
    2-图片权重编号对应参考图编号
    3-latent_image: 生成图尺寸，如果没有连接，则从上下文获取latent尺寸
    """
    def process(self, context=None, model=None, lora_stack=None,latent_image=None, latent_mask=None, prompt="",union_stack=None,
                main_prompt_ratio=0.5,
                ref_latent_img1=None, img1_strength=1.0,
                ref_latent_img2=None, img2_strength=1.0,
                ref_latent_img3=None, img3_strength=1.0,
                ref_latent_img4=None, img4_strength=1.0,
                ref_latent_img5=None, img5_strength=1.0):  

        latent = None
        clip = context.get("clip", None)
        vae = context.get("vae", None)
        negative= context.get("negative", None)
        negative = condi_zero_out(negative)[0]  
        
        if model is None:
            model = context.get("model", None)

        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        if prompt == "":
            prompt = context.get("pos", "a boy in a forest, ")
        prompt_weight_processor = pre_Unit_PromptWeight()
        positive, prompt_info = prompt_weight_processor.process(clip, prompt, main_prompt_ratio)


        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack (positive, negative, vae, union_stack, extra_concat=[] )
        
        if latent_image is not None:
            positive, negative, latent = self.addConditioning(positive, negative, latent_image, vae, latent_mask)


        if ref_latent_img1 is not None:
            encoded_latent = vae.encode(ref_latent_img1) * img1_strength
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent]},  append=True)

        if ref_latent_img2 is not None:
            encoded_latent2 = vae.encode(ref_latent_img2) * img2_strength
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent2]},  append=True)

        if ref_latent_img3 is not None:
            encoded_latent3 = vae.encode(ref_latent_img3) * img3_strength
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent3]},  append=True)
            
        if ref_latent_img4 is not None:
            encoded_latent4 = vae.encode(ref_latent_img4) * img4_strength
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent4]},  append=True)

        if ref_latent_img5 is not None:
            encoded_latent5 = vae.encode(ref_latent_img5) * img5_strength
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [encoded_latent5]},  append=True)


        context = new_context(context, model=model, positive=positive, negative=negative, latent=latent)

        return (context,positive,negative, prompt_info)

    def _process_image_channels(self, pixels):
        if pixels.shape[-1] > 3:
            pixels = pixels[..., :3]
        pixels = pixels.clamp(0.0, 1.0)
        return pixels
    
    def addConditioning(self, positive, negative, pixels, vae, mask=None):
        pixels = self._process_image_channels(pixels)
        orig_pixels = pixels.clone()
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] - x) // 2
            y_offset = (pixels.shape[2] - y) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        
        concat_latent = vae.encode(pixels)
        out_latent = {"samples": concat_latent}

        if mask is not None:
            mask = torch.nn.functional.interpolate(
                mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
                size=(x, y),
                mode="bilinear",
                align_corners=False
            )
            mask = mask.round().clamp(0.0, 1.0)
            m = (1.0 - mask).squeeze(1).unsqueeze(-1)
            pixels = pixels * m + 0.5 * (1 - m)
            pixels = pixels.clamp(0.0, 1.0)

            concat_latent = vae.encode(pixels)
            out_latent = {
                "samples": vae.encode(self._process_image_channels(orig_pixels)),
                "noise_mask": mask
            }
        
        out = []
        for cond in [positive, negative]:
            c = node_helpers.conditioning_set_values(cond, {"concat_latent_image": concat_latent})
            if mask is not None:
                c = node_helpers.conditioning_set_values(c, {"concat_mask": mask})
            out.append(c)
        
        return (out[0], out[1], out_latent,)
















