
#region-------------------------------import-----------------------
import folder_paths
import torch
from nodes import common_ksampler, CLIPTextEncode, ControlNetApplyAdvanced

import math
import comfy
import node_helpers
import comfy.utils
from ..office_unit import *


from comfy_extras.nodes_model_advanced import ModelSamplingAuraFlow



from ..office_unit import FluxKontextMultiReferenceLatentMethod


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


#endregion-----------------------------import----------------------------





#region---------------------kontext------------------


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
    CATEGORY = "Apt_Preset/chx_tool/Kontext"

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
    CATEGORY = "Apt_Preset/chx_tool/Kontext"


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
    CATEGORY = "Apt_Preset/chx_tool/Kontext"

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
    CATEGORY = "Apt_Preset/chx_tool"

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
    CATEGORY = "Apt_Preset/chx_tool/qwen"
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
                        image1=None, image2=None, image3=None, vae=None,latent_image=None,latent_mask=None,):



        vae = context.get("vae", None)
        model = context.get("model", None)
        positive = context.get("positive", None)
        negative = context.get("negative", None)
        latent = context.get("latent", None)

        if controlnet1 != "None" and image1 is not None:
            cn1=ModelPatchLoader().load_model_patch(controlnet1)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn1, vae, image1, strength1, latent_mask)[0]


        if controlnet2 != "None" and image2 is not None:
            cn2=ModelPatchLoader().load_model_patch(controlnet2)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn2, vae, image2, strength2, latent_mask)[0]


        if controlnet3 != "None" and image3 is not None:
            cn3=ModelPatchLoader().load_model_patch(controlnet3)[0]
            model=QwenImageDiffsynthControlnet().diffsynth_controlnet(model, cn3, vae, image3, strength3, latent_mask)[0]


        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)

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



class sum_stack_QwenEditPlus:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),

            },
            "optional": {
                "model":("MODEL", ),               
                "lora_stack": ("LORASTACK",),

                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                
                "union_controlnet": ("UNION_STACK",),
                "vl_size": ("INT", {"default":384, "min": 64, "max": 2048, "step": 64}),   
                "auto_resize":("BOOLEAN", {"default": True}),                 
                "prompt": ("STRING", {"multiline": True, "default": ""}),    
                "latent_image": ("IMAGE", ),
                "latent_mask": ("MASK", ),

            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","LATENT","CLIP","VAE")
    RETURN_NAMES = ("context", "model","positive","latent","clip","vae")
    FUNCTION = "QWENencode"
    CATEGORY = "Apt_Preset/chx_tool"

    DESCRIPTION = """注释：
    vl_size:视觉尺寸，会影响细节    
    auto_resize: 按latent_image统一尺寸
    latent_image: 生成图尺寸。（没接入，按原始1024*1024算法）
    latent_mask: 生成图遮罩
    """

    def QWENencode(self,context=None, prompt="", model=None, lora_stack=None,union_controlnet=None,image1=None, image2=None, image3=None,
                   vl_size=384,latent_image=None, latent_mask=None,auto_resize=True):

        if model is None:
            model = context.get("model", None)

        clip = context.get("clip", None)

        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        negative= context.get("negative", None)
        vae = context.get("vae", None)
        latent = context.get("latent", None)
       
        if auto_resize and latent_image is not None:
            getsamples = latent_image.movedim(-1, 1)
            if image1 is not None: 
                image1 = image1.movedim(-1, 1)
                image1 = self.auto_resize(image1, getsamples)[0]
                image1 = image1.movedim(1, -1)  
            if image2 is not None:
                image2 = image2.movedim(-1, 1)
                image2 = self.auto_resize(image2, getsamples)[0]
                image2 = image2.movedim(1, -1)
            if image3 is not None:
                image3 = image3.movedim(-1, 1)
                image3 = self.auto_resize(image3, getsamples)[0]
                image3 = image3.movedim(1, -1)

#-----------------------------------------------------
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(vl_size * vl_size)   #视觉统一处理
                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)
                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))

                if vae is not None:
                    if latent_image is not None:
                        getsamples = latent_image.movedim(-1, 1)
                        getwidth = round(getsamples.shape[3])
                        getheight = round(getsamples.shape[2])
                        total = int(getwidth * getheight)
                        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                        width = round(samples.shape[3] * scale_by / 8.0) * 8
                        height = round(samples.shape[2] * scale_by / 8.0) * 8

                    else:
                        total = int(1024 * 1024)
                        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                        width = round(samples.shape[3] * scale_by / 8.0) * 8
                        height = round(samples.shape[2] * scale_by / 8.0) * 8

                    K = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(K.movedim(1, -1)[:, :, :, :3]))


                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
      
#------------------------------------------------------------------------

        if union_controlnet is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_controlnet, extra_concat=[])

        if latent_image is not None:
            positive, negative, latent = self.addConditioning(
                positive, negative, latent_image, vae, 
                mask=latent_mask if latent_mask is not None else None)

        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent=latent,)
        return (context, model, positive, latent, clip, vae)


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


    def auto_resize(self, image, get_image_size):
        if len(image.shape) == 3:
            H, W, C = image.shape
        else:  
            B, H, W, C = image.shape

        _, height_max, width_max, _ = get_image_size.shape
            
        image = image.movedim(-1,1)
        outimage = common_upscale(image, width_max, height_max, "bicubic", "center")
        image = outimage.movedim(1,-1)
        
        width = max(image.shape[2], 64)
        height = max(image.shape[1], 64)
        
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        return(image,)



class sum_stack_QwenEdit:

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),

            },
            "optional": {
                "model":("MODEL", ),               
                "lora_stack": ("LORASTACK",),

                "image1": ("IMAGE", ),
                "image2": ("IMAGE", ),
                "image3": ("IMAGE", ),
                
                "union_stack": ("UNION_STACK",),
                "latent_stack": ("LATENT_STACK",),
                      
                "prompt": ("STRING", {"multiline": True, "default": ""}),    

            },
            "hidden": {},
        }
        
    RETURN_TYPES = ("RUN_CONTEXT","MODEL", "CONDITIONING","LATENT","CLIP","VAE")
    RETURN_NAMES = ("context", "model","positive","latent","clip","vae")
    FUNCTION = "QWENencode"

    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"


    def QWENencode(self,context=None, prompt="", model=None, lora_stack=None,union_stack=None,latent_stack=None, image1=None, image2=None, image3=None):

        if model is None:
            model = context.get("model", None)

        clip = context.get("clip", None)

        if lora_stack is not None:
            model, clip = Apply_LoRAStack().apply_lora_stack(model, clip, lora_stack)

        negative= context.get("negative", None)
        vae = context.get("vae", None)
        latent = context.get("latent", None)

#-----------------------------------------------------
        ref_latents = []
        images = [image1, image2, image3]
        images_vl = []
        llama_template = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        image_prompt = ""

        for i, image in enumerate(images):
            if image is not None:
                samples = image.movedim(-1, 1)
                total = int(384 * 384)

                scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                width = round(samples.shape[3] * scale_by)
                height = round(samples.shape[2] * scale_by)

                s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                images_vl.append(s.movedim(1, -1))
                if vae is not None:
                    total = int(1024 * 1024)
                    scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
                    width = round(samples.shape[3] * scale_by / 8.0) * 8
                    height = round(samples.shape[2] * scale_by / 8.0) * 8

                    s = comfy.utils.common_upscale(samples, width, height, "area", "disabled")
                    ref_latents.append(vae.encode(s.movedim(1, -1)[:, :, :, :3]))

                image_prompt += "Picture {}: <|vision_start|><|image_pad|><|vision_end|>".format(i + 1)

        tokens = clip.tokenize(image_prompt + prompt, images=images_vl, llama_template=llama_template)
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        if len(ref_latents) > 0:
            conditioning = node_helpers.conditioning_set_values(conditioning, {"reference_latents": ref_latents}, append=True)
        positive = conditioning
      
#------------------------------------------------------------------------

        if union_stack is not None:
            positive, negative = Apply_CN_union().apply_union_stack(positive, negative, vae, union_stack, extra_concat=[])

        if latent_stack is not None:
            model, positive, negative, latent = Apply_latent().apply_latent_stack(model, positive, negative, vae, latent_stack)

        context = new_context(context, clip=clip, positive=positive, negative=negative, model=model, latent=latent,)
        return (context, model, positive, latent, clip, vae)



#endregion-------------------qwen------------------



class pre_ref_condition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "context": ("RUN_CONTEXT",),
            },
            "optional": {
                "latnet_image1": ("IMAGE", ),
                "latnet_image2": ("IMAGE", ),
                "latnet_image3": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("RUN_CONTEXT","CONDITIONING",)
    RETURN_NAMES = ("context","positive", )
    FUNCTION = "process"
    CATEGORY = "Apt_Preset/chx_tool/conditioning"

    def process(self, context=None, latnet_image1=None, latnet_image2=None, latnet_image3=None):  
        vae = context.get("vae", None)
        positive = context.get("positive", None)

        if latnet_image1 is not None:
            encoded_latent = vae.encode(latnet_image1)
            latent = {"samples": encoded_latent}
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [latent]},  append=True)

        if latnet_image2 is not None:
            encoded_latent2 = vae.encode(latnet_image2)
            latent2 = {"samples": encoded_latent2}
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [latent2]},  append=True)

        if latnet_image3 is not None:
            encoded_latent3 = vae.encode(latnet_image3)
            latent3 = {"samples": encoded_latent3}
            positive = node_helpers.conditioning_set_values( positive, {"reference_latents": [latent3]},  append=True)

        return (context,positive,)





class pre_MulCondiMode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mode": (["combine", "average", "concat"], ),
            },
            "optional": {
                "conditioning_1": ("CONDITIONING", ),
                "conditioning_2": ("CONDITIONING", ),
                "conditioning_3": ("CONDITIONING",),
                "conditioning_4": ("CONDITIONING",),
                "strength1": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength2": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength3": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength4": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})  
            }
        }
    
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "merge"
    CATEGORY = "Apt_Preset/chx_tool/conditioning"

    DESCRIPTION = """
    - combine：列表拼接，让模型 “同时做两件事”
    - average：数值融合，让模型 “做一件中间的事”
    - concat：维度拼接，让模型 “用更完整的信息做两件事”"""

    def merge(self, mode, conditioning_1=None, conditioning_2=None, 
              conditioning_3=None, conditioning_4=None,
              strength1=0.5, strength2=0.5, strength3=0.5, strength4=0.5):
        
        conditionings = []
        strengths = []
        inputs = [
            (conditioning_1, strength1),
            (conditioning_2, strength2),
            (conditioning_3, strength3),
            (conditioning_4, strength4)
        ]
        for cond, strength in inputs:
            if cond is not None:
                conditionings.append(cond)
                strengths.append(strength)
        
        if not conditionings:
            logging.warning("No valid conditioning inputs provided.")
            return ([], )
        
        if len(conditionings) == 1:
            result = []
            cond = conditionings[0]
            strength = strengths[0]
            for item in cond:
                tensor = torch.mul(item[0], strength)
                meta = item[1].copy()
                if "pooled_output" in meta and meta["pooled_output"] is not None:
                    meta["pooled_output"] = torch.mul(meta["pooled_output"], strength)
                result.append([tensor, meta])
            return (result, )
        
        if mode == "combine":
            result = []
            for cond, strength in zip(conditionings, strengths):
                for item in cond:
                    if item[0].numel() == 0:
                        continue
                    tensor = torch.mul(item[0], strength)
                    meta = item[1].copy()
                    if "pooled_output" in meta and meta["pooled_output"] is not None:
                        meta["pooled_output"] = torch.mul(meta["pooled_output"], strength)
                    result.append([tensor, meta])
            return (result, )
        
        elif mode == "average":
            total_strength = sum(strengths)
            if total_strength <= 0:
                normalized = [1.0 / len(strengths)] * len(strengths)
            else:
                normalized = [s / total_strength for s in strengths]
            
            base = conditionings[0]
            out = []
            for i in range(len(base)):
                result_tensor = torch.mul(base[i][0], normalized[0])
                result_meta = base[i][1].copy()
                pooled = None
                if "pooled_output" in result_meta and result_meta["pooled_output"] is not None:
                    pooled = torch.mul(result_meta["pooled_output"], normalized[0])
                
                for j in range(1, len(conditionings)):
                    cond = conditionings[j]
                    if i >= len(cond):

                        continue
                    curr_tensor = cond[i][0]
                    curr_meta = cond[i][1]
                    
                    if curr_tensor.shape[1] != result_tensor.shape[1]:
                        if curr_tensor.shape[1] > result_tensor.shape[1]:
                            curr_tensor = curr_tensor[:, :result_tensor.shape[1], :]
                        else:
                            pad = torch.zeros(
                                (curr_tensor.shape[0], result_tensor.shape[1] - curr_tensor.shape[1], curr_tensor.shape[2]),
                                dtype=curr_tensor.dtype, device=curr_tensor.device
                            )
                            curr_tensor = torch.cat([curr_tensor, pad], dim=1)
                    
                    result_tensor += torch.mul(curr_tensor, normalized[j])
                    
                    if pooled is not None and "pooled_output" in curr_meta and curr_meta["pooled_output"] is not None:
                        curr_pooled = curr_meta["pooled_output"]
                        if curr_pooled.shape != pooled.shape:
                            if curr_pooled.numel() > pooled.numel():
                                curr_pooled = curr_pooled.flatten()[:pooled.numel()].reshape(pooled.shape)
                            else:
                                curr_pooled = torch.cat([curr_pooled.flatten(), torch.zeros(pooled.numel() - curr_pooled.numel(), device=curr_pooled.device)]).reshape(pooled.shape)
                        pooled += torch.mul(curr_pooled, normalized[j])
                
                if pooled is not None:
                    result_meta["pooled_output"] = pooled
                out.append([result_tensor, result_meta])
            
            return (out, )
        
        elif mode == "concat":
            out = []
            base = conditionings[0]
            for i in range(len(base)):
                tensors = [torch.mul(base[i][0], strengths[0])]
                for j in range(1, len(conditionings)):
                    cond = conditionings[j]
                    if i >= len(cond):
                        logging.warning(f"Conditioning {j+1} has fewer items than the first, skipping index {i}")
                        continue
                    curr_tensor = torch.mul(cond[i][0], strengths[j])
                    if curr_tensor.shape[0] != tensors[0].shape[0] or curr_tensor.shape[2] != tensors[0].shape[2]:
                        logging.warning(f"Conditioning {j+1} has incompatible dimensions with the first, skipping.")
                        continue
                    tensors.append(curr_tensor)
            
                result_tensor = torch.cat(tensors, dim=1)
                result_meta = base[i][1].copy()
                
                if "pooled_output" in result_meta and result_meta["pooled_output"] is not None:
                    pooled_list = [torch.mul(result_meta["pooled_output"], strengths[0])]
                    for j in range(1, len(conditionings)):
                        cond = conditionings[j]
                        if i < len(cond) and "pooled_output" in cond[i][1] and cond[i][1]["pooled_output"] is not None:
                            pooled_list.append(torch.mul(cond[i][1]["pooled_output"], strengths[j]))
                    result_meta["pooled_output"] = torch.cat(pooled_list, dim=0)
                    logging.info("Concatenated pooled_output may not be compatible with some models.")
                
                out.append([result_tensor, result_meta])
            
            return (out, )
        
        else:
            raise ValueError(f"Unknown mode: {mode}")




































