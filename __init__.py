WEB_DIRECTORY = "./web"


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY", ]

import sys; print(sys.executable)





from .NodeChx.main_nodes import *
from .NodeChx.main_stack import *
from .NodeChx.AdvancedCN import AD_sch_latent
from .NodeChx.IPAdapterPlus import chx_IPA_basic,chx_IPA_faceID,chx_IPA_XL,chx_IPA_region_combine, chx_IPA_apply_combine,chx_IPA_adv,chx_IPA_faceID_adv
from .NodeChx.style_node import *
from .NodeChx.video_node import *
from .NodeChx.Ksampler_all import *
from .NodeChx.Keyframe_schedule import *

from .NodeBasic.lay_img_canvas import *
from .NodeBasic.highway import Data_Highway
from .NodeBasic.C_packdata import *
from .NodeBasic.C_math import *
from .NodeBasic.C_model import *
from .NodeBasic.C_mask import *
from .NodeBasic.C_latent import *
from .NodeBasic.C_viewIO import *
from .NodeBasic.C_AD import *
from .NodeBasic.C_image import *
from .NodeBasic.C_promp import *
from .NodeBasic.C_imgEffect import *
from .NodeBasic.C_type import *
from .NodeBasic.C_basinInput import *
from .NodeExcel.ExcelOP import *
from .NodeExcel.AIagent import *
from .NodeBasic.C_flow import *
from .NodeBasic.C_condition import *


from .NodeCollect.mask_face import *
from .NodeChx.sum_text_yaml import *
from .NodeChx.edit_imge import*




from .NodeBasic.C_test import *

#-load------------------------------------------#



NODE_CLASS_MAPPINGS= {



#-------------------------------------------------------N
"sum_load_adv": sum_load_adv,   
"sum_editor": sum_editor,                      
"sum_Ksampler": sum_Ksampler,  
"chx_input_data": chx_input_data, 

"load_Nanchaku":load_Nanchaku,
"load_GGUF": UnetLoaderGGUF2,
#-------------------------------------------------------S

"Data_Highway":Data_Highway,
"Data_bus_chx":Data_bus_chx,
"Data_basic": Data_basic,                     
"Data_select": Data_select,
"Data_chx_Merge":Data_chx_Merge,
"Data_preset_save": Data_preset_save,

"sum_latent": sum_latent,     
"sum_lora": sum_lora,
"sum_stack_image": sum_stack_image,     
"sum_stack_Wan": sum_stack_Wan,
"sum_stack_AD": sum_stack_AD,

"sum_stack_Kontext":sum_stack_Kontext,
"sum_stack_QwenEditPlus":sum_stack_QwenEditPlus,

#-sample-------------------------------------------#

"basic_Ksampler_simple": basic_Ksampler_simple,  
"basic_Ksampler_full": basic_Ksampler_full, 
"chx_Ksampler_inpaint": chx_Ksampler_inpaint,        
"sampler_DynamicTileSplit": DynamicTileSplit, 
"sampler_DynamicTileMerge": DynamicTileMerge,
"sampler_enhance": sampler_enhance,

#--------***----------------------------#
"basic_Ksampler_mid": basic_Ksampler_mid,        
"basic_Ksampler_custom": basic_Ksampler_custom,
"basic_Ksampler_adv": basic_Ksampler_adv,
"basic_Ksampler_low_gpu": basic_Ksampler_low_gpu,
"chx_Ksampler_refine": chx_Ksampler_refine,
"chx_ksampler_tile": chx_ksampler_tile,   
"chx_Ksampler_highAndLow":chx_Ksampler_highAndLow,


"chx_Ksampler_dual_paint": chx_Ksampler_dual_paint, 
"chx_Ksampler_mix": chx_Ksampler_mix,
"chx_ksampler_Deforum_sch":chx_ksampler_Deforum_sch,
"basic_KSampler_variant_seed": basic_KSampler_variant_seed,
"sum_create_chx": sum_create_chx,  

#---------control tool------------------------------------------#


"pre_qwen_controlnet": pre_qwen_controlnet,    

"pre_Kontext": pre_Kontext,                           
"pre_Kontext_mul_mask": pre_Kontext_mul,
"pre_Kontext_mul_Image":pre_Kontext_mul_Image,

"pre_controlnet": pre_controlnet,      
"pre_controlnet_union": pre_controlnet_union, 
"pre_inpaint_sum": pre_inpaint_sum,


"pre_mul_Mulcondi": pre_mul_Mulcondi,   
"pre_condi_combine_switch":pre_condi_combine_switch,
"pre_ref_condition": pre_ref_condition,  # New
"pre_MulCondiMode": pre_MulCondiMode,

#---------****------------------
"pre_ic_light_sd15": pre_ic_light_sd15,
"pre_USO": pre_USO, 
"pre_Flex2": pre_Flex2,    #    CATEGORY = "Apt_Preset/chx_tool/😺backup"
"pre_latent_light": pre_latent_light,
"pre_guide": pre_guide,
"pre_sample_data": pre_sample_data,


#-IPA-------------------------------------------#

"chx_IPA_basic": chx_IPA_basic,
"chx_IPA_faceID": chx_IPA_faceID,
"chx_IPA_faceID_adv": chx_IPA_faceID_adv,
"chx_IPA_XL": chx_IPA_XL,
"chx_IPA_adv":chx_IPA_adv,
"chx_IPA_region_combine": chx_IPA_region_combine,
"chx_IPA_apply_combine": chx_IPA_apply_combine,

"chx_YC_LG_Redux": chx_YC_LG_Redux,


"IPA_XL_PromptInjection": IPA_PromptInjection,
"IPA_clip_vision": IPA_clip_vision,



#-stack#-----------------**----------------------------------------------------#

"Stack_latent": Stack_latent,
"Stack_pre_Mark2": Stack_pre_Mark2,

"Stack_Kontext_MulCondi":Stack_Kontext_MulCondi,
"Stack_Kontext_MulImg":Stack_Kontext_MulImg,
"Stack_sample_data": Stack_sample_data,
"Stack_LoRA": Stack_LoRA,
"Stack_IPA": Stack_IPA,
"Stack_text": Stack_text,

"Stack_Redux":Stack_Redux,
"Stack_condi": Stack_condi,
"Stack_adv_CN": Stack_adv_CN,   
"Stack_ControlNet":Stack_ControlNet,
"Stack_CN_union":Stack_CN_union,   
"Stack_inpaint": Stack_inpaint,
"Stack_CN_union3": Stack_CN_union3,



"Stack_VAEDecodeTiled": Stack_VAEDecodeTiled,
"Stack_Ksampler_adv": Stack_Ksampler_adv,
"Stack_Ksampler_basic": Stack_Ksampler_basic,
"Stack_Ksampler_custom": Stack_Ksampler_custom,
"Stack_Ksampler_dual_paint": Stack_Ksampler_dual_paint,
"Stack_Ksampler_highAndLow": Stack_Ksampler_highAndLow,
"Stack_Ksampler_refine": Stack_Ksampler_refine,
"Stack_ksampler_tile":Stack_ksampler_tile,

#-------------wan---------------------------------------

"Stack_WanFunControlToVideo": Stack_WanFunControlToVideo,
"Stack_Wan22FunControlToVideo": Stack_Wan22FunControlToVideo,
"Stack_WanFunInpaintToVideo": Stack_WanFunInpaintToVideo,

"Stack_WanImageToVideo": Stack_WanImageToVideo,
"Stack_WanFirstLastFrameToVideo": Stack_WanFirstLastFrameToVideo,
"Stack_WanVaceToVideo": Stack_WanVaceToVideo,
"Stack_WanAnimateToVideo": Stack_WanAnimateToVideo,

"Stack_WanCameraImageToVideo": Stack_WanCameraImageToVideo,
"Stack_WanTrackToVideo": Stack_WanTrackToVideo,
"Stack_WanSoundImageToVideo": Stack_WanSoundImageToVideo,
"Stack_WanSoundImageToVideoExtend": Stack_WanSoundImageToVideoExtend,
"Stack_WanHuMoImageToVideo": Stack_WanHuMoImageToVideo,
"Stack_WanPhantomSubjectToVideo": Stack_WanPhantomSubjectToVideo,







#--------AD------------------------------------------

#"AD_sch_prompt_preset": AD_sch_prompt_preset,
"AD_sch_adv_CN":AD_sch_adv_CN,
"AD_sch_IPA": AD_sch_IPA,
"AD_sch_prompt_adv": AD_sch_prompt_adv,
"AD_sch_prompt_stack": AD_sch_prompt_stack,
"AD_sch_latent": AD_sch_latent,
"AD_sch_value": AD_sch_value,
"AD_sch_prompt_basic": AD_sch_prompt_basic,
"AD_sch_mask_weigh":AD_sch_mask_weigh,



"AD_sch_image_merge":AD_sch_image_merge,
"AD_DrawSchedule": AD_DrawSchedule,
"AD_slice_Condi": AD_slice_Condi,
"AD_MaskExpandBatch": AD_MaskExpandBatch, 
"AD_ImageExpandBatch": AD_ImageExpandBatch,
"AD_frame_replace": AD_frame_replace,
"AD_pingpong_vedio":AD_pingpong_vedio,
"AD_FILM_VFI":AD_FILM_VFI,


"Amp_drive_value": Amp_drive_value,
"Amp_drive_String": Amp_drive_String,
"Amp_audio_Normalized": Amp_audio_Normalized,
"Amp_drive_mask": Amp_drive_mask,


#--unpack------------------------------------------#
"param_preset_pack": param_preset,
"param_preset_Unpack": Unpack_param,
"Model_Preset_pack":Model_Preset,
"Model_Preset_Unpack": Unpack_Model,

"CN_preset1_pack": CN_preset1,
"CN_preset1_Unpack": Unpack_CN,
"photoshop_preset_pack":photoshop_preset,
"photoshop_preset_Unpack":Unpack_photoshop,
"unpack_box2":unpack_box2,


#-------------view-IO-------------------


"IO_inputfile": IO_inputfile,
"IO_outputfile":IO_outputfile,
"IO_input_any": IO_input_any,
"IO_load_anyimage":IO_load_anyimage,
"IO_save_image": IO_save_image,  
"IO_video_encode": IO_video_encode,
"IO_store_image":IO_store_image,
"IO_image_select": IO_image_select,



"view_Data": view_Data,  #wed
"view_bridge_image": view_bridge_image,  #wed
"view_bridge_Text":view_bridge_Text, #wed
"view_mask": view_mask,
"view_Mask_And_Img": view_Mask_And_Img,

"view_combo": view_combo,#wed
"view_node_Script": view_node_Script,
"view_GetLength": view_GetLength, #wed----utils
"view_GetShape": view_GetShape, #wed----utils
"view_GetWidgetsValues": view_GetWidgetsValues, #wed----utils


#-------------***-------------------
"basicIn_Remap_slide": basicIn_Remap_slide,
"basicIn_color": basicIn_color,        
"basicIn_int": basicIn_int,
"basicIn_float": basicIn_float,
"basicIn_string": basicIn_string,
"basicIn_Scheduler": basicIn_Scheduler,
"basicIn_Sampler": basicIn_Sampler,
"basicIn_Seed": basicIn_Seed,
"IO_inputbasic": IO_inputbasic,
"IO_clear_cache": IO_clear_cache,



#-------------data-------------------

"sch_Prompt":sch_Prompt,
"sch_Value":sch_Value,
"sch_text" :sch_text,
"sch_split_text":sch_split_text,
"sch_mask":sch_mask,
"sch_image":sch_image,



"type_AnyIndex":type_AnyIndex,
"type_AnyCast": type_AnyCast, 

"list_Value":list_Value,
"math_calculate": math_calculate, 


"create_mask_batch": create_mask_batch, #wed
"create_image_batch": create_image_batch, #wed
"create_any_List": create_any_List,#wed
"create_any_batch": create_any_batch,  #wed


#-------------**-------------------

"list_ListSlice": list_Slice, 
"list_MergeList": list_Merge, #wed   
"list_num_range": list_num_range,
"batch_BatchSlice": BatchSlice, 
"batch_MergeBatch": MergeBatch, #wed
"pack_Pack": Pack, #wed
"pack_Unpack": Unpack, #wed
"math_Remap_data": math_Remap_data,  





#-----------math---type-------------------


#---------------model--------

"model_tool_assy":model_tool_assy,
"latent_chx_noise": latent_chx_noise,
"latent_Image2Noise": latent_Image2Noise,
"chx_latent_adjust": chx_latent_adjust,



#----------------image------------------------

"Image_pad_keep": Image_pad_keep,  
"Image_pad_adjust": Image_pad_adjust,  
"Image_pad_adjust_restore": Image_pad_adjust_restore,   #TEST----------------------



#-------------------------------------------------------------
"Image_solo_crop": Image_solo_crop,  
"Image_solo_stitch": Image_solo_stitch,    
"Image_transform_crop":Image_transform_crop,  

"Image_Channel_Apply": Image_Channel_Apply,
"Image_Solo_data":Image_Solo_data,
"Image_smooth_blur": Image_smooth_blur,    


"Image_target_adjust":Image_target_adjust,
"Image_Resize_sum": Image_Resize_sum,    
"Image_Resize_sum_restore":Image_Resize_sum_restore,     
"Image_Resize_sum_data":Image_Resize_sum_data,
"Image_Upscaletile": Image_Upscaletile,    
"Image_Resize_longsize": Image_Resize_longsize,



"Image_transform_layer":Image_transform_layer,   
"Image_Pair_Merge": Image_Pair_Merge,  
"Image_Pair_crop": Image_Pair_crop, 
"Image_CnMapMix":Image_CnMapMix,    #TEST----------------------
"Image_Detail_HL_frequencye": Image_Detail_HL_frequencye,

#-------------------------------------------------------------

"color_adjust_HSL": color_adjust_HSL,
"color_adjust_HDR": color_adjust_HDR,
"color_match_adv":color_match_adv,


#---****------------------

"Image_batch_composite": Image_batch_composite,
"color_OneColor_replace": color_OneColor_replace,
"color_OneColor_keep": color_OneColor_keep,     
  
"color_tool": color_tool,
"color_balance_adv": color_balance_adv,
#"color_selector":color_selector,   # bug
"Image_tensor_Converter":Image_tensor_Converter,   #TEST----------------------



#----------------imgEffect--------

"create_lineGradient": create_lineGradient,
"create_RadialGradient": create_RadialGradient,
"create_mulcolor_img": create_mulcolor_img,   

"stack_Mask2color": stack_Mask2color,
"img_effect_Load": img_effect_Load,

"lay_edge_cut": lay_edge_cut,   
"lay_text_sum":lay_text_sum,
"lay_image_grid_note": lay_image_grid_note,
"lay_mul_image":lay_mul_image,


"texture_Offset": texture_Offset,
"texture_create": texture_create,
"texture_Ksampler": texture_Ksampler,
"texture_apply": texture_apply,


#-----------------mask----------------------
"create_Mask_visual_tag":create_Mask_visual_tag,     
"create_Mask_match_shape": create_Mask_match_shape,   

"Image_matchByMask": Image_matchByMask,

"Mask_image2mask": Mask_image2mask,  
"Mask_Remove_bg": Mask_Remove_bg,
"Mask_split_mulMask":Mask_split_mulMask,   
"Mask_splitMask_by_color": Mask_splitMask_by_color,  


"Mask_transform_sum":Mask_transform_sum, 
"Mask_simple_adjust":Mask_simple_adjust,

#----------***------------------
"create_mask_solo": create_mask_solo,     
"create_AD_mask": create_AD_mask,
"create_mask_array": create_mask_array,  
"Mask_Detect_label": Mask_Detect_label,
"Mask_face_detect": Mask_face_detect,
"Mask_math": Mask_math,
"Mask_splitMask":Mask_splitMask,          




#----------prompt----------------
"excel_qwen_font":excel_qwen_font,
"excel_qwen_artistic":excel_qwen_artistic,    #N------------     
"excel_VedioPrompt":excel_VedioPrompt,       #N------------
"excel_roles":excel_roles,  
"excel_object":excel_object,  
"excel_Prompter":excel_Prompter,       


"excel_read_easy":excel_read_easy,
"excel_write_data_easy":excel_write_data_easy,
"excel_insert_image_easy":excel_insert_image_easy,
"excel_search_data":excel_search_data,
"excel_row_diff":excel_row_diff,
"excel_column_diff":excel_column_diff,         


"text_mul_Split":text_mul_Split,
"text_mul_Join":text_mul_Join,
"text_mul_replace": text_mul_replace,
"text_mul_remove": text_mul_remove,
"text_list_combine ": text_list_combine ,  
 

"text_stack_wildcards": text_stack_wildcards,
"text_sum": text_sum,#web
"AI_GLM4":AI_GLM4,

#---****------------------
"text_SuperPrompter": text_SuperPrompter,
"text_free_wildcards": text_free_wildcards,
"AI_Ollama": AI_Ollama,
"excel_insert_image":excel_insert_image,
"excel_write_data":excel_write_data,
"excel_read":excel_read,
"excel_imgEditor_helper":excel_imgEditor_helper,  

#------------------------流程相关-------------------------


"flow_judge":flow_judge,
"flow_auto_pixel":flow_auto_pixel, 
"flow_switch":flow_switch,
"flow_sch_control":flow_sch_control,
"flow_low_gpu":flow_low_gpu,
"flow_case_tentor":flow_case_tentor,


#region------------------------准备废弃-------------------------
"type_BasiPIPE": type_BasiPIPE, #(Deprecated)
"load_FLUX": load_FLUX,   #TITLE = "load_FLUX (Deprecated)"    CATEGORY = "Apt_Preset/🚫Deprecated/🚫"
"load_basic": load_basic, #(Deprecated)
"load_SD35": load_SD35,   #(Deprecated)
"lay_imgCanvas":lay_imgCanvasNode, #(Deprecated)
"Image_Resize2": Image_Resize2,#(Deprecated)
"chx_Ksampler_Kontext": chx_Ksampler_Kontext,   #(Deprecated)
"chx_Ksampler_Kontext_adv": chx_Ksampler_Kontext_adv,  #(Deprecated)
"chx_Ksampler_Kontext_inpaint": chx_Ksampler_Kontext_inpaint,  #(Deprecated)
"sum_stack_all": sum_stack_all,#(Deprecated)
"stack_sum_pack": stack_sum_pack,#(Deprecated)
"IO_adjust_image": IO_adjust_image,#(Deprecated)

"type_Image_List2Batch":type_Image_List2Batch,#(Deprecated)
"type_Image_Batch2List":type_Image_Batch2List,#(Deprecated)
"type_Mask_Batch2List":type_Mask_Batch2List,#(Deprecated)
"type_Mask_List2Batch":type_Mask_List2Batch,#(Deprecated)
"type_ListToBatch": type_ListToBatch, #(Deprecated)
"type_BatchToList": type_BatchToList,#(Deprecated)
"Data_sampleData": Data_sampleData,#(Deprecated)
"Data_presetData":Data_presetData,#(Deprecated)
"chx_StyleModelApply":chx_StyleModelApply,#(Deprecated)
"chx_Style_Redux":chx_Style_Redux,#(Deprecated)

"view_latent": view_LatentAdvanced,#(Deprecated)

"img_effect_CircleWarp": img_effect_CircleWarp,#(Deprecated)
"img_effect_Stretch": img_effect_Stretch,#(Deprecated)
"img_effect_WaveWarp": img_effect_WaveWarp,#(Deprecated)
"img_effect_Liquify": img_effect_Liquify,#(Deprecated)
"model_adjust_color": model_adjust_color,#(Deprecated)
"lay_ImageGrid": lay_ImageGrid,#(Deprecated)
"lay_MaskGrid": lay_MaskGrid,#(Deprecated)  CATEGORY = "Apt_Preset/Deprecated"
"latent_ratio": latent_ratio,#(Deprecated)
"pre_QwenEdit":pre_QwenEdit,   #(Deprecated)
"sum_stack_QwenEdit":sum_stack_QwenEdit, #(Deprecated)
"model_Style_Align":model_Style_Align,#(Deprecated)
"chx_Ksampler_VisualStyle": chx_Ksampler_VisualStyle, #(Deprecated)
"lay_image_mul":lay_image_mul,#(Deprecated)
"chx_Ksampler_dual_area": chx_Ksampler_dual_area,#(Deprecated)
"Image_batch_select": Image_batch_select,#(Deprecated)



#------------------------隐藏节点-------------------------

#"model_diff_inpaint": model_diff_inpaint,
#"model_Regional": model_Regional,
#"Apply_IPA": Apply_IPA,
#"Apply_IPA_SD3":Apply_IPA_SD3,
#"Apply_adv_CN": Apply_adv_CN,
#"Apply_condiStack": Apply_condiStack,
#"Apply_textStack": Apply_textStack,
#"Apply_Redux": Apply_Redux,
#"Apply_latent": Apply_latent,
#"Apply_CN_union":Apply_CN_union,
#"Apply_ControlNetStack": Apply_ControlNetStack,
#"Apply_LoRAStack": Apply_LoRAStack,
#"text_CSV_load": text_CSV_load,
#"lay_compare_img": lay_compare_img,
#"lay_images_free_layout":lay_images_free_layout,
#"lay_iamge_conbine":lay_iamge_conbine,

#endregion------------------------隐藏节点-------------------------







}







NODE_DISPLAY_NAME_MAPPINGS = {
"Data_Highway": "数据_高速通道",

}
