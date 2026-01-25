import os
import torch
import torch.nn as nn
import numpy as np
from typing import Union
from PIL import Image, ImageFilter
import onnxruntime
import folder_paths


try:
    from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
    REMOVER_AVAILABLE = True  
except ImportError:
    SegformerImageProcessor = None
    AutoModelForSemanticSegmentation = None
    REMOVER_AVAILABLE = False 


try:
    from torchvision import transforms
    REMOVER_AVAILABLE = True  
except ImportError:
    transforms = None
    REMOVER_AVAILABLE = False 







# 公共工具函数
def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32)/255.0)[None,]

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255.*image.cpu().numpy(),0,255).astype(np.uint8))

def image2mask(image: Image.Image) -> torch.Tensor:
    if isinstance(image, Image.Image):
        image = pil2tensor(image)
    return image.squeeze()[...,0]

def mask2image(mask: torch.Tensor) -> Image.Image:
    if len(mask.shape)==2:
        mask = mask.unsqueeze(0)
    return tensor2pil(mask)

def RGB2RGBA(image: Image.Image, mask: Union[Image.Image, torch.Tensor]) -> Image.Image:
    if isinstance(mask, torch.Tensor):
        mask = mask2image(mask)
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.Resampling.LANCZOS)
    return Image.merge('RGBA',(*image.convert('RGB').split(),mask.convert('L')))

def hex_to_rgba(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color)==6:
        r,g,b = int(hex_color[0:2],16),int(hex_color[2:4],16),int(hex_color[4:6],16)
        a = 255
    elif len(hex_color)==8:
        r,g,b,a = int(hex_color[0:2],16),int(hex_color[2:4],16),int(hex_color[4:6],16),int(hex_color[6:8],16)
    else:
        raise ValueError("Invalid color format")
    return (r,g,b,a)

# 定义背景颜色映射表
BACKGROUND_COLORS = {
    "white": "#FFFFFF",
    "black": "#000000",
    "green": "#00FF00",
    "red": "#FF0000",
    "blue": "#0000FF",
    "gray": "#808080"
}

# 公共配置
device = "cuda" if torch.cuda.is_available() else "cpu"
# 定义模型根目录
MODEL_ROOT_DIR = os.path.join(folder_paths.models_dir, "Apt_File")

# 基础分割类（提取公共逻辑）
class BaseSegment:
    def __init__(self):
        self.model = None
        self.processor = None
        self.cache_dir = ""
        self.model_file = ""

    def check_model_cache(self):
        raise NotImplementedError
    
    def clear_model(self):
        if self.model is not None:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()

    # 移除自动下载模型的方法
    def download_model_files(self, *args, **kwargs):
        return False, "自动下载模型功能已禁用，请手动下载模型文件到指定目录"

    def process_mask_visualization(self, batch_masks):
        mask_images = []
        for mask_tensor in batch_masks:
            mask_image = mask_tensor.reshape((-1,1,mask_tensor.shape[-2],mask_tensor.shape[-1])).movedim(1,-1).expand(-1,-1,-1,3)
            mask_images.append(mask_image)
        return torch.cat(mask_images,dim=0)

    def get_background_color(self, background):
        """获取背景颜色的RGBA值"""
        if background == "Alpha":
            return None
        return BACKGROUND_COLORS.get(background, "#222222")

# 人脸分割节点
class Mask_FaceSegment(BaseSegment):
    def __init__(self):
        super().__init__()
        # 修改模型路径
        self.cache_dir = os.path.join(MODEL_ROOT_DIR, "segformer_face")
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = ["Skin","Nose","Eyeglasses","Left-eye","Right-eye","Left-eyebrow","Right-eyebrow","Left-ear","Right-ear","Mouth","Upper-lip","Lower-lip","Hair","Earring","Neck"]
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                **{cls_name:("BOOLEAN",{"default":False}) for cls_name in available_classes},
                "background":(["Alpha","white", "black", "green", "red", "blue", "gray"],{"default":"Alpha"})
            }
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("IMAGE","MASK",)
    FUNCTION = "segment_face"
    CATEGORY = "Apt_Preset/mask"

    def check_model_cache(self):
        if not os.path.exists(self.cache_dir):
            error_msg = f"""人脸分割模型目录不存在！
请手动下载人脸分割模型文件，并放到以下目录：
{self.cache_dir}
模型地址：https://huggingface.co/1038lab/segformer_face
需要的文件：config.json、model.safetensors、preprocessor_config.json"""
            return False, error_msg
        
        required_files = ['config.json','model.safetensors','preprocessor_config.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.cache_dir,f))]
        if missing_files:
            error_msg = f"""人脸分割模型文件缺失！
缺失文件：{', '.join(missing_files)}
请从以下地址下载缺失文件并放到目录：{self.cache_dir}
模型地址：https://huggingface.co/1038lab/segformer_face"""
            return False, error_msg
        return True,"Cache verified"

    def segment_face(self, images, background="Alpha",**class_selections):
        try:
            cache_status,message = self.check_model_cache()
            if not cache_status:
                raise RuntimeError(message)
            
            if self.processor is None:
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.to(device)

            class_map = {"Background":0,"Skin":1,"Nose":2,"Eyeglasses":3,"Left-eye":4,"Right-eye":5,"Left-eyebrow":6,"Right-eyebrow":7,"Left-ear":8,"Right-ear":9,"Mouth":10,"Upper-lip":11,"Lower-lip":12,"Hair":13,"Hat":14,"Earring":15,"Necklace":16,"Neck":17,"Clothing":18}
            selected_classes = [name for name,selected in class_selections.items() if selected] or ["Skin","Nose","Left-eye","Right-eye","Mouth"]
            invalid_classes = [cls for cls in selected_classes if cls not in class_map]
            if invalid_classes:
                raise ValueError(f"Invalid classes: {', '.join(invalid_classes)}")

            transform_image = transforms.Compose([transforms.Resize((512,512)),transforms.ToTensor()])
            batch_tensor, batch_masks = [], []
            
            for image in images:
                orig_image = tensor2pil(image)
                w,h = orig_image.size
                input_tensor = transform_image(orig_image)
                if input_tensor.shape[0]==4:
                    input_tensor = input_tensor[:3]
                input_tensor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(input_tensor).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits.cpu()
                    upsampled_logits = nn.functional.interpolate(logits,size=(h,w),mode="bilinear",align_corners=False)
                    pred_seg = upsampled_logits.argmax(dim=1)[0]

                    combined_mask = None
                    for class_name in selected_classes:
                        mask = (pred_seg == class_map[class_name]).float()
                        combined_mask = mask if combined_mask is None else torch.clamp(combined_mask + mask,0,1)

                    mask_image = Image.fromarray((combined_mask.numpy()*255).astype(np.uint8))
                    rgba_image = RGB2RGBA(orig_image,mask_image)
                    
                    if background == "Alpha":
                        result_image = pil2tensor(rgba_image)
                    else:
                        # 获取对应的背景颜色
                        bg_color = self.get_background_color(background)
                        bg_image = Image.new('RGBA',orig_image.size,hex_to_rgba(bg_color))
                        composite_image = Image.alpha_composite(bg_image,rgba_image)
                        result_image = pil2tensor(composite_image.convert('RGB'))

                    batch_tensor.append(result_image)
                    batch_masks.append(pil2tensor(mask_image))

            mask_image_output = self.process_mask_visualization(batch_masks)
            batch_tensor = torch.cat(batch_tensor,dim=0)
            batch_masks = torch.cat(batch_masks,dim=0)
            
            return (batch_tensor,batch_masks,mask_image_output)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Face segmentation error: {str(e)}")
        finally:
            if self.model is not None and not self.model.training:
                self.clear_model()

# 衣物分割节点
class Mask_ClothesSegment(BaseSegment):
    def __init__(self):
        super().__init__()
        # 修改模型路径
        self.cache_dir = os.path.join(MODEL_ROOT_DIR, "segformer_clothes")
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = ["Hat","Hair","Face","Sunglasses","Upper-clothes","Skirt","Dress","Belt","Pants","Left-arm","Right-arm","Left-leg","Right-leg","Bag","Scarf","Left-shoe","Right-shoe"]
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                **{cls_name:("BOOLEAN",{"default":False}) for cls_name in available_classes},
                "background":(["Alpha","white", "black", "green", "red", "blue", "gray"],{"default":"Alpha"})
            }
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("IMAGE","MASK",)
    FUNCTION = "segment_clothes"
    CATEGORY = "Apt_Preset/mask"

    def check_model_cache(self):
        if not os.path.exists(self.cache_dir):
            error_msg = f"""衣物分割模型目录不存在！
请手动下载衣物分割模型文件，并放到以下目录：
{self.cache_dir}
模型地址：https://huggingface.co/1038lab/segformer_clothes
需要的文件：config.json、model.safetensors、preprocessor_config.json"""
            return False, error_msg
        
        required_files = ['config.json','model.safetensors','preprocessor_config.json']
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(self.cache_dir,f))]
        if missing_files:
            error_msg = f"""衣物分割模型文件缺失！
缺失文件：{', '.join(missing_files)}
请从以下地址下载缺失文件并放到目录：{self.cache_dir}
模型地址：https://huggingface.co/1038lab/segformer_clothes"""
            return False, error_msg
        return True,"Cache verified"

    def segment_clothes(self, images, background="Alpha",**class_selections):
        try:
            cache_status,message = self.check_model_cache()
            if not cache_status:
                raise RuntimeError(message)
            
            if self.processor is None:
                self.processor = SegformerImageProcessor.from_pretrained(self.cache_dir)
                self.model = AutoModelForSemanticSegmentation.from_pretrained(self.cache_dir)
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False
                self.model.to(device)

            class_map = {"Background":0,"Hat":1,"Hair":2,"Sunglasses":3,"Upper-clothes":4,"Skirt":5,"Pants":6,"Dress":7,"Belt":8,"Left-shoe":9,"Right-shoe":10,"Face":11,"Left-leg":12,"Right-leg":13,"Left-arm":14,"Right-arm":15,"Bag":16,"Scarf":17}
            selected_classes = [name for name,selected in class_selections.items() if selected] or ["Upper-clothes"]

            transform_image = transforms.Compose([transforms.Resize((1024,1024)),transforms.ToTensor()])
            batch_tensor, batch_masks = [], []
            
            for image in images:
                orig_image = tensor2pil(image)
                w,h = orig_image.size
                input_tensor = transform_image(orig_image)
                if input_tensor.shape[0]==4:
                    input_tensor = input_tensor[:3]
                input_tensor = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(input_tensor).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    logits = outputs.logits.cpu()
                    upsampled_logits = nn.functional.interpolate(logits,size=(h,w),mode="bilinear",align_corners=False)
                    pred_seg = upsampled_logits.argmax(dim=1)[0]

                    combined_mask = None
                    for class_name in selected_classes:
                        mask = (pred_seg == class_map[class_name]).float()
                        combined_mask = mask if combined_mask is None else torch.clamp(combined_mask + mask,0,1)

                    mask_image = Image.fromarray((combined_mask.numpy()*255).astype(np.uint8))
                    rgba_image = RGB2RGBA(orig_image,mask_image)
                    
                    if background == "Alpha":
                        result_image = pil2tensor(rgba_image)
                    else:
                        # 获取对应的背景颜色
                        bg_color = self.get_background_color(background)
                        bg_image = Image.new('RGBA',orig_image.size,hex_to_rgba(bg_color))
                        composite_image = Image.alpha_composite(bg_image,rgba_image)
                        result_image = pil2tensor(composite_image.convert('RGB'))

                    batch_tensor.append(result_image)
                    batch_masks.append(pil2tensor(mask_image))

            mask_image_output = self.process_mask_visualization(batch_masks)
            batch_tensor = torch.cat(batch_tensor,dim=0)
            batch_masks = torch.cat(batch_masks,dim=0)
            
            return (batch_tensor,batch_masks,mask_image_output)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Clothes segmentation error: {str(e)}")
        finally:
            if self.model is not None and not self.model.training:
                self.clear_model()

# 人体分割节点
class Mask_BodySegment(BaseSegment):
    def __init__(self):
        super().__init__()
        # 修改模型路径
        self.cache_dir = os.path.join(MODEL_ROOT_DIR, "body_segment")
        self.model_file = "deeplabv3p-resnet50-human.onnx"
    
    @classmethod
    def INPUT_TYPES(cls):
        available_classes = ["Hair","Glasses","Top-clothes","Bottom-clothes","Torso-skin","Face","Left-arm","Right-arm","Left-leg","Right-leg","Left-foot","Right-foot"]
        return {
            "required": {"images": ("IMAGE",)},
            "optional": {
                **{cls_name:("BOOLEAN",{"default":False}) for cls_name in available_classes},
                "background":(["Alpha","white", "black", "green", "red", "blue", "gray"],{"default":"Alpha"})
            }
        }

    RETURN_TYPES = ("IMAGE","MASK",)
    RETURN_NAMES = ("IMAGE","MASK",)
    FUNCTION = "segment_body"
    CATEGORY = "Apt_Preset/mask"

    def check_model_cache(self):
        model_path = os.path.join(self.cache_dir,self.model_file)
        if not os.path.exists(self.cache_dir):
            error_msg = f"""人体分割模型目录不存在！
请手动下载人体分割模型文件，并放到以下目录：
{self.cache_dir}
模型地址：https://huggingface.co/Metal3d/deeplabv3p-resnet50-human
需要的文件：deeplabv3p-resnet50-human.onnx"""
            return False, error_msg
        
        if not os.path.exists(model_path):
            error_msg = f"""人体分割模型文件缺失！
缺失文件：{self.model_file}
请从以下地址下载缺失文件并放到目录：{self.cache_dir}
模型地址：https://huggingface.co/Metal3d/deeplabv3p-resnet50-human"""
            return False, error_msg
        return True,"Cache verified"

    def segment_body(self, images, background="Alpha",**class_selections):
        try:
            cache_status,message = self.check_model_cache()
            if not cache_status:
                raise RuntimeError(message)
            
            if self.model is None:
                self.model = onnxruntime.InferenceSession(os.path.join(self.cache_dir,self.model_file))

            class_map = {"Hair":2,"Glasses":4,"Top-clothes":5,"Bottom-clothes":9,"Torso-skin":10,"Face":13,"Left-arm":14,"Right-arm":15,"Left-leg":16,"Right-leg":17,"Left-foot":18,"Right-foot":19}
            selected_classes = [name for name,selected in class_selections.items() if selected] or ["Face","Hair","Top-clothes","Bottom-clothes"]

            batch_tensor, batch_masks = [], []
            
            for image in images:
                orig_image = tensor2pil(image)
                w,h = orig_image.size
                input_image = orig_image.resize((512,512))
                input_array = np.array(input_image).astype(np.float32)/127.5 - 1
                input_array = np.expand_dims(input_array,axis=0)

                input_name = self.model.get_inputs()[0].name
                output_name = self.model.get_outputs()[0].name
                result = self.model.run([output_name],{input_name: input_array})

                result = np.array(result[0])
                pred_seg = result.argmax(axis=3).squeeze(0)

                combined_mask = np.zeros_like(pred_seg,dtype=np.float32)
                for class_name in selected_classes:
                    mask = (pred_seg == class_map[class_name]).astype(np.float32)
                    combined_mask = np.clip(combined_mask + mask,0,1)

                mask_image = Image.fromarray((combined_mask*255).astype(np.uint8)).resize((w,h),Image.Resampling.LANCZOS)
                rgba_image = RGB2RGBA(orig_image,mask_image)
                
                if background == "Alpha":
                    result_image = pil2tensor(rgba_image)
                else:
                    # 获取对应的背景颜色
                    bg_color = self.get_background_color(background)
                    bg_image = Image.new('RGBA',orig_image.size,hex_to_rgba(bg_color))
                    composite_image = Image.alpha_composite(bg_image,rgba_image)
                    result_image = pil2tensor(composite_image.convert('RGB'))

                batch_tensor.append(result_image)
                batch_masks.append(pil2tensor(mask_image))

            mask_image_output = self.process_mask_visualization(batch_masks)
            batch_tensor = torch.cat(batch_tensor,dim=0)
            batch_masks = torch.cat(batch_masks,dim=0)
            
            return (batch_tensor,batch_masks,)

        except Exception as e:
            self.clear_model()
            raise RuntimeError(f"Body segmentation error: {str(e)}")
        finally:
            self.clear_model()






