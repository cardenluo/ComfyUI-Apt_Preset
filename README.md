

<img width="6085" height="413" alt="11123" src="https://github.com/user-attachments/assets/87d0f55b-c3bb-4621-862d-8f65f83815a2" />

# Overview

The current workflow connections are becoming increasingly dense, making them difficult to understand and operate, and there are issues with poor consistency. As the number of saved workflows increases, frequent errors occur during actual use, and the operation methods need to be relearned. To address this, a plugin has been developed to simplify workflows, clarify control concepts, and design nodes around the "Loader ---- Controller ----- Sampler" framework.

## <a href="./README.md">English</a> >> <a href="./README_ZH.md">中文版</a>


# I. Update Record


2025.11.2 Add, input_text, output_text, support reading and storing text in multiple formats
          add sum_ksample Includes the switching of multiple sampling methods

2025.11.24 Add API calls to the Qwen model, AI_Qwen, AI_Qwen_text, 
           or local calls to the Qwen model: AI_Ollama_image, AI_Ollama_text

2025.11.25  Quickly debug and add an update button: view Data, view_mask, view_GetShape, view_Mask_And_Img

# II. Usage Guide

## 1. Universal Loader Sum_load_adv, Supports GGUF Models

1) Model combinations for various workflows are the same as official workflows.

① XL, SD mode: Load model checkpoint or Unet or over model.

② wan2.1, wan2.2, QwenEdit mode: Load model Unet + clip1

③ Flux\Kontext mode: Load model Unet + clip1 + clip2 (Note: In order, not clip1+clip3)

④ SD3.5 mode: Load model Unet+clip1 +clip2+clip3

⑤ Hi-dream mode: Load model Unet+clip1 +clip2+clip3+clip4

2) Override mode over_model and over_clip

① When using over_model, the internal corresponding model or Unet of the loader becomes invalid and will directly output model

② When using over_clip, all internal corresponding clip1,2,3,4 of the loader become invalid and will directly output clip

3) Preset saving, which can save loaded models and sampling methods uniformly

① At least one arbitrary preset must be selected, otherwise an error will occur

② After selecting any preset, all parameters can be modified and take effect arbitrarily.

③ Newly set parameters can be saved as new presets, but ComfyUI needs to be restarted to select and use them

![image](https://github.com/user-attachments/assets/c937203d-6ada-4b58-a882-512290e30dcd)

## 2. Controller Stack: Functional modules with associated control tools centralized together

① General Control image Stack: Universal control for SD, XL, Flux, etc., Ipa style, redux transfer, Union_controlnet, controlnet_adv, inpaint redrawing

② General Control wan Stack: All wan video generation nodes officially supported

③ General Control AD Stack: Animatediff animation generation control, prompt scheduling, CN scheduling, IPA scheduling, etc.

④ General Control Kontext Stack: Multi-image reference, partitioned generation, redux transfer, union_controlnet

⑤ General Control QwenEdit Stack: Multi-image reference, union_controlnet

<img width="2890" height="715" alt="image" src="https://github.com/user-attachments/assets/27e6733d-4d2e-49d3-8249-8a6f86ff883f" />


## 3. Sampler: Rich sampling methods to eliminate repetitive connections

① Basic sampler: Packages ComfyUI's built-in sampler input ports into a single port replacement, with functionality completely consistent with the official version

② Special function sampler: A combination of basic sampler + special functions.

As in the example below, operations like secondary sampling repair and refine enlargement can be achieved in one step:

![image](https://github.com/user-attachments/assets/0c62a1f7-e92f-41bc-a6fb-447b3cc7ea48)

## 4. Utility Nodes: Image, mask, data processing tools

1. Data types: Data conversion, data creation，data operation

<img width="3405" height="1278" alt="image" src="https://github.com/user-attachments/assets/fb743a4b-0406-4a73-b8ae-4e8e0d8e38b2" />


2. Image processing: A powerful image processing combination: pre-processing, intermediate generation, and post-recovery

<img width="2962" height="1149" alt="PixPin_2025-10-01_08-38-57" src="https://github.com/user-attachments/assets/5d46bfde-bfd9-47a2-966b-1de24acca80c" />


3. Mask processing: Creation, conversion, and transformation of masks

<img width="3562" height="1079" alt="image" src="https://github.com/user-attachments/assets/8adf81a2-a777-45b2-9e67-19d85fb66855" />


# III. Installation
Clone the repository to the **custom_nodes** directory and install dependencies

```

#1. Git download
git clone https://github.com/cardenluo/ComfyUI-Apt_Preset.git

#2. Install dependencies
Double-click install.bat to install dependencies

```

Note:

To use the controlNet schedule control feature, please first install [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)

To load GGUF models, please first install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF)

To use the load_Nanchaku node, please first install [ComfyUI-nunchaku](https://github.com/nunchaku-tech/ComfyUI-nunchaku).

## Disclaimer
This open-source project and its contents are provided "AS IS" without any express or implied warranties, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are responsible for ensuring compliance with all applicable laws and regulations in their respective jurisdictions when using this software or publishing content generated by it. The authors and copyright holders are not responsible for any violations of laws or regulations by users in their respective locations.

## Follow Me, Share Methods for Easily Building Workflows
Bilibili: https://space.bilibili.com/2008798642?spm_id_from=333.33.0.0
