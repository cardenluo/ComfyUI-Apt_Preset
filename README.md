
## **Make the workflow easier**

## 👨update record更新记录

1、2025.3.29  add node “sum_stack_all”  



## 👨🏻‍🎨 Usage Guide使用指南
1.The main loader sum\_load integrates 4 loading modes: basic, clip, flux, and sd 3.5, all parameters can be loaded directly from the preset.
主加载器sum\_load集成了4种加载模式：basic, clip, flux, and sd 3.5，所有参数可以从 preset 中直接加载.

| **Basic**  | **Load checkpoint + clip set last layer**  |
| ---------- | ------------------------------------------ |
| **Clip**   | **Load checkpoint + load clip**            |
| **Flux**   | **Load diffusion model +DualCLIPLoader**   |
| **SD 3.5** | **Load diffusion model +TripleCLIPLoader** |
****

![image](https://github.com/user-attachments/assets/73d64eb6-fc41-44e7-9766-dce8f9ab74e6)



Examlpe: Flux Mode (Arbitrary Transfer)
![image](https://github.com/user-attachments/assets/15099be1-25d7-42d7-876d-948ef54f99cc)



2. Preset can save sum_load presets (requires restart to take effect),Save path: ComfyUI-Apt_Preset\presets
   Preset可以保存sum_load预设（需要重启才能生效），保存路径：ComfyUI-Apt_Preset\presets
![image](https://github.com/user-attachments/assets/9c68e1d5-a5ee-454b-b343-f94f9e6d47eb)




3. A very simple integrated control node. All controls can be derived from this node.
   一个非常简单的控制节点，所有的控制方法，只需从该节点引出即可
![image](https://github.com/user-attachments/assets/6448859c-f968-4dc1-b6aa-17e6d416f416)


## 👨🏻‍🔧 Installation

Clone the repository to the **custom_nodes** directory and install dependencies
```
#1. git下载
git clone https://github.com/cardenluo/ComfyUI-Apt_Preset.git
#2. 安装依赖
双击install.bat安装依赖
```
## Reference Node Packages参考节点包

The code of this open-source project has referred to the following code during the development process. We express our sincere gratitude for their contributions in the relevant fields.

| [ComfyUI](https://github.com/comfyanonymous/ComfyUI)                          | [ComfyUI\_VisualStylePrompting](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting) | [ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) |
| ----------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [ComfyUI-EasyDeforum](https://github.com/Chan-0312/ComfyUI-EasyDeforum)   | [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)  | [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)   |
| [rgthree-comfy](https://github.com/rgthree/rgthree-comfy)                     | [ComfyUI\_mittimiLoadPreset2](https://github.com/mittimi/ComfyUI_mittimiLoadPreset2)       | [ComfyUI-Keyframed](https://github.com/dmarx/ComfyUI-Keyframed)      |
| [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) | [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved)      | [ComfyUI-Easy-Use](https://github.com/yolain/ComfyUI-Easy-Use)           |
| [ComfyUI_essentials](https://github.com/cubiq/ComfyUI_essentials)         | [Comfyui__Flux_Style_Adjust](https://github.com/yichengup/Comfyui_Flux_Style_Adjust)           |                                                                          |
|                                                                               |                                                                                                |                                                                          |


## Disclaimer免责声明

This open-source project and its contents are provided "AS IS" without any express or implied warranties, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are responsible for ensuring compliance with all applicable laws and regulations in their respective jurisdictions when using this software or publishing content generated by it. The authors and copyright holders are not responsible for any violations of laws or regulations by users in their respective locations.
