
**ComfyUI-Apt_Preset** is a node package designed to simplify workflows by merging commonly used basic nodes into new nodes, reducing connections between basic nodes, and simplifying the overall workflow.
**ComfyUI-Apt_Preset**是一个简化工作流的节点包，将常用的基本节点合并到一个新节点，减少连线。

## 👨🏻‍🎨 Usage Guide使用指南

1. The main loader sum_load integrates 4 loading modes: basic, clip, flux, and sd 3.5
   主加载器sum_load集成了4种加载模式：basic, clip, flux, and sd 3.5

| Basic  | Load checkpoint  + clip set last layer |
| ------ | -------------------------------------- |
| Clip   | Load checkpoint   + load clip          |
| Flux   | Load diffusion model +DualCLIPLoader   |
| SD 3.5 | Load diffusion model +TripleCLIPLoader |

![image](https://github.com/user-attachments/assets/73d64eb6-fc41-44e7-9766-dce8f9ab74e6)


Flux Mode (Arbitrary Transfer)
![image](https://github.com/user-attachments/assets/e6ebec1f-b000-42f5-8c0e-cadc3a6d437c)


2. Preset can save sum_load presets (requires restart to take effect),Save path: ComfyUI-Apt_Preset\presets
   Preset可以保存sum_load预设（需要重启才能生效），保存路径：ComfyUI-Apt_Preset\presets
![image](https://github.com/user-attachments/assets/01d5793f-5703-420b-b57a-6e120527bc19)



## 👨🏻‍🔧 Installation

Clone the repository to the **custom_nodes** directory and install dependencies
```
#1. git下载
git clone https://github.com/cardenluo/ComfyUI-Apt_Preset.git
#2. 安装依赖
双击install.bat安装依赖
```


## Reference Node Packages参考节点包

Declaration: I highly respect the efforts of the original authors. Open source is not easy. I just did some integration.

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) -The most powerful and modular diffusion model GUI

[ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack) - helps to conveniently enhance images through Detector, Detailer

[ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) - This repository offers various extension nodes for ComfyUI

**[ComfyUI_mittimiLoadPreset2](https://github.com/mittimi/ComfyUI_mittimiLoadPreset2)**-easily switch between models and prompts by saving presets

**[ComfyUI-EasyDeforum](https://github.com/Chan-0312/ComfyUI-EasyDeforum)**- easy deforum

[ComfyUI_VisualStylePrompting](https://github.com/ExponentialML/ComfyUI_VisualStylePrompting) -Visual Style Prompting with Swapping Self-Attention

[rgthree-comfy](https://github.com/rgthree/rgthree-comfy) - _Making ComfyUI more comfortable!_


## Disclaimer免责声明

This open-source project and its contents are provided "AS IS" without any express or implied warranties, including but not limited to warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

Users are responsible for ensuring compliance with all applicable laws and regulations in their respective jurisdictions when using this software or publishing content generated by it. The authors and copyright holders are not responsible for any violations of laws or regulations by users in their respective locations.
