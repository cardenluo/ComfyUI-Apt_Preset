# flow_stage 桥张量读取与查看

保留两个节点，每个输出可分别连接查看节点：

`flow_stage_bridge_decode_range → bridge_latent_data1 / bridge_latent_data2 → View_bridge_tentor.bridge_latent`

## 按 ID 读取

`flow_stage_bridge_decode_range` 位于 `Apt_Preset/flow`。

- `run_id`：已有阶段任务的ID。
- `start_id` / `end_id`：从1开始的闭区间；相等时读取单段，倒序填写时仍按ID升序读取。
- 无需选择channel，同时读取两个通道。
- 输出 `bridge_latent_data1` 和 `bridge_latent_data2`：各自保留该通道每段原始载荷和元数据的LATENT列表；此节点不加载VAE、不进行图像/音频解码。
- 两路独立按ID升序输出，缺失数据分别跳过；某一路完全没有数据时输出空列表，不影响另一路。

例如 `start_id=3, end_id=3` 读取第3段的data1和data2；`start_id=1, end_id=5` 分别读取第1至5段两个通道的已有数据。

文件来源是 `<output>/.apt_stage_bridge/<run_id对应目录>/`：优先读取已提交的 `stage_{ID-1:05d}.safetensors`（data1）或 `stage_{ID-1:05d}_2.safetensors`（data2），缺失时读取对应检查点。缺失阶段跳过，文件损坏则报错；文件路径、修改时间或大小变化会使读取缓存失效。

## 解码与合并

`View_bridge_tentor` 保留原名称及 `Apt_Preset/PreView` 菜单路径。

- 唯一数据输入 `bridge_latent`，既可连接普通单个LATENT，也可接上述阶段列表。
- `vae`、`audio_vae`、`fps` 在此设置；VAE选择None时跳过对应latent解码。
- 单段按原方式解码；多段依次解码、应用各段裁剪和调色元数据，再按输入顺序合并。
- 输出保留 `image / video / audio / mask / text`，没有对应数据时阻断该路输出。
- 已保存的图像、视频、音频、遮罩和文本载荷按类型直接读取，不重复VAE解码。
- 多段视频以第一段的分辨率、帧率和位深为准。音频以第一份有效音频的格式为准，各段时长对齐；没有音频的分段填充静音。
- 多段文本以换行连接。全视频输入的image输出复用合并视频帧，audio输出使用其对齐后的音轨。

多段解码和合并仍会保留帧数据，长视频的内存占用取决于总帧数；本次节点整合不改变采样或提供流式解码。

## 旧工作流

`flow_stage_bridge_decode` 和 `flow_stage_bridge_merge` 不再注册。旧的单段读取改用起止ID相同的范围节点；旧的五路列表连接改接对应的 `bridge_latent_data1` 或 `bridge_latent_data2` 输出，VAE/FPS移到查看节点。原来的单LATENT `View_bridge_tentor` 连接方式保持可用。移除channel选择后，旧范围节点需重新添加并连接对应通道输出。
