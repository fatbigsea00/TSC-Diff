# 训练样本示例 (Sample Illustrations)

本目录提供每个类别的少量训练样本示例,用于展示数据格式。完整数据集将在论文录用后公开。

每个类别(`aircraft` / `ship` / `artificial fishing reef`)各含 3 组样本,每组包含:

- `<id>.jpg` — 原始侧扫声呐图像(Stage 2 的生成目标 / ground truth)
- `<id>_mask.png` — 对应的分割条件图(segmentation map),由 labelme 标注的多边形渲染得到,即 Stage 2 ControlNet 的输入条件

分割图的颜色映射(见 [step2/utils/controlnet_dataset.py](../../../step2/utils/controlnet_dataset.py) 的 `CATEGORY_COLORS`):

| 区域 | 颜色 (RGB) |
|------|-----------|
| aircraft | (0, 0, 255) 蓝 |
| ship | (0, 255, 0) 绿 |
| human | (255, 0, 0) 红 |
| artificial fishing reef | (255, 255, 0) 黄 |
| shadow | (128, 128, 128) 灰 |
| background | (64, 64, 64) 深灰 |

> 注:实际训练时分割图由标注 JSON 在线渲染,此处的 `_mask.png` 仅为可视化示例。
