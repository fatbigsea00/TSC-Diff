# TSC-Diff

TSC-Diff 是一个面向**侧扫声呐图像生成 / 数据增强**的两阶段扩散模型框架。第一阶段在声呐数据上微调 Stable Diffusion 以学习声呐图像分布;第二阶段训练一个受分割图(segmentation map)控制的 ControlNet,按指定类别与位置生成完整声呐图像。

本仓库为**论文配套代码发布版**。当前版本提供方法的核心代码及部分训练样本示例;论文录用后,我们将公开完整的实验代码与训练数据集。

## 目录结构

```
TSC-Diff/
├── requirements.txt
├── step1/                       # Stage 1: SD 微调 + DFDB
│   ├── utils/train.py
│   ├── utils/dataset.py
│   ├── plugins/DSR.py           # PFA / LFA 增强模块
│   └── run_train_dfdb.bat
└── step2/                       # Stage 2: ControlNet (TSCG)
    ├── models/                  # RBE / MapCA / MaskCA / 兼容回退
    ├── utils/train_controlnet.py
    ├── utils/inference_controlnet.py
    ├── utils/controlnet_dataset.py
    ├── run_train.bat            # 训练 + 推理
    └── run_inference.bat        # 仅推理
```

## 环境安装

```bash
conda create -n tsc-diff python=3.10 -y
conda activate tsc-diff
pip install -r requirements.txt
```

## 数据集

本文涉及两个侧扫声呐数据集:**SCTD**(方法训练/推理所用)与 **SSTar**(本文构建的侧扫声呐目标数据集)。两者均仅在本仓库提供**少量样本实例**,**完整数据集、类别定义、标注与划分等信息将在论文录用后公开。**

### 1. SCTD

TSC-Diff 两阶段训练与推理直接使用的声呐数据集,含 `aircraft / ship / human / artificial fishing reef` 四类,每个样本由「原始图像 + labelme 分割标注」组成,标注经渲染得到 ControlNet 所需的分割条件图。

> 本发布版提供样本示例(见 [dataset/SCTD/samples/](dataset/SCTD/samples/),每类 3 组「原图 + 分割条件图」)。

完整数据集预期目录结构:

```
dataset/SCTD/
├── samples/                     # 已提供:每类少量样本示例(图 + 掩码)
├── aircraft/                    # 待补充:各类别图像 + 标注
├── ship/
├── human/
├── artificial fishing reef/
├── metadata.jsonl               # 图像-提示词-掩码元信息
└── split_4cat_70_plus_afr.json  # train/test 划分
```

### 2. SSTar (Side-Scan Sonar Targets)

本文构建的侧扫声呐目标数据集,采集自真实侧扫声呐作业数据,覆盖多种典型水下目标。完整数据集共 **489 张**图像,按目标类型分为 **5 类**:

| 类别(原始编码) | 完整数据量 | 本仓库样本数 |
|------------------|-----------|--------------|
| RGYJ | 389 | 4 |
| SXJS | 54 | 4 |
| MTZ | 29 | 4 |
| JZX | 15 | 4 |
| shipwreck(沉船) | 2 | 2 |
| **合计** | **489** | **18** |

> 本发布版仅提供少量样本实例(见 [dataset/SSTar/samples/](dataset/SSTar/samples/),详见 [dataset/SSTar/README.md](dataset/SSTar/README.md))。

数据特点:真实侧扫声呐影像(非仿真),单波段声呐强度图,呈典型成像特征(目标高亮回波 + 后方声学阴影 + 海底背景纹理),各图分辨率不固定。

```
dataset/SSTar/
├── samples/                 # 已提供:每类少量样本实例
│   ├── RGYJ/  SXJS/  MTZ/  JZX/  shipwreck/
├── <各类别完整图像>          # 待补充
├── metadata.jsonl           # 待补充:图像-提示词-标注元信息
└── split.json               # 待补充:train/test 划分
```

## 使用方法

### Stage 1 — SD 微调(DFDB)

需先准备 `stable-diffusion-v1-5` 基础权重(放到 `pretrained/sd-v1-5` 或修改脚本中的 `SD_BASE`)。

```bash
# Windows
step1\run_train_dfdb.bat

# 或手动:
accelerate launch --mixed_precision=fp16 step1/utils/train.py \
  --pretrained_model_name_or_path pretrained/sd-v1-5 \
  --train_data_dir dataset/SCTD \
  --split train --split_file dataset/SCTD/split_4cat_70_plus_afr.json \
  --output_dir step1/checkpoints/sd-dfdb \
  --resolution 512 --train_batch_size 4 --max_train_steps 5000 \
  --learning_rate 1e-5 --gradient_checkpointing \
  --use_dsr --dsr_prob 0.5 --dsr_slic_segments 64 \
  --use_lfa --lfa_alpha 0.05 --lfa_prob 0.15
```

### Stage 2 — ControlNet 训练 + 推理(TSCG)

```bash
# Windows: 训练 + 推理一键
step2\run_train.bat

# 仅推理(已有训练好的 ControlNet)
step2\run_inference.bat
```

推理 `inference_controlnet.py` 支持三种 `--mode`:

| 模式 | 用途 | 关键参数 |
|------|------|----------|
| `dataset` | 从数据集划分批量生成 | `--data_dir` `--split` `--categories` |
| `mask` | 从掩码图片生成 | `--mask_path` `--prompt` |
| `xml` | 从 XML 标注生成 | `--xml_path` |

## 说明

- 脚本默认使用相对路径(以脚本所在目录为基准),路径变量集中在每个 `.bat` 顶部,按需修改。
- Stage 2 依赖 Stage 1 产出的微调 SD 模型作为基础模型。
- 显存参考:训练约 10–12GB,推理约 6–8GB。
