# TSC-Diff

TSC-Diff 是一个面向**侧扫声呐图像生成 / 数据增强**的两阶段扩散模型框架。第一阶段在声呐数据上微调 Stable Diffusion 以学习声呐图像分布;第二阶段训练一个受分割图(segmentation map)控制的 ControlNet,按指定类别与位置生成完整声呐图像。

本仓库为**论文配套代码发布版**,仅包含两阶段的训练与推理代码,不含指标计算、消融实验与下游任务(分割/检测)代码。

## 方法与模块对应

| 论文术语 | 作用 | 代码位置 / 开关 |
|----------|------|----------------|
| **Stage 1 - SD 微调** | 学习声呐图像分布 | [step1/utils/train.py](step1/utils/train.py) |
| **DFDB** | 双频域数据增强,= PFA + LFA | `--use_dsr`(PFA)+ `--use_lfa`(LFA) |
| PFA (Pixel-level Frequency Augmentation) | 像素级频率扰动 | `DoublePerturb` @ [step1/plugins/DSR.py](step1/plugins/DSR.py) |
| LFA (Latent Frequency Augmentation) | 隐空间频率扰动 | `LatentFrequencyPerturb` @ [step1/plugins/DSR.py](step1/plugins/DSR.py) |
| **Stage 2 - ControlNet** | 受控生成 | [step2/utils/train_controlnet.py](step2/utils/train_controlnet.py) / [step2/utils/inference_controlnet.py](step2/utils/inference_controlnet.py) |
| **TSCG** | 完整控制方案,= RBE + MapCA + Region/Boundary Loss | `--use_rbe --use_mask_ca --ca_mild --ca_timestep_gate --use_region_loss` |
| RBE | 残差边界增强的条件嵌入 | [step2/models/rbe_controlnet.py](step2/models/rbe_controlnet.py) |
| MapCA | 掩码引导的交叉注意力 | 训练 [step2/models/map_cross_attention.py](step2/models/map_cross_attention.py),推理 [step2/models/mask_cross_attention.py](step2/models/mask_cross_attention.py) |
| Region/Boundary Loss | 区域 + 边界加权损失(时间步门控) | `--use_region_loss --rl_*` |

> [step2/models/erm_controlnet.py](step2/models/erm_controlnet.py) 仅用于推理时兼容旧 checkpoint 的回退加载。

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

> 本发布版**暂未包含**数据集,请按下述结构放置后再运行。

```
dataset/SCTD/
├── aircraft/                    # 各类别图像
├── ship/
├── human/
├── artificial fishing reef/
├── metadata.jsonl               # 图像-提示词-掩码元信息
└── split_4cat_70_plus_afr.json  # train/test 划分
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
