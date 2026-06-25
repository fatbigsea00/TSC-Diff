# TSC-Diff

TSC-Diff is a two-stage diffusion framework for **side-scan sonar image generation / data augmentation**. Stage 1 fine-tunes Stable Diffusion on sonar data to learn the sonar image distribution; Stage 2 trains a ControlNet conditioned on a segmentation map to generate complete sonar images at specified categories and locations.

This repository is the **code release accompanying the paper**. The current version provides the core method code together with a small set of training sample illustrations; the full experiment code and training datasets will be released after the paper is accepted.

## Repository Structure

```
TSC-Diff/
├── requirements.txt
├── step1/                       # Stage 1: SD fine-tuning + DFDB
│   ├── utils/train.py
│   ├── utils/dataset.py
│   ├── plugins/DSR.py           # PFA / LFA augmentation modules
│   └── run_train_dfdb.bat
└── step2/                       # Stage 2: ControlNet (TSCG)
    ├── models/                  # RBE / MapCA / MaskCA
    ├── utils/train_controlnet.py
    ├── utils/inference_controlnet.py
    ├── utils/controlnet_dataset.py
    ├── run_train.bat            # train + inference
    └── run_inference.bat        # inference only
```

## Installation

```bash
conda create -n tsc-diff python=3.10 -y
conda activate tsc-diff
pip install -r requirements.txt
```

## Datasets

This work involves two side-scan sonar datasets: the **constructed dataset** (used for training/inference of the method) and **SSTar** (the side-scan sonar target dataset constructed in this work). Only a small set of sample instances is provided in this repository for both; the **full datasets, category definitions, annotations, and splits will be released after the paper is accepted.**

### 1. Constructed Dataset

The sonar dataset directly used by the two-stage TSC-Diff training and inference. It contains four categories (`aircraft / ship / human / artificial fishing reef`); each sample consists of an original image plus a labelme segmentation annotation, which is rendered into the segmentation condition map required by ControlNet.

> This release provides sample illustrations (see [dataset/ConstructedDataset/samples/](dataset/ConstructedDataset/samples/), 3 pairs of "image + segmentation condition map" per category).

Expected directory layout of the full dataset:

```
dataset/ConstructedDataset/
├── samples/                     # provided: a few sample illustrations per category (image + mask)
├── aircraft/                    # to be added: per-category images + annotations
├── ship/
├── human/
├── artificial fishing reef/
├── metadata.jsonl               # image-prompt-mask metadata
└── split_4cat_70_plus_afr.json  # train/test split
```

### 2. SSTar (Side-Scan Sonar Targets)

The side-scan sonar target dataset constructed in this work, collected from real side-scan sonar surveys and covering several typical underwater targets. The full dataset contains **489 images** grouped into **5 target categories**:

| Category (original code) | Full size | Samples in repo |
|--------------------------|-----------|-----------------|
| RGYJ | 389 | 4 |
| SXJS | 54 | 4 |
| MTZ | 29 | 4 |
| JZX | 15 | 4 |
| shipwreck | 2 | 2 |
| **Total** | **489** | **18** |

> This release provides only a few sample instances (see [dataset/SSTar/samples/](dataset/SSTar/samples/); details in [dataset/SSTar/README.md](dataset/SSTar/README.md)).

Characteristics: real side-scan sonar imagery (not synthetic), single-band sonar intensity images with typical imaging characteristics (bright target echo + acoustic shadow behind the target + seabed background texture), with varying per-image resolution.

```
dataset/SSTar/
├── samples/                 # provided: a few sample instances per category
│   ├── RGYJ/  SXJS/  MTZ/  JZX/  shipwreck/
├── <full per-category images>   # to be added
├── metadata.jsonl           # to be added: image-prompt-annotation metadata
└── split.json               # to be added: train/test split
```

## Usage

### Stage 1 — SD fine-tuning (DFDB)

First prepare the `stable-diffusion-v1-5` base weights (place them under `pretrained/sd-v1-5` or edit `SD_BASE` in the script).

```bash
# Windows
step1\run_train_dfdb.bat

# Or manually:
accelerate launch --mixed_precision=fp16 step1/utils/train.py \
  --pretrained_model_name_or_path pretrained/sd-v1-5 \
  --train_data_dir dataset/ConstructedDataset \
  --split train --split_file dataset/ConstructedDataset/split_4cat_70_plus_afr.json \
  --output_dir step1/checkpoints/sd-dfdb \
  --resolution 512 --train_batch_size 4 --max_train_steps 5000 \
  --learning_rate 1e-5 --gradient_checkpointing \
  --use_dsr --dsr_prob 0.5 --dsr_slic_segments 64 \
  --use_lfa --lfa_alpha 0.05 --lfa_prob 0.15
```

### Stage 2 — ControlNet training + inference (TSCG)

```bash
# Windows: train + inference in one shot
step2\run_train.bat

# Inference only (with an already trained ControlNet)
step2\run_inference.bat
```

The inference script `inference_controlnet.py` supports three `--mode` values:

| Mode | Purpose | Key arguments |
|------|---------|---------------|
| `dataset` | Batch generation from a dataset split | `--data_dir` `--split` `--categories` |
| `mask` | Generation from a mask image | `--mask_path` `--prompt` |
| `xml` | Generation from an XML annotation | `--xml_path` |

## Notes

- Scripts use relative paths by default (resolved from the script directory); path variables are grouped at the top of each `.bat` file and can be edited as needed.
- Stage 2 depends on the fine-tuned SD model produced by Stage 1 as its base model.
- GPU memory: ~10-12GB for training, ~6-8GB for inference.
