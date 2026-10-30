# TSC-Diff

TSC-Diff is a two-stage diffusion framework for **side-scan sonar image generation / data augmentation**. Stage 1 fine-tunes Stable Diffusion on sonar data with Dual-space Frequency Distribution Broadening (DFDB = PFA + LFA). Stage 2 trains a ControlNet with Target-Shadow Coupled Guidance to generate sonar images at specified categories and locations.

This repository currently provides the core modules and a few sample images. The full training/inference pipeline and datasets will be released after the paper is accepted.

## Contents

```
TSC-Diff/
├── requirements.txt
├── step1/plugins/DSR.py              # PFA / LFA (DFDB) modules
├── step2/models/rbe_controlnet.py    # RBE: edge-enhanced ControlNet conditioning embedding
├── step2/models/map_cross_attention.py  # MapCA: map-guided cross-attention modulation
└── dataset/
    ├── ConstructedDataset/samples/   # image + condition map pairs
    └── SSTar/samples/                # raw SSTar sample images
```

## Installation

```bash
conda create -n tsc-diff python=3.10 -y
conda activate tsc-diff
pip install -r requirements.txt
```

## Sample Data

- `dataset/ConstructedDataset/samples/`: 3 categories (aircraft / ship / artificial fishing reef), 3 image–condition-map pairs each.
- `dataset/SSTar/samples/`: 18 raw images from 5 categories (reefs / underwater_reefs / dock_piles / containers / shipwrecks).
