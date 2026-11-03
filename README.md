# TSC-Diff

TSC-Diff is a two-stage diffusion framework for **side-scan sonar image generation / data augmentation**. Stage 1 fine-tunes Stable Diffusion on sonar data with Dual-space Frequency Distribution Broadening (DFDB = PFA + LFA). Stage 2 trains a ControlNet conditional branch with Target-Shadow Coupled Guidance (TSCG = TRM + RBE + MapCA), using the sonar-adapted SD U-Net as a frozen backbone.

TSCG incorporates target-shadow layout information through three guidance components:

- **Condition guidance (TRM):** a Tri-Region Map represents the target, shadow, and background regions and conditions the trainable branch.
- **Feature guidance (RBE):** Region Boundary Enhancement strengthens boundary cues in the conditional features.
- **Attention guidance (MapCA):** Map-guided Cross-Attention uses TRM-derived target and shadow masks to spatially reweight cross-attention outputs in the frozen backbone.

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

## Paper Configuration

The module defaults match the settings used in the paper experiments:

```python
from step1.plugins.DSR import DoublePerturb, LatentFrequencyPerturb
from step2.models.map_cross_attention import set_map_ca_data

pfa = DoublePerturb(feat_dim=3, prob=0.5)            # PFA
lfa = LatentFrequencyPerturb(alpha=0.05, prob=0.15)  # LFA

# MapCA (obj, shadow) boost per resolution:
# REGION_BOOST = {64: (0.35, 0.2), 32: (0.5, 0.3), 16: (0.5, 0.3), 8: (0.5, 0.3)}
set_map_ca_data(colormap, timestep=t, gate_mid=400.0, gate_temp=100.0)
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
