# SSTar Dataset (Side-Scan Sonar Targets)

SSTar is a **side-scan sonar target dataset** constructed in this work, collected from real side-scan sonar (SSS) surveys and covering several typical underwater targets. The dataset targets sonar image generation and data augmentation, and serves as the target source and evaluation basis for the TSC-Diff method.

SSTar contains **489 target samples** from **five** representative underwater object categories: 2 shipwrecks, 15 containers, 29 dock piles, 389 reefs, and 54 underwater reefs. The data were collected from 133 survey lines, with a total survey distance of about 317 km, and cover diverse imaging conditions, target scales, seabed environments, and target orientations.

> This repository currently provides only a small number of sample instances to illustrate the data format. **The full dataset, category definitions, annotations, and splits will be released after the paper is accepted.**

## Scale

| Category (original code) | Full size | Samples in repo |
|--------------------------|-----------|-----------------|
| reefs (RGYJ) | 389 | 4 |
| underwater_reefs (SXJS) | 54 | 4 |
| dock_piles (MTZ) | 29 | 4 |
| containers (JZX) | 15 | 4 |
| shipwrecks | 2 | 2 |
| **Total** | **489** | **18** |

## Characteristics

- **Source**: real side-scan sonar survey imagery, not synthetic; collected from 133 survey lines (~317 km in total).
- **Form**: single-band sonar intensity images (JPG) with typical sonar imaging characteristics — bright target echo + acoustic shadow behind the target + seabed background texture.
- **Diversity**: covers diverse imaging conditions, target scales, seabed environments, and target orientations.
- **Resolution**: image sizes are not fixed (on the order of a few hundred pixels), preserving the original survey aspect ratios.
- **Use**: real target samples for the two-stage TSC-Diff generation framework, used to learn the sonar target distribution and for controlled generation.

## Directory Structure

```
dataset/SSTar/
├── README.md
└── samples/                 # provided: a few sample instances per category (raw images)
    ├── reefs/
    ├── underwater_reefs/
    ├── dock_piles/
    ├── containers/
    └── shipwrecks/
```

> The samples here are for format illustration only. The full category definitions, annotation files, and train/test splits will be released together with the full dataset.
