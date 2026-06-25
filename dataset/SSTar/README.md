# SSTar Dataset (Side-Scan Sonar Targets)

SSTar is a **side-scan sonar target dataset** constructed in this work, collected from real side-scan sonar (SSS) surveys and covering several typical underwater targets. The dataset targets sonar image generation and data augmentation, and serves as the target source and evaluation basis for the TSC-Diff method.

> This repository currently provides only a small number of sample instances to illustrate the data format. **The full dataset, category definitions, annotations, and splits will be released after the paper is accepted.**

## Scale

The full dataset contains **489** side-scan sonar images, grouped into **5** target categories (category folders follow the dataset's original naming codes):

| Category (original code) | Full size | Samples in repo |
|--------------------------|-----------|-----------------|
| RGYJ | 389 | 4 |
| SXJS | 54 | 4 |
| MTZ | 29 | 4 |
| JZX | 15 | 4 |
| shipwreck | 2 | 2 |
| **Total** | **489** | **18** |

## Characteristics

- **Source**: real side-scan sonar survey imagery, not synthetic.
- **Form**: single-band sonar intensity images (JPG) with typical sonar imaging characteristics — bright target echo + acoustic shadow behind the target + seabed background texture.
- **Resolution**: image sizes are not fixed (on the order of a few hundred pixels), preserving the original survey aspect ratios.
- **Use**: real target samples for the two-stage TSC-Diff generation framework, used to learn the sonar target distribution and for controlled generation.

## Directory Structure

```
dataset/SSTar/
├── README.md
└── samples/                 # provided: a few sample instances per category (raw images)
    ├── RGYJ/
    ├── SXJS/
    ├── MTZ/
    ├── JZX/
    └── shipwreck/
```

> Category folders follow the dataset's original naming codes; the samples here are for format illustration only. The full category semantics, annotation files, and train/test splits will be released together with the full dataset.
