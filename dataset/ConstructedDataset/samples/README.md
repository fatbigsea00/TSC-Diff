# Training Sample Illustrations

This directory provides a small number of training sample illustrations per category to demonstrate the data format. The full dataset will be released after the paper is accepted.

Each category (`aircraft` / `ship` / `artificial fishing reef`) contains 3 sample pairs. Each pair includes:

- `<id>.jpg` — the original side-scan sonar image (the generation target / ground truth for Stage 2)
- `<id>_mask.png` — the corresponding segmentation condition map, rendered from the labelme polygon annotations; this is the conditioning input of the Stage 2 ControlNet

Color mapping of the segmentation map (see `CATEGORY_COLORS` in [step2/utils/controlnet_dataset.py](../../../step2/utils/controlnet_dataset.py)):

| Region | Color (RGB) |
|--------|-------------|
| aircraft | (0, 0, 255) blue |
| ship | (0, 255, 0) green |
| human | (255, 0, 0) red |
| artificial fishing reef | (255, 255, 0) yellow |
| shadow | (128, 128, 128) gray |
| background | (64, 64, 64) dark gray |

> Note: during actual training the segmentation map is rendered online from the annotation JSON; the `_mask.png` files here are only visualization examples.
