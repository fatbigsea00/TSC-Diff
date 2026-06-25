"""
Constructed dataset loader - for ControlNet-Seg training
Uses a mask/segmentation map as the condition to generate a complete image (background + objects)

Difference from Inpainting:
- Inpainting: requires the original image + mask -> repair the masked region
- ControlNet-Seg: only requires a mask/segmentation map -> generate a complete new image
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision import transforms


# Fixed mapping from category to RGB color
# Note: the background cannot be pure black (0,0,0); otherwise, after normalization ControlNet would get no information
CATEGORY_COLORS = {
    'human': (255, 0, 0),      # red
    'ship': (0, 255, 0),       # green
    'aircraft': (0, 0, 255),   # blue
    'artificial fishing reef': (255, 255, 0),  # yellow
    'shadow': (128, 128, 128), # gray
    'background': (64, 64, 64),   # dark gray - background (must be non-zero!)
}


class ConstructedControlNetDataset(Dataset):
    """
    Constructed dataset loader - for ControlNet training
    
    Core idea: 
    - Input: segmentation map (mask)
    - Output: complete image (background + objects)
    - Does not require an "original image" as input; the model learns to generate a complete image from the segmentation
    
    Data format:
    - One folder per category (aircraft, ship, human)
    - Each sample contains: an image file (.jpg) and a corresponding JSON annotation file (.json)
    """
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        prompt_prefix: str = "an underwater sonar image of ",
        use_colormap: bool = True,
        grid_size: int = 8,
        split: str = None,
        split_file: str = None,
        no_shadow: bool = False,
    ):
        """
        Args:
            data_dir: Path to the dataset root directory
            resolution: Output image resolution
            prompt_prefix: Unified prompt prefix
            use_colormap: Whether to use an RGB colormap (True) or a binary mask (False)
            grid_size: Grid size for the grid superpixels
            split: "train" / "test" / None (all)
            split_file: Path to split.json (default: data_dir/split.json)
            no_shadow: Do not draw shadows in the mask, keeping only objects and background (shadow treated as background)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.prompt_prefix = prompt_prefix
        self.use_colormap = use_colormap
        self.grid_size = grid_size
        self.split = split
        self.no_shadow = no_shadow

        self._allowed_files = None
        if split is not None:
            sf = Path(split_file) if split_file else self.data_dir / "split.json"
            import json as _json
            with open(sf, "r", encoding="utf-8") as f:
                split_data = _json.load(f)
            self._allowed_files = split_data.get(split, {})

        self.categories = ["aircraft", "ship", "human", "artificial fishing reef"]
        
        self.samples = []
        self._collect_samples()
        
        # Define image transforms
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # normalize to [-1, 1]
        ])
        
        # Transforms for the conditioning image (segmentation map) - only normalize to [0, 1]
        self.conditioning_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        print(f"[OK] Constructed ControlNet dataset loaded: {len(self.samples)} samples")
        for cat in self.categories:
            count = sum(1 for s in self.samples if s["category"] == cat)
            print(f"  - {cat}: {count} samples")
    
    def _collect_samples(self):
        """Collect data samples, filtering by split"""
        for category in self.categories:
            cat_dir = self.data_dir / category
            if not cat_dir.exists():
                print(f"  [WARN] Category dir not found: {cat_dir}")
                continue

            allowed = None
            if self._allowed_files is not None:
                allowed = set(self._allowed_files.get(category, []))
            
            img_paths = sorted(cat_dir.glob("*.jpg"))
            for img_path in img_paths:
                if allowed is not None and img_path.name not in allowed:
                    continue

                json_path = img_path.with_suffix(".json")
                if json_path.exists():
                    has_shadow = self._check_has_shadow(json_path)
                    if has_shadow:
                        prompt = f"{self.prompt_prefix}{category} with shadow"
                    else:
                        prompt = f"{self.prompt_prefix}{category}"
                    
                    self.samples.append({
                        "image_path": str(img_path),
                        "json_path": str(json_path),
                        "category": category,
                        "prompt": prompt,
                        "has_shadow": has_shadow,
                    })
    
    def _check_has_shadow(self, json_path: str) -> bool:
        """Check whether the annotation contains a shadow"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower().strip()
                if 'shadow' in label:
                    return True
        except Exception as e:
            print(f"Warning: failed to read JSON file {json_path}: {e}")
        return False
    
    def _create_segmentation_map(self, json_path: str, img_size: Tuple[int, int], category: str) -> Image.Image:
        """
        Generate a segmentation map from a JSON annotation file

        Args:
            json_path: Path to the JSON file
            img_size: Original image size (width, height)
            category: Main object category

        Returns:
            PIL Image: segmentation map
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        if self.use_colormap:
            # RGB colormap mode
            seg_map = Image.new('RGB', img_size, CATEGORY_COLORS['background'])
            draw = ImageDraw.Draw(seg_map)

            shadow_shapes = []
            object_shapes = []

            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower().strip()
                if 'shadow' in label:
                    shadow_shapes.append(shape)
                else:
                    object_shapes.append(shape)

            if not self.no_shadow:
                for shape in shadow_shapes:
                    points = shape.get('points', [])
                    color = CATEGORY_COLORS['shadow']
                    if len(points) >= 3:
                        flattened_points = [coord for point in points for coord in point]
                        draw.polygon(flattened_points, fill=color)

            for shape in object_shapes:
                points = shape.get('points', [])
                color = CATEGORY_COLORS.get(category, (128, 128, 128))
                if len(points) >= 3:
                    flattened_points = [coord for point in points for coord in point]
                    draw.polygon(flattened_points, fill=color)
        else:
            # Binary mask mode
            seg_map = Image.new('L', img_size, 0)
            draw = ImageDraw.Draw(seg_map)

            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower().strip()
                if self.no_shadow and 'shadow' in label:
                    continue
                points = shape.get('points', [])
                if len(points) >= 3:
                    flattened_points = [coord for point in points for coord in point]
                    draw.polygon(flattened_points, fill=255)

        return seg_map

    def _generate_grid_spixel_mask(self, height: int, width: int) -> torch.Tensor:
        """
        Generate a grid superpixel mask

        Args:
            height: Image height
            width: Image width

        Returns:
            torch.Tensor: grid superpixel label map [1, H, W]
        """
        # Generate grid indices
        h_idx = torch.arange(height) * self.grid_size // height
        w_idx = torch.arange(width) * self.grid_size // width

        # Generate the label map: each grid cell has a unique label
        label_map = h_idx.unsqueeze(1) * self.grid_size + w_idx.unsqueeze(0)

        return label_map.float().unsqueeze(0)  # [1, H, W]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Returns:
            dict: {
                'pixel_values': target image tensor [3, H, W], range [-1, 1]
                'conditioning_pixel_values': segmentation map tensor [3 or 1, H, W], range [0, 1]
                'spixel_mask': superpixel mask tensor [1, H, W], where each pixel value is the label of its superpixel
                'prompt': text prompt
                'category': category name
            }
        """
        sample = self.samples[idx]

        # Load the target image (ground truth)
        image = Image.open(sample["image_path"]).convert("RGB")
        original_size = image.size

        # Create the segmentation map (conditioning image)
        seg_map = self._create_segmentation_map(
            sample["json_path"],
            original_size,
            sample["category"]
        )

        # Generate the grid superpixel mask (based on the original size)
        spixel_mask = self._generate_grid_spixel_mask(original_size[1], original_size[0])  # (height, width)

        # Apply transforms
        pixel_values = self.image_transforms(image)
        conditioning_pixel_values = self.conditioning_transforms(seg_map)

        # Resize the superpixel mask to the target resolution (use nearest-neighbor interpolation to keep labels intact)
        spixel_mask = torch.nn.functional.interpolate(
            spixel_mask.unsqueeze(0),
            size=(self.resolution, self.resolution),
            mode='nearest'
        ).squeeze(0)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "spixel_mask": spixel_mask,
            "prompt": sample["prompt"],
            "category": sample["category"],
        }


def controlnet_collate_fn(examples):
    """Collate function for batching"""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    spixel_masks = torch.stack([example["spixel_mask"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    categories = [example["category"] for example in examples]

    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "spixel_masks": spixel_masks,
        "prompts": prompts,
        "categories": categories,
    }


if __name__ == "__main__":
    # Test the data loader
    dataset = ConstructedControlNetDataset(
        data_dir="./dataset/ConstructedDataset",
        resolution=512,
        prompt_prefix="an underwater sonar image of ",
        use_colormap=True,
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a single sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  - pixel_values: {sample['pixel_values'].shape}")
    print(f"  - conditioning_pixel_values: {sample['conditioning_pixel_values'].shape}")
    print(f"  - prompt: {sample['prompt']}")
    print(f"  - category: {sample['category']}")
    
    # Test batching
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=controlnet_collate_fn)
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  - pixel_values: {batch['pixel_values'].shape}")
    print(f"  - conditioning_pixel_values: {batch['conditioning_pixel_values'].shape}")
    print(f"  - number of prompts: {len(batch['prompts'])}")

