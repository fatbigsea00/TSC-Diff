"""
Constructed dataset loader - for Inpainting LoRA training
Supports the merged main-object + shadow mask strategy
Supports colormap mode: converts the mask into a 3-channel RGB semantic segmentation map
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

try:
    from skimage.segmentation import slic as _slic
    _SLIC_AVAILABLE = True
except ImportError:
    _SLIC_AVAILABLE = False


# Fixed mapping from category to RGB color
CATEGORY_COLORS = {
    'background': (64, 64, 64),   # dark gray - background
    'human': (255, 0, 0),         # red
    'ship': (0, 255, 0),          # green
    'aircraft': (0, 0, 255),      # blue
    'artificial fishing reef': (255, 255, 0),  # yellow
    'shadow': (128, 128, 128),    # gray
}


def colormap_to_binary_mask(colormap_tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert a colormap tensor into a binary mask

    Args:
        colormap_tensor: a colormap of shape [3, H, W] or [B, 3, H, W], range [0, 1]

    Returns:
        binary_mask: a binary mask of shape [1, H, W] or [B, 1, H, W], where non-background regions are 1
    """
    # Normalize the background color
    bg_color = torch.tensor([c / 255.0 for c in CATEGORY_COLORS['background']], 
                           device=colormap_tensor.device, dtype=colormap_tensor.dtype)
    
    if colormap_tensor.dim() == 3:
        # [3, H, W]
        bg_color = bg_color.view(3, 1, 1)
        is_background = (torch.abs(colormap_tensor - bg_color) < 0.02).all(dim=0, keepdim=True)
    else:
        # [B, 3, H, W]
        bg_color = bg_color.view(1, 3, 1, 1)
        is_background = (torch.abs(colormap_tensor - bg_color) < 0.02).all(dim=1, keepdim=True)
    
    return (~is_background).float()


class ConstructedInpaintingDataset(Dataset):
    """
    Constructed dataset loader - for Inpainting training

    Mask strategy: merged main object + shadow (option A)
    Mask format: binary mask (0/1)

    Data format:
    - One folder per category (aircraft, ship, human)
    - Each sample contains: an image file (.jpg) and the corresponding JSON annotation file (.json)
    - The JSON contains polygon annotation points used to generate the mask
    """
    
    def __init__(
        self,
        data_dir: str,
        resolution: int = 512,
        prompt_prefix: str = "an underwater sonar image of ",
        center_crop: bool = False,
        random_flip: bool = False,
        use_colormap: bool = False,
        precompute_slic: bool = False,
        slic_segments: int = 64,
        split: str = None,
        split_file: str = None,
    ):
        """
        Args:
            data_dir: path to the dataset root directory
            resolution: output image resolution
            prompt_prefix: unified prompt prefix
            center_crop: whether to center crop
            random_flip: whether to randomly flip horizontally
            use_colormap: whether to use colormap mode
            split: "train" / "test" / None (all)
            split_file: path to split.json (default data_dir/split.json)
        """
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.prompt_prefix = prompt_prefix
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.use_colormap = use_colormap
        self.precompute_slic = precompute_slic and _SLIC_AVAILABLE
        self.slic_segments = slic_segments
        self.split = split

        self._allowed_files = None
        if split is not None:
            sf = Path(split_file) if split_file else self.data_dir / "split.json"
            with open(sf, "r", encoding="utf-8") as f:
                split_data = json.load(f)
            self._allowed_files = split_data.get(split, {})

        self.categories = ["aircraft", "ship", "human", "artificial fishing reef"]
        
        self.samples = []
        self._collect_samples()
        
        # Define the image transforms
        self.image_transforms = self._get_image_transforms()
        
        print(f"[OK] Constructed Inpainting dataset loaded: {len(self.samples)} samples")
        for cat in self.categories:
            count = sum(1 for s in self.samples if s["category"] == cat)
            print(f"  - {cat}: {count} samples")
    
    def _collect_samples(self):
        """Collect data samples, filtered by split"""
        for category in self.categories:
            cat_dir = self.data_dir / category
            if not cat_dir.exists():
                print(f"  [WARN] Category directory does not exist: {cat_dir}")
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
    
    def _get_image_transforms(self):
        """Define the image preprocessing transforms"""
        transforms_list = [
            # Force resize to a square without preserving aspect ratio, to match the behavior of the Stable Diffusion Inpainting Pipeline
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # normalize to [-1, 1]
        ])
        
        return transforms.Compose(transforms_list)
    
    def _load_mask_from_json(self, json_path: str, img_size: Tuple[int, int], category: str = None) -> Image.Image:
        """
        Generate a mask from a JSON annotation file

        Mask strategy: merged main object + shadow (option A)
        - If use_colormap=False: generate a binary mask (0=background, 255=target+shadow)
        - If use_colormap=True: generate an RGB semantic segmentation map (different categories use different colors)

        Args:
            json_path: path to the JSON file
            img_size: original image size (width, height)
            category: main object category

        Returns:
            PIL Image: the mask image
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not self.use_colormap:
            # Original binary mask mode
            mask = Image.new('L', img_size, 0)
            draw = ImageDraw.Draw(mask)
            
            # Draw all annotated polygons (including the target body and the shadow)
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                
                if len(points) >= 3:  # at least 3 points are needed to form a polygon
                    flattened_points = [coord for point in points for coord in point]
                    draw.polygon(flattened_points, fill=255)
            
            return mask
        else:
            # Colormap mode: generate an RGB semantic segmentation map
            # The background is marked with dark gray instead of black
            mask = Image.new('RGB', img_size, CATEGORY_COLORS['background'])
            draw = ImageDraw.Draw(mask)
            
            # Draw all annotated polygons, using different colors based on the label
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                label = shape.get('label', '').lower().strip()
                
                # Determine the color
                if 'shadow' in label:
                    color = CATEGORY_COLORS['shadow']
                elif category:
                    # Use the color of the main object category
                    color = CATEGORY_COLORS.get(category, (128, 128, 128))
                else:
                    color = (128, 128, 128)  # default gray
                
                if len(points) >= 3:
                    flattened_points = [coord for point in points for coord in point]
                    draw.polygon(flattened_points, fill=color)
            
            return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample

        Returns:
            dict: {
                'pixel_values': original image tensor [3, H, W], range [-1, 1]
                'mask': binary mask tensor [1, H, W], range [0, 1]
                'masked_image': masked image tensor [3, H, W]
                'prompt': text prompt
                'category': category name
            }
        """
        sample = self.samples[idx]
        
        # Load the image
        image = Image.open(sample["image_path"]).convert("RGB")
        original_size = image.size
        
        # Load and generate the mask
        mask = self._load_mask_from_json(sample["json_path"], original_size, sample["category"])
        
        # Random horizontal flip (if enabled)
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # Apply the image transforms
        pixel_values = self.image_transforms(image)
        
        # Transform the mask - ensure size matches, force resize to a square
        mask_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        mask_tensor = mask_transform(mask)  # [1 or 3, H, W], range [0, 1]
        
        # Create the masked image (set the masked region to 0)
        if self.use_colormap:
            # Colormap mode: use the helper function to compute the binary mask
            mask_binary = colormap_to_binary_mask(mask_tensor)  # [1, H, W]
            masked_image = pixel_values * (1 - mask_binary)
        else:
            # Binary mask mode: mask is [1, H, W]
            masked_image = pixel_values * (1 - mask_tensor)
        
        result = {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "masked_image": masked_image,
            "prompt": sample["prompt"],
            "category": sample["category"],
        }

        # SLIC superpixel mask: precomputed in the worker process to avoid blocking the GPU in the training loop
        if self.precompute_slic:
            # pixel_values ∈ [-1,1] → [0,1], numpy [H,W,3]
            img_np = ((pixel_values + 1.0) / 2.0).permute(1, 2, 0).numpy()
            segments = _slic(
                img_np,
                n_segments=self.slic_segments,
                compactness=10.0,
                sigma=1.0,
                start_label=0,
                channel_axis=2,
            )
            # [H,W] → [1,H,W] float32
            result["spixel_mask"] = torch.from_numpy(segments).float().unsqueeze(0)

        return result


def collate_fn(examples):
    """
    Batch collation function
    """
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    masks = torch.stack([example["mask"] for example in examples])
    masked_images = torch.stack([example["masked_image"] for example in examples])
    prompts = [example["prompt"] for example in examples]
    categories = [example["category"] for example in examples]
    
    return {
        "pixel_values": pixel_values,
        "masks": masks,
        "masked_images": masked_images,
        "prompts": prompts,
        "categories": categories,
    }


def get_samples_by_category(dataset, category: str) -> List[int]:
    """Get the sample indices for the specified category"""
    return [i for i, s in enumerate(dataset.samples) if s["category"] == category]


if __name__ == "__main__":
    # Test the data loader
    dataset = ConstructedInpaintingDataset(
        data_dir="./dataset/ConstructedDataset",
        resolution=512,
        prompt_prefix="an underwater sonar image of "
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test loading a single sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    print(f"  - pixel_values: {sample['pixel_values'].shape}")
    print(f"  - mask: {sample['mask'].shape}")
    print(f"  - masked_image: {sample['masked_image'].shape}")
    print(f"  - prompt: {sample['prompt']}")
    print(f"  - category: {sample['category']}")
    
    # Test batch collation
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(f"\nBatch shapes:")
    print(f"  - pixel_values: {batch['pixel_values'].shape}")
    print(f"  - masks: {batch['masks'].shape}")
    print(f"  - masked_images: {batch['masked_images'].shape}")
    print(f"  - number of prompts: {len(batch['prompts'])}")
    
    # Count by category
    print("\nCount by category:")
    for cat in dataset.categories:
        indices = get_samples_by_category(dataset, cat)
        print(f"  - {cat}: {len(indices)} samples")
