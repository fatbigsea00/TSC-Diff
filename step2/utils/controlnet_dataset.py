"""
SCTD数据集加载器 - 用于ControlNet-Seg训练
使用掩码/分割图作为条件，生成完整图像（背景+物体）

与Inpainting不同：
- Inpainting: 需要原图 + 掩码 -> 修复掩码区域
- ControlNet-Seg: 只需要掩码/分割图 -> 生成完整新图像
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


# 定义类别到RGB颜色的固定映射
# 注意：背景不能是纯黑(0,0,0)，否则归一化后ControlNet无法获取任何信息
CATEGORY_COLORS = {
    'human': (255, 0, 0),      # 红色
    'ship': (0, 255, 0),       # 绿色
    'aircraft': (0, 0, 255),   # 蓝色
    'artificial fishing reef': (255, 255, 0),  # 黄色
    'shadow': (128, 128, 128), # 灰色
    'background': (64, 64, 64),   # 深灰色 - 背景（必须非零！）
}


class SCTDControlNetDataset(Dataset):
    """
    SCTD数据集加载器 - 用于ControlNet训练
    
    核心思想: 
    - 输入: segmentation map (掩码)
    - 输出: 完整图像 (背景 + 物体)
    - 不需要"原图"作为输入，让模型学习从segmentation生成完整图像
    
    数据格式:
    - 每个类别一个文件夹 (aircraft, ship, human)
    - 每个样本包含: 图像文件(.jpg) 和 对应的JSON标注文件(.json)
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
            data_dir: 数据集根目录路径
            resolution: 输出图像分辨率
            prompt_prefix: 统一的prompt前缀
            use_colormap: 是否使用RGB colormap (True) 或二值mask (False)
            grid_size: 网格超像素的网格大小
            split: "train" / "test" / None(全部)
            split_file: split.json 路径 (默认 data_dir/split.json)
            no_shadow: 掩码中不绘制阴影，仅保留目标和背景 (shadow视为background)
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
        
        # 定义图像转换
        self.image_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])
        
        # 条件图像（segmentation map）的转换 - 只归一化到[0, 1]
        self.conditioning_transforms = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        
        print(f"[OK] SCTD ControlNet数据集加载完成: {len(self.samples)} 个样本")
        for cat in self.categories:
            count = sum(1 for s in self.samples if s["category"] == cat)
            print(f"  - {cat}: {count} 样本")
    
    def _collect_samples(self):
        """收集数据样本，按 split 过滤"""
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
        """检查标注中是否包含shadow"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for shape in data.get('shapes', []):
                label = shape.get('label', '').lower().strip()
                if 'shadow' in label:
                    return True
        except Exception as e:
            print(f"警告: 读取JSON文件失败 {json_path}: {e}")
        return False
    
    def _create_segmentation_map(self, json_path: str, img_size: Tuple[int, int], category: str) -> Image.Image:
        """
        从JSON标注文件生成分割图 (segmentation map)

        Args:
            json_path: JSON文件路径
            img_size: 原始图像尺寸 (width, height)
            category: 主物体类别

        Returns:
            PIL Image: 分割图
        """
        with open(json_path, 'r') as f:
            data = json.load(f)

        if self.use_colormap:
            # RGB colormap模式
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
            # 二值掩码模式
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
        生成网格超像素掩码

        Args:
            height: 图像高度
            width: 图像宽度

        Returns:
            torch.Tensor: 网格超像素标签图 [1, H, W]
        """
        # 生成网格索引
        h_idx = torch.arange(height) * self.grid_size // height
        w_idx = torch.arange(width) * self.grid_size // width

        # 生成标签图：每个网格块有唯一的标签
        label_map = h_idx.unsqueeze(1) * self.grid_size + w_idx.unsqueeze(0)

        return label_map.float().unsqueeze(0)  # [1, H, W]

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本

        Returns:
            dict: {
                'pixel_values': 目标图像 tensor [3, H, W], 范围[-1, 1]
                'conditioning_pixel_values': 分割图 tensor [3 or 1, H, W], 范围[0, 1]
                'spixel_mask': 超像素掩码 tensor [1, H, W]，每个像素值为其所属超像素标签
                'prompt': 文本prompt
                'category': 类别名称
            }
        """
        sample = self.samples[idx]

        # 加载目标图像（ground truth）
        image = Image.open(sample["image_path"]).convert("RGB")
        original_size = image.size

        # 创建分割图（conditioning image）
        seg_map = self._create_segmentation_map(
            sample["json_path"],
            original_size,
            sample["category"]
        )

        # 生成网格超像素掩码（基于原始尺寸）
        spixel_mask = self._generate_grid_spixel_mask(original_size[1], original_size[0])  # (height, width)

        # 应用转换
        pixel_values = self.image_transforms(image)
        conditioning_pixel_values = self.conditioning_transforms(seg_map)

        # Resize 超像素掩码到目标分辨率（使用最近邻插值保持标签完整性）
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
    """批处理函数"""
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
    # 测试数据加载器
    dataset = SCTDControlNetDataset(
        data_dir="./dataset/SCTD",
        resolution=512,
        prompt_prefix="an underwater sonar image of ",
        use_colormap=True,
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载单个样本
    sample = dataset[0]
    print(f"\n样本形状:")
    print(f"  - pixel_values: {sample['pixel_values'].shape}")
    print(f"  - conditioning_pixel_values: {sample['conditioning_pixel_values'].shape}")
    print(f"  - prompt: {sample['prompt']}")
    print(f"  - category: {sample['category']}")
    
    # 测试批处理
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=controlnet_collate_fn)
    batch = next(iter(dataloader))
    print(f"\n批次形状:")
    print(f"  - pixel_values: {batch['pixel_values'].shape}")
    print(f"  - conditioning_pixel_values: {batch['conditioning_pixel_values'].shape}")
    print(f"  - prompts数量: {len(batch['prompts'])}")

