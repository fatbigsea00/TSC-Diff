"""
SCTD数据集加载器 - 用于Inpainting LoRA训练
支持主物体+阴影合并掩码策略
支持colormap模式：将掩码转换为3通道RGB语义分割图
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


# 定义类别到RGB颜色的固定映射
CATEGORY_COLORS = {
    'background': (64, 64, 64),   # 深灰色 - 背景
    'human': (255, 0, 0),         # 红色
    'ship': (0, 255, 0),          # 绿色
    'aircraft': (0, 0, 255),      # 蓝色
    'artificial fishing reef': (255, 255, 0),  # 黄色
    'shadow': (128, 128, 128),    # 灰色
}


def colormap_to_binary_mask(colormap_tensor: torch.Tensor) -> torch.Tensor:
    """
    将 colormap tensor 转换为二值掩码
    
    Args:
        colormap_tensor: [3, H, W] 或 [B, 3, H, W] 的 colormap，范围 [0, 1]
    
    Returns:
        binary_mask: [1, H, W] 或 [B, 1, H, W] 的二值掩码，非背景区域为1
    """
    # 背景颜色归一化
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


class SCTDInpaintingDataset(Dataset):
    """
    SCTD数据集加载器 - 用于Inpainting训练
    
    掩码策略: 主物体+阴影合并 (方案A)
    掩码格式: 二值掩码 (0/1)
    
    数据格式:
    - 每个类别一个文件夹 (aircraft, ship, human)
    - 每个样本包含: 图像文件(.jpg) 和 对应的JSON标注文件(.json)
    - JSON包含多边形标注点，用于生成掩码
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
            data_dir: 数据集根目录路径
            resolution: 输出图像分辨率
            prompt_prefix: 统一的prompt前缀
            center_crop: 是否中心裁剪
            random_flip: 是否随机水平翻转
            use_colormap: 是否使用colormap模式
            split: "train" / "test" / None(全部)
            split_file: split.json 路径 (默认 data_dir/split.json)
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
        
        # 定义图像转换
        self.image_transforms = self._get_image_transforms()
        
        print(f"[OK] SCTD Inpainting dataset loaded: {len(self.samples)} samples")
        for cat in self.categories:
            count = sum(1 for s in self.samples if s["category"] == cat)
            print(f"  - {cat}: {count} 样本")
    
    def _collect_samples(self):
        """收集数据样本，按 split 过滤"""
        for category in self.categories:
            cat_dir = self.data_dir / category
            if not cat_dir.exists():
                print(f"  [WARN] 类别目录不存在: {cat_dir}")
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
    
    def _get_image_transforms(self):
        """定义图像预处理转换"""
        transforms_list = [
            # 强制resize到正方形，不保持宽高比，确保与Stable Diffusion Inpainting Pipeline行为一致
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        ]
        
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # 归一化到[-1, 1]
        ])
        
        return transforms.Compose(transforms_list)
    
    def _load_mask_from_json(self, json_path: str, img_size: Tuple[int, int], category: str = None) -> Image.Image:
        """
        从JSON标注文件生成掩码
        
        掩码策略: 主物体+阴影合并 (方案A)
        - 如果use_colormap=False: 生成二值掩码 (0=背景, 255=目标+影子)
        - 如果use_colormap=True: 生成RGB语义分割图（不同类别用不同颜色）
        
        Args:
            json_path: JSON文件路径
            img_size: 原始图像尺寸 (width, height)
            category: 主物体类别
        
        Returns:
            PIL Image: 掩码图像
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if not self.use_colormap:
            # 原始二值掩码模式
            mask = Image.new('L', img_size, 0)
            draw = ImageDraw.Draw(mask)
            
            # 绘制所有标注的多边形（包括目标本体和影子）
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                
                if len(points) >= 3:  # 至少需要3个点才能构成多边形
                    flattened_points = [coord for point in points for coord in point]
                    draw.polygon(flattened_points, fill=255)
            
            return mask
        else:
            # Colormap模式：生成RGB语义分割图
            # 背景使用深灰色标记，不再是黑色
            mask = Image.new('RGB', img_size, CATEGORY_COLORS['background'])
            draw = ImageDraw.Draw(mask)
            
            # 绘制所有标注的多边形，根据标签使用不同颜色
            for shape in data.get('shapes', []):
                points = shape.get('points', [])
                label = shape.get('label', '').lower().strip()
                
                # 确定颜色
                if 'shadow' in label:
                    color = CATEGORY_COLORS['shadow']
                elif category:
                    # 使用主物体类别的颜色
                    color = CATEGORY_COLORS.get(category, (128, 128, 128))
                else:
                    color = (128, 128, 128)  # 默认灰色
                
                if len(points) >= 3:
                    flattened_points = [coord for point in points for coord in point]
                    draw.polygon(flattened_points, fill=color)
            
            return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        Returns:
            dict: {
                'pixel_values': 原始图像 tensor [3, H, W], 范围[-1, 1]
                'mask': 二值掩码 tensor [1, H, W], 范围[0, 1]
                'masked_image': 掩码后的图像 tensor [3, H, W]
                'prompt': 文本prompt
                'category': 类别名称
            }
        """
        sample = self.samples[idx]
        
        # 加载图像
        image = Image.open(sample["image_path"]).convert("RGB")
        original_size = image.size
        
        # 加载并生成掩码
        mask = self._load_mask_from_json(sample["json_path"], original_size, sample["category"])
        
        # 随机水平翻转 (如果启用)
        if self.random_flip and torch.rand(1).item() > 0.5:
            image = transforms.functional.hflip(image)
            mask = transforms.functional.hflip(mask)
        
        # 应用图像转换
        pixel_values = self.image_transforms(image)
        
        # 转换掩码 - 确保尺寸匹配，强制resize到正方形
        mask_transform = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        mask_tensor = mask_transform(mask)  # [1 or 3, H, W], 范围[0, 1]
        
        # 创建masked image (将掩码区域设为0)
        if self.use_colormap:
            # Colormap模式：使用辅助函数计算二值掩码
            mask_binary = colormap_to_binary_mask(mask_tensor)  # [1, H, W]
            masked_image = pixel_values * (1 - mask_binary)
        else:
            # 二值掩码模式：mask是[1, H, W]
            masked_image = pixel_values * (1 - mask_tensor)
        
        result = {
            "pixel_values": pixel_values,
            "mask": mask_tensor,
            "masked_image": masked_image,
            "prompt": sample["prompt"],
            "category": sample["category"],
        }

        # SLIC 超像素掩码：在 worker 进程里预计算，避免训练循环里阻塞 GPU
        if self.precompute_slic:
            # pixel_values ∈ [-1,1] → [0,1]，numpy [H,W,3]
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
    批处理函数
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
    """获取指定类别的样本索引"""
    return [i for i, s in enumerate(dataset.samples) if s["category"] == category]


if __name__ == "__main__":
    # 测试数据加载器
    dataset = SCTDInpaintingDataset(
        data_dir="/home/lxd/data/spackle/SCTD/SD_SCTD-1.0",
        resolution=512,
        prompt_prefix="an underwater sonar image of "
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试加载单个样本
    sample = dataset[0]
    print(f"\n样本形状:")
    print(f"  - pixel_values: {sample['pixel_values'].shape}")
    print(f"  - mask: {sample['mask'].shape}")
    print(f"  - masked_image: {sample['masked_image'].shape}")
    print(f"  - prompt: {sample['prompt']}")
    print(f"  - category: {sample['category']}")
    
    # 测试批处理
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    batch = next(iter(dataloader))
    print(f"\n批次形状:")
    print(f"  - pixel_values: {batch['pixel_values'].shape}")
    print(f"  - masks: {batch['masks'].shape}")
    print(f"  - masked_images: {batch['masked_images'].shape}")
    print(f"  - prompts数量: {len(batch['prompts'])}")
    
    # 按类别统计
    print("\n按类别统计:")
    for cat in dataset.categories:
        indices = get_samples_by_category(dataset, cat)
        print(f"  - {cat}: {len(indices)} 样本")
