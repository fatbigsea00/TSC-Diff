"""
ControlNet-Seg 推理脚本

支持三种推理模式:
1. dataset: 从SCTD数据集读取mask推理
2. mask:    从掩码图片文件/目录推理
3. xml:     从X-AnyLabeling XML标注文件推理
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from controlnet_dataset import SCTDControlNetDataset, CATEGORY_COLORS


# ===================== XML 解析工具 =====================

def parse_xml_annotation(xml_path: str) -> Dict:
    """解析X-AnyLabeling格式的XML文件"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)
    filename = root.find('filename').text

    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text.lower()
        polygon_elem = obj.find('polygon')
        if polygon_elem is not None:
            points = []
            i = 1
            while True:
                x_elem = polygon_elem.find(f'x{i}')
                y_elem = polygon_elem.find(f'y{i}')
                if x_elem is None or y_elem is None:
                    break
                points.append((float(x_elem.text), float(y_elem.text)))
                i += 1
            objects.append({'name': name, 'points': points})

    return {
        'width': width, 'height': height, 'depth': depth,
        'filename': filename, 'objects': objects,
    }


def create_mask_from_xml(xml_data: Dict, target_size: Tuple[int, int] = (512, 512)) -> Image.Image:
    """从XML数据创建RGB colormap掩码"""
    mask = Image.new('RGB', target_size, CATEGORY_COLORS['background'])
    draw = ImageDraw.Draw(mask)

    orig_width = xml_data['width']
    orig_height = xml_data['height']
    scale_x = target_size[0] / orig_width
    scale_y = target_size[1] / orig_height

    shadow_objects = []
    other_objects = []
    for obj in xml_data['objects']:
        if 'shadow' in obj['name']:
            shadow_objects.append(obj)
        else:
            other_objects.append(obj)

    # 先绘制shadow（下层），再绘制物体（上层）
    for obj in shadow_objects:
        scaled_points = [(x * scale_x, y * scale_y) for x, y in obj['points']]
        if len(scaled_points) >= 3:
            draw.polygon(scaled_points, fill=CATEGORY_COLORS['shadow'])

    for obj in other_objects:
        label = obj['name']
        scaled_points = [(x * scale_x, y * scale_y) for x, y in obj['points']]
        color = CATEGORY_COLORS.get(label, (128, 128, 128))
        if len(scaled_points) >= 3:
            draw.polygon(scaled_points, fill=color)

    return mask


# ===================== 通用工具 =====================

def create_mask_from_polygon(
    polygons: List[List[tuple]],
    labels: List[str],
    image_size: tuple = (512, 512),
    use_colormap: bool = True,
) -> Image.Image:
    """从多边形坐标创建mask"""
    if use_colormap:
        mask = Image.new('RGB', image_size, CATEGORY_COLORS['background'])
    else:
        mask = Image.new('L', image_size, 0)

    draw = ImageDraw.Draw(mask)
    for polygon, label in zip(polygons, labels):
        if use_colormap:
            label_lower = label.lower()
            if 'shadow' in label_lower:
                color = CATEGORY_COLORS['shadow']
            elif label_lower in CATEGORY_COLORS:
                color = CATEGORY_COLORS[label_lower]
            else:
                color = (128, 128, 128)
        else:
            color = 255
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=color)
    return mask


def create_comparison_image(images: List[Image.Image]) -> Image.Image:
    """创建横向拼接对比图"""
    width = images[0].width
    height = images[0].height
    comparison = Image.new('RGB', (width * len(images), height))
    for i, img in enumerate(images):
        comparison.paste(img.resize((width, height), Image.BILINEAR), (width * i, 0))
    return comparison


# ===================== Pipeline 加载 =====================

def load_pipeline(args):
    """加载ControlNet + SD pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # 加载 ControlNet
    print(f"加载ControlNet模型: {args.controlnet_model_path}")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_path, torch_dtype=dtype,
    )

    # RBE/ERM: 区域边界增强模块
    if args.use_erm:
        # 新版本训练保存的子模块名为 rbe，需用 rbe_controlnet 注入；
        # 兼容老 erm_controlnet 训练出的 ckpt 时再回退。
        try:
            from models.rbe_controlnet import apply_rbe_to_controlnet
            controlnet = apply_rbe_to_controlnet(controlnet, input_resolution=args.resolution // 8)
        except ImportError:
            from models.erm_controlnet import apply_erm_to_controlnet
            controlnet = apply_erm_to_controlnet(controlnet, input_resolution=args.resolution // 8)
        # 优先 rbe_controlnet.pth（新名），回退 erm_controlnet.pth（兼容旧 ckpt）
        weight_path = None
        for cand in ("rbe_controlnet.pth", "erm_controlnet.pth"):
            p = os.path.join(args.controlnet_model_path, cand)
            if os.path.exists(p):
                weight_path = p
                break
        if weight_path is None:
            print(f"[WARN] 未找到 rbe_controlnet.pth / erm_controlnet.pth，"
                  f"RBE 模块将保持随机初始化 -> 推理结果可能不正确")
        else:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
            print(f"已加载 RBE/ERM 训练权重: {weight_path}")
            if missing:
                print(f"  [info] missing keys: {len(missing)} (前 3 项: {missing[:3]})")
            if unexpected:
                print(f"  [info] unexpected keys: {len(unexpected)} (前 3 项: {unexpected[:3]})")
        controlnet = controlnet.to(dtype=dtype)

    # 创建 Pipeline
    print(f"加载SD模型: {args.pretrained_model_name_or_path}")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    # 掩码引导交叉注意力
    if args.use_mask_ca:
        from models.mask_cross_attention import apply_mask_guided_attention
        apply_mask_guided_attention(pipeline.unet)
        if args.ca_timestep_gate:
            from models.mask_cross_attention import install_timestep_hook
            pipeline.unet._mask_ca_gate_mid = args.ca_gate_mid
            pipeline.unet._mask_ca_gate_temp = args.ca_gate_temp
            install_timestep_hook(pipeline.unet)
            print(f"  Timestep gate: mid={args.ca_gate_mid}, temp={args.ca_gate_temp}")

    return pipeline, device


# ===================== 模式: dataset =====================

def run_dataset_mode(args):
    """从SCTD数据集读取mask推理"""
    pipeline, device = load_pipeline(args)

    print("加载数据集...")
    dataset = SCTDControlNetDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        prompt_prefix=args.prompt_prefix,
        use_colormap=args.use_colormap,
        split=args.split,
        split_file=args.split_file,
        no_shadow=getattr(args, 'no_shadow', False),
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for category in args.categories:
        print(f"\n处理类别: {category}")
        category_indices = [
            i for i, s in enumerate(dataset.samples)
            if s["category"] == category
        ]
        if args.num_images_per_category:
            category_indices = category_indices[:args.num_images_per_category]

        category_dir = os.path.join(args.output_dir, category)
        os.makedirs(category_dir, exist_ok=True)

        if args.save_comparison:
            comparison_dir = os.path.join(category_dir, "comparison")
            os.makedirs(comparison_dir, exist_ok=True)

        for idx in tqdm(category_indices, desc=f"Generating {category}"):
            sample = dataset[idx]
            cond_tensor = sample["conditioning_pixel_values"]
            if args.use_colormap:
                cond_np = (cond_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            else:
                cond_np = (cond_tensor.squeeze().numpy() * 255).astype("uint8")
            cond_image = Image.fromarray(cond_np)

            # 二值掩码需要传 tensor 给 pipeline，避免 PIL 自动转 RGB 导致通道不匹配
            if args.use_colormap:
                pipeline_cond = cond_image
            else:
                pipeline_cond = cond_tensor.unsqueeze(0)  # [1, 1, H, W]

            original_image = Image.open(dataset.samples[idx]["image_path"]).convert("RGB")
            prompt = sample["prompt"]
            original_name = Path(dataset.samples[idx]["image_path"]).stem

            if args.use_mask_ca:
                from torchvision import transforms as T
                cond_tensor = T.ToTensor()(cond_image).unsqueeze(0).to(device)
                from models.mask_cross_attention import BOOST_LAYERED, BOOST_MILD
                if args.ca_mild:
                    _ab = BOOST_MILD
                elif args.ca_layered:
                    _ab = BOOST_LAYERED
                else:
                    _ab = None
                _bs = getattr(args, 'ca_boost_scale', 1.0)
                if _ab is not None and args.ca_timestep_gate:
                    from models.mask_cross_attention import set_mask_ca_data_with_bases
                    ob = {r: v[0] * _bs for r, v in _ab.items()}
                    sb = {r: v[1] * _bs for r, v in _ab.items()}
                    set_mask_ca_data_with_bases(cond_tensor, ob, sb)
                else:
                    from models.mask_cross_attention import set_mask_ca_data
                    if _ab is not None:
                        ob = {r: v[0] * _bs for r, v in _ab.items()}
                        sb = {r: v[1] * _bs for r, v in _ab.items()}
                    else:
                        ob, sb = args.ca_obj_boost * _bs, args.ca_shadow_boost * _bs
                    set_mask_ca_data(cond_tensor, ob, sb)

            for var_idx in range(args.num_variations_per_image):
                unique_seed = args.seed + idx * 1000 + var_idx
                generator = torch.Generator(device=device).manual_seed(unique_seed)

                with torch.no_grad():
                    output = pipeline(
                        prompt=prompt,
                        image=pipeline_cond,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                        generator=generator,
                    ).images[0]

                output.save(os.path.join(category_dir, f"{original_name}_var{var_idx}.png"))

                if args.save_comparison:
                    comparison = create_comparison_image([original_image, cond_image, output])
                    comparison.save(os.path.join(
                        comparison_dir, f"comparison_{original_name}_var{var_idx}.png"
                    ))

            cond_image.save(os.path.join(category_dir, f"{original_name}_condition.png"))

    print(f"\n生成完成! 结果保存在: {args.output_dir}")


# ===================== 模式: mask =====================

def run_mask_mode(args):
    """从掩码图片文件/目录推理"""
    pipeline, device = load_pipeline(args)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_comparison:
        comparison_dir = os.path.join(args.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)

    mask_path = Path(args.mask_path)
    if mask_path.is_file():
        mask_files = [mask_path]
    elif mask_path.is_dir():
        mask_files = list(mask_path.glob("*.png")) + list(mask_path.glob("*.jpg"))
    else:
        raise ValueError(f"无效的mask路径: {args.mask_path}")

    print(f"找到 {len(mask_files)} 个掩码文件")

    for mask_file in tqdm(mask_files, desc="生成图像"):
        mask_image = Image.open(mask_file).convert('RGB')
        if mask_image.size != (args.resolution, args.resolution):
            mask_image = mask_image.resize((args.resolution, args.resolution), Image.LANCZOS)

        if args.use_mask_ca:
            from torchvision import transforms as T
            cond_tensor = T.ToTensor()(mask_image).unsqueeze(0).to(device)
            from models.mask_cross_attention import BOOST_LAYERED, BOOST_MILD
            if args.ca_mild:
                _ab = BOOST_MILD
            elif args.ca_layered:
                _ab = BOOST_LAYERED
            else:
                _ab = None
            _bs = getattr(args, 'ca_boost_scale', 1.0)
            if _ab is not None and args.ca_timestep_gate:
                from models.mask_cross_attention import set_mask_ca_data_with_bases
                ob = {r: v[0] * _bs for r, v in _ab.items()}
                sb = {r: v[1] * _bs for r, v in _ab.items()}
                set_mask_ca_data_with_bases(cond_tensor, ob, sb)
            else:
                from models.mask_cross_attention import set_mask_ca_data
                if _ab is not None:
                    ob = {r: v[0] * _bs for r, v in _ab.items()}
                    sb = {r: v[1] * _bs for r, v in _ab.items()}
                else:
                    ob, sb = args.ca_obj_boost * _bs, args.ca_shadow_boost * _bs
                set_mask_ca_data(cond_tensor, ob, sb)

        for var_idx in range(args.num_variations_per_image):
            unique_seed = args.seed + var_idx
            generator = torch.Generator(device=device).manual_seed(unique_seed)

            with torch.no_grad():
                output = pipeline(
                    prompt=args.prompt,
                    image=mask_image,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                    generator=generator,
                ).images[0]

            output.save(os.path.join(args.output_dir, f"{mask_file.stem}_var{var_idx}.png"))

            if args.save_comparison:
                comparison = create_comparison_image([mask_image, output])
                comparison.save(os.path.join(
                    comparison_dir, f"comparison_{mask_file.stem}_var{var_idx}.png"
                ))

    print(f"\n生成完成! 结果保存在: {args.output_dir}")


# ===================== 模式: xml =====================

def run_xml_mode(args):
    """从XML标注文件推理"""
    pipeline, device = load_pipeline(args)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"解析XML文件: {args.xml_path}")
    xml_data = parse_xml_annotation(args.xml_path)
    print(f"图像尺寸: {xml_data['width']}x{xml_data['height']}, 对象数: {len(xml_data['objects'])}")

    mask_image = create_mask_from_xml(xml_data, (args.resolution, args.resolution))

    xml_stem = Path(args.xml_path).stem
    if args.save_mask:
        mask_path = os.path.join(args.output_dir, f"{xml_stem}_mask.png")
        mask_image.save(mask_path)
        print(f"保存掩码: {mask_path}")

    # 生成 prompt
    if args.categories:
        categories = args.categories
    else:
        categories = list(set([
            obj['name'] for obj in xml_data['objects'] if 'shadow' not in obj['name']
        ]))
    prompt = args.prompt_prefix + (" and ".join(categories) if categories else "object")
    print(f"使用prompt: {prompt}")

    for var_idx in range(args.num_variations_per_image):
        unique_seed = args.seed + var_idx
        generator = torch.Generator(device=device).manual_seed(unique_seed)

        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                image=mask_image,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                generator=generator,
            ).images[0]

        output.save(os.path.join(args.output_dir, f"{xml_stem}_generated_var{var_idx}.png"))

    print(f"\n生成完成! 结果保存在: {args.output_dir}")


# ===================== 参数解析 =====================

def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet-Seg 推理脚本")

    # 推理模式
    parser.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "mask", "xml"],
        help="推理模式: dataset(从数据集), mask(从掩码图片), xml(从XML标注)",
    )

    # 模型参数
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="预训练SD模型路径",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        required=True,
        help="训练好的ControlNet模型路径",
    )

    # 输出参数
    parser.add_argument("--output_dir", type=str, default="./outputs/controlnet")
    parser.add_argument("--resolution", type=int, default=512)

    # dataset 模式参数
    parser.add_argument("--data_dir", type=str, default="./dataset/SCTD", help="数据集目录路径")
    parser.add_argument("--split", type=str, default=None, choices=["train", "test"],
        help="使用数据集划分 (train/test)，默认None=全部数据")
    parser.add_argument("--split_file", type=str, default=None,
        help="split.json 路径 (默认 data_dir/split.json)")
    parser.add_argument("--categories", nargs="+", default=["aircraft", "ship", "human", "artificial fishing reef"])
    parser.add_argument("--num_images_per_category", type=int, default=None,
                        help="每个类别生成的图像数 (None=全部)")
    parser.add_argument("--prompt_prefix", type=str, default="an underwater sonar image of ")
    parser.add_argument("--use_colormap", action="store_true", default=True)
    parser.add_argument("--use_binary_mask", action="store_true")
    parser.add_argument("--no_shadow", action="store_true",
                        help="掩码中不包含阴影区域，仅保留目标和背景")

    # mask 模式参数
    parser.add_argument("--mask_path", type=str, help="掩码图片路径 (文件或目录)")
    parser.add_argument("--prompt", type=str, default="an underwater sonar image of ship",
                        help="mask模式的文本提示")

    # xml 模式参数
    parser.add_argument("--xml_path", type=str, help="XML标注文件路径")
    parser.add_argument("--save_mask", action="store_true", help="保存生成的掩码图像")

    # 生成参数
    parser.add_argument("--num_variations_per_image", type=int, default=4, help="每个mask生成的变体数")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                        help="ControlNet条件强度 (0-1)")
    parser.add_argument("--seed", type=int, default=42)

    # 模块开关 (与训练脚本保持一致)
    # 历史上 ERM 模块改名为 RBE (Region Boundary Enhancement)，两个 flag 等价
    parser.add_argument("--use_erm", "--use_rbe", dest="use_erm", action="store_true",
                        help="使用 RBE/ERM 区域边界增强模块（同义）")
    parser.add_argument("--use_mask_ca", action="store_true", help="使用掩码引导交叉注意力 (MaskCA)")
    parser.add_argument("--ca_obj_boost", type=float, default=0.5, help="物体区域交叉注意力增强系数")
    parser.add_argument("--ca_shadow_boost", type=float, default=0.3, help="阴影区域交叉注意力增强系数")
    parser.add_argument("--ca_layered", action="store_true", help="使用分层 boost")
    parser.add_argument("--ca_mild", action="store_true", help="使用温和分层 boost")
    parser.add_argument("--ca_timestep_gate", action="store_true", help="使用时步门控")
    parser.add_argument("--ca_gate_mid", type=float, default=400.0, help="时步门控 sigmoid 中心点")
    parser.add_argument("--ca_gate_temp", type=float, default=100.0, help="时步门控 sigmoid 温度")
    parser.add_argument("--ca_boost_scale", type=float, default=1.0,
                        help="MapCA boost 全局缩放因子 (0=关闭MapCA效果, 1=原始, <1=减弱, >1=增强)")
    # 其他参数
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_comparison", action="store_true", help="保存对比图")

    args = parser.parse_args()

    if args.use_binary_mask:
        args.use_colormap = False

    # 校验模式参数
    if args.mode == "mask" and not args.mask_path:
        parser.error("mask模式需要 --mask_path 参数")
    if args.mode == "xml" and not args.xml_path:
        parser.error("xml模式需要 --xml_path 参数")

    return args


# ===================== 入口 =====================

def main():
    args = parse_args()

    if args.mode == "dataset":
        run_dataset_mode(args)
    elif args.mode == "mask":
        run_mask_mode(args)
    elif args.mode == "xml":
        run_xml_mode(args)


if __name__ == "__main__":
    main()
