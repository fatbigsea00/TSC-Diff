"""
ControlNet-Seg inference script

Supports three inference modes:
1. dataset: read masks from the constructed dataset for inference
2. mask:    inference from mask image files/directories
3. xml:     inference from X-AnyLabeling XML annotation files
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
from controlnet_dataset import ConstructedControlNetDataset, CATEGORY_COLORS


# ===================== XML parsing utilities =====================

def parse_xml_annotation(xml_path: str) -> Dict:
    """Parse an XML file in X-AnyLabeling format"""
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
    """Create an RGB colormap mask from XML data"""
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

    # Draw shadows first (lower layer), then objects (upper layer)
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


# ===================== General utilities =====================

def create_mask_from_polygon(
    polygons: List[List[tuple]],
    labels: List[str],
    image_size: tuple = (512, 512),
    use_colormap: bool = True,
) -> Image.Image:
    """Create a mask from polygon coordinates"""
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
    """Create a horizontally concatenated comparison image"""
    width = images[0].width
    height = images[0].height
    comparison = Image.new('RGB', (width * len(images), height))
    for i, img in enumerate(images):
        comparison.paste(img.resize((width, height), Image.BILINEAR), (width * i, 0))
    return comparison


# ===================== Pipeline loading =====================

def load_pipeline(args):
    """Load the ControlNet + SD pipeline"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mixed_precision == "fp16":
        dtype = torch.float16
    elif args.mixed_precision == "bf16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Load ControlNet
    print(f"Loading ControlNet model: {args.controlnet_model_path}")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model_path, torch_dtype=dtype,
    )

    # RBE: Region Boundary Enhancement module
    if args.use_rbe:
        from models.rbe_controlnet import apply_rbe_to_controlnet
        controlnet = apply_rbe_to_controlnet(controlnet, input_resolution=args.resolution // 8)
        weight_path = os.path.join(args.controlnet_model_path, "rbe_controlnet.pth")
        if not os.path.exists(weight_path):
            print(f"[WARN] rbe_controlnet.pth not found; "
                  f"the RBE module will remain randomly initialized -> inference results may be incorrect")
            weight_path = None
        if weight_path is not None:
            state_dict = torch.load(weight_path, map_location="cpu", weights_only=True)
            missing, unexpected = controlnet.load_state_dict(state_dict, strict=False)
            print(f"Loaded RBE trained weights: {weight_path}")
            if missing:
                print(f"  [info] missing keys: {len(missing)} (first 3: {missing[:3]})")
            if unexpected:
                print(f"  [info] unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
        controlnet = controlnet.to(dtype=dtype)

    # Create the pipeline
    print(f"Loading SD model: {args.pretrained_model_name_or_path}")
    pipeline = StableDiffusionControlNetPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)

    # Mask-guided cross-attention
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


# ===================== Mode: dataset =====================

def run_dataset_mode(args):
    """Read masks from the constructed dataset for inference"""
    pipeline, device = load_pipeline(args)

    print("Loading dataset...")
    dataset = ConstructedControlNetDataset(
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
        print(f"\nProcessing category: {category}")
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

            # Binary masks must be passed as a tensor to the pipeline to avoid PIL auto-converting to RGB and causing a channel mismatch
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

    print(f"\nGeneration complete! Results saved to: {args.output_dir}")


# ===================== Mode: mask =====================

def run_mask_mode(args):
    """Inference from mask image files/directories"""
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
        raise ValueError(f"Invalid mask path: {args.mask_path}")

    print(f"Found {len(mask_files)} mask files")

    for mask_file in tqdm(mask_files, desc="Generating images"):
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

    print(f"\nGeneration complete! Results saved to: {args.output_dir}")


# ===================== Mode: xml =====================

def run_xml_mode(args):
    """Inference from XML annotation files"""
    pipeline, device = load_pipeline(args)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Parsing XML file: {args.xml_path}")
    xml_data = parse_xml_annotation(args.xml_path)
    print(f"Image size: {xml_data['width']}x{xml_data['height']}, object count: {len(xml_data['objects'])}")

    mask_image = create_mask_from_xml(xml_data, (args.resolution, args.resolution))

    xml_stem = Path(args.xml_path).stem
    if args.save_mask:
        mask_path = os.path.join(args.output_dir, f"{xml_stem}_mask.png")
        mask_image.save(mask_path)
        print(f"Saved mask: {mask_path}")

    # Generate prompt
    if args.categories:
        categories = args.categories
    else:
        categories = list(set([
            obj['name'] for obj in xml_data['objects'] if 'shadow' not in obj['name']
        ]))
    prompt = args.prompt_prefix + (" and ".join(categories) if categories else "object")
    print(f"Using prompt: {prompt}")

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

    print(f"\nGeneration complete! Results saved to: {args.output_dir}")


# ===================== Argument parsing =====================

def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet-Seg inference script")

    # Inference mode
    parser.add_argument(
        "--mode",
        type=str,
        default="dataset",
        choices=["dataset", "mask", "xml"],
        help="Inference mode: dataset (from dataset), mask (from mask images), xml (from XML annotations)",
    )

    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to the pretrained SD model",
    )
    parser.add_argument(
        "--controlnet_model_path",
        type=str,
        required=True,
        help="Path to the trained ControlNet model",
    )

    # Output parameters
    parser.add_argument("--output_dir", type=str, default="./outputs/controlnet")
    parser.add_argument("--resolution", type=int, default=512)

    # dataset mode parameters
    parser.add_argument("--data_dir", type=str, default="./dataset/ConstructedDataset", help="Path to the dataset root directory")
    parser.add_argument("--split", type=str, default=None, choices=["train", "test"],
        help="Which dataset split to use (train/test); default None = all data")
    parser.add_argument("--split_file", type=str, default=None,
        help="Path to split.json (default: data_dir/split.json)")
    parser.add_argument("--categories", nargs="+", default=["aircraft", "ship", "human", "artificial fishing reef"])
    parser.add_argument("--num_images_per_category", type=int, default=None,
                        help="Number of images to generate per category (None = all)")
    parser.add_argument("--prompt_prefix", type=str, default="an underwater sonar image of ")
    parser.add_argument("--use_colormap", action="store_true", default=True)
    parser.add_argument("--use_binary_mask", action="store_true")
    parser.add_argument("--no_shadow", action="store_true",
                        help="Exclude shadow regions from the mask, keeping only objects and background")

    # mask mode parameters
    parser.add_argument("--mask_path", type=str, help="Path to mask image (file or directory)")
    parser.add_argument("--prompt", type=str, default="an underwater sonar image of ship",
                        help="Text prompt for mask mode")

    # xml mode parameters
    parser.add_argument("--xml_path", type=str, help="Path to the XML annotation file")
    parser.add_argument("--save_mask", action="store_true", help="Save the generated mask image")

    # Generation parameters
    parser.add_argument("--num_variations_per_image", type=int, default=4, help="Number of variations to generate per mask")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0,
                        help="ControlNet conditioning strength (0-1)")
    parser.add_argument("--seed", type=int, default=42)

    # Module switches (kept consistent with the training script)
    parser.add_argument("--use_rbe", dest="use_rbe", action="store_true",
                        help="Use the RBE Region Boundary Enhancement module")
    parser.add_argument("--use_mask_ca", action="store_true", help="Use mask-guided cross-attention (MaskCA)")
    parser.add_argument("--ca_obj_boost", type=float, default=0.5, help="Cross-attention boost factor for object regions")
    parser.add_argument("--ca_shadow_boost", type=float, default=0.3, help="Cross-attention boost factor for shadow regions")
    parser.add_argument("--ca_layered", action="store_true", help="Use layered boost")
    parser.add_argument("--ca_mild", action="store_true", help="Use mild layered boost")
    parser.add_argument("--ca_timestep_gate", action="store_true", help="Use timestep gating")
    parser.add_argument("--ca_gate_mid", type=float, default=400.0, help="Center point of the timestep gating sigmoid")
    parser.add_argument("--ca_gate_temp", type=float, default=100.0, help="Temperature of the timestep gating sigmoid")
    parser.add_argument("--ca_boost_scale", type=float, default=1.0,
                        help="Global scaling factor for the MapCA boost (0=disable MapCA effect, 1=original, <1=weaken, >1=strengthen)")
    # Other parameters
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--save_comparison", action="store_true", help="Save the comparison image")

    args = parser.parse_args()

    if args.use_binary_mask:
        args.use_colormap = False

    # Validate mode-specific parameters
    if args.mode == "mask" and not args.mask_path:
        parser.error("mask mode requires the --mask_path argument")
    if args.mode == "xml" and not args.xml_path:
        parser.error("xml mode requires the --xml_path argument")

    return args


# ===================== Entry point =====================

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
