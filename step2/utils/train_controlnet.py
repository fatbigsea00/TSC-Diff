"""
ControlNet-Seg training script (final unified version)

Core idea:
- Input: segmentation map (colormap / binary mask) + text prompt
- Output: complete sonar image (background + object + shadow)

Optional modules (all controlled via command-line switches for convenient ablation studies):
  1. --use_colormap (default True)  : RGB Colormap as conditioning input (vs binary mask)
  2. --zero_conv_lr_mult        : Zero Conv layered learning rate
  3. --use_rbe                  : Region Boundary Enhancement module (RBE), enhances conditioning_embedding
  4. --use_mask_ca              : Map-Guided Cross-Attention (MapCA)
  5. --use_region_loss          : Region-weighted + boundary-focused noise loss

The SD backbone can point to different fine-tuned models via --pretrained_model_name_or_path
(e.g. sd-baseline / sd-dsr) to control whether DSR enhancement is used.
"""

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    StableDiffusionControlNetPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controlnet_dataset import ConstructedControlNetDataset, controlnet_collate_fn

logger = get_logger(__name__)


def build_region_boundary_weight(cond_img, latent_h=64, latent_w=64,
                                  w_obj=1.0, w_shadow=0.5, w_boundary=1.0,
                                  boundary_width=3, timesteps=None,
                                  gate_mid=400.0, gate_temp=100.0):
    """
    From a colormap [B,3,H,W] in [0,1], build a spatial weight map [B,1,h,w].
    Three improvements over v1:
      1. Additive-only: background stays 1.0, regions get extra weight (no normalization)
      2. Timestep-aware: gate(t) fades extra weight to 0 at low noise (protects detail stage)
      3. Area-adaptive: smaller object regions get proportionally higher weight
    """
    with torch.no_grad():
        ch_max = torch.amax(cond_img, dim=1, keepdim=True)
        ch_min = torch.amin(cond_img, dim=1, keepdim=True)
        obj_mask = ((ch_max > 0.75) & (ch_min < 0.25)).float()

        mean_c = cond_img.mean(dim=1, keepdim=True)
        var_c = cond_img.var(dim=1, keepdim=True)
        shadow_mask = ((mean_c > 0.4) & (mean_c < 0.6) & (var_c < 0.005)).float()

        # area-adaptive: smaller object area → higher weight (clamped 1~4×)
        obj_ratio = obj_mask.flatten(1).mean(1, keepdim=True).unsqueeze(-1).unsqueeze(-1)
        adaptive_scale = (0.1 / obj_ratio.clamp(min=0.01)).clamp(1.0, 4.0)

        # boundary detection at latent resolution
        label = obj_mask * 2.0 + shadow_mask * 1.0
        label_small = F.interpolate(label, size=(latent_h, latent_w), mode='nearest')
        pad = boundary_width // 2
        lp = F.pad(label_small, [pad]*4, mode='replicate')
        patches = lp.unfold(2, boundary_width, 1).unfold(3, boundary_width, 1)
        is_boundary = (patches.amax(dim=(-1, -2)) != patches.amin(dim=(-1, -2))).float()

        # downsample masks to latent resolution
        obj_small = F.interpolate(obj_mask, size=(latent_h, latent_w), mode='bilinear', align_corners=False)
        shadow_small = F.interpolate(shadow_mask, size=(latent_h, latent_w), mode='bilinear', align_corners=False)

        # additive-only: background = 1.0, extra weight on regions/boundary
        extra = (w_obj * adaptive_scale * obj_small
                 + w_shadow * shadow_small
                 + w_boundary * is_boundary)

        # timestep gating: fade extra to 0 at low noise steps
        if timesteps is not None:
            gate = torch.sigmoid((timesteps.float() - gate_mid) / gate_temp)
            gate = gate.view(-1, 1, 1, 1)
            extra = extra * gate

        wmap = 1.0 + extra
    return wmap


def parse_args():
    parser = argparse.ArgumentParser(description="ControlNet-Seg training on the constructed dataset")
    
    # Data parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./dataset/ConstructedDataset",
        help="Path to the dataset root directory",
    )
    parser.add_argument("--split", type=str, default=None, choices=["train", "test"],
        help="Dataset split to use (train/test); default None = all data")
    parser.add_argument("--split_file", type=str, default=None,
        help="Path to split.json (default: data_dir/split.json)")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution",
    )
    parser.add_argument(
        "--prompt_prefix",
        type=str,
        default="an underwater sonar image of ",
        help="Prompt prefix",
    )
    parser.add_argument(
        "--use_colormap",
        action="store_true",
        default=True,
        help="Use RGB colormap as conditioning (default True)",
    )
    parser.add_argument(
        "--use_binary_mask",
        action="store_true",
        help="Use binary mask instead of colormap",
    )
    parser.add_argument(
        "--no_shadow",
        action="store_true",
        help="Exclude shadow regions from the mask, keeping only object and background",
    )
    
    # Model parameters
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Path to pretrained SD model",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained ControlNet model (optional, for continued training)",
    )
    
    # Training parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/controlnet_sctd",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Training batch size",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Maximum number of training steps",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Learning rate warmup steps",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    
    # Validation and saving
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=500,
        help="Validation frequency",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
    )
    
    # Misc
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="./logs/controlnet",
    )
    parser.add_argument(
        "--zero_conv_lr_mult",
        type=float,
        default=1.0,
        help="Zero conv layer learning rate multiplier (recommended 5-10)",
    )
    # ── Module switch: RBE ──
    parser.add_argument(
        "--use_rbe",
        action="store_true",
        help="Use RBE (Region Boundary Enhancement module) to enhance ControlNet conditioning_embedding",
    )
    # ── Module switch: MapCA ──
    parser.add_argument(
        "--use_mask_ca",
        action="store_true",
        help="Use Map-Guided Cross-Attention (MapCA)",
    )
    parser.add_argument(
        "--ca_obj_boost",
        type=float,
        default=0.5,
        help="MapCA: object region boost coefficient (uniform mode)",
    )
    parser.add_argument(
        "--ca_shadow_boost",
        type=float,
        default=0.3,
        help="MapCA: shadow region boost coefficient (uniform mode)",
    )
    parser.add_argument(
        "--ca_layered",
        action="store_true",
        help="Use layered boost (different boost coefficient per resolution)",
    )
    parser.add_argument(
        "--ca_mild",
        action="store_true",
        help="Use mild layered boost (only the 64px layer is slightly reduced, the rest same as uniform)",
    )
    parser.add_argument(
        "--ca_timestep_gate",
        action="store_true",
        help="Use timestep gating (strong modulation at high noise, weak modulation at low noise)",
    )
    parser.add_argument(
        "--ca_gate_mid",
        type=float,
        default=400.0,
        help="Timestep gating: sigmoid center point",
    )
    parser.add_argument(
        "--ca_gate_temp",
        type=float,
        default=100.0,
        help="Timestep gating: sigmoid temperature",
    )
    # ── Module switch: Region Loss ──
    parser.add_argument("--use_region_loss", action="store_true",
        help="Enable region-weighted + boundary-focused noise prediction loss")
    parser.add_argument("--rl_w_obj", type=float, default=1.0,
        help="Extra loss weight for object regions (additive)")
    parser.add_argument("--rl_w_shadow", type=float, default=0.5,
        help="Extra loss weight for shadow regions (additive)")
    parser.add_argument("--rl_w_boundary", type=float, default=1.0,
        help="Extra loss weight for boundary regions (additive)")
    parser.add_argument("--rl_boundary_width", type=int, default=3,
        help="Boundary dilation kernel size")
    parser.add_argument("--rl_gate_mid", type=float, default=400.0,
        help="Region loss timestep gating: sigmoid center point")
    parser.add_argument("--rl_gate_temp", type=float, default=100.0,
        help="Region loss timestep gating: sigmoid temperature")
    args = parser.parse_args()
    
    # Handle colormap arguments
    if args.use_binary_mask:
        args.use_colormap = False
    
    return args


def main():
    args = parse_args()
    
    # Initialize accelerator
    logging_dir = Path(args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
    )
    
    if args.seed is not None:
        set_seed(args.seed)
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model components
    print("Loading pretrained model...")
    
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae"
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    
    # Initialize or load ControlNet
    if args.controlnet_model_name_or_path:
        print(f"Loading pretrained ControlNet: {args.controlnet_model_name_or_path}")
        controlnet = ControlNetModel.from_pretrained(args.controlnet_model_name_or_path)
    else:
        print("Initializing new ControlNet from UNet...")
        # Determine number of input channels
        conditioning_channels = 3 if args.use_colormap else 1
        controlnet = ControlNetModel.from_unet(
            unet,
            conditioning_channels=conditioning_channels,
        )
    
    # RBE: Region Boundary Enhancement module —— replaces conditioning_embedding
    if args.use_rbe:
        from models.rbe_controlnet import apply_rbe_to_controlnet
        controlnet = apply_rbe_to_controlnet(controlnet, input_resolution=args.resolution // 8)
        print("  [RBE] Region Boundary Enhancement module enabled")
    
    # Freeze other components, train only ControlNet
    vae.requires_grad_(False)
    unet.requires_grad_(False)
    text_encoder.requires_grad_(False)
    controlnet.train()

    # MapCA: replace the UNet's cross-attention processors
    if args.use_mask_ca:
        from models.map_cross_attention import apply_map_guided_attention, BOOST_LAYERED, BOOST_MILD
        apply_map_guided_attention(unet)
        if args.ca_mild:
            _active_boost = BOOST_MILD
            print(f"  Mild layered boost: {_active_boost}")
        elif args.ca_layered:
            _active_boost = BOOST_LAYERED
            print(f"  Layered boost: {_active_boost}")
        else:
            _active_boost = None
            print(f"  Uniform boost: obj={args.ca_obj_boost}, shadow={args.ca_shadow_boost}")
        if args.ca_timestep_gate:
            print(f"  Timestep gate: mid={args.ca_gate_mid}, temp={args.ca_gate_temp}")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in controlnet.parameters() if p.requires_grad)
    print(f"ControlNet trainable parameters: {trainable_params:,}")
    
    # Set precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    vae.to(accelerator.device, dtype=torch.float32)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = ConstructedControlNetDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        prompt_prefix=args.prompt_prefix,
        use_colormap=args.use_colormap,
        split=args.split,
        split_file=args.split_file,
        no_shadow=args.no_shadow,
    )
    
    _num_workers = args.dataloader_num_workers if args.dataloader_num_workers > 0 else 4
    if os.name == "nt":
        _num_workers = 0

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=controlnet_collate_fn,
        num_workers=_num_workers,
        pin_memory=(_num_workers > 0),
        persistent_workers=(_num_workers > 0),
    )
    
    # Optimizer - supports layered learning rate
    # Zero conv layers use a higher learning rate because they are initialized to 0
    if args.zero_conv_lr_mult != 1.0:
        zero_conv_params = []
        other_params = []
        
        for name, param in controlnet.named_parameters():
            if param.requires_grad:
                if 'controlnet_down_blocks' in name or 'controlnet_mid_block' in name:
                    zero_conv_params.append(param)
                else:
                    other_params.append(param)
        
        zero_conv_lr = args.learning_rate * args.zero_conv_lr_mult
        print(f"Using layered learning rate:")
        print(f"  - Base learning rate: {args.learning_rate}")
        print(f"  - Zero Conv learning rate: {zero_conv_lr} ({args.zero_conv_lr_mult}x)")
        print(f"  - Number of base parameters: {sum(p.numel() for p in other_params):,}")
        print(f"  - Number of Zero Conv parameters: {sum(p.numel() for p in zero_conv_params):,}")
        
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': args.learning_rate},
            {'params': zero_conv_params, 'lr': zero_conv_lr}
        ],
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    else:
        optimizer = torch.optim.AdamW(
            controlnet.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    
    # Compute number of training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    
    # Learning rate scheduling
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )
    
    # Accelerator preparation
    controlnet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        controlnet, optimizer, train_dataloader, lr_scheduler
    )
    
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # Initialize swanlab (optional)
    try:
        import swanlab
        if accelerator.is_main_process:
            swanlab.init(
                project="controlnet-seg-sctd",
                experiment_name="sctd_controlnet_training",
                config=vars(args),
                logdir=args.logging_dir,
            )
        use_swanlab = True
    except ImportError:
        use_swanlab = False
        print("swanlab not installed, skipping logging")
    
    # Start training
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    
    print("=" * 60)
    print("  ControlNet-Seg training (unified version)")
    print("=" * 60)
    print(f"  SD backbone   = {args.pretrained_model_name_or_path}")
    print(f"  Condition type = {'RGB Colormap' if args.use_colormap else 'Binary Mask'}")
    print(f"  Num samples   = {len(train_dataset)}")
    print(f"  Epochs        = {num_train_epochs}")
    print(f"  Batch size    = {args.train_batch_size}")
    print(f"  Grad accum    = {args.gradient_accumulation_steps}")
    print(f"  Total batch size  = {total_batch_size}")
    print(f"  Total training steps = {args.max_train_steps}")
    print(f"  Learning rate = {args.learning_rate}")
    print(f"  Zero Conv LR  = {args.learning_rate * args.zero_conv_lr_mult} ({args.zero_conv_lr_mult}x)")
    print("-" * 60)
    print(f"  [RBE]          = {'ON' if args.use_rbe else 'OFF'}")
    print(f"  [MapCA]        = {'ON' if args.use_mask_ca else 'OFF'}")
    print(f"  [Region Loss]  = {'ON' if args.use_region_loss else 'OFF'}")
    print(f"  [No Shadow]    = {'ON (object+background only)' if args.no_shadow else 'OFF (includes shadow)'}")
    print("=" * 60)
    
    global_step = 0
    first_epoch = 0
    
    # Resume training
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        
        if path is not None:
            accelerator.print(f"Resuming from checkpoint: {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            first_epoch = global_step // num_update_steps_per_epoch
    
    # Training loop
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    if args.use_region_loss:
        print(f"  Region+Boundary loss v2: w_obj={args.rl_w_obj}, w_shadow={args.rl_w_shadow}, "
              f"w_boundary={args.rl_w_boundary}, boundary_width={args.rl_boundary_width}, "
              f"gate_mid={args.rl_gate_mid}, gate_temp={args.rl_gate_temp}")

    for epoch in range(first_epoch, num_train_epochs):
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(controlnet):
                # Encode target image into latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=torch.float32)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()
                
                # Add noise
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                noisy_latents = noisy_latents.to(dtype=weight_dtype)
                
                # Get conditioning image (segmentation map)
                controlnet_image = batch["conditioning_pixel_values"].to(dtype=weight_dtype)

                # Encode text
                text_input_ids = tokenizer(
                    batch["prompts"],
                    max_length=tokenizer.model_max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).input_ids.to(accelerator.device)
                
                encoder_hidden_states = text_encoder(text_input_ids)[0].to(dtype=weight_dtype)
                
                # ControlNet forward pass
                down_block_res_samples, mid_block_res_sample = controlnet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    controlnet_cond=controlnet_image,
                    return_dict=False,
                )
                
                # Ensure ControlNet output dtype matches UNet
                down_block_res_samples = [
                    sample.to(dtype=weight_dtype) for sample in down_block_res_samples
                ]
                mid_block_res_sample = mid_block_res_sample.to(dtype=weight_dtype)
                
                # UNet predicts noise (with ControlNet conditioning)
                if args.use_mask_ca:
                    from models.map_cross_attention import set_map_ca_data
                    if _active_boost is not None:
                        ob = {r: v[0] for r, v in _active_boost.items()}
                        sb = {r: v[1] for r, v in _active_boost.items()}
                    else:
                        ob, sb = args.ca_obj_boost, args.ca_shadow_boost
                    set_map_ca_data(
                        controlnet_image, ob, sb,
                        timestep=timesteps if args.ca_timestep_gate else None,
                        gate_mid=args.ca_gate_mid, gate_temp=args.ca_gate_temp,
                    )
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=encoder_hidden_states,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample
                
                # Compute loss
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise.to(dtype=weight_dtype)
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps).to(dtype=weight_dtype)
                else:
                    raise ValueError(f"Unknown prediction type")
                
                if args.use_region_loss:
                    wmap = build_region_boundary_weight(
                        controlnet_image.float(),
                        latent_h=model_pred.shape[2], latent_w=model_pred.shape[3],
                        w_obj=args.rl_w_obj, w_shadow=args.rl_w_shadow,
                        w_boundary=args.rl_w_boundary,
                        boundary_width=args.rl_boundary_width,
                        timesteps=timesteps,
                        gate_mid=args.rl_gate_mid, gate_temp=args.rl_gate_temp,
                    )
                    per_pixel = (model_pred.float() - target.float()) ** 2
                    loss = (per_pixel * wmap).mean()
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # Backward pass
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(controlnet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                train_loss += loss.detach().item()
                
                # Logging
                if global_step % 100 == 0:
                    avg_loss = train_loss / 100
                    if accelerator.is_main_process and use_swanlab:
                        swanlab.log({
                            "train_loss": avg_loss, 
                            "lr": lr_scheduler.get_last_lr()[0]
                        }, step=global_step)
                    train_loss = 0.0
                
                # Save checkpoint
                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        print(f"Saved checkpoint: {save_path}")
                
                # Validation
                if global_step % args.validation_steps == 0:
                    if accelerator.is_main_process:
                        print(f"\nValidating... (step {global_step})")
                        validation(
                            args,
                            accelerator,
                            controlnet,
                            unet,
                            vae,
                            text_encoder,
                            tokenizer,
                            noise_scheduler,
                            weight_dtype,
                            global_step,
                        )
            
            if global_step >= args.max_train_steps:
                break
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        controlnet_model = accelerator.unwrap_model(controlnet)
        save_dir = os.path.join(args.output_dir, "controlnet")
        os.makedirs(save_dir, exist_ok=True)

        if args.use_rbe:
            weight_file = os.path.join(save_dir, "rbe_controlnet.pth")
            torch.save(controlnet_model.state_dict(), weight_file)
            try:
                controlnet_model.save_pretrained(save_dir)
            except Exception as e:
                print(f"  Warning: save_pretrained failed ({e}), falling back to torch.save")
                torch.save(controlnet_model.state_dict(),
                           os.path.join(save_dir, "diffusion_pytorch_model.bin"))
                controlnet_model.save_config(save_dir)
            print(f"Training complete! RBE-ControlNet saved to: {save_dir}")
        else:
            controlnet_model.save_pretrained(save_dir)
            print(f"Training complete! ControlNet saved to: {save_dir}")
        
        if use_swanlab:
            swanlab.finish()
    
    accelerator.end_training()


def validation(args, accelerator, controlnet, unet, vae, text_encoder, tokenizer, 
               noise_scheduler, weight_dtype, global_step):
    """Validation function - generates sample images"""
    import random
    from PIL import Image
    import numpy as np
    
    # Set to evaluation mode
    controlnet.eval()
    controlnet_model = accelerator.unwrap_model(controlnet)
    
    # Ensure all models use float32 to avoid dtype mismatch during validation
    # Save original dtype for restoration
    orig_unet_dtype = next(unet.parameters()).dtype
    orig_text_encoder_dtype = next(text_encoder.parameters()).dtype
    
    vae.to(dtype=torch.float32)
    unet.to(dtype=torch.float32)
    text_encoder.to(dtype=torch.float32)
    controlnet_model.to(dtype=torch.float32)
    
    # Create pipeline - use float32 to avoid dtype mismatch
    pipeline = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet_model,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline = pipeline.to(accelerator.device, dtype=torch.float32)
    pipeline.set_progress_bar_config(disable=True)
    
    # Load validation data
    val_dataset = ConstructedControlNetDataset(
        data_dir=args.data_dir,
        resolution=args.resolution,
        prompt_prefix=args.prompt_prefix,
        use_colormap=args.use_colormap,
        no_shadow=args.no_shadow,
    )
    
    val_indices = random.sample(
        range(len(val_dataset)), 
        min(args.num_validation_images, len(val_dataset))
    )
    
    validation_dir = os.path.join(args.output_dir, "validation", f"step-{global_step}")
    os.makedirs(validation_dir, exist_ok=True)
    
    for i, idx in enumerate(val_indices):
        sample = val_dataset[idx]

        # Prepare conditioning image
        cond_image = sample["conditioning_pixel_values"]  # [C, H, W]
        if args.use_colormap:
            cond_image_np = (cond_image.permute(1, 2, 0).numpy() * 255).astype("uint8")
        else:
            cond_image_np = (cond_image.squeeze().numpy() * 255).astype("uint8")
        cond_image_pil = Image.fromarray(cond_image_np)

        # Binary masks must be passed to the pipeline as a tensor, to avoid PIL auto-converting to RGB and causing a channel mismatch
        if args.use_colormap:
            pipeline_cond = cond_image_pil
        else:
            pipeline_cond = cond_image.unsqueeze(0)  # [1, 1, H, W]

        # Target image (ground truth)
        gt_image = (sample["pixel_values"].permute(1, 2, 0).numpy() + 1) / 2
        gt_image = (gt_image * 255).clip(0, 255).astype("uint8")
        gt_image_pil = Image.fromarray(gt_image)

        prompt = sample["prompt"]

        # Generate
        if args.use_mask_ca:
            from models.map_cross_attention import BOOST_LAYERED, BOOST_MILD
            if args.ca_mild:
                _val_boost = BOOST_MILD
            elif args.ca_layered:
                _val_boost = BOOST_LAYERED
            else:
                _val_boost = None
            cond_tensor_val = cond_image.unsqueeze(0).to(accelerator.device)
            if _val_boost is not None and args.ca_timestep_gate:
                from models.map_cross_attention import set_map_ca_data_with_bases, install_timestep_hook
                ob = {r: v[0] for r, v in _val_boost.items()}
                sb = {r: v[1] for r, v in _val_boost.items()}
                set_map_ca_data_with_bases(cond_tensor_val, ob, sb)
                unet._map_ca_gate_mid = args.ca_gate_mid
                unet._map_ca_gate_temp = args.ca_gate_temp
                install_timestep_hook(unet)
            else:
                from models.map_cross_attention import set_map_ca_data
                if _val_boost is not None:
                    ob = {r: v[0] for r, v in _val_boost.items()}
                    sb = {r: v[1] for r, v in _val_boost.items()}
                else:
                    ob, sb = args.ca_obj_boost, args.ca_shadow_boost
                set_map_ca_data(cond_tensor_val, ob, sb)
        with torch.no_grad():
            output = pipeline(
                prompt=prompt,
                image=pipeline_cond,
                num_inference_steps=50,
                guidance_scale=7.5,
            ).images[0]
        
        # Save results
        output.save(os.path.join(validation_dir, f"generated_{i}.png"))
        gt_image_pil.save(os.path.join(validation_dir, f"ground_truth_{i}.png"))
        cond_image_pil.save(os.path.join(validation_dir, f"condition_{i}.png"))
        
        with open(os.path.join(validation_dir, f"prompt_{i}.txt"), "w") as f:
            f.write(prompt)
    
    print(f"Validation images saved to: {validation_dir}")
    
    del pipeline
    torch.cuda.empty_cache()
    
    # Restore dtypes used during training
    vae.to(dtype=torch.float32)  # VAE always stays float32
    unet.to(dtype=orig_unet_dtype)
    text_encoder.to(dtype=orig_text_encoder_dtype)
    
    # Restore training mode
    controlnet.train()


if __name__ == "__main__":
    main()

