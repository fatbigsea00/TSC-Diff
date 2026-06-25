@echo off
REM ============================================================
REM  TSC-Diff Stage 2 : ControlNet train + inference (TSCG full)
REM  TSCG = RBE + MapCA (mild, timestep-gated) + Region/Boundary Loss
REM  Requires Stage 1 fine-tuned SD as the base model.
REM ============================================================
setlocal
cd /d "%~dp0"
set REPO_ROOT=%~dp0..

REM ---- paths (edit to your environment) ----
set SD_BASE=%REPO_ROOT%\step1\checkpoints\sd-dfdb
set DATA_DIR=%REPO_ROOT%\dataset\ConstructedDataset
set SPLIT_FILE=%DATA_DIR%\split_4cat_70_plus_afr.json
set CN_DIR=%~dp0checkpoints\controlnet_full
set OUT_DIR=%~dp0outputs\full

if not exist "%SD_BASE%" (
  echo [ERROR] Stage 1 SD base not found: %SD_BASE%
  echo         Run step1\run_train_dfdb.bat first.
  exit /b 1
)

echo [1/2] Training ControlNet (TSCG full) ...
python utils\train_controlnet.py ^
  --pretrained_model_name_or_path "%SD_BASE%" ^
  --data_dir "%DATA_DIR%" ^
  --split train ^
  --split_file "%SPLIT_FILE%" ^
  --output_dir "%CN_DIR%" ^
  --logging_dir "%~dp0logs\controlnet_full" ^
  --resolution 512 ^
  --prompt_prefix "an underwater sonar image of " ^
  --train_batch_size 2 ^
  --gradient_accumulation_steps 4 ^
  --num_train_epochs 100 ^
  --learning_rate 5e-5 ^
  --zero_conv_lr_mult 10.0 ^
  --lr_scheduler cosine ^
  --lr_warmup_steps 500 ^
  --validation_steps 500 ^
  --checkpointing_steps 500 ^
  --num_validation_images 4 ^
  --mixed_precision fp16 ^
  --dataloader_num_workers 0 ^
  --max_grad_norm 1.0 ^
  --seed 42 ^
  --use_rbe ^
  --use_mask_ca ^
  --ca_mild ^
  --ca_timestep_gate ^
  --ca_gate_mid 400.0 ^
  --ca_gate_temp 100.0 ^
  --use_region_loss ^
  --rl_w_obj 1.0 ^
  --rl_w_shadow 0.5 ^
  --rl_w_boundary 1.0 ^
  --rl_boundary_width 3 ^
  --rl_gate_mid 400.0 ^
  --rl_gate_temp 100.0
if errorlevel 1 (
  echo [ERROR] training failed
  exit /b 1
)

echo.
echo [2/2] Inference on test split (4 cats, 4 variations/image) ...
python utils\inference_controlnet.py ^
  --mode dataset ^
  --pretrained_model_name_or_path "%SD_BASE%" ^
  --controlnet_model_path "%CN_DIR%\controlnet" ^
  --data_dir "%DATA_DIR%" ^
  --split test ^
  --split_file "%SPLIT_FILE%" ^
  --categories aircraft ship human "artificial fishing reef" ^
  --output_dir "%OUT_DIR%" ^
  --num_variations_per_image 4 ^
  --num_inference_steps 50 ^
  --guidance_scale 7.5 ^
  --resolution 512 ^
  --seed 42 ^
  --mixed_precision fp16 ^
  --use_rbe ^
  --use_mask_ca ^
  --ca_mild ^
  --ca_timestep_gate ^
  --ca_gate_mid 400.0 ^
  --ca_gate_temp 100.0 ^
  --save_comparison

echo.
echo [DONE] ControlNet: %CN_DIR%   Generated images: %OUT_DIR%
endlocal
