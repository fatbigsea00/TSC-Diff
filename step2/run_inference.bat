@echo off
REM ============================================================
REM  TSC-Diff Stage 2 : ControlNet inference only (TSCG full)
REM  Uses a trained ControlNet to generate sonar images.
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

if not exist "%CN_DIR%\controlnet" (
  echo [ERROR] trained ControlNet not found: %CN_DIR%\controlnet
  echo         Run step2\run_train.bat first.
  exit /b 1
)

echo [Inference] dataset mode (test split) ...
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
echo [DONE] Generated images: %OUT_DIR%
endlocal
