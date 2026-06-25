@echo off
REM ============================================================
REM  TSC-Diff Stage 1 : SD fine-tune + DFDB (PFA + LFA)
REM  Base SD (sd-v1-5)  ->  fine-tuned SD for Stage 2 ControlNet
REM  DFDB = --use_dsr (PFA) + --use_lfa (LFA)
REM ============================================================
setlocal
cd /d "%~dp0"
set REPO_ROOT=%~dp0..

REM ---- paths (edit to your environment) ----
set SD_BASE=%REPO_ROOT%\pretrained\sd-v1-5
set DATA_DIR=%REPO_ROOT%\dataset\ConstructedDataset
set SPLIT_FILE=%DATA_DIR%\split_4cat_70_plus_afr.json
set OUT_DIR=%~dp0checkpoints\sd-dfdb

if not exist "%SD_BASE%" (
  echo [ERROR] base SD model not found: %SD_BASE%
  echo         Download stable-diffusion-v1-5 and put it there, or edit SD_BASE.
  exit /b 1
)

echo [Training] SD fine-tune with DFDB ...
accelerate launch --num_processes=1 --mixed_precision="fp16" ^
  utils\train.py ^
  --pretrained_model_name_or_path "%SD_BASE%" ^
  --train_data_dir "%DATA_DIR%" ^
  --split train ^
  --split_file "%SPLIT_FILE%" ^
  --output_dir "%OUT_DIR%" ^
  --resolution 512 ^
  --random_flip ^
  --train_batch_size 4 ^
  --dataloader_num_workers 0 ^
  --gradient_accumulation_steps 1 ^
  --gradient_checkpointing ^
  --max_train_steps 5000 ^
  --checkpointing_steps 500 ^
  --learning_rate 1e-05 ^
  --max_grad_norm 1 ^
  --lr_scheduler "constant" ^
  --lr_warmup_steps 0 ^
  --mixed_precision fp16 ^
  --seed 42 ^
  --use_dsr ^
  --dsr_prob 0.5 ^
  --dsr_slic_segments 64 ^
  --use_lfa ^
  --lfa_alpha 0.05 ^
  --lfa_prob 0.15

echo.
echo [DONE] fine-tuned SD saved to: %OUT_DIR%
endlocal
