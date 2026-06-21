#!/bin/bash

LOG_FILE="SOTA.train_5e.txt"

# Activate conda properly
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mt_env

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

echo "=====================================" | tee $LOG_FILE
echo "Running on: $(hostname)" | tee -a $LOG_FILE
echo "Start time: $(date)" | tee -a $LOG_FILE
echo "=====================================" | tee -a $LOG_FILE

# GPU info
nvidia-smi | tee -a $LOG_FILE
export HF_TOKEN="hf_key_use_it"

# Optional but recommended
export TOKENIZERS_PARALLELISM=false





# ✅ Correct argument passing
python /home/vikas_blackwell/vikas/IT2_ft/uni_dir/rs_lora3/step5e_subword_reg.py --pipeline A B C D 2>&1 | tee -a $LOG_FILE

