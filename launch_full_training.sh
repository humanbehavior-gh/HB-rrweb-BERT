#!/bin/bash

# Full-scale RRWebBERT training script
# This will train on all 30,000+ RRWEB files with optimized parameters

echo "Starting full-scale RRWebBERT training..."
echo "This will train on all RRWEB files in the dataset"
echo "Using the tokenizer that's currently being built"

# Check if tokenizer exists
TOKENIZER_PATH="/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest"
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "Warning: tokenizer_model_latest not found, using test tokenizer"
    TOKENIZER_PATH="/home/ubuntu/rrweb_tokenizer/tokenizer_model_test"
fi

# Set wandb environment
export WANDB_PROJECT="rrweb-bert"
export WANDB_MODE="online"

# Launch training with optimized parameters for full dataset
python train_rrweb_bert.py \
    --data_dir /home/ubuntu/embeddingV2/rrweb_data \
    --tokenizer_path "$TOKENIZER_PATH" \
    --output_dir ./rrweb-bert-full \
    --model_size base \
    --max_length 2048 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_epochs 5 \
    --warmup_steps 1000 \
    --save_steps 5000 \
    --logging_steps 500 \
    --mlm_probability 0.15 \
    --wandb_project "rrweb-bert" \
    --seed 42

echo "Training complete!"