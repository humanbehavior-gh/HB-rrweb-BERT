#!/bin/bash

# Launch RRWebBERT training in background with nohup
# This allows training to continue even if SSH session disconnects

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/rrweb_bert_training_$TIMESTAMP.log"

echo "Launching RRWebBERT training in background..."
echo "Log file: $LOG_FILE"
echo "PID file: $LOG_DIR/rrweb_bert.pid"

# Check if tokenizer exists
TOKENIZER_PATH="/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest"
if [ ! -d "$TOKENIZER_PATH" ]; then
    echo "Warning: tokenizer_model_latest not found, using test tokenizer"
    TOKENIZER_PATH="/home/ubuntu/rrweb_tokenizer/tokenizer_model_test"
fi

# Set wandb environment
export WANDB_PROJECT="rrweb-bert"
export WANDB_MODE="online"

# Launch with nohup
nohup python train_rrweb_bert.py \
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
    --seed 42 \
    > "$LOG_FILE" 2>&1 &

# Save PID
echo $! > "$LOG_DIR/rrweb_bert.pid"

echo "Training started with PID: $(cat $LOG_DIR/rrweb_bert.pid)"
echo "Monitor progress with: tail -f $LOG_FILE"
echo "Check wandb at: https://wandb.ai/$USER/rrweb-bert"
echo ""
echo "To stop training: kill \$(cat $LOG_DIR/rrweb_bert.pid)"