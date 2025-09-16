#!/bin/bash
# Run Production Training for RRWEB BERT Model

echo "=============================================="
echo "Starting RRWEB BERT Production Training"
echo "Single GPU configuration with full dataset"
echo "=============================================="
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Set Python path
export PYTHONPATH=/home/ubuntu/rrweb-bert:$PYTHONPATH

# Set CUDA visible devices (single GPU to avoid DataParallel issues)
export CUDA_VISIBLE_DEVICES=0

# Enable better CUDA error messages
export CUDA_LAUNCH_BLOCKING=1

# Create log directory
mkdir -p logs

# Run the production training with output to both console and log
echo "Starting training at $(date)"
echo "Configuration:"
echo "  - Dataset: 28,750 training files, 3,195 validation files"
echo "  - Batch size: 64"
echo "  - Max sequence length: 2048"
echo "  - Epochs: 10"
echo "  - Single GPU (H100 80GB)"
echo ""

python -u production_training.py 2>&1 | tee logs/production_training_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Production training completed at $(date)"
echo "Check logs directory for detailed output"