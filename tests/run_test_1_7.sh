#!/bin/bash
# Run Test 1.7: End-to-End Training Pipeline Test

echo "=============================================="
echo "Starting Test 1.7: End-to-End Training Pipeline"
echo "Testing complete training workflow with single GPU (batch size 64)"
echo "=============================================="
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Set Python path
export PYTHONPATH=/home/ubuntu/rrweb-bert:$PYTHONPATH

# Set CUDA visible devices (use single GPU to avoid DataParallel hanging)
export CUDA_VISIBLE_DEVICES=0

# Enable better CUDA error messages
export CUDA_LAUNCH_BLOCKING=1

# Run the test with output to both console and log
python tests/test_1_7_training_pipeline.py 2>&1 | tee test_1_7.log

echo ""
echo "Test 1.7 completed. Results saved to test_1_7.log"