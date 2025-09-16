#!/bin/bash
# Run Test 1.6: Model Architecture Testing with Multi-GPU Support (FIXED VERSION)

echo "=============================================="
echo "Starting Test 1.6: Model Architecture Testing (FIXED)"
echo "Production-ready version with proper multi-GPU handling"
echo "=============================================="
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Set Python path
export PYTHONPATH=/home/ubuntu/rrweb-bert:$PYTHONPATH

# Set CUDA visible devices (use both GPUs if available)
export CUDA_VISIBLE_DEVICES=0,1

# Enable CUDA launch blocking for better error messages
export CUDA_LAUNCH_BLOCKING=1

# Run the fixed test with output to both console and log
python tests/test_1_6_model_architecture_fixed.py 2>&1 | tee test_1_6_fixed.log

echo ""
echo "Test 1.6 (fixed) completed. Results saved to test_1_6_fixed.log"