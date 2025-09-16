#!/bin/bash
# Run Test 1.8: Minimal Forward Pass Test

echo "=============================================="
echo "Starting Test 1.8: Minimal Forward Pass Test"
echo "Testing basic forward pass performance"
echo "=============================================="
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
echo ""

# Set Python path
export PYTHONPATH=/home/ubuntu/rrweb-bert:$PYTHONPATH

# Set CUDA visible devices (use single GPU for testing)
export CUDA_VISIBLE_DEVICES=0

# Enable better CUDA error messages
export CUDA_LAUNCH_BLOCKING=1

# Run the test with unbuffered output to both console and log (10 minute timeout)
timeout 600 python -u tests/test_1_8_minimal_forward.py 2>&1 | tee test_1_8.log

echo ""
echo "Test 1.8 completed. Results saved to test_1_8.log"