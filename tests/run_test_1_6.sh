#!/bin/bash
# Run Test 1.6: Model Architecture Testing with Multi-GPU Support

echo "=============================================="
echo "Starting Test 1.6: Model Architecture Testing"
echo "Testing RRWebBERT with multi-GPU support"
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

# Run the test with output to both console and log
python tests/test_1_6_model_architecture.py 2>&1 | tee test_1_6.log

echo ""
echo "Test 1.6 completed. Results saved to test_1_6.log"