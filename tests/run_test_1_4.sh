#!/bin/bash

echo "Starting Test 1.4: Dataset Production Testing"
echo "============================================"
echo "Start time: $(date)"
echo "This test will:"
echo "  1. Test memory efficiency with lazy loading"
echo "  2. Test data integrity and consistency"
echo "  3. Test caching performance"
echo "  4. Test edge case handling"
echo "  5. Test production scale with DataLoader"
echo ""
echo "Expected runtime: ~10-15 minutes"
echo "Progress will be logged to wandb and test_1_4.log"
echo ""

# Run the test in background with nohup
nohup python /home/ubuntu/rrweb-bert/tests/test_1_4_dataset_production.py > test_1_4.log 2>&1 &

# Get the PID
PID=$!
echo "Test started with PID: $PID"
echo "To monitor progress:"
echo "  tail -f test_1_4.log"
echo "  Or check wandb at: https://wandb.ai/rrweb-tokenizer-tests"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop the test:"
echo "  kill $PID"