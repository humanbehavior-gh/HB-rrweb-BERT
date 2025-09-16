#!/bin/bash

# Test 1.2: Tokenizer Performance at Scale
# Run with nohup for background execution with logging

echo "Starting Test 1.2: Tokenizer Performance at Scale"
echo "================================================"
echo "Start time: $(date)"
echo "This test will:"
echo "  1. Process 1000 files to test scale"
echo "  2. Test parallel processing with 4 workers"
echo "  3. Test memory stability over iterations"
echo "  4. Test error handling"
echo ""
echo "Expected runtime: ~20-30 minutes"
echo "Progress will be logged to wandb and test_1_2.log"
echo ""

# Set up environment
export PYTHONUNBUFFERED=1
cd /home/ubuntu/rrweb-bert

# Run the test
nohup python -u tests/test_tokenizer_performance_production.py > test_1_2.log 2>&1 &

# Get the PID
PID=$!
echo "Test started with PID: $PID"
echo "To monitor progress:"
echo "  tail -f test_1_2.log"
echo "  Or check wandb at: https://wandb.ai/rrweb-tokenizer-tests"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop the test:"
echo "  kill $PID"