#!/bin/bash

# Test 1.3: Event Type Extraction
# Run with nohup for background execution with logging

echo "Starting Test 1.3: Event Type Extraction"
echo "========================================"
echo "Start time: $(date)"
echo "This test will:"
echo "  1. Test event type coverage (all 7 types)"
echo "  2. Test token-to-event mapping"
echo "  3. Test event type embeddings"
echo "  4. Test at production scale (5000 files)"
echo ""
echo "Expected runtime: ~15-20 minutes"
echo "Progress will be logged to wandb and test_1_3.log"
echo ""

# Set up environment
export PYTHONUNBUFFERED=1
cd /home/ubuntu/rrweb-bert

# Run the test
nohup python -u tests/test_event_type_extraction.py > test_1_3.log 2>&1 &

# Get the PID
PID=$!
echo "Test started with PID: $PID"
echo "To monitor progress:"
echo "  tail -f test_1_3.log"
echo "  Or check wandb at: https://wandb.ai/rrweb-tokenizer-tests"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop the test:"
echo "  kill $PID"