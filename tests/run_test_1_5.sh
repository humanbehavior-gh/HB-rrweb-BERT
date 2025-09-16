#!/bin/bash
# Run Test 1.5: Collator Production Testing

echo "=============================================="
echo "Starting Test 1.5: Collator Production Testing"
echo "Testing with full 31,951 file dataset"
echo "=============================================="
echo ""

# Set Python path
export PYTHONPATH=/home/ubuntu/rrweb-bert:$PYTHONPATH

# Run the test with output to both console and log
python tests/test_1_5_collator_production.py 2>&1 | tee test_1_5.log

echo ""
echo "Test 1.5 completed. Results saved to test_1_5.log"