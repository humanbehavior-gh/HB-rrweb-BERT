#!/bin/bash

# Script to download 200,000 RRWEB files with high concurrency
# Uses the S3 download script from HumanBehaviorWorker

SCRIPT_DIR="/home/ubuntu/HumanBehaviorWorker/EmbeddingBehavior"
OUTPUT_DIR="/home/ubuntu/rrweb-bert/rrweb_data"
LOG_FILE="/home/ubuntu/rrweb-bert/download_200k.log"
COUNT=200000
CONCURRENCY=100  # Reduced from 1000 to prevent memory issues

echo "========================================" | tee -a "$LOG_FILE"
echo "🚀 Starting download of $COUNT RRWEB files" | tee -a "$LOG_FILE"
echo "📁 Target directory: $OUTPUT_DIR" | tee -a "$LOG_FILE"
echo "⚡ Concurrency: $CONCURRENCY threads" | tee -a "$LOG_FILE"
echo "📝 Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "🕐 Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Change to the script directory
cd "$SCRIPT_DIR"

# Check for .env file
if [ ! -f "../.env" ]; then
    echo "❌ Error: .env file not found at $SCRIPT_DIR/../.env" | tee -a "$LOG_FILE"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Skip dependency check - already installed
echo "✓ Using existing node_modules" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "📥 Starting download with $CONCURRENCY concurrent connections..." | tee -a "$LOG_FILE"
echo "This may take several hours. Check the log file for progress." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Run the TypeScript file directly with tsx (no compilation needed)
# Use stdbuf to disable output buffering for real-time logging
# Increase Node.js heap size to 8GB to handle concurrent downloads
NODE_OPTIONS="--max-old-space-size=8192" stdbuf -oL -eL npx tsx download_rrweb_from_s3.ts \
    --count "$COUNT" \
    --concurrency "$CONCURRENCY" \
    --output "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Download script completed successfully" | tee -a "$LOG_FILE"
else
    echo "❌ Download script failed with exit code: $EXIT_CODE" | tee -a "$LOG_FILE"
fi
echo "🕐 End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"

# Count downloaded files
if [ -d "$OUTPUT_DIR" ]; then
    FILE_COUNT=$(ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | wc -l)
    echo "📊 Total files in $OUTPUT_DIR: $FILE_COUNT" | tee -a "$LOG_FILE"
fi

echo "📝 Full log saved to: $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE