# RRWebBERT Training Guide

## Scaled Training Parameters

The `train_rrweb_bert.py` script has been optimized for full-scale training on 30,000+ RRWEB files.

### Key Improvements

1. **Batch Size & Gradient Accumulation**
   - Base batch size: 16
   - Gradient accumulation: 4 steps
   - Effective batch size: 64

2. **Training Schedule**
   - Learning rate: 2e-5 (optimized for stability)
   - Warmup steps: 1000
   - Epochs: 5 (increased from 3)

3. **Logging & Checkpointing**
   - WandB integration for real-time monitoring
   - Save checkpoints every 5000 steps
   - Log metrics every 500 steps

4. **Data Loading**
   - 8 dataloader workers for faster preprocessing
   - Pin memory for GPU transfers
   - Optional max_files parameter for testing

## Launch Commands

### Quick Test (50 files)
```bash
python train_rrweb_bert.py \
    --data_dir /home/ubuntu/embeddingV2/rrweb_data \
    --tokenizer_path /home/ubuntu/rrweb_tokenizer/tokenizer_model_test \
    --output_dir ./rrweb-bert-test \
    --model_size small \
    --max_files 50 \
    --num_epochs 1
```

### Full Training (Interactive)
```bash
./launch_full_training.sh
```

### Full Training (Background)
```bash
./launch_training_nohup.sh
```

## Model Sizes

- **Small**: 67M parameters (512 hidden, 6 layers)
- **Base**: 110M parameters (768 hidden, 12 layers)  [Recommended]
- **Large**: 340M parameters (1024 hidden, 24 layers)

## Monitoring

### WandB Dashboard
Track training metrics at: https://wandb.ai/[your-username]/rrweb-bert

### Local Logs
```bash
# View training logs
tail -f logs/rrweb_bert_training_*.log

# Check training status
ps aux | grep train_rrweb_bert
```

## Resource Requirements

### Base Model Training
- GPU Memory: ~8-12GB
- RAM: ~32GB
- Disk: ~10GB for checkpoints
- Time: ~24-48 hours on single GPU

### Optimizations Applied
- Mixed precision training (fp16)
- Gradient checkpointing for large model
- Efficient data loading with workers
- Smart tokenization caching

## Integration with Tokenizer

The training script automatically uses the latest tokenizer:
1. Checks for `tokenizer_model_latest` (full tokenizer)
2. Falls back to `tokenizer_model_test` if needed
3. Handles both structural and BPE text tokens

## Output

Trained model will be saved in `./rrweb-bert-full/` with:
- `pytorch_model.bin`: Model weights
- `config.json`: Model configuration
- `training_args.bin`: Training configuration
- Checkpoints in subdirectories

## Next Steps

After training completes:
1. Test inference with `test_inference.py`
2. Push to Hugging Face Hub
3. Integrate with trimodal embedding system