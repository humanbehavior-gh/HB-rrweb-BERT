# RRWebBERT Training Guide

## Production Training Configuration

The production training uses `production_training.py` optimized for single H100 GPU with 31,951 RRWEB files.

### Key Features

1. **Random Window Sampling**
   - Samples random 2048-token windows from long sessions
   - Prevents training bias (not just seeing session beginnings)
   - 90,000 samples per epoch for better coverage

2. **Batch Configuration**
   - Batch size: 32 (optimized for 80GB GPU memory)
   - No gradient accumulation needed
   - Single GPU training (DataParallel disabled)

3. **Training Schedule**
   - Learning rate: 5e-5 with AdamW optimizer
   - Weight decay: 0.01
   - Cosine annealing warm restarts (T_0=10, T_mult=2)
   - MLM probability: 15%

4. **Data Loading**
   - Lazy loading with LRU cache (2000 files)
   - 4 dataloader workers
   - Train: 28,750 files (90%)
   - Validation: 3,195 files (10%)

## Launch Commands

### Production Training (Single GPU)
```bash
python production_training.py
```

### Production Training (Background with nohup)
```bash
nohup python -u production_training.py > production_training.out 2>&1 &
```

### Testing Pipeline (Batch Size 8)
```bash
python tests/test_1_7_training_pipeline.py
```

### Monitor Training
```bash
# Check WandB logs
tail -f production_training.out

# Check GPU usage
nvidia-smi -l 1

# Check process
ps aux | grep production_training
```

## Model Architecture

### RRWebBERT (Production)
- **Parameters**: ~110M
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Intermediate Size**: 3072
- **Max Position Embeddings**: 2048
- **Vocab Size**: 12,520 (520 structural + 12,000 BPE text)
- **Dropout**: 0.1 (hidden and attention)

## Monitoring

### WandB Integration
- Project: `rrweb-bert-production`
- Tracks: loss, perplexity, learning rate, GPU memory
- Updates every 50 steps
- Dashboard: Check your WandB account

### Checkpointing
- Directory: `./checkpoints_production/`
- Saves best model based on validation loss
- Includes optimizer state for resuming
- Format: `checkpoint_epoch{N}_step{M}.pt`

## Resource Requirements

### Production Training
- **GPU**: Single H100 80GB (or A100 80GB)
- **GPU Memory Usage**: ~69GB with batch size 32
- **System RAM**: 32GB+ recommended
- **Storage**: 100GB+ for dataset and checkpoints
- **Training Time**: ~2-3 days for 10 epochs

### Memory Optimizations
- Mixed precision training (FP16) with GradScaler
- LRU cache limiting to 2000 files in memory
- Random window sampling (avoids loading full sessions)
- Efficient tokenizer with pre-computed vocabularies

## Tokenizer Integration

### TokenizerWrapper
Production uses `TokenizerWrapper` class that:
1. Loads structural vocab (520 tokens)
2. Loads BPE text tokenizer (12,000 tokens)
3. Combines into unified vocabulary (12,520 total)
4. Handles special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4

### Tokenizer Path
```python
tokenizer = TokenizerWrapper(
    structural_vocab_path='/home/ubuntu/rrweb_tokenizer/structural_vocab.json',
    bpe_model_path='/home/ubuntu/rrweb_tokenizer/text_bpe.model'
)
```

## Training Output

### Checkpoint Structure
```
checkpoints_production/
├── best_model.pt           # Best validation loss
├── checkpoint_epoch1_step1000.pt
├── checkpoint_epoch2_step2000.pt
└── training_history.json   # Metrics history
```

### Model Components
- **Model weights**: BERT parameters
- **Optimizer state**: AdamW state for resuming
- **Scheduler state**: Cosine annealing state
- **Training metrics**: Loss, perplexity, LR history

## Dataset Details

### Random Window Sampling
```python
# Prevents bias of only seeing session starts
if len(tokens) > max_length:
    start_idx = random.randint(0, len(tokens) - max_length)
    tokens = tokens[start_idx:start_idx + max_length]
```

### Data Split
- **Training**: 28,750 files (90%)
- **Validation**: 3,195 files (10%)
- **Test**: Reserve separately

## Production Training Status

The current production training:
- Uses random window sampling
- Processes 90,000 samples per epoch
- Runs on single H100 GPU
- Batch size 32 (reduced from 64 to prevent OOM)
- Logs to WandB for monitoring