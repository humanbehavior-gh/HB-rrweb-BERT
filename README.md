# RRWebBERT

BERT-based encoder for RRWEB session recordings. Uses custom RRWeb tokenizer with structural and BPE text vocabularies trained on 31,000 RRWEB session files.

## Architecture

### Model Configuration
- **Vocabulary Size**: 12,520 tokens (520 structural + 12,000 BPE text tokens)
- **Max Position Embeddings**: 2,048
- **Hidden Size**: 768
- **Number of Layers**: 12
- **Number of Attention Heads**: 12
- **Intermediate Size**: 3,072
- **Hidden Activation**: GELU
- **Model Parameters**: ~110M

### Key Features
- Custom embedding layer with separate structural and text token embeddings
- Token type embeddings to distinguish structural (0) vs text (1) tokens
- Optimized for RRWEB event sequences with random window sampling
- Mixed precision training (FP16) support
- Gradient checkpointing for memory efficiency

## Training Infrastructure

### Dataset
- **RRWebLazyDataset**: Memory-efficient lazy loading with LRU caching
- **Random Window Sampling**: Prevents training bias by sampling random 2048-token windows from long sessions
- **Samples per Epoch**: 90,000 (configurable)
- **Data Augmentation**: 15% MLM masking probability with 80/10/10 mask/random/keep strategy

### Production Training Configuration
```python
{
    'batch_size': 32,  # Optimized for H100 80GB
    'max_length': 2048,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'mlm_probability': 0.15,
    'num_workers': 4,
    'cache_size': 2000,  # LRU cache size
    'train_files': 28750,  # 90% of 31,951 files
    'val_files': 3195,     # 10% of 31,951 files
    'samples_per_epoch': 90000,  # Random windows per epoch
    'use_random_window': True,
    'T_0': 10,  # Cosine annealing restart period
    'T_mult': 2,  # Period multiplier after restart
}
```

### Hardware Requirements
- Single GPU with 80GB VRAM (A100 or H100)
- 32GB+ system RAM recommended
- Fast NVMe storage for dataset caching

## Training

### Prerequisites

1. Train RRWeb tokenizer first:
```bash
cd /home/ubuntu/rrweb_tokenizer
./build_vocab_pipeline.sh /path/to/rrweb/data 12000 1.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Production Training

```bash
# Run production training with random window sampling
python production_training.py

# Or run in background with nohup
nohup python -u production_training.py > production_training.out 2>&1 &

# Monitor training
tail -f production_training.out
```

The production script automatically:
- Uses random window sampling to prevent bias
- Loads data from `/home/ubuntu/embeddingV2/rrweb_data`
- Uses pre-trained tokenizer from `/home/ubuntu/rrweb_tokenizer`
- Saves checkpoints to `./checkpoints_production/`
- Logs to WandB project `rrweb-bert-production`

## Usage

```python
import torch
import sys
sys.path.append('/home/ubuntu/rrweb-bert')
sys.path.append('/home/ubuntu/rrweb_tokenizer')

from src.tokenizer_wrapper import TokenizerWrapper
from production_training import RRWebBERT

# Load tokenizer
tokenizer = TokenizerWrapper(
    structural_vocab_path='/home/ubuntu/rrweb_tokenizer/structural_vocab.json',
    bpe_model_path='/home/ubuntu/rrweb_tokenizer/text_bpe.model'
)

# Load trained model
checkpoint = torch.load('./checkpoints_production/best_model.pt')
model = RRWebBERT(vocab_size=12520)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Tokenize RRWEB session
with open('session.json', 'r') as f:
    events = json.load(f)
tokens = tokenizer.tokenize_session(events)

# Get embeddings
input_ids = torch.tensor([tokens[:2048]])  # Truncate to max length
with torch.no_grad():
    outputs = model(input_ids)
    embeddings = outputs.loss  # Note: Returns MLM loss in training mode
```

## Integration with Trimodal Model

Use RRWebBERT as the RRWEB encoder in your trimodal embedding:

```python
class TrimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Text encoder (pretrained)
        self.text_encoder = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        # Image encoder (pretrained)  
        self.image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
        
        # RRWEB encoder (your trained model)
        self.rrweb_encoder = RRWebBERTModel.from_pretrained('./rrweb-bert-base')
        
        # Projection heads to common dimension
        self.text_proj = nn.Linear(384, 512)
        self.image_proj = nn.Linear(768, 512)
        self.rrweb_proj = nn.Linear(768, 512)
```

## Model Checkpoints

### Available Checkpoints
- `checkpoints_production/`: Active training checkpoints with optimizer states
- Model weights saved every 5000 steps during training
- Best checkpoint selection based on validation loss

### Hugging Face Hub

Model available at: `HumanBehaviorLabs/HB-rrweb-BERT`

To load from Hugging Face:
```python
from transformers import AutoModel, AutoConfig

config = AutoConfig.from_pretrained("HumanBehaviorLabs/HB-rrweb-BERT")
model = AutoModel.from_pretrained("HumanBehaviorLabs/HB-rrweb-BERT")
```

## Technical Implementation Details

### Tokenizer Architecture
- **Structural Tokens (0-519)**: Direct mapping of RRWEB event types and DOM elements
- **Text Tokens (520-12519)**: BPE-encoded text content from DOM nodes
- **Special Tokens**: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4

### Training Optimizations
1. **Memory Management**:
   - LRU cache with 2000 file limit
   - Lazy loading with on-demand decompression
   - Random window sampling prevents loading full sessions
   - Mixed precision (FP16) training with GradScaler

2. **Performance**:
   - DataLoader with 4 workers for parallel data loading
   - Pin memory for faster GPU transfer
   - Single GPU optimization (avoids DataParallel overhead)
   - 90,000 samples per epoch for better coverage

3. **Stability**:
   - Gradient clipping at 1.0
   - Cosine annealing with warm restarts
   - AdamW optimizer with weight decay 0.01
   - Validation every epoch for early stopping

### Known Limitations
- Maximum sequence length of 2048 tokens
- Requires preprocessing RRWEB events to extract text and structure
- Single GPU training (DataParallel hangs with batch_size ≤ num_gpus)
- Random window sampling may split semantic boundaries

### Production Training Details
- **Dataset**: 31,951 RRWEB session files from real user interactions
- **Training Strategy**: Random window sampling to see diverse parts of sessions
- **Validation**: 10% holdout for model selection
- **Hardware**: Single NVIDIA H100 80GB GPU
- **Training Time**: ~2-3 days for 10 epochs

## Citation

```bibtex
@software{hb-rrweb_bert,
  title = {HB-RRWebBERT: BERT for Web Session Recordings},
  author = {Aneesh Muppidi},
  year = {2025},
  url = {https://github.com/humanbehavior-gh/HB-rrweb-BERT}
}
```