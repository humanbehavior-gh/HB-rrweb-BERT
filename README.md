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
    'batch_size': 32,  # Reduced from 64 to prevent OOM on 80GB GPU
    'max_length': 2048,
    'learning_rate': 5e-5,
    'warmup_steps': 10000,
    'max_steps': 1000000,
    'gradient_accumulation_steps': 4,
    'fp16': True,
    'gradient_checkpointing': True,
    'use_random_window': True,
    'samples_per_epoch': 90000,
    'num_workers': 4
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

### Pretraining with MLM

```bash
python train_rrweb_bert.py \
    --data_dir /home/ubuntu/embeddingV2/rrweb_data \
    --tokenizer_path /home/ubuntu/rrweb_tokenizer/tokenizer_model_latest \
    --output_dir ./rrweb-bert-base \
    --model_size base \
    --batch_size 8 \
    --num_epochs 3 \
    --max_length 2048
```

## Usage

```python
from configuration_rrweb import RRWebBERTConfig
from modeling_rrweb import RRWebBERTModel
import sys
sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

# Load tokenizer
tokenizer = RRWebTokenizer.load('/path/to/tokenizer_model')

# Load model
config = RRWebBERTConfig.from_pretrained('./rrweb-bert-base')
model = RRWebBERTModel.from_pretrained('./rrweb-bert-base')

# Tokenize RRWEB events
events = [...]  # Your RRWEB events
tokens = tokenizer.tokenize_session(events)

# Get embeddings
import torch
input_ids = torch.tensor([tokens])
outputs = model(input_ids)
embeddings = outputs.pooler_output  # [batch_size, hidden_size]
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
   - LRU cache with 1000 file limit
   - Lazy loading with on-demand decompression
   - Random window sampling to avoid loading full sessions

2. **Performance**:
   - DataLoader with 4 workers for parallel data loading
   - Pin memory for faster GPU transfer
   - Gradient accumulation for larger effective batch sizes

3. **Stability**:
   - Gradient clipping at 1.0
   - Warmup over 10,000 steps
   - AdamW optimizer with weight decay 0.01

### Known Limitations
- Maximum sequence length of 2048 tokens
- Requires preprocessing RRWEB events to extract text and structure
- Single GPU training (DataParallel not supported with small batch sizes)

## Citation

```bibtex
@software{rrweb_bert,
  title = {RRWebBERT: BERT for Web Session Recordings},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/your-username/rrweb-bert}
}
```