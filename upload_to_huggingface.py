#!/usr/bin/env python3
"""
Upload RRWebBERT model checkpoint to Hugging Face Hub
"""

import torch
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
import sys
sys.path.append('/home/ubuntu/rrweb-bert')
from configuration_rrweb import RRWebBERTConfig
from modeling_rrweb import RRWebBERTModel

def prepare_model_for_hub(checkpoint_path, output_dir):
    """Prepare model files for Hugging Face upload"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Create config
    config = RRWebBERTConfig(
        vocab_size=12520,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=2048,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        structural_vocab_size=520,
        text_vocab_size=12000
    )
    
    # Initialize model and load state dict
    model = RRWebBERTModel(config)
    model.load_state_dict(state_dict, strict=False)
    
    # Save in HuggingFace format
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    # Add model card
    model_card = """---
language: en
tags:
- rrweb
- bert
- web-sessions
- behavior-modeling
license: mit
datasets:
- custom-rrweb-sessions
---

# HB-rrweb-BERT

BERT model trained on RRWEB session recordings for behavioral understanding.

## Model Details

- **Architecture**: BERT-based encoder
- **Parameters**: 110M
- **Vocabulary**: 12,520 tokens (520 structural + 12,000 BPE text)
- **Max sequence length**: 2,048 tokens
- **Training data**: 31,000 RRWEB session files

## Usage

```python
from transformers import AutoModel, AutoConfig
import torch

# Load model
config = AutoConfig.from_pretrained("HumanBehavior/HB-rrweb-BERT")
model = AutoModel.from_pretrained("HumanBehavior/HB-rrweb-BERT")

# Example usage (requires custom tokenizer)
input_ids = torch.tensor([[2, 100, 200, 300, 3]])  # [CLS] tokens [SEP]
outputs = model(input_ids)
embeddings = outputs.last_hidden_state
```

## Training Details

- Mixed precision training (FP16)
- Batch size: 32 with gradient accumulation (effective 128)
- Learning rate: 5e-5 with linear warmup
- MLM objective with 15% masking
- Random window sampling from long sequences

## Limitations

- Requires custom RRWeb tokenizer (not included)
- Maximum sequence length of 2,048 tokens
- Single GPU training optimized

## Citation

```bibtex
@software{hb_rrweb_bert,
  title = {HB-rrweb-BERT: BERT for Web Session Recordings},
  author = {Human Behavior Labs},
  year = {2024},
  url = {https://github.com/Aneeshers/HB-rrweb-BERT}
}
```
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(model_card)
    
    print(f"Model prepared in {output_dir}")
    return output_dir

def upload_to_hub(model_dir, repo_id="HumanBehavior/HB-rrweb-BERT", private=False):
    """Upload model to Hugging Face Hub"""
    
    api = HfApi()
    
    # Create repository if it doesn't exist
    try:
        create_repo(repo_id, private=private, repo_type="model")
        print(f"Created repository: {repo_id}")
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload model files
    print(f"Uploading model to {repo_id}...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload HB-rrweb-BERT v0.6 checkpoint"
    )
    
    print(f"Model uploaded successfully to https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # Use best checkpoint
    checkpoint_path = "/home/ubuntu/rrweb-bert/checkpoints_production/best_model.pt"
    output_dir = "/tmp/hb-rrweb-bert-upload"
    
    # Prepare model
    model_dir = prepare_model_for_hub(checkpoint_path, output_dir)
    
    # Upload to Hugging Face
    upload_to_hub(model_dir, repo_id="HumanBehavior/HB-rrweb-BERT", private=False)