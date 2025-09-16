#!/usr/bin/env python3
"""
Test script to verify RRWebBERT model can be loaded and used for inference.
"""

import sys
import json
import torch

# Add parent directory to path for tokenizer
sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

from configuration_rrweb import RRWebBERTConfig
from modeling_rrweb import RRWebBERTModel

def test_inference():
    # Load the trained model
    print("Loading RRWebBERT model...")
    config = RRWebBERTConfig.from_pretrained('./rrweb-bert-test')
    model = RRWebBERTModel.from_pretrained('./rrweb-bert-test', config=config)
    model.eval()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_test')
    
    # Load a sample RRWEB file
    print("Loading sample RRWEB session...")
    with open('/home/ubuntu/embeddingV2/rrweb_data/01993243-0eb4-7c37-9514-67433ebc2082.json', 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    events = []
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and 'events' in data:
        events = data['events']
    
    # Tokenize
    print("Tokenizing session...")
    tokens = tokenizer.tokenize_session(events[:10])  # Use first 10 events
    print(f"Tokenized to {len(tokens)} tokens")
    
    # Prepare input
    input_ids = torch.tensor([tokens])
    attention_mask = torch.ones_like(input_ids)
    
    # Get embeddings
    print("Getting embeddings...")
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    # Extract embeddings
    embeddings = outputs.pooler_output  # [batch_size, hidden_size]
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding vector (first 10 dims): {embeddings[0][:10].tolist()}")
    
    print("\nSuccess! RRWebBERT model is working correctly.")
    print(f"Model produces {embeddings.shape[1]}-dimensional embeddings for RRWEB sessions.")

if __name__ == "__main__":
    test_inference()