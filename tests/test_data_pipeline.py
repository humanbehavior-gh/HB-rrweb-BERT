#!/usr/bin/env python3
"""
Test script to verify the data processing pipeline is working correctly.
"""

import sys
import json
import torch
import numpy as np
from collections import Counter

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

# Import training components
from train_rrweb_bert import RRWebDataset, DataCollatorForRRWebMLM, TokenizerWrapper

def test_data_pipeline():
    print("=" * 60)
    print("Testing RRWebBERT Data Processing Pipeline")
    print("=" * 60)
    
    # 1. Test Dataset Loading
    print("\n1. Testing Dataset Loading...")
    dataset = RRWebDataset(
        data_dir="/home/ubuntu/embeddingV2/rrweb_data",
        tokenizer_path="/home/ubuntu/rrweb_tokenizer/tokenizer_model_test",
        max_length=512,
        max_files=10
    )
    
    print(f"   ✓ Loaded {len(dataset)} sessions")
    print(f"   ✓ Tokenizer vocab size: {dataset.tokenizer.vocab.structural_vocab_size}")
    
    # 2. Test individual samples
    print("\n2. Testing Individual Samples...")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"   Sample {i}:")
        print(f"     - input_ids shape: {sample['input_ids'].shape}")
        print(f"     - attention_mask shape: {sample['attention_mask'].shape}")
        print(f"     - token_type_ids shape: {sample['token_type_ids'].shape}")
        print(f"     - Unique tokens: {len(torch.unique(sample['input_ids']))}")
        print(f"     - Non-zero tokens: {(sample['input_ids'] > 0).sum().item()}")
    
    # 3. Test Data Collator
    print("\n3. Testing Data Collator...")
    tokenizer_wrapper = TokenizerWrapper(dataset.tokenizer)
    data_collator = DataCollatorForRRWebMLM(
        tokenizer=tokenizer_wrapper,
        mlm_probability=0.15
    )
    
    # Create a batch
    batch_samples = [dataset[i] for i in range(min(4, len(dataset)))]
    batch = data_collator(batch_samples)
    
    print(f"   Batch shapes:")
    print(f"     - input_ids: {batch['input_ids'].shape}")
    print(f"     - attention_mask: {batch['attention_mask'].shape}")
    print(f"     - token_type_ids: {batch['token_type_ids'].shape}")
    print(f"     - labels: {batch['labels'].shape}")
    
    # 4. Verify Masking is Working
    print("\n4. Verifying MLM Masking...")
    
    # Compare original vs masked
    original_batch = [dataset[i] for i in range(min(4, len(dataset)))]
    masked_batch = data_collator(original_batch)
    
    total_tokens = 0
    masked_tokens = 0
    mask_token_count = 0
    
    for i in range(len(original_batch)):
        original_ids = original_batch[i]['input_ids']
        masked_ids = masked_batch['input_ids'][i]
        labels = masked_batch['labels'][i]
        attention = masked_batch['attention_mask'][i]
        
        # Count actual content tokens (not padding)
        content_mask = attention == 1
        content_tokens = content_mask.sum().item()
        total_tokens += content_tokens
        
        # Count masked positions (where labels != -100)
        masked_positions = labels != -100
        masked_count = masked_positions.sum().item()
        masked_tokens += masked_count
        
        # Count [MASK] tokens
        mask_token_positions = masked_ids == 4  # [MASK] token id
        mask_token_count += mask_token_positions.sum().item()
        
        print(f"   Sample {i}:")
        print(f"     - Content tokens: {content_tokens}")
        print(f"     - Masked tokens: {masked_count}")
        print(f"     - [MASK] tokens: {mask_token_positions.sum().item()}")
        print(f"     - Changed tokens: {(original_ids != masked_ids).sum().item()}")
    
    mask_percentage = (masked_tokens / total_tokens * 100) if total_tokens > 0 else 0
    print(f"\n   Overall Statistics:")
    print(f"     - Total content tokens: {total_tokens}")
    print(f"     - Total masked tokens: {masked_tokens}")
    print(f"     - Actual mask percentage: {mask_percentage:.1f}%")
    print(f"     - [MASK] tokens used: {mask_token_count}")
    
    # 5. Check Loss Computation
    print("\n5. Testing Loss Computation...")
    
    # Create sample predictions
    vocab_size = tokenizer_wrapper.vocab_size
    batch_size = batch['input_ids'].shape[0]
    seq_length = batch['input_ids'].shape[1]
    
    # Simulate model output
    logits = torch.randn(batch_size, seq_length, vocab_size)
    labels = batch['labels']
    
    # Compute loss manually
    loss_fct = torch.nn.CrossEntropyLoss()
    active_loss = labels.view(-1) != -100
    active_logits = logits.view(-1, vocab_size)[active_loss]
    active_labels = labels.view(-1)[active_loss]
    
    if len(active_labels) > 0:
        loss = loss_fct(active_logits, active_labels)
        print(f"   ✓ Loss computed: {loss.item():.4f}")
        print(f"   ✓ Active positions: {active_loss.sum().item()}")
    else:
        print(f"   ✗ WARNING: No active labels for loss computation!")
    
    # 6. Data Distribution Analysis
    print("\n6. Analyzing Token Distribution...")
    all_tokens = []
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        all_tokens.extend(sample['input_ids'].tolist())
    
    token_counts = Counter(all_tokens)
    print(f"   - Unique tokens used: {len(token_counts)}")
    print(f"   - Most common tokens: {token_counts.most_common(5)}")
    print(f"   - Token range: [{min(all_tokens)}, {max(all_tokens)}]")
    
    print("\n" + "=" * 60)
    print("Pipeline Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_data_pipeline()