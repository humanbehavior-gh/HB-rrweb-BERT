#!/usr/bin/env python3
"""
Test the lazy dataset implementation to ensure it doesn't load everything into memory.
"""

import sys
import os
import gc
import psutil
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/ubuntu/rrweb_tokenizer')
sys.path.append('/home/ubuntu/rrweb-bert')
sys.path.append('/home/ubuntu/rrweb-bert/src')

from rrweb_tokenizer import RRWebTokenizer
from dataset import RRWebLazyDataset, RRWebDatasetSplitter
from collator import ImprovedDataCollatorForMLM


def test_memory_usage():
    """Test that memory usage stays constant with lazy loading."""
    print("=" * 80)
    print("TESTING LAZY DATASET MEMORY USAGE")
    print("=" * 80)
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"\nInitial memory: {initial_memory:.2f} MB")
    
    # Load tokenizer
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest')
    print("Tokenizer loaded")
    
    # Create dataset with many files
    dataset = RRWebLazyDataset(
        data_dir='/home/ubuntu/embeddingV2/rrweb_data',
        tokenizer=tokenizer,
        max_length=512,
        max_files=1000,  # Load paths for 1000 files
        cache_size=10     # But only cache 10
    )
    
    after_init_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = after_init_memory - initial_memory
    print(f"\nAfter dataset init (1000 files):")
    print(f"  Memory: {after_init_memory:.2f} MB")
    print(f"  Increase: {memory_increase:.2f} MB")
    
    # Access some samples
    print("\nAccessing samples...")
    for i in range(20):
        sample = dataset[i]
        if i % 5 == 0:
            current_memory = process.memory_info().rss / 1024 / 1024
            print(f"  After sample {i}: {current_memory:.2f} MB (+{current_memory - after_init_memory:.2f} MB)")
    
    final_memory = process.memory_info().rss / 1024 / 1024
    total_increase = final_memory - initial_memory
    
    print(f"\nFinal memory: {final_memory:.2f} MB")
    print(f"Total increase: {total_increase:.2f} MB")
    
    # Get detailed stats
    stats = dataset.get_memory_usage()
    print(f"\nDataset statistics:")
    print(f"  File paths count: {stats['file_paths_count']}")
    print(f"  File paths size: {stats['file_paths_size_mb']:.3f} MB")
    print(f"  Cache hits: {stats['cache_info']['hits']}")
    print(f"  Cache misses: {stats['cache_info']['misses']}")
    print(f"  Cache size: {stats['cache_info']['currsize']}/{stats['cache_info']['maxsize']}")
    
    # Check if memory increase is reasonable
    if total_increase < 500:  # Less than 500MB increase
        print("\n✅ Memory usage is reasonable for lazy loading")
    else:
        print(f"\n⚠️  Memory usage higher than expected: {total_increase:.2f} MB")
    
    return dataset


def test_data_loading():
    """Test that data loads correctly."""
    print("\n" + "=" * 80)
    print("TESTING DATA LOADING")
    print("=" * 80)
    
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest')
    
    dataset = RRWebLazyDataset(
        data_dir='/home/ubuntu/embeddingV2/rrweb_data',
        tokenizer=tokenizer,
        max_length=512,
        max_files=10,
        cache_size=5
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test single sample
    sample = dataset[0]
    print(f"\nFirst sample keys: {sample.keys()}")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  attention_mask shape: {sample['attention_mask'].shape}")
    print(f"  token_type_ids shape: {sample['token_type_ids'].shape}")
    print(f"  event_type_ids shape: {sample['event_type_ids'].shape}")
    
    # Check token type assignment
    structural_tokens = (sample['token_type_ids'] == 0).sum()
    text_tokens = (sample['token_type_ids'] == 1).sum()
    print(f"\nToken types:")
    print(f"  Structural (type=0): {structural_tokens}")
    print(f"  Text (type=1): {text_tokens}")
    
    # Verify boundary
    input_ids = sample['input_ids']
    token_types = sample['token_type_ids']
    
    errors = 0
    for i in range(len(input_ids)):
        token_id = input_ids[i].item()
        token_type = token_types[i].item()
        
        if token_id < 520 and token_type == 1:
            errors += 1
        elif token_id >= 520 and token_type == 0:
            errors += 1
    
    if errors == 0:
        print("\n✅ Token types correctly assigned")
    else:
        print(f"\n⚠️  Found {errors} token type errors")
    
    return dataset


def test_dataloader():
    """Test with PyTorch DataLoader."""
    print("\n" + "=" * 80)
    print("TESTING DATALOADER INTEGRATION")
    print("=" * 80)
    
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest')
    
    dataset = RRWebLazyDataset(
        data_dir='/home/ubuntu/embeddingV2/rrweb_data',
        tokenizer=tokenizer,
        max_length=512,
        max_files=20
    )
    
    # Create collator
    from train_rrweb_bert import TokenizerWrapper
    tokenizer_wrapper = TokenizerWrapper(tokenizer)
    collator = ImprovedDataCollatorForMLM(
        tokenizer=tokenizer_wrapper,
        mlm_probability=0.15
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collator,
        num_workers=0  # Single process for testing
    )
    
    print(f"\nDataLoader created with batch_size=4")
    
    # Test batching
    for i, batch in enumerate(dataloader):
        if i >= 3:
            break
        
        print(f"\nBatch {i}:")
        print(f"  input_ids: {batch['input_ids'].shape}")
        print(f"  attention_mask: {batch['attention_mask'].shape}")
        print(f"  token_type_ids: {batch['token_type_ids'].shape}")
        print(f"  event_type_ids: {batch['event_type_ids'].shape}")
        print(f"  labels: {batch['labels'].shape}")
        
        # Check masking
        masked_positions = (batch['labels'] != -100).sum()
        total_positions = batch['attention_mask'].sum()
        mask_ratio = masked_positions.float() / total_positions
        
        print(f"  Masked: {masked_positions}/{total_positions} ({mask_ratio:.2%})")
    
    # Test collator statistics
    samples = [dataset[i] for i in range(10)]
    stats = collator.get_masking_stats(samples)
    
    print(f"\nMasking statistics over 10 samples:")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Mask ratio: {stats['mask_ratio']:.2%}")
    print(f"  [MASK] token: {stats['mask_token_ratio']:.2%} of masked")
    print(f"  Random token: {stats['random_ratio']:.2%} of masked")
    print(f"  Kept original: {stats['kept_ratio']:.2%} of masked")
    
    print("\n✅ DataLoader integration successful")


def test_splits():
    """Test train/val/test splitting."""
    print("\n" + "=" * 80)
    print("TESTING DATASET SPLITS")
    print("=" * 80)
    
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest')
    
    dataset = RRWebLazyDataset(
        data_dir='/home/ubuntu/embeddingV2/rrweb_data',
        tokenizer=tokenizer,
        max_length=512,
        max_files=100
    )
    
    # Create splits
    train_idx, val_idx, test_idx = RRWebDatasetSplitter.create_splits(
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_idx)}")
    print(f"  Val: {len(val_idx)}")
    print(f"  Test: {len(test_idx)}")
    
    # Check no overlap
    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)
    
    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set
    
    print(f"\nOverlap check:")
    print(f"  Train-Val: {len(overlap_train_val)}")
    print(f"  Train-Test: {len(overlap_train_test)}")
    print(f"  Val-Test: {len(overlap_val_test)}")
    
    if len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0:
        print("\n✅ No overlap between splits")
    else:
        print("\n⚠️  Found overlap between splits!")


def main():
    print("LAZY DATASET COMPREHENSIVE TEST")
    print("=" * 80)
    
    # Run all tests
    dataset = test_memory_usage()
    test_data_loading()
    test_dataloader()
    test_splits()
    
    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()