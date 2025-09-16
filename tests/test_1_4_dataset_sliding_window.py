#!/usr/bin/env python3
"""
Test 1.4: Dataset Production Testing with Sliding Window - PRODUCTION TEST
Verify the lazy-loading dataset with sliding window works at production scale.
"""

import sys
import os
import json
import time
import gc
import random
import psutil
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append('/home/ubuntu/rrweb_tokenizer')
sys.path.append('/home/ubuntu/rrweb-bert/src')

from rrweb_tokenizer import RRWebTokenizer
from dataset_sliding_window import RRWebLazyDataset


class DatasetProductionTester:
    """Test dataset with sliding window at production scale."""
    
    def __init__(self):
        self.tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
        self.data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
        
        # Initialize wandb
        wandb.init(
            project="rrweb-tokenizer-tests",
            name="test_1_4_dataset_sliding_window",
            config={
                "test_type": "dataset_sliding_window_production",
                "tokenizer_path": self.tokenizer_path,
                "data_dir": self.data_dir
            }
        )
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = RRWebTokenizer.load(self.tokenizer_path)
        print(f"Tokenizer loaded: structural_vocab_size={self.tokenizer.vocab.structural_vocab_size}")
    
    def test_sliding_window_coverage(self):
        """Test that sliding window creates proper coverage."""
        print("\n" + "="*80)
        print("TEST: Sliding Window Coverage")
        print("="*80)
        
        # Create dataset with sliding window
        print("\nCreating dataset with sliding window (stride=1024)...")
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=100,  # Start with 100 files for testing
            cache_size=50,
            use_sliding_window=True,
            window_stride=1024,  # 50% overlap
            shuffle_windows=False  # Don't shuffle for testing
        )
        
        print(f"\nDataset stats:")
        print(f"  Files: {len(dataset.file_paths)}")
        print(f"  Windows: {len(dataset.windows)}")
        print(f"  Windows per file: {len(dataset.windows) / len(dataset.file_paths):.2f}")
        
        # Test a few windows
        print("\nTesting first 5 windows...")
        for i in range(min(5, len(dataset))):
            sample = dataset[i]
            print(f"  Window {i}: shape={sample['input_ids'].shape}, "
                  f"non-zero tokens={(sample['input_ids'] != 0).sum().item()}")
        
        wandb.log({
            "sliding_window/num_files": len(dataset.file_paths),
            "sliding_window/num_windows": len(dataset.windows),
            "sliding_window/windows_per_file": len(dataset.windows) / len(dataset.file_paths)
        })
        
        return dataset
    
    def test_production_scale(self):
        """Test with full 31k files."""
        print("\n" + "="*80)
        print("TEST: Production Scale with 31k Files")
        print("="*80)
        
        # Create dataset with ALL files
        print("\nCreating dataset with ALL files and sliding window...")
        start_time = time.time()
        
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=None,  # Use ALL files
            cache_size=2000,  # Large cache for production
            use_sliding_window=True,
            window_stride=1024,
            shuffle_windows=True  # Shuffle for training
        )
        
        creation_time = time.time() - start_time
        
        print(f"\nDataset creation took {creation_time:.2f}s")
        print(f"Dataset stats:")
        print(f"  Files: {len(dataset.file_paths)}")
        print(f"  Windows: {len(dataset.windows)}")
        print(f"  Windows per file: {len(dataset.windows) / len(dataset.file_paths):.2f}")
        print(f"  Memory usage: {dataset.get_memory_usage()}")
        
        # Create DataLoader
        print("\nCreating DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        # Test loading a few batches
        print("\nLoading first 3 batches...")
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:
                break
            
            print(f"\nBatch {batch_idx}:")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  token_type_ids shape: {batch['token_type_ids'].shape}")
            print(f"  Non-padding tokens: {(batch['attention_mask'] == 1).sum().item()}")
            
            # Check diversity in batch
            first_tokens = batch['input_ids'][:, :10]
            unique_starts = len(torch.unique(first_tokens, dim=0))
            print(f"  Unique sequence starts in batch: {unique_starts}/{batch['input_ids'].shape[0]}")
        
        wandb.log({
            "production/num_files": len(dataset.file_paths),
            "production/num_windows": len(dataset.windows),
            "production/windows_per_file": len(dataset.windows) / len(dataset.file_paths),
            "production/dataset_creation_time": creation_time
        })
        
        return dataset, dataloader
    
    def test_memory_efficiency(self):
        """Test memory usage with sliding window."""
        print("\n" + "="*80)
        print("TEST: Memory Efficiency")
        print("="*80)
        
        # Get initial memory
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"Initial memory: {initial_memory:.2f} MB")
        
        # Create large dataset
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=None,  # ALL files
            cache_size=2000,
            use_sliding_window=True,
            window_stride=1024
        )
        
        after_dataset_memory = process.memory_info().rss / 1024 / 1024
        print(f"After dataset creation: {after_dataset_memory:.2f} MB")
        print(f"Dataset overhead: {after_dataset_memory - initial_memory:.2f} MB")
        
        # Load some samples
        print("\nLoading 100 random windows...")
        indices = random.sample(range(len(dataset)), min(100, len(dataset)))
        for idx in tqdm(indices):
            _ = dataset[idx]
        
        after_loading_memory = process.memory_info().rss / 1024 / 1024
        print(f"After loading 100 windows: {after_loading_memory:.2f} MB")
        print(f"Loading overhead: {after_loading_memory - after_dataset_memory:.2f} MB")
        
        wandb.log({
            "memory/initial_mb": initial_memory,
            "memory/after_dataset_mb": after_dataset_memory,
            "memory/after_loading_mb": after_loading_memory,
            "memory/dataset_overhead_mb": after_dataset_memory - initial_memory,
            "memory/loading_overhead_mb": after_loading_memory - after_dataset_memory
        })
    
    def run_all_tests(self):
        """Run all dataset tests."""
        print("\n" + "="*80)
        print("RUNNING ALL DATASET SLIDING WINDOW TESTS")
        print("="*80)
        
        # Test 1: Basic sliding window coverage
        dataset = self.test_sliding_window_coverage()
        
        # Test 2: Production scale
        dataset, dataloader = self.test_production_scale()
        
        # Test 3: Memory efficiency
        self.test_memory_efficiency()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
        
        wandb.finish()


if __name__ == "__main__":
    tester = DatasetProductionTester()
    tester.run_all_tests()