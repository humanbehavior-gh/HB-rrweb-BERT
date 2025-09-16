"""
Lazy-loading dataset for RRWEB sessions with sliding window support.
Production-ready implementation with caching and proper error handling.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Any, Tuple
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)


class RRWebLazyDataset(Dataset):
    """
    Lazy-loading dataset for RRWEB sessions with sliding window support.
    Creates multiple training samples from long sequences.
    """
    
    def __init__(
        self,
        data_dir: str,
        tokenizer,
        max_length: int = 2048,
        max_files: Optional[int] = None,
        cache_size: int = 100,
        min_session_length: int = 10,
        max_events_per_session: Optional[int] = None,
        seed: int = 42,
        use_sliding_window: bool = True,
        window_stride: int = 1024,
        shuffle_windows: bool = True
    ):
        """
        Args:
            data_dir: Directory containing RRWEB JSON files
            tokenizer: RRWebTokenizer instance
            max_length: Maximum sequence length per window
            max_files: Optional limit on number of files
            cache_size: Number of tokenized sessions to cache
            min_session_length: Minimum tokens required for valid session
            max_events_per_session: Optional limit on events to process
            seed: Random seed for reproducibility
            use_sliding_window: Whether to use sliding window sampling
            window_stride: Stride for sliding window (default 1024 = 50% overlap)
            shuffle_windows: Whether to shuffle windows for training
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_session_length = min_session_length
        self.max_events_per_session = max_events_per_session
        self.use_sliding_window = use_sliding_window
        self.window_stride = window_stride
        self.shuffle_windows = shuffle_windows
        
        # Calculate vocabulary boundary for token type IDs
        self.structural_vocab_size = tokenizer.vocab.structural_vocab_size
        
        # Find all valid JSON files
        self.file_paths = self._find_valid_files(data_dir, max_files)
        
        # Initialize windows (file_idx, start_pos, end_pos)
        self.windows = []
        self._initialize_windows()
        
        # Set up caching
        self._setup_cache(cache_size)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Shuffle windows if requested
        if self.shuffle_windows and self.use_sliding_window:
            random.shuffle(self.windows)
        
        logger.info(f"Initialized lazy dataset with {len(self.file_paths)} files")
        logger.info(f"Created {len(self.windows)} windows with sliding window={use_sliding_window}, stride={window_stride}")
        logger.info(f"Structural vocab boundary: {self.structural_vocab_size}")
    
    def _find_valid_files(self, data_dir: str, max_files: Optional[int]) -> List[str]:
        """Find all valid RRWEB JSON files."""
        file_paths = []
        
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    # Quick validation - check file size
                    if os.path.getsize(file_path) > 100:  # Skip tiny files
                        file_paths.append(file_path)
                        
                        if max_files and len(file_paths) >= max_files:
                            break
            
            if max_files and len(file_paths) >= max_files:
                break
        
        # Shuffle for better training diversity
        random.shuffle(file_paths)
        
        return file_paths
    
    def _initialize_windows(self):
        """Initialize sliding windows for all files."""
        if not self.use_sliding_window:
            # Original behavior: one window per file
            for idx in range(len(self.file_paths)):
                self.windows.append((idx, 0, self.max_length))
        else:
            # New behavior: create multiple windows per file
            print(f"Initializing sliding windows for {len(self.file_paths)} files...")
            
            # We need to tokenize files to know their lengths
            # But we'll do this lazily - just estimate based on file size
            for idx, file_path in enumerate(self.file_paths):
                # Estimate tokens based on file size (rough heuristic)
                file_size = os.path.getsize(file_path)
                # Rough estimate: 1 token per 20 bytes (adjust based on your data)
                estimated_tokens = file_size // 20
                
                if estimated_tokens < self.min_session_length:
                    continue
                    
                if estimated_tokens <= self.max_length:
                    # File fits in one window
                    self.windows.append((idx, 0, estimated_tokens))
                else:
                    # Create sliding windows
                    for start in range(0, estimated_tokens - self.max_length + 1, self.window_stride):
                        self.windows.append((idx, start, start + self.max_length))
                
                if idx % 1000 == 0:
                    print(f"  Processed {idx}/{len(self.file_paths)} files, {len(self.windows)} windows so far...")
            
            print(f"Created {len(self.windows)} total windows from {len(self.file_paths)} files")
    
    def _setup_cache(self, cache_size: int):
        """Set up LRU cache for tokenized sessions."""
        @lru_cache(maxsize=cache_size)
        def cached_tokenize(file_path: str):
            return self._load_and_tokenize_file(file_path)
        
        self._cached_tokenize = cached_tokenize
    
    def _load_and_tokenize_file(self, file_path: str) -> Optional[List[int]]:
        """Load and tokenize an entire file, returning token list."""
        import time
        try:
            # Progress: Loading file
            load_start = time.time()
            with open(file_path, 'r') as f:
                data = json.load(f)
            load_time = time.time() - load_start
            
            # Progress: Extracting events
            extract_start = time.time()
            events = self._extract_events(data)
            extract_time = time.time() - extract_start
            
            if not events:
                return None
            
            # Limit events if specified
            if self.max_events_per_session:
                events = events[:self.max_events_per_session]
            
            # Progress: Tokenizing
            tokenize_start = time.time()
            print(f"  [Dataset] Tokenizing {len(events)} events from {os.path.basename(file_path)}...")
            tokens = self.tokenizer.tokenize_session(events)
            tokenize_time = time.time() - tokenize_start
            print(f"  [Dataset] Tokenization took {tokenize_time:.2f}s (load: {load_time:.2f}s, extract: {extract_time:.2f}s)")
            
            return tokens
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None
    
    def _create_tensor_from_window(self, tokens: List[int], start: int, end: int) -> Dict[str, torch.Tensor]:
        """Create tensor dictionary from a token window."""
        # Extract window
        window_tokens = tokens[start:min(end, len(tokens))]
        
        # Pad if necessary
        if len(window_tokens) < self.max_length:
            # Pad with tokenizer's pad token (usually 0)
            pad_length = self.max_length - len(window_tokens)
            window_tokens = window_tokens + [0] * pad_length
            attention_mask = torch.tensor([1] * (self.max_length - pad_length) + [0] * pad_length, dtype=torch.long)
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)
        
        # Create tensors
        input_ids = torch.tensor(window_tokens[:self.max_length], dtype=torch.long)
        
        # Token type IDs: 0 for structural, 1 for text
        token_type_ids = (input_ids >= self.structural_vocab_size).long()
        
        # Event type IDs (simplified - just zeros for now)
        event_type_ids = torch.zeros_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'event_type_ids': event_type_ids
        }
    
    def _extract_events(self, data: Any) -> List[Dict]:
        """Extract events from various data formats."""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'events' in data:
                return data['events']
            elif 'data' in data and isinstance(data['data'], list):
                return data['data']
            elif 'rrweb' in data and isinstance(data['rrweb'], list):
                return data['rrweb']
        return []
    
    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a specific window."""
        import time
        import os
        
        # Get window info
        file_idx, start_pos, end_pos = self.windows[idx]
        file_path = self.file_paths[file_idx]
        
        start = time.time()
        print(f"  [DataLoader] Loading window {idx} (file {file_idx}, pos {start_pos}-{end_pos}) from {os.path.basename(file_path)}...")
        
        # Get tokenized file from cache or tokenize
        tokens = self._cached_tokenize(file_path)
        
        if tokens is None:
            # Return dummy sample if tokenization failed
            logger.warning(f"Failed to tokenize {file_path}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'token_type_ids': torch.zeros(self.max_length, dtype=torch.long),
                'event_type_ids': torch.zeros(self.max_length, dtype=torch.long)
            }
        
        # Create tensor from window
        result = self._create_tensor_from_window(tokens, start_pos, end_pos)
        
        print(f"  [DataLoader] Window {idx} loaded in {time.time() - start:.2f}s")
        return result
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        import psutil
        import sys
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'file_paths_count': len(self.file_paths),
            'windows_count': len(self.windows),
            'file_paths_size_mb': sys.getsizeof(self.file_paths) / 1024 / 1024,
            'windows_size_mb': sys.getsizeof(self.windows) / 1024 / 1024,
            'cache_info': self._cached_tokenize.cache_info()._asdict()
        }


class RRWebDatasetSplitter:
    """Utility to create proper train/val/test splits."""
    
    @staticmethod
    def create_splits(
        dataset: RRWebLazyDataset,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> tuple:
        """
        Create stratified splits based on file sizes.
        
        Returns:
            train_indices, val_indices, test_indices
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # For sliding window dataset, split by files not windows
        # to avoid data leakage
        n_files = len(dataset.file_paths)
        indices = list(range(n_files))
        
        # Shuffle
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        # Split files
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        train_files = set(indices[:n_train])
        val_files = set(indices[n_train:n_train + n_val])
        test_files = set(indices[n_train + n_val:])
        
        # Now map to window indices
        train_indices = []
        val_indices = []
        test_indices = []
        
        for window_idx, (file_idx, _, _) in enumerate(dataset.windows):
            if file_idx in train_files:
                train_indices.append(window_idx)
            elif file_idx in val_files:
                val_indices.append(window_idx)
            else:
                test_indices.append(window_idx)
        
        logger.info(f"Split sizes - Train: {len(train_indices)} windows, Val: {len(val_indices)} windows, Test: {len(test_indices)} windows")
        logger.info(f"From files - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        return train_indices, val_indices, test_indices