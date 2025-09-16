"""
Lazy-loading dataset for RRWEB sessions with random window sampling.
More efficient implementation that samples windows dynamically.
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
    Lazy-loading dataset for RRWEB sessions with random window sampling.
    Samples random windows from sequences during training for better coverage.
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
        use_random_window: bool = True,
        samples_per_epoch: Optional[int] = None
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
            use_random_window: Whether to use random window sampling
            samples_per_epoch: Total samples per epoch (default: 3x number of files)
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_session_length = min_session_length
        self.max_events_per_session = max_events_per_session
        self.use_random_window = use_random_window
        
        # Calculate vocabulary boundary for token type IDs
        self.structural_vocab_size = tokenizer.vocab.structural_vocab_size
        
        # Find all valid JSON files
        self.file_paths = self._find_valid_files(data_dir, max_files)
        
        # Set samples per epoch (for random window mode)
        if samples_per_epoch is None:
            # Default: 3 samples per file to get good coverage
            self.samples_per_epoch = len(self.file_paths) * 3
        else:
            self.samples_per_epoch = samples_per_epoch
        
        # Set up caching
        self._setup_cache(cache_size)
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        logger.info(f"Initialized lazy dataset with {len(self.file_paths)} files")
        logger.info(f"Random window mode: {use_random_window}, samples per epoch: {self.samples_per_epoch}")
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
    
    def _extract_window(self, tokens: List[int], start_idx: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Extract a window from tokens, either from start_idx or random position."""
        if len(tokens) <= self.max_length:
            # Entire sequence fits in one window
            window_tokens = tokens
            pad_length = self.max_length - len(window_tokens)
            if pad_length > 0:
                window_tokens = window_tokens + [0] * pad_length
                attention_mask = torch.tensor([1] * (self.max_length - pad_length) + [0] * pad_length, dtype=torch.long)
            else:
                attention_mask = torch.ones(self.max_length, dtype=torch.long)
        else:
            # Need to extract a window
            if start_idx is None:
                # Random window sampling
                max_start = len(tokens) - self.max_length
                start_idx = self.rng.randint(0, max_start + 1)
            
            window_tokens = tokens[start_idx:start_idx + self.max_length]
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
        """Return number of samples per epoch."""
        if self.use_random_window:
            return self.samples_per_epoch
        else:
            return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample (random window if enabled)."""
        import time
        import os
        
        # Map idx to file (with wraparound for random window mode)
        file_idx = idx % len(self.file_paths)
        file_path = self.file_paths[file_idx]
        
        start = time.time()
        
        # Get tokenized file from cache or tokenize
        tokens = self._cached_tokenize(file_path)
        
        if tokens is None or len(tokens) < self.min_session_length:
            # Return dummy sample if tokenization failed
            logger.warning(f"Failed to tokenize or too short: {file_path}")
            return {
                'input_ids': torch.zeros(self.max_length, dtype=torch.long),
                'attention_mask': torch.zeros(self.max_length, dtype=torch.long),
                'token_type_ids': torch.zeros(self.max_length, dtype=torch.long),
                'event_type_ids': torch.zeros(self.max_length, dtype=torch.long)
            }
        
        # Extract window (random if enabled)
        if self.use_random_window:
            print(f"  [DataLoader] Loading random window from file {file_idx} ({os.path.basename(file_path)})...")
            result = self._extract_window(tokens, start_idx=None)
        else:
            print(f"  [DataLoader] Loading full/truncated sequence from file {file_idx} ({os.path.basename(file_path)})...")
            result = self._extract_window(tokens, start_idx=0)
        
        print(f"  [DataLoader] Sample loaded in {time.time() - start:.2f}s")
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
            'samples_per_epoch': self.samples_per_epoch,
            'file_paths_size_mb': sys.getsizeof(self.file_paths) / 1024 / 1024,
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
        Create stratified splits based on files.
        
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n_files = len(dataset.file_paths)
        indices = list(range(n_files))
        
        # Shuffle
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        # Split files
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)
        
        train_files = [dataset.file_paths[i] for i in indices[:n_train]]
        val_files = [dataset.file_paths[i] for i in indices[n_train:n_train + n_val]]
        test_files = [dataset.file_paths[i] for i in indices[n_train + n_val:]]
        
        # Create separate datasets
        train_dataset = RRWebLazyDataset(
            data_dir="",  # Not used since we provide file_paths directly
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            cache_size=dataset._cached_tokenize.cache_info().maxsize,
            min_session_length=dataset.min_session_length,
            max_events_per_session=dataset.max_events_per_session,
            seed=seed,
            use_random_window=dataset.use_random_window,
            samples_per_epoch=len(train_files) * 3
        )
        train_dataset.file_paths = train_files
        
        val_dataset = RRWebLazyDataset(
            data_dir="",
            tokenizer=dataset.tokenizer,
            max_length=dataset.max_length,
            cache_size=100,
            min_session_length=dataset.min_session_length,
            max_events_per_session=dataset.max_events_per_session,
            seed=seed,
            use_random_window=False,  # No random window for validation
            samples_per_epoch=len(val_files)
        )
        val_dataset.file_paths = val_files
        
        logger.info(f"Split sizes - Train: {len(train_files)} files, Val: {len(val_files)} files, Test: {len(test_files)} files")
        
        return train_dataset, val_dataset, None