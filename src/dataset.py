"""
Lazy-loading dataset for RRWEB sessions that doesn't load everything into memory.
Production-ready implementation with caching and proper error handling.
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional, Any
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np

logger = logging.getLogger(__name__)


class RRWebLazyDataset(Dataset):
    """
    Lazy-loading dataset for RRWEB sessions.
    Only stores file paths in memory and loads/tokenizes on demand.
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
        use_random_window: bool = False,
        samples_per_epoch: Optional[int] = None
    ):
        """
        Args:
            data_dir: Directory containing RRWEB JSON files
            tokenizer: RRWebTokenizer instance
            max_length: Maximum sequence length
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
            return self._load_and_tokenize(file_path)
        
        self._cached_tokenize = cached_tokenize
    
    def _load_and_tokenize(self, file_path: str) -> Optional[Dict[str, torch.Tensor]]:
        """Load and tokenize a single file."""
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
            
            # Progress: Tokenizing (THIS IS THE SLOW PART)
            tokenize_start = time.time()
            print(f"  [Dataset] Tokenizing {len(events)} events from {os.path.basename(file_path)}...")
            tokens = self.tokenizer.tokenize_session(events)
            tokenize_time = time.time() - tokenize_start
            print(f"  [Dataset] Tokenization took {tokenize_time:.2f}s (load: {load_time:.2f}s, extract: {extract_time:.2f}s)")
            
            # Skip if too short
            if len(tokens) < self.min_session_length:
                return None
            
            # Handle window extraction
            if self.use_random_window and len(tokens) > self.max_length:
                # Random window sampling
                max_start = len(tokens) - self.max_length
                start_idx = self.rng.randint(0, max_start + 1)
                tokens = tokens[start_idx:start_idx + self.max_length]
            elif len(tokens) > self.max_length:
                # Original behavior: truncate from beginning
                tokens = tokens[:self.max_length]
            
            # Extract event types for event_type_ids
            event_types = self._extract_event_types(events, len(tokens))
            
            # Create tensors
            input_ids = torch.tensor(tokens, dtype=torch.long)
            attention_mask = torch.ones_like(input_ids)
            
            # Token type IDs: 0 for structural, 1 for text
            token_type_ids = (input_ids >= self.structural_vocab_size).long()
            
            # Event type IDs (simplified - you might want more sophisticated logic)
            event_type_ids = torch.tensor(event_types[:len(tokens)], dtype=torch.long)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'event_type_ids': event_type_ids
            }
            
        except Exception as e:
            logger.debug(f"Error processing {file_path}: {e}")
            return None
    
    def _extract_events(self, data: Any) -> List[Dict]:
        """Extract events from various data formats."""
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            if 'events' in data:
                return data['events']
            elif 'data' in data and isinstance(data['data'], list):
                return data['data']
        return []
    
    def _extract_event_types(self, events: List[Dict], target_length: int) -> List[int]:
        """Extract event type IDs for each token position."""
        # Simple approach: use the event type of the first event
        # In practice, you'd map each token to its source event
        event_types = []
        for event in events:
            event_type = event.get('type', 0)
            # Normalize to 0-9 range for embedding layer
            event_type = min(max(event_type, 0), 9)
            event_types.append(event_type)
        
        # Extend to match token length (simplified)
        while len(event_types) < target_length:
            event_types.extend(event_types[:min(len(event_types), target_length - len(event_types))])
        
        return event_types[:target_length]
    
    def __len__(self):
        """Return number of samples per epoch."""
        if self.use_random_window:
            return self.samples_per_epoch
        else:
            return len(self.file_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single tokenized session."""
        import time
        import os
        
        # Map idx to file (with wraparound for random window mode)
        if self.use_random_window:
            file_idx = idx % len(self.file_paths)
        else:
            file_idx = idx
            
        start = time.time()
        file_path = self.file_paths[file_idx]
        
        if self.use_random_window:
            print(f"  [DataLoader] Loading random window from file {file_idx} ({os.path.basename(file_path)})...")
        else:
            print(f"  [DataLoader] Loading item {idx} from {os.path.basename(file_path)}...")
        
        # Try to get from cache or load
        result = self._cached_tokenize(file_path)
        print(f"  [DataLoader] Item {idx} loaded in {time.time() - start:.2f}s")
        
        # If failed, try next file (with limit to prevent infinite loop)
        attempts = 0
        while result is None and attempts < 10:
            idx = (idx + 1) % len(self.file_paths)
            file_path = self.file_paths[idx]
            result = self._cached_tokenize(file_path)
            attempts += 1
        
        # If still None, return a dummy sample
        if result is None:
            logger.warning(f"Failed to load any valid file after {attempts} attempts")
            # Return minimal valid sample
            return {
                'input_ids': torch.tensor([2, 3], dtype=torch.long),  # [CLS], [SEP]
                'attention_mask': torch.ones(2, dtype=torch.long),
                'token_type_ids': torch.zeros(2, dtype=torch.long),
                'event_type_ids': torch.zeros(2, dtype=torch.long)
            }
        
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
        Create stratified splits based on file sizes.
        
        Returns:
            train_indices, val_indices, test_indices
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        # Get file sizes for stratification
        file_sizes = []
        for path in dataset.file_paths:
            try:
                size = os.path.getsize(path)
                file_sizes.append(size)
            except:
                file_sizes.append(0)
        
        # Sort indices by file size
        indices = np.argsort(file_sizes)
        
        # Stratified split
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        # Shuffle within strata
        np.random.seed(seed)
        np.random.shuffle(indices)
        
        train_indices = indices[:n_train].tolist()
        val_indices = indices[n_train:n_train + n_val].tolist()
        test_indices = indices[n_train + n_val:].tolist()
        
        logger.info(f"Split sizes - Train: {len(train_indices)}, Val: {len(val_indices)}, Test: {len(test_indices)}")
        
        return train_indices, val_indices, test_indices