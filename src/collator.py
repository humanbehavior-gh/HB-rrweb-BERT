"""
Improved data collator for masked language modeling with proper 80/10/10 split.
Production-ready with configurable masking strategies.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ImprovedDataCollatorForMLM:
    """
    Data collator for masked language modeling with exact probability control.
    
    Properly implements:
    - 80% masked tokens → [MASK]
    - 10% masked tokens → random token
    - 10% masked tokens → keep original
    """
    
    tokenizer: Any  # TokenizerWrapper
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None
    
    def __post_init__(self):
        """Extract special token IDs from tokenizer."""
        # Get special tokens from tokenizer
        if hasattr(self.tokenizer, 'tokenizer'):
            # TokenizerWrapper
            actual_tokenizer = self.tokenizer.tokenizer
            self.pad_token_id = 0  # [PAD]
            self.cls_token_id = 2  # [CLS]
            self.sep_token_id = 3  # [SEP]
            self.mask_token_id = 4  # [MASK]
            self.unk_token_id = 1  # [UNK]
            self.vocab_size = self.tokenizer.vocab_size
            self.structural_vocab_size = actual_tokenizer.vocab.structural_vocab_size
        else:
            # Fallback defaults
            self.pad_token_id = 0
            self.cls_token_id = 2
            self.sep_token_id = 3
            self.mask_token_id = 4
            self.unk_token_id = 1
            self.vocab_size = 12520
            self.structural_vocab_size = 520
        
        # Special tokens that should never be masked
        self.special_token_ids = {
            self.pad_token_id,
            self.cls_token_id,
            self.sep_token_id,
            self.mask_token_id,
            self.unk_token_id
        }
        
        logger.info(f"Collator initialized - Vocab: {self.vocab_size}, Structural: {self.structural_vocab_size}")
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch with proper masking."""
        
        # Find max length in batch
        max_length = max(len(ex['input_ids']) for ex in examples)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Initialize batch tensors
        batch_size = len(examples)
        input_ids = torch.full((batch_size, max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
        token_type_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        event_type_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        
        # Fill batch and apply masking
        for i, example in enumerate(examples):
            seq_len = len(example['input_ids'])
            
            # Copy original sequences
            input_ids[i, :seq_len] = example['input_ids']
            attention_mask[i, :seq_len] = example['attention_mask']
            token_type_ids[i, :seq_len] = example['token_type_ids']
            
            if 'event_type_ids' in example:
                event_type_ids[i, :seq_len] = example['event_type_ids']
            
            # Apply MLM masking
            masked_input_ids, mlm_labels = self._mask_tokens(
                input_ids[i, :seq_len].clone(),
                attention_mask[i, :seq_len]
            )
            
            input_ids[i, :seq_len] = masked_input_ids
            labels[i, :seq_len] = mlm_labels
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'event_type_ids': event_type_ids,
            'labels': labels
        }
    
    def _mask_tokens(
        self,
        inputs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> tuple:
        """
        Apply masking with exact 80/10/10 split.
        
        Returns:
            (masked_inputs, labels)
        """
        labels = inputs.clone()
        
        # Create probability matrix for masking
        probability_matrix = torch.full(inputs.shape, self.mlm_probability)
        
        # Don't mask special tokens
        for token_id in self.special_token_ids:
            probability_matrix[inputs == token_id] = 0.0
        
        # Don't mask padding
        probability_matrix[attention_mask == 0] = 0.0
        
        # Sample masked indices
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Only compute loss on masked tokens
        labels[~masked_indices] = -100
        
        # Get indices to be masked
        indices_to_mask = torch.where(masked_indices)[0]
        
        if len(indices_to_mask) == 0:
            return inputs, labels
        
        # Shuffle indices for random assignment
        shuffled_indices = indices_to_mask[torch.randperm(len(indices_to_mask))]
        
        # Calculate split points for 80/10/10
        n_mask = len(shuffled_indices)
        n_replaced = int(0.8 * n_mask)
        n_random = int(0.1 * n_mask)
        # Rest are kept original
        
        # 80% -> [MASK]
        mask_indices = shuffled_indices[:n_replaced]
        inputs[mask_indices] = self.mask_token_id
        
        # 10% -> random token
        if n_random > 0:
            random_indices = shuffled_indices[n_replaced:n_replaced + n_random]
            # Generate random tokens (avoiding special tokens)
            random_tokens = torch.randint(
                low=5,  # Start after special tokens
                high=self.vocab_size,
                size=(len(random_indices),),
                dtype=torch.long
            )
            inputs[random_indices] = random_tokens
        
        # 10% -> keep original (do nothing)
        
        return inputs, labels
    
    def get_masking_stats(
        self,
        examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Get statistics about masking for debugging."""
        total_tokens = 0
        total_masked = 0
        total_mask_token = 0
        total_random = 0
        total_kept = 0
        
        for example in examples:
            seq_len = len(example['input_ids'])
            original = example['input_ids'][:seq_len].clone()
            
            masked, labels = self._mask_tokens(
                original.clone(),
                torch.ones_like(original)
            )
            
            masked_positions = labels != -100
            total_tokens += seq_len
            total_masked += masked_positions.sum().item()
            
            # Count each type
            for i in range(seq_len):
                if masked_positions[i]:
                    if masked[i] == self.mask_token_id:
                        total_mask_token += 1
                    elif masked[i] != original[i]:
                        total_random += 1
                    else:
                        total_kept += 1
        
        return {
            'total_tokens': total_tokens,
            'total_masked': total_masked,
            'mask_ratio': total_masked / max(total_tokens, 1),
            'mask_token_ratio': total_mask_token / max(total_masked, 1),
            'random_ratio': total_random / max(total_masked, 1),
            'kept_ratio': total_kept / max(total_masked, 1)
        }


@dataclass 
class DataCollatorForPreTraining:
    """
    Data collator for both MLM and next sentence prediction (if needed).
    Extended version with additional features.
    """
    
    tokenizer: Any
    mlm_probability: float = 0.15
    nsp_probability: float = 0.5  # For future NSP task
    pad_to_multiple_of: Optional[int] = None
    mask_structural_tokens: bool = True
    mask_text_tokens: bool = True
    
    def __post_init__(self):
        """Initialize from base collator."""
        self.mlm_collator = ImprovedDataCollatorForMLM(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
            pad_to_multiple_of=self.pad_to_multiple_of
        )
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate with optional NSP task."""
        # For now, just use MLM
        # NSP can be added later if needed for session continuity
        return self.mlm_collator(examples)