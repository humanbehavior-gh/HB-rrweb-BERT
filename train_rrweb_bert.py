#!/usr/bin/env python3
"""
Training script for RRWebBERT model with masked language modeling.
"""

import os
import sys
import json
import argparse
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed
)

# Add parent directory to path for tokenizer
sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

from configuration_rrweb import RRWebBERTConfig
from modeling_rrweb import RRWebBERTForMaskedLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RRWebDataset(Dataset):
    """Dataset for RRWEB sessions."""
    
    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_length: int = 2048,
        max_files: Optional[int] = None
    ):
        self.max_length = max_length
        
        # Load RRWeb tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        self.tokenizer = RRWebTokenizer.load(tokenizer_path)
        
        # Find all JSON files
        self.files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    self.files.append(os.path.join(root, file))
        
        if max_files:
            self.files = self.files[:max_files]
        
        logger.info(f"Found {len(self.files)} RRWEB files")
        
        # Pre-tokenize all files (for faster training)
        self.tokenized_sessions = []
        logger.info("Pre-tokenizing sessions...")
        
        for file_path in tqdm(self.files[:max_files] if max_files else self.files, desc="Tokenizing"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Handle different formats
                events = []
                if isinstance(data, list):
                    events = data
                elif isinstance(data, dict) and 'events' in data:
                    events = data['events']
                
                if events:
                    tokens = self.tokenizer.tokenize_session(events)
                    if len(tokens) > 10:  # Skip very short sessions
                        self.tokenized_sessions.append(tokens)
            except Exception as e:
                logger.debug(f"Error processing {file_path}: {e}")
                continue
        
        logger.info(f"Tokenized {len(self.tokenized_sessions)} sessions")
    
    def __len__(self):
        return len(self.tokenized_sessions)
    
    def __getitem__(self, idx):
        tokens = self.tokenized_sessions[idx]
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        input_ids = torch.tensor(tokens, dtype=torch.long)
        
        # Create attention mask
        attention_mask = torch.ones_like(input_ids)
        
        # Token type IDs (0 for structural, 1 for text)
        token_type_ids = torch.zeros_like(input_ids)
        for i, token_id in enumerate(tokens):
            if token_id >= self.tokenizer.text_token_offset:
                token_type_ids[i] = 1
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids
        }


class TokenizerWrapper:
    """Wrapper to make RRWebTokenizer compatible with HuggingFace trainer."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.bpe_tokenizer = tokenizer.bpe_tokenizer
        self.vocab_size = tokenizer.vocab.structural_vocab_size + (len(tokenizer.bpe_tokenizer.vocab) if tokenizer.bpe_tokenizer else 0)
        
    def save_pretrained(self, save_directory):
        """Dummy method for HuggingFace compatibility."""
        pass

@dataclass
class DataCollatorForRRWebMLM:
    """Data collator for masked language modeling on RRWEB tokens."""
    
    tokenizer: Any
    mlm_probability: float = 0.15
    pad_token_id: int = 0
    mask_token_id: int = 4
    
    def __call__(self, examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_length = max(len(ex['input_ids']) for ex in examples)
        
        # Pad all examples to max length
        batch = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }
        
        for example in examples:
            length = len(example['input_ids'])
            padding_length = max_length - length
            
            # Pad tensors
            input_ids = F.pad(example['input_ids'], (0, padding_length), value=self.pad_token_id)
            attention_mask = F.pad(example['attention_mask'], (0, padding_length), value=0)
            token_type_ids = F.pad(example['token_type_ids'], (0, padding_length), value=0)
            
            # Create labels (copy of input_ids for MLM)
            labels = input_ids.clone()
            
            # Apply masking
            probability_matrix = torch.full(labels.shape, self.mlm_probability)
            # Don't mask special tokens (PAD, CLS, SEP, MASK, UNK)
            special_tokens_mask = (input_ids == self.pad_token_id) | (input_ids < 5)
            probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
            
            # Also don't mask padding
            probability_matrix[attention_mask == 0] = 0.0
            
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[~masked_indices] = -100  # Only compute loss on masked tokens
            
            # 80% of the time, replace with [MASK]
            indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
            input_ids[indices_replaced] = self.mask_token_id
            
            # 10% of the time, replace with random token
            indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
            # Use the actual vocab size from tokenizer
            vocab_size = self.tokenizer.vocab_size if hasattr(self.tokenizer, 'vocab_size') else 640
            random_tokens = torch.randint(5, vocab_size, labels.shape, dtype=torch.long)
            input_ids[indices_random] = random_tokens[indices_random]
            
            # 10% of the time, keep original token
            
            batch['input_ids'].append(input_ids)
            batch['attention_mask'].append(attention_mask)
            batch['token_type_ids'].append(token_type_ids)
            batch['labels'].append(labels)
        
        # Stack into tensors
        return {
            'input_ids': torch.stack(batch['input_ids']),
            'attention_mask': torch.stack(batch['attention_mask']),
            'token_type_ids': torch.stack(batch['token_type_ids']),
            'labels': torch.stack(batch['labels'])
        }


def train():
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with RRWEB JSON files')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to trained RRWeb tokenizer')
    parser.add_argument('--output_dir', type=str, default='./rrweb-bert-model', help='Output directory')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='Masking probability')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--save_steps', type=int, default=5000, help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=500, help='Log every N steps')
    parser.add_argument('--max_files', type=int, default=None, help='Maximum number of files to use (None for all)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--wandb_project', type=str, default='rrweb-bert', help='WandB project name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize wandb if available
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=f"rrweb-bert-{args.model_size}-{args.num_epochs}ep",
            config=vars(args)
        )
    
    # Load tokenizer to get vocab size
    rrweb_tokenizer = RRWebTokenizer.load(args.tokenizer_path)
    vocab_size = rrweb_tokenizer.vocab.structural_vocab_size + len(rrweb_tokenizer.bpe_tokenizer.vocab)
    
    # Configure model based on size
    model_configs = {
        'small': {
            'hidden_size': 512,
            'num_hidden_layers': 6,
            'num_attention_heads': 8,
            'intermediate_size': 2048
        },
        'base': {
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072
        },
        'large': {
            'hidden_size': 1024,
            'num_hidden_layers': 24,
            'num_attention_heads': 16,
            'intermediate_size': 4096
        }
    }
    
    # Create config
    config = RRWebBERTConfig(
        vocab_size=vocab_size,
        max_position_embeddings=args.max_length,
        structural_vocab_size=rrweb_tokenizer.vocab.structural_vocab_size,
        text_vocab_size=len(rrweb_tokenizer.bpe_tokenizer.vocab) if rrweb_tokenizer.bpe_tokenizer else 0,
        **model_configs[args.model_size]
    )
    
    # Create model
    logger.info(f"Creating RRWebBERT model ({args.model_size})")
    model = RRWebBERTForMaskedLM(config)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = RRWebDataset(
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        max_length=args.max_length,
        max_files=args.max_files
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data collator with tokenizer wrapper
    tokenizer_wrapper = TokenizerWrapper(rrweb_tokenizer)
    data_collator = DataCollatorForRRWebMLM(
        tokenizer=tokenizer_wrapper,
        mlm_probability=args.mlm_probability
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        logging_dir='./logs',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        bf16=False,  # Use fp16 instead of bf16
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=f"rrweb-bert-{args.model_size}-{args.num_epochs}ep",
        logging_first_step=True,
        save_safetensors=False,  # Avoid tokenizer save issue
        gradient_checkpointing=True if args.model_size == 'large' else False,
        remove_unused_columns=False,
        label_names=['labels']
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save final model
    logger.info(f"Saving model to {args.output_dir}")
    trainer.save_model()
    
    # Save config
    config.save_pretrained(args.output_dir)
    
    logger.info("Training complete!")
    
    # Log final metrics
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.log({
            "final_model_size": args.model_size,
            "total_params": sum(p.numel() for p in model.parameters()),
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "vocab_size": vocab_size,
            "dataset_size": len(dataset),
            "train_size": train_size,
            "val_size": val_size
        })
        wandb.finish()


if __name__ == "__main__":
    train()