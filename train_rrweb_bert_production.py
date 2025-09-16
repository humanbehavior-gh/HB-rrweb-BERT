#!/usr/bin/env python3
"""
Production training script for RRWebBERT with all optimizations from testing.
Incorporates lessons from Test 1.2:
- Lazy loading dataset (no OOM)
- Proper 80/10/10 masking
- Comprehensive monitoring
- No timeouts, iterate until working
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import wandb
import psutil
import numpy as np

# Add parent directory for imports
sys.path.append('/home/ubuntu/rrweb_tokenizer')
sys.path.append('/home/ubuntu/rrweb-bert/src')

from rrweb_tokenizer import RRWebTokenizer
from configuration_rrweb import RRWebBERTConfig
from modeling_rrweb import RRWebBERTForMaskedLM
from dataset import RRWebLazyDataset, RRWebDatasetSplitter
from collator import ImprovedDataCollatorForMLM
from metrics import RRWebMetrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """Wrapper to make RRWebTokenizer compatible with HuggingFace."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab
        self.bpe_tokenizer = tokenizer.bpe_tokenizer
        self.vocab_size = tokenizer.vocab.structural_vocab_size + len(tokenizer.bpe_tokenizer.vocab)
        
    def save_pretrained(self, save_directory):
        """Save tokenizer for HuggingFace compatibility."""
        os.makedirs(save_directory, exist_ok=True)
        # Save the actual tokenizer model
        save_path = os.path.join(save_directory, 'rrweb_tokenizer')
        self.tokenizer.save(save_path)


class ProductionTrainer:
    """Production trainer with comprehensive monitoring and no timeouts."""
    
    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        
        # Set seeds for reproducibility
        set_seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Initialize wandb
        if not args.no_wandb:
            wandb.init(
                project=args.wandb_project,
                name=f"rrweb-bert-{args.model_size}-production-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=vars(args),
                resume="allow"
            )
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer from {args.tokenizer_path}")
        self.tokenizer = RRWebTokenizer.load(args.tokenizer_path)
        self.tokenizer_wrapper = TokenizerWrapper(self.tokenizer)
        
        # Setup model
        self.setup_model()
        
        # Setup data
        self.setup_data()
        
        # Setup training components
        self.setup_training()
        
        # Log initial state
        self.log_initial_state()
    
    def setup_model(self):
        """Initialize model with proper configuration."""
        vocab_size = self.tokenizer_wrapper.vocab_size
        structural_vocab_size = self.tokenizer.vocab.structural_vocab_size
        
        # Model configurations
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
        self.config = RRWebBERTConfig(
            vocab_size=vocab_size,
            max_position_embeddings=self.args.max_length,
            structural_vocab_size=structural_vocab_size,
            text_vocab_size=vocab_size - structural_vocab_size,
            **model_configs[self.args.model_size]
        )
        
        # Create or load model
        if self.args.resume_from_checkpoint:
            logger.info(f"Loading checkpoint from {self.args.resume_from_checkpoint}")
            self.model = RRWebBERTForMaskedLM.from_pretrained(self.args.resume_from_checkpoint)
        else:
            logger.info(f"Creating new {self.args.model_size} model")
            self.model = RRWebBERTForMaskedLM(self.config)
        
        self.model.to(self.device)
        
        # Log model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    def setup_data(self):
        """Setup datasets with lazy loading and proper splits."""
        logger.info("Setting up datasets...")
        
        # Create lazy dataset (no OOM issues)
        full_dataset = RRWebLazyDataset(
            data_dir=self.args.data_dir,
            tokenizer=self.tokenizer,
            max_length=self.args.max_length,
            max_files=self.args.max_files,
            cache_size=self.args.cache_size,
            min_session_length=10,
            seed=self.args.seed
        )
        
        logger.info(f"Total dataset size: {len(full_dataset)} files")
        
        # Create proper splits
        train_indices, val_indices, test_indices = RRWebDatasetSplitter.create_splits(
            dataset=full_dataset,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            seed=self.args.seed
        )
        
        # Create subset datasets
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = Subset(full_dataset, test_indices)
        
        logger.info(f"Dataset splits - Train: {len(self.train_dataset)}, Val: {len(self.val_dataset)}, Test: {len(self.test_dataset)}")
        
        # Create data collator with proper masking
        self.data_collator = ImprovedDataCollatorForMLM(
            tokenizer=self.tokenizer_wrapper,
            mlm_probability=self.args.mlm_probability,
            pad_to_multiple_of=8  # For tensor core efficiency
        )
        
        # Test collator statistics
        if self.args.test_collator:
            self.test_collator_stats()
    
    def test_collator_stats(self):
        """Test that collator produces correct 80/10/10 split."""
        logger.info("Testing collator masking statistics...")
        
        # Get a small batch
        test_batch = [self.train_dataset[i] for i in range(min(10, len(self.train_dataset)))]
        stats = self.data_collator.get_masking_stats(test_batch)
        
        logger.info(f"Masking stats - Total: {stats['total_tokens']}, Masked: {stats['total_masked']} ({stats['mask_ratio']:.1%})")
        logger.info(f"  [MASK]: {stats['mask_token_ratio']:.1%}")
        logger.info(f"  Random: {stats['random_ratio']:.1%}")
        logger.info(f"  Kept: {stats['kept_ratio']:.1%}")
        
        # Log to wandb
        if not self.args.no_wandb:
            wandb.log({"collator_stats": stats})
    
    def setup_training(self):
        """Setup training components."""
        # Calculate training steps
        self.total_steps = (len(self.train_dataset) // self.args.batch_size) * self.args.num_epochs
        self.warmup_steps = min(self.args.warmup_steps, self.total_steps // 10)
        
        logger.info(f"Training steps - Total: {self.total_steps}, Warmup: {self.warmup_steps}")
        
        # Setup optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Setup metrics tracker
        self.metrics = RRWebMetrics(
            vocab_size=self.tokenizer_wrapper.vocab_size,
            structural_vocab_size=self.tokenizer.vocab.structural_vocab_size
        )
    
    def log_initial_state(self):
        """Log initial state to wandb."""
        if self.args.no_wandb:
            return
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        wandb.log({
            "initial/model_size": self.args.model_size,
            "initial/total_params": sum(p.numel() for p in self.model.parameters()),
            "initial/vocab_size": self.tokenizer_wrapper.vocab_size,
            "initial/dataset_size": len(self.train_dataset) + len(self.val_dataset) + len(self.test_dataset),
            "initial/memory_mb": memory_info.rss / 1024 / 1024,
            "initial/device": str(self.device)
        })
    
    def train(self):
        """Main training loop with no timeouts."""
        logger.info("Starting production training...")
        
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )
        
        # Training state
        global_step = 0
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop - NO TIMEOUTS
        for epoch in range(self.args.num_epochs):
            epoch_start = time.time()
            logger.info(f"\n{'='*50}")
            logger.info(f"Epoch {epoch + 1}/{self.args.num_epochs}")
            logger.info(f"{'='*50}")
            
            # Training phase
            self.model.train()
            train_loss = 0
            train_steps = 0
            
            train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            for batch_idx, batch in enumerate(train_pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                
                # Optimizer step
                if (batch_idx + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Log metrics
                    if global_step % self.args.logging_steps == 0:
                        self.log_training_metrics(
                            loss=loss.item(),
                            global_step=global_step,
                            epoch=epoch + 1,
                            learning_rate=self.scheduler.get_last_lr()[0]
                        )
                
                train_loss += loss.item()
                train_steps += 1
                
                # Update progress bar
                train_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{train_loss/train_steps:.4f}",
                    'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                })
                
                # Validation check
                if global_step % self.args.eval_steps == 0:
                    val_loss = self.validate(val_loader, global_step)
                    
                    # Check for improvement
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        self.save_checkpoint(global_step, val_loss)
                    else:
                        patience_counter += 1
                        
                    # Early stopping check (but we iterate until working)
                    if patience_counter >= self.args.patience:
                        logger.warning(f"No improvement for {patience_counter} evaluations")
                        # Don't actually stop - iterate until working
                    
                    self.model.train()
            
            # Epoch complete
            avg_train_loss = train_loss / train_steps
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Epoch {epoch + 1} complete - Loss: {avg_train_loss:.4f}, Time: {epoch_time/60:.1f} min")
            
            # End of epoch validation
            val_loss = self.validate(val_loader, global_step)
            
            # Log epoch metrics
            if not self.args.no_wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "epoch/train_loss": avg_train_loss,
                    "epoch/val_loss": val_loss,
                    "epoch/time_minutes": epoch_time / 60,
                    "epoch/best_val_loss": best_val_loss
                })
        
        # Training complete
        total_time = time.time() - self.start_time
        logger.info(f"\n{'='*50}")
        logger.info(f"Training complete! Total time: {total_time/3600:.1f} hours")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final model
        self.save_checkpoint(global_step, val_loss, final=True)
        
        # Final test evaluation
        self.test_model()
    
    def validate(self, val_loader, global_step):
        """Validation loop."""
        self.model.eval()
        val_loss = 0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                val_loss += outputs.loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        logger.info(f"Validation - Loss: {avg_val_loss:.4f}")
        
        if not self.args.no_wandb:
            wandb.log({
                "val/loss": avg_val_loss,
                "val/perplexity": np.exp(avg_val_loss),
                "global_step": global_step
            })
        
        return avg_val_loss
    
    def test_model(self):
        """Final test evaluation."""
        logger.info("\nRunning final test evaluation...")
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True,
            collate_fn=self.data_collator
        )
        
        test_loss = self.validate(test_loader, -1)
        
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Perplexity: {np.exp(test_loss):.2f}")
        
        if not self.args.no_wandb:
            wandb.log({
                "test/loss": test_loss,
                "test/perplexity": np.exp(test_loss)
            })
    
    def log_training_metrics(self, loss, global_step, epoch, learning_rate):
        """Log detailed training metrics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        metrics = {
            "train/loss": loss,
            "train/perplexity": np.exp(loss),
            "train/learning_rate": learning_rate,
            "train/epoch": epoch,
            "train/global_step": global_step,
            "system/memory_mb": memory_info.rss / 1024 / 1024,
            "system/memory_percent": process.memory_percent()
        }
        
        if not self.args.no_wandb:
            wandb.log(metrics)
    
    def save_checkpoint(self, global_step, val_loss, final=False):
        """Save model checkpoint."""
        if final:
            save_dir = os.path.join(self.args.output_dir, "final_model")
        else:
            save_dir = os.path.join(self.args.output_dir, f"checkpoint-{global_step}")
        
        logger.info(f"Saving checkpoint to {save_dir}")
        
        # Save model
        self.model.save_pretrained(save_dir)
        self.config.save_pretrained(save_dir)
        self.tokenizer_wrapper.save_pretrained(save_dir)
        
        # Save training state
        torch.save({
            'global_step': global_step,
            'val_loss': val_loss,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, os.path.join(save_dir, 'training_state.pt'))
        
        logger.info(f"Checkpoint saved with val_loss: {val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Production training for RRWebBERT")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory with RRWEB JSON files')
    parser.add_argument('--tokenizer_path', type=str, required=True, help='Path to trained tokenizer')
    parser.add_argument('--output_dir', type=str, default='./rrweb-bert-production', help='Output directory')
    
    # Model arguments
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--max_length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--mlm_probability', type=float, default=0.15, help='Masking probability')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per device')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Gradient accumulation')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm')
    
    # Checkpointing
    parser.add_argument('--logging_steps', type=int, default=100, help='Log every N steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluate every N steps')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save every N steps')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint')
    
    # Data loading
    parser.add_argument('--max_files', type=int, default=None, help='Max files to use')
    parser.add_argument('--cache_size', type=int, default=100, help='Dataset cache size')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    # Monitoring
    parser.add_argument('--wandb_project', type=str, default='rrweb-bert-production')
    parser.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--test_collator', action='store_true', help='Test collator statistics')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = ProductionTrainer(args)
    
    # Start training - NO TIMEOUTS
    trainer.train()
    
    # Finish wandb
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()