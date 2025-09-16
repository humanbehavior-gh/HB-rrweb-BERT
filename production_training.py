#!/usr/bin/env python3
"""
Production Training Script for RRWEB BERT Model
Single GPU configuration with full dataset
"""

import os
import sys
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from datetime import datetime
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import wandb
from tqdm import tqdm
import psutil

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer_wrapper import TokenizerWrapper
from src.dataset import RRWebLazyDataset
from src.collator import ImprovedDataCollatorForMLM
from transformers import BertConfig, BertForMaskedLM
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


class RRWebBERT(nn.Module):
    """Production-ready RRWEB BERT model with enhanced architecture"""
    
    def __init__(self, vocab_size: int, config: Optional[BertConfig] = None):
        super().__init__()
        
        if config is None:
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=2048,
                type_vocab_size=2
            )
        
        self.bert = BertForMaskedLM(config)
        
    def forward(self, input_ids, attention_mask=None, labels=None, token_type_ids=None, event_type_ids=None):
        # Standard BERT uses token_type_ids, not event_type_ids
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )


class ProductionTrainingPipeline:
    """Production training pipeline for RRWEB BERT"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_gpus = torch.cuda.device_count()
        
        # Components to be initialized
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Setup directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Initializing Production Training Pipeline...")
        print(f"Device: {self.device}")
        print(f"Number of GPUs: {self.num_gpus}")
    
    def setup(self):
        """Setup all training components"""
        print("\n" + "="*80)
        print("SETTING UP PRODUCTION TRAINING PIPELINE")
        print("="*80)
        
        # 1. Load tokenizer
        print("\n1. Loading tokenizer...")
        tokenizer_path = "/home/ubuntu/rrweb_tokenizer/tokenizer_model_20250911_234222"
        self.tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
        print(f"   Vocabulary size: {self.tokenizer.vocab_size}")
        
        # 2. Create model
        print("\n2. Creating model...")
        self.model = RRWebBERT(vocab_size=self.tokenizer.vocab_size)
        self.model = self.model.to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # 3. Setup optimizer
        print("\n3. Setting up optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # 4. Setup scheduler
        print("\n4. Setting up learning rate scheduler...")
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['T_0'],
            T_mult=self.config['T_mult'],
            eta_min=self.config['min_lr']
        )
        
        # 5. Setup mixed precision
        print("\n5. Setting up mixed precision training...")
        self.scaler = GradScaler('cuda')
        
        # 6. Setup data loaders
        print("\n6. Setting up data loaders...")
        
        # First, get all files and split them
        import random
        from pathlib import Path
        
        data_dir = "/home/ubuntu/embeddingV2/rrweb_data"
        all_files = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    if os.path.getsize(file_path) > 100:  # Skip tiny files
                        all_files.append(file_path)
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(all_files)
        
        # Split into train and val
        total_files = min(len(all_files), self.config['train_files'] + self.config['val_files'])
        all_files = all_files[:total_files]
        
        train_split = self.config['train_files']
        train_files = all_files[:train_split]
        val_files = all_files[train_split:]
        
        print(f"   Total files found: {len(all_files)}")
        print(f"   Train files: {len(train_files)}")
        print(f"   Val files: {len(val_files)}")
        
        # Training dataset with random window sampling
        train_dataset = RRWebLazyDataset(
            data_dir="/home/ubuntu/embeddingV2/rrweb_data",
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            cache_size=self.config['cache_size'],
            max_files=self.config['train_files'],
            use_random_window=True,  # Enable random window sampling
            samples_per_epoch=self.config.get('samples_per_epoch', 90000),  # Default 90k samples/epoch
            seed=42  # Same seed for consistency
        )
        
        # Validation dataset - no random window for consistent validation
        val_dataset = RRWebLazyDataset(
            data_dir="/home/ubuntu/embeddingV2/rrweb_data",
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            cache_size=self.config['cache_size'],
            max_files=self.config['val_files'],
            use_random_window=False,  # No random window for validation
            seed=43  # Different seed to get different files
        )
        
        # Data collator
        collator = ImprovedDataCollatorForMLM(
            tokenizer=self.tokenizer,
            mlm_probability=self.config['mlm_probability']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collator,
            num_workers=self.config['num_workers'],
            pin_memory=True
        )
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
        
        print("\n✅ Production pipeline setup complete!")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to GPU
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch_size = batch['input_ids'].size(0)
            
            # Remove event_type_ids if present
            if 'event_type_ids' in batch:
                del batch['event_type_ids']
            
            # Forward pass with mixed precision
            with autocast('cuda'):
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Handle DataParallel loss
                if loss.dim() > 0:
                    loss = loss.mean()
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step(epoch + batch_idx / len(self.train_loader))
            
            # Track metrics
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            
            # Update progress bar
            current_loss = epoch_loss / epoch_samples
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'lr': f'{current_lr:.2e}',
                'samples/s': f'{batch_size / progress_bar.format_dict["elapsed"]:.1f}'
            })
            
            # Log to W&B
            if self.config['use_wandb'] and batch_idx % self.config['log_interval'] == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/learning_rate': current_lr,
                    'train/epoch': epoch,
                    'train/step': epoch * len(self.train_loader) + batch_idx
                })
        
        return epoch_loss / epoch_samples
    
    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].size(0)
                
                # Remove event_type_ids if present
                if 'event_type_ids' in batch:
                    del batch['event_type_ids']
                
                with autocast('cuda'):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    if loss.dim() > 0:
                        loss = loss.mean()
                
                val_loss += loss.item() * batch_size
                val_samples += batch_size
        
        return val_loss / val_samples
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'training_history': self.training_history
        }
        
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"   Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if metrics['val_loss'] == min(self.training_history['val_loss']):
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"   Best model updated: {best_path}")
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING PRODUCTION TRAINING")
        print("="*80)
        
        # Initialize W&B if configured
        if self.config['use_wandb']:
            wandb.init(
                project="rrweb-production-training",
                name=f"production_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n" + "="*80)
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print("="*80)
            
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate()
            
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_loss:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   Epoch Time: {epoch_time:.1f}s")
            
            # Log to W&B
            if self.config['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_loss,
                    'val/epoch_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch_time': epoch_time
                })
            
            # Save checkpoint
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'epoch': epoch + 1
            }
            self.save_checkpoint(epoch, metrics)
            
            # Track best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"   🎉 New best validation loss: {val_loss:.4f}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETED")
        print("="*80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final summary
        self._save_training_summary()
        
        if self.config['use_wandb']:
            wandb.finish()
    
    def _save_training_summary(self):
        """Save training summary to JSON"""
        summary = {
            'config': self.config,
            'training_history': self.training_history,
            'final_metrics': {
                'final_train_loss': self.training_history['train_loss'][-1],
                'final_val_loss': self.training_history['val_loss'][-1],
                'best_val_loss': min(self.training_history['val_loss']),
                'total_epochs': len(self.training_history['train_loss']),
                'total_time': sum(self.training_history['epoch_time'])
            }
        }
        
        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        print(f"Training summary saved: {summary_path}")


def main():
    """Main production training function"""
    
    # Production configuration
    config = {
        'batch_size': 32,  # Reduced to avoid OOM on H100 GPU
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_length': 2048,  # Full sequence length
        'mlm_probability': 0.15,
        'num_workers': 4,
        'cache_size': 2000,  # Increased for better performance
        'max_grad_norm': 1.0,
        'log_interval': 50,
        'checkpoint_dir': './checkpoints_production',
        'use_wandb': True,
        # Dataset sizes
        'train_files': 28750,  # 90% of 31951 files
        'val_files': 3195,     # 10% of 31951 files
        'samples_per_epoch': 90000,  # Number of random windows per epoch
        # Scheduler params
        'T_0': 10,
        'T_mult': 2,
        'min_lr': 1e-6
    }
    
    print("="*80)
    print("RRWEB BERT PRODUCTION TRAINING")
    print("="*80)
    print(f"Training configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize pipeline
    pipeline = ProductionTrainingPipeline(config)
    
    # Setup components
    pipeline.setup()
    
    # Run training
    num_epochs = 10  # Can be adjusted based on convergence
    pipeline.train(num_epochs=num_epochs)
    
    print("\n✅ Production training completed successfully!")


if __name__ == "__main__":
    main()