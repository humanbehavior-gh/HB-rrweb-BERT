#!/usr/bin/env python3
"""
Test 1.7: End-to-End Training Pipeline Test
Production-ready training pipeline with multi-GPU support, checkpointing, and monitoring
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import DataParallel
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
                max_position_embeddings=8192,
                type_vocab_size=2,
                layer_norm_eps=1e-12,
            )
        
        self.bert = BertForMaskedLM(config)
        self.vocab_size = vocab_size
        
        # Custom embeddings for RRWEB event types
        self.event_type_embeddings = nn.Embedding(100, config.hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        self.event_type_embeddings.weight.data.normal_(mean=0.0, std=0.02)
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with enhanced processing"""
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs


class TrainingPipeline:
    """Production-ready training pipeline with comprehensive monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.multi_gpu = torch.cuda.device_count() > 1
        self.num_gpus = torch.cuda.device_count()
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        
        # Metrics tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': [],
            'gpu_memory': [],
            'throughput': []
        }
        
        # Checkpoint management
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"Initializing Training Pipeline...")
        print(f"Device: {self.device}")
        print(f"Multi-GPU: {self.multi_gpu} ({self.num_gpus} GPUs)")
    
    def setup(self):
        """Setup all training components"""
        print("\n" + "="*80)
        print("SETTING UP TRAINING PIPELINE")
        print("="*80)
        
        # Load tokenizer
        print("\n1. Loading tokenizer...")
        self.tokenizer = TokenizerWrapper.from_pretrained('/home/ubuntu/rrweb_tokenizer/tokenizer_model_20250911_234222')
        print(f"   Vocabulary size: {self.tokenizer.vocab.structural_vocab_size + 12000}")
        
        # Create model
        print("\n2. Creating model...")
        vocab_size = self.tokenizer.vocab.structural_vocab_size + 12000
        self.model = RRWebBERT(vocab_size=vocab_size)
        
        if self.multi_gpu:
            print(f"   Wrapping model with DataParallel for {self.num_gpus} GPUs")
            self.model = DataParallel(self.model)
        
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        
        # Setup optimizer
        print("\n3. Setting up optimizer...")
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler
        print("\n4. Setting up learning rate scheduler...")
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('T_0', 10),
            T_mult=self.config.get('T_mult', 2),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # Setup mixed precision
        print("\n5. Setting up mixed precision training...")
        self.scaler = GradScaler('cuda')
        
        # Setup data loaders
        print("\n6. Setting up data loaders...")
        self._setup_data_loaders()
        
        print("\n✅ Training pipeline setup complete!")
    
    def _setup_data_loaders(self):
        """Setup training and validation data loaders with random window sampling"""
        # Training dataset with random window sampling
        train_dataset = RRWebLazyDataset(
            data_dir="/home/ubuntu/embeddingV2/rrweb_data",
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            cache_size=self.config.get('cache_size', 1000),
            max_files=None,  # Use ALL files for production
            use_random_window=True,  # Enable random window sampling
            samples_per_epoch=self.config.get('samples_per_epoch', 90000)  # Default 90k samples
        )
        
        # Validation dataset (no random window for consistent validation)
        val_dataset = RRWebLazyDataset(
            data_dir="/home/ubuntu/embeddingV2/rrweb_data",
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
            cache_size=100,
            max_files=100,  # Use smaller subset for validation
            use_random_window=False  # No random window for validation
        )
        
        # Data collator
        collator = ImprovedDataCollatorForMLM(
            tokenizer=self.tokenizer,
            mlm_probability=self.config.get('mlm_probability', 0.15)
        )
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=collator,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"   Training samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Training batches: {len(self.train_loader)}")
        print(f"   Validation batches: {len(self.val_loader)}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_samples = 0
        batch_times = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} Training")
        
        print(f"\n[Training] Starting epoch {epoch+1}, {len(self.train_loader)} batches to process...")
        for batch_idx, batch in enumerate(pbar):
            print(f"\n[Training] Processing batch {batch_idx+1}/{len(self.train_loader)}...")
            batch_start = time.time()
            
            # Move batch to device
            print(f"  [Batch {batch_idx+1}] Moving data to GPU...")
            move_start = time.time()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            batch_size = batch['input_ids'].size(0)
            print(f"  [Batch {batch_idx+1}] Data moved to GPU in {time.time() - move_start:.2f}s (batch_size={batch_size})")
            
            # Mixed precision forward pass
            print(f"  [Batch {batch_idx+1}] Starting forward pass...")
            forward_start = time.time()
            with autocast('cuda'):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss
            print(f"  [Batch {batch_idx+1}] Forward pass completed in {time.time() - forward_start:.2f}s")
            
            # Handle DataParallel loss aggregation
            if loss.dim() > 0:
                loss = loss.mean()
            
            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item() * batch_size
            epoch_samples += batch_size
            batch_times.append(time.time() - batch_start)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
                'samples/s': f"{batch_size / batch_times[-1]:.1f}"
            })
            
            # Periodic logging
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self._log_training_step(epoch, batch_idx, loss.item())
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / epoch_samples
        avg_batch_time = np.mean(batch_times)
        throughput = self.config['batch_size'] / avg_batch_time
        
        return {
            'loss': avg_loss,
            'batch_time': avg_batch_time,
            'throughput': throughput
        }
    
    def validate(self) -> float:
        """Run validation"""
        self.model.eval()
        val_loss = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].size(0)
                
                with autocast('cuda'):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels']
                    )
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
            'model_state_dict': self.model.module.state_dict() if self.multi_gpu else self.model.state_dict(),
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
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        
        if self.multi_gpu:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.training_history = checkpoint['training_history']
        
        print(f"Checkpoint loaded from epoch {checkpoint['epoch'] + 1}")
        return checkpoint['epoch']
    
    def _log_training_step(self, epoch: int, batch_idx: int, loss: float):
        """Log training step metrics"""
        if wandb.run:
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                'train/epoch': epoch,
                'train/step': epoch * len(self.train_loader) + batch_idx
            })
    
    def _get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage"""
        if not torch.cuda.is_available():
            return {}
        
        memory_stats = {}
        for i in range(self.num_gpus):
            memory_stats[f'gpu_{i}_memory_mb'] = torch.cuda.memory_allocated(i) / 1024 / 1024
            memory_stats[f'gpu_{i}_memory_percent'] = (
                torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                if torch.cuda.max_memory_allocated(i) > 0 else 0
            )
        
        return memory_stats
    
    def train(self, num_epochs: int):
        """Main training loop"""
        print("\n" + "="*80)
        print("STARTING TRAINING")
        print("="*80)
        
        # Initialize W&B if configured
        if self.config.get('use_wandb', True):
            wandb.init(
                project="rrweb-training-pipeline",
                name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config
            )
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n{'='*80}")
            print(f"EPOCH {epoch+1}/{num_epochs}")
            print(f"{'='*80}")
            
            epoch_start = time.time()
            
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Validation
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Get GPU memory stats
            gpu_memory = self._get_gpu_memory()
            
            # Update history
            self.training_history['train_loss'].append(train_metrics['loss'])
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.training_history['epoch_time'].append(epoch_time)
            self.training_history['throughput'].append(train_metrics['throughput'])
            self.training_history['gpu_memory'].append(gpu_memory)
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch+1} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_loss:.4f}")
            print(f"   Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"   Throughput: {train_metrics['throughput']:.1f} samples/sec")
            print(f"   Epoch Time: {epoch_time:.1f}s")
            
            if gpu_memory:
                for key, value in gpu_memory.items():
                    if 'percent' in key:
                        print(f"   {key}: {value:.1f}%")
            
            # Log to W&B
            if wandb.run:
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_loss,
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'throughput': train_metrics['throughput'],
                    'epoch_time': epoch_time,
                    **gpu_memory
                })
            
            # Save checkpoint
            checkpoint_metrics = {
                'train_loss': train_metrics['loss'],
                'val_loss': val_loss,
                'best_val_loss': min(best_val_loss, val_loss)
            }
            self.save_checkpoint(epoch, checkpoint_metrics)
            
            # Update best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"   🏆 New best validation loss: {best_val_loss:.4f}")
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Save final metrics
        self._save_training_summary()
        
        if wandb.run:
            wandb.finish()
    
    def _save_training_summary(self):
        """Save training summary to JSON"""
        summary = {
            'config': self.config,
            'final_metrics': {
                'final_train_loss': self.training_history['train_loss'][-1],
                'final_val_loss': self.training_history['val_loss'][-1],
                'best_val_loss': min(self.training_history['val_loss']),
                'total_epochs': len(self.training_history['train_loss']),
                'avg_epoch_time': np.mean(self.training_history['epoch_time']),
                'avg_throughput': np.mean(self.training_history['throughput'])
            },
            'training_history': self.training_history
        }
        
        summary_path = self.checkpoint_dir / "training_summary.json"
        with open(summary_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                elif isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                return obj
            
            json.dump(summary, f, indent=2, default=convert)
        
        print(f"Training summary saved: {summary_path}")


def test_training_pipeline():
    """Test the complete training pipeline"""
    
    # Training configuration  
    config = {
        'batch_size': 8,  # Small batch size as requested
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'max_length': 2048,  # Full production sequence length
        'mlm_probability': 0.15,
        'num_workers': 4,
        'cache_size': 2000,  # Larger cache for production
        'samples_per_epoch': 1000,  # Number of random windows per epoch for testing
        'max_grad_norm': 1.0,
        'log_interval': 10,
        'checkpoint_dir': './checkpoints_test',
        'use_wandb': True,
        # Scheduler params
        'T_0': 10,
        'T_mult': 2,
        'min_lr': 1e-6
    }
    
    print("="*80)
    print("END-TO-END TRAINING PIPELINE TEST")
    print("="*80)
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Setup components
    pipeline.setup()
    
    # Run training for 2 epochs (production would use more)
    pipeline.train(num_epochs=2)
    
    # Test checkpoint loading
    print("\n" + "="*80)
    print("TESTING CHECKPOINT LOADING")
    print("="*80)
    
    checkpoint_path = Path(config['checkpoint_dir']) / "checkpoint_epoch_1.pt"
    if checkpoint_path.exists():
        pipeline.load_checkpoint(str(checkpoint_path))
        print("✅ Checkpoint loading successful")
    
    # Test inference
    print("\n" + "="*80)
    print("TESTING INFERENCE")
    print("="*80)
    
    pipeline.model.eval()
    with torch.no_grad():
        # Create dummy input
        dummy_input = torch.randint(0, 12520, (1, 256)).to(pipeline.device)
        dummy_mask = torch.ones_like(dummy_input)
        
        with autocast('cuda'):
            outputs = pipeline.model(
                input_ids=dummy_input,
                attention_mask=dummy_mask
            )
        
        print(f"Inference output shape: {outputs.logits.shape}")
        print("✅ Inference test successful")
    
    print("\n" + "="*80)
    print("ALL PIPELINE TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    test_training_pipeline()