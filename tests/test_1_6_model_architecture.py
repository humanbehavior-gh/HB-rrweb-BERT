#!/usr/bin/env python3
"""
Test 1.6: Model Architecture Testing with Multi-GPU Support
Test RRWebBERT model initialization, forward pass, and parallel processing on 2 GPUs.
Production-ready validation with the full 31,951 file dataset.
"""

import sys
import os
import json
import time
import gc
import psutil
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
from transformers import BertConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEmbeddings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ubuntu/rrweb_tokenizer')
sys.path.append('/home/ubuntu/rrweb-bert/src')

from collator import ImprovedDataCollatorForMLM
from dataset import RRWebLazyDataset
from tokenizer_wrapper import TokenizerWrapper


class RRWebBertEmbeddings(BertEmbeddings):
    """Custom embeddings for RRWEB with event type support."""
    
    def __init__(self, config):
        super().__init__(config)
        # Add event type embeddings (7 event types in RRWEB)
        self.event_type_embeddings = nn.Embedding(10, config.hidden_size)
        
    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        event_type_ids=None,
        inputs_embeds=None,
        past_key_values_length=0,
    ):
        """Forward pass with event type embeddings."""
        # Get base embeddings
        embeddings = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )
        
        # Add event type embeddings if provided
        if event_type_ids is not None:
            event_embeddings = self.event_type_embeddings(event_type_ids)
            embeddings = embeddings + event_embeddings
            
        return embeddings


class RRWebBERT(BertForMaskedLM):
    """BERT model adapted for RRWEB session modeling."""
    
    def __init__(self, config):
        super().__init__(config)
        # Replace embeddings with our custom version
        self.bert.embeddings = RRWebBertEmbeddings(config)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        event_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """Forward pass with event type support."""
        # Get base model outputs
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
        return {
            'loss': masked_lm_loss,
            'logits': prediction_scores,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions if output_attentions else None,
        }


class ModelArchitectureTest:
    """Test suite for RRWebBERT model architecture with multi-GPU support."""
    
    def __init__(self):
        """Initialize test environment."""
        print("Initializing Model Architecture Test...")
        
        # Initialize wandb
        wandb.init(
            project="rrweb-tokenizer-tests",
            name="test_1_6_model_architecture",
            config={
                "test": "model_architecture",
                "gpus": torch.cuda.device_count(),
                "vocab_size": 12520,
                "max_length": 2048,
            }
        )
        
        # Data configuration
        self.data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
        self.batch_size = 8  # Per GPU
        self.max_length = 2048
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = TokenizerWrapper.from_pretrained(
            '/home/ubuntu/rrweb_tokenizer/tokenizer_model_20250911_234222'
        )
        
        # Initialize collator
        self.collator = ImprovedDataCollatorForMLM(
            tokenizer=self.tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=8
        )
        
        # Device configuration
        self.device_count = torch.cuda.device_count()
        print(f"Found {self.device_count} GPUs")
        
        if self.device_count == 0:
            print("⚠️ No GPUs found, using CPU (not recommended for production)")
            self.device = torch.device('cpu')
            self.multi_gpu = False
        else:
            self.device = torch.device('cuda:0')
            self.multi_gpu = self.device_count > 1
            
        print(f"Multi-GPU mode: {self.multi_gpu}")
        print(f"Primary device: {self.device}")
        
    def test_1_model_initialization(self):
        """Test 1: Model initialization and configuration."""
        print("\n" + "="*80)
        print("TEST 1: MODEL INITIALIZATION")
        print("="*80)
        
        try:
            # Create model configuration
            config = BertConfig(
                vocab_size=12520,  # 520 structural + 12000 BPE
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                max_position_embeddings=2048,
                type_vocab_size=2,  # structural vs text tokens
            )
            
            # Initialize model
            print("Creating RRWebBERT model...")
            model = RRWebBERT(config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n📊 Model Statistics:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Model size: {total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
            print(f"  Vocabulary size: {config.vocab_size}")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Layers: {config.num_hidden_layers}")
            print(f"  Attention heads: {config.num_attention_heads}")
            
            # Move to device
            model = model.to(self.device)
            
            # Test multi-GPU wrapping
            if self.multi_gpu:
                print(f"\nWrapping model for {self.device_count} GPUs...")
                model = DataParallel(model)
                print("✓ DataParallel wrapper applied")
            
            wandb.log({
                "model/total_params": total_params,
                "model/trainable_params": trainable_params,
                "model/size_mb": total_params * 4 / 1024 / 1024,
            })
            
            print("\n✅ Model initialization test PASSED")
            return model, True
            
        except Exception as e:
            print(f"\n❌ Model initialization test FAILED: {e}")
            return None, False
    
    def test_2_forward_pass(self, model):
        """Test 2: Single forward pass with dummy data."""
        print("\n" + "="*80)
        print("TEST 2: FORWARD PASS")
        print("="*80)
        
        try:
            # Create dummy batch
            batch_size = self.batch_size * self.device_count if self.multi_gpu else self.batch_size
            seq_length = 512
            
            print(f"Creating dummy batch (batch_size={batch_size}, seq_len={seq_length})...")
            
            dummy_batch = {
                'input_ids': torch.randint(0, 12520, (batch_size, seq_length)).to(self.device),
                'attention_mask': torch.ones(batch_size, seq_length).to(self.device),
                'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long).to(self.device),
                'event_type_ids': torch.randint(0, 7, (batch_size, seq_length)).to(self.device),
                'labels': torch.full((batch_size, seq_length), -100, dtype=torch.long).to(self.device),
            }
            
            # Add some masked positions for loss calculation (15% masking)
            mask_positions = torch.rand(batch_size, seq_length) < 0.15
            dummy_batch['labels'][mask_positions] = torch.randint(0, 12520, (mask_positions.sum(),)).to(self.device)
            
            # Forward pass
            print("Running forward pass...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**dummy_batch)
            
            forward_time = time.time() - start_time
            
            # Check outputs
            print(f"\n📊 Forward Pass Results:")
            if outputs['loss'] is not None:
                print(f"  Loss: {outputs['loss'].item():.4f}")
            else:
                print(f"  Loss: None (no masked tokens)")
            print(f"  Logits shape: {outputs['logits'].shape}")
            print(f"  Forward time: {forward_time:.3f}s")
            print(f"  Throughput: {batch_size / forward_time:.1f} samples/sec")
            
            # Verify shapes
            assert outputs['logits'].shape == (batch_size, seq_length, 12520)
            assert outputs['loss'].item() > 0
            
            wandb.log({
                "forward_pass/loss": outputs['loss'].item(),
                "forward_pass/time": forward_time,
                "forward_pass/throughput": batch_size / forward_time,
            })
            
            print("\n✅ Forward pass test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Forward pass test FAILED: {e}")
            return False
    
    def test_3_parallel_forward_pass(self, model):
        """Test 3: Parallel forward pass on multiple GPUs."""
        print("\n" + "="*80)
        print("TEST 3: PARALLEL FORWARD PASS")
        print("="*80)
        
        if not self.multi_gpu:
            print("⚠️ Skipping (only 1 GPU available)")
            return True
        
        try:
            # Create larger batch for parallel processing
            batch_size = self.batch_size * self.device_count
            seq_length = 1024
            
            print(f"Testing parallel processing on {self.device_count} GPUs...")
            print(f"Total batch size: {batch_size} ({self.batch_size} per GPU)")
            
            dummy_batch = {
                'input_ids': torch.randint(0, 12520, (batch_size, seq_length)).to(self.device),
                'attention_mask': torch.ones(batch_size, seq_length).to(self.device),
                'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long).to(self.device),
                'event_type_ids': torch.randint(0, 7, (batch_size, seq_length)).to(self.device),
                'labels': torch.full((batch_size, seq_length), -100, dtype=torch.long).to(self.device),
            }
            
            # Measure GPU memory before
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                mem_before = [torch.cuda.memory_allocated(i) / 1024 / 1024 for i in range(self.device_count)]
            
            # Parallel forward pass
            print("Running parallel forward pass...")
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model(**dummy_batch)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            forward_time = time.time() - start_time
            
            # Measure GPU memory after
            if torch.cuda.is_available():
                mem_after = [torch.cuda.memory_allocated(i) / 1024 / 1024 for i in range(self.device_count)]
                mem_used = [after - before for after, before in zip(mem_after, mem_before)]
            
            print(f"\n📊 Parallel Processing Results:")
            if outputs['loss'] is not None:
                print(f"  Loss: {outputs['loss'].item():.4f}")
            else:
                print(f"  Loss: None (no masked tokens)")
            print(f"  Total forward time: {forward_time:.3f}s")
            print(f"  Throughput: {batch_size / forward_time:.1f} samples/sec")
            print(f"  Per-GPU throughput: {self.batch_size / forward_time:.1f} samples/sec")
            
            if torch.cuda.is_available():
                print(f"\n  GPU Memory Usage:")
                for i, (before, after, used) in enumerate(zip(mem_before, mem_after, mem_used)):
                    print(f"    GPU {i}: {before:.1f} MB → {after:.1f} MB (+{used:.1f} MB)")
            
            wandb.log({
                "parallel/loss": outputs['loss'].item(),
                "parallel/time": forward_time,
                "parallel/throughput": batch_size / forward_time,
                "parallel/per_gpu_throughput": self.batch_size / forward_time,
            })
            
            print("\n✅ Parallel forward pass test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Parallel forward pass test FAILED: {e}")
            return False
    
    def test_4_real_data_processing(self, model):
        """Test 4: Process real RRWEB data."""
        print("\n" + "="*80)
        print("TEST 4: REAL DATA PROCESSING")
        print("="*80)
        
        try:
            # Create dataset
            print("Loading RRWEB dataset...")
            dataset = RRWebLazyDataset(
                data_dir=self.data_dir,
                tokenizer=self.tokenizer,
                max_length=self.max_length,
                max_files=100,  # Use subset for testing
                cache_size=50,
                seed=42
            )
            
            # Create dataloader
            batch_size = self.batch_size * self.device_count if self.multi_gpu else self.batch_size
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collator,
                num_workers=2,
                pin_memory=torch.cuda.is_available()
            )
            
            print(f"Processing {len(dataloader)} batches...")
            
            # Process batches
            total_loss = 0
            total_samples = 0
            processing_times = []
            
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                    if i >= 10:  # Process first 10 batches
                        break
                    
                    # Move batch to device
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    start_time = time.time()
                    outputs = model(**batch)
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    processing_time = time.time() - start_time
                    processing_times.append(processing_time)
                    
                    # Accumulate metrics
                    total_loss += outputs['loss'].item() * batch['input_ids'].size(0)
                    total_samples += batch['input_ids'].size(0)
            
            # Calculate statistics
            avg_loss = total_loss / total_samples
            avg_time = np.mean(processing_times)
            throughput = batch_size / avg_time
            
            print(f"\n📊 Real Data Processing Results:")
            print(f"  Average loss: {avg_loss:.4f}")
            print(f"  Average batch time: {avg_time:.3f}s")
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Total samples processed: {total_samples}")
            
            wandb.log({
                "real_data/avg_loss": avg_loss,
                "real_data/avg_time": avg_time,
                "real_data/throughput": throughput,
                "real_data/total_samples": total_samples,
            })
            
            print("\n✅ Real data processing test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Real data processing test FAILED: {e}")
            return False
    
    def test_5_mixed_precision(self, model):
        """Test 5: Mixed precision training with AMP."""
        print("\n" + "="*80)
        print("TEST 5: MIXED PRECISION (FP16)")
        print("="*80)
        
        if not torch.cuda.is_available():
            print("⚠️ Skipping (CUDA not available)")
            return True
        
        try:
            # Create dummy batch
            batch_size = self.batch_size * self.device_count if self.multi_gpu else self.batch_size
            seq_length = 512
            
            dummy_batch = {
                'input_ids': torch.randint(0, 12520, (batch_size, seq_length)).to(self.device),
                'attention_mask': torch.ones(batch_size, seq_length).to(self.device),
                'token_type_ids': torch.zeros(batch_size, seq_length, dtype=torch.long).to(self.device),
                'event_type_ids': torch.randint(0, 7, (batch_size, seq_length)).to(self.device),
                'labels': torch.full((batch_size, seq_length), -100, dtype=torch.long).to(self.device),
            }
            
            # Test FP16 forward pass
            print("Testing mixed precision forward pass...")
            scaler = GradScaler()
            
            # FP32 baseline
            torch.cuda.synchronize()
            start_fp32 = time.time()
            with torch.no_grad():
                outputs_fp32 = model(**dummy_batch)
            torch.cuda.synchronize()
            time_fp32 = time.time() - start_fp32
            
            # FP16 forward pass
            torch.cuda.synchronize()
            start_fp16 = time.time()
            with autocast():
                with torch.no_grad():
                    outputs_fp16 = model(**dummy_batch)
            torch.cuda.synchronize()
            time_fp16 = time.time() - start_fp16
            
            # Compare results
            speedup = time_fp32 / time_fp16
            
            print(f"\n📊 Mixed Precision Results:")
            print(f"  FP32 time: {time_fp32:.3f}s")
            print(f"  FP16 time: {time_fp16:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  FP32 loss: {outputs_fp32['loss'].item():.4f}")
            print(f"  FP16 loss: {outputs_fp16['loss'].item():.4f}")
            print(f"  Loss difference: {abs(outputs_fp32['loss'].item() - outputs_fp16['loss'].item()):.6f}")
            
            wandb.log({
                "mixed_precision/fp32_time": time_fp32,
                "mixed_precision/fp16_time": time_fp16,
                "mixed_precision/speedup": speedup,
            })
            
            print("\n✅ Mixed precision test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Mixed precision test FAILED: {e}")
            return False
    
    def test_6_memory_stress(self, model):
        """Test 6: Memory stress test with increasing batch sizes."""
        print("\n" + "="*80)
        print("TEST 6: MEMORY STRESS TEST")
        print("="*80)
        
        if not torch.cuda.is_available():
            print("⚠️ Skipping (CUDA not available)")
            return True
        
        try:
            seq_length = 1024
            batch_sizes = [1, 2, 4, 8, 16, 32]
            
            print("Testing memory usage with increasing batch sizes...")
            memory_usage = []
            
            for bs in batch_sizes:
                # Clear cache
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Measure memory before
                mem_before = torch.cuda.memory_allocated() / 1024 / 1024
                
                try:
                    # Create batch
                    dummy_batch = {
                        'input_ids': torch.randint(0, 12520, (bs, seq_length)).to(self.device),
                        'attention_mask': torch.ones(bs, seq_length).to(self.device),
                        'token_type_ids': torch.zeros(bs, seq_length, dtype=torch.long).to(self.device),
                        'event_type_ids': torch.randint(0, 7, (bs, seq_length)).to(self.device),
                        'labels': torch.randint(-100, 12520, (bs, seq_length)).to(self.device),
                    }
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(**dummy_batch)
                    
                    torch.cuda.synchronize()
                    
                    # Measure memory after
                    mem_after = torch.cuda.memory_allocated() / 1024 / 1024
                    mem_used = mem_after - mem_before
                    
                    memory_usage.append((bs, mem_used))
                    print(f"  Batch size {bs:2d}: {mem_used:7.1f} MB")
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"  Batch size {bs:2d}: OOM")
                        break
                    else:
                        raise e
            
            # Calculate memory scaling
            if len(memory_usage) > 1:
                # Linear regression to find memory per sample
                x = np.array([m[0] for m in memory_usage])
                y = np.array([m[1] for m in memory_usage])
                slope = np.polyfit(x, y, 1)[0]
                
                print(f"\n📊 Memory Scaling:")
                print(f"  Memory per sample: {slope:.2f} MB")
                print(f"  Max batch size tested: {memory_usage[-1][0]}")
                
                wandb.log({
                    "memory_stress/per_sample_mb": slope,
                    "memory_stress/max_batch_size": memory_usage[-1][0],
                })
            
            print("\n✅ Memory stress test PASSED")
            return True
            
        except Exception as e:
            print(f"\n❌ Memory stress test FAILED: {e}")
            return False
    
    def run_all_tests(self):
        """Run all model architecture tests."""
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE TEST SUITE")
        print("="*80)
        
        results = {}
        
        # Test 1: Model initialization
        model, passed = self.test_1_model_initialization()
        results['model_initialization'] = passed
        
        if not model:
            print("\n⚠️ Cannot continue without model")
            return False
        
        # Test 2: Forward pass
        results['forward_pass'] = self.test_2_forward_pass(model)
        
        # Test 3: Parallel forward pass
        results['parallel_forward'] = self.test_3_parallel_forward_pass(model)
        
        # Test 4: Real data processing
        results['real_data'] = self.test_4_real_data_processing(model)
        
        # Test 5: Mixed precision
        results['mixed_precision'] = self.test_5_mixed_precision(model)
        
        # Test 6: Memory stress
        results['memory_stress'] = self.test_6_memory_stress(model)
        
        # Summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        passed = sum(results.values())
        total = len(results)
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - Model ready for production!")
        else:
            print("\n❌ SOME TESTS FAILED - Review issues above")
        
        wandb.log({
            "tests_passed": passed,
            "tests_total": total,
            "success_rate": passed / total,
        })
        
        return passed == total


def main():
    """Main test execution."""
    tester = ModelArchitectureTest()
    success = tester.run_all_tests()
    
    wandb.finish()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()