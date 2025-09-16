#!/usr/bin/env python3
"""
Test 1.8: Minimal Forward Pass Test
Using EXACT same setup as Test 1.7 but with minimal data to test forward pass
"""

import os
import sys
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ubuntu/rrweb_tokenizer')

from src.tokenizer_wrapper import TokenizerWrapper
from src.dataset import RRWebLazyDataset
from src.collator import ImprovedDataCollatorForMLM
from transformers import BertForMaskedLM, BertConfig

def test_minimal_forward():
    print("=" * 80, flush=True)
    print("MINIMAL FORWARD PASS TEST - EXACT PRODUCTION SETUP", flush=True)
    print("Using same model, tokenizer, collator, dataset as Test 1.7", flush=True)
    print("Only difference: minimal data (1 file, batch size 1)", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}", flush=True)
    print(f"GPU count: {torch.cuda.device_count()}", flush=True)
    
    # 1. EXACT SAME TOKENIZER AS TEST 1.7
    print("\n1. Loading production tokenizer...", flush=True)
    tokenizer_path = "/home/ubuntu/rrweb_tokenizer/tokenizer_model_20250911_234222"
    tokenizer = TokenizerWrapper.from_pretrained(tokenizer_path)
    print(f"   Vocabulary size: {tokenizer.vocab_size}", flush=True)
    
    # 2. EXACT SAME MODEL CONFIG AS TEST 1.7
    print("\n2. Creating model with production config...", flush=True)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        type_vocab_size=2
    )
    
    model = BertForMaskedLM(config)
    model = model.to(device)
    
    # EXACT SAME MULTI-GPU SETUP AS TEST 1.7
    if torch.cuda.device_count() > 1:
        print(f"   Wrapping model with DataParallel for {torch.cuda.device_count()} GPUs", flush=True)
        model = DataParallel(model)
    
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    
    # 3. EXACT SAME OPTIMIZER AS TEST 1.7
    print("\n3. Setting up optimizer...", flush=True)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 4. EXACT SAME MIXED PRECISION AS TEST 1.7
    print("\n4. Setting up mixed precision training...", flush=True)
    scaler = GradScaler('cuda')
    
    # 5. MINIMAL DATASET - only 1 file instead of 1000
    print("\n5. Setting up minimal dataset (1 file only)...", flush=True)
    train_dataset = RRWebLazyDataset(
        data_dir="/home/ubuntu/embeddingV2/rrweb_data",
        tokenizer=tokenizer,
        max_length=2048,  # SAME AS TEST 1.7
        cache_size=100,   # SAME AS TEST 1.7
        max_files=1       # ONLY DIFFERENCE: 1 file instead of 1000
    )
    
    # 6. EXACT SAME COLLATOR AS TEST 1.7
    collator = ImprovedDataCollatorForMLM(
        tokenizer=tokenizer,
        mlm_probability=0.15
    )
    
    # 7. MINIMAL DATALOADER - batch size 1
    print("\n6. Creating dataloader with batch size 1...", flush=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,  # Minimal batch size
        shuffle=False,
        collate_fn=collator,
        num_workers=0
    )
    
    print(f"   Dataset files: {len(train_dataset.file_paths)}", flush=True)
    print(f"   Batch size: 1", flush=True)
    print(f"   Total batches: {len(train_loader)}", flush=True)
    
    # 8. TEST SINGLE FORWARD PASS
    print("\n" + "=" * 80, flush=True)
    print("TESTING FORWARD PASS", flush=True)
    print("=" * 80, flush=True)
    
    model.train()
    
    print("\nLoading first batch...", flush=True)
    start_load = time.time()
    batch = next(iter(train_loader))
    load_time = time.time() - start_load
    print(f"   Batch loaded in {load_time:.2f}s", flush=True)
    
    print("\nBatch info:", flush=True)
    print(f"   Input shape: {batch['input_ids'].shape}", flush=True)
    print(f"   Attention mask shape: {batch['attention_mask'].shape}", flush=True)
    print(f"   Labels shape: {batch['labels'].shape}", flush=True)
    
    print("\nMoving batch to GPU...", flush=True)
    start_move = time.time()
    batch = {k: v.to(device) for k, v in batch.items()}
    move_time = time.time() - start_move
    print(f"   Data moved to GPU in {move_time:.2f}s", flush=True)
    
    # Remove event_type_ids as standard BERT doesn't expect it
    if 'event_type_ids' in batch:
        del batch['event_type_ids']
    
    print("\nRunning forward pass with mixed precision...", flush=True)
    print(f"   Starting forward pass at {time.strftime('%H:%M:%S')}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    torch.cuda.synchronize()
    start_forward = time.time()
    
    with autocast('cuda'):
        outputs = model(**batch)
        loss = outputs.loss
    
    torch.cuda.synchronize()
    forward_time = time.time() - start_forward
    print(f"   Forward pass completed in {forward_time:.2f}s", flush=True)
    print(f"   Loss: {loss.item():.4f}", flush=True)
    
    print("\nRunning backward pass...")
    torch.cuda.synchronize()
    start_backward = time.time()
    
    optimizer.zero_grad()
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    torch.cuda.synchronize()
    backward_time = time.time() - start_backward
    print(f"   Backward pass completed in {backward_time:.2f}s")
    
    print("\nOptimizer step...")
    start_step = time.time()
    scaler.step(optimizer)
    scaler.update()
    step_time = time.time() - start_step
    print(f"   Optimizer step completed in {step_time:.2f}s")
    
    print("\n" + "=" * 80)
    print("TIMING SUMMARY")
    print("=" * 80)
    print(f"   Data loading: {load_time:.2f}s")
    print(f"   Move to GPU: {move_time:.2f}s")
    print(f"   Forward pass: {forward_time:.2f}s")
    print(f"   Backward pass: {backward_time:.2f}s")
    print(f"   Optimizer step: {step_time:.2f}s")
    print(f"   TOTAL: {load_time + move_time + forward_time + backward_time + step_time:.2f}s")
    
    # Test with different batch sizes
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT BATCH SIZES")
    print("=" * 80)
    
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    for bs in batch_sizes:
        print(f"\nTesting batch size {bs}...")
        
        # Create new dataset with more files if needed
        train_dataset = RRWebLazyDataset(
            data_dir="/home/ubuntu/embeddingV2/rrweb_data",
            tokenizer=tokenizer,
            max_length=2048,
            cache_size=100,
            max_files=bs  # Load enough files for the batch
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=False,
            collate_fn=collator,
            num_workers=0
        )
        
        try:
            batch = next(iter(train_loader))
            print(f"   Batch loaded, moving to GPU...", flush=True)
            
            batch = {k: v.to(device) for k, v in batch.items()}
            print(f"   Batch on GPU. Shape: {batch['input_ids'].shape}", flush=True)
            
            # Remove event_type_ids as standard BERT doesn't expect it
            if 'event_type_ids' in batch:
                del batch['event_type_ids']
            
            print(f"   About to start forward pass...", flush=True)
            torch.cuda.synchronize()
            start = time.time()
            
            print(f"   Entering autocast context...", flush=True)
            with autocast('cuda'):
                print(f"   Calling model forward (DataParallel={isinstance(model, DataParallel)})...", flush=True)
                print(f"   Input device: {batch['input_ids'].device}", flush=True)
                print(f"   Model device: {next(model.parameters()).device if not isinstance(model, DataParallel) else 'DataParallel'}", flush=True)
                
                # Add detailed timing for DataParallel
                if isinstance(model, DataParallel):
                    print(f"   DataParallel will split batch across {len(model.device_ids)} GPUs", flush=True)
                    print(f"   Device IDs: {model.device_ids}", flush=True)
                    print(f"   Batch size {bs} will be split as: {[bs // len(model.device_ids)] * len(model.device_ids)}", flush=True)
                
                outputs = model(**batch)
                print(f"   Model forward completed!", flush=True)
                loss = outputs.loss
                if hasattr(loss, 'item'):
                    print(f"   Loss extracted: {loss.item():.4f}", flush=True)
                else:
                    print(f"   Loss extracted: {loss.mean().item():.4f}", flush=True)
            
            torch.cuda.synchronize()
            forward_time = time.time() - start
            print(f"   Forward pass finished in {forward_time:.3f}s", flush=True)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            
            loss_value = loss.item() if hasattr(loss, 'item') else loss.mean().item()
            print(f"   Batch size {bs}: {forward_time:.3f}s, Memory: {memory_used:.1f}GB, Loss: {loss_value:.4f}")
            
            # Clear memory
            del batch, outputs, loss
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"   Batch size {bs}: OOM")
                torch.cuda.empty_cache()
                break
            else:
                print(f"   Batch size {bs}: Error - {e}")
    
    print("\n✅ Minimal forward pass test completed successfully!")
    return True

if __name__ == "__main__":
    test_minimal_forward()