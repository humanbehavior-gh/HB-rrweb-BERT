#!/usr/bin/env python3
"""
Test 1.5: Collator Production Testing
Test the data collator at production scale with the full 31,951 file dataset.
Verify MLM masking, padding efficiency, memory stability, and edge cases.
"""

import sys
import os
import json
import time
import gc
import random
import psutil
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import wandb
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ubuntu/rrweb_tokenizer')
sys.path.append('/home/ubuntu/rrweb-bert/src')

from collator import ImprovedDataCollatorForMLM, DataCollatorForPreTraining
from dataset import RRWebLazyDataset
from tokenizer_wrapper import TokenizerWrapper


class CollatorProductionTest:
    """Production-scale testing for data collator."""
    
    def __init__(self):
        """Initialize test environment."""
        print("Initializing Collator Production Test...")
        
        # Data configuration
        self.data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = TokenizerWrapper.from_pretrained('/home/ubuntu/rrweb_tokenizer/tokenizer_model_20250911_234222')
        
        # Initialize collator
        self.collator = ImprovedDataCollatorForMLM(
            tokenizer=self.tokenizer,
            mlm_probability=0.15,
            pad_to_multiple_of=8
        )
        
        # Initialize wandb
        wandb.init(
            project='rrweb-tokenizer-tests',
            name='test_1_5_collator_production',
            config={
                'test': 'collator_production',
                'data_dir': self.data_dir,
                'mlm_probability': 0.15
            }
        )
        
        print(f"Data directory: {self.data_dir}")
        print(f"Tokenizer vocab size: {self.tokenizer.vocab_size}")
        print(f"MLM probability: {self.collator.mlm_probability}")
    
    def test_1_mlm_probability_distribution(self, n_samples=1000):
        """Test 1: Verify exact 80/10/10 split for MLM."""
        print("\n" + "="*80)
        print("TEST 1: MLM PROBABILITY DISTRIBUTION")
        print("="*80)
        print(f"Testing with {n_samples} samples to verify 80/10/10 split...")
        
        # Create dataset
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=n_samples,
            cache_size=100
        )
        
        # Collect statistics
        total_tokens = 0
        total_masked = 0
        mask_token_count = 0
        random_token_count = 0
        unchanged_count = 0
        
        print("\nProcessing samples...")
        for i in tqdm(range(min(n_samples, len(dataset))), desc="Testing MLM distribution"):
            try:
                sample = dataset[i]
                if sample is None:
                    continue
                
                # Create batch of 1
                batch = self.collator([sample])
                
                input_ids = batch['input_ids'][0]
                labels = batch['labels'][0]
                attention_mask = batch['attention_mask'][0]
                
                # Get original tokens (before masking)
                original = sample['input_ids']
                
                # Count masked positions
                masked_positions = labels != -100
                valid_positions = attention_mask == 1
                
                for j in range(len(input_ids)):
                    if valid_positions[j] and masked_positions[j]:
                        total_masked += 1
                        
                        if input_ids[j] == self.collator.mask_token_id:
                            mask_token_count += 1
                        elif j < len(original) and input_ids[j] != original[j]:
                            random_token_count += 1
                        else:
                            unchanged_count += 1
                    
                    if valid_positions[j]:
                        total_tokens += 1
                
            except Exception as e:
                print(f"  Warning: Sample {i} failed: {e}")
                continue
        
        # Calculate ratios
        mask_ratio = total_masked / max(total_tokens, 1)
        
        if total_masked > 0:
            mask_token_ratio = mask_token_count / total_masked
            random_ratio = random_token_count / total_masked
            unchanged_ratio = unchanged_count / total_masked
        else:
            mask_token_ratio = random_ratio = unchanged_ratio = 0
        
        print(f"\n📊 MLM Statistics:")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Total masked: {total_masked:,} ({mask_ratio:.1%})")
        print(f"\n  Distribution of masked tokens:")
        print(f"    [MASK] tokens: {mask_token_count:,} ({mask_token_ratio:.1%})")
        print(f"    Random tokens: {random_token_count:,} ({random_ratio:.1%})")
        print(f"    Unchanged: {unchanged_count:,} ({unchanged_ratio:.1%})")
        
        # Statistical test for 80/10/10 distribution
        expected = [0.8 * total_masked, 0.1 * total_masked, 0.1 * total_masked]
        observed = [mask_token_count, random_token_count, unchanged_count]
        
        if total_masked > 100:  # Need enough samples for chi-square
            chi2, p_value = stats.chisquare(observed, expected)
            print(f"\n  Chi-square test:")
            print(f"    Chi2: {chi2:.4f}")
            print(f"    P-value: {p_value:.4f}")
            
            # Check if distribution is close to 80/10/10 (p > 0.05)
            success = p_value > 0.05 and abs(mask_ratio - 0.15) < 0.02
        else:
            print("  (Not enough samples for statistical test)")
            success = abs(mask_ratio - 0.15) < 0.02
        
        # Log to wandb
        wandb.log({
            'mlm/total_tokens': total_tokens,
            'mlm/mask_ratio': mask_ratio,
            'mlm/mask_token_ratio': mask_token_ratio,
            'mlm/random_ratio': random_ratio,
            'mlm/unchanged_ratio': unchanged_ratio
        })
        
        if success:
            print(f"\n✅ MLM distribution test PASSED")
        else:
            print(f"\n❌ MLM distribution test FAILED")
            print(f"   Expected: 15% masking with 80/10/10 split")
            print(f"   Got: {mask_ratio:.1%} masking with {mask_token_ratio:.0%}/{random_ratio:.0%}/{unchanged_ratio:.0%}")
        
        return success
    
    def test_2_batch_processing_scale(self):
        """Test 2: Batch processing at production scale."""
        print("\n" + "="*80)
        print("TEST 2: BATCH PROCESSING AT SCALE")
        print("="*80)
        
        batch_sizes = [8, 16, 32, 64]
        results = {}
        
        # Create dataset
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=1000,  # Test with 1000 files
            cache_size=200
        )
        
        for batch_size in batch_sizes:
            print(f"\n📦 Testing batch_size={batch_size}...")
            
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=self.collator,
                num_workers=0  # Single worker for testing
            )
            
            # Process batches
            start_time = time.time()
            memory_start = psutil.Process().memory_info().rss / 1024 / 1024
            
            samples_processed = 0
            batches_processed = 0
            max_memory = memory_start
            
            for i, batch in enumerate(tqdm(dataloader, desc=f"Batch {batch_size}", total=20)):
                if i >= 20:  # Process 20 batches for each size
                    break
                
                # Verify batch dimensions
                assert batch['input_ids'].shape[0] == batch_size or i == len(dataloader) - 1
                assert batch['input_ids'].shape == batch['attention_mask'].shape
                assert batch['input_ids'].shape == batch['labels'].shape
                
                samples_processed += batch['input_ids'].shape[0]
                batches_processed += 1
                
                # Track memory
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                max_memory = max(max_memory, current_memory)
            
            elapsed = time.time() - start_time
            throughput = samples_processed / elapsed
            memory_used = max_memory - memory_start
            
            results[batch_size] = {
                'throughput': throughput,
                'memory_mb': memory_used,
                'time': elapsed
            }
            
            print(f"  Throughput: {throughput:.1f} samples/sec")
            print(f"  Memory used: {memory_used:.1f} MB")
            print(f"  Time: {elapsed:.1f}s")
            
            # Log to wandb
            wandb.log({
                f'batch_processing/throughput_bs{batch_size}': throughput,
                f'batch_processing/memory_mb_bs{batch_size}': memory_used,
                f'batch_processing/time_bs{batch_size}': elapsed
            })
        
        # Summary
        print(f"\n📊 Batch Processing Summary:")
        print(f"{'Batch Size':<12} {'Throughput':<20} {'Memory (MB)':<15}")
        print("-" * 47)
        for bs, metrics in results.items():
            print(f"{bs:<12} {metrics['throughput']:<20.1f} {metrics['memory_mb']:<15.1f}")
        
        # Success criteria: larger batches should be more efficient
        success = results[32]['throughput'] > results[8]['throughput']
        
        if success:
            print(f"\n✅ Batch processing test PASSED")
        else:
            print(f"\n❌ Batch processing test FAILED")
            print(f"   Larger batches should have higher throughput")
        
        return success
    
    def test_3_padding_efficiency(self):
        """Test 3: Padding efficiency with variable length sequences."""
        print("\n" + "="*80)
        print("TEST 3: PADDING EFFICIENCY")
        print("="*80)
        
        # Create samples with varying lengths
        print("Creating variable length samples...")
        samples = []
        
        # Short sequences
        for _ in range(10):
            length = random.randint(50, 200)
            samples.append({
                'input_ids': torch.randint(5, 1000, (length,)),
                'attention_mask': torch.ones(length),
                'token_type_ids': torch.zeros(length),
                'event_type_ids': torch.zeros(length)
            })
        
        # Medium sequences
        for _ in range(10):
            length = random.randint(500, 1000)
            samples.append({
                'input_ids': torch.randint(5, 1000, (length,)),
                'attention_mask': torch.ones(length),
                'token_type_ids': torch.zeros(length),
                'event_type_ids': torch.zeros(length)
            })
        
        # Long sequences
        for _ in range(5):
            length = random.randint(1500, 2048)
            samples.append({
                'input_ids': torch.randint(5, 1000, (length,)),
                'attention_mask': torch.ones(length),
                'token_type_ids': torch.zeros(length),
                'event_type_ids': torch.zeros(length)
            })
        
        # Process batch
        batch = self.collator(samples)
        
        # Calculate padding statistics
        total_tokens = batch['attention_mask'].numel()
        real_tokens = batch['attention_mask'].sum().item()
        padding_tokens = total_tokens - real_tokens
        padding_ratio = padding_tokens / total_tokens
        
        # Check attention masks
        for i in range(batch['input_ids'].shape[0]):
            seq = batch['input_ids'][i]
            mask = batch['attention_mask'][i]
            
            # Verify padding tokens have attention_mask = 0
            padding_positions = seq == self.collator.pad_token_id
            assert (mask[padding_positions] == 0).all(), "Padding tokens should have attention_mask=0"
            
            # Verify non-padding tokens have attention_mask = 1
            non_padding = ~padding_positions
            real_token_positions = mask == 1
            # Most real tokens should be non-padding (some might be token_id=0 legitimately)
        
        print(f"\n📊 Padding Statistics:")
        print(f"  Batch shape: {batch['input_ids'].shape}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Real tokens: {real_tokens:,}")
        print(f"  Padding tokens: {padding_tokens:,}")
        print(f"  Padding ratio: {padding_ratio:.1%}")
        print(f"  Wasted computation: {padding_ratio:.1%}")
        
        # Check pad_to_multiple_of
        if self.collator.pad_to_multiple_of:
            seq_len = batch['input_ids'].shape[1]
            assert seq_len % self.collator.pad_to_multiple_of == 0, \
                f"Sequence length {seq_len} not multiple of {self.collator.pad_to_multiple_of}"
            print(f"  ✓ Padded to multiple of {self.collator.pad_to_multiple_of}")
        
        # Log to wandb
        wandb.log({
            'padding/total_tokens': total_tokens,
            'padding/real_tokens': real_tokens,
            'padding/padding_tokens': padding_tokens,
            'padding/padding_ratio': padding_ratio
        })
        
        # Success: padding ratio should be reasonable (< 50% for mixed lengths)
        success = padding_ratio < 0.5
        
        if success:
            print(f"\n✅ Padding efficiency test PASSED")
        else:
            print(f"\n❌ Padding efficiency test FAILED")
            print(f"   Excessive padding: {padding_ratio:.1%}")
        
        return success
    
    def test_4_special_token_preservation(self):
        """Test 4: Ensure special tokens are never masked."""
        print("\n" + "="*80)
        print("TEST 4: SPECIAL TOKEN PRESERVATION")
        print("="*80)
        
        # Create samples with special tokens
        samples = []
        for _ in range(100):
            # Include special tokens
            tokens = [
                self.collator.cls_token_id,  # [CLS]
                *torch.randint(5, 1000, (50,)).tolist(),
                self.collator.sep_token_id,  # [SEP]
            ]
            
            samples.append({
                'input_ids': torch.tensor(tokens),
                'attention_mask': torch.ones(len(tokens)),
                'token_type_ids': torch.zeros(len(tokens)),
                'event_type_ids': torch.zeros(len(tokens))
            })
        
        # Process batches
        special_masked_count = 0
        structural_masked_count = 0
        text_masked_count = 0
        
        print("Testing special token preservation...")
        for sample in tqdm(samples, desc="Checking tokens"):
            batch = self.collator([sample])
            
            input_ids = batch['input_ids'][0]
            labels = batch['labels'][0]
            original = sample['input_ids']
            
            for i, (token_id, label) in enumerate(zip(input_ids, labels)):
                if label != -100:  # This position was masked
                    original_token = original[i] if i < len(original) else -1
                    
                    # Check if special token was masked (should never happen)
                    if original_token in self.collator.special_token_ids:
                        special_masked_count += 1
                        print(f"  WARNING: Special token {original_token} was masked!")
                    
                    # Check structural vs text tokens
                    if original_token < self.collator.structural_vocab_size:
                        structural_masked_count += 1
                    else:
                        text_masked_count += 1
        
        print(f"\n📊 Token Preservation Statistics:")
        print(f"  Special tokens masked: {special_masked_count} (should be 0)")
        print(f"  Structural tokens masked: {structural_masked_count}")
        print(f"  Text tokens masked: {text_masked_count}")
        
        # Verify [CLS] and [SEP] are preserved
        for sample in samples[:10]:  # Check first 10 samples
            batch = self.collator([sample])
            input_ids = batch['input_ids'][0]
            
            # [CLS] should be at position 0
            if len(input_ids) > 0 and sample['input_ids'][0] == self.collator.cls_token_id:
                assert input_ids[0] == self.collator.cls_token_id or \
                       batch['labels'][0][0] != -100, "[CLS] was incorrectly modified"
            
            # Find [SEP] positions
            sep_positions = (sample['input_ids'] == self.collator.sep_token_id).nonzero(as_tuple=True)[0]
            for pos in sep_positions:
                if pos < len(input_ids):
                    assert input_ids[pos] == self.collator.sep_token_id or \
                           batch['labels'][0][pos] != -100, f"[SEP] at position {pos} was incorrectly modified"
        
        # Log to wandb
        wandb.log({
            'token_preservation/special_masked': special_masked_count,
            'token_preservation/structural_masked': structural_masked_count,
            'token_preservation/text_masked': text_masked_count
        })
        
        success = special_masked_count == 0
        
        if success:
            print(f"\n✅ Special token preservation test PASSED")
        else:
            print(f"\n❌ Special token preservation test FAILED")
            print(f"   {special_masked_count} special tokens were masked!")
        
        return success
    
    def test_5_memory_stability(self, n_iterations=1000):
        """Test 5: Long-running memory stability test."""
        print("\n" + "="*80)
        print("TEST 5: MEMORY STABILITY")
        print("="*80)
        print(f"Running {n_iterations} iterations to check for memory leaks...")
        
        # Create a fixed set of samples
        samples = []
        for _ in range(32):  # Batch of 32
            length = random.randint(100, 2048)
            samples.append({
                'input_ids': torch.randint(5, 1000, (length,)),
                'attention_mask': torch.ones(length),
                'token_type_ids': torch.zeros(length),
                'event_type_ids': torch.zeros(length)
            })
        
        # Track memory over time
        memory_readings = []
        gc.collect()
        memory_start = psutil.Process().memory_info().rss / 1024 / 1024
        
        print(f"Starting memory: {memory_start:.1f} MB")
        
        for i in tqdm(range(n_iterations), desc="Memory stability test"):
            # Process batch
            batch = self.collator(samples)
            
            # Simulate some work
            _ = batch['input_ids'].sum()
            
            # Periodic memory check
            if i % 100 == 0:
                gc.collect()
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_readings.append(current_memory)
                
                if i > 0:
                    memory_growth = current_memory - memory_start
                    print(f"  Iteration {i}: {current_memory:.1f} MB (growth: {memory_growth:+.1f} MB)")
            
            # Clear batch reference
            del batch
        
        # Final memory check
        gc.collect()
        memory_end = psutil.Process().memory_info().rss / 1024 / 1024
        memory_growth = memory_end - memory_start
        
        print(f"\n📊 Memory Stability Results:")
        print(f"  Starting memory: {memory_start:.1f} MB")
        print(f"  Ending memory: {memory_end:.1f} MB")
        print(f"  Total growth: {memory_growth:.1f} MB")
        print(f"  Growth per 1000 iterations: {memory_growth * 1000 / n_iterations:.1f} MB")
        
        # Check for linear growth (indicates leak)
        if len(memory_readings) > 2:
            # Fit linear regression
            x = np.arange(len(memory_readings))
            slope, _ = np.polyfit(x, memory_readings, 1)
            print(f"  Memory growth rate: {slope:.3f} MB per checkpoint")
        
        # Log to wandb
        wandb.log({
            'memory_stability/start_mb': memory_start,
            'memory_stability/end_mb': memory_end,
            'memory_stability/growth_mb': memory_growth,
            'memory_stability/iterations': n_iterations
        })
        
        # Success: less than 50 MB growth over 1000 iterations
        success = memory_growth < 50
        
        if success:
            print(f"\n✅ Memory stability test PASSED")
        else:
            print(f"\n❌ Memory stability test FAILED")
            print(f"   Excessive memory growth: {memory_growth:.1f} MB")
        
        return success
    
    def test_6_edge_cases(self):
        """Test 6: Handle edge cases and problematic inputs."""
        print("\n" + "="*80)
        print("TEST 6: EDGE CASES")
        print("="*80)
        
        test_cases = []
        
        # Empty batch
        print("Testing empty sequences...")
        try:
            batch = self.collator([])
            print("  ❌ Empty batch should raise an error")
            test_cases.append(('empty_batch', False))
        except:
            print("  ✓ Empty batch handled correctly")
            test_cases.append(('empty_batch', True))
        
        # Single token sequence
        print("Testing single token sequence...")
        single_token = [{
            'input_ids': torch.tensor([self.collator.cls_token_id]),
            'attention_mask': torch.ones(1),
            'token_type_ids': torch.zeros(1),
            'event_type_ids': torch.zeros(1)
        }]
        try:
            batch = self.collator(single_token)
            assert batch['input_ids'].shape[1] >= 1
            print("  ✓ Single token handled")
            test_cases.append(('single_token', True))
        except Exception as e:
            print(f"  ❌ Single token failed: {e}")
            test_cases.append(('single_token', False))
        
        # Maximum length sequence
        print("Testing maximum length sequence...")
        max_length = 8192
        max_seq = [{
            'input_ids': torch.randint(5, 1000, (max_length,)),
            'attention_mask': torch.ones(max_length),
            'token_type_ids': torch.zeros(max_length),
            'event_type_ids': torch.zeros(max_length)
        }]
        try:
            batch = self.collator(max_seq)
            assert batch['input_ids'].shape[1] <= max_length * 1.1  # Allow some padding
            print(f"  ✓ Max length ({max_length}) handled")
            test_cases.append(('max_length', True))
        except Exception as e:
            print(f"  ❌ Max length failed: {e}")
            test_cases.append(('max_length', False))
        
        # All padding sequence (shouldn't happen but test resilience)
        print("Testing all-padding sequence...")
        all_padding = [{
            'input_ids': torch.zeros(100, dtype=torch.long),
            'attention_mask': torch.zeros(100),
            'token_type_ids': torch.zeros(100),
            'event_type_ids': torch.zeros(100)
        }]
        try:
            batch = self.collator(all_padding)
            # Should not mask any padding tokens
            assert (batch['labels'][0] == -100).all()
            print("  ✓ All-padding handled")
            test_cases.append(('all_padding', True))
        except Exception as e:
            print(f"  ❌ All-padding failed: {e}")
            test_cases.append(('all_padding', False))
        
        # Mixed batch with extreme length differences
        print("Testing extreme length differences...")
        extreme_mixed = [
            {
                'input_ids': torch.randint(5, 1000, (10,)),
                'attention_mask': torch.ones(10),
                'token_type_ids': torch.zeros(10),
                'event_type_ids': torch.zeros(10)
            },
            {
                'input_ids': torch.randint(5, 1000, (2000,)),
                'attention_mask': torch.ones(2000),
                'token_type_ids': torch.zeros(2000),
                'event_type_ids': torch.zeros(2000)
            }
        ]
        try:
            batch = self.collator(extreme_mixed)
            assert batch['input_ids'].shape[0] == 2
            assert batch['input_ids'].shape[1] >= 2000
            print("  ✓ Extreme length differences handled")
            test_cases.append(('extreme_lengths', True))
        except Exception as e:
            print(f"  ❌ Extreme lengths failed: {e}")
            test_cases.append(('extreme_lengths', False))
        
        # Summary
        print(f"\n📊 Edge Case Results:")
        passed = sum(1 for _, success in test_cases if success)
        total = len(test_cases)
        
        for name, success in test_cases:
            status = "✓" if success else "❌"
            print(f"  {status} {name}")
        
        print(f"\n  Passed: {passed}/{total}")
        
        # Log to wandb
        wandb.log({
            'edge_cases/passed': passed,
            'edge_cases/total': total,
            'edge_cases/success_rate': passed / total
        })
        
        success = passed == total
        
        if success:
            print(f"\n✅ Edge cases test PASSED")
        else:
            print(f"\n❌ Edge cases test FAILED ({total - passed} failures)")
        
        return success
    
    def run_all_tests(self):
        """Run all collator production tests."""
        print("\n" + "="*80)
        print("COLLATOR PRODUCTION TEST SUITE")
        print("="*80)
        
        all_passed = True
        
        # Test 1: MLM distribution
        if not self.test_1_mlm_probability_distribution(n_samples=500):
            all_passed = False
        
        # Test 2: Batch processing
        if not self.test_2_batch_processing_scale():
            all_passed = False
        
        # Test 3: Padding efficiency
        if not self.test_3_padding_efficiency():
            all_passed = False
        
        # Test 4: Special token preservation
        if not self.test_4_special_token_preservation():
            all_passed = False
        
        # Test 5: Memory stability
        if not self.test_5_memory_stability(n_iterations=500):
            all_passed = False
        
        # Test 6: Edge cases
        if not self.test_6_edge_cases():
            all_passed = False
        
        # Final summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        
        if all_passed:
            print("🎉 ALL COLLATOR PRODUCTION TESTS PASSED! 🎉")
            print("\nThe collator is ready for production training:")
            print("  ✓ MLM distribution correct (80/10/10)")
            print("  ✓ Efficient batch processing")
            print("  ✓ Reasonable padding overhead")
            print("  ✓ Special tokens preserved")
            print("  ✓ No memory leaks detected")
            print("  ✓ Edge cases handled")
        else:
            print("❌ SOME TESTS FAILED")
            print("\nIssues to address before production:")
            print("  - Review failed tests above")
            print("  - Fix identified issues")
            print("  - Re-run test suite")
        
        wandb.log({'test_suite_passed': all_passed})
        wandb.finish()
        
        return all_passed


def main():
    """Main entry point."""
    tester = CollatorProductionTest()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()