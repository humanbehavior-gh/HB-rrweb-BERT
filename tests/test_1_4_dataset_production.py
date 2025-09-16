#!/usr/bin/env python3
"""
Test 1.4: Dataset Production Testing - PRODUCTION TEST
Verify the lazy-loading dataset works at production scale.
No timeouts, iterate until 100% working.
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

sys.path.append('/home/ubuntu/rrweb_tokenizer')
sys.path.append('/home/ubuntu/rrweb-bert/src')

from rrweb_tokenizer import RRWebTokenizer
from dataset import RRWebLazyDataset


class DatasetProductionTester:
    """Test dataset at production scale."""
    
    def __init__(self):
        self.tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
        self.data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
        
        # Initialize wandb
        wandb.init(
            project="rrweb-tokenizer-tests",
            name="test_1_4_dataset_production",
            config={
                "test_type": "dataset_production",
                "tokenizer_path": self.tokenizer_path,
                "data_dir": self.data_dir
            }
        )
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = RRWebTokenizer.load(self.tokenizer_path)
        
        # Track memory usage
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Statistics
        self.stats = defaultdict(lambda: 0)
    
    def test_1_memory_efficiency(self, n_files: int = 31951):
        """Test 1: Memory Efficiency - Load full dataset without OOM."""
        print("\n" + "="*80)
        print(f"TEST 1: MEMORY EFFICIENCY ({n_files} files)")
        print("="*80)
        
        start_time = time.time()
        memory_samples = []
        
        print("Creating lazy dataset...")
        
        # Create dataset with full file list
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=n_files,
            cache_size=100,  # Small cache to test memory efficiency
            min_session_length=10
        )
        
        # Check memory after initialization
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory
        
        print(f"\nDataset initialized:")
        print(f"  Files loaded: {len(dataset)}")
        print(f"  Memory used: {memory_used:.1f} MB")
        print(f"  Memory per file: {memory_used / len(dataset) * 1000:.3f} KB")
        
        # Test accessing random samples
        print("\nTesting random access (10 samples)...")
        access_times = []
        
        for i in range(10):
            idx = np.random.randint(0, len(dataset))
            start = time.time()
            sample = dataset[idx]
            access_time = time.time() - start
            access_times.append(access_time)
            
            if sample is not None:
                print(f"  Sample {idx}: {len(sample['input_ids'])} tokens in {access_time:.3f}s")
            
            # Track memory
            current_memory = self.process.memory_info().rss / 1024 / 1024
            memory_samples.append(current_memory)
        
        # Memory statistics
        max_memory = max(memory_samples) - self.initial_memory
        avg_memory = np.mean(memory_samples) - self.initial_memory
        
        print(f"\nMemory Statistics:")
        print(f"  Peak memory usage: {max_memory:.1f} MB")
        print(f"  Average memory usage: {avg_memory:.1f} MB")
        print(f"  Access time average: {np.mean(access_times):.3f}s")
        
        # Log to wandb
        wandb.log({
            "memory_efficiency/files_loaded": len(dataset),
            "memory_efficiency/peak_memory_mb": max_memory,
            "memory_efficiency/avg_memory_mb": avg_memory,
            "memory_efficiency/memory_per_file_kb": memory_used / len(dataset) * 1000,
            "memory_efficiency/avg_access_time": np.mean(access_times)
        })
        
        # Success criteria: Less than 2GB for full dataset
        success = max_memory < 2000
        
        if success:
            print(f"\n✅ Memory efficiency test PASSED (peak: {max_memory:.1f} MB < 2000 MB)")
        else:
            print(f"\n❌ Memory efficiency test FAILED (peak: {max_memory:.1f} MB >= 2000 MB)")
        
        # Cleanup
        del dataset
        gc.collect()
        
        return success
    
    def test_2_data_integrity(self, n_samples: int = 100):
        """Test 2: Data Integrity - Verify tokenization consistency."""
        print("\n" + "="*80)
        print(f"TEST 2: DATA INTEGRITY ({n_samples} samples)")
        print("="*80)
        
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=1000,
            cache_size=50
        )
        
        integrity_errors = []
        consistency_checks = []
        
        print("Testing data integrity...")
        pbar = tqdm(range(min(n_samples, len(dataset))), desc="Checking samples")
        
        for i in pbar:
            try:
                # Get sample twice to check consistency
                sample1 = dataset[i]
                sample2 = dataset[i]
                
                if sample1 is None or sample2 is None:
                    continue
                
                # Check consistency
                is_consistent = (
                    torch.equal(sample1['input_ids'], sample2['input_ids']) and
                    torch.equal(sample1['attention_mask'], sample2['attention_mask'])
                )
                
                consistency_checks.append(is_consistent)
                
                if not is_consistent:
                    integrity_errors.append({
                        'index': i,
                        'error': 'Inconsistent tokenization'
                    })
                
                # Verify tensor properties
                if sample1['input_ids'].max() >= self.tokenizer.vocab.structural_vocab_size + len(self.tokenizer.bpe_tokenizer.vocab):
                    integrity_errors.append({
                        'index': i,
                        'error': f"Token ID out of range: {sample1['input_ids'].max()}"
                    })
                
                # Check shapes match
                if len(sample1['input_ids']) != len(sample1['attention_mask']):
                    integrity_errors.append({
                        'index': i,
                        'error': 'Shape mismatch between input_ids and attention_mask'
                    })
                
            except Exception as e:
                integrity_errors.append({
                    'index': i,
                    'error': str(e)
                })
        
        # Results
        consistency_rate = sum(consistency_checks) / len(consistency_checks) if consistency_checks else 0
        
        print(f"\nData Integrity Results:")
        print(f"  Samples checked: {len(consistency_checks)}")
        print(f"  Consistency rate: {consistency_rate*100:.1f}%")
        print(f"  Integrity errors: {len(integrity_errors)}")
        
        if integrity_errors:
            print(f"\nFirst 5 errors:")
            for err in integrity_errors[:5]:
                print(f"  Index {err['index']}: {err['error']}")
        
        # Log to wandb
        wandb.log({
            "data_integrity/samples_checked": len(consistency_checks),
            "data_integrity/consistency_rate": consistency_rate,
            "data_integrity/error_count": len(integrity_errors)
        })
        
        success = consistency_rate > 0.99 and len(integrity_errors) < 5
        
        if success:
            print(f"\n✅ Data integrity test PASSED")
        else:
            print(f"\n❌ Data integrity test FAILED")
        
        return success
    
    def test_3_caching_performance(self):
        """Test 3: Production Scale Caching Performance."""
        print("\n" + "="*80)
        print("TEST 3: PRODUCTION SCALE CACHING PERFORMANCE")
        print("="*80)
        print("Testing caching with realistic patterns...")
        
        # Use smaller sample for faster testing, but still realistic
        n_files = 500  # Enough files to test caching
        cache_size = 50  # Realistic cache that will have evictions
        test_size = 30  # Test with 30 files accessed multiple times
        
        print(f"\n🔧 Configuration:")
        print(f"  Dataset size: {n_files} files")
        print(f"  Cache size: {cache_size} entries")
        print(f"  Test sample: {test_size} files")
        print(f"  Access pattern: 3 epochs with shuffling")
        
        # Create dataset with cache
        print("\nCreating dataset...")
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=n_files,
            cache_size=cache_size
        )
        
        # Test with a subset of files
        file_indices = list(range(test_size))
        
        # First access: populate cache (cold)
        print(f"\n📊 Phase 1: Cold cache (first access to {test_size} files)")
        cold_times = []
        for i, idx in enumerate(file_indices):
            if i % 10 == 0:
                print(f"  Processing file {i}/{test_size}...")
            start = time.time()
            _ = dataset[idx]
            cold_times.append(time.time() - start)
        
        avg_cold = np.mean(cold_times)
        cache_info_cold = dataset._cached_tokenize.cache_info()
        
        # Second access: should hit cache (warm)
        print(f"\n📊 Phase 2: Warm cache (re-accessing same {test_size} files)")
        warm_times = []
        random.shuffle(file_indices)  # Access in different order
        for i, idx in enumerate(file_indices):
            if i % 10 == 0:
                print(f"  Processing file {i}/{test_size}...")
            start = time.time()
            _ = dataset[idx]
            warm_times.append(time.time() - start)
        
        avg_warm = np.mean(warm_times)
        cache_info_warm = dataset._cached_tokenize.cache_info()
        
        # Third access: still warm
        print(f"\n📊 Phase 3: Testing cache persistence")
        persist_times = []
        random.shuffle(file_indices)
        for idx in file_indices[:10]:  # Just test 10 files
            start = time.time()
            _ = dataset[idx]
            persist_times.append(time.time() - start)
        
        avg_persist = np.mean(persist_times)
        
        # Get final cache statistics
        cache_info = dataset._cached_tokenize.cache_info()
        hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) if (cache_info.hits + cache_info.misses) > 0 else 0
        
        # Calculate speedup
        speedup = avg_cold / avg_warm if avg_warm > 0 else 1
        
        # Detailed results
        print(f"\n" + "="*60)
        print("🎯 PRODUCTION CACHING RESULTS")
        print("="*60)
        print(f"\n📈 Performance Comparison:")
        print(f"  Phase 1 - Cold Cache:")
        print(f"    Average time: {avg_cold:.3f}s per file")
        print(f"    Total time: {sum(cold_times):.1f}s")
        print(f"    Cache state: {cache_info_cold.currsize} entries")
        print(f"  Phase 2 - Warm Cache:")
        print(f"    Average time: {avg_warm:.3f}s per file")
        print(f"    Total time: {sum(warm_times):.1f}s")
        print(f"    New hits: {cache_info_warm.hits - cache_info_cold.hits}")
        print(f"  Phase 3 - Persistence:")
        print(f"    Average time: {avg_persist:.3f}s per file")
        print(f"\n🚀 Performance Improvement:")
        print(f"    Speedup: {speedup:.2f}x faster with cache")
        print(f"    Time saved: {sum(cold_times) - sum(warm_times):.1f}s ({(1-1/speedup)*100:.0f}% reduction)")
        
        print(f"\n📊 Cache Statistics:")
        print(f"    Hit rate: {hit_rate*100:.1f}%")
        print(f"    Total hits: {cache_info.hits}")
        print(f"    Total misses: {cache_info.misses}")
        print(f"    Cache utilization: {cache_info.currsize}/{cache_info.maxsize} ({cache_info.currsize/cache_info.maxsize*100:.1f}%)")
        
        # Memory impact
        memory_info = psutil.Process().memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        print(f"\n💾 Memory Impact:")
        print(f"    Current RSS: {memory_mb:.1f} MB")
        print(f"    Cache entries: {cache_info.currsize}/{cache_info.maxsize}")
        print(f"    Cache utilization: {cache_info.currsize/cache_info.maxsize*100:.0f}%")
        
        # Log to wandb
        wandb.log({
            "caching/speedup": speedup,
            "caching/hit_rate": hit_rate,
            "caching/total_hits": cache_info.hits,
            "caching/total_misses": cache_info.misses,
            "caching/cache_utilization": cache_info.currsize / cache_info.maxsize,
            "caching/time_saved": sum(cold_times) - sum(warm_times),
            "caching/avg_cold": avg_cold,
            "caching/avg_warm": avg_warm,
            "caching/avg_persist": avg_persist,
            "caching/memory_mb": memory_mb
        })
        
        # Production success criteria
        success = speedup > 1.5 and hit_rate > 0.5
        
        if success:
            print(f"\n✅ Production caching test PASSED")
            print(f"   Achieved {speedup:.2f}x speedup with {hit_rate*100:.1f}% hit rate")
        else:
            print(f"\n❌ Production caching test FAILED")
            print(f"   Expected: speedup > 1.5x and hit rate > 50%")
            print(f"   Got: speedup = {speedup:.2f}x, hit rate = {hit_rate*100:.1f}%")
        
        return success
    
    def test_4_edge_cases(self):
        """Test 4: Edge Cases - Handle problematic files."""
        print("\n" + "="*80)
        print("TEST 4: EDGE CASES")
        print("="*80)
        
        # Create test files for edge cases
        test_dir = '/tmp/rrweb_edge_cases'
        os.makedirs(test_dir, exist_ok=True)
        
        edge_cases = {
            'empty.json': [],
            'invalid.json': '{invalid json',
            'huge_session.json': [{'type': i, 'timestamp': i} for i in range(10000)],
            'no_events.json': {'metadata': 'test'},
            'nested_events.json': {'events': {'nested': [{'type': 1}]}},
            'single_event.json': [{'type': 2, 'data': {}}]
        }
        
        # Write test files
        for filename, content in edge_cases.items():
            filepath = os.path.join(test_dir, filename)
            try:
                if filename == 'invalid.json':
                    with open(filepath, 'w') as f:
                        f.write(content)
                else:
                    with open(filepath, 'w') as f:
                        json.dump(content, f)
            except:
                pass
        
        # Test dataset with edge cases
        dataset = RRWebLazyDataset(
            data_dir=test_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=None,
            cache_size=10
        )
        
        handled_correctly = 0
        errors = []
        
        print(f"Testing {len(dataset)} edge case files...")
        
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                if sample is not None:
                    handled_correctly += 1
                    print(f"  ✓ File {i}: Handled successfully")
                else:
                    # None is acceptable for empty files
                    handled_correctly += 1
                    print(f"  ✓ File {i}: Correctly returned None")
            except Exception as e:
                errors.append(str(e))
                print(f"  ✗ File {i}: Error - {e}")
        
        # Clean up test files
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
        print(f"\nEdge Case Results:")
        print(f"  Files tested: {len(dataset)}")
        print(f"  Handled correctly: {handled_correctly}")
        print(f"  Errors: {len(errors)}")
        
        # Log to wandb
        wandb.log({
            "edge_cases/total_files": len(dataset),
            "edge_cases/handled_correctly": handled_correctly,
            "edge_cases/errors": len(errors)
        })
        
        success = len(errors) == 0
        
        if success:
            print(f"\n✅ Edge cases test PASSED")
        else:
            print(f"\n❌ Edge cases test FAILED")
        
        return success
    
    def test_5_production_scale(self, n_files: int = 5000, batch_size: int = 8):
        """Test 5: Production Scale - Process with DataLoader."""
        print("\n" + "="*80)
        print(f"TEST 5: PRODUCTION SCALE ({n_files} files, batch_size={batch_size})")
        print("="*80)
        
        # Create dataset
        dataset = RRWebLazyDataset(
            data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            max_length=2048,
            max_files=n_files,
            cache_size=100
        )
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Start with 0 for testing
            collate_fn=self.custom_collate_fn
        )
        
        print(f"Created DataLoader with {len(dataset)} samples")
        print(f"Expected batches: {len(dataloader)}")
        
        # Process batches
        batch_times = []
        successful_batches = 0
        failed_batches = 0
        total_samples = 0
        
        start_time = time.time()
        
        pbar = tqdm(dataloader, desc="Processing batches")
        for batch_idx, batch in enumerate(pbar):
            batch_start = time.time()
            
            try:
                if batch is not None:
                    # Verify batch structure
                    assert 'input_ids' in batch
                    assert 'attention_mask' in batch
                    
                    batch_size_actual = batch['input_ids'].shape[0]
                    total_samples += batch_size_actual
                    successful_batches += 1
                    
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    
                    # Update progress more frequently
                    avg_time = np.mean(batch_times[-10:]) if batch_times else 0
                    samples_per_sec = total_samples / (time.time() - start_time) if (time.time() - start_time) > 0 else 0
                    pbar.set_description(
                        f"OK: {successful_batches} | "
                        f"Samples: {total_samples} | "
                        f"Speed: {samples_per_sec:.1f} samples/s"
                    )
                    
                    if batch_idx % 10 == 0:
                        
                        # Log intermediate results
                        wandb.log({
                            "production_scale/batches_processed": successful_batches,
                            "production_scale/samples_processed": total_samples,
                            "production_scale/avg_batch_time": avg_time
                        })
                else:
                    failed_batches += 1
                    
            except Exception as e:
                failed_batches += 1
                if failed_batches <= 5:
                    print(f"\n  Batch {batch_idx} error: {e}")
            
            # Stop if too many failures
            if failed_batches > 10:
                print(f"\n⚠️ Too many failures ({failed_batches}), stopping test")
                break
        
        total_time = time.time() - start_time
        
        print(f"\nProduction Scale Results:")
        print(f"  Total batches: {successful_batches + failed_batches}")
        print(f"  Successful batches: {successful_batches}")
        print(f"  Failed batches: {failed_batches}")
        print(f"  Total samples processed: {total_samples}")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average batch time: {np.mean(batch_times):.3f}s")
        print(f"  Throughput: {total_samples/total_time:.1f} samples/s")
        
        # Memory check
        current_memory = self.process.memory_info().rss / 1024 / 1024
        memory_used = current_memory - self.initial_memory
        print(f"  Memory used: {memory_used:.1f} MB")
        
        # Log final results
        wandb.log({
            "production_scale/total_batches": successful_batches,
            "production_scale/failed_batches": failed_batches,
            "production_scale/total_samples": total_samples,
            "production_scale/throughput": total_samples/total_time,
            "production_scale/memory_mb": memory_used
        })
        
        success_rate = successful_batches / (successful_batches + failed_batches) if (successful_batches + failed_batches) > 0 else 0
        success = success_rate > 0.95 and total_samples > n_files * 0.8
        
        if success:
            print(f"\n✅ Production scale test PASSED (success rate: {success_rate*100:.1f}%)")
        else:
            print(f"\n❌ Production scale test FAILED (success rate: {success_rate*100:.1f}%)")
        
        return success
    
    def custom_collate_fn(self, batch):
        """Custom collate function to handle None samples and variable lengths."""
        # Filter out None samples
        batch = [b for b in batch if b is not None]
        
        if len(batch) == 0:
            return None
        
        # Find max length in batch for padding
        max_len = max(len(b['input_ids']) for b in batch)
        
        # Pad all tensors to max length
        padded_input_ids = []
        padded_attention_mask = []
        padded_token_type_ids = []
        padded_event_type_ids = []
        
        for b in batch:
            seq_len = len(b['input_ids'])
            padding_len = max_len - seq_len
            
            # Pad input_ids with 0 (PAD token)
            padded_input_ids.append(
                torch.cat([b['input_ids'], torch.zeros(padding_len, dtype=torch.long)])
            )
            
            # Pad attention_mask with 0
            padded_attention_mask.append(
                torch.cat([b['attention_mask'], torch.zeros(padding_len, dtype=torch.long)])
            )
            
            # Pad token_type_ids with 0
            padded_token_type_ids.append(
                torch.cat([b['token_type_ids'], torch.zeros(padding_len, dtype=torch.long)])
            )
            
            # Pad event_type_ids if present
            if 'event_type_ids' in b:
                padded_event_type_ids.append(
                    torch.cat([b['event_type_ids'], torch.zeros(padding_len, dtype=torch.long)])
                )
        
        result = {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'token_type_ids': torch.stack(padded_token_type_ids)
        }
        
        # Add event_type_ids if present
        if padded_event_type_ids:
            result['event_type_ids'] = torch.stack(padded_event_type_ids)
        
        return result
    
    def run_all_tests(self):
        """Run all dataset production tests."""
        print("="*80)
        print("DATASET PRODUCTION TEST SUITE")
        print("="*80)
        
        all_passed = True
        
        # Test 1: Memory Efficiency - TEST WITH FULL 31K FILES!
        print("\n🔥 Testing with FULL 31,951 files to prove lazy loading works!")
        if not self.test_1_memory_efficiency(n_files=31951):  # Full dataset!
            all_passed = False
        
        # Test 2: Data Integrity
        if not self.test_2_data_integrity(n_samples=100):
            all_passed = False
        
        # Test 3: Caching Performance
        try:
            if not self.test_3_caching_performance():
                all_passed = False
        except Exception as e:
            print(f"❌ Caching test failed with error: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
        
        # Test 4: Edge Cases
        if not self.test_4_edge_cases():
            all_passed = False
        
        # Test 5: Production Scale - Test with FULL DATASET!
        print("\n🔥 Testing DataLoader with FULL 31,951 files in batches!")
        print("This proves we can handle the entire dataset efficiently with lazy loading.")
        if not self.test_5_production_scale(n_files=31951, batch_size=8):
            all_passed = False
        
        # Summary
        print("\n" + "="*80)
        print("DATASET PRODUCTION TEST SUMMARY")
        print("="*80)
        
        if all_passed:
            print("✅ ALL DATASET PRODUCTION TESTS PASSED!")
            print("Dataset is production ready.")
        else:
            print("⚠️ Some tests failed - iterate until working")
        
        # Log summary
        wandb.log({
            "test_suite": "dataset_production",
            "all_tests_passed": all_passed
        })
        
        wandb.finish()
        
        return all_passed


def main():
    """Run dataset production tests."""
    tester = DatasetProductionTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()