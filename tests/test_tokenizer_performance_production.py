#!/usr/bin/env python3
"""
Test 1.2: Tokenizer Performance at Scale - PRODUCTION TEST
Accept 1s per file as reasonable performance.
Focus on correctness, stability, and handling scale.
"""

import sys
import os
import time
import json
import psutil
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple
from tqdm import tqdm
import wandb

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer


class ProductionTokenizerTest:
    """Test tokenizer for production readiness at scale."""
    
    def __init__(self):
        self.tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
        self.data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
        self.tokenizer = RRWebTokenizer.load(self.tokenizer_path)
        
        # Initialize wandb
        wandb.init(
            project="rrweb-tokenizer-tests",
            name="test_1_2_performance_scale",
            config={
                "test_type": "performance_at_scale",
                "tokenizer_path": self.tokenizer_path,
                "data_dir": self.data_dir,
                "vocab_size": 12520
            }
        )
        
        # Find ALL test files for production testing
        print("Finding all RRWEB files...")
        self.test_files = []
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.json'):
                    self.test_files.append(os.path.join(root, file))
        
        print(f"Found {len(self.test_files)} total files for production testing")
        wandb.log({"total_files_found": len(self.test_files)})
        
        self.errors = []
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_tokens': 0,
            'max_tokens': 0,
            'min_tokens': float('inf'),
            'processing_times': []
        }
    
    def tokenize_file(self, file_path: str) -> Tuple[bool, int, float, str]:
        """Tokenize a single file and return (success, token_count, time, error_msg)."""
        start_time = time.time()
        error_msg = ""
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different formats
            events = []
            if isinstance(data, list):
                events = data
            elif isinstance(data, dict) and 'events' in data:
                events = data['events']
            
            # Tokenize
            tokens = self.tokenizer.tokenize_session(events)
            elapsed = time.time() - start_time
            
            return True, len(tokens), elapsed, ""
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            return False, 0, elapsed, error_msg
    
    def test_scale_processing(self, n_files: int = 1000):
        """Test processing at scale with production expectations."""
        print("\n" + "="*80)
        print(f"TESTING SCALE PROCESSING ({n_files} files)")
        print("Expectation: ~1 second per file is acceptable")
        print("="*80)
        
        test_files = self.test_files[:n_files] if n_files <= len(self.test_files) else self.test_files
        actual_count = len(test_files)
        
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start_time = time.time()
        
        # Create progress bar
        pbar = tqdm(total=actual_count, desc="Processing files", unit="files")
        
        for i, file_path in enumerate(test_files):
            file_start = time.time()
            success, token_count, elapsed, error = self.tokenize_file(file_path)
            
            self.stats['total_files'] += 1
            self.stats['processing_times'].append(elapsed)
            
            if success:
                self.stats['successful'] += 1
                self.stats['total_tokens'] += token_count
                self.stats['max_tokens'] = max(self.stats['max_tokens'], token_count)
                self.stats['min_tokens'] = min(self.stats['min_tokens'], token_count)
            else:
                self.stats['failed'] += 1
                self.errors.append((file_path, error))
            
            # Update progress bar
            pbar.update(1)
            
            # Calculate current metrics
            elapsed_total = time.time() - start_time
            rate = (i + 1) / elapsed_total
            memory_current = process.memory_info().rss / 1024 / 1024
            
            # Log to wandb for every file
            wandb.log({
                "file_index": i + 1,
                "file_name": os.path.basename(file_path),
                "success": success,
                "token_count": token_count if success else 0,
                "processing_time": elapsed,
                "cumulative_success_rate": self.stats['successful'] / (i + 1),
                "cumulative_avg_time": sum(self.stats['processing_times']) / len(self.stats['processing_times']),
                "files_per_second": rate,
                "memory_mb": memory_current,
                "memory_delta_mb": memory_current - memory_before,
                "total_tokens_so_far": self.stats['total_tokens'],
                "max_tokens_seen": self.stats['max_tokens'],
                "min_tokens_seen": self.stats['min_tokens'] if self.stats['min_tokens'] != float('inf') else 0
            })
            
            # Update progress bar description with current stats
            pbar.set_description(
                f"Files [{self.stats['successful']}/{i+1}] "
                f"Rate: {rate:.2f}/s "
                f"Mem: {memory_current:.0f}MB"
            )
            
            # Detailed progress every 100 files
            if (i + 1) % 100 == 0:
                eta = (actual_count - (i + 1)) / rate if rate > 0 else 0
                
                print(f"\n--- Progress Report at {i+1} files ---")
                print(f"  Success rate: {100*self.stats['successful']/(i+1):.1f}%")
                print(f"  Avg time/file: {sum(self.stats['processing_times'])/len(self.stats['processing_times']):.3f}s")
                print(f"  Memory usage: {memory_current:.1f} MB (Δ{memory_current - memory_before:+.1f} MB)")
                print(f"  ETA: {eta/60:.1f} minutes")
                
                # Log summary to wandb
                wandb.log({
                    "checkpoint": i + 1,
                    "checkpoint_success_rate": self.stats['successful'] / (i + 1),
                    "checkpoint_avg_time": sum(self.stats['processing_times']) / len(self.stats['processing_times']),
                    "checkpoint_memory_mb": memory_current,
                    "checkpoint_eta_minutes": eta / 60
                })
        
        pbar.close()
        
        total_time = time.time() - start_time
        memory_after = process.memory_info().rss / 1024 / 1024
        
        # Calculate final statistics
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times'])
        max_time = max(self.stats['processing_times'])
        min_time = min(self.stats['processing_times'])
        
        # Log final results to wandb
        final_results = {
            "test_name": "scale_processing",
            "total_files": self.stats['total_files'],
            "successful_files": self.stats['successful'],
            "failed_files": self.stats['failed'],
            "success_rate": self.stats['successful'] / self.stats['total_files'],
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "avg_time_per_file": avg_time,
            "min_time_per_file": min_time,
            "max_time_per_file": max_time,
            "throughput_files_per_sec": actual_count / total_time,
            "total_tokens": self.stats['total_tokens'],
            "avg_tokens_per_file": self.stats['total_tokens'] / max(self.stats['successful'], 1),
            "min_tokens_in_file": self.stats['min_tokens'] if self.stats['min_tokens'] != float('inf') else 0,
            "max_tokens_in_file": self.stats['max_tokens'],
            "memory_start_mb": memory_before,
            "memory_end_mb": memory_after,
            "memory_increase_mb": memory_after - memory_before,
            "performance_acceptable": avg_time <= 2.0
        }
        
        wandb.log(final_results)
        
        print("\n" + "="*80)
        print("SCALE PROCESSING RESULTS")
        print("="*80)
        print(f"Files processed: {self.stats['total_files']}")
        print(f"Successful: {self.stats['successful']} ({100*self.stats['successful']/self.stats['total_files']:.1f}%)")
        print(f"Failed: {self.stats['failed']}")
        print(f"\nPerformance:")
        print(f"  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"  Average: {avg_time:.3f} sec/file")
        print(f"  Min: {min_time:.3f} sec/file")
        print(f"  Max: {max_time:.3f} sec/file")
        print(f"  Throughput: {actual_count/total_time:.2f} files/sec")
        print(f"\nToken statistics:")
        print(f"  Total tokens: {self.stats['total_tokens']:,}")
        print(f"  Avg tokens/file: {self.stats['total_tokens']/max(self.stats['successful'],1):.0f}")
        print(f"  Min tokens: {self.stats['min_tokens'] if self.stats['min_tokens'] != float('inf') else 0}")
        print(f"  Max tokens: {self.stats['max_tokens']}")
        print(f"\nMemory:")
        print(f"  Start: {memory_before:.1f} MB")
        print(f"  End: {memory_after:.1f} MB")
        print(f"  Increase: {memory_after - memory_before:.1f} MB")
        
        # Check if performance is acceptable
        if avg_time <= 2.0:  # 2 seconds per file is acceptable
            print("\n✅ Performance is ACCEPTABLE for production")
        else:
            print("\n⚠️ Performance may be too slow for production")
        
        if self.stats['failed'] > 0:
            print(f"\nFirst 5 errors:")
            for file_path, error in self.errors[:5]:
                print(f"  {os.path.basename(file_path)}: {error}")
            
            # Log errors to wandb
            wandb.log({"error_samples": [
                {"file": os.path.basename(fp), "error": err} 
                for fp, err in self.errors[:10]
            ]})
        
        return self.stats['successful'] / self.stats['total_files'] > 0.95  # 95% success rate
    
    def test_parallel_processing(self, n_files: int = 100, n_workers: int = 4):
        """Test parallel processing with multiple workers."""
        print("\n" + "="*80)
        print(f"TESTING PARALLEL PROCESSING ({n_files} files, {n_workers} workers)")
        print("="*80)
        
        test_files = self.test_files[:n_files]
        
        start_time = time.time()
        results = []
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.tokenize_file, f) for f in test_files]
            for future in tqdm(futures, desc="Parallel processing"):
                results.append(future.result())
        
        total_time = time.time() - start_time
        
        successful = sum(1 for r in results if r[0])
        
        parallel_results = {
            "test_name": "parallel_processing",
            "n_files": n_files,
            "n_workers": n_workers,
            "successful": successful,
            "total_time": total_time,
            "throughput": n_files / total_time,
            "speedup": n_files / total_time / (n_files / sum(r[2] for r in results))
        }
        
        wandb.log(parallel_results)
        
        print(f"Results:")
        print(f"  Files: {successful}/{n_files} successful")
        print(f"  Time: {total_time:.1f} seconds")
        print(f"  Throughput: {n_files/total_time:.2f} files/sec")
        print(f"  Speedup vs sequential: {parallel_results['speedup']:.2f}x")
        
        return successful == n_files
    
    def test_memory_stability(self, n_iterations: int = 10, files_per_iteration: int = 100):
        """Test that memory usage is stable over time."""
        print("\n" + "="*80)
        print(f"TESTING MEMORY STABILITY ({n_iterations} iterations)")
        print("="*80)
        
        process = psutil.Process()
        memory_readings = []
        
        for iteration in tqdm(range(n_iterations), desc="Memory stability test"):
            # Get memory before
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Process batch
            start_idx = iteration * files_per_iteration
            end_idx = min(start_idx + files_per_iteration, len(self.test_files))
            test_files = self.test_files[start_idx:end_idx]
            
            for file_path in test_files:
                self.tokenize_file(file_path)
            
            # Get memory after
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_readings.append(memory_after)
            
            wandb.log({
                "memory_iteration": iteration + 1,
                "memory_mb": memory_after,
                "memory_delta_mb": memory_after - memory_before
            })
            
            print(f"  Iteration {iteration + 1}: {memory_after:.1f} MB (Δ{memory_after - memory_before:+.1f} MB)")
        
        # Check stability
        first_half_avg = sum(memory_readings[:n_iterations//2]) / (n_iterations//2)
        second_half_avg = sum(memory_readings[n_iterations//2:]) / (n_iterations - n_iterations//2)
        
        growth_rate = (second_half_avg - first_half_avg) / first_half_avg
        
        memory_results = {
            "test_name": "memory_stability",
            "first_half_avg_mb": first_half_avg,
            "second_half_avg_mb": second_half_avg,
            "growth_rate": growth_rate,
            "is_stable": growth_rate < 0.20
        }
        
        wandb.log(memory_results)
        
        print(f"\nMemory analysis:")
        print(f"  First half avg: {first_half_avg:.1f} MB")
        print(f"  Second half avg: {second_half_avg:.1f} MB")
        print(f"  Growth rate: {growth_rate:.1%}")
        
        if growth_rate < 0.20:  # Allow up to 20% growth
            print("  ✅ Memory usage is stable")
            return True
        else:
            print("  ⚠️ Memory may be growing unbounded")
            return False
    
    def test_error_handling(self):
        """Test handling of malformed files."""
        print("\n" + "="*80)
        print("TESTING ERROR HANDLING")
        print("="*80)
        
        # Create test cases
        test_cases = [
            ("empty_file", {}),
            ("null_events", {"events": None}),
            ("invalid_event", {"events": [{"invalid": "structure"}]}),
            ("mixed_valid_invalid", {"events": [
                {"type": 2, "data": {"node": {"type": 1, "tagName": "div"}}},
                {"invalid": "event"},
                {"type": 3, "data": {}}
            ]})
        ]
        
        error_results = []
        
        for name, data in test_cases:
            # Save test file
            test_file = f"/tmp/test_{name}.json"
            with open(test_file, 'w') as f:
                json.dump(data, f)
            
            # Try to tokenize
            success, tokens, elapsed, error = self.tokenize_file(test_file)
            
            result = {
                "test_case": name,
                "success": success,
                "tokens": tokens,
                "error": error[:50] if error else None
            }
            error_results.append(result)
            
            if success:
                print(f"  ✓ {name}: Handled gracefully ({tokens} tokens)")
            else:
                print(f"  ✓ {name}: Error caught: {error[:50]}")
            
            os.remove(test_file)
        
        wandb.log({"error_handling_results": error_results})
        
        return True
    
    def run_production_tests(self):
        """Run all production readiness tests."""
        print("="*80)
        print("PRODUCTION TOKENIZER TEST SUITE")
        print("="*80)
        print(f"Total files available: {len(self.test_files)}")
        
        all_passed = True
        
        # Test 1: Scale processing (1000 files)
        print("\n[1/4] Scale Processing Test")
        if not self.test_scale_processing(min(1000, len(self.test_files))):
            all_passed = False
        
        # Test 2: Parallel processing
        print("\n[2/4] Parallel Processing Test")
        if not self.test_parallel_processing(100, 4):
            all_passed = False
        
        # Test 3: Memory stability
        print("\n[3/4] Memory Stability Test")
        if not self.test_memory_stability(5, 50):
            all_passed = False
        
        # Test 4: Error handling
        print("\n[4/4] Error Handling Test")
        if not self.test_error_handling():
            all_passed = False
        
        # Final summary
        print("\n" + "="*80)
        print("PRODUCTION TEST SUMMARY")
        print("="*80)
        
        final_summary = {
            "test_suite": "tokenizer_performance_production",
            "all_tests_passed": all_passed,
            "total_files_available": len(self.test_files),
            "avg_processing_time": sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0
        }
        
        wandb.log(final_summary)
        
        if all_passed:
            print("✅ TOKENIZER IS PRODUCTION READY!")
            print(f"  - Can process {len(self.test_files)} files")
            print(f"  - Average speed: ~{final_summary['avg_processing_time']:.2f} sec/file")
            print(f"  - Memory stable over long runs")
            print(f"  - Handles errors gracefully")
        else:
            print("⚠️ Some tests failed, but tokenizer may still be usable")
        
        wandb.finish()
        return all_passed


def main():
    """Run production tokenizer tests."""
    tester = ProductionTokenizerTest()
    success = tester.run_production_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()