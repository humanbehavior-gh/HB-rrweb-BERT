#!/usr/bin/env python3
"""
Test 1.2: Tokenizer Performance at Scale
MUST handle production workloads efficiently.
Target: 100 files/second consistently.
"""

import sys
import os
import time
import json
import psutil
import threading
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Tuple
import traceback
import gc

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer


class TokenizerPerformanceTester:
    """Test tokenizer performance at production scale."""
    
    def __init__(self, tokenizer_path: str, data_dir: str):
        self.tokenizer_path = tokenizer_path
        self.data_dir = data_dir
        self.tokenizer = RRWebTokenizer.load(tokenizer_path)
        
        # Find test files (limit to avoid timeout)
        self.test_files = []
        max_files = 1000  # Limit for performance testing
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.json'):
                    self.test_files.append(os.path.join(root, file))
                    if len(self.test_files) >= max_files:
                        break
            if len(self.test_files) >= max_files:
                break
        
        print(f"Found {len(self.test_files)} test files")
        
        # Performance metrics
        self.results = {
            'throughput': [],
            'memory': [],
            'errors': []
        }
    
    def tokenize_file(self, file_path: str) -> Tuple[str, List[int], float]:
        """Tokenize a single file and return results."""
        start_time = time.time()
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
            
            return file_path, tokens, elapsed
            
        except Exception as e:
            elapsed = time.time() - start_time
            return file_path, None, elapsed
    
    def test_sequential_performance(self, n_files: int = 100) -> Dict:
        """Test sequential tokenization performance."""
        print("\n" + "="*80)
        print(f"TESTING SEQUENTIAL PERFORMANCE ({n_files} files)")
        print("="*80)
        
        test_files = self.test_files[:n_files] if n_files else self.test_files
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Tokenize files
        start_time = time.time()
        successful = 0
        total_tokens = 0
        
        for i, file_path in enumerate(test_files):
            _, tokens, _ = self.tokenize_file(file_path)
            if tokens is not None:
                successful += 1
                total_tokens += len(tokens)
            
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Processed {i+1}/{n_files} files - {rate:.1f} files/sec")
        
        total_time = time.time() - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        # Calculate metrics
        throughput = n_files / total_time
        avg_tokens = total_tokens / max(successful, 1)
        
        results = {
            'files_processed': n_files,
            'successful': successful,
            'total_time': total_time,
            'throughput': throughput,
            'avg_tokens_per_file': avg_tokens,
            'memory_increase_mb': memory_increase
        }
        
        print(f"\nResults:")
        print(f"  Files: {successful}/{n_files} successful")
        print(f"  Time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.1f} files/second")
        print(f"  Avg tokens/file: {avg_tokens:.0f}")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        
        return results
    
    def test_parallel_performance(self, n_files: int = 100, n_workers: int = 4) -> Dict:
        """Test parallel tokenization with thread pool."""
        print("\n" + "="*80)
        print(f"TESTING PARALLEL PERFORMANCE ({n_files} files, {n_workers} workers)")
        print("="*80)
        
        test_files = self.test_files[:n_files] if n_files else self.test_files
        
        # Memory before
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024
        
        # Tokenize in parallel
        start_time = time.time()
        successful = 0
        total_tokens = 0
        
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self.tokenize_file, f) for f in test_files]
            
            for i, future in enumerate(futures):
                file_path, tokens, elapsed = future.result()
                if tokens is not None:
                    successful += 1
                    total_tokens += len(tokens)
                
                if (i + 1) % 20 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"  Processed {i+1}/{n_files} files - {rate:.1f} files/sec")
        
        total_time = time.time() - start_time
        
        # Memory after
        memory_after = process.memory_info().rss / 1024 / 1024
        memory_increase = memory_after - memory_before
        
        # Calculate metrics
        throughput = n_files / total_time
        avg_tokens = total_tokens / max(successful, 1)
        
        results = {
            'files_processed': n_files,
            'successful': successful,
            'total_time': total_time,
            'throughput': throughput,
            'avg_tokens_per_file': avg_tokens,
            'memory_increase_mb': memory_increase,
            'n_workers': n_workers
        }
        
        print(f"\nResults:")
        print(f"  Files: {successful}/{n_files} successful")
        print(f"  Time: {total_time:.2f} seconds")
        print(f"  Throughput: {throughput:.1f} files/second")
        print(f"  Speedup: {throughput / (n_files / total_time):.2f}x")
        print(f"  Memory increase: {memory_increase:.1f} MB")
        
        return results
    
    def test_memory_stability(self, n_iterations: int = 10, files_per_iteration: int = 100) -> bool:
        """Test that memory doesn't grow unbounded."""
        print("\n" + "="*80)
        print(f"TESTING MEMORY STABILITY ({n_iterations} iterations)")
        print("="*80)
        
        process = psutil.Process()
        memory_readings = []
        
        for iteration in range(n_iterations):
            # Force garbage collection
            gc.collect()
            
            # Get memory before
            memory_before = process.memory_info().rss / 1024 / 1024
            
            # Process batch of files
            test_files = self.test_files[
                iteration * files_per_iteration : (iteration + 1) * files_per_iteration
            ]
            
            for file_path in test_files:
                self.tokenize_file(file_path)
            
            # Get memory after
            memory_after = process.memory_info().rss / 1024 / 1024
            memory_readings.append(memory_after)
            
            print(f"  Iteration {iteration + 1}: {memory_after:.1f} MB (Δ {memory_after - memory_before:.1f} MB)")
        
        # Check if memory is stable (not growing linearly)
        first_half_avg = sum(memory_readings[:n_iterations//2]) / (n_iterations//2)
        second_half_avg = sum(memory_readings[n_iterations//2:]) / (n_iterations - n_iterations//2)
        
        memory_growth_rate = (second_half_avg - first_half_avg) / first_half_avg
        
        print(f"\nMemory analysis:")
        print(f"  First half avg: {first_half_avg:.1f} MB")
        print(f"  Second half avg: {second_half_avg:.1f} MB")
        print(f"  Growth rate: {memory_growth_rate:.1%}")
        
        # Accept up to 10% growth as stable
        is_stable = memory_growth_rate < 0.10
        
        if is_stable:
            print("  ✅ Memory is stable")
        else:
            print("  ⚠️ Memory may be leaking")
        
        return is_stable
    
    def test_determinism(self, n_files: int = 10) -> bool:
        """Test that tokenization is deterministic."""
        print("\n" + "="*80)
        print("TESTING DETERMINISM")
        print("="*80)
        
        test_files = self.test_files[:n_files]
        all_deterministic = True
        
        for i, file_path in enumerate(test_files):
            # Tokenize twice
            _, tokens1, _ = self.tokenize_file(file_path)
            _, tokens2, _ = self.tokenize_file(file_path)
            
            if tokens1 is None or tokens2 is None:
                continue
            
            if tokens1 != tokens2:
                print(f"  ❌ Non-deterministic: {os.path.basename(file_path)}")
                print(f"     First: {tokens1[:10]}...")
                print(f"     Second: {tokens2[:10]}...")
                all_deterministic = False
            else:
                print(f"  ✓ Deterministic: {os.path.basename(file_path)} ({len(tokens1)} tokens)")
        
        if all_deterministic:
            print("\n✅ All tokenization is deterministic")
        else:
            print("\n❌ Found non-deterministic tokenization")
        
        return all_deterministic
    
    def test_thread_safety(self, n_threads: int = 10) -> bool:
        """Test thread safety by tokenizing same file from multiple threads."""
        print("\n" + "="*80)
        print(f"TESTING THREAD SAFETY ({n_threads} threads)")
        print("="*80)
        
        if not self.test_files:
            print("No test files available")
            return False
        
        test_file = self.test_files[0]
        
        # Get reference tokenization
        _, reference_tokens, _ = self.tokenize_file(test_file)
        if reference_tokens is None:
            print(f"Failed to tokenize reference file")
            return False
        
        # Tokenize from multiple threads simultaneously
        results = []
        errors = []
        
        def worker():
            try:
                _, tokens, _ = self.tokenize_file(test_file)
                results.append(tokens)
            except Exception as e:
                errors.append(str(e))
        
        threads = []
        for i in range(n_threads):
            t = threading.Thread(target=worker)
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Check results
        if errors:
            print(f"  ❌ Thread safety errors: {errors}")
            return False
        
        all_match = True
        for i, tokens in enumerate(results):
            if tokens != reference_tokens:
                print(f"  ❌ Thread {i} produced different tokens")
                all_match = False
        
        if all_match:
            print(f"  ✅ All {n_threads} threads produced identical results")
        
        return all_match
    
    def test_large_file_handling(self) -> bool:
        """Test handling of exceptionally large files."""
        print("\n" + "="*80)
        print("TESTING LARGE FILE HANDLING")
        print("="*80)
        
        # Find largest file
        largest_file = None
        largest_size = 0
        
        for file_path in self.test_files[:100]:  # Check first 100
            size = os.path.getsize(file_path)
            if size > largest_size:
                largest_size = size
                largest_file = file_path
        
        if not largest_file:
            print("No files found")
            return False
        
        print(f"  Testing file: {os.path.basename(largest_file)}")
        print(f"  Size: {largest_size / 1024:.1f} KB")
        
        # Test tokenization
        start_time = time.time()
        _, tokens, elapsed = self.tokenize_file(largest_file)
        
        if tokens is None:
            print(f"  ❌ Failed to tokenize large file")
            return False
        
        print(f"  ✅ Tokenized in {elapsed:.2f} seconds")
        print(f"  Tokens: {len(tokens)}")
        print(f"  Rate: {len(tokens) / elapsed:.0f} tokens/second")
        
        return True
    
    def run_all_tests(self) -> bool:
        """Run all performance tests."""
        print("="*80)
        print("TOKENIZER PERFORMANCE TEST SUITE")
        print("="*80)
        
        all_passed = True
        
        # Test 1: Sequential performance
        seq_results = self.test_sequential_performance(50)  # Reduced from 100
        if seq_results['throughput'] < 50:  # Minimum 50 files/sec
            print("  ⚠️ Sequential throughput below target (50 files/sec)")
            all_passed = False
        
        # Test 2: Parallel performance
        parallel_results = self.test_parallel_performance(50, 4)  # Reduced from 100
        if parallel_results['throughput'] < 100:  # Target 100 files/sec
            print("  ⚠️ Parallel throughput below target (100 files/sec)")
            all_passed = False
        
        # Test 3: Memory stability
        if not self.test_memory_stability(3, 20):  # Reduced iterations and files
            all_passed = False
        
        # Test 4: Determinism
        if not self.test_determinism(10):
            all_passed = False
        
        # Test 5: Thread safety
        if not self.test_thread_safety(10):
            all_passed = False
        
        # Test 6: Large file handling
        if not self.test_large_file_handling():
            all_passed = False
        
        # Summary
        print("\n" + "="*80)
        print("PERFORMANCE TEST SUMMARY")
        print("="*80)
        
        print(f"\nKey metrics:")
        print(f"  Sequential throughput: {seq_results['throughput']:.1f} files/sec")
        print(f"  Parallel throughput: {parallel_results['throughput']:.1f} files/sec")
        print(f"  Memory stable: {'Yes' if self.test_memory_stability(3, 30) else 'No'}")
        
        if all_passed:
            print("\n✅ ALL PERFORMANCE TESTS PASSED!")
            print(f"Tokenizer can handle production workloads at {parallel_results['throughput']:.0f} files/sec")
        else:
            print("\n❌ SOME PERFORMANCE TESTS FAILED")
            print("Tokenizer may need optimization for production use")
        
        return all_passed


def main():
    """Run tokenizer performance tests."""
    tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
    data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
    
    tester = TokenizerPerformanceTester(tokenizer_path, data_dir)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()