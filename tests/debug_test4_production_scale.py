#!/usr/bin/env python3
"""
Focused debugging for Test 4: Production Scale failures.
"""

import sys
import os
import json
import time
import traceback
from typing import Dict, List, Optional
import numpy as np

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

def debug_single_file(file_path: str, tokenizer, verbose: bool = True):
    """Debug a single file processing with detailed logging."""
    result = {
        'success': False,
        'error': None,
        'events': 0,
        'tokens': 0,
        'time': 0,
        'error_type': None,
        'error_details': None
    }
    
    start_time = time.time()
    
    try:
        # Step 1: Load file
        if verbose:
            print(f"  Loading: {os.path.basename(file_path)}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Step 2: Extract events
        events = data if isinstance(data, list) else data.get('events', [])
        result['events'] = len(events)
        
        if verbose:
            print(f"    Events: {len(events)}")
        
        if not events:
            result['error'] = "No events in file"
            result['error_type'] = 'EMPTY_FILE'
            return result
        
        # Step 3: Extract event info (as in original test)
        event_types = []
        mutation_types = []
        
        for event in events:
            if not isinstance(event, dict):
                continue
            
            event_type = event.get('type')
            if event_type is not None:
                event_types.append(event_type)
                
                # For incremental snapshots, get mutation type
                if event_type == 3 and 'data' in event:
                    mutation_type = event['data'].get('source')
                    if mutation_type is not None:
                        mutation_types.append(mutation_type)
        
        if verbose:
            print(f"    Event types: {len(set(event_types))} unique")
        
        # Step 4: Tokenize session
        if verbose:
            print(f"    Tokenizing...")
        
        tokens = tokenizer.tokenize_session(events)
        result['tokens'] = len(tokens)
        
        if verbose:
            print(f"    Tokens: {len(tokens)}")
        
        # Success!
        result['success'] = True
        result['time'] = time.time() - start_time
        
        if verbose:
            print(f"    ✅ Success in {result['time']:.3f}s")
        
    except json.JSONDecodeError as e:
        result['error'] = f"JSON decode error: {str(e)}"
        result['error_type'] = 'JSON_ERROR'
        result['error_details'] = {
            'line': e.lineno if hasattr(e, 'lineno') else None,
            'column': e.colno if hasattr(e, 'colno') else None
        }
        if verbose:
            print(f"    ❌ JSON Error: {e}")
    
    except KeyError as e:
        result['error'] = f"Key error: {str(e)}"
        result['error_type'] = 'KEY_ERROR'
        result['error_details'] = {'missing_key': str(e)}
        if verbose:
            print(f"    ❌ Key Error: {e}")
    
    except Exception as e:
        result['error'] = f"Unexpected error: {str(e)}"
        result['error_type'] = 'UNKNOWN_ERROR'
        result['error_details'] = {
            'exception_type': type(e).__name__,
            'traceback': traceback.format_exc()
        }
        if verbose:
            print(f"    ❌ Error: {type(e).__name__}: {e}")
            print(f"    Traceback:\n{traceback.format_exc()}")
    
    result['time'] = time.time() - start_time
    return result

def test_production_scale_debug(n_files: int = 100):
    """Debug production scale test with detailed error tracking."""
    print("="*80)
    print(f"DEBUG: PRODUCTION SCALE TEST ({n_files} files)")
    print("="*80)
    
    # Load tokenizer
    tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
    print(f"\nLoading tokenizer from: {tokenizer_path}")
    tokenizer = RRWebTokenizer.load(tokenizer_path)
    print("✅ Tokenizer loaded")
    
    # Get test files
    data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
    print(f"\nScanning for files in: {data_dir}")
    
    test_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                test_files.append(os.path.join(root, file))
    
    print(f"Found {len(test_files)} total files")
    test_files = test_files[:n_files]
    print(f"Testing first {len(test_files)} files")
    
    # Track statistics
    stats = {
        'processed': 0,
        'failed': 0,
        'error_types': {},
        'processing_times': [],
        'token_counts': [],
        'event_counts': [],
        'failures': []
    }
    
    print("\n" + "-"*80)
    print("PROCESSING FILES")
    print("-"*80)
    
    # Process files with detailed logging
    for i, file_path in enumerate(test_files):
        if i < 10 or i % 100 == 0:  # Verbose for first 10 and every 100th
            print(f"\n[{i+1}/{len(test_files)}] Processing:")
            result = debug_single_file(file_path, tokenizer, verbose=True)
        else:
            result = debug_single_file(file_path, tokenizer, verbose=False)
        
        # Update statistics
        if result['success']:
            stats['processed'] += 1
            stats['processing_times'].append(result['time'])
            stats['token_counts'].append(result['tokens'])
            stats['event_counts'].append(result['events'])
        else:
            stats['failed'] += 1
            error_type = result['error_type']
            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
            
            # Store detailed failure info
            stats['failures'].append({
                'file': os.path.basename(file_path),
                'error': result['error'],
                'error_type': result['error_type'],
                'details': result['error_details']
            })
        
        # Progress update every 10 files
        if (i + 1) % 10 == 0:
            avg_time = np.mean(stats['processing_times'][-10:]) if stats['processing_times'] else 0
            print(f"\n[Progress] {i+1}/{len(test_files)}: "
                  f"Success={stats['processed']}, Failed={stats['failed']}, "
                  f"Avg time={avg_time:.3f}s/file")
    
    # Final statistics
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Total files: {len(test_files)}")
    print(f"  Successful: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Success rate: {100*stats['processed']/len(test_files):.1f}%")
    
    if stats['processing_times']:
        print(f"\nTiming Statistics:")
        print(f"  Mean time: {np.mean(stats['processing_times']):.3f}s")
        print(f"  Median time: {np.median(stats['processing_times']):.3f}s")
        print(f"  Min time: {np.min(stats['processing_times']):.3f}s")
        print(f"  Max time: {np.max(stats['processing_times']):.3f}s")
    
    if stats['token_counts']:
        print(f"\nToken Statistics:")
        print(f"  Mean tokens: {np.mean(stats['token_counts']):.0f}")
        print(f"  Median tokens: {np.median(stats['token_counts']):.0f}")
        print(f"  Min tokens: {np.min(stats['token_counts'])}")
        print(f"  Max tokens: {np.max(stats['token_counts'])}")
    
    if stats['error_types']:
        print(f"\nError Breakdown:")
        for error_type, count in sorted(stats['error_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count} failures")
    
    # Show first few failures in detail
    if stats['failures']:
        print(f"\nFirst 5 Failures (out of {len(stats['failures'])}):")
        for i, failure in enumerate(stats['failures'][:5]):
            print(f"\n  {i+1}. {failure['file']}")
            print(f"     Type: {failure['error_type']}")
            print(f"     Error: {failure['error']}")
            if failure['details'] and 'traceback' not in failure['details']:
                print(f"     Details: {failure['details']}")
    
    return stats

def compare_with_original_test():
    """Compare our debug results with original Test 4 behavior."""
    print("\n" + "="*80)
    print("COMPARING WITH ORIGINAL TEST 4")
    print("="*80)
    
    # Test with exact same logic as original
    tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
    tokenizer = RRWebTokenizer.load(tokenizer_path)
    data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
    
    # Get first file that might be problematic
    test_file = None
    for root, _, files in os.walk(data_dir):
        for file in files[:10]:  # Check first 10
            if file.endswith('.json'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    if test_file:
        print(f"\nTesting single file: {os.path.basename(test_file)}")
        
        try:
            # Exact logic from original test_production_scale
            with open(test_file, 'r') as f:
                data = json.load(f)
            events = data if isinstance(data, list) else data.get('events', [])
            
            print(f"  Events: {len(events)}")
            
            # This is where original test might fail
            # The extract_event_info method is called but might not exist
            print("  Attempting to extract event info...")
            
            # Check if extract_event_info exists
            print("  Note: Original test calls self.extract_event_info(file_path)")
            print("  This method processes events and tracks statistics")
            
            # Try tokenization
            tokens = tokenizer.tokenize_session(events)
            print(f"  Tokens: {len(tokens)}")
            print("  ✅ File processed successfully")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            print(f"  Traceback:\n{traceback.format_exc()}")

def main():
    """Run focused debugging."""
    # First, test a small sample
    print("Phase 1: Testing 20 files with detailed logging")
    stats_small = test_production_scale_debug(20)
    
    if stats_small['failed'] > 0:
        print("\n⚠️ FAILURES DETECTED IN SMALL SAMPLE")
        print("Analyzing failure pattern...")
        
        # Compare with original test logic
        compare_with_original_test()
    else:
        print("\n✅ Small sample successful, testing larger set...")
        # Test larger sample
        stats_large = test_production_scale_debug(100)
        
        if stats_large['failed'] > 0:
            print("\n⚠️ FAILURES IN LARGER SAMPLE")
        else:
            print("\n✅ ALL TESTS PASSING")
    
    return 0 if stats_small['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())