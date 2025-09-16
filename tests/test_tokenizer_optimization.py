#!/usr/bin/env python3
"""
Analyze and optimize tokenizer performance.
Goal: Achieve 100 files/second for production.
"""

import sys
import os
import time
import json
import cProfile
import pstats
from io import StringIO

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer


def profile_tokenization():
    """Profile tokenizer to find bottlenecks."""
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest')
    
    # Find a test file
    data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
    test_file = None
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                test_file = os.path.join(root, file)
                break
        if test_file:
            break
    
    print(f"Test file: {test_file}")
    
    # Load events
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    events = []
    if isinstance(data, list):
        events = data
    elif isinstance(data, dict) and 'events' in data:
        events = data['events']
    
    print(f"Events: {len(events)}")
    
    # Profile tokenization
    profiler = cProfile.Profile()
    
    print("\nProfiling tokenization...")
    profiler.enable()
    
    for _ in range(5):  # Run 5 times
        tokens = tokenizer.tokenize_session(events)
    
    profiler.disable()
    
    # Print stats
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    
    print("\nTop time-consuming functions:")
    print(s.getvalue())
    
    # Measure different parts
    print("\n" + "="*60)
    print("Component timing analysis:")
    
    # Test event tokenization
    start = time.time()
    for _ in range(10):
        for event in events[:10]:  # First 10 events
            tokenizer.tokenize_event(event)
    event_time = (time.time() - start) / 100  # Per event
    print(f"  Event tokenization: {event_time*1000:.2f} ms/event")
    
    # Test text tokenization
    test_text = "This is a sample text for testing tokenization performance"
    start = time.time()
    for _ in range(100):
        tokenizer.tokenize_text(test_text)
    text_time = (time.time() - start) / 100
    print(f"  Text tokenization: {text_time*1000:.2f} ms/call")
    
    # Test DOM node tokenization
    if events and 'data' in events[0] and 'node' in events[0]['data']:
        node = events[0]['data']['node']
        start = time.time()
        for _ in range(100):
            tokenizer._tokenize_dom_node(node)
        node_time = (time.time() - start) / 100
        print(f"  DOM node tokenization: {node_time*1000:.2f} ms/node")
    
    return tokens


def test_batch_tokenization():
    """Test if batch processing improves performance."""
    print("\n" + "="*60)
    print("Testing batch vs sequential tokenization:")
    
    tokenizer = RRWebTokenizer.load('/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest')
    
    # Get test files
    data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
    test_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                test_files.append(os.path.join(root, file))
                if len(test_files) >= 10:
                    break
        if len(test_files) >= 10:
            break
    
    # Sequential tokenization
    start = time.time()
    sequential_tokens = []
    for file_path in test_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
        events = data if isinstance(data, list) else data.get('events', [])
        tokens = tokenizer.tokenize_session(events)
        sequential_tokens.append(len(tokens))
    seq_time = time.time() - start
    
    print(f"Sequential (10 files): {seq_time:.2f}s = {10/seq_time:.1f} files/sec")
    print(f"  Tokens: {sequential_tokens}")
    
    # Check if we can optimize by reusing tokenizer state
    # (Currently not implemented, but this is where we'd test it)


def suggest_optimizations():
    """Suggest optimizations based on profiling."""
    print("\n" + "="*60)
    print("OPTIMIZATION SUGGESTIONS:")
    print("="*60)
    
    suggestions = [
        "1. Cache BPE tokenization results for common strings",
        "2. Use numpy arrays instead of lists for token operations",
        "3. Implement batch tokenization for multiple files",
        "4. Pre-compile regex patterns if using any",
        "5. Use multiprocessing for parallel tokenization",
        "6. Optimize DOM traversal with iterative instead of recursive",
        "7. Use C extension for hot path (BPE merging)",
        "8. Implement token caching for repeated elements",
        "9. Use memory-mapped files for large datasets",
        "10. Profile and optimize the BPE tokenizer itself"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")
    
    print("\nCurrent bottleneck analysis:")
    print("  - BPE tokenization is likely the slowest part")
    print("  - DOM traversal may be inefficient for deep trees")
    print("  - String operations in Python are slow")
    print("  - No caching of repeated patterns")


def main():
    """Run performance analysis."""
    print("TOKENIZER PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Profile tokenization
    tokens = profile_tokenization()
    print(f"\nTokenized to {len(tokens)} tokens")
    
    # Test batch processing
    test_batch_tokenization()
    
    # Suggest optimizations
    suggest_optimizations()
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("  Current performance: ~1-2 files/second")
    print("  Target performance: 100 files/second")
    print("  Required speedup: 50-100x")
    print("\nRecommendation: Implement caching and batch processing first,")
    print("then consider C extensions for BPE if still needed.")


if __name__ == "__main__":
    main()