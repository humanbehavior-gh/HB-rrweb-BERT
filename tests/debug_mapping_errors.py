#!/usr/bin/env python3
"""
Focused debugging of the 28 mapping errors.
"""

import sys
import os
import json
import numpy as np
from typing import List, Dict, Tuple

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

def analyze_session_structure(session_tokens: List[int], tokenizer) -> Dict:
    """Analyze the structure of session tokens."""
    analysis = {
        'total_tokens': len(session_tokens),
        'has_cls': False,
        'cls_position': -1,
        'sep_count': 0,
        'sep_positions': [],
        'last_token': None,
        'is_truncated': len(session_tokens) in [8191, 8192]
    }
    
    # Check for [CLS] token (usually 2)
    if session_tokens and session_tokens[0] == 2:
        analysis['has_cls'] = True
        analysis['cls_position'] = 0
    
    # Count and locate [SEP] tokens (usually 3)
    for i, token in enumerate(session_tokens):
        if token == 3:
            analysis['sep_count'] += 1
            analysis['sep_positions'].append(i)
    
    if session_tokens:
        analysis['last_token'] = session_tokens[-1]
    
    return analysis

def debug_specific_file(file_path: str, tokenizer):
    """Debug a specific file in detail."""
    print(f"\n{'='*80}")
    print(f"DEBUGGING: {os.path.basename(file_path)}")
    print('='*80)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    events = data if isinstance(data, list) else data.get('events', [])
    print(f"Number of events: {len(events)}")
    
    # Tokenize session
    session_tokens = tokenizer.tokenize_session(events)
    session_analysis = analyze_session_structure(session_tokens, tokenizer)
    
    print(f"\nSession Tokenization:")
    print(f"  Total tokens: {session_analysis['total_tokens']}")
    print(f"  Has [CLS]: {session_analysis['has_cls']}")
    print(f"  [SEP] count: {session_analysis['sep_count']}")
    print(f"  Is truncated: {session_analysis['is_truncated']}")
    print(f"  Last token: {session_analysis['last_token']}")
    
    # Tokenize events individually
    individual_tokens = []
    event_boundaries = []  # Track where each event starts/ends
    
    for i, event in enumerate(events):
        start_pos = len(individual_tokens)
        event_tokens = tokenizer.tokenize_event(event)
        individual_tokens.extend(event_tokens)
        end_pos = len(individual_tokens)
        event_boundaries.append((start_pos, end_pos, len(event_tokens)))
        
        if i < 5:  # Show first 5 events
            print(f"  Event {i}: {len(event_tokens)} tokens")
    
    print(f"\nIndividual Tokenization:")
    print(f"  Total tokens: {len(individual_tokens)}")
    print(f"  Total events: {len(events)}")
    
    # Calculate expected session length
    # Expected: [CLS] + individual_tokens + (N * [SEP])
    expected_with_seps = 1 + len(individual_tokens) + len(events)
    
    print(f"\nExpected vs Actual:")
    print(f"  Expected (with [CLS] and [SEP]s): {expected_with_seps}")
    print(f"  Actual session length: {len(session_tokens)}")
    print(f"  Difference: {len(session_tokens) - expected_with_seps}")
    
    # Check if it's a near-truncation case
    if expected_with_seps > 8192:
        print(f"\n⚠️ TRUNCATION CASE:")
        print(f"  Expected tokens ({expected_with_seps}) exceed 8192 limit")
        print(f"  Session was {'truncated' if session_analysis['is_truncated'] else 'NOT truncated'}")
        if session_analysis['is_truncated']:
            print(f"  Truncated at: {len(session_tokens)} tokens")
            # Find where truncation happened
            cumulative = 1  # Start with [CLS]
            for i, (start, end, length) in enumerate(event_boundaries):
                cumulative += length + 1  # event + [SEP]
                if cumulative > len(session_tokens):
                    print(f"  Truncation occurred around event {i} of {len(events)}")
                    break
    
    # Check for anomalies
    if session_analysis['sep_count'] != len(events) and not session_analysis['is_truncated']:
        print(f"\n❌ ANOMALY: Expected {len(events)} [SEP] tokens, found {session_analysis['sep_count']}")
    
    # Check if last token is [SEP]
    if session_analysis['last_token'] != 3 and not session_analysis['is_truncated']:
        print(f"\n❌ ANOMALY: Last token is {session_analysis['last_token']}, expected [SEP] (3)")
    
    return {
        'file': os.path.basename(file_path),
        'events': len(events),
        'expected': expected_with_seps,
        'actual': len(session_tokens),
        'difference': len(session_tokens) - expected_with_seps,
        'truncated': session_analysis['is_truncated'],
        'sep_count': session_analysis['sep_count']
    }

def find_problem_files():
    """Find files that have mapping errors."""
    tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
    tokenizer = RRWebTokenizer.load(tokenizer_path)
    data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
    
    print("="*80)
    print("FINDING FILES WITH MAPPING ERRORS")
    print("="*80)
    
    # Get test files
    test_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                test_files.append(os.path.join(root, file))
    
    # Test first 100 files (same as in Test 1.3)
    test_files = test_files[:100]
    
    problem_files = []
    truncated_files = []
    success_files = []
    
    print(f"\nTesting {len(test_files)} files...")
    
    for i, file_path in enumerate(test_files):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            events = data if isinstance(data, list) else data.get('events', [])
            if not events:
                continue
            
            # Tokenize session
            session_tokens = tokenizer.tokenize_session(events)
            
            # Check if truncated
            if len(session_tokens) == 8192 or len(session_tokens) == 8191:
                truncated_files.append(file_path)
                continue
            
            # Tokenize individually
            individual_tokens = []
            for event in events:
                event_tokens = tokenizer.tokenize_event(event)
                individual_tokens.extend(event_tokens)
            
            # Calculate expected
            expected_len = 1 + len(individual_tokens) + len(events)
            actual_len = len(session_tokens)
            
            # Check for mismatch (allow small variance)
            if abs(actual_len - expected_len) > 2:
                problem_files.append({
                    'path': file_path,
                    'events': len(events),
                    'expected': expected_len,
                    'actual': actual_len,
                    'diff': actual_len - expected_len
                })
            else:
                success_files.append(file_path)
                
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
    
    print(f"\nResults:")
    print(f"  Success: {len(success_files)}")
    print(f"  Truncated: {len(truncated_files)}")
    print(f"  Errors: {len(problem_files)}")
    
    # Analyze problem files
    if problem_files:
        print(f"\n{'='*80}")
        print("ANALYZING PROBLEM FILES")
        print('='*80)
        
        # Sort by difference magnitude
        problem_files.sort(key=lambda x: abs(x['diff']), reverse=True)
        
        print("\nTop 5 files with largest differences:")
        for i, pf in enumerate(problem_files[:5]):
            print(f"\n{i+1}. {os.path.basename(pf['path'])}")
            print(f"   Events: {pf['events']}")
            print(f"   Expected: {pf['expected']}")
            print(f"   Actual: {pf['actual']}")
            print(f"   Difference: {pf['diff']:+d}")
        
        # Debug the worst case in detail
        if problem_files:
            worst_case = problem_files[0]
            debug_specific_file(worst_case['path'], tokenizer)
            
            # Debug one more for comparison
            if len(problem_files) > 1:
                debug_specific_file(problem_files[1]['path'], tokenizer)
    
    return problem_files, truncated_files, success_files

def analyze_patterns(problem_files):
    """Look for patterns in the problem files."""
    print(f"\n{'='*80}")
    print("PATTERN ANALYSIS")
    print('='*80)
    
    if not problem_files:
        print("No problem files to analyze")
        return
    
    # Analyze differences
    differences = [pf['diff'] for pf in problem_files]
    
    print(f"\nDifference Statistics:")
    print(f"  Mean: {np.mean(differences):.1f}")
    print(f"  Median: {np.median(differences):.1f}")
    print(f"  Min: {min(differences)}")
    print(f"  Max: {max(differences)}")
    
    # Check if differences are consistent
    positive_diffs = [d for d in differences if d > 0]
    negative_diffs = [d for d in differences if d < 0]
    
    print(f"\nDifference Distribution:")
    print(f"  Positive (actual > expected): {len(positive_diffs)}")
    print(f"  Negative (actual < expected): {len(negative_diffs)}")
    
    # Check for specific patterns
    print(f"\nChecking for patterns...")
    
    # Pattern 1: Files with many events
    many_events = [pf for pf in problem_files if pf['events'] > 100]
    if many_events:
        print(f"  Files with >100 events: {len(many_events)}")
        print(f"    These tend to have diff: {np.mean([pf['diff'] for pf in many_events]):.1f}")
    
    # Pattern 2: Files near truncation limit
    near_truncation = [pf for pf in problem_files if pf['expected'] > 7000]
    if near_truncation:
        print(f"  Files near truncation (>7000 expected): {len(near_truncation)}")
        print(f"    These tend to have diff: {np.mean([pf['diff'] for pf in near_truncation]):.1f}")
    
    # Pattern 3: Check if actual is close to 8191/8190
    near_8192 = [pf for pf in problem_files if 8189 <= pf['actual'] <= 8192]
    if near_8192:
        print(f"  Files with actual tokens near 8192: {len(near_8192)}")
        for pf in near_8192:
            print(f"    {os.path.basename(pf['path'])}: actual={pf['actual']}, expected={pf['expected']}")

def main():
    """Run focused debugging."""
    problem_files, truncated_files, success_files = find_problem_files()
    
    if problem_files:
        analyze_patterns(problem_files)
        
        print(f"\n{'='*80}")
        print("RECOMMENDATIONS")
        print('='*80)
        
        print("\nBased on the analysis:")
        print("1. Some files are being truncated at 8191 instead of 8192")
        print("2. This happens when the last event would exceed the limit")
        print("3. The tokenizer stops BEFORE adding an event that would exceed 8192")
        print("\nSolution options:")
        print("a) Mark these as truncated (8189-8192 range)")
        print("b) Document this behavior clearly")
        print("c) Update the test to handle near-truncation cases")
    else:
        print("\n✅ No mapping errors found beyond truncation!")
    
    return len(problem_files)

if __name__ == "__main__":
    sys.exit(main())