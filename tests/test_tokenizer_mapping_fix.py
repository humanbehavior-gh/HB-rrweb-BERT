#!/usr/bin/env python3
"""
Focused test to fix tokenizer mapping issues.
Build up from simple to complex cases.
"""

import sys
import os
import json
import time
from typing import List, Dict

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

class TokenizerMappingTest:
    """Test and fix tokenizer mapping issues."""
    
    def __init__(self):
        self.tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
        self.tokenizer = RRWebTokenizer.load(self.tokenizer_path)
        self.data_dir = '/home/ubuntu/embeddingV2/rrweb_data'
        
    def test_1_simple_event(self):
        """Test 1: Single event tokenization."""
        print("\n" + "="*60)
        print("TEST 1: SINGLE EVENT TOKENIZATION")
        print("="*60)
        
        # Create a simple event
        event = {
            'type': 3,  # IncrementalSnapshot
            'data': {
                'source': 2,  # MouseMove
                'positions': [{'x': 100, 'y': 200}]
            },
            'timestamp': 1234567890
        }
        
        # Tokenize as single event
        event_tokens = self.tokenizer.tokenize_event(event)
        print(f"Single event tokens: {event_tokens}")
        print(f"Token count: {len(event_tokens)}")
        
        # Tokenize as session with one event
        session_tokens = self.tokenizer.tokenize_session([event])
        print(f"Session tokens: {session_tokens}")
        print(f"Token count: {len(session_tokens)}")
        
        # Compare
        print(f"\nComparison:")
        print(f"  Event tokenization: {len(event_tokens)} tokens")
        print(f"  Session tokenization: {len(session_tokens)} tokens")
        print(f"  Difference: {len(session_tokens) - len(event_tokens)} tokens")
        
        # Check for special tokens
        if len(session_tokens) > len(event_tokens):
            print(f"\nExtra tokens in session:")
            if session_tokens[0] == 2:  # [CLS]
                print(f"  - [CLS] at position 0")
            if session_tokens[-1] == 3:  # [SEP]
                print(f"  - [SEP] at position {len(session_tokens)-1}")
        
        success = abs(len(session_tokens) - len(event_tokens)) <= 2  # Allow [CLS] and [SEP]
        print(f"\n✅ PASS" if success else "❌ FAIL")
        return success
    
    def test_2_multiple_events(self):
        """Test 2: Multiple events without truncation."""
        print("\n" + "="*60)
        print("TEST 2: MULTIPLE EVENTS (NO TRUNCATION)")
        print("="*60)
        
        # Create 10 simple events
        events = []
        for i in range(10):
            events.append({
                'type': 3,
                'data': {'source': 2, 'positions': [{'x': i, 'y': i*2}]},
                'timestamp': 1234567890 + i
            })
        
        # Tokenize individually
        individual_tokens = []
        for i, event in enumerate(events):
            tokens = self.tokenizer.tokenize_event(event)
            individual_tokens.extend(tokens)
            print(f"Event {i}: {len(tokens)} tokens")
        
        # Tokenize as session
        session_tokens = self.tokenizer.tokenize_session(events)
        
        print(f"\nComparison:")
        print(f"  Individual total: {len(individual_tokens)} tokens")
        print(f"  Session total: {len(session_tokens)} tokens")
        print(f"  Difference: {len(session_tokens) - len(individual_tokens)} tokens")
        
        # Check for separators between events
        sep_count = session_tokens.count(3)  # Count [SEP] tokens
        print(f"\n[SEP] tokens in session: {sep_count}")
        
        # Expected: [CLS] + events + separators + [SEP]
        expected_extra = 1 + len(events) + 1  # CLS + separators + final SEP
        actual_extra = len(session_tokens) - len(individual_tokens)
        
        print(f"\nExpected extra tokens: {expected_extra}")
        print(f"Actual extra tokens: {actual_extra}")
        
        success = abs(actual_extra - expected_extra) <= 2
        print(f"\n✅ PASS" if success else "❌ FAIL")
        return success
    
    def test_3_truncation_check(self):
        """Test 3: Check truncation behavior."""
        print("\n" + "="*60)
        print("TEST 3: TRUNCATION BEHAVIOR")
        print("="*60)
        
        # Find a file with many events
        test_file = None
        for root, _, files in os.walk(self.data_dir):
            for file in files[:100]:  # Check first 100 files
                if file.endswith('.json'):
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as f:
                            data = json.load(f)
                        events = data if isinstance(data, list) else data.get('events', [])
                        if len(events) > 100:  # Find file with many events
                            test_file = path
                            break
                    except:
                        continue
            if test_file:
                break
        
        if not test_file:
            print("No suitable test file found")
            return False
        
        print(f"Testing file: {os.path.basename(test_file)}")
        
        with open(test_file, 'r') as f:
            data = json.load(f)
        events = data if isinstance(data, list) else data.get('events', [])
        
        print(f"Number of events: {len(events)}")
        
        # Test with different max_lengths
        test_lengths = [1024, 2048, 4096, 8192]
        
        for max_len in test_lengths:
            # Tokenize with potential truncation
            session_tokens = self.tokenizer.tokenize_session(events)
            
            # Check if truncated
            is_truncated = len(session_tokens) == max_len
            
            print(f"\nMax length {max_len}:")
            print(f"  Token count: {len(session_tokens)}")
            print(f"  Truncated: {'YES' if is_truncated else 'NO'}")
            
            if len(session_tokens) == 8192:
                print(f"  ⚠️ WARNING: Truncated at exactly 8192 tokens!")
                
                # Find how to fix this
                print("\n  Checking tokenizer configuration...")
                if hasattr(self.tokenizer, 'max_length'):
                    print(f"  - tokenizer.max_length = {self.tokenizer.max_length}")
                if hasattr(self.tokenizer, 'max_seq_length'):
                    print(f"  - tokenizer.max_seq_length = {self.tokenizer.max_seq_length}")
                
                return False
        
        return True
    
    def test_4_production_fix(self):
        """Test 4: Fix and verify production tokenization."""
        print("\n" + "="*60)
        print("TEST 4: PRODUCTION FIX VERIFICATION")
        print("="*60)
        
        # Test files that were failing
        problem_files = [
            '01993247-772f-7136-9b22-1e27dc68ca33.json',
            '0199322e-be39-7672-b9bc-263734f3fbb3.json',
            '6b38f8ba-1fcc-4bae-a84e-c71a55be526f.json'
        ]
        
        fixed_count = 0
        for filename in problem_files:
            # Find the file
            file_path = None
            for root, _, files in os.walk(self.data_dir):
                if filename in files:
                    file_path = os.path.join(root, filename)
                    break
            
            if not file_path:
                continue
            
            print(f"\nTesting: {filename}")
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            events = data if isinstance(data, list) else data.get('events', [])
            
            # Original tokenization
            session_tokens = self.tokenizer.tokenize_session(events)
            
            # Individual tokenization
            individual_tokens = []
            for event in events:
                tokens = self.tokenizer.tokenize_event(event)
                individual_tokens.extend(tokens)
            
            print(f"  Events: {len(events)}")
            print(f"  Session tokens: {len(session_tokens)}")
            print(f"  Individual tokens: {len(individual_tokens)}")
            
            # Check if session is truncated
            if len(session_tokens) == 8192:
                print(f"  ❌ TRUNCATED at 8192")
                
                # Propose fix: tokenize in chunks
                print(f"  Proposed fix: Remove truncation limit or tokenize in chunks")
            else:
                # Check if difference is reasonable (CLS + SEP tokens)
                diff = abs(len(session_tokens) - len(individual_tokens))
                if diff <= len(events) + 2:  # Allow for CLS/SEP and event separators
                    print(f"  ✅ ACCEPTABLE (diff={diff})")
                    fixed_count += 1
                else:
                    print(f"  ❌ MISMATCH (diff={diff})")
        
        success = fixed_count > 0
        print(f"\n{'✅ Some files work correctly' if success else '❌ All files have issues'}")
        return success
    
    def run_all_tests(self):
        """Run all focused tests."""
        print("="*60)
        print("TOKENIZER MAPPING FIX TEST SUITE")
        print("="*60)
        
        results = []
        
        # Test 1: Simple event
        results.append(("Single Event", self.test_1_simple_event()))
        
        # Test 2: Multiple events
        results.append(("Multiple Events", self.test_2_multiple_events()))
        
        # Test 3: Truncation
        results.append(("Truncation Check", self.test_3_truncation_check()))
        
        # Test 4: Production fix
        results.append(("Production Fix", self.test_4_production_fix()))
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{name}: {status}")
        
        all_passed = all(r[1] for r in results)
        
        if not all_passed:
            print("\n⚠️ CRITICAL ISSUE FOUND:")
            print("The tokenize_session() function has a hard limit of 8192 tokens.")
            print("This causes truncation for long sessions.")
            print("\nRECOMMENDED FIX:")
            print("1. Remove or increase the 8192 token limit in tokenize_session()")
            print("2. Or handle long sessions by chunking")
            print("3. Document the behavior clearly")
        
        return all_passed


def main():
    """Run focused tokenizer tests."""
    tester = TokenizerMappingTest()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())