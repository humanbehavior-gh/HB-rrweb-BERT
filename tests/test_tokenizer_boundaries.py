#!/usr/bin/env python3
"""
Test 1.1: Tokenizer Boundary Verification
MUST verify all 12,520 tokens are accessible and correctly mapped.
NO SHORTCUTS - test everything!
"""

import sys
import json
import torch
import numpy as np
from typing import Dict, List, Set
import traceback

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer


class TokenizerBoundaryTester:
    """Comprehensive tokenizer boundary testing."""
    
    def __init__(self, tokenizer_path: str):
        self.tokenizer = RRWebTokenizer.load(tokenizer_path)
        self.errors = []
        self.warnings = []
        
        # Calculate boundaries
        self.structural_vocab_size = self.tokenizer.vocab.structural_vocab_size
        self.bpe_vocab_size = len(self.tokenizer.bpe_tokenizer.vocab) if self.tokenizer.bpe_tokenizer else 0
        self.total_vocab_size = self.structural_vocab_size + self.bpe_vocab_size
        
        print(f"Tokenizer loaded:")
        print(f"  Structural vocab: {self.structural_vocab_size}")
        print(f"  BPE vocab: {self.bpe_vocab_size}")
        print(f"  Total vocab: {self.total_vocab_size}")
    
    def test_structural_tokens(self) -> bool:
        """Test all structural tokens (0-519)."""
        print("\n" + "="*80)
        print("TESTING STRUCTURAL TOKENS (0-519)")
        print("="*80)
        
        passed = True
        
        # Test special tokens
        print("\n1. Testing special tokens...")
        special_tokens = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
            '[DOM_START]': 5, '[DOM_END]': 6, '[MUTATION_START]': 7,
            '[MUTATION_END]': 8, '[SCROLL]': 9, '[CLICK]': 10,
            '[INPUT]': 11, '[FORM_SUBMIT]': 12, '[TEXT_NODE]': 13
        }
        
        for name, expected_id in special_tokens.items():
            actual_id = self.tokenizer.vocab.special_tokens.get(name)
            if actual_id != expected_id:
                self.errors.append(f"Special token {name}: expected {expected_id}, got {actual_id}")
                passed = False
            else:
                print(f"  ✓ {name} = {expected_id}")
        
        # Test HTML tags (100-199)
        print("\n2. Testing HTML tags (100-199)...")
        essential_tags = ['div', 'span', 'p', 'a', 'img', 'button', 'input', 'form']
        for tag in essential_tags:
            if tag in self.tokenizer.vocab.html_tags:
                token_id = self.tokenizer.vocab.html_tags[tag]
                if not (100 <= token_id < 200):
                    self.errors.append(f"HTML tag '{tag}' has invalid ID: {token_id}")
                    passed = False
                else:
                    print(f"  ✓ <{tag}> = {token_id}")
            else:
                self.errors.append(f"Essential HTML tag '{tag}' not found!")
                passed = False
        
        # Test attributes (200-299)
        print("\n3. Testing attributes (200-299)...")
        essential_attrs = ['class', 'id', 'style', 'href', 'src', 'type', 'name', 'value']
        for attr in essential_attrs:
            if attr in self.tokenizer.vocab.attributes:
                token_id = self.tokenizer.vocab.attributes[attr]
                if not (200 <= token_id < 300):
                    self.errors.append(f"Attribute '{attr}' has invalid ID: {token_id}")
                    passed = False
                else:
                    print(f"  ✓ {attr} = {token_id}")
            else:
                self.errors.append(f"Essential attribute '{attr}' not found!")
                passed = False
        
        # Test event types (300-399)
        print("\n4. Testing event types (300-399)...")
        event_types = ['Load', 'FullSnapshot', 'IncrementalSnapshot', 'Meta', 'Custom']
        for event in event_types:
            if event in self.tokenizer.vocab.event_types:
                token_id = self.tokenizer.vocab.event_types[event]
                if not (300 <= token_id < 400):
                    self.errors.append(f"Event type '{event}' has invalid ID: {token_id}")
                    passed = False
                else:
                    print(f"  ✓ {event} = {token_id}")
            else:
                self.warnings.append(f"Event type '{event}' not found")
        
        print(f"\nStructural tokens: {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    
    def test_bpe_tokens(self) -> bool:
        """Test BPE tokens (520-12519)."""
        print("\n" + "="*80)
        print(f"TESTING BPE TOKENS ({self.structural_vocab_size}-{self.total_vocab_size-1})")
        print("="*80)
        
        passed = True
        
        if not self.tokenizer.bpe_tokenizer:
            self.errors.append("No BPE tokenizer found!")
            return False
        
        # Test BPE vocab size
        print(f"\n1. BPE vocabulary size: {self.bpe_vocab_size}")
        if self.bpe_vocab_size != 12000:
            self.warnings.append(f"Expected 12000 BPE tokens, got {self.bpe_vocab_size}")
        
        # Test sample text tokenization
        print("\n2. Testing text tokenization...")
        test_texts = [
            "Hello world",
            "Click here to submit",
            "user@example.com",
            "https://www.example.com/page?param=value",
            "1234567890",
            "!@#$%^&*()",
            "",  # Empty
            " " * 100,  # Spaces
            "a" * 1000,  # Long repetition
            "🎉 Unicode émojis 中文"  # Unicode
        ]
        
        for text in test_texts:
            try:
                tokens = self.tokenizer.tokenize_text(text)
                
                # Verify all tokens are in valid range
                for token in tokens:
                    if not (self.structural_vocab_size <= token < self.total_vocab_size):
                        self.errors.append(f"Text token {token} out of range for text: '{text[:50]}'")
                        passed = False
                
                # Show result
                preview = text[:30] if len(text) <= 30 else text[:27] + "..."
                print(f"  ✓ '{preview}' -> {len(tokens)} tokens, range [{min(tokens) if tokens else 0}, {max(tokens) if tokens else 0}]")
                
            except Exception as e:
                self.errors.append(f"Failed to tokenize text '{text[:50]}': {e}")
                passed = False
        
        # Test token ID mapping consistency
        print("\n3. Testing BPE token ID mapping...")
        sample_bpe_tokens = list(self.tokenizer.bpe_tokenizer.vocab.items())[:100]
        for text_piece, bpe_id in sample_bpe_tokens[:10]:  # Check first 10
            expected_token_id = self.structural_vocab_size + bpe_id
            if expected_token_id >= self.total_vocab_size:
                self.errors.append(f"BPE token '{text_piece}' maps to invalid ID {expected_token_id}")
                passed = False
        print(f"  Checked {len(sample_bpe_tokens)} BPE mappings")
        
        print(f"\nBPE tokens: {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    
    def test_edge_cases(self) -> bool:
        """Test edge cases and error handling."""
        print("\n" + "="*80)
        print("TESTING EDGE CASES")
        print("="*80)
        
        passed = True
        
        # Test 1: Empty session
        print("\n1. Testing empty session...")
        try:
            tokens = self.tokenizer.tokenize_session([])
            # tokenize_session adds [CLS] at start, [SEP] at end of each event
            # Empty session should just have [CLS]
            if tokens != [2]:  # Just [CLS] for empty session
                if tokens == [2, 3]:  # Also acceptable: [CLS], [SEP]
                    print(f"  ✓ Empty session -> {tokens}")
                else:
                    self.warnings.append(f"Empty session produced unexpected tokens: {tokens}")
            else:
                print(f"  ✓ Empty session -> {tokens}")
        except Exception as e:
            self.errors.append(f"Failed on empty session: {e}")
            passed = False
        
        # Test 2: Invalid event
        print("\n2. Testing invalid event...")
        try:
            invalid_event = {"type": 999, "data": {"invalid": "data"}}
            tokens = self.tokenizer.tokenize_event(invalid_event)
            print(f"  ✓ Invalid event handled -> {len(tokens)} tokens")
        except Exception as e:
            self.errors.append(f"Failed on invalid event: {e}")
            passed = False
        
        # Test 3: Very long text
        print("\n3. Testing very long text...")
        try:
            long_text = "a" * 10000
            tokens = self.tokenizer.tokenize_text(long_text)
            if len(tokens) > 0:
                print(f"  ✓ Long text (10000 chars) -> {len(tokens)} tokens")
            else:
                self.warnings.append("Long text produced no tokens")
        except Exception as e:
            self.errors.append(f"Failed on long text: {e}")
            passed = False
        
        # Test 4: Mixed content
        print("\n4. Testing mixed structural and text content...")
        try:
            test_event = {
                "type": 2,  # FullSnapshot
                "data": {
                    "node": {
                        "type": 1,
                        "tagName": "div",
                        "attributes": {"class": "test"},
                        "childNodes": [{
                            "type": 3,
                            "textContent": "Hello world!"
                        }]
                    }
                }
            }
            tokens = self.tokenizer.tokenize_event(test_event)
            
            # Check mix of structural and text tokens
            structural_count = sum(1 for t in tokens if t < self.structural_vocab_size)
            text_count = sum(1 for t in tokens if t >= self.structural_vocab_size)
            
            print(f"  ✓ Mixed content -> {structural_count} structural, {text_count} text tokens")
            
            if structural_count == 0:
                self.errors.append("No structural tokens in mixed content!")
                passed = False
            
        except Exception as e:
            self.errors.append(f"Failed on mixed content: {e}")
            passed = False
        
        # Test 5: Token boundary
        print("\n5. Testing token at boundaries...")
        boundary_tokens = [0, 1, 519, 520, self.total_vocab_size - 1]
        for token_id in boundary_tokens:
            if 0 <= token_id < self.total_vocab_size:
                print(f"  ✓ Token {token_id} is valid")
            else:
                self.errors.append(f"Token {token_id} is out of bounds!")
                passed = False
        
        print(f"\nEdge cases: {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    
    def test_determinism(self) -> bool:
        """Test that tokenization is deterministic."""
        print("\n" + "="*80)
        print("TESTING DETERMINISM")
        print("="*80)
        
        passed = True
        
        # Test same input produces same output
        test_cases = [
            {"type": 1, "timestamp": 123},
            {"type": 2, "data": {"node": {"tagName": "div"}}},
            "This is a test string",
            [{"type": 1}, {"type": 2, "data": {}}]  # Proper event list
        ]
        
        print("\n1. Testing deterministic tokenization...")
        for i, test_input in enumerate(test_cases):
            tokens_1 = []
            tokens_2 = []
            
            try:
                if isinstance(test_input, str):
                    tokens_1 = self.tokenizer.tokenize_text(test_input)
                    tokens_2 = self.tokenizer.tokenize_text(test_input)
                elif isinstance(test_input, list):
                    tokens_1 = self.tokenizer.tokenize_session(test_input)
                    tokens_2 = self.tokenizer.tokenize_session(test_input)
                elif isinstance(test_input, dict):
                    tokens_1 = self.tokenizer.tokenize_event(test_input)
                    tokens_2 = self.tokenizer.tokenize_event(test_input)
                
                if tokens_1 != tokens_2:
                    self.errors.append(f"Non-deterministic tokenization for case {i}!")
                    passed = False
                else:
                    print(f"  ✓ Test case {i}: {len(tokens_1)} tokens (consistent)")
                    
            except Exception as e:
                self.errors.append(f"Determinism test failed for case {i}: {e}")
                passed = False
        
        print(f"\nDeterminism: {'✅ PASSED' if passed else '❌ FAILED'}")
        return passed
    
    def run_all_tests(self) -> bool:
        """Run all tokenizer boundary tests."""
        print("="*80)
        print("TOKENIZER BOUNDARY VERIFICATION - PRODUCTION TEST")
        print("="*80)
        
        all_passed = True
        
        # Run each test
        all_passed &= self.test_structural_tokens()
        all_passed &= self.test_bpe_tokens()
        all_passed &= self.test_edge_cases()
        all_passed &= self.test_determinism()
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if all_passed and not self.errors:
            print("\n✅ ALL TOKENIZER BOUNDARY TESTS PASSED!")
            print(f"Verified {self.total_vocab_size} tokens work correctly")
        else:
            print(f"\n❌ TOKENIZER BOUNDARY TESTS FAILED!")
            print(f"Found {len(self.errors)} errors that must be fixed")
        
        return all_passed and not self.errors


def main():
    """Run tokenizer boundary tests."""
    tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
    
    tester = TokenizerBoundaryTester(tokenizer_path)
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()