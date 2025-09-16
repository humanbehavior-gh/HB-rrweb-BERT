#!/usr/bin/env python3
"""
Debug the token mapping issue.
"""

import sys
import os
import json

sys.path.append('/home/ubuntu/rrweb_tokenizer')
from rrweb_tokenizer import RRWebTokenizer

def debug_mapping():
    """Debug token mapping mismatch."""
    tokenizer_path = '/home/ubuntu/rrweb_tokenizer/tokenizer_model_latest'
    tokenizer = RRWebTokenizer.load(tokenizer_path)
    
    # Test with one of the problematic files
    test_file = '/home/ubuntu/embeddingV2/rrweb_data/01993247-772f-7136-9b22-1e27dc68ca33.json'
    
    print(f"Testing file: {test_file}")
    
    with open(test_file, 'r') as f:
        data = json.load(f)
    
    events = data if isinstance(data, list) else data.get('events', [])
    print(f"Number of events: {len(events)}")
    
    # Method 1: tokenize_session
    session_tokens = tokenizer.tokenize_session(events)
    print(f"\nSession tokenization: {len(session_tokens)} tokens")
    print(f"First 20 tokens: {session_tokens[:20]}")
    print(f"Last 20 tokens: {session_tokens[-20:]}")
    
    # Method 2: tokenize individual events
    individual_tokens = []
    for i, event in enumerate(events[:10]):  # Test first 10 events
        event_tokens = tokenizer.tokenize_event(event)
        individual_tokens.extend(event_tokens)
        print(f"Event {i}: {len(event_tokens)} tokens")
    
    print(f"\nIndividual tokenization (first 10 events): {len(individual_tokens)} tokens")
    print(f"First 20 tokens: {individual_tokens[:20]}")
    
    # Check if tokenize_session adds special tokens
    print("\n=== Checking for special tokens ===")
    if len(session_tokens) > 0:
        if session_tokens[0] == 2:  # [CLS]
            print("Session starts with [CLS] token")
        if session_tokens[-1] == 3:  # [SEP]
            print("Session ends with [SEP] token")
    
    # Check if there's truncation
    print(f"\n=== Checking for truncation ===")
    print(f"Session token count: {len(session_tokens)}")
    if len(session_tokens) == 8192:
        print("Session might be truncated at 8192 tokens!")
    
    # Tokenize all events individually
    all_individual = []
    for event in events:
        event_tokens = tokenizer.tokenize_event(event)
        all_individual.extend(event_tokens)
    
    print(f"\nAll events tokenized individually: {len(all_individual)} tokens")
    
    # Compare
    print(f"\n=== COMPARISON ===")
    print(f"Session tokenization: {len(session_tokens)} tokens")
    print(f"Individual tokenization: {len(all_individual)} tokens")
    print(f"Difference: {len(session_tokens) - len(all_individual)} tokens")
    
    # Check if session tokenizer has a max_length
    if hasattr(tokenizer, 'max_length'):
        print(f"Tokenizer max_length: {tokenizer.max_length}")

if __name__ == "__main__":
    debug_mapping()