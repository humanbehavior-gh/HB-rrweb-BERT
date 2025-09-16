"""
Simple wrapper for RRWebTokenizer to provide consistent interface.
"""

class TokenizerWrapper:
    """Wrapper to provide consistent tokenizer interface."""
    
    def __init__(self, tokenizer):
        """Initialize with RRWebTokenizer instance."""
        self.tokenizer = tokenizer
        self.vocab = tokenizer.vocab  # Expose vocab directly
        # Get total vocab size (structural + BPE)
        self.vocab_size = tokenizer.vocab.structural_vocab_size + 12000  # 520 structural + 12000 BPE
        
    def tokenize_session(self, events):
        """Tokenize a session of events."""
        return self.tokenizer.tokenize_session(events)
        
    @classmethod
    def from_pretrained(cls, path):
        """Load tokenizer from path."""
        import sys
        sys.path.append('/home/ubuntu/rrweb_tokenizer')
        from rrweb_tokenizer import RRWebTokenizer
        tokenizer = RRWebTokenizer.load(path)
        return cls(tokenizer)