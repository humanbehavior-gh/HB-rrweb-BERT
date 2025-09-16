"""RRWebBERT configuration"""

from transformers import PretrainedConfig


class RRWebBERTConfig(PretrainedConfig):
    """
    Configuration class for RRWebBERT model.
    
    Args:
        vocab_size: Vocabulary size of the RRWebBERT model (structural tokens + BPE tokens)
        hidden_size: Dimensionality of encoder layers
        num_hidden_layers: Number of hidden layers in the Transformer encoder
        num_attention_heads: Number of attention heads for each attention layer
        intermediate_size: Dimensionality of the "intermediate" (feed-forward) layer
        hidden_act: Non-linear activation function
        hidden_dropout_prob: Dropout probability for fully connected layers
        attention_probs_dropout_prob: Dropout probability for attention weights
        max_position_embeddings: Maximum sequence length
        type_vocab_size: Vocabulary size for token types (structural vs text)
        initializer_range: Standard deviation for weight initialization
        layer_norm_eps: Epsilon for layer normalization
        position_embedding_type: Type of position embeddings
        use_cache: Whether to use key/value cache for generation
        classifier_dropout: Dropout for classification head
        
        structural_vocab_size: Size of structural vocabulary (HTML, events, etc.)
        text_vocab_size: Size of BPE text vocabulary
        max_dom_depth: Maximum DOM tree depth for traversal
        event_type_embeddings: Whether to use separate embeddings for event types
    """
    
    model_type = "rrweb-bert"
    
    def __init__(
        self,
        vocab_size=13000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=8192,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        # RRWeb specific parameters
        structural_vocab_size=1000,
        text_vocab_size=12000,
        max_dom_depth=20,
        event_type_embeddings=True,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        
        # RRWeb specific
        self.structural_vocab_size = structural_vocab_size
        self.text_vocab_size = text_vocab_size
        self.max_dom_depth = max_dom_depth
        self.event_type_embeddings = event_type_embeddings