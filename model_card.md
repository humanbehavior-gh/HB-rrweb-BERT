# Model Card: RRWebBERT

## Model Details

### Model Description
RRWebBERT is a BERT-based encoder specifically designed for RRWEB session recordings. It processes structured web interaction data and generates embeddings suitable for downstream tasks like session understanding, anomaly detection, and user behavior analysis.

**Developed by:** Human Behavior Labs  
**Model type:** Masked Language Model (MLM)  
**Language(s):** Multi-modal (RRWEB events + text)  
**License:** Apache 2.0  
**Fine-tuned from:** Trained from scratch  

### Model Architecture
- **Base Model:** BERT architecture
- **Parameters:** ~110M
- **Hidden Size:** 768
- **Layers:** 12
- **Attention Heads:** 12
- **Max Sequence Length:** 2048 tokens
- **Vocabulary Size:** 12,520 tokens
  - Structural tokens (0-519): RRWEB event types, DOM elements
  - BPE text tokens (520-12,519): Text content from DOM nodes
  - Special tokens: [PAD]=0, [UNK]=1, [CLS]=2, [SEP]=3, [MASK]=4

## Uses

### Direct Use
- Generate embeddings for RRWEB session recordings
- Pre-training for downstream tasks on web interaction data
- Session similarity and clustering
- Anomaly detection in user behavior

### Downstream Use
- Fine-tuning for specific task classification
- Integration into trimodal embedding systems (text + image + RRWEB)
- User behavior prediction and analysis
- Session quality assessment

### Out-of-Scope Use
- Natural language processing tasks (use standard BERT models)
- Image processing (this model doesn't process screenshots)
- Real-time streaming analysis (designed for batch processing)

## Bias, Risks, and Limitations

### Technical Limitations
- Maximum sequence length of 2048 tokens may truncate long sessions
- Single GPU training may limit model capacity
- Random window sampling might split semantic boundaries

### Biases
- Trained on specific dataset of 31,951 RRWEB files
- May not generalize to all types of web applications
- Performance depends on quality of RRWEB recordings

### Recommendations
- Validate on your specific domain before production use
- Consider fine-tuning for specialized applications
- Monitor for drift when deployed on new web applications

## Training Details

### Training Data
- **Dataset Size:** 31,951 RRWEB session files
- **Data Split:** 
  - Training: 28,750 files (90%)
  - Validation: 3,195 files (10%)
- **Preprocessing:** 
  - Structural token extraction from RRWEB events
  - BPE tokenization for text content
  - Random window sampling (2048 tokens) from long sessions

### Training Procedure

#### Training Hyperparameters
- **Batch Size:** 32
- **Learning Rate:** 5e-5
- **Weight Decay:** 0.01
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing with Warm Restarts (T_0=10, T_mult=2)
- **MLM Probability:** 15%
- **Masking Strategy:** 80% mask, 10% random, 10% unchanged
- **Mixed Precision:** FP16 with GradScaler
- **Gradient Clipping:** 1.0

#### Hardware
- **GPU:** Single NVIDIA H100 80GB
- **Training Time:** ~2-3 days for 10 epochs
- **Peak Memory:** ~69GB

#### Software
- **Framework:** PyTorch 2.0+
- **Transformers:** Hugging Face Transformers
- **Additional:** WandB for monitoring

### Training Innovations
- **Random Window Sampling:** Prevents bias of only seeing session starts
- **Lazy Loading:** Memory-efficient data loading with LRU cache
- **Hybrid Tokenization:** Combines structural and text understanding

## Evaluation

### Testing Data
Reserved test set from production RRWEB recordings (separate from training/validation)

### Metrics
- **Perplexity:** Measured on masked token prediction
- **MLM Accuracy:** Accuracy of masked token predictions
- **Validation Loss:** Cross-entropy loss on validation set

### Results
- Training monitored via WandB project: `rrweb-bert-production`
- Best model selected based on validation loss
- Checkpoints saved every epoch

## Environmental Impact
- **Hardware Type:** NVIDIA H100 80GB
- **Hours used:** ~60-72 hours
- **Cloud Provider:** AWS/Local cluster
- **Carbon Efficiency:** Optimized single-GPU training

## Technical Specifications

### Model Input Format
```python
{
    "input_ids": [2, 145, 267, ..., 3],  # Token IDs with [CLS] and [SEP]
    "attention_mask": [1, 1, 1, ..., 0],  # 1 for real tokens, 0 for padding
    "token_type_ids": [0, 0, 1, ..., 0],  # 0 for structural, 1 for text tokens
}
```

### Model Output Format
```python
{
    "loss": 2.45,  # MLM loss (during training)
    "logits": torch.Tensor,  # [batch_size, seq_length, vocab_size]
    "hidden_states": torch.Tensor,  # [batch_size, seq_length, hidden_size]
}
```

## Citation

```bibtex
@software{rrweb_bert_2024,
  title = {RRWebBERT: BERT-based Encoder for Web Session Recordings},
  author = {HumanBehaviorLabs},
  year = {2024},
  url = {https://github.com/HumanBehaviorLabs/rrweb-bert},
  version = {1.0.0}
}
```

## Model Card Authors
HumanBehaviorLabs Team

## Model Card Contact
For questions and feedback: [Create an issue on GitHub](https://github.com/HumanBehaviorLabs/rrweb-bert/issues)

## Updates
- **2024-01-16:** Initial model release with random window sampling
- **2024-01-16:** Production training on 31,951 RRWEB files