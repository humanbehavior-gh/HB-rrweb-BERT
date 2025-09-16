# BERT Training Code Production Issues Audit

## 🚨 Critical Issues Found

### 1. **Dataset Loading - Memory Issues**
**Problem**: Pre-tokenizing ALL files into memory
```python
# Line 63-87: Loading everything at once!
for file_path in tqdm(self.files...):
    tokens = self.tokenizer.tokenize_session(events)
    self.tokenized_sessions.append(tokens)  # Stores ALL in memory
```
**Impact**: Will OOM with 30k+ files
**Fix Needed**: Lazy loading, tokenize on-demand in `__getitem__`

### 2. **Missing text_token_offset Attribute**
**Problem**: Line 108 references non-existent attribute
```python
if token_id >= self.tokenizer.text_token_offset:  # This doesn't exist!
```
**Impact**: Will crash when creating token_type_ids
**Fix Needed**: Use actual vocab boundary (520)

### 3. **No Validation Split**
**Problem**: Training without validation set
```python
# Lines 273-275: Random split, not stratified
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
```
**Impact**: Can't properly monitor overfitting
**Fix Needed**: Proper train/val/test splits

### 4. **Loss Always Shows 0.0**
**Problem**: Something wrong with loss computation
**Impact**: Can't track training progress
**Fix Needed**: Debug why loss is always 0

### 5. **Event Type Embeddings Not Used**
**Problem**: Model has event_type_embeddings but never receives event_type_ids
```python
# Line 50: Defined but never used
self.event_type_embeddings = nn.Embedding(10, config.hidden_size)
```
**Impact**: Missing important positional information
**Fix Needed**: Extract and pass event types

### 6. **Inefficient Masking Logic**
**Problem**: Lines 176-180 use multiple Bernoulli calls
```python
indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
```
**Impact**: Not guaranteed 80/10/10 split
**Fix Needed**: Proper probabilistic sampling

### 7. **No Checkpointing for Large Models**
**Problem**: Large model will OOM without gradient checkpointing
**Impact**: Can't train large models
**Fix Needed**: Enable gradient checkpointing properly

### 8. **Missing Data Augmentation**
**Problem**: No augmentation strategies
**Impact**: Model may overfit
**Fix Needed**: Add session chunking, event sampling

### 9. **No Mixed Precision Training**
**Problem**: FP16 enabled but not properly configured
```python
fp16=torch.cuda.is_available(),  # Too simplistic
```
**Impact**: Slower training, more memory usage
**Fix Needed**: Proper AMP setup

### 10. **No Distributed Training Support**
**Problem**: Single GPU only
**Impact**: Can't scale to large datasets
**Fix Needed**: Add DDP support

### 11. **Hardcoded Special Token IDs**
**Problem**: Lines 136-137, 166 hardcode token IDs
```python
pad_token_id: int = 0
mask_token_id: int = 4
special_tokens_mask = (input_ids == self.pad_token_id) | (input_ids < 5)
```
**Impact**: Brittle, will break if tokenizer changes
**Fix Needed**: Get from tokenizer

### 12. **No Learning Rate Scheduling**
**Problem**: Fixed learning rate throughout training
**Impact**: Suboptimal convergence
**Fix Needed**: Add scheduler (cosine, linear decay)

### 13. **Missing Metrics**
**Problem**: Only tracking loss
**Impact**: No perplexity, accuracy, or other metrics
**Fix Needed**: Add comprehensive metrics

### 14. **No Early Stopping**
**Problem**: Always trains full epochs
**Impact**: May overfit
**Fix Needed**: Add early stopping callback

### 15. **Token Type ID Logic Wrong**
**Problem**: Using non-existent text_token_offset
**Impact**: Token types incorrectly assigned
**Fix Needed**: Use vocab.structural_vocab_size (520)

## 🔧 Required Fixes Priority

### HIGH Priority (Blocking):
1. Fix memory issue with dataset loading
2. Fix text_token_offset attribute error  
3. Fix loss computation showing 0.0
4. Fix token type ID assignment

### MEDIUM Priority (Performance):
5. Add proper validation split
6. Fix masking probability logic
7. Add learning rate scheduling
8. Add proper metrics tracking

### LOW Priority (Nice to have):
9. Add distributed training
10. Add data augmentation
11. Add gradient checkpointing
12. Add early stopping

## 📝 Summary

The current implementation is a prototype that will:
- **Crash** on large datasets (memory issues)
- **Not train properly** (loss always 0)
- **Not scale** (no distributed training)
- **Not monitor properly** (no validation metrics)

These need to be fixed before production training!