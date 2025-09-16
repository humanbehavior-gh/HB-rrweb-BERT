"""
Comprehensive metrics for BERT training evaluation.
Includes perplexity, accuracy, and per-token-type metrics.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import accuracy_score, f1_score
import logging

logger = logging.getLogger(__name__)


class RRWebBERTMetrics:
    """Compute comprehensive metrics for RRWebBERT training."""
    
    def __init__(self, vocab_size: int, structural_vocab_size: int = 520):
        """
        Args:
            vocab_size: Total vocabulary size
            structural_vocab_size: Size of structural vocabulary (boundary for token types)
        """
        self.vocab_size = vocab_size
        self.structural_vocab_size = structural_vocab_size
        
        # Track running statistics
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.total_loss = 0.0
        self.total_tokens = 0
        self.correct_predictions = 0
        self.structural_correct = 0
        self.structural_total = 0
        self.text_correct = 0
        self.text_total = 0
        self.n_batches = 0
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute metrics for HuggingFace trainer.
        
        Args:
            eval_pred: EvalPrediction object with predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        
        # Handle different prediction formats
        if isinstance(predictions, tuple):
            # (loss, logits) format
            logits = predictions[1] if len(predictions) > 1 else predictions[0]
        else:
            logits = predictions
        
        # Get predictions
        preds = np.argmax(logits, axis=-1)
        
        # Flatten everything
        labels_flat = labels.reshape(-1)
        preds_flat = preds.reshape(-1)
        
        # Filter out padding (-100 labels)
        mask = labels_flat != -100
        labels_masked = labels_flat[mask]
        preds_masked = preds_flat[mask]
        
        if len(labels_masked) == 0:
            return {
                'accuracy': 0.0,
                'perplexity': float('inf'),
                'structural_accuracy': 0.0,
                'text_accuracy': 0.0
            }
        
        # Overall accuracy
        accuracy = accuracy_score(labels_masked, preds_masked)
        
        # Per-token-type accuracy
        structural_mask = labels_masked < self.structural_vocab_size
        text_mask = labels_masked >= self.structural_vocab_size
        
        structural_acc = 0.0
        if structural_mask.sum() > 0:
            structural_acc = accuracy_score(
                labels_masked[structural_mask],
                preds_masked[structural_mask]
            )
        
        text_acc = 0.0
        if text_mask.sum() > 0:
            text_acc = accuracy_score(
                labels_masked[text_mask],
                preds_masked[text_mask]
            )
        
        # Compute perplexity from logits
        loss = self._compute_loss(logits, labels)
        perplexity = np.exp(loss) if loss < 10 else float('inf')
        
        return {
            'accuracy': accuracy,
            'perplexity': perplexity,
            'structural_accuracy': structural_acc,
            'text_accuracy': text_acc,
            'loss': loss
        }
    
    def _compute_loss(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Compute cross-entropy loss."""
        # Convert to torch for loss computation
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        # Reshape for loss
        batch_size, seq_len, vocab_size = logits_tensor.shape
        logits_flat = logits_tensor.view(-1, vocab_size)
        labels_flat = labels_tensor.view(-1)
        
        # Compute loss only on non-padding tokens
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
        loss = loss_fct(logits_flat, labels_flat)
        
        return loss.item()
    
    def update_batch(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor,
        labels: torch.Tensor
    ):
        """Update metrics with a batch."""
        self.n_batches += 1
        
        # Get predictions
        preds = torch.argmax(logits, dim=-1)
        
        # Mask for valid tokens
        mask = labels != -100
        
        if mask.sum() == 0:
            return
        
        # Update loss
        self.total_loss += loss.item()
        
        # Update token counts
        self.total_tokens += mask.sum().item()
        
        # Overall accuracy
        correct = (preds[mask] == labels[mask]).sum().item()
        self.correct_predictions += correct
        
        # Per-token-type accuracy
        labels_masked = labels[mask]
        preds_masked = preds[mask]
        
        structural_mask = labels_masked < self.structural_vocab_size
        if structural_mask.sum() > 0:
            structural_correct = (preds_masked[structural_mask] == labels_masked[structural_mask]).sum().item()
            self.structural_correct += structural_correct
            self.structural_total += structural_mask.sum().item()
        
        text_mask = labels_masked >= self.structural_vocab_size
        if text_mask.sum() > 0:
            text_correct = (preds_masked[text_mask] == labels_masked[text_mask]).sum().item()
            self.text_correct += text_correct
            self.text_total += text_mask.sum().item()
    
    def get_metrics(self) -> Dict[str, float]:
        """Get current metrics."""
        if self.n_batches == 0:
            return {
                'loss': 0.0,
                'perplexity': float('inf'),
                'accuracy': 0.0,
                'structural_accuracy': 0.0,
                'text_accuracy': 0.0
            }
        
        avg_loss = self.total_loss / self.n_batches
        perplexity = np.exp(avg_loss) if avg_loss < 10 else float('inf')
        
        accuracy = self.correct_predictions / max(self.total_tokens, 1)
        structural_acc = self.structural_correct / max(self.structural_total, 1)
        text_acc = self.text_correct / max(self.text_total, 1)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'accuracy': accuracy,
            'structural_accuracy': structural_acc,
            'text_accuracy': text_acc,
            'total_tokens': self.total_tokens,
            'n_batches': self.n_batches
        }


class TopKAccuracy:
    """Compute top-k accuracy for masked token prediction."""
    
    @staticmethod
    def compute(
        logits: torch.Tensor,
        labels: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            labels: True labels (batch_size, seq_len)
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        # Get top k predictions
        _, top_k_preds = torch.topk(logits, k=k, dim=-1)
        
        # Expand labels for comparison
        labels_expanded = labels.unsqueeze(-1).expand_as(top_k_preds)
        
        # Check if true label is in top k
        correct = (top_k_preds == labels_expanded).any(dim=-1)
        
        # Mask out padding
        mask = labels != -100
        
        if mask.sum() == 0:
            return 0.0
        
        accuracy = correct[mask].float().mean().item()
        return accuracy


class TokenPredictionAnalyzer:
    """Analyze token prediction patterns for debugging."""
    
    def __init__(self, tokenizer, vocab_size: int):
        """
        Args:
            tokenizer: Tokenizer wrapper
            vocab_size: Total vocabulary size
        """
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        
        # Track prediction statistics
        self.token_confusion = {}  # {true_token: {predicted_token: count}}
        self.token_accuracy = {}   # {token: (correct, total)}
    
    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor
    ):
        """Update statistics with batch predictions."""
        preds_flat = predictions.view(-1)
        labels_flat = labels.view(-1)
        
        # Filter valid predictions
        mask = labels_flat != -100
        preds_masked = preds_flat[mask]
        labels_masked = labels_flat[mask]
        
        for true_token, pred_token in zip(labels_masked.tolist(), preds_masked.tolist()):
            # Update confusion matrix
            if true_token not in self.token_confusion:
                self.token_confusion[true_token] = {}
            
            if pred_token not in self.token_confusion[true_token]:
                self.token_confusion[true_token][pred_token] = 0
            
            self.token_confusion[true_token][pred_token] += 1
            
            # Update accuracy
            if true_token not in self.token_accuracy:
                self.token_accuracy[true_token] = [0, 0]
            
            self.token_accuracy[true_token][1] += 1  # Total
            if true_token == pred_token:
                self.token_accuracy[true_token][0] += 1  # Correct
    
    def get_worst_tokens(self, n: int = 10) -> list:
        """Get tokens with worst prediction accuracy."""
        token_acc_list = []
        
        for token, (correct, total) in self.token_accuracy.items():
            if total > 10:  # Minimum sample size
                accuracy = correct / total
                token_acc_list.append((token, accuracy, total))
        
        # Sort by accuracy (ascending)
        token_acc_list.sort(key=lambda x: x[1])
        
        return token_acc_list[:n]
    
    def get_confusion_patterns(self, token: int, n: int = 5) -> list:
        """Get most common confusion patterns for a token."""
        if token not in self.token_confusion:
            return []
        
        confusions = self.token_confusion[token]
        sorted_confusions = sorted(confusions.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_confusions[:n]