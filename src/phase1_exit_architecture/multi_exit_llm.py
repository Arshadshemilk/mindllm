"""
Phase 1: Core Multi-Exit Architecture & Cache Management
Implements multi-exit heads and KV cache propagation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, PreTrainedModel
from dataclasses import dataclass


@dataclass
class ExitOutput:
    """Container for multi-exit forward pass output"""
    logits: torch.Tensor  # Predicted logits
    exit_layer: int  # Layer at which exit occurred
    exit_confidence: float  # Confidence score
    hidden_state: torch.Tensor  # Hidden state at exit layer
    kv_cache: Dict[str, torch.Tensor]  # KV cache for continuation


class ExitHead(nn.Module):
    """
    Exit head for generating predictions at intermediate layers.
    Takes hidden state and produces logits.
    """
    
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_state: [batch_size, seq_len, hidden_dim]
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        hidden = torch.relu(self.dense(hidden_state))
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


class MultiExitLLM(nn.Module):
    """
    Multi-exit LLM wrapper that injects exit heads at designated layers.
    Supports early exiting with KV cache propagation.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        hidden_dim: int,
        exit_layer_indices: List[int],
        vocab_size: int
    ):
        super().__init__()
        self.base_model = base_model
        self.hidden_dim = hidden_dim
        self.exit_layer_indices = sorted(exit_layer_indices)
        self.vocab_size = vocab_size
        
        # Create exit heads for each designated layer
        self.exit_heads = nn.ModuleDict()
        for layer_idx in self.exit_layer_indices:
            self.exit_heads[str(layer_idx)] = ExitHead(hidden_dim, vocab_size)
        
        # Freeze base model weights
        for param in self.base_model.parameters():
            param.requires_grad = False
        
    def forward(
        self,
        input_ids: torch.Tensor,
        exit_threshold: Optional[float] = None,
        past_key_values: Optional[Dict[str, torch.Tensor]] = None,
        return_all_exits: bool = False
    ) -> ExitOutput:
        """
        Forward pass with optional early exiting.
        
        Args:
            input_ids: [batch_size, seq_len]
            exit_threshold: If confidence at any exit layer exceeds this,
                          exit early (None = no early exit)
            past_key_values: KV cache from previous tokens
            return_all_exits: If True, return predictions from all exit layers
            
        Returns:
            ExitOutput containing logits, exit layer, confidence, and updated cache
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Access the transformer model
        transformer = self.base_model
        if hasattr(transformer, 'model'):  # For some model architectures
            transformer = transformer.model
        
        # Get embeddings
        embeddings_fn = self.base_model.get_input_embeddings()
        hidden_state = embeddings_fn(input_ids)
        
        # Initialize attention mask and position ids if needed
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        
        # Cache for KV for continuation
        cached_kv = {}
        exit_logits = None
        exit_layer_idx = None
        exit_confidence = 0.0
        exit_hidden_state = hidden_state
        
        # Process through layers
        for layer_idx, layer in enumerate(transformer.layers if hasattr(transformer, 'layers') else transformer.h if hasattr(transformer, 'h') else []):
            # Forward through layer
            if past_key_values and f"layer_{layer_idx}" in past_key_values:
                # Use cached KV if available
                layer_output = layer(
                    hidden_state,
                    attention_mask=attention_mask,
                    past_key_value=past_key_values[f"layer_{layer_idx}"],
                    use_cache=True
                )
            else:
                layer_output = layer(
                    hidden_state,
                    attention_mask=attention_mask,
                    use_cache=True
                )
            
            hidden_state = layer_output[0]
            
            # Cache KV for this layer
            if len(layer_output) > 1 and layer_output[1] is not None:
                cached_kv[f"layer_{layer_idx}"] = layer_output[1]
            
            # Check for early exit
            if (layer_idx + 1) in self.exit_layer_indices and exit_threshold is not None:
                exit_head = self.exit_heads[str(layer_idx + 1)]
                logits = exit_head(hidden_state)
                
                # Calculate confidence (max softmax probability of last token)
                probs = torch.softmax(logits[:, -1, :], dim=-1)
                confidence = probs.max(dim=-1)[0].mean().item()
                
                if confidence >= exit_threshold:
                    exit_logits = logits
                    exit_layer_idx = layer_idx + 1
                    exit_confidence = confidence
                    exit_hidden_state = hidden_state
                    
                    # Compute KV projections for remaining layers
                    remaining_kv = self._compute_remaining_kv(
                        hidden_state, 
                        layer_idx + 1,
                        transformer.layers if hasattr(transformer, 'layers') else transformer.h
                    )
                    cached_kv.update(remaining_kv)
                    break
        
        # If no early exit, use final layer prediction
        if exit_logits is None:
            exit_layer_idx = self.exit_layer_indices[-1]
            exit_head = self.exit_heads[str(exit_layer_idx)]
            exit_logits = exit_head(hidden_state)
            probs = torch.softmax(exit_logits[:, -1, :], dim=-1)
            exit_confidence = probs.max(dim=-1)[0].mean().item()
            exit_hidden_state = hidden_state
        
        return ExitOutput(
            logits=exit_logits,
            exit_layer=exit_layer_idx,
            exit_confidence=exit_confidence,
            hidden_state=exit_hidden_state,
            kv_cache=cached_kv
        )
    
    def _compute_remaining_kv(
        self,
        hidden_state: torch.Tensor,
        current_layer: int,
        layers: List[nn.Module]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute K and V projections for remaining layers after early exit.
        
        Args:
            hidden_state: Hidden state at exit layer
            current_layer: Current layer index
            layers: All transformer layers
            
        Returns:
            Dictionary of computed KV for remaining layers
        """
        remaining_kv = {}
        
        with torch.no_grad():
            for layer_idx in range(current_layer, len(layers)):
                layer = layers[layer_idx]
                layer_output = layer(
                    hidden_state,
                    use_cache=True
                )
                hidden_state = layer_output[0]
                
                if len(layer_output) > 1 and layer_output[1] is not None:
                    remaining_kv[f"layer_{layer_idx}"] = layer_output[1]
        
        return remaining_kv
    
    def generate_with_early_exit(
        self,
        input_ids: torch.Tensor,
        exit_threshold: float,
        max_new_tokens: int = 128,
        return_exit_info: bool = False
    ) -> Dict:
        """
        Generate tokens with early exiting at each step.
        
        Args:
            input_ids: Initial token ids [batch_size, seq_len]
            exit_threshold: Confidence threshold for early exit
            max_new_tokens: Maximum tokens to generate
            return_exit_info: If True, return info about exits at each step
            
        Returns:
            Dictionary with generated tokens and optional exit information
        """
        generated_ids = input_ids.clone()
        exit_info = []
        
        past_key_values = None
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                output = self.forward(
                    generated_ids[:, -1:],  # Only last token
                    exit_threshold=exit_threshold,
                    past_key_values=past_key_values,
                )
            
            # Get next token
            next_logits = output.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            past_key_values = output.kv_cache
            
            if return_exit_info:
                exit_info.append({
                    'step': step,
                    'exit_layer': output.exit_layer,
                    'confidence': output.exit_confidence
                })
        
        result = {'generated_ids': generated_ids}
        if return_exit_info:
            result['exit_info'] = exit_info
        
        return result
