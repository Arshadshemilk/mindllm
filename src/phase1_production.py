"""
Real Production Pipeline for Phase 1: Multi-Exit LLM Training & Inference

This is the actual codebase for loading Gemma 3 1B, adding exit heads,
and managing token generation with early exiting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExitHead(nn.Module):
    """Exit head for generating predictions at intermediate layers"""
    
    def __init__(self, hidden_dim: int, vocab_size: int, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Generate logits from hidden state"""
        hidden = F.relu(self.dense(hidden_state))
        hidden = self.norm(hidden)
        logits = self.lm_head(hidden)
        return logits


class MultiExitLLMProduction(nn.Module):
    """Production-grade multi-exit LLM wrapper"""
    
    def __init__(
        self,
        model_name: str = "unsloth/gemma-3-1b-it",
        exit_layer_indices: List[int] = None,
        freeze_base: bool = True,
        use_cache: bool = True
    ):
        super().__init__()
        
        logger.info(f"Loading base model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Get model config
        self.hidden_dim = self.base_model.config.hidden_size
        self.vocab_size = self.base_model.config.vocab_size
        self.num_layers = self.base_model.config.num_hidden_layers
        
        if exit_layer_indices is None:
            # Default exit points for Gemma 3 1B (26 layers)
            exit_layer_indices = [6, 10, 14, 18, 22, 26]
        
        self.exit_layer_indices = sorted(exit_layer_indices)
        self.use_cache = use_cache
        
        # Create exit heads for each layer
        self.exit_heads = nn.ModuleDict()
        for layer_idx in self.exit_layer_indices:
            self.exit_heads[str(layer_idx)] = ExitHead(
                self.hidden_dim, 
                self.vocab_size,
                layer_idx
            )
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        logger.info(f"Multi-exit LLM ready. Exit points: {self.exit_layer_indices}")
    
    def get_hidden_states_at_layers(
        self,
        input_ids: torch.Tensor,
        exit_threshold: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get hidden states at all exit layers during forward pass.
        
        Returns dictionary with:
        - 'hidden_states': Dict[layer_idx] -> hidden states
        - 'logits_at_exits': Dict[layer_idx] -> logits
        - 'confidences': Dict[layer_idx] -> confidence scores
        - 'exit_layer': which layer exited (if threshold set)
        """
        
        outputs = self.base_model(
            input_ids,
            output_hidden_states=True,
            return_dict=True
        )
        
        all_hidden_states = outputs.hidden_states
        result = {
            'hidden_states': {},
            'logits_at_exits': {},
            'confidences': {},
            'exit_layer': None,
            'full_logits': outputs.logits
        }
        
        for layer_idx in self.exit_layer_indices:
            if layer_idx <= len(all_hidden_states):
                # Get hidden state at this layer (layer_idx - 1 because of embedding)
                hidden_state = all_hidden_states[layer_idx - 1]
                
                # Generate logits
                exit_head = self.exit_heads[str(layer_idx)]
                logits = exit_head(hidden_state)
                
                # Calculate confidence (max softmax probability of last token)
                probs = F.softmax(logits[:, -1, :], dim=-1)
                confidence = probs.max(dim=-1)[0].mean().item()
                
                result['hidden_states'][layer_idx] = hidden_state
                result['logits_at_exits'][layer_idx] = logits
                result['confidences'][layer_idx] = confidence
                
                # Check for early exit
                if exit_threshold is not None and confidence >= exit_threshold:
                    result['exit_layer'] = layer_idx
                    break
        
        return result
    
    def generate_tokens(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        exit_threshold: Optional[float] = None,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict:
        """Generate tokens using early exiting"""
        
        generated_ids = input_ids.clone()
        exit_layers_used = []
        confidences = []
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Get hidden states and exit information
                result = self.get_hidden_states_at_layers(
                    generated_ids,
                    exit_threshold=exit_threshold
                )
                
                # Use appropriate logits (exit or final)
                if result['exit_layer'] is not None:
                    logits = result['logits_at_exits'][result['exit_layer']]
                    exit_layers_used.append(result['exit_layer'])
                    confidences.append(result['confidences'][result['exit_layer']])
                else:
                    logits = result['full_logits']
                    exit_layers_used.append(self.num_layers)
                    confidences.append(result['confidences'][self.exit_layer_indices[-1]])
                
                # Sample next token
                next_logits = logits[:, -1, :] / temperature
                
                # Top-p sampling
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True
                )
                cumsum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_keep = cumsum_probs <= top_p
                sorted_indices_to_keep[..., 0] = True
                
                indices_to_remove = sorted_indices[~sorted_indices_to_keep]
                next_logits[:, indices_to_remove] = -float('inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        return {
            'generated_ids': generated_ids,
            'num_tokens_generated': generated_ids.shape[1] - input_ids.shape[1],
            'exit_layers': exit_layers_used,
            'confidences': confidences,
            'avg_exit_layer': sum(exit_layers_used) / len(exit_layers_used),
            'avg_confidence': sum(confidences) / len(confidences)
        }
    
    def extract_training_data(
        self,
        input_ids: torch.Tensor,
        batch_size: int = 32
    ) -> Dict[int, List[torch.Tensor]]:
        """Extract hidden states and targets for training confidence classifiers"""
        
        hidden_states_by_layer = {layer: [] for layer in self.exit_layer_indices}
        targets_by_layer = {layer: [] for layer in self.exit_layer_indices}
        
        with torch.no_grad():
            result = self.get_hidden_states_at_layers(input_ids)
            
            # Get final layer predictions for labels
            final_preds = result['full_logits'].argmax(dim=-1)
            
            for layer_idx in self.exit_layer_indices:
                if layer_idx in result['logits_at_exits']:
                    exit_preds = result['logits_at_exits'][layer_idx].argmax(dim=-1)
                    
                    # Label as correct (1) or incorrect (0)
                    targets = (exit_preds == final_preds).float()
                    
                    hidden_states_by_layer[layer_idx].append(
                        result['hidden_states'][layer_idx]
                    )
                    targets_by_layer[layer_idx].append(targets)
        
        return hidden_states_by_layer, targets_by_layer


class TextDataset(Dataset):
    """Simple text dataset for training"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }


def load_model() -> MultiExitLLMProduction:
    """Production model loading"""
    return MultiExitLLMProduction(
        model_name="unsloth/gemma-3-1b-it",
        exit_layer_indices=[6, 10, 14, 18, 22, 26],
        freeze_base=True
    )
