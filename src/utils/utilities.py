"""Utilities for MindEdgeAI system"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


def load_model_with_unsloth(model_name: str = "unsloth/gemma-3-1b-it"):
    """
    Load Gemma 3 1B model directly from Hugging Face.
    
    Args:
        model_name: Model identifier on Hugging Face
        
    Returns:
        Tuple of (model, tokenizer)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"Loading {model_name} from Hugging Face...")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer


class KVCacheManager:
    """Manages KV cache during inference for efficient generation."""
    
    def __init__(self, max_cache_size: int = 2048):
        self.cache: Dict[str, torch.Tensor] = {}
        self.max_cache_size = max_cache_size
    
    def update(self, new_cache: Dict[str, torch.Tensor]):
        """Update cache with new entries"""
        self.cache.update(new_cache)
    
    def clear(self):
        """Clear all cached values"""
        self.cache.clear()
    
    def get_cache_size(self) -> int:
        """Estimate cache size in MB"""
        total_bytes = 0
        for tensor in self.cache.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes // (1024 ** 2)


class EnergyEstimator:
    """
    Estimates energy consumption based on model operations.
    Energy is approximated as proportional to the number of layers used.
    """
    
    def __init__(self, max_layers: int = 18, base_energy: float = 1.0):
        self.max_layers = max_layers
        self.base_energy = base_energy
    
    def estimate_token_energy(self, exit_layer: int) -> float:
        """
        Estimate energy for processing one token.
        
        Args:
            exit_layer: Layer at which exit occurred
            
        Returns:
            Estimated energy units (0 to base_energy)
        """
        return self.base_energy * (exit_layer / self.max_layers)
    
    def estimate_sequence_energy(self, exit_layers: List[int]) -> float:
        """
        Estimate total energy for a sequence.
        
        Args:
            exit_layers: List of exit layers for each token
            
        Returns:
            Total estimated energy
        """
        return sum(self.estimate_token_energy(layer) for layer in exit_layers)
    
    def get_remaining_budget(
        self,
        total_budget: float,
        used_energy: float,
        num_generated_tokens: int,
        avg_tokens_to_generate: int
    ) -> float:
        """
        Estimate remaining energy budget.
        
        Args:
            total_budget: Total energy budget
            used_energy: Energy consumed so far
            num_generated_tokens: Tokens generated so far
            avg_tokens_to_generate: Average tokens to generate
            
        Returns:
            Remaining energy budget as fraction [0, 1]
        """
        if avg_tokens_to_generate <= 0:
            return 0.0
        
        remaining_energy = total_budget - used_energy
        remaining_tokens = avg_tokens_to_generate - num_generated_tokens
        
        if remaining_tokens <= 0:
            return 0.0
        
        fraction_remaining = remaining_energy / total_budget
        return max(0.0, min(1.0, fraction_remaining))


class MetricsTracker:
    """Tracks metrics during inference and training."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.confidence_scores: List[float] = []
        self.exit_layers: List[int] = []
        self.rewards: List[float] = []
    
    def record(self, key: str, value: float):
        """Record a metric value"""
        if key not in self.metrics:
            self.metrics[key] = []
        self.metrics[key].append(value)
    
    def record_confidence(self, confidence: float):
        """Record confidence score"""
        self.confidence_scores.append(confidence)
    
    def record_exit_layer(self, layer: int):
        """Record exit layer"""
        self.exit_layers.append(layer)
    
    def record_reward(self, reward: float):
        """Record reward"""
        self.rewards.append(reward)
    
    def get_summary(self) -> Dict:
        """Get summary of all tracked metrics"""
        summary = {}
        
        # Aggregate custom metrics
        for key, values in self.metrics.items():
            if values:
                summary[f'{key}_mean'] = np.mean(values)
                summary[f'{key}_std'] = np.std(values)
                summary[f'{key}_min'] = np.min(values)
                summary[f'{key}_max'] = np.max(values)
        
        # Add confidence statistics
        if self.confidence_scores:
            summary['confidence_mean'] = np.mean(self.confidence_scores)
            summary['confidence_std'] = np.std(self.confidence_scores)
            summary['confidence_min'] = np.min(self.confidence_scores)
            summary['confidence_max'] = np.max(self.confidence_scores)
        
        # Add exit layer statistics
        if self.exit_layers:
            summary['exit_layer_mean'] = np.mean(self.exit_layers)
            summary['exit_layer_std'] = np.std(self.exit_layers)
            summary['avg_layers_used'] = np.mean(self.exit_layers)
            summary['early_exit_rate'] = sum(1 for layer in self.exit_layers if layer < 18) / len(self.exit_layers)
        
        # Add reward statistics
        if self.rewards:
            summary['reward_mean'] = np.mean(self.rewards)
            summary['reward_std'] = np.std(self.rewards)
        
        return summary
    
    def reset(self):
        """Clear all recorded metrics"""
        self.metrics.clear()
        self.confidence_scores.clear()
        self.exit_layers.clear()
        self.rewards.clear()


class HyperparameterScheduler:
    """Manages dynamic hyperparameter scheduling."""
    
    def __init__(self, warmup_steps: int = 100):
        self.warmup_steps = warmup_steps
        self.step = 0
    
    def get_learning_rate_scale(self) -> float:
        """Get learning rate scale (cosine annealing)"""
        if self.step < self.warmup_steps:
            return self.step / self.warmup_steps
        else:
            progress = (self.step - self.warmup_steps) / max(1, self.warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    def get_dropout_scale(self) -> float:
        """Get dropout scale during training"""
        return max(0.0, 1.0 - self.step / (5 * self.warmup_steps))
    
    def step_update(self):
        """Increment step counter"""
        self.step += 1


def save_config_to_json(config: Dict, filepath: str):
    """Save configuration to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)


def load_config_from_json(filepath: str) -> Dict:
    """Load configuration from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
