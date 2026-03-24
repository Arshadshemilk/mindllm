"""Configuration for MindEdgeAI system"""

from dataclasses import dataclass
from typing import List


@dataclass
class ModelConfig:
    """Configuration for the multi-exit LLM model"""
    model_name: str = "unsloth/gemma-3-1b-it"  # Gemma 3 1B instruction-tuned
    num_layers: int = 26  # Gemma 3 1B has 26 layers
    hidden_dim: int = 2304  # Gemma 3 1B hidden dimension
    exit_layer_indices: List[int] = None
    
    def __post_init__(self):
        if self.exit_layer_indices is None:
            # Exit points for Gemma 3 1B (26 layers, evenly distributed)
            self.exit_layer_indices = [6, 10, 14, 18, 22, 26]


@dataclass
class ConfidenceClassifierConfig:
    """Configuration for confidence classifiers"""
    input_dim: int = 2048
    hidden_dim1: int = 256
    hidden_dim2: int = 64
    output_dim: int = 1
    dropout_rate: float = 0.1
    temperature: float = 1.0
    target_ece: float = 0.05
    gamma: float = 0.9  # Discount factor for loss


@dataclass
class BanditConfig:
    """Configuration for UCB bandit controller"""
    # Action space: confidence thresholds
    min_threshold: float = 0.50
    max_threshold: float = 0.99
    num_arms: int = 20  # Number of discrete threshold levels
    
    # Exploration parameters
    exploration_constant_warmup: float = 1.0  # for tokens 0-100
    exploration_constant_high: float = 3.0    # for tokens 100-500
    exploration_constant_standard: float = 2.0  # for tokens > 500
    
    # Reward function coefficients
    confidence_weight: float = 0.7
    energy_weight: float = 0.3
    
    # Thresholds
    warmup_tokens: int = 100
    extended_explore_tokens: int = 500
    
    def get_threshold_at_arm(self, arm_index: int) -> float:
        """Get confidence threshold for a given arm index"""
        if arm_index < 0 or arm_index >= self.num_arms:
            raise ValueError(f"Arm index must be in [0, {self.num_arms-1}]")
        return self.min_threshold + (self.max_threshold - self.min_threshold) * (arm_index / (self.num_arms - 1))


# Default configurations
DEFAULT_MODEL_CONFIG = ModelConfig()
DEFAULT_CLASSIFIER_CONFIG = ConfidenceClassifierConfig()
DEFAULT_BANDIT_CONFIG = BanditConfig()
