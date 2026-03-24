"""
Integration module that ties together all three phases of MindEdgeAI.
This is the main inference engine.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
import numpy as np

from .phase1_exit_architecture.multi_exit_llm import MultiExitLLM, ExitOutput
from .phase2_confidence_classifier.confidence_heads import ConfidenceClassifierEnsemble
from .phase3_bandit_controller.ucb_controller import UCBBanditController, AdaptiveThresholdManager, BanditState
from .utils.utilities import EnergyEstimator, MetricsTracker, KVCacheManager


class MindEdgeAIEngine:
    """
    Complete MindEdgeAI inference engine integrating:
    - Phase 1: Multi-exit LLM architecture
    - Phase 2: Confidence classifiers
    - Phase 3: UCB bandit controller for adaptive thresholds
    """
    
    def __init__(
        self,
        multi_exit_model: MultiExitLLM,
        confidence_classifier: ConfidenceClassifierEnsemble,
        bandit_controller: UCBBanditController,
        max_layers: int = 18,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the MindEdgeAI engine.
        
        Args:
            multi_exit_model: The multi-exit LLM
            confidence_classifier: Confidence estimation heads
            bandit_controller: UCB bandit for threshold adaptation
            max_layers: Total number of layers in the model
            device: Device to use for computation
        """
        self.multi_exit_model = multi_exit_model.to(device)
        self.confidence_classifier = confidence_classifier.to(device)
        self.bandit_controller = bandit_controller
        self.adaptive_threshold_manager = AdaptiveThresholdManager(
            bandit_controller,
            max_layers=max_layers
        )
        self.max_layers = max_layers
        self.device = device
        
        # Utilities
        self.energy_estimator = EnergyEstimator(max_layers=max_layers)
        self.cache_manager = KVCacheManager()
        self.metrics_tracker = MetricsTracker()
    
    def generate_token_with_adaptive_threshold(
        self,
        input_ids: torch.Tensor,
        token_position: int,
        energy_budget_remaining: float = 1.0,
        use_confidence_classifiers: bool = True
    ) -> Dict:
        """
        Generate a single token with adaptive threshold selection.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_position: Position in the generated sequence
            energy_budget_remaining: Fraction of energy budget remaining
            use_confidence_classifiers: Whether to use trained confidence classifiers
            
        Returns:
            Dictionary with generation results and metadata
        """
        # Select threshold using bandit
        threshold, arm_idx = self.adaptive_threshold_manager.select_and_apply_threshold(
            token_position=token_position,
            remaining_energy=energy_budget_remaining
        )
        
        with torch.no_grad():
            # Forward pass through multi-exit model
            exit_output: ExitOutput = self.multi_exit_model.forward(
                input_ids,
                exit_threshold=threshold,
                past_key_values=self.cache_manager.cache if self.cache_manager.cache else None
            )
            
            # Update KV cache
            self.cache_manager.update(exit_output.kv_cache)
            
            # Get next token prediction
            next_logits = exit_output.logits[:, -1, :]
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            
            # Compute actual confidence (could be refined if confidence classifiers are used)
            actual_confidence = exit_output.exit_confidence
            
            # Record metrics
            self.metrics_tracker.record_confidence(actual_confidence)
            self.metrics_tracker.record_exit_layer(exit_output.exit_layer)
            
            # Update bandit with actual outcome
            self.adaptive_threshold_manager.record_decision(
                arm_idx=arm_idx,
                exit_layer=exit_output.exit_layer,
                exit_confidence=actual_confidence
            )
            
            # Estimate energy
            token_energy = self.energy_estimator.estimate_token_energy(exit_output.exit_layer)
            
            result = {
                'next_token': next_token,
                'exit_layer': exit_output.exit_layer,
                'confidence': actual_confidence,
                'threshold_used': threshold,
                'arm_idx': arm_idx,
                'token_energy': token_energy,
                'logits': next_logits
            }
            
            return result
    
    def generate_sequence(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        total_energy_budget: float = 1.0,
        use_bandit_control: bool = True
    ) -> Dict:
        """
        Generate a complete sequence with adaptive threshold control.
        
        Args:
            input_ids: Initial token IDs
            max_new_tokens: Maximum tokens to generate
            total_energy_budget: Total energy budget for generation
            use_bandit_control: Whether to use bandit controller for thresholds
            
        Returns:
            Dictionary with generated sequence and statistics
        """
        self.metrics_tracker.reset()
        self.cache_manager.clear()
        
        generated_ids = input_ids.clone()
        total_energy_used = 0.0
        
        for step in range(max_new_tokens):
            # Check energy constraint
            energy_used_fraction = total_energy_used / total_energy_budget
            remaining_budget_fraction = 1.0 - energy_used_fraction
            
            if remaining_budget_fraction <= 0.01:  # Stop if <1% budget remains
                break
            
            # Generate next token
            result = self.generate_token_with_adaptive_threshold(
                input_ids=generated_ids[:, -1:] if step > 0 else generated_ids,
                token_position=step,
                energy_budget_remaining=remaining_budget_fraction,
                use_confidence_classifiers=True
            )
            
            # Append token
            generated_ids = torch.cat([generated_ids, result['next_token']], dim=1)
            total_energy_used += result['token_energy']
            
            # Record reward
            reward = self.bandit_controller.compute_reward(
                exit_layer=result['exit_layer'],
                exit_confidence=result['confidence'],
                max_layers=self.max_layers
            )
            self.metrics_tracker.record_reward(reward)
        
        # Compile results
        summary = self.metrics_tracker.get_summary()
        bandit_summary = self.bandit_controller.get_summary_statistics()
        
        return {
            'generated_ids': generated_ids,
            'num_tokens_generated': generated_ids.shape[1] - input_ids.shape[1],
            'total_energy_used': total_energy_used,
            'energy_budget': total_energy_budget,
            'energy_efficiency': (generated_ids.shape[1] - input_ids.shape[1]) / (total_energy_used + 1e-6),
            'metrics': summary,
            'bandit_stats': bandit_summary,
            'exit_layers': self.metrics_tracker.exit_layers,
            'confidences': self.metrics_tracker.confidence_scores
        }
    
    def get_model_config(self) -> Dict:
        """Get configuration of the entire system"""
        return {
            'max_layers': self.max_layers,
            'exit_layer_indices': self.multi_exit_model.exit_layer_indices,
            'num_confidence_heads': len(self.confidence_classifier.heads),
            'num_bandit_arms': self.bandit_controller.num_arms,
            'threshold_range': (
                self.bandit_controller.min_threshold,
                self.bandit_controller.max_threshold
            ),
            'device': str(self.device)
        }
    
    def get_statistics(self) -> Dict:
        """Get current statistics of all components"""
        return {
            'metrics': self.metrics_tracker.get_summary(),
            'bandit_stats': self.bandit_controller.get_summary_statistics(),
            'arm_statistics': self.bandit_controller.get_arm_statistics(),
            'cache_size_mb': self.cache_manager.get_cache_size()
        }
