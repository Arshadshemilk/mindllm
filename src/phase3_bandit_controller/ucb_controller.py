"""
Phase 3: Bandit Controller Implementation (Online Learning)
Implements UCB-based controller for dynamic threshold adjustment
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
import math


@dataclass
class BanditArm:
    """Represents a single arm (threshold) in the bandit"""
    arm_index: int
    threshold: float
    mean_reward: float = 0.0
    n_pulls: int = 0
    reward_sum: float = 0.0
    reward_squared_sum: float = 0.0
    
    def update(self, reward: float):
        """Update arm statistics with new reward"""
        self.n_pulls += 1
        self.reward_sum += reward
        self.reward_squared_sum += reward ** 2
        self.mean_reward = self.reward_sum / self.n_pulls
    
    def get_std(self) -> float:
        """Compute standard deviation of rewards"""
        if self.n_pulls < 2:
            return float('inf')
        variance = (self.reward_squared_sum / self.n_pulls) - (self.mean_reward ** 2)
        return math.sqrt(max(0, variance))
    
    def get_ucb(self, exploration_constant: float) -> float:
        """
        Compute Upper Confidence Bound.
        UCB = mean_reward + c * sqrt(ln(t) / n_i)
        """
        if self.n_pulls == 0:
            return float('inf')
        
        exploration_bonus = exploration_constant * math.sqrt(math.log(1.0) / self.n_pulls)
        return self.mean_reward + exploration_bonus


@dataclass
class BanditState:
    """State of the bandit at a given time"""
    total_pulls: int = 0
    remaining_energy_budget: float = 1.0  # Fraction of budget remaining
    token_position: int = 0  # Position in token sequence
    historical_confidence_mean: float = 0.5  # Mean of recent confidences
    historical_confidence_std: float = 0.1


class UCBBanditController:
    """
    UCB (Upper Confidence Bound) based bandit controller for dynamic 
    confidence threshold selection during inference.
    """
    
    def __init__(
        self,
        min_threshold: float = 0.50,
        max_threshold: float = 0.99,
        num_arms: int = 20,
        exploration_constant_warmup: float = 1.0,
        exploration_constant_high: float = 3.0,
        exploration_constant_standard: float = 2.0,
        confidence_weight: float = 0.7,
        energy_weight: float = 0.3,
        warmup_tokens: int = 100,
        extended_explore_tokens: int = 500,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize the bandit controller.
        
        Args:
            min_threshold: Minimum confidence threshold
            max_threshold: Maximum confidence threshold
            num_arms: Number of discrete threshold levels
            exploration_constant_warmup: Exploration constant for warmup phase (0-100 tokens)
            exploration_constant_high: Exploration constant for exploration phase (100-500 tokens)
            exploration_constant_standard: Exploration constant for standard phase (>500 tokens)
            confidence_weight: Weight for confidence in reward function
            energy_weight: Weight for energy cost in reward function
            warmup_tokens: Number of tokens for warmup phase
            extended_explore_tokens: Number of tokens for extended exploration
            device: Device to use
        """
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold
        self.num_arms = num_arms
        self.exploration_constant_warmup = exploration_constant_warmup
        self.exploration_constant_high = exploration_constant_high
        self.exploration_constant_standard = exploration_constant_standard
        self.confidence_weight = confidence_weight
        self.energy_weight = energy_weight
        self.warmup_tokens = warmup_tokens
        self.extended_explore_tokens = extended_explore_tokens
        self.device = device
        
        # Initialize arms
        self.arms = []
        for arm_idx in range(num_arms):
            threshold = self._get_threshold_at_arm(arm_idx)
            self.arms.append(BanditArm(arm_index=arm_idx, threshold=threshold))
        
        # Tracking
        self.total_pulls = 0
        self.confidence_history = []
        self.threshold_history = []
        self.exit_layer_history = []
        self.reward_history = []
    
    def _get_threshold_at_arm(self, arm_idx: int) -> float:
        """Map arm index to confidence threshold"""
        if self.num_arms == 1:
            return self.min_threshold
        return self.min_threshold + (self.max_threshold - self.min_threshold) * (
            arm_idx / (self.num_arms - 1)
        )
    
    def _get_exploration_constant(self, token_position: int) -> float:
        """
        Get exploration constant based on token position.
        
        Warm-up strategy:
        - 0-100 tokens: Forced round-robin (no exploration bonus)
        - 100-500 tokens: High exploration (c=3.0)
        - >500 tokens: Standard exploration (c=2.0)
        """
        if token_position < self.warmup_tokens:
            return 0.0  # Forced round-robin
        elif token_position < self.extended_explore_tokens:
            return self.exploration_constant_high
        else:
            return self.exploration_constant_standard
    
    def select_threshold(
        self,
        state: BanditState
    ) -> Tuple[float, int]:
        """
        Select a confidence threshold using UCB algorithm.
        
        Args:
            state: Current state of the system
            
        Returns:
            (threshold, arm_index) tuple
        """
        if state.token_position < self.warmup_tokens:
            # Forced round-robin during warmup
            arm_idx = state.token_position % self.num_arms
        else:
            # UCB-based selection
            exploration_const = self._get_exploration_constant(state.token_position)
            ucb_values = [arm.get_ucb(exploration_const) for arm in self.arms]
            arm_idx = int(np.argmax(ucb_values))
        
        selected_arm = self.arms[arm_idx]
        self.threshold_history.append(selected_arm.threshold)
        
        return selected_arm.threshold, arm_idx
    
    def compute_reward(
        self,
        exit_layer: int,
        exit_confidence: float,
        max_layers: int = 18
    ) -> float:
        """
        Compute reward for an action (exit decision).
        
        Reward = 0.7 * confidence - 0.3 * (exit_layer / L)
        
        This balances quality (confidence) with energy cost (layer ratio).
        
        Args:
            exit_layer: Layer at which exit occurred (1 to max_layers)
            exit_confidence: Confidence score [0, 1]
            max_layers: Total number of layers
            
        Returns:
            Reward value
        """
        # Compute energy cost as normalized layer ratio
        energy_cost = exit_layer / max_layers
        
        # Compute reward
        reward = (
            self.confidence_weight * exit_confidence -
            self.energy_weight * energy_cost
        )
        
        return reward
    
    def update_arm(
        self,
        arm_idx: int,
        exit_layer: int,
        exit_confidence: float,
        max_layers: int = 18
    ):
        """
        Update arm statistics after action execution.
        
        Args:
            arm_idx: Index of selected arm
            exit_layer: Layer at which exit occurred
            exit_confidence: Confidence score at exit
            max_layers: Total number of layers
        """
        # Compute reward
        reward = self.compute_reward(exit_layer, exit_confidence, max_layers)
        
        # Update arm
        self.arms[arm_idx].update(reward)
        
        # Update history
        self.total_pulls += 1
        self.confidence_history.append(exit_confidence)
        self.exit_layer_history.append(exit_layer)
        self.reward_history.append(reward)
    
    def get_system_state(
        self,
        token_position: int,
        energy_budget_remaining: float = 1.0,
        window_size: int = 10
    ) -> BanditState:
        """
        Get current system state for threshold selection.
        
        Args:
            token_position: Current position in token sequence
            energy_budget_remaining: Fraction of energy budget remaining
            window_size: Window size for computing historical statistics
            
        Returns:
            BanditState object
        """
        # Compute historical statistics over recent window
        if len(self.confidence_history) > 0:
            recent_confidences = self.confidence_history[-window_size:]
            historical_confidence_mean = float(np.mean(recent_confidences))
            historical_confidence_std = float(np.std(recent_confidences))
        else:
            historical_confidence_mean = 0.5
            historical_confidence_std = 0.1
        
        return BanditState(
            total_pulls=self.total_pulls,
            remaining_energy_budget=energy_budget_remaining,
            token_position=token_position,
            historical_confidence_mean=historical_confidence_mean,
            historical_confidence_std=historical_confidence_std
        )
    
    def get_arm_statistics(self) -> Dict:
        """
        Get detailed statistics for all arms.
        
        Returns:
            Dictionary with arm statistics
        """
        stats = {}
        for arm in self.arms:
            stats[f'arm_{arm.arm_index}'] = {
                'threshold': arm.threshold,
                'mean_reward': arm.mean_reward,
                'num_pulls': arm.n_pulls,
                'ucb_value': arm.get_ucb(self.exploration_constant_standard),
                'std_reward': arm.get_std()
            }
        return stats
    
    def get_summary_statistics(self) -> Dict:
        """
        Get summary statistics of the bandit controller.
        
        Returns:
            Dictionary with overall statistics
        """
        if len(self.reward_history) == 0:
            return {
                'total_pulls': 0,
                'mean_reward': 0.0,
                'mean_confidence': 0.0,
                'mean_exit_layer': 0.0
            }
        
        return {
            'total_pulls': self.total_pulls,
            'mean_reward': float(np.mean(self.reward_history)),
            'mean_confidence': float(np.mean(self.confidence_history)),
            'mean_exit_layer': float(np.mean(self.exit_layer_history)),
            'std_reward': float(np.std(self.reward_history)),
            'best_arm': int(np.argmax([arm.mean_reward for arm in self.arms])),
            'best_arm_threshold': float(self.arms[int(np.argmax([arm.mean_reward for arm in self.arms]))].threshold)
        }


class AdaptiveThresholdManager:
    """
    Manager for adaptive threshold adjustment during inference.
    Integrates confidence classifiers, bandit controller, and the multi-exit model.
    """
    
    def __init__(
        self,
        bandit_controller: UCBBanditController,
        max_layers: int = 18
    ):
        self.bandit = bandit_controller
        self.max_layers = max_layers
        self.token_counter = 0
        self.energy_budget = 1.0
    
    def select_and_apply_threshold(
        self,
        token_position: int,
        remaining_energy: float = 1.0
    ) -> float:
        """
        Select a confidence threshold for the current token.
        
        Args:
            token_position: Position in token sequence
            remaining_energy: Fraction of energy budget remaining [0, 1]
            
        Returns:
            Confidence threshold for early exit decision
        """
        # Get system state
        state = self.bandit.get_system_state(
            token_position=token_position,
            energy_budget_remaining=remaining_energy
        )
        
        # Select threshold using UCB
        threshold, arm_idx = self.bandit.select_threshold(state)
        
        return threshold, arm_idx
    
    def record_decision(
        self,
        arm_idx: int,
        exit_layer: int,
        exit_confidence: float
    ):
        """
        Record the decision and update bandit statistics.
        
        Args:
            arm_idx: Selected arm index
            exit_layer: Layer where exit occurred
            exit_confidence: Confidence at exit
        """
        self.bandit.update_arm(arm_idx, exit_layer, exit_confidence, self.max_layers)
        self.token_counter += 1
