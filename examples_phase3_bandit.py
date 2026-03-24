"""
Example script demonstrating Phase 3: Bandit controller online learning.

This script shows:
1. Initializing the UCB bandit controller
2. Simulating inference with adaptive threshold selection
3. Tracking bandit arm performance
4. Analyzing learned policies
"""

import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.phase3_bandit_controller.ucb_controller import (
    UCBBanditController,
    BanditState,
    AdaptiveThresholdManager
)
from src.config import BanditConfig, ModelConfig
from src.utils.utilities import set_seed


def simulate_inference_episode(
    bandit_controller: UCBBanditController,
    num_tokens: int = 100,
    max_layers: int = 18
):
    """
    Simulate an inference episode with the bandit controller.
    
    Args:
        bandit_controller: The UCB bandit controller
        num_tokens: Number of tokens to generate
        max_layers: Total number of layers
    """
    print("\n" + "=" * 60)
    print("Simulating Inference with Bandit Control")
    print("=" * 60)
    
    for token_idx in range(num_tokens):
        # Get system state
        state = bandit_controller.get_system_state(
            token_position=token_idx,
            energy_budget_remaining=1.0 - (token_idx / num_tokens)
        )
        
        # Select threshold
        threshold, arm_idx = bandit_controller.select_threshold(state)
        
        # Simulate exit layer and confidence (in real inference, from model)
        # Simulate better performance for middle layers
        simulated_exit_layer = 4 + (arm_idx * 14) // bandit_controller.num_arms
        simulated_exit_layer = min(max_layers, max(4, simulated_exit_layer))
        
        # Confidence inversely related to threshold
        simulated_confidence = min(0.99, threshold + np.random.normal(0, 0.05))
        simulated_confidence = max(0.5, simulated_confidence)
        
        # Update bandit
        bandit_controller.update_arm(
            arm_idx=arm_idx,
            exit_layer=simulated_exit_layer,
            exit_confidence=simulated_confidence,
            max_layers=max_layers
        )
        
        # Print progress
        if (token_idx + 1) % 20 == 0 or token_idx == 0:
            print(f"\nToken {token_idx + 1}/{num_tokens}")
            print(f"  State: Position={state.token_position}, "
                  f"Energy={state.remaining_energy_budget:.3f}, "
                  f"ConfMean={state.historical_confidence_mean:.3f}")
            print(f"  Selected: Arm {arm_idx}, Threshold={threshold:.4f}")
            print(f"  Outcome: ExitLayer={simulated_exit_layer}, "
                  f"Confidence={simulated_confidence:.4f}")


def analyze_bandit_performance(
    bandit_controller: UCBBanditController
):
    """Analyze the learned bandit policy."""
    print("\n" + "=" * 60)
    print("Bandit Controller Analysis")
    print("=" * 60)
    
    # Get summary statistics
    summary = bandit_controller.get_summary_statistics()
    
    print(f"\nOverall Performance:")
    print(f"  Total pulls: {summary['total_pulls']}")
    print(f"  Mean reward: {summary['mean_reward']:.4f}")
    print(f"  Mean confidence: {summary['mean_confidence']:.4f}")
    print(f"  Mean exit layer: {summary['mean_exit_layer']:.2f}")
    print(f"  Std reward: {summary['std_reward']:.4f}")
    
    # Best arm
    print(f"\nBest Performing Arm:")
    print(f"  Arm index: {summary['best_arm']}")
    print(f"  Threshold: {summary['best_arm_threshold']:.4f}")
    
    # Arm statistics
    print(f"\nArm-level Statistics:")
    arm_stats = bandit_controller.get_arm_statistics()
    
    # Show top 5 arms
    sorted_arms = sorted(
        arm_stats.items(),
        key=lambda x: x[1]['mean_reward'],
        reverse=True
    )
    
    print(f"Top 5 performing arms:")
    for arm_name, stats in sorted_arms[:5]:
        print(f"  {arm_name}:")
        print(f"    Threshold: {stats['threshold']:.4f}")
        print(f"    Mean Reward: {stats['mean_reward']:.4f}")
        print(f"    Pulls: {stats['num_pulls']}")
    
    # Exploration pattern
    print(f"\nExploration Pattern:")
    pull_counts = [arm['num_pulls'] for arm in arm_stats.values()]
    print(f"  Min pulls per arm: {min(pull_counts)}")
    print(f"  Max pulls per arm: {max(pull_counts)}")
    print(f"  Mean pulls per arm: {np.mean(pull_counts):.1f}")


def print_threshold_schedule(bandit_config: BanditConfig):
    """Print the threshold schedule for all arms."""
    print("\n" + "=" * 60)
    print("Confidence Threshold Schedule")
    print("=" * 60)
    
    print(f"\nArm thresholds (min={bandit_config.min_threshold}, max={bandit_config.max_threshold}):")
    for arm_idx in range(bandit_config.num_arms):
        threshold = bandit_config.get_threshold_at_arm(arm_idx)
        print(f"  Arm {arm_idx:2d}: {threshold:.4f}")


def print_exploration_schedule(bandit_config: BanditConfig):
    """Print the exploration schedule."""
    print("\n" + "=" * 60)
    print("Exploration Schedule")
    print("=" * 60)
    
    print(f"\nWarmup Phase (tokens 0-{bandit_config.warmup_tokens-1}):")
    print(f"  Strategy: Forced round-robin")
    print(f"  Exploration constant: 0.0")
    
    print(f"\nExtended Exploration Phase (tokens {bandit_config.warmup_tokens}-{bandit_config.extended_explore_tokens-1}):")
    print(f"  Strategy: UCB with high exploration")
    print(f"  Exploration constant: {bandit_config.exploration_constant_high}")
    
    print(f"\nStandard Phase (tokens {bandit_config.extended_explore_tokens}+):")
    print(f"  Strategy: Standard UCB")
    print(f"  Exploration constant: {bandit_config.exploration_constant_standard}")


def main():
    """Main demonstration function."""
    print("MindEdgeAI - Phase 3: Bandit Controller Demo")
    print("=" * 60)
    
    set_seed(42)
    
    # Configuration
    bandit_config = BanditConfig()
    model_config = ModelConfig()
    
    # Print schedules
    print_threshold_schedule(bandit_config)
    print_exploration_schedule(bandit_config)
    
    # Initialize bandit controller
    bandit_controller = UCBBanditController(
        min_threshold=bandit_config.min_threshold,
        max_threshold=bandit_config.max_threshold,
        num_arms=bandit_config.num_arms,
        exploration_constant_warmup=bandit_config.exploration_constant_warmup,
        exploration_constant_high=bandit_config.exploration_constant_high,
        exploration_constant_standard=bandit_config.exploration_constant_standard,
        confidence_weight=bandit_config.confidence_weight,
        energy_weight=bandit_config.energy_weight,
        warmup_tokens=bandit_config.warmup_tokens,
        extended_explore_tokens=bandit_config.extended_explore_tokens
    )
    
    # Simulate inference
    simulate_inference_episode(
        bandit_controller=bandit_controller,
        num_tokens=200,
        max_layers=model_config.num_layers
    )
    
    # Analyze results
    analyze_bandit_performance(bandit_controller)


if __name__ == "__main__":
    main()
